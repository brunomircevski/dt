#include "tree_cuda.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validateCudaLaunchOptions(TrainingOptions &options) {
  options.cudaRowsPerTile =
      std::clamp(options.cudaRowsPerTile, std::size_t{1024},
                 std::size_t{1 << 20});
  options.cudaMaxTilesPerFeature =
      std::clamp(options.cudaMaxTilesPerFeature, 1, 512);
  options.cudaScoreThreadsPerBlock =
      std::clamp(options.cudaScoreThreadsPerBlock, 32, 1024);
  options.cudaGatherBlockSize =
      std::clamp(options.cudaGatherBlockSize, 32, 1024);
}

} // namespace

// =============================================================================
// GPU split search
// -----------------------------------------------------------------------------
// The CPU still drives the recursion (see TreeBase::buildNode). For each node it
// asks the GPU one question: "given these rows, what is the best split?"
//
// The old version was slow because, per node, it launched ~160 tiny GPU
// operations (a sort + several Thrust passes + a host copy PER feature) and then
// scanned all rows with a SINGLE GPU thread (<<<1,1>>>). Launch overhead and a
// serial scan dominated everything.
//
// This version does the whole node with only a few launches:
//   1. gather      : copy this node's (value, rowId) pairs into scratch          (1 kernel)
//   2. segmentedSort: sort all features' slices by value in ONE CUB launch       (1 sort)
//   3. scoreSplits : evaluate every feature's thresholds in parallel             (1 kernel)
//   4. one copy of the per-feature results back to the host.
//
// We deliberately do NOT carry class ids through the sort. Instead each row's
// class is looked up from the (tiny, L2-resident) global class array using the
// row id. That keeps the sort to a single key/value pair, which lets us use a
// fast segmented radix sort instead of a slow tuple sort.
//
// Sorting matters because, for a numeric feature, the best threshold always sits
// between two adjacent values once the rows are ordered. After sorting we can
// sweep left-to-right and keep running class counts in O(1) per step.
// =============================================================================

namespace {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                          \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +               \
                               cudaGetErrorString(error));                     \
    }                                                                          \
  } while (0)

// Training knobs copied into a small POD so we can hand them to kernels by value.
struct GpuTrainConfig {
  int impurityMeasure = 0;    // 0 = entropy, 1 = gini
  int splitSelectionMode = 0; // 0 = C4.5 gain ratio, 1 = CART max gain
  double epsilon = 1e-9;
  std::size_t minSamplesPerLeaf = 1;
  std::uint16_t numClasses = 0;
};

// One candidate split: "feature <= threshold?" plus its scores.
struct GpuSplitCandidate {
  bool valid = false;
  std::size_t featureIndex = 0;
  double threshold = 0.0;
  double informationGain = 0.0;
  double splitInformation = 0.0;
  double gainRatio = 0.0;
  std::size_t leftRowCount = 0;
};

// Class counts live in small per-thread arrays, so we cap the number of classes.
constexpr std::uint16_t kMaxGpuClasses = 64;

// Feature values are stored on the GPU as 32-bit floats instead of 64-bit
// doubles. This halves the two largest memory consumers (the uploaded feature
// matrix and the per-node value scratch) and makes the radix sort ~2x faster
// (32-bit keys = 4 radix passes instead of 8). For datasets like SUSY whose
// features are already float-precision measurements this does not change the
// resulting tree in any meaningful way; thresholds are still returned as
// doubles. If you ever need bit-exact agreement with the CPU backend, change
// this alias back to double.
using GpuValue = float;

// Impurity (entropy or Gini) of a node described purely by its class counts.
__device__ double deviceImpurity(const std::uint32_t *counts,
                                 std::uint16_t numClasses, std::size_t total,
                                 int impurityMeasure) {
  if (total == 0) {
    return 0.0;
  }
  const double totalDouble = static_cast<double>(total);

  if (impurityMeasure == 1) { // Gini = 1 - sum(p^2)
    double sumSquared = 0.0;
    for (std::uint16_t c = 0; c < numClasses; ++c) {
      if (counts[c] == 0) {
        continue;
      }
      const double p = static_cast<double>(counts[c]) / totalDouble;
      sumSquared += p * p;
    }
    return 1.0 - sumSquared;
  }

  double result = 0.0; // Entropy = -sum(p * log2 p)
  for (std::uint16_t c = 0; c < numClasses; ++c) {
    if (counts[c] == 0) {
      continue;
    }
    const double p = static_cast<double>(counts[c]) / totalDouble;
    result -= p * log2(p);
  }
  return result;
}

// CART tie-break: highest gain, then earliest feature, then lowest threshold.
__device__ bool deviceIsBetterMaxGain(const GpuSplitCandidate &lhs,
                                      const GpuSplitCandidate &rhs,
                                      double epsilon) {
  if (!lhs.valid) return false;
  if (!rhs.valid) return true;
  if (lhs.informationGain > rhs.informationGain + epsilon) return true;
  if (rhs.informationGain > lhs.informationGain + epsilon) return false;
  if (lhs.featureIndex != rhs.featureIndex) return lhs.featureIndex < rhs.featureIndex;
  return lhs.threshold < rhs.threshold - epsilon;
}

// C4.5 tie-break: highest gain ratio, then gain, then feature, then threshold.
__device__ bool deviceIsBetterC45(const GpuSplitCandidate &lhs,
                                  const GpuSplitCandidate &rhs, double epsilon) {
  if (!lhs.valid) return false;
  if (!rhs.valid) return true;
  if (lhs.gainRatio > rhs.gainRatio + epsilon) return true;
  if (rhs.gainRatio > lhs.gainRatio + epsilon) return false;
  if (lhs.informationGain > rhs.informationGain + epsilon) return true;
  if (rhs.informationGain > lhs.informationGain + epsilon) return false;
  if (lhs.featureIndex != rhs.featureIndex) return lhs.featureIndex < rhs.featureIndex;
  return lhs.threshold < rhs.threshold - epsilon;
}

__device__ bool deviceIsBetter(const GpuSplitCandidate &lhs,
                               const GpuSplitCandidate &rhs,
                               const GpuTrainConfig &config) {
  return config.splitSelectionMode == 1
             ? deviceIsBetterMaxGain(lhs, rhs, config.epsilon)
             : deviceIsBetterC45(lhs, rhs, config.epsilon);
}

// Score one threshold given the left/right class counts it would produce.
// Mirrors TreeBase::scoreCandidateFromCounts so GPU and CPU agree.
__device__ GpuSplitCandidate deviceScoreCandidate(
    std::size_t featureIndex, std::size_t totalRows, double parentImpurity,
    const std::uint32_t *leftCounts, const std::uint32_t *rightCounts,
    std::size_t leftSize, double threshold, const GpuTrainConfig &config) {
  GpuSplitCandidate result;
  result.featureIndex = featureIndex;
  result.threshold = threshold;
  result.leftRowCount = leftSize;

  const std::size_t rightSize = totalRows - leftSize;
  if (leftSize == 0 || rightSize == 0) {
    return result;
  }
  if (config.minSamplesPerLeaf != 0 &&
      (leftSize < config.minSamplesPerLeaf ||
       rightSize < config.minSamplesPerLeaf)) {
    return result;
  }

  const double leftImpurity =
      deviceImpurity(leftCounts, config.numClasses, leftSize, config.impurityMeasure);
  const double rightImpurity = deviceImpurity(rightCounts, config.numClasses,
                                              rightSize, config.impurityMeasure);
  const double totalDouble = static_cast<double>(totalRows);
  const double afterSplit =
      (static_cast<double>(leftSize) / totalDouble) * leftImpurity +
      (static_cast<double>(rightSize) / totalDouble) * rightImpurity;

  result.informationGain = parentImpurity - afterSplit;

  if (config.splitSelectionMode == 1) { // CART: gain alone decides.
    result.valid = (result.informationGain > config.epsilon);
    return result;
  }

  // C4.5: penalise lopsided splits via the gain ratio.
  const double leftProbability = static_cast<double>(leftSize) / totalDouble;
  const double rightProbability = static_cast<double>(rightSize) / totalDouble;
  result.splitInformation = -leftProbability * log2(leftProbability) -
                            rightProbability * log2(rightProbability);
  if (result.informationGain <= config.epsilon ||
      result.splitInformation <= config.epsilon) {
    return result;
  }
  result.gainRatio = result.informationGain / result.splitInformation;
  result.valid = true;
  return result;
}

// -----------------------------------------------------------------------------
// Kernel 1: gather this node's rows into dense, feature-major scratch arrays.
// Each thread handles one (feature, localRow) pair and writes the feature value
// (sort key) and the original row id (sort payload). Class ids are NOT copied;
// they are looked up later from the global array using the row id.
// -----------------------------------------------------------------------------
__global__ void gatherNodeFeaturesKernel(const std::uint32_t *currentRows,
                                         const GpuValue *features,
                                         std::size_t nodeRows,
                                         std::size_t totalRows,
                                         std::size_t featureCount,
                                         GpuValue *nodeValues,
                                         std::uint32_t *nodeRowIds) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t workItems = featureCount * nodeRows;
  if (tid >= workItems) {
    return;
  }

  const std::size_t featureIndex = tid / nodeRows;
  const std::size_t localRow = tid % nodeRows;
  const std::size_t rowIndex = currentRows[localRow];
  const std::size_t offset = featureIndex * nodeRows + localRow;

  nodeValues[offset] = features[featureIndex * totalRows + rowIndex];
  nodeRowIds[offset] = static_cast<std::uint32_t>(rowIndex);
}

// -----------------------------------------------------------------------------
// Kernel 2: find the best split for every feature, ONE BLOCK PER FEATURE.
//
// Each feature's slice (already sorted by value) is split into `blockDim`
// contiguous chunks, one per thread. The work happens in three cooperative
// phases so the whole block — not a single thread — does the sweep:
//
//   Phase 1: each thread builds the class histogram of its own chunk.
//   Phase 2: an exclusive prefix scan over the chunk histograms tells each
//            thread how many rows of each class lie BEFORE its chunk. That is
//            exactly the "left" class counts at the start of the chunk.
//   Phase 3: each thread sweeps its chunk, evaluating every threshold (gap
//            between two distinct values with a class change) and keeping its
//            best. A final reduction picks the feature's overall best.
//
// Class ids are looked up from the global array (a few MB, stays hot in L2).
// Shared memory is laid out manually: candidates first (8-byte aligned because
// they contain doubles), then the per-chunk histograms, then the totals.
// -----------------------------------------------------------------------------
__global__ void scoreSplitsKernel(const GpuValue *values,
                                   const std::uint32_t *rowIds,
                                   const std::uint16_t *classIds,
                                   std::size_t nodeRows, GpuTrainConfig config,
                                   GpuSplitCandidate *out) {
  const std::size_t feature = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const std::uint16_t numClasses = config.numClasses;

  const GpuValue *v = values + feature * nodeRows;
  const std::uint32_t *ids = rowIds + feature * nodeRows;

  extern __shared__ unsigned char sharedRaw[];
  GpuSplitCandidate *sBest = reinterpret_cast<GpuSplitCandidate *>(sharedRaw);
  std::uint32_t *sHist =
      reinterpret_cast<std::uint32_t *>(sBest + threads); // [threads][numClasses]
  std::uint32_t *sTotal = sHist + threads * numClasses;   // [numClasses]
  __shared__ double sParentImpurity;

  // Each thread owns rows [chunkBegin, chunkEnd).
  const std::size_t chunkSize = (nodeRows + threads - 1) / threads;
  const std::size_t chunkBegin = static_cast<std::size_t>(tid) * chunkSize;
  const std::size_t chunkEnd =
      chunkBegin + chunkSize < nodeRows ? chunkBegin + chunkSize : nodeRows;

  // Phase 1: histogram of this thread's chunk.
  std::uint32_t *hist = sHist + tid * numClasses;
  for (std::uint16_t k = 0; k < numClasses; ++k) {
    hist[k] = 0;
  }
  for (std::size_t i = chunkBegin; i < chunkEnd; ++i) {
    hist[classIds[ids[i]]]++;
  }
  __syncthreads();

  // Phase 2: exclusive scan of the histograms across chunks (one thread per
  // class). Afterwards sHist[t][k] = rows of class k strictly before chunk t,
  // and sTotal[k] = total rows of class k in the node.
  if (tid < numClasses) {
    std::uint32_t accumulated = 0;
    for (int t = 0; t < threads; ++t) {
      std::uint32_t &slot = sHist[t * numClasses + tid];
      const std::uint32_t value = slot;
      slot = accumulated;
      accumulated += value;
    }
    sTotal[tid] = accumulated;
  }
  __syncthreads();

  if (tid == 0) {
    sParentImpurity =
        deviceImpurity(sTotal, numClasses, nodeRows, config.impurityMeasure);
  }
  __syncthreads();

  // Phase 3: sweep this chunk. left[] starts at the prefix counts from phase 2
  // and grows as the split point moves right through the chunk.
  std::uint32_t left[kMaxGpuClasses];
  for (std::uint16_t k = 0; k < numClasses; ++k) {
    left[k] = hist[k];
  }

  GpuSplitCandidate best; // invalid until a real split is found.
  for (std::size_t i = chunkBegin; i < chunkEnd; ++i) {
    if (i >= 1) {
      const std::uint16_t previousClass = classIds[ids[i - 1]];
      const std::uint16_t currentClass = classIds[ids[i]];
      const double previousValue = v[i - 1];
      const double currentValue = v[i];
      // Identical values can't be separated; equal neighbours can't help purity.
      if (currentClass != previousClass &&
          fabs(previousValue - currentValue) >= config.epsilon) {
        std::uint32_t right[kMaxGpuClasses];
        for (std::uint16_t k = 0; k < numClasses; ++k) {
          right[k] = sTotal[k] - left[k];
        }
        const double threshold = (previousValue + currentValue) / 2.0;
        const GpuSplitCandidate candidate =
            deviceScoreCandidate(feature, nodeRows, sParentImpurity, left, right,
                                 i, threshold, config);
        if (candidate.valid && deviceIsBetter(candidate, best, config)) {
          best = candidate;
        }
      }
    }
    left[classIds[ids[i]]]++; // row i now belongs to the left side.
  }

  // Reduce per-thread bests into the feature's overall best.
  sBest[tid] = best;
  __syncthreads();
  if (tid == 0) {
    GpuSplitCandidate winner = sBest[0];
    for (int t = 1; t < threads; ++t) {
      if (deviceIsBetter(sBest[t], winner, config)) {
        winner = sBest[t];
      }
    }
    out[feature] = winner;
  }
}

// =============================================================================
// Multi-block scan (used only for LARGE nodes).
//
// scoreSplitsKernel above uses one block per feature. With only ~18 features
// that leaves most of a 36-SM GPU idle on big nodes. For large nodes we instead
// split each feature's sorted slice into many TILES (one block each) so every
// SM stays busy. A block can only evaluate a threshold if it knows the class
// counts of all rows BEFORE its tile ("left" counts), so we run three steps:
//
//   A. tileHistogramKernel : each tile counts its own classes.
//   B. tilePrefixKernel    : exclusive-scan tile histograms per feature/class,
//                            giving each tile its starting "left" counts plus
//                            the per-feature totals.
//   C. tileScanKernel      : each tile sweeps its rows (same chunked logic as
//                            the single-block kernel) and writes its best split.
//   D. reduceTilesKernel   : combine each feature's tile winners into one best.
// =============================================================================

// A: per-tile class histogram. Grid = (tile, feature).
__global__ void tileHistogramKernel(const std::uint32_t *rowIds,
                                    const std::uint16_t *classIds,
                                    std::size_t nodeRows, int numTiles,
                                    std::size_t tileSize,
                                    std::uint16_t numClasses,
                                    std::uint32_t *tileHist) {
  const int tile = blockIdx.x;
  const std::size_t feature = blockIdx.y;
  const std::uint32_t *ids = rowIds + feature * nodeRows;
  const std::size_t begin = static_cast<std::size_t>(tile) * tileSize;
  const std::size_t end = begin + tileSize < nodeRows ? begin + tileSize : nodeRows;

  extern __shared__ std::uint32_t sh[]; // [numClasses]
  for (int k = threadIdx.x; k < numClasses; k += blockDim.x) {
    sh[k] = 0;
  }
  __syncthreads();
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    atomicAdd(&sh[classIds[ids[i]]], 1u);
  }
  __syncthreads();
  std::uint32_t *out = tileHist + (feature * numTiles + tile) * numClasses;
  for (int k = threadIdx.x; k < numClasses; k += blockDim.x) {
    out[k] = sh[k];
  }
}

// B: exclusive scan of tile histograms (one thread per class, one block per
// feature). Afterwards tileHist[feature][tile][k] = rows of class k before the
// tile, and featureTotals[feature][k] = total rows of class k.
__global__ void tilePrefixKernel(std::uint32_t *tileHist, int numTiles,
                                 std::uint16_t numClasses,
                                 std::uint32_t *featureTotals) {
  const std::size_t feature = blockIdx.x;
  const int k = threadIdx.x;
  if (k >= numClasses) {
    return;
  }
  std::uint32_t accumulated = 0;
  for (int t = 0; t < numTiles; ++t) {
    std::uint32_t &slot = tileHist[(feature * numTiles + t) * numClasses + k];
    const std::uint32_t value = slot;
    slot = accumulated;
    accumulated += value;
  }
  featureTotals[feature * numClasses + k] = accumulated;
}

// C: sweep each tile. Grid = (tile, feature). Same per-thread chunked logic as
// scoreSplitsKernel, but "left" starts at the tile's prefix counts from step B.
__global__ void tileScanKernel(const GpuValue *values,
                               const std::uint32_t *rowIds,
                               const std::uint16_t *classIds,
                               std::size_t nodeRows, int numTiles,
                               std::size_t tileSize, GpuTrainConfig config,
                               const std::uint32_t *tilePrefix,
                               const std::uint32_t *featureTotals,
                               GpuSplitCandidate *blockBest) {
  const int tile = blockIdx.x;
  const std::size_t feature = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const std::uint16_t numClasses = config.numClasses;

  const GpuValue *v = values + feature * nodeRows;
  const std::uint32_t *ids = rowIds + feature * nodeRows;
  const std::size_t tileBegin = static_cast<std::size_t>(tile) * tileSize;
  const std::size_t tileEnd =
      tileBegin + tileSize < nodeRows ? tileBegin + tileSize : nodeRows;
  const std::size_t tileLen = tileEnd - tileBegin;

  extern __shared__ unsigned char sharedRaw[];
  GpuSplitCandidate *sBest = reinterpret_cast<GpuSplitCandidate *>(sharedRaw);
  std::uint32_t *sHist =
      reinterpret_cast<std::uint32_t *>(sBest + threads); // [threads][numClasses]
  __shared__ std::uint32_t sTotal[kMaxGpuClasses];
  __shared__ double sParentImpurity;

  // Per-feature totals and parent impurity are shared by all tiles/threads.
  for (int k = tid; k < numClasses; k += threads) {
    sTotal[k] = featureTotals[feature * numClasses + k];
  }
  __syncthreads();
  if (tid == 0) {
    sParentImpurity =
        deviceImpurity(sTotal, numClasses, nodeRows, config.impurityMeasure);
  }

  // Each thread owns a sub-chunk [chunkBegin, chunkEnd) inside the tile.
  const std::size_t chunkSize = (tileLen + threads - 1) / threads;
  const std::size_t chunkBegin = tileBegin + static_cast<std::size_t>(tid) * chunkSize;
  const std::size_t chunkEnd =
      chunkBegin + chunkSize < tileEnd ? chunkBegin + chunkSize : tileEnd;

  std::uint32_t *hist = sHist + tid * numClasses;
  for (std::uint16_t k = 0; k < numClasses; ++k) {
    hist[k] = 0;
  }
  for (std::size_t i = chunkBegin; i < chunkEnd; ++i) {
    hist[classIds[ids[i]]]++;
  }
  __syncthreads();

  // Exclusive scan of sub-chunk histograms WITHIN the tile.
  if (tid < numClasses) {
    std::uint32_t accumulated = 0;
    for (int t = 0; t < threads; ++t) {
      std::uint32_t &slot = sHist[t * numClasses + tid];
      const std::uint32_t value = slot;
      slot = accumulated;
      accumulated += value;
    }
  }
  __syncthreads();

  // left = (rows before this tile) + (rows before this sub-chunk in the tile).
  const std::uint32_t *prefix =
      tilePrefix + (feature * numTiles + tile) * numClasses;
  std::uint32_t left[kMaxGpuClasses];
  for (std::uint16_t k = 0; k < numClasses; ++k) {
    left[k] = prefix[k] + hist[k];
  }

  GpuSplitCandidate best;
  for (std::size_t i = chunkBegin; i < chunkEnd; ++i) {
    if (i >= 1) {
      const std::uint16_t previousClass = classIds[ids[i - 1]];
      const std::uint16_t currentClass = classIds[ids[i]];
      const double previousValue = v[i - 1];
      const double currentValue = v[i];
      if (currentClass != previousClass &&
          fabs(previousValue - currentValue) >= config.epsilon) {
        std::uint32_t right[kMaxGpuClasses];
        for (std::uint16_t k = 0; k < numClasses; ++k) {
          right[k] = sTotal[k] - left[k];
        }
        const double threshold = (previousValue + currentValue) / 2.0;
        const GpuSplitCandidate candidate =
            deviceScoreCandidate(feature, nodeRows, sParentImpurity, left, right,
                                 i, threshold, config);
        if (candidate.valid && deviceIsBetter(candidate, best, config)) {
          best = candidate;
        }
      }
    }
    left[classIds[ids[i]]]++;
  }

  sBest[tid] = best;
  __syncthreads();
  if (tid == 0) {
    GpuSplitCandidate winner = sBest[0];
    for (int t = 1; t < threads; ++t) {
      if (deviceIsBetter(sBest[t], winner, config)) {
        winner = sBest[t];
      }
    }
    blockBest[feature * numTiles + tile] = winner;
  }
}

// D: reduce each feature's per-tile winners into a single best split.
__global__ void reduceTilesKernel(const GpuSplitCandidate *blockBest,
                                  int numTiles, GpuTrainConfig config,
                                  GpuSplitCandidate *out) {
  const std::size_t feature = blockIdx.x;
  GpuSplitCandidate winner = blockBest[feature * numTiles];
  for (int t = 1; t < numTiles; ++t) {
    const GpuSplitCandidate &candidate = blockBest[feature * numTiles + t];
    if (deviceIsBetter(candidate, winner, config)) {
      winner = candidate;
    }
  }
  out[feature] = winner;
}

// Maps segment index s -> s * nodeRows, used to describe the per-feature
// segment boundaries for the segmented sort without allocating an array.
struct SegmentOffset {
  std::size_t nodeRows;
  __host__ __device__ int operator()(int segment) const {
    return segment * static_cast<int>(nodeRows);
  }
};

} // namespace

// Persistent + reusable device buffers for one fit().
struct TreeCuda::CudaState {
  // Whole-dataset, uploaded once.
  GpuValue *d_features = nullptr;      // feature-major: [feature][row]
  std::uint16_t *d_classIds = nullptr; // class id per row
  std::size_t totalRows = 0;
  std::size_t featureCount = 0;
  std::uint16_t numClasses = 0;

  // Per-node scratch, grown on demand and reused across nodes.
  std::uint32_t *d_currentRows = nullptr; // this node's row indices
  GpuValue *d_values = nullptr;           // gathered values (sort keys, in)
  std::uint32_t *d_rowIds = nullptr;      // gathered row ids (sort values, in)
  GpuValue *d_valuesSorted = nullptr;     // sorted values (out)
  std::uint32_t *d_rowIdsSorted = nullptr;// sorted row ids (out)
  void *d_temp = nullptr;                 // CUB segmented-sort temp storage
  std::size_t tempCapacity = 0;
  GpuSplitCandidate *d_candidates = nullptr; // best split per feature

  // Small fixed-size buffers for the multi-block scan (large nodes only).
  std::uint32_t *d_tileHist = nullptr;       // [feature][tile][class]
  std::uint32_t *d_featureTotals = nullptr;  // [feature][class]
  GpuSplitCandidate *d_blockBest = nullptr;  // [feature][tile]

  int maxTilesPerFeature = 128; // from TrainingOptions::cudaMaxTilesPerFeature
  std::size_t scratchCapacity = 0; // current featureCount*nodeRows capacity
};

void TreeCuda::releaseCudaState() noexcept {
  if (!cuda_) {
    return;
  }
  auto freeDevice = [](auto *&pointer) {
    if (pointer) {
      cudaFree(pointer);
      pointer = nullptr;
    }
  };
  freeDevice(cuda_->d_features);
  freeDevice(cuda_->d_classIds);
  freeDevice(cuda_->d_currentRows);
  freeDevice(cuda_->d_values);
  freeDevice(cuda_->d_rowIds);
  freeDevice(cuda_->d_valuesSorted);
  freeDevice(cuda_->d_rowIdsSorted);
  freeDevice(cuda_->d_temp);
  freeDevice(cuda_->d_candidates);
  freeDevice(cuda_->d_tileHist);
  freeDevice(cuda_->d_featureTotals);
  freeDevice(cuda_->d_blockBest);
  delete cuda_;
  cuda_ = nullptr;
}

TreeCuda::~TreeCuda() { releaseCudaState(); }

void TreeCuda::fit(const Dataset &dataset, const TrainingOptions &options) {
  prepareFit(dataset, options);
  validateCudaLaunchOptions(options_);

  cuda_ = new CudaState();
  CudaState &gpu = *cuda_;
  gpu.totalRows = dataset.samples.size();
  gpu.featureCount = dataset.featureNames.size();
  gpu.numClasses = numClasses_;
  gpu.maxTilesPerFeature = options_.cudaMaxTilesPerFeature;
  if (gpu.numClasses > kMaxGpuClasses) {
    throw std::runtime_error("TreeCuda supports at most 64 classes.");
  }

  // Upload the dataset once in feature-major layout (column f, row r).
  std::vector<GpuValue> hostFeatures(gpu.featureCount * gpu.totalRows);
  std::vector<std::uint16_t> hostClassIds(gpu.totalRows);
  for (std::size_t row = 0; row < gpu.totalRows; ++row) {
    hostClassIds[row] = classIdForRow(row);
    for (std::size_t f = 0; f < gpu.featureCount; ++f) {
      hostFeatures[f * gpu.totalRows + row] =
          static_cast<GpuValue>(dataset.samples[row].features[f]);
    }
  }

  CUDA_CHECK(cudaMalloc(&gpu.d_features, hostFeatures.size() * sizeof(GpuValue)));
  CUDA_CHECK(cudaMalloc(&gpu.d_classIds,
                        hostClassIds.size() * sizeof(std::uint16_t)));
  CUDA_CHECK(cudaMemcpy(gpu.d_features, hostFeatures.data(),
                        hostFeatures.size() * sizeof(GpuValue),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gpu.d_classIds, hostClassIds.data(),
                        hostClassIds.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice));

  // One small buffer of per-feature results, reused by every node.
  CUDA_CHECK(cudaMalloc(&gpu.d_candidates,
                        gpu.featureCount * sizeof(GpuSplitCandidate)));

  // Small fixed buffers for the multi-block (large-node) scan path.
  CUDA_CHECK(cudaMalloc(&gpu.d_tileHist,
                        gpu.featureCount * gpu.maxTilesPerFeature *
                            gpu.numClasses * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu.d_featureTotals,
                        gpu.featureCount * gpu.numClasses *
                            sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu.d_blockBest, gpu.featureCount * gpu.maxTilesPerFeature *
                                              sizeof(GpuSplitCandidate)));

  const std::vector<std::size_t> rowIndices = makeRootRowIndices();
  const BuildTimePoint buildStart = startBuildTimer();
  root_ = buildNode(rowIndices, 0); // CPU recursion; GPU answers each node.
  finishBuildTimer(buildStart);

  finalizeFit(options);
  releaseCudaState();
}

// Grow the reusable per-node scratch if this node needs more room. Buffers only
// ever grow, so deep trees with many small nodes never re-allocate.
void TreeCuda::ensureNodeScratch(std::size_t nodeRows) const {
  CudaState &gpu = *cuda_;
  const std::size_t needed = gpu.featureCount * nodeRows;
  if (needed <= gpu.scratchCapacity && gpu.d_currentRows != nullptr) {
    return;
  }

  auto freeIf = [](auto *&pointer) {
    if (pointer) {
      cudaFree(pointer);
      pointer = nullptr;
    }
  };
  freeIf(gpu.d_currentRows);
  freeIf(gpu.d_values);
  freeIf(gpu.d_rowIds);
  freeIf(gpu.d_valuesSorted);
  freeIf(gpu.d_rowIdsSorted);

  gpu.scratchCapacity = needed;
  CUDA_CHECK(cudaMalloc(&gpu.d_currentRows, nodeRows * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu.d_values, needed * sizeof(GpuValue)));
  CUDA_CHECK(cudaMalloc(&gpu.d_rowIds, needed * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu.d_valuesSorted, needed * sizeof(GpuValue)));
  CUDA_CHECK(cudaMalloc(&gpu.d_rowIdsSorted, needed * sizeof(std::uint32_t)));
}

TreeCuda::SplitSearchResult TreeCuda::findBestSplitAtNode(
    const std::vector<std::size_t> &rowIndices) const {
  if (!cuda_) {
    throw std::runtime_error("TreeCuda GPU state is not initialized.");
  }
  const std::size_t featureCount = dataset_->featureNames.size();
  if (featureCount == 0) {
    return {};
  }

  CudaState &gpu = *cuda_;
  const std::size_t nodeRows = rowIndices.size();
  ensureNodeScratch(nodeRows);

  // Row indices fit in 32 bits (datasets have far fewer than 4B rows); the
  // narrower type halves this transfer and buffer.
  std::vector<std::uint32_t> hostRows(nodeRows);
  for (std::size_t i = 0; i < nodeRows; ++i) {
    hostRows[i] = static_cast<std::uint32_t>(rowIndices[i]);
  }
  CUDA_CHECK(cudaMemcpy(gpu.d_currentRows, hostRows.data(),
                        nodeRows * sizeof(std::uint32_t), cudaMemcpyHostToDevice));

  // 1) Gather this node's (value, rowId) pairs into dense, feature-major scratch.
  const std::size_t workItems = featureCount * nodeRows;
  const int blockSize = options_.cudaGatherBlockSize;
  const int gridSize = static_cast<int>((workItems + blockSize - 1) / blockSize);
  gatherNodeFeaturesKernel<<<gridSize, blockSize>>>(
      gpu.d_currentRows, gpu.d_features, nodeRows, gpu.totalRows, featureCount,
      gpu.d_values, gpu.d_rowIds);
  CUDA_CHECK(cudaGetLastError());

  // 2) Sort every feature's slice by value with ONE segmented radix sort.
  //    Each feature is a segment of length nodeRows; the offset iterator gives
  //    each segment's [begin, end) without us allocating an offsets array.
  //
  //    DoubleBuffer lets CUB ping-pong between the two buffers we already own,
  //    which needs far less temporary scratch than the plain SortPairs form.
  //    The sorted data ends up in whichever buffer .Current() points to.
  auto beginOffsets = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int>(0), SegmentOffset{nodeRows});
  auto endOffsets = beginOffsets + 1;
  const int numItems = static_cast<int>(workItems);
  const int numSegments = static_cast<int>(featureCount);

  cub::DoubleBuffer<GpuValue> keys(gpu.d_values, gpu.d_valuesSorted);
  cub::DoubleBuffer<std::uint32_t> vals(gpu.d_rowIds, gpu.d_rowIdsSorted);

  // First call (temp = null) only reports how much scratch CUB needs.
  std::size_t tempBytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(nullptr, tempBytes, keys, vals,
                                           numItems, numSegments, beginOffsets,
                                           endOffsets);
  if (tempBytes > gpu.tempCapacity) {
    if (gpu.d_temp) {
      cudaFree(gpu.d_temp);
    }
    CUDA_CHECK(cudaMalloc(&gpu.d_temp, tempBytes));
    gpu.tempCapacity = tempBytes;
  }
  cub::DeviceSegmentedRadixSort::SortPairs(gpu.d_temp, tempBytes, keys, vals,
                                           numItems, numSegments, beginOffsets,
                                           endOffsets);
  CUDA_CHECK(cudaGetLastError());

  // Sorted results live in the "current" side of each double buffer.
  GpuValue *valuesSorted = keys.Current();
  std::uint32_t *rowIdsSorted = vals.Current();

  // 3) Evaluate all features' thresholds in parallel (one thread per feature).
  GpuTrainConfig config;
  config.impurityMeasure =
      options_.impurityMeasure == ImpurityMeasure::Gini ? 1 : 0;
  config.splitSelectionMode =
      options_.splitSelectionMode == SplitSelectionMode::MaxGain ? 1 : 0;
  config.epsilon = options_.epsilon;
  config.minSamplesPerLeaf = options_.minSamplesPerLeaf;
  config.numClasses = numClasses_;

  // Decide how many tiles (blocks) per feature. Small nodes use one block per
  // feature (numTiles == 1); large nodes are split into tiles so all SMs stay
  // busy. ~32k rows per tile keeps each block well-fed.
  const std::size_t rowsPerTile = options_.cudaRowsPerTile;
  const int maxTiles = cuda_->maxTilesPerFeature;
  int numTiles = static_cast<int>((nodeRows + rowsPerTile - 1) / rowsPerTile);
  if (numTiles < 1) numTiles = 1;
  if (numTiles > maxTiles) numTiles = maxTiles;

  // Threads per block, capped so each thread still gets a non-trivial chunk.
  int scoreThreads = options_.cudaScoreThreadsPerBlock;
  const std::size_t perBlockRows =
      (nodeRows + static_cast<std::size_t>(numTiles) - 1) / numTiles;
  while (scoreThreads > 1 &&
         static_cast<std::size_t>(scoreThreads) > perBlockRows) {
    scoreThreads /= 2;
  }
  const std::size_t scanShared =
      static_cast<std::size_t>(scoreThreads) * sizeof(GpuSplitCandidate) +
      static_cast<std::size_t>(scoreThreads) * numClasses_ *
          sizeof(std::uint32_t);

  if (numTiles == 1) {
    // Single block per feature: self-contained (histogram + prefix + sweep).
    const std::size_t sharedBytes =
        scanShared + numClasses_ * sizeof(std::uint32_t);
    scoreSplitsKernel<<<static_cast<int>(featureCount), scoreThreads,
                        sharedBytes>>>(valuesSorted, rowIdsSorted,
                                       gpu.d_classIds, nodeRows, config,
                                       gpu.d_candidates);
    CUDA_CHECK(cudaGetLastError());
  } else {
    // Multi-block tiled path for large nodes (A: histogram, B: prefix,
    // C: per-tile sweep, D: per-feature reduce).
    const std::size_t tileSize =
        (nodeRows + static_cast<std::size_t>(numTiles) - 1) / numTiles;
    const dim3 grid(static_cast<unsigned>(numTiles),
                    static_cast<unsigned>(featureCount));

    tileHistogramKernel<<<grid, 256, numClasses_ * sizeof(std::uint32_t)>>>(
        rowIdsSorted, gpu.d_classIds, nodeRows, numTiles, tileSize,
        numClasses_, gpu.d_tileHist);
    CUDA_CHECK(cudaGetLastError());

    const int prefixThreads = numClasses_ < 32 ? 32 : numClasses_;
    tilePrefixKernel<<<static_cast<int>(featureCount), prefixThreads>>>(
        gpu.d_tileHist, numTiles, numClasses_, gpu.d_featureTotals);
    CUDA_CHECK(cudaGetLastError());

    tileScanKernel<<<grid, scoreThreads, scanShared>>>(
        valuesSorted, rowIdsSorted, gpu.d_classIds, nodeRows, numTiles,
        tileSize, config, gpu.d_tileHist, gpu.d_featureTotals,
        gpu.d_blockBest);
    CUDA_CHECK(cudaGetLastError());

    reduceTilesKernel<<<static_cast<int>(featureCount), 1>>>(
        gpu.d_blockBest, numTiles, config, gpu.d_candidates);
    CUDA_CHECK(cudaGetLastError());
  }

  // 4) One copy of the per-feature winners back to the host (also syncs the GPU).
  std::vector<GpuSplitCandidate> hostCandidates(featureCount);
  CUDA_CHECK(cudaMemcpy(hostCandidates.data(), gpu.d_candidates,
                        featureCount * sizeof(GpuSplitCandidate),
                        cudaMemcpyDeviceToHost));

  // Pick the best feature overall using the shared CPU tie-breaking rules.
  std::vector<SplitResult> featureBestCandidates;
  featureBestCandidates.reserve(featureCount);
  for (const GpuSplitCandidate &candidate : hostCandidates) {
    if (!candidate.valid) {
      continue;
    }
    SplitResult split;
    split.valid = true;
    split.featureIndex = candidate.featureIndex;
    split.featureName = dataset_->featureNames[candidate.featureIndex];
    split.threshold = candidate.threshold;
    split.informationGain = candidate.informationGain;
    split.splitInformation = candidate.splitInformation;
    split.gainRatio = candidate.gainRatio;
    split.leftRowCount = candidate.leftRowCount;
    featureBestCandidates.push_back(split);
  }

  SplitSearchResult bestSearch;
  bestSearch.split = chooseBestSplit(featureBestCandidates);
  if (!bestSearch.split.valid) {
    return bestSearch;
  }

  // Hand back the winning feature's sorted row ids so the CPU can partition the
  // node without re-scanning (matches TreeBase::partitionFromSortedView).
  //
  // partitionFromSortedView only reads SortedFeatureRow::rowIndex, so we copy
  // *only* the sorted row ids — not the values, and we do not look up class ids
  // (those host-side map lookups, repeated for every row of every node, used to
  // dominate the runtime).
  const std::size_t winningFeature = bestSearch.split.featureIndex;
  const std::size_t sliceOffset = winningFeature * nodeRows;

  bestSearch.view.featureIndex = winningFeature;
  bestSearch.view.totalRows = nodeRows;
  bestSearch.view.rows.resize(nodeRows);

  std::vector<std::uint32_t> hostRowIds(nodeRows);
  CUDA_CHECK(cudaMemcpy(hostRowIds.data(), rowIdsSorted + sliceOffset,
                        nodeRows * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

  for (std::size_t index = 0; index < nodeRows; ++index) {
    bestSearch.view.rows[index].rowIndex = hostRowIds[index];
  }
  bestSearch.hasView = true;
  return bestSearch;
}
