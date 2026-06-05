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

void validateCudaLaunchOptions(Options &options) {
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

// Same type as Sample::features in the dataset. Keeps host packing and GPU
// storage aligned; thresholds and impurity scores still use double for math.
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

// Sweep a sorted range and score every useful threshold in it.
//
// `leftCounts` must already contain the class histogram of all rows before
// `begin`. As the loop moves forward, row i is added to the left side, so the
// split tested at position i means:
//
//   rows [0, i)       go left
//   rows [i, nodeRows) go right
//
// The caller decides how the sorted feature is divided among threads/tiles; this
// helper only contains the decision-tree math. Keeping it shared makes the
// single-block and tiled kernels much easier to compare.
__device__ __forceinline__ GpuSplitCandidate scanSortedThresholds(
    std::size_t featureIndex, const GpuValue *values,
    const std::uint32_t *rowIds, const std::uint16_t *classIds,
    std::size_t nodeRows, std::size_t begin, std::size_t end,
    const std::uint32_t *totalCounts, double parentImpurity,
    std::uint32_t *leftCounts, const GpuTrainConfig &config) {
  GpuSplitCandidate best; // invalid until a real split is found.

  for (std::size_t i = begin; i < end; ++i) {
    if (i >= 1) {
      const std::uint16_t previousClass = classIds[rowIds[i - 1]];
      const std::uint16_t currentClass = classIds[rowIds[i]];
      const double previousValue = static_cast<double>(values[i - 1]);
      const double currentValue = static_cast<double>(values[i]);

      // Only gaps between different values can create a legal threshold.
      // If the adjacent classes are equal, moving the boundary here cannot
      // improve purity, so skip that work too.
      if (currentClass != previousClass &&
          fabs(previousValue - currentValue) >= config.epsilon) {
        std::uint32_t rightCounts[kMaxGpuClasses];
        for (std::uint16_t k = 0; k < config.numClasses; ++k) {
          rightCounts[k] = totalCounts[k] - leftCounts[k];
        }

        const double threshold = (previousValue + currentValue) / 2.0;
        const GpuSplitCandidate candidate = deviceScoreCandidate(
            featureIndex, nodeRows, parentImpurity, leftCounts, rightCounts, i,
            threshold, config);
        if (candidate.valid && deviceIsBetter(candidate, best, config)) {
          best = candidate;
        }
      }
    }

    // After testing the boundary before row i, row i moves to the left child.
    leftCounts[classIds[rowIds[i]]]++;
  }

  return best;
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

  const GpuSplitCandidate best =
      scanSortedThresholds(feature, v, ids, classIds, nodeRows, chunkBegin,
                           chunkEnd, sTotal, sParentImpurity, left, config);

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

  const GpuSplitCandidate best =
      scanSortedThresholds(feature, v, ids, classIds, nodeRows, chunkBegin,
                           chunkEnd, sTotal, sParentImpurity, left, config);

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

GpuTrainConfig makeGpuTrainConfig(const Options &options,
                                  std::uint16_t numClasses) {
  GpuTrainConfig config;
  config.impurityMeasure =
      options.impurityMeasure == ImpurityMeasure::Gini ? 1 : 0;
  config.splitSelectionMode =
      options.splitSelectionMode == SplitSelectionMode::MaxGain ? 1 : 0;
  config.epsilon = options.epsilon;
  config.minSamplesPerLeaf = options.minSamplesPerLeaf;
  config.numClasses = numClasses;
  return config;
}

struct SplitScanLaunch {
  int numTiles = 1;
  std::size_t tileSize = 0;
  int threadsPerBlock = 1;
  std::size_t sharedBytesPerScanBlock = 0;
  std::size_t sharedBytesForSingleBlock = 0;
};

SplitScanLaunch makeSplitScanLaunch(std::size_t nodeRows,
                                     std::uint16_t numClasses,
                                     const Options &options,
                                     int maxTilesPerFeature) {
  SplitScanLaunch launch;

  // Small nodes use one block per feature. Large nodes are split into tiles so
  // more SMs can work on the same feature at once.
  launch.numTiles =
      static_cast<int>((nodeRows + options.cudaRowsPerTile - 1) /
                       options.cudaRowsPerTile);
  launch.numTiles = std::clamp(launch.numTiles, 1, maxTilesPerFeature);
  launch.tileSize =
      (nodeRows + static_cast<std::size_t>(launch.numTiles) - 1) /
      static_cast<std::size_t>(launch.numTiles);

  // Do not launch more threads than useful rows in a tile; otherwise many
  // threads would own empty chunks and only add reduction overhead.
  launch.threadsPerBlock = options.cudaScoreThreadsPerBlock;
  while (launch.threadsPerBlock > 1 &&
         static_cast<std::size_t>(launch.threadsPerBlock) > launch.tileSize) {
    launch.threadsPerBlock /= 2;
  }

  launch.sharedBytesPerScanBlock =
      static_cast<std::size_t>(launch.threadsPerBlock) *
      (sizeof(GpuSplitCandidate) +
       static_cast<std::size_t>(numClasses) * sizeof(std::uint32_t));
  launch.sharedBytesForSingleBlock =
      launch.sharedBytesPerScanBlock +
      static_cast<std::size_t>(numClasses) * sizeof(std::uint32_t);
  return launch;
}

} // namespace

// CPU-side bookkeeping for one training run. The struct itself lives in host RAM;
// its pointer fields (d_*) point at buffers allocated on the GPU with cudaMalloc.
struct TreeCuda::CudaState {
  // Whole-dataset, uploaded once.
  GpuValue *d_features = nullptr;      // feature-major: [feature][row]
  std::uint16_t *d_classIds = nullptr; // class id per row
  std::size_t totalRows = 0;
  std::size_t featureCount = 0;
  std::uint16_t numClasses = 0;

  // Per-node scratch, pre-allocated at root size and reused across nodes.
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

  int maxTilesPerFeature = 128; // from Options::cudaMaxTilesPerFeature
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

void TreeCuda::fit(const Dataset &dataset, const Options &options) {
  prepareFit(dataset, options);
  validateCudaLaunchOptions(options_);

  // Create GPU session state for this fit(). cuda_ is owned by the CPU; it only
  // holds pointers to device memory, freed in releaseCudaState() below.
  cuda_ = new CudaState();
  CudaState &gpu = *cuda_;
  gpu.totalRows = dataset.samples.size();
  gpu.featureCount = dataset.featureNames.size();
  gpu.numClasses = numClasses_;
  gpu.maxTilesPerFeature = options_.cudaMaxTilesPerFeature;
  if (gpu.numClasses > kMaxGpuClasses) {
    throw std::runtime_error("TreeCuda supports at most 64 classes.");
  }

  // --- Stage A: pack the full dataset on the host, then upload once to the GPU ---
  //
  // hostFeatures: every feature value of every row (feature-major: all of feature 0,
  // then all of feature 1, ...). For large datasets this vector is huge, but we
  // pay the copy cost only once; each tree node later sends just a list of row ids.
  std::vector<GpuValue> hostFeatures(gpu.featureCount * gpu.totalRows);
  // hostClassIds: numeric class id per row. Labels in the CSV are strings; we map
  // them to small integers (0, 1, 2, ...) in buildClassMapping() so counting and
  // impurity math on the GPU use cheap numbers instead of string compares.
  std::vector<std::uint16_t> hostClassIds(gpu.totalRows);
  for (std::size_t row = 0; row < gpu.totalRows; ++row) {
    hostClassIds[row] = classIdForRow(row);
    for (std::size_t f = 0; f < gpu.featureCount; ++f) {
      // Reorder row-major samples into feature-major GPU layout.
      hostFeatures[f * gpu.totalRows + row] = dataset.samples[row].features[f];
    }
  }

  // cudaMalloc allocates memory on the device (GPU VRAM). cudaMemcpy copies our
  // host staging vectors into those device buffers (HostToDevice = CPU -> GPU).
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

  // Per node scratch space.
  const std::size_t scratchItems = gpu.featureCount * gpu.totalRows;
  CUDA_CHECK(cudaMalloc(&gpu.d_currentRows,
                        gpu.totalRows * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu.d_values, scratchItems * sizeof(GpuValue)));
  CUDA_CHECK(cudaMalloc(&gpu.d_rowIds, scratchItems * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu.d_valuesSorted, scratchItems * sizeof(GpuValue)));
  CUDA_CHECK(cudaMalloc(&gpu.d_rowIdsSorted,
                        scratchItems * sizeof(std::uint32_t)));

  const std::vector<std::size_t> rowIndices = makeRootRowIndices();
  const BuildTimePoint buildStart = startBuildTimer();
  root_ = buildNode(rowIndices, 0); // CPU recursion; GPU answers each node.
  finishBuildTimer(buildStart);

  finalizeFit(options);
  releaseCudaState();
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

  // 3) Evaluate all features' thresholds in parallel. Each CUDA block scans one
  // feature slice (or one tile of a feature slice for large nodes).
  const GpuTrainConfig config = makeGpuTrainConfig(options_, numClasses_);
  const SplitScanLaunch scanLaunch = makeSplitScanLaunch(
      nodeRows, numClasses_, options_, cuda_->maxTilesPerFeature);

  if (scanLaunch.numTiles == 1) {
    // Single block per feature: self-contained (histogram + prefix + sweep).
    scoreSplitsKernel<<<static_cast<int>(featureCount),
                        scanLaunch.threadsPerBlock,
                        scanLaunch.sharedBytesForSingleBlock>>>(
        valuesSorted, rowIdsSorted, gpu.d_classIds, nodeRows, config,
        gpu.d_candidates);
    CUDA_CHECK(cudaGetLastError());
  } else {
    // Multi-block tiled path for large nodes (A: histogram, B: prefix,
    // C: per-tile sweep, D: per-feature reduce).
    const dim3 grid(static_cast<unsigned>(scanLaunch.numTiles),
                    static_cast<unsigned>(featureCount));

    tileHistogramKernel<<<grid, 256, numClasses_ * sizeof(std::uint32_t)>>>(
        rowIdsSorted, gpu.d_classIds, nodeRows, scanLaunch.numTiles,
        scanLaunch.tileSize, numClasses_, gpu.d_tileHist);
    CUDA_CHECK(cudaGetLastError());

    const int prefixThreads = numClasses_ < 32 ? 32 : numClasses_;
    tilePrefixKernel<<<static_cast<int>(featureCount), prefixThreads>>>(
        gpu.d_tileHist, scanLaunch.numTiles, numClasses_, gpu.d_featureTotals);
    CUDA_CHECK(cudaGetLastError());

    tileScanKernel<<<grid, scanLaunch.threadsPerBlock,
                     scanLaunch.sharedBytesPerScanBlock>>>(
        valuesSorted, rowIdsSorted, gpu.d_classIds, nodeRows,
        scanLaunch.numTiles, scanLaunch.tileSize, config, gpu.d_tileHist,
        gpu.d_featureTotals, gpu.d_blockBest);
    CUDA_CHECK(cudaGetLastError());

    reduceTilesKernel<<<static_cast<int>(featureCount), 1>>>(
        gpu.d_blockBest, scanLaunch.numTiles, config, gpu.d_candidates);
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
