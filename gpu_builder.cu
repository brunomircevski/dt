// GPU tree builder (Cuda backend).
//
// The same algorithm as CpuTreeBuilder (presorted columns, one threshold sweep
// per feature, stable partition), reorganised for the GPU:
//
//  * Presort: every feature column is sorted once with CUB radix sort.
//  * Breadth-first: all large nodes of one tree level are processed together
//    by a handful of kernel launches, so even deep levels with hundreds of
//    nodes keep the whole GPU busy. Work is cut into tiles of kTile entries of
//    one (node, feature) column range; every kernel runs one block per tile.
//  * The host knows every node's class counts before the node is processed
//    (from its parent's split), so it decides up front which nodes are leaves;
//    only nodes that search for a split enter a level. Per level:
//      1. tileHistogram   class counts of every tile
//      2. segmentPrefix   per (node, feature): exclusive scan over its tiles,
//                         giving each tile the class counts left of it
//      3. tileEstimate / segmentMax / tileExact
//                         each tile sweeps its cuts: first in single precision,
//                         then exactly (the CPU's double-precision function)
//                         for the cuts that can still be the best
//      4. segmentBest     best cut per (node, feature) and the class counts
//                         left of it                         -> host picks the
//                                                               split (SplitRules)
//                                                               and the children
//      5. markGoesLeft / tileLeftCount / segmentPrefix / scatter:
//                         stable partition of every column of every split node
//                         into the other buffer (ping-pong between two copies),
//                         skipped for nodes whose children are both leaves
//    so each level needs one round trip to the host (step 4).
//  * Children with fewer than gpuMinRows rows are gathered into one block,
//    copied to the host in one transfer and grown by CpuTreeBuilder tasks on
//    the thread pool while the GPU continues with the next level.

#include "gpu_builder.h"

#include "cpu_builder.h"
#include "split_math.h"
#include "timing.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    const cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                      \
      throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +           \
                               std::to_string(__LINE__) + ": " +                      \
                               cudaGetErrorString(status));                           \
    }                                                                                 \
  } while (0)

using dt::CutCandidate;
using dt::Entry;
using dt::EntryCodec;

constexpr int kThreads = 256;
constexpr int kItems = 8; // entries per thread in a tile
constexpr std::uint32_t kTile = kThreads * kItems;
constexpr int kMaxGpuClasses = 64;

// One (node, feature) column range.
struct Segment {
  unsigned long long base; // offset of the node's first entry in this column
  std::uint32_t count;     // rows of the node
  std::uint32_t node;      // index into the level's node table
  std::uint32_t feature;
  std::uint32_t firstTile;
  std::uint32_t tileCount;
};

// Per node of the current level (every one of them searches for a split).
struct NodeDesc {
  double parentWeighted = 0.0; // weightedImpurity of the node
  std::uint32_t count = 0;
  std::uint32_t minChild = 0;
  std::int32_t splitFeature = -1; // chosen split (-1: none)
  std::uint32_t leftCount = 0;
  std::int32_t partition = 0;     // split, and not both children are leaves
  float filterMargin = 0.0f; // see floatGainMargin()
};

struct ScanParams {
  EntryCodec codec;
  int classCount;
  int criterion;
  double minGap;
  dt::LogTable logs;        // device copy of the host table
  const float *floatLogs;   // the same table in single precision
};

// Single-precision estimate of dt::cutGain, used only to rule out cuts that
// cannot be the best one (consumer GPUs run double precision ~64x slower).
__device__ __forceinline__ float floatXlog2x(const float *table, std::uint32_t count) {
  return count < dt::kLogTableSize ? table[count]
                                   : static_cast<float>(count) * log2f(static_cast<float>(count));
}

template <int MAXK>
__device__ __forceinline__ float floatCutGain(const std::uint32_t *total,
                                              const std::uint32_t *left, int classCount,
                                              std::uint32_t n, std::uint32_t nLeft,
                                              float parentWeighted, int criterion,
                                              const float *table) {
  const std::uint32_t nRight = n - nLeft;
  float leftWeighted;
  float rightWeighted;
  if (criterion == dt::kGini) {
    float leftSquares = 0.0f;
    float rightSquares = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      if (k < classCount) {
        const float l = static_cast<float>(left[k]);
        const float r = static_cast<float>(total[k] - left[k]);
        leftSquares += l * l;
        rightSquares += r * r;
      }
    }
    leftWeighted = static_cast<float>(nLeft) - leftSquares / static_cast<float>(nLeft);
    rightWeighted = static_cast<float>(nRight) - rightSquares / static_cast<float>(nRight);
  } else {
    float leftSum = 0.0f;
    float rightSum = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      if (k < classCount) {
        leftSum += floatXlog2x(table, left[k]);
        rightSum += floatXlog2x(table, total[k] - left[k]);
      }
    }
    leftWeighted = floatXlog2x(table, nLeft) - leftSum;
    rightWeighted = floatXlog2x(table, nRight) - rightSum;
  }
  return (parentWeighted - leftWeighted - rightWeighted) / static_cast<float>(n);
}

// A generous bound on |floatCutGain - cutGain| for a node of n rows (a few
// float roundings on each of the ~2K+4 terms, each at most n * log2(n) in
// size for entropy and n for Gini, divided by n). Cuts whose float gain is
// more than 2 * bound + kTieEps below the best float gain of the tile cannot
// be the best cut, or tie with it, so only the others are scored in double.
float floatGainMargin(int classCount, std::uint32_t n, int criterion) {
  const double unit = 1.0 / (1 << 23);
  const double scale = criterion == dt::kGini ? 1.0 : std::max(1.0, std::log2(double(n)));
  const double bound = 4.0 * (2.0 * classCount + 8.0) * unit * scale;
  return static_cast<float>(2.0 * bound + dt::kTieEps);
}

struct CopyJob {
  unsigned long long from;
  unsigned long long to;
  std::uint32_t count;
};

// Best cut found by part of the block, with tie-breaking on position.
struct BestCut {
  double gain;
  std::uint32_t position;
  float leftValue;
  float rightValue;
};

struct FloatMax {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct BetterCut {
  __device__ BestCut operator()(const BestCut &a, const BestCut &b) const {
    if (b.gain == -INFINITY) {
      return a;
    }
    if (a.gain == -INFINITY) {
      return b;
    }
    return dt::isBetterCut(b.gain, b.position, a.gain, a.position) ? b : a;
  }
};

__device__ __forceinline__ void tileRange(const Segment &segment, std::uint32_t tile,
                                          std::uint32_t &start, std::uint32_t &length) {
  start = (tile - segment.firstTile) * kTile;
  length = min(kTile, segment.count - start);
}

// ---------------------------------------------------------------------------
// Presort helpers
// ---------------------------------------------------------------------------
__global__ void preparePresortKernel(float *keys, std::uint32_t *ids,
                                     const std::uint16_t *labels, std::size_t rows,
                                     std::size_t total, EntryCodec codec) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const std::uint32_t row = static_cast<std::uint32_t>(index % rows);
  keys[index] = keys[index] + 0.0f; // -0.0 -> +0.0, as on the CPU
  ids[index] = codec.pack(row, labels[row]);
}

__global__ void interleaveKernel(const float *keys, const std::uint32_t *ids, Entry *entries,
                                 std::size_t total) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    entries[index] = {keys[index], ids[index]};
  }
}

// ---------------------------------------------------------------------------
// 1. Class histogram of every tile
// ---------------------------------------------------------------------------
template <int MAXK>
__global__ void __launch_bounds__(kThreads)
    tileHistogramKernel(const Entry *src, const Segment *segments,
                        const std::uint32_t *tileSegment, ScanParams params,
                        std::uint32_t *tileHist) {
  __shared__ std::uint32_t histogram[MAXK];
  const std::uint32_t tile = blockIdx.x;
  const Segment segment = segments[tileSegment[tile]];
  std::uint32_t start;
  std::uint32_t length;
  tileRange(segment, tile, start, length);
  const Entry *entries = src + segment.base + start;
  const int classCount = params.classCount;

  if (threadIdx.x < MAXK) {
    histogram[threadIdx.x] = 0;
  }
  __syncthreads();

  // Each warp counts 32 entries per step with one ballot per class.
  const int lane = threadIdx.x & 31;
  std::uint32_t counts[MAXK];
#pragma unroll
  for (int k = 0; k < MAXK; ++k) {
    counts[k] = 0;
  }
  for (std::uint32_t base = threadIdx.x - lane; base < length; base += kThreads) {
    const std::uint32_t index = base + lane;
    const std::uint32_t cls =
        index < length ? params.codec.cls(entries[index].packed) : 0xFFFFFFFFu;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      if (k < classCount) {
        counts[k] += __popc(__ballot_sync(0xFFFFFFFFu, cls == static_cast<std::uint32_t>(k)));
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      if (k < classCount && counts[k] > 0) {
        atomicAdd(&histogram[k], counts[k]);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < classCount) {
    tileHist[static_cast<std::size_t>(tile) * classCount + threadIdx.x] = histogram[threadIdx.x];
  }
}

// ---------------------------------------------------------------------------
// 2. / 7. Exclusive scan over the tiles of each segment (one block per
// segment), `width` values per tile. Writes the per-segment totals.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads)
    segmentPrefixKernel(std::uint32_t *perTile, int width, const Segment *segments,
                        const NodeDesc *nodes, bool partitionedOnly,
                        std::uint32_t *segmentTotals) {
  using Scan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename Scan::TempStorage scanStorage;
  const Segment segment = segments[blockIdx.x];
  if (partitionedOnly && !nodes[segment.node].partition) {
    return;
  }
  for (int k = 0; k < width; ++k) {
    std::uint32_t carry = 0;
    for (std::uint32_t chunk = 0; chunk < segment.tileCount; chunk += kThreads) {
      const std::uint32_t tile = chunk + threadIdx.x;
      std::uint32_t *slot =
          perTile + static_cast<std::size_t>(segment.firstTile + tile) * width + k;
      const std::uint32_t value = tile < segment.tileCount ? *slot : 0;
      std::uint32_t prefix;
      std::uint32_t sum;
      Scan(scanStorage).ExclusiveSum(value, prefix, sum);
      if (tile < segment.tileCount) {
        *slot = carry + prefix;
      }
      carry += sum;
      __syncthreads();
    }
    if (segmentTotals && threadIdx.x == 0) {
      segmentTotals[static_cast<std::size_t>(blockIdx.x) * width + k] = carry;
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Sweep every cut of every tile (one block per tile)
// ---------------------------------------------------------------------------
// Cut at segment position p (between entries p-1 and p) sends the first p rows
// left. Warp w of the block sweeps tile entries [w * 256, (w + 1) * 256) in 8
// steps of 32 consecutive entries (coalesced loads). Within a step, lane i
// gets the class counts left of its entry from a ballot over the warp:
//     left[k] = (rows of class k before this step) + popc(ballot(class == k) & lanes below i)
// and its neighbours' values through warp shuffles.
//
// Two passes keep double precision (slow on consumer GPUs) rare:
//   3a. tileEstimate: count the cuts (C4.5's "tries") and estimate every
//       cut's gain in single precision; keep the best estimate per tile.
//   3b. segmentMax:   best estimate per (node, feature).
//   3c. tileExact:    score in double precision (the exact CPU function) only
//       the cuts whose estimate is within the proven float error of that best.
constexpr int kWarps = kThreads / 32;
constexpr int kWarpSteps = kTile / kThreads; // 32-entry steps per warp

template <int MAXK> struct TileSweep {
  std::uint32_t start;
  std::uint32_t length;
  const Entry *column;
  float value[kWarpSteps];
  std::uint32_t cls[kWarpSteps];
  std::uint32_t warpLeft[MAXK]; // class counts before this warp's first entry
};

// Load a tile and work out the class counts before each warp's entries.
// Uses `warpCounts` and `totals` (shared) and one __syncthreads.
template <int MAXK>
__device__ __forceinline__ void loadTile(TileSweep<MAXK> &sweep, const Entry *src,
                                         const Segment &segment, std::uint32_t segmentIndex,
                                         std::uint32_t tile, const ScanParams &params,
                                         const std::uint32_t *tilePrefix,
                                         const std::uint32_t *segmentTotals,
                                         std::uint32_t (*warpCounts)[MAXK],
                                         std::uint32_t *totals) {
  const unsigned full = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int classCount = params.classCount;
  tileRange(segment, tile, sweep.start, sweep.length);
  sweep.column = src + segment.base;
  if (threadIdx.x < classCount) {
    totals[threadIdx.x] =
        segmentTotals[static_cast<std::size_t>(segmentIndex) * classCount + threadIdx.x];
  }
#pragma unroll
  for (int step = 0; step < kWarpSteps; ++step) {
    const std::uint32_t index = warp * (kWarpSteps * 32) + step * 32 + lane;
    if (index < sweep.length) {
      const Entry entry = sweep.column[sweep.start + index];
      sweep.value[step] = entry.value;
      sweep.cls[step] = params.codec.cls(entry.packed);
    } else {
      sweep.value[step] = 0.0f;
      sweep.cls[step] = 0xFFFFFFFFu;
    }
  }
#pragma unroll
  for (int k = 0; k < MAXK; ++k) {
    std::uint32_t mine = 0;
    if (k < classCount) {
#pragma unroll
      for (int step = 0; step < kWarpSteps; ++step) {
        mine += __popc(__ballot_sync(full, sweep.cls[step] == static_cast<std::uint32_t>(k)));
      }
    }
    if (lane == 0) {
      warpCounts[warp][k] = mine;
    }
  }
  __syncthreads();
  const std::uint32_t *prefix = tilePrefix + static_cast<std::size_t>(tile) * classCount;
#pragma unroll
  for (int k = 0; k < MAXK; ++k) {
    std::uint32_t before = k < classCount ? prefix[k] : 0;
    for (int other = 0; other < warp; ++other) {
      before += warpCounts[other][k];
    }
    sweep.warpLeft[k] = before;
  }
}

// Visit every cut of this lane that must be scored: onCut(step, position,
// leftCounts, previousValue, estimate). Returns the number of real cuts.
template <int MAXK, typename OnCut>
__device__ __forceinline__ std::uint32_t sweepTile(const TileSweep<MAXK> &sweep,
                                                   const Segment &segment, const NodeDesc &node,
                                                   const ScanParams &params,
                                                   const std::uint32_t *totals, OnCut &&onCut) {
  const unsigned full = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const unsigned lanesBelow = (1u << lane) - 1u;
  const std::uint32_t count = segment.count;
  const std::uint32_t minChild = node.minChild;
  const float parentWeighted = static_cast<float>(node.parentWeighted);
  const Entry *column = sweep.column;

  std::uint32_t left[MAXK];
#pragma unroll
  for (int k = 0; k < MAXK; ++k) {
    left[k] = sweep.warpLeft[k];
  }
  std::uint32_t tries = 0;
#pragma unroll
  for (int step = 0; step < kWarpSteps; ++step) {
    const std::uint32_t index = warp * (kWarpSteps * 32) + step * 32 + lane;
    const std::uint32_t position = sweep.start + index; // segment-relative
    const bool valid = index < sweep.length;
    const float value = sweep.value[step];

    // Neighbours: previous entry, the one before it, and the next one.
    float previous = __shfl_up_sync(full, value, 1);
    std::uint32_t previousClass = __shfl_up_sync(full, sweep.cls[step], 1);
    float beforePrevious = __shfl_up_sync(full, value, 2);
    float next = __shfl_down_sync(full, value, 1);
    if (valid) {
      if (lane < 1 && position >= 1) {
        const Entry entry = column[position - 1];
        previous = entry.value;
        previousClass = params.codec.cls(entry.packed);
      }
      if (lane < 2 && position >= 2) {
        beforePrevious = column[position - 2].value;
      }
      if ((lane == 31 || index + 1 >= sweep.length) && position + 1 < count) {
        next = column[position + 1].value;
      }
    }

    std::uint32_t leftHere[MAXK];
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      const unsigned votes = __ballot_sync(full, sweep.cls[step] == static_cast<std::uint32_t>(k));
      leftHere[k] = left[k] + __popc(votes & lanesBelow);
      left[k] += __popc(votes);
    }

    if (valid && position >= minChild && position + minChild <= count &&
        dt::isCut(previous, value, params.minGap)) {
      ++tries;
      bool skip = false;
      if (previousClass == sweep.cls[step]) {
        const bool leftSingleton = position < 2 || dt::isCut(beforePrevious, previous, params.minGap);
        const bool rightSingleton = position + 1 >= count || dt::isCut(value, next, params.minGap);
        skip = dt::isSkippableCut(true, leftSingleton, rightSingleton, position, count, minChild);
      }
      if (!skip) {
        const float estimate =
            floatCutGain<MAXK>(totals, leftHere, params.classCount, count, position,
                               parentWeighted, params.criterion, params.floatLogs);
        onCut(step, position, leftHere, previous, estimate);
      }
    }
  }
  return tries;
}

template <int MAXK>
__global__ void __launch_bounds__(kThreads)
    tileEstimateKernel(const Entry *src, const Segment *segments,
                       const std::uint32_t *tileSegment, const NodeDesc *nodes,
                       ScanParams params, const std::uint32_t *tilePrefix,
                       const std::uint32_t *segmentTotals, float *tileMax,
                       std::uint32_t *tileTries) {
  using SumReduce = cub::BlockReduce<std::uint32_t, kThreads>;
  using FloatReduce = cub::BlockReduce<float, kThreads>;
  __shared__ union {
    typename SumReduce::TempStorage sum;
    typename FloatReduce::TempStorage max;
  } storage;
  __shared__ std::uint32_t totals[MAXK];
  __shared__ std::uint32_t warpCounts[kWarps][MAXK];

  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  const NodeDesc node = nodes[segment.node];
  TileSweep<MAXK> sweep;
  loadTile<MAXK>(sweep, src, segment, segmentIndex, tile, params, tilePrefix, segmentTotals,
                 warpCounts, totals);
  float best = -INFINITY;
  const std::uint32_t tries = sweepTile<MAXK>(
      sweep, segment, node, params, totals,
      [&](int, std::uint32_t, const std::uint32_t *, float, float estimate) {
        best = fmaxf(best, estimate);
      });
  const float tileBest = FloatReduce(storage.max).Reduce(best, FloatMax());
  __syncthreads();
  const std::uint32_t tileTotalTries = SumReduce(storage.sum).Sum(tries);
  if (threadIdx.x == 0) {
    tileMax[tile] = tileBest;
    tileTries[tile] = tileTotalTries;
  }
}

__global__ void __launch_bounds__(kThreads)
    segmentMaxKernel(const Segment *segments, const float *tileMax, float *segmentMax) {
  using FloatReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename FloatReduce::TempStorage storage;
  const Segment segment = segments[blockIdx.x];
  float best = -INFINITY;
  for (std::uint32_t tile = threadIdx.x; tile < segment.tileCount; tile += kThreads) {
    best = fmaxf(best, tileMax[segment.firstTile + tile]);
  }
  const float segmentBest = FloatReduce(storage).Reduce(best, FloatMax());
  if (threadIdx.x == 0) {
    segmentMax[blockIdx.x] = segmentBest;
  }
}

template <int MAXK>
__global__ void __launch_bounds__(kThreads)
    tileExactKernel(const Entry *src, const Segment *segments, const std::uint32_t *tileSegment,
                    const NodeDesc *nodes, ScanParams params, const std::uint32_t *tilePrefix,
                    const std::uint32_t *segmentTotals, const float *tileMax,
                    const float *segmentMax, BestCut *tileBest) {
  using CutReduce = cub::BlockReduce<BestCut, kThreads>;
  __shared__ typename CutReduce::TempStorage storage;
  __shared__ std::uint32_t totals[MAXK];
  __shared__ std::uint32_t warpCounts[kWarps][MAXK];

  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  const NodeDesc node = nodes[segment.node];
  const float cutoff = segmentMax[segmentIndex] - node.filterMargin;
  if (tileMax[tile] < cutoff) {
    if (threadIdx.x == 0) {
      tileBest[tile] = {-INFINITY, 0xFFFFFFFFu, 0.0f, 0.0f};
    }
    return; // no cut of this tile can be the best one
  }
  TileSweep<MAXK> sweep;
  loadTile<MAXK>(sweep, src, segment, segmentIndex, tile, params, tilePrefix, segmentTotals,
                 warpCounts, totals);
  BestCut best{-INFINITY, 0xFFFFFFFFu, 0.0f, 0.0f};
  sweepTile<MAXK>(sweep, segment, node, params, totals,
                  [&](int step, std::uint32_t position, const std::uint32_t *left,
                      float previous, float estimate) {
                    if (estimate < cutoff) {
                      return;
                    }
                    const double gain =
                        dt::cutGain(totals, left, params.classCount, segment.count, position,
                                    node.parentWeighted, params.criterion, params.logs);
                    if (best.gain == -INFINITY ||
                        dt::isBetterCut(gain, position, best.gain, best.position)) {
                      best = {gain, position, previous, sweep.value[step]};
                    }
                  });
  const BestCut tileWinner = CutReduce(storage).Reduce(best, BetterCut());
  if (threadIdx.x == 0) {
    tileBest[tile] = tileWinner;
  }
}

// ---------------------------------------------------------------------------
// 4. Best cut per segment (one block per segment), with the class counts left
//    of it (the left child's counts if this feature wins).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads)
    segmentBestKernel(const Entry *src, const Segment *segments, ScanParams params,
                      const std::uint32_t *tilePrefix, const BestCut *tileBest,
                      const std::uint32_t *tileTries, CutCandidate *segmentBest,
                      std::uint32_t *segmentLeft) {
  using CutReduce = cub::BlockReduce<BestCut, kThreads>;
  using SumReduce = cub::BlockReduce<std::uint32_t, kThreads>;
  __shared__ union {
    typename CutReduce::TempStorage cut;
    typename SumReduce::TempStorage sum;
  } storage;
  __shared__ BestCut winner;
  __shared__ std::uint32_t left[kMaxGpuClasses];
  const Segment segment = segments[blockIdx.x];
  const int classCount = params.classCount;
  BestCut best{-INFINITY, 0xFFFFFFFFu, 0.0f, 0.0f};
  std::uint32_t tries = 0;
  for (std::uint32_t tile = threadIdx.x; tile < segment.tileCount; tile += kThreads) {
    best = BetterCut()(best, tileBest[segment.firstTile + tile]);
    tries += tileTries[segment.firstTile + tile];
  }
  const BestCut blockBest = CutReduce(storage.cut).Reduce(best, BetterCut());
  __syncthreads();
  const std::uint32_t totalTries = SumReduce(storage.sum).Sum(tries);
  if (threadIdx.x == 0) {
    winner = blockBest;
    CutCandidate result;
    result.gain = blockBest.gain;
    result.leftCount = blockBest.position;
    result.tries = totalTries;
    result.leftValue = blockBest.leftValue;
    result.rightValue = blockBest.rightValue;
    segmentBest[blockIdx.x] = result;
  }
  if (threadIdx.x < classCount) {
    left[threadIdx.x] = 0;
  }
  __syncthreads();
  // Left of the cut: the counts before its tile plus the tile's entries
  // before the cut (zeros if this feature has no cut).
  const bool found = winner.gain != -INFINITY;
  const std::uint32_t position = found ? winner.position : 0;
  const std::uint32_t tile = position / kTile;
  const Entry *entries = src + segment.base;
  for (std::uint32_t index = tile * kTile + threadIdx.x; index < position; index += kThreads) {
    atomicAdd(&left[params.codec.cls(entries[index].packed)], 1u);
  }
  __syncthreads();
  if (threadIdx.x < classCount) {
    const std::uint32_t before =
        found ? tilePrefix[static_cast<std::size_t>(segment.firstTile + tile) * classCount +
                           threadIdx.x]
              : 0;
    segmentLeft[static_cast<std::size_t>(blockIdx.x) * classCount + threadIdx.x] =
        before + left[threadIdx.x];
  }
}

// ---------------------------------------------------------------------------
// 5. Stable partition of every column of every split node
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads)
    markGoesLeftKernel(const Entry *src, const Segment *segments,
                       const std::uint32_t *tileSegment, const NodeDesc *nodes,
                       EntryCodec codec, std::uint8_t *goesLeft) {
  const std::uint32_t tile = blockIdx.x;
  const Segment segment = segments[tileSegment[tile]];
  const NodeDesc node = nodes[segment.node];
  if (!node.partition || node.splitFeature != static_cast<std::int32_t>(segment.feature)) {
    return;
  }
  std::uint32_t start;
  std::uint32_t length;
  tileRange(segment, tile, start, length);
  const Entry *column = src + segment.base;
  for (std::uint32_t index = threadIdx.x; index < length; index += kThreads) {
    const std::uint32_t position = start + index;
    goesLeft[codec.row(column[position].packed)] = position < node.leftCount;
  }
}

__global__ void __launch_bounds__(kThreads)
    tileLeftCountKernel(const Entry *src, const Segment *segments,
                        const std::uint32_t *tileSegment, const NodeDesc *nodes,
                        EntryCodec codec, const std::uint8_t *goesLeft,
                        std::uint32_t *tileLeft) {
  using SumReduce = cub::BlockReduce<std::uint32_t, kThreads>;
  __shared__ typename SumReduce::TempStorage storage;
  const std::uint32_t tile = blockIdx.x;
  const Segment segment = segments[tileSegment[tile]];
  if (!nodes[segment.node].partition) {
    return;
  }
  std::uint32_t start;
  std::uint32_t length;
  tileRange(segment, tile, start, length);
  const Entry *entries = src + segment.base + start;
  std::uint32_t lefts = 0;
  for (std::uint32_t index = threadIdx.x; index < length; index += kThreads) {
    lefts += goesLeft[codec.row(entries[index].packed)];
  }
  const std::uint32_t total = SumReduce(storage).Sum(lefts);
  if (threadIdx.x == 0) {
    tileLeft[tile] = total;
  }
}

__global__ void __launch_bounds__(kThreads)
    scatterKernel(const Entry *src, Entry *dst, const Segment *segments,
                  const std::uint32_t *tileSegment, const NodeDesc *nodes, EntryCodec codec,
                  const std::uint8_t *goesLeft, const std::uint32_t *tileLeftPrefix) {
  using Scan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename Scan::TempStorage storage;
  const std::uint32_t tile = blockIdx.x;
  const Segment segment = segments[tileSegment[tile]];
  const NodeDesc node = nodes[segment.node];
  if (!node.partition) {
    return;
  }
  std::uint32_t start;
  std::uint32_t length;
  tileRange(segment, tile, start, length);
  const Entry *from = src + segment.base;
  Entry *to = dst + segment.base;

  const std::uint32_t first = threadIdx.x * kItems;
  Entry items[kItems];
  bool left[kItems];
  std::uint32_t lefts = 0;
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    left[j] = false;
    if (first + j < length) {
      items[j] = from[start + first + j];
      left[j] = goesLeft[codec.row(items[j].packed)];
      lefts += left[j];
    }
  }
  std::uint32_t leftsBefore;
  Scan(storage).ExclusiveSum(lefts, leftsBefore);
  leftsBefore += tileLeftPrefix[tile]; // left rows before my first entry (segment)
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    if (first + j < length) {
      const std::uint32_t position = start + first + j;
      if (left[j]) {
        to[leftsBefore++] = items[j];
      } else {
        to[node.leftCount + (position - leftsBefore)] = items[j];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 6. Gather the columns of nodes handed to the CPU into one block
// ---------------------------------------------------------------------------
__global__ void copyKernel(const Entry *src, Entry *dst, const CopyJob *jobs) {
  const CopyJob job = jobs[blockIdx.x];
  for (std::uint32_t index = threadIdx.x; index < job.count; index += blockDim.x) {
    dst[job.to + index] = src[job.from + index];
  }
}

template <typename T> class DeviceArray {
public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  ~DeviceArray() {
    if (data_) {
      cudaFree(data_);
    }
  }

  void reserve(std::size_t count) {
    if (count <= capacity_) {
      return;
    }
    if (data_) {
      CUDA_CHECK(cudaFree(data_));
      data_ = nullptr;
    }
    capacity_ = std::max(count, capacity_ + capacity_ / 2);
    CUDA_CHECK(cudaMalloc(&data_, capacity_ * sizeof(T)));
  }
  T *get() const { return data_; }

private:
  T *data_ = nullptr;
  std::size_t capacity_ = 0;
};

// Optional per-kernel timing (DT_GPU_VERBOSE=1): CUDA events around launches.
class KernelProfile {
public:
  explicit KernelProfile(bool enabled) : enabled_(enabled) {}
  ~KernelProfile() {
    for (auto &entry : pending_) {
      cudaEventDestroy(entry.start);
      cudaEventDestroy(entry.stop);
    }
  }
  bool enabled() const { return enabled_; }

  template <typename Launch> void run(const char *name, cudaStream_t stream, Launch &&launch) {
    if (!enabled_) {
      launch();
      return;
    }
    Pending entry{name, nullptr, nullptr};
    cudaEventCreate(&entry.start);
    cudaEventCreate(&entry.stop);
    cudaEventRecord(entry.start, stream);
    launch();
    cudaEventRecord(entry.stop, stream);
    pending_.push_back(entry);
  }

  void report() {
    for (Pending &entry : pending_) {
      float ms = 0.0f;
      cudaEventSynchronize(entry.stop);
      cudaEventElapsedTime(&ms, entry.start, entry.stop);
      cudaEventDestroy(entry.start);
      cudaEventDestroy(entry.stop);
      auto found = std::find_if(totals_.begin(), totals_.end(),
                                [&](const auto &total) { return total.first == entry.name; });
      if (found == totals_.end()) {
        totals_.push_back({entry.name, ms});
      } else {
        found->second += ms;
      }
    }
    pending_.clear();
    for (const auto &total : totals_) {
      std::fprintf(stderr, "  %-16s %9.2f ms\n", total.first.c_str(), total.second);
    }
  }

private:
  struct Pending {
    std::string name;
    cudaEvent_t start;
    cudaEvent_t stop;
  };
  bool enabled_;
  std::vector<Pending> pending_;
  std::vector<std::pair<std::string, float>> totals_;
};

// One node of the current GPU level, on the host side.
struct LevelNode {
  Node *node;
  std::uint32_t begin; // range in the current buffer (all columns)
  std::uint32_t count;
  int depth;
};

// A child handed to the CPU: CpuTreeBuilder grows it into parent->left/right.
struct HandOff {
  Node *parent;
  bool left;
  std::uint32_t begin;
  std::uint32_t count;
  int depth;
};

class GpuTreeBuilder {
public:
  GpuTreeBuilder(const Dataset &dataset, const SplitRules &rules, const EntryCodec &codec,
                 const Options &options, ThreadPool &pool,
                 std::vector<std::vector<float>> *distinctValues)
      : dataset_(dataset), rules_(rules), codec_(codec), options_(options), pool_(pool),
        distinctValues_(distinctValues),
        rows_(dataset.rowCount), features_(dataset.featureCount()),
        classCount_(static_cast<int>(dataset.classCount())),
        hostGoesLeft_(new std::uint8_t[dataset.rowCount]),
        cpuBuilder_(rules, codec, dataset.featureCount(), hostGoesLeft_.get(), &pool, options) {
    if (classCount_ > kMaxGpuClasses) {
      throw std::runtime_error("The Cuda backend supports at most 64 classes.");
    }
    params_.codec = codec;
    params_.classCount = classCount_;
    params_.criterion = rules.criterion();
    params_.minGap = rules.minValueGap();
  }

  ~GpuTreeBuilder() {
    if (arenaPinning_.valid()) {
      arenaPinning_.wait();
    }
    if (arenaPinned_) {
      cudaHostUnregister(arena_.get());
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  std::unique_ptr<Node> run(GpuTimings &timings) {
    {
      ScopedTimer timer(timings.setupSeconds);
      setup();
    }
    std::unique_ptr<Node> root;
    {
      ScopedTimer timer(timings.buildSeconds);
      grow(root);
      double waitSeconds = 0.0;
      {
        ScopedTimer timer(waitSeconds);
        pool_.waitIdle();
      }
      if (profile_.enabled()) {
        std::fprintf(stderr, "  waiting for CPU subtrees: %.2f ms\n", waitSeconds * 1000.0);
        profile_.report();
      }
    }
    return root;
  }

private:
  void setup() {
    double contextSeconds = 0.0;
    {
      ScopedTimer timer(contextSeconds);
      CUDA_CHECK(cudaFree(nullptr)); // creates the CUDA context if needed
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    const auto setupStart = std::chrono::steady_clock::now();
    auto phase = [&](const char *name) {
      if (profile_.enabled()) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        std::fprintf(stderr, "  setup %-12s %8.2f ms\n", name,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - setupStart)
                         .count());
      }
    };
    if (profile_.enabled()) {
      std::fprintf(stderr, "  setup context      %8.2f ms\n", contextSeconds * 1e3);
    }
    const std::size_t total = features_ * rows_;
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const std::size_t needed = 2 * total * sizeof(Entry) + rows_ * 3 + (64u << 20);
    if (needed > freeBytes) {
      throw std::runtime_error("Not enough GPU memory: need about " +
                               std::to_string(needed >> 20) + " MiB, " +
                               std::to_string(freeBytes >> 20) +
                               " MiB free. Use --parallel instead.");
    }
    entries_[0].reserve(total);
    entries_[1].reserve(total);
    goesLeft_.reserve(rows_);

    // Presort. Buffer 1 holds keys/ids before sorting, buffer 0 after; the
    // sorted entries are then interleaved back into buffer 1.
    float *keys = reinterpret_cast<float *>(entries_[1].get());
    std::uint32_t *ids = reinterpret_cast<std::uint32_t *>(keys + total);
    float *sortedKeys = reinterpret_cast<float *>(entries_[0].get());
    std::uint32_t *sortedIds = reinterpret_cast<std::uint32_t *>(sortedKeys + total);
    DeviceArray<std::uint16_t> labels;
    labels.reserve(rows_);
    // Page-locking the source first makes the copy ~4x faster, which more
    // than pays for the registration.
    void *values = const_cast<float *>(dataset_.values.data());
    const bool pinned =
        cudaHostRegister(values, total * sizeof(float), cudaHostRegisterReadOnly) == cudaSuccess;
    if (!pinned) {
      cudaGetLastError(); // fall back to a pageable copy
    }
    CUDA_CHECK(cudaMemcpyAsync(keys, dataset_.values.data(), total * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(labels.get(), dataset_.labels.data(),
                               rows_ * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream_));
    if (pinned) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      CUDA_CHECK(cudaHostUnregister(values));
    }
    phase("upload");
    const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
    preparePresortKernel<<<blocks, 256, 0, stream_>>>(keys, ids, labels.get(), rows_, total,
                                                       codec_);
    CUDA_CHECK(cudaGetLastError());

    std::size_t tempBytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, keys, sortedKeys, ids, sortedIds,
                                    static_cast<int>(rows_), 0, 32, stream_);
    DeviceArray<unsigned char> temp;
    temp.reserve(tempBytes);
    for (std::size_t feature = 0; feature < features_; ++feature) {
      const std::size_t offset = feature * rows_;
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp.get(), tempBytes, keys + offset,
                                                 sortedKeys + offset, ids + offset,
                                                 sortedIds + offset, static_cast<int>(rows_),
                                                 0, 32, stream_));
    }
    phase("sort");
    if (distinctValues_) {
      // C4.5 needs each feature's sorted distinct values. Copy the sorted keys
      // now (the buffer is reused later) and dedupe them on the CPU pool,
      // which is idle while the GPU works on the first levels.
      sortedValues_.resize(total);
      CUDA_CHECK(cudaMemcpyAsync(sortedValues_.data(), sortedKeys, total * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
    }
    interleaveKernel<<<blocks, 256, 0, stream_>>>(sortedKeys, sortedIds, entries_[1].get(),
                                                  total);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    phase("interleave");
    current_ = 1;
    if (distinctValues_) {
      distinctValues_->assign(features_, {});
      for (std::size_t feature = 0; feature < features_; ++feature) {
        pool_.submit([this, feature]() {
          (*distinctValues_)[feature] =
              distinctSortedValues(sortedValues_.data() + feature * rows_, rows_);
        });
      }
    }

    upload(logTable_, rules_.logTableValues());
    params_.logs = dt::LogTable{logTable_.get()};
    const std::vector<float> floatLogs(rules_.logTableValues().begin(),
                                       rules_.logTableValues().end());
    upload(floatLogTable_, floatLogs);
    params_.floatLogs = floatLogTable_.get();

    // Host memory for the columns of nodes handed to the CPU (every row is
    // handed over at most once, so features * rows entries always suffice).
    // Page-locked so the copies run at full speed; locking touches every page,
    // so it runs in the background while the GPU works on the first levels.
    arena_.reset(new Entry[total]);
    arenaUsed_ = 0;
    arenaPinning_ = std::async(std::launch::async, [this, total]() {
      if (cudaHostRegister(arena_.get(), total * sizeof(Entry), cudaHostRegisterDefault) ==
          cudaSuccess) {
        arenaPinned_ = true;
      } else {
        cudaGetLastError(); // pageable copies still work, just slower
      }
    });
  }

  void grow(std::unique_ptr<Node> &root) {
    if (rows_ < options_.gpuMinRows) {
      // Too small for the GPU: copy the presorted columns and use the CPU.
      arenaPinning_.get();
      CUDA_CHECK(cudaMemcpy(arena_.get(), entries_[current_].get(),
                            features_ * rows_ * sizeof(Entry), cudaMemcpyDeviceToHost));
      cpuBuilder_.build(Columns{arena_.get(), rows_, features_}, 0,
                        static_cast<std::uint32_t>(rows_), 0, root);
      return;
    }

    std::vector<std::uint32_t> counts(static_cast<std::size_t>(classCount_), 0);
    for (const std::uint16_t label : dataset_.labels) {
      ++counts[label];
    }
    root = std::make_unique<Node>();
    root->setCounts(std::move(counts));
    if (rules_.isTerminal(root->classCounts.data(), root->count, 0)) {
      return;
    }
    std::vector<LevelNode> level{{root.get(), 0, static_cast<std::uint32_t>(rows_), 0}};
    const bool verbose = profile_.enabled();
    int depth = 0;
    while (!level.empty()) {
      double seconds = 0.0;
      std::size_t rows = 0;
      for (const LevelNode &node : level) {
        rows += node.count;
      }
      const std::size_t nodes = level.size();
      {
        ScopedTimer timer(seconds);
        level = processLevel(level);
      }
      if (verbose) {
        std::fprintf(stderr, "  gpu level %2d: %6zu nodes %9zu rows %8.2f ms\n", depth, nodes,
                     rows, seconds * 1000.0);
      }
      ++depth;
    }
  }

  template <int MAXK> void launchScans(std::uint32_t tiles, std::size_t segmentCount) {
    const unsigned segmentBlocks = static_cast<unsigned>(segmentCount);
    profile_.run("tileEstimate", stream_, [&]() {
      tileEstimateKernel<MAXK><<<tiles, kThreads, 0, stream_>>>(
          entries_[current_].get(), segments_.get(), tileSegment_.get(), nodes_.get(), params_,
          tileHist_.get(), segmentTotals_.get(), tileMax_.get(), tileTries_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("segmentMax", stream_, [&]() {
      segmentMaxKernel<<<segmentBlocks, kThreads, 0, stream_>>>(segments_.get(), tileMax_.get(),
                                                               segmentMax_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("tileExact", stream_, [&]() {
      tileExactKernel<MAXK><<<tiles, kThreads, 0, stream_>>>(
          entries_[current_].get(), segments_.get(), tileSegment_.get(), nodes_.get(), params_,
          tileHist_.get(), segmentTotals_.get(), tileMax_.get(), segmentMax_.get(),
          tileBest_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("segmentBest", stream_, [&]() {
      segmentBestKernel<<<segmentBlocks, kThreads, 0, stream_>>>(
          entries_[current_].get(), segments_.get(), params_, tileHist_.get(), tileBest_.get(),
          tileTries_.get(), segmentBest_.get(), segmentLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
  }

  template <int MAXK> void launchHistogram(std::uint32_t tiles) {
    profile_.run("tileHistogramKernel", stream_, [&]() {
      tileHistogramKernel<MAXK><<<tiles, kThreads, 0, stream_>>>(
        entries_[current_].get(), segments_.get(), tileSegment_.get(), params_,
        tileHist_.get());
    });
    CUDA_CHECK(cudaGetLastError());
  }

  void dispatchHistogram(std::uint32_t tiles) {
    if (classCount_ <= 2) launchHistogram<2>(tiles);
    else if (classCount_ <= 4) launchHistogram<4>(tiles);
    else if (classCount_ <= 8) launchHistogram<8>(tiles);
    else if (classCount_ <= 16) launchHistogram<16>(tiles);
    else if (classCount_ <= 32) launchHistogram<32>(tiles);
    else launchHistogram<64>(tiles);
  }

  void dispatchScans(std::uint32_t tiles, std::size_t segmentCount) {
    if (classCount_ <= 2) launchScans<2>(tiles, segmentCount);
    else if (classCount_ <= 4) launchScans<4>(tiles, segmentCount);
    else if (classCount_ <= 8) launchScans<8>(tiles, segmentCount);
    else if (classCount_ <= 16) launchScans<16>(tiles, segmentCount);
    else if (classCount_ <= 32) launchScans<32>(tiles, segmentCount);
    else launchScans<64>(tiles, segmentCount);
  }

  template <typename T>
  void upload(DeviceArray<T> &target, const std::vector<T> &source) {
    target.reserve(source.size());
    CUDA_CHECK(cudaMemcpyAsync(target.get(), source.data(), source.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream_));
  }

  template <typename T>
  void download(std::vector<T> &target, const DeviceArray<T> &source, std::size_t count,
                bool wait = true) {
    target.resize(count);
    CUDA_CHECK(cudaMemcpyAsync(target.data(), source.get(), count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream_));
    if (wait) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
  }

  // Split the nodes of one level. Every node in `level` has its class counts
  // set and is not terminal. Returns the children that stay on the GPU.
  std::vector<LevelNode> processLevel(const std::vector<LevelNode> &level) {
    const std::size_t nodeCount = level.size();
    const std::size_t segmentCount = nodeCount * features_;

    // Tiles: segment = node * features + feature.
    std::vector<Segment> segments(segmentCount);
    std::vector<std::uint32_t> tileSegment;
    std::vector<NodeDesc> nodes(nodeCount);
    for (std::size_t n = 0; n < nodeCount; ++n) {
      const std::uint32_t tilesPerSegment = (level[n].count + kTile - 1) / kTile;
      for (std::size_t f = 0; f < features_; ++f) {
        const std::size_t index = n * features_ + f;
        segments[index] = {f * rows_ + level[n].begin,
                           level[n].count,
                           static_cast<std::uint32_t>(n),
                           static_cast<std::uint32_t>(f),
                           static_cast<std::uint32_t>(tileSegment.size()),
                           tilesPerSegment};
        tileSegment.insert(tileSegment.end(), tilesPerSegment,
                           static_cast<std::uint32_t>(index));
      }
      const std::uint32_t *counts = level[n].node->classCounts.data();
      nodes[n].count = level[n].count;
      nodes[n].minChild = rules_.minChildRows(level[n].count);
      nodes[n].parentWeighted = dt::weightedImpurity(counts, classCount_, level[n].count,
                                                     rules_.criterion(), rules_.logTable());
      nodes[n].filterMargin = floatGainMargin(classCount_, level[n].count, rules_.criterion());
    }
    const std::uint32_t tiles = static_cast<std::uint32_t>(tileSegment.size());
    upload(segments_, segments);
    upload(tileSegment_, tileSegment);
    upload(nodes_, nodes);
    tileHist_.reserve(static_cast<std::size_t>(tiles) * classCount_);
    segmentTotals_.reserve(segmentCount * classCount_);
    tileBest_.reserve(tiles);
    tileMax_.reserve(tiles);
    tileTries_.reserve(tiles);
    segmentMax_.reserve(segmentCount);
    tileLeft_.reserve(tiles);
    segmentBest_.reserve(segmentCount);
    segmentLeft_.reserve(segmentCount * classCount_);

    // 1-4: class counts left of every tile, then the best cut per (node,
    // feature) with its left class counts.
    dispatchHistogram(tiles);
    profile_.run("segmentPrefixKernel", stream_, [&]() {
      segmentPrefixKernel<<<static_cast<unsigned>(segmentCount), kThreads, 0, stream_>>>(
        tileHist_.get(), classCount_, segments_.get(), nodes_.get(), false,
        segmentTotals_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    dispatchScans(tiles, segmentCount);
    std::vector<CutCandidate> best;
    std::vector<std::uint32_t> lefts;
    download(best, segmentBest_, segmentCount, false);
    download(lefts, segmentLeft_, segmentCount * classCount_);

    // Split decisions. Children that are leaves are finished right here.
    bool anyPartition = false;
    std::vector<LevelNode> next;
    std::vector<HandOff> toCpu;
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    for (std::size_t n = 0; n < nodeCount; ++n) {
      const SplitRules::Decision decision =
          rules_.choose(best.data() + n * features_, features_, level[n].count);
      if (decision.feature < 0) {
        continue;
      }
      Node *node = level[n].node;
      node->feature = decision.feature;
      node->threshold = decision.threshold;
      nodes[n].splitFeature = decision.feature;
      nodes[n].leftCount = decision.leftCount;

      const std::uint32_t *leftCounts =
          lefts.data() + (n * features_ + static_cast<std::size_t>(decision.feature)) * classes;
      for (const bool left : {true, false}) {
        std::vector<std::uint32_t> counts(leftCounts, leftCounts + classes);
        if (!left) {
          for (std::size_t k = 0; k < classes; ++k) {
            counts[k] = node->classCounts[k] - counts[k];
          }
        }
        const std::uint32_t begin = left ? level[n].begin : level[n].begin + decision.leftCount;
        const std::uint32_t count =
            left ? decision.leftCount : level[n].count - decision.leftCount;
        const int depth = level[n].depth + 1;
        std::unique_ptr<Node> &slot = left ? node->left : node->right;
        if (rules_.isTerminal(counts.data(), count, depth)) {
          slot = std::make_unique<Node>();
          slot->setCounts(std::move(counts));
          continue;
        }
        nodes[n].partition = 1;
        if (count >= options_.gpuMinRows) {
          slot = std::make_unique<Node>();
          slot->setCounts(std::move(counts));
          next.push_back({slot.get(), begin, count, depth});
        } else {
          toCpu.push_back({node, left, begin, count, depth});
        }
      }
      anyPartition = anyPartition || nodes[n].partition;
    }
    if (!anyPartition) {
      return {};
    }

    // 5: stable partition of all columns of split nodes into the other buffer.
    upload(nodes_, nodes);
    const int other = 1 - current_;
    profile_.run("markGoesLeftKernel", stream_, [&]() {
      markGoesLeftKernel<<<tiles, kThreads, 0, stream_>>>(entries_[current_].get(),
                                                        segments_.get(), tileSegment_.get(),
                                                        nodes_.get(), codec_, goesLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("tileLeftCountKernel", stream_, [&]() {
      tileLeftCountKernel<<<tiles, kThreads, 0, stream_>>>(
        entries_[current_].get(), segments_.get(), tileSegment_.get(), nodes_.get(), codec_,
        goesLeft_.get(), tileLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("segmentPrefixKernel", stream_, [&]() {
      segmentPrefixKernel<<<static_cast<unsigned>(segmentCount), kThreads, 0, stream_>>>(
        tileLeft_.get(), 1, segments_.get(), nodes_.get(), true, nullptr);
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("scatterKernel", stream_, [&]() {
      scatterKernel<<<tiles, kThreads, 0, stream_>>>(
        entries_[current_].get(), entries_[other].get(), segments_.get(), tileSegment_.get(),
        nodes_.get(), codec_, goesLeft_.get(), tileLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());

    // 6: small children -> CPU. The old buffer is free now: gather there.
    if (!toCpu.empty()) {
      handOff(toCpu, other);
    }
    current_ = other;
    return next;
  }

  // Gather the columns of `children` (in buffer `buffer`) into one block,
  // copy it to the host and start a CPU task per child.
  void handOff(const std::vector<HandOff> &children, int buffer) {
    std::size_t handOffRows = 0;
    for (const HandOff &child : children) {
      handOffRows += child.count;
    }
    // Block layout: column f of all children back to back, stride handOffRows.
    std::vector<CopyJob> jobs;
    std::size_t offset = 0;
    for (const HandOff &child : children) {
      for (std::size_t f = 0; f < features_; ++f) {
        for (std::uint32_t done = 0; done < child.count; done += kTile) {
          jobs.push_back({f * rows_ + child.begin + done, f * handOffRows + offset + done,
                          std::min(kTile, child.count - done)});
        }
      }
      offset += child.count;
    }
    upload(copyJobs_, jobs);
    Entry *staging = entries_[1 - buffer].get(); // fully consumed by this level
    profile_.run("copyKernel", stream_, [&]() {
      copyKernel<<<static_cast<unsigned>(jobs.size()), kThreads, 0, stream_>>>(
        entries_[buffer].get(), staging, copyJobs_.get());
    });
    CUDA_CHECK(cudaGetLastError());

    if (arenaPinning_.valid()) {
      arenaPinning_.get();
    }
    Entry *host = arena_.get() + arenaUsed_;
    arenaUsed_ += features_ * handOffRows;
    profile_.run("handOffCopy", stream_, [&]() {
      CUDA_CHECK(cudaMemcpyAsync(host, staging, features_ * handOffRows * sizeof(Entry),
                                 cudaMemcpyDeviceToHost, stream_));
    });
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    const Columns columns{host, handOffRows, features_};
    offset = 0;
    for (const HandOff &child : children) {
      std::unique_ptr<Node> &slot = child.left ? child.parent->left : child.parent->right;
      const std::uint32_t begin = static_cast<std::uint32_t>(offset);
      pool_.submit([this, columns, begin, count = child.count, depth = child.depth, &slot]() {
        cpuBuilder_.build(columns, begin, count, depth, slot);
      });
      offset += child.count;
    }
  }

  const Dataset &dataset_;
  const SplitRules &rules_;
  EntryCodec codec_;
  const Options &options_;
  ThreadPool &pool_;
  std::vector<std::vector<float>> *distinctValues_;
  std::vector<float> sortedValues_;
  std::size_t rows_;
  std::size_t features_;
  int classCount_;
  ScanParams params_{};

  std::unique_ptr<std::uint8_t[]> hostGoesLeft_;
  CpuTreeBuilder cpuBuilder_;
  std::unique_ptr<Entry[]> arena_;
  std::size_t arenaUsed_ = 0;
  std::future<void> arenaPinning_;
  bool arenaPinned_ = false;

  cudaStream_t stream_ = nullptr;
  DeviceArray<Entry> entries_[2];
  int current_ = 0;
  DeviceArray<std::uint8_t> goesLeft_;
  DeviceArray<Segment> segments_;
  DeviceArray<std::uint32_t> tileSegment_;
  DeviceArray<std::uint32_t> tileHist_;
  DeviceArray<std::uint32_t> segmentTotals_;
  DeviceArray<NodeDesc> nodes_;
  DeviceArray<BestCut> tileBest_;
  DeviceArray<float> tileMax_;
  DeviceArray<float> segmentMax_;
  DeviceArray<std::uint32_t> tileTries_;
  DeviceArray<std::uint32_t> tileLeft_;
  DeviceArray<CutCandidate> segmentBest_;
  DeviceArray<std::uint32_t> segmentLeft_;
  DeviceArray<CopyJob> copyJobs_;
  DeviceArray<double> logTable_;
  DeviceArray<float> floatLogTable_;
  KernelProfile profile_{std::getenv("DT_GPU_VERBOSE") != nullptr};
};

} // namespace

void gpuPrepare() { CUDA_CHECK(cudaFree(nullptr)); }

std::unique_ptr<Node> gpuGrowTree(const Dataset &dataset, const SplitRules &rules,
                                  const dt::EntryCodec &codec, const Options &options,
                                  ThreadPool &pool, GpuTimings &timings,
                                  std::vector<std::vector<float>> *distinctValues) {
  GpuTreeBuilder builder(dataset, rules, codec, options, pool, distinctValues);
  return builder.run(timings);
}
