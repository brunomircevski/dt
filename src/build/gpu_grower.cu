// GPU tree builder (Cuda backend).
//
// The same algorithm as CpuTreeBuilder (presorted columns, one threshold sweep
// per feature, stable partition), reorganised for the GPU:
//
//  * Presort: every feature column is sorted once per grow with CUB radix sort.
//  * Breadth-first: all large nodes of one tree level are processed together
//    by a handful of kernel launches, so even deep levels with hundreds of
//    nodes keep the whole GPU busy. Work is cut into tiles of kTile entries of
//    one (node, feature) column range (a segment); every kernel runs one block
//    per tile or per segment.
//  * The host knows every node's class counts before the node is processed
//    (from its parent's split), so it decides up front which nodes are leaves;
//    only nodes that search for a split enter a level. Per level:
//      0. segmentConstant  segments whose values are all equal (they are
//                          skipped by every kernel, and dropped below)
//      1. tileHistogram    class counts of every tile
//      2. segmentPrefix    per segment: exclusive scan over its tiles, giving
//                          each tile the class counts left of it
//      3. tileEstimate / segmentMax / tileExact
//                          each tile sweeps its cuts: first in single
//                          precision, then exactly (the CPU's double-precision
//                          function) for the cuts that can still be the best.
//                          On GPUs with fast double precision, one exact sweep.
//      4. segmentBest      best cut per segment and the class counts left of
//                          it -> host picks the split (SplitRules) and the
//                          children
//      5. markGoesLeft / tileLeftCount / segmentPrefix / scatter:
//                          stable partition of the columns of split nodes into
//                          the other buffer (ping-pong between two copies),
//                          skipped for nodes whose children are both leaves
//    so each level needs one round trip to the host (step 4).
//  * Children with fewer than gpu.minRows rows are gathered into one block,
//    copied to the host in one transfer and grown by CpuTreeBuilder tasks on
//    the thread pool while the GPU continues with the next level.
//  * A grower used for cross-validation keeps the raw feature values on the
//    device and builds each fold's sorted columns there.

#include "build/cpu_builder.h"
#include "build/grower.h"
#include "core/timing.h"

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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace dt {

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
  float filterMargin = 0.0f;      // see floatGainMargin()
};

struct ScanParams {
  EntryCodec codec;
  int classCount;
  Criterion criterion;
  double minGap;
  LogTable logs;          // device copy of the host table
  const float *floatLogs; // the same table in single precision
};

// Single-precision estimate of cutGain, used only to rule out cuts that
// cannot be the best one (consumer GPUs run double precision ~64x slower).
__device__ __forceinline__ float floatXlog2x(const float *table, std::uint32_t count) {
  return count < kLogTableSize ? table[count]
                               : static_cast<float>(count) * log2f(static_cast<float>(count));
}

template <int MAXK>
__device__ __forceinline__ float floatCutGain(const std::uint32_t *total,
                                              const std::uint32_t *left, int classCount,
                                              std::uint32_t n, std::uint32_t nLeft,
                                              float parentWeighted, Criterion criterion,
                                              const float *table) {
  const std::uint32_t nRight = n - nLeft;
  float leftWeighted;
  float rightWeighted;
  if (criterion == Criterion::Gini) {
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
// more than 2 * bound + kTieEps below the best float gain of the segment
// cannot be the best cut, or tie with it, so only the others are scored in
// double precision.
float floatGainMargin(int classCount, std::uint32_t n, Criterion criterion) {
  const double unit = 1.0 / (1 << 23);
  const double scale =
      criterion == Criterion::Gini ? 1.0 : std::max(1.0, std::log2(double(n)));
  const double bound = 4.0 * (2.0 * classCount + 8.0) * unit * scale;
  return static_cast<float>(2.0 * bound + kTieEps);
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

constexpr BestCut kNoCut{-INFINITY, 0xFFFFFFFFu, 0.0f, 0.0f};

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
    return isBetterCut(b.gain, b.position, a.gain, a.position) ? b : a;
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
// keys/ids of every (feature, selected row): keys[f * count + i] is the value
// of row rows[i] (or row i if rows is null). `values` may alias `keys` only
// when rows is null (in place). Grid: x over rows, y = feature.
__global__ void gatherKernel(const float *values, std::size_t stride, const std::uint32_t *rows,
                             std::uint32_t count, const std::uint16_t *labels, EntryCodec codec,
                             float *keys, std::uint32_t *ids) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::size_t feature = blockIdx.y;
  const std::uint32_t row = rows ? rows[index] : index;
  const std::size_t target = feature * count + index;
  keys[target] = values[feature * stride + row] + 0.0f; // -0.0 -> +0.0, as on the CPU
  ids[target] = codec.pack(row, labels[row]);
}

__global__ void interleaveKernel(const float *keys, const std::uint32_t *ids, Entry *entries,
                                 std::size_t total) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    entries[index] = {keys[index], ids[index]};
  }
}

// ---------------------------------------------------------------------------
// 0. Segments without a cut: all values (nearly) equal (sorted: first vs last)
// ---------------------------------------------------------------------------
__global__ void segmentConstantKernel(const Entry *src, const Segment *segments,
                                      std::size_t segmentCount, double minGap,
                                      std::uint8_t *constant) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < segmentCount) {
    const Segment segment = segments[index];
    const Entry *column = src + segment.base;
    constant[index] = !isCut(column[0].value, column[segment.count - 1].value, minGap);
  }
}

// ---------------------------------------------------------------------------
// 1. Class histogram of every tile
// ---------------------------------------------------------------------------
template <int MAXK>
__global__ void __launch_bounds__(kThreads)
    tileHistogramKernel(const Entry *src, const Segment *segments,
                        const std::uint32_t *tileSegment, const std::uint8_t *constant,
                        ScanParams params, std::uint32_t *tileHist) {
  __shared__ std::uint32_t histogram[MAXK];
  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t segmentIndex = tileSegment[tile];
  if (constant[segmentIndex]) {
    return;
  }
  const Segment segment = segments[segmentIndex];
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
// 2. / 5. Exclusive scan over the tiles of each segment (one block per
// segment), `width` values per tile. Writes the per-segment totals.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads)
    segmentPrefixKernel(std::uint32_t *perTile, int width, const Segment *segments,
                        const NodeDesc *nodes, const std::uint8_t *constant,
                        bool partitionedOnly, std::uint32_t *segmentTotals) {
  using Scan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename Scan::TempStorage scanStorage;
  const Segment segment = segments[blockIdx.x];
  if (constant[blockIdx.x] || (partitionedOnly && !nodes[segment.node].partition)) {
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
// With fast double precision, tileExact alone scores every cut and counts.
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
// leftCounts, previousValue, estimate); the estimate is only computed if
// ESTIMATE. Returns the number of real cuts.
template <int MAXK, bool ESTIMATE, typename OnCut>
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
        isCut(previous, value, params.minGap)) {
      ++tries;
      bool skip = false;
      if (previousClass == sweep.cls[step]) {
        const bool leftSingleton = position < 2 || isCut(beforePrevious, previous, params.minGap);
        const bool rightSingleton = position + 1 >= count || isCut(value, next, params.minGap);
        skip = isSkippableCut(true, leftSingleton, rightSingleton, position, count, minChild);
      }
      if (!skip) {
        float estimate = 0.0f;
        if constexpr (ESTIMATE) {
          estimate = floatCutGain<MAXK>(totals, leftHere, params.classCount, count, position,
                                        parentWeighted, params.criterion, params.floatLogs);
        }
        onCut(step, position, leftHere, previous, estimate);
      }
    }
  }
  return tries;
}

template <int MAXK>
__global__ void __launch_bounds__(kThreads)
    tileEstimateKernel(const Entry *src, const Segment *segments,
                       const std::uint32_t *tileSegment, const std::uint8_t *constant,
                       const NodeDesc *nodes, ScanParams params, const std::uint32_t *tilePrefix,
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
  if (constant[segmentIndex]) {
    if (threadIdx.x == 0) {
      tileMax[tile] = -INFINITY;
      tileTries[tile] = 0;
    }
    return;
  }
  const Segment segment = segments[segmentIndex];
  const NodeDesc node = nodes[segment.node];
  TileSweep<MAXK> sweep;
  loadTile<MAXK>(sweep, src, segment, segmentIndex, tile, params, tilePrefix, segmentTotals,
                 warpCounts, totals);
  float best = -INFINITY;
  const std::uint32_t tries = sweepTile<MAXK, true>(
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

// FILTER: two-pass mode (only cuts whose estimate reaches the cutoff are
// scored; tries come from tileEstimate). Otherwise every cut is scored and
// the tile's tries are counted here.
template <int MAXK, bool FILTER>
__global__ void __launch_bounds__(kThreads)
    tileExactKernel(const Entry *src, const Segment *segments, const std::uint32_t *tileSegment,
                    const std::uint8_t *constant, const NodeDesc *nodes, ScanParams params,
                    const std::uint32_t *tilePrefix, const std::uint32_t *segmentTotals,
                    const float *tileMax, const float *segmentMax, BestCut *tileBest,
                    std::uint32_t *tileTries) {
  using CutReduce = cub::BlockReduce<BestCut, kThreads>;
  using SumReduce = cub::BlockReduce<std::uint32_t, kThreads>;
  __shared__ union {
    typename CutReduce::TempStorage cut;
    typename SumReduce::TempStorage sum;
  } storage;
  __shared__ std::uint32_t totals[MAXK];
  __shared__ std::uint32_t warpCounts[kWarps][MAXK];

  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  const NodeDesc node = nodes[segment.node];
  float cutoff = -INFINITY;
  bool skip = constant[segmentIndex];
  if (FILTER && !skip) {
    cutoff = segmentMax[segmentIndex] - node.filterMargin;
    skip = tileMax[tile] < cutoff; // no cut of this tile can be the best one
  }
  if (skip) {
    if (threadIdx.x == 0) {
      tileBest[tile] = kNoCut;
      if (!FILTER) {
        tileTries[tile] = 0;
      }
    }
    return;
  }
  TileSweep<MAXK> sweep;
  loadTile<MAXK>(sweep, src, segment, segmentIndex, tile, params, tilePrefix, segmentTotals,
                 warpCounts, totals);
  BestCut best = kNoCut;
  const std::uint32_t tries = sweepTile<MAXK, FILTER>(
      sweep, segment, node, params, totals,
      [&](int step, std::uint32_t position, const std::uint32_t *left, float previous,
          float estimate) {
        if (FILTER && estimate < cutoff) {
          return;
        }
        const double gain = cutGain(totals, left, params.classCount, segment.count, position,
                                    node.parentWeighted, params.criterion, params.logs);
        if (best.gain == -INFINITY || isBetterCut(gain, position, best.gain, best.position)) {
          best = {gain, position, previous, sweep.value[step]};
        }
      });
  const BestCut tileWinner = CutReduce(storage.cut).Reduce(best, BetterCut());
  if (threadIdx.x == 0) {
    tileBest[tile] = tileWinner;
  }
  if (!FILTER) {
    __syncthreads();
    const std::uint32_t tileTotalTries = SumReduce(storage.sum).Sum(tries);
    if (threadIdx.x == 0) {
      tileTries[tile] = tileTotalTries;
    }
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
  BestCut best = kNoCut;
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
// 5. Stable partition of every (non-constant) column of every split node
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
                        const std::uint32_t *tileSegment, const std::uint8_t *constant,
                        const NodeDesc *nodes, EntryCodec codec, const std::uint8_t *goesLeft,
                        std::uint32_t *tileLeft) {
  using SumReduce = cub::BlockReduce<std::uint32_t, kThreads>;
  __shared__ typename SumReduce::TempStorage storage;
  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  if (constant[segmentIndex] || !nodes[segment.node].partition) {
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
                  const std::uint32_t *tileSegment, const std::uint8_t *constant,
                  const NodeDesc *nodes, EntryCodec codec, const std::uint8_t *goesLeft,
                  const std::uint32_t *tileLeftPrefix) {
  using Scan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename Scan::TempStorage storage;
  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  const NodeDesc node = nodes[segment.node];
  if (constant[segmentIndex] || !node.partition) {
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
      std::fprintf(stderr, "  %-20s %9.2f ms\n", total.first.c_str(), total.second);
    }
    totals_.clear();
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
  std::uint32_t node; // NodeStore id
  std::uint32_t begin; // range in the current buffer (all columns)
  std::uint32_t count;
  int depth;
  std::vector<std::uint32_t> features; // features that can still split it
};

class GpuGrower final : public Grower {
public:
  GpuGrower(const Dataset &train, const Options &options, ThreadPool &pool, bool reusable,
            double &setupSeconds)
      : train_(train), options_(options), pool_(pool), reusable_(reusable),
        rows_(train.rowCount), features_(train.featureCount()),
        classCount_(static_cast<int>(train.classCount())),
        codec_(makeEntryCodec(train.classCount(), train.rowCount)),
        hostGoesLeft_(new std::uint8_t[train.rowCount]),
        profile_(std::getenv("DT_GPU_VERBOSE") != nullptr) {
    if (classCount_ > kMaxGpuClasses) {
      throw std::runtime_error("The Cuda backend supports at most 64 classes.");
    }
    if (features_ > 65535) {
      throw std::runtime_error("The Cuda backend supports at most 65535 features.");
    }
    ScopedTimer timer(setupSeconds);
    setup();
  }

  ~GpuGrower() override {
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

  Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows,
            GrowTimings &timings, std::vector<float> *sortedValues) override {
    stride_ = rows.empty() ? rows_ : rows.size();
    params_.minGap = rules.minValueGap();
    params_.criterion = rules.criterion();
    {
      ScopedTimer timer(timings.prepareSeconds);
      presort(rows, sortedValues);
    }
    Tree tree;
    tree.featureNames = train_.featureNames;
    tree.classNames = train_.classNames;
    ScopedTimer timer(timings.buildSeconds);
    NodeStore store(train_.classCount(), 2 * stride_);
    CpuTreeBuilder cpuBuilder(rules, codec_, features_, store, hostGoesLeft_.get(), &pool_,
                              options_.parallel);
    rules_ = &rules;
    store_ = &store;
    cpuBuilder_ = &cpuBuilder;
    arenaUsed_ = 0;
    scratchUsed_ = 0;

    std::vector<std::uint32_t> counts(static_cast<std::size_t>(classCount_), 0);
    if (rows.empty()) {
      for (const std::uint16_t label : train_.labels) {
        ++counts[label];
      }
    } else {
      for (const std::uint32_t row : rows) {
        ++counts[train_.labels[row]];
      }
    }
    std::vector<std::uint32_t> allFeatures(features_);
    std::iota(allFeatures.begin(), allFeatures.end(), 0u);
    LevelNode root{store.add(counts.data()), 0, static_cast<std::uint32_t>(stride_), 0,
                   std::move(allFeatures)};
    if (stride_ < options_.gpu.minRows) {
      growOnCpu(std::move(root));
    } else {
      growLevels(std::move(root));
    }
    double waitSeconds = 0.0;
    {
      ScopedTimer wait(waitSeconds);
      pool_.waitIdle();
    }
    if (profile_.enabled()) {
      std::fprintf(stderr, "  waiting for CPU subtrees: %.2f ms\n", waitSeconds * 1000.0);
      profile_.report();
    }
    store.toTree(tree);
    rules_ = nullptr;
    store_ = nullptr;
    cpuBuilder_ = nullptr;
    return tree;
  }

private:
  void setup() {
    double contextSeconds = 0.0;
    {
      ScopedTimer timer(contextSeconds);
      CUDA_CHECK(cudaFree(nullptr)); // creates the CUDA context if needed
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int fp32PerFp64 = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&fp32PerFp64, cudaDevAttrSingleToDoublePrecisionPerfRatio,
                                      device));
    onePass_ = options_.gpu.sweep == GpuSweep::OnePass ||
               (options_.gpu.sweep == GpuSweep::Auto && fp32PerFp64 <= 4);
    if (profile_.enabled()) {
      cudaDeviceProp properties{};
      CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      std::fprintf(stderr, "  gpu %s (sm_%d%d), fp32:fp64 = %d:1, %s sweep\n", properties.name,
                   properties.major, properties.minor, fp32PerFp64,
                   onePass_ ? "one-pass" : "two-pass");
      std::fprintf(stderr, "  setup context      %8.2f ms\n", contextSeconds * 1e3);
    }

    const std::size_t total = features_ * rows_;
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const std::size_t needed = 2 * total * sizeof(Entry) +
                               (reusable_ ? total * sizeof(float) : 0) + rows_ * 7 + (64u << 20);
    if (needed > freeBytes) {
      throw std::runtime_error("Not enough GPU memory: need about " +
                               std::to_string(needed >> 20) + " MiB, " +
                               std::to_string(freeBytes >> 20) +
                               " MiB free. Use --parallel instead.");
    }
    entries_[0].reserve(total);
    entries_[1].reserve(total);
    goesLeft_.reserve(rows_);
    labels_.reserve(rows_);
    if (reusable_) {
      values_.reserve(total);
      rowList_.reserve(rows_);
    }

    // Raw values: kept in their own buffer for repeated use, otherwise
    // uploaded straight into the key area of the presort (buffer 1).
    // Page-locking the source first makes the copy ~4x faster, which more than
    // pays for the registration.
    float *target = reusable_ ? values_.get() : reinterpret_cast<float *>(entries_[1].get());
    void *source = const_cast<float *>(train_.values.data());
    const bool pinned =
        cudaHostRegister(source, total * sizeof(float), cudaHostRegisterReadOnly) == cudaSuccess;
    if (!pinned) {
      cudaGetLastError(); // fall back to a pageable copy
    }
    CUDA_CHECK(cudaMemcpyAsync(target, train_.values.data(), total * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(labels_.get(), train_.labels.data(),
                               rows_ * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (pinned) {
      CUDA_CHECK(cudaHostUnregister(source));
    }

    std::size_t tempBytes = 0;
    float *keys = reinterpret_cast<float *>(entries_[1].get());
    std::uint32_t *ids = reinterpret_cast<std::uint32_t *>(entries_[0].get());
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, keys, keys, ids, ids,
                                               static_cast<int>(rows_), 0, 32, stream_));
    sortTemp_.reserve(tempBytes);
    sortTempBytes_ = tempBytes;

    // c * log2(c) tables (the same for every grow).
    const std::vector<double> logs = xlog2xTable();
    upload(logTable_, logs);
    upload(floatLogTable_, std::vector<float>(logs.begin(), logs.end()));
    params_.codec = codec_;
    params_.classCount = classCount_;
    params_.logs = LogTable{logTable_.get()};
    params_.floatLogs = floatLogTable_.get();

    // Host memory for the columns of nodes handed to the CPU (every row is
    // handed over at most once per grow, so features * rows entries always
    // suffice), and their partition scratch. Page-locked so the copies run at
    // full speed; locking touches every page, so it runs in the background.
    arena_.reset(new Entry[total]);
    arenaScratch_.reset(new Entry[rows_]);
    arenaPinning_ = std::async(std::launch::async, [this, total]() {
      if (cudaHostRegister(arena_.get(), total * sizeof(Entry), cudaHostRegisterDefault) ==
          cudaSuccess) {
        arenaPinned_ = true;
      } else {
        cudaGetLastError(); // pageable copies still work, just slower
      }
    });
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  // Sorted columns of the selected rows in buffer 1, stride = their count.
  // The keys/ids to sort go to buffer 1, the sorted ones to buffer 0, and are
  // then interleaved back into buffer 1.
  void presort(std::span<const std::uint32_t> rows, std::vector<float> *sortedValues) {
    const std::size_t total = features_ * stride_;
    float *keys = reinterpret_cast<float *>(entries_[1].get());
    std::uint32_t *ids = reinterpret_cast<std::uint32_t *>(keys + total);
    float *sortedKeys = reinterpret_cast<float *>(entries_[0].get());
    std::uint32_t *sortedIds = reinterpret_cast<std::uint32_t *>(sortedKeys + total);
    const float *values = keys; // in place: uploaded there by setup()
    if (reusable_) {
      values = values_.get();
    } else if (presorted_ || !rows.empty()) {
      throw std::logic_error("GpuGrower: not created for repeated use");
    }
    presorted_ = true;
    const std::uint32_t *rowList = nullptr;
    if (!rows.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(rowList_.get(), rows.data(), rows.size() * sizeof(std::uint32_t),
                                 cudaMemcpyHostToDevice, stream_));
      rowList = rowList_.get();
    }
    const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
    profile_.run("presortGather", stream_, [&]() {
      const dim3 grid(static_cast<unsigned>((stride_ + 255) / 256), static_cast<unsigned>(features_));
      gatherKernel<<<grid, 256, 0, stream_>>>(values, rows_, rowList,
                                              static_cast<std::uint32_t>(stride_), labels_.get(),
                                              codec_, keys, ids);
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("presortSort", stream_, [&]() {
      for (std::size_t feature = 0; feature < features_; ++feature) {
        const std::size_t offset = feature * stride_;
        std::size_t tempBytes = sortTempBytes_;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            sortTemp_.get(), tempBytes, keys + offset, sortedKeys + offset, ids + offset,
            sortedIds + offset, static_cast<int>(stride_), 0, 32, stream_));
      }
    });
    if (sortedValues) {
      // Before level 0 reuses buffer 0 (the copy is ordered on the stream).
      sortedValues->resize(total);
      CUDA_CHECK(cudaMemcpyAsync(sortedValues->data(), sortedKeys, total * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
    }
    profile_.run("presortInterleave", stream_, [&]() {
      interleaveKernel<<<blocks, 256, 0, stream_>>>(sortedKeys, sortedIds, entries_[1].get(),
                                                    total);
    });
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    current_ = 1;
  }

  // The arena is page-locked in the background; wait before the first copy.
  void waitForArena() {
    if (arenaPinning_.valid()) {
      arenaPinning_.get();
    }
  }

  // Too small for the GPU: copy the presorted columns and use the CPU.
  void growOnCpu(LevelNode root) {
    waitForArena();
    CUDA_CHECK(cudaMemcpy(arena_.get(), entries_[current_].get(),
                          features_ * stride_ * sizeof(Entry), cudaMemcpyDeviceToHost));
    cpuBuilder_->grow(Columns{arena_.get(), stride_, arenaScratch_.get()},
                      Subtree{root.node, 0, root.count, 0, std::move(root.features)});
  }

  void growLevels(LevelNode root) {
    if (rules_->isTerminal(store_->counts(root.node), root.count, 0)) {
      return;
    }
    std::vector<LevelNode> level;
    level.push_back(std::move(root));
    for (int depth = 0; !level.empty(); ++depth) {
      double seconds = 0.0;
      std::size_t rows = 0;
      std::size_t segments = 0;
      for (const LevelNode &node : level) {
        rows += node.count;
        segments += node.features.size();
      }
      const std::size_t nodes = level.size();
      {
        ScopedTimer timer(seconds);
        level = processLevel(level);
      }
      if (profile_.enabled()) {
        std::fprintf(stderr, "  gpu level %2d: %6zu nodes %9zu rows %7zu segments %8.2f ms\n",
                     depth, nodes, rows, segments, seconds * 1000.0);
      }
    }
  }

  template <int MAXK> void launchScans(std::uint32_t tiles, std::size_t segmentCount) {
    const unsigned segmentBlocks = static_cast<unsigned>(segmentCount);
    const Entry *src = entries_[current_].get();
    if (onePass_) {
      profile_.run("tileExact", stream_, [&]() {
        tileExactKernel<MAXK, false><<<tiles, kThreads, 0, stream_>>>(
            src, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
            tileHist_.get(), segmentTotals_.get(), nullptr, nullptr, tileBest_.get(),
            tileTries_.get());
      });
      CUDA_CHECK(cudaGetLastError());
    } else {
      profile_.run("tileEstimate", stream_, [&]() {
        tileEstimateKernel<MAXK><<<tiles, kThreads, 0, stream_>>>(
            src, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
            tileHist_.get(), segmentTotals_.get(), tileMax_.get(), tileTries_.get());
      });
      CUDA_CHECK(cudaGetLastError());
      profile_.run("segmentMax", stream_, [&]() {
        segmentMaxKernel<<<segmentBlocks, kThreads, 0, stream_>>>(segments_.get(),
                                                                 tileMax_.get(),
                                                                 segmentMax_.get());
      });
      CUDA_CHECK(cudaGetLastError());
      profile_.run("tileExact", stream_, [&]() {
        tileExactKernel<MAXK, true><<<tiles, kThreads, 0, stream_>>>(
            src, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
            tileHist_.get(), segmentTotals_.get(), tileMax_.get(), segmentMax_.get(),
            tileBest_.get(), nullptr);
      });
      CUDA_CHECK(cudaGetLastError());
    }
    profile_.run("segmentBest", stream_, [&]() {
      segmentBestKernel<<<segmentBlocks, kThreads, 0, stream_>>>(
          src, segments_.get(), params_, tileHist_.get(), tileBest_.get(), tileTries_.get(),
          segmentBest_.get(), segmentLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
  }

  template <int MAXK> void launchHistogram(std::uint32_t tiles) {
    profile_.run("tileHistogram", stream_, [&]() {
      tileHistogramKernel<MAXK><<<tiles, kThreads, 0, stream_>>>(
          entries_[current_].get(), segments_.get(), tileSegment_.get(), constant_.get(),
          params_, tileHist_.get());
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

  template <typename T> void upload(DeviceArray<T> &target, const std::vector<T> &source) {
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
    const std::size_t classes = static_cast<std::size_t>(classCount_);

    // Segments: one per (node, feature that can still split it).
    std::vector<Segment> segments;
    std::vector<std::size_t> firstSegment(nodeCount + 1, 0);
    std::vector<std::uint32_t> tileSegment;
    std::vector<NodeDesc> nodes(nodeCount);
    for (std::size_t n = 0; n < nodeCount; ++n) {
      const LevelNode &node = level[n];
      const std::uint32_t tilesPerSegment = (node.count + kTile - 1) / kTile;
      firstSegment[n] = segments.size();
      for (const std::uint32_t f : node.features) {
        const std::uint32_t index = static_cast<std::uint32_t>(segments.size());
        segments.push_back({f * stride_ + node.begin, node.count, static_cast<std::uint32_t>(n),
                            f, static_cast<std::uint32_t>(tileSegment.size()),
                            tilesPerSegment});
        tileSegment.insert(tileSegment.end(), tilesPerSegment, index);
      }
      const std::uint32_t *counts = store_->counts(node.node);
      nodes[n].count = node.count;
      nodes[n].minChild = rules_->minChildRows(node.count);
      nodes[n].parentWeighted = weightedImpurity(counts, classCount_, node.count,
                                                 rules_->criterion(), rules_->logTable());
      nodes[n].filterMargin = floatGainMargin(classCount_, node.count, rules_->criterion());
    }
    firstSegment[nodeCount] = segments.size();
    const std::size_t segmentCount = segments.size();
    const std::uint32_t tiles = static_cast<std::uint32_t>(tileSegment.size());
    upload(segments_, segments);
    upload(tileSegment_, tileSegment);
    upload(nodes_, nodes);
    constant_.reserve(segmentCount);
    tileHist_.reserve(static_cast<std::size_t>(tiles) * classes);
    segmentTotals_.reserve(segmentCount * classes);
    tileBest_.reserve(tiles);
    tileMax_.reserve(tiles);
    tileTries_.reserve(tiles);
    segmentMax_.reserve(segmentCount);
    tileLeft_.reserve(tiles);
    segmentBest_.reserve(segmentCount);
    segmentLeft_.reserve(segmentCount * classes);

    // 0-4: constant segments, class counts left of every tile, then the best
    // cut per segment with its left class counts.
    profile_.run("segmentConstant", stream_, [&]() {
      segmentConstantKernel<<<static_cast<unsigned>((segmentCount + 255) / 256), 256, 0,
                              stream_>>>(entries_[current_].get(), segments_.get(), segmentCount,
                                         params_.minGap, constant_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    dispatchHistogram(tiles);
    profile_.run("segmentPrefix", stream_, [&]() {
      segmentPrefixKernel<<<static_cast<unsigned>(segmentCount), kThreads, 0, stream_>>>(
          tileHist_.get(), classCount_, segments_.get(), nodes_.get(), constant_.get(), false,
          segmentTotals_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    dispatchScans(tiles, segmentCount);
    std::vector<CutCandidate> best;
    std::vector<std::uint32_t> lefts;
    std::vector<std::uint8_t> constant;
    download(best, segmentBest_, segmentCount, false);
    download(constant, constant_, segmentCount, false);
    download(lefts, segmentLeft_, segmentCount * classes);

    // Split decisions. Children that are leaves are finished right here.
    bool anyPartition = false;
    std::vector<LevelNode> next;
    std::vector<LevelNode> toCpu;
    thread_local std::vector<CutCandidate> cuts;
    for (std::size_t n = 0; n < nodeCount; ++n) {
      const LevelNode &parent = level[n];
      cuts.assign(features_, CutCandidate{});
      std::vector<std::uint32_t> features; // still useful below this node
      int winnerSegment = -1;
      for (std::size_t s = firstSegment[n]; s < firstSegment[n + 1]; ++s) {
        cuts[segments[s].feature] = best[s];
        if (!constant[s]) {
          features.push_back(segments[s].feature);
        }
      }
      const SplitRules::Decision decision =
          rules_->choose(cuts.data(), features_, parent.count);
      if (decision.feature < 0) {
        continue;
      }
      for (std::size_t s = firstSegment[n]; s < firstSegment[n + 1]; ++s) {
        if (static_cast<int>(segments[s].feature) == decision.feature) {
          winnerSegment = static_cast<int>(s);
        }
      }
      nodes[n].splitFeature = decision.feature;
      nodes[n].leftCount = decision.leftCount;

      const std::uint32_t *parentCounts = store_->counts(parent.node);
      std::vector<std::uint32_t> childCounts(2 * classes);
      std::uint32_t *leftCounts = childCounts.data();
      std::uint32_t *rightCounts = leftCounts + classes;
      std::copy_n(lefts.data() + static_cast<std::size_t>(winnerSegment) * classes, classes,
                  leftCounts);
      for (std::size_t k = 0; k < classes; ++k) {
        rightCounts[k] = parentCounts[k] - leftCounts[k];
      }
      const std::uint32_t leftId = store_->add(leftCounts);
      const std::uint32_t rightId = store_->add(rightCounts);
      Node &node = store_->node(parent.node);
      node.feature = decision.feature;
      node.threshold = decision.threshold;
      node.left = leftId;
      node.right = rightId;

      for (const bool left : {true, false}) {
        const std::uint32_t begin = left ? parent.begin : parent.begin + decision.leftCount;
        const std::uint32_t count =
            left ? decision.leftCount : parent.count - decision.leftCount;
        const int depth = parent.depth + 1;
        if (rules_->isTerminal(left ? leftCounts : rightCounts, count, depth)) {
          continue;
        }
        nodes[n].partition = 1;
        LevelNode child{left ? leftId : rightId, begin, count, depth, features};
        (count >= options_.gpu.minRows ? next : toCpu).push_back(std::move(child));
      }
      anyPartition = anyPartition || nodes[n].partition;
    }
    if (!anyPartition) {
      return {};
    }

    // 5: stable partition of the non-constant columns of split nodes into the
    //    other buffer.
    upload(nodes_, nodes);
    const int other = 1 - current_;
    profile_.run("markGoesLeft", stream_, [&]() {
      markGoesLeftKernel<<<tiles, kThreads, 0, stream_>>>(entries_[current_].get(),
                                                          segments_.get(), tileSegment_.get(),
                                                          nodes_.get(), codec_, goesLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("tileLeftCount", stream_, [&]() {
      tileLeftCountKernel<<<tiles, kThreads, 0, stream_>>>(
          entries_[current_].get(), segments_.get(), tileSegment_.get(), constant_.get(),
          nodes_.get(), codec_, goesLeft_.get(), tileLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("segmentPrefix", stream_, [&]() {
      segmentPrefixKernel<<<static_cast<unsigned>(segmentCount), kThreads, 0, stream_>>>(
          tileLeft_.get(), 1, segments_.get(), nodes_.get(), constant_.get(), true, nullptr);
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("scatter", stream_, [&]() {
      scatterKernel<<<tiles, kThreads, 0, stream_>>>(
          entries_[current_].get(), entries_[other].get(), segments_.get(), tileSegment_.get(),
          constant_.get(), nodes_.get(), codec_, goesLeft_.get(), tileLeft_.get());
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
  // copy it to the host and start a CPU task per child. Only the columns of
  // features that can still split a child are copied.
  void handOff(std::vector<LevelNode> &children, int buffer) {
    std::size_t handOffRows = 0;
    for (const LevelNode &child : children) {
      handOffRows += child.count;
    }
    // Block layout: column f of all children back to back, stride handOffRows.
    std::vector<CopyJob> jobs;
    std::size_t offset = 0;
    for (const LevelNode &child : children) {
      for (const std::uint32_t f : child.features) {
        for (std::uint32_t done = 0; done < child.count; done += kTile) {
          jobs.push_back({f * stride_ + child.begin + done, f * handOffRows + offset + done,
                          std::min(kTile, child.count - done)});
        }
      }
      offset += child.count;
    }
    const std::size_t blockEntries = features_ * handOffRows;
    Entry *staging = entries_[1 - buffer].get(); // fully consumed by this level
    if (!jobs.empty()) {
      upload(copyJobs_, jobs);
      profile_.run("handOffGather", stream_, [&]() {
        copyKernel<<<static_cast<unsigned>(jobs.size()), kThreads, 0, stream_>>>(
            entries_[buffer].get(), staging, copyJobs_.get());
      });
      CUDA_CHECK(cudaGetLastError());
    }

    waitForArena();
    Entry *host = arena_.get() + arenaUsed_;
    arenaUsed_ += blockEntries;
    Entry *scratch = arenaScratch_.get() + scratchUsed_;
    scratchUsed_ += handOffRows;
    // One transfer from the first to the last copied column.
    std::size_t firstEntry = blockEntries;
    std::size_t endEntry = 0;
    for (const CopyJob &job : jobs) {
      firstEntry = std::min<std::size_t>(firstEntry, job.to);
      endEntry = std::max<std::size_t>(endEntry, job.to + job.count);
    }
    if (endEntry > firstEntry) {
      profile_.run("handOffCopy", stream_, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(host + firstEntry, staging + firstEntry,
                                   (endEntry - firstEntry) * sizeof(Entry),
                                   cudaMemcpyDeviceToHost, stream_));
      });
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    const Columns columns{host, handOffRows, scratch};
    offset = 0;
    for (LevelNode &child : children) {
      Subtree subtree{child.node, static_cast<std::uint32_t>(offset), child.count, child.depth,
                      std::move(child.features)};
      pool_.submit([this, columns, subtree = std::move(subtree)]() mutable {
        cpuBuilder_->grow(columns, std::move(subtree));
      });
      offset += child.count;
    }
  }

  const Dataset &train_;
  const Options &options_;
  ThreadPool &pool_;
  bool reusable_;
  bool presorted_ = false;
  bool onePass_ = false;
  std::size_t rows_;
  std::size_t features_;
  int classCount_;
  EntryCodec codec_;
  ScanParams params_{};

  // Per grow.
  std::size_t stride_ = 0; // rows of the current grow (column stride)
  const SplitRules *rules_ = nullptr;
  NodeStore *store_ = nullptr;
  CpuTreeBuilder *cpuBuilder_ = nullptr;
  int current_ = 1;

  std::unique_ptr<std::uint8_t[]> hostGoesLeft_;
  std::unique_ptr<Entry[]> arena_;
  std::unique_ptr<Entry[]> arenaScratch_;
  std::size_t arenaUsed_ = 0;
  std::size_t scratchUsed_ = 0;
  std::future<void> arenaPinning_;
  bool arenaPinned_ = false;

  cudaStream_t stream_ = nullptr;
  DeviceArray<Entry> entries_[2];
  DeviceArray<float> values_;
  DeviceArray<std::uint16_t> labels_;
  DeviceArray<std::uint32_t> rowList_;
  DeviceArray<unsigned char> sortTemp_;
  std::size_t sortTempBytes_ = 0;
  DeviceArray<std::uint8_t> goesLeft_;
  DeviceArray<Segment> segments_;
  DeviceArray<std::uint32_t> tileSegment_;
  DeviceArray<std::uint8_t> constant_;
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
  KernelProfile profile_;
};

} // namespace

void gpuPrepare() { CUDA_CHECK(cudaFree(nullptr)); }

std::unique_ptr<Grower> makeGpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool &pool, bool reusable, double &setupSeconds) {
  return std::make_unique<GpuGrower>(train, options, pool, reusable, setupSeconds);
}

} // namespace dt
