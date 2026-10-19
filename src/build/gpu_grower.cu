// GPU tree builder (Cuda backend).
//
// The same algorithm as CpuTreeBuilder (presorted columns, one threshold sweep
// per feature, stable partition), reorganised for the GPU:
//
//  * Presort: every feature column is sorted once per grow with CUB radix sort.
//  * Memory: one copy of the sorted columns, in a buffer of a slot per column
//    and kSpareSlots more (about 8 bytes per value). The partition moves each
//    column to a free slot, and the hand-offs to the CPU go through a small
//    ring (kRingBytes), so the peak is ~(features + 1) * rows * 8 bytes + 512 MB.
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
//      1. tileHistogram    class counts of every tile (and its cuts); below
//                          the root, the parent level's partition counts them
//      2. segmentPrefix    per segment: exclusive scan over its tiles, giving
//                          each tile the class counts left of it
//      3. tileEstimate / segmentMax / tileCandidate / tileExact
//                          each tile sweeps its cuts: first in single
//                          precision, then exactly (the CPU's double-precision
//                          function) for the cuts that can still be the best
//                          (with few classes, from the few the first sweep
//                          kept). On GPUs with fast double precision, one
//                          exact sweep.
//      4. segmentBest      best cut per segment and the class counts left of
//                          it -> host picks the split (SplitRules) and the
//                          children
//      5. markGoesLeft / partition:
//                          stable partition of the columns of split nodes, each
//                          into a free slot, in one pass, skipped for nodes
//                          whose children are both leaves
//    so each level needs one round trip to the host (step 4).
//  * Children with fewer than minRows_ (gpu.minRows) rows are gathered into one block,
//    copied to the host (through the ring) and grown by CpuTreeBuilder tasks
//    on the thread pool while the GPU continues with the next level.
//  * A grower used for cross-validation keeps the raw feature values on the
//    device and builds each fold's sorted columns there.

#include "build/cpu_builder.h"
#include "build/grower.h"
#include "core/nvtx.h"
#include "core/timing.h"
#include "core/trace.h"

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <numeric>
#include <pthread.h>
#include <sched.h>
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
  float filterScale = 0.0f;       // see floatGainFilter()
  float filterMargin = 0.0f;
};

struct ScanParams {
  EntryCodec codec;
  int classCount;
  Criterion criterion;
  double minGap;
  LogTable logs;          // device copy of the host table
};

// Single-precision estimate of cutGain, used only to rule out cuts that
// cannot be the best one (consumer GPUs run double precision ~64x slower).
template <int MAXK>
__device__ __forceinline__ float floatCutGain(const std::uint32_t *total,
                                              const std::uint32_t *left, int classCount,
                                              std::uint32_t n, std::uint32_t nLeft,
                                              float parentWeighted, Criterion criterion) {
  const std::uint32_t nRight = n - nLeft;
  float leftWeighted;
  float rightWeighted;
  if (criterion == Criterion::Gini) {
    // gain = sum_k (l_k nR - r_k nL)^2 / (n^2 nL nR), where
    // l_k nR - r_k nL = l_k n - t_k nL is exact in 64 bits (its magnitude is
    // at most nL nR < 2^62): no term cancels another, so the estimate is
    // within a small relative error of the gain (floatGainFilter).
    const auto difference = [&](int k) {
      const unsigned long long leftTimesRows = static_cast<unsigned long long>(left[k]) * n;
      const unsigned long long totalTimesLeft = static_cast<unsigned long long>(total[k]) * nLeft;
      return static_cast<float>(static_cast<long long>(leftTimesRows - totalTimesLeft));
    };
    float squares = 0.0f;
    if (MAXK == 2 && classCount == 2) { // the second difference is minus the first
      const float d = difference(0);
      squares = 2.0f * (d * d);
    } else {
#pragma unroll
      for (int k = 0; k < MAXK; ++k) {
        if (k < classCount) {
          const float d = difference(k);
          squares += d * d;
        }
      }
    }
    const float rows = static_cast<float>(n);
    return squares / (rows * rows * static_cast<float>(nLeft) * static_cast<float>(nRight));
  } else {
    // m * entropy of a side as sum_k c_k * log2(m / c_k): every term is at
    // most its count times log2(m), never a difference of terms of size
    // m * log2(m), so single precision stays accurate (floatGainMargin).
    const float leftRows = static_cast<float>(nLeft);
    const float rightRows = static_cast<float>(nRight);
    leftWeighted = 0.0f;
    rightWeighted = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      if (k < classCount) {
        const std::uint32_t l = left[k];
        const std::uint32_t r = total[k] - left[k];
        if (l > 0) {
          leftWeighted += static_cast<float>(l) * log2f(leftRows / static_cast<float>(l));
        }
        if (r > 0) {
          rightWeighted += static_cast<float>(r) * log2f(rightRows / static_cast<float>(r));
        }
      }
    }
  }
  return (parentWeighted - leftWeighted - rightWeighted) / static_cast<float>(n);
}

// Cuts whose float gain is below filterCutoff(best float gain of the segment)
// cannot be the best cut, or tie with it, so only the others are scored in
// double precision. The cutoff is best - (best * scale + margin), with a
// generous bound B on |floatCutGain - cutGain| over the node's cuts: scale and
// margin hold 2 B + kTieEps (a cut can be best only if its gain reaches the
// best one's minus kTieEps, and each of the two estimates may be off by B).
//   Gini: the estimate is within (K + 10) 2^-24 of the gain relatively (to
//   first order: 3 roundings per square, K - 1 additions of non-negative
//   terms, 7 for the denominator, 1 for the division); the gains, and so
//   their error, are at most best / (1 - that). cutGain's own error is a few
//   double roundings on each of the ~2K+4 terms, each at most n, divided by
//   n. Twice these bounds, and 1% more, cover the cutoff's float roundings.
//   Entropy: a term c * log2f(m / c) is within c * (0.9 + 1.8 log2(m / c)) *
//   1e-7 of c * log2(m / c) (the division and the product are correctly
//   rounded, log2f is within 1 ulp), so a side's sum is within m * (0.9 +
//   1.8 log2 K + 0.6 (K - 1) log2 K) * 1e-7 with the additions; the
//   subtractions from the parent's value (at most n log2 K) and the division
//   add 2.4e-7 log2 K. Counts of 2^24 and more are not exact in float: twice
//   the bound.
void floatGainFilter(int classCount, std::uint32_t n, Criterion criterion, float &scale,
                     float &margin) {
  if (criterion == Criterion::Gini) {
    const double relative = 2.0 * (classCount + 10) * std::ldexp(1.0, -24);
    const double doubleBound = 2.0 * 4.0 * (2.0 * classCount + 8.0) * std::ldexp(1.0, -52);
    scale = static_cast<float>(1.01 * 2.0 * relative / (1.0 - relative));
    margin = static_cast<float>(1.01 * (2.0 * doubleBound + kTieEps));
    return;
  }
  const double classBits = std::max(1.0, std::log2(double(classCount)));
  double bound = 4.0 * (1.0 + 2.0 * classBits + 0.6 * (classCount - 1) * classBits) * 1e-7;
  if (n >= (1u << 24)) {
    bound *= 2.0;
  }
  scale = 0.0f;
  margin = static_cast<float>(2.0 * bound + kTieEps);
}

__device__ __forceinline__ float filterCutoff(float best, const NodeDesc &node) {
  return best - fmaxf(best, 0.0f) * node.filterScale - node.filterMargin;
}

struct CopyJob {
  unsigned long long from;
  unsigned long long to;
  std::uint32_t count;
};

// The rows of `count` entries from `from` (one column of a node) are in leaf `leaf`.
struct LeafJob {
  unsigned long long from;
  std::uint32_t count;
  std::uint32_t leaf;
};

// Best cut found by part of the block, with tie-breaking on position.
struct BestCut {
  double gain;
  std::uint32_t position;
  float leftValue;
  float rightValue;
};

constexpr BestCut kNoCut{-INFINITY, 0xFFFFFFFFu, 0.0f, 0.0f};

struct Max {
  template <typename T> __device__ T operator()(T a, T b) const { return a > b ? a : b; }
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
// The unsorted column of one feature: entry i is row rows[i] (or row i if
// rows is null), with its value from `values` (the feature's raw column).
__global__ void gatherKernel(const float *values, const std::uint32_t *rows, std::uint32_t count,
                             const std::uint16_t *labels, EntryCodec codec, Entry *column) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t row = rows ? rows[index] : index;
  column[index] = {values[row] + 0.0f, codec.pack(row, labels[row])}; // -0.0 -> +0.0, as on the CPU
}

// The columns are sorted by value alone (radix sort, stable: equal values
// stay in row order).
struct EntryValue {
  __host__ __device__ cuda::std::tuple<float &> operator()(Entry &entry) const {
    return {entry.value};
  }
};

// The bit pattern of an entry's value (distinct values: cub's unique on them).
struct ValueBits {
  __device__ std::uint32_t operator()(const Entry &entry) const {
    return __float_as_uint(entry.value);
  }
};

// The distinct values of the features, back to back in a pool, as far as
// they fit: where feature `feature`'s go (kNotPlaced once one did not fit).
constexpr unsigned long long kNotPlaced = ~0ull;
__global__ void placeDistinctKernel(const int *counts, std::uint32_t feature,
                                    unsigned long long capacity, unsigned long long *used,
                                    unsigned long long *placement) {
  const unsigned long long count = static_cast<unsigned long long>(counts[feature]);
  if (*used != kNotPlaced && *used + count <= capacity) {
    placement[feature] = *used;
    *used += count;
  } else {
    placement[feature] = kNotPlaced;
    *used = kNotPlaced;
  }
}

__global__ void copyDistinctKernel(const std::uint32_t *unique, const int *counts,
                                   std::uint32_t feature, const unsigned long long *placement,
                                   std::uint32_t *pool) {
  const unsigned long long at = placement[feature];
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (at != kNotPlaced && index < static_cast<std::uint32_t>(counts[feature])) {
    pool[at + index] = unique[index];
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
// Also counts the tile's cuts (C4.5's "tries"; distinct values within the
// node's allowed cut positions), so that the sweeps can skip whole tiles. A
// segment of one tile needs neither (tileEstimate counts its cuts).
template <int MAXK>
__global__ void __launch_bounds__(kThreads)
    tileHistogramKernel(const Entry *src, const Segment *segments,
                        const std::uint32_t *tileSegment, const std::uint8_t *constant,
                        const NodeDesc *nodes, ScanParams params, std::uint32_t *tileHist,
                        std::uint32_t *tileTries) {
  __shared__ std::uint32_t histogram[MAXK];
  __shared__ std::uint32_t tries;
  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t segmentIndex = tileSegment[tile];
  if (constant[segmentIndex]) {
    if (threadIdx.x == 0) {
      tileTries[tile] = 0;
    }
    return;
  }
  const Segment segment = segments[segmentIndex];
  const int classCount = params.classCount;
  if (segment.tileCount == 1) {
    // Nothing before the tile; tileEstimate counts its cuts.
    if (threadIdx.x < classCount) {
      tileHist[static_cast<std::size_t>(tile) * classCount + threadIdx.x] = 0;
    }
    return;
  }
  const NodeDesc node = nodes[segment.node];
  std::uint32_t start;
  std::uint32_t length;
  tileRange(segment, tile, start, length);
  const Entry *column = src + segment.base;
  const Entry *entries = column + start;

  if (threadIdx.x < MAXK) {
    histogram[threadIdx.x] = 0;
  }
  if (threadIdx.x == 0) {
    tries = 0;
  }
  __syncthreads();

  // Each warp counts 32 entries per step with one ballot per class.
  const unsigned full = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  std::uint32_t counts[MAXK];
#pragma unroll
  for (int k = 0; k < MAXK; ++k) {
    counts[k] = 0;
  }
  std::uint32_t cuts = 0;
  for (std::uint32_t base = threadIdx.x - lane; base < length; base += kThreads) {
    const std::uint32_t index = base + lane;
    const std::uint32_t position = start + index; // cut between position - 1 and position
    Entry entry{0.0f, 0xFFFFFFFFu};
    if (index < length) {
      entry = entries[index];
    }
    const std::uint32_t cls = index < length ? params.codec.cls(entry.packed) : 0xFFFFFFFFu;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      if (k < classCount) {
        counts[k] += __popc(__ballot_sync(full, cls == static_cast<std::uint32_t>(k)));
      }
    }
    float previous = __shfl_up_sync(full, entry.value, 1);
    if (lane == 0 && index < length && position >= 1) {
      previous = column[position - 1].value;
    }
    const bool isCutHere = index < length && position >= node.minChild &&
                           position + node.minChild <= segment.count &&
                           isCut(previous, entry.value, params.minGap);
    cuts += __popc(__ballot_sync(full, isCutHere));
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      if (k < classCount && counts[k] > 0) {
        atomicAdd(&histogram[k], counts[k]);
      }
    }
    if (cuts > 0) {
      atomicAdd(&tries, cuts);
    }
  }
  __syncthreads();
  if (threadIdx.x < classCount) {
    tileHist[static_cast<std::size_t>(tile) * classCount + threadIdx.x] = histogram[threadIdx.x];
  }
  if (threadIdx.x == 0) {
    tileTries[tile] = tries;
  }
}

// ---------------------------------------------------------------------------
// 2b. Two classes: an upper bound of every tile's gains, and the exact gain of
//     one cut of the tile (a lower bound of the segment's best).
// ---------------------------------------------------------------------------
// The gain is convex in the left class counts (the impurities are concave in
// a child's counts), and the tile's cuts have left counts in the box between
// the counts before the tile and those after it, so the gains of all its cuts
// are at most the largest of the box's four corners. A tile whose bound is
// kTileBoundGap below the exact gain of another cut of the segment holds no
// cut that can be the segment's best, nor one that can tie with it (the
// reduction's near-ties are kTieEps = 1e-12 wide, so its choice among the
// other cuts cannot drift that far), so the sweeps skip it.
constexpr double kTileBoundGap = 1e-8;

__global__ void tileBoundKernel(const Entry *src, const Segment *segments, std::uint32_t tileCount,
                                const std::uint32_t *tileSegment, const std::uint8_t *constant,
                                const NodeDesc *nodes, ScanParams params,
                                const std::uint32_t *tilePrefix,
                                const std::uint32_t *nodeCounts, double *tileUpper,
                                double *tileLower) {
  const std::uint32_t tile = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile >= tileCount) {
    return;
  }
  const std::uint32_t segmentIndex = tileSegment[tile];
  tileUpper[tile] = INFINITY;
  tileLower[tile] = -INFINITY;
  if (constant[segmentIndex]) {
    return;
  }
  const Segment segment = segments[segmentIndex];
  if (segment.tileCount == 1) {
    return; // the segment's best cut is in this tile
  }
  const NodeDesc node = nodes[segment.node];
  const std::uint32_t count = segment.count;
  std::uint32_t start;
  std::uint32_t length;
  tileRange(segment, tile, start, length);
  const std::uint32_t total[2] = {nodeCounts[2 * std::size_t{segment.node}],
                                  nodeCounts[2 * std::size_t{segment.node} + 1]};
  const std::uint32_t before[2] = {tilePrefix[2 * std::size_t{tile}],
                                   tilePrefix[2 * std::size_t{tile} + 1]};
  const bool last = tile + 1 == segment.firstTile + segment.tileCount;
  const std::uint32_t after[2] = {last ? total[0] : tilePrefix[2 * std::size_t{tile} + 2],
                                  last ? total[1] : tilePrefix[2 * std::size_t{tile} + 3]};
  const std::uint32_t first = max(start, node.minChild);
  if (count < 2 * node.minChild || first > min(start + length - 1, count - node.minChild)) {
    tileUpper[tile] = -INFINITY; // no allowed cut in this tile
    return;
  }
  // The gain at a corner (0 if a side is empty: the extension that keeps it
  // convex).
  auto gainAt = [&](std::uint32_t left0, std::uint32_t left1) {
    const std::uint32_t left[2] = {left0, left1};
    const std::uint32_t leftCount = left0 + left1;
    if (leftCount == 0 || leftCount >= count) {
      return 0.0;
    }
    return cutGain(total, left, 2, count, leftCount, node.parentWeighted, params.criterion,
                   params.logs);
  };
  tileUpper[tile] = fmax(fmax(gainAt(before[0], before[1]), gainAt(after[0], after[1])),
                         fmax(gainAt(after[0], before[1]), gainAt(before[0], after[1])));
  // The tile's first cut, if the sweeps score it (as sweepTile decides).
  if (first != start || start < 1) {
    return;
  }
  const Entry *column = src + segment.base;
  const Entry previous = column[start - 1];
  const Entry current = column[start];
  if (!isCut(previous.value, current.value, params.minGap)) {
    return;
  }
  const std::uint32_t previousClass = params.codec.cls(previous.packed);
  if (previousClass == params.codec.cls(current.packed)) {
    const bool leftSingleton =
        start < 2 || isCut(column[start - 2].value, previous.value, params.minGap);
    const bool rightSingleton =
        start + 1 >= count || isCut(current.value, column[start + 1].value, params.minGap);
    if (isSkippableCut(true, leftSingleton, rightSingleton, start, count, node.minChild)) {
      return;
    }
  }
  tileLower[tile] = cutGain(total, before, 2, count, start, node.parentWeighted,
                            params.criterion, params.logs);
}

// Whether tileBoundKernel ruled out every cut of the tile.
__device__ __forceinline__ bool tileRuledOut(const double *tileUpper, const double *segmentLower,
                                             std::uint32_t tile, std::uint32_t segment) {
  return tileUpper && tileUpper[tile] < segmentLower[segment] - kTileBoundGap;
}

// ---------------------------------------------------------------------------
// 2. Exclusive scan over the tiles of each segment (one block per segment),
// `width` values per tile.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads)
    segmentPrefixKernel(std::uint32_t *perTile, int width, const Segment *segments,
                        const std::uint8_t *constant) {
  using Scan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename Scan::TempStorage scanStorage;
  const Segment segment = segments[blockIdx.x];
  if (constant[blockIdx.x]) {
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
//       With at most 8 classes, tileCandidate scores the cuts tileEstimate
//       kept instead, and tileExact sweeps only the tiles that had too many.
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

// With few classes, tileEstimate also keeps the cuts of the tile whose
// estimate is within the float error of the tile's best one (a superset of
// those within that error of the segment's best) when there are at most
// kCandidates of them, so tileCandidate scores them without sweeping the tile
// again; tileExact sweeps only the tiles with more of them.
constexpr std::uint32_t kCandidates = 8;
constexpr int kMaxCandidateClasses = 8;

template <int MAXK> struct alignas(8) Candidate {
  float estimate;
  std::uint32_t position;
  float leftValue;
  float rightValue;
  std::uint32_t left[MAXK]; // class counts left of the cut
};

// Load a tile and work out the class counts before each warp's entries.
// Uses `warpCounts` and `totals` (shared) and one __syncthreads.
template <int MAXK>
__device__ __forceinline__ void loadTile(TileSweep<MAXK> &sweep, const Entry *src,
                                         const Segment &segment, std::uint32_t segmentIndex,
                                         std::uint32_t tile, const ScanParams &params,
                                         const std::uint32_t *tilePrefix,
                                         const std::uint32_t *nodeCounts,
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
        nodeCounts[static_cast<std::size_t>(segment.node) * classCount + threadIdx.x];
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
                                        parentWeighted, params.criterion);
        }
        onCut(step, position, leftHere, previous, estimate);
      }
    }
  }
  return tries;
}

// Write the tile's cuts whose estimate reaches `cutoff` to `slots`, in
// position order, if there are at most kCandidates of them; their number
// goes to *count either way. Bit `step` of `scored` says whether this thread
// scored a cut at that step, and estimates[step * kThreads + threadIdx.x]
// (shared, written by this thread) is then its estimate. Uses
// `warpCandidates` (shared) and one __syncthreads.
template <int MAXK>
__device__ __forceinline__ void keepCandidates(const TileSweep<MAXK> &sweep,
                                               const ScanParams &params, unsigned scored,
                                               const float *estimates, float cutoff,
                                               Candidate<MAXK> *slots, std::uint32_t *count,
                                               std::uint32_t *warpCandidates) {
  const unsigned full = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const unsigned lanesBelow = (1u << lane) - 1u;
  unsigned mine = 0; // bit per step
#pragma unroll
  for (int step = 0; step < kWarpSteps; ++step) {
    if (((scored >> step) & 1u) && !(estimates[step * kThreads + threadIdx.x] < cutoff)) {
      mine |= 1u << step;
    }
  }
  std::uint32_t warpCount = 0;
  if (__any_sync(full, mine != 0)) {
#pragma unroll
    for (int step = 0; step < kWarpSteps; ++step) {
      warpCount += __popc(__ballot_sync(full, (mine >> step) & 1u));
    }
  }
  if (lane == 0) {
    warpCandidates[warp] = warpCount;
  }
  __syncthreads();
  std::uint32_t slot = 0;
  std::uint32_t total = 0;
#pragma unroll
  for (int other = 0; other < kWarps; ++other) {
    const std::uint32_t n = warpCandidates[other];
    slot += other < warp ? n : 0;
    total += n;
  }
  if (threadIdx.x == 0) {
    *count = total;
  }
  if (total > kCandidates || warpCount == 0) {
    return;
  }
  std::uint32_t left[MAXK];
#pragma unroll
  for (int k = 0; k < MAXK; ++k) {
    left[k] = sweep.warpLeft[k];
  }
  // The tile's entries are read again (from cache): keeping them in registers
  // would cost the sweep its occupancy.
  for (int step = 0; step < kWarpSteps; ++step) {
    const std::uint32_t index = warp * (kWarpSteps * 32) + step * 32 + lane;
    float value = 0.0f;
    std::uint32_t cls = 0xFFFFFFFFu;
    if (index < sweep.length) {
      const Entry entry = sweep.column[sweep.start + index];
      value = entry.value;
      cls = params.codec.cls(entry.packed);
    }
    float previous = __shfl_up_sync(full, value, 1);
    std::uint32_t leftHere[MAXK];
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
      const unsigned votes = __ballot_sync(full, cls == static_cast<std::uint32_t>(k));
      leftHere[k] = left[k] + __popc(votes & lanesBelow);
      left[k] += __popc(votes);
    }
    const unsigned kept = __ballot_sync(full, (mine >> step) & 1u);
    if ((mine >> step) & 1u) {
      const std::uint32_t position = sweep.start + index; // >= minChild >= 1
      if (lane == 0) {
        previous = sweep.column[position - 1].value;
      }
      Candidate<MAXK> &candidate = slots[slot + __popc(kept & lanesBelow)];
      candidate.estimate = estimates[step * kThreads + threadIdx.x];
      candidate.position = position;
      candidate.leftValue = previous;
      candidate.rightValue = value;
#pragma unroll
      for (int k = 0; k < MAXK; ++k) {
        candidate.left[k] = leftHere[k];
      }
    }
    slot += __popc(kept);
  }
}

// Also counts the cuts of single-tile segments (tileHistogram skipped them).
// With `candidates`, also keeps the tile's best cuts (keepCandidates).
template <int MAXK>
__global__ void __launch_bounds__(kThreads)
    tileEstimateKernel(const Entry *src, const Segment *segments,
                       const std::uint32_t *tileSegment, const std::uint8_t *constant,
                       const NodeDesc *nodes, ScanParams params, const std::uint32_t *tilePrefix,
                       const std::uint32_t *nodeCounts, const double *tileUpper,
                       const double *segmentLower, float *tileMax, std::uint32_t *tileTries,
                       Candidate<MAXK> *candidates, std::uint32_t *candidateCount) {
  using FloatReduce = cub::BlockReduce<float, kThreads>;
  using SumReduce = cub::BlockReduce<std::uint32_t, kThreads>;
  __shared__ union {
    typename FloatReduce::TempStorage max;
    typename SumReduce::TempStorage sum;
  } storage;
  __shared__ std::uint32_t totals[MAXK];
  __shared__ std::uint32_t warpCounts[kWarps][MAXK];
  __shared__ std::uint32_t warpCandidates[kWarps];
  __shared__ float sharedMax;
  // Few classes: each thread's estimate per step, for keepCandidates.
  __shared__ float estimates[MAXK <= kMaxCandidateClasses ? kTile : 1];

  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  const bool single = segment.tileCount == 1;
  if (constant[segmentIndex] || tileRuledOut(tileUpper, segmentLower, tile, segmentIndex)) {
    if (threadIdx.x == 0) {
      tileMax[tile] = -INFINITY;
      if (single) {
        tileTries[tile] = 0;
      }
    }
    return;
  }
  const NodeDesc node = nodes[segment.node];
  TileSweep<MAXK> sweep;
  loadTile<MAXK>(sweep, src, segment, segmentIndex, tile, params, tilePrefix, nodeCounts,
                 warpCounts, totals);
  float best = -INFINITY;
  const bool keep = MAXK <= kMaxCandidateClasses && candidates;
  unsigned scored = 0; // keep: bit per step
  const std::uint32_t tries = sweepTile<MAXK, true>(
      sweep, segment, node, params, totals,
      [&](int step, std::uint32_t, const std::uint32_t *, float, float estimate) {
        best = fmaxf(best, estimate);
        if (keep) {
          estimates[step * kThreads + threadIdx.x] = estimate;
          scored |= 1u << step;
        }
      });
  const float tileBest = FloatReduce(storage.max).Reduce(best, Max());
  if (threadIdx.x == 0) {
    tileMax[tile] = tileBest;
    sharedMax = tileBest;
  }
  if constexpr (MAXK <= kMaxCandidateClasses) {
    if (candidates) {
      __syncthreads();
      const float localMax = sharedMax;
      if (localMax == -INFINITY) {
        if (threadIdx.x == 0) {
          candidateCount[tile] = 0;
        }
      } else {
        keepCandidates<MAXK>(sweep, params, scored, estimates, filterCutoff(localMax, node),
                             candidates + static_cast<std::size_t>(tile) * kCandidates,
                             candidateCount + tile, warpCandidates);
      }
    }
  }
  if (single) {
    __syncthreads();
    const std::uint32_t tileTotalTries = SumReduce(storage.sum).Sum(tries);
    if (threadIdx.x == 0) {
      tileTries[tile] = tileTotalTries;
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    segmentMaxKernel(const Segment *segments, const T *tileMax, T *segmentMax) {
  using Reduce = cub::BlockReduce<T, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  const Segment segment = segments[blockIdx.x];
  T best = -INFINITY;
  for (std::uint32_t tile = threadIdx.x; tile < segment.tileCount; tile += kThreads) {
    best = Max()(best, tileMax[segment.firstTile + tile]);
  }
  const T segmentBest = Reduce(storage).Reduce(best, Max());
  if (threadIdx.x == 0) {
    segmentMax[blockIdx.x] = segmentBest;
  }
}


// The tile's best cut in double precision, if it may be the segment's best.
// Whether the tile is skipped depends only on the tile (the whole block
// returns early together).
template <int MAXK, bool FILTER>
__device__ __forceinline__ void exactTile(std::uint32_t tile, const Entry *src,
                                          const Segment *segments,
                                          const std::uint32_t *tileSegment,
                                          const std::uint8_t *constant, const NodeDesc *nodes,
                                          const ScanParams &params,
                                          const std::uint32_t *tilePrefix,
                                          const std::uint32_t *nodeCounts,
                                          const double *tileUpper, const double *segmentLower,
                                          const float *tileMax, const float *segmentMax,
                                          BestCut *tileBest, std::uint32_t *tileTries) {
  using CutReduce = cub::BlockReduce<BestCut, kThreads>;
  using SumReduce = cub::BlockReduce<std::uint32_t, kThreads>;
  __shared__ union {
    typename CutReduce::TempStorage cut;
    typename SumReduce::TempStorage sum;
  } storage;
  __shared__ std::uint32_t totals[MAXK];
  __shared__ std::uint32_t warpCounts[kWarps][MAXK];

  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  const bool single = segment.tileCount == 1;
  const NodeDesc node = nodes[segment.node];
  float cutoff = -INFINITY;
  bool skip = constant[segmentIndex] || tileRuledOut(tileUpper, segmentLower, tile, segmentIndex);
  if (FILTER && !skip) {
    cutoff = filterCutoff(segmentMax[segmentIndex], node);
    skip = tileMax[tile] < cutoff; // no cut of this tile can be the best one
  }
  if (skip) {
    if (threadIdx.x == 0) {
      tileBest[tile] = kNoCut;
      if (!FILTER && single) {
        tileTries[tile] = 0;
      }
    }
    return;
  }
  TileSweep<MAXK> sweep;
  loadTile<MAXK>(sweep, src, segment, segmentIndex, tile, params, tilePrefix, nodeCounts,
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
  if (!FILTER && single) {
    __syncthreads();
    const std::uint32_t tileTotalTries = SumReduce(storage.sum).Sum(tries);
    if (threadIdx.x == 0) {
      tileTries[tile] = tileTotalTries;
    }
  }
}

// FILTER: two-pass mode (only cuts whose estimate reaches the cutoff are
// scored). Otherwise every cut is scored, and the cuts of single-tile
// segments are counted here. With `tileList`, the blocks loop over the
// *listed tiles instead (those tileCandidate left).
template <int MAXK, bool FILTER>
__global__ void __launch_bounds__(kThreads)
    tileExactKernel(const Entry *src, const Segment *segments, const std::uint32_t *tileSegment,
                    const std::uint8_t *constant, const NodeDesc *nodes, ScanParams params,
                    const std::uint32_t *tilePrefix, const std::uint32_t *nodeCounts,
                    const double *tileUpper, const double *segmentLower, const float *tileMax,
                    const float *segmentMax, BestCut *tileBest, std::uint32_t *tileTries,
                    const std::uint32_t *tileList, const std::uint32_t *listed) {
  if (!tileList) {
    exactTile<MAXK, FILTER>(blockIdx.x, src, segments, tileSegment, constant, nodes, params,
                            tilePrefix, nodeCounts, tileUpper, segmentLower, tileMax, segmentMax,
                            tileBest, tileTries);
    return;
  }
  // loadTile's __syncthreads keeps one tile's shared memory from the next.
  const std::uint32_t count = *listed;
  for (std::uint32_t i = blockIdx.x; i < count; i += gridDim.x) {
    exactTile<MAXK, FILTER>(tileList[i], src, segments, tileSegment, constant, nodes, params,
                            tilePrefix, nodeCounts, tileUpper, segmentLower, tileMax, segmentMax,
                            tileBest, tileTries);
  }
}

// Two-pass mode with few classes: one warp per tile scores the candidates
// tileEstimate kept (in double precision) and lists the tiles that had too
// many for tileExact.
template <int MAXK>
__global__ void __launch_bounds__(kThreads)
    tileCandidateKernel(std::uint32_t tiles, const Segment *segments,
                        const std::uint32_t *tileSegment, const std::uint8_t *constant,
                        const NodeDesc *nodes, ScanParams params, const std::uint32_t *nodeCounts,
                        const double *tileUpper, const double *segmentLower, const float *tileMax,
                        const float *segmentMax, const Candidate<MAXK> *candidates,
                        const std::uint32_t *candidateCount, BestCut *tileBest,
                        std::uint32_t *tileList, std::uint32_t *listed) {
  const unsigned full = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  const std::uint32_t tile = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (tile >= tiles) {
    return;
  }
  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  const NodeDesc node = nodes[segment.node];
  float cutoff = -INFINITY;
  bool skip = constant[segmentIndex] || tileRuledOut(tileUpper, segmentLower, tile, segmentIndex);
  if (!skip) {
    cutoff = filterCutoff(segmentMax[segmentIndex], node);
    skip = tileMax[tile] < cutoff; // no cut of this tile can be the best one
  }
  if (skip) {
    if (lane == 0) {
      tileBest[tile] = kNoCut;
    }
    return;
  }
  // tileEstimate kept the cuts reaching the tile's own cutoff, which is at
  // most this one (checked: if not, the tile is swept again).
  const std::uint32_t count = candidateCount[tile];
  if (count > kCandidates || filterCutoff(tileMax[tile], node) > cutoff) {
    if (lane == 0) {
      tileList[atomicAdd(listed, 1u)] = tile;
    }
    return;
  }
  BestCut best = kNoCut;
  if (static_cast<std::uint32_t>(lane) < count) {
    const Candidate<MAXK> candidate =
        candidates[static_cast<std::size_t>(tile) * kCandidates + lane];
    if (!(candidate.estimate < cutoff)) {
      std::uint32_t totals[MAXK];
      const std::uint32_t *counts =
          nodeCounts + static_cast<std::size_t>(segment.node) * params.classCount;
#pragma unroll
      for (int k = 0; k < MAXK; ++k) {
        totals[k] = k < params.classCount ? counts[k] : 0;
      }
      const double gain = cutGain(totals, candidate.left, params.classCount, segment.count,
                                  candidate.position, node.parentWeighted, params.criterion,
                                  params.logs);
      best = {gain, candidate.position, candidate.leftValue, candidate.rightValue};
    }
  }
  // Candidates are in position order: reduce in a fixed order.
#pragma unroll
  for (int offset = kCandidates / 2; offset > 0; offset >>= 1) {
    BestCut other;
    other.gain = __shfl_down_sync(full, best.gain, offset);
    other.position = __shfl_down_sync(full, best.position, offset);
    other.leftValue = __shfl_down_sync(full, best.leftValue, offset);
    other.rightValue = __shfl_down_sync(full, best.rightValue, offset);
    best = BetterCut()(best, other);
  }
  if (lane == 0) {
    tileBest[tile] = best;
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
                       EntryCodec codec, std::uint32_t *goesLeft) {
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
  // Only the left rows: the bits start at zero on every level.
  const std::uint32_t end = min(length, node.leftCount > start ? node.leftCount - start : 0u);
  for (std::uint32_t index = threadIdx.x; index < end; index += kThreads) {
    const std::uint32_t row = codec.row(column[start + index].packed);
    atomicOr(goesLeft + (row >> 5), 1u << (row & 31));
  }
}

// Single pass (decoupled look-back, as in CUB's scans): each tile counts its
// left rows and publishes the count, then adds up the counts of the tiles
// before it in its segment, walking back to a tile that has published its
// running total, and publishes its own. Tiles are taken in order from a
// counter (not by blockIdx), so a tile only ever waits for tiles that have
// started. The tile is staged in shared memory, left rows first, so that both
// runs are written with consecutive stores.
//
// The partition also does step 1 of the next level for the children that
// stay on the GPU: it knows where every entry lands in the child's column, so
// it adds up the class histogram and the cuts ("tries") of every next-level
// tile, instead of a tileHistogram pass over them. A cut at the start of a
// run lies between two runs; the tiles also publish the last value of each
// side, so the walk back finds the entry before the run too.
//
// Tile states start at zero. Every word carries its own flag, so a tile can
// wait for any of them, in any order, with no memory fences.
constexpr unsigned long long kTileLefts = 1ull << 32; // left rows of the tile
constexpr unsigned long long kTileTotal = 2ull << 32; // ... of the segment up to the tile
constexpr unsigned long long kValue = 1ull << 32;     // a value (float bits)
constexpr unsigned long long kNoValue = 2ull << 32;   // no row of that side
struct TileState {
  unsigned long long lefts;       // kTileLefts or kTileTotal | count
  unsigned long long last[2];     // last value of each side in the tile (if any)
  unsigned long long lastUpTo[2]; // ... in the segment up to the tile
};
constexpr std::uint32_t kNoTarget = 0xFFFFFFFFu;

// Where the partition of a segment sends its runs (side 0: left, 1: right):
// the next-level segment of the child for the same feature.
//
// The columns live in slots of one buffer with a few more slots than columns
// (GpuGrower::slot_). The partition moves every column it writes to a free
// slot: the free ones first, then the slots that earlier columns leave. A
// node's rows keep their range in every column, so in a vacated slot a
// segment only overwrites the same node's segment of the column that left
// it: a tile writes once every tile of that segment has loaded its entries,
// which the segment's last tile tells by publishing its running total (each
// tile publishes its count right after its load, and the total needs all of
// them). Within a node the segments are in column order, so the tiles it
// waits for have started before it.
struct PartitionTarget {
  std::uint32_t firstTile[2]; // its first tile, kNoTarget if the child leaves the GPU
  std::uint32_t count[2];     // the child's rows
  std::uint32_t minChild[2];  // and its NodeDesc::minChild
};

// Where the partition of a segment writes (a struct of its own: in
// PartitionTarget, which is indexed by side, it pushed that to local memory).
struct PartitionMove {
  long long shift;        // the column's new slot minus its old one (entries)
  std::uint32_t waitTile; // last tile of the segment that leaves the new range, or kNoTarget
};

__device__ __forceinline__ unsigned long long waitFor(const unsigned long long *word) {
  unsigned long long value;
  do {
    value = *reinterpret_cast<const volatile unsigned long long *>(word);
  } while (value == 0);
  return value;
}

__device__ __forceinline__ void waitForTotal(const unsigned long long *word) {
  while (*reinterpret_cast<const volatile unsigned long long *>(word) < kTileTotal) {
  }
}

__device__ __forceinline__ float valueOf(unsigned long long word) {
  return __uint_as_float(static_cast<std::uint32_t>(word));
}

__global__ void __launch_bounds__(kThreads)
    partitionKernel(const Entry *src, Entry *dst, const Segment *segments,
                    const std::uint32_t *tileSegment, const std::uint8_t *constant,
                    const NodeDesc *nodes, ScanParams params, const std::uint32_t *goesLeft,
                    const PartitionTarget *targets, const PartitionMove *moves,
                    TileState *tileStates, unsigned int *tileCounter, std::uint32_t *nextHist,
                    std::uint32_t *nextTries) {
  using Load = cub::BlockLoad<Entry, kThreads, kItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using Scan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ union {
    typename Load::TempStorage load;
    typename Scan::TempStorage scan;
    Entry staged[kTile];
  } storage;
  __shared__ std::uint32_t histogram[2][2][kMaxGpuClasses]; // [side][next tile][class]
  __shared__ std::uint32_t tries[2][2];
  __shared__ std::uint32_t shared;
  __shared__ long long shift;       // PartitionMove::shift
  __shared__ float previousLast[2]; // last value of each side before this tile

  if (threadIdx.x == 0) {
    shared = atomicAdd(tileCounter, 1u);
  }
  __syncthreads();
  const std::uint32_t tile = shared;
  const std::uint32_t segmentIndex = tileSegment[tile];
  const Segment segment = segments[segmentIndex];
  if (constant[segmentIndex] || !nodes[segment.node].partition) {
    return;
  }
  const std::uint32_t leftCount = nodes[segment.node].leftCount;
  const PartitionTarget target = targets[segmentIndex];
  // Only thread 0 reads the move (read by every thread, or waited for by
  // another one, it made the kernel 3% slower), early: the tile's load
  // hides it.
  PartitionMove move{0, kNoTarget};
  if (threadIdx.x == 0) {
    move = moves[segmentIndex];
  }
  const int classCount = params.classCount;
  std::uint32_t start;
  std::uint32_t length;
  tileRange(segment, tile, start, length);
  for (std::uint32_t index = threadIdx.x; index < 4 * kMaxGpuClasses; index += kThreads) {
    (&histogram[0][0][0])[index] = 0;
  }
  if (threadIdx.x < 4) {
    (&tries[0][0])[threadIdx.x] = 0;
  }

  Entry items[kItems];
  Load(storage.load).Load(src + segment.base + start, items, length);
  const std::uint32_t first = threadIdx.x * kItems;
  bool left[kItems];
  std::uint32_t lefts = 0;
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const std::uint32_t row = params.codec.row(items[j].packed);
    left[j] = first + j < length && ((goesLeft[row >> 5] >> (row & 31)) & 1u);
    lefts += left[j];
  }
  __syncthreads(); // the scan reuses the load's shared memory (and `shared`)
  std::uint32_t leftsBefore; // in the tile, before my first entry
  std::uint32_t tileLefts;
  Scan(storage.scan).ExclusiveSum(lefts, leftsBefore, tileLefts);
  const std::uint32_t tileRights = length - tileLefts;
  const bool firstTile = tile == segment.firstTile;
  TileState *state = tileStates + tile;

  // Publish: the count (thread 0), then the last value of each side (the
  // thread that holds it).
  if (threadIdx.x == 0) {
    atomicExch(&state->lefts, (firstTile ? kTileTotal : kTileLefts) | tileLefts);
    if (firstTile && tileLefts == 0) {
      atomicExch(&state->lastUpTo[0], kNoValue);
    }
    if (firstTile && tileRights == 0) {
      atomicExch(&state->lastUpTo[1], kNoValue);
    }
  }
  const std::uint32_t valid = first < length ? min(kItems, length - first) : 0;
  const std::uint32_t rights = valid - lefts;
  const bool lastLeft = lefts > 0 && leftsBefore + lefts == tileLefts;
  const bool lastRight = rights > 0 && (first - leftsBefore) + rights == tileRights;
  if (lastLeft || lastRight) {
    float values[2] = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      if (static_cast<std::uint32_t>(j) < valid) {
        values[left[j] ? 0 : 1] = items[j].value;
      }
    }
#pragma unroll
    for (int side = 0; side < 2; ++side) {
      if (side == 0 ? lastLeft : lastRight) {
        const unsigned long long word = kValue | __float_as_uint(values[side]);
        atomicExch(&state->last[side], word);
        if (firstTile) {
          atomicExch(&state->lastUpTo[side], word);
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    std::uint32_t before = 0; // left rows of the segment before this tile
    float last[2] = {0.0f, 0.0f};
    bool have[2] = {false, false};
    if (!firstTile) {
      bool found[2] = {false, false};
      // Earlier tiles of the segment are full.
      for (std::uint32_t previous = tile - 1;; --previous) {
        const TileState *other = tileStates + previous;
        const unsigned long long status = waitFor(&other->lefts);
        const std::uint32_t otherLefts = static_cast<std::uint32_t>(status);
        before += otherLefts;
        const bool total = status >= kTileTotal;
#pragma unroll
        for (int side = 0; side < 2; ++side) {
          if (found[side]) {
            continue;
          }
          if (total) {
            const unsigned long long word = waitFor(&other->lastUpTo[side]);
            have[side] = word < kNoValue;
            last[side] = valueOf(word);
          } else if (side == 0 ? otherLefts > 0 : otherLefts < kTile) {
            last[side] = valueOf(waitFor(&other->last[side]));
            have[side] = found[side] = true;
          }
        }
        if (total) {
          break;
        }
      }
#pragma unroll
      for (int side = 0; side < 2; ++side) {
        const unsigned long long word = (side == 0 ? tileLefts : tileRights) > 0
                                            ? waitFor(&state->last[side])
                                        : have[side] ? kValue | __float_as_uint(last[side])
                                                     : kNoValue;
        atomicExch(&state->lastUpTo[side], word);
      }
      atomicExch(&state->lefts, kTileTotal | (before + tileLefts));
    }
    // The new range is free once the segment that was there is loaded.
    if (move.waitTile != kNoTarget) {
      waitForTotal(&tileStates[move.waitTile].lefts);
    }
    shared = before;
    shift = move.shift;
    previousLast[0] = last[0];
    previousLast[1] = last[1];
  }
  __syncthreads(); // `shared`; the staging reuses the scan's shared memory
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const std::uint32_t index = first + j;
    if (index < length) {
      storage.staged[left[j] ? leftsBefore : tileLefts + (index - leftsBefore)] = items[j];
      leftsBefore += left[j];
    }
  }
  __syncthreads();
  const std::uint32_t segmentLeftsBefore = shared;
  const std::uint32_t rightsBefore = start - segmentLeftsBefore; // in the segment

  // Write both runs; count the next level's classes and cuts on the way. The
  // entry at `index` is at `position` of its child's column.
  Entry *to = dst + (static_cast<long long>(segment.base) + shift);
  const std::uint32_t runStart[2] = {segmentLeftsBefore, rightsBefore};
  const unsigned full = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  for (std::uint32_t base = threadIdx.x - lane; base < length; base += kThreads) {
    const std::uint32_t index = base + lane;
    const bool inTile = index < length;
    const int side = index < tileLefts ? 0 : 1;
    const std::uint32_t run = side == 0 ? index : index - tileLefts;
    const std::uint32_t position = runStart[side] + run;
    Entry entry{0.0f, 0u};
    if (inTile) {
      entry = storage.staged[index];
      to[side == 0 ? position : leftCount + position] = entry;
    }
    const bool counted = inTile && target.firstTile[side] != kNoTarget;
    const int nextTile = static_cast<int>(position / kTile - runStart[side] / kTile);
    const std::uint32_t key =
        counted ? static_cast<std::uint32_t>((side * 2 + nextTile) * kMaxGpuClasses) +
                      params.codec.cls(entry.packed)
                : 0xFFFFFFFFu;
    const unsigned peers = __match_any_sync(full, key);
    if (key != 0xFFFFFFFFu && lane == __ffs(peers) - 1) {
      atomicAdd(&histogram[0][0][0] + key, static_cast<std::uint32_t>(__popc(peers)));
    }
    bool cut = false;
    if (counted && position >= 1 && position >= target.minChild[side] &&
        position + target.minChild[side] <= target.count[side]) {
      const float previous = run > 0 ? storage.staged[index - 1].value : previousLast[side];
      cut = isCut(previous, entry.value, params.minGap);
    }
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      const unsigned cuts = __ballot_sync(full, cut && side * 2 + nextTile == slot);
      if (lane == 0 && cuts != 0) {
        atomicAdd(&tries[0][0] + slot, static_cast<std::uint32_t>(__popc(cuts)));
      }
    }
  }
  __syncthreads();
  for (std::uint32_t index = threadIdx.x; index < 4u * classCount; index += kThreads) {
    const int side = index / (2 * classCount);
    const int nextTile = (index / classCount) % 2;
    const int k = index % classCount;
    const std::uint32_t count = histogram[side][nextTile][k];
    if (count > 0) {
      const std::size_t at = target.firstTile[side] + runStart[side] / kTile + nextTile;
      atomicAdd(nextHist + at * classCount + k, count);
    }
  }
  if (threadIdx.x < 4) {
    const int side = threadIdx.x / 2;
    const int nextTile = threadIdx.x % 2;
    const std::uint32_t count = tries[side][nextTile];
    if (count > 0) {
      atomicAdd(nextTries + target.firstTile[side] + runStart[side] / kTile + nextTile, count);
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

// Record the leaf of the rows of every job (GpuGrower::recordLeaves).
__global__ void leafKernel(const Entry *src, const LeafJob *jobs, EntryCodec codec,
                           std::uint32_t *leafOfRow) {
  const LeafJob job = jobs[blockIdx.x];
  for (std::uint32_t index = threadIdx.x; index < job.count; index += blockDim.x) {
    leafOfRow[codec.row(src[job.from + index].packed)] = job.leaf;
  }
}

// C4.5's pruning order (GpuGrower::arrangeRows): the leaf slot of every row.
__global__ void leafSlotKernel(const std::uint32_t *ids, const std::uint32_t *slotOfId,
                               std::size_t rows, std::uint32_t *slots, std::uint32_t *order) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows) {
    slots[row] = slotOfId[ids[row]];
    order[row] = static_cast<std::uint32_t>(row);
  }
}

// Then, in that order: the row-major features, the labels, and the class
// counts of every slot. A thread per (position, feature): the writes are
// contiguous.
__global__ void arrangeKernel(const float *values, std::size_t rows, std::uint32_t features,
                              const std::uint32_t *order, const std::uint32_t *slots,
                              const std::uint16_t *labels, int classCount, float *rowMajor,
                              std::uint16_t *sortedLabels, std::uint32_t *slotCounts) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * features) {
    return;
  }
  const std::size_t position = index / features;
  const std::uint32_t feature = static_cast<std::uint32_t>(index - position * features);
  const std::uint32_t row = order[position];
  rowMajor[index] = values[feature * rows + row];
  if (feature == 0) {
    const std::uint16_t label = labels[row];
    sortedLabels[position] = label;
    atomicAdd(&slotCounts[static_cast<std::size_t>(slots[position]) * classCount + label], 1u);
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

  // With a stream, the old memory is freed and the new allocated in stream
  // order (cudaFree would wait for the whole device, the hand-off copies on
  // another stream too).
  void reserve(std::size_t count, cudaStream_t stream = nullptr) {
    if (count <= capacity_) {
      return;
    }
    if (data_) {
      CUDA_CHECK(stream ? cudaFreeAsync(data_, stream) : cudaFree(data_));
      data_ = nullptr;
    }
    capacity_ = std::max(count, capacity_ + capacity_ / 2);
    CUDA_CHECK(stream ? cudaMallocAsync(&data_, capacity_ * sizeof(T), stream)
                      : cudaMalloc(&data_, capacity_ * sizeof(T)));
  }
  T *get() const { return data_; }
  std::size_t capacity() const { return capacity_; }

private:
  T *data_ = nullptr;
  std::size_t capacity_ = 0;
};

// Optional per-kernel timing: CUDA events around launches and copies. With
// --gpu-profile report() prints the totals per name; with --profile every
// span goes to the trace, timed from an origin event recorded at a known host
// time (setOrigin()).
class KernelProfile {
public:
  KernelProfile(bool print, bool trace) : print_(print), trace_(trace) {}
  ~KernelProfile() {
    for (auto &entry : pending_) {
      cudaEventDestroy(entry.start);
      cudaEventDestroy(entry.stop);
    }
    if (origin_) {
      cudaEventDestroy(origin_);
    }
  }
  bool enabled() const { return print_ || trace_; }
  bool printing() const { return print_; }

  // `stream` must be idle.
  void setOrigin(cudaStream_t stream) {
    if (!trace_) {
      return;
    }
    CUDA_CHECK(cudaEventCreate(&origin_));
    CUDA_CHECK(cudaEventRecord(origin_, stream));
    CUDA_CHECK(cudaEventSynchronize(origin_));
    hostOrigin_ = trace::now();
  }

  template <typename Launch>
  void run(const char *name, cudaStream_t stream, Launch &&launch, const char *lane = "compute") {
    if (!enabled()) {
      launch();
      return;
    }
    Pending entry{name, lane, nullptr, nullptr};
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
      if (origin_) {
        float startMs = 0.0f;
        cudaEventElapsedTime(&startMs, origin_, entry.start);
        trace::gpuSpan(entry.name, entry.lane, hostOrigin_ + startMs * 1e-3,
                       hostOrigin_ + (startMs + ms) * 1e-3);
      }
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
      if (print_) {
        std::fprintf(stderr, "  %-20s %9.2f ms\n", total.first.c_str(), total.second);
      }
    }
    totals_.clear();
  }

private:
  struct Pending {
    std::string name;
    const char *lane;
    cudaEvent_t start;
    cudaEvent_t stop;
  };
  bool print_;
  bool trace_;
  cudaEvent_t origin_ = nullptr;
  double hostOrigin_ = 0.0;
  std::vector<Pending> pending_;
  std::vector<std::pair<std::string, float>> totals_;
};

// The CPUs of the GPU's NUMA node that this process may use; empty if that
// is unknown or the machine has a single node.
std::vector<int> gpuNodeCpus() {
  int device = 0;
  char busId[32] = {};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetPCIBusId(busId, sizeof busId, device) != cudaSuccess) {
    return {};
  }
  std::string id(busId);
  std::transform(id.begin(), id.end(), id.begin(), [](unsigned char c) { return std::tolower(c); });
  int node = -1;
  if (std::FILE *file = std::fopen(("/sys/bus/pci/devices/" + id + "/numa_node").c_str(), "r")) {
    if (std::fscanf(file, "%d", &node) != 1) {
      node = -1;
    }
    std::fclose(file);
  }
  std::FILE *file =
      node < 0 ? nullptr
               : std::fopen(("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist").c_str(), "r");
  if (!file) {
    return {};
  }
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  sched_getaffinity(0, sizeof allowed, &allowed);
  std::vector<int> cpus;
  int first = 0;
  while (std::fscanf(file, "%d", &first) == 1) {
    int last = first;
    if (std::fscanf(file, "-%d", &last) != 1) {
      last = first;
    }
    for (int cpu = first; cpu <= last; ++cpu) {
      if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) {
        cpus.push_back(cpu);
      }
    }
    if (std::fgetc(file) != ',') {
      break;
    }
  }
  std::fclose(file);
  if (static_cast<int>(cpus.size()) == CPU_COUNT(&allowed)) {
    return {}; // one node (or only its CPUs allowed): nothing to choose
  }
  return cpus;
}

// Page-locked staging buffers for copies between the GPU and pageable memory
// (GpuGrower::uploadStaged, downloadStaged). Allocating them takes tens of
// milliseconds, so gpuPrepare() does it while the dataset loads. They live as
// long as the process (freeing them at exit would race with the CUDA
// runtime's teardown).
//
// On a machine with several NUMA nodes, the buffers are on the GPU's node
// and the copies into and out of them are made by threads pinned to its
// CPUs: with the copies spread over both sockets of the A100 node, the
// traffic between the sockets slowed the transfers to half the PCIe speed
// (6 GB uploaded in 0.55-0.75 s instead of 0.3 s).
struct Staging {
  static constexpr int kBuffers = 4;
  static constexpr std::size_t kBytes = std::size_t{32} << 20;
  static constexpr std::size_t kCopyThreads = 32;
  char *buffers[kBuffers] = {};
  // The hand-offs' own (GpuGrower::handOff), one per copy stream, each a
  // part of the device's hand-off ring.
  static constexpr int kHandOffBuffers = 2;
  static constexpr std::size_t kHandOffBytes = std::size_t{128} << 20;
  char *handOff[kHandOffBuffers] = {};
  cudaEvent_t done[kBuffers] = {};
  std::unique_ptr<ThreadPool> copiers; // on the GPU's node, if it is known
  std::once_flag once;

  // `threads`: at most this many threads copy (the caller included); the
  // first call decides.
  void prepare(unsigned threads = kCopyThreads) {
    std::call_once(once, [this, threads]() {
      const std::vector<int> cpus = gpuNodeCpus();
      auto pin = [cpus]() {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (const int cpu : cpus) {
          CPU_SET(cpu, &set);
        }
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
      };
      // Page-locked memory is placed when it is allocated: on the node of
      // the allocating thread.
      std::thread([&]() {
        if (!cpus.empty()) {
          pin();
        }
        for (int buffer = 0; buffer < kBuffers; ++buffer) {
          void *memory = nullptr;
          CUDA_CHECK(cudaMallocHost(&memory, kBytes));
          buffers[buffer] = static_cast<char *>(memory);
          CUDA_CHECK(cudaEventCreateWithFlags(&done[buffer], cudaEventDisableTiming));
        }
        for (char *&buffer : handOff) {
          void *memory = nullptr;
          CUDA_CHECK(cudaMallocHost(&memory, kHandOffBytes));
          buffer = static_cast<char *>(memory);
        }
      }).join();
      const std::size_t copyThreads =
          std::min({cpus.size(), kCopyThreads, static_cast<std::size_t>(threads)});
      if (copyThreads > 1) {
        copiers = std::make_unique<ThreadPool>(static_cast<unsigned>(copyThreads - 1), pin);
      }
    });
  }

  // The threads for the copies: the caller works too (ThreadPool::parallelFor).
  ThreadPool &copyPool(ThreadPool &fallback) { return copiers ? *copiers : fallback; }
};

Staging &staging() {
  static Staging *instance = new Staging; // never destroyed, see above
  return *instance;
}

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
        profile_(options.gpu.profile, trace::on()) {
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
    if (toucher_.joinable()) {
      toucher_.join();
    }
    for (cudaEvent_t event : {gathered_, copied_}) {
      if (event) {
        cudaEventDestroy(event);
      }
    }
    for (cudaEvent_t event : ringFree_) {
      if (event) {
        cudaEventDestroy(event);
      }
    }
    for (cudaStream_t stream : copyStreams_) {
      if (stream) {
        cudaStreamDestroy(stream);
      }
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows,
            GrowTimings &timings, DistinctValues *distinct, RowLeaves *leafOfRow) override {
    if (leafOfRow && !rows.empty()) {
      throw std::logic_error("GpuGrower: leaves of rows only with all rows");
    }
    stride_ = rows.empty() ? rows_ : rows.size();
    params_.minGap = rules.minValueGap();
    params_.criterion = rules.criterion();
    {
      ScopedTimer timer(timings.prepareSeconds);
      trace::Phase phase("presort");
      trace::Doing doing(trace::Work::GpuHost);
      presort(rows, distinct);
    }
    Tree tree;
    tree.featureNames = train_.featureNames;
    tree.classNames = train_.classNames;
    ScopedTimer timer(timings.buildSeconds);
    std::optional<trace::Phase> setupPhase(std::in_place, "grow setup");
    NodeStore store(train_.classCount(), 2 * stride_);
    // Leaves of rows: the CPU tasks write theirs into `leafOfRow`, the GPU
    // levels into leafOfRow_ (all ones elsewhere), merged at the end.
    leafJobs_.clear();
    if (leafOfRow) {
      leafOfRow->ids.resize(stride_);
      leafOfRow->map.clear();
      leafOfRow_.reserve(stride_);
      CUDA_CHECK(cudaMemsetAsync(leafOfRow_.get(), 0xff, stride_ * sizeof(std::uint32_t),
                                 stream_));
    }
    recordLeaves_ = leafOfRow != nullptr;
    CpuTreeBuilder cpuBuilder(rules, codec_, features_, store, hostGoesLeft_.get(), &pool_,
                              options_.parallel, {}, leafOfRow ? leafOfRow->ids.data() : nullptr);
    rules_ = &rules;
    store_ = &store;
    cpuBuilder_ = &cpuBuilder;
    arenaUsed_ = 0;
    scratchUsed_ = 0;

    const std::vector<std::uint32_t> counts = countClasses(train_, rows, &pool_);
    std::vector<std::uint32_t> allFeatures(features_);
    std::iota(allFeatures.begin(), allFeatures.end(), 0u);
    LevelNode root{store.add(counts.data()), 0, static_cast<std::uint32_t>(stride_), 0,
                   std::move(allFeatures)};
    minRows_ = options_.gpu.only ? 1
               : options_.gpu.minRows > 0
                   ? options_.gpu.minRows
                   : options_.algorithm == Algorithm::Cart
                       ? std::clamp(stride_ / GpuOptions::kAutoMinRowsShareCart,
                                    GpuOptions::kAutoMinRows, GpuOptions::kAutoMaxMinRowsCart)
                       : std::clamp(stride_ / GpuOptions::kAutoMinRowsShare,
                                    GpuOptions::kAutoMinRows, GpuOptions::kAutoMaxMinRows);
    setupPhase.reset();
    if (stride_ < minRows_) {
      growOnCpu(std::move(root));
    } else {
      trace::Phase phase("gpu levels");
      trace::Doing doing(trace::Work::GpuHost);
      growLevels(std::move(root));
    }
    double waitSeconds = 0.0;
    {
      ScopedTimer wait(waitSeconds);
      trace::Phase phase("wait cpu subtrees");
      {
        // Every hand-off has started its CPU tasks once its copy is done.
        trace::Doing doing(trace::Work::GpuWait);
        for (cudaStream_t stream : copyStreams_) {
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
      }
      pool_.waitIdle();
    }
    if (profile_.printing()) {
      std::fprintf(stderr, "  waiting for CPU subtrees: %.2f ms\n", waitSeconds * 1000.0);
    }
    if (profile_.enabled()) {
      profile_.report();
    }
    trace::Phase phase("assemble tree");
    trace::Doing doing(trace::Work::Assemble);
    if (leafOfRow) {
      mergeLeaves(leafOfRow->ids);
    }
    store.toTree(tree, &pool_, leafOfRow);
    rules_ = nullptr;
    store_ = nullptr;
    cpuBuilder_ = nullptr;
    return tree;
  }

  bool arrangeRows(std::span<const std::uint32_t> slotOfId, std::size_t slots,
                   std::span<const std::uint32_t> ids, float *rowMajor,
                   std::uint16_t *labels, std::uint32_t *slotCounts) override {
    if (!rawValues_.get() || ids.size() != rows_ || slots == 0) {
      return false;
    }
    // The result, then scratch, in the column buffer: the grow is done with it.
    const std::size_t rows = rows_;
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    unsigned keyBits = 1;
    while ((std::size_t{1} << keyBits) < slots) {
      ++keyBits;
    }
    std::size_t tempBytes = 0;
    std::uint32_t *none = nullptr; // only the size is asked for
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, none, none, none, none,
                                               static_cast<int>(rows), 0,
                                               static_cast<int>(keyBits), stream_));
    char *scratch = reinterpret_cast<char *>(columns_.get());
    std::size_t used = (rows * features_ * sizeof(float) + 255) / 256 * 256; // the result
    auto take = [&](std::size_t bytes) {
      char *at = scratch + used;
      used += (bytes + 255) / 256 * 256;
      return at;
    };
    auto *nodeSlots = reinterpret_cast<std::uint32_t *>(take(slotOfId.size_bytes()));
    auto *leaves = reinterpret_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
    auto *keys = reinterpret_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
    auto *sortedKeys = reinterpret_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
    auto *rowIds = reinterpret_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
    auto *sortedOrder = reinterpret_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
    auto *sortedLabels = reinterpret_cast<std::uint16_t *>(take(rows * sizeof(std::uint16_t)));
    auto *counts = reinterpret_cast<std::uint32_t *>(take(slots * classes * sizeof(std::uint32_t)));
    void *temp = take(tempBytes);
    if (used > slotCount_ * slotEntries_ * sizeof(Entry)) {
      return false;
    }
    float *arranged = reinterpret_cast<float *>(columns_.get());
    uploadStaged(nodeSlots, slotOfId.data(), slotOfId.size_bytes(), "uploadArrange");
    uploadStaged(leaves, ids.data(), rows * sizeof(std::uint32_t), "uploadArrange");
    profile_.run("arrangeSort", stream_, [&]() {
      leafSlotKernel<<<static_cast<unsigned>((rows + 255) / 256), 256, 0, stream_>>>(
          leaves, nodeSlots, rows, keys, rowIds);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp, tempBytes, keys, sortedKeys, rowIds,
                                                 sortedOrder, static_cast<int>(rows), 0,
                                                 static_cast<int>(keyBits), stream_));
    });
    CUDA_CHECK(cudaMemsetAsync(counts, 0, slots * classes * sizeof(std::uint32_t), stream_));
    profile_.run("arrangeRows", stream_, [&]() {
      const std::size_t total = rows * features_;
      arrangeKernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream_>>>(
          rawValues_.get(), rows, static_cast<std::uint32_t>(features_), sortedOrder, sortedKeys,
          labels_.get(), classCount_, arranged, sortedLabels, counts);
    });
    CUDA_CHECK(cudaGetLastError());
    downloadStaged(rowMajor, arranged, rows * features_ * sizeof(float), "downloadArranged");
    downloadStaged(labels, sortedLabels, rows * sizeof(std::uint16_t), "downloadArranged");
    downloadStaged(slotCounts, counts, slots * classes * sizeof(std::uint32_t), "downloadArranged");
    {
      trace::Doing waiting(trace::Work::GpuWait);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    if (profile_.enabled()) {
      profile_.report();
    }
    return true;
  }

  // The host arena: it only holds the columns of nodes handed to the CPU.
  std::span<std::byte> workspace() override {
    if (!arena_) { // --gpu-only
      arena_.reset(new Entry[features_ * rows_]);
    }
    waitTouched(features_ * rows_ * sizeof(Entry)); // the caller writes all of it
    return {reinterpret_cast<std::byte *>(arena_.get()), features_ * rows_ * sizeof(Entry)};
  }

private:
  void setup() {
    double contextSeconds = 0.0;
    {
      ScopedTimer timer(contextSeconds);
      CUDA_CHECK(cudaFree(nullptr)); // creates the CUDA context if needed
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    for (cudaStream_t &stream : copyStreams_) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&gathered_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    // The level arrays grow in stream order (DeviceArray::reserve): keep what
    // the pool frees for reuse instead of returning it at every sync.
    cudaMemPool_t memoryPool = nullptr;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memoryPool, device));
    std::uint64_t keep = UINT64_MAX;
    CUDA_CHECK(cudaMemPoolSetAttribute(memoryPool, cudaMemPoolAttrReleaseThreshold, &keep));
    int fp32PerFp64 = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&fp32PerFp64, cudaDevAttrSingleToDoublePrecisionPerfRatio,
                                      device));
    onePass_ = options_.gpu.sweep == GpuSweep::OnePass ||
               (options_.gpu.sweep == GpuSweep::Auto && fp32PerFp64 <= 4);
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount_, cudaDevAttrMultiProcessorCount, device));
    profile_.setOrigin(stream_);
    if (trace::on()) {
      cudaDeviceProp properties{};
      CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      trace::info("gpu", properties.name);
      trace::info("gpu_sweep", onePass_ ? "one-pass" : "two-pass");
      trace::add("gpu_context_seconds", contextSeconds);
    }
    if (profile_.printing()) {
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
    const bool leaves = options_.algorithm == Algorithm::C45 && options_.c45.prune;
    // The columns: one slot each, and kSpareSlots to move them to (see
    // PartitionTarget). The hand-offs go through a ring of kRingParts parts.
    slotCount_ = features_ + kSpareSlots;
    ringPartEntries_ = std::max<std::size_t>(
        kTile, std::min(kRingBytes, total * sizeof(Entry)) / kRingParts / sizeof(Entry));
    // Slots start at page boundaries: then a column's old and new place
    // have the same alignment, as in two separate buffers (unaligned slots
    // made the partition 5% slower on airline).
    slotEntries_ = (rows_ + kSlotAlign - 1) / kSlotAlign * kSlotAlign;
    const std::size_t needed = (slotCount_ * slotEntries_ + kRingParts * ringPartEntries_) * sizeof(Entry) +
                               (reusable_ ? total * sizeof(float) : 0) + rows_ * 7 +
                               (leaves ? rows_ * sizeof(std::uint32_t) : 0) + (64u << 20);
    if (needed > freeBytes) {
      throw std::runtime_error("Not enough GPU memory: need about " +
                               std::to_string(needed >> 20) + " MiB, " +
                               std::to_string(freeBytes >> 20) +
                               " MiB free. Use --parallel instead.");
    }
    // C4.5's pruning starts by ordering the rows by leaf (arrangeRows()),
    // which needs the raw values again: a copy stays on the GPU if it fits.
    const bool keepValues = leaves && !reusable_ && needed + total * sizeof(float) <= freeBytes;
    double allocSeconds = 0.0;
    std::optional<ScopedTimer> allocTimer(std::in_place, allocSeconds);
    columns_.reserve(slotCount_ * slotEntries_);
    ring_.reserve(kRingParts * ringPartEntries_);
    for (cudaEvent_t &event : ringFree_) {
      CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    goesLeft_.reserve(rows_ / 32 + 1);
    labels_.reserve(rows_);
    if (reusable_) {
      values_.reserve(total);
      rowList_.reserve(rows_);
    }

    // Raw values: kept in their own buffer for repeated use, otherwise
    // uploaded into the end of the column buffer (see presort()). Through
    // page-locked staging buffers, without page-locking the source (see
    // uploadStaged()).
    float *target = reusable_ ? values_.get() : uploadedValues();
    allocTimer.reset();
    double uploadSeconds = 0.0;
    {
      ScopedTimer timer(uploadSeconds);
      trace::Phase phase("upload");
      uploadStaged(target, train_.values.data(), total * sizeof(float));
      uploadStaged(labels_.get(), train_.labels.data(), rows_ * sizeof(std::uint16_t));
      if (keepValues) {
        rawValues_.reserve(total);
        profile_.run("copyValues", stream_, [&]() {
          CUDA_CHECK(cudaMemcpyAsync(rawValues_.get(), target, total * sizeof(float),
                                     cudaMemcpyDeviceToDevice, stream_));
        });
      }
    }
    trace::add("gpu_alloc_seconds", allocSeconds);
    if (profile_.printing()) {
      std::fprintf(stderr, "  setup alloc %.2f ms, upload %.2f ms\n", allocSeconds * 1e3,
                   uploadSeconds * 1e3);
    }

    std::size_t tempBytes = 0;
    cub::DoubleBuffer<Entry> buffers(columns_.get(), columns_.get());
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, tempBytes, buffers, static_cast<int>(rows_),
                                              EntryValue{}, 0, 32, stream_));
    // ... and room for C4.5's distinct values (distinctValues()): growing it
    // there would synchronize.
    std::size_t uniqueBytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Unique(
        nullptr, uniqueBytes, thrust::make_transform_iterator(columns_.get(), ValueBits{}),
        reinterpret_cast<std::uint32_t *>(columns_.get()), static_cast<int *>(nullptr),
        static_cast<int>(rows_), stream_));
    sortTemp_.reserve(std::max(tempBytes, uniqueBytes));
    sortTempBytes_ = tempBytes;

    // c * log2(c) tables (the same for every grow).
    const std::vector<double> logs = xlog2xTable();
    upload(logTable_, logs);
    params_.codec = codec_;
    params_.classCount = classCount_;
    params_.logs = LogTable{logTable_.get()};

    // Host memory for the columns of nodes handed to the CPU (every row is
    // handed over at most once per grow, so features * rows entries always
    // suffice), and their partition scratch. Pageable: the hand-offs go
    // through page-locked buffers (see handOff()). --gpu-only hands nothing
    // over (workspace() allocates the arena if the pruning asks for it).
    if (!options_.gpu.only) {
      arena_.reset(new Entry[total]);
      arenaScratch_.reset(new Entry[rows_]);
    }
    // CART with a fixed alpha stops early and hands few rows to the CPU
    // (airline: 2 of 12 GB): the copies fault in the few pages they need.
    const bool fewHandOffs = options_.gpu.only ||
                             (options_.algorithm == Algorithm::Cart &&
                              options_.cart.pruning == CartPruning::Alpha);
    if (fewHandOffs) {
      touchedBytes_ = total * sizeof(Entry);
    } else {
      toucher_ = std::thread([this, bytes = total * sizeof(Entry)]() { touchArena(bytes); });
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  // Copy `bytes` from pageable host memory to the device at the speed of
  // page-locked memory, without page-locking the source: the pool's threads
  // copy chunks into a few page-locked staging buffers while the previous
  // chunks are uploaded from them. Page-locking is slow with HMM (both
  // cluster GPU nodes): on the H100 node 6 GB took 0.17-0.22 s this way,
  // 0.32 s page-locked in pipelined chunks and 0.35 s page-locked at once; on
  // the A100 node 0.45-0.55 s either way. Smaller chunks cost more per chunk
  // than they gain, more threads gained nothing.
  void uploadStaged(void *target, const void *source, std::size_t bytes,
                    const char *name = "upload") {
    constexpr std::size_t kChunk = Staging::kBytes;
    trace::add("h2d_bytes", static_cast<double>(bytes));
    if (bytes <= kChunk) {
      profile_.run(name, stream_, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(target, source, bytes, cudaMemcpyHostToDevice, stream_));
      }, "copy");
      trace::Doing waiting(trace::Work::GpuWait);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      return;
    }
    trace::Doing copying(trace::Work::Staging);
    staging().prepare();
    Staging &staging = dt::staging();
    const char *from = static_cast<const char *>(source);
    char *to = static_cast<char *>(target);
    ThreadPool &copiers = staging.copyPool(pool_);
    const std::size_t parts = std::min<std::size_t>(Staging::kCopyThreads, copiers.workerCount() + 1);
    for (std::size_t offset = 0, chunk = 0; offset < bytes; offset += kChunk, ++chunk) {
      const int buffer = static_cast<int>(chunk % Staging::kBuffers);
      const std::size_t size = std::min(kChunk, bytes - offset);
      {
        trace::Doing waiting(trace::Work::GpuWait);
        CUDA_CHECK(cudaEventSynchronize(staging.done[buffer])); // its last upload is done
      }
      char *stage = staging.buffers[buffer];
      copiers.parallelFor(parts, [&](std::size_t part) {
        const std::size_t begin = size * part / parts;
        const std::size_t end = size * (part + 1) / parts;
        std::memcpy(stage + begin, from + offset + begin, end - begin);
      });
      profile_.run(name, stream_, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(to + offset, stage, size, cudaMemcpyHostToDevice, stream_));
      }, "copy");
      CUDA_CHECK(cudaEventRecord(staging.done[buffer], stream_));
    }
    trace::Doing waiting(trace::Work::GpuWait);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  // The reverse of uploadStaged(): chunks are downloaded into the staging
  // buffers, and the pool's threads copy each one out while the next ones
  // are downloaded.
  void downloadStaged(void *target, const void *source, std::size_t bytes,
                      const char *name = "download") {
    if (bytes <= Staging::kBytes / 8) {
      trace::add("d2h_bytes", static_cast<double>(bytes));
      profile_.run(name, stream_, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToHost, stream_));
      }, "copy");
      trace::Doing waiting(trace::Work::GpuWait);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      return;
    }
    char *to = static_cast<char *>(target);
    staging().prepare();
    ThreadPool &copiers = staging().copyPool(pool_);
    const std::size_t parts = std::min<std::size_t>(Staging::kCopyThreads, copiers.workerCount() + 1);
    downloadChunks(source, bytes, name, [&](std::size_t offset, const char *stage, std::size_t size) {
      trace::Doing copying(trace::Work::Staging);
      copiers.parallelFor(parts, [&](std::size_t part) {
        const std::size_t begin = size * part / parts;
        const std::size_t end = size * (part + 1) / parts;
        std::memcpy(to + offset + begin, stage + begin, end - begin);
      });
    });
  }

  // Download `bytes` chunk by chunk through the staging buffers; `consume`
  // (offset, chunk, size) takes each chunk in order, while the next ones are
  // downloaded.
  template <typename Consume>
  void downloadChunks(const void *source, std::size_t bytes, const char *name, Consume &&consume) {
    constexpr std::size_t kChunk = Staging::kBytes;
    trace::add("d2h_bytes", static_cast<double>(bytes));
    staging().prepare();
    Staging &staging = dt::staging();
    const char *from = static_cast<const char *>(source);
    const std::size_t chunks = (bytes + kChunk - 1) / kChunk;
    auto sizeOf = [&](std::size_t chunk) { return std::min(kChunk, bytes - chunk * kChunk); };
    auto download = [&](std::size_t chunk) {
      const int buffer = static_cast<int>(chunk % Staging::kBuffers);
      profile_.run(name, stream_, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(staging.buffers[buffer], from + chunk * kChunk, sizeOf(chunk),
                                   cudaMemcpyDeviceToHost, stream_));
      }, "copy");
      CUDA_CHECK(cudaEventRecord(staging.done[buffer], stream_));
    };
    for (std::size_t chunk = 0; chunk < std::min<std::size_t>(chunks, Staging::kBuffers); ++chunk) {
      download(chunk);
    }
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      const int buffer = static_cast<int>(chunk % Staging::kBuffers);
      {
        trace::Doing waiting(trace::Work::GpuWait);
        CUDA_CHECK(cudaEventSynchronize(staging.done[buffer]));
      }
      consume(chunk * kChunk, staging.buffers[buffer], sizeOf(chunk));
      if (chunk + Staging::kBuffers < chunks) {
        download(chunk + Staging::kBuffers);
      }
    }
  }

  // Touch the arena's pages on all threads, in the background (the pool is
  // idle while the GPU presorts), so the hand-offs' copies into it do not
  // fault them in: that made C4.5 on airline 0.2-0.9 s slower on the H100
  // node. The pages are spread over the NUMA nodes, like the threads that
  // read them. In chunks, in order: a hand-off waits only for the part it
  // fills (waitTouched()). (MADV_POPULATE_WRITE, which leaves the contents
  // alone, holds the process's mmap lock: other threads' allocations and
  // page faults waited for it, and the H100 node's presort went from 0.5 to
  // 4.2 s when its memory was low.)
  void touchArena(std::size_t bytes) {
    constexpr std::size_t kChunks = 256;
    trace::Doing doing(trace::Work::PageLock);
    std::vector<char> done(kChunks, 0);
    std::size_t next = 0; // chunks below it are done (touchMutex_)
    char *data = reinterpret_cast<char *>(arena_.get());
    pool_.parallelFor(kChunks, [&](std::size_t chunk) {
      const std::size_t begin = bytes * chunk / kChunks;
      const std::size_t end = bytes * (chunk + 1) / kChunks;
      std::memset(data + begin, 0, end - begin);
      std::lock_guard<std::mutex> lock(touchMutex_);
      done[chunk] = 1;
      while (next < kChunks && done[next]) {
        ++next;
      }
      touchedBytes_ = bytes * next / kChunks;
      touchCv_.notify_all();
    });
  }

  // Wait until the first `bytes` of the arena are touched.
  void waitTouched(std::size_t bytes) {
    trace::Doing waiting(trace::Work::PageLock);
    std::unique_lock<std::mutex> lock(touchMutex_);
    touchCv_.wait(lock, [&]() { return touchedBytes_ >= bytes; });
  }

  // Where setup() uploads the raw values (column-major, rows_ per feature)
  // when the grower is used once: the end of the column buffer.
  float *uploadedValues() const {
    return reinterpret_cast<float *>(columns_.get() + slotCount_ * slotEntries_) - features_ * rows_;
  }

  // The sorted column of every feature, of the selected rows, in a slot each.
  // One feature at a time: its entries are gathered into a free slot and
  // sorted with another one as the radix sort's second buffer; the column
  // stays in the one that holds the result. Uploaded raw values take the
  // last slots, which free up as their features are sorted: a slot is taken
  // only once the raw values in it are no longer needed.
  void presort(std::span<const std::uint32_t> rows, DistinctValues *distinct) {
    const float *values = uploadedValues();
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
    // Whether slot `slot` overlaps the raw values of features `first` onwards.
    const std::size_t rawStart = 2 * slotCount_ * slotEntries_ - features_ * rows_; // in floats
    auto holdsValues = [&](std::size_t slot, std::size_t first) {
      return !reusable_ && first < features_ &&
             2 * (slot + 1) * slotEntries_ > rawStart + first * rows_;
    };
    std::vector<bool> taken(slotCount_, false);
    auto freeSlot = [&](std::size_t first, std::size_t other) {
      for (std::size_t slot = 0; slot < slotCount_; ++slot) {
        if (!taken[slot] && slot != other && !holdsValues(slot, first)) {
          return slot;
        }
      }
      throw std::logic_error("GpuGrower: no free column slot for the presort");
    };
    if (distinct) {
      distinctCounts_.reserve(features_);
      distinctPlaces_.reserve(features_ + 1);
      CUDA_CHECK(cudaMemsetAsync(distinctPlaces_.get() + features_, 0, sizeof(unsigned long long),
                                 stream_));
    }
    slot_.assign(features_, 0);
    for (std::size_t feature = 0; feature < features_; ++feature) {
      const std::size_t first = freeSlot(feature, slotCount_);
      const std::size_t second = freeSlot(feature + 1, first); // written after the gather
      Entry *a = columns_.get() + first * slotEntries_;
      Entry *b = columns_.get() + second * slotEntries_;
      profile_.run("presortGather", stream_, [&]() {
        gatherKernel<<<static_cast<unsigned>((stride_ + 255) / 256), 256, 0, stream_>>>(
            values + feature * rows_, rowList, static_cast<std::uint32_t>(stride_), labels_.get(),
            codec_, a);
      });
      CUDA_CHECK(cudaGetLastError());
      cub::DoubleBuffer<Entry> buffers(a, b);
      profile_.run("presortSort", stream_, [&]() {
        std::size_t tempBytes = sortTempBytes_;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(sortTemp_.get(), tempBytes, buffers,
                                                  static_cast<int>(stride_), EntryValue{}, 0, 32,
                                                  stream_));
      });
      const bool inFirst = buffers.Current() == a;
      slot_[feature] = static_cast<std::uint32_t>(inFirst ? first : second);
      taken[slot_[feature]] = true;
      if (distinct) { // the other slot is free
        distinctValues(feature, buffers.Current(), inFirst ? b : a);
      }
    }
    if (distinct) {
      downloadDistinct(*distinct, taken);
    }
    trace::Doing waiting(trace::Work::GpuWait);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  // The distinct values of a feature, from its sorted column: cub's unique on
  // their bit patterns (into `scratch`, a free slot), appended to the pool (the
  // hand-off ring, free until the build) if they fit.
  void distinctValues(std::size_t feature, const Entry *column, Entry *scratch) {
    const auto bits = thrust::make_transform_iterator(column, ValueBits{});
    auto *unique = reinterpret_cast<std::uint32_t *>(scratch);
    const int count = static_cast<int>(stride_);
    int *counts = distinctCounts_.get();
    std::size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, tempBytes, bits, unique, counts + feature, count,
                                         stream_));
    sortTemp_.reserve(tempBytes);
    const unsigned blocks = static_cast<unsigned>((stride_ + 255) / 256);
    profile_.run("presortDistinct", stream_, [&]() {
      CUDA_CHECK(cub::DeviceSelect::Unique(sortTemp_.get(), tempBytes, bits, unique,
                                           counts + feature, count, stream_));
      placeDistinctKernel<<<1, 1, 0, stream_>>>(counts, static_cast<std::uint32_t>(feature),
                                                distinctPoolCapacity(),
                                                distinctPlaces_.get() + features_,
                                                distinctPlaces_.get());
      copyDistinctKernel<<<blocks, 256, 0, stream_>>>(unique, counts,
                                                      static_cast<std::uint32_t>(feature),
                                                      distinctPlaces_.get(), distinctPool());
    });
    CUDA_CHECK(cudaGetLastError());
  }

  std::uint32_t *distinctPool() const { return reinterpret_cast<std::uint32_t *>(ring_.get()); }
  std::size_t distinctPoolCapacity() const {
    return kRingParts * ringPartEntries_ * sizeof(Entry) / sizeof(std::uint32_t);
  }

  // Download the pooled distinct values; the features that did not fit are
  // done again one at a time from their sorted columns (in a free slot).
  void downloadDistinct(DistinctValues &distinct, const std::vector<bool> &taken) {
    std::vector<int> counts(features_);
    std::vector<unsigned long long> places(features_);
    CUDA_CHECK(cudaMemcpyAsync(counts.data(), distinctCounts_.get(), features_ * sizeof(int),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(places.data(), distinctPlaces_.get(),
                               features_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    distinct.offsets.assign(features_ + 1, 0);
    std::size_t placed = 0; // features 0..placed-1 are in the pool, back to back
    for (std::size_t feature = 0; feature < features_; ++feature) {
      distinct.offsets[feature + 1] = distinct.offsets[feature] + static_cast<std::size_t>(counts[feature]);
      if (places[feature] == kNotPlaced) {
        continue;
      }
      placed = feature + 1;
    }
    distinct.values.resize(distinct.offsets[features_]);
    downloadStaged(distinct.values.data(), distinctPool(), distinct.offsets[placed] * sizeof(float),
                   "downloadDistinct");
    if (placed == features_) {
      return;
    }
    const std::size_t free = static_cast<std::size_t>(
        std::find(taken.begin(), taken.end(), false) - taken.begin());
    auto *unique = reinterpret_cast<std::uint32_t *>(columns_.get() + free * slotEntries_);
    for (std::size_t feature = placed; feature < features_; ++feature) {
      std::size_t tempBytes = sortTemp_.capacity();
      const auto bits = thrust::make_transform_iterator(column(feature), ValueBits{});
      CUDA_CHECK(cub::DeviceSelect::Unique(sortTemp_.get(), tempBytes, bits, unique,
                                           distinctCounts_.get() + feature,
                                           static_cast<int>(stride_), stream_));
      downloadStaged(distinct.values.data() + distinct.offsets[feature], unique,
                     static_cast<std::size_t>(counts[feature]) * sizeof(float), "downloadDistinct");
    }
  }

  // Too small for the GPU: copy the presorted columns and use the CPU.
  void growOnCpu(LevelNode root) {
    waitTouched(features_ * stride_ * sizeof(Entry));
    for (std::size_t feature = 0; feature < features_; ++feature) {
      downloadStaged(arena_.get() + feature * stride_, column(feature), stride_ * sizeof(Entry),
                     "downloadColumns");
    }
    cpuBuilder_->grow(Columns{arena_.get(), stride_, arenaScratch_.get()},
                      Subtree{root.node, 0, root.count, 0, std::move(root.features), {}, {}});
  }

  void growLevels(LevelNode root) {
    if (rules_->isTerminal(store_->counts(root.node), root.count, 0)) {
      addLeafJob(columnBase(0), root.count, root.node);
      launchLeafJobs();
      return;
    }
    std::vector<LevelNode> level;
    level.push_back(std::move(root));
    histogramReady_ = false;
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
        const std::string name = "gpu level " + std::to_string(depth);
        NvtxRange range(name.c_str());
        level = processLevel(level);
      }
      trace::row("gpu_levels", {{"depth", depth}, {"nodes", static_cast<double>(nodes)},
                                {"rows", static_cast<double>(rows)},
                                {"segments", static_cast<double>(segments)},
                                {"end", trace::now()}, {"seconds", seconds}});
      if (profile_.printing()) {
        std::fprintf(stderr, "  gpu level %2d: %6zu nodes %9zu rows %7zu segments %8.2f ms\n",
                     depth, nodes, rows, segments, seconds * 1000.0);
      }
    }
  }

  template <int MAXK> void launchScans(std::uint32_t tiles, std::size_t segmentCount) {
    const unsigned segmentBlocks = static_cast<unsigned>(segmentCount);
    const Entry *src = columns_.get();
    // Two classes: bound every tile, so the sweeps skip the hopeless ones.
    const double *tileUpper = nullptr;
    const double *segmentLower = nullptr;
    if (classCount_ == 2) {
      profile_.run("tileBound", stream_, [&]() {
        tileBoundKernel<<<(tiles + 255) / 256, 256, 0, stream_>>>(
            src, segments_.get(), tiles, tileSegment_.get(), constant_.get(), nodes_.get(),
            params_, tileHist_.get(), nodeCounts_.get(), tileUpper_.get(), tileLower_.get());
      });
      CUDA_CHECK(cudaGetLastError());
      profile_.run("segmentLower", stream_, [&]() {
        segmentMaxKernel<double><<<segmentBlocks, kThreads, 0, stream_>>>(
            segments_.get(), tileLower_.get(), segmentLower_.get());
      });
      CUDA_CHECK(cudaGetLastError());
      tileUpper = tileUpper_.get();
      segmentLower = segmentLower_.get();
    }
    if (onePass_) {
      profile_.run("tileExact", stream_, [&]() {
        tileExactKernel<MAXK, false><<<tiles, kThreads, 0, stream_>>>(
            src, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
            tileHist_.get(), nodeCounts_.get(), tileUpper, segmentLower, nullptr, nullptr,
            tileBest_.get(), tileTries_.get(), nullptr, nullptr);
      });
      CUDA_CHECK(cudaGetLastError());
    } else {
      constexpr bool kKeep = MAXK <= kMaxCandidateClasses;
      Candidate<MAXK> *candidates =
          kKeep ? reinterpret_cast<Candidate<MAXK> *>(candidates_.get()) : nullptr;
      profile_.run("tileEstimate", stream_, [&]() {
        tileEstimateKernel<MAXK><<<tiles, kThreads, 0, stream_>>>(
            src, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
            tileHist_.get(), nodeCounts_.get(), tileUpper, segmentLower, tileMax_.get(),
            tileTries_.get(), candidates, candidateCount_.get());
      });
      CUDA_CHECK(cudaGetLastError());
      profile_.run("segmentMax", stream_, [&]() {
        segmentMaxKernel<float><<<segmentBlocks, kThreads, 0, stream_>>>(
            segments_.get(), tileMax_.get(), segmentMax_.get());
      });
      CUDA_CHECK(cudaGetLastError());
      if constexpr (kKeep) {
        std::uint32_t *listed = tileList_.get() + tiles;
        CUDA_CHECK(cudaMemsetAsync(listed, 0, sizeof(std::uint32_t), stream_));
        profile_.run("tileCandidate", stream_, [&]() {
          tileCandidateKernel<MAXK><<<(tiles + kWarps - 1) / kWarps, kThreads, 0, stream_>>>(
              tiles, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
              nodeCounts_.get(), tileUpper, segmentLower, tileMax_.get(), segmentMax_.get(),
              candidates, candidateCount_.get(), tileBest_.get(), tileList_.get(), listed);
        });
        CUDA_CHECK(cudaGetLastError());
        // The listed tiles are few: a grid of resident blocks loops over them.
        const unsigned blocks = std::min<unsigned>(tiles, static_cast<unsigned>(smCount_) * 4);
        profile_.run("tileExact", stream_, [&]() {
          tileExactKernel<MAXK, true><<<blocks, kThreads, 0, stream_>>>(
              src, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
              tileHist_.get(), nodeCounts_.get(), tileUpper, segmentLower, tileMax_.get(),
              segmentMax_.get(), tileBest_.get(), nullptr, tileList_.get(), listed);
        });
      } else {
        profile_.run("tileExact", stream_, [&]() {
          tileExactKernel<MAXK, true><<<tiles, kThreads, 0, stream_>>>(
              src, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
              tileHist_.get(), nodeCounts_.get(), tileUpper, segmentLower, tileMax_.get(),
              segmentMax_.get(), tileBest_.get(), nullptr, nullptr, nullptr);
        });
      }
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
          columns_.get(), segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(),
          params_, tileHist_.get(), tileTries_.get());
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
    target.reserve(source.size(), stream_);
    profile_.run("uploadLevel", stream_, [&]() {
      CUDA_CHECK(cudaMemcpyAsync(target.get(), source.data(), source.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_));
    }, "copy");
  }

  template <typename T>
  void download(std::vector<T> &target, const DeviceArray<T> &source, std::size_t count,
                bool wait = true) {
    target.resize(count);
    profile_.run("downloadLevel", stream_, [&]() {
      CUDA_CHECK(cudaMemcpyAsync(target.data(), source.get(), count * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream_));
    }, "copy");
    if (wait) {
      trace::Doing waiting(trace::Work::GpuWait);
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
    std::vector<std::uint32_t> nodeCounts(nodeCount * classes);
    for (std::size_t n = 0; n < nodeCount; ++n) {
      const LevelNode &node = level[n];
      const std::uint32_t tilesPerSegment = (node.count + kTile - 1) / kTile;
      firstSegment[n] = segments.size();
      for (const std::uint32_t f : node.features) {
        const std::uint32_t index = static_cast<std::uint32_t>(segments.size());
        segments.push_back({columnBase(f) + node.begin, node.count, static_cast<std::uint32_t>(n),
                            f, static_cast<std::uint32_t>(tileSegment.size()),
                            tilesPerSegment});
        tileSegment.insert(tileSegment.end(), tilesPerSegment, index);
      }
      const std::uint32_t *counts = store_->counts(node.node);
      std::copy_n(counts, classes, nodeCounts.data() + n * classes);
      nodes[n].count = node.count;
      nodes[n].minChild = rules_->minChildRows(node.count);
      nodes[n].parentWeighted = weightedImpurity(counts, classCount_, node.count,
                                                 rules_->criterion(), rules_->logTable());
      floatGainFilter(classCount_, node.count, rules_->criterion(), nodes[n].filterScale,
                      nodes[n].filterMargin);
    }
    firstSegment[nodeCount] = segments.size();
    const std::size_t segmentCount = segments.size();
    const std::uint32_t tiles = static_cast<std::uint32_t>(tileSegment.size());
    upload(segments_, segments);
    upload(tileSegment_, tileSegment);
    upload(nodes_, nodes);
    upload(nodeCounts_, nodeCounts);
    constant_.reserve(segmentCount, stream_);
    tileHist_.reserve(static_cast<std::size_t>(tiles) * classes, stream_);
    tileBest_.reserve(tiles, stream_);
    tileMax_.reserve(tiles, stream_);
    tileTries_.reserve(tiles, stream_);
    segmentMax_.reserve(segmentCount, stream_);
    if (!onePass_ && classCount_ <= kMaxCandidateClasses) {
      const std::size_t candidateBytes = classCount_ <= 2   ? sizeof(Candidate<2>)
                                         : classCount_ <= 4 ? sizeof(Candidate<4>)
                                                            : sizeof(Candidate<8>);
      candidates_.reserve(static_cast<std::size_t>(tiles) * kCandidates * candidateBytes, stream_);
      candidateCount_.reserve(tiles, stream_);
      tileList_.reserve(static_cast<std::size_t>(tiles) + 1, stream_);
    }
    if (classCount_ == 2) {
      tileUpper_.reserve(tiles, stream_);
      tileLower_.reserve(tiles, stream_);
      segmentLower_.reserve(segmentCount, stream_);
    }
    tileStates_.reserve(static_cast<std::size_t>(tiles) + 1, stream_);
    segmentBest_.reserve(segmentCount, stream_);
    segmentLeft_.reserve(segmentCount * classes, stream_);

    // 0-4: constant segments, class counts left of every tile, then the best
    // cut per segment with its left class counts.
    profile_.run("segmentConstant", stream_, [&]() {
      segmentConstantKernel<<<static_cast<unsigned>((segmentCount + 255) / 256), 256, 0,
                              stream_>>>(columns_.get(), segments_.get(), segmentCount,
                                         params_.minGap, constant_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    if (!histogramReady_) { // else counted by the previous level's partition
      dispatchHistogram(tiles);
    }
    histogramReady_ = false;
    profile_.run("segmentPrefix", stream_, [&]() {
      segmentPrefixKernel<<<static_cast<unsigned>(segmentCount), kThreads, 0, stream_>>>(
          tileHist_.get(), classCount_, segments_.get(), constant_.get());
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
    std::vector<std::pair<std::size_t, int>> nextParent; // (node, side) of each of `next`
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
        addLeafJob(segments[firstSegment[n]].base, parent.count, parent.node);
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
          addLeafJob(segments[static_cast<std::size_t>(winnerSegment)].base + (begin - parent.begin),
                     count, left ? leftId : rightId);
          continue;
        }
        nodes[n].partition = 1;
        LevelNode child{left ? leftId : rightId, begin, count, depth, features};
        if (count >= minRows_) {
          next.push_back(std::move(child));
          nextParent.emplace_back(n, left ? 0 : 1);
        } else {
          toCpu.push_back(std::move(child));
        }
      }
      anyPartition = anyPartition || nodes[n].partition;
    }
    launchLeafJobs(); // reads this level's buffer, before handOff() reuses it
    if (!anyPartition) {
      return {};
    }

    // 5: stable partition of the non-constant columns of split nodes, each
    //    into a free slot (see PartitionTarget). It also counts the class
    //    histogram and the cuts of every tile of the next level, laid out as
    //    the next processLevel() lays it out (children in order, then their
    //    features).
    upload(nodes_, nodes);
    std::vector<PartitionTarget> targets(
        segmentCount, PartitionTarget{{kNoTarget, kNoTarget}, {0, 0}, {0, 0}});
    std::size_t nextTiles = 0;
    for (std::size_t c = 0; c < next.size(); ++c) {
      const LevelNode &child = next[c];
      const auto [n, side] = nextParent[c];
      const std::uint32_t tilesPerSegment = (child.count + kTile - 1) / kTile;
      std::size_t s = firstSegment[n];
      for (const std::uint32_t f : child.features) { // in the order of n's segments
        while (segments[s].feature != f) {
          ++s;
        }
        targets[s].firstTile[side] = static_cast<std::uint32_t>(nextTiles);
        targets[s].count[side] = child.count;
        targets[s].minChild[side] = rules_->minChildRows(child.count);
        nextTiles += tilesPerSegment;
      }
    }
    upload(partitionTargets_, targets);
    if (nextTiles > 0) {
      // This level is done with them (the kernels are ordered on the stream).
      tileHist_.reserve(nextTiles * classes, stream_);
      tileTries_.reserve(nextTiles, stream_);
      CUDA_CHECK(cudaMemsetAsync(tileHist_.get(), 0, nextTiles * classes * sizeof(std::uint32_t),
                                 stream_));
      CUDA_CHECK(cudaMemsetAsync(tileTries_.get(), 0, nextTiles * sizeof(std::uint32_t), stream_));
      histogramReady_ = true;
    }
    // The columns that move (those with a segment to partition), in feature
    // order. Each takes the first free slot: at first the slots that hold no
    // column of this level, then those that the moved columns leave.
    auto partitioned = [&](std::size_t s) {
      return !constant[s] && nodes[segments[s].node].partition;
    };
    std::vector<std::uint32_t> moveOf(features_, kNoTarget);
    for (std::size_t s = 0; s < segmentCount; ++s) {
      if (partitioned(s)) {
        moveOf[segments[s].feature] = 0;
      }
    }
    std::vector<bool> occupied(slotCount_, false);
    for (std::uint32_t f = 0; f < features_; ++f) {
      if (moveOf[f] != kNoTarget) {
        occupied[slot_[f]] = true;
      }
    }
    std::deque<std::pair<std::uint32_t, std::uint32_t>> freeSlots; // (slot, column leaving it)
    for (std::uint32_t slot = 0; slot < slotCount_; ++slot) {
      if (!occupied[slot]) {
        freeSlots.emplace_back(slot, kNoTarget);
      }
    }
    std::vector<std::uint32_t> leaving(features_, kNoTarget); // the column leaving f's new slot
    std::vector<long long> shift(features_, 0);
    for (std::uint32_t f = 0; f < features_; ++f) {
      if (moveOf[f] == kNoTarget) {
        continue;
      }
      const auto [slot, left] = freeSlots.front();
      freeSlots.pop_front();
      leaving[f] = left;
      shift[f] = (static_cast<long long>(slot) - static_cast<long long>(slot_[f])) *
                 static_cast<long long>(slotEntries_);
      freeSlots.emplace_back(slot_[f], f);
      slot_[f] = slot;
    }
    // What each segment waits for.
    std::vector<PartitionMove> moves(segmentCount, PartitionMove{0, kNoTarget});
    std::vector<std::uint32_t> segmentOf(features_, kNoTarget); // of the node, by feature
    for (std::size_t n = 0; n < nodeCount; ++n) {
      if (!nodes[n].partition) {
        continue;
      }
      for (std::size_t s = firstSegment[n]; s < firstSegment[n + 1]; ++s) {
        segmentOf[segments[s].feature] = partitioned(s) ? static_cast<std::uint32_t>(s) : kNoTarget;
      }
      for (std::size_t s = firstSegment[n]; s < firstSegment[n + 1]; ++s) {
        if (!partitioned(s)) {
          continue;
        }
        const std::uint32_t f = segments[s].feature;
        moves[s].shift = shift[f];
        const std::uint32_t waited = leaving[f] == kNoTarget ? kNoTarget : segmentOf[leaving[f]];
        if (waited != kNoTarget) {
          moves[s].waitTile = segments[waited].firstTile + segments[waited].tileCount - 1;
        }
      }
    }
    upload(partitionMoves_, moves);
    profile_.run("markGoesLeft", stream_, [&]() {
      CUDA_CHECK(cudaMemsetAsync(goesLeft_.get(), 0, (rows_ / 32 + 1) * sizeof(std::uint32_t),
                                 stream_));
      markGoesLeftKernel<<<tiles, kThreads, 0, stream_>>>(columns_.get(), segments_.get(),
                                                          tileSegment_.get(), nodes_.get(), codec_,
                                                          goesLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("partition", stream_, [&]() {
      // Tile states and the tile counter start at zero.
      CUDA_CHECK(cudaMemsetAsync(tileStates_.get(), 0, (tiles + 1) * sizeof(TileState), stream_));
      partitionKernel<<<tiles, kThreads, 0, stream_>>>(
          columns_.get(), columns_.get(), segments_.get(), tileSegment_.get(), constant_.get(),
          nodes_.get(), params_, goesLeft_.get(), partitionTargets_.get(), partitionMoves_.get(),
          tileStates_.get(),
          reinterpret_cast<unsigned int *>(tileStates_.get() + tiles), tileHist_.get(),
          tileTries_.get());
    });
    CUDA_CHECK(cudaGetLastError());

    // 6: small children -> CPU.
    if (!toCpu.empty()) {
      handOff(toCpu);
    }
    return next;
  }

  // The rows of a leaf made on the GPU: `count` entries of one of its columns
  // at `from` in the current buffer.
  void addLeafJob(std::size_t from, std::uint32_t count, std::uint32_t leaf) {
    if (!recordLeaves_) {
      return;
    }
    constexpr std::uint32_t kPiece = 1u << 14; // entries per block
    for (std::uint32_t done = 0; done < count; done += kPiece) {
      leafJobs_.push_back({from + done, std::min(kPiece, count - done), leaf});
    }
  }

  void launchLeafJobs() {
    if (leafJobs_.empty()) {
      return;
    }
    upload(leafJobsDevice_, leafJobs_);
    profile_.run("leafRows", stream_, [&]() {
      leafKernel<<<static_cast<unsigned>(leafJobs_.size()), kThreads, 0, stream_>>>(
          columns_.get(), leafJobsDevice_.get(), codec_, leafOfRow_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    leafJobs_.clear(); // a copy from pageable memory has read it on return
  }

  // The leaves of the rows that ended on the GPU, into `leafOfRow` (which
  // holds those of the CPU tasks).
  void mergeLeaves(decltype(RowLeaves::ids) &leafOfRow) {
    constexpr std::size_t kPiece = 1 << 16; // rows
    downloadChunks(leafOfRow_.get(), stride_ * sizeof(std::uint32_t), "downloadLeaves",
                   [&](std::size_t offset, const char *stage, std::size_t size) {
                     const auto *gpuLeaves = reinterpret_cast<const std::uint32_t *>(stage);
                     std::uint32_t *leaves = leafOfRow.data() + offset / sizeof(std::uint32_t);
                     const std::size_t count = size / sizeof(std::uint32_t);
                     pool_.parallelFor((count + kPiece - 1) / kPiece, [&](std::size_t piece) {
                       const std::size_t end = std::min(count, (piece + 1) * kPiece);
                       for (std::size_t row = piece * kPiece; row < end; ++row) {
                         if (gpuLeaves[row] != UINT32_MAX) {
                           leaves[row] = gpuLeaves[row];
                         }
                       }
                     });
                   });
  }

  // Copy the columns of `children` to the host arena as one block and start
  // a CPU task per child. Only the columns of features that can still split
  // a child are copied.
  //
  // The block goes in pieces of at most a ring part: each is gathered into
  // the next part of the ring (once that part's last copy is done) and
  // copied to the host on the copy streams, which start the CPU tasks
  // themselves once the last piece is copied (startTasks()). The GPU goes on
  // with the next levels meanwhile; it only waits when it gets a whole ring
  // ahead of the copies.
  //
  // The arena is pageable: a piece is downloaded into a page-locked
  // hand-off buffer, and a host function on the copy stream copies it into
  // the arena (drain()). The two copy streams take turns, so one downloads
  // while the other's buffer is copied out. (Buffers of 32 MB, four times
  // as many host functions, left the GPU levels of C4.5 on airline 0.2-0.4 s
  // slower on both GPU nodes: the host functions run one at a time.)
  // Page-locking the arena instead (75 GB for ECBDL'14) held the CUDA
  // driver's lock 1.2-1.9 s, sometimes 4-6 s on the H100 node, and the GPU
  // loop waited for it.
  void handOff(std::vector<LevelNode> &children) {
    NvtxRange range("hand-off copy");
    std::size_t handOffRows = 0;
    for (const LevelNode &child : children) {
      handOffRows += child.count;
    }
    // Block layout: column f of all children back to back, stride handOffRows,
    // cut into pieces of at most a ring part: copy jobs of up to kTile
    // entries, made in block order, relative to their piece.
    struct Piece {
      std::size_t begin, end; // in the block
      std::size_t firstJob, jobCount;
    };
    std::vector<Piece> pieces;
    std::vector<CopyJob> jobs;
    std::vector<std::size_t> next(children.size(), 0); // into each child's features
    for (std::uint32_t f = 0; f < features_; ++f) {
      std::size_t offset = 0;
      for (std::size_t c = 0; c < children.size(); ++c) {
        const LevelNode &child = children[c];
        if (next[c] < child.features.size() && child.features[next[c]] == f) {
          ++next[c];
          const std::size_t from = columnBase(f) + child.begin;
          const std::size_t to = f * handOffRows + offset;
          for (std::size_t at = to; at < to + child.count;) {
            if (pieces.empty() || at >= pieces.back().begin + ringPartEntries_) {
              pieces.push_back({at, at, jobs.size(), 0});
            }
            Piece &piece = pieces.back();
            const std::size_t end = std::min({to + child.count, piece.begin + ringPartEntries_,
                                              at + kTile});
            jobs.push_back({from + (at - to), at - piece.begin, static_cast<std::uint32_t>(end - at)});
            ++piece.jobCount;
            piece.end = end;
            at = end;
          }
        }
        offset += child.count;
      }
    }
    const std::size_t blockEntries = features_ * handOffRows;
    trace::add("handoff_rows", static_cast<double>(handOffRows));
    trace::add("handoff_nodes", static_cast<double>(children.size()));
    trace::add("handoff_bytes", static_cast<double>(blockEntries * sizeof(Entry)));
    trace::add("d2h_bytes", static_cast<double>(blockEntries * sizeof(Entry)));
    trace::add("handoffs", 1);
    waitTouched((arenaUsed_ + blockEntries) * sizeof(Entry));
    Entry *host = arena_.get() + arenaUsed_;
    arenaUsed_ += blockEntries;
    Entry *scratch = arenaScratch_.get() + scratchUsed_;
    scratchUsed_ += handOffRows;
    if (!jobs.empty()) {
      upload(copyJobs_, jobs);
    }
    staging().prepare();
    Staging &staging = dt::staging();
    for (const Piece &piece : pieces) {
      const std::size_t part = ringNext_++ % kRingParts;
      Entry *ring = ring_.get() + part * ringPartEntries_;
      CUDA_CHECK(cudaStreamWaitEvent(stream_, ringFree_[part], 0));
      profile_.run("handOffGather", stream_, [&]() {
        copyKernel<<<static_cast<unsigned>(piece.jobCount), kThreads, 0, stream_>>>(
            columns_.get(), ring, copyJobs_.get() + piece.firstJob);
      });
      CUDA_CHECK(cudaGetLastError());
      // Buffer b is only used on stream b: its next download waits there for
      // its copy into the arena.
      const int buffer = static_cast<int>(handOffNext_++ % Staging::kHandOffBuffers);
      cudaStream_t stream = copyStreams_[buffer];
      CUDA_CHECK(cudaEventRecord(gathered_, stream_));
      CUDA_CHECK(cudaStreamWaitEvent(stream, gathered_, 0));
      const std::size_t bytes = (piece.end - piece.begin) * sizeof(Entry);
      char *stage = staging.handOff[buffer];
      profile_.run("handOffCopy", stream, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(stage, ring, bytes, cudaMemcpyDeviceToHost, stream));
      }, "copy");
      CUDA_CHECK(cudaLaunchHostFunc(stream, drain, new Drain{this, host + piece.begin, stage, bytes}));
      CUDA_CHECK(cudaEventRecord(ringFree_[part], stream));
    }
    // Both streams' pieces are in the arena before the tasks start.
    CUDA_CHECK(cudaEventRecord(copied_, copyStreams_[1]));
    CUDA_CHECK(cudaStreamWaitEvent(copyStreams_[0], copied_, 0));
    auto tasks = std::make_unique<PendingTasks>();
    tasks->grower = this;
    tasks->columns = Columns{host, handOffRows, scratch};
    std::size_t offset = 0;
    for (LevelNode &child : children) {
      tasks->subtrees.push_back({child.node, static_cast<std::uint32_t>(offset), child.count,
                                 child.depth, std::move(child.features), {}, {}});
      offset += child.count;
    }
    CUDA_CHECK(cudaLaunchHostFunc(copyStreams_[0], startTasks, tasks.release()));
  }

  // One hand-off buffer to copy into the arena (a stream callback: no CUDA
  // calls). The copier threads are idle during the GPU levels.
  struct Drain {
    GpuGrower *grower;
    void *target;
    const char *stage;
    std::size_t bytes;
  };

  static void CUDART_CB drain(void *data) {
    std::unique_ptr<Drain> drain(static_cast<Drain *>(data));
    constexpr std::size_t kMinPart = std::size_t{1} << 20;
    trace::Doing copying(trace::Work::Staging);
    ThreadPool &copiers = staging().copyPool(drain->grower->pool_);
    const std::size_t parts = std::max<std::size_t>(
        1, std::min({Staging::kCopyThreads, std::size_t{copiers.workerCount()} + 1,
                     drain->bytes / kMinPart}));
    char *to = static_cast<char *>(drain->target);
    copiers.parallelFor(parts, [&](std::size_t part) {
      const std::size_t begin = drain->bytes * part / parts;
      const std::size_t end = drain->bytes * (part + 1) / parts;
      std::memcpy(to + begin, drain->stage + begin, end - begin);
    });
  }

  // The CPU tasks of one hand-off (a stream callback: no CUDA calls).
  struct PendingTasks {
    GpuGrower *grower;
    Columns columns;
    std::vector<Subtree> subtrees;
  };

  static void CUDART_CB startTasks(void *data) {
    std::unique_ptr<PendingTasks> tasks(static_cast<PendingTasks *>(data));
    GpuGrower *grower = tasks->grower;
    for (Subtree &subtree : tasks->subtrees) {
      grower->pool_.submit(
          [grower, columns = tasks->columns, subtree = std::move(subtree)]() mutable {
            NvtxRange range("cpu subtree");
            grower->cpuBuilder_->grow(columns, std::move(subtree));
          });
    }
  }

  const Dataset &train_;
  const Options &options_;
  ThreadPool &pool_;
  bool reusable_;
  bool presorted_ = false;
  bool onePass_ = false;
  int smCount_ = 1;
  std::size_t rows_;
  std::size_t features_;
  int classCount_;
  EntryCodec codec_;
  ScanParams params_{};

  // Per grow.
  std::size_t stride_ = 0; // rows of the current grow (column stride)
  std::size_t minRows_ = 0; // smaller nodes go to the CPU (gpu.minRows, or auto)
  const SplitRules *rules_ = nullptr;
  NodeStore *store_ = nullptr;
  CpuTreeBuilder *cpuBuilder_ = nullptr;

  std::unique_ptr<std::uint8_t[]> hostGoesLeft_;
  std::unique_ptr<Entry[]> arena_;
  std::unique_ptr<Entry[]> arenaScratch_;
  std::size_t arenaUsed_ = 0;
  std::size_t scratchUsed_ = 0;
  std::size_t handOffNext_ = 0; // the next of Staging::handOff
  std::thread toucher_; // touchArena()
  std::mutex touchMutex_;
  std::condition_variable touchCv_;
  std::size_t touchedBytes_ = 0; // the first bytes of the arena are touched (touchMutex_)

  cudaStream_t stream_ = nullptr;
  cudaStream_t copyStreams_[2] = {}; // hand-off copies to the host (handOff())
  cudaEvent_t gathered_ = nullptr;    // the last hand-off piece is gathered
  cudaEvent_t copied_ = nullptr;      // copyStreams_[1] is done with a hand-off

  // The sorted columns, one per slot of slotEntries_ entries; slot_[f] holds
  // feature f (see PartitionTarget).
  // One is enough: a column then waits for the one before it (3-5 ms of 500
  // on airline); every further slot costs 8 bytes per row.
  static constexpr std::size_t kSpareSlots = 1;
  DeviceArray<Entry> columns_;
  std::size_t slotCount_ = 0;
  std::vector<std::uint32_t> slot_;
  Entry *column(std::size_t feature) const { return columns_.get() + columnBase(feature); }
  std::size_t columnBase(std::size_t feature) const {
    return std::size_t{slot_[feature]} * slotEntries_;
  }
  static constexpr std::size_t kSlotAlign = (std::size_t{2} << 20) / sizeof(Entry); // 2 MiB
  std::size_t slotEntries_ = 0; // rows_, rounded up to kSlotAlign
  // Hand-off staging (see handOff()).
  static constexpr std::size_t kRingBytes = std::size_t{512} << 20;
  static constexpr std::size_t kRingParts = 4;
  static_assert(kRingBytes / kRingParts <= Staging::kHandOffBytes);
  DeviceArray<Entry> ring_;
  std::size_t ringPartEntries_ = 0;
  std::size_t ringNext_ = 0;
  cudaEvent_t ringFree_[kRingParts] = {}; // the last copy out of each part is done
  DeviceArray<float> values_;
  DeviceArray<float> rawValues_; // C4.5 with pruning: for arrangeRows()
  DeviceArray<std::uint16_t> labels_;
  DeviceArray<std::uint32_t> rowList_;
  DeviceArray<unsigned char> sortTemp_;
  std::size_t sortTempBytes_ = 0;
  DeviceArray<int> distinctCounts_; // distinctValues(): per feature
  DeviceArray<unsigned long long> distinctPlaces_; // ... where in the pool, then the pool's use
  DeviceArray<std::uint32_t> goesLeft_; // a bit per row: goes left (fits in L2)
  DeviceArray<Segment> segments_;
  DeviceArray<std::uint32_t> tileSegment_;
  DeviceArray<std::uint8_t> constant_;
  DeviceArray<std::uint32_t> tileHist_;
  DeviceArray<std::uint32_t> nodeCounts_; // class counts of the level's nodes
  DeviceArray<NodeDesc> nodes_;
  DeviceArray<BestCut> tileBest_;
  DeviceArray<float> tileMax_;
  DeviceArray<float> segmentMax_;
  DeviceArray<unsigned char> candidates_;    // tileEstimate: Candidate<MAXK>[kCandidates] per tile
  DeviceArray<std::uint32_t> candidateCount_;
  DeviceArray<std::uint32_t> tileList_;      // tiles left to tileExact, then their number
  DeviceArray<double> tileUpper_;    // two classes: tileBoundKernel
  DeviceArray<double> tileLower_;
  DeviceArray<double> segmentLower_;
  DeviceArray<std::uint32_t> tileTries_;
  DeviceArray<TileState> tileStates_; // partition: per tile, then the tile counter
  DeviceArray<PartitionTarget> partitionTargets_;
  DeviceArray<PartitionMove> partitionMoves_;
  bool histogramReady_ = false; // tileHist_ / tileTries_ hold the level's step 1
  DeviceArray<CutCandidate> segmentBest_;
  DeviceArray<std::uint32_t> segmentLeft_;
  DeviceArray<CopyJob> copyJobs_;
  DeviceArray<LeafJob> leafJobsDevice_;
  DeviceArray<std::uint32_t> leafOfRow_; // leaves of rows made on the GPU
  std::vector<LeafJob> leafJobs_;
  bool recordLeaves_ = false;
  DeviceArray<double> logTable_;
  KernelProfile profile_;
};

} // namespace

void gpuPrepare(unsigned threads) {
  CUDA_CHECK(cudaFree(nullptr));
  staging().prepare(threads);
}

std::unique_ptr<Grower> makeGpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool &pool, bool reusable, double &setupSeconds) {
  return std::make_unique<GpuGrower>(train, options, pool, reusable, setupSeconds);
}

} // namespace dt
