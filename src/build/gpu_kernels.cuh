#pragma once

// The GPU builder's kernels and the device-side structures they share with
// the host code. Included only by gpu_grower.cu (one translation unit); the
// steps are described at the top of that file.

#include "algo/split_math.h"

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace dt {

namespace {

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
  std::int32_t marksRight = 0;    // goesLeft holds the right rows (more left than right)
};

__host__ __device__ inline bool marksRight(const NodeDesc &node) { return node.marksRight != 0; }

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
// The unsorted column of one feature: entry i is row i, with its value from
// `values` (the feature's raw column).
__global__ void gatherKernel(const float *values, std::uint32_t count,
                             const std::uint16_t *labels, EntryCodec codec, Entry *column) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count) {
    return;
  }
  column[row] = {values[row] + 0.0f, codec.pack(row, labels[row])}; // -0.0 -> +0.0, as on the CPU
}

// The class counts of all rows (the root's), during the presort
// instead of countClasses() on the host (airline: 58 ms less on the H100
// node). A shared histogram per block, one atomic per distinct label in a
// warp; `counts` starts at zero.
__global__ void countClassesKernel(const std::uint16_t *labels, std::uint32_t count,
                                   int classCount, std::uint32_t *counts) {
  __shared__ std::uint32_t local[kMaxGpuClasses];
  for (int k = threadIdx.x; k < classCount; k += blockDim.x) {
    local[k] = 0;
  }
  __syncthreads();
  const std::uint32_t lane = threadIdx.x & 31;
  const std::size_t step = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t base = std::size_t{blockIdx.x} * blockDim.x; base < count; base += step) {
    const std::size_t index = base + threadIdx.x;
    const bool valid = index < count;
    const std::uint32_t label = valid ? labels[index] : 0xFFFFu;
    const unsigned same = __match_any_sync(0xFFFFFFFFu, label);
    if (valid && lane == static_cast<std::uint32_t>(__ffs(same) - 1)) {
      atomicAdd(local + label, static_cast<std::uint32_t>(__popc(same)));
    }
  }
  __syncthreads();
  for (int k = threadIdx.x; k < classCount; k += blockDim.x) {
    if (local[k]) {
      atomicAdd(counts + k, local[k]);
    }
  }
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
  // Only the rows of the smaller side: the bits start at zero on every
  // level. The left ones, or (marksRight) the right ones; the partition
  // flips the bit then.
  const bool right = marksRight(node);
  const std::uint32_t from = right ? node.leftCount : 0u;
  const std::uint32_t to = right ? node.count : node.leftCount;
  const std::uint32_t begin = from > start ? min(length, from - start) : 0u;
  const std::uint32_t end = min(length, to > start ? to - start : 0u);
  for (std::uint32_t index = begin + threadIdx.x; index < end; index += kThreads) {
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
  const std::uint32_t flip = marksRight(nodes[segment.node]) ? 1u : 0u;
  // In the split feature's own column the left rows are the first
  // leftCount entries: no goesLeft read.
  const bool byPosition =
      nodes[segment.node].splitFeature == static_cast<std::int32_t>(segment.feature);
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
    left[j] = first + j < length &&
              (byPosition ? start + first + j < leftCount
                          : (((goesLeft[row >> 5] >> (row & 31)) & 1u) ^ flip) != 0);
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

// Then, in that order: the row-major features (unless `rowMajor` is null:
// then `features` is 1), the labels, and the class counts of every slot. A
// thread per (position, feature): the writes are contiguous.
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
  if (rowMajor) {
    rowMajor[index] = values[feature * rows + row];
  }
  if (feature == 0) {
    const std::uint16_t label = labels[row];
    sortedLabels[position] = label;
    atomicAdd(&slotCounts[static_cast<std::size_t>(slots[position]) * classCount + label], 1u);
  }
}

} // namespace

} // namespace dt
