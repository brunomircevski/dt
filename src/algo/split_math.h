#pragma once

// Split-scoring math shared by the CPU builder (g++) and the GPU kernels
// (nvcc). Keeping one copy guarantees both backends score cuts identically.

#include "core/options.h"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define DT_HD __host__ __device__ __forceinline__
#else
#define DT_HD inline
#endif

namespace dt {

// ---------------------------------------------------------------------------
// Presorted entries
// ---------------------------------------------------------------------------
// For every feature the builders keep the node's rows sorted by that feature's
// value. One entry = the value plus (row id, class id) packed in 32 bits:
//     packed = row << classBits | class
// Carrying the class in the entry means a threshold sweep never has to look
// anything up by row id.
struct Entry {
  float value;
  std::uint32_t packed;
};
static_assert(sizeof(Entry) == 8, "Entry must stay 8 bytes");

struct EntryCodec {
  std::uint32_t classBits = 1;
  std::uint32_t classMask = 1;

  DT_HD std::uint32_t row(std::uint32_t packed) const { return packed >> classBits; }
  DT_HD std::uint32_t cls(std::uint32_t packed) const { return packed & classMask; }
  DT_HD std::uint32_t pack(std::uint32_t row, std::uint32_t cls) const {
    return (row << classBits) | cls;
  }
};

// ---------------------------------------------------------------------------
// Impurity
// ---------------------------------------------------------------------------
// Two gains closer than this are treated as equal; the earlier cut / lower
// feature index then wins. This makes results reproducible across CPU and GPU
// even though their log2() can differ in the last bit.
constexpr double kTieEps = 1e-12;

DT_HD double xlog2x(double x) { return x > 0.0 ? x * log2(x) : 0.0; }

// Entropy needs c * log2(c) for integer counts c. Small counts are looked up in
// a table (built once on the host and also copied to the GPU), which is much
// faster than log2() and gives both backends bit-identical values.
constexpr std::uint32_t kLogTableSize = 1u << 16;

struct LogTable {
  const double *values = nullptr; // values[c] = c * log2(c), c < kLogTableSize

  DT_HD double xlog2x(std::uint32_t count) const {
    return count < kLogTableSize ? values[count] : dt::xlog2x(static_cast<double>(count));
  }
};

// "Weighted impurity" of a node: n * impurity(node), from class counts.
//   Gini:    n - sum(c^2) / n                (n * (1 - sum p^2))
//   Entropy: n*log2(n) - sum(c * log2(c))     (n * H, C4.5's TotalInfo)
// Splitting a node with weighted impurity P into children L and R gives
//   gain = (P - L - R) / n
// which is the usual impurity decrease / information gain.
DT_HD double weightedImpurity(const std::uint32_t *counts, int classCount,
                              std::uint32_t n, Criterion criterion, LogTable logs) {
  if (n == 0) {
    return 0.0;
  }
  if (criterion == Criterion::Gini) {
    double sumSquares = 0.0;
    for (int k = 0; k < classCount; ++k) {
      const double c = static_cast<double>(counts[k]);
      sumSquares += c * c;
    }
    return static_cast<double>(n) - sumSquares / static_cast<double>(n);
  }
  double sum = 0.0;
  for (int k = 0; k < classCount; ++k) {
    sum += logs.xlog2x(counts[k]);
  }
  return logs.xlog2x(n) - sum;
}

// Gain of the cut that sends the rows counted in `left` to the left child.
// `total` are the node's class counts; right = total - left.
DT_HD double cutGain(const std::uint32_t *total, const std::uint32_t *left,
                     int classCount, std::uint32_t n, std::uint32_t nLeft,
                     double parentWeighted, Criterion criterion, LogTable logs) {
  const std::uint32_t nRight = n - nLeft;
  double leftWeighted;
  double rightWeighted;
  if (criterion == Criterion::Gini) {
    double leftSquares = 0.0;
    double rightSquares = 0.0;
    for (int k = 0; k < classCount; ++k) {
      const double l = static_cast<double>(left[k]);
      const double r = static_cast<double>(total[k] - left[k]);
      leftSquares += l * l;
      rightSquares += r * r;
    }
    leftWeighted = static_cast<double>(nLeft) - leftSquares / static_cast<double>(nLeft);
    rightWeighted = static_cast<double>(nRight) - rightSquares / static_cast<double>(nRight);
  } else {
    double leftSum = 0.0;
    double rightSum = 0.0;
    for (int k = 0; k < classCount; ++k) {
      leftSum += logs.xlog2x(left[k]);
      rightSum += logs.xlog2x(total[k] - left[k]);
    }
    leftWeighted = logs.xlog2x(nLeft) - leftSum;
    rightWeighted = logs.xlog2x(nRight) - rightSum;
  }
  return (parentWeighted - leftWeighted - rightWeighted) / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// Best cut of one feature
// ---------------------------------------------------------------------------
// Both algorithms pick the threshold of a feature by raw gain (C4.5 subtracts
// its MDL threshold cost afterwards, which is the same for every cut of the
// feature, so it does not change which cut wins).
struct CutCandidate {
  double gain = -INFINITY;      // impurity decrease of the best cut
  std::uint32_t leftCount = 0;  // rows going left (= position of the cut)
  std::uint32_t tries = 0;      // C4.5: distinct-value cuts allowed by MinSplit
  float leftValue = 0.0f;       // largest value going left
  float rightValue = 0.0f;      // smallest value going right

  DT_HD bool valid() const { return gain > -INFINITY; }
};

// Higher gain wins; (near-)ties go to the earlier cut.
DT_HD bool isBetterCut(double gain, std::uint32_t position, double bestGain,
                       std::uint32_t bestPosition) {
  if (gain > bestGain + kTieEps) {
    return true;
  }
  if (bestGain > gain + kTieEps) {
    return false;
  }
  return position < bestPosition;
}

DT_HD bool isBetterCandidate(const CutCandidate &a, const CutCandidate &b) {
  if (!a.valid()) {
    return false;
  }
  if (!b.valid()) {
    return true;
  }
  return isBetterCut(a.gain, a.leftCount, b.gain, b.leftCount);
}

// Can a threshold be placed between two adjacent sorted values? CART needs the
// values to differ; C4.5 (contin.c) needs them at least 1e-5 apart.
DT_HD bool isCut(float lower, float upper, double minGap) {
  return static_cast<double>(lower) < static_cast<double>(upper) - minGap;
}

// Boundary-point shortcut (Fayyad & Irani). Take cut i (between entries i-1
// and i) where both entries have the same class and each is the only entry with
// its value. Then cuts i-1, i, i+1 are all real thresholds and moving from one
// to the next moves one row of that same class, and the gain is convex in that
// move (true for entropy and Gini), so gain(i) <= max(gain(i-1), gain(i+1)):
// cut i can never be the first best cut and its gain need not be computed.
// The neighbours must be allowed cuts too (minChild rows on each side);
// otherwise cut i may be the best allowed one. Cuts next to a run of equal
// values are always evaluated: a run that mixes classes can make them optimal.
DT_HD bool isSkippableCut(bool sameClass, bool leftIsSingleton, bool rightIsSingleton,
                          std::uint32_t cut, std::uint32_t count, std::uint32_t minChild) {
  return sameClass && leftIsSingleton && rightIsSingleton && cut > minChild &&
         cut + minChild < count;
}

} // namespace dt
