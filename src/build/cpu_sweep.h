#pragma once

// The threshold sweep of the CPU builder for two classes (TwoClassSweep) and
// its shared pieces. Internal to src/build: cpu_sweep.cpp runs the sweeps of
// CpuTreeBuilder::scanFeature(), cpu_builder.cpp the ones fused into the
// partition of a node's columns (partitionAndSweep()).

#include "algo/split_math.h"
#include "build/cpu_builder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace dt {

// Candidates a sweep buffers before settling them (see scanFeatureK()).
constexpr std::uint32_t kSweepBlock = 256;
// Candidates buffered before the first settling (then twice as many each
// time, up to kSweepBlock), so that small nodes get a threshold early.
constexpr std::uint32_t kFirstSettle = 16;
// Unit roundoff of double.
constexpr double kRoundoff = 0x1p-53;

// An upper bound, times n, on how much a cut's gain can grow when one more
// row moves left. With S = sum l_k^2 / n_L + sum r_k^2 / n_R, the Gini gain is
// (P - n + S) / n, and moving a row raises the left part of S by less than 2
// and the right part by at most 1. The entropy gain is (P - W) / n with W the
// weighted entropy, and moving a row of class c changes W by
// d(n_L) - d(l_c) - d(n_R - 1) + d(r_c - 1), d(x) = (x + 1) log2(x + 1) -
// x log2(x) increasing, so it lowers W by at most d(n_R - 1) <=
// log2(n) + log2(e). Both rounded up.
template <Criterion Crit> double maxGainStep(std::uint32_t count) {
  if constexpr (Crit == Criterion::Gini) {
    return 4.0;
  } else {
    return std::log2(static_cast<double>(count)) + 2.0;
  }
}

// Skipping cuts that cannot win. After a cut with gain estimate `gain`, the
// exact gain of the cut j steps further is at most gain + margin + j * step / n
// (maxGainStep), so the next skippableSteps() cuts provably cannot beat
// `bestGain` by kTieEps (rounding: another margin, and a small factor on the
// step). Short skips do not pay off (the loop exit is mispredicted), so a
// sweep only skips after estimates below skipBelow(): at least kMinSkip steps.
constexpr double kMinSkip = 8.0;

inline double skipBelow(double bestGain, double margin, double rowsPerGain) {
  return bestGain + kTieEps - 2.0 * margin - kMinSkip / rowsPerGain;
}

inline std::uint32_t skippableSteps(double bestGain, double gain, double margin, double rowsPerGain,
                             std::uint32_t limit) {
  const double steps = (bestGain + kTieEps - gain - 2.0 * margin) * rowsPerGain * (1.0 - 1e-9);
  return steps >= static_cast<double>(limit) ? limit : static_cast<std::uint32_t>(steps);
}

// Bounded blocks (two classes): the cuts of a node with at least
// kBoundedSweepRows rows are taken in blocks of kBoundBlock, and a block whose
// upper bound cannot beat the best cut so far is skipped. The bound: both
// impurities are concave in a child's class counts (n * Gini and n * entropy
// are perspectives of concave functions), so the gain is convex in the left
// class counts, and its maximum over the box of counts a block passes through
// is at one of the box's four corners. Rounding: the estimate is within
// `margin` of cutGain(), and both are within a few roundings of the exact
// function (entropy: of terms up to n * log2(n), the tabled c * log2(c)
// included), which `boundSlack` covers twice.
constexpr std::uint32_t kBoundBlock = 16;
constexpr std::uint32_t kBoundedSweepRows = 64;
// Parallel Gini sweeps of nodes with at least this many rows divide instead
// of looking up 1/n in the count tables: those are as big as the training set,
// and with many threads reading them costs more memory bandwidth than the
// entries (on one thread the lookups are faster).
constexpr std::uint32_t kDividingSweepRows = 1u << 16;
// After a skipped block, further cuts are skipped by the per-row bound
// (maxGainStep) if that covers at least this many.
constexpr std::uint32_t kMinFurtherSkip = 32;


// Exact part of a sweep (see scanFeatureK()): the buffered candidates, in
// sweep order, go through the rule "keep a cut if its gain beats the best by
// more than kTieEps", with cutGain(). Candidates before the last one whose
// estimate beats all earlier buffered estimates by more than
// 2 * margin + kTieEps are skipped. `countsAt(i, out)` writes the class counts
// left of candidate i into `counts` (classCount values).
// Returns the buffer index of the last candidate kept, or -1.
template <class CountsAt>
int settleCandidates(const double *estimate, const std::uint32_t *position, std::uint32_t size,
                     double margin, const std::uint32_t *total, int classCount,
                     std::uint32_t count, double parentWeighted, Criterion criterion,
                     LogTable logs, CountsAt &&countsAt, std::uint32_t *counts,
                     double &bestGain, std::uint32_t &bestCut, std::uint32_t *bestLeft) {
  int kept = -1;
  std::uint32_t from = 0;
  double highest = -INFINITY;
  const double clearLead = 2.0 * margin + kTieEps;
  for (std::uint32_t index = 0; index < size; ++index) {
    if (estimate[index] > highest + clearLead) {
      from = index;
    }
    highest = std::max(highest, estimate[index]);
  }
  for (std::uint32_t index = from; index < size; ++index) {
    if (estimate[index] <= bestGain + kTieEps - margin) {
      continue;
    }
    countsAt(index, counts);
    const double gain =
        cutGain(total, counts, classCount, count, position[index], parentWeighted, criterion, logs);
    if (gain > bestGain + kTieEps) {
      bestGain = gain;
      bestCut = position[index];
      std::copy_n(counts, classCount, bestLeft);
      kept = static_cast<int>(index);
    }
  }
  return kept;
}

// The sweep of scanFeatureK() for two classes (e.g. SUSY, HIGGS), with the
// same estimate / candidate logic but leaner: the class counts at a cut follow
// from the number of class-1 rows left of it, the estimates need no running
// sums (Gini: no division, with tabled reciprocals; entropy: cutGain() with a
// multiplication by 1/n instead of the division), the boundary-point shortcut
// is decided without a branch (on continuous features it is a coin flip), and
// whole blocks of cuts are skipped when a bound shows that none of them can
// beat the best cut so far (see kBoundBlock).
//
// It can run in pieces, so that a child's sweep runs while the child's column
// is being written by the parent's partition: advance() takes the cuts whose
// entries are available. The count tables must cover `count`; HasGap: C4.5's
// minimum gap between values (CART: values must just differ).
struct SweepSetup {
  EntryCodec codec;
  const double *xlog = nullptr;    // CountTables
  const double *inverse = nullptr;
  double gap = 0.0;                // SplitRules::minValueGap()
  LogTable logs;
  bool divide = false;             // see kDividingSweepRows
};

template <Criterion Crit, bool HasGap> class TwoClassSweep {
public:
  using Setup = SweepSetup;

  TwoClassSweep(const Setup &setup, std::uint32_t count, const std::uint32_t *total,
                double parentWeighted, std::uint32_t minChild)
      : setup_(setup), count_(count), total_{total[0], total[1]},
        parentWeighted_(parentWeighted), minChild_(minChild), rows_(count),
        inverseRows_(1.0 / rows_), giniBase_(parentWeighted - rows_),
        boundSlack_(Crit == Criterion::Gini
                        ? 4.0 * kMargin
                        : 4.0 * kMargin + 64.0 * (std::log2(rows_) + 2.0) * kRoundoff),
        empty_(count < 2 * minChild), lastCut_(empty_ ? 0 : count - minChild),
        blockCuts_(count >= kBoundedSweepRows ? kBoundBlock : lastCut_ + 1), cut_(minChild) {}

  // The first entry that later advance() calls still read.
  std::uint32_t from() const { return started_ && cut_ >= 2 ? cut_ - 2 : 0; }

  // Takes the cuts whose entries are among the first `available` (all cuts if
  // available == count). entries[i] must be readable for i in
  // [from(), available).
  void advance(const Entry *entries, std::uint32_t available) {
    if (empty_) {
      return;
    }
    const EntryCodec codec = setup_.codec;
    if (!started_) {
      if (available < minChild_ + 1 && available < count_) {
        return;
      }
      for (std::uint32_t index = 0; index + 1 < minChild_; ++index) {
        ones_ += codec.cls(entries[index].packed);
      }
      started_ = true;
    }
    // Locals, so that they stay in registers (the buffers are members).
    std::uint32_t cut = cut_;
    std::uint32_t ones = ones_;
    std::uint32_t tries = tries_;
    std::uint32_t found = found_;
    double threshold = threshold_;
    double bestGain = bestGain_;
    const std::uint32_t count = count_;
    const std::uint32_t minChild = minChild_;
    const std::uint32_t lastCut = lastCut_;
    const std::uint32_t blockCuts = blockCuts_;
    const bool bounded = blockCuts == kBoundBlock && count >= kBoundedSweepRows;
    const double boundSlack = boundSlack_;
    const double inverseRows = inverseRows_;
    const double giniBase = giniBase_;
    const double parentWeighted = parentWeighted_;
    const double rowsPerGain = rows_ / maxGainStep<Crit>(count);
    const std::uint32_t total0 = total_[0];
    const std::uint32_t total1 = total_[1];
    const double total0Value = total0;
    const double total1Value = total1;
    const double *xlog = setup_.xlog;
    const double *inverse = setup_.inverse;
    const double gap = setup_.gap;
    auto distinct = [gap](float lower, float upper) {
      if constexpr (HasGap) {
        return isCut(lower, upper, gap);
      } else {
        return lower < upper; // isCut() with no gap
      }
    };
    // The estimate at a cut with left class counts (left0, left1).
    const bool divide = setup_.divide && count >= kDividingSweepRows;
    auto estimateAt = [&](std::uint32_t left0, std::uint32_t left1) {
      const std::uint32_t leftCount = left0 + left1;
      if constexpr (Crit == Criterion::Gini) {
        const double l0 = left0;
        const double l1 = left1;
        const double r0 = total0Value - l0;
        const double r1 = total1Value - l1;
        const double leftSquares = l0 * l0 + l1 * l1;
        const double rightSquares = r0 * r0 + r1 * r1;
        if (divide) {
          // S_L / n_L + S_R / n_R with one division and no table (big
          // tables cost memory bandwidth): the squares and n_L * n_R are
          // exact, so this is within a few roundings too.
          const double leftRows = leftCount;
          const double rightRows = count - leftCount;
          return (giniBase + (leftSquares * rightRows + rightSquares * leftRows) /
                                 (leftRows * rightRows)) *
                 inverseRows;
        }
        return (giniBase + leftSquares * inverse[leftCount] +
                rightSquares * inverse[count - leftCount]) *
               inverseRows;
      } else {
        const double leftWeighted = xlog[leftCount] - (xlog[left0] + xlog[left1]);
        const double rightWeighted =
            xlog[count - leftCount] - (xlog[total0 - left0] + xlog[total1 - left1]);
        return (parentWeighted - leftWeighted - rightWeighted) * inverseRows;
      }
    };
    // Candidates are settled before advance() returns, so that the entries
    // around them can still be read.
    auto settleNow = [&]() {
      found_ = found;
      settle(entries);
      found = 0;
      threshold = threshold_;
      bestGain = bestGain_;
    };
    double *estimate = estimate_;
    std::uint32_t *position = position_;
    std::uint32_t *onesAt = onesAt_;
    while (cut <= lastCut) {
      const std::uint32_t blockEnd = std::min(cut + blockCuts, lastCut + 1); // cuts [cut, blockEnd)
      if (blockEnd >= available && available < count) {
        break; // its last cut needs entry blockEnd
      }
      if (bounded) {
        // Moving from cut `cut` to cut blockEnd - 1 moves entries [cut - 1,
        // blockEnd - 2] left, so the block's left class counts lie in the box
        // [l0lo, l0hi] x [l1lo, l1hi]. The gain is convex in them, so its
        // largest value over the box is at a corner; if that cannot beat the
        // best cut so far, no cut of the block can (see kBoundBlock).
        std::uint32_t blockOnes = 0;
        for (std::uint32_t index = cut - 1; index + 1 < blockEnd; ++index) {
          blockOnes += codec.cls(entries[index].packed);
        }
        const std::uint32_t l1lo = ones + codec.cls(entries[cut - 1].packed);
        const std::uint32_t l1hi = ones + blockOnes;
        const std::uint32_t l0lo = cut - l1lo;
        const std::uint32_t l0hi = blockEnd - 1 - l1hi;
        const double upper =
            std::max(std::max(estimateAt(l0lo, l1lo), estimateAt(l0hi, l1hi)),
                     std::max(estimateAt(l0hi, l1lo), estimateAt(l0lo, l1hi))) +
            boundSlack;
        bool skip = upper <= bestGain + kTieEps;
        if (!skip && found > 0) {
          settleNow(); // the exact best so far may already be high enough
          skip = upper <= bestGain + kTieEps;
        }
        if (skip) {
          // Far below the best cut, the cuts after the block cannot reach it
          // either (the gain grows by at most maxGainStep / n per row): they
          // are skipped too, without the bound's table lookups.
          const std::uint32_t limit = (available < count ? available - 1 : lastCut + 1) - blockEnd;
          const std::uint32_t further =
              skippableSteps(bestGain, upper, kMargin, rowsPerGain, limit);
          const std::uint32_t skipEnd = blockEnd + (further >= kMinFurtherSkip ? further : 0);
          for (std::uint32_t index = blockEnd - 1; index + 1 < skipEnd; ++index) {
            blockOnes += codec.cls(entries[index].packed);
          }
          ones += blockOnes;
          if constexpr (HasGap) { // C4.5 counts the thresholds (MDL cost)
            std::uint32_t blockTries = 0;
            for (std::uint32_t index = cut; index < skipEnd; ++index) {
              blockTries += distinct(entries[index - 1].value, entries[index].value);
            }
            tries += blockTries;
          }
          cut = skipEnd;
          continue;
        }
      }
      for (; cut < blockEnd; ++cut) {
        const Entry previous = entries[cut - 1];
        const Entry current = entries[cut];
        const std::uint32_t previousClass = codec.cls(previous.packed);
        ones += previousClass;
        if (!distinct(previous.value, current.value)) {
          continue;
        }
        ++tries;
        const bool leftDistinct = cut < 2 || distinct(entries[cut - 2].value, previous.value);
        const bool rightDistinct =
            cut + 1 >= count || distinct(current.value, entries[cut + 1].value);
        const bool skippable = (previousClass == codec.cls(current.packed)) & leftDistinct &
                               rightDistinct & (cut > minChild) & (cut + minChild < count);
        const double gain = estimateAt(cut - ones, ones);
        // Buffered unless skipped, without a branch on `skippable`.
        estimate[found] = gain;
        position[found] = cut;
        onesAt[found] = ones;
        found += !skippable & (gain > threshold);
        if (found == settleAt_) {
          settleNow();
        }
      }
    }
    if (found > 0) {
      settleNow();
    }
    cut_ = cut;
    ones_ = ones;
    tries_ = tries;
    found_ = found;
  }

  // The best cut, after advance(..., count); `bestLeft` receives its class
  // counts.
  CutCandidate finish(std::uint32_t *bestLeft) {
    CutCandidate best;
    if (empty_) {
      return best;
    }
    best.tries = tries_;
    if (bestCut_ > 0) {
      best.gain = bestGain_;
      best.leftCount = bestCut_;
      best.leftValue = bestValues_.first;
      best.rightValue = bestValues_.second;
      bestLeft[0] = bestLeft_[0];
      bestLeft[1] = bestLeft_[1];
    }
    return best;
  }

private:
  static constexpr double kMargin = 256.0 * kRoundoff; // a few roundings of terms <= n, / n

  void settle(const Entry *entries) {
    std::uint32_t counts[2];
    const int kept = settleCandidates(
        estimate_, position_, found_, kMargin, total_, 2, count_, parentWeighted_, Crit,
        setup_.logs,
        [&](std::uint32_t index, std::uint32_t *out) {
          out[0] = position_[index] - onesAt_[index];
          out[1] = onesAt_[index];
        },
        counts, bestGain_, bestCut_, bestLeft_);
    if (kept >= 0) {
      bestValues_ = {entries[bestCut_ - 1].value, entries[bestCut_].value};
    }
    found_ = 0;
    settleAt_ = std::min(2 * settleAt_, kSweepBlock);
    threshold_ = bestGain_ + kTieEps - kMargin;
  }

  Setup setup_;
  std::uint32_t count_;
  std::uint32_t total_[2];
  double parentWeighted_;
  std::uint32_t minChild_;
  double rows_;
  double inverseRows_;
  double giniBase_;
  // Bound on |estimate - exact gain| at any count vector, the rounding of the
  // tabled c*log2(c) (terms up to n*log2(n)) included: see kBoundBlock.
  double boundSlack_;
  bool empty_;
  std::uint32_t lastCut_;
  std::uint32_t blockCuts_;

  bool started_ = false;
  std::uint32_t cut_;      // next cut to take
  std::uint32_t ones_ = 0; // class-1 rows among the entries before cut_ - 1
  std::uint32_t tries_ = 0;
  double bestGain_ = -INFINITY;
  std::uint32_t bestCut_ = 0;
  std::uint32_t bestLeft_[2] = {0, 0};
  std::pair<float, float> bestValues_{0.0f, 0.0f};
  double threshold_ = -INFINITY; // estimates at or below it cannot beat the best cut
  std::uint32_t found_ = 0;
  std::uint32_t settleAt_ = kFirstSettle;
  double estimate_[kSweepBlock];
  std::uint32_t position_[kSweepBlock];
  std::uint32_t onesAt_[kSweepBlock];
};

} // namespace dt
