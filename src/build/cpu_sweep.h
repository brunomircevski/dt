#pragma once

// The threshold sweeps of the CPU builder, for two classes (TwoClassSweep)
// and for any number (KClassSweep), and their shared pieces. Internal to
// src/build: cpu_sweep.cpp runs the sweeps of CpuTreeBuilder::scanFeature(),
// cpu_builder.cpp the ones fused into the partition of a node's columns
// (partitionAndSweep()).

#include "algo/split_math.h"
#include "build/cpu_builder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace dt {

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

// Skipping cuts that cannot win. After a cut with gain `gain` (within
// `margin` of the exact function), the exact gain of the cut j steps further is at most gain + margin + j * step / n
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
// is at one of the box's four corners (as in tileBoundKernel on the GPU).
// Rounding: the corners' estimate is within
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


// The sweep of scanFeatureK() for two classes (e.g. SUSY, HIGGS): the class
// counts at a cut follow from the number of class-1 rows left of it, every
// cut the boundary-point shortcut leaves gets cutGain(), and whole blocks of
// cuts are skipped when they hold no threshold or a bound shows that none of
// them can beat the best cut so far (see kBoundBlock; only the bound's
// corners use a cheaper estimate: Gini with tabled reciprocals, entropy with
// tabled c*log2(c), both times 1/n).
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
  int classCount = 2;
};

template <Criterion Crit, bool HasGap, bool InChunks = false> class TwoClassSweep {
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

  // InChunks: a chunk of a chunked sweep (CpuTreeBuilder::sweepInChunks()).
  // It takes only the cuts [firstCut, endCut) (within [minChild, count -
  // minChild]), as if the sweep had reached firstCut with `ones` class-1
  // entries before entry firstCut - 1 and `bestGain` as the best gain so far;
  // then advance(entries, count). It appends every pick to `record` if not
  // null, else it stops at its first pick that is one of [stopFrom, stopEnd)
  // (increasing cuts).
  struct Pick {
    double gain;
    std::uint32_t cut;
    std::uint32_t ones; // class-1 entries left of the cut
  };
  void restrict(std::uint32_t firstCut, std::uint32_t endCut, std::uint32_t ones, double bestGain,
                std::vector<Pick> *record, const Pick *stopFrom = nullptr,
                const Pick *stopEnd = nullptr) {
    static_assert(InChunks);
    cut_ = firstCut;
    lastCut_ = endCut - 1;
    ones_ = ones;
    bestGain_ = bestGain;
    started_ = true;
    record_ = record;
    stop_ = stopFrom;
    stopEnd_ = stopEnd;
  }

  // Whether advance() stopped at a pick of [stopFrom, stopEnd).
  bool stopped() const { return stopped_; }

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
    // The estimate at a bound's corner with left class counts (left0, left1).
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
        if (blockEnd - cut == kBoundBlock) { // a fixed count: the compiler vectorizes it
          for (std::uint32_t index = 0; index < kBoundBlock; ++index) {
            blockOnes += codec.cls(entries[cut - 1 + index].packed);
          }
        } else {
          for (std::uint32_t index = cut - 1; index + 1 < blockEnd; ++index) {
            blockOnes += codec.cls(entries[index].packed);
          }
        }
        if (!distinct(entries[cut - 1].value, entries[blockEnd - 1].value)) {
          // No threshold in the block (sorted values: if its first and last
          // ones are too close for a cut, so are all its neighbours), e.g.
          // inside a run of equal values: only its rows move.
          ones += blockOnes;
          cut = blockEnd;
          continue;
        }
        const std::uint32_t l1lo = ones + codec.cls(entries[cut - 1].packed);
        const std::uint32_t l1hi = ones + blockOnes;
        const std::uint32_t l0lo = cut - l1lo;
        const std::uint32_t l0hi = blockEnd - 1 - l1hi;
        const double upper =
            std::max(std::max(estimateAt(l0lo, l1lo), estimateAt(l0hi, l1hi)),
                     std::max(estimateAt(l0hi, l1lo), estimateAt(l0lo, l1hi))) +
            boundSlack;
        if (upper <= bestGain + kTieEps) {
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
        if (skippable) {
          continue;
        }
        const std::uint32_t counts[2] = {cut - ones, ones};
        const double gain =
            cutGain(total_, counts, 2, count, cut, parentWeighted, Crit, setup_.logs);
        if (gain > bestGain + kTieEps) {
          bestGain = gain;
          bestCut_ = cut;
          bestLeft_[0] = counts[0];
          bestLeft_[1] = counts[1];
          bestValues_ = {previous.value, current.value};
          if constexpr (InChunks) {
            if (record_) {
              record_->push_back({gain, cut, ones});
            } else {
              while (stop_ != stopEnd_ && stop_->cut < cut) {
                ++stop_;
              }
              if (stop_ != stopEnd_ && stop_->cut == cut) {
                stopped_ = true;
                cut = lastCut + 1;
                break;
              }
            }
          }
        }
      }
    }
    cut_ = cut;
    ones_ = ones;
    tries_ = tries;
    bestGain_ = bestGain;
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
  // InChunks (see restrict()).
  std::vector<Pick> *record_ = nullptr;
  const Pick *stop_ = nullptr;
  const Pick *stopEnd_ = nullptr;
  bool stopped_ = false;
};

// The sweep of scanFeatureK() for any number of classes: FixedK of them, or
// 0 for setup.classCount. Cut i sits between entries i-1 and i and sends the
// first i rows left. Rows are moved left one by one, which only updates the
// class counts (most rows of discrete features sit inside runs of equal
// values, which a tight loop moves). Every cut with a threshold that the
// boundary-point shortcut leaves gets cutGain(), and a cut is kept when its
// gain beats the best so far by more than kTieEps. After a cut far below the
// best, the next cuts that provably cannot reach it (maxGainStep) only move
// their rows.
//
// It runs in pieces like TwoClassSweep (same interface): a piece stops before
// a cut whose entries are not all available (cut i reads entries i-2 to i+1,
// a run or a skip up to the cut it stops at) and the next one goes on in the
// same state, so the cuts, their order and the result do not depend on the
// pieces. `total` must stay valid until finish().
template <int FixedK, Criterion Crit> class KClassSweep {
public:
  using Setup = SweepSetup;

  KClassSweep(const Setup &setup, std::uint32_t count, const std::uint32_t *total,
              double parentWeighted, std::uint32_t minChild)
      : setup_(setup), count_(count), total_(total), parentWeighted_(parentWeighted),
        minChild_(minChild), classCount_(FixedK > 0 ? FixedK : setup.classCount),
        empty_(count < 2 * minChild), lastCut_(empty_ ? 0 : count - minChild), cut_(minChild) {
    const double rows = static_cast<double>(count);
    // Bound on the rounding of cutGain(): a generous count of roundings times
    // the size of the terms (Gini: at most n; entropy: at most n * log2(n)),
    // divided by n.
    if constexpr (Crit == Criterion::Gini) {
      margin_ = 256.0 * kRoundoff;
    } else {
      const double scale = std::max(1.0, setup.logs.xlog2x(count)) / rows;
      margin_ = 2.0 * ((4.0 * classCount_ + 16.0) * scale + 64.0) * kRoundoff;
    }
    rowsPerGain_ = rows / maxGainStep<Crit>(count);
    if constexpr (FixedK == 0) {
      dynamic_.assign(2 * static_cast<std::size_t>(classCount_), 0u);
    }
  }

  // The first entry that later advance() calls still read.
  std::uint32_t from() const { return started_ && cut_ >= 2 ? cut_ - 2 : 0; }

  // Takes the cuts whose entries are among the first `available` (all cuts if
  // available == count). entries[i] must be readable for i in
  // [from(), available).
  void advance(const Entry *entries, std::uint32_t available) {
    if (empty_) {
      return;
    }
    // A constant for FixedK (the class loops unroll; with the member, 7
    // classes ran 35% more instructions than a sweep in one function).
    const int classCount = FixedK > 0 ? FixedK : classCount_;
    const std::size_t classes = static_cast<std::size_t>(classCount);
    // left: class counts of the rows moved left so far (fixed K: a local
    // copy, which the entries cannot alias).
    std::uint32_t fixedLeft[kFixedClasses];
    std::uint32_t *left = fixedLeft;
    if constexpr (FixedK == 0) {
      left = dynamic_.data();
    } else {
      std::copy_n(fixed_, FixedK, fixedLeft);
    }
    std::uint32_t *const bestLeft = FixedK > 0 ? fixed_ + kFixedClasses : left + classes;
    const EntryCodec codec = setup_.codec;
    if (!started_) {
      if (available < minChild_ + 1 && available < count_) {
        return;
      }
      for (std::uint32_t index = 0; index + 1 < minChild_; ++index) {
        ++left[codec.cls(entries[index].packed)];
      }
      started_ = true;
    }

    const double gap = setup_.gap;
    const LogTable logs = setup_.logs;
    const std::uint32_t count = count_;
    const std::uint32_t minChild = minChild_;
    const std::uint32_t lastCut = lastCut_;
    const std::uint32_t *const total = total_;
    const double parentWeighted = parentWeighted_;
    const double margin = margin_;
    const double rowsPerGain = rowsPerGain_;
    std::uint32_t tries = tries_;
    double bestGain = bestGain_;
    std::uint32_t bestCut = bestCut_;
    std::pair<float, float> bestValues = bestValues_;
    double skipThreshold = skipThreshold_;
    std::uint32_t skipEnd = skipEnd_;
    // Cuts below `limit` have all their entries (cut i reads up to i + 1).
    const std::uint32_t limit =
        available >= count ? lastCut + 1 : std::min(lastCut + 1, available - 1);

    constexpr std::size_t kLanes = 4;
    constexpr std::uint32_t kRunStep = 16; // see moveRows
    std::uint32_t fixedLaneCounts[kLanes * kFixedClasses];
    std::uint32_t *laneCounts = fixedLaneCounts;
    if constexpr (FixedK == 0) {
      thread_local std::vector<std::uint32_t> dynamicLaneCounts;
      dynamicLaneCounts.resize(kLanes * classes);
      laneCounts = dynamicLaneCounts.data();
    }
    // Moves the rows of the cuts from `cut` on left, up to `end` or, with
    // `untilThreshold`, up to the first cut with a threshold, into interleaved
    // counters (rows of the same class in a row then do not wait for each
    // other's increment). Returns where it stopped and how many of the passed
    // cuts have a threshold.
    auto moveRows = [&](std::uint32_t cut, std::uint32_t end, bool untilThreshold) {
      std::fill_n(laneCounts, kLanes * classes, 0u);
      std::uint32_t thresholds = 0;
      if (untilThreshold) {
        // Long runs (discrete features) without comparing every pair: the
        // values are sorted, so if entry `last` is too close to entry
        // `safe - 1` for a threshold, no cut in [safe, last] has one. The step
        // doubles while that holds and halves when it does not. (So where
        // `end` cuts a run does not change where it stops.)
        std::uint32_t safe = cut; // the cuts in [cut, safe) have no threshold
        for (std::uint32_t step = kRunStep; step >= kRunStep;) {
          const std::uint32_t last = safe - 1 + step;
          if (last < end && !isCut(entries[safe - 1].value, entries[last].value, gap)) {
            safe = last + 1;
            step *= 2;
          } else {
            step /= 2;
          }
        }
        std::uint32_t *const lane[kLanes] = {laneCounts, laneCounts + classes,
                                             laneCounts + 2 * classes, laneCounts + 3 * classes};
        for (; cut + kLanes <= safe; cut += kLanes) {
          for (std::size_t k = 0; k < kLanes; ++k) {
            ++lane[k][codec.cls(entries[cut - 1 + k].packed)];
          }
        }
        for (; cut < safe; ++cut) {
          ++laneCounts[codec.cls(entries[cut - 1].packed)];
        }
      }
      for (; cut < end; ++cut) {
        const bool threshold = isCut(entries[cut - 1].value, entries[cut].value, gap);
        if (untilThreshold && threshold) {
          break;
        }
        ++laneCounts[(cut & (kLanes - 1)) * classes + codec.cls(entries[cut - 1].packed)];
        thresholds += threshold;
      }
      for (std::size_t lane = 0; lane < kLanes; ++lane) {
        for (std::size_t k = 0; k < classes; ++k) {
          left[k] += laneCounts[lane * classes + k];
        }
      }
      return std::pair<std::uint32_t, std::uint32_t>{cut, thresholds};
    };

    std::uint32_t cut = cut_;
    if (cut < skipEnd) { // a skip the last piece left unfinished
      const std::uint32_t end = std::min(skipEnd, limit);
      tries += moveRows(cut, end, false).second;
      cut = end;
    }
    for (; cut < limit; ++cut) {
      const Entry previous = entries[cut - 1];
      const float value = entries[cut].value;
      const std::uint32_t cls = codec.cls(previous.packed);
      ++left[cls];
      if (!isCut(previous.value, value, gap)) {
        // (Nearly) equal values: no threshold fits between them. Move the
        // rest of the run in a tight loop. The loop goes on with the next cut
        // with a threshold (or the end).
        cut = moveRows(cut + 1, limit, true).first - 1;
        continue;
      }
      ++tries;
      const bool leftDistinct = cut < 2 || isCut(entries[cut - 2].value, previous.value, gap);
      const bool rightDistinct = cut + 1 >= count || isCut(value, entries[cut + 1].value, gap);
      const bool skippable = (cls == codec.cls(entries[cut].packed)) & leftDistinct &
                             rightDistinct & (cut > minChild) & (cut + minChild < count);
      if (skippable) {
        continue;
      }
      const double gain =
          cutGain(total, left, classCount, count, cut, parentWeighted, Crit, logs);
      if (gain > bestGain + kTieEps) {
        bestGain = gain;
        bestCut = cut;
        std::copy_n(left, classes, bestLeft);
        bestValues = {previous.value, value};
        skipThreshold = skipBelow(bestGain, margin, rowsPerGain);
      }
      if (!(gain < skipThreshold)) {
        continue;
      }
      // The next cuts cannot beat the best cut while their gain provably
      // stays below it: only move their rows and count their thresholds.
      skipEnd = cut + 1 + skippableSteps(bestGain, gain, margin, rowsPerGain, lastCut - cut);
      const std::uint32_t end = std::min(skipEnd, limit);
      tries += moveRows(cut + 1, end, false).second;
      cut = end - 1;
    }

    if constexpr (FixedK > 0) {
      std::copy_n(fixedLeft, FixedK, fixed_);
    }
    cut_ = cut;
    tries_ = tries;
    bestGain_ = bestGain;
    bestCut_ = bestCut;
    bestValues_ = bestValues;
    skipThreshold_ = skipThreshold;
    skipEnd_ = skipEnd;
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
      const std::uint32_t *counts = FixedK > 0 ? fixed_ + kFixedClasses : dynamic_.data() + classCount_;
      std::copy_n(counts, classCount_, bestLeft);
    }
    return best;
  }

private:
  static constexpr std::size_t kFixedClasses = FixedK > 0 ? FixedK : 1;

  Setup setup_;
  std::uint32_t count_;
  const std::uint32_t *total_;
  double parentWeighted_;
  std::uint32_t minChild_;
  int classCount_;
  double margin_;
  double rowsPerGain_;
  bool empty_;
  std::uint32_t lastCut_;

  bool started_ = false;
  std::uint32_t cut_; // next cut to take
  std::uint32_t tries_ = 0;
  double bestGain_ = -INFINITY;
  std::uint32_t bestCut_ = 0;
  std::pair<float, float> bestValues_{0.0f, 0.0f};
  double skipThreshold_ = -INFINITY; // gains below it start a skip
  std::uint32_t skipEnd_ = 0;        // cuts below it are skipped
  // Class counts left of cut_ - 1, then the best cut's: in fixed_ for FixedK
  // classes, else in dynamic_.
  std::uint32_t fixed_[2 * kFixedClasses] = {};
  std::vector<std::uint32_t> dynamic_;
};

} // namespace dt
