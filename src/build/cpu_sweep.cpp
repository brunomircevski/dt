#include "build/cpu_sweep.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dt {

SweepSetup CpuTreeBuilder::sweepSetup() const {
  return {codec_,           tables_.xlog,      tables_.inverse, rules_.minValueGap(),
          rules_.logTable(), pool_ != nullptr};
}

CutCandidate CpuTreeBuilder::scanFeature(const Entry *entries, std::uint32_t count,
                                         const std::uint32_t *total, double parentWeighted,
                                         std::uint32_t minChild, std::uint32_t *bestLeft) const {
  if (rules_.criterion() == Criterion::Gini) {
    return scanFeatureFor<Criterion::Gini>(entries, count, total, parentWeighted, minChild,
                                           bestLeft);
  }
  return scanFeatureFor<Criterion::Entropy>(entries, count, total, parentWeighted, minChild,
                                            bestLeft);
}

template <Criterion Crit>
CutCandidate CpuTreeBuilder::scanFeatureFor(const Entry *entries, std::uint32_t count,
                                            const std::uint32_t *total, double parentWeighted,
                                            std::uint32_t minChild,
                                            std::uint32_t *bestLeft) const {
  switch (classCount_) {
  case 2:
    if (count < tables_.size) {
      return rules_.minValueGap() > 0.0
                 ? scanTwoClasses<Crit, true>(entries, count, total, parentWeighted, minChild,
                                              bestLeft)
                 : scanTwoClasses<Crit, false>(entries, count, total, parentWeighted, minChild,
                                               bestLeft);
    }
    return scanFeatureK<2, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 3: return scanFeatureK<3, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 4: return scanFeatureK<4, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 5: return scanFeatureK<5, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 6: return scanFeatureK<6, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 7: return scanFeatureK<7, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 8: return scanFeatureK<8, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  default: return scanFeatureK<0, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  }
}

template <Criterion Crit, bool HasGap>
CutCandidate CpuTreeBuilder::scanTwoClasses(const Entry *entries, std::uint32_t count,
                                            const std::uint32_t *total, double parentWeighted,
                                            std::uint32_t minChild,
                                            std::uint32_t *bestLeft) const {
  TwoClassSweep<Crit, HasGap> sweep(sweepSetup(), count, total, parentWeighted, minChild);
  sweep.advance(entries, count);
  return sweep.finish(bestLeft);
}

// One left-to-right sweep. Cut i sits between entries i-1 and i and sends the
// first i rows left. The result is exactly that of evaluating cutGain() at
// every allowed cut in order and keeping a cut when its gain beats the best so
// far by more than kTieEps, but cutGain() itself runs only for few cuts:
//
//  1. Rows are moved left one by one, which only updates the class counts
//     (most rows of discrete features sit inside runs of equal values, which
//     a tight loop moves). At each cut with a threshold a cheap estimate of
//     the gain is computed, one that needs no loop over the classes when a
//     single row moved since the previous one, nor a division: Gini from the
//     sums of squared class counts, which change by an integer, and tabled
//     reciprocals; entropy from running sums of c*log2(c), recomputed after
//     runs and every kSweepBlock updates, which keeps the rounding errors
//     small. The estimate is provably within `margin` of cutGain().
//  2. A cut whose estimate cannot beat the best cut by kTieEps cannot change
//     the result. The others (unless the boundary-point shortcut skips them)
//     are buffered with their class counts and settled in sweep order when
//     the buffer is full: every candidate before the last estimate that beats
//     all earlier buffered ones by more than 2 * margin + kTieEps is skipped,
//     because that cut's exact gain beats all of them by more than kTieEps,
//     so whatever they did to the best cut, it ends up the same once that cut
//     is reached (this keeps rising stretches of the gain cheap). The others
//     get cutGain().
template <int FixedK, Criterion Crit>
CutCandidate CpuTreeBuilder::scanFeatureK(const Entry *entries, std::uint32_t count,
                                          const std::uint32_t *total, double parentWeighted,
                                          std::uint32_t minChild, std::uint32_t *bestLeft) const {
  CutCandidate best;
  if (count < 2 * minChild) {
    return best;
  }
  const int classCount = FixedK > 0 ? FixedK : classCount_;
  const std::size_t classes = static_cast<std::size_t>(classCount);
  // left: class counts of the rows moved left so far; scratch: for
  // settleCandidates(); stored: the counts at each buffered candidate.
  constexpr std::size_t kFixedClasses = FixedK > 0 ? FixedK : 1;
  std::uint32_t fixedLeft[2 * kFixedClasses] = {};
  std::uint32_t fixedStored[kSweepBlock * kFixedClasses];
  std::uint32_t *left = fixedLeft;
  std::uint32_t *stored = fixedStored;
  if constexpr (FixedK == 0) {
    thread_local std::vector<std::uint32_t> dynamicCounts;
    dynamicCounts.assign((kSweepBlock + 2) * classes, 0);
    left = dynamicCounts.data();
    stored = left + 2 * classes;
  }
  std::uint32_t *scratch = left + classes;
  for (std::uint32_t index = 0; index + 1 < minChild; ++index) {
    ++left[codec_.cls(entries[index].packed)];
  }

  const double gap = rules_.minValueGap();
  const LogTable logs = rules_.logTable();
  const CountTables &tables = tables_;
  const double rows = static_cast<double>(count);
  const double inverseRows = 1.0 / rows;
  // Bound on |estimate - cutGain()|: a generous count of roundings times the
  // size of the terms (Gini: at most n; entropy: at most n * log2(n), and up
  // to kSweepBlock updates add 4 roundings each to the running sums), divided
  // by n.
  double margin;
  if constexpr (Crit == Criterion::Gini) {
    margin = 256.0 * kRoundoff;
  } else {
    const double scale = std::max(1.0, tables.xlog2x(count)) * inverseRows;
    margin =
        2.0 * ((4.0 * kSweepBlock + 4.0 * classCount + 16.0) * scale + 64.0) * kRoundoff;
  }
  const double giniBase = parentWeighted - rows;

  // Gini: exact sums of squares of the left and right class counts (signed:
  // converts to double in one instruction; n^2 < 2^62 as n < 2^31). Entropy:
  // sums of c*log2(c) over them.
  std::int64_t leftSquares = 0;
  std::int64_t rightSquares = 0;
  double leftLogs = 0.0;
  double rightLogs = 0.0;
  std::uint32_t updates = 0; // entropy: incremental updates since the last recompute
  auto recompute = [&]() {
    if constexpr (Crit == Criterion::Gini) {
      leftSquares = 0;
      rightSquares = 0;
      for (std::size_t k = 0; k < classes; ++k) {
        const std::int64_t l = left[k];
        const std::int64_t r = total[k] - left[k];
        leftSquares += l * l;
        rightSquares += r * r;
      }
    } else {
      leftLogs = 0.0;
      rightLogs = 0.0;
      for (std::size_t k = 0; k < classes; ++k) {
        leftLogs += tables.xlog2x(left[k]);
        rightLogs += tables.xlog2x(total[k] - left[k]);
      }
      updates = 0;
    }
  };
  recompute();

  std::uint32_t tries = 0;
  double bestGain = -INFINITY;
  std::uint32_t bestCut = 0;
  double threshold = -INFINITY;     // estimates at or below it cannot beat the best cut
  double skipThreshold = -INFINITY; // estimates below it start a skip
  double estimate[kSweepBlock];
  std::uint32_t position[kSweepBlock];
  std::uint32_t found = 0;
  std::uint32_t settleAt = kFirstSettle;
  auto settle = [&]() {
    settleCandidates(
        estimate, position, found, margin, total, classCount, count, parentWeighted, Crit, logs,
        [&](std::uint32_t index, std::uint32_t *out) {
          std::copy_n(stored + index * classes, classes, out);
        },
        scratch, bestGain, bestCut, bestLeft);
    found = 0;
    settleAt = std::min(2 * settleAt, kSweepBlock);
    threshold = bestGain + kTieEps - margin;
    skipThreshold = skipBelow(bestGain, margin, rows / maxGainStep<Crit>(count));
  };
  const double rowsPerGain = rows / maxGainStep<Crit>(count);
  const EntryCodec codec = codec_;
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
      // doubles while that holds and halves when it does not.
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
      for (; cut < safe; ++cut) {
        ++laneCounts[(cut & (kLanes - 1)) * classes + codec.cls(entries[cut - 1].packed)];
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

  const std::uint32_t lastCut = count - minChild;
  std::uint32_t moved = 0; // rows moved since the sums were updated
  for (std::uint32_t cut = minChild; cut <= lastCut; ++cut) {
    const Entry previous = entries[cut - 1];
    const float value = entries[cut].value;
    const std::uint32_t cls = codec.cls(previous.packed);
    ++left[cls];
    ++moved;
    if (!isCut(previous.value, value, gap)) {
      // (Nearly) equal values: no threshold fits between them. Move the rest
      // of the run in a tight loop.
      const std::uint32_t next = moveRows(cut + 1, lastCut + 1, true).first;
      moved += next - cut - 1;
      cut = next - 1; // the loop goes on with `next`, a cut with a threshold (or the end)
      continue;
    }
    ++tries;
    const bool leftDistinct = cut < 2 || isCut(entries[cut - 2].value, previous.value, gap);
    const bool rightDistinct = cut + 1 >= count || isCut(value, entries[cut + 1].value, gap);
    const bool skippable = (cls == codec.cls(entries[cut].packed)) & leftDistinct &
                           rightDistinct & (cut > minChild) & (cut + minChild < count);
    if (moved > 1 || (Crit == Criterion::Entropy && updates == kSweepBlock)) {
      recompute();
    } else {
      const std::uint32_t l = left[cls] - 1; // before this row moved
      const std::uint32_t r = total[cls] - l;
      if constexpr (Crit == Criterion::Gini) {
        leftSquares += 2 * std::int64_t{l} + 1;
        rightSquares -= 2 * std::int64_t{r} - 1;
      } else {
        leftLogs += tables.xlog2x(l + 1) - tables.xlog2x(l);
        rightLogs += tables.xlog2x(r - 1) - tables.xlog2x(r);
        ++updates;
      }
    }
    moved = 0;
    double gain;
    if constexpr (Crit == Criterion::Gini) {
      gain = (giniBase + static_cast<double>(leftSquares) * tables.reciprocal(cut) +
              static_cast<double>(rightSquares) * tables.reciprocal(count - cut)) *
             inverseRows;
    } else {
      gain = (parentWeighted - (tables.xlog2x(cut) - leftLogs) -
              (tables.xlog2x(count - cut) - rightLogs)) *
             inverseRows;
    }
    // Buffered unless skipped, without a branch on `skippable`.
    estimate[found] = gain;
    position[found] = cut;
    std::copy_n(left, classes, stored + found * classes);
    found += !skippable & (gain > threshold);
    if (found == settleAt) {
      settle();
    }
    if (!(gain < skipThreshold)) {
      continue;
    }
    // The next cuts cannot beat the best cut while their gain provably stays
    // below it: only move their rows and count their thresholds.
    const std::uint32_t steps =
        skippableSteps(bestGain, gain, margin, rowsPerGain, lastCut - cut);
    tries += moveRows(cut + 1, cut + 1 + steps, false).second;
    moved += steps;
    cut += steps;
  }
  settle();
  best.tries = tries;
  if (bestCut > 0) {
    best.gain = bestGain;
    best.leftCount = bestCut;
    best.leftValue = entries[bestCut - 1].value;
    best.rightValue = entries[bestCut].value;
  }
  return best;
}

} // namespace dt
