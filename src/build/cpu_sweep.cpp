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
// first i rows left. Rows are moved left one by one, which only updates the
// class counts (most rows of discrete features sit inside runs of equal
// values, which a tight loop moves). Every cut with a threshold that the
// boundary-point shortcut leaves gets cutGain(), and a cut is kept when its
// gain beats the best so far by more than kTieEps. After a cut far below the
// best, the next cuts that provably cannot reach it (maxGainStep) only move
// their rows.
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
  // left: class counts of the rows moved left so far.
  constexpr std::size_t kFixedClasses = FixedK > 0 ? FixedK : 1;
  std::uint32_t fixedLeft[kFixedClasses] = {};
  std::uint32_t *left = fixedLeft;
  if constexpr (FixedK == 0) {
    thread_local std::vector<std::uint32_t> dynamicCounts;
    dynamicCounts.assign(classes, 0);
    left = dynamicCounts.data();
  }
  for (std::uint32_t index = 0; index + 1 < minChild; ++index) {
    ++left[codec_.cls(entries[index].packed)];
  }

  const double gap = rules_.minValueGap();
  const LogTable logs = rules_.logTable();
  const double rows = static_cast<double>(count);
  // Bound on the rounding of cutGain(): a generous count of roundings times
  // the size of the terms (Gini: at most n; entropy: at most n * log2(n)),
  // divided by n.
  double margin;
  if constexpr (Crit == Criterion::Gini) {
    margin = 256.0 * kRoundoff;
  } else {
    const double scale = std::max(1.0, logs.xlog2x(count)) / rows;
    margin = 2.0 * ((4.0 * classCount + 16.0) * scale + 64.0) * kRoundoff;
  }

  std::uint32_t tries = 0;
  double bestGain = -INFINITY;
  std::uint32_t bestCut = 0;
  double skipThreshold = -INFINITY; // gains below it start a skip
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

  const std::uint32_t lastCut = count - minChild;
  for (std::uint32_t cut = minChild; cut <= lastCut; ++cut) {
    const Entry previous = entries[cut - 1];
    const float value = entries[cut].value;
    const std::uint32_t cls = codec.cls(previous.packed);
    ++left[cls];
    if (!isCut(previous.value, value, gap)) {
      // (Nearly) equal values: no threshold fits between them. Move the rest
      // of the run in a tight loop.
      // The loop goes on with the next cut with a threshold (or the end).
      cut = moveRows(cut + 1, lastCut + 1, true).first - 1;
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
    const double gain = cutGain(total, left, classCount, count, cut, parentWeighted, Crit, logs);
    if (gain > bestGain + kTieEps) {
      bestGain = gain;
      bestCut = cut;
      std::copy_n(left, classes, bestLeft);
      skipThreshold = skipBelow(bestGain, margin, rowsPerGain);
    }
    if (!(gain < skipThreshold)) {
      continue;
    }
    // The next cuts cannot beat the best cut while their gain provably stays
    // below it: only move their rows and count their thresholds.
    const std::uint32_t steps =
        skippableSteps(bestGain, gain, margin, rowsPerGain, lastCut - cut);
    tries += moveRows(cut + 1, cut + 1 + steps, false).second;
    cut += steps;
  }
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
