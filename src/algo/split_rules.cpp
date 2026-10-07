#include "algo/split_rules.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dt {

namespace {

// C4.5's Epsilon (defns.i): tolerance of the average-gain filter and the
// minimum useful split information.
constexpr double kC45Epsilon = 1e-3;

double midpoint(float low, float high) {
  // Exact in double: the midpoint of two floats needs 25 mantissa bits.
  return (static_cast<double>(low) + static_cast<double>(high)) / 2.0;
}

} // namespace

std::vector<double> xlog2xTable() {
  std::vector<double> table(kLogTableSize);
  for (std::uint32_t count = 0; count < kLogTableSize; ++count) {
    table[count] = xlog2x(static_cast<double>(count));
  }
  return table;
}

SplitRules::SplitRules(const Options &options, std::size_t classCount)
    : algorithm_(options.algorithm),
      criterion_(options.algorithm == Algorithm::C45 ? Criterion::Entropy : Criterion::Gini),
      classCount_(static_cast<int>(classCount)),
      maxDepth_(options.maxDepth),
      minLeaf_(static_cast<std::uint32_t>(
          options.minLeaf > 0 ? options.minLeaf
                              : (options.algorithm == Algorithm::C45 ? 2 : 1))),
      xlog2xTable_(xlog2xTable()) {}

std::uint32_t SplitRules::minChildRows(std::uint32_t n) const {
  if (algorithm_ == Algorithm::Cart) {
    return minLeaf_;
  }
  // contin.c: MinSplit = 0.10 * KnownItems / (MaxClass + 1), stored as float,
  // then "if (MinSplit <= MINOBJS) MinSplit = MINOBJS; else if (> 25) 25".
  // Counts are integers, so "count >= MinSplit" is "count >= ceil(MinSplit)".
  // (ComputeGain additionally needs MINOBJS cases per branch, which only
  // matters when -m is above 25.)
  float minSplit = static_cast<float>(0.10 * static_cast<double>(n) / classCount_);
  if (minSplit <= static_cast<float>(minLeaf_)) {
    minSplit = static_cast<float>(minLeaf_);
  } else if (minSplit > 25.0f) {
    minSplit = 25.0f;
  }
  return std::max(static_cast<std::uint32_t>(std::ceil(minSplit)), minLeaf_);
}

bool SplitRules::isTerminal(const std::uint32_t *classCounts, std::uint32_t n,
                            int depth) const {
  const std::uint32_t largest = *std::max_element(classCounts, classCounts + classCount_);
  if (largest == n) {
    return true; // pure
  }
  if (maxDepth_ >= 0 && depth >= maxDepth_) {
    return true;
  }
  // No cut can give both children enough rows (for C4.5 this includes
  // build.c's "Cases < 2 * MINOBJS").
  return n < 2 * minChildRows(n);
}

SplitRules::Decision SplitRules::choose(const CutCandidate *cuts,
                                        std::size_t featureCount, std::uint32_t n) const {
  return algorithm_ == Algorithm::Cart ? chooseCart(cuts, featureCount)
                                       : chooseC45(cuts, featureCount, n);
}

SplitRules::Decision SplitRules::chooseCart(const CutCandidate *cuts,
                                            std::size_t featureCount) const {
  int best = -1;
  for (std::size_t feature = 0; feature < featureCount; ++feature) {
    if (!cuts[feature].valid()) {
      continue;
    }
    if (best < 0 || cuts[feature].gain > cuts[best].gain + kTieEps) {
      best = static_cast<int>(feature);
    }
  }
  Decision decision;
  if (best < 0) {
    return decision;
  }
  // Any split is accepted, even one with zero gain (CART needs it for e.g.
  // XOR-like data); pruning removes the useless ones afterwards.
  decision.feature = best;
  decision.leftCount = cuts[best].leftCount;
  decision.threshold = midpoint(cuts[best].leftValue, cuts[best].rightValue);
  return decision;
}

SplitRules::Decision SplitRules::chooseC45(const CutCandidate *cuts,
                                           std::size_t featureCount,
                                           std::uint32_t n) const {
  // EvalContinuousAtt: Gain[Att] = best gain - Log(Tries) / Items, and the
  // attribute is usable only if that is > 0.
  std::vector<double> gain(featureCount, -1.0);
  double gainSum = 0.0;
  int possible = 0;
  for (std::size_t feature = 0; feature < featureCount; ++feature) {
    const CutCandidate &cut = cuts[feature];
    if (!cut.valid()) {
      continue;
    }
    const double thresholdCost = std::log2(static_cast<double>(cut.tries)) / n;
    const double corrected = cut.gain - thresholdCost;
    if (corrected > 0.0) {
      gain[feature] = corrected;
      gainSum += corrected;
      ++possible;
    }
  }
  Decision decision;
  if (possible == 0) {
    return decision;
  }
  const double averageGain = gainSum / possible;

  // FormTree + Worth(): among attributes whose gain is at least the average,
  // take the highest gain ratio = gain / split information. First wins ties.
  double bestRatio = -kC45Epsilon;
  int best = -1;
  for (std::size_t feature = 0; feature < featureCount; ++feature) {
    if (gain[feature] <= 0.0) {
      continue;
    }
    const std::uint32_t leftCount = cuts[feature].leftCount;
    const std::uint32_t counts[2] = {leftCount, n - leftCount};
    const double splitInfo = weightedImpurity(counts, 2, n, Criterion::Entropy, logTable()) / n;
    if (gain[feature] < averageGain - kC45Epsilon || splitInfo <= kC45Epsilon) {
      continue;
    }
    const double ratio = gain[feature] / splitInfo;
    // Equal ratios (up to rounding, which differs between CPU and GPU log2)
    // go to the first feature, as c4.5 intends with its strict ">".
    if (best < 0 ? ratio > bestRatio : ratio > bestRatio + kTieEps) {
      bestRatio = ratio;
      best = static_cast<int>(feature);
    }
  }
  if (best < 0) {
    return decision;
  }
  decision.feature = best;
  decision.leftCount = cuts[best].leftCount;
  // C4.5 later moves the threshold down to the largest training value not
  // above this midpoint (see c45UseTrainingValueThresholds); same partition.
  decision.threshold = midpoint(cuts[best].leftValue, cuts[best].rightValue);
  return decision;
}

} // namespace dt
