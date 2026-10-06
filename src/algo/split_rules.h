#pragma once

#include "algo/split_math.h"
#include "core/options.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dt {

// The values behind LogTable: c * log2(c) for c < kLogTableSize.
std::vector<double> xlog2xTable();

// Algorithm-specific rules around the shared split math: when a node must be a
// leaf, how many rows each child needs, and which feature's cut to use.
// Every backend calls these same functions, so they all grow the same tree.
class SplitRules {
public:
  // `totalRows`: rows of the training set the tree is grown on.
  SplitRules(const Options &options, std::size_t classCount, std::size_t totalRows);

  Algorithm algorithm() const { return algorithm_; }
  Criterion criterion() const { return criterion_; }
  int classCount() const { return classCount_; }

  // c * log2(c) for small counts (see LogTable).
  LogTable logTable() const { return {xlog2xTable_.data()}; }

  // Two sorted values only admit a threshold between them if they are more
  // than this apart: 0 for CART, 1e-5 for C4.5 (contin.c). A node whose
  // values of a feature span no more than this has no cut on that feature,
  // and neither has any node below it.
  double minValueGap() const { return algorithm_ == Algorithm::C45 ? 1e-5 : 0.0; }

  // Minimum number of rows on each side of a cut, for a node with n rows.
  //   CART: minLeaf.
  //   C4.5: MinSplit = 10% of the average class size, clamped to [m, 25].
  std::uint32_t minChildRows(std::uint32_t n) const;

  // True if the node must become a leaf without looking for a split.
  bool isTerminal(const std::uint32_t *classCounts, std::uint32_t n, int depth) const;

  struct Decision {
    int feature = -1;           // -1: make a leaf
    std::uint32_t leftCount = 0;
    double threshold = 0.0;     // midpoint between the two values around the cut
  };

  // Choose the split of a node from the best cut of every feature (invalid
  // candidates for features without a cut).
  //   CART: highest impurity decrease (lowest feature index on ties).
  //   C4.5: subtract the MDL threshold cost log2(tries)/n from each gain, keep
  //         features whose gain is at least the average, take the highest gain
  //         ratio.
  Decision choose(const CutCandidate *bestCutPerFeature, std::size_t featureCount,
                  std::uint32_t n) const;

private:
  Decision chooseCart(const CutCandidate *cuts, std::size_t featureCount,
                      std::uint32_t n) const;
  Decision chooseC45(const CutCandidate *cuts, std::size_t featureCount,
                     std::uint32_t n) const;

  Algorithm algorithm_;
  Criterion criterion_;
  int classCount_;
  double totalRows_;
  int maxDepth_;
  std::uint32_t minSplit_;
  std::uint32_t minLeaf_;
  double minDecrease_;
  std::uint32_t c45MinObjects_;
  std::vector<double> xlog2xTable_;
};

} // namespace dt
