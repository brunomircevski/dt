// The Simple backend: CART and C4.5 written the way the textbooks describe
// them, with no performance tricks. It is the reference the optimized
// backends are compared against: same SplitRules and split math, so it must
// grow the same tree, only slower.
//
//   1. Sort the rows by every feature once.
//   2. Grow a node: unless it must be a leaf, try every threshold of every
//      feature (between two neighbouring sorted values), keep the best cut of
//      each feature, and let the CART or C4.5 rule choose the feature.
//   3. Send every row with value <= threshold left, the others right, keeping
//      each feature's order, and grow both children the same way.

#include "build/grower.h"
#include "core/timing.h"

#include <algorithm>
#include <bit>
#include <numeric>
#include <stdexcept>

namespace dt {

namespace {

// A node's rows, once per feature, sorted by that feature's value.
using SortedRows = std::vector<std::vector<std::uint32_t>>;

class SimpleGrower final : public Grower {
public:
  explicit SimpleGrower(const Dataset &train) : train_(train) {}

  Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows, GrowTimings &timings,
            DistinctValues *distinct, RowLeaves *leafOfRow) override {
    if (leafOfRow && !rows.empty()) {
      throw std::logic_error("SimpleGrower: leaves of rows only with all rows");
    }
    std::vector<std::uint32_t> all;
    if (rows.empty()) {
      all.resize(train_.rowCount);
      std::iota(all.begin(), all.end(), 0u);
      rows = all;
    }

    SortedRows sorted(train_.featureCount());
    {
      ScopedTimer timer(timings.prepareSeconds);
      for (std::size_t feature = 0; feature < sorted.size(); ++feature) {
        sorted[feature].assign(rows.begin(), rows.end());
        std::stable_sort(sorted[feature].begin(), sorted[feature].end(),
                         [&](std::uint32_t a, std::uint32_t b) {
                           return value(a, feature) < value(b, feature);
                         });
      }
      if (distinct) {
        distinctValues(sorted, *distinct);
      }
    }

    ScopedTimer timer(timings.buildSeconds);
    Tree tree;
    tree.featureNames = train_.featureNames;
    tree.classNames = train_.classNames;
    if (leafOfRow) {
      leafOfRow->ids.resize(train_.rowCount);
      leafOfRow->map.clear();
    }
    rules_ = &rules;
    tree_ = &tree;
    leafOfRow_ = leafOfRow;
    growNode(std::move(sorted), 0);
    return tree;
  }

  // No memory to lend: C4.5's pruning allocates its own.
  std::span<std::byte> workspace() override { return {}; }

private:
  // -0.0 and 0.0 are the same value (no threshold between them).
  float value(std::uint32_t row, std::size_t feature) const {
    return train_.value(row, feature) + 0.0f;
  }

  // Every feature's distinct values, in increasing order.
  void distinctValues(const SortedRows &sorted, DistinctValues &distinct) const {
    distinct.values.clear();
    distinct.offsets.assign(1, 0);
    for (std::size_t feature = 0; feature < sorted.size(); ++feature) {
      for (std::size_t index = 0; index < sorted[feature].size(); ++index) {
        const float current = value(sorted[feature][index], feature);
        if (index == 0 || std::bit_cast<std::uint32_t>(current) !=
                              std::bit_cast<std::uint32_t>(distinct.values.back())) {
          distinct.values.push_back(current);
        }
      }
      distinct.offsets.push_back(distinct.values.size());
    }
  }

  // Adds the node with the rows `sorted` (and, in preorder, its subtree) to
  // the tree. Returns its id.
  std::uint32_t growNode(SortedRows sorted, int depth) {
    const std::vector<std::uint32_t> &rows = sorted.front();
    const std::uint32_t count = static_cast<std::uint32_t>(rows.size());
    const int classCount = rules_->classCount();

    std::vector<std::uint32_t> counts(classCount, 0);
    for (std::uint32_t row : rows) {
      ++counts[train_.labels[row]];
    }
    const std::uint32_t id = static_cast<std::uint32_t>(tree_->nodes.size());
    Node node;
    node.count = count;
    node.label = majorityClass(counts.data(), counts.size());
    tree_->nodes.push_back(node);
    tree_->classCounts.insert(tree_->classCounts.end(), counts.begin(), counts.end());

    auto makeLeaf = [&]() {
      if (leafOfRow_) {
        for (std::uint32_t row : rows) {
          leafOfRow_->ids[row] = id;
        }
      }
      return id;
    };
    if (rules_->isTerminal(counts.data(), count, depth)) {
      return makeLeaf();
    }

    // The best cut of every feature, then the CART or C4.5 choice among them.
    std::vector<CutCandidate> cuts(sorted.size());
    for (std::size_t feature = 0; feature < sorted.size(); ++feature) {
      cuts[feature] = bestCut(sorted[feature], feature, counts);
    }
    const SplitRules::Decision decision = rules_->choose(cuts.data(), cuts.size(), count);
    if (decision.feature < 0) {
      return makeLeaf();
    }

    // Split every feature's sorted rows; each side keeps the order.
    std::vector<bool> goesLeft(train_.rowCount, false);
    for (std::uint32_t row : rows) {
      goesLeft[row] = value(row, decision.feature) <= decision.threshold;
    }
    SortedRows left(sorted.size());
    SortedRows right(sorted.size());
    for (std::size_t feature = 0; feature < sorted.size(); ++feature) {
      for (std::uint32_t row : sorted[feature]) {
        (goesLeft[row] ? left : right)[feature].push_back(row);
      }
      sorted[feature] = {}; // the parent's rows are no longer needed
    }

    // Children in preorder: the left subtree right after this node.
    const std::uint32_t leftId = growNode(std::move(left), depth + 1);
    const std::uint32_t rightId = growNode(std::move(right), depth + 1);
    Node &parent = tree_->nodes[id];
    parent.feature = decision.feature;
    parent.threshold = decision.threshold;
    parent.left = leftId;
    parent.right = rightId;
    return id;
  }

  // The best threshold of one feature. Cut i sends the first i sorted rows
  // left; it exists where the values on either side differ (by more than
  // C4.5's minimum gap) and both sides keep at least minChildRows rows.
  // Every cut's gain is computed; a later cut must beat the best by kTieEps.
  CutCandidate bestCut(const std::vector<std::uint32_t> &rows, std::size_t feature,
                       const std::vector<std::uint32_t> &total) const {
    const std::uint32_t count = static_cast<std::uint32_t>(rows.size());
    const int classCount = rules_->classCount();
    const std::uint32_t minChild = rules_->minChildRows(count);
    const double parentWeighted = weightedImpurity(total.data(), classCount, count,
                                                   rules_->criterion(), rules_->logTable());
    CutCandidate best;
    std::vector<std::uint32_t> left(classCount, 0); // class counts of rows [0, cut)
    for (std::uint32_t cut = 1; cut < count; ++cut) {
      ++left[train_.labels[rows[cut - 1]]];
      if (cut < minChild || count - cut < minChild) {
        continue;
      }
      const float lower = value(rows[cut - 1], feature);
      const float upper = value(rows[cut], feature);
      if (!isCut(lower, upper, rules_->minValueGap())) {
        continue;
      }
      ++best.tries; // C4.5's threshold cost counts the possible cuts
      const double gain = cutGain(total.data(), left.data(), classCount, count, cut,
                                  parentWeighted, rules_->criterion(), rules_->logTable());
      if (gain > best.gain + kTieEps) {
        best.gain = gain;
        best.leftCount = cut;
        best.leftValue = lower;
        best.rightValue = upper;
      }
    }
    return best;
  }

  const Dataset &train_;
  // Set for one grow().
  const SplitRules *rules_ = nullptr;
  Tree *tree_ = nullptr;
  RowLeaves *leafOfRow_ = nullptr;
};

} // namespace

std::unique_ptr<Grower> makeSimpleGrower(const Dataset &train) {
  return std::make_unique<SimpleGrower>(train);
}

} // namespace dt
