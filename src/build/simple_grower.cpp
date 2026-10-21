// The Simple backend: CART and C4.5 written the way the textbooks describe
// them, with no performance tricks. It is the reference the optimized
// backends are compared against: same SplitRules and split math, so it must
// grow the same tree, only slower.
//
//   1. Grow a node: unless it must be a leaf, sort its rows by every feature,
//      try every threshold (between two neighbouring sorted values), keep the
//      best cut of each feature, and let the CART or C4.5 rule choose the
//      feature.
//   2. Send every row with value <= threshold left, the others right, and
//      grow both children the same way.

#include "build/grower.h"
#include "core/timing.h"

#include <algorithm>
#include <numeric>

namespace dt {

namespace {

// Grows one tree.
struct Builder {
  const Dataset &train;
  const SplitRules &rules;
  Tree &tree;

  // Adds the node with the rows `rows` and, in preorder, its subtree.
  void growNode(std::vector<std::uint32_t> rows, int depth) {
    const std::uint32_t count = static_cast<std::uint32_t>(rows.size());
    std::vector<std::uint32_t> counts(rules.classCount(), 0);
    for (std::uint32_t row : rows) {
      ++counts[train.labels[row]];
    }
    const std::uint32_t id = static_cast<std::uint32_t>(tree.nodes.size());
    Node node;
    node.count = count;
    node.label = majorityClass(counts.data(), counts.size());
    tree.nodes.push_back(node);
    tree.classCounts.insert(tree.classCounts.end(), counts.begin(), counts.end());
    if (rules.isTerminal(counts.data(), count, depth)) {
      return;
    }

    // The best cut of every feature, then the CART or C4.5 choice among them.
    const double parentWeighted = weightedImpurity(counts.data(), rules.classCount(), count,
                                                   rules.criterion(), rules.logTable());
    std::vector<CutCandidate> cuts(train.featureCount());
    for (std::size_t feature = 0; feature < cuts.size(); ++feature) {
      std::vector<std::uint32_t> sorted = rows;
      std::sort(sorted.begin(), sorted.end(), [&](std::uint32_t a, std::uint32_t b) {
        return train.value(a, feature) < train.value(b, feature);
      });
      cuts[feature] = bestCut(sorted, feature, counts, parentWeighted);
    }
    const SplitRules::Decision decision = rules.choose(cuts.data(), cuts.size(), count);
    if (decision.feature < 0) {
      return;
    }

    std::vector<std::uint32_t> left;
    std::vector<std::uint32_t> right;
    for (std::uint32_t row : rows) {
      (train.value(row, decision.feature) <= decision.threshold ? left : right).push_back(row);
    }
    rows = {}; // the parent's rows are no longer needed

    // Preorder: the left subtree right after this node, then the right one.
    tree.nodes[id].feature = decision.feature;
    tree.nodes[id].threshold = decision.threshold;
    tree.nodes[id].left = id + 1;
    growNode(std::move(left), depth + 1);
    tree.nodes[id].right = static_cast<std::uint32_t>(tree.nodes.size());
    growNode(std::move(right), depth + 1);
  }

  // The best threshold of one feature, `rows` sorted by it. Cut i sends the
  // first i rows left; it exists where the values on either side differ (by
  // more than C4.5's minimum gap) and both sides keep at least minChildRows
  // rows. Every cut's gain is computed; a later cut must beat the best by
  // kTieEps. Rows with equal values may be in any order: no cut falls
  // between them.
  CutCandidate bestCut(const std::vector<std::uint32_t> &rows, std::size_t feature,
                       const std::vector<std::uint32_t> &total, double parentWeighted) const {
    const std::uint32_t count = static_cast<std::uint32_t>(rows.size());
    const std::uint32_t minChild = rules.minChildRows(count);
    CutCandidate best;
    std::vector<std::uint32_t> left(rules.classCount(), 0); // class counts of rows [0, cut)
    for (std::uint32_t cut = 1; cut < count; ++cut) {
      ++left[train.labels[rows[cut - 1]]];
      if (cut < minChild || count - cut < minChild) {
        continue;
      }
      const float lower = train.value(rows[cut - 1], feature);
      const float upper = train.value(rows[cut], feature);
      if (!isCut(lower, upper, rules.minValueGap())) {
        continue;
      }
      ++best.tries; // C4.5's threshold cost counts the possible cuts
      const double gain = cutGain(total.data(), left.data(), rules.classCount(), count, cut,
                                  parentWeighted, rules.criterion(), rules.logTable());
      if (gain > best.gain + kTieEps) {
        best.gain = gain;
        best.leftCount = cut;
        best.leftValue = lower;
        best.rightValue = upper;
      }
    }
    return best;
  }
};

// Fills neither `distinct` nor `leafOfRow`: the trainer then computes the
// distinct values itself and C4.5's pruning routes the rows itself.
class SimpleGrower final : public Grower {
public:
  explicit SimpleGrower(const Dataset &train) : train_(train) {}

  Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows, GrowTimings &timings,
            DistinctValues * /*distinct*/, RowLeaves * /*leafOfRow*/) override {
    ScopedTimer timer(timings.buildSeconds);
    std::vector<std::uint32_t> all(rows.begin(), rows.end());
    if (all.empty()) {
      all.resize(train_.rowCount);
      std::iota(all.begin(), all.end(), 0u);
    }
    Tree tree;
    tree.featureNames = train_.featureNames;
    tree.classNames = train_.classNames;
    Builder{train_, rules, tree}.growNode(std::move(all), 0);
    return tree;
  }

  // No memory to lend: C4.5's pruning allocates its own.
  std::span<std::byte> workspace() override { return {}; }

private:
  const Dataset &train_;
};

} // namespace

std::unique_ptr<Grower> makeSimpleGrower(const Dataset &train) {
  return std::make_unique<SimpleGrower>(train);
}

} // namespace dt
