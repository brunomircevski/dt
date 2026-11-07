#include "algo/cart_pruning.h"

#include <algorithm>
#include <vector>

// All passes rely on the preorder layout of Tree: a loop over decreasing ids
// visits children before their parents, increasing ids parents first.

namespace dt {

namespace {

double trainingErrors(const Tree &tree, std::uint32_t id) {
  const Node &node = tree.nodes[id];
  return static_cast<double>(node.count - tree.counts(id)[node.label]);
}

} // namespace

void cartCostComplexityPrune(Tree &tree, double alpha) {
  if (tree.nodes.empty()) {
    return;
  }
  const double penalty = alpha * static_cast<double>(tree.nodes[0].count);
  // Minimal cost (errors + penalty * leaves) of every subtree.
  std::vector<double> cost(tree.nodes.size());
  for (std::size_t id = tree.nodes.size(); id-- > 0;) {
    Node &node = tree.nodes[id];
    const double leafCost = trainingErrors(tree, static_cast<std::uint32_t>(id)) + penalty;
    if (node.isLeaf()) {
      cost[id] = leafCost;
      continue;
    }
    const double subtreeCost = cost[node.left] + cost[node.right];
    // "<=": T(alpha) is the *smallest* minimising subtree.
    if (leafCost <= subtreeCost + 1e-9 * std::max(1.0, subtreeCost)) {
      node.makeLeaf();
      cost[id] = leafCost;
    } else {
      cost[id] = subtreeCost;
    }
  }
  tree.compact();
}

} // namespace dt
