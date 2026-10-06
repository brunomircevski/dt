#include "algo/cart_pruning.h"

#include <algorithm>
#include <cmath>

// All passes rely on the preorder layout of Tree: a loop over decreasing ids
// visits children before their parents, increasing ids parents first.

namespace dt {

namespace {

double trainingErrors(const Tree &tree, std::uint32_t id) {
  const Node &node = tree.nodes[id];
  return static_cast<double>(node.count - tree.counts(id)[node.label]);
}

// Minimal cost of a subtree as a function of the per-leaf penalty p:
//   f(p) = min over pruned subtrees T' of errors(T') + p * leaves(T').
// It is concave and piecewise linear; segment i holds on [start_i, start_i+1).
struct Segment {
  double start;
  double slope;     // leaves of the optimal subtree on this segment
  double intercept; // its errors
};

std::vector<Segment> addEnvelopes(const std::vector<Segment> &a, const std::vector<Segment> &b) {
  std::vector<Segment> sum;
  sum.reserve(a.size() + b.size());
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a.size() && j < b.size()) {
    const double start = std::max(a[i].start, b[j].start);
    sum.push_back({start, a[i].slope + b[j].slope, a[i].intercept + b[j].intercept});
    const double endA = i + 1 < a.size() ? a[i + 1].start : INFINITY;
    const double endB = j + 1 < b.size() ? b[j + 1].start : INFINITY;
    if (endA <= endB) {
      ++i;
    }
    if (endB <= endA) {
      ++j;
    }
  }
  return sum;
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

PruningSequence cartPruningSequence(const Tree &tree) {
  PruningSequence result;
  const std::size_t nodeCount = tree.nodes.size();
  if (nodeCount == 0) {
    return result;
  }
  // Collapse points in errors per leaf, bottom-up. Children's envelopes are
  // released as soon as the parent has merged them.
  std::vector<double> &collapse = result.collapseAlpha;
  collapse.assign(nodeCount, INFINITY);
  std::vector<std::vector<Segment>> envelope(nodeCount);
  for (std::size_t id = nodeCount; id-- > 0;) {
    const Node &node = tree.nodes[id];
    const double leafErrors = trainingErrors(tree, static_cast<std::uint32_t>(id));
    if (node.isLeaf()) {
      envelope[id] = {{0.0, 1.0, leafErrors}};
      continue;
    }
    const std::vector<Segment> children = addEnvelopes(envelope[node.left], envelope[node.right]);
    std::vector<Segment>().swap(envelope[node.left]);
    std::vector<Segment>().swap(envelope[node.right]);

    // Children cost grows faster (slope >= 2) than the leaf cost (slope 1) and
    // starts no higher, so they cross exactly once: that is the collapse point.
    std::vector<Segment> &own = envelope[id];
    double collapseAt = 0.0;
    for (std::size_t index = 0; index < children.size(); ++index) {
      const Segment &segment = children[index];
      const double end = index + 1 < children.size() ? children[index + 1].start : INFINITY;
      const double crossing = (leafErrors - segment.intercept) / (segment.slope - 1.0);
      if (crossing < end) {
        collapseAt = std::max(segment.start, crossing);
        if (collapseAt > segment.start) {
          own.push_back(segment);
        }
        break;
      }
      own.push_back(segment);
    }
    own.push_back({collapseAt, 1.0, leafErrors});
    collapse[id] = collapseAt;
  }

  // Penalty per leaf in errors -> alpha as a misclassification rate.
  const double scale = 1.0 / static_cast<double>(tree.nodes[0].count);
  for (double &value : collapse) {
    value *= scale;
  }

  // A node starts a new subtree of the sequence if it collapses before all of
  // its ancestors (top-down, so parents first).
  std::vector<double> alphas{0.0};
  std::vector<double> ancestorMin(nodeCount, INFINITY);
  for (std::size_t id = 0; id < nodeCount; ++id) {
    const Node &node = tree.nodes[id];
    if (node.isLeaf()) {
      continue;
    }
    if (collapse[id] < ancestorMin[id]) {
      alphas.push_back(collapse[id]);
    }
    ancestorMin[node.left] = ancestorMin[node.right] = std::min(ancestorMin[id], collapse[id]);
  }
  std::sort(alphas.begin(), alphas.end());
  for (double alpha : alphas) {
    if (result.alphas.empty() || alpha > result.alphas.back() * (1 + 1e-12) + 1e-15) {
      result.alphas.push_back(alpha);
    }
  }
  return result;
}

} // namespace dt
