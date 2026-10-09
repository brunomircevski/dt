#pragma once

#include "core/dataset.h"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace dt {

class ThreadPool;

// One tree node. Decision nodes ask "value[feature] <= threshold ?":
// yes -> left, no -> right. A leaf has feature == -1.
struct Node {
  double threshold = 0.0;
  std::int32_t feature = -1;
  std::uint32_t left = 0;  // child node ids (decision nodes only)
  std::uint32_t right = 0;
  std::uint32_t count = 0; // training rows reaching this node
  std::uint16_t label = 0; // majority class

  bool isLeaf() const { return feature < 0; }
  void makeLeaf() { feature = -1; }
};

// The most frequent class. Ties go to `preferred` if it is one of the most
// frequent classes, otherwise to the lowest class id.
std::uint16_t majorityClass(const std::uint32_t *counts, std::size_t classCount,
                            std::uint16_t preferred = 0);

// A tree stored as flat arrays. nodes[0] is the root and nodes are in
// preorder: a node's children always come after it, so a loop over
// decreasing ids visits children before their parents. Every node keeps the
// class histogram of the training rows that reached it; pruning and printing
// use it, so no pass over the data is needed later.
class Tree {
public:
  std::vector<std::string> featureNames;
  std::vector<std::string> classNames;
  std::vector<Node> nodes;
  std::vector<std::uint32_t> classCounts; // node id * classCount() + class

  std::size_t classCount() const { return classNames.size(); }
  const std::uint32_t *counts(std::uint32_t id) const {
    return classCounts.data() + std::size_t{id} * classCount();
  }
  std::uint32_t *counts(std::uint32_t id) { return classCounts.data() + std::size_t{id} * classCount(); }

  // `row` points at one row's feature values (featureNames order).
  std::uint16_t predict(const float *row) const;

  // Fraction of rows classified correctly.
  double accuracy(const Dataset &dataset, ThreadPool *pool) const;

  std::size_t nodeCount() const { return nodes.size(); }
  std::size_t leafCount() const;
  int depth() const;

  // Indented text form (--dump).
  void print(std::ostream &output) const;

  // After pruning: drop the nodes that are no longer reachable from the root
  // and renumber the others in preorder.
  void compact();

  // Replace the nodes by the tree under `root` of another node storage,
  // renumbered in preorder. nodeAt(id) -> const Node&, countsAt(id) -> const
  // uint32_t* (classCount() values).
  template <class NodeAt, class CountsAt>
  void assignPreorder(std::uint32_t root, NodeAt &&nodeAt, CountsAt &&countsAt);
};

template <class NodeAt, class CountsAt>
void Tree::assignPreorder(std::uint32_t root, NodeAt &&nodeAt, CountsAt &&countsAt) {
  nodes.clear();
  classCounts.clear();
  // A child's new id is only known when it is visited; it patches its parent.
  struct Visit {
    std::uint32_t id;
    std::int64_t parent; // new id of the parent, -1 for the root
    bool left;
  };
  std::vector<Visit> stack{{root, -1, false}};
  while (!stack.empty()) {
    const Visit visit = stack.back();
    stack.pop_back();
    const std::uint32_t newId = static_cast<std::uint32_t>(nodes.size());
    const Node &node = nodeAt(visit.id);
    nodes.push_back(node);
    const std::uint32_t *counts = countsAt(visit.id);
    classCounts.insert(classCounts.end(), counts, counts + classCount());
    if (visit.parent >= 0) {
      Node &parent = nodes[static_cast<std::size_t>(visit.parent)];
      (visit.left ? parent.left : parent.right) = newId;
    }
    if (!node.isLeaf()) {
      stack.push_back({node.right, newId, false});
      stack.push_back({node.left, newId, true});
    }
  }
}

} // namespace dt
