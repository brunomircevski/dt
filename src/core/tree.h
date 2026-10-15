#pragma once

#include "core/dataset.h"
#include "core/thread_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace dt {

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
  // and renumber the others in preorder. `pool` may be null. `newIdOf` (if
  // not null, a value per old id) receives the new id of every kept node.
  void compact(ThreadPool *pool = nullptr, std::uint32_t *newIdOf = nullptr);

  // Replace the nodes by the tree under `root` of another node storage,
  // renumbered in preorder. nodeAt(id) -> const Node&, countsAt(id) -> const
  // uint32_t* (classCount() values); with a pool both are called from several
  // threads at once. `pool` may be null. `newIdOf` (if not null, indexed by
  // the other storage's ids) receives the new id of every copied node.
  template <class NodeAt, class CountsAt>
  void assignPreorder(std::uint32_t root, NodeAt &&nodeAt, CountsAt &&countsAt,
                      ThreadPool *pool = nullptr, std::uint32_t *newIdOf = nullptr);

  // Call visit(id) for every node, children before their parent. Subtrees are
  // visited in parallel (nodes of different subtrees at the same time), the
  // nodes above them last, on this thread. Needs the layout that
  // assignPreorder() and compact() leave: a decision node's left child is the
  // next id, and its subtree is a contiguous range of ids. `pool` may be null.
  template <class Visit> void bottomUp(ThreadPool *pool, Visit &&visit) const;

  // Call visit(id) for every id, parents before their children: the nodes
  // above the subtrees of bottomUp() first, then the subtrees in parallel,
  // each by increasing id. A node made a leaf whose children's ids are still
  // in place (makeLeaf) is visited with its old subtree.
  template <class Visit> void topDown(ThreadPool *pool, Visit &&visit) const;

private:
  // Copy the subtree under `root` (of another node storage) to the ids
  // [first, first + its size) in preorder; returns the next free id.
  template <class NodeAt, class CountsAt>
  std::uint32_t copyPreorder(std::uint32_t root, std::uint32_t first, NodeAt &nodeAt,
                             CountsAt &countsAt, std::uint32_t *newIdOf);
};

// ids[i] = newIdOf[ids[i]] for the `count` ids, in parallel (`pool` may be
// null).
void remapIds(std::uint32_t *ids, std::size_t count, const std::uint32_t *newIdOf,
              ThreadPool *pool);

// The parallel tree passes split a tree into subtrees of at most this share of
// the work (rows or nodes) per thread, so threads finish together.
inline constexpr std::size_t kSubtreesPerThread = 8;

template <class NodeAt, class CountsAt>
std::uint32_t Tree::copyPreorder(std::uint32_t root, std::uint32_t first, NodeAt &nodeAt,
                                 CountsAt &countsAt, std::uint32_t *newIdOf) {
  const std::size_t classes = classCount();
  // A child's new id is only known when it is visited; it patches its parent.
  struct Visit {
    std::uint32_t id;
    std::int64_t parent; // new id of the parent, -1 for the subtree's root
    bool left;
  };
  std::vector<Visit> stack{{root, -1, false}};
  std::uint32_t next = first;
  while (!stack.empty()) {
    const Visit visit = stack.back();
    stack.pop_back();
    const std::uint32_t newId = next++;
    const Node &node = nodeAt(visit.id);
    nodes[newId] = node;
    if (newIdOf) {
      newIdOf[visit.id] = newId;
    }
    const std::uint32_t *counts = countsAt(visit.id);
    std::copy(counts, counts + classes, classCounts.data() + std::size_t{newId} * classes);
    if (visit.parent >= 0) {
      Node &parent = nodes[static_cast<std::size_t>(visit.parent)];
      (visit.left ? parent.left : parent.right) = newId;
    }
    if (!node.isLeaf()) {
      stack.push_back({node.right, newId, false});
      stack.push_back({node.left, newId, true});
    }
  }
  return next;
}

template <class NodeAt, class CountsAt>
void Tree::assignPreorder(std::uint32_t root, NodeAt &&nodeAt, CountsAt &&countsAt,
                          ThreadPool *pool, std::uint32_t *newIdOf) {
  // The tree is cut into blocks: subtrees of at most `grain` training rows
  // (a subtree has fewer than twice as many nodes as rows), copied by one
  // task each. The few nodes above them ("top" nodes) are copied by this
  // thread. Items are the top nodes and blocks in preorder.
  const std::size_t threads = pool ? pool->workerCount() + 1 : 1;
  const std::uint32_t grain =
      threads > 1 ? std::max<std::uint32_t>(
                        1, static_cast<std::uint32_t>(nodeAt(root).count /
                                                      (kSubtreesPerThread * threads)))
                  : UINT32_MAX;
  struct Item {
    std::uint32_t id;
    bool block;
    std::uint32_t size = 1;               // nodes (blocks: counted below)
    std::uint32_t children[2] = {0, 0};   // top nodes: items of the children
  };
  std::vector<Item> items;
  struct Pending {
    std::uint32_t id;
    std::int64_t parentItem; // -1 for the root
    int side;
  };
  std::vector<Pending> stack{{root, -1, 0}};
  while (!stack.empty()) {
    const Pending pending = stack.back();
    stack.pop_back();
    const auto item = static_cast<std::uint32_t>(items.size());
    if (pending.parentItem >= 0) {
      items[static_cast<std::size_t>(pending.parentItem)].children[pending.side] = item;
    }
    const Node &node = nodeAt(pending.id);
    const bool top = !node.isLeaf() && node.count > grain;
    items.push_back({pending.id, !top});
    if (top) {
      stack.push_back({node.right, item, 1});
      stack.push_back({node.left, item, 0});
    }
  }
  std::vector<std::uint32_t> blocks;
  for (std::uint32_t item = 0; item < items.size(); ++item) {
    if (items[item].block) {
      blocks.push_back(item);
    }
  }
  parallelFor(pool, blocks.size(), [&](std::size_t index) {
    Item &block = items[blocks[index]];
    std::uint32_t size = 0;
    std::vector<std::uint32_t> pending{block.id};
    while (!pending.empty()) {
      const Node &node = nodeAt(pending.back());
      pending.pop_back();
      ++size;
      if (!node.isLeaf()) {
        pending.push_back(node.right);
        pending.push_back(node.left);
      }
    }
    block.size = size;
  });
  // In preorder the items follow one another: item i starts at first[i].
  std::vector<std::uint32_t> first(items.size());
  std::uint32_t total = 0;
  for (std::size_t item = 0; item < items.size(); ++item) {
    first[item] = total;
    total += items[item].size;
  }
  nodes.resize(total);
  classCounts.resize(std::size_t{total} * classCount());
  for (std::size_t item = 0; item < items.size(); ++item) {
    if (!items[item].block) {
      Node node = nodeAt(items[item].id);
      node.left = first[items[item].children[0]];
      node.right = first[items[item].children[1]];
      nodes[first[item]] = node;
      if (newIdOf) {
        newIdOf[items[item].id] = first[item];
      }
      const std::uint32_t *counts = countsAt(items[item].id);
      std::copy(counts, counts + classCount(),
                classCounts.data() + std::size_t{first[item]} * classCount());
    }
  }
  parallelFor(pool, blocks.size(), [&](std::size_t index) {
    copyPreorder(items[blocks[index]].id, first[blocks[index]], nodeAt, countsAt, newIdOf);
  });
}

// The blocks and top nodes of bottomUp() and topDown(): blocks are subtrees of
// at most `grain` nodes, as ranges of ids; top nodes are the nodes above them,
// in preorder.
inline void splitIntoBlocks(const std::vector<Node> &nodes, std::size_t threads,
                            std::vector<std::pair<std::uint32_t, std::uint32_t>> &blocks,
                            std::vector<std::uint32_t> &top) {
  const std::size_t grain =
      threads > 1 ? std::max<std::size_t>(4096, nodes.size() / (kSubtreesPerThread * threads))
                  : nodes.size();
  std::vector<std::pair<std::uint32_t, std::uint32_t>> stack{
      {0, static_cast<std::uint32_t>(nodes.size())}};
  while (!stack.empty()) {
    const auto [id, end] = stack.back();
    stack.pop_back();
    const Node &node = nodes[id];
    if (node.isLeaf() || end - id <= grain) {
      blocks.emplace_back(id, end);
      continue;
    }
    top.push_back(id);
    stack.emplace_back(node.right, end);
    stack.emplace_back(node.left, node.right); // left == id + 1
  }
}

template <class Visit> void Tree::bottomUp(ThreadPool *pool, Visit &&visit) const {
  if (nodes.empty()) {
    return;
  }
  std::vector<std::pair<std::uint32_t, std::uint32_t>> blocks;
  std::vector<std::uint32_t> top;
  splitIntoBlocks(nodes, pool ? pool->workerCount() + 1 : 1, blocks, top);
  parallelFor(pool, blocks.size(), [&](std::size_t index) {
    for (std::uint32_t id = blocks[index].second; id-- > blocks[index].first;) {
      visit(id);
    }
  });
  for (auto it = top.rbegin(); it != top.rend(); ++it) {
    visit(*it);
  }
}

template <class Visit> void Tree::topDown(ThreadPool *pool, Visit &&visit) const {
  if (nodes.empty()) {
    return;
  }
  std::vector<std::pair<std::uint32_t, std::uint32_t>> blocks;
  std::vector<std::uint32_t> top;
  splitIntoBlocks(nodes, pool ? pool->workerCount() + 1 : 1, blocks, top);
  for (const std::uint32_t id : top) {
    visit(id);
  }
  parallelFor(pool, blocks.size(), [&](std::size_t index) {
    for (std::uint32_t id = blocks[index].first; id < blocks[index].second; ++id) {
      visit(id);
    }
  });
}

} // namespace dt
