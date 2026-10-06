#pragma once

#include "dataset.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <istream>
#include <ostream>
#include <vector>

// One tree node. A leaf has feature == -1.
//
// Decision nodes ask "value[feature] <= threshold ?": yes -> left, no -> right.
// Every node keeps the class histogram of the training rows that reached it;
// pruning and printing use it, so no pass over the data is needed later.
struct Node {
  std::int32_t feature = -1;
  double threshold = 0.0;
  std::uint16_t label = 0;                 // majority class (ties: lowest id)
  std::uint32_t count = 0;                 // training rows reaching this node
  std::vector<std::uint32_t> classCounts;  // per class
  double errors = 0.0;     // scratch for pruning (C4.5: estimated errors of subtree)
  std::uint32_t index = 0; // scratch for algorithms that number nodes
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;

  bool isLeaf() const { return feature < 0; }

  // Turn this node into a leaf (drops the subtree).
  void makeLeaf() {
    feature = -1;
    threshold = 0.0;
    left.reset();
    right.reset();
  }

  // Set count/classCounts/label from a class histogram.
  void setCounts(const std::uint32_t *counts, std::size_t classCount);
  void setCounts(std::vector<std::uint32_t> counts);
};

class DecisionTree {
public:
  std::unique_ptr<Node> root;
  std::vector<std::string> featureNames;
  std::vector<std::string> classNames;

  // `features` points at one row's feature values (featureNames order).
  std::uint16_t predict(const float *features) const;
  std::uint16_t predict(const Dataset &dataset, std::size_t row) const;

  // Fraction of rows classified correctly (runs on `threads` threads).
  double accuracy(const Dataset &dataset, unsigned threads) const;

  std::size_t nodeCount() const;
  std::size_t leafCount() const;
  int depth() const;

  // Indented text form, readable by python/render_tree_svg.py.
  void print(std::ostream &output) const;
};

// Read a tree written by DecisionTree::print (feature and class names must
// exist in `dataset`), then recount every node's class histogram on `dataset`.
std::unique_ptr<Node> readTree(std::istream &input, const Dataset &dataset);
void recountTree(Node *root, const Dataset &dataset);

std::size_t countNodes(const Node *node);
std::size_t countLeaves(const Node *node);
int subtreeDepth(const Node *node);
