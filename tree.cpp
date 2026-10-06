#include "tree.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <string_view>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

void Node::setCounts(const std::uint32_t *counts, std::size_t classCount) {
  classCounts.assign(counts, counts + classCount);
  count = 0;
  label = 0;
  for (std::size_t index = 0; index < classCount; ++index) {
    count += counts[index];
    if (counts[index] > counts[label]) {
      label = static_cast<std::uint16_t>(index);
    }
  }
}

void Node::setCounts(std::vector<std::uint32_t> counts) {
  classCounts = std::move(counts);
  count = 0;
  label = 0;
  for (std::size_t index = 0; index < classCounts.size(); ++index) {
    count += classCounts[index];
    if (classCounts[index] > classCounts[label]) {
      label = static_cast<std::uint16_t>(index);
    }
  }
}

std::uint16_t DecisionTree::predict(const float *features) const {
  if (!root) {
    throw std::runtime_error("Cannot predict before training.");
  }
  const Node *node = root.get();
  while (!node->isLeaf()) {
    node = features[node->feature] <= node->threshold ? node->left.get()
                                                      : node->right.get();
  }
  return node->label;
}

std::uint16_t DecisionTree::predict(const Dataset &dataset, std::size_t row) const {
  if (!root) {
    throw std::runtime_error("Cannot predict before training.");
  }
  const Node *node = root.get();
  while (!node->isLeaf()) {
    node = dataset.value(row, static_cast<std::size_t>(node->feature)) <= node->threshold
               ? node->left.get()
               : node->right.get();
  }
  return node->label;
}

double DecisionTree::accuracy(const Dataset &dataset, unsigned threads) const {
  if (dataset.rowCount == 0) {
    return 0.0;
  }
  threads = std::max(1u, std::min<unsigned>(threads, 64));
  std::atomic<std::size_t> correct{0};
  std::vector<std::thread> workers;
  const std::size_t features = dataset.featureCount();
  for (unsigned worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker]() {
      const std::size_t begin = dataset.rowCount * worker / threads;
      const std::size_t end = dataset.rowCount * (worker + 1) / threads;
      // Copy blocks of rows into row-major order first: walking the tree then
      // reads one row's values from one place instead of one cache line per
      // visited feature column.
      constexpr std::size_t kBlock = 512;
      std::vector<float> block(kBlock * features);
      std::size_t local = 0;
      for (std::size_t first = begin; first < end; first += kBlock) {
        const std::size_t rows = std::min(kBlock, end - first);
        for (std::size_t feature = 0; feature < features; ++feature) {
          const float *column = dataset.column(feature) + first;
          for (std::size_t row = 0; row < rows; ++row) {
            block[row * features + feature] = column[row];
          }
        }
        for (std::size_t row = 0; row < rows; ++row) {
          local += predict(block.data() + row * features) == dataset.labels[first + row];
        }
      }
      correct += local;
    });
  }
  for (std::thread &worker : workers) {
    worker.join();
  }
  return static_cast<double>(correct.load()) / static_cast<double>(dataset.rowCount);
}

std::size_t countNodes(const Node *node) {
  if (!node) {
    return 0;
  }
  return 1 + countNodes(node->left.get()) + countNodes(node->right.get());
}

std::size_t countLeaves(const Node *node) {
  if (!node) {
    return 0;
  }
  if (node->isLeaf()) {
    return 1;
  }
  return countLeaves(node->left.get()) + countLeaves(node->right.get());
}

int subtreeDepth(const Node *node) {
  if (!node) {
    return -1;
  }
  if (node->isLeaf()) {
    return 0;
  }
  return 1 + std::max(subtreeDepth(node->left.get()), subtreeDepth(node->right.get()));
}

std::size_t DecisionTree::nodeCount() const { return countNodes(root.get()); }
std::size_t DecisionTree::leafCount() const { return countLeaves(root.get()); }
int DecisionTree::depth() const { return subtreeDepth(root.get()); }

namespace {

void printNode(const DecisionTree &tree, const Node *node, std::ostream &output,
               int depth, const char *edge) {
  output << std::string(static_cast<std::size_t>(depth) * 2, ' ') << edge
         << " [n=" << node->count << "]: ";
  if (node->isLeaf()) {
    output << "Leaf -> " << tree.classNames[node->label] << '\n';
    return;
  }
  // Shortest text that reads back as exactly the same double.
  char threshold[32];
  const auto written = std::to_chars(threshold, threshold + sizeof(threshold), node->threshold);
  output << "if " << tree.featureNames[static_cast<std::size_t>(node->feature)] << " <= "
         << std::string_view(threshold, static_cast<std::size_t>(written.ptr - threshold))
         << '\n';
  printNode(tree, node->left.get(), output, depth + 1, "yes");
  printNode(tree, node->right.get(), output, depth + 1, "no");
}

} // namespace

void DecisionTree::print(std::ostream &output) const {
  if (!root) {
    output << "Tree is empty.\n";
    return;
  }
  printNode(*this, root.get(), output, 0, "ROOT");
}

namespace {

std::size_t findName(const std::vector<std::string> &names, const std::string &name,
                     const char *what) {
  for (std::size_t index = 0; index < names.size(); ++index) {
    if (names[index] == name) {
      return index;
    }
  }
  throw std::runtime_error(std::string("Unknown ") + what + " in tree file: " + name);
}

std::unique_ptr<Node> readNode(std::istream &input, const Dataset &dataset) {
  static const std::regex lineFormat(R"(^\s*\w+ \[n=\d+\]: (.*)$)");
  static const std::regex splitFormat(R"(^if (.+) <= (\S+)$)");
  std::string line;
  while (std::getline(input, line) && line.find_first_not_of(" \t\r") == std::string::npos) {
  }
  std::smatch match;
  if (!std::regex_match(line, match, lineFormat)) {
    throw std::runtime_error("Bad tree line: " + line);
  }
  const std::string body = match[1];
  auto node = std::make_unique<Node>();
  if (body.rfind("Leaf -> ", 0) == 0) {
    node->label = static_cast<std::uint16_t>(
        findName(dataset.classNames, body.substr(8), "class"));
    return node;
  }
  if (!std::regex_match(body, match, splitFormat)) {
    throw std::runtime_error("Bad tree line: " + line);
  }
  node->feature = static_cast<std::int32_t>(findName(dataset.featureNames, match[1], "feature"));
  node->threshold = std::stod(match[2]);
  node->left = readNode(input, dataset);
  node->right = readNode(input, dataset);
  return node;
}

void addRow(Node *node, const Dataset &dataset, std::size_t row) {
  while (true) {
    ++node->classCounts[dataset.labels[row]];
    if (node->isLeaf()) {
      return;
    }
    node = dataset.value(row, static_cast<std::size_t>(node->feature)) <= node->threshold
               ? node->left.get()
               : node->right.get();
  }
}

void finishCounts(Node *node) {
  const std::uint16_t label = node->label;
  const std::vector<std::uint32_t> counts = node->classCounts;
  node->setCounts(counts.data(), counts.size());
  if (node->count == 0) {
    node->label = label; // keep the stored class of an empty node
  }
  if (!node->isLeaf()) {
    finishCounts(node->left.get());
    finishCounts(node->right.get());
  }
}

void clearCounts(Node *node, std::size_t classCount) {
  node->classCounts.assign(classCount, 0);
  if (!node->isLeaf()) {
    clearCounts(node->left.get(), classCount);
    clearCounts(node->right.get(), classCount);
  }
}

} // namespace

std::unique_ptr<Node> readTree(std::istream &input, const Dataset &dataset) {
  std::unique_ptr<Node> root = readNode(input, dataset);
  recountTree(root.get(), dataset);
  return root;
}

void recountTree(Node *root, const Dataset &dataset) {
  clearCounts(root, dataset.classCount());
  for (std::size_t row = 0; row < dataset.rowCount; ++row) {
    addRow(root, dataset, row);
  }
  finishCounts(root);
}
