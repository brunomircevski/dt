#include "core/tree.h"

#include "core/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <stdexcept>
#include <string_view>

namespace dt {

std::uint16_t majorityClass(const std::uint32_t *counts, std::size_t classCount,
                            std::uint16_t preferred) {
  std::uint16_t best = preferred;
  for (std::size_t k = 0; k < classCount; ++k) {
    if (counts[k] > counts[best]) {
      best = static_cast<std::uint16_t>(k);
    }
  }
  return best;
}

std::uint16_t Tree::predict(const float *row) const {
  if (nodes.empty()) {
    throw std::runtime_error("Cannot predict before training.");
  }
  const Node *node = nodes.data();
  while (!node->isLeaf()) {
    node = &nodes[row[node->feature] <= node->threshold ? node->left : node->right];
  }
  return node->label;
}

double Tree::accuracy(const Dataset &dataset, ThreadPool *pool) const {
  if (dataset.rowCount == 0) {
    return 0.0;
  }
  // Blocks of rows are copied to row-major order first: walking the tree then
  // reads one row's values from one place instead of one cache line per
  // visited feature column.
  constexpr std::size_t kBlock = 512;
  const std::size_t features = dataset.featureCount();
  const std::size_t blocks = (dataset.rowCount + kBlock - 1) / kBlock;
  std::atomic<std::size_t> correct{0};
  parallelFor(pool, blocks, [&](std::size_t block) {
    const std::size_t first = block * kBlock;
    const std::size_t rows = std::min(kBlock, dataset.rowCount - first);
    thread_local std::vector<float> buffer;
    buffer.resize(kBlock * features);
    for (std::size_t feature = 0; feature < features; ++feature) {
      const float *column = dataset.column(feature) + first;
      for (std::size_t row = 0; row < rows; ++row) {
        buffer[row * features + feature] = column[row];
      }
    }
    std::size_t local = 0;
    for (std::size_t row = 0; row < rows; ++row) {
      local += predict(buffer.data() + row * features) == dataset.labels[first + row];
    }
    correct += local;
  });
  return static_cast<double>(correct.load()) / static_cast<double>(dataset.rowCount);
}

std::size_t Tree::leafCount() const {
  return static_cast<std::size_t>(
      std::count_if(nodes.begin(), nodes.end(), [](const Node &node) { return node.isLeaf(); }));
}

int Tree::depth() const {
  // Preorder: parents come before their children.
  std::vector<int> depths(nodes.size(), 0);
  int deepest = nodes.empty() ? -1 : 0;
  for (std::size_t id = 0; id < nodes.size(); ++id) {
    const Node &node = nodes[id];
    if (!node.isLeaf()) {
      depths[node.left] = depths[node.right] = depths[id] + 1;
      deepest = std::max(deepest, depths[id] + 1);
    }
  }
  return deepest;
}

void Tree::print(std::ostream &output) const {
  if (nodes.empty()) {
    output << "Tree is empty.\n";
    return;
  }
  struct Item {
    std::uint32_t id;
    int depth;
    const char *edge;
  };
  std::vector<Item> stack{{0, 0, "ROOT"}};
  while (!stack.empty()) {
    const Item item = stack.back();
    stack.pop_back();
    const Node &node = nodes[item.id];
    output << std::string(static_cast<std::size_t>(item.depth) * 2, ' ') << item.edge
           << " [n=" << node.count << "]: ";
    if (node.isLeaf()) {
      output << "Leaf -> " << classNames[node.label] << '\n';
      continue;
    }
    // Shortest text that reads back as exactly the same double.
    char threshold[32];
    const auto written = std::to_chars(threshold, threshold + sizeof(threshold), node.threshold);
    output << "if " << featureNames[static_cast<std::size_t>(node.feature)] << " <= "
           << std::string_view(threshold, static_cast<std::size_t>(written.ptr - threshold))
           << '\n';
    stack.push_back({node.right, item.depth + 1, "no"});
    stack.push_back({node.left, item.depth + 1, "yes"});
  }
}

void Tree::compact() {
  if (nodes.empty()) {
    return;
  }
  const std::vector<Node> oldNodes = std::move(nodes);
  const std::vector<std::uint32_t> oldCounts = std::move(classCounts);
  const std::size_t classes = classCount();
  assignPreorder(
      0, [&](std::uint32_t id) -> const Node & { return oldNodes[id]; },
      [&](std::uint32_t id) { return oldCounts.data() + std::size_t{id} * classes; });
}

} // namespace dt
