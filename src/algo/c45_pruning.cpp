#include "algo/c45_pruning.h"

#include "core/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace dt {

namespace {

constexpr double kEpsilon = 1e-3; // C4.5's Epsilon (defns.i)

// The nodes of `root`'s subtree in preorder, left branch first.
void subtreePreorder(const Tree &tree, std::uint32_t root, std::vector<std::uint32_t> &out) {
  out.clear();
  std::vector<std::uint32_t> stack{root};
  while (!stack.empty()) {
    const std::uint32_t id = stack.back();
    stack.pop_back();
    out.push_back(id);
    const Node &node = tree.nodes[id];
    if (!node.isLeaf()) {
      stack.push_back(node.right);
      stack.push_back(node.left);
    }
  }
}

// stats.c AddErrs(): extra errors to add to `errors` observed among `cases`,
// so that the total is the upper limit of the CF confidence interval. The
// normal deviate for CF is interpolated from c4.5's table, like the original.
class ErrorEstimator {
public:
  explicit ErrorEstimator(double confidenceFactor) : cf_(confidenceFactor) {
    static const double kValue[] = {0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00};
    static const double kDeviation[] = {4.0, 3.09, 2.58, 2.33, 1.65, 1.28, 0.84, 0.25, 0.00};
    int index = 0;
    while (cf_ > kValue[index]) {
      ++index;
    }
    index = std::max(index, 1);
    const double deviation =
        kDeviation[index - 1] + (kDeviation[index] - kDeviation[index - 1]) *
                                    (cf_ - kValue[index - 1]) /
                                    (kValue[index] - kValue[index - 1]);
    coefficient_ = deviation * deviation;
  }

  double extraErrors(double cases, double errors) const {
    if (errors < 1e-6) {
      return cases * (1.0 - std::exp(std::log(cf_) / cases));
    }
    if (errors < 0.9999) {
      const double zeroErrors = cases * (1.0 - std::exp(std::log(cf_) / cases));
      return zeroErrors + errors * (extraErrors(cases, 1.0) - zeroErrors);
    }
    if (errors + 0.5 >= cases) {
      return 0.67 * (cases - errors);
    }
    const double probability =
        (errors + 0.5 + coefficient_ / 2 +
         std::sqrt(coefficient_ * ((errors + 0.5) * (1 - (errors + 0.5) / cases) +
                                   coefficient_ / 4))) /
        (cases + coefficient_);
    return cases * probability - errors;
  }

  // Estimated errors if this class distribution were a leaf. The leaf class
  // is the most frequent one, ties going to `currentLabel` (as in prune.c).
  double leafErrors(const std::uint32_t *counts, std::size_t classCount,
                    std::uint16_t currentLabel) const {
    double cases = 0.0;
    for (std::size_t k = 0; k < classCount; ++k) {
      cases += counts[k];
    }
    if (cases == 0.0) {
      return 0.0;
    }
    const double errors = cases - counts[majorityClass(counts, classCount, currentLabel)];
    return errors + extraErrors(cases, errors);
  }

private:
  double cf_;
  double coefficient_;
};

// prune.c EstimateErrors(), organised for speed:
//  * rows are routed through the tree once, using a row-major copy of the
//    features (a row's values share a cache line), and grouped by leaf in
//    preorder, so the rows of every subtree form one contiguous slice;
//  * the class distribution of every node is therefore known up front, and
//    only subtree raising needs to send rows through another subtree.
// Pruned nodes are only marked (leaf / raised); Tree::compact() removes what
// is no longer reachable at the end.
class C45Pruner {
public:
  C45Pruner(Tree &tree, const Dataset &train, const float *rowMajor, double confidenceFactor,
            ThreadPool *pool)
      : tree_(tree), train_(train), rowMajor_(rowMajor), features_(train.featureCount()),
        classCount_(train.classCount()), estimator_(confidenceFactor), pool_(pool),
        errors_(tree.nodes.size(), 0.0), leafSlot_(tree.nodes.size(), 0) {}

  void run() {
    std::vector<std::uint32_t> rows(train_.rowCount);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      rows[row] = static_cast<std::uint32_t>(row);
    }
    arrange(0, rows.data(), rows.size());
    prune(0, rows.data(), rows.size());
    tree_.compact();
  }

private:
  static constexpr std::size_t kParallelRows = 50000;
  static constexpr std::size_t kRouteChunks = 64;

  const float *features(std::uint32_t row) const {
    return rowMajor_ + std::size_t{row} * features_;
  }

  // The leaf of `root`'s subtree that a row reaches.
  std::uint32_t leafOf(std::uint32_t root, std::uint32_t row) const {
    const float *x = features(row);
    const Node *nodes = tree_.nodes.data();
    std::uint32_t id = root;
    while (!nodes[id].isLeaf()) {
      id = x[nodes[id].feature] <= nodes[id].threshold ? nodes[id].left : nodes[id].right;
    }
    return id;
  }

  // Leaf of every row, in parallel for many rows.
  void route(std::uint32_t root, const std::uint32_t *rows, std::size_t count,
             std::vector<std::uint32_t> &leaves) const {
    leaves.resize(count);
    auto chunk = [&](std::size_t part) {
      const std::size_t begin = count * part / kRouteChunks;
      const std::size_t end = count * (part + 1) / kRouteChunks;
      for (std::size_t index = begin; index < end; ++index) {
        leaves[index] = leafOf(root, rows[index]);
      }
    };
    parallelFor(count >= kParallelRows ? pool_ : nullptr, kRouteChunks, chunk);
  }

  // Like setting the counts on a new node, but ties keep the current class
  // (prune.c starts its search for the best class at T->Leaf).
  void setCounts(std::uint32_t id, const std::uint32_t *counts) {
    Node &node = tree_.nodes[id];
    std::copy(counts, counts + classCount_, tree_.counts(id));
    node.count = 0;
    for (std::size_t k = 0; k < classCount_; ++k) {
      node.count += counts[k];
    }
    node.label = majorityClass(counts, classCount_, node.label);
  }

  // Send `rows` through `root`'s subtree, recount the class distribution of
  // every node in it, and reorder the rows by leaf (preorder) so that the rows
  // of each node are contiguous: the left child's rows come first.
  void arrange(std::uint32_t root, std::uint32_t *rows, std::size_t count) {
    std::vector<std::uint32_t> order;
    subtreePreorder(tree_, root, order);
    std::vector<std::uint32_t> leaves;
    for (std::uint32_t id : order) {
      if (tree_.nodes[id].isLeaf()) {
        leafSlot_[id] = static_cast<std::uint32_t>(leaves.size());
        leaves.push_back(id);
      }
    }
    std::vector<std::uint32_t> leafOfRow;
    route(root, rows, count, leafOfRow);

    // Stable sort by leaf, and the class histogram of every leaf.
    std::vector<std::uint32_t> leafCounts(leaves.size() * classCount_, 0);
    if (pool_ && count >= kParallelRows) {
      sortByLeafInParallel(rows, count, leafOfRow, leaves.size(), leafCounts);
    } else {
      std::vector<std::size_t> offset(leaves.size() + 1, 0);
      for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t slot = leafSlot_[leafOfRow[index]];
        leafOfRow[index] = slot;
        ++offset[slot + 1];
        ++leafCounts[slot * classCount_ + train_.labels[rows[index]]];
      }
      for (std::size_t slot = 0; slot < leaves.size(); ++slot) {
        offset[slot + 1] += offset[slot];
      }
      std::vector<std::uint32_t> sorted(count);
      for (std::size_t index = 0; index < count; ++index) {
        sorted[offset[leafOfRow[index]]++] = rows[index];
      }
      std::copy(sorted.begin(), sorted.end(), rows);
    }
    for (std::size_t slot = 0; slot < leaves.size(); ++slot) {
      setCounts(leaves[slot], leafCounts.data() + slot * classCount_);
    }

    // Decision nodes: sum of their children, bottom-up.
    std::vector<std::uint32_t> sum(classCount_);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      const Node &node = tree_.nodes[*it];
      if (node.isLeaf()) {
        continue;
      }
      const std::uint32_t *left = tree_.counts(node.left);
      const std::uint32_t *right = tree_.counts(node.right);
      for (std::size_t k = 0; k < classCount_; ++k) {
        sum[k] = left[k] + right[k];
      }
      setCounts(*it, sum.data());
    }
  }

  // The same as arrange()'s counting sort, for many rows: `leaves` holds the
  // leaf of every row and receives its slot; a stable LSD radix sort of
  // (slot, row) in 11-bit digits orders the rows exactly as the counting sort
  // does, and the class counts are added up run by run.
  void sortByLeafInParallel(std::uint32_t *rows, std::size_t count,
                            std::vector<std::uint32_t> &leaves, std::size_t slotCount,
                            std::vector<std::uint32_t> &leafCounts) {
    constexpr unsigned kBits = 11;
    constexpr std::size_t kBuckets = std::size_t{1} << kBits;
    const std::size_t chunks = 4 * (pool_->workerCount() + 1);
    auto forChunks = [&](auto &&body) {
      parallelFor(pool_, chunks, [&](std::size_t chunk) {
        body(chunk, count * chunk / chunks, count * (chunk + 1) / chunks);
      });
    };
    forChunks([&](std::size_t, std::size_t begin, std::size_t end) {
      for (std::size_t index = begin; index < end; ++index) {
        leaves[index] = leafSlot_[leaves[index]];
      }
    });
    std::vector<std::uint32_t> keyTemp(count);
    std::vector<std::uint32_t> rowTemp(count);
    std::vector<std::uint32_t> histogram(chunks * kBuckets);
    std::uint32_t *keys = leaves.data();
    std::uint32_t *values = rows;
    std::uint32_t *keysOut = keyTemp.data();
    std::uint32_t *valuesOut = rowTemp.data();
    for (unsigned shift = 0; (std::size_t{1} << shift) < slotCount; shift += kBits) {
      forChunks([&](std::size_t chunk, std::size_t begin, std::size_t end) {
        std::uint32_t *counts = histogram.data() + chunk * kBuckets;
        std::fill_n(counts, kBuckets, 0u);
        for (std::size_t index = begin; index < end; ++index) {
          ++counts[(keys[index] >> shift) & (kBuckets - 1)];
        }
      });
      std::uint32_t sum = 0;
      for (std::size_t digit = 0; digit < kBuckets; ++digit) {
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
          const std::uint32_t size = histogram[chunk * kBuckets + digit];
          histogram[chunk * kBuckets + digit] = sum;
          sum += size;
        }
      }
      forChunks([&](std::size_t chunk, std::size_t begin, std::size_t end) {
        std::uint32_t *next = histogram.data() + chunk * kBuckets;
        for (std::size_t index = begin; index < end; ++index) {
          const std::uint32_t at = next[(keys[index] >> shift) & (kBuckets - 1)]++;
          keysOut[at] = keys[index];
          valuesOut[at] = values[index];
        }
      });
      std::swap(keys, keysOut);
      std::swap(values, valuesOut);
    }
    forChunks([&](std::size_t, std::size_t begin, std::size_t end) {
      if (values != rows) {
        std::copy(values + begin, values + end, rows + begin);
      }
      // Runs of one slot; a run may continue in the next chunk.
      for (std::size_t index = begin; index < end;) {
        const std::uint32_t slot = keys[index];
        std::uint32_t *target = leafCounts.data() + std::size_t{slot} * classCount_;
        thread_local std::vector<std::uint32_t> local;
        local.assign(classCount_, 0);
        for (; index < end && keys[index] == slot; ++index) {
          ++local[train_.labels[values[index]]];
        }
        for (std::size_t k = 0; k < classCount_; ++k) {
          if (local[k] > 0) {
            std::atomic_ref<std::uint32_t>(target[k]).fetch_add(local[k],
                                                                std::memory_order_relaxed);
          }
        }
      }
    });
  }

  // EstimateErrors(T, rows, UpdateTree = true). `rows` is the node's slice,
  // arranged as above; its counts are already up to date.
  double prune(std::uint32_t id, std::uint32_t *rows, std::size_t count) {
    const double leafEstimate =
        estimator_.leafErrors(tree_.counts(id), classCount_, tree_.nodes[id].label);
    if (tree_.nodes[id].isLeaf()) {
      return errors_[id] = leafEstimate;
    }

    // Branches without cases are skipped, as in c4.5.
    const std::uint32_t left = tree_.nodes[id].left;
    const std::uint32_t right = tree_.nodes[id].right;
    const std::size_t leftCount = tree_.nodes[left].count;
    double branchErrors[2] = {0.0, 0.0};
    auto pruneBranch = [&](std::size_t side) {
      if (side == 0 && leftCount > 0) {
        branchErrors[0] = prune(left, rows, leftCount);
      } else if (side == 1 && leftCount < count) {
        branchErrors[1] = prune(right, rows + leftCount, count - leftCount);
      }
    };
    parallelFor(count >= kParallelRows ? pool_ : nullptr, 2, pruneBranch);
    const double treeErrors = branchErrors[0] + branchErrors[1];

    // The branch with most cases (ties: the later branch, as in c4.5), and its
    // errors if it also received the other branch's cases (EstimateErrors with
    // UpdateTree = false).
    const bool rightIsLargest = count - leftCount >= leftCount;
    const std::uint32_t largest = rightIsLargest ? right : left;
    const double largestBranchErrors =
        rightIsLargest ? errorsWithExtraRows(right, rows, leftCount)
                       : errorsWithExtraRows(left, rows + leftCount, count - leftCount);

    if (leafEstimate <= largestBranchErrors + 0.1 && leafEstimate <= treeErrors + 0.1) {
      tree_.nodes[id].makeLeaf();
      return errors_[id] = leafEstimate;
    }
    if (largestBranchErrors <= treeErrors + 0.1) {
      // Subtree raising: the largest branch replaces this node and is pruned
      // again with all of this node's cases.
      tree_.nodes[id] = tree_.nodes[largest];
      arrange(id, rows, count);
      return prune(id, rows, count);
    }
    return errors_[id] = treeErrors;
  }

  // Estimated errors of `root`'s subtree (as currently pruned) if the `extra`
  // rows were sent through it in addition to the rows it already holds
  // (EstimateErrors with UpdateTree = false). That is the sum of the leaf
  // estimates, so it equals the stored estimate of the subtree plus, for each
  // leaf that receives extra rows, the change of that leaf's estimate.
  double errorsWithExtraRows(std::uint32_t root, const std::uint32_t *extra,
                             std::size_t count) {
    if (count == 0) {
      return errors_[root];
    }
    // Per-thread buffers: this runs for every decision node.
    thread_local std::vector<std::uint32_t> arrival;
    thread_local std::vector<std::uint32_t> leaves;
    thread_local std::vector<std::uint32_t> extraCounts;
    thread_local std::vector<std::uint32_t> counts;
    route(root, extra, count, arrival);
    // Class counts of the extra rows per reached leaf. A leaf's slot is kept
    // in leafSlot_ (validated against `leaves`, as arrange() reuses it).
    leaves.clear();
    extraCounts.clear();
    for (std::size_t index = 0; index < count; ++index) {
      const std::uint32_t leaf = arrival[index];
      std::uint32_t &slot = leafSlot_[leaf];
      if (slot >= leaves.size() || leaves[slot] != leaf) {
        slot = static_cast<std::uint32_t>(leaves.size());
        leaves.push_back(leaf);
        extraCounts.resize(extraCounts.size() + classCount_, 0);
      }
      ++extraCounts[std::size_t{slot} * classCount_ + train_.labels[extra[index]]];
    }
    double errors = errors_[root];
    counts.resize(classCount_);
    for (std::size_t slot = 0; slot < leaves.size(); ++slot) {
      const std::uint32_t leaf = leaves[slot];
      const std::uint32_t *own = tree_.counts(leaf);
      for (std::size_t k = 0; k < classCount_; ++k) {
        counts[k] = own[k] + extraCounts[slot * classCount_ + k];
      }
      errors += estimator_.leafErrors(counts.data(), classCount_, tree_.nodes[leaf].label) -
                (tree_.nodes[leaf].count > 0 ? errors_[leaf] : 0.0);
    }
    return errors;
  }

  Tree &tree_;
  const Dataset &train_;
  const float *rowMajor_;
  std::size_t features_;
  std::size_t classCount_;
  ErrorEstimator estimator_;
  ThreadPool *pool_;
  std::vector<double> errors_;          // per node: estimated errors of its subtree
  std::vector<std::uint32_t> leafSlot_; // per leaf: scratch slot number
};

} // namespace

void c45CollapseUselessSplits(Tree &tree) {
  // Bottom-up (preorder ids: children have larger ids than their parent).
  std::vector<double> subtreeErrors(tree.nodes.size());
  for (std::size_t id = tree.nodes.size(); id-- > 0;) {
    Node &node = tree.nodes[id];
    const double leafErrors = node.count - tree.counts(static_cast<std::uint32_t>(id))[node.label];
    if (node.isLeaf()) {
      subtreeErrors[id] = leafErrors;
      continue;
    }
    subtreeErrors[id] = subtreeErrors[node.left] + subtreeErrors[node.right];
    if (subtreeErrors[id] >= leafErrors - kEpsilon) { // build.c: Errors >= Cases - NoBestClass - Epsilon
      node.makeLeaf();
    }
  }
  tree.compact();
}

void c45UseTrainingValueThresholds(Tree &tree, std::span<const float> sortedValues,
                                   std::size_t rowCount, ThreadPool *pool) {
  constexpr std::size_t kChunk = 4096;
  const std::size_t chunks = (tree.nodes.size() + kChunk - 1) / kChunk;
  parallelFor(pool, chunks, [&](std::size_t chunk) {
    const std::size_t end = std::min(tree.nodes.size(), (chunk + 1) * kChunk);
    for (std::size_t id = chunk * kChunk; id < end; ++id) {
      Node &node = tree.nodes[id];
      if (node.isLeaf()) {
        continue;
      }
      const float *begin =
          sortedValues.data() + static_cast<std::size_t>(node.feature) * rowCount;
      const float *above = std::upper_bound(begin, begin + rowCount, node.threshold,
                                            [](double threshold, float value) {
                                              return threshold < static_cast<double>(value);
                                            });
      // A grown tree always has a value at or below its midpoint (the one
      // just left of the cut); keep the threshold otherwise.
      if (above != begin) {
        node.threshold = static_cast<double>(above[-1]);
      }
    }
  });
}

void c45PessimisticPrune(Tree &tree, const Dataset &train, double confidenceFactor,
                         ThreadPool *pool, std::span<std::byte> workspace) {
  if (tree.nodes.empty()) {
    return;
  }
  const std::size_t values = train.rowCount * train.featureCount();
  std::unique_ptr<float[]> owned;
  float *rowMajor = reinterpret_cast<float *>(workspace.data());
  if (workspace.size() < values * sizeof(float) ||
      reinterpret_cast<std::uintptr_t>(rowMajor) % alignof(float) != 0) {
    owned.reset(new float[values]);
    rowMajor = owned.get();
  }
  rowMajorFeatures(train, pool, rowMajor);
  C45Pruner(tree, train, rowMajor, confidenceFactor, pool).run();
}

} // namespace dt
