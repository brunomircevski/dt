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
//    features, and grouped by leaf in preorder, so the rows of every subtree
//    form one contiguous slice; the row-major copy (and the labels) is kept
//    in that order, so a slice is also one contiguous block of memory, read
//    sequentially and often still in cache from the branches below;
//  * the class distribution of every node is therefore known up front, and
//    only subtree raising needs to send rows through another subtree.
// Rows are therefore known by their position in this order. Pruned nodes are
// only marked (leaf / raised); Tree::compact() removes what is no longer
// reachable at the end.
class C45Pruner {
public:
  // `rowMajor` and `spare` each have room for all the feature values.
  C45Pruner(Tree &tree, const Dataset &train, float *rowMajor, float *spare,
            double confidenceFactor, ThreadPool *pool)
      : tree_(tree), train_(train), data_(rowMajor), spare_(spare),
        features_(train.featureCount()), classCount_(train.classCount()),
        estimator_(confidenceFactor), pool_(pool), labels_(train.labels),
        errors_(tree.nodes.size(), 0.0), leafSlot_(tree.nodes.size(), 0),
        fresh_(tree.nodes.size(), 0), hops_(tree.nodes.size()) {
    constexpr std::size_t kChunk = 4096;
    parallelFor(pool_, (hops_.size() + kChunk - 1) / kChunk, [&](std::size_t chunk) {
      const std::size_t end = std::min(hops_.size(), (chunk + 1) * kChunk);
      for (std::size_t id = chunk * kChunk; id < end; ++id) {
        hops_[id] = hopOf(static_cast<std::uint32_t>(id));
      }
    });
  }

  void run() {
    arrange(0, 0, train_.rowCount);
    prune(0, 0, train_.rowCount);
    tree_.compact();
  }

private:
  static constexpr std::size_t kParallelRows = 50000;
  static constexpr std::size_t kRouteChunks = 64;

  const float *features(std::size_t position) const { return data_ + position * features_; }

  // A node reduced to what routing needs: 16 bytes, four per cache line. The
  // threshold is the largest float at or below the node's, which sends every
  // float value the same way; a leaf leads back to itself.
  struct Hop {
    float threshold;
    std::uint32_t feature;
    std::uint32_t next[2]; // left, right
  };

  Hop hopOf(std::uint32_t id) const {
    const Node &node = tree_.nodes[id];
    if (node.isLeaf()) {
      return {0.0f, 0, {id, id}};
    }
    float threshold = static_cast<float>(node.threshold);
    if (static_cast<double>(threshold) > node.threshold) {
      threshold = std::nextafter(threshold, -std::numeric_limits<float>::infinity());
    }
    return {threshold, static_cast<std::uint32_t>(node.feature), {node.left, node.right}};
  }

  // The leaf of `root`'s subtree that each of the rows at [begin, begin +
  // count) reaches, read from the row-major copy or, before it exists
  // (Columns), from the dataset's columns: rows side by side then share the
  // cache lines of each column. A walk is a chain of dependent loads, so
  // kLanes rows walk side by side and a lane that reaches its leaf takes the
  // next row.
  template <bool Columns>
  void routeRange(std::uint32_t root, std::size_t begin, std::size_t count,
                  std::uint32_t *leaves) const {
    constexpr std::size_t kLanes = 16;
    const Hop *hops = hops_.data();
    const std::size_t columnSize = train_.rowCount;
    const float *x[kLanes];
    std::uint32_t id[kLanes];
    std::size_t at[kLanes];
    std::size_t lanes = 0;
    std::size_t taken = 0;
    auto take = [&](std::size_t lane) {
      at[lane] = taken;
      x[lane] = Columns ? train_.values.data() + begin + taken : features(begin + taken);
      id[lane] = root;
      ++taken;
    };
    for (; lanes < kLanes && lanes < count; ++lanes) {
      take(lanes);
    }
    while (lanes > 0) {
      for (std::size_t lane = 0; lane < lanes;) {
        const Hop &hop = hops[id[lane]];
        const float value = Columns ? x[lane][hop.feature * columnSize] : x[lane][hop.feature];
        const std::uint32_t next = hop.next[value > hop.threshold];
        if (next != id[lane]) {
          id[lane++] = next;
          continue;
        }
        leaves[at[lane]] = next;
        if (taken < count) {
          take(lane++);
        } else {
          --lanes;
          at[lane] = at[lanes];
          x[lane] = x[lanes];
          id[lane] = id[lanes];
        }
      }
    }
  }

  // Leaf of every row, in parallel for many rows.
  void route(std::uint32_t root, std::size_t begin, std::size_t count,
             std::vector<std::uint32_t> &leaves) const {
    leaves.resize(count);
    auto range = [&](std::size_t from, std::size_t to) {
      if (arranged_) {
        routeRange<false>(root, begin + from, to - from, leaves.data() + from);
      } else {
        routeRange<true>(root, begin + from, to - from, leaves.data() + from);
      }
    };
    if (!pool_ || count < kParallelRows) {
      range(0, count);
      return;
    }
    pool_->parallelFor(kRouteChunks, [&](std::size_t part) {
      range(count * part / kRouteChunks, count * (part + 1) / kRouteChunks);
    });
  }

  // Like setting the counts on a new node, but ties keep the current class
  // (prune.c starts its search for the best class at T->Leaf).
  // A node that gains rows is no longer pruned (fresh_).
  void setCounts(std::uint32_t id, const std::uint32_t *counts) {
    Node &node = tree_.nodes[id];
    std::copy(counts, counts + classCount_, tree_.counts(id));
    std::uint32_t total = 0;
    for (std::size_t k = 0; k < classCount_; ++k) {
      total += counts[k];
    }
    if (total != node.count) {
      fresh_[id] = 0;
    }
    node.count = total;
    node.label = majorityClass(counts, classCount_, node.label);
  }

  // Send the rows at [begin, begin + count) through `root`'s subtree, recount
  // the class distribution of every node in it, and reorder the rows by leaf
  // (preorder, stably) so that the rows of each node are contiguous: the left
  // child's rows come first.
  void arrange(std::uint32_t root, std::size_t begin, std::size_t count) {
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
    route(root, begin, count, leafOfRow);

    // Stable sort by leaf (`source`: the old place of each row, relative to
    // `begin`), and the class histogram of every leaf.
    std::vector<std::uint32_t> leafCounts(leaves.size() * classCount_, 0);
    std::vector<std::uint32_t> source(count);
    const bool parallel = pool_ && count >= kParallelRows;
    if (parallel) {
      sortByLeafInParallel(begin, count, leafOfRow, leaves.size(), leafCounts, source);
    } else {
      std::vector<std::size_t> offset(leaves.size() + 1, 0);
      for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t slot = leafSlot_[leafOfRow[index]];
        leafOfRow[index] = slot;
        ++offset[slot + 1];
        ++leafCounts[slot * classCount_ + labels_[begin + index]];
      }
      for (std::size_t slot = 0; slot < leaves.size(); ++slot) {
        offset[slot + 1] += offset[slot];
      }
      for (std::size_t index = 0; index < count; ++index) {
        source[offset[leafOfRow[index]]++] = static_cast<std::uint32_t>(index);
      }
    }
    if (arranged_) {
      permute(begin, count, source, parallel);
    } else {
      transposeArranged(source);
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

  // The row-major copy, written in the arranged order: the row at `position`
  // is row source[position] of the dataset. Blocks of rows are read from the
  // columns (sequentially) and each row is copied to its place.
  void transposeArranged(const std::vector<std::uint32_t> &source) {
    const std::size_t rows = train_.rowCount;
    const std::size_t width = features_;
    std::vector<std::uint32_t> place(rows);
    std::vector<std::uint16_t> labels(rows);
    constexpr std::size_t kBlock = 1024;
    const std::size_t blocks = (rows + kBlock - 1) / kBlock;
    parallelFor(pool_, blocks, [&](std::size_t block) {
      const std::size_t end = std::min(rows, (block + 1) * kBlock);
      for (std::size_t position = block * kBlock; position < end; ++position) {
        place[source[position]] = static_cast<std::uint32_t>(position);
      }
    });
    parallelFor(pool_, blocks, [&](std::size_t block) {
      const std::size_t begin = block * kBlock;
      const std::size_t end = std::min(rows, begin + kBlock);
      thread_local std::vector<float> tile;
      tile.resize(kBlock * width);
      for (std::size_t feature = 0; feature < width; ++feature) {
        const float *column = train_.column(feature);
        for (std::size_t row = begin; row < end; ++row) {
          tile[(row - begin) * width + feature] = column[row];
        }
      }
      for (std::size_t row = begin; row < end; ++row) {
        std::copy_n(tile.data() + (row - begin) * width, width, data_ + place[row] * width);
        labels[place[row]] = labels_[row];
      }
    });
    labels_ = std::move(labels);
    arranged_ = true;
  }

  // Reorder the rows (features and labels) at [begin, begin + count), after
  // subtree raising: row `index` moves from begin + source[index]. The
  // features go through the same slice of `spare_` (slices of concurrent
  // tasks are disjoint); all rows at once simply swap the buffers.
  void permute(std::size_t begin, std::size_t count, const std::vector<std::uint32_t> &source,
               bool parallel) {
    std::vector<std::uint16_t> labels(count);
    const std::size_t width = features_;
    auto gather = [&](std::size_t from, std::size_t to) {
      for (std::size_t index = from; index < to; ++index) {
        const std::size_t old = begin + source[index];
        std::copy_n(data_ + old * width, width, spare_ + (begin + index) * width);
        labels[index] = labels_[old];
      }
    };
    const bool whole = begin == 0 && count == train_.rowCount;
    auto copyBack = [&](std::size_t from, std::size_t to) {
      if (!whole) {
        std::copy(spare_ + (begin + from) * width, spare_ + (begin + to) * width,
                  data_ + (begin + from) * width);
      }
      std::copy(labels.begin() + from, labels.begin() + to, labels_.begin() + begin + from);
    };
    if (parallel) {
      pool_->parallelFor(kRouteChunks, [&](std::size_t part) {
        gather(count * part / kRouteChunks, count * (part + 1) / kRouteChunks);
      });
      pool_->parallelFor(kRouteChunks, [&](std::size_t part) {
        copyBack(count * part / kRouteChunks, count * (part + 1) / kRouteChunks);
      });
    } else {
      gather(0, count);
      copyBack(0, count);
    }
    if (whole) {
      std::swap(data_, spare_);
    }
  }

  // The same as arrange()'s counting sort, for many rows: `leaves` holds the
  // leaf of every row and receives its slot; a stable LSD radix sort of
  // (slot, index) in 11-bit digits orders the rows exactly as the counting
  // sort does, and the class counts are added up run by run.
  void sortByLeafInParallel(std::size_t begin, std::size_t count,
                            std::vector<std::uint32_t> &leaves, std::size_t slotCount,
                            std::vector<std::uint32_t> &leafCounts,
                            std::vector<std::uint32_t> &source) {
    constexpr unsigned kBits = 11;
    constexpr std::size_t kBuckets = std::size_t{1} << kBits;
    const std::size_t chunks = 4 * (pool_->workerCount() + 1);
    auto forChunks = [&](auto &&body) {
      pool_->parallelFor(chunks, [&](std::size_t chunk) {
        body(chunk, count * chunk / chunks, count * (chunk + 1) / chunks);
      });
    };
    forChunks([&](std::size_t, std::size_t from, std::size_t to) {
      for (std::size_t index = from; index < to; ++index) {
        leaves[index] = leafSlot_[leaves[index]];
        source[index] = static_cast<std::uint32_t>(index);
      }
    });
    std::vector<std::uint32_t> keyTemp(count);
    std::vector<std::uint32_t> valueTemp(count);
    std::vector<std::uint32_t> histogram(chunks * kBuckets);
    std::uint32_t *keys = leaves.data();
    std::uint32_t *values = source.data();
    std::uint32_t *keysOut = keyTemp.data();
    std::uint32_t *valuesOut = valueTemp.data();
    for (unsigned shift = 0; (std::size_t{1} << shift) < slotCount; shift += kBits) {
      forChunks([&](std::size_t chunk, std::size_t from, std::size_t to) {
        std::uint32_t *counts = histogram.data() + chunk * kBuckets;
        std::fill_n(counts, kBuckets, 0u);
        for (std::size_t index = from; index < to; ++index) {
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
      forChunks([&](std::size_t chunk, std::size_t from, std::size_t to) {
        std::uint32_t *next = histogram.data() + chunk * kBuckets;
        for (std::size_t index = from; index < to; ++index) {
          const std::uint32_t at = next[(keys[index] >> shift) & (kBuckets - 1)]++;
          keysOut[at] = keys[index];
          valuesOut[at] = values[index];
        }
      });
      std::swap(keys, keysOut);
      std::swap(values, valuesOut);
    }
    forChunks([&](std::size_t, std::size_t from, std::size_t to) {
      if (values != source.data()) {
        std::copy(values + from, values + to, source.data() + from);
      }
      // Runs of one slot; a run may continue in the next chunk.
      for (std::size_t index = from; index < to;) {
        const std::uint32_t slot = keys[index];
        std::uint32_t *target = leafCounts.data() + std::size_t{slot} * classCount_;
        thread_local std::vector<std::uint32_t> local;
        local.assign(classCount_, 0);
        for (; index < to && keys[index] == slot; ++index) {
          ++local[labels_[begin + values[index]]];
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

  // EstimateErrors(T, rows, UpdateTree = true) for the node's rows, at
  // [begin, begin + count) and arranged as above; its counts are up to date.
  //
  // A subtree pruned before (fresh_) whose rows have not changed since is
  // left as it is: pruning it again would repeat the same decisions on the
  // same rows (in the same order, as arrange() sorts stably) and end with
  // the same estimate. Only subtree raising prunes a subtree again, and it
  // adds rows to only some of its nodes.
  double prune(std::uint32_t id, std::size_t begin, std::size_t count) {
    if (fresh_[id]) {
      return errors_[id];
    }
    for (;;) {
      if (tree_.nodes[id].isLeaf()) {
        fresh_[id] = 1;
        return errors_[id] = leafErrors(id);
      }
      // Branches without cases are skipped, as in c4.5.
      const std::uint32_t left = tree_.nodes[id].left;
      const std::size_t leftCount = tree_.nodes[left].count;
      const std::uint32_t right = tree_.nodes[id].right;
      double branchErrors[2] = {0.0, 0.0};
      parallelFor(count >= kParallelRows ? pool_ : nullptr, 2, [&](std::size_t side) {
        if (side == 0 && leftCount > 0) {
          branchErrors[0] = prune(left, begin, leftCount);
        } else if (side == 1 && leftCount < count) {
          branchErrors[1] = prune(right, begin + leftCount, count - leftCount);
        }
      });
      double errors;
      if (settle(id, begin, count, branchErrors[0] + branchErrors[1], errors)) {
        return errors;
      }
    }
  }

  double leafErrors(std::uint32_t id) const {
    return estimator_.leafErrors(tree_.counts(id), classCount_, tree_.nodes[id].label);
  }

  // The rest of prune() for decision node `id` once its branches are pruned
  // (`treeErrors`): keep the subtree, make a leaf, or raise the largest
  // branch. Returns false after raising, when `id` must be pruned again.
  bool settle(std::uint32_t id, std::size_t begin, std::size_t count, double treeErrors,
              double &errors) {
    const double leafEstimate = leafErrors(id);
    const std::uint32_t left = tree_.nodes[id].left;
    const std::uint32_t right = tree_.nodes[id].right;
    const std::size_t leftCount = tree_.nodes[left].count;

    // The branch with most cases (ties: the later branch, as in c4.5), and its
    // errors if it also received the other branch's cases (EstimateErrors with
    // UpdateTree = false).
    const bool rightIsLargest = count - leftCount >= leftCount;
    const std::uint32_t largest = rightIsLargest ? right : left;
    const double largestBranchErrors =
        rightIsLargest ? errorsWithExtraRows(right, begin, leftCount)
                       : errorsWithExtraRows(left, begin + leftCount, count - leftCount);

    if (leafEstimate <= largestBranchErrors + 0.1 && leafEstimate <= treeErrors + 0.1) {
      tree_.nodes[id].makeLeaf();
      hops_[id] = hopOf(id);
      fresh_[id] = 1;
      errors = errors_[id] = leafEstimate;
      return true;
    }
    if (largestBranchErrors <= treeErrors + 0.1) {
      // Subtree raising: the largest branch replaces this node and is pruned
      // again with all of this node's cases.
      tree_.nodes[id] = tree_.nodes[largest];
      hops_[id] = hopOf(id);
      arrange(id, begin, count);
      return false;
    }
    fresh_[id] = 1;
    errors = errors_[id] = treeErrors;
    return true;
  }

  // Estimated errors of `root`'s subtree (as currently pruned) if the `extra`
  // rows were sent through it in addition to the rows it already holds
  // (EstimateErrors with UpdateTree = false). That is the sum of the leaf
  // estimates, so it equals the stored estimate of the subtree plus, for each
  // leaf that receives extra rows, the change of that leaf's estimate.
  double errorsWithExtraRows(std::uint32_t root, std::size_t begin, std::size_t count) {
    if (count == 0) {
      return errors_[root];
    }
    // Per-thread buffers: this runs for every decision node.
    thread_local std::vector<std::uint32_t> arrival;
    thread_local std::vector<std::uint32_t> leaves;
    thread_local std::vector<std::uint32_t> extraCounts;
    thread_local std::vector<std::uint32_t> counts;
    route(root, begin, count, arrival);
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
      ++extraCounts[std::size_t{slot} * classCount_ + labels_[begin + index]];
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
  float *data_;  // row-major features, in the arranged order
  float *spare_; // as large, for reordering
  std::size_t features_;
  std::size_t classCount_;
  ErrorEstimator estimator_;
  ThreadPool *pool_;
  std::vector<std::uint16_t> labels_;   // in the arranged order
  bool arranged_ = false;               // data_ written yet (by the first arrange())
  std::vector<double> errors_;          // per node: estimated errors of its subtree
  std::vector<std::uint32_t> leafSlot_; // per leaf: scratch slot number
  std::vector<std::uint8_t> fresh_;     // per node: pruned, for its current rows
  std::vector<Hop> hops_;               // per node, kept in step with tree_.nodes
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
  // The row-major copy and the room to reorder it come from the grower's
  // workspace when it is large enough (it holds two copies for C4.5).
  const std::size_t values = train.rowCount * train.featureCount();
  std::unique_ptr<float[]> owned;
  float *rowMajor = reinterpret_cast<float *>(workspace.data());
  const std::size_t room =
      reinterpret_cast<std::uintptr_t>(rowMajor) % alignof(float) == 0
          ? workspace.size() / sizeof(float)
          : 0;
  if (room < 2 * values) {
    owned.reset(new float[room < values ? 2 * values : values]);
    if (room < values) {
      rowMajor = owned.get();
    }
  }
  float *spare = room >= 2 * values ? rowMajor + values
                 : room >= values   ? owned.get()
                                    : owned.get() + values;
  C45Pruner(tree, train, rowMajor, spare, confidenceFactor, pool).run();
}

} // namespace dt
