#include "pruning.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// =============================================================================
// C4.5
// =============================================================================

namespace {

std::uint32_t trainingErrors(const Node *node) {
  return node->count - node->classCounts[node->label];
}

// Returns the training errors of the (possibly collapsed) subtree.
double collapse(Node *node) {
  const double leafErrors = trainingErrors(node);
  if (node->isLeaf()) {
    return leafErrors;
  }
  const double subtreeErrors = collapse(node->left.get()) + collapse(node->right.get());
  if (subtreeErrors >= leafErrors - 1e-3) { // build.c: Errors >= Cases - NoBestClass - Epsilon
    node->makeLeaf();
  }
  return subtreeErrors;
}

void collectDecisionNodes(Node *node, std::vector<std::vector<Node *>> &byFeature) {
  if (!node || node->isLeaf()) {
    return;
  }
  byFeature[static_cast<std::size_t>(node->feature)].push_back(node);
  collectDecisionNodes(node->left.get(), byFeature);
  collectDecisionNodes(node->right.get(), byFeature);
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
                    std::uint16_t currentLabel, std::uint16_t *bestLabel = nullptr) const {
    double cases = 0.0;
    std::uint16_t best = currentLabel;
    for (std::size_t index = 0; index < classCount; ++index) {
      cases += counts[index];
      if (counts[index] > counts[best]) {
        best = static_cast<std::uint16_t>(index);
      }
    }
    if (bestLabel) {
      *bestLabel = best;
    }
    if (cases == 0.0) {
      return 0.0;
    }
    const double errors = cases - counts[best];
    return errors + extraErrors(cases, errors);
  }

private:
  double cf_;
  double coefficient_;
};

// prune.c EstimateErrors(), organised for speed:
//  * rows are routed through the tree once, using a row-major copy of the
//    features (a row's values share a cache line), and grouped by leaf in DFS
//    order, so the rows of every subtree form one contiguous slice;
//  * the class distribution of every node is therefore known up front, and
//    only subtree raising needs to send rows through another subtree.
class C45Pruner {
public:
  C45Pruner(const Dataset &train, const float *rowMajor, double confidenceFactor,
            bool subtreeRaising, ThreadPool *pool)
      : train_(train), rowMajor_(rowMajor), features_(train.featureCount()),
        estimator_(confidenceFactor), raising_(subtreeRaising), pool_(pool),
        classCount_(train.classCount()) {}

  void run(std::unique_ptr<Node> &root) {
    std::vector<std::uint32_t> rows(train_.rowCount);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      rows[row] = static_cast<std::uint32_t>(row);
    }
    arrange(root.get(), rows.data(), rows.size());
    prune(root, rows.data(), rows.size());
  }

private:
  static constexpr std::size_t kParallelRows = 50000;

  const float *features(std::uint32_t row) const { return rowMajor_ + std::size_t{row} * features_; }

  static bool goesLeft(const Node &node, const float *x) {
    return x[node.feature] <= node.threshold;
  }

  // Send `rows` through `node`'s subtree, recount the class distribution of
  // every node in it, and reorder the rows by leaf (DFS order) so that the
  // rows of each node are contiguous: the left child's rows come first.
  void arrange(Node *node, std::uint32_t *rows, std::size_t count) {
    std::vector<Node *> leaves;
    collectLeaves(node, leaves);
    for (std::size_t index = 0; index < leaves.size(); ++index) {
      leaves[index]->index = static_cast<std::uint32_t>(index);
    }
    std::vector<std::uint32_t> leafOf(count);
    auto route = [&](std::size_t chunk) {
      const std::size_t begin = count * chunk / 64;
      const std::size_t end = count * (chunk + 1) / 64;
      for (std::size_t index = begin; index < end; ++index) {
        const float *x = features(rows[index]);
        const Node *at = node;
        while (!at->isLeaf()) {
          at = goesLeft(*at, x) ? at->left.get() : at->right.get();
        }
        leafOf[index] = at->index;
      }
    };
    if (pool_ && count >= kParallelRows) {
      pool_->parallelFor(64, route);
    } else {
      for (std::size_t chunk = 0; chunk < 64; ++chunk) {
        route(chunk);
      }
    }

    // Counting sort by leaf, and the class histogram of every leaf.
    std::vector<std::size_t> offset(leaves.size() + 1, 0);
    std::vector<std::uint32_t> leafCounts(leaves.size() * classCount_, 0);
    for (std::size_t index = 0; index < count; ++index) {
      ++offset[leafOf[index] + 1];
      ++leafCounts[leafOf[index] * classCount_ + train_.labels[rows[index]]];
    }
    for (std::size_t leaf = 0; leaf < leaves.size(); ++leaf) {
      offset[leaf + 1] += offset[leaf];
    }
    std::vector<std::uint32_t> sorted(count);
    for (std::size_t index = 0; index < count; ++index) {
      sorted[offset[leafOf[index]]++] = rows[index];
    }
    std::copy(sorted.begin(), sorted.end(), rows);
    for (std::size_t leaf = 0; leaf < leaves.size(); ++leaf) {
      setCounts(*leaves[leaf], leafCounts.data() + leaf * classCount_);
    }
    sumCounts(node);
  }

  static void collectLeaves(Node *node, std::vector<Node *> &leaves) {
    if (node->isLeaf()) {
      leaves.push_back(node);
      return;
    }
    collectLeaves(node->left.get(), leaves);
    collectLeaves(node->right.get(), leaves);
  }

  // Like Node::setCounts, but ties keep the current class (prune.c starts its
  // search for the best class at T->Leaf).
  void setCounts(Node &node, const std::uint32_t *counts) const {
    node.classCounts.assign(counts, counts + classCount_);
    node.count = 0;
    std::uint16_t best = node.label;
    for (std::size_t k = 0; k < classCount_; ++k) {
      node.count += counts[k];
      if (counts[k] > counts[best]) {
        best = static_cast<std::uint16_t>(k);
      }
    }
    node.label = best;
  }

  void sumCounts(Node *node) const {
    if (node->isLeaf()) {
      return;
    }
    sumCounts(node->left.get());
    sumCounts(node->right.get());
    std::vector<std::uint32_t> counts(classCount_);
    for (std::size_t k = 0; k < classCount_; ++k) {
      counts[k] = node->left->classCounts[k] + node->right->classCounts[k];
    }
    setCounts(*node, counts.data());
  }

  // EstimateErrors(T, rows, UpdateTree = true). `rows` is the node's slice,
  // arranged as above; its counts are already up to date.
  double prune(std::unique_ptr<Node> &slot, std::uint32_t *rows, std::size_t count) {
    Node &node = *slot;
    const double leafEstimate =
        estimator_.leafErrors(node.classCounts.data(), classCount_, node.label);
    if (node.isLeaf()) {
      node.errors = leafEstimate;
      return node.errors;
    }

    // Branches without cases are skipped, as in c4.5.
    const std::size_t leftCount = node.left->count;
    double branchErrors[2] = {0.0, 0.0};
    auto pruneBranch = [&](std::size_t side) {
      if (side == 0 && leftCount > 0) {
        branchErrors[0] = prune(node.left, rows, leftCount);
      } else if (side == 1 && leftCount < count) {
        branchErrors[1] = prune(node.right, rows + leftCount, count - leftCount);
      }
    };
    if (pool_ && count >= kParallelRows) {
      pool_->parallelFor(2, pruneBranch);
    } else {
      pruneBranch(0);
      pruneBranch(1);
    }
    const double treeErrors = branchErrors[0] + branchErrors[1];

    // The branch with most cases (ties: the later branch, as in c4.5).
    const bool rightIsLargest = count - leftCount >= leftCount;
    double largestBranchErrors = std::numeric_limits<double>::infinity();
    if (raising_) {
      // Errors of the largest branch if it also received the other branch's
      // cases (EstimateErrors with UpdateTree = false).
      largestBranchErrors =
          rightIsLargest ? errorsWithExtraRows(node.right.get(), rows, leftCount)
                         : errorsWithExtraRows(node.left.get(), rows + leftCount,
                                               count - leftCount);
    }

    if (leafEstimate <= largestBranchErrors + 0.1 && leafEstimate <= treeErrors + 0.1) {
      node.makeLeaf();
      node.errors = leafEstimate;
      return node.errors;
    }
    if (raising_ && largestBranchErrors <= treeErrors + 0.1) {
      // Subtree raising: the largest branch replaces this node and is pruned
      // again with all of this node's cases.
      std::unique_ptr<Node> branch = std::move(rightIsLargest ? node.right : node.left);
      arrange(branch.get(), rows, count);
      prune(branch, rows, count);
      slot = std::move(branch);
      return slot->errors;
    }
    node.errors = treeErrors;
    return node.errors;
  }

  // Estimated errors of `node`'s subtree (as currently pruned) if the `extra`
  // rows were sent through it in addition to the rows it already holds
  // (EstimateErrors with UpdateTree = false). That is the sum of the leaf
  // estimates, so it equals the stored estimate of the subtree plus, for each
  // leaf that receives extra rows, the change of that leaf's estimate.
  double errorsWithExtraRows(Node *node, const std::uint32_t *extra, std::size_t count) const {
    if (count == 0) {
      return node->errors;
    }
    // Class counts of the extra rows per leaf. A leaf's slot is kept in its
    // scratch index (validated against `leaves`, as other code reuses it).
    std::vector<Node *> arrival(count);
    auto route = [&](std::size_t chunk) {
      const std::size_t begin = count * chunk / 64;
      const std::size_t end = count * (chunk + 1) / 64;
      for (std::size_t index = begin; index < end; ++index) {
        const float *x = features(extra[index]);
        Node *at = node;
        while (!at->isLeaf()) {
          at = goesLeft(*at, x) ? at->left.get() : at->right.get();
        }
        arrival[index] = at;
      }
    };
    if (pool_ && count >= kParallelRows) {
      pool_->parallelFor(64, route);
    } else {
      for (std::size_t chunk = 0; chunk < 64; ++chunk) {
        route(chunk);
      }
    }
    std::vector<const Node *> leaves;
    std::vector<std::uint32_t> extraCounts;
    for (std::size_t index = 0; index < count; ++index) {
      Node *at = arrival[index];
      if (at->index >= leaves.size() || leaves[at->index] != at) {
        at->index = static_cast<std::uint32_t>(leaves.size());
        leaves.push_back(at);
        extraCounts.resize(extraCounts.size() + classCount_, 0);
      }
      ++extraCounts[std::size_t{at->index} * classCount_ + train_.labels[extra[index]]];
    }
    double errors = node->errors;
    std::vector<std::uint32_t> counts(classCount_);
    for (std::size_t slot = 0; slot < leaves.size(); ++slot) {
      const Node *leaf = leaves[slot];
      for (std::size_t k = 0; k < classCount_; ++k) {
        counts[k] = leaf->classCounts[k] + extraCounts[slot * classCount_ + k];
      }
      errors += estimator_.leafErrors(counts.data(), classCount_, leaf->label) -
                (leaf->count > 0 ? leaf->errors : 0.0);
    }
    return errors;
  }

  const Dataset &train_;
  const float *rowMajor_;
  std::size_t features_;
  ErrorEstimator estimator_;
  bool raising_;
  ThreadPool *pool_;
  std::size_t classCount_;
};

} // namespace

void c45CollapseUselessSplits(Node *root) {
  if (root) {
    collapse(root);
  }
}

void c45UseTrainingValueThresholds(Node *root, const Dataset &train,
                                   std::vector<std::vector<float>> &distinctValues) {
  if (distinctValues.empty()) {
    distinctValues.resize(train.featureCount());
    for (std::size_t feature = 0; feature < train.featureCount(); ++feature) {
      std::vector<float> values(train.column(feature), train.column(feature) + train.rowCount);
      for (float &value : values) {
        value += 0.0f; // -0.0 -> 0.0
      }
      std::sort(values.begin(), values.end());
      values.erase(std::unique(values.begin(), values.end()), values.end());
      distinctValues[feature] = std::move(values);
    }
  }
  std::vector<std::vector<Node *>> byFeature(train.featureCount());
  collectDecisionNodes(root, byFeature);
  for (std::size_t feature = 0; feature < byFeature.size(); ++feature) {
    const std::vector<float> &values = distinctValues[feature];
    for (Node *node : byFeature[feature]) {
      // Largest value <= threshold. The value just left of the cut always
      // qualifies, so the search never runs off the front.
      const auto above = std::upper_bound(values.begin(), values.end(), node->threshold,
                                          [](double threshold, float value) {
                                            return threshold < static_cast<double>(value);
                                          });
      node->threshold = static_cast<double>(*(above - 1));
    }
  }
}

void c45PessimisticPrune(std::unique_ptr<Node> &root, const Dataset &train,
                         double confidenceFactor, bool subtreeRaising, ThreadPool *pool) {
  if (!root) {
    return;
  }
  const std::vector<float> rowMajor = rowMajorFeatures(train, pool);
  C45Pruner pruner(train, rowMajor.data(), confidenceFactor, subtreeRaising, pool);
  pruner.run(root);
}

// =============================================================================
// CART
// =============================================================================

namespace {

// Returns the minimal cost (errors + penalty * leaves) of the subtree.
double costComplexityPrune(Node *node, double penalty) {
  const double leafCost = trainingErrors(node) + penalty;
  if (node->isLeaf()) {
    return leafCost;
  }
  const double subtreeCost = costComplexityPrune(node->left.get(), penalty) +
                             costComplexityPrune(node->right.get(), penalty);
  // "<=": T(alpha) is the *smallest* minimising subtree.
  if (leafCost <= subtreeCost + 1e-9 * std::max(1.0, subtreeCost)) {
    node->makeLeaf();
    return leafCost;
  }
  return subtreeCost;
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

// Fills node->errors with the penalty at which the node becomes a leaf.
std::vector<Segment> pruningEnvelope(Node *node) {
  const double leafErrors = trainingErrors(node);
  if (node->isLeaf()) {
    node->errors = INFINITY;
    return {{0.0, 1.0, leafErrors}};
  }
  std::vector<Segment> children =
      addEnvelopes(pruningEnvelope(node->left.get()), pruningEnvelope(node->right.get()));

  // Children cost grows faster (slope >= 2) than the leaf cost (slope 1) and
  // starts no higher, so they cross exactly once: that is the collapse point.
  std::vector<Segment> result;
  double collapseAt = 0.0;
  for (std::size_t index = 0; index < children.size(); ++index) {
    const Segment &segment = children[index];
    const double end = index + 1 < children.size() ? children[index + 1].start : INFINITY;
    const double crossing = (leafErrors - segment.intercept) / (segment.slope - 1.0);
    if (crossing < end) {
      collapseAt = std::max(segment.start, crossing);
      if (collapseAt > segment.start) {
        result.push_back(segment);
      }
      break;
    }
    result.push_back(segment);
  }
  result.push_back({collapseAt, 1.0, leafErrors});
  node->errors = collapseAt;
  return result;
}

void collectSequence(const Node *node, double ancestorMin, std::vector<double> &alphas) {
  if (node->isLeaf()) {
    return;
  }
  if (node->errors < ancestorMin) {
    alphas.push_back(node->errors);
  }
  const double below = std::min(ancestorMin, node->errors);
  collectSequence(node->left.get(), below, alphas);
  collectSequence(node->right.get(), below, alphas);
}

void scaleCollapse(Node *node, double scale) {
  if (node->isLeaf()) {
    return;
  }
  node->errors *= scale;
  scaleCollapse(node->left.get(), scale);
  scaleCollapse(node->right.get(), scale);
}

} // namespace

void cartCostComplexityPrune(Node *root, double alpha, std::size_t totalRows) {
  if (root) {
    costComplexityPrune(root, alpha * static_cast<double>(totalRows));
  }
}

std::vector<double> cartPruningSequence(Node *root, std::size_t totalRows) {
  if (!root) {
    return {};
  }
  pruningEnvelope(root);
  // Penalty per leaf in errors -> alpha in misclassification rate.
  scaleCollapse(root, 1.0 / static_cast<double>(totalRows));

  std::vector<double> alphas{0.0};
  collectSequence(root, INFINITY, alphas);
  std::sort(alphas.begin(), alphas.end());
  std::vector<double> distinct;
  for (double alpha : alphas) {
    if (distinct.empty() || alpha > distinct.back() * (1 + 1e-12) + 1e-15) {
      distinct.push_back(alpha);
    }
  }
  return distinct;
}
