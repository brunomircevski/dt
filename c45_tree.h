#pragma once

#include "dataset.h"
#include "node.h"
#include "task_executor.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

enum class ImpurityMeasure {
  // Classic C4.5 uses entropy.
  Entropy,
  // Gini is kept only as an experiment/comparison mode.
  Gini
};

enum class SplitSelectionMode {
  // C4.5 variant: filters candidates with below-average gain.
  MeanGainFiltered,
  // CART style: purely maximizes information/Gini gain.
  MaxGain
};

enum class PruningMode {
  None,
  // C4.5 pessimistic pruning: compares a replacement leaf's estimated error
  // with the sum of estimated errors of all leaves in the subtree (uses pruningConfidenceFactor).
  PessimisticError,
  // CART: Minimal Cost-Complexity Pruning (uses ccpAlpha).
  CostComplexity
};

enum class GleamsMode {
  Serial,  // recursive buildNode + sequential findBestSplit
  VDa,     // async per-feature sort/score pipeline in findBestSplit
  Ta,      // async node expansion (no workers blocking on children)
  VDTa     // both VD and Ta (separate executors to avoid deadlock)
};

struct TrainingOptions {
  // TYPICAL ALGORITHM CONFIGURATIONS:
  // CART: impurityMeasure=Gini, splitSelectionMode=MaxGain, pruningMode=CostComplexity
  // C4.5: impurityMeasure=Entropy, splitSelectionMode=MeanGainFiltered, pruningMode=PessimisticError

  // Maximum depth of the tree. Prevents the model from becoming overly complex.
  int maxDepth = -1;

  // Minimum number of samples in a node to attempt a split.
  // Prevents splitting small groups that may represent noise.
  std::size_t minSamplesToSplit = 2;

  // Minimum number of samples required in each new leaf after a split.
  // Guarantees statistical significance of the resulting leaves.
  std::size_t minSamplesPerLeaf = 1;

  // Selects the method for simplifying (pruning) the tree after the growing phase.
  PruningMode pruningMode = PruningMode::None;

  // Criterion for selecting the best split among candidates.
  SplitSelectionMode splitSelectionMode = SplitSelectionMode::MeanGainFiltered;

  // Tolerance for floating-point comparisons (double).
  // Prevents creating artificial thresholds for nearly identical values.
  double epsilon = 1e-9;

  // Confidence factor for C4.5 pessimistic pruning (PEP).
  // Smaller values result in stronger, more aggressive pruning.
  double pruningConfidenceFactor = 0.03;

  // Complexity parameter for Cost-Complexity Pruning (CART).
  // Penalty for each additional leaf; larger values lead to smaller trees (e.g., 0.01).
  double ccpAlpha = 0.5;

  // Impurity measure method (Entropy for C4.5, Gini for CART).
  // Used to evaluate split quality (e.g., ImpurityMeasure::Gini).
  ImpurityMeasure impurityMeasure = ImpurityMeasure::Entropy;

  // GLEAMS parallelism mode (see wielowątkowość.md).
  GleamsMode gleamsMode = GleamsMode::Serial;

  // Thread count for vdExecutor and taExecutor (when gleamsMode uses them).
  int maxThreadCount = 1;

  // VD pipeline only runs when the dataset has at least this many features.
  std::size_t minFeaturesToParallelize = 4;

  // Ta/VDTa async node tasks only when a node has at least this many rows.
  std::size_t minRowsToParallelize = 32;
};

struct SplitResult {
  // If false, we did not find any useful split.
  bool valid = false;

  // Which feature gave the best split.
  std::size_t featureIndex = 0;
  std::string featureName;

  // Threshold for the question:
  // "Is feature <= threshold?"
  double threshold = 0.0;

  // These values are stored mainly for learning/debugging.
  // They let us print how good the chosen split was.
  double informationGain = 0.0;
  double splitInformation = 0.0;
  double gainRatio = 0.0;
};

class C45Tree {
public:
  C45Tree();
  ~C45Tree();

  C45Tree(const C45Tree &) = delete;
  C45Tree &operator=(const C45Tree &) = delete;

  void fit(const Dataset &dataset,
           const TrainingOptions &options = TrainingOptions{});
  std::string predict(const Sample &sample) const;
  void print(std::ostream &output) const;
  int treeDepth() const;
  std::size_t nodeCount() const;
  double buildTimeSeconds() const;
  double pruneTimeSeconds() const;

  // The next functions are public so they can be explored from main()
  // and studied separately.
  double entropy(const std::vector<std::size_t> &rowIndices) const;
  double giniIndex(const std::vector<std::size_t> &rowIndices) const;
  double impurity(const std::vector<std::size_t> &rowIndices) const;
  double informationGain(
      const std::vector<std::size_t> &rowIndices,
      const std::vector<std::vector<std::size_t>> &partitions) const;
  double splitInformation(
      const std::vector<std::size_t> &rowIndices,
      const std::vector<std::vector<std::size_t>> &partitions) const;
  SplitResult findBestSplit(const std::vector<std::size_t> &rowIndices) const;

private:
  struct FitContext {
    std::unique_ptr<TaskExecutor> vdExecutor;
    std::unique_ptr<TaskExecutor> taExecutor;
    std::atomic<std::size_t> pendingNodeTasks{0};
    std::mutex completionMutex;
    std::condition_variable completionCv;
  };

  struct PartitionedRows {
    std::vector<std::size_t> leftRows;
    std::vector<std::size_t> rightRows;
  };

  using RowIndexList = std::shared_ptr<const std::vector<std::size_t>>;

  struct NodeBuildJob {
    RowIndexList rowIndices;
    int depth = 0;
    std::function<void(std::unique_ptr<Node>)> onComplete;
  };

  // Result of growing one node: leaf, or decision node + row sets for children.
  struct NodeExpandResult {
    std::unique_ptr<Node> node;
    PartitionedRows partitions;
  };

  // We only keep a pointer to the dataset. The tree does not own the dataset.
  // "const" means the tree promises not to modify it.
  const Dataset *dataset_ = nullptr;

  // root_ points to the first node in the tree.
  std::unique_ptr<Node> root_;

  // The last training call stores its configuration here so all helper
  // functions can use the same settings while the tree is being built.
  TrainingOptions options_;

  double buildTimeSeconds_ = 0.0;
  double pruneTimeSeconds_ = 0.0;

  mutable std::unique_ptr<FitContext> fitContext_;

  bool usesVerticalParallelism() const;
  bool usesAsyncTreeBuilding() const;
  bool shouldParallelizeVD(std::size_t featureCount) const;
  bool shouldParallelizeTa(std::size_t rowCount) const;
  void waitForAllNodeTasks() const;
  void notifyNodeTaskFinished() const;
  std::function<void(std::unique_ptr<Node>)>
  wrapNodeOnComplete(std::function<void(std::unique_ptr<Node>)> userCallback) const;

  SplitResult
  reduceBestPerFeature(const std::vector<SplitResult> &scoredCandidates) const;

  // One training row sorted by a single feature value at the current node.
  struct SortedFeatureRow {
    std::size_t rowIndex = 0;
    double value = 0.0;
    std::string label;
  };

  // Per-feature cache for split search: sort once, score many thresholds cheaply.
  struct SortedFeatureView {
    std::size_t featureIndex = 0;
    std::vector<SortedFeatureRow> rows;  // sorted by value, then label (legacy order)
    std::vector<double> thresholds;      // candidate cut points
    std::vector<std::size_t> leftSizes;  // rows[0..leftSize-1] go left for each threshold
    // prefixClassCounts[i] = class histogram of rows[0..i-1]
    std::vector<std::map<std::string, int>> prefixClassCounts;
    double parentImpurity = 0.0;  // impurity before any split on this feature
    std::size_t totalRows = 0;
  };

  std::map<std::string, int>
  computeClassCounts(const std::vector<std::size_t> &rowIndices) const;
  // Entropy or Gini from a class histogram (no row scan).
  double impurityFromClassCounts(const std::map<std::string, int> &counts,
                                 std::size_t total) const;
  // Sort rows by one feature and precompute thresholds + prefix class counts.
  SortedFeatureView buildSortedFeatureView(
      const std::vector<std::size_t> &rowIndices, std::size_t featureIndex) const;
  // Score one threshold using the sorted view (no partitionRows).
  SplitResult scoreSplitFromSorted(const SortedFeatureView &view,
                                   std::size_t thresholdIndex) const;
  SplitResult scoreAllThresholdsForFeature(const SortedFeatureView &view) const;
  NodeExpandResult expandOneNode(const std::vector<std::size_t> &rowIndices,
                                 int depth) const;
  std::unique_ptr<Node> buildNode(const std::vector<std::size_t> &rowIndices,
                                  int depth) const;
  void expandNodeAsync(NodeBuildJob job) const;
  void scheduleAsyncChildren(std::unique_ptr<Node> parent,
                             PartitionedRows partitions, int depth,
                             std::function<void(std::unique_ptr<Node>)> onComplete) const;
  bool isSplitRejected(const SplitResult &split) const;
  SplitResult chooseBestSplit(const std::vector<SplitResult> &candidates) const;
  bool shouldStopGrowing(const std::vector<std::size_t> &rowIndices,
                         int depth) const;
  PartitionedRows partitionRows(const std::vector<std::size_t> &rowIndices,
                                std::size_t featureIndex,
                                double threshold) const;
  void applySelectedPruning(const std::vector<std::size_t> &rowIndices);
  void prunePessimisticError(std::unique_ptr<Node> &node,
                             const std::vector<std::size_t> &rowIndices);
  void pruneCostComplexity(std::unique_ptr<Node> &node,
                           const std::vector<std::size_t> &rowIndices);
  double estimatePessimisticLeafErrorCount(std::size_t observedErrors,
                                           std::size_t sampleCount) const;
  double estimateSubtreePessimisticErrorCount(
      const Node *node, const std::vector<std::size_t> &rowIndices) const;
  std::size_t
  countCorrectPredictions(const Node *node,
                          const std::vector<std::size_t> &rowIndices) const;
  std::size_t countLeafNodes(const Node *node) const;
  std::size_t countNodes(const Node *node) const;
  int computeTreeDepth(const Node *node) const;
  bool allSameLabel(const std::vector<std::size_t> &rowIndices) const;
  std::string
  getMajorityLabel(const std::vector<std::size_t> &rowIndices) const;
  void printNode(const Node *node, std::ostream &output, int depth,
                 const std::string &edgeText) const;
};
