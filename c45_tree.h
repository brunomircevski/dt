#pragma once

#include "dataset.h"
#include "node.h"

#include <limits>
#include <map>
#include <memory>
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
  void fit(const Dataset &dataset,
           const TrainingOptions &options = TrainingOptions{});
  std::string predict(const Sample &sample) const;
  void print(std::ostream &output) const;
  int treeDepth() const;
  std::size_t nodeCount() const;

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
  struct PartitionedRows {
    std::vector<std::size_t> leftRows;
    std::vector<std::size_t> rightRows;
  };

  // We only keep a pointer to the dataset. The tree does not own the dataset.
  // "const" means the tree promises not to modify it.
  const Dataset *dataset_ = nullptr;

  // root_ points to the first node in the tree.
  std::unique_ptr<Node> root_;

  // The last training call stores its configuration here so all helper
  // functions can use the same settings while the tree is being built.
  TrainingOptions options_;

  std::map<std::string, int>
  computeClassCounts(const std::vector<std::size_t> &rowIndices) const;
  std::unique_ptr<Node> buildNode(const std::vector<std::size_t> &rowIndices,
                                  int depth) const;
  std::vector<double>
  collectNumericThresholdCandidates(const std::vector<std::size_t> &rowIndices,
                                    std::size_t featureIndex) const;
  SplitResult scoreSplit(const std::vector<std::size_t> &rowIndices,
                         std::size_t featureIndex, double threshold) const;
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
