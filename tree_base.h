#pragma once

#include "dataset.h"
#include "node.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

// --- Algorithm options (CART vs C4.5) ---
//
// CART:  ImpurityMeasure::Gini, SplitSelectionMode::MaxGain,
//        PruningMode::CostComplexity
// C4.5: ImpurityMeasure::Entropy, SplitSelectionMode::MeanGainFiltered,
//        PruningMode::PessimisticError

enum class ImpurityMeasure {
  Entropy,
  Gini
};

enum class SplitSelectionMode {
  MeanGainFiltered,
  MaxGain
};

enum class PruningMode {
  None,
  PessimisticError,
  CostComplexity
};

struct TrainingOptions {
  int maxDepth = -1;
  std::size_t minSamplesToSplit = 2;
  std::size_t minSamplesPerLeaf = 1;

  PruningMode pruningMode = PruningMode::None;
  SplitSelectionMode splitSelectionMode = SplitSelectionMode::MeanGainFiltered;
  double epsilon = 1e-9;
  double pruningConfidenceFactor = 0.25;
  double ccpAlpha = 0.5;
  ImpurityMeasure impurityMeasure = ImpurityMeasure::Entropy;

  int maxFeatureThreadCount = 4;
  int maxNodeThreadCount = 4;
  std::size_t minFeaturesToParallelize = 4;
  std::size_t minRowsToParallelize = 32;
};

class TreeBase {
public:
  virtual ~TreeBase() = default;

  virtual void fit(const Dataset &dataset,
                   const TrainingOptions &options = TrainingOptions{}) = 0;

  virtual void prune(const TrainingOptions &options = TrainingOptions{});
  virtual std::string predict(const Sample &sample) const;
  virtual void print(std::ostream &output) const;
  virtual int treeDepth() const;
  virtual std::size_t nodeCount() const;
  virtual double buildTimeSeconds() const;
  virtual double pruneTimeSeconds() const;

protected:
  struct PartitionedRows {
    std::vector<std::size_t> leftRows;
    std::vector<std::size_t> rightRows;
  };

  struct SortedFeatureRow {
    std::size_t rowIndex = 0;
    double value = 0.0;
    std::uint16_t classId = 0;
  };

  struct SortedFeatureView {
    std::size_t featureIndex = 0;
    std::vector<SortedFeatureRow> rows;
    std::size_t totalRows = 0;
  };

  struct SplitResult {
    bool valid = false;
    std::size_t featureIndex = 0;
    std::string featureName;
    double threshold = 0.0;
    double informationGain = 0.0;
    double splitInformation = 0.0;
    double gainRatio = 0.0;
    std::size_t leftRowCount = 0;
  };

  struct SplitSearchResult {
    SplitResult split;
    SortedFeatureView view;
    bool hasView = false;
  };

  struct NodeExpandResult {
    std::unique_ptr<Node> node;
    PartitionedRows partitions;
  };

  using BuildTimePoint = std::chrono::steady_clock::time_point;

  const Dataset *dataset_ = nullptr;
  std::unique_ptr<Node> root_;
  TrainingOptions options_;
  double buildTimeSeconds_ = 0.0;
  double pruneTimeSeconds_ = 0.0;

  std::vector<std::string> classLabels_;
  std::unordered_map<std::string, std::uint16_t> labelToClassId_;
  std::uint16_t numClasses_ = 0;

  void prepareFit(const Dataset &dataset, const TrainingOptions &options);
  BuildTimePoint startBuildTimer() const;
  void finishBuildTimer(BuildTimePoint start);
  void finalizeFit(const TrainingOptions &options);

  std::vector<std::size_t> makeRootRowIndices() const;

  void buildClassMapping(const Dataset &dataset);
  std::uint16_t classIdForRow(std::size_t rowIndex) const;
  std::vector<std::uint32_t> computeClassCountsArray(
      const std::vector<std::size_t> &rowIndices) const;
  double impurityFromCounts(const std::vector<std::uint32_t> &counts,
                            std::size_t total) const;
  SplitResult scoreCandidateFromCounts(
      const SortedFeatureView &view,
      const std::vector<std::uint32_t> &totalCounts, double parentImpurity,
      const std::vector<std::uint32_t> &leftCounts,
      const std::vector<std::uint32_t> &rightCounts, std::size_t leftSize,
      double threshold) const;
  SortedFeatureView buildSortedFeatureView(
      const std::vector<std::size_t> &rowIndices, std::size_t featureIndex) const;
  SplitResult scoreAllThresholdsForFeature(const SortedFeatureView &view) const;
  PartitionedRows partitionFromSortedView(const SortedFeatureView &view,
                                          std::size_t leftSize) const;

  SplitSearchResult evaluateFeatureSplit(
      const std::vector<std::size_t> &rowIndices, std::size_t featureIndex) const;
  SplitSearchResult reduceBestSplitSearch(
      std::vector<SplitSearchResult> featureResults,
      const std::vector<std::size_t> &rowIndices) const;

  virtual SplitSearchResult findBestSplitAtNode(
      const std::vector<std::size_t> &rowIndices) const = 0;
  NodeExpandResult expandOneNode(const std::vector<std::size_t> &rowIndices,
                                 int depth) const;
  std::unique_ptr<Node> buildNode(const std::vector<std::size_t> &rowIndices,
                                  int depth) const;

  bool isSplitRejected(const SplitResult &split) const;
  SplitResult chooseBestSplit(const std::vector<SplitResult> &candidates) const;
  bool shouldStopGrowing(const std::vector<std::size_t> &rowIndices, int depth) const;
  PartitionedRows partitionRows(const std::vector<std::size_t> &rowIndices,
                                std::size_t featureIndex, double threshold) const;
  std::size_t countCorrectPredictions(const Node *node,
                                      const std::vector<std::size_t> &rowIndices) const;
  std::size_t countLeafNodes(const Node *node) const;
  std::size_t countNodes(const Node *node) const;
  int computeTreeDepth(const Node *node) const;
  bool allSameLabel(const std::vector<std::size_t> &rowIndices) const;
  std::string getMajorityLabel(const std::vector<std::size_t> &rowIndices) const;
  void printNode(const Node *node, std::ostream &output, int depth,
                 const std::string &edgeText) const;

  double entropy(const std::vector<std::size_t> &rowIndices) const;
  double giniIndex(const std::vector<std::size_t> &rowIndices) const;
  double impurity(const std::vector<std::size_t> &rowIndices) const;
  double informationGain(const std::vector<std::size_t> &rowIndices,
                         const std::vector<std::vector<std::size_t>> &partitions) const;
  double splitInformation(const std::vector<std::size_t> &rowIndices,
                          const std::vector<std::vector<std::size_t>> &partitions) const;

  static bool areEffectivelyEqual(double left, double right, double epsilon);
  static bool isBetterC45(const SplitResult &lhs, const SplitResult &rhs,
                          double epsilon);
  static bool isBetterMaxGain(const SplitResult &lhs, const SplitResult &rhs,
                              double epsilon);

  friend void pruneTree(TreeBase &tree, const TrainingOptions &options);
};
