#pragma once

#include "dataset.h"
#include "node.h"

#include <memory>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

enum class ImpurityMeasure {
    Entropy,
    Gini
};

enum class PruningMode {
    // Keep every branch created during tree growth.
    None,
    // Replace a subtree with one majority-class leaf when the leaf is at least
    // as accurate on the training samples that reached that subtree.
    TrainingAccuracyPrune,
    // Replace a subtree with one majority-class leaf when the leaf has
    // no worse pessimistic error estimate than the whole subtree.
    PessimisticErrorPrune
};

struct TrainingOptions {
    // Maximum recursion depth of the tree.
    // Larger values can fit training data more closely, but may overfit.
    int maxDepth = 10;

    // A node must contain at least this many samples before we even try
    // to split it into children.
    std::size_t minSamplesToSplit = 2;

    // Each child created by a split must contain at least this many samples.
    // This prevents very tiny leaves supported by only one or two rows.
    std::size_t minSamplesPerLeaf = 1;

    // Selects whether the finished tree should be simplified after training.
    // The enum above describes how each pruning method makes that decision.
    PruningMode pruningMode = PruningMode::None;

    // Small tolerance used when comparing floating-point values.
    // It helps us avoid creating fake thresholds between almost equal numbers.
    double epsilon = 1e-9;

    // Entropy and Gini are two common ways to measure how mixed a node is.
    // Lower impurity means a "cleaner" group of samples.
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
    void fit(const Dataset& dataset, const TrainingOptions& options = TrainingOptions{});
    std::string predict(const Sample& sample) const;
    void print(std::ostream& output) const;

    // The next functions are public so they can be explored from main()
    // and studied separately.
    double entropy(const std::vector<std::size_t>& rowIndices) const;
    double giniIndex(const std::vector<std::size_t>& rowIndices) const;
    double impurity(const std::vector<std::size_t>& rowIndices) const;
    double informationGain(
        const std::vector<std::size_t>& rowIndices,
        const std::vector<std::vector<std::size_t>>& partitions
    ) const;
    double splitInformation(
        const std::vector<std::size_t>& rowIndices,
        const std::vector<std::vector<std::size_t>>& partitions
    ) const;
    SplitResult findBestSplit(const std::vector<std::size_t>& rowIndices) const;

private:
    // We only keep a pointer to the dataset. The tree does not own the dataset.
    // "const" means the tree promises not to modify it.
    const Dataset* dataset_ = nullptr;

    // root_ points to the first node in the tree.
    std::unique_ptr<Node> root_;

    // The last training call stores its configuration here so all helper
    // functions can use the same settings while the tree is being built.
    TrainingOptions options_;

    std::unique_ptr<Node> buildNode(const std::vector<std::size_t>& rowIndices, int depth) const;
    void pruneTree(std::unique_ptr<Node>& node, const std::vector<std::size_t>& rowIndices) const;
    std::size_t countCorrectPredictions(
        const Node* node,
        const std::vector<std::size_t>& rowIndices
    ) const;
    std::size_t countLeafNodes(const Node* node) const;
    bool allSameLabel(const std::vector<std::size_t>& rowIndices) const;
    std::string getMajorityLabel(const std::vector<std::size_t>& rowIndices) const;
    void printNode(const Node* node, std::ostream& output, int depth, const std::string& edgeText) const;
};
