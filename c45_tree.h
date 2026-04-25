#pragma once

#include "dataset.h"
#include "node.h"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

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
    void fit(const Dataset& dataset);
    std::string predict(const Sample& sample) const;
    void print(std::ostream& output) const;

    // The next functions are public so they can be explored from main()
    // and studied separately.
    double entropy(const std::vector<std::size_t>& rowIndices) const;
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

    std::unique_ptr<Node> buildNode(const std::vector<std::size_t>& rowIndices, int depth) const;
    bool allSameLabel(const std::vector<std::size_t>& rowIndices) const;
    void printNode(const Node* node, std::ostream& output, int depth, const std::string& edgeText) const;
};
