#pragma once

#include "dataset.h"
#include "node.h"

#include <map>
#include <memory>
#include <limits>
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
    // Closest simple numeric-only version of classic C4.5:
    // Dodatni informationGain i największy gainRatio
    ClassicC45,
    // Liczy średni informationGain, odrzuca te, poniżej średniej, wybiera największy gainRatio.
    MeanGainFiltered
};

enum class PruningMode {
    // Bez przycinania: zostawia całe drzewo jak urosło.
    None,
    // Zamienia poddrzewo na liść klasy większościowej,
    // jeśli taki liść ma na treningu nie gorszą trafność niż poddrzewo.
    TrainingAccuracyPrune,
    // Zamienia poddrzewo na liść klasy większościowej,
    // jeśli pessymistyczny błąd liścia nie jest gorszy niż poddrzewa.
    PessimisticErrorPrune,
    // Pessymistyczne przycinanie bardziej w stylu C4.5:
    // porównuje liść vs sumę błędów wszystkich liści w poddrzewie.
    C45PessimisticPrune
};

struct TrainingOptions {
    // Maximum recursion depth of the tree.
    // Use -1 to mean "no fixed depth limit".
    //
    // In classic C4.5 a hard depth cap is not the main control knob,
    // so the learning-friendly default is unlimited depth.
    int maxDepth = -1;

    // A node must contain at least this many samples before we even try
    // to split it into children.
    std::size_t minSamplesToSplit = 2;

    // Each child created by a split must contain at least this many samples.
    // This prevents very tiny leaves supported by only one or two rows.
    std::size_t minSamplesPerLeaf = 1;

    // Selects whether the finished tree should be simplified after training.
    // The enum above describes how each pruning method makes that decision.
    PruningMode pruningMode = PruningMode::None;

    // Controls how we choose among valid numeric split candidates.
    // The default is the simpler, more faithful C4.5-style rule.
    SplitSelectionMode splitSelectionMode = SplitSelectionMode::ClassicC45;

    // Small tolerance used when comparing floating-point values.
    // It helps us avoid creating fake thresholds between almost equal numbers.
    double epsilon = 1e-9;

    // Confidence factor used by the more C4.5-like pessimistic pruning mode.
    // Smaller values make the pruning estimate more conservative.
    double pruningConfidenceFactor = 0.25;

    // Entropy is the C4.5 default and should stay the default here.
    // Gini is available only as a non-C4.5 comparison mode.
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
    int treeDepth() const;
    std::size_t nodeCount() const;

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
    struct PartitionedRows {
        std::vector<std::size_t> leftRows;
        std::vector<std::size_t> rightRows;
    };

    // We only keep a pointer to the dataset. The tree does not own the dataset.
    // "const" means the tree promises not to modify it.
    const Dataset* dataset_ = nullptr;

    // root_ points to the first node in the tree.
    std::unique_ptr<Node> root_;

    // The last training call stores its configuration here so all helper
    // functions can use the same settings while the tree is being built.
    TrainingOptions options_;

    std::map<std::string, int> computeClassCounts(
        const std::vector<std::size_t>& rowIndices
    ) const;
    std::unique_ptr<Node> buildNode(const std::vector<std::size_t>& rowIndices, int depth) const;
    std::vector<double> collectNumericThresholdCandidates(
        const std::vector<std::size_t>& rowIndices,
        std::size_t featureIndex
    ) const;
    SplitResult scoreSplit(
        const std::vector<std::size_t>& rowIndices,
        std::size_t featureIndex,
        double threshold
    ) const;
    SplitResult chooseBestSplit(const std::vector<SplitResult>& candidates) const;
    bool shouldStopGrowing(const std::vector<std::size_t>& rowIndices, int depth) const;
    PartitionedRows partitionRows(
        const std::vector<std::size_t>& rowIndices,
        std::size_t featureIndex,
        double threshold
    ) const;
    void applySelectedPruning(const std::vector<std::size_t>& rowIndices);
    void pruneWithTrainingAccuracy(
        std::unique_ptr<Node>& node,
        const std::vector<std::size_t>& rowIndices
    );
    void pruneWithPessimisticError(
        std::unique_ptr<Node>& node,
        const std::vector<std::size_t>& rowIndices
    );
    void pruneWithC45Pessimistic(
        std::unique_ptr<Node>& node,
        const std::vector<std::size_t>& rowIndices
    );
    double estimatePessimisticLeafErrorCount(
        std::size_t observedErrors,
        std::size_t sampleCount
    ) const;
    double estimateSubtreePessimisticErrorCount(
        const Node* node,
        const std::vector<std::size_t>& rowIndices
    ) const;
    std::size_t countCorrectPredictions(
        const Node* node,
        const std::vector<std::size_t>& rowIndices
    ) const;
    std::size_t countLeafNodes(const Node* node) const;
    std::size_t countNodes(const Node* node) const;
    int computeTreeDepth(const Node* node) const;
    bool allSameLabel(const std::vector<std::size_t>& rowIndices) const;
    std::string getMajorityLabel(const std::vector<std::size_t>& rowIndices) const;
    void printNode(const Node* node, std::ostream& output, int depth, const std::string& edgeText) const;
};
