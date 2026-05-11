#include "c45_tree.h"
#include "dataset.h"

#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void printDatasetSummary(const Dataset& dataset) {
    std::cout << "Loaded samples: " << dataset.samples.size() << '\n';
    std::cout << "Features:";
    for (const std::string& featureName : dataset.featureNames) {
        std::cout << ' ' << featureName;
    }
    std::cout << "\n\n";

    if (!dataset.samples.empty()) {
        const Sample& firstSample = dataset.samples.front();
        std::cout << "First sample: ";
        for (std::size_t index = 0; index < firstSample.features.size(); ++index) {
            std::cout << dataset.featureNames[index] << '=' << firstSample.features[index] << ' ';
        }
        std::cout << "label=" << firstSample.label << "\n\n";
    }
}

void runPredictionChecks(const C45Tree& tree, const Dataset& dataset) {
    std::cout << "Summary:\n";

    std::size_t correctPredictions = 0;
    for (std::size_t index = 0; index < dataset.samples.size(); ++index) {
        const Sample& sample = dataset.samples[index];
        const std::string prediction = tree.predict(sample);
        if (prediction == sample.label) {
            ++correctPredictions;
        }
    }

    const double accuracy = static_cast<double>(correctPredictions)
        / static_cast<double>(dataset.samples.size());

    std::cout << "  checked samples = " << dataset.samples.size() << '\n';
    std::cout << "  correct predictions = " << correctPredictions << '\n';
    std::cout << "  tree depth = " << tree.treeDepth() << '\n';
    std::cout << "  node count = " << tree.nodeCount() << '\n';
    std::cout << "  accuracy = "
              << std::fixed << std::setprecision(4)
              << accuracy * 100.0 << "%\n";
}

}  // namespace

int main() {
    try {
        // 1. Load the dataset from disk.
        const Dataset dataset = loadDataset("datasets/diabetes.csv");
        printDatasetSummary(dataset);

        // 2. Train the decision tree.
        C45Tree tree;
        TrainingOptions options;
        options.impurityMeasure = ImpurityMeasure::Entropy;
        //options.minSamplesPerLeaf = 10;
        options.pruningMode = PruningMode::PessimisticErrorPrune;
        options.splitSelectionMode = SplitSelectionMode::MeanGainFiltered;

        tree.fit(dataset, options);

        // 3. Print tree
        std::cout << "Learned tree:\n";
        tree.print(std::cout);
        std::cout << '\n';

        // 4. Check accuarcy
        runPredictionChecks(tree, dataset);
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
