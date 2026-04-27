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

void runMathChecks(C45Tree& tree, const Dataset& dataset) {
    // Small hand-picked subset used to make the entropy/gain numbers easier
    // to reason about when reading the output.
    std::vector<std::size_t> toyRows = {0, 1, 2, 50, 51, 52};
    const double toyEntropy = tree.entropy(toyRows);

    std::vector<std::vector<std::size_t>> obviousSplit = {
        {0, 1, 2},
        {50, 51, 52}
    };

    const double gain = tree.informationGain(toyRows, obviousSplit);
    const double splitInfo = tree.splitInformation(toyRows, obviousSplit);
    const double gainRatio = splitInfo > 0.0 ? gain / splitInfo : 0.0;

    std::cout << "Math check on a tiny hand-picked subset:\n";
    std::cout << "  entropy = " << std::fixed << std::setprecision(4) << toyEntropy << '\n';
    std::cout << "  information gain = " << gain << '\n';
    std::cout << "  split information = " << splitInfo << '\n';
    std::cout << "  gain ratio = " << gainRatio << "\n\n";

    // Then inspect the best first split on the full dataset.
    std::vector<std::size_t> allRows(dataset.samples.size());
    for (std::size_t index = 0; index < allRows.size(); ++index) {
        allRows[index] = index;
    }

    const SplitResult bestSplit = tree.findBestSplit(allRows);
    if (bestSplit.valid) {
        std::cout << "Best root split found before training:\n";
        std::cout << "  feature = " << bestSplit.featureName << '\n';
        std::cout << "  threshold = " << bestSplit.threshold << '\n';
        std::cout << "  information gain = " << bestSplit.informationGain << '\n';
        std::cout << "  split information = " << bestSplit.splitInformation << '\n';
        std::cout << "  gain ratio = " << bestSplit.gainRatio << "\n\n";
    }
}

void runPredictionChecks(const C45Tree& tree, const Dataset& dataset) {
    std::cout << "Prediction summary:\n";

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
    std::cout << "  accuracy = "
              << std::fixed << std::setprecision(4)
              << accuracy * 100.0 << "%\n";
}

}  // namespace

int main() {
    try {
        // This file is intentionally written like a small learning script.
        // The steps below match the main ideas of the algorithm:
        // 1. read data
        // 2. compute a tree from that data
        // 3. print the learned questions
        // 4. test a few predictions

        // 1. Load the Iris dataset from disk.
        const Dataset dataset = loadIrisDataset("datasets/iris.csv");
        printDatasetSummary(dataset);

        // 2. Train the decision tree.
        //
        // C++ note:
        // "C45Tree tree;" creates an object on the stack.
        // We can call methods on it with tree.fit(...), tree.print(...), etc.
        C45Tree tree;
        tree.fit(dataset);
        // runMathChecks(tree, dataset);

        // 3. Print the learned structure so we can inspect the splits.
        std::cout << "Learned tree:\n";
        tree.print(std::cout);
        std::cout << '\n';

        // 4. Run a few direct predictions as a sanity check.
        runPredictionChecks(tree, dataset);
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
