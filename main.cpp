#include "c45_tree.h"
#include "dataset.h"
#include "tree_visualization.h"

#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void runPredictionChecks(const C45Tree &tree, const Dataset &dataset) {
  std::cout << "Summary:\n";

  std::size_t correctPredictions = 0;
  for (std::size_t index = 0; index < dataset.samples.size(); ++index) {
    const Sample &sample = dataset.samples[index];
    const std::string prediction = tree.predict(sample);
    if (prediction == sample.label) {
      ++correctPredictions;
    }
  }

  const double accuracy = static_cast<double>(correctPredictions) /
                          static_cast<double>(dataset.samples.size());

  std::cout << "  checked samples = " << dataset.samples.size() << '\n';
  std::cout << "  correct predictions = " << correctPredictions << '\n';
  std::cout << "  tree depth = " << tree.treeDepth() << '\n';
  std::cout << "  node count = " << tree.nodeCount() << '\n';
  std::cout << "  accuracy = " << std::fixed << std::setprecision(4)
            << accuracy * 100.0 << "%\n";
  std::cout << "  build time = " << std::fixed << std::setprecision(4)
            << tree.buildTimeSeconds() * 1000.0 << " ms\n";
  std::cout << "  prune time = " << std::fixed << std::setprecision(4)
            << tree.pruneTimeSeconds() * 1000.0 << " ms\n";
}

} // namespace

int main() {
  try {
    // 1. Load the dataset from disk.
    const Dataset dataset = loadDataset("datasets/covertype_100x_smaller.csv");
    printDatasetSummary(dataset);

    // 2. Train the decision tree.
    C45Tree tree;
    TrainingOptions options;

    options.maxDepth = 6;
    options.parallelMode = ParallelMode::VDTa;
    options.minFeaturesToParallelize = 4;
    options.minRowsToParallelize = 16;
    options.maxThreadCount = 28;

    // --- CART Configuration ---
    // options.impurityMeasure = ImpurityMeasure::Gini;
    // options.splitSelectionMode = SplitSelectionMode::MaxGain;
    // options.pruningMode = PruningMode::CostComplexity;
    options.ccpAlpha = 2000;

    // --- C.45 Configuration ---
    options.impurityMeasure = ImpurityMeasure::Entropy;
    options.splitSelectionMode = SplitSelectionMode::MeanGainFiltered;
    options.pruningMode = PruningMode::PessimisticError;
    options.pruningConfidenceFactor = 0.0005;

    // options.minSamplesPerLeaf = 1;

    tree.fit(dataset, options);

    // 3. Print tree
    std::cout << "Learned tree:\n";
    tree.print(std::cout);
    std::cout << '\n';

    generateTreeSvg(tree, "tree.svg", options, dataset);

    // 4. Check accuarcy
    runPredictionChecks(tree, dataset);
  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }

  return 0;
}
