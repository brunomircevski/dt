#include "dataset.h"
#include "tree_parallel.h"
#include "tree_serial.h"
#include "tree_visualization.h"

#include <iostream>

int main() {
  try {
    const Dataset dataset = loadDataset("datasets/covertype.csv");
    printDatasetSummary(dataset);

    TrainingOptions options;

    options.maxDepth = 5;
    options.minFeaturesToParallelize = 4;
    options.minRowsToParallelize = 16;
    options.maxThreadCount = 28;

    // --- CART Configuration ---
    options.impurityMeasure = ImpurityMeasure::Gini;
    options.splitSelectionMode = SplitSelectionMode::MaxGain;
    options.pruningMode = PruningMode::CostComplexity;
    options.ccpAlpha = 1000;

    // --- C4.5 Configuration ---
    // options.impurityMeasure = ImpurityMeasure::Entropy;
    // options.splitSelectionMode = SplitSelectionMode::MeanGainFiltered;
    // options.pruningMode = PruningMode::PessimisticError;
    // options.pruningConfidenceFactor = 0.0005;

    // Switch backend: TreeSerial, TreeParallel, or TreeCuda.
    TreeParallel tree;

    tree.fit(dataset, options);

    std::cout << "Learned tree:\n";
    std::cout << '\n';

    generateTreeSvg(tree, "tree.svg", options, dataset);
    printSummary(tree, dataset);

  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }

  return 0;
}
