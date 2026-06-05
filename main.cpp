#include "dataset.h"
#include "options.h"
#include "tree_visualization.h"

#include <iostream>

int main(int argc, char *argv[]) {
  try {
    Options options;
    options.backend = Backend::Cuda;
    options.datasetPath = "datasets/supersymmetry.csv";
    options.maxDepth = 7;
    options.minFeaturesToParallelize = 4;
    options.minRowsToParallelize = 32;
    options.maxFeatureThreadCount = 20;
    options.maxNodeThreadCount = 8;

    // TreeCuda only (ignored by TreeSerial / TreeParallel).
    // options.cudaRowsPerTile = 16384;
    // options.cudaMaxTilesPerFeature = 256;
    // options.cudaScoreThreadsPerBlock = 256;
    // options.cudaGatherBlockSize = 256;

    // --- CART Configuration ---
    options.impurityMeasure = ImpurityMeasure::Gini;
    options.splitSelectionMode = SplitSelectionMode::MaxGain;
    options.pruningMode = PruningMode::None;
    options.ccpAlpha = 1000;

    // --- C4.5 Configuration ---
    // options.impurityMeasure = ImpurityMeasure::Entropy;
    // options.splitSelectionMode = SplitSelectionMode::MeanGainFiltered;
    // options.pruningMode = PruningMode::PessimisticError;
    // options.pruningConfidenceFactor = 0.0005;

    applyCommandLine(argc, argv, options);

    std::cout << "Backend: " << backendName(options.backend) << "\n";
    std::cout << "Loading dataset: " << options.datasetPath << "\n";
    const Dataset dataset = loadDataset(options.datasetPath);
    printDatasetSummary(dataset);

    auto tree = createTree(options.backend);

    std::cout << "Fitting tree...\n";
    tree->fit(dataset, options);

    generateTreeSvg(*tree, "tree.svg", options, dataset);
    printSummary(*tree, dataset);

  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }

  return 0;
}
