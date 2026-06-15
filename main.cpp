#include "dataset.h"
#include "options.h"
#include "timing.h"
#include "tree_visualization.h"

#include <iostream>

int main(int argc, char *argv[]) {
  try {
    double loadSeconds = 0.0;
    Options options;
    options.backend = Backend::Cuda;
    options.datasetPath = "datasets/supersymmetry.csv";
    options.demoDatasetMultiplier = 1;
    options.maxDepth = 30;

    // TreeParallel only.
    options.parallelMinFeaturesToParallelize = 4;
    options.parallelMaxFeatureThreadCount = 20;
    options.parallelMaxNodeThreadCount = 4;

    // TreeParallel + TreeCuda.
    options.minRowsToParallelize = 20; //16-32 optimal

    // TreeCuda only.
    options.cudaCpuThreadCount = 120; // 120 is 10% faster than 20
    options.cudaGpuWorkerCount = 8; // 4-8 optimal, higher needs more cpu threads
    options.cudaMinRowsForGpu = 400; // around 400 is the fastest
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
    if (options.demoDatasetMultiplier > 1) {
      std::cout << "Demo dataset multiply: " << options.demoDatasetMultiplier
                << "x (in-memory)\n";
    }
    Dataset dataset;
    {
      ScopedTimer loadTimer(loadSeconds);
      dataset = loadDataset(options.datasetPath, options.demoDatasetMultiplier);
    }
    printDatasetSummary(dataset);

    auto tree = createTree(options.backend);
    tree->setLoadTimeSeconds(loadSeconds);

    std::cout << "Fitting tree...\n";
    tree->fit(dataset, options);

    //generateTreeSvg(*tree, "tree.svg", options, dataset);
    printSummary(*tree, dataset);

  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }

  return 0;
}
