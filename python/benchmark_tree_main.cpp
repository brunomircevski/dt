// Temporary benchmark entry point (compiled only by benchmark_comparison.py).
#include "c45_tree.h"
#include "dataset.h"

#include <iomanip>
#include <iostream>
#include <string>

namespace {

void printSummary(const C45Tree &tree, const Dataset &dataset) {
  std::size_t correctPredictions = 0;
  for (const Sample &sample : dataset.samples) {
    if (tree.predict(sample) == sample.label) {
      ++correctPredictions;
    }
  }

  const double accuracy = static_cast<double>(correctPredictions) /
                          static_cast<double>(dataset.samples.size());

  std::cout << "Summary:\n";
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

int main(int argc, char *argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "usage: tree_benchmark <cart|c45>\n";
      return 1;
    }

    const std::string mode = argv[1];
    const Dataset dataset = loadDataset("datasets/covertype_10x_smaller.csv");

    C45Tree tree;
    TrainingOptions options;
    options.pruningMode = PruningMode::None;
    options.maxDepth = 5;
    options.minFeaturesToParallelize = 3;
    options.maxThreadCount = 28;

    if (mode == "cart") {
      options.impurityMeasure = ImpurityMeasure::Gini;
      options.splitSelectionMode = SplitSelectionMode::MaxGain;
    } else if (mode == "c45") {
      options.impurityMeasure = ImpurityMeasure::Entropy;
      options.splitSelectionMode = SplitSelectionMode::MeanGainFiltered;
    } else {
      std::cerr << "Unknown mode: " << mode << '\n';
      return 1;
    }

    tree.fit(dataset, options);
    printSummary(tree, dataset);
  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }

  return 0;
}
