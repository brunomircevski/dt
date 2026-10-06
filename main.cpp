#include "dataset.h"
#include "gpu_builder.h"
#include "options.h"
#include "timing.h"
#include "trainer.h"
#include "tree.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

std::string shellQuote(const std::string &value) {
  std::string quoted = "'";
  for (char character : value) {
    quoted += character == '\'' ? std::string("'\\''") : std::string(1, character);
  }
  return quoted + "'";
}

void renderSvg(const DecisionTree &tree, const std::string &svgPath) {
  const std::string textPath = svgPath + ".txt";
  {
    std::ofstream output(textPath);
    if (!output) {
      throw std::runtime_error("Could not write " + textPath);
    }
    tree.print(output);
  }
  const std::string command = "python3 python/render_tree_svg.py " + shellQuote(textPath) +
                              " " + shellQuote(svgPath);
  const int status = std::system(command.c_str());
  std::remove(textPath.c_str());
  if (status != 0) {
    throw std::runtime_error("Could not render " + svgPath);
  }
}

void printMs(const char *label, double seconds) {
  std::cout << "  " << std::left << std::setw(18) << label << std::right << std::fixed
            << std::setprecision(1) << std::setw(10) << seconds * 1000.0 << " ms\n";
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    // Defaults for a run without arguments. Everything can be overridden on
    // the command line (see --help).
    Options options;
    options.datasetPath = "datasets/covertype.csv";
    options.backend = Backend::Parallel;
    options.algorithm = Algorithm::Cart;
    options.maxDepth = -1;

    // --- CART (Breiman): Gini, grow fully, then cost-complexity prune.
    options.criterion = Criterion::Gini;
    options.cartPrune = false;
    options.ccpAlpha = 0.0;

    // --- C4.5 (Quinlan, Release 8 defaults: -m 2 -c 25, subtree raising).
    options.c45MinObjects = 2;
    options.c45Prune = true;
    options.c45ConfidenceFactor = 0.25;

    if (!applyCommandLine(argc, argv, options)) {
      printUsage(argv[0]);
      return 0;
    }
    validateOptions(options);

    const unsigned threads = options.threads > 0
                                 ? static_cast<unsigned>(options.threads)
                                 : std::max(1u, std::thread::hardware_concurrency());

    std::cout << algorithmName(options.algorithm) << " on " << options.datasetPath
              << " (backend " << backendName(options.backend) << ")\n";

    // CUDA context creation is slow; do it while the CSV is being parsed.
    std::future<void> gpuReady;
    if (options.backend == Backend::Cuda) {
      gpuReady = std::async(std::launch::async, gpuPrepare);
    }

    double loadSeconds = 0.0;
    Dataset data;
    {
      ScopedTimer timer(loadSeconds);
      data = loadDataset(options.datasetPath, threads);
      multiplyDataset(data, options.datasetMultiplier);
    }

    Dataset train;
    Dataset test;
    if (options.holdoutFraction > 0.0) {
      splitHoldout(data, options.holdoutFraction, train, test);
      data = Dataset{};
    } else {
      train = std::move(data);
    }
    std::cout << "  rows " << train.rowCount << " train";
    if (test.rowCount > 0) {
      std::cout << " / " << test.rowCount << " test";
    }
    std::cout << ", " << train.featureCount() << " features, " << train.classCount()
              << " classes\n";
    if (options.algorithm == Algorithm::Cart) {
      std::cout << "  criterion " << criterionName(options.criterion) << ", pruning "
                << (!options.cartPrune ? std::string("off")
                    : options.ccpFolds > 1
                        ? std::to_string(options.ccpFolds) + "-fold CV"
                        : "alpha " + std::to_string(options.ccpAlpha))
                << "\n";
    } else {
      std::cout << "  pruning "
                << (options.c45Prune ? "CF " + std::to_string(options.c45ConfidenceFactor) +
                                           (options.c45SubtreeRaising ? " + raising" : "")
                                     : std::string("off"))
                << "\n";
    }

    if (gpuReady.valid()) {
      gpuReady.get();
    }
    TrainTimings timings;
    const DecisionTree tree = trainTree(train, options, timings);

    double evalSeconds = 0.0;
    double trainAccuracy = 0.0;
    double testAccuracy = 0.0;
    {
      ScopedTimer timer(evalSeconds);
      trainAccuracy = tree.accuracy(train, threads);
      if (test.rowCount > 0) {
        testAccuracy = tree.accuracy(test, threads);
      }
    }

    std::cout << "Tree:\n"
              << "  nodes " << tree.nodeCount() << ", leaves " << tree.leafCount()
              << ", depth " << tree.depth() << "\n"
              << std::fixed << std::setprecision(4) << "  train accuracy "
              << trainAccuracy * 100.0 << "%\n";
    if (test.rowCount > 0) {
      std::cout << "  test accuracy  " << testAccuracy * 100.0 << "%\n";
    }
    std::cout << "Timings:\n";
    printMs("load", loadSeconds);
    if (timings.presortSeconds > 0) {
      printMs("presort", timings.presortSeconds);
    }
    if (timings.gpuSetupSeconds > 0) {
      printMs("gpu setup", timings.gpuSetupSeconds);
    }
    printMs("build", timings.buildSeconds);
    printMs("prune", timings.pruneSeconds);
    printMs("train total", timings.presortSeconds + timings.gpuSetupSeconds +
                               timings.buildSeconds + timings.pruneSeconds);
    printMs("evaluate", evalSeconds);

    if (options.printTree) {
      tree.print(std::cout);
    }
    if (!options.dumpPath.empty()) {
      std::ofstream output(options.dumpPath);
      if (!output) {
        throw std::runtime_error("Could not write " + options.dumpPath);
      }
      tree.print(output);
    }
    if (!options.svgPath.empty()) {
      renderSvg(tree, options.svgPath);
    }
  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }
  return 0;
}
