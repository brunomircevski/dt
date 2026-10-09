#include "app/trainer.h"
#include "build/grower.h"
#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"
#include "core/timing.h"
#include "core/tree.h"

#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printMs(const char *label, double seconds) {
  std::cout << "  " << std::left << std::setw(18) << label << std::right << std::fixed
            << std::setprecision(1) << std::setw(10) << seconds * 1000.0 << " ms\n";
}

std::string pruningDescription(const dt::Options &options) {
  if (options.algorithm == dt::Algorithm::C45) {
    return options.c45.prune ? "CF " + std::to_string(options.c45.confidence) + " + raising"
                             : "off";
  }
  switch (options.cart.pruning) {
  case dt::CartPruning::None: return "off";
  case dt::CartPruning::Alpha: return "alpha " + std::to_string(options.cart.alpha);
  case dt::CartPruning::CrossValidation:
    return "alpha by " + std::to_string(options.cart.folds) + "-fold CV (seed " +
           std::to_string(options.seed) + ")";
  }
  return "?";
}

} // namespace

int main(int argc, char *argv[]) {
  using namespace dt;
  try {
    Options options; // defaults: see options.h
    if (!applyCommandLine(argc, argv, options)) {
      printUsage(argv[0]);
      return 0;
    }
    validateOptions(options);

    std::cout << algorithmName(options.algorithm) << " on " << options.datasetPath
              << " (backend " << backendName(options.backend) << ")\n";

    // CUDA context creation is slow; do it while the CSV is being parsed.
    std::future<void> gpuReady;
    if (options.backend == Backend::Cuda) {
      gpuReady = std::async(std::launch::async, gpuPrepare);
    }

    // The calling thread works too, so threads - 1 workers. Loading and
    // evaluation always use it; training only with --parallel / --cuda.
    ThreadPool pool(threadCount(options) - 1);

    double loadSeconds = 0.0;
    Dataset data;
    {
      ScopedTimer timer(loadSeconds);
      data = loadDataset(options.datasetPath, &pool);
      multiplyDataset(data, options.multiplier);
    }

    Dataset train;
    Dataset test;
    if (options.holdout > 0.0) {
      splitHoldout(data, options.holdout, options.seed, RandomStream::Holdout, train, test,
                   &pool);
      data = Dataset{};
    } else {
      train = std::move(data);
    }
    std::cout << "  rows " << train.rowCount << " train";
    if (test.rowCount > 0) {
      std::cout << " / " << test.rowCount << " test (seed " << options.seed << ")";
    }
    std::cout << ", " << train.featureCount() << " features, " << train.classCount()
              << " classes\n";
    std::cout << "  pruning " << pruningDescription(options) << "\n";

    if (gpuReady.valid()) {
      gpuReady.get();
    }
    TrainTimings timings;
    const Tree tree = trainTree(train, options, &pool, timings);

    double evalSeconds = 0.0;
    double trainAccuracy = 0.0;
    double testAccuracy = 0.0;
    {
      ScopedTimer timer(evalSeconds);
      trainAccuracy = tree.accuracy(train, &pool);
      if (test.rowCount > 0) {
        testAccuracy = tree.accuracy(test, &pool);
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
    if (timings.gpuSetupSeconds > 0) {
      printMs("gpu setup", timings.gpuSetupSeconds);
    }
    printMs("presort", timings.presortSeconds);
    printMs("build", timings.buildSeconds);
    if (timings.cvSeconds > 0) {
      printMs("cross-validation", timings.cvSeconds);
    }
    printMs("prune", timings.pruneSeconds);
    printMs("train total", timings.total());
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
  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }
  return 0;
}
