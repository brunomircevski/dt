#include "app/repeat.h"
#include "app/trainer.h"
#include "build/grower.h"
#include "core/dataset.h"
#include "core/nvtx.h"
#include "core/options.h"
#include "core/thread_pool.h"
#include "core/timing.h"
#include "core/trace.h"
#include "core/tree.h"

#include <cstdio>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

using namespace dt;

struct LoadedData {
  Dataset train;
  Dataset test;              // empty without a test set
  std::size_t unseenTestRows = 0; // test rows of classes the training set lacks
  double seconds = 0.0;
};

// The training set and the test set (--test, --test-last or --holdout).
LoadedData loadData(const Options &options, ThreadPool &pool) {
  LoadedData data;
  {
    ScopedTimer timer(data.seconds);
    NvtxRange range("load");
    data.train = loadDataset(options.datasetPath, &pool);
    if (!options.testPath.empty()) {
      data.test = loadDataset(options.testPath, &pool);
      data.unseenTestRows = matchTrainingSchema(data.train, data.test);
    } else if (options.testLast > 0) {
      const Dataset all = std::move(data.train);
      splitTail(all, options.testLast, data.train, data.test, &pool);
    }
  }
  if (options.holdout > 0.0) { // not part of the load time
    const Dataset all = std::move(data.train);
    splitHoldout(all, options.holdout, options.seed, data.train, data.test, &pool);
  }
  return data;
}

std::string pruningDescription(const Options &options) {
  if (options.algorithm == Algorithm::C45) {
    return options.c45.prune ? "CF " + std::to_string(options.c45.confidence) + " + raising"
                             : "off";
  }
  return options.cart.prune ? "alpha " + std::to_string(options.cart.alpha) : "off";
}

void printDataSummary(const Options &options, const LoadedData &data) {
  std::cout << "  rows " << data.train.rowCount << " train";
  if (!options.testPath.empty()) {
    std::cout << " / " << data.test.rowCount << " test (" << options.testPath << ")";
  } else if (options.testLast > 0) {
    std::cout << " / " << data.test.rowCount << " test (the last rows)";
  } else if (data.test.rowCount > 0) {
    std::cout << " / " << data.test.rowCount << " test (seed " << options.seed << ")";
  }
  std::cout << ", " << data.train.featureCount() << " features, " << data.train.classCount()
            << " classes\n";
  if (data.unseenTestRows > 0) {
    std::cout << "  " << data.unseenTestRows
              << " test rows have a class the training set lacks (always misclassified)\n";
  }
  std::cout << "  pruning " << pruningDescription(options) << "\n";
}

// The process's peak resident memory so far in KiB (VmHWM; Linux only, -1
// elsewhere). Read right after training, before evaluation allocates more.
long peakResidentKiB() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      return std::stol(line.substr(6));
    }
  }
  return -1;
}

void printMs(const char *label, double seconds) {
  std::cout << "  " << std::left << std::setw(18) << label << std::right << std::fixed
            << std::setprecision(1) << std::setw(10) << seconds * 1000.0 << " ms\n";
}

struct Results {
  double trainAccuracy = 0.0;
  double testAccuracy = 0.0;
  double evalSeconds = 0.0;
  long trainPeakKiB = -1;
};

void printResults(const Tree &tree, const LoadedData &data, const TrainTimings &timings,
                  const Results &results) {
  std::cout << "Tree:\n"
            << "  nodes " << tree.nodeCount() << ", leaves " << tree.leafCount() << ", depth "
            << tree.depth() << "\n"
            << std::fixed << std::setprecision(4) << "  train accuracy "
            << results.trainAccuracy * 100.0 << "%\n";
  if (data.test.rowCount > 0) {
    std::cout << "  test accuracy  " << results.testAccuracy * 100.0 << "%\n";
  }
  std::cout << "Timings:\n";
  printMs("load", data.seconds);
  if (timings.gpuSetupSeconds > 0) {
    printMs("gpu setup", timings.gpuSetupSeconds);
  }
  printMs("presort", timings.presortSeconds);
  printMs("build", timings.buildSeconds);
  printMs("prune", timings.pruneSeconds);
  printMs("train total", timings.total());
  printMs("evaluate", results.evalSeconds);
  if (results.trainPeakKiB >= 0) {
    std::cout << "Memory:\n  peak after training " << results.trainPeakKiB << " KiB\n";
  }
}

// --profile: the trace with the facts of this run.
void writeProfile(const Options &options, const Tree &tree, const LoadedData &data,
                  const TrainTimings &timings, const Results &results) {
  auto text = [](const std::string &value) {
    std::string quoted = "\"";
    for (const char c : value) {
      quoted += c == '"' || c == '\\' ? std::string("\\") + c : std::string(1, c);
    }
    return quoted + "\"";
  };
  auto number = [](double value) {
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
  };
  trace::write(options.profilePath,
               {{"dataset", text(options.datasetPath)},
                {"backend", text(backendName(options.backend))},
                {"algorithm", text(algorithmName(options.algorithm))},
                {"pruning", text(pruningDescription(options))},
                {"threads", number(threadCount(options))},
                {"rows", number(static_cast<double>(data.train.rowCount))},
                {"features", number(static_cast<double>(data.train.featureCount()))},
                {"nodes", number(static_cast<double>(tree.nodeCount()))},
                {"depth", number(tree.depth())},
                {"load_seconds", number(data.seconds)},
                {"train_seconds", number(timings.total())},
                {"timings", "{\"gpu_setup\": " + number(timings.gpuSetupSeconds) +
                                ", \"presort\": " + number(timings.presortSeconds) +
                                ", \"build\": " + number(timings.buildSeconds) +
                                ", \"prune\": " + number(timings.pruneSeconds) + "}"},
                {"peak_rss_train_kib", number(static_cast<double>(results.trainPeakKiB))}});
}

int run(const Options &options, bool &exitWithoutCleanup) {
  std::cout << algorithmName(options.algorithm) << " on " << options.datasetPath
            << " (backend " << backendName(options.backend) << ")\n";

  // CUDA context creation is slow; do it while the CSV is being parsed. Not
  // with --repeat N > 1: a process may only fork before it uses CUDA.
  const bool forks = options.repeat > 1;
  std::future<void> gpuReady;
  if (options.backend == Backend::Cuda && !forks) {
    gpuReady = std::async(std::launch::async, gpuPrepare, threadCount(options));
  }

  // The calling thread works too, so threads - 1 workers. Loading and
  // evaluation always use it; training only with --parallel / --cuda.
  auto pool = std::make_unique<ThreadPool>(threadCount(options) - 1);
  const LoadedData data = loadData(options, *pool);
  printDataSummary(options, data);
  if (gpuReady.valid()) {
    gpuReady.get();
  }

  if (forks) {
    if (const std::optional<int> status = forkRepeatedTrees(data.train, options, pool)) {
      return *status; // the child of the last tree did the rest
    }
    exitWithoutCleanup = true; // the child of the last tree
  }

  TrainTimings timings;
  const bool profile = !options.profilePath.empty();
  if (profile) {
    trace::start();
  }
  const double start = unixNow();
  const Tree tree = trainTree(data.train, options, pool.get(), timings);
  if (options.repeat > 0) {
    printRepeatWindow(options, options.repeat, timings, start);
  }
  if (profile) {
    trace::stop();
  }

  Results results;
  results.trainPeakKiB = peakResidentKiB();
  {
    ScopedTimer timer(results.evalSeconds);
    NvtxRange range("accuracy");
    results.trainAccuracy = tree.accuracy(data.train, pool.get());
    if (data.test.rowCount > 0) {
      results.testAccuracy = tree.accuracy(data.test, pool.get());
    }
  }
  printResults(tree, data, timings, results);
  if (profile) {
    writeProfile(options, tree, data, timings, results);
  }
  if (!options.dumpPath.empty()) {
    std::ofstream output(options.dumpPath);
    if (!output) {
      throw std::runtime_error("Could not write " + options.dumpPath);
    }
    tree.print(output);
  }
  return 0;
}

} // namespace

int main(int argc, char *argv[]) {
  // The child that trains --repeat's last tree leaves with _exit(): no
  // destructors, as the parent's pool threads do not exist in it.
  bool exitWithoutCleanup = false;
  int status = 0;
  try {
    dt::Options options; // defaults: see options.h
    if (!dt::applyCommandLine(argc, argv, options)) {
      dt::printUsage(argv[0]);
      return 0;
    }
    dt::validateOptions(options);
    status = run(options, exitWithoutCleanup);
  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    status = 1;
  }
  if (exitWithoutCleanup) {
    std::cout.flush();
    std::cerr.flush();
    _exit(status);
  }
  return status;
}
