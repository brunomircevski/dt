#include "app/trainer.h"
#include "build/grower.h"
#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"
#include "core/nvtx.h"
#include "core/timing.h"
#include "core/trace.h"
#include "core/tree.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace {

void printMs(const char *label, double seconds) {
  std::cout << "  " << std::left << std::setw(18) << label << std::right << std::fixed
            << std::setprecision(1) << std::setw(10) << seconds * 1000.0 << " ms\n";
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
  // --repeat N > 1: the child that trains the last tree also evaluates and
  // prints it, then leaves with _exit (no destructors: the parent's pool
  // threads do not exist in it).
  bool lastChild = false;
  auto finish = [&lastChild](int status) {
    if (lastChild) {
      std::cout.flush();
      std::cerr.flush();
      _exit(status);
    }
    return status;
  };
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
    // Not with --repeat N > 1: the trees before the last one are trained in
    // forked children, and a process may only fork before it uses CUDA.
    const bool forks = options.repeat > 1;
    std::future<void> gpuReady;
    if (options.backend == Backend::Cuda && !forks) {
      gpuReady = std::async(std::launch::async, gpuPrepare, threadCount(options));
    }

    // The calling thread works too, so threads - 1 workers. Loading and
    // evaluation always use it; training only with --parallel / --cuda.
    // (On the heap: the child of --repeat's last tree leaks it, see below.)
    auto poolOwner = std::make_unique<ThreadPool>(threadCount(options) - 1);
    ThreadPool &pool = *poolOwner;

    double loadSeconds = 0.0;
    Dataset data;
    Dataset train;
    Dataset test;
    {
      ScopedTimer timer(loadSeconds);
      NvtxRange range("load");
      data = loadDataset(options.datasetPath, &pool);
      if (options.testLast > 0) {
        splitTail(data, options.testLast, train, test, &pool);
        data = std::move(train);
      }
    }

    if (options.holdout > 0.0) {
      splitHoldout(data, options.holdout, options.seed, RandomStream::Holdout, train, test,
                   &pool);
      data = Dataset{};
    } else {
      train = std::move(data);
    }
    std::cout << "  rows " << train.rowCount << " train";
    if (options.testLast > 0) {
      std::cout << " / " << test.rowCount << " test (the last rows)";
    } else if (test.rowCount > 0) {
      std::cout << " / " << test.rowCount << " test (seed " << options.seed << ")";
    }
    std::cout << ", " << train.featureCount() << " features, " << train.classCount()
              << " classes\n";
    std::cout << "  pruning " << pruningDescription(options) << "\n";

    if (gpuReady.valid()) {
      gpuReady.get();
    }
    // --repeat: the windows let an outside power sampler integrate the
    // energy of training only (bench/power). A window runs from the start of
    // trainTree() to the end of training, before the grower frees its memory.
    auto unixNow = [] {
      return std::chrono::duration<double>(
                 std::chrono::system_clock::now().time_since_epoch())
          .count();
    };
    auto printWindow = [&](std::size_t run, const TrainTimings &done, double start) {
      std::cout << "  repeat " << run << "/" << options.repeat << ": train " << std::fixed
                << std::setprecision(1) << done.total() * 1000.0 << " ms, unix "
                << std::setprecision(6) << start << " " << done.endUnix << std::endl;
    };
    // --repeat N > 1: every tree in a forked child, so none of them inherits
    // what an earlier tree left behind (on the GPU, a later tree in the same
    // process took up to 40% longer to set up, and so did a tree trained by
    // this process after forking). A child gets what a single run has when
    // its timer starts: the loaded data (shared with this process), its own
    // threads (fork copies only the calling one) and, with --cuda, the CUDA
    // context and the staging buffers. Trees 1..N-1 are trained in this loop;
    // the child of tree N goes on below like a single run (evaluation, output)
    // while this process waits for it and exits with its status.
    for (std::size_t run = 1; forks && run < options.repeat; ++run) {
      std::cout.flush();
      std::fflush(nullptr);
      const pid_t child = fork();
      if (child < 0) {
        throw std::runtime_error("--repeat: fork failed");
      }
      if (child == 0) {
        int status = 0;
        try {
          ThreadPool childPool(threadCount(options) - 1);
          if (options.backend == Backend::Cuda) {
            gpuPrepare(threadCount(options));
          }
          TrainTimings childTimings;
          const double start = unixNow();
          trainTree(train, options, &childPool, childTimings, /*verbose=*/false);
          printWindow(run, childTimings, start);
        } catch (const std::exception &exception) {
          std::cerr << "Error (--repeat tree " << run << "): " << exception.what() << '\n';
          status = 1;
        }
        std::cout.flush();
        std::cerr.flush();
        _exit(status); // no destructors: the parent's pool threads do not exist here
      }
      int status = 0;
      if (waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
          WEXITSTATUS(status) != 0) {
        throw std::runtime_error("--repeat: tree " + std::to_string(run) + " failed");
      }
    }
    std::optional<ThreadPool> childPool;
    ThreadPool *work = &pool; // training and evaluation
    if (forks) {
      std::cout.flush();
      std::fflush(nullptr);
      const pid_t child = fork();
      if (child < 0) {
        throw std::runtime_error("--repeat: fork failed");
      }
      if (child > 0) {
        int status = 0;
        if (waitpid(child, &status, 0) != child || !WIFEXITED(status)) {
          throw std::runtime_error("--repeat: tree " + std::to_string(options.repeat) + " failed");
        }
        return WEXITSTATUS(status);
      }
      lastChild = true;
      // Its threads do not exist here: destroying it would wait for them.
      static_cast<void>(poolOwner.release());
      childPool.emplace(threadCount(options) - 1);
      work = &*childPool;
      if (options.backend == Backend::Cuda) {
        gpuPrepare(threadCount(options));
      }
    }
    TrainTimings timings;
    const bool profile = !options.profilePath.empty();
    if (profile) {
      trace::start();
    }
    Tree tree;
    {
      const double start = unixNow();
      tree = trainTree(train, options, work, timings);
      if (options.repeat > 0) {
        printWindow(options.repeat, timings, start);
      }
    }
    if (profile) {
      trace::stop();
    }
    const long trainPeakKiB = peakResidentKiB();

    double evalSeconds = 0.0;
    double trainAccuracy = 0.0;
    double testAccuracy = 0.0;
    {
      ScopedTimer timer(evalSeconds);
      NvtxRange range("accuracy");
      trainAccuracy = tree.accuracy(train, work);
      if (test.rowCount > 0) {
        testAccuracy = tree.accuracy(test, work);
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
    if (trainPeakKiB >= 0) {
      std::cout << "Memory:\n  peak after training " << trainPeakKiB << " KiB\n";
    }

    if (profile) {
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
                    {"rows", number(static_cast<double>(train.rowCount))},
                    {"features", number(static_cast<double>(train.featureCount()))},
                    {"nodes", number(static_cast<double>(tree.nodeCount()))},
                    {"depth", number(tree.depth())},
                    {"load_seconds", number(loadSeconds)},
                    {"train_seconds", number(timings.total())},
                    {"timings", "{\"gpu_setup\": " + number(timings.gpuSetupSeconds) +
                                    ", \"presort\": " + number(timings.presortSeconds) +
                                    ", \"build\": " + number(timings.buildSeconds) +
                                    ", \"cross_validation\": " + number(timings.cvSeconds) +
                                    ", \"prune\": " + number(timings.pruneSeconds) + "}"},
                    {"peak_rss_train_kib", number(static_cast<double>(trainPeakKiB))}});
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
    return finish(1);
  }
  return finish(0);
}
