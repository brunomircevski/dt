#include "app/repeat.h"

#include "build/grower.h"

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace dt {

namespace {

// --repeat-warmup: every thread of `threads` spins until the deadline.
void warmUp(const Options &options, ThreadPool &threads) {
  if (options.repeatWarmup <= 0.0) {
    return;
  }
  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::duration<double>(options.repeatWarmup));
  threads.parallelFor(threads.workerCount() + 1, [end](std::size_t) {
    while (std::chrono::steady_clock::now() < end) {
    }
  });
}

// What a child has before its tree: its own pool, the CUDA context, the warm-up.
std::unique_ptr<ThreadPool> prepareChild(const Options &options) {
  auto pool = std::make_unique<ThreadPool>(threadCount(options) - 1);
  if (options.backend == Backend::Cuda) {
    gpuPrepare(threadCount(options));
  }
  warmUp(options, *pool);
  return pool;
}

pid_t forkFlushed() {
  std::cout.flush();
  std::fflush(nullptr);
  const pid_t child = fork();
  if (child < 0) {
    throw std::runtime_error("--repeat: fork failed");
  }
  return child;
}

} // namespace

double unixNow() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void printRepeatWindow(const Options &options, std::size_t run, const TrainTimings &timings,
                       double start) {
  std::cout << "  repeat " << run << "/" << options.repeat << ": train " << std::fixed
            << std::setprecision(1) << timings.total() * 1000.0 << " ms, unix "
            << std::setprecision(6) << start << " " << timings.endUnix << std::endl;
}

std::optional<int> forkRepeatedTrees(const Dataset &train, const Options &options,
                                     std::unique_ptr<ThreadPool> &pool) {
  for (std::size_t run = 1; run < options.repeat; ++run) {
    const pid_t child = forkFlushed();
    if (child == 0) {
      int status = 0;
      try {
        const std::unique_ptr<ThreadPool> childPool = prepareChild(options);
        TrainTimings timings;
        const double start = unixNow();
        trainTree(train, options, childPool.get(), timings, /*verbose=*/false);
        printRepeatWindow(options, run, timings, start);
      } catch (const std::exception &exception) {
        std::cerr << "Error (--repeat tree " << run << "): " << exception.what() << '\n';
        status = 1;
      }
      std::cout.flush();
      std::cerr.flush();
      _exit(status); // no destructors: the parent's pool threads do not exist here
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      throw std::runtime_error("--repeat: tree " + std::to_string(run) + " failed");
    }
  }
  const pid_t child = forkFlushed();
  if (child > 0) {
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status)) {
      throw std::runtime_error("--repeat: tree " + std::to_string(options.repeat) + " failed");
    }
    return WEXITSTATUS(status);
  }
  // Its threads do not exist here: destroying it would wait for them.
  static_cast<void>(pool.release());
  pool = prepareChild(options);
  return std::nullopt;
}

} // namespace dt
