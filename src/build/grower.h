#pragma once

#include "algo/split_rules.h"
#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"
#include "core/tree.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace dt {

struct GrowTimings {
  double prepareSeconds = 0.0; // presorting / selecting the rows' sorted columns
  double buildSeconds = 0.0;   // growing the tree
};

// Grows the unpruned tree of one training set with one backend, once.
class Grower {
public:
  virtual ~Grower() = default;

  // Grow the tree on every training row. If `distinct` is not null it
  // receives every feature's distinct values. If `leafOfRow` is not null it
  // receives, for every training row, the id of the leaf it ends in. A grower
  // may leave either one empty (the Simple backend does).
  virtual Tree grow(const SplitRules &rules, GrowTimings &timings,
                    DistinctValues *distinct = nullptr, RowLeaves *leafOfRow = nullptr) = 0;

  // After a grow that returned the leaves of rows: C4.5's RowArranger (see
  // algo/c45_pruning.h). False if this grower cannot.
  virtual bool arrangeRows(std::span<const std::uint32_t> /*slotOfId*/, std::size_t /*slots*/,
                           std::span<const std::uint32_t> /*ids*/, float * /*rowMajor*/,
                           std::uint32_t * /*rowOf*/, std::uint16_t * /*labels*/,
                           std::uint32_t * /*slotCounts*/) {
    return false;
  }

  // Memory the grower only uses inside grow(): the caller may use it until
  // the next grow() (it is already mapped, so no page faults). At least
  // 8 bytes per training row and feature.
  virtual std::span<std::byte> workspace() = 0;
};

// Serial (pool == nullptr) and Parallel backends.
std::unique_ptr<Grower> makeCpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool *pool);

// Simple backend: the textbook algorithm on one thread, no optimizations
// (simple_grower.cpp). The reference the others are compared against.
std::unique_ptr<Grower> makeSimpleGrower(const Dataset &train);

// The GPU backends' timed setup in its steps; their sum is the setup time.
struct GpuSetupTimings {
  double buffersSeconds = 0.0;  // page-locked staging (and --hybrid's hand-off) buffers, copy threads
  double allocateSeconds = 0.0; // streams, events, device memory
  double uploadSeconds = 0.0;   // labels and values to the GPU, each feature presorted as it arrives
  double finishSeconds = 0.0;   // the presort's end, kernel tables, host arena
  double total() const { return buffersSeconds + allocateSeconds + uploadSeconds + finishSeconds; }
};

// Cuda backend; small nodes are finished by CPU tasks on `pool`. Its setup
// (allocation, upload, presort) is timed in `setup`. `releasable` (null, or
// `train` itself): --hybrid may give the memory of its values back once they
// are on the GPU (see trainTree()).
std::unique_ptr<Grower> makeGpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool &pool, GpuSetupTimings &setup,
                                      Dataset *releasable = nullptr);

// The GPU's one-time start-up: the CUDA context (slow: 0.4-0.7 s). Call it
// early on another thread, e.g. while the dataset loads, so training does not
// wait for it. Everything after the context (the kernels, loaded lazily at
// their first launch, the memory pool, the page-locked staging buffers and
// the copy threads) is made in the timed setup (GpuGrower::setup), as cuML
// and XGBoost make theirs inside their fit (bench/tools/README.md "How time is
// measured"). Returns its duration in seconds (0 in the CPU-only build).
double gpuPrepare();

} // namespace dt
