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

// Cuda backend; small nodes are finished by CPU tasks on `pool`. Allocation
// and upload time goes to `setupSeconds`.
std::unique_ptr<Grower> makeGpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool &pool, double &setupSeconds);

// Create the CUDA context (slow: a few hundred ms). Call it early on another
// thread, e.g. while the dataset loads, so training does not wait for it.
// `threads`: the most CPU threads its staged copies use (the caller included).
void gpuPrepare(unsigned threads);

} // namespace dt
