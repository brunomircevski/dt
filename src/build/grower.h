#pragma once

#include "algo/split_rules.h"
#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"
#include "core/tree.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace dt {

struct GrowTimings {
  double prepareSeconds = 0.0; // presorting / selecting the rows' sorted columns
  double buildSeconds = 0.0;   // growing the tree
};

// Grows unpruned trees on (subsets of) one training set with one backend.
// It is created once per training run, so cross-validation reuses the
// presorted columns and (Cuda) the device buffers.
class Grower {
public:
  virtual ~Grower() = default;

  // Grow on the rows `rows` (increasing row ids) of the training set, or on
  // all rows if `rows` is empty. `rules` must be made for that many rows.
  // If `sortedValues` is not null it receives every feature's values of those
  // rows in increasing order (column-major), a by-product of the presort that
  // C4.5's thresholds need.
  virtual Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows,
                    GrowTimings &timings, std::vector<float> *sortedValues = nullptr) = 0;
};

// Serial (pool == nullptr) and Parallel backends. With `reusable` false the
// grower may be used once, on all rows, and saves a copy of the columns.
std::unique_ptr<Grower> makeCpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool *pool, bool reusable);

// Cuda backend; small nodes are finished by CPU tasks on `pool`. Allocation
// and upload time goes to `setupSeconds`.
std::unique_ptr<Grower> makeGpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool &pool, bool reusable, double &setupSeconds);

// Create the CUDA context (slow: a few hundred ms). Call it early on another
// thread, e.g. while the dataset loads, so training does not wait for it.
void gpuPrepare();

} // namespace dt
