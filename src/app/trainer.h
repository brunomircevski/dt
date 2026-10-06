#pragma once

#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"
#include "core/tree.h"

namespace dt {

struct TrainTimings {
  double gpuSetupSeconds = 0.0; // Cuda: allocation, upload
  double presortSeconds = 0.0;  // sorting every feature once
  double buildSeconds = 0.0;    // growing the tree
  double cvSeconds = 0.0;       // CART: cross-validation (growing the fold trees)
  double pruneSeconds = 0.0;    // post-processing and pruning

  double total() const {
    return gpuSetupSeconds + presortSeconds + buildSeconds + cvSeconds + pruneSeconds;
  }
};

// Train a CART or C4.5 tree on `train` as configured by `options`. `pool`
// is used by the Parallel and Cuda backends (the Serial backend ignores it);
// the Cuda backend needs one (it may have zero workers).
Tree trainTree(const Dataset &train, const Options &options, ThreadPool *pool,
               TrainTimings &timings, bool verbose = true);

} // namespace dt
