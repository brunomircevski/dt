#pragma once

#include "build/grower.h"
#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"
#include "core/tree.h"

namespace dt {

struct TrainTimings {
  double gpuSetupSeconds = 0.0; // Cuda: allocation, upload (gpuSetup.total())
  GpuSetupTimings gpuSetup;     // Cuda: gpuSetupSeconds in its steps
  double presortSeconds = 0.0;  // sorting every feature once
  double buildSeconds = 0.0;    // growing the tree
  double pruneSeconds = 0.0;    // pruning
  // Cuda: the one-time start-up before training (CUDA context, memory pool,
  // page-locked staging buffers), made while the dataset loads. Not training
  // work, so not in total(); reported apart.
  double gpuStartupSeconds = 0.0;
  // Unix time when training ended: after the pruning, before the grower frees
  // its memory (the end of --repeat's energy window).
  double endUnix = 0.0;

  double total() const {
    return gpuSetupSeconds + presortSeconds + buildSeconds + pruneSeconds;
  }
};

// Train a CART or C4.5 tree on `train` as configured by `options`. `pool`
// is used by the Parallel and Cuda backends (the Serial backend ignores it);
// the Cuda backend needs one (it may have zero workers).
// `releasable` (null, or `train` itself, used for nothing else afterwards):
// the training may give the memory of its feature values back once it no
// longer needs them (then train.valuesReleased is set; the labels stay).
// Only where nothing reads them later (see trainingMayReleaseValues()).
Tree trainTree(const Dataset &train, const Options &options, ThreadPool *pool,
               TrainTimings &timings, bool verbose = true, Dataset *releasable = nullptr);

// Whether trainTree() could give the training values back: one tree, grown
// by --hybrid, and no C4.5 pruning (which reads them).
bool trainingMayReleaseValues(const Options &options);

} // namespace dt
