#pragma once

#include "dataset.h"
#include "options.h"
#include "tree.h"

struct TrainTimings {
  double presortSeconds = 0.0;  // CPU backends: sorting every feature once
  double gpuSetupSeconds = 0.0; // Cuda backend: allocation, upload, GPU presort
  double buildSeconds = 0.0;    // growing the tree
  double pruneSeconds = 0.0;    // all post-processing (C4.5 steps, pruning, CV)
};

// Train a CART or C4.5 tree on `train` as configured by `options`.
DecisionTree trainTree(const Dataset &train, const Options &options,
                       TrainTimings &timings, bool verbose = true);
