#pragma once

#include "dataset.h"
#include "options.h"
#include "split_math.h"
#include "split_rules.h"
#include "thread_pool.h"
#include "tree.h"

#include <memory>
#include <vector>

struct GpuTimings {
  double setupSeconds = 0.0; // allocation + upload + presort on the GPU
  double buildSeconds = 0.0; // growing the tree (GPU levels + CPU subtrees)
};

// Create the CUDA context (slow: a few hundred ms). Call it early on another
// thread, e.g. while the dataset loads, so training does not wait for it.
void gpuPrepare();

// Grow the unpruned tree: the GPU processes large nodes one tree level at a
// time; nodes below options.gpuMinRows are copied back and finished by
// CpuTreeBuilder tasks on `pool` while the GPU continues.
// If `distinctValues` is not null it receives every feature's distinct values
// in increasing order (a by-product of the presort; C4.5 needs them).
std::unique_ptr<Node> gpuGrowTree(const Dataset &dataset, const SplitRules &rules,
                                  const dt::EntryCodec &codec, const Options &options,
                                  ThreadPool &pool, GpuTimings &timings,
                                  std::vector<std::vector<float>> *distinctValues);
