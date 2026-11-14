#pragma once

#include "app/trainer.h"
#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"

#include <cstddef>
#include <memory>
#include <optional>

namespace dt {

// --repeat N: train N trees on the data loaded once and print each one's
// training window as Unix times, so that an outside power sampler can
// integrate the energy of training only (bench/power). A window runs from
// the start of trainTree() to the end of training, before the grower frees
// its memory.

// Seconds since the Unix epoch.
double unixNow();

// "repeat RUN/N: train X ms, unix START END".
void printRepeatWindow(const Options &options, std::size_t run, const TrainTimings &timings,
                       double start);

// --repeat N > 1: every tree is trained in a forked child, so none of them
// inherits what an earlier tree left behind (on the GPU, a later tree in the
// same process took up to 40% longer to set up, and so did a tree trained by
// a process after forking). A child gets what a single run has when its timer
// starts: the loaded data (shared with the parent), its own threads (fork
// copies only the calling one) and, with --gpu or --hybrid, the CUDA context and the
// staging buffers. With --repeat-warmup S its threads spin for S seconds
// first, as a single run's CSV load does right before its tree.
//
// Trains trees 1 .. N-1, then forks the child of tree N. In that child it
// returns nullopt with `pool` replaced by the child's own pool (the parent's
// is leaked: its threads do not exist in the child); the caller goes on like
// a single run and must leave with _exit(). In the parent it returns the
// exit status of that child, once it is done.
std::optional<int> forkRepeatedTrees(const Dataset &train, const Options &options,
                                     std::unique_ptr<ThreadPool> &pool);

} // namespace dt
