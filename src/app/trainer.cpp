#include "app/trainer.h"

#include "algo/c45_pruning.h"
#include "algo/cart_pruning.h"
#include "algo/split_rules.h"
#include "build/grower.h"
#include "core/timing.h"
#include "core/trace.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>

namespace dt {

Tree trainTree(const Dataset &train, const Options &options, ThreadPool *pool,
               TrainTimings &timings, bool verbose) {
  validateOptions(options);
  if (train.classCount() == 0 || train.rowCount == 0) {
    throw std::runtime_error("Cannot train on an empty dataset.");
  }
  ThreadPool *buildPool = options.backend == Backend::Serial || options.backend == Backend::Simple
                               ? nullptr
                               : pool;
  const bool isCart = options.algorithm == Algorithm::Cart;

  std::unique_ptr<Grower> grower;
  // --profile: the trace ends with the training, before the grower frees its
  // memory (destroyed after this).
  struct TraceEnd {
    ~TraceEnd() { trace::stop(); }
  } traceEnd;
  // Declared after the grower, so it runs before the grower frees its memory.
  struct TrainEnd {
    TrainTimings &timings;
    ~TrainEnd() {
      timings.endUnix =
          std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
              .count();
    }
  } trainEnd{timings};
  if (usesGpu(options.backend)) {
    if (!pool) {
      throw std::logic_error("The GPU backends need a thread pool");
    }
    trace::Phase phase("gpu setup");
    trace::Doing doing(trace::Work::GpuHost);
    grower = makeGpuGrower(train, options, *pool, timings.gpuSetupSeconds);
  } else if (options.backend == Backend::Simple) {
    grower = makeSimpleGrower(train);
  } else {
    grower = makeCpuGrower(train, options, buildPool);
  }

  const SplitRules rules(options, train.classCount(), train.rowCount);
  GrowTimings growTimings;
  DistinctValues distinct;      // C4.5's thresholds need them
  RowLeaves leafOfRow;          // C4.5's pruning starts from them
  const bool c45Prune = !isCart && options.c45.prune;
  Tree tree = grower->grow(rules, growTimings, isCart ? nullptr : &distinct,
                           c45Prune ? &leafOfRow : nullptr);
  timings.presortSeconds += growTimings.prepareSeconds;
  timings.buildSeconds += growTimings.buildSeconds;

  if (!isCart) {
    ScopedTimer timer(timings.pruneSeconds);
    {
      trace::Phase phase("c45 collapse, thresholds");
      trace::Doing doing(trace::Work::PruneOther);
      // Part of growing in c4.5 (done with or without pruning).
      c45CollapseUselessSplits(tree, buildPool, &leafOfRow);
      if (distinct.offsets.empty()) {
        distinct = distinctValues(train); // the grower did not provide them
      }
      c45UseTrainingValueThresholds(tree, distinct, buildPool);
    }
    if (options.c45.prune) {
      if (verbose) {
        std::cout << "  unpruned: " << tree.nodeCount() << " nodes\n";
      }
      const RowArranger arranger = [&](auto... args) { return grower->arrangeRows(args...); };
      // (--gpu allocates the workspace when asked for it)
      const std::span<std::byte> workspace = c45PruneUsesRowMajor(train.featureCount())
                                                 ? grower->workspace()
                                                 : std::span<std::byte>{};
      c45PessimisticPrune(tree, train, options.c45.confidence, buildPool, workspace, &leafOfRow,
                          arranger);
    }
    return tree;
  }
  if (options.cart.prune) {
    if (verbose) {
      std::cout << "  unpruned: " << tree.nodeCount() << " nodes\n";
    }
    ScopedTimer timer(timings.pruneSeconds);
    trace::Phase phase("cart prune");
    trace::Doing doing(trace::Work::Prune);
    cartCostComplexityPrune(tree, options.cart.alpha);
  }
  return tree;
}

} // namespace dt
