#pragma once

#include "core/dataset.h"
#include "core/tree.h"

#include <cstddef>
#include <span>
#include <vector>

namespace dt {

class ThreadPool;

// C4.5 (Release 8) post-processing, in the order c4.5 applies it. `pool` may
// be null (serial).

// build.c, end of FormTree: a subtree that makes as many training errors as a
// single leaf would is collapsed into that leaf.
void c45CollapseUselessSplits(Tree &tree);

// contin.c, ContinTest: a threshold is the largest training value of the
// feature (over the whole training set) that is <= the cut's midpoint.
// The training partition does not change; only unseen values in the gap do.
// `sortedValues`: every feature's training values in increasing order
// (column-major, `rowCount` per feature).
void c45UseTrainingValueThresholds(Tree &tree, std::span<const float> sortedValues,
                                   std::size_t rowCount, ThreadPool *pool);

// prune.c: error-based pruning. Each node's error is estimated with the upper
// limit of a binomial confidence interval (confidence factor CF). A subtree is
// replaced by a leaf, or by its most-used branch ("subtree raising"), when that
// is not estimated to be worse. `workspace` (if at least 4 bytes per training
// row and feature) is used instead of allocating the row-major copy of the
// features.
void c45PessimisticPrune(Tree &tree, const Dataset &train, double confidenceFactor,
                         ThreadPool *pool, std::span<std::byte> workspace = {});

} // namespace dt
