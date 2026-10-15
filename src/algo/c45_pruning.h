#pragma once

#include "core/dataset.h"
#include "core/tree.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace dt {

class ThreadPool;

// C4.5 (Release 8) post-processing, in the order c4.5 applies it. `pool` may
// be null (serial).

// build.c, end of FormTree: a subtree that makes as many training errors as a
// single leaf would is collapsed into that leaf. `leafOfRow` (optional: the
// leaf of every training row) is changed to the leaves of the new tree.
void c45CollapseUselessSplits(Tree &tree, ThreadPool *pool, RowLeaves *leafOfRow = nullptr);

// contin.c, ContinTest: a threshold is the largest training value of the
// feature (over the whole training set) that is <= the cut's midpoint.
// The training partition does not change; only unseen values in the gap do.
// `distinct`: every feature's distinct training values.
void c45UseTrainingValueThresholds(Tree &tree, const DistinctValues &distinct, ThreadPool *pool);

// Puts the training rows in the order the pruning starts from, elsewhere (the
// GPU): row r goes to slot slotOfId[ids[r]] (< slots), the rows sorted
// by slot stably. Fills `rowMajor` with their features (row by row, in that
// order), `labels` with their labels and `slotCounts` (zero, slots x
// classes) with the class counts of every slot. False if it cannot; the
// pruning then does it on the CPU.
using RowArranger = std::function<bool(
    std::span<const std::uint32_t> slotOfId, std::size_t slots,
    std::span<const std::uint32_t> ids, float *rowMajor, std::uint16_t *labels,
    std::uint32_t *slotCounts)>;


// prune.c: error-based pruning. Each node's error is estimated with the upper
// limit of a binomial confidence interval (confidence factor CF). A subtree is
// replaced by a leaf, or by its most-used branch ("subtree raising"), when that
// is not estimated to be worse. `workspace` (if at least 8 bytes per training
// row and feature) holds the two row-major copies of the features that the
// pruning works with, instead of allocating them. `leafOfRow` (optional) is
// the leaf of every training row, which saves sending them through the tree;
// the pruning uses its ids up (as scratch); with it, `arranger` (optional) may
// order the rows.
void c45PessimisticPrune(Tree &tree, const Dataset &train, double confidenceFactor,
                         ThreadPool *pool, std::span<std::byte> workspace = {},
                         RowLeaves *leafOfRow = nullptr, const RowArranger &arranger = {});

} // namespace dt
