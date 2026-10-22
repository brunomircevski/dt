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
// by slot stably. Fills `rowMajor` (if not null) with their features (row by
// row, in that order), `rowOf` (if not null) with the training row at every
// position, `labels` with their labels and `slotCounts` (zero, slots x
// classes) with the class counts of every slot. False if it cannot; the
// pruning then does it on the CPU.
using RowArranger = std::function<bool(
    std::span<const std::uint32_t> slotOfId, std::size_t slots,
    std::span<const std::uint32_t> ids, float *rowMajor, std::uint32_t *rowOf,
    std::uint16_t *labels, std::uint32_t *slotCounts)>;

// Whether the pruning keeps a row-major copy of the features (narrow rows:
// a row's features share cache lines) or reads the dataset's columns
// through the training row of every position (wide rows: a walk touches one
// cache line per node either way, and reordering moves 6 bytes per row
// instead of the whole row). Only the row-major copy needs the workspace.
bool c45PruneUsesRowMajor(std::size_t featureCount);

// A node reduced to what routing needs: 16 bytes, four per cache line. The
// threshold is the largest float at or below the node's, which sends every
// float value the same way (value > threshold: right); a leaf leads back to
// itself.
struct PruneHop {
  float threshold;
  std::uint32_t feature;
  std::uint32_t next[2]; // left, right
};

// Subtree raising's work on big row slices, done elsewhere (the GPU), while
// the pruning keeps every decision. A slice is given as the training rows at
// its positions, in the pruning's order. Calls for disjoint subtrees may come
// from several threads at once. Only integer counts come back, so the
// pruning is the same to the last bit.
class SliceWorker {
public:
  virtual ~SliceWorker() = default;

  // Slices of fewer rows stay on the CPU.
  virtual std::size_t minRows() const = 0;

  // After the first arrangement (the RowArranger is done): every node's hop.
  // False if the worker cannot be used.
  virtual bool start(std::span<const PruneHop> hops) = 0;

  // New hops of nodes `ids` (hops[id]), for every call that starts after this
  // one.
  virtual void updateHops(std::span<const std::uint32_t> ids, std::span<const PruneHop> hops) = 0;

  // The rows sent through `root`'s subtree: for every leaf reached, in no
  // particular order, 2 + classes numbers: the leaf, the index in `rows` of
  // its first row and its class counts.
  virtual void countReached(std::uint32_t root, std::span<const std::uint32_t> rows,
                            std::vector<std::uint32_t> &reached) = 0;

  // The rows sorted stably by the slot of the leaf they reach (the index of
  // the leaf in `leaves`, which lists every leaf of `root`'s subtree):
  // source[index] is the index in `rows` of the row that moves to `index`,
  // and `slotCounts` (zero, leaves x classes) receives the class counts of
  // every slot.
  virtual void arrange(std::uint32_t root, std::span<const std::uint32_t> rows,
                       std::span<const std::uint32_t> leaves, std::uint32_t *source,
                       std::uint32_t *slotCounts) = 0;

  // The pruning is done (--profile: its GPU spans are recorded).
  virtual void finish() {}
};

// prune.c: error-based pruning. Each node's error is estimated with the upper
// limit of a binomial confidence interval (confidence factor CF). A subtree is
// replaced by a leaf, or by its most-used branch ("subtree raising"), when that
// is not estimated to be worse. `workspace` (if at least 8 bytes per training
// row and feature) holds the two row-major copies of the features that the
// pruning works with, instead of allocating them. `leafOfRow` (optional) is
// the leaf of every training row, which saves sending them through the tree;
// the pruning uses its ids up (as scratch); with it, `arranger` (optional) may
// order the rows.
// `worker` (optional) takes the big slices of subtree raising.
void c45PessimisticPrune(Tree &tree, const Dataset &train, double confidenceFactor,
                         ThreadPool *pool, std::span<std::byte> workspace = {},
                         RowLeaves *leafOfRow = nullptr, const RowArranger &arranger = {},
                         SliceWorker *worker = nullptr);

} // namespace dt
