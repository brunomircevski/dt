#pragma once

#include "algo/split_math.h"
#include "algo/split_rules.h"
#include "build/node_store.h"
#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dt {

// Presorted feature columns: for feature f, data[f * stride + i] is the i-th
// entry in value order. A tree node owns the same index range [begin, end) in
// every column. Splitting a node stably partitions each column's range into
// "left rows" then "right rows", so both children again own sorted ranges and
// nothing is ever sorted twice (the SLIQ/SPRINT idea).
//
// `scratch` (stride entries) is partition workspace; a node only uses its own
// range [begin, end) of it, so nodes can be split concurrently.
//
// `other` (optional) is a second copy with the layout of `data`. Big nodes
// are then partitioned out of place: from the side that holds their columns
// into the other side, where their children continue (ping-pong). That moves
// every entry once, which matters because big nodes are bound by memory
// bandwidth; in place, the rows of one child are also moved through a buffer.
//
// With `other`, `leftBits` (a bit per row id) says which rows of a node that
// is partitioned out of place go left: 1/8 of a byte per row, so it stays in
// the caches while the columns stream through them.
struct Columns {
  Entry *data = nullptr;
  std::size_t stride = 0;
  Entry *scratch = nullptr;
  Entry *other = nullptr;
  std::uint64_t *leftBits = nullptr;

  Entry *feature(std::size_t f, int side = 0) const { return (side ? other : data) + f * stride; }
};

// Pick the packing of (row, class) into 32 bits for this dataset.
EntryCodec makeEntryCodec(std::size_t classCount, std::size_t rowCount);

// Fill `out` (featureCount * rowCount entries) with every feature column
// sorted by value (LSD radix sort, stable, so equal values keep row order),
// and `sortedValues` (if not null, as many floats) with the values alone.
// `temp` (if not null, as many entries as `out`) is the sort's workspace;
// otherwise every sorting thread allocates a column's worth.
void presortColumns(const Dataset &dataset, const EntryCodec &codec, Entry *out,
                    ThreadPool *pool, float *sortedValues = nullptr, Entry *temp = nullptr);

// A node that still has to be split, and the features that can still split
// it: a feature whose values are all equal in a node is dropped for the whole
// subtree (its column range is not partitioned any more).
struct FeatureCuts;

struct Subtree {
  std::uint32_t node = 0; // id in the NodeStore (class counts already set)
  std::uint32_t begin = 0;
  std::uint32_t count = 0;
  int depth = 0;
  std::vector<std::uint32_t> features;
  int side = 0; // where its columns are: Columns::data (0) or Columns::other (1)
  // If not null: the best cut of every feature, found while the parent's
  // columns were partitioned (the node does not sweep its columns again).
  std::shared_ptr<FeatureCuts> cuts;
};

// What a node's sweeps found, per feature (see Subtree::cuts).
struct FeatureCuts {
  FeatureCuts(std::size_t featureCount, std::size_t classCount)
      : cuts(featureCount), lefts(featureCount * classCount), constant(featureCount, 0) {}

  std::vector<CutCandidate> cuts;     // best cut (invalid for unswept features)
  std::vector<std::uint32_t> lefts;   // feature * classCount + class: counts left of it
  std::vector<std::uint8_t> constant; // all values (nearly) equal: no cut here or below
};

// Per-count tables for the sweep's gain estimates, for counts below `size`:
// c * log2(c) (the same values as LogTable and dt::xlog2x) and 1 / c. Above
// `size` the values are computed. The CPU grower passes tables that cover all
// training rows; the default ones cover counts below kLogTableSize.
struct CountTables {
  const double *xlog = nullptr;
  const double *inverse = nullptr;
  std::uint32_t size = 0;

  double xlog2x(std::uint32_t count) const {
    return count < size ? xlog[count] : dt::xlog2x(static_cast<double>(count));
  }
  double reciprocal(std::uint32_t count) const {
    return count < size ? inverse[count] : 1.0 / static_cast<double>(count);
  }
};

// Fill `xlog` and `inverse` with the tables for counts 0 .. size - 1.
void fillCountTables(std::size_t size, std::vector<double> &xlog, std::vector<double> &inverse,
                     ThreadPool *pool);

struct SweepSetup; // what a threshold sweep needs from the builder (cpu_builder.cpp)

// Grows (sub)trees on the CPU from presorted columns. Used by the Serial and
// Parallel backends, and by the Cuda backend for nodes that are too small to be
// worth a GPU launch.
class CpuTreeBuilder {
public:
  // `goesLeft` is a scratch byte per row id (rows of different nodes never
  // collide). `pool` may be null (serial build). Without `tables`, tables
  // for counts below kLogTableSize are used.
  CpuTreeBuilder(const SplitRules &rules, const EntryCodec &codec, std::size_t featureCount,
                 NodeStore &store, std::uint8_t *goesLeft, ThreadPool *pool,
                 const ParallelOptions &options, CountTables tables = {});

  // Grow the subtree under `root`. With a pool, parts of it are grown by pool
  // tasks: call pool->waitIdle() before using the tree.
  void grow(const Columns &columns, Subtree root);

private:
  // Which children of a split are grown further, i.e. need sorted columns.
  enum class Keep { Both, Left, Right };

  // A child whose sweeps run while its parent's columns are partitioned.
  struct ChildSweep {
    std::uint32_t count = 0;
    std::uint32_t total[2] = {0, 0}; // class counts (two classes only)
    double parentWeighted = 0.0;
    std::uint32_t minChild = 0;
    FeatureCuts *out = nullptr;
  };

  void split(const Columns &columns, Subtree &item, std::vector<Subtree> &stack);
  // Best cut of one feature; `bestLeft` receives the class counts left of it.
  CutCandidate scanFeature(const Entry *entries, std::uint32_t count,
                           const std::uint32_t *total, double parentWeighted,
                           std::uint32_t minChild, std::uint32_t *bestLeft) const;
  template <Criterion Crit>
  CutCandidate scanFeatureFor(const Entry *entries, std::uint32_t count,
                              const std::uint32_t *total, double parentWeighted,
                              std::uint32_t minChild, std::uint32_t *bestLeft) const;
  SweepSetup sweepSetup() const;
  template <Criterion Crit, bool HasGap>
  CutCandidate scanTwoClasses(const Entry *entries, std::uint32_t count,
                              const std::uint32_t *total, double parentWeighted,
                              std::uint32_t minChild, std::uint32_t *bestLeft) const;
  template <int FixedK, Criterion Crit>
  CutCandidate scanFeatureK(const Entry *entries, std::uint32_t count,
                            const std::uint32_t *total, double parentWeighted,
                            std::uint32_t minChild, std::uint32_t *bestLeft) const;
  // Returns the side the children's columns are on. `children` (left,
  // right; null entries for children without) are swept on the way.
  int partition(const Columns &columns, const Subtree &item, int winner,
                std::uint32_t leftCount, Keep keep, ChildSweep *const *children);
  template <Criterion Crit, bool HasGap>
  void partitionAndSweep(const Entry *source, Entry *target, std::uint32_t count,
                         std::uint32_t leftCount, Keep keep, const std::uint64_t *leftBits,
                         bool isWinner, std::uint32_t feature,
                         ChildSweep *const *children) const;
  void partitionFeatureOutOfPlace(const Entry *source, Entry *target, std::uint32_t count,
                                  std::uint32_t leftCount, Keep keep,
                                  const std::uint64_t *leftBits) const;
  void partitionFeature(Entry *entries, std::uint32_t count, std::uint32_t leftCount,
                        Entry *buffer, Keep keep) const;
  void partitionFeatureInBlocks(Entry *entries, std::uint32_t count, std::uint32_t leftCount,
                                Entry *buffer) const;

  const SplitRules &rules_;
  EntryCodec codec_;
  std::size_t featureCount_;
  int classCount_;
  NodeStore &store_;
  std::uint8_t *goesLeft_;
  ThreadPool *pool_;
  ParallelOptions options_;
  CountTables tables_;
};

} // namespace dt
