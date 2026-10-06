#pragma once

#include "algo/split_math.h"
#include "algo/split_rules.h"
#include "build/node_store.h"
#include "core/dataset.h"
#include "core/options.h"
#include "core/thread_pool.h"

#include <cstddef>
#include <cstdint>
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
struct Columns {
  Entry *data = nullptr;
  std::size_t stride = 0;
  Entry *scratch = nullptr;

  Entry *feature(std::size_t f) const { return data + f * stride; }
};

// Pick the packing of (row, class) into 32 bits for this dataset.
EntryCodec makeEntryCodec(std::size_t classCount, std::size_t rowCount);

// Fill `out` (featureCount * rowCount entries) with every feature column
// sorted by value (LSD radix sort, stable, so equal values keep row order).
void presortColumns(const Dataset &dataset, const EntryCodec &codec, Entry *out,
                    ThreadPool *pool);

// A node that still has to be split, and the features that can still split
// it: a feature whose values are all equal in a node is dropped for the whole
// subtree (its column range is not partitioned any more).
struct Subtree {
  std::uint32_t node = 0; // id in the NodeStore (class counts already set)
  std::uint32_t begin = 0;
  std::uint32_t count = 0;
  int depth = 0;
  std::vector<std::uint32_t> features;
};

// Grows (sub)trees on the CPU from presorted columns. Used by the Serial and
// Parallel backends, and by the Cuda backend for nodes that are too small to be
// worth a GPU launch.
class CpuTreeBuilder {
public:
  // `goesLeft` is a scratch byte per row id (rows of different nodes never
  // collide). `pool` may be null (serial build).
  CpuTreeBuilder(const SplitRules &rules, const EntryCodec &codec, std::size_t featureCount,
                 NodeStore &store, std::uint8_t *goesLeft, ThreadPool *pool,
                 const ParallelOptions &options);

  // Grow the subtree under `root`. With a pool, parts of it are grown by pool
  // tasks: call pool->waitIdle() before using the tree.
  void grow(const Columns &columns, Subtree root);

private:
  void split(const Columns &columns, Subtree &item, std::vector<Subtree> &stack);
  CutCandidate scanFeature(const Entry *entries, std::uint32_t count,
                           const std::uint32_t *total, double parentWeighted,
                           std::uint32_t minChild) const;
  template <int FixedK>
  CutCandidate scanFeatureK(const Entry *entries, std::uint32_t count,
                            const std::uint32_t *total, double parentWeighted,
                            std::uint32_t minChild) const;
  void partition(const Columns &columns, const Subtree &item, int winner,
                 std::uint32_t leftCount);
  void partitionFeature(Entry *entries, std::uint32_t count, Entry *buffer) const;
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
};

} // namespace dt
