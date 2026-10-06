#pragma once

#include "dataset.h"
#include "split_math.h"
#include "split_rules.h"
#include "thread_pool.h"
#include "tree.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Presorted feature columns: for feature f, data[f * stride + i] is the i-th
// entry in value order. A tree node owns the same index range [begin, end) in
// every column. Splitting a node stably partitions each column's range into
// "left rows" then "right rows", so both children again own sorted ranges and
// nothing is ever sorted twice (the SLIQ/SPRINT idea).
struct Columns {
  dt::Entry *data = nullptr;
  std::size_t stride = 0;
  std::size_t featureCount = 0;

  dt::Entry *feature(std::size_t f) const { return data + f * stride; }
};

// Pick the packing of (row, class) into 32 bits for this dataset.
dt::EntryCodec makeEntryCodec(std::size_t classCount, std::size_t rowCount);

// Fill `out` (featureCount * rowCount entries) with every feature column
// sorted by value (LSD radix sort, stable, so equal values keep row order).
void presortColumns(const Dataset &dataset, const dt::EntryCodec &codec, dt::Entry *out,
                    ThreadPool *pool);

// The distinct values of a column, from its entries in sorted order.
std::vector<float> distinctSortedValues(const dt::Entry *sorted, std::size_t count);
std::vector<float> distinctSortedValues(const float *sorted, std::size_t count);

// Grows (sub)trees on the CPU from presorted columns. Used by the Serial and
// Parallel backends, and by the Cuda backend for nodes that are too small to be
// worth a GPU launch.
class CpuTreeBuilder {
public:
  // `goesLeft` is a scratch byte per dataset row (shared by all nodes: every
  // node only touches its own rows). `pool` may be null (serial build).
  CpuTreeBuilder(const SplitRules &rules, const dt::EntryCodec &codec,
                 std::size_t featureCount, std::uint8_t *goesLeft, ThreadPool *pool,
                 const Options &options);

  // Build the subtree for the rows in [begin, begin + count) of `columns` and
  // store it in `slot`. With a pool, parts of the subtree are built by pool
  // tasks: call pool->waitIdle() before using the tree.
  void build(Columns columns, std::uint32_t begin, std::uint32_t count, int depth,
             std::unique_ptr<Node> &slot);

private:
  void buildNode(Columns columns, std::uint32_t begin, std::uint32_t count, int depth,
                 std::vector<std::uint32_t> classCounts, std::unique_ptr<Node> &slot);
  dt::CutCandidate scanFeature(const dt::Entry *entries, std::uint32_t count,
                               const std::uint32_t *total, double parentWeighted,
                               std::uint32_t minChild) const;
  template <int FixedK>
  dt::CutCandidate scanFeatureK(const dt::Entry *entries, std::uint32_t count,
                                const std::uint32_t *total, double parentWeighted,
                                std::uint32_t minChild) const;
  void partitionFeature(dt::Entry *entries, std::uint32_t count) const;

  const SplitRules &rules_;
  dt::EntryCodec codec_;
  std::size_t featureCount_;
  int classCount_;
  std::uint8_t *goesLeft_;
  ThreadPool *pool_;
  std::size_t minRowsForNodeTask_;
  std::size_t minRowsForFeatureParallel_;
};
