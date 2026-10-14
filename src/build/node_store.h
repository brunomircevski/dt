#pragma once

#include "core/tree.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace dt {

// Where the builders put nodes while they grow a tree. add() may be called
// from many threads at once and never moves existing nodes (they live in
// fixed-size chunks), so a builder can keep working on a node while other
// threads add theirs. Every thread takes a whole chunk of ids at a time, so
// threads neither contend on one counter or lock nor write to the same
// memory, and each chunk is allocated (and its pages placed) by the thread
// that fills it. Ids depend
// on thread timing and may have gaps; toTree() renumbers the nodes in
// preorder, so the result does not depend on them.
class NodeStore {
public:
  // `maxNodes`: an upper bound (a tree on n rows has at most 2n - 1 nodes).
  NodeStore(std::size_t classCount, std::size_t maxNodes);

  // New node with the given class histogram (sets count and label). The
  // first node added is the root (id 0).
  std::uint32_t add(const std::uint32_t *counts);

  Node &node(std::uint32_t id) { return chunk(id).nodes[id & kChunkMask]; }
  const std::uint32_t *counts(std::uint32_t id) {
    return chunk(id).counts.get() + std::size_t{id & kChunkMask} * classCount_;
  }

  // The tree below node 0 (the first node added), in preorder. `pool` may
  // be null.
  void toTree(Tree &tree, ThreadPool *pool);

private:
  static constexpr unsigned kChunkBits = 12;
  static constexpr std::uint32_t kChunkSize = 1u << kChunkBits;
  static constexpr std::uint32_t kChunkMask = kChunkSize - 1;
  static constexpr std::uint32_t kBlockSize = kChunkSize; // ids a thread takes at once
  static constexpr std::size_t kMaxThreads = 4096; // bounds the unused ids

  struct Chunk {
    std::unique_ptr<Node[]> nodes;
    std::unique_ptr<std::uint32_t[]> counts;
  };

  Chunk &chunk(std::uint32_t id) { return *chunks_[id >> kChunkBits].load(std::memory_order_acquire); }
  Chunk &ensureChunk(std::size_t index);
  std::uint32_t nextId();

  std::size_t classCount_;
  std::size_t maxChunks_;
  std::unique_ptr<std::atomic<Chunk *>[]> chunks_;
  std::vector<std::unique_ptr<Chunk>> owned_; // guarded by mutex_
  std::mutex mutex_;
  std::atomic<std::uint32_t> size_{0}; // ids handed out in blocks
  std::uint64_t serial_;                // tells this store from earlier ones
};

} // namespace dt
