#include "build/node_store.h"

#include <algorithm>
#include <stdexcept>

namespace dt {

namespace {
std::atomic<std::uint64_t> nextSerial{1};
} // namespace

NodeStore::NodeStore(std::size_t classCount, std::size_t maxNodes)
    : classCount_(classCount),
      maxChunks_((maxNodes + kBlockSize * kMaxThreads) / kChunkSize + 1),
      chunks_(new std::atomic<Chunk *>[maxChunks_]), serial_(nextSerial.fetch_add(1)) {
  for (std::size_t index = 0; index < maxChunks_; ++index) {
    chunks_[index].store(nullptr, std::memory_order_relaxed);
  }
}

NodeStore::Chunk &NodeStore::ensureChunk(std::size_t index) {
  if (index >= maxChunks_) {
    throw std::logic_error("NodeStore: more nodes than the given maximum");
  }
  Chunk *existing = chunks_[index].load(std::memory_order_acquire);
  if (existing) {
    return *existing;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  existing = chunks_[index].load(std::memory_order_acquire);
  if (!existing) {
    auto fresh = std::make_unique<Chunk>();
    fresh->nodes.reset(new Node[kChunkSize]);
    fresh->counts.reset(new std::uint32_t[std::size_t{kChunkSize} * classCount_]);
    existing = fresh.get();
    owned_.push_back(std::move(fresh));
    chunks_[index].store(existing, std::memory_order_release);
  }
  return *existing;
}

std::uint32_t NodeStore::nextId() {
  struct Cursor {
    std::uint64_t store = 0;
    std::uint32_t next = 0;
    std::uint32_t end = 0;
  };
  thread_local Cursor cursor;
  if (cursor.store != serial_ || cursor.next == cursor.end) {
    const std::uint32_t first = size_.fetch_add(kBlockSize, std::memory_order_relaxed);
    cursor = {serial_, first, first + kBlockSize};
  }
  return cursor.next++;
}

std::uint32_t NodeStore::add(const std::uint32_t *counts) {
  const std::uint32_t id = nextId();
  Chunk &target = ensureChunk(id >> kChunkBits);
  Node &node = target.nodes[id & kChunkMask];
  std::uint32_t *own = target.counts.get() + std::size_t{id & kChunkMask} * classCount_;
  std::copy(counts, counts + classCount_, own);
  node = Node{};
  for (std::size_t k = 0; k < classCount_; ++k) {
    node.count += counts[k];
  }
  node.label = majorityClass(counts, classCount_);
  return id;
}

void NodeStore::toTree(Tree &tree) {
  if (size_.load() == 0) {
    tree.nodes.clear();
    tree.classCounts.clear();
    return;
  }
  tree.assignPreorder(
      0, [&](std::uint32_t id) -> const Node & { return node(id); },
      [&](std::uint32_t id) { return counts(id); });
}

} // namespace dt
