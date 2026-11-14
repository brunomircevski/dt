#include "build/cpu_builder.h"
#include "build/cpu_sweep.h"

#include "core/trace.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <deque>
#include <numeric>
#include <optional>
#include <thread>
#include <utility>

#if defined(__SSE2__)
#include <immintrin.h>
#endif

namespace dt {

namespace {

// Nodes with at least this many rows are partitioned out of place, if the
// builder has a spare column slot (Columns::slots).
constexpr std::uint32_t kOutOfPlaceRows = 1u << 15;

// Such nodes with at least 1 / kRawMarkShare of all rows mark every row id
// from the winner's raw values, on all threads (markByValue()); smaller ones
// mark their own rows on the splitting thread alone, about 2 ns per row.
// 4 to 32 built the same within noise (HIGGS, SUSY, chess_100m, 48 threads);
// 2 was 20% slower on dell chess_100m (marks of the 2nd level on one thread).
constexpr std::size_t kRawMarkShare = 8;

// Nodes with at least this many rows are partitioned out of place in blocks
// of kPartitionBlock entries by all threads (partitionInBlocks()) when they
// have fewer free slots than threads (and columns to move), so that the
// slots do not limit the threads; their children are not swept on the way.
constexpr std::uint32_t kBlockedRows = 1u << 20;
constexpr std::uint32_t kPartitionBlock = 1u << 14;
// waitNonZero(): pause loops it spins, how long it then yields its CPU, and
// how long it then sleeps between looks.
constexpr unsigned kSpinBeforeYield = 2048;
constexpr std::chrono::milliseconds kYieldBeforeSleep{2};
constexpr std::chrono::microseconds kSleepBetweenLooks{50};

// Waits until value is not 0 and returns it. Not with atomic::wait(): libstdc++
// (13) notifies only if it sees a waiter, and after a release store that check
// can run before the store is visible (x86 lets a load pass an earlier store),
// so the waiter slept on the old value for good (about 1 in 40 --multicore runs
// on 1G rows hung on a 48-thread Xeon). A seq_cst store closes that but waits for the
// partition's streaming stores (ECBDL 15M C4.5 7% slower). So it spins briefly,
// then yields its CPU (the thread it waits for may be descheduled, and 47
// spinning threads would keep it off a CPU), then sleeps between looks: a
// futex sleep right after the spin made a build on 100M rows 2x slower
// (waking is slow).
std::uint32_t waitNonZero(const std::atomic<std::uint32_t> &value) {
  std::uint32_t seen;
  unsigned spin = 0;
  std::chrono::steady_clock::time_point yieldingSince;
  while ((seen = value.load(std::memory_order_acquire)) == 0) {
    if (spin < kSpinBeforeYield) {
      ++spin;
#if defined(__SSE2__)
      _mm_pause();
#endif
      continue;
    }
    const auto now = std::chrono::steady_clock::now();
    if (spin == kSpinBeforeYield) {
      ++spin;
      yieldingSince = now;
    }
    if (now - yieldingSince < kYieldBeforeSleep) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(kSleepBetweenLooks);
    }
  }
  return seen;
}

// Such nodes are also partitioned in blocks when their columns to move would
// keep at most 1 / kFewColumns of the threads busy (one thread per column),
// although the blocks lose the fused sweeps. With 2 features (1 column to
// move) one thread moved all the rows of the root and its children while the
// others waited, and a single thread's throughput drops as the columns grow:
// on 1G rows the build took 30-48 s instead of 5 (48 threads, AMD EPYC and
// Intel Xeon; 2.3x faster on 100M rows). On 48 threads, 12 columns (airline) were 0-15%
// slower in blocks, 17 and 27 (SUSY, HIGGS) up to 50%: they keep the fused
// sweeps. Between 1 and 12 columns the cut-off is not measured.
constexpr std::size_t kFewColumns = 8;

// Two-class nodes with at least this many rows whose sweeps were not fused
// into the parent's partition (the root, children of nodes partitioned in
// blocks) and with fewer features than threads are swept in chunks of
// kSweepChunk cuts by all threads (sweepInChunks()), not a feature per thread.
constexpr std::uint32_t kChunkedSweepRows = 1u << 20;
constexpr std::uint32_t kSweepChunk = 1u << 15;

// Children with at least this many rows are swept during the out-of-place
// partition of their parent, through staging buffers that are flushed when
// they hold kStageEntries (partitionAndSweep()).
constexpr std::uint32_t kSweptChildRows = 1u << 12;
constexpr std::uint32_t kStageEntries = 1024;

// The partitions move entries as one 64-bit word: GCC copies an Entry field
// by field (the float through an SSE register), two loads and two stores.
static_assert(std::endian::native == std::endian::little, "Entry::packed is the high half");
std::uint64_t entryWord(const Entry &entry) { return std::bit_cast<std::uint64_t>(entry); }
std::uint32_t packedOf(std::uint64_t word) { return static_cast<std::uint32_t>(word >> 32); }
void putEntry(Entry *at, std::uint64_t word) { std::memcpy(at, &word, sizeof word); }

// A non-temporal store: the columns of a big node do not fit in the caches,
// so their stores bypass them (no read of the target line first, no eviction
// of data still needed). An _mm_sfence() makes them visible to other threads.
void streamEntry(Entry *at, std::uint64_t word) {
#if defined(__SSE2__)
  _mm_stream_si64(reinterpret_cast<long long *>(at), static_cast<long long>(word));
#else
  putEntry(at, word);
#endif
}

// 1 if the entry's row goes left (its bit in the splitting thread's bitmap).
std::uint32_t leftBit(const std::uint64_t *leftBits, EntryCodec codec, std::uint64_t word) {
  const std::uint32_t row = codec.row(packedOf(word));
  return static_cast<std::uint32_t>(leftBits[row >> 6] >> (row & 63)) & 1u;
}

// The bitmap of the rows that go left (a bit per row id, `rows` row ids) of
// the big nodes this thread splits. Each thread has its own: a shared one
// had the rows of every node that is being split in each 64-byte line, so
// every mark took lines that the threads partitioning other nodes were
// reading (atomic marks cost 24-54 ns per row on 48 threads instead of 2).
// A thread uses it from the mark to the end of the node's partition, and
// never splits another node in between (ThreadPool::parallelFor runs only
// its own items on the calling thread); the partition's helpers only read
// it. Not initialised: a mark writes the bit of each of the node's rows.
std::uint64_t *threadLeftBits(std::size_t rows) {
  thread_local std::vector<std::uint64_t, UninitializedAllocator<std::uint64_t>> bits;
  if (bits.size() < rows / 64 + 1) {
    bits.resize(rows / 64 + 1);
  }
  return bits.data();
}

// Bit `row` of `bits` = values[row] <= leftValue, for the rows of the words
// [fromWord, toWord) (of `rows` rows): sequential reads, whole words
// written, no atomics. The node's left rows are exactly its rows with a value
// up to the cut's leftValue (the next value is larger; the entries hold
// value + 0.0f, and -0.0 <= x compares as 0.0 <= x); the bits of rows of
// other nodes are not read.
void markByValue(const float *values, std::size_t rows, float leftValue, std::uint64_t *bits,
                 std::size_t fromWord, std::size_t toWord) {
  for (std::size_t word = fromWord; word < toWord; ++word) {
    const std::size_t first = word * 64;
    const std::size_t length = std::min<std::size_t>(64, rows - std::min(rows, first));
    std::uint64_t mask = 0;
#if defined(__SSE2__)
    if (length == 64) { // four values per compare (GCC does not vectorize the loop)
      const __m128 limit = _mm_set1_ps(leftValue);
      for (std::size_t quad = 0; quad < 16; ++quad) {
        const __m128 four = _mm_loadu_ps(values + first + 4 * quad);
        mask |= std::uint64_t{static_cast<std::uint32_t>(_mm_movemask_ps(_mm_cmple_ps(four, limit)))}
                << (4 * quad);
      }
      bits[word] = mask;
      continue;
    }
#endif
    for (std::size_t bit = 0; bit < length; ++bit) {
      mask |= std::uint64_t{values[first + bit] <= leftValue} << bit;
    }
    bits[word] = mask;
  }
}

} // namespace

void fillCountTables(std::size_t size, CountTable *xlog, CountTable *inverse, ThreadPool *pool) {
  if (xlog) {
    xlog->resize(size);
  }
  if (inverse) {
    inverse->resize(size);
  }
  constexpr std::size_t kChunks = 64;
  parallelFor(pool, kChunks, [&](std::size_t chunk) {
    const std::size_t end = size * (chunk + 1) / kChunks;
    for (std::size_t c = size * chunk / kChunks; c < end; ++c) {
      if (xlog) {
        (*xlog)[c] = xlog2x(static_cast<double>(c));
      }
      if (inverse) {
        (*inverse)[c] = 1.0 / static_cast<double>(c); // inverse[0] is never used
      }
    }
  });
}

namespace {

CountTables defaultCountTables() {
  static const auto tables = [] {
    std::pair<CountTable, CountTable> both;
    fillCountTables(kLogTableSize, &both.first, &both.second, nullptr);
    return both;
  }();
  return {tables.first.data(), tables.second.data(), kLogTableSize};
}

} // namespace

CpuTreeBuilder::CpuTreeBuilder(const SplitRules &rules, const EntryCodec &codec,
                               std::size_t featureCount, NodeStore &store,
                               std::uint8_t *goesLeft, ThreadPool *pool,
                               const ParallelOptions &options, CountTables tables,
                               std::uint32_t *leafOfRow)
    : rules_(rules), codec_(codec), featureCount_(featureCount),
      classCount_(rules.classCount()), store_(store), goesLeft_(goesLeft), pool_(pool),
      options_(options), tables_(tables.size > 0 ? tables : defaultCountTables()),
      leafOfRow_(leafOfRow) {}

void CpuTreeBuilder::recordLeaf(const Entry *entries, std::uint32_t count,
                                std::uint32_t leaf) const {
  if (!leafOfRow_) {
    return;
  }
  trace::Doing doing(trace::Work::Leaves);
  auto range = [&](std::uint32_t from, std::uint32_t to) {
    for (std::uint32_t index = from; index < to; ++index) {
      leafOfRow_[codec_.row(entries[index].packed)] = leaf;
    }
  };
  if (pool_ && count >= options_.featureParallelRows) {
    const std::size_t blocks = 4 * (pool_->workerCount() + 1);
    pool_->parallelFor(blocks, [&](std::size_t block) {
      range(static_cast<std::uint32_t>(std::size_t{count} * block / blocks),
            static_cast<std::uint32_t>(std::size_t{count} * (block + 1) / blocks));
    });
  } else {
    range(0, count);
  }
}

namespace {

// A node's column of `feature`.
Entry *columnOf(const Columns &columns, const Subtree &item, std::uint32_t feature) {
  return columns.slot(item.slots ? (*item.slots)[feature] : feature);
}

} // namespace

void CpuTreeBuilder::grow(const Columns &columns, Subtree root) {
  trace::Doing doing(trace::Work::Build);
  std::vector<Subtree> stack;
  stack.push_back(std::move(root));
  while (!stack.empty()) {
    Subtree item = std::move(stack.back());
    stack.pop_back();
    split(columns, item, stack);
  }
}

void CpuTreeBuilder::split(const Columns &columns, Subtree &item, std::vector<Subtree> &stack) {
  const std::uint32_t begin = item.begin;
  const std::uint32_t count = item.count;
  const std::uint32_t *counts = store_.counts(item.node);
  // Every column of item.features holds the node's rows in its range.
  const Entry *rows = columnOf(columns, item, item.features.front()) + begin;
  if (rules_.isTerminal(counts, count, item.depth)) {
    recordLeaf(rows, count, item.node);
    return;
  }

  // Features whose values are all (nearly) equal here cannot split this node
  // or any node below it: the range is sorted, so first vs last tells.
  const double gap = rules_.minValueGap();
  std::vector<std::uint32_t> &features = item.features;
  std::erase_if(features, [&](std::uint32_t feature) {
    if (item.cuts) {
      return item.cuts->constant[feature] != 0;
    }
    const Entry *entries = columnOf(columns, item, feature) + begin;
    return !isCut(entries[0].value, entries[count - 1].value, gap);
  });
  if (features.empty()) {
    recordLeaf(rows, count, item.node);
    return;
  }

  // 1. Best cut of every feature: one sweep over its sorted range, which also
  //    gives the class counts left of that cut (unless the parent's partition
  //    did these sweeps).
  const bool wide = pool_ && count >= options_.featureParallelRows;
  ThreadPool *const widePool = wide ? pool_ : nullptr;
  const std::uint32_t minChild = rules_.minChildRows(count);
  const double parentWeighted =
      weightedImpurity(counts, classCount_, count, rules_.criterion(), rules_.logTable());
  const std::size_t classes = static_cast<std::size_t>(classCount_);
  // Reused per thread (no allocation per node). Named references, because a
  // thread_local used inside the parallelFor lambda would be each worker's own.
  thread_local std::vector<CutCandidate> threadCuts;
  thread_local std::vector<std::uint32_t> threadLefts;
  std::vector<CutCandidate> &cuts = threadCuts;
  std::vector<std::uint32_t> &cutLefts = threadLefts;
  cuts.assign(featureCount_, CutCandidate{});
  cutLefts.resize(featureCount_ * classes);
  if (item.cuts) {
    for (const std::uint32_t feature : features) {
      cuts[feature] = item.cuts->cuts[feature];
      std::copy_n(item.cuts->lefts.data() + feature * classes, classes,
                  cutLefts.data() + feature * classes);
    }
    item.cuts.reset();
  } else if (pool_ && classCount_ == 2 && count >= kChunkedSweepRows && count < tables_.size &&
             count >= 2 * minChild &&
             features.size() < pool_->workerCount() + 1) {
    // Fewer features than threads: chunks of every column on all threads.
    trace::Doing doing(trace::Work::Sweep);
    sweepInChunks(columns, item, counts, parentWeighted, minChild, cuts.data(), cutLefts.data());
  } else {
    trace::Doing doing(trace::Work::Sweep);
    parallelFor(widePool, features.size(), [&](std::size_t index) {
      const std::uint32_t feature = features[index];
      cuts[feature] = scanFeature(columnOf(columns, item, feature) + begin, count, counts,
                                  parentWeighted, minChild, cutLefts.data() + feature * classes);
    });
  }

  // 2. Pick the feature (CART or C4.5 rule).
  const SplitRules::Decision decision = rules_.choose(cuts.data(), featureCount_, count);
  if (decision.feature < 0) {
    recordLeaf(rows, count, item.node);
    return;
  }
  const std::uint32_t leftCount = decision.leftCount;
  const Entry *winner =
      columnOf(columns, item, static_cast<std::uint32_t>(decision.feature)) + begin;

  // 3. The children's class counts: the winner's sweep counted the left side.
  thread_local std::vector<std::uint32_t> threadChildCounts;
  std::vector<std::uint32_t> &childCounts = threadChildCounts;
  childCounts.resize(2 * classes);
  std::uint32_t *leftCounts = childCounts.data();
  std::uint32_t *rightCounts = leftCounts + classes;
  std::copy_n(cutLefts.data() + static_cast<std::size_t>(decision.feature) * classes, classes,
              leftCounts);
  for (std::size_t k = 0; k < classes; ++k) {
    rightCounts[k] = counts[k] - leftCounts[k];
  }
  const std::uint32_t leftId = store_.add(leftCounts);
  const std::uint32_t rightId = store_.add(rightCounts);
  Node &node = store_.node(item.node);
  node.feature = decision.feature;
  node.threshold = decision.threshold;
  node.left = leftId;
  node.right = rightId;

  // 4. Unless both children are leaves (often the case deep in the tree), mark
  //    the rows that go left and stable-partition every other column so both
  //    children stay sorted (only the side that is grown further, if the other
  //    child is a leaf).
  const std::uint32_t rightCount = count - leftCount;
  const bool leftTerminal = rules_.isTerminal(leftCounts, leftCount, item.depth + 1);
  const bool rightTerminal = rules_.isTerminal(rightCounts, rightCount, item.depth + 1);
  if (leftTerminal) {
    recordLeaf(winner, leftCount, leftId);
  }
  if (rightTerminal) {
    recordLeaf(winner + leftCount, rightCount, rightId);
  }
  if (leftTerminal && rightTerminal) {
    return;
  }
  const std::size_t blocks = wide ? 4 * (pool_->workerCount() + 1) : 1;
  // Out of place needs a slot that holds none of the node's columns.
  const bool outOfPlace = columns.train && count >= kOutOfPlaceRows &&
                          (columns.slots > 0 ? columns.slots : featureCount_) > features.size();
  // The CPU growers mark in the thread's bitmap, also in place: a byte
  // array shared by all threads had 64 rows of every node split at once in
  // each cache line. On one thread the bytes stay for in place: a byte is
  // stored without loading its line first, a bit is not (HIGGS full depth
  // 3% faster). The CUDA backend's hand-offs mark goesLeft_.
  std::uint64_t *const leftBits =
      columns.train && (outOfPlace || pool_) ? threadLeftBits(columns.stride) : nullptr;
  std::optional<trace::Doing> marking(std::in_place, trace::Work::Mark);
  if (outOfPlace && count >= columns.stride / kRawMarkShare) {
    const float *values = columns.train->column(static_cast<std::size_t>(decision.feature));
    const float leftValue = winner[leftCount - 1].value;
    const std::size_t words = columns.stride / 64 + 1;
    parallelFor(widePool, blocks, [&](std::size_t block) {
      markByValue(values, columns.stride, leftValue, leftBits, words * block / blocks,
                  words * (block + 1) / blocks);
    });
  } else if (leftBits) {
    for (std::uint32_t index = 0; index < leftCount; ++index) {
      const std::uint32_t row = codec_.row(winner[index].packed);
      leftBits[row >> 6] |= std::uint64_t{1} << (row & 63);
    }
    for (std::uint32_t index = leftCount; index < count; ++index) {
      const std::uint32_t row = codec_.row(winner[index].packed);
      leftBits[row >> 6] &= ~(std::uint64_t{1} << (row & 63));
    }
  } else {
    parallelFor(widePool, blocks, [&](std::size_t block) {
      const std::uint32_t end =
          static_cast<std::uint32_t>(std::size_t{count} * (block + 1) / blocks);
      for (std::uint32_t index = static_cast<std::uint32_t>(std::size_t{count} * block / blocks);
           index < end; ++index) {
        goesLeft_[codec_.row(winner[index].packed)] = index < leftCount;
      }
    });
  }
  marking.reset();
  const Keep keep = leftTerminal ? Keep::Right : rightTerminal ? Keep::Left : Keep::Both;
  // Big children are swept while the columns are partitioned; with more than
  // two classes only on a pool, where the second read of the columns costs
  // memory bandwidth (MNIST8M 21-23% faster on 48 threads). On one thread it
  // depends on the CPU (MNIST8M 3% slower on an EPYC 9255, 6% faster on a
  // Xeon 6542Y), so there they keep main's separate sweeps.
  ChildSweep sweeps[2];
  ChildSweep *swept[2] = {nullptr, nullptr};
  std::shared_ptr<FeatureCuts> childCuts[2];
  // In blocks only if the free slots cannot keep the threads busy, as the
  // blocks lose the fused sweeps.
  const std::size_t freeSlots =
      outOfPlace ? (columns.slots > 0 ? columns.slots : featureCount_) - features.size() : 0;
  const std::size_t moved = features.size() - 1; // every column but the winner's
  const std::size_t threads = pool_ ? pool_->workerCount() + 1 : 1;
  const bool blocked = outOfPlace && pool_ && count >= kBlockedRows &&
                       (freeSlots < std::min(moved, threads) ||
                        (moved > 0 && moved * kFewColumns <= threads));
  if (outOfPlace && !blocked && (classCount_ == 2 || pool_)) {
    for (int child = 0; child < 2; ++child) {
      const std::uint32_t *total = store_.counts(child == 0 ? leftId : rightId);
      const std::uint32_t rows = child == 0 ? leftCount : rightCount;
      const std::uint32_t childMin = rules_.minChildRows(rows);
      // The stage keeps up to childMin + 2 entries for the sweep (see
      // partitionAndSweep()); TwoClassSweep needs the count tables.
      if ((child == 0 ? leftTerminal : rightTerminal) || rows < kSweptChildRows ||
          (classCount_ == 2 && rows >= tables_.size) || childMin + 64 > kStageEntries) {
        continue;
      }
      childCuts[child] = std::make_shared<FeatureCuts>(featureCount_, classes);
      sweeps[child] = {rows,
                       total,
                       weightedImpurity(total, classCount_, rows, rules_.criterion(),
                                        rules_.logTable()),
                       childMin,
                       childCuts[child].get()};
      swept[child] = &sweeps[child];
    }
  }
  std::optional<trace::Doing> partitioning(
      std::in_place, swept[0] || swept[1] ? trace::Work::PartitionSweep : trace::Work::Partition);
  std::shared_ptr<const SlotMap> slots =
      partition(columns, item, decision.feature, leftCount, keep, leftBits, outOfPlace, blocked,
                swept);
  partitioning.reset();

  // 5. Children. A big left child becomes a pool task so idle threads can pick
  //    it up; this thread continues with the right one.
  Subtree left{leftId, begin, leftCount, item.depth + 1, {}, slots, std::move(childCuts[0])};
  Subtree right{rightId, begin + leftCount, rightCount, item.depth + 1, {}, std::move(slots),
                std::move(childCuts[1])};
  if (!leftTerminal && !rightTerminal) {
    left.features = features;
  }
  (rightTerminal ? left : right).features = std::move(features);
  if (!rightTerminal) {
    stack.push_back(std::move(right));
  }
  if (!leftTerminal) {
    if (pool_ && count >= options_.nodeTaskRows) {
      pool_->submit([this, columns, child = std::move(left)]() mutable {
        grow(columns, std::move(child));
      });
    } else {
      stack.push_back(std::move(left));
    }
  }
}

std::shared_ptr<const SlotMap> CpuTreeBuilder::partition(const Columns &columns,
                                                         const Subtree &item, int winner,
                                                         std::uint32_t leftCount, Keep keep,
                                                         const std::uint64_t *leftBits,
                                                         bool outOfPlace, bool blocked,
                                                         ChildSweep *const *children) {
  const std::uint32_t count = item.count;
  if (outOfPlace) {
    // Every column but the winner's into a slot that holds none of the
    // node's columns (see Columns): the free ones first, then those that the
    // moved columns leave. The winner's column is already in order.
    constexpr std::uint32_t kNone = ~0u;
    const std::vector<std::uint32_t> &features = item.features;
    const std::size_t slotCount = columns.slots > 0 ? columns.slots : featureCount_;
    auto slots = std::make_shared<SlotMap>(featureCount_);
    if (item.slots) {
      *slots = *item.slots;
    } else {
      std::iota(slots->begin(), slots->end(), 0u);
    }
    std::vector<bool> taken(slotCount, false);
    for (const std::uint32_t feature : features) {
      taken[(*slots)[feature]] = true;
    }
    std::deque<std::pair<std::uint32_t, std::uint32_t>> free; // (slot, index of the column leaving it)
    for (std::uint32_t slot = 0; slot < slotCount; ++slot) {
      if (!taken[slot]) {
        free.emplace_back(slot, kNone);
      }
    }
    std::vector<std::uint32_t> from(features.size());
    std::vector<std::uint32_t> waitFor(features.size(), kNone);
    for (std::size_t index = 0; index < features.size(); ++index) {
      const std::uint32_t feature = features[index];
      from[index] = (*slots)[feature];
      if (static_cast<int>(feature) == winner) {
        continue;
      }
      const auto [slot, leaving] = free.front();
      free.pop_front();
      waitFor[index] = leaving;
      free.emplace_back(from[index], static_cast<std::uint32_t>(index));
      (*slots)[feature] = slot;
    }
    if (blocked) {
      std::vector<const Entry *> sources;
      std::vector<Entry *> targets;
      std::vector<std::uint32_t> moveOf(features.size(), kNone);
      std::vector<std::uint32_t> moveWaits;
      for (std::size_t index = 0; index < features.size(); ++index) {
        if (static_cast<int>(features[index]) != winner) {
          moveOf[index] = static_cast<std::uint32_t>(sources.size());
          sources.push_back(columns.slot(from[index]) + item.begin);
          targets.push_back(columns.slot((*slots)[features[index]]) + item.begin);
          moveWaits.push_back(waitFor[index] == kNone ? kNone : moveOf[waitFor[index]]);
        }
      }
      partitionInBlocks(sources, targets, moveWaits, count, leftCount, keep, leftBits);
      return slots;
    }
    // Columns are taken in order, so the one a column waits for has started.
    std::unique_ptr<std::atomic<std::uint32_t>[]> done(
        new std::atomic<std::uint32_t>[features.size()]());
    const bool sweep = children[0] || children[1];
    const bool gini = rules_.criterion() == Criterion::Gini;
    parallelFor(pool_ && count >= options_.featureParallelRows ? pool_ : nullptr,
                features.size(), [&](std::size_t index) {
                  const std::uint32_t feature = features[index];
                  const bool isWinner = static_cast<int>(feature) == winner;
                  const Entry *source = columns.slot(from[index]) + item.begin;
                  Entry *target = isWinner ? nullptr : columns.slot((*slots)[feature]) + item.begin;
                  if (waitFor[index] != kNone) {
                    trace::Doing waiting(trace::Work::Idle);
                    waitNonZero(done[waitFor[index]]);
                  }
                  if (sweep) {
                    // CART: Gini, values must differ; C4.5: entropy, gap.
                    if (gini) {
                      partitionAndSweepFor<Criterion::Gini, false>(
                          source, target, count, leftCount, keep, leftBits, isWinner,
                          feature, children);
                    } else {
                      partitionAndSweepFor<Criterion::Entropy, true>(
                          source, target, count, leftCount, keep, leftBits, isWinner,
                          feature, children);
                    }
                  } else if (!isWinner) {
                    partitionFeatureOutOfPlace(source, target, count, leftCount, keep,
                                               leftBits);
                  }
                  done[index].store(1, std::memory_order_release);
                });
    return slots;
  }
  thread_local std::vector<std::uint32_t> threadOthers;
  std::vector<std::uint32_t> &others = threadOthers;
  others.clear();
  for (std::uint32_t feature : item.features) {
    if (static_cast<int>(feature) != winner) {
      others.push_back(feature);
    }
  }
  // In place, for nodes below kOutOfPlaceRows (and the GPU's hand-offs, below
  // --handoff-rows): a per-thread buffer (stays in cache), one column per
  // thread for wide nodes.
  auto partitionOne = [&](std::size_t index) {
    thread_local std::vector<Entry> buffer;
    if (keep == Keep::Both && buffer.size() < count) {
      buffer.resize(count);
    }
    Entry *entries = columnOf(columns, item, others[index]) + item.begin;
    if (leftBits) {
      partitionFeature(entries, count, leftCount, buffer.data(), keep, [&](std::uint64_t word) {
        return leftBit(leftBits, codec_, word);
      });
    } else {
      partitionFeature(entries, count, leftCount, buffer.data(), keep, [&](std::uint64_t word) {
        return std::uint32_t{goesLeft_[codec_.row(packedOf(word))]};
      });
    }
  };
  parallelFor(pool_ && count >= options_.featureParallelRows ? pool_ : nullptr, others.size(),
              partitionOne);
  return item.slots;
}

// Moves `count` entries of `source` to `target` by leftBits: left rows to
// target[left], target[left + 1], ..., right rows to target[right], ... (only
// those of `keep`'s side if one child is a leaf), with non-temporal stores
// (streamEntry(); the caller fences).
void CpuTreeBuilder::moveRows(const Entry *source, std::uint32_t count, Entry *target,
                              std::uint32_t left, std::uint32_t right, Keep keep,
                              const std::uint64_t *leftBits) const {
  const EntryCodec codec = codec_;
  if (keep == Keep::Both) {
    for (std::uint32_t index = 0; index < count; ++index) {
      const std::uint64_t word = entryWord(source[index]);
      const std::uint32_t isLeft = leftBit(leftBits, codec, word);
      // isLeft ? left : right, without a branch: GCC turned the conditional
      // into one, mispredicted on every other row of an even split.
      streamEntry(target + (right ^ ((left ^ right) & (0u - isLeft))), word);
      left += isLeft;
      right += 1 - isLeft;
    }
    return;
  }
  // One child: usually a lopsided split, so the branch is predictable.
  const std::uint32_t wanted = keep == Keep::Left;
  Entry *write = target + (keep == Keep::Left ? left : right);
  for (std::uint32_t index = 0; index < count; ++index) {
    const std::uint64_t word = entryWord(source[index]);
    if (leftBit(leftBits, codec, word) == wanted) {
      streamEntry(write++, word);
    }
  }
}

// Stable partition of `source` into `target`: left rows to [0, leftCount),
// right rows to [leftCount, count), in order.
void CpuTreeBuilder::partitionFeatureOutOfPlace(const Entry *source, Entry *target,
                                                std::uint32_t count, std::uint32_t leftCount,
                                                Keep keep, const std::uint64_t *leftBits) const {
  moveRows(source, count, target, 0, leftCount, keep, leftBits);
#if defined(__SSE2__)
  _mm_sfence(); // the stores are visible before another thread reads them
#endif
}

// partitionFeatureOutOfPlace() for the columns `sources` of a big node, into
// `targets`, in blocks of kPartitionBlock entries taken in order (column by
// column) by all threads. A block counts its left rows (which brings it into
// the cache), takes the left rows before it from the block before it (a
// chain: that block has started), publishes its own, and writes its rows. A
// column writes once every block of the column that leaves its target slot
// (waitFor, an earlier one) is done with it.
void CpuTreeBuilder::partitionInBlocks(const std::vector<const Entry *> &sources,
                                       const std::vector<Entry *> &targets,
                                       const std::vector<std::uint32_t> &waitFor,
                                       std::uint32_t count, std::uint32_t leftCount, Keep keep,
                                       const std::uint64_t *leftBits) const {
  const std::size_t columns = sources.size();
  const std::size_t blocks = (count + kPartitionBlock - 1) / kPartitionBlock;
  // Left rows up to and including each block, plus one (0: not yet known).
  std::unique_ptr<std::atomic<std::uint32_t>[]> lefts(
      new std::atomic<std::uint32_t>[columns * blocks]());
  std::unique_ptr<std::atomic<std::uint32_t>[]> read(new std::atomic<std::uint32_t>[columns]());
  const EntryCodec codec = codec_;
  pool_->parallelFor(columns * blocks, [&](std::size_t item) {
    const std::size_t column = item / blocks;
    const std::size_t block = item % blocks;
    const std::uint32_t start = static_cast<std::uint32_t>(block * kPartitionBlock);
    const std::uint32_t length = std::min(kPartitionBlock, count - start);
    const Entry *source = sources[column] + start;
    std::uint32_t mine = 0;
    for (std::uint32_t index = 0; index < length; ++index) {
      mine += leftBit(leftBits, codec, entryWord(source[index]));
    }
    std::uint32_t before = 0; // left rows of the column before this block
    if (block > 0) {
      before = waitNonZero(lefts[item - 1]) - 1; // (it is counting)
    }
    lefts[item].store(before + mine + 1, std::memory_order_release);
    if (waitFor[column] != ~0u) {
      std::atomic<std::uint32_t> &other = read[waitFor[column]];
      for (std::uint32_t value; (value = other.load(std::memory_order_acquire)) < blocks;) {
        other.wait(value, std::memory_order_acquire);
      }
    }
    moveRows(source, length, targets[column], before, leftCount + (start - before), keep,
             leftBits);
#if defined(__SSE2__)
    _mm_sfence(); // the stores are visible before another thread reads them
#endif
    // Done with its source (read twice: counting, then writing).
    if (read[column].fetch_add(1, std::memory_order_acq_rel) + 1 == blocks) {
      read[column].notify_all();
    }
  });
}

void CpuTreeBuilder::sweepInChunks(const Columns &columns, const Subtree &item,
                                   const std::uint32_t *total, double parentWeighted,
                                   std::uint32_t minChild, CutCandidate *cuts,
                                   std::uint32_t *lefts) const {
  // CART: Gini, values must differ; C4.5: entropy, gap.
  if (rules_.criterion() == Criterion::Gini) {
    sweepInChunksFor<Criterion::Gini, false>(columns, item, total, parentWeighted, minChild, cuts,
                                             lefts);
  } else {
    sweepInChunksFor<Criterion::Entropy, true>(columns, item, total, parentWeighted, minChild,
                                               cuts, lefts);
  }
}

// The sweep of a column is a chain: a cut is picked when its gain beats the
// best one so far by kTieEps, so near-ties go to the earlier cut. Every chunk
// of kSweepChunk cuts is swept on its own first (phase A), as if the column
// started there with a best gain L that is at most the sequential sweep's
// best b at that point (-inf, or see `floors`): it counts its class-1
// entries, takes the count before it from the chunk before it (a chain, as in
// partitionInBlocks()) and records its picks, whose gains grow by more than
// kTieEps each; a cut before the first pick is at most L + kTieEps <= b +
// kTieEps, one between two picks at most kTieEps above the earlier one. Then
// each column's chunks are joined in order (phase B) with the chain's best
// gain b:
//
// * a chunk whose picks are all at most b cannot beat b by kTieEps anywhere;
// * otherwise its cuts before its first pick p above b are at most kTieEps
//   above b. If p beats b by kTieEps, the sequential sweep picks p too (no
//   bound skips it), and from there on both make the same picks: the chunk's
//   own best stands;
// * else (b < gain(p) <= b + kTieEps) the cuts after p are swept again from b
//   until they pick a cut the chunk picked too (the same state again), or to
//   the chunk's end.
//
// So the result is the sequential sweep's, near-ties included.
template <Criterion Crit, bool HasGap>
void CpuTreeBuilder::sweepInChunksFor(const Columns &columns, const Subtree &item,
                                      const std::uint32_t *total, double parentWeighted,
                                      std::uint32_t minChild, CutCandidate *cuts,
                                      std::uint32_t *lefts) const {
  using Sweep = TwoClassSweep<Crit, HasGap, true>;
  using Pick = typename Sweep::Pick;
  const std::vector<std::uint32_t> &features = item.features;
  const std::uint32_t count = item.count;
  const std::uint32_t endCut = count - minChild + 1; // cuts [minChild, endCut)
  const std::size_t chunks = (endCut - minChild + kSweepChunk - 1) / kSweepChunk;
  struct Chunk {
    CutCandidate best;
    std::uint32_t left[2] = {0, 0};
    std::vector<Pick> picks;
  };
  std::vector<Chunk> found(features.size() * chunks);
  // Class-1 entries up to the end of each chunk, plus one (0: not yet known).
  std::unique_ptr<std::atomic<std::uint32_t>[]> ones(
      new std::atomic<std::uint32_t>[features.size() * chunks]());
  // The largest gain of a cut that the sequential sweep takes (or provably
  // skips) up to the end of each chunk, written before its `ones`.
  std::unique_ptr<double[]> floors(new double[features.size() * chunks]);
  const double gap = rules_.minValueGap();
  const LogTable logs = rules_.logTable();
  const SweepSetup setup = sweepSetup();
  const EntryCodec codec = codec_;
  auto cutsOf = [&](std::size_t chunk) {
    const std::uint32_t first = minChild + static_cast<std::uint32_t>(chunk) * kSweepChunk;
    return std::pair<std::uint32_t, std::uint32_t>{first, std::min(first + kSweepChunk, endCut)};
  };
  pool_->parallelFor(features.size() * chunks, [&](std::size_t index) {
    const std::size_t column = index / chunks;
    const std::size_t chunk = index % chunks;
    const Entry *entries = columnOf(columns, item, features[column]) + item.begin;
    const auto [first, end] = cutsOf(chunk);
    // The sweep at cut c has counted the class-1 entries of [0, c - 1).
    std::uint32_t mine = 0;
    for (std::uint32_t entry = first - 1; entry + 1 < end; ++entry) {
      mine += codec.cls(entries[entry].packed);
    }
    std::uint32_t before = 0;
    double floor = -INFINITY;
    if (chunk > 0) {
      before = waitNonZero(ones[index - 1]) - 1; // (it is counting)
      floor = floors[index - 1];
    } else {
      for (std::uint32_t entry = 0; entry + 1 < first; ++entry) {
        before += codec.cls(entries[entry].packed);
      }
    }
    // The chunk's last cut, unless the boundary-point shortcut skips it.
    const std::uint32_t last = end - 1;
    const Entry previous = entries[last - 1];
    const Entry current = entries[last];
    auto distinct = [&](float lower, float upper) {
      return HasGap ? isCut(lower, upper, gap) : lower < upper;
    };
    if (distinct(previous.value, current.value) &&
        !(codec.cls(previous.packed) == codec.cls(current.packed) &&
          (last < 2 || distinct(entries[last - 2].value, previous.value)) &&
          (last + 1 >= count || distinct(current.value, entries[last + 1].value)) &&
          last > minChild && last + minChild < count)) {
      const std::uint32_t lefts[2] = {last - (before + mine), before + mine};
      const std::uint32_t totals[2] = {total[0], total[1]};
      floor = std::max(floor, cutGain(totals, lefts, 2, count, last, parentWeighted, Crit, logs));
    }
    floors[index] = floor;
    ones[index].store(before + mine + 1, std::memory_order_release);
    Chunk &out = found[index];
    Sweep sweep(setup, count, total, parentWeighted, minChild);
    // A cut the sequential sweep takes, or provably skips, is at most kTieEps
    // above its best one so far (up to the rounding of best + kTieEps), so
    // the best gain at the start of this chunk is above floor - 2 kTieEps.
    // Starting there instead of at -inf drops only picks that cannot beat
    // it, and the bounds skip most of the chunk.
    sweep.restrict(first, end, before, floor - 2.0 * kTieEps, &out.picks);
    sweep.advance(entries, count);
    out.best = sweep.finish(out.left);
  });
  pool_->parallelFor(features.size(), [&](std::size_t column) {
    const Entry *entries = columnOf(columns, item, features[column]) + item.begin;
    CutCandidate best;
    std::uint32_t left[2] = {0, 0};
    std::uint32_t tries = 0; // C4.5: every cut with a threshold, once
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      const Chunk &own = found[column * chunks + chunk];
      tries += own.best.tries;
      const std::vector<Pick> &picks = own.picks;
      const auto above = std::find_if(picks.begin(), picks.end(),
                                      [&](const Pick &pick) { return pick.gain > best.gain; });
      if (above == picks.end()) {
        continue;
      }
      if (above->gain > best.gain + kTieEps) {
        best = own.best;
        std::copy_n(own.left, 2, left);
        continue;
      }
      Sweep sweep(setup, count, total, parentWeighted, minChild);
      sweep.restrict(above->cut + 1, cutsOf(chunk).second, above->ones, best.gain, nullptr,
                     picks.data() + (above - picks.begin()) + 1, picks.data() + picks.size());
      sweep.advance(entries, count);
      std::uint32_t again[2];
      const CutCandidate picked = sweep.finish(again);
      if (sweep.stopped()) {
        best = own.best;
        std::copy_n(own.left, 2, left);
      } else if (picked.valid()) {
        best = picked;
        std::copy_n(again, 2, left);
      }
    }
    best.tries = tries;
    const std::uint32_t feature = features[column];
    cuts[feature] = best;
    std::copy_n(left, 2, lefts + 2 * std::size_t{feature});
  });
}

// Stable partition by isLeft(entry word) (`leftCount` rows go left). The larger side is
// compacted in place (the write index never passes the read index), the
// smaller one waits in `buffer`, which needs min(left, right) + 1 entries. If
// only one child is grown further, only its rows are moved, in place: left
// rows forward, or right rows backward from the end.
template <class IsLeft>
void CpuTreeBuilder::partitionFeature(Entry *entries, std::uint32_t count,
                                      std::uint32_t leftCount, Entry *buffer, Keep keep,
                                      IsLeft isLeft) const {
  if (keep == Keep::Left) {
    std::uint32_t write = 0;
    for (std::uint32_t index = 0; index < count; ++index) {
      const std::uint64_t word = entryWord(entries[index]);
      putEntry(entries + write, word);
      write += isLeft(word);
    }
    return;
  }
  if (keep == Keep::Right) {
    std::uint32_t write = count; // right rows fill [write, count)
    for (std::uint32_t index = count; index-- > 0;) {
      const std::uint64_t word = entryWord(entries[index]);
      putEntry(entries + write - 1, word);
      write -= 1u - isLeft(word);
    }
    return;
  }
  if (leftCount < count - leftCount) {
    // Right rows compacted backward from the end, left rows into
    // buffer[1, leftCount], also backward.
    std::uint32_t rightWrite = count;
    std::uint32_t leftWrite = leftCount;
    for (std::uint32_t index = count; index-- > 0;) {
      const std::uint64_t word = entryWord(entries[index]);
      const std::uint32_t left = isLeft(word);
      putEntry(entries + rightWrite - 1, word);
      putEntry(buffer + leftWrite, word);
      rightWrite -= 1 - left;
      leftWrite -= left;
    }
    std::memcpy(entries, buffer + 1, leftCount * sizeof(Entry));
    return;
  }
  std::uint32_t leftWrite = 0;
  std::uint32_t rightWrite = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    const std::uint64_t word = entryWord(entries[index]);
    const std::uint32_t left = isLeft(word);
    putEntry(entries + leftWrite, word);
    putEntry(buffer + rightWrite, word);
    leftWrite += left;
    rightWrite += 1 - left;
  }
  std::memcpy(entries + leftWrite, buffer, rightWrite * sizeof(Entry));
}

template <Criterion Crit, bool HasGap>
void CpuTreeBuilder::partitionAndSweepFor(const Entry *source, Entry *target,
                                          std::uint32_t count, std::uint32_t leftCount,
                                          Keep keep, const std::uint64_t *leftBits,
                                          bool isWinner, std::uint32_t feature,
                                          ChildSweep *const *children) const {
  auto run = [&]<class Sweep>() {
    partitionAndSweep<Sweep>(source, target, count, leftCount, keep, leftBits, isWinner,
                             feature, children);
  };
  switch (classCount_) {
  case 2: return run.template operator()<TwoClassSweep<Crit, HasGap>>();
  case 3: return run.template operator()<KClassSweep<3, Crit>>();
  case 4: return run.template operator()<KClassSweep<4, Crit>>();
  case 5: return run.template operator()<KClassSweep<5, Crit>>();
  case 6: return run.template operator()<KClassSweep<6, Crit>>();
  case 7: return run.template operator()<KClassSweep<7, Crit>>();
  case 8: return run.template operator()<KClassSweep<8, Crit>>();
  default: return run.template operator()<KClassSweep<0, Crit>>();
  }
}

// Out-of-place partition of one column (see partitionFeatureOutOfPlace())
// that also sweeps the column for the children in `children`: entries go
// through a small staging buffer per child (in L1), where the child's sweep
// reads them, and from there to the target with non-temporal stores. So a big
// node's columns are read once per level instead of twice (partition, then
// the children's sweeps). The winner's column is already in order: it stays
// where it is (no target), and the sweeps read it there. Sweep: TwoClassSweep
// or KClassSweep, as partitionAndSweepFor() picks it.
template <class Sweep>
void CpuTreeBuilder::partitionAndSweep(const Entry *source, Entry *target, std::uint32_t count,
                                       std::uint32_t leftCount, Keep keep,
                                       const std::uint64_t *leftBits, bool isWinner,
                                       std::uint32_t feature,
                                       ChildSweep *const *children) const {
  const SweepSetup setup = sweepSetup();
  // --profile: the sweeps count as Sweep, the rest as PartitionSweep.
  const bool traced = trace::on();
  const bool kept[2] = {keep != Keep::Right, keep != Keep::Left};
  const std::uint32_t childBegin[2] = {0, leftCount};
  const std::uint32_t childCount[2] = {leftCount, count - leftCount};
  const double gap = rules_.minValueGap();
  // A child whose values of this column are all (nearly) equal is not swept:
  // its split drops the feature (FeatureCuts::constant), as split() does
  // before its own sweeps. Its first and last entries tell; they are usually
  // within a few entries of the column's ends.
  auto constantIn = [&](int child) {
    if (isWinner) {
      const Entry *entries = source + childBegin[child];
      return !isCut(entries[0].value, entries[childCount[child] - 1].value, gap);
    }
    constexpr std::uint32_t kEndScan = 64;
    const std::uint32_t wanted = child == 0;
    const std::uint32_t scan = std::min(count, kEndScan);
    std::uint32_t first = 0;
    while (first < scan && leftBit(leftBits, codec_, entryWord(source[first])) != wanted) {
      ++first;
    }
    std::uint32_t last = count - 1;
    while (last >= count - scan && leftBit(leftBits, codec_, entryWord(source[last])) != wanted) {
      --last;
    }
    return first < scan && last >= count - scan &&
           !isCut(source[first].value, source[last].value, gap);
  };
  std::optional<Sweep> sweeps[2];
  for (int child = 0; child < 2; ++child) {
    if (children[child] && !constantIn(child)) {
      const ChildSweep &info = *children[child];
      sweeps[child].emplace(setup, info.count, info.total, info.parentWeighted, info.minChild);
    }
  }

  if (isWinner) { // in order where it is (no target)
    trace::Doing sweeping(trace::Work::Sweep);
    for (int child = 0; child < 2; ++child) {
      if (kept[child] && sweeps[child]) {
        sweeps[child]->advance(source + childBegin[child], childCount[child]);
      }
    }
  } else {
    // Stage of one child: its entries [base, base + size) are in `data`
    // (the first ones only kept for the sweep), [0, written) are in the
    // target.
    struct Stage {
      Entry *data;
      std::uint32_t base = 0;
      std::uint32_t size = 0;
      std::uint32_t written = 0;
    };
    constexpr std::uint32_t kCapacity = kStageEntries + 64;
    thread_local std::vector<Entry> memory(2 * kCapacity);
    Stage stages[2] = {{memory.data()}, {memory.data() + kCapacity}};
    auto flush = [&](int child) {
      Stage &stage = stages[child];
      Entry *to = target + childBegin[child];
      const std::uint32_t end = stage.base + stage.size;
      for (std::uint32_t index = stage.written; index < end; ++index) {
        streamEntry(to + index, entryWord(stage.data[index - stage.base]));
      }
      stage.written = end;
      std::uint32_t from = end;
      if (sweeps[child]) {
        // The sweep reads entry i at window[i]: the stage, shifted by base.
        const Entry *window = reinterpret_cast<const Entry *>(
            reinterpret_cast<std::uintptr_t>(stage.data) - std::uintptr_t{stage.base} * sizeof(Entry));
        std::optional<trace::Doing> sweeping;
        if (traced) {
          sweeping.emplace(trace::Work::Sweep);
        }
        sweeps[child]->advance(window, end);
        sweeping.reset();
        from = std::min(sweeps[child]->from(), end);
      }
      std::memmove(stage.data, stage.data + (from - stage.base), (end - from) * sizeof(Entry));
      stage.size = end - from;
      stage.base = from;
    };
    const EntryCodec codec = codec_;
    // The entries go to the stages in runs that cannot fill a stage, and
    // stages of kStageEntries or more are flushed between runs (a flush
    // leaves only the few entries the sweep still reads, so every run takes
    // at least 64 entries). So the loops keep the stage sizes in registers.
    // With the capacity checks inside the loop the sizes went through memory
    // on every entry (flush() changes them), and the loop's speed depended
    // on how the compiler arranged flush(): when a4d89adf moved the sweep to
    // its own files, GCC 13 inlined flush() here, and C4.5's build on HIGGS
    // took 15% longer on a Xeon 6542Y (docs/PERFORMANCE.md).
    if (keep == Keep::Both) {
      Stage &left = stages[0];
      Stage &right = stages[1];
      for (std::uint32_t index = 0; index < count;) {
        const std::uint32_t end =
            index + std::min(count - index, kCapacity - std::max(left.size, right.size));
        Entry *const leftData = left.data;
        Entry *const rightData = right.data;
        std::uint32_t leftSize = left.size;
        std::uint32_t rightSize = right.size;
        for (; index < end; ++index) {
          const std::uint64_t word = entryWord(source[index]);
          const std::uint32_t isLeft = leftBit(leftBits, codec, word);
          putEntry(leftData + leftSize, word);
          putEntry(rightData + rightSize, word);
          leftSize += isLeft;
          rightSize += 1 - isLeft;
        }
        left.size = leftSize;
        right.size = rightSize;
        for (int child = 0; child < 2; ++child) {
          if (stages[child].size >= kStageEntries) {
            flush(child);
          }
        }
      }
      flush(0);
      flush(1);
    } else {
      const int child = keep == Keep::Left ? 0 : 1;
      const std::uint32_t wanted = keep == Keep::Left;
      Stage &stage = stages[child];
      for (std::uint32_t index = 0; index < count;) {
        const std::uint32_t end = index + std::min(count - index, kCapacity - stage.size);
        Entry *const data = stage.data;
        std::uint32_t size = stage.size;
        for (; index < end; ++index) {
          const std::uint64_t word = entryWord(source[index]);
          putEntry(data + size, word);
          size += leftBit(leftBits, codec, word) == wanted;
        }
        stage.size = size;
        if (size >= kStageEntries) {
          flush(child);
        }
      }
      flush(child);
    }
  }
#if defined(__SSE2__)
  _mm_sfence(); // the stores are visible before another thread reads them
#endif
  for (int child = 0; child < 2; ++child) {
    if (!children[child]) {
      continue;
    }
    FeatureCuts &out = *children[child]->out;
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    if (sweeps[child]) {
      out.cuts[feature] = sweeps[child]->finish(out.lefts.data() + feature * classes);
    }
    const Entry *entries = (isWinner ? source : target) + childBegin[child];
    out.constant[feature] = !isCut(entries[0].value, entries[childCount[child] - 1].value, gap);
  }
}

} // namespace dt
