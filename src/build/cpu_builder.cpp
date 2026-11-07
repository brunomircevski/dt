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
// so the waiter slept on the old value for good (about 1 in 40 --parallel runs
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

// Children with at least this many rows are swept during the out-of-place
// partition of their parent, through staging buffers of kStageEntries.
constexpr std::uint32_t kSweptChildRows = 1u << 12;
constexpr std::uint32_t kStageEntries = 1024;

} // namespace

void fillCountTables(std::size_t size, CountTable &xlog, CountTable &inverse, ThreadPool *pool) {
  xlog.resize(size);
  inverse.resize(size);
  constexpr std::size_t kChunks = 64;
  parallelFor(pool, kChunks, [&](std::size_t chunk) {
    const std::size_t end = size * (chunk + 1) / kChunks;
    for (std::size_t c = size * chunk / kChunks; c < end; ++c) {
      xlog[c] = xlog2x(static_cast<double>(c));
      inverse[c] = 1.0 / static_cast<double>(c); // inverse[0] is never used
    }
  });
}

namespace {

CountTables defaultCountTables() {
  static const auto tables = [] {
    std::pair<CountTable, CountTable> both;
    fillCountTables(kLogTableSize, both.first, both.second, nullptr);
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
      options_(options), tables_(tables.xlog ? tables : defaultCountTables()),
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
  const bool outOfPlace = columns.leftBits && count >= kOutOfPlaceRows &&
                          (columns.slots > 0 ? columns.slots : featureCount_) > features.size();
  std::optional<trace::Doing> marking(std::in_place, trace::Work::Mark);
  parallelFor(widePool, blocks, [&](std::size_t block) {
    const std::uint32_t end = static_cast<std::uint32_t>(std::size_t{count} * (block + 1) / blocks);
    for (std::uint32_t index = static_cast<std::uint32_t>(std::size_t{count} * block / blocks);
         index < end; ++index) {
      const std::uint32_t row = codec_.row(winner[index].packed);
      if (outOfPlace) {
        const std::uint64_t bit = std::uint64_t{1} << (row & 63);
        if (!pool_) { // one thread: nothing else writes the words
          std::uint64_t &word = columns.leftBits[row >> 6];
          word = index < leftCount ? word | bit : word & ~bit;
          continue;
        }
        // Atomic: the rows of other nodes share the words.
        std::atomic_ref<std::uint64_t> word(columns.leftBits[row >> 6]);
        if (index < leftCount) {
          word.fetch_or(bit, std::memory_order_relaxed);
        } else {
          word.fetch_and(~bit, std::memory_order_relaxed);
        }
      } else {
        goesLeft_[row] = index < leftCount;
      }
    }
  });
  marking.reset();
  const Keep keep = leftTerminal ? Keep::Right : rightTerminal ? Keep::Left : Keep::Both;
  // Big children (two classes) are swept while the columns are partitioned.
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
  if (outOfPlace && !blocked && classCount_ == 2) {
    for (int child = 0; child < 2; ++child) {
      const std::uint32_t *total = child == 0 ? leftCounts : rightCounts;
      const std::uint32_t rows = child == 0 ? leftCount : rightCount;
      const std::uint32_t childMin = rules_.minChildRows(rows);
      if ((child == 0 ? leftTerminal : rightTerminal) || rows < kSweptChildRows ||
          rows >= tables_.size || childMin + 64 > kStageEntries) {
        continue;
      }
      childCuts[child] = std::make_shared<FeatureCuts>(featureCount_, classes);
      sweeps[child] = {rows,
                       {total[0], total[1]},
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
      partition(columns, item, decision.feature, leftCount, keep, outOfPlace, blocked, swept);
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
      partitionInBlocks(sources, targets, moveWaits, count, leftCount, keep, columns.leftBits);
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
                      partitionAndSweep<Criterion::Gini, false>(source, target, count,
                                                                leftCount, keep, columns.leftBits,
                                                                isWinner, feature, children);
                    } else {
                      partitionAndSweep<Criterion::Entropy, true>(source, target, count,
                                                                  leftCount, keep, columns.leftBits,
                                                                  isWinner, feature, children);
                    }
                  } else if (!isWinner) {
                    partitionFeatureOutOfPlace(source, target, count, leftCount, keep,
                                               columns.leftBits);
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
  // --gpu-min-rows): a per-thread buffer (stays in cache), one column per
  // thread for wide nodes.
  auto partitionOne = [&](std::size_t index) {
    thread_local std::vector<Entry> buffer;
    if (keep == Keep::Both && buffer.size() < count) {
      buffer.resize(count);
    }
    partitionFeature(columnOf(columns, item, others[index]) + item.begin, count, leftCount,
                     buffer.data(), keep);
  };
  parallelFor(pool_ && count >= options_.featureParallelRows ? pool_ : nullptr, others.size(),
              partitionOne);
  return item.slots;
}

// Stable partition of `source` into `target` by goesLeft: left rows to
// [0, leftCount), right rows to [leftCount, count), in order (only one of the
// two if the other child is a leaf). The columns of a big node do not fit in
// the caches, so the stores bypass them (non-temporal: no read of the target
// lines first, no eviction of data still needed).
void CpuTreeBuilder::partitionFeatureOutOfPlace(const Entry *source, Entry *target,
                                                std::uint32_t count, std::uint32_t leftCount,
                                                Keep keep, const std::uint64_t *leftBits) const {
  auto store = [](Entry *at, Entry entry) {
#if defined(__SSE2__)
    _mm_stream_si64(reinterpret_cast<long long *>(at), std::bit_cast<long long>(entry));
#else
    *at = entry;
#endif
  };
  const EntryCodec codec = codec_;
  auto goesLeft = [leftBits, codec](Entry entry) {
    const std::uint32_t row = codec.row(entry.packed);
    return static_cast<std::uint32_t>(leftBits[row >> 6] >> (row & 63)) & 1u;
  };
  if (keep == Keep::Both) {
    std::uint32_t left = 0;
    std::uint32_t right = leftCount;
    for (std::uint32_t index = 0; index < count; ++index) {
      const Entry entry = source[index];
      const std::uint32_t isLeft = goesLeft(entry);
      store(target + (isLeft ? left : right), entry);
      left += isLeft;
      right += 1 - isLeft;
    }
  } else {
    // One child: usually a lopsided split, so the branch is predictable.
    const std::uint32_t wanted = keep == Keep::Left;
    Entry *write = target + (keep == Keep::Left ? 0 : leftCount);
    for (std::uint32_t index = 0; index < count; ++index) {
      const Entry entry = source[index];
      if (goesLeft(entry) == wanted) {
        store(write++, entry);
      }
    }
  }
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
  auto goesLeft = [leftBits, codec](Entry entry) {
    const std::uint32_t row = codec.row(entry.packed);
    return static_cast<std::uint32_t>(leftBits[row >> 6] >> (row & 63)) & 1u;
  };
  pool_->parallelFor(columns * blocks, [&](std::size_t item) {
    const std::size_t column = item / blocks;
    const std::size_t block = item % blocks;
    const std::uint32_t start = static_cast<std::uint32_t>(block * kPartitionBlock);
    const std::uint32_t length = std::min(kPartitionBlock, count - start);
    const Entry *source = sources[column] + start;
    std::uint32_t mine = 0;
    for (std::uint32_t index = 0; index < length; ++index) {
      mine += goesLeft(source[index]);
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
    Entry *target = targets[column];
    std::uint32_t left = before;
    std::uint32_t right = leftCount + (start - before);
    for (std::uint32_t index = 0; index < length; ++index) {
      const Entry entry = source[index];
      const std::uint32_t isLeft = goesLeft(entry);
      if (isLeft ? keep != Keep::Right : keep != Keep::Left) {
#if defined(__SSE2__)
        _mm_stream_si64(reinterpret_cast<long long *>(target + (isLeft ? left : right)),
                        std::bit_cast<long long>(entry));
#else
        target[isLeft ? left : right] = entry;
#endif
      }
      left += isLeft;
      right += 1 - isLeft;
    }
#if defined(__SSE2__)
    _mm_sfence(); // the stores are visible before another thread reads them
#endif
    // Done with its source (read twice: counting, then writing).
    if (read[column].fetch_add(1, std::memory_order_acq_rel) + 1 == blocks) {
      read[column].notify_all();
    }
  });
}

// Stable partition by goesLeft (`leftCount` rows go left). The larger side is
// compacted in place (the write index never passes the read index), the
// smaller one waits in `buffer`, which needs min(left, right) + 1 entries. If
// only one child is grown further, only its rows are moved, in place: left
// rows forward, or right rows backward from the end.
void CpuTreeBuilder::partitionFeature(Entry *entries, std::uint32_t count,
                                      std::uint32_t leftCount, Entry *buffer, Keep keep) const {
  if (keep == Keep::Left) {
    std::uint32_t write = 0;
    for (std::uint32_t index = 0; index < count; ++index) {
      const Entry entry = entries[index];
      entries[write] = entry;
      write += goesLeft_[codec_.row(entry.packed)];
    }
    return;
  }
  if (keep == Keep::Right) {
    std::uint32_t write = count; // right rows fill [write, count)
    for (std::uint32_t index = count; index-- > 0;) {
      const Entry entry = entries[index];
      entries[write - 1] = entry;
      write -= 1u - goesLeft_[codec_.row(entry.packed)];
    }
    return;
  }
  if (leftCount < count - leftCount) {
    // Right rows compacted backward from the end, left rows into
    // buffer[1, leftCount], also backward.
    std::uint32_t rightWrite = count;
    std::uint32_t leftWrite = leftCount;
    for (std::uint32_t index = count; index-- > 0;) {
      const Entry entry = entries[index];
      const std::uint32_t goesLeft = goesLeft_[codec_.row(entry.packed)];
      entries[rightWrite - 1] = entry;
      buffer[leftWrite] = entry;
      rightWrite -= 1 - goesLeft;
      leftWrite -= goesLeft;
    }
    std::memcpy(entries, buffer + 1, leftCount * sizeof(Entry));
    return;
  }
  std::uint32_t leftWrite = 0;
  std::uint32_t rightWrite = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    const Entry entry = entries[index];
    const std::uint32_t goesLeft = goesLeft_[codec_.row(entry.packed)];
    entries[leftWrite] = entry;
    buffer[rightWrite] = entry;
    leftWrite += goesLeft;
    rightWrite += 1 - goesLeft;
  }
  std::memcpy(entries + leftWrite, buffer, rightWrite * sizeof(Entry));
}

// Out-of-place partition of one column (see partitionFeatureOutOfPlace())
// that also sweeps the column for the children in `children`: entries go
// through a small staging buffer per child (in L1), where the child's sweep
// reads them, and from there to the target with non-temporal stores. So a big
// node's columns are read once per level instead of twice (partition, then
// the children's sweeps). The winner's column is already in order: it stays
// where it is (no target), and the sweeps read it there.
template <Criterion Crit, bool HasGap>
void CpuTreeBuilder::partitionAndSweep(const Entry *source, Entry *target, std::uint32_t count,
                                       std::uint32_t leftCount, Keep keep,
                                       const std::uint64_t *leftBits, bool isWinner,
                                       std::uint32_t feature,
                                       ChildSweep *const *children) const {
  using Sweep = TwoClassSweep<Crit, HasGap>;
  auto store = [](Entry *at, Entry entry) {
#if defined(__SSE2__)
    _mm_stream_si64(reinterpret_cast<long long *>(at), std::bit_cast<long long>(entry));
#else
    *at = entry;
#endif
  };
  const SweepSetup setup = sweepSetup();
  // --profile: the sweeps count as Sweep, the rest as PartitionSweep.
  const bool traced = trace::on();
  const bool kept[2] = {keep != Keep::Right, keep != Keep::Left};
  const std::uint32_t childBegin[2] = {0, leftCount};
  const std::uint32_t childCount[2] = {leftCount, count - leftCount};
  std::optional<Sweep> sweeps[2];
  for (int child = 0; child < 2; ++child) {
    if (children[child]) {
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
        store(to + index, stage.data[index - stage.base]);
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
    auto goesLeft = [leftBits, codec](Entry entry) {
      const std::uint32_t row = codec.row(entry.packed);
      return static_cast<std::uint32_t>(leftBits[row >> 6] >> (row & 63)) & 1u;
    };
    if (keep == Keep::Both) {
      Stage &left = stages[0];
      Stage &right = stages[1];
      for (std::uint32_t index = 0; index < count; ++index) {
        const Entry entry = source[index];
        const std::uint32_t isLeft = goesLeft(entry);
        left.data[left.size] = entry;
        right.data[right.size] = entry;
        left.size += isLeft;
        right.size += 1 - isLeft;
        if (left.size == kCapacity) {
          flush(0);
        }
        if (right.size == kCapacity) {
          flush(1);
        }
      }
      flush(0);
      flush(1);
    } else {
      // One child: usually a lopsided split, so the branch is predictable.
      const int child = keep == Keep::Left ? 0 : 1;
      const std::uint32_t wanted = keep == Keep::Left;
      Stage &stage = stages[child];
      for (std::uint32_t index = 0; index < count; ++index) {
        const Entry entry = source[index];
        if (goesLeft(entry) == wanted) {
          stage.data[stage.size++] = entry;
          if (stage.size == kCapacity) {
            flush(child);
          }
        }
      }
      flush(child);
    }
  }
#if defined(__SSE2__)
  _mm_sfence(); // the stores are visible before another thread reads them
#endif
  const double gap = rules_.minValueGap();
  for (int child = 0; child < 2; ++child) {
    if (!sweeps[child]) {
      continue;
    }
    FeatureCuts &out = *children[child]->out;
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    out.cuts[feature] = sweeps[child]->finish(out.lefts.data() + feature * classes);
    const Entry *entries = (isWinner ? source : target) + childBegin[child];
    out.constant[feature] = !isCut(entries[0].value, entries[childCount[child] - 1].value, gap);
  }
}

} // namespace dt
