#include "build/cpu_builder.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>

namespace dt {

EntryCodec makeEntryCodec(std::size_t classCount, std::size_t rowCount) {
  EntryCodec codec;
  codec.classBits = 1;
  while ((std::size_t{1} << codec.classBits) < classCount) {
    ++codec.classBits;
  }
  codec.classMask = (1u << codec.classBits) - 1;
  if (codec.classBits >= 32 || rowCount > (std::size_t{1} << (32 - codec.classBits))) {
    throw std::runtime_error("Too many rows (" + std::to_string(rowCount) + ") for " +
                             std::to_string(classCount) +
                             " classes: row and class id must fit in 32 bits");
  }
  return codec;
}

namespace {

// Nodes with more rows partition their columns one at a time, each with all
// threads, through the node's own range of Columns::scratch. Smaller nodes
// partition one column per thread with a per-thread buffer of at most this
// size, so memory does not grow with threads * rows.
constexpr std::uint32_t kBlockPartitionRows = 1u << 18;

// Map a float to an unsigned key with the same order (negatives flipped).
std::uint32_t sortKey(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

// Stable LSD radix sort of entries by value: 3 passes of 11 bits. Passes in
// which every key has the same digit (common for small integer features) are
// skipped.
void radixSortByValue(Entry *entries, std::size_t count, std::vector<Entry> &temp) {
  constexpr int kBits = 11;
  constexpr std::size_t kBuckets = std::size_t{1} << kBits;
  std::vector<std::size_t> histogram(3 * kBuckets, 0);
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint32_t key = sortKey(entries[index].value);
    ++histogram[key & (kBuckets - 1)];
    ++histogram[kBuckets + ((key >> kBits) & (kBuckets - 1))];
    ++histogram[2 * kBuckets + (key >> (2 * kBits))];
  }

  temp.resize(count);
  Entry *source = entries;
  Entry *target = temp.data();
  for (int pass = 0; pass < 3; ++pass) {
    std::size_t *bucket = histogram.data() + pass * kBuckets;
    const std::uint32_t shift = static_cast<std::uint32_t>(pass * kBits);
    const std::uint32_t firstDigit = (sortKey(source[0].value) >> shift) & (kBuckets - 1);
    if (bucket[firstDigit] == count) {
      continue; // all keys share this digit
    }
    std::size_t offset = 0;
    for (std::size_t digit = 0; digit < kBuckets; ++digit) {
      const std::size_t size = bucket[digit];
      bucket[digit] = offset;
      offset += size;
    }
    for (std::size_t index = 0; index < count; ++index) {
      const std::uint32_t digit = (sortKey(source[index].value) >> shift) & (kBuckets - 1);
      target[bucket[digit]++] = source[index];
    }
    std::swap(source, target);
  }
  if (source != entries) {
    std::memcpy(entries, source, count * sizeof(Entry));
  }
}

} // namespace

void presortColumns(const Dataset &dataset, const EntryCodec &codec, Entry *out,
                    ThreadPool *pool) {
  const std::size_t rows = dataset.rowCount;
  parallelFor(pool, dataset.featureCount(), [&](std::size_t feature) {
    const float *values = dataset.column(feature);
    Entry *entries = out + feature * rows;
    for (std::size_t row = 0; row < rows; ++row) {
      // +0.0f: -0.0 and 0.0 are the same value (no threshold between them).
      entries[row] = {values[row] + 0.0f,
                      codec.pack(static_cast<std::uint32_t>(row), dataset.labels[row])};
    }
    std::vector<Entry> temp;
    radixSortByValue(entries, rows, temp);
  });
}

CpuTreeBuilder::CpuTreeBuilder(const SplitRules &rules, const EntryCodec &codec,
                               std::size_t featureCount, NodeStore &store,
                               std::uint8_t *goesLeft, ThreadPool *pool,
                               const ParallelOptions &options)
    : rules_(rules), codec_(codec), featureCount_(featureCount),
      classCount_(rules.classCount()), store_(store), goesLeft_(goesLeft), pool_(pool),
      options_(options) {}

void CpuTreeBuilder::grow(const Columns &columns, Subtree root) {
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
  if (rules_.isTerminal(counts, count, item.depth)) {
    return;
  }

  // Features whose values are all (nearly) equal here cannot split this node
  // or any node below it: the range is sorted, so first vs last tells.
  const double gap = rules_.minValueGap();
  std::vector<std::uint32_t> &features = item.features;
  std::erase_if(features, [&](std::uint32_t feature) {
    const Entry *entries = columns.feature(feature) + begin;
    return !isCut(entries[0].value, entries[count - 1].value, gap);
  });
  if (features.empty()) {
    return;
  }

  // 1. Best cut of every feature: one sweep over its sorted range.
  const bool wide = pool_ && count >= options_.featureParallelRows;
  ThreadPool *const widePool = wide ? pool_ : nullptr;
  const std::uint32_t minChild = rules_.minChildRows(count);
  const double parentWeighted =
      weightedImpurity(counts, classCount_, count, rules_.criterion(), rules_.logTable());
  // Reused per thread (no allocation per node). A named reference, because a
  // thread_local used inside the parallelFor lambda would be each worker's own.
  thread_local std::vector<CutCandidate> threadCuts;
  std::vector<CutCandidate> &cuts = threadCuts;
  cuts.assign(featureCount_, CutCandidate{});
  parallelFor(widePool, features.size(), [&](std::size_t index) {
    const std::uint32_t feature = features[index];
    cuts[feature] =
        scanFeature(columns.feature(feature) + begin, count, counts, parentWeighted, minChild);
  });

  // 2. Pick the feature (CART or C4.5 rule).
  const SplitRules::Decision decision = rules_.choose(cuts.data(), featureCount_, count);
  if (decision.feature < 0) {
    return;
  }
  const std::uint32_t leftCount = decision.leftCount;
  const Entry *winner = columns.feature(static_cast<std::size_t>(decision.feature)) + begin;

  // 3. In the winning column the first leftCount entries go left: count the
  //    children's classes (in blocks, in parallel for wide nodes).
  const std::size_t classes = static_cast<std::size_t>(classCount_);
  const std::size_t blocks = wide ? 4 * (pool_->workerCount() + 1) : 1;
  std::vector<std::uint32_t> blockCounts(blocks * classes, 0);
  parallelFor(widePool, blocks, [&](std::size_t block) {
    std::uint32_t *local = blockCounts.data() + block * classes;
    const std::uint32_t end = static_cast<std::uint32_t>(leftCount * (block + 1) / blocks);
    for (std::uint32_t index = static_cast<std::uint32_t>(leftCount * block / blocks);
         index < end; ++index) {
      ++local[codec_.cls(winner[index].packed)];
    }
  });
  std::vector<std::uint32_t> childCounts(2 * classes, 0);
  std::uint32_t *leftCounts = childCounts.data();
  std::uint32_t *rightCounts = leftCounts + classes;
  for (std::size_t block = 0; block < blocks; ++block) {
    for (std::size_t k = 0; k < classes; ++k) {
      leftCounts[k] += blockCounts[block * classes + k];
    }
  }
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
  //    children stay sorted.
  const std::uint32_t rightCount = count - leftCount;
  const bool leftTerminal = rules_.isTerminal(leftCounts, leftCount, item.depth + 1);
  const bool rightTerminal = rules_.isTerminal(rightCounts, rightCount, item.depth + 1);
  if (leftTerminal && rightTerminal) {
    return;
  }
  parallelFor(widePool, blocks, [&](std::size_t block) {
    const std::uint32_t end = static_cast<std::uint32_t>(std::size_t{count} * (block + 1) / blocks);
    for (std::uint32_t index = static_cast<std::uint32_t>(std::size_t{count} * block / blocks);
         index < end; ++index) {
      goesLeft_[codec_.row(winner[index].packed)] = index < leftCount;
    }
  });
  partition(columns, item, decision.feature, leftCount);

  // 5. Children. A big left child becomes a pool task so idle threads can pick
  //    it up; this thread continues with the right one.
  Subtree left{leftId, begin, leftCount, item.depth + 1, {}};
  Subtree right{rightId, begin + leftCount, rightCount, item.depth + 1, {}};
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

void CpuTreeBuilder::partition(const Columns &columns, const Subtree &item, int winner,
                               std::uint32_t leftCount) {
  const std::uint32_t count = item.count;
  thread_local std::vector<std::uint32_t> threadOthers;
  std::vector<std::uint32_t> &others = threadOthers;
  others.clear();
  for (std::uint32_t feature : item.features) {
    if (static_cast<int>(feature) != winner) {
      others.push_back(feature);
    }
  }
  if (count > kBlockPartitionRows) {
    // Big node: the node's own range of the shared scratch array; with a
    // pool, all threads work on one column at a time.
    Entry *nodeScratch = columns.scratch + item.begin;
    for (std::uint32_t feature : others) {
      Entry *entries = columns.feature(feature) + item.begin;
      if (pool_) {
        partitionFeatureInBlocks(entries, count, leftCount, nodeScratch);
      } else {
        partitionFeature(entries, count, nodeScratch);
      }
    }
    return;
  }
  // Small node: a per-thread buffer (stays in cache), one column per thread
  // for wide nodes.
  auto partitionOne = [&](std::size_t index) {
    thread_local std::vector<Entry> buffer;
    if (buffer.size() < count) {
      buffer.resize(count);
    }
    partitionFeature(columns.feature(others[index]) + item.begin, count, buffer.data());
  };
  parallelFor(pool_ && count >= options_.featureParallelRows ? pool_ : nullptr, others.size(),
              partitionOne);
}

// Stable partition by goesLeft: left rows are compacted forward in place (the
// write index never passes the read index), right rows wait in `buffer`.
void CpuTreeBuilder::partitionFeature(Entry *entries, std::uint32_t count, Entry *buffer) const {
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

// The same for one big column, with all threads: every block splits its
// entries into its own range of `buffer` (left rows from the front, right rows
// from the back), then copies them to their final places.
void CpuTreeBuilder::partitionFeatureInBlocks(Entry *entries, std::uint32_t count,
                                              std::uint32_t leftCount, Entry *buffer) const {
  const std::size_t threads = pool_->workerCount() + 1;
  const std::uint32_t blockSize = std::max<std::uint32_t>(
      1u << 14, static_cast<std::uint32_t>((count + 4 * threads - 1) / (4 * threads)));
  const std::size_t blocks = (count + blockSize - 1) / blockSize;
  std::vector<std::uint32_t> leftBefore(blocks + 1, 0);
  pool_->parallelFor(blocks, [&](std::size_t block) {
    const std::uint32_t start = static_cast<std::uint32_t>(block * blockSize);
    const std::uint32_t end = std::min(count, start + blockSize);
    std::uint32_t leftWrite = start;
    std::uint32_t rightWrite = end; // right rows fill [rightWrite, end) backwards
    for (std::uint32_t index = start; index < end; ++index) {
      const Entry entry = entries[index];
      const std::uint32_t goesLeft = goesLeft_[codec_.row(entry.packed)];
      buffer[leftWrite] = entry;
      buffer[rightWrite - 1] = entry;
      leftWrite += goesLeft;
      rightWrite -= 1 - goesLeft;
    }
    leftBefore[block + 1] = leftWrite - start;
  });
  for (std::size_t block = 0; block < blocks; ++block) {
    leftBefore[block + 1] += leftBefore[block];
  }
  pool_->parallelFor(blocks, [&](std::size_t block) {
    const std::uint32_t start = static_cast<std::uint32_t>(block * blockSize);
    const std::uint32_t end = std::min(count, start + blockSize);
    const std::uint32_t lefts = leftBefore[block + 1] - leftBefore[block];
    std::memcpy(entries + leftBefore[block], buffer + start, lefts * sizeof(Entry));
    Entry *right = entries + leftCount + (start - leftBefore[block]);
    for (std::uint32_t index = 0; index < end - start - lefts; ++index) {
      right[index] = buffer[end - 1 - index];
    }
  });
}

CutCandidate CpuTreeBuilder::scanFeature(const Entry *entries, std::uint32_t count,
                                         const std::uint32_t *total, double parentWeighted,
                                         std::uint32_t minChild) const {
  switch (classCount_) {
  case 2: return scanFeatureK<2>(entries, count, total, parentWeighted, minChild);
  case 3: return scanFeatureK<3>(entries, count, total, parentWeighted, minChild);
  case 4: return scanFeatureK<4>(entries, count, total, parentWeighted, minChild);
  case 5: return scanFeatureK<5>(entries, count, total, parentWeighted, minChild);
  case 6: return scanFeatureK<6>(entries, count, total, parentWeighted, minChild);
  case 7: return scanFeatureK<7>(entries, count, total, parentWeighted, minChild);
  case 8: return scanFeatureK<8>(entries, count, total, parentWeighted, minChild);
  default: return scanFeatureK<0>(entries, count, total, parentWeighted, minChild);
  }
}

// One left-to-right sweep. Cut i sits between entries i-1 and i and sends the
// first i rows left; `left` holds the class counts of those rows.
template <int FixedK>
CutCandidate CpuTreeBuilder::scanFeatureK(const Entry *entries, std::uint32_t count,
                                          const std::uint32_t *total, double parentWeighted,
                                          std::uint32_t minChild) const {
  const int classCount = FixedK > 0 ? FixedK : classCount_;
  const Criterion criterion = rules_.criterion();
  const LogTable logs = rules_.logTable();
  std::uint32_t fixedLeft[FixedK > 0 ? FixedK : 1] = {};
  std::vector<std::uint32_t> dynamicLeft;
  std::uint32_t *left = fixedLeft;
  if constexpr (FixedK == 0) {
    dynamicLeft.assign(static_cast<std::size_t>(classCount), 0);
    left = dynamicLeft.data();
  }

  CutCandidate best;
  if (count < 2 * minChild) {
    return best;
  }
  for (std::uint32_t index = 0; index + 1 < minChild; ++index) {
    ++left[codec_.cls(entries[index].packed)];
  }

  const double gap = rules_.minValueGap();
  std::uint32_t tries = 0;
  const std::uint32_t lastCut = count - minChild;
  for (std::uint32_t cut = minChild; cut <= lastCut; ++cut) {
    const Entry previous = entries[cut - 1];
    const Entry current = entries[cut];
    const std::uint32_t previousClass = codec_.cls(previous.packed);
    ++left[previousClass];
    if (!isCut(previous.value, current.value, gap)) {
      continue; // (nearly) equal values: no threshold fits between them
    }
    ++tries;
    if (previousClass == codec_.cls(current.packed)) {
      const bool leftSingleton = cut < 2 || isCut(entries[cut - 2].value, previous.value, gap);
      const bool rightSingleton =
          cut + 1 >= count || isCut(current.value, entries[cut + 1].value, gap);
      if (isSkippableCut(true, leftSingleton, rightSingleton, cut, count, minChild)) {
        continue;
      }
    }
    const double gain =
        cutGain(total, left, classCount, count, cut, parentWeighted, criterion, logs);
    if (!best.valid() || isBetterCut(gain, cut, best.gain, best.leftCount)) {
      best.gain = gain;
      best.leftCount = cut;
      best.leftValue = previous.value;
      best.rightValue = current.value;
    }
  }
  best.tries = tries;
  return best;
}

} // namespace dt
