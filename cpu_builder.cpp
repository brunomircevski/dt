#include "cpu_builder.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>

dt::EntryCodec makeEntryCodec(std::size_t classCount, std::size_t rowCount) {
  dt::EntryCodec codec;
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

// Map a float to an unsigned key with the same order (negatives flipped).
std::uint32_t sortKey(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

// Stable LSD radix sort of entries by value: 3 passes of 11 bits. Passes in
// which every key has the same digit (common for small integer features) are
// skipped.
void radixSortByValue(dt::Entry *entries, std::size_t count, std::vector<dt::Entry> &temp) {
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
  dt::Entry *source = entries;
  dt::Entry *target = temp.data();
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
    std::memcpy(entries, source, count * sizeof(dt::Entry));
  }
}

} // namespace

void presortColumns(const Dataset &dataset, const dt::EntryCodec &codec, dt::Entry *out,
                    ThreadPool *pool) {
  const std::size_t rows = dataset.rowCount;
  auto sortFeature = [&](std::size_t feature) {
    const float *values = dataset.column(feature);
    dt::Entry *entries = out + feature * rows;
    for (std::size_t row = 0; row < rows; ++row) {
      // +0.0f: -0.0 and 0.0 are the same value (no threshold between them).
      entries[row] = {values[row] + 0.0f,
                      codec.pack(static_cast<std::uint32_t>(row), dataset.labels[row])};
    }
    thread_local std::vector<dt::Entry> temp;
    radixSortByValue(entries, rows, temp);
    temp.clear();
    temp.shrink_to_fit();
  };
  if (pool) {
    pool->parallelFor(dataset.featureCount(), sortFeature);
  } else {
    for (std::size_t feature = 0; feature < dataset.featureCount(); ++feature) {
      sortFeature(feature);
    }
  }
}

std::vector<float> distinctSortedValues(const dt::Entry *sorted, std::size_t count) {
  std::vector<float> values;
  for (std::size_t index = 0; index < count; ++index) {
    if (values.empty() || values.back() < sorted[index].value) {
      values.push_back(sorted[index].value);
    }
  }
  values.shrink_to_fit();
  return values;
}

std::vector<float> distinctSortedValues(const float *sorted, std::size_t count) {
  std::vector<float> values;
  for (std::size_t index = 0; index < count; ++index) {
    if (values.empty() || values.back() < sorted[index]) {
      values.push_back(sorted[index]);
    }
  }
  values.shrink_to_fit();
  return values;
}

CpuTreeBuilder::CpuTreeBuilder(const SplitRules &rules, const dt::EntryCodec &codec,
                               std::size_t featureCount, std::uint8_t *goesLeft,
                               ThreadPool *pool, const Options &options)
    : rules_(rules), codec_(codec), featureCount_(featureCount),
      classCount_(rules.classCount()), goesLeft_(goesLeft), pool_(pool),
      minRowsForNodeTask_(options.minRowsForNodeTask),
      minRowsForFeatureParallel_(options.minRowsForFeatureParallel) {}

void CpuTreeBuilder::build(Columns columns, std::uint32_t begin, std::uint32_t count,
                           int depth, std::unique_ptr<Node> &slot) {
  std::vector<std::uint32_t> classCounts(static_cast<std::size_t>(classCount_), 0);
  const dt::Entry *entries = columns.feature(0) + begin;
  for (std::uint32_t index = 0; index < count; ++index) {
    ++classCounts[codec_.cls(entries[index].packed)];
  }
  buildNode(columns, begin, count, depth, std::move(classCounts), slot);
}

void CpuTreeBuilder::buildNode(Columns columns, std::uint32_t begin, std::uint32_t count,
                               int depth, std::vector<std::uint32_t> classCounts,
                               std::unique_ptr<Node> &slot) {
  auto node = std::make_unique<Node>();
  node->setCounts(std::move(classCounts));
  const std::vector<std::uint32_t> &counts = node->classCounts;
  if (rules_.isTerminal(counts.data(), count, depth)) {
    slot = std::move(node);
    return;
  }

  // 1. Best cut of every feature: one sweep over its sorted range.
  const std::uint32_t minChild = rules_.minChildRows(count);
  const double parentWeighted =
      dt::weightedImpurity(counts.data(), classCount_, count, rules_.criterion(),
                           rules_.logTable());
  // Reused per thread (no allocation per node). A named reference, because a
  // thread_local used inside the parallelFor lambda would be each worker's own.
  thread_local std::vector<dt::CutCandidate> threadCuts;
  std::vector<dt::CutCandidate> &cuts = threadCuts;
  cuts.assign(featureCount_, dt::CutCandidate{});
  const bool wide = pool_ && count >= minRowsForFeatureParallel_;
  auto scan = [&](std::size_t feature) {
    cuts[feature] = scanFeature(columns.feature(feature) + begin, count, counts.data(),
                                parentWeighted, minChild);
  };
  if (wide) {
    pool_->parallelFor(featureCount_, scan);
  } else {
    for (std::size_t feature = 0; feature < featureCount_; ++feature) {
      scan(feature);
    }
  }

  // 2. Pick the feature (CART or C4.5 rule).
  const SplitRules::Decision decision = rules_.choose(cuts.data(), featureCount_, count);
  if (decision.feature < 0) {
    slot = std::move(node);
    return;
  }
  const std::uint32_t leftCount = decision.leftCount;

  // 3. In the winning column the first leftCount entries go left: count the
  //    children's classes.
  std::vector<std::uint32_t> leftCounts(counts.size(), 0);
  const dt::Entry *winner = columns.feature(static_cast<std::size_t>(decision.feature)) + begin;
  for (std::uint32_t index = 0; index < leftCount; ++index) {
    ++leftCounts[codec_.cls(winner[index].packed)];
  }
  std::vector<std::uint32_t> rightCounts = counts;
  for (std::size_t cls = 0; cls < counts.size(); ++cls) {
    rightCounts[cls] -= leftCounts[cls];
  }

  // 4. Unless both children are leaves (often the case deep in the tree), mark
  //    the rows that go left and stable-partition every other column so both
  //    children stay sorted.
  const bool leftTerminal = rules_.isTerminal(leftCounts.data(), leftCount, depth + 1);
  const bool rightTerminal =
      rules_.isTerminal(rightCounts.data(), count - leftCount, depth + 1);
  if (!leftTerminal || !rightTerminal) {
    for (std::uint32_t index = 0; index < count; ++index) {
      goesLeft_[codec_.row(winner[index].packed)] = index < leftCount;
    }
    auto partition = [&](std::size_t feature) {
      if (static_cast<int>(feature) != decision.feature) {
        partitionFeature(columns.feature(feature) + begin, count);
      }
    };
    if (wide) {
      pool_->parallelFor(featureCount_, partition);
    } else {
      for (std::size_t feature = 0; feature < featureCount_; ++feature) {
        partition(feature);
      }
    }
  }

  node->feature = decision.feature;
  node->threshold = decision.threshold;
  Node *parent = node.get();
  slot = std::move(node);

  // 5. Children. Big ones become pool tasks so idle threads can pick them up.
  if (pool_ && count >= minRowsForNodeTask_ && !leftTerminal) {
    pool_->submit([this, columns, begin, leftCount, depth, parent,
                   counts = std::move(leftCounts)]() mutable {
      buildNode(columns, begin, leftCount, depth + 1, std::move(counts), parent->left);
    });
  } else {
    buildNode(columns, begin, leftCount, depth + 1, std::move(leftCounts), parent->left);
  }
  buildNode(columns, begin + leftCount, count - leftCount, depth + 1,
            std::move(rightCounts), parent->right);
}

dt::CutCandidate CpuTreeBuilder::scanFeature(const dt::Entry *entries, std::uint32_t count,
                                             const std::uint32_t *total,
                                             double parentWeighted,
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
dt::CutCandidate CpuTreeBuilder::scanFeatureK(const dt::Entry *entries, std::uint32_t count,
                                              const std::uint32_t *total,
                                              double parentWeighted,
                                              std::uint32_t minChild) const {
  const int classCount = FixedK > 0 ? FixedK : classCount_;
  const int criterion = rules_.criterion();
  const dt::LogTable logs = rules_.logTable();
  std::uint32_t fixedLeft[FixedK > 0 ? FixedK : 1] = {};
  std::vector<std::uint32_t> dynamicLeft;
  std::uint32_t *left = fixedLeft;
  if constexpr (FixedK == 0) {
    dynamicLeft.assign(static_cast<std::size_t>(classCount), 0);
    left = dynamicLeft.data();
  }

  dt::CutCandidate best;
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
    const dt::Entry previous = entries[cut - 1];
    const dt::Entry current = entries[cut];
    const std::uint32_t previousClass = codec_.cls(previous.packed);
    ++left[previousClass];
    if (!dt::isCut(previous.value, current.value, gap)) {
      continue; // (nearly) equal values: no threshold fits between them
    }
    ++tries;
    if (previousClass == codec_.cls(current.packed)) {
      const bool leftSingleton = cut < 2 || dt::isCut(entries[cut - 2].value, previous.value, gap);
      const bool rightSingleton =
          cut + 1 >= count || dt::isCut(current.value, entries[cut + 1].value, gap);
      if (dt::isSkippableCut(true, leftSingleton, rightSingleton, cut, count, minChild)) {
        continue;
      }
    }
    const double gain =
        dt::cutGain(total, left, classCount, count, cut, parentWeighted, criterion, logs);
    if (!best.valid() || dt::isBetterCut(gain, cut, best.gain, best.leftCount)) {
      best.gain = gain;
      best.leftCount = cut;
      best.leftValue = previous.value;
      best.rightValue = current.value;
    }
  }
  best.tries = tries;
  return best;
}

void CpuTreeBuilder::partitionFeature(dt::Entry *entries, std::uint32_t count) const {
  // Stable in-place partition: left rows are compacted forward (the write
  // index never passes the read index), right rows wait in a scratch buffer.
  thread_local std::vector<dt::Entry> scratch;
  if (scratch.size() < count) {
    scratch.resize(count);
  }
  std::uint32_t leftWrite = 0;
  std::uint32_t rightWrite = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    const dt::Entry entry = entries[index];
    const std::uint32_t goesLeft = goesLeft_[codec_.row(entry.packed)];
    entries[leftWrite] = entry;
    scratch[rightWrite] = entry;
    leftWrite += goesLeft;
    rightWrite += 1 - goesLeft;
  }
  std::memcpy(entries + leftWrite, scratch.data(), rightWrite * sizeof(dt::Entry));
}
