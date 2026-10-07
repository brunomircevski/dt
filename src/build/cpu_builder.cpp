#include "build/cpu_builder.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

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

// Stable LSD radix sort by value: 3 passes of 11 bits. Passes in which every
// key has the same digit (common for small integer and binary features) are
// skipped.
constexpr int kRadixBits = 11;
constexpr std::size_t kRadixBuckets = std::size_t{1} << kRadixBits;
constexpr int kRadixPasses = 3;

struct RadixHistogram {
  std::uint32_t counts[kRadixPasses][kRadixBuckets];
  bool active[kRadixPasses]; // the pass changes the order
  int activePasses;
};

void radixHistogram(const float *values, std::size_t count, RadixHistogram &histogram) {
  std::fill_n(&histogram.counts[0][0], kRadixPasses * kRadixBuckets, 0u);
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint32_t key = sortKey(values[index] + 0.0f);
    ++histogram.counts[0][key & (kRadixBuckets - 1)];
    ++histogram.counts[1][(key >> kRadixBits) & (kRadixBuckets - 1)];
    ++histogram.counts[2][key >> (2 * kRadixBits)];
  }
  histogram.activePasses = 0;
  const std::uint32_t firstKey = count > 0 ? sortKey(values[0] + 0.0f) : 0;
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    const std::uint32_t digit = (firstKey >> (pass * kRadixBits)) & (kRadixBuckets - 1);
    histogram.active[pass] = histogram.counts[pass][digit] != count;
    histogram.activePasses += histogram.active[pass];
  }
}

// Sorted entries of one feature column into `out` (and their values into
// `sortedValues` if not null). The first active pass reads the raw values, and
// the buffers alternate so that the last one writes `out`.
void sortColumn(const float *values, const std::uint16_t *labels, std::size_t count,
                const EntryCodec &codec, RadixHistogram &histogram, Entry *out,
                float *sortedValues) {
  auto entryOf = [&](std::size_t row) {
    // +0.0f: -0.0 and 0.0 are the same value (no threshold between them).
    return Entry{values[row] + 0.0f, codec.pack(static_cast<std::uint32_t>(row), labels[row])};
  };
  if (histogram.activePasses == 0) {
    for (std::size_t row = 0; row < count; ++row) {
      out[row] = entryOf(row);
    }
    if (sortedValues) {
      std::fill_n(sortedValues, count, count > 0 ? out[0].value : 0.0f);
    }
    return;
  }
  thread_local std::vector<Entry> temp;
  if (histogram.activePasses > 1 && temp.size() < count) {
    temp.resize(count);
  }
  Entry *target = histogram.activePasses % 2 == 1 ? out : temp.data();
  Entry *other = target == out ? temp.data() : out;
  const Entry *source = nullptr; // null: the raw values
  int remaining = histogram.activePasses;
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    if (!histogram.active[pass]) {
      continue;
    }
    const bool last = --remaining == 0;
    std::uint32_t *bucket = histogram.counts[pass];
    std::uint32_t offset = 0;
    for (std::size_t digit = 0; digit < kRadixBuckets; ++digit) {
      const std::uint32_t size = bucket[digit];
      bucket[digit] = offset;
      offset += size;
    }
    const std::uint32_t shift = static_cast<std::uint32_t>(pass * kRadixBits);
    auto place = [&](const Entry &entry) {
      const std::uint32_t at = bucket[(sortKey(entry.value) >> shift) & (kRadixBuckets - 1)]++;
      target[at] = entry;
      if (last && sortedValues) {
        sortedValues[at] = entry.value;
      }
    };
    if (source == nullptr) {
      for (std::size_t row = 0; row < count; ++row) {
        place(entryOf(row));
      }
    } else {
      for (std::size_t index = 0; index < count; ++index) {
        place(source[index]);
      }
    }
    source = target;
    std::swap(target, other);
  }
}

} // namespace

void presortColumns(const Dataset &dataset, const EntryCodec &codec, Entry *out,
                    ThreadPool *pool, float *sortedValues) {
  const std::size_t rows = dataset.rowCount;
  const std::size_t features = dataset.featureCount();
  if (rows > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("presortColumns: too many rows");
  }
  // Histograms first, so that the features with the most passes are sorted
  // first (features differ a lot: binary ones need one pass, continuous ones
  // three), which balances the threads.
  std::vector<RadixHistogram> histograms(features);
  parallelFor(pool, features, [&](std::size_t feature) {
    radixHistogram(dataset.column(feature), rows, histograms[feature]);
  });
  std::vector<std::size_t> order(features);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return histograms[a].activePasses > histograms[b].activePasses;
  });
  parallelFor(pool, features, [&](std::size_t index) {
    const std::size_t feature = order[index];
    sortColumn(dataset.column(feature), dataset.labels.data(), rows, codec, histograms[feature],
               out + feature * rows, sortedValues ? sortedValues + feature * rows : nullptr);
  });
}

void fillCountTables(std::size_t size, std::vector<double> &xlog, std::vector<double> &inverse,
                     ThreadPool *pool) {
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
    std::pair<std::vector<double>, std::vector<double>> both;
    fillCountTables(kLogTableSize, both.first, both.second, nullptr);
    return both;
  }();
  return {tables.first.data(), tables.second.data(), kLogTableSize};
}

} // namespace

CpuTreeBuilder::CpuTreeBuilder(const SplitRules &rules, const EntryCodec &codec,
                               std::size_t featureCount, NodeStore &store,
                               std::uint8_t *goesLeft, ThreadPool *pool,
                               const ParallelOptions &options, CountTables tables)
    : rules_(rules), codec_(codec), featureCount_(featureCount),
      classCount_(rules.classCount()), store_(store), goesLeft_(goesLeft), pool_(pool),
      options_(options), tables_(tables.xlog ? tables : defaultCountTables()) {}

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

  // 1. Best cut of every feature: one sweep over its sorted range, which also
  //    gives the class counts left of that cut.
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
  parallelFor(widePool, features.size(), [&](std::size_t index) {
    const std::uint32_t feature = features[index];
    cuts[feature] = scanFeature(columns.feature(feature) + begin, count, counts, parentWeighted,
                                minChild, cutLefts.data() + feature * classes);
  });

  // 2. Pick the feature (CART or C4.5 rule).
  const SplitRules::Decision decision = rules_.choose(cuts.data(), featureCount_, count);
  if (decision.feature < 0) {
    return;
  }
  const std::uint32_t leftCount = decision.leftCount;
  const Entry *winner = columns.feature(static_cast<std::size_t>(decision.feature)) + begin;

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
  if (leftTerminal && rightTerminal) {
    return;
  }
  const std::size_t blocks = wide ? 4 * (pool_->workerCount() + 1) : 1;
  parallelFor(widePool, blocks, [&](std::size_t block) {
    const std::uint32_t end = static_cast<std::uint32_t>(std::size_t{count} * (block + 1) / blocks);
    for (std::uint32_t index = static_cast<std::uint32_t>(std::size_t{count} * block / blocks);
         index < end; ++index) {
      goesLeft_[codec_.row(winner[index].packed)] = index < leftCount;
    }
  });
  const Keep keep = leftTerminal ? Keep::Right : rightTerminal ? Keep::Left : Keep::Both;
  partition(columns, item, decision.feature, leftCount, keep);

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
                               std::uint32_t leftCount, Keep keep) {
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
        partitionFeature(entries, count, nodeScratch, keep);
      }
    }
    return;
  }
  // Small node: a per-thread buffer (stays in cache), one column per thread
  // for wide nodes.
  auto partitionOne = [&](std::size_t index) {
    thread_local std::vector<Entry> buffer;
    if (keep == Keep::Both && buffer.size() < count) {
      buffer.resize(count);
    }
    partitionFeature(columns.feature(others[index]) + item.begin, count, buffer.data(), keep);
  };
  parallelFor(pool_ && count >= options_.featureParallelRows ? pool_ : nullptr, others.size(),
              partitionOne);
}

// Stable partition by goesLeft: left rows are compacted forward in place (the
// write index never passes the read index), right rows wait in `buffer`. If
// only one child is grown further, only its rows are moved, in place: left
// rows forward, or right rows backward from the end.
void CpuTreeBuilder::partitionFeature(Entry *entries, std::uint32_t count, Entry *buffer,
                                      Keep keep) const {
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
                                         std::uint32_t minChild, std::uint32_t *bestLeft) const {
  if (rules_.criterion() == Criterion::Gini) {
    return scanFeatureFor<Criterion::Gini>(entries, count, total, parentWeighted, minChild,
                                           bestLeft);
  }
  return scanFeatureFor<Criterion::Entropy>(entries, count, total, parentWeighted, minChild,
                                            bestLeft);
}

template <Criterion Crit>
CutCandidate CpuTreeBuilder::scanFeatureFor(const Entry *entries, std::uint32_t count,
                                            const std::uint32_t *total, double parentWeighted,
                                            std::uint32_t minChild,
                                            std::uint32_t *bestLeft) const {
  switch (classCount_) {
  case 2:
    if (count < tables_.size) {
      return rules_.minValueGap() > 0.0
                 ? scanTwoClasses<Crit, true>(entries, count, total, parentWeighted, minChild,
                                              bestLeft)
                 : scanTwoClasses<Crit, false>(entries, count, total, parentWeighted, minChild,
                                               bestLeft);
    }
    return scanFeatureK<2, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 3: return scanFeatureK<3, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 4: return scanFeatureK<4, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 5: return scanFeatureK<5, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 6: return scanFeatureK<6, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 7: return scanFeatureK<7, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  case 8: return scanFeatureK<8, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  default: return scanFeatureK<0, Crit>(entries, count, total, parentWeighted, minChild, bestLeft);
  }
}

namespace {

// Candidates a sweep buffers before settling them (see scanFeatureK()).
constexpr std::uint32_t kSweepBlock = 256;
// Candidates buffered before the first settling (then twice as many each
// time, up to kSweepBlock), so that small nodes get a threshold early.
constexpr std::uint32_t kFirstSettle = 16;
// Unit roundoff of double.
constexpr double kRoundoff = 0x1p-53;

// An upper bound, times n, on how much a cut's gain can grow when one more
// row moves left. With S = sum l_k^2 / n_L + sum r_k^2 / n_R, the Gini gain is
// (P - n + S) / n, and moving a row raises the left part of S by less than 2
// and the right part by at most 1. The entropy gain is (P - W) / n with W the
// weighted entropy, and moving a row of class c changes W by
// d(n_L) - d(l_c) - d(n_R - 1) + d(r_c - 1), d(x) = (x + 1) log2(x + 1) -
// x log2(x) increasing, so it lowers W by at most d(n_R - 1) <=
// log2(n) + log2(e). Both rounded up.
template <Criterion Crit> double maxGainStep(std::uint32_t count) {
  if constexpr (Crit == Criterion::Gini) {
    return 4.0;
  } else {
    return std::log2(static_cast<double>(count)) + 2.0;
  }
}

// Skipping cuts that cannot win. After a cut with gain estimate `gain`, the
// exact gain of the cut j steps further is at most gain + margin + j * step / n
// (maxGainStep), so the next skippableSteps() cuts provably cannot beat
// `bestGain` by kTieEps (rounding: another margin, and a small factor on the
// step). Short skips do not pay off (the loop exit is mispredicted), so a
// sweep only skips after estimates below skipBelow(): at least kMinSkip steps.
constexpr double kMinSkip = 8.0;

double skipBelow(double bestGain, double margin, double rowsPerGain) {
  return bestGain + kTieEps - 2.0 * margin - kMinSkip / rowsPerGain;
}

std::uint32_t skippableSteps(double bestGain, double gain, double margin, double rowsPerGain,
                             std::uint32_t limit) {
  const double steps = (bestGain + kTieEps - gain - 2.0 * margin) * rowsPerGain * (1.0 - 1e-9);
  return steps >= static_cast<double>(limit) ? limit : static_cast<std::uint32_t>(steps);
}

} // namespace

// Exact part of a sweep (see scanFeatureK()): the buffered candidates, in
// sweep order, go through the rule "keep a cut if its gain beats the best by
// more than kTieEps", with cutGain(). Candidates before the last one whose
// estimate beats all earlier buffered estimates by more than
// 2 * margin + kTieEps are skipped. `countsAt(i, out)` writes the class counts
// left of candidate i into `counts` (classCount values).
template <class CountsAt>
void settleCandidates(const double *estimate, const std::uint32_t *position, std::uint32_t size,
                      double margin, const std::uint32_t *total, int classCount,
                      std::uint32_t count, double parentWeighted, Criterion criterion,
                      LogTable logs, CountsAt &&countsAt, std::uint32_t *counts,
                      double &bestGain, std::uint32_t &bestCut, std::uint32_t *bestLeft) {
  std::uint32_t from = 0;
  double highest = -INFINITY;
  const double clearLead = 2.0 * margin + kTieEps;
  for (std::uint32_t index = 0; index < size; ++index) {
    if (estimate[index] > highest + clearLead) {
      from = index;
    }
    highest = std::max(highest, estimate[index]);
  }
  for (std::uint32_t index = from; index < size; ++index) {
    if (estimate[index] <= bestGain + kTieEps - margin) {
      continue;
    }
    countsAt(index, counts);
    const double gain =
        cutGain(total, counts, classCount, count, position[index], parentWeighted, criterion, logs);
    if (gain > bestGain + kTieEps) {
      bestGain = gain;
      bestCut = position[index];
      std::copy_n(counts, classCount, bestLeft);
    }
  }
}

// The sweep of scanFeatureK() for two classes (e.g. SUSY, HIGGS), with the
// same estimate / candidate / skip logic but leaner: the class counts at a cut
// follow from the number of class-1 rows left of it, the estimates need no
// running sums (Gini: no division, with tabled reciprocals; entropy: cutGain()
// with a multiplication by 1/n instead of the division), the boundary-point
// shortcut is decided without a branch (on continuous features it is a coin
// flip), and skipped stretches only add up class bits and thresholds.
// `tables_` must cover `count` (the caller checks); HasGap: C4.5's minimum
// gap between values (CART: values must just differ).
template <Criterion Crit, bool HasGap>
CutCandidate CpuTreeBuilder::scanTwoClasses(const Entry *entries, std::uint32_t count,
                                            const std::uint32_t *total, double parentWeighted,
                                            std::uint32_t minChild,
                                            std::uint32_t *bestLeft) const {
  CutCandidate best;
  if (count < 2 * minChild) {
    return best;
  }
  // Locals, so that they stay in registers across the settle() calls.
  const EntryCodec codec = codec_;
  const double *xlog = tables_.xlog;
  const double *inverse = tables_.inverse;
  const double gap = rules_.minValueGap();
  auto distinctValues = [gap](float lower, float upper) {
    if constexpr (HasGap) {
      return isCut(lower, upper, gap);
    } else {
      return lower < upper; // isCut() with no gap
    }
  };
  const LogTable logs = rules_.logTable();
  const double rows = static_cast<double>(count);
  const double inverseRows = 1.0 / rows;
  const double margin = 256.0 * kRoundoff; // a few roundings of terms <= n, divided by n
  const double rowsPerGain = rows / maxGainStep<Crit>(count);
  const double giniBase = parentWeighted - rows;
  const double total0 = total[0];
  const double total1 = total[1];

  std::uint32_t ones = 0; // class-1 rows left of the current cut
  for (std::uint32_t index = 0; index + 1 < minChild; ++index) {
    ones += codec.cls(entries[index].packed);
  }
  std::uint32_t tries = 0;
  double bestGain = -INFINITY;
  std::uint32_t bestCut = 0;
  double threshold = -INFINITY;     // estimates at or below it cannot beat the best cut
  double skipThreshold = -INFINITY; // estimates below it start a skip
  double estimate[kSweepBlock];
  std::uint32_t position[kSweepBlock];
  std::uint32_t onesAt[kSweepBlock];
  std::uint32_t found = 0;
  std::uint32_t settleAt = kFirstSettle;
  std::uint32_t counts[2];
  auto settle = [&]() {
    settleCandidates(
        estimate, position, found, margin, total, 2, count, parentWeighted, Crit, logs,
        [&](std::uint32_t index, std::uint32_t *out) {
          out[0] = position[index] - onesAt[index];
          out[1] = onesAt[index];
        },
        counts, bestGain, bestCut, bestLeft);
    found = 0;
    settleAt = std::min(2 * settleAt, kSweepBlock);
    threshold = bestGain + kTieEps - margin;
    skipThreshold = skipBelow(bestGain, margin, rowsPerGain);
  };

  const std::uint32_t lastCut = count - minChild;
  for (std::uint32_t cut = minChild; cut <= lastCut;) {
    const Entry previous = entries[cut - 1];
    const Entry current = entries[cut];
    const std::uint32_t previousClass = codec.cls(previous.packed);
    ones += previousClass;
    if (!distinctValues(previous.value, current.value)) {
      ++cut;
      continue;
    }
    ++tries;
    const bool leftDistinct = cut < 2 || distinctValues(entries[cut - 2].value, previous.value);
    const bool rightDistinct =
        cut + 1 >= count || distinctValues(current.value, entries[cut + 1].value);
    const bool skippable = (previousClass == codec.cls(current.packed)) & leftDistinct &
                           rightDistinct & (cut > minChild) & (cut + minChild < count);
    const double l1 = ones;
    const double l0 = static_cast<double>(cut - ones);
    const double r0 = total0 - l0;
    const double r1 = total1 - l1;
    double gain;
    if constexpr (Crit == Criterion::Gini) {
      gain = (giniBase + (l0 * l0 + l1 * l1) * inverse[cut] +
              (r0 * r0 + r1 * r1) * inverse[count - cut]) *
             inverseRows;
    } else {
      const std::uint32_t left0 = cut - ones;
      const double leftWeighted = xlog[cut] - (xlog[left0] + xlog[ones]);
      const double rightWeighted =
          xlog[count - cut] - (xlog[total[0] - left0] + xlog[total[1] - ones]);
      gain = (parentWeighted - leftWeighted - rightWeighted) * inverseRows;
    }
    // Buffered unless skipped, without a branch on `skippable`.
    estimate[found] = gain;
    position[found] = cut;
    onesAt[found] = ones;
    found += !skippable & (gain > threshold);
    if (found == settleAt) {
      settle();
    }
    ++cut;
    if (!(gain < skipThreshold)) {
      continue;
    }
    // The next cuts cannot beat the best cut while their gain provably stays
    // below it: only count their class-1 rows and their thresholds.
    const std::uint32_t steps =
        skippableSteps(bestGain, gain, margin, rowsPerGain, lastCut + 1 - cut);
    std::uint32_t skippedOnes = 0;
    std::uint32_t skippedTries = 0;
    for (std::uint32_t index = cut; index < cut + steps; ++index) {
      skippedOnes += codec.cls(entries[index - 1].packed);
      skippedTries += distinctValues(entries[index - 1].value, entries[index].value);
    }
    ones += skippedOnes;
    tries += skippedTries;
    cut += steps;
  }
  settle();
  best.tries = tries;
  if (bestCut > 0) {
    best.gain = bestGain;
    best.leftCount = bestCut;
    best.leftValue = entries[bestCut - 1].value;
    best.rightValue = entries[bestCut].value;
  }
  return best;
}

// One left-to-right sweep. Cut i sits between entries i-1 and i and sends the
// first i rows left. The result is exactly that of evaluating cutGain() at
// every allowed cut in order and keeping a cut when its gain beats the best so
// far by more than kTieEps, but cutGain() itself runs only for few cuts:
//
//  1. Rows are moved left one by one, which only updates the class counts
//     (most rows of discrete features sit inside runs of equal values, which
//     a tight loop moves). At each cut with a threshold a cheap estimate of
//     the gain is computed, one that needs no loop over the classes when a
//     single row moved since the previous one, nor a division: Gini from the
//     sums of squared class counts, which change by an integer, and tabled
//     reciprocals; entropy from running sums of c*log2(c), recomputed after
//     runs and every kSweepBlock updates, which keeps the rounding errors
//     small. The estimate is provably within `margin` of cutGain().
//  2. A cut whose estimate cannot beat the best cut by kTieEps cannot change
//     the result. The others (unless the boundary-point shortcut skips them)
//     are buffered with their class counts and settled in sweep order when
//     the buffer is full: every candidate before the last estimate that beats
//     all earlier buffered ones by more than 2 * margin + kTieEps is skipped,
//     because that cut's exact gain beats all of them by more than kTieEps,
//     so whatever they did to the best cut, it ends up the same once that cut
//     is reached (this keeps rising stretches of the gain cheap). The others
//     get cutGain().
template <int FixedK, Criterion Crit>
CutCandidate CpuTreeBuilder::scanFeatureK(const Entry *entries, std::uint32_t count,
                                          const std::uint32_t *total, double parentWeighted,
                                          std::uint32_t minChild, std::uint32_t *bestLeft) const {
  CutCandidate best;
  if (count < 2 * minChild) {
    return best;
  }
  const int classCount = FixedK > 0 ? FixedK : classCount_;
  const std::size_t classes = static_cast<std::size_t>(classCount);
  // left: class counts of the rows moved left so far; scratch: for
  // settleCandidates(); stored: the counts at each buffered candidate.
  constexpr std::size_t kFixedClasses = FixedK > 0 ? FixedK : 1;
  std::uint32_t fixedLeft[2 * kFixedClasses] = {};
  std::uint32_t fixedStored[kSweepBlock * kFixedClasses];
  std::uint32_t *left = fixedLeft;
  std::uint32_t *stored = fixedStored;
  if constexpr (FixedK == 0) {
    thread_local std::vector<std::uint32_t> dynamicCounts;
    dynamicCounts.assign((kSweepBlock + 2) * classes, 0);
    left = dynamicCounts.data();
    stored = left + 2 * classes;
  }
  std::uint32_t *scratch = left + classes;
  for (std::uint32_t index = 0; index + 1 < minChild; ++index) {
    ++left[codec_.cls(entries[index].packed)];
  }

  const double gap = rules_.minValueGap();
  const LogTable logs = rules_.logTable();
  const CountTables &tables = tables_;
  const double rows = static_cast<double>(count);
  const double inverseRows = 1.0 / rows;
  // Bound on |estimate - cutGain()|: a generous count of roundings times the
  // size of the terms (Gini: at most n; entropy: at most n * log2(n), and up
  // to kSweepBlock updates add 4 roundings each to the running sums), divided
  // by n.
  double margin;
  if constexpr (Crit == Criterion::Gini) {
    margin = 256.0 * kRoundoff;
  } else {
    const double scale = std::max(1.0, tables.xlog2x(count)) * inverseRows;
    margin =
        2.0 * ((4.0 * kSweepBlock + 4.0 * classCount + 16.0) * scale + 64.0) * kRoundoff;
  }
  const double giniBase = parentWeighted - rows;

  // Gini: exact sums of squares of the left and right class counts (signed:
  // converts to double in one instruction; n^2 < 2^62 as n < 2^31). Entropy:
  // sums of c*log2(c) over them.
  std::int64_t leftSquares = 0;
  std::int64_t rightSquares = 0;
  double leftLogs = 0.0;
  double rightLogs = 0.0;
  std::uint32_t updates = 0; // entropy: incremental updates since the last recompute
  auto recompute = [&]() {
    if constexpr (Crit == Criterion::Gini) {
      leftSquares = 0;
      rightSquares = 0;
      for (std::size_t k = 0; k < classes; ++k) {
        const std::int64_t l = left[k];
        const std::int64_t r = total[k] - left[k];
        leftSquares += l * l;
        rightSquares += r * r;
      }
    } else {
      leftLogs = 0.0;
      rightLogs = 0.0;
      for (std::size_t k = 0; k < classes; ++k) {
        leftLogs += tables.xlog2x(left[k]);
        rightLogs += tables.xlog2x(total[k] - left[k]);
      }
      updates = 0;
    }
  };
  recompute();

  std::uint32_t tries = 0;
  double bestGain = -INFINITY;
  std::uint32_t bestCut = 0;
  double threshold = -INFINITY;     // estimates at or below it cannot beat the best cut
  double skipThreshold = -INFINITY; // estimates below it start a skip
  double estimate[kSweepBlock];
  std::uint32_t position[kSweepBlock];
  std::uint32_t found = 0;
  std::uint32_t settleAt = kFirstSettle;
  auto settle = [&]() {
    settleCandidates(
        estimate, position, found, margin, total, classCount, count, parentWeighted, Crit, logs,
        [&](std::uint32_t index, std::uint32_t *out) {
          std::copy_n(stored + index * classes, classes, out);
        },
        scratch, bestGain, bestCut, bestLeft);
    found = 0;
    settleAt = std::min(2 * settleAt, kSweepBlock);
    threshold = bestGain + kTieEps - margin;
    skipThreshold = skipBelow(bestGain, margin, rows / maxGainStep<Crit>(count));
  };
  const double rowsPerGain = rows / maxGainStep<Crit>(count);
  const EntryCodec codec = codec_;
  constexpr std::size_t kLanes = 4;
  std::uint32_t fixedLaneCounts[kLanes * kFixedClasses];
  std::uint32_t *laneCounts = fixedLaneCounts;
  if constexpr (FixedK == 0) {
    thread_local std::vector<std::uint32_t> dynamicLaneCounts;
    dynamicLaneCounts.resize(kLanes * classes);
    laneCounts = dynamicLaneCounts.data();
  }
  // Moves the rows of the cuts from `cut` on left, up to `end` or, with
  // `untilThreshold`, up to the first cut with a threshold, into interleaved
  // counters (rows of the same class in a row then do not wait for each
  // other's increment). Returns where it stopped and how many of the passed
  // cuts have a threshold.
  auto moveRows = [&](std::uint32_t cut, std::uint32_t end, bool untilThreshold) {
    std::fill_n(laneCounts, kLanes * classes, 0u);
    std::uint32_t thresholds = 0;
    for (; cut < end; ++cut) {
      const bool threshold = isCut(entries[cut - 1].value, entries[cut].value, gap);
      if (untilThreshold && threshold) {
        break;
      }
      ++laneCounts[(cut & (kLanes - 1)) * classes + codec.cls(entries[cut - 1].packed)];
      thresholds += threshold;
    }
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
      for (std::size_t k = 0; k < classes; ++k) {
        left[k] += laneCounts[lane * classes + k];
      }
    }
    return std::pair<std::uint32_t, std::uint32_t>{cut, thresholds};
  };

  const std::uint32_t lastCut = count - minChild;
  std::uint32_t moved = 0; // rows moved since the sums were updated
  for (std::uint32_t cut = minChild; cut <= lastCut; ++cut) {
    const Entry previous = entries[cut - 1];
    const float value = entries[cut].value;
    const std::uint32_t cls = codec.cls(previous.packed);
    ++left[cls];
    ++moved;
    if (!isCut(previous.value, value, gap)) {
      // (Nearly) equal values: no threshold fits between them. Move the rest
      // of the run in a tight loop.
      const std::uint32_t next = moveRows(cut + 1, lastCut + 1, true).first;
      moved += next - cut - 1;
      cut = next - 1; // the loop goes on with `next`, a cut with a threshold (or the end)
      continue;
    }
    ++tries;
    const bool leftDistinct = cut < 2 || isCut(entries[cut - 2].value, previous.value, gap);
    const bool rightDistinct = cut + 1 >= count || isCut(value, entries[cut + 1].value, gap);
    const bool skippable = (cls == codec.cls(entries[cut].packed)) & leftDistinct &
                           rightDistinct & (cut > minChild) & (cut + minChild < count);
    if (moved > 1 || (Crit == Criterion::Entropy && updates == kSweepBlock)) {
      recompute();
    } else {
      const std::uint32_t l = left[cls] - 1; // before this row moved
      const std::uint32_t r = total[cls] - l;
      if constexpr (Crit == Criterion::Gini) {
        leftSquares += 2 * std::int64_t{l} + 1;
        rightSquares -= 2 * std::int64_t{r} - 1;
      } else {
        leftLogs += tables.xlog2x(l + 1) - tables.xlog2x(l);
        rightLogs += tables.xlog2x(r - 1) - tables.xlog2x(r);
        ++updates;
      }
    }
    moved = 0;
    double gain;
    if constexpr (Crit == Criterion::Gini) {
      gain = (giniBase + static_cast<double>(leftSquares) * tables.reciprocal(cut) +
              static_cast<double>(rightSquares) * tables.reciprocal(count - cut)) *
             inverseRows;
    } else {
      gain = (parentWeighted - (tables.xlog2x(cut) - leftLogs) -
              (tables.xlog2x(count - cut) - rightLogs)) *
             inverseRows;
    }
    // Buffered unless skipped, without a branch on `skippable`.
    estimate[found] = gain;
    position[found] = cut;
    std::copy_n(left, classes, stored + found * classes);
    found += !skippable & (gain > threshold);
    if (found == settleAt) {
      settle();
    }
    if (!(gain < skipThreshold)) {
      continue;
    }
    // The next cuts cannot beat the best cut while their gain provably stays
    // below it: only move their rows and count their thresholds.
    const std::uint32_t steps =
        skippableSteps(bestGain, gain, margin, rowsPerGain, lastCut - cut);
    tries += moveRows(cut + 1, cut + 1 + steps, false).second;
    moved += steps;
    cut += steps;
  }
  settle();
  best.tries = tries;
  if (bestCut > 0) {
    best.gain = bestGain;
    best.leftCount = bestCut;
    best.leftValue = entries[bestCut - 1].value;
    best.rightValue = entries[bestCut].value;
  }
  return best;
}

} // namespace dt
