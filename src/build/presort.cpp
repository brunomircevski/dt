#include "build/presort.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

// Map a float to an unsigned key with the same order (negatives flipped).
std::uint32_t sortKey(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

// Stable LSD radix sort by value: 3 passes of 11 bits. Passes in which every
// key has the same digit (common for small integer and binary features) are
// skipped. Columns of 1M+ rows are counted in 11-bit digits, then radixPlan()
// may sort them in fewer passes of up to 12 bits.
constexpr int kRadixBits = 11;
constexpr std::size_t kRadixBuckets = std::size_t{1} << kRadixBits;
constexpr int kRadixPasses = 3;
constexpr int kRadixMaxBits = 12;
constexpr std::size_t kRadixMaxBuckets = std::size_t{1} << kRadixMaxBits;

struct RadixHistogram {
  std::uint32_t counts[kRadixPasses][kRadixMaxBuckets];
  bool active[kRadixPasses]; // the pass changes the order
  int activePasses;
  // Digit of pass p: (key >> shift[p]) & mask[p]. The key is sortKey(value),
  // or with `integer` int(value) - base: the same order.
  std::uint32_t shift[kRadixPasses] = {0, kRadixBits, 2 * kRadixBits};
  std::uint32_t mask[kRadixPasses] = {kRadixBuckets - 1, kRadixBuckets - 1, kRadixBuckets - 1};
  bool integer = false;
  std::int32_t base = 0;
  bool counted = true; // counts holds the active passes' histograms

  std::size_t buckets(int pass) const { return std::size_t{mask[pass]} + 1; }
};

// What radixPlan() needs to know about a column's keys: the
// bits in which they differ, and whether every value is an integer.
struct ColumnStats {
  std::uint32_t all = ~0u; // AND of the keys
  std::uint32_t any = 0;   // OR of the keys
  float low = INFINITY;
  float high = -INFINITY;
  bool integral = true;
};

// The 11-bit digit counts of rows [begin, end) (into `counts`, zeroed here).
void radixCount(const float *values, std::size_t begin, std::size_t end,
                std::uint32_t (*counts)[kRadixMaxBuckets]) {
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    std::fill_n(counts[pass], kRadixBuckets, 0u);
  }
  for (std::size_t index = begin; index < end; ++index) {
    const std::uint32_t key = sortKey(values[index] + 0.0f);
    ++counts[0][key & (kRadixBuckets - 1)];
    ++counts[1][(key >> kRadixBits) & (kRadixBuckets - 1)];
    ++counts[2][key >> (2 * kRadixBits)];
  }
}

// The AND and OR of the keys counted in `histogram`: per 11-bit digit, those
// of the digits that occur.
void keyBits(const RadixHistogram &histogram, ColumnStats &stats) {
  stats.all = 0;
  stats.any = 0;
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    std::uint32_t all = kRadixBuckets - 1;
    std::uint32_t any = 0;
    for (std::uint32_t digit = 0; digit < kRadixBuckets; ++digit) {
      if (histogram.counts[pass][digit] != 0) {
        all &= digit;
        any |= digit;
      }
    }
    stats.all |= all << (pass * kRadixBits);
    stats.any |= any << (pass * kRadixBits);
  }
}

// Whether rows [begin, end) are all integers within +-2^30 (stops at the
// first block that is not), and their range.
bool integerRange(const float *values, std::size_t begin, std::size_t end, float &low,
                  float &high) {
  constexpr std::size_t kBlock = 4096;
  float least = INFINITY;
  float most = -INFINITY;
  for (std::size_t block = begin; block < end; block += kBlock) {
    const std::size_t stop = std::min(end, block + kBlock);
    std::uint32_t integral = 1;
    for (std::size_t index = block; index < stop; ++index) {
      const float value = values[index] + 0.0f;
      least = std::min(least, value);
      most = std::max(most, value);
      const bool inRange = value >= -0x1p30f && value <= 0x1p30f; // false for NaN
      const float safe = inRange ? value : 0.0f;
      integral &= static_cast<std::uint32_t>(inRange) &
                  static_cast<std::uint32_t>(
                      static_cast<float>(static_cast<std::int32_t>(safe)) == value);
    }
    if (!integral) {
      return false;
    }
  }
  low = least;
  high = most;
  return true;
}

// The digits for a column with these stats: only the bits in which its keys
// differ, in as few passes of at most 12 bits as possible; integer columns
// whose range needs fewer passes than that use int(value) - min as the key.
// The order is the same (stable sort, the other bits are equal everywhere).
// Taken only if that needs fewer passes than the 11-bit digits counted
// already (`histogram`'s active passes): the digits are then counted again.
void radixPlan(const ColumnStats &stats, RadixHistogram &histogram) {
  auto passesFor = [](int span) { return (span + kRadixMaxBits - 1) / kRadixMaxBits; };
  const std::uint32_t varying = stats.any ^ stats.all;
  int low = 0;
  int span = 0;
  if (varying != 0) {
    low = std::countr_zero(varying);
    span = 32 - std::countl_zero(varying) - low;
  }
  bool integer = false;
  if (stats.integral && varying != 0) {
    const std::int64_t range =
        static_cast<std::int64_t>(stats.high) - static_cast<std::int64_t>(stats.low);
    const int rangeSpan = std::bit_width(static_cast<std::uint64_t>(range));
    if (passesFor(rangeSpan) < passesFor(span)) {
      integer = true;
      low = 0;
      span = rangeSpan;
    }
  }
  const int passes = passesFor(span);
  if (passes >= histogram.activePasses) {
    return;
  }
  histogram.integer = integer;
  histogram.base = integer ? static_cast<std::int32_t>(stats.low) : 0;
  histogram.activePasses = 0;
  int shift = low;
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    const int bits = pass < passes ? span / passes + (pass < span % passes ? 1 : 0) : 1;
    histogram.shift[pass] = static_cast<std::uint32_t>(std::min(shift, 31));
    histogram.mask[pass] = (1u << bits) - 1;
    histogram.active[pass] =
        pass < passes && (histogram.integer || ((varying >> shift) & histogram.mask[pass]) != 0);
    histogram.activePasses += histogram.active[pass];
    shift += bits;
  }
  histogram.counted = false;
}

// Sorted entries of one feature column, in `out` or `temp` (returned): the
// first active pass reads the raw values into `out`, then the buffers
// alternate (`temp` is needed only with more than one active pass). With
// `parts` > 1, every pass is split into that many ranges of rows on `pool`:
// each range counts its digits, and scatters from its own offsets (after
// those of the earlier ranges, so the sort stays stable).
template <class KeyOf>
Entry *sortColumnBy(const float *values, const std::uint16_t *labels, std::size_t count,
                    const EntryCodec &codec, RadixHistogram &histogram, Entry *out, Entry *temp,
                    ThreadPool *pool, std::size_t parts, KeyOf keyOf) {
  auto entryOf = [&](std::size_t row) {
    // +0.0f: -0.0 and 0.0 are the same value (no threshold between them).
    return Entry{values[row] + 0.0f, codec.pack(static_cast<std::uint32_t>(row), labels[row])};
  };
  if (histogram.activePasses == 0) {
    for (std::size_t row = 0; row < count; ++row) {
      out[row] = entryOf(row);
    }
    return out;
  }
  if (parts <= 1 && !histogram.counted) { // radixPlan()'s digits, in one read
    int active[kRadixPasses] = {0, 0, 0};
    int activeCount = 0;
    for (int pass = 0; pass < kRadixPasses; ++pass) {
      if (histogram.active[pass]) {
        std::fill_n(histogram.counts[pass], histogram.buckets(pass), 0u);
        active[activeCount++] = pass;
      }
    }
    std::uint32_t *c0 = histogram.counts[active[0]];
    std::uint32_t *c1 = histogram.counts[active[1]];
    const std::uint32_t s0 = histogram.shift[active[0]], m0 = histogram.mask[active[0]];
    const std::uint32_t s1 = histogram.shift[active[1]], m1 = histogram.mask[active[1]];
    if (activeCount == 1) {
      for (std::size_t row = 0; row < count; ++row) {
        ++c0[(keyOf(values[row] + 0.0f) >> s0) & m0];
      }
    } else { // 2: radixPlan() takes fewer than 3 passes only
      for (std::size_t row = 0; row < count; ++row) {
        const std::uint32_t key = keyOf(values[row] + 0.0f);
        ++c0[(key >> s0) & m0];
        ++c1[(key >> s1) & m1];
      }
    }
    histogram.counted = true;
  }
  Entry *target = out;
  Entry *other = temp;
  Entry *source = nullptr; // null: the raw values
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    if (!histogram.active[pass]) {
      continue;
    }
    const std::uint32_t shift = histogram.shift[pass];
    const std::uint32_t mask = histogram.mask[pass];
    const std::size_t bucketCount = histogram.buckets(pass);
    if (parts > 1) {
      auto digitAt = [&](std::size_t index) {
        const float value = source ? source[index].value : values[index] + 0.0f;
        return (keyOf(value) >> shift) & mask;
      };
      std::vector<std::uint32_t> buckets(parts * bucketCount, 0);
      parallelFor(pool, parts, [&](std::size_t part) {
        std::uint32_t *bucket = buckets.data() + part * bucketCount;
        const std::size_t end = count * (part + 1) / parts;
        for (std::size_t index = count * part / parts; index < end; ++index) {
          ++bucket[digitAt(index)];
        }
      });
      std::uint32_t offset = 0;
      for (std::size_t digit = 0; digit < bucketCount; ++digit) {
        for (std::size_t part = 0; part < parts; ++part) {
          const std::uint32_t size = buckets[part * bucketCount + digit];
          buckets[part * bucketCount + digit] = offset;
          offset += size;
        }
      }
      parallelFor(pool, parts, [&](std::size_t part) {
        std::uint32_t *bucket = buckets.data() + part * bucketCount;
        const std::size_t end = count * (part + 1) / parts;
        for (std::size_t index = count * part / parts; index < end; ++index) {
          const Entry entry = source ? source[index] : entryOf(index);
          target[bucket[(keyOf(entry.value) >> shift) & mask]++] = entry;
        }
      });
      source = target;
      std::swap(target, other);
      continue;
    }
    std::uint32_t *bucket = histogram.counts[pass];
    std::uint32_t offset = 0;
    for (std::size_t digit = 0; digit < bucketCount; ++digit) {
      const std::uint32_t size = bucket[digit];
      bucket[digit] = offset;
      offset += size;
    }
    auto place = [&](const Entry &entry) {
      const std::uint32_t at = bucket[(keyOf(entry.value) >> shift) & mask]++;
      target[at] = entry;
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
  return source;
}

Entry *sortColumn(const float *values, const std::uint16_t *labels, std::size_t count,
                  const EntryCodec &codec, RadixHistogram &histogram, Entry *out, Entry *temp,
                  ThreadPool *pool, std::size_t parts) {
  if (histogram.integer) {
    const std::int32_t base = histogram.base;
    return sortColumnBy(values, labels, count, codec, histogram, out, temp, pool, parts,
                        [base](float value) {
                          return static_cast<std::uint32_t>(static_cast<std::int32_t>(value) - base);
                        });
  }
  return sortColumnBy(values, labels, count, codec, histogram, out, temp, pool, parts,
                      [](float value) { return sortKey(value); });
}

} // namespace

SlotMap presortColumns(const Dataset &dataset, const EntryCodec &codec, Entry *data,
                       std::size_t slots, ThreadPool *pool) {
  const std::size_t rows = dataset.rowCount;
  const std::size_t features = dataset.featureCount();
  if (rows > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("presortColumns: too many rows");
  }
  // Histograms first, so that the features with the most passes are sorted
  // first (features differ a lot: binary ones need one pass, continuous ones
  // three), which balances the threads.
  std::unique_ptr<RadixHistogram[]> histograms(new RadixHistogram[features]);
  // Columns of 1M+ rows are counted in chunks on all threads (when there are
  // fewer columns than threads: one thread per column made the counts most of
  // the presort), and radixPlan() picks their digits. Smaller ones stay in
  // cache, where the pass it saves costs about as much as its extra reads.
  const bool big = rows >= (std::size_t{1} << 20);
  const std::size_t workers = pool ? pool->workerCount() + 1 : 1;
  const std::size_t chunks =
      big ? std::max<std::size_t>(1, (4 * workers + features - 1) / features) : 1;
  std::unique_ptr<RadixHistogram[]> chunkCounts(chunks > 1 ? new RadixHistogram[features * chunks]
                                                           : nullptr);
  parallelFor(pool, features * chunks, [&](std::size_t task) {
    const std::size_t feature = task / chunks;
    const std::size_t chunk = task % chunks;
    radixCount(dataset.column(feature), rows * chunk / chunks, rows * (chunk + 1) / chunks,
               chunks > 1 ? chunkCounts[task].counts : histograms[feature].counts);
  });
  std::vector<ColumnStats> stats(features);
  parallelFor(pool, features, [&](std::size_t feature) {
    RadixHistogram &histogram = histograms[feature];
    if (chunks > 1) {
      for (int pass = 0; pass < kRadixPasses; ++pass) {
        std::uint32_t *sum = histogram.counts[pass];
        std::copy_n(chunkCounts[feature * chunks].counts[pass], kRadixBuckets, sum);
        for (std::size_t chunk = 1; chunk < chunks; ++chunk) {
          const std::uint32_t *add = chunkCounts[feature * chunks + chunk].counts[pass];
          for (std::size_t digit = 0; digit < kRadixBuckets; ++digit) {
            sum[digit] += add[digit];
          }
        }
      }
    }
    // The passes whose digit is not the same everywhere.
    const std::uint32_t firstKey = rows > 0 ? sortKey(dataset.column(feature)[0] + 0.0f) : 0;
    histogram.activePasses = 0;
    for (int pass = 0; pass < kRadixPasses; ++pass) {
      const std::uint32_t digit = (firstKey >> (pass * kRadixBits)) & (kRadixBuckets - 1);
      histogram.active[pass] = histogram.counts[pass][digit] != rows;
      histogram.activePasses += histogram.active[pass];
    }
    if (big) {
      keyBits(histogram, stats[feature]);
    }
  });
  if (big) {
    // Integer keys can save a pass only where there are two or more.
    std::vector<ColumnStats> ranges(features * chunks);
    parallelFor(pool, ranges.size(), [&](std::size_t task) {
      const std::size_t feature = task / chunks;
      const std::size_t chunk = task % chunks;
      ColumnStats &range = ranges[task];
      range.integral = histograms[feature].activePasses >= 2 &&
                       integerRange(dataset.column(feature), rows * chunk / chunks,
                                    rows * (chunk + 1) / chunks, range.low, range.high);
    });
    for (std::size_t feature = 0; feature < features; ++feature) {
      ColumnStats &merged = stats[feature];
      for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const ColumnStats &range = ranges[feature * chunks + chunk];
        merged.integral = merged.integral && range.integral;
        merged.low = std::min(merged.low, range.low);
        merged.high = std::max(merged.high, range.high);
      }
      radixPlan(merged, histograms[feature]);
    }
  }
  std::vector<std::size_t> order(features);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return histograms[a].activePasses > histograms[b].activePasses;
  });
  if (slots <= features) {
    throw std::logic_error("presortColumns: needs a slot more than features");
  }
  // The free slots, handed out under a lock: a sort takes one for its column
  // and, with more than one pass, one more as workspace, and gives back the
  // one that does not end up with the result. Fewer sorts at once than
  // threads: every column is sorted in parts.
  const std::size_t threads = pool ? pool->workerCount() + 1 : 1;
  SlotMap slotOf(features);
  std::vector<std::uint32_t> free(slots);
  std::iota(free.rbegin(), free.rend(), 0u); // taken from the back: lowest first
  std::size_t sorted = 0;
  std::mutex mutex;
  std::condition_variable freed;
  parallelFor(pool, features, [&](std::size_t index) {
    const std::size_t feature = order[index];
    const bool workspace = histograms[feature].activePasses > 1;
    std::uint32_t slot;
    std::uint32_t spare = 0;
    std::size_t parts = 1;
    {
      std::unique_lock<std::mutex> lock(mutex);
      freed.wait(lock, [&]() { return free.size() >= (workspace ? 2u : 1u); });
      slot = free.back();
      free.pop_back();
      if (workspace) {
        spare = free.back();
        free.pop_back();
      }
      // Sorts that can run at once: the features left, and the slots for them.
      const std::size_t room =
          std::max<std::size_t>(1, std::min(features - index, (slots - sorted) / 2));
      if (rows >= (std::size_t{1} << 20)) {
        parts = (threads + room - 1) / room;
      }
    }
    Entry *out = data + std::size_t{slot} * rows;
    const Entry *result =
        sortColumn(dataset.column(feature), dataset.labels.data(), rows, codec,
                   histograms[feature], out, workspace ? data + std::size_t{spare} * rows : nullptr,
                   pool, parts);
    {
      std::lock_guard<std::mutex> lock(mutex);
      slotOf[feature] = result == out ? slot : spare;
      if (workspace) {
        free.push_back(result == out ? spare : slot);
      }
      ++sorted;
    }
    freed.notify_all();
  });
  return slotOf;
}

} // namespace dt
