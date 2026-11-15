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
// alternate (`temp` is needed only with more than one active pass).
template <class KeyOf>
Entry *sortColumnBy(const float *values, const std::uint16_t *labels, std::size_t count,
                    const EntryCodec &codec, RadixHistogram &histogram, Entry *out, Entry *temp,
                    KeyOf keyOf) {
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
  if (!histogram.counted) { // radixPlan()'s digits, in one read
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
                  const EntryCodec &codec, RadixHistogram &histogram, Entry *out, Entry *temp) {
  if (histogram.integer) {
    const std::int32_t base = histogram.base;
    return sortColumnBy(values, labels, count, codec, histogram, out, temp,
                        [base](float value) {
                          return static_cast<std::uint32_t>(static_cast<std::int32_t>(value) - base);
                        });
  }
  return sortColumnBy(values, labels, count, codec, histogram, out, temp,
                      [](float value) { return sortKey(value); });
}

// ---------------------------------------------------------------------------
// Columns of 1M+ rows: one scatter out of the caches, the rest in them.
// ---------------------------------------------------------------------------
// 1. One read of the raw values (per chunk of rows): the AND and OR of the
//    keys, the range, and whether all are integers (ColumnStats).
// 2. The key (as radixPlan(): sortKey(), or int(value) - min where that needs
//    fewer bits) and the bits in which the keys differ. A column whose keys
//    differ in at most kDirectBits bits is sorted by one counting pass over
//    them (done).
// 3. Otherwise the top kPrefixBits of those bits are counted per chunk, and
//    consecutive prefixes are grouped into buckets of about kBucketEntries
//    entries (float keys are skewed: a few exponents hold most rows, so equal
//    digit widths would make a few huge buckets), and the raw rows are
//    scattered once, stably, into their buckets in the column's slot.
// 4. Each bucket is sorted where it lies by the bits in which its own keys
//    differ (LSD through a per-thread buffer, in cache).
// The result is a stable sort by the key, as the LSD sort's: the same column.
constexpr std::size_t kBigColumnRows = std::size_t{1} << 20;
constexpr int kDirectBits = 12;
constexpr int kPrefixBits = 16;
constexpr std::size_t kBucketEntries = std::size_t{1} << 15;
constexpr std::size_t kMaxBuckets = 4096;
constexpr int kBucketDigitBits = 11;

struct KeyPlan {
  bool integer = false;
  std::int32_t base = 0;
  int low = 0;  // lowest varying bit
  int span = 0; // varying bits [low, low + span)
  int prefixBits = 0;
  int prefixShift = 0;
  std::vector<std::uint16_t> bucketOf; // prefix -> bucket (span > kDirectBits)
  std::size_t buckets = 0;
  std::vector<std::uint32_t> bucketStart; // buckets + 1

  std::uint32_t key(float value) const {
    return integer ? static_cast<std::uint32_t>(static_cast<std::int32_t>(value) - base)
                   : sortKey(value);
  }
  std::uint32_t prefix(std::uint32_t key) const {
    return (key >> prefixShift) & ((1u << prefixBits) - 1);
  }
};

void columnStats(const float *values, std::size_t begin, std::size_t end, ColumnStats &stats) {
  std::uint32_t all = ~0u;
  std::uint32_t any = 0;
  float least = INFINITY;
  float most = -INFINITY;
  std::uint32_t integral = 1;
  for (std::size_t index = begin; index < end; ++index) {
    const float value = values[index] + 0.0f;
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t key = (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
    all &= key;
    any |= key;
    least = std::min(least, value);
    most = std::max(most, value);
    const bool inRange = value >= -0x1p30f && value <= 0x1p30f; // false for NaN
    const float safe = inRange ? value : 0.0f;
    integral &= static_cast<std::uint32_t>(inRange) &
                static_cast<std::uint32_t>(static_cast<float>(static_cast<std::int32_t>(safe)) ==
                                           value);
  }
  stats.all = all;
  stats.any = any;
  stats.low = least;
  stats.high = most;
  stats.integral = integral != 0;
}

void planKey(const ColumnStats &stats, KeyPlan &plan) {
  const std::uint32_t varying = stats.any ^ stats.all;
  if (varying == 0) {
    return; // span 0: one value
  }
  plan.low = std::countr_zero(varying);
  plan.span = 32 - std::countl_zero(varying) - plan.low;
  if (stats.integral) {
    const std::int64_t range =
        static_cast<std::int64_t>(stats.high) - static_cast<std::int64_t>(stats.low);
    const int rangeSpan = std::bit_width(static_cast<std::uint64_t>(range));
    if (rangeSpan < plan.span) {
      plan.integer = true;
      plan.base = static_cast<std::int32_t>(stats.low);
      plan.low = 0;
      plan.span = rangeSpan;
    }
  }
  plan.prefixBits = std::min(plan.span, plan.span <= kDirectBits ? kDirectBits : kPrefixBits);
  plan.prefixShift = plan.low + plan.span - plan.prefixBits;
}

// Buckets from the prefix counts (summed over the chunks): consecutive
// prefixes up to about rows / buckets entries each.
void planBuckets(const std::uint32_t *prefixCounts, std::size_t rows, KeyPlan &plan) {
  const std::size_t prefixes = std::size_t{1} << plan.prefixBits;
  plan.bucketOf.resize(prefixes);
  if (plan.span <= kDirectBits) { // every prefix (the whole varying key) its own bucket
    std::iota(plan.bucketOf.begin(), plan.bucketOf.end(), std::uint16_t{0});
    plan.buckets = prefixes;
    return;
  }
  const std::size_t wanted =
      std::clamp<std::size_t>((rows + kBucketEntries - 1) / kBucketEntries, 1, kMaxBuckets / 2);
  const std::size_t target = (rows + wanted - 1) / wanted;
  std::size_t bucket = 0;
  std::size_t filled = 0;
  for (std::size_t p = 0; p < prefixes; ++p) {
    const std::size_t count = prefixCounts[p];
    if (filled > 0 && filled + count > target && bucket + 1 < kMaxBuckets) {
      ++bucket;
      filled = 0;
    }
    plan.bucketOf[p] = static_cast<std::uint16_t>(bucket);
    filled += count;
  }
  plan.buckets = bucket + 1;
}

using EntryBuffer = std::vector<Entry, UninitializedAllocator<Entry>>;

// Stable sort of entries[0, count) by plan.key(), in cache: LSD over the bits
// in which these keys differ, through `buffer`.
void sortBucket(Entry *entries, std::size_t count, const KeyPlan &plan, EntryBuffer &buffer) {
  if (count < 2) {
    return;
  }
  std::uint32_t all = ~0u;
  std::uint32_t any = 0;
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint32_t key = plan.key(entries[index].value);
    all &= key;
    any |= key;
  }
  const std::uint32_t varying = any ^ all;
  if (varying == 0) {
    return;
  }
  if (count <= 32) { // insertion sort (stable)
    for (std::size_t index = 1; index < count; ++index) {
      const Entry entry = entries[index];
      const std::uint32_t key = plan.key(entry.value);
      std::size_t at = index;
      while (at > 0 && plan.key(entries[at - 1].value) > key) {
        entries[at] = entries[at - 1];
        --at;
      }
      entries[at] = entry;
    }
    return;
  }
  const int low = std::countr_zero(varying);
  const int span = 32 - std::countl_zero(varying) - low;
  const int passes = (span + kBucketDigitBits - 1) / kBucketDigitBits;
  int shift[kRadixPasses];
  std::uint32_t mask[kRadixPasses];
  int at = low;
  for (int pass = 0; pass < passes; ++pass) {
    const int bits = span / passes + (pass < span % passes ? 1 : 0);
    shift[pass] = at;
    mask[pass] = (1u << bits) - 1;
    at += bits;
  }
  thread_local std::vector<std::uint32_t> countStore(std::size_t{kRadixPasses}
                                                     << kBucketDigitBits);
  std::uint32_t *counts[kRadixPasses];
  for (int pass = 0; pass < passes; ++pass) {
    counts[pass] = countStore.data() + (static_cast<std::size_t>(pass) << kBucketDigitBits);
    std::fill_n(counts[pass], std::size_t{mask[pass]} + 1, 0u);
  }
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint32_t key = plan.key(entries[index].value);
    for (int pass = 0; pass < passes; ++pass) {
      ++counts[pass][(key >> shift[pass]) & mask[pass]];
    }
  }
  if (buffer.size() < count) {
    buffer.resize(count);
  }
  Entry *source = entries;
  Entry *target = buffer.data();
  for (int pass = 0; pass < passes; ++pass) {
    std::uint32_t *bucket = counts[pass];
    if (bucket[(plan.key(source[0].value) >> shift[pass]) & mask[pass]] == count) {
      continue; // one digit throughout
    }
    std::uint32_t offset = 0;
    for (std::uint32_t digit = 0; digit <= mask[pass]; ++digit) {
      const std::uint32_t size = bucket[digit];
      bucket[digit] = offset;
      offset += size;
    }
    for (std::size_t index = 0; index < count; ++index) {
      const Entry entry = source[index];
      target[bucket[(plan.key(entry.value) >> shift[pass]) & mask[pass]]++] = entry;
    }
    std::swap(source, target);
  }
  if (source != entries) {
    std::memcpy(entries, source, count * sizeof(Entry));
  }
}

// One column on this thread (as presortBig() with one chunk), into `out`.
void sortWholeColumn(const float *values, const std::uint16_t *labels, std::size_t rows,
                     const EntryCodec &codec, Entry *out) {
  ColumnStats stats;
  columnStats(values, 0, rows, stats);
  KeyPlan plan;
  planKey(stats, plan);
  auto entryOf = [&](std::size_t row, float value) {
    return Entry{value, codec.pack(static_cast<std::uint32_t>(row), labels[row])};
  };
  if (plan.span == 0) {
    for (std::size_t row = 0; row < rows; ++row) {
      out[row] = entryOf(row, values[row] + 0.0f);
    }
    return;
  }
  thread_local std::vector<std::uint32_t> counts;
  counts.assign(std::size_t{1} << plan.prefixBits, 0u);
  for (std::size_t row = 0; row < rows; ++row) {
    ++counts[plan.prefix(plan.key(values[row] + 0.0f))];
  }
  planBuckets(counts.data(), rows, plan);
  thread_local std::vector<std::uint32_t> offset;
  offset.assign(plan.buckets + 1, 0u);
  for (std::size_t p = 0; p < counts.size(); ++p) {
    offset[plan.bucketOf[p] + 1] += counts[p];
  }
  std::partial_sum(offset.begin(), offset.end(), offset.begin());
  plan.bucketStart.assign(offset.begin(), offset.end());
  const std::uint16_t *bucketOf = plan.bucketOf.data();
  for (std::size_t row = 0; row < rows; ++row) {
    const float value = values[row] + 0.0f;
    out[offset[bucketOf[plan.prefix(plan.key(value))]]++] = entryOf(row, value);
  }
  if (plan.span <= kDirectBits) {
    return;
  }
  thread_local EntryBuffer buffer;
  for (std::size_t bucket = 0; bucket < plan.buckets; ++bucket) {
    const std::uint32_t begin = plan.bucketStart[bucket];
    sortBucket(out + begin, plan.bucketStart[bucket + 1] - begin, plan, buffer);
  }
}

// Columns of kBigColumnRows+ rows, each into its own slot (feature f in slot
// f). `chunks` row chunks per column for the passes over the raw values; with
// at least as many columns as threads, a column per task (sortWholeColumn()).
SlotMap presortBig(const Dataset &dataset, const EntryCodec &codec, Entry *data,
                   ThreadPool *pool) {
  const std::size_t rows = dataset.rowCount;
  const std::size_t features = dataset.featureCount();
  const std::size_t workers = pool ? pool->workerCount() + 1 : 1;
  const std::size_t chunks =
      std::clamp<std::size_t>((4 * workers + features - 1) / features, 1, rows / 65536 + 1);
  const std::uint16_t *labels = dataset.labels.data();
  SlotMap slotOf(features);
  std::iota(slotOf.begin(), slotOf.end(), 0u);
  if (chunks == 1) {
    parallelFor(pool, features, [&](std::size_t feature) {
      sortWholeColumn(dataset.column(feature), labels, rows, codec, data + feature * rows);
    });
    return slotOf;
  }
  auto chunkBegin = [&](std::size_t chunk) { return rows * chunk / chunks; };

  // 1-2. Stats and key plans.
  std::vector<ColumnStats> chunkStats(features * chunks);
  parallelFor(pool, features * chunks, [&](std::size_t task) {
    const std::size_t chunk = task % chunks;
    columnStats(dataset.column(task / chunks), chunkBegin(chunk), chunkBegin(chunk + 1),
                chunkStats[task]);
  });
  std::vector<KeyPlan> plans(features);
  for (std::size_t feature = 0; feature < features; ++feature) {
    ColumnStats merged = chunkStats[feature * chunks];
    for (std::size_t chunk = 1; chunk < chunks; ++chunk) {
      const ColumnStats &part = chunkStats[feature * chunks + chunk];
      merged.all &= part.all;
      merged.any |= part.any;
      merged.low = std::min(merged.low, part.low);
      merged.high = std::max(merged.high, part.high);
      merged.integral = merged.integral && part.integral;
    }
    planKey(merged, plans[feature]);
  }

  // 3. Prefix counts per chunk, buckets, the offsets of every (bucket, chunk)
  //    (bucket-major: the scatter is stable), and the scatter.
  std::vector<std::vector<std::uint32_t>> chunkCounts(features * chunks);
  parallelFor(pool, features * chunks, [&](std::size_t task) {
    const KeyPlan &plan = plans[task / chunks];
    if (plan.span == 0) {
      return;
    }
    std::vector<std::uint32_t> &counts = chunkCounts[task];
    counts.assign(std::size_t{1} << plan.prefixBits, 0u);
    const float *values = dataset.column(task / chunks);
    const std::size_t chunk = task % chunks;
    const std::size_t end = chunkBegin(chunk + 1);
    for (std::size_t row = chunkBegin(chunk); row < end; ++row) {
      ++counts[plan.prefix(plan.key(values[row] + 0.0f))];
    }
  });
  // offsets[task][bucket]: where the chunk's first entry of the bucket goes.
  std::vector<std::vector<std::uint32_t>> offsets(features * chunks);
  parallelFor(pool, features, [&](std::size_t feature) {
    KeyPlan &plan = plans[feature];
    if (plan.span == 0) {
      return;
    }
    const std::size_t prefixes = std::size_t{1} << plan.prefixBits;
    std::vector<std::uint32_t> total(chunkCounts[feature * chunks]);
    for (std::size_t chunk = 1; chunk < chunks; ++chunk) {
      const std::uint32_t *add = chunkCounts[feature * chunks + chunk].data();
      for (std::size_t p = 0; p < prefixes; ++p) {
        total[p] += add[p];
      }
    }
    planBuckets(total.data(), rows, plan);
    std::vector<std::uint32_t> sizes(plan.buckets * chunks, 0u); // chunk-major
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      const std::uint32_t *counts = chunkCounts[feature * chunks + chunk].data();
      std::uint32_t *size = sizes.data() + chunk * plan.buckets;
      for (std::size_t p = 0; p < prefixes; ++p) {
        size[plan.bucketOf[p]] += counts[p];
      }
    }
    plan.bucketStart.assign(plan.buckets + 1, 0u);
    std::uint32_t offset = 0;
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      offsets[feature * chunks + chunk].resize(plan.buckets);
    }
    for (std::size_t bucket = 0; bucket < plan.buckets; ++bucket) {
      plan.bucketStart[bucket] = offset;
      for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        offsets[feature * chunks + chunk][bucket] = offset;
        offset += sizes[chunk * plan.buckets + bucket];
      }
    }
    plan.bucketStart[plan.buckets] = offset;
  });
  parallelFor(pool, features * chunks, [&](std::size_t task) {
    const std::size_t feature = task / chunks;
    const KeyPlan &plan = plans[feature];
    const float *values = dataset.column(feature);
    Entry *out = data + feature * rows;
    const std::size_t chunk = task % chunks;
    const std::size_t end = chunkBegin(chunk + 1);
    auto entryOf = [&](std::size_t row, float value) {
      return Entry{value, codec.pack(static_cast<std::uint32_t>(row), labels[row])};
    };
    if (plan.span == 0) {
      for (std::size_t row = chunkBegin(chunk); row < end; ++row) {
        out[row] = entryOf(row, values[row] + 0.0f);
      }
      return;
    }
    std::uint32_t *offset = offsets[task].data();
    const std::uint16_t *bucketOf = plan.bucketOf.data();
    for (std::size_t row = chunkBegin(chunk); row < end; ++row) {
      const float value = values[row] + 0.0f;
      out[offset[bucketOf[plan.prefix(plan.key(value))]]++] = entryOf(row, value);
    }
  });

  // 4. The buckets of the columns with more varying bits than the prefix's,
  //    in groups of about the same number of entries.
  struct Group {
    std::uint32_t feature;
    std::uint32_t first; // buckets [first, last)
    std::uint32_t last;
  };
  std::vector<Group> groups;
  const std::size_t groupEntries =
      std::max<std::size_t>(kBucketEntries, rows * features / (8 * workers) + 1);
  for (std::size_t feature = 0; feature < features; ++feature) {
    const KeyPlan &plan = plans[feature];
    if (plan.span <= kDirectBits) {
      continue;
    }
    std::uint32_t first = 0;
    for (std::uint32_t bucket = 0; bucket < plan.buckets; ++bucket) {
      if (plan.bucketStart[bucket + 1] - plan.bucketStart[first] >= groupEntries) {
        groups.push_back({static_cast<std::uint32_t>(feature), first, bucket + 1});
        first = bucket + 1;
      }
    }
    if (first < plan.buckets) {
      groups.push_back({static_cast<std::uint32_t>(feature), first,
                        static_cast<std::uint32_t>(plan.buckets)});
    }
  }
  parallelFor(pool, groups.size(), [&](std::size_t index) {
    const Group &group = groups[index];
    const KeyPlan &plan = plans[group.feature];
    Entry *out = data + std::size_t{group.feature} * rows;
    thread_local EntryBuffer buffer;
    for (std::uint32_t bucket = group.first; bucket < group.last; ++bucket) {
      const std::uint32_t begin = plan.bucketStart[bucket];
      sortBucket(out + begin, plan.bucketStart[bucket + 1] - begin, plan, buffer);
    }
  });
  return slotOf;
}

} // namespace

SlotMap presortColumns(const Dataset &dataset, const EntryCodec &codec, Entry *data,
                       std::size_t slots, ThreadPool *pool) {
  const std::size_t rows = dataset.rowCount;
  const std::size_t features = dataset.featureCount();
  if (rows > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("presortColumns: too many rows");
  }
  if (slots <= features) {
    throw std::logic_error("presortColumns: needs a slot more than features");
  }
  // On one thread the LSD passes below are faster (one thread does not
  // saturate the memory: HIGGS and chess presorts 3-8% slower this way).
  if (rows >= kBigColumnRows && pool && pool->workerCount() > 0) {
    return presortBig(dataset, codec, data, pool);
  }
  // Histograms first, so that the features with the most passes are sorted
  // first (features differ a lot: binary ones need one pass, continuous ones
  // three), which balances the threads.
  std::unique_ptr<RadixHistogram[]> histograms(new RadixHistogram[features]);
  // Columns of 1M+ rows (here on one thread) get their digits from
  // radixPlan(). Smaller ones stay in cache, where the pass it saves costs
  // about as much as its extra reads.
  const bool big = rows >= kBigColumnRows;
  std::vector<ColumnStats> stats(features);
  parallelFor(pool, features, [&](std::size_t feature) {
    RadixHistogram &histogram = histograms[feature];
    radixCount(dataset.column(feature), 0, rows, histogram.counts);
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
    for (std::size_t feature = 0; feature < features; ++feature) {
      ColumnStats &column = stats[feature];
      column.integral = histograms[feature].activePasses >= 2 &&
                        integerRange(dataset.column(feature), 0, rows, column.low, column.high);
      radixPlan(column, histograms[feature]);
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
  // one that does not end up with the result.
  SlotMap slotOf(features);
  std::vector<std::uint32_t> free(slots);
  std::iota(free.rbegin(), free.rend(), 0u); // taken from the back: lowest first
  std::mutex mutex;
  std::condition_variable freed;
  parallelFor(pool, features, [&](std::size_t index) {
    const std::size_t feature = order[index];
    const bool workspace = histograms[feature].activePasses > 1;
    std::uint32_t slot;
    std::uint32_t spare = 0;
    {
      std::unique_lock<std::mutex> lock(mutex);
      freed.wait(lock, [&]() { return free.size() >= (workspace ? 2u : 1u); });
      slot = free.back();
      free.pop_back();
      if (workspace) {
        spare = free.back();
        free.pop_back();
      }
    }
    Entry *out = data + std::size_t{slot} * rows;
    const Entry *result =
        sortColumn(dataset.column(feature), dataset.labels.data(), rows, codec,
                   histograms[feature], out, workspace ? data + std::size_t{spare} * rows : nullptr);
    {
      std::lock_guard<std::mutex> lock(mutex);
      slotOf[feature] = result == out ? slot : spare;
      if (workspace) {
        free.push_back(result == out ? spare : slot);
      }
    }
    freed.notify_all();
  });
  return slotOf;
}

} // namespace dt
