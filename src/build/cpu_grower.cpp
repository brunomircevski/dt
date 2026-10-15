#include "build/cpu_builder.h"
#include "build/grower.h"
#include "core/timing.h"

#include <bit>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace dt {

namespace {

class CpuGrower final : public Grower {
public:
  CpuGrower(const Dataset &train, const Options &options, ThreadPool *pool, bool reusable)
      : train_(train), options_(options), pool_(pool), reusable_(reusable),
        rows_(train.rowCount), features_(train.featureCount()),
        codec_(makeEntryCodec(train.classCount(), train.rowCount)),
        work_(new Entry[features_ * rows_]), scratch_(new Entry[rows_]),
        goesLeft_(new std::uint8_t[rows_]) {
    if (reusable_) {
      sorted_.reset(new Entry[features_ * rows_]);
    }
    if (pool_) {
      other_.reset(new Entry[features_ * rows_]);
      leftBits_.reset(new std::uint64_t[rows_ / 64 + 1]);
    }
  }

  Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows,
            GrowTimings &timings, DistinctValues *distinct, RowLeaves *leafOfRow) override {
    if (leafOfRow && !rows.empty()) {
      throw std::logic_error("CpuGrower: leaves of rows only with all rows");
    }
    const std::size_t count = rows.empty() ? rows_ : rows.size();
    {
      ScopedTimer timer(timings.prepareSeconds);
      prepare(rows);
      if (distinct) {
        distinctValues(count, *distinct);
      }
    }
    Tree tree;
    tree.featureNames = train_.featureNames;
    tree.classNames = train_.classNames;
    ScopedTimer timer(timings.buildSeconds);
    const std::vector<std::uint32_t> counts = countClasses(train_, rows, pool_);
    NodeStore store(train_.classCount(), 2 * count);
    Subtree root;
    root.node = store.add(counts.data());
    root.count = static_cast<std::uint32_t>(count);
    root.features.resize(features_);
    std::iota(root.features.begin(), root.features.end(), 0u);
    if (leafOfRow) {
      leafOfRow->ids.resize(count);
      leafOfRow->map.clear();
    }
    CpuTreeBuilder builder(rules, codec_, features_, store, goesLeft_.get(), pool_,
                           options_.parallel, countTables(),
                           leafOfRow ? leafOfRow->ids.data() : nullptr);
    builder.grow(Columns{work_.get(), count, scratch_.get(), other_.get(), leftBits_.get()},
                 std::move(root));
    if (pool_) {
      pool_->waitIdle();
    }
    store.toTree(tree, pool_, leafOfRow);
    return tree;
  }

  std::span<std::byte> workspace() override {
    return {reinterpret_cast<std::byte *>(work_.get()), features_ * rows_ * sizeof(Entry)};
  }

private:
  // The sweep's per-count tables, for every count up to the number of
  // training rows (the builder's default tables cover small training sets).
  CountTables countTables() {
    if (rows_ < kLogTableSize) {
      return {};
    }
    if (countXlog_.empty()) {
      fillCountTables(rows_ + 1, countXlog_, countInverse_, pool_);
    }
    return {countXlog_.data(), countInverse_.data(), static_cast<std::uint32_t>(rows_ + 1)};
  }

  // Every feature's distinct values, from the sorted columns in work_ (stride
  // `count`): the last value of each run, in chunks on the pool.
  void distinctValues(std::size_t count, DistinctValues &distinct) {
    constexpr std::size_t kChunk = std::size_t{1} << 20;
    const std::size_t chunks = (count + kChunk - 1) / kChunk;
    std::vector<std::vector<float>> parts(features_ * chunks);
    parallelFor(pool_, parts.size(), [&](std::size_t part) {
      const Entry *column = work_.get() + part / chunks * count;
      const std::size_t begin = part % chunks * kChunk;
      const std::size_t end = std::min(count, begin + kChunk);
      std::vector<float> &out = parts[part];
      for (std::size_t index = begin; index < end; ++index) {
        const float value = column[index].value;
        if (index + 1 == count ||
            std::bit_cast<std::uint32_t>(value) != std::bit_cast<std::uint32_t>(column[index + 1].value)) {
          out.push_back(value);
        }
      }
    });
    distinct.values.clear();
    distinct.offsets.assign(1, 0);
    for (std::size_t part = 0; part < parts.size(); ++part) {
      distinct.values.insert(distinct.values.end(), parts[part].begin(), parts[part].end());
      if (part % chunks == chunks - 1) {
        distinct.offsets.push_back(distinct.values.size());
      }
    }
    if (chunks == 0) {
      distinct.offsets.assign(features_ + 1, 0);
    }
  }

  // Fill work_ with the sorted columns of the selected rows (stride = their
  // count). Without reuse, presort straight into work_.
  void prepare(std::span<const std::uint32_t> rows) {
    if (!reusable_) {
      if (used_ || !rows.empty()) {
        throw std::logic_error("CpuGrower: not created for repeated use");
      }
      used_ = true;
      presortColumns(train_, codec_, work_.get(), pool_, other_.get());
      return;
    }
    if (!presorted_) {
      presortColumns(train_, codec_, sorted_.get(), pool_, other_.get());
      presorted_ = true;
    }
    if (rows.empty()) {
      parallelFor(pool_, features_, [&](std::size_t feature) {
        std::memcpy(work_.get() + feature * rows_, sorted_.get() + feature * rows_,
                    rows_ * sizeof(Entry));
      });
      return;
    }
    // Keep the selected rows of every sorted column: still sorted, and the
    // row ids stay those of the full training set.
    std::vector<std::uint8_t> keep(rows_, 0);
    for (std::uint32_t row : rows) {
      keep[row] = 1;
    }
    const std::size_t stride = rows.size();
    parallelFor(pool_, features_, [&](std::size_t feature) {
      const Entry *source = sorted_.get() + feature * rows_;
      Entry *target = work_.get() + feature * stride;
      std::size_t written = 0;
      for (std::size_t index = 0; index < rows_; ++index) {
        if (keep[codec_.row(source[index].packed)]) {
          target[written++] = source[index];
        }
      }
    });
  }

  const Dataset &train_;
  const Options &options_;
  ThreadPool *pool_;
  bool reusable_;
  bool presorted_ = false;
  bool used_ = false;
  std::size_t rows_;
  std::size_t features_;
  EntryCodec codec_;
  std::unique_ptr<Entry[]> sorted_; // presorted columns of all rows (reusable only)
  std::unique_ptr<Entry[]> work_;   // columns being partitioned by the builder
  std::unique_ptr<Entry[]> other_;  // parallel: their second copy (Columns::other)
  std::unique_ptr<std::uint64_t[]> leftBits_; // with other_: Columns::leftBits
  std::unique_ptr<Entry[]> scratch_;
  std::unique_ptr<std::uint8_t[]> goesLeft_;
  std::vector<double> countXlog_; // see countTables()
  std::vector<double> countInverse_;
};

} // namespace

std::unique_ptr<Grower> makeCpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool *pool, bool reusable) {
  return std::make_unique<CpuGrower>(train, options, pool, reusable);
}

} // namespace dt
