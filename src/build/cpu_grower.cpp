#include "build/cpu_builder.h"
#include "build/grower.h"
#include "core/timing.h"

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
    }
  }

  Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows,
            GrowTimings &timings, std::vector<float> *sortedValues) override {
    const std::size_t count = rows.empty() ? rows_ : rows.size();
    {
      ScopedTimer timer(timings.prepareSeconds);
      prepare(rows, sortedValues);
      if (sortedValues && reusable_) { // the presort wrote them otherwise
        parallelFor(pool_, features_, [&](std::size_t feature) {
          const Entry *column = work_.get() + feature * count;
          float *values = sortedValues->data() + feature * count;
          for (std::size_t index = 0; index < count; ++index) {
            values[index] = column[index].value;
          }
        });
      }
    }
    Tree tree;
    tree.featureNames = train_.featureNames;
    tree.classNames = train_.classNames;
    ScopedTimer timer(timings.buildSeconds);
    std::vector<std::uint32_t> counts(train_.classCount(), 0);
    for (std::size_t index = 0; index < count; ++index) {
      ++counts[codec_.cls(work_[index].packed)];
    }
    NodeStore store(train_.classCount(), 2 * count);
    Subtree root;
    root.node = store.add(counts.data());
    root.count = static_cast<std::uint32_t>(count);
    root.features.resize(features_);
    std::iota(root.features.begin(), root.features.end(), 0u);
    CpuTreeBuilder builder(rules, codec_, features_, store, goesLeft_.get(), pool_,
                           options_.parallel, countTables());
    builder.grow(Columns{work_.get(), count, scratch_.get(), other_.get()}, std::move(root));
    if (pool_) {
      pool_->waitIdle();
    }
    store.toTree(tree);
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

  // Fill work_ with the sorted columns of the selected rows (stride = their
  // count), and `sortedValues` (if not null) with their values. Without
  // reuse, presort straight into work_.
  void prepare(std::span<const std::uint32_t> rows, std::vector<float> *sortedValues) {
    const std::size_t count = rows.empty() ? rows_ : rows.size();
    if (sortedValues) {
      sortedValues->resize(features_ * count);
    }
    if (!reusable_) {
      if (used_ || !rows.empty()) {
        throw std::logic_error("CpuGrower: not created for repeated use");
      }
      used_ = true;
      presortColumns(train_, codec_, work_.get(), pool_,
                     sortedValues ? sortedValues->data() : nullptr, other_.get());
      return;
    }
    if (!presorted_) {
      presortColumns(train_, codec_, sorted_.get(), pool_, nullptr, other_.get());
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
