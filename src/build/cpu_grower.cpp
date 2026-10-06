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
  }

  Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows,
            GrowTimings &timings, std::vector<float> *sortedValues) override {
    const std::size_t count = rows.empty() ? rows_ : rows.size();
    {
      ScopedTimer timer(timings.prepareSeconds);
      prepare(rows);
      if (sortedValues) {
        sortedValues->resize(features_ * count);
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
    Subtree root{store.add(counts.data()), 0, static_cast<std::uint32_t>(count), 0,
                 std::vector<std::uint32_t>(features_)};
    std::iota(root.features.begin(), root.features.end(), 0u);
    CpuTreeBuilder builder(rules, codec_, features_, store, goesLeft_.get(), pool_,
                           options_.parallel);
    builder.grow(Columns{work_.get(), count, scratch_.get()}, std::move(root));
    if (pool_) {
      pool_->waitIdle();
    }
    store.toTree(tree);
    return tree;
  }

private:
  // Fill work_ with the sorted columns of the selected rows (stride = their
  // count). Without reuse, presort straight into work_.
  void prepare(std::span<const std::uint32_t> rows) {
    if (!reusable_) {
      if (used_ || !rows.empty()) {
        throw std::logic_error("CpuGrower: not created for repeated use");
      }
      used_ = true;
      presortColumns(train_, codec_, work_.get(), pool_);
      return;
    }
    if (!presorted_) {
      presortColumns(train_, codec_, sorted_.get(), pool_);
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
  std::unique_ptr<Entry[]> scratch_;
  std::unique_ptr<std::uint8_t[]> goesLeft_;
};

} // namespace

std::unique_ptr<Grower> makeCpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool *pool, bool reusable) {
  return std::make_unique<CpuGrower>(train, options, pool, reusable);
}

} // namespace dt
