#include "build/cpu_builder.h"
#include "build/grower.h"
#include "core/timing.h"
#include "core/trace.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace dt {

namespace {

class CpuGrower final : public Grower {
public:
  CpuGrower(const Dataset &train, const Options &options, ThreadPool *pool, bool reusable)
      : train_(train), options_(options), pool_(pool), reusable_(reusable),
        rows_(train.rowCount), features_(train.featureCount()),
        codec_(makeEntryCodec(train.classCount(), train.rowCount)),
        slots_(features_ + spareSlots(options, pool, features_)), scratch_(new Entry[rows_]),
        goesLeft_(new std::uint8_t[rows_]) {
    // A slot per column and a few more (see Columns): serial and parallel
    // builds run the same algorithm, big nodes are partitioned out of place
    // in both, only on one thread or on many.
    work_.reset(new Entry[slots_ * rows_]);
    if (reusable_) {
      sorted_.reset(new Entry[features_ * rows_]);
    }
    leftBits_.reset(new std::uint64_t[rows_ / 64 + 1]);
  }

  Tree grow(const SplitRules &rules, std::span<const std::uint32_t> rows,
            GrowTimings &timings, DistinctValues *distinct, RowLeaves *leafOfRow) override {
    if (leafOfRow && !rows.empty()) {
      throw std::logic_error("CpuGrower: leaves of rows only with all rows");
    }
    const std::size_t count = rows.empty() ? rows_ : rows.size();
    {
      ScopedTimer timer(timings.prepareSeconds);
      trace::Phase phase("presort");
      trace::Doing doing(trace::Work::Presort);
      prepare(rows);
      if (distinct) {
        distinctValues(count, *distinct);
      }
    }
    Tree tree;
    tree.featureNames = train_.featureNames;
    tree.classNames = train_.classNames;
    ScopedTimer timer(timings.buildSeconds);
    std::optional<trace::Phase> phase(std::in_place, "build");
    const std::vector<std::uint32_t> counts = countClasses(train_, rows, pool_);
    NodeStore store(train_.classCount(), 2 * count);
    Subtree root;
    root.slots = std::make_shared<const SlotMap>(slotOf_);
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
    builder.grow(Columns{work_.get(), count, scratch_.get(), slots_, leftBits_.get()},
                 std::move(root));
    if (pool_) {
      pool_->waitIdle();
    }
    phase.emplace("assemble tree");
    trace::Doing doing(trace::Work::Assemble);
    store.toTree(tree, pool_, leafOfRow);
    return tree;
  }

  std::span<std::byte> workspace() override {
    return {reinterpret_cast<std::byte *>(work_.get()), slots_ * rows_ * sizeof(Entry)};
  }

private:
  // See ParallelOptions::spareSlots.
  static std::size_t spareSlots(const Options &options, ThreadPool *pool, std::size_t features) {
    if (!pool) {
      return 1;
    }
    if (options.parallel.spareSlots > 0) {
      return options.parallel.spareSlots;
    }
    const std::size_t threads = pool->workerCount() + 1;
    return std::max<std::size_t>(1, std::min(features > 0 ? features - 1 : 0, threads));
  }

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
  // `count`, slots slotOf_): the last value of each run, in chunks on the pool.
  void distinctValues(std::size_t count, DistinctValues &distinct) {
    constexpr std::size_t kChunk = std::size_t{1} << 20;
    const std::size_t chunks = (count + kChunk - 1) / kChunk;
    std::vector<std::vector<float>> parts(features_ * chunks);
    parallelFor(pool_, parts.size(), [&](std::size_t part) {
      const Entry *column = work_.get() + slotOf_[part / chunks] * count;
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
  // count; slotOf_ says where). Without reuse, presort straight into work_.
  void prepare(std::span<const std::uint32_t> rows) {
    if (!reusable_) {
      if (used_ || !rows.empty()) {
        throw std::logic_error("CpuGrower: not created for repeated use");
      }
      used_ = true;
      slotOf_ = presortColumns(train_, codec_, work_.get(), slots_, pool_);
      return;
    }
    if (!presorted_) { // in work_'s slots, then in feature order into sorted_
      const SlotMap slots = presortColumns(train_, codec_, work_.get(), slots_, pool_);
      parallelFor(pool_, features_, [&](std::size_t feature) {
        std::memcpy(sorted_.get() + feature * rows_, work_.get() + slots[feature] * rows_,
                    rows_ * sizeof(Entry));
      });
      presorted_ = true;
    }
    slotOf_.resize(features_);
    std::iota(slotOf_.begin(), slotOf_.end(), 0u);
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
  std::size_t slots_; // features_ + spareSlots()
  std::unique_ptr<Entry[]> sorted_; // presorted columns of all rows (reusable only)
  std::unique_ptr<Entry[]> work_;   // slots_ slots: the columns being partitioned
  SlotMap slotOf_;                  // where in work_ the grow starts
  std::unique_ptr<std::uint64_t[]> leftBits_; // Columns::leftBits
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
