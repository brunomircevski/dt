#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace dt {

class ThreadPool;

// Numeric features + one class label per row, stored column-major.
//
// Column-major ("all values of feature 0, then all of feature 1, ...") is the
// layout every split search wants: it reads one feature at a time.
// Class labels are mapped to small integer ids once, at load time, so no hot
// loop ever compares strings.
// An allocator whose resize() leaves new elements uninitialised: the feature
// values are always written right after (by the parser or by threads copying
// rows), so zero-filling hundreds of MB first on one thread is wasted time.
template <class T> struct UninitializedAllocator : std::allocator<T> {
  template <class U> struct rebind {
    using other = UninitializedAllocator<U>;
  };
  UninitializedAllocator() = default;
  template <class U> UninitializedAllocator(const UninitializedAllocator<U> &) {}
  template <class U> void construct(U *place) noexcept { ::new (static_cast<void *>(place)) U; }
  template <class U, class... Args> void construct(U *place, Args &&...args) {
    ::new (static_cast<void *>(place)) U(std::forward<Args>(args)...);
  }
};

struct Dataset {
  using Values = std::vector<float, UninitializedAllocator<float>>;

  std::vector<std::string> featureNames;
  std::vector<std::string> classNames; // class id -> label text (sorted)

  std::size_t rowCount = 0;
  Values values;                       // values[feature * rowCount + row]
  std::vector<std::uint16_t> labels;   // class id per row

  std::size_t featureCount() const { return featureNames.size(); }
  std::size_t classCount() const { return classNames.size(); }

  const float *column(std::size_t feature) const {
    return values.data() + feature * rowCount;
  }
  float value(std::size_t row, std::size_t feature) const {
    return values[feature * rowCount + row];
  }
};

// Load a CSV with a header line. The last column is the class label; a first
// column named "Id" (any case) is ignored; every other column must be numeric.
// Parsing runs on the pool (or on the calling thread if `pool` is null).
Dataset loadDataset(const std::string &filePath, ThreadPool *pool);

// Stress tests: append (multiplier - 1) slightly rescaled copies of every row.
void multiplyDataset(Dataset &dataset, std::size_t multiplier);

// The features as a row-major matrix: result[row * featureCount + feature].
// Walking a tree for one row then touches one or two cache lines only.
std::unique_ptr<float[]> rowMajorFeatures(const Dataset &dataset, ThreadPool *pool);
// The same into `out` (rowCount * featureCount floats).
void rowMajorFeatures(const Dataset &dataset, ThreadPool *pool, float *out);

// Independent random streams derived from one --seed, so that e.g. changing
// the CV folds does not change the holdout split.
enum class RandomStream : std::uint32_t { Holdout = 1, CrossValidation = 2 };
std::mt19937_64 seededRandom(std::uint64_t seed, RandomStream stream);

// Shuffle the rows (same seed and stream = same split) and split off
// `testFraction` of them as the test set.
void splitHoldout(const Dataset &dataset, double testFraction, std::uint64_t seed,
                  RandomStream stream, Dataset &train, Dataset &test,
                  ThreadPool *pool = nullptr);

} // namespace dt
