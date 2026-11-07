#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
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

// Every feature's distinct values (bit patterns), in increasing order: those
// of feature f are values[offsets[f]] .. values[offsets[f + 1] - 1]. C4.5's
// thresholds need them (a by-product of the presort).
struct DistinctValues {
  Dataset::Values values; // uninitialised: the threads that fill it touch its pages
  std::vector<std::size_t> offsets;
};

// The distinct values of every feature over all rows (the backends that
// presort get them from the presort instead).
DistinctValues distinctValues(const Dataset &dataset);

// Load a CSV with a header line. The last column is the class label; a first
// column named "Id" (any case) is ignored; every other column must be numeric.
// Parsing runs on the pool (or on the calling thread if `pool` is null).
Dataset loadDataset(const std::string &filePath, ThreadPool *pool);


// Rows of each class, counted on `pool`.
std::vector<std::uint32_t> countClasses(const Dataset &dataset, ThreadPool *pool);

// Shuffle the rows (same seed = same split on every machine) and split off
// `testFraction` of them as the test set.
void splitHoldout(const Dataset &dataset, double testFraction, std::uint64_t seed,
                  Dataset &train, Dataset &test, ThreadPool *pool = nullptr);

// Make `test` (loaded from another file) comparable with `train`: it must
// have the same features, in the same order, and its labels are renumbered
// to train's class ids. A class that only the test set has gets an id after
// train's, which no tree predicts. Returns the test rows of such classes.
std::size_t matchTrainingSchema(const Dataset &train, Dataset &test);

// The last `testRows` rows are the test set, the others the training set (both
// in file order): a published split such as HIGGS's and SUSY's.
void splitTail(const Dataset &dataset, std::size_t testRows, Dataset &train, Dataset &test,
               ThreadPool *pool = nullptr);

} // namespace dt
