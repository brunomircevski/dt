#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dt {

class ThreadPool;

// Numeric features + one class label per row, stored column-major.
//
// Column-major ("all values of feature 0, then all of feature 1, ...") is the
// layout every split search wants: it reads one feature at a time.
// Class labels are mapped to small integer ids once, at load time, so no hot
// loop ever compares strings.
struct Dataset {
  std::vector<std::string> featureNames;
  std::vector<std::string> classNames; // class id -> label text (sorted)

  std::size_t rowCount = 0;
  std::vector<float> values;           // values[feature * rowCount + row]
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
std::vector<float> rowMajorFeatures(const Dataset &dataset, ThreadPool *pool);

// Deterministically shuffle the rows and split off `testFraction` of them.
void splitHoldout(const Dataset &dataset, double testFraction, Dataset &train,
                  Dataset &test);

} // namespace dt
