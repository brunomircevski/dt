#include "dataset.h"

#include "thread_pool.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace {

class MappedFile {
public:
  explicit MappedFile(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("Could not open dataset file: " + path);
    }
    struct stat info {};
    if (::fstat(fd_, &info) != 0 || info.st_size == 0) {
      ::close(fd_);
      throw std::runtime_error("Dataset file is empty: " + path);
    }
    size_ = static_cast<std::size_t>(info.st_size);
    mapped_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
      ::close(fd_);
      throw std::runtime_error("Could not mmap dataset file: " + path);
    }
    ::madvise(mapped_, size_, MADV_SEQUENTIAL);
  }
  ~MappedFile() {
    ::munmap(mapped_, size_);
    ::close(fd_);
  }
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  const char *begin() const { return static_cast<const char *>(mapped_); }
  const char *end() const { return begin() + size_; }

private:
  int fd_ = -1;
  void *mapped_ = MAP_FAILED;
  std::size_t size_ = 0;
};

// One line without its '\n' and optional trailing '\r'.
std::string_view lineAt(const char *&cursor, const char *end) {
  const char *lineBegin = cursor;
  const char *newline = static_cast<const char *>(
      std::memchr(cursor, '\n', static_cast<std::size_t>(end - cursor)));
  const char *lineEnd = newline ? newline : end;
  cursor = newline ? newline + 1 : end;
  if (lineEnd > lineBegin && lineEnd[-1] == '\r') {
    --lineEnd;
  }
  return {lineBegin, static_cast<std::size_t>(lineEnd - lineBegin)};
}

std::vector<std::string_view> splitFields(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = line.find(',', start);
    if (comma == std::string_view::npos) {
      fields.push_back(line.substr(start));
      return fields;
    }
    fields.push_back(line.substr(start, comma - start));
    start = comma + 1;
  }
}

bool isIdColumn(std::string_view name) {
  return name.size() == 2 && (name[0] == 'I' || name[0] == 'i') &&
         (name[1] == 'D' || name[1] == 'd');
}

// A contiguous byte range of the file that starts at a line start.
struct Chunk {
  const char *begin = nullptr;
  const char *end = nullptr;
  std::size_t firstRow = 0;
  std::size_t rowCount = 0;
  std::vector<std::string_view> labelNames; // chunk-local label id -> text
};

std::size_t countRows(const char *begin, const char *end) {
  std::size_t rows = 0;
  const char *cursor = begin;
  while (cursor < end) {
    const std::string_view line = lineAt(cursor, end);
    if (!line.empty()) {
      ++rows;
    }
  }
  return rows;
}

void parseChunk(Chunk &chunk, std::size_t columnCount, bool skipFirstColumn,
                std::size_t rowCount, float *values, std::uint16_t *labels) {
  std::size_t row = chunk.firstRow;
  const char *cursor = chunk.begin;

  while (cursor < chunk.end) {
    const std::string_view line = lineAt(cursor, chunk.end);
    if (line.empty()) {
      continue;
    }

    const char *field = line.data();
    const char *lineEnd = line.data() + line.size();
    std::size_t column = 0;
    std::size_t feature = 0;
    while (true) {
      const char *fieldEnd = static_cast<const char *>(
          std::memchr(field, ',', static_cast<std::size_t>(lineEnd - field)));
      if (!fieldEnd) {
        fieldEnd = lineEnd;
      }
      if (column + 1 == columnCount) {
        // Last column: the class label.
        if (fieldEnd != lineEnd) {
          throw std::runtime_error("Malformed CSV row " + std::to_string(row + 2) +
                                   ": too many columns");
        }
        const std::string_view label(field, static_cast<std::size_t>(fieldEnd - field));
        auto found = std::find(chunk.labelNames.begin(), chunk.labelNames.end(), label);
        if (found == chunk.labelNames.end()) {
          if (chunk.labelNames.size() >= 65535) {
            throw std::runtime_error("Too many distinct class labels");
          }
          chunk.labelNames.push_back(label);
          found = chunk.labelNames.end() - 1;
        }
        labels[row] = static_cast<std::uint16_t>(found - chunk.labelNames.begin());
        break;
      }
      if (column > 0 || !skipFirstColumn) {
        float value = 0.0f;
        const auto result = std::from_chars(field, fieldEnd, value);
        if (result.ec != std::errc{} || result.ptr != fieldEnd || std::isnan(value)) {
          throw std::runtime_error("Could not parse numeric value '" +
                                   std::string(field, fieldEnd) + "' on CSV row " +
                                   std::to_string(row + 2));
        }
        values[feature * rowCount + row] = value;
        ++feature;
      }
      if (fieldEnd == lineEnd) {
        throw std::runtime_error("Malformed CSV row " + std::to_string(row + 2) +
                                 ": too few columns");
      }
      field = fieldEnd + 1;
      ++column;
    }
    ++row;
  }
}

} // namespace

Dataset loadDataset(const std::string &filePath, unsigned threadCount) {
  const MappedFile file(filePath);
  const char *cursor = file.begin();

  // Header: feature names. The last column is the label; "Id" is skipped.
  const std::vector<std::string_view> header = splitFields(lineAt(cursor, file.end()));
  const bool skipFirstColumn = isIdColumn(header.front());
  const std::size_t columnCount = header.size();
  const std::size_t featureColumns = columnCount - 1 - (skipFirstColumn ? 1 : 0);
  if (columnCount < 2 || featureColumns == 0) {
    throw std::runtime_error("CSV needs at least one feature column and a label column");
  }

  Dataset dataset;
  for (std::size_t column = skipFirstColumn ? 1 : 0; column + 1 < columnCount; ++column) {
    dataset.featureNames.emplace_back(header[column]);
  }

  // Cut the body into one chunk per thread, each starting at a line start.
  threadCount = std::max(1u, threadCount);
  const std::size_t bodySize = static_cast<std::size_t>(file.end() - cursor);
  std::vector<Chunk> chunks;
  const char *chunkBegin = cursor;
  for (unsigned index = 0; index < threadCount && chunkBegin < file.end(); ++index) {
    const char *chunkEnd =
        index + 1 == threadCount ? file.end()
                                 : std::min(file.end(), cursor + bodySize * (index + 1) / threadCount);
    if (chunkEnd < file.end()) {
      const void *newline =
          std::memchr(chunkEnd, '\n', static_cast<std::size_t>(file.end() - chunkEnd));
      chunkEnd = newline ? static_cast<const char *>(newline) + 1 : file.end();
    }
    if (chunkEnd > chunkBegin) {
      chunks.push_back({chunkBegin, chunkEnd, 0, 0, {}});
    }
    chunkBegin = chunkEnd;
  }

  auto runOnChunks = [&](auto &&work) {
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(chunks.size());
    for (std::size_t index = 0; index < chunks.size(); ++index) {
      workers.emplace_back([&, index]() {
        try {
          work(chunks[index]);
        } catch (...) {
          errors[index] = std::current_exception();
        }
      });
    }
    for (std::thread &worker : workers) {
      worker.join();
    }
    for (const std::exception_ptr &error : errors) {
      if (error) {
        std::rethrow_exception(error);
      }
    }
  };

  // Pass 1: count rows per chunk so each chunk knows where its rows go.
  runOnChunks([](Chunk &chunk) { chunk.rowCount = countRows(chunk.begin, chunk.end); });
  std::size_t rowCount = 0;
  for (Chunk &chunk : chunks) {
    chunk.firstRow = rowCount;
    rowCount += chunk.rowCount;
  }
  if (rowCount == 0) {
    throw std::runtime_error("Dataset has no rows: " + filePath);
  }
  if (rowCount > 0xFFFFFFFFull) {
    throw std::runtime_error("Datasets are limited to 2^32 - 1 rows");
  }

  // Pass 2: parse straight into the column-major arrays.
  dataset.rowCount = rowCount;
  dataset.values.resize(featureColumns * rowCount);
  dataset.labels.resize(rowCount);
  runOnChunks([&](Chunk &chunk) {
    parseChunk(chunk, columnCount, skipFirstColumn, rowCount, dataset.values.data(),
               dataset.labels.data());
  });

  // Merge chunk-local label ids into global ids ordered by label text.
  std::vector<std::string> names;
  for (const Chunk &chunk : chunks) {
    for (std::string_view name : chunk.labelNames) {
      names.emplace_back(name);
    }
  }
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  dataset.classNames = names;

  std::unordered_map<std::string_view, std::uint16_t> globalId;
  for (std::size_t id = 0; id < dataset.classNames.size(); ++id) {
    globalId[dataset.classNames[id]] = static_cast<std::uint16_t>(id);
  }
  runOnChunks([&](Chunk &chunk) {
    std::vector<std::uint16_t> remap;
    for (std::string_view name : chunk.labelNames) {
      remap.push_back(globalId.at(name));
    }
    for (std::size_t row = chunk.firstRow; row < chunk.firstRow + chunk.rowCount; ++row) {
      dataset.labels[row] = remap[dataset.labels[row]];
    }
  });

  return dataset;
}

void multiplyDataset(Dataset &dataset, std::size_t multiplier) {
  if (multiplier <= 1) {
    return;
  }
  const std::size_t oldRows = dataset.rowCount;
  const std::size_t newRows = oldRows * multiplier;
  if (newRows > 0xFFFFFFFFull) {
    throw std::runtime_error("Datasets are limited to 2^32 - 1 rows");
  }
  std::vector<float> values(dataset.featureCount() * newRows);
  for (std::size_t feature = 0; feature < dataset.featureCount(); ++feature) {
    const float *source = dataset.column(feature);
    float *target = values.data() + feature * newRows;
    for (std::size_t copy = 0; copy < multiplier; ++copy) {
      const float scale = 1.0f + static_cast<float>(copy) * 0.0001f;
      for (std::size_t row = 0; row < oldRows; ++row) {
        target[copy * oldRows + row] = source[row] * scale;
      }
    }
  }
  std::vector<std::uint16_t> labels(newRows);
  for (std::size_t copy = 0; copy < multiplier; ++copy) {
    std::copy(dataset.labels.begin(), dataset.labels.end(),
              labels.begin() + static_cast<std::ptrdiff_t>(copy * oldRows));
  }
  dataset.values = std::move(values);
  dataset.labels = std::move(labels);
  dataset.rowCount = newRows;
}

Dataset selectRows(const Dataset &dataset, const std::vector<std::uint32_t> &rows) {
  Dataset subset;
  subset.featureNames = dataset.featureNames;
  subset.classNames = dataset.classNames;
  subset.rowCount = rows.size();
  subset.values.resize(dataset.featureCount() * rows.size());
  subset.labels.resize(rows.size());
  for (std::size_t feature = 0; feature < dataset.featureCount(); ++feature) {
    const float *source = dataset.column(feature);
    float *target = subset.values.data() + feature * rows.size();
    for (std::size_t index = 0; index < rows.size(); ++index) {
      target[index] = source[rows[index]];
    }
  }
  for (std::size_t index = 0; index < rows.size(); ++index) {
    subset.labels[index] = dataset.labels[rows[index]];
  }
  return subset;
}

void splitHoldout(const Dataset &dataset, double testFraction, Dataset &train,
                  Dataset &test) {
  std::vector<std::uint32_t> order(dataset.rowCount);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937_64 random(12345);
  std::shuffle(order.begin(), order.end(), random);

  const std::size_t testRows =
      static_cast<std::size_t>(std::llround(testFraction * static_cast<double>(order.size())));
  std::vector<std::uint32_t> testIndex(order.begin(), order.begin() + testRows);
  std::vector<std::uint32_t> trainIndex(order.begin() + testRows, order.end());
  // Keep the original row order inside each part (nicer for debugging).
  std::sort(testIndex.begin(), testIndex.end());
  std::sort(trainIndex.begin(), trainIndex.end());
  train = selectRows(dataset, trainIndex);
  test = selectRows(dataset, testIndex);
}

std::vector<float> rowMajorFeatures(const Dataset &dataset, ThreadPool *pool) {
  const std::size_t rows = dataset.rowCount;
  const std::size_t features = dataset.featureCount();
  std::vector<float> matrix(rows * features);
  constexpr std::size_t kBlock = 4096;
  const std::size_t blocks = (rows + kBlock - 1) / kBlock;
  auto transpose = [&](std::size_t block) {
    const std::size_t begin = block * kBlock;
    const std::size_t end = std::min(rows, begin + kBlock);
    for (std::size_t feature = 0; feature < features; ++feature) {
      const float *column = dataset.column(feature);
      for (std::size_t row = begin; row < end; ++row) {
        matrix[row * features + feature] = column[row];
      }
    }
  };
  if (pool) {
    pool->parallelFor(blocks, transpose);
  } else {
    for (std::size_t block = 0; block < blocks; ++block) {
      transpose(block);
    }
  }
  return matrix;
}
