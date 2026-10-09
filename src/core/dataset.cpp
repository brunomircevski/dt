#include "core/dataset.h"

#include "core/thread_pool.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace dt {

namespace {

// The CSV is read with pread() through a small buffer per thread, never mapped
// or read whole: the loader holds at most a few MiB of text at a time, so the
// process's peak memory is the dataset and the training structures, not a
// copy of the file as well (as for a tool that reads a binary file).
class InputFile {
public:
  explicit InputFile(const std::string &path) {
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
    ::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
  }
  ~InputFile() { ::close(fd_); }
  InputFile(const InputFile &) = delete;
  InputFile &operator=(const InputFile &) = delete;

  std::size_t size() const { return size_; }

  // Up to `length` bytes from `offset` into `out`; fewer only at the end of the file.
  std::size_t read(std::size_t offset, char *out, std::size_t length) const {
    std::size_t done = 0;
    while (done < length) {
      const ssize_t got =
          ::pread(fd_, out + done, length - done, static_cast<off_t>(offset + done));
      if (got < 0 && errno == EINTR) {
        continue;
      }
      if (got < 0) {
        throw std::runtime_error("Could not read dataset file");
      }
      if (got == 0) {
        break;
      }
      done += static_cast<std::size_t>(got);
    }
    return done;
  }

  // The offset just past the first '\n' at or after `offset` (the file size if none).
  std::size_t nextLineStart(std::size_t offset) const {
    char block[4096];
    while (offset < size_) {
      const std::size_t got = read(offset, block, std::min(sizeof block, size_ - offset));
      if (const void *newline = std::memchr(block, '\n', got)) {
        return offset + static_cast<std::size_t>(static_cast<const char *>(newline) - block) + 1;
      }
      offset += got;
    }
    return size_;
  }

private:
  int fd_ = -1;
  std::size_t size_ = 0;
};

// Calls visit(begin, end) on the bytes [first, last) of the file, in pieces
// that hold whole lines only (`last` must be a line start or the file size).
// A line longer than the buffer grows it. 64 KiB stays below glibc's mmap
// threshold: freeing a bigger buffer would raise that threshold, and the
// training's later allocations would then stay on the heap (+28 MiB peak RSS
// on SUSY).
template <class Visit>
void forEachLines(const InputFile &file, std::size_t first, std::size_t last, Visit &&visit) {
  std::vector<char> buffer(std::size_t{64} << 10);
  std::size_t kept = 0; // bytes of an unfinished line at the front of the buffer
  std::size_t offset = first;
  while (offset < last) {
    if (kept == buffer.size()) {
      buffer.resize(buffer.size() * 2);
    }
    const std::size_t got =
        file.read(offset, buffer.data() + kept, std::min(buffer.size() - kept, last - offset));
    if (got == 0) {
      throw std::runtime_error("Dataset file changed while it was read");
    }
    offset += got;
    const std::size_t filled = kept + got;
    std::size_t whole = filled; // at `last` every line is complete
    if (offset < last) {
      whole = 0;
      for (std::size_t index = filled; index > 0; --index) {
        if (buffer[index - 1] == '\n') {
          whole = index;
          break;
        }
      }
    }
    visit(buffer.data(), buffer.data() + whole);
    kept = filled - whole;
    std::memmove(buffer.data(), buffer.data() + whole, kept);
  }
}

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
  std::size_t begin = 0; // file offsets
  std::size_t end = 0;
  std::size_t firstRow = 0;
  std::size_t rowCount = 0;
  std::vector<std::string> labelNames; // chunk-local label id -> text
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

// Parse the whole lines in [begin, end); `row` is the dataset row of the first one.
void parseLines(const char *begin, const char *end, Chunk &chunk, std::size_t &row,
                std::size_t columnCount, bool skipFirstColumn, std::size_t rowCount,
                float *values, std::uint16_t *labels) {
  const char *cursor = begin;

  while (cursor < end) {
    const std::string_view line = lineAt(cursor, end);
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
          chunk.labelNames.emplace_back(label);
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

Dataset loadDataset(const std::string &filePath, ThreadPool *pool) {
  const InputFile file(filePath);

  // Header: feature names. The last column is the label; "Id" is skipped.
  const std::size_t bodyStart = file.nextLineStart(0);
  std::string headerLine(bodyStart, '\0');
  file.read(0, headerLine.data(), bodyStart);
  const char *headerCursor = headerLine.data();
  const std::vector<std::string_view> header =
      splitFields(lineAt(headerCursor, headerLine.data() + headerLine.size()));
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
  const unsigned threadCount = pool ? pool->workerCount() + 1 : 1;
  const std::size_t bodySize = file.size() - bodyStart;
  std::vector<Chunk> chunks;
  std::size_t chunkBegin = bodyStart;
  for (unsigned index = 0; index < threadCount && chunkBegin < file.size(); ++index) {
    std::size_t chunkEnd = file.size();
    if (index + 1 < threadCount) {
      chunkEnd = std::min(chunkEnd, bodyStart + bodySize * (index + 1) / threadCount);
    }
    if (chunkEnd < file.size()) {
      chunkEnd = file.nextLineStart(std::max(chunkEnd, chunkBegin));
    }
    if (chunkEnd > chunkBegin) {
      chunks.push_back({chunkBegin, chunkEnd, 0, 0, {}});
    }
    chunkBegin = chunkEnd;
  }

  // Each chunk reads its own bytes through its own descriptor (so the kernel
  // sees one sequential stream per thread and reads ahead for each).
  auto runOnChunks = [&](auto &&work) {
    parallelFor(pool, chunks.size(), [&](std::size_t index) {
      const InputFile chunkFile(filePath);
      work(chunkFile, chunks[index]);
    });
  };

  // Pass 1: count rows per chunk so each chunk knows where its rows go.
  runOnChunks([](const InputFile &chunkFile, Chunk &chunk) {
    forEachLines(chunkFile, chunk.begin, chunk.end, [&](const char *begin, const char *end) {
      chunk.rowCount += countRows(begin, end);
    });
  });
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
  runOnChunks([&](const InputFile &chunkFile, Chunk &chunk) {
    std::size_t row = chunk.firstRow;
    forEachLines(chunkFile, chunk.begin, chunk.end, [&](const char *begin, const char *end) {
      parseLines(begin, end, chunk, row, columnCount, skipFirstColumn, rowCount,
                 dataset.values.data(), dataset.labels.data());
    });
  });

  // Merge chunk-local label ids into global ids ordered by label text.
  std::vector<std::string> names;
  for (const Chunk &chunk : chunks) {
    names.insert(names.end(), chunk.labelNames.begin(), chunk.labelNames.end());
  }
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  dataset.classNames = names;

  std::unordered_map<std::string_view, std::uint16_t> globalId;
  for (std::size_t id = 0; id < dataset.classNames.size(); ++id) {
    globalId[dataset.classNames[id]] = static_cast<std::uint16_t>(id);
  }
  parallelFor(pool, chunks.size(), [&](std::size_t index) {
    const Chunk &chunk = chunks[index];
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
  Dataset::Values values(dataset.featureCount() * newRows);
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

namespace {

// Copy the given rows (in order) into a new dataset with the same schema.
Dataset selectRows(const Dataset &dataset, const std::vector<std::uint32_t> &rows,
                   ThreadPool *pool) {
  Dataset subset;
  subset.featureNames = dataset.featureNames;
  subset.classNames = dataset.classNames;
  subset.rowCount = rows.size();
  subset.values.resize(dataset.featureCount() * rows.size());
  subset.labels.resize(rows.size());
  parallelFor(pool, dataset.featureCount(), [&](std::size_t feature) {
    const float *source = dataset.column(feature);
    float *target = subset.values.data() + feature * rows.size();
    for (std::size_t index = 0; index < rows.size(); ++index) {
      target[index] = source[rows[index]];
    }
  });
  for (std::size_t index = 0; index < rows.size(); ++index) {
    subset.labels[index] = dataset.labels[rows[index]];
  }
  return subset;
}

} // namespace

std::mt19937_64 seededRandom(std::uint64_t seed, RandomStream stream) {
  std::seed_seq seeds{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
                      static_cast<std::uint32_t>(stream)};
  return std::mt19937_64(seeds);
}

void splitHoldout(const Dataset &dataset, double testFraction, std::uint64_t seed,
                  RandomStream stream, Dataset &train, Dataset &test, ThreadPool *pool) {
  std::vector<std::uint32_t> order(dataset.rowCount);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937_64 random = seededRandom(seed, stream);
  std::shuffle(order.begin(), order.end(), random);

  const std::size_t testRows =
      static_cast<std::size_t>(std::llround(testFraction * static_cast<double>(order.size())));
  // Both parts keep the original row order (nicer for debugging, and the
  // reads in selectRows stay sequential).
  std::vector<std::uint8_t> inTest(order.size(), 0);
  for (std::size_t index = 0; index < testRows; ++index) {
    inTest[order[index]] = 1;
  }
  std::vector<std::uint32_t> testIndex;
  std::vector<std::uint32_t> trainIndex;
  testIndex.reserve(testRows);
  trainIndex.reserve(order.size() - testRows);
  for (std::uint32_t row = 0; row < order.size(); ++row) {
    (inTest[row] ? testIndex : trainIndex).push_back(row);
  }
  train = selectRows(dataset, trainIndex, pool);
  test = selectRows(dataset, testIndex, pool);
}

std::unique_ptr<float[]> rowMajorFeatures(const Dataset &dataset, ThreadPool *pool) {
  const std::size_t rows = dataset.rowCount;
  const std::size_t features = dataset.featureCount();
  // Not zero-filled first: the threads write (and so first touch) it.
  std::unique_ptr<float[]> matrix(new float[rows * features]);
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
  parallelFor(pool, blocks, transpose);
  return matrix;
}

} // namespace dt
