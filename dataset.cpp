#include "dataset.h"

#include <charconv>
#include <cstddef>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

class MmappedFile {
public:
  explicit MmappedFile(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("Could not open dataset file: " + path);
    }

    struct stat stat {};
    if (::fstat(fd_, &stat) != 0 || stat.st_size == 0) {
      ::close(fd_);
      throw std::runtime_error("Dataset file is empty: " + path);
    }

    size_ = static_cast<std::size_t>(stat.st_size);
    mapped_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
      ::close(fd_);
      throw std::runtime_error("Could not mmap dataset file: " + path);
    }

    data_ = static_cast<const char *>(mapped_);
  }

  ~MmappedFile() {
    if (mapped_ != MAP_FAILED) {
      ::munmap(mapped_, size_);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  MmappedFile(const MmappedFile &) = delete;
  MmappedFile &operator=(const MmappedFile &) = delete;

  const char *data() const { return data_; }
  std::size_t size() const { return size_; }

private:
  int fd_ = -1;
  void *mapped_ = MAP_FAILED;
  const char *data_ = nullptr;
  std::size_t size_ = 0;
};

class LineCursor {
public:
  LineCursor(const char *data, std::size_t size)
      : begin_(data), end_(data + size), cursor_(data) {}

  bool next() {
    if (cursor_ >= end_) {
      return false;
    }

    lineBegin_ = cursor_;
    while (cursor_ < end_ && *cursor_ != '\n') {
      ++cursor_;
    }

    lineEnd_ = cursor_;
    if (lineEnd_ > lineBegin_ && lineEnd_[-1] == '\r') {
      --lineEnd_;
    }

    if (cursor_ < end_) {
      ++cursor_;
    }

    return true;
  }

  const char *lineBegin() const { return lineBegin_; }
  const char *lineEnd() const { return lineEnd_; }
  bool isEmptyLine() const { return lineBegin_ == lineEnd_; }

private:
  const char *begin_;
  const char *end_;
  const char *cursor_;
  const char *lineBegin_ = nullptr;
  const char *lineEnd_ = nullptr;
};

std::size_t countFields(const char *lineBegin, const char *lineEnd) {
  if (lineBegin == lineEnd) {
    return 0;
  }

  std::size_t fieldCount = 1;
  for (const char *cursor = lineBegin; cursor < lineEnd; ++cursor) {
    if (*cursor == ',') {
      ++fieldCount;
    }
  }
  return fieldCount;
}

float parseFloatField(const char *begin, const char *end) {
  float value = 0.0f;
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end) {
    throw std::runtime_error("Could not parse float value in dataset");
  }
  return value;
}

std::string fieldToString(const char *begin, const char *end) {
  return std::string(begin, static_cast<std::size_t>(end - begin));
}

void parseHeaderLine(const char *lineBegin, const char *lineEnd,
                     std::vector<std::string> &featureNames) {
  const std::size_t columnCount = countFields(lineBegin, lineEnd);
  if (columnCount < 3) {
    throw std::runtime_error("Unexpected CSV header format");
  }

  featureNames.clear();
  featureNames.reserve(columnCount - 2);

  const char *fieldBegin = lineBegin;
  std::size_t columnIndex = 0;
  for (const char *cursor = lineBegin; cursor <= lineEnd; ++cursor) {
    if (cursor == lineEnd || *cursor == ',') {
      if (columnIndex > 0 && columnIndex + 1 < columnCount) {
        featureNames.push_back(fieldToString(fieldBegin, cursor));
      }
      fieldBegin = cursor + 1;
      ++columnIndex;
    }
  }
}

std::size_t countDataRows(LineCursor &cursor, std::size_t expectedColumnCount) {
  std::size_t rowCount = 0;
  while (cursor.next()) {
    if (cursor.isEmptyLine()) {
      continue;
    }
    if (countFields(cursor.lineBegin(), cursor.lineEnd()) !=
        expectedColumnCount) {
      throw std::runtime_error("Malformed CSV row in dataset");
    }
    ++rowCount;
  }
  return rowCount;
}

void parseDataRows(LineCursor &cursor, Dataset &dataset,
                   std::size_t expectedColumnCount,
                   std::size_t featureCount) {
  while (cursor.next()) {
    if (cursor.isEmptyLine()) {
      continue;
    }

    const char *lineBegin = cursor.lineBegin();
    const char *lineEnd = cursor.lineEnd();
    if (countFields(lineBegin, lineEnd) != expectedColumnCount) {
      throw std::runtime_error("Malformed CSV row in dataset");
    }

    Sample sample;
    sample.features.resize(featureCount);

    const char *fieldBegin = lineBegin;
    std::size_t columnIndex = 0;
    std::size_t featureIndex = 0;
    for (const char *fieldCursor = lineBegin; fieldCursor <= lineEnd;
         ++fieldCursor) {
      if (fieldCursor == lineEnd || *fieldCursor == ',') {
        if (columnIndex > 0 && columnIndex + 1 < expectedColumnCount) {
          sample.features[featureIndex++] =
              parseFloatField(fieldBegin, fieldCursor);
        } else if (columnIndex + 1 == expectedColumnCount) {
          sample.label = fieldToString(fieldBegin, fieldCursor);
        }
        fieldBegin = fieldCursor + 1;
        ++columnIndex;
      }
    }

    dataset.samples.push_back(std::move(sample));
  }
}

void multiplyDatasetInMemory(Dataset &dataset, std::size_t multiplier) {
  if (multiplier <= 1) {
    return;
  }

  const std::size_t originalCount = dataset.samples.size();
  dataset.samples.reserve(originalCount * multiplier);

  for (std::size_t copyIndex = 1; copyIndex < multiplier; ++copyIndex) {
    const float featureScale =
        1.0f + static_cast<float>(copyIndex) * 0.0001f;
    for (std::size_t sampleIndex = 0; sampleIndex < originalCount;
         ++sampleIndex) {
      Sample duplicated = dataset.samples[sampleIndex];
      for (float &feature : duplicated.features) {
        feature *= featureScale;
      }
      dataset.samples.push_back(std::move(duplicated));
    }
  }
}

} // namespace

Dataset loadDataset(const std::string &filePath,
                    const std::size_t demoDatasetMultiplier) {
  const MmappedFile file(filePath);

  LineCursor headerCursor(file.data(), file.size());
  if (!headerCursor.next() || headerCursor.isEmptyLine()) {
    throw std::runtime_error("Dataset file is empty: " + filePath);
  }

  Dataset dataset;
  parseHeaderLine(headerCursor.lineBegin(), headerCursor.lineEnd(),
                  dataset.featureNames);

  const std::size_t expectedColumnCount =
      dataset.featureNames.size() + 2;
  const std::size_t featureCount = dataset.featureNames.size();

  LineCursor countCursor(file.data(), file.size());
  countCursor.next(); // skip header
  const std::size_t rowCount = countDataRows(countCursor, expectedColumnCount);
  dataset.samples.reserve(rowCount);

  LineCursor dataCursor(file.data(), file.size());
  dataCursor.next(); // skip header
  parseDataRows(dataCursor, dataset, expectedColumnCount, featureCount);
  multiplyDatasetInMemory(dataset, demoDatasetMultiplier);

  return dataset;
}
