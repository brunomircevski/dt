#pragma once

#include <chrono>

struct FitTimings {
  double loadSeconds = 0.0;
  double gpuAllocSeconds = 0.0;
  double gpuUploadSeconds = 0.0;
  double buildSeconds = 0.0;
  double pruneSeconds = 0.0;

  void resetFitPhases() {
    gpuAllocSeconds = 0.0;
    gpuUploadSeconds = 0.0;
    buildSeconds = 0.0;
    pruneSeconds = 0.0;
  }

  double endToEndSeconds() const {
    return loadSeconds + gpuAllocSeconds + gpuUploadSeconds + buildSeconds +
           pruneSeconds;
  }
};

// Records elapsed wall time into outSeconds when destroyed.
class ScopedTimer {
public:
  explicit ScopedTimer(double &outSeconds)
      : outSeconds_(outSeconds),
        start_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    outSeconds_ = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_)
                        .count();
  }

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
  double &outSeconds_;
  std::chrono::steady_clock::time_point start_;
};
