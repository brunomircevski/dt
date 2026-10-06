#pragma once

#include <chrono>

// Adds the elapsed wall time to `outSeconds` when destroyed.
class ScopedTimer {
public:
  explicit ScopedTimer(double &outSeconds)
      : outSeconds_(outSeconds), start_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    outSeconds_ +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
  double &outSeconds_;
  std::chrono::steady_clock::time_point start_;
};
