#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/nvtx.h"

namespace dt {

// Opt-in timers for the parts of a phase (DT_PROFILE=1 in the environment):
// each accumulates the wall time of its scopes, summed over the threads that
// run them, and print() writes them to stderr. Off, a scope costs one branch.
class ProfileCounter {
public:
  explicit ProfileCounter(const char *name) : name_(name) {}

  const char *name() const { return name_; }

  static bool enabled() {
    static const bool on = [] {
      const char *value = std::getenv("DT_PROFILE");
      return value && *value && *value != '0';
    }();
    return on;
  }

  void add(double seconds) {
    nanoseconds_.fetch_add(static_cast<long long>(seconds * 1e9), std::memory_order_relaxed);
    calls_.fetch_add(1, std::memory_order_relaxed);
  }

  void print() const {
    if (enabled()) {
      std::fprintf(stderr, "  profile %-28s %10.2f ms %10lld calls\n", name_,
                   static_cast<double>(nanoseconds_.load()) * 1e-6, calls_.load());
    }
  }

  void reset() {
    nanoseconds_ = 0;
    calls_ = 0;
  }

private:
  const char *name_;
  std::atomic<long long> nanoseconds_{0};
  std::atomic<long long> calls_{0};
};

class ProfileScope {
public:
  explicit ProfileScope(ProfileCounter &counter)
      : range_(counter.name()), counter_(ProfileCounter::enabled() ? &counter : nullptr) {
    if (counter_) {
      start_ = std::chrono::steady_clock::now();
    }
  }
  ~ProfileScope() {
    if (counter_) {
      counter_->add(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count());
    }
  }
  ProfileScope(const ProfileScope &) = delete;
  ProfileScope &operator=(const ProfileScope &) = delete;

private:
  NvtxRange range_;
  ProfileCounter *counter_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace dt
