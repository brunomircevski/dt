#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// The timeline of one training: when each of its sections started and ended,
// in seconds since begin() (trainTree's start), printed as "span" lines
// (bench/tools/gpu_steps_chart.py draws them). Always on: a section costs two clock
// reads and a lock, and a training has about a dozen.
namespace dt::timeline {

struct Span {
  std::string name;
  double start;
  double end;
  double busy = 0.0;      // tasks on several threads: their summed seconds of work
  std::uint64_t tasks = 0; // ... and how many
};

struct State {
  std::mutex mutex;
  std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
  std::vector<Span> spans;
};

inline State &state() {
  static State instance;
  return instance;
}

// A new timeline starts now (earlier spans are dropped).
inline void begin() {
  State &s = state();
  std::lock_guard lock(s.mutex);
  s.origin = std::chrono::steady_clock::now();
  s.spans.clear();
}

inline double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - state().origin).count();
}

inline void add(std::string name, double start, double end, double busy = 0.0,
                std::uint64_t tasks = 0) {
  State &s = state();
  std::lock_guard lock(s.mutex);
  s.spans.push_back({std::move(name), start, end, busy, tasks});
}

inline std::vector<Span> spans() {
  State &s = state();
  std::lock_guard lock(s.mutex);
  return s.spans;
}

// A section of the calling thread, from its construction to its destruction.
class Section {
public:
  explicit Section(const char *name) : name_(name), start_(now()) {}
  ~Section() { add(name_, start_, now()); }
  Section(const Section &) = delete;
  Section &operator=(const Section &) = delete;

private:
  const char *name_;
  double start_;
};

} // namespace dt::timeline
