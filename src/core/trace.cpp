#include "core/trace.h"

#include <sys/resource.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace dt::trace {

namespace {

constexpr std::size_t kWorks = static_cast<std::size_t>(Work::Count);
constexpr double kSampleSeconds = 0.005;

const char *const kWorkNames[kWorks] = {
    "idle",      "other",           "presort", "build",     "sweep",  "mark",
    "partition", "partition_sweep", "leaves",  "assemble",  "gpu_host", "gpu_wait",
    "staging",   "page_lock",       "prune_arrange", "prune", "prune_other"};

using Clock = std::chrono::steady_clock;
std::atomic<Clock::rep> origin{Clock::now().time_since_epoch().count()};

// One per thread that did any traced work. Only its own thread writes it;
// the sampler reads `work`, stop() reads all of it.
struct ThreadState {
  std::atomic<std::uint8_t> work{static_cast<std::uint8_t>(Work::Idle)};
  std::atomic<double> since{0.0};
  std::array<std::atomic<double>, kWorks> seconds{};
};

struct Span {
  std::string name;
  const char *lane; // or the phase depth's text
  double start, end;
  int depth;
};

struct Sample {
  double time;
  double cpuSeconds; // user + system of the process so far
  std::array<std::uint16_t, kWorks> threads;
};

struct Recorder {
  std::mutex mutex;
  std::vector<std::unique_ptr<ThreadState>> threads;
  std::vector<Span> phases;
  std::vector<Span> gpu;
  std::map<std::string, std::vector<std::vector<std::pair<const char *, double>>>> tables;
  std::map<std::string, double> facts;
  std::map<std::string, std::string> texts;
  std::vector<Sample> samples;
  std::array<double, kWorks> totals{};
  double stopped = 0.0;
  std::thread sampler;
  std::atomic<bool> sampling{false};
};

Recorder &recorder() {
  static Recorder *instance = new Recorder; // threads may still use it at exit
  return *instance;
}

thread_local ThreadState *self = nullptr;
thread_local int phaseDepth = 0;

ThreadState &state() {
  if (!self) {
    auto fresh = std::make_unique<ThreadState>();
    fresh->since.store(now(), std::memory_order_relaxed);
    self = fresh.get();
    Recorder &r = recorder();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.threads.push_back(std::move(fresh));
  }
  return *self;
}

double cpuSeconds() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
         1e-6 * static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
}

void sample(Recorder &r) {
  Sample s{now(), cpuSeconds(), {}};
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    for (const auto &thread : r.threads) {
      ++s.threads[thread->work.load(std::memory_order_relaxed)];
    }
    r.samples.push_back(s);
  }
}

void writeString(std::ostream &out, const std::string &text) {
  out << '"';
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out << '\\' << c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out << ' ';
    } else {
      out << c;
    }
  }
  out << '"';
}

std::string number(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "%.9g", value);
  return text;
}

} // namespace

const char *workName(Work work) { return kWorkNames[static_cast<std::size_t>(work)]; }

double now() {
  const Clock::rep ticks = Clock::now().time_since_epoch().count() - origin.load(std::memory_order_relaxed);
  return std::chrono::duration<double>(Clock::duration(ticks)).count();
}

Work current() {
  if (!on() || !self) {
    return Work::Other;
  }
  return static_cast<Work>(self->work.load(std::memory_order_relaxed));
}

Work swap(Work work) {
  ThreadState &s = state();
  const double time = now();
  const auto previous = s.work.load(std::memory_order_relaxed);
  std::atomic<double> &seconds = s.seconds[previous];
  seconds.store(seconds.load(std::memory_order_relaxed) + time - s.since.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
  s.since.store(time, std::memory_order_relaxed);
  s.work.store(static_cast<std::uint8_t>(work), std::memory_order_relaxed);
  return static_cast<Work>(previous);
}

Phase::Phase(const char *name) : range_(name), name_(name), on_(on()) {
  if (on_) {
    start_ = now();
    depth_ = phaseDepth++;
  }
}

Phase::~Phase() {
  if (on_) {
    --phaseDepth;
    const double end = now();
    Recorder &r = recorder();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.phases.push_back({name_, "", start_, end, depth_});
  }
}

void gpuSpan(const std::string &name, const char *lane, double start, double end) {
  if (!on()) {
    return;
  }
  Recorder &r = recorder();
  std::lock_guard<std::mutex> lock(r.mutex);
  r.gpu.push_back({name, lane, start, end, 0});
}

void row(const char *table, std::vector<std::pair<const char *, double>> fields) {
  if (!on()) {
    return;
  }
  Recorder &r = recorder();
  std::lock_guard<std::mutex> lock(r.mutex);
  r.tables[table].push_back(std::move(fields));
}

void add(const char *name, double value) {
  if (!on()) {
    return;
  }
  Recorder &r = recorder();
  std::lock_guard<std::mutex> lock(r.mutex);
  r.facts[name] += value;
}

void info(const char *name, const std::string &value) {
  if (!on()) {
    return;
  }
  Recorder &r = recorder();
  std::lock_guard<std::mutex> lock(r.mutex);
  r.texts[name] = value;
}

void start() {
  Recorder &r = recorder();
  origin.store(Clock::now().time_since_epoch().count(), std::memory_order_relaxed);
  active.store(true, std::memory_order_relaxed);
  swap(Work::Other); // the training thread is busy unless it says otherwise
  r.sampling = true;
  r.sampler = std::thread([&r]() {
    for (auto next = Clock::now(); r.sampling.load(std::memory_order_relaxed);) {
      sample(r);
      next += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(kSampleSeconds));
      std::this_thread::sleep_until(next);
    }
  });
}

void stop() {
  if (!on()) {
    return;
  }
  Recorder &r = recorder();
  r.sampling = false;
  if (r.sampler.joinable()) {
    r.sampler.join();
  }
  sample(r);
  active.store(false, std::memory_order_relaxed);
  r.stopped = now();
  std::lock_guard<std::mutex> lock(r.mutex);
  for (const auto &thread : r.threads) {
    for (std::size_t work = 0; work < kWorks; ++work) {
      r.totals[work] += thread->seconds[work].load(std::memory_order_relaxed);
    }
    // What it is doing now, since it started.
    r.totals[thread->work.load(std::memory_order_relaxed)] +=
        r.stopped - thread->since.load(std::memory_order_relaxed);
  }
}

void write(const std::string &path, const std::vector<std::pair<std::string, std::string>> &facts) {
  Recorder &r = recorder();
  std::lock_guard<std::mutex> lock(r.mutex);
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not write " + path);
  }
  out << "{\n  \"version\": 1,\n  \"seconds\": " << number(r.stopped) << ",\n";
  for (const auto &[name, value] : facts) {
    out << "  ";
    writeString(out, name);
    out << ": " << value << ",\n";
  }
  out << "  \"info\": {";
  const char *comma = "";
  for (const auto &[name, value] : r.texts) {
    out << comma;
    writeString(out, name);
    out << ": ";
    writeString(out, value);
    comma = ", ";
  }
  out << "},\n  \"facts\": {";
  comma = "";
  for (const auto &[name, value] : r.facts) {
    out << comma;
    writeString(out, name);
    out << ": " << number(value);
    comma = ", ";
  }
  out << "},\n  \"threads_seen\": " << r.threads.size() << ",\n  \"work_seconds\": {";
  for (std::size_t work = 0; work < kWorks; ++work) {
    out << (work ? ", " : "") << '"' << kWorkNames[work] << "\": " << number(r.totals[work]);
  }
  out << "},\n  \"phases\": [";
  comma = "\n    ";
  for (const Span &span : r.phases) {
    out << comma << "{\"name\": ";
    writeString(out, span.name);
    out << ", \"depth\": " << span.depth << ", \"start\": " << number(span.start)
        << ", \"end\": " << number(span.end) << "}";
    comma = ",\n    ";
  }
  out << "\n  ],\n  \"gpu\": [";
  comma = "\n    ";
  for (const Span &span : r.gpu) {
    out << comma << "[";
    writeString(out, span.name);
    out << ", \"" << span.lane << "\", " << number(span.start) << ", " << number(span.end) << "]";
    comma = ",\n    ";
  }
  out << "\n  ],\n  \"tables\": {";
  comma = "\n    ";
  for (const auto &[name, rows] : r.tables) {
    out << comma;
    writeString(out, name);
    out << ": [";
    const char *rowComma = "\n      ";
    for (const auto &fields : rows) {
      out << rowComma << "{";
      const char *fieldComma = "";
      for (const auto &[field, value] : fields) {
        out << fieldComma << '"' << field << "\": " << number(value);
        fieldComma = ", ";
      }
      out << "}";
      rowComma = ",\n      ";
    }
    out << "\n    ]";
    comma = ",\n    ";
  }
  out << "\n  },\n  \"samples\": {\"interval\": " << number(kSampleSeconds) << ", \"time\": [";
  for (std::size_t i = 0; i < r.samples.size(); ++i) {
    out << (i ? ", " : "") << number(r.samples[i].time);
  }
  out << "],\n    \"cpu_seconds\": [";
  for (std::size_t i = 0; i < r.samples.size(); ++i) {
    out << (i ? ", " : "") << number(r.samples[i].cpuSeconds);
  }
  out << "],\n    \"threads\": {";
  for (std::size_t work = 0; work < kWorks; ++work) {
    out << (work ? ",\n      " : "\n      ") << '"' << kWorkNames[work] << "\": [";
    for (std::size_t i = 0; i < r.samples.size(); ++i) {
      out << (i ? "," : "") << r.samples[i].threads[work];
    }
    out << "]";
  }
  out << "}}\n}\n";
}

} // namespace dt::trace
