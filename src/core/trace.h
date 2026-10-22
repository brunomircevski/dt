#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/nvtx.h"

// Where training time goes (--profile FILE): a JSON file with
//  * phases: wall-time spans of the training thread (gpu setup, presort, ...);
//  * work: what every thread is doing, as thread-seconds per kind of work
//    (Work), and every 5 ms how many threads do each kind (the timeline);
//  * gpu spans: every kernel and copy of the Cuda backend (CUDA events);
//  * tables (e.g. one row per GPU level) and summed facts (bytes copied).
// Off (the default), every hook costs one relaxed load and a branch.
namespace dt::trace {

// What a thread is doing. A thread starts as Idle; Doing scopes nest. Pool
// tasks and parallelFor items inherit the kind of work of the thread that
// made them (ThreadPool), so the helpers of a sweep count as sweeping.
enum class Work : std::uint8_t {
  Idle,           // blocked: waiting for work or for other threads
  Other,          // anything not listed below
  Presort,        // sorting the columns once (CPU)
  Build,          // growing: bookkeeping of nodes (choosing the split, the node store)
  Sweep,          // growing: scanning sorted columns for the best cut
  Mark,           // growing: marking the rows that go left
  Partition,      // growing: stable partition of the columns
  PartitionSweep, // growing: partition fused with the children's sweeps (the sweeps count as Sweep)
  Leaves,         // growing: recording the leaf of every row (C4.5)
  Assemble,       // the node store into the tree
  GpuHost,        // Cuda: preparing and launching GPU work
  GpuWait,        // Cuda: waiting for the GPU or a copy to finish
  Staging,        // Cuda: copying between pageable and page-locked memory
  PageLock,       // Cuda: touching host memory ahead of copies, or waiting for it
  PruneArrange,   // C4.5 pruning: rows ordered by leaf, row-major
  Prune,          // pruning proper: the error estimates of subtrees
  PruneOther,     // pruning: setup, collapsing, thresholds, compaction
  Count
};
const char *workName(Work work);

inline std::atomic<bool> active{false};
inline bool on() { return active.load(std::memory_order_relaxed); }

double now(); // seconds since start()

Work current();       // this thread's work (Other when off or not yet seen)
Work swap(Work work); // this thread now does `work`; returns what it did

class Doing {
public:
  explicit Doing(Work work) : on_(on()) {
    if (on_) {
      previous_ = swap(work);
    }
  }
  ~Doing() {
    if (on_) {
      swap(previous_);
    }
  }
  Doing(const Doing &) = delete;
  Doing &operator=(const Doing &) = delete;

private:
  bool on_;
  Work previous_ = Work::Other;
};

// A wall-time span of the training thread (phases may nest), also an NVTX range.
class Phase {
public:
  explicit Phase(const char *name);
  ~Phase();
  Phase(const Phase &) = delete;
  Phase &operator=(const Phase &) = delete;

private:
  NvtxRange range_;
  const char *name_;
  double start_ = 0.0;
  int depth_ = 0;
  bool on_;
};

// Cuda: a kernel or copy that ran on the GPU from `start` to `end` (now()'s
// clock), on `lane` ("compute" or "copy").
void gpuSpan(const std::string &name, const char *lane, double start, double end);
// A row of the table `table`, e.g. one per GPU level.
void row(const char *table, std::vector<std::pair<const char *, double>> fields);
// Add `value` to the fact `name` (e.g. bytes copied).
void add(const char *name, double value);
// A text fact, e.g. the GPU's name.
void info(const char *name, const std::string &value);

// Start recording on the calling thread (the training thread).
void start();
// Stop recording (once; later calls do nothing), then write everything to
// `path` with `facts` (name, JSON value).
void stop();
void write(const std::string &path, const std::vector<std::pair<std::string, std::string>> &facts);

} // namespace dt::trace
