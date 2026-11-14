#pragma once

// Host-side helpers of the GPU builder: CUDA_CHECK, device arrays, kernel
// timing for --profile, and the page-locked staging buffers for copies
// between host and device. Included only by gpu_grower.cu.

#include "core/thread_pool.h"
#include "core/trace.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <sys/mman.h>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    const cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                      \
      throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +           \
                               std::to_string(__LINE__) + ": " +                      \
                               cudaGetErrorString(status));                           \
    }                                                                                 \
  } while (0)

namespace dt {

namespace {

// A growable host array that is page-locked as it grows, in place: its
// address range is reserved once, and each growth makes the next part usable
// and page-locks (cudaHostRegister) only that part. Nothing is copied or
// freed until the buffer goes, so a buffer that grows level by level costs
// about its largest size, page-locked once. (std::vector with page-locked
// memory kept every outgrown copy until the end: 1.1 GB more for a GPU-only
// HIGGS tree; freeing them, 0.3 ms per MB in cudaFreeHost.) Elements are
// trivially copyable and resize() leaves new ones uninitialised. The copies
// go range by range (copyRanges()): one page-locked range is one copy.
template <typename T> class PinnedBuffer {
public:
  PinnedBuffer() = default;
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  ~PinnedBuffer() {
    for (const Range &range : ranges_) {
      if (range.locked) {
        cudaHostUnregister(base_ + range.begin);
      }
    }
    if (base_) {
      munmap(base_, kReserve);
    }
  }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  T *data() { return reinterpret_cast<T *>(base_); }
  const T *data() const { return reinterpret_cast<const T *>(base_); }
  T &operator[](std::size_t index) { return data()[index]; }
  const T &operator[](std::size_t index) const { return data()[index]; }
  T *begin() { return data(); }
  T *end() { return data() + size_; }
  const T *begin() const { return data(); }
  const T *end() const { return data() + size_; }

  void resize(std::size_t count) {
    reserve(count);
    size_ = count;
  }

  void reserve(std::size_t count) {
    const std::size_t bytes = count * sizeof(T);
    if (bytes <= committed_) {
      return;
    }
    if (!base_) {
      void *memory = mmap(nullptr, kReserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                          -1, 0);
      if (memory == MAP_FAILED) {
        throw std::bad_alloc();
      }
      base_ = static_cast<char *>(memory);
      // Page-locking faults the pages in first: with huge pages 4x faster
      // (RTX 2080 Ti node: 256 MB in 28 ms instead of 124 ms).
      madvise(base_, kReserve, MADV_HUGEPAGE);
    }
    // Half again as much (fewer ranges), in whole 64 KiB: the first level's
    // buffers are small, and each range costs about 0.5 ms per MB of 4 KiB
    // pages (a 2 MiB first range for each of them made the root level of a
    // hybrid HIGGS tree 40 ms longer).
    constexpr std::size_t kGrain = std::size_t{64} << 10;
    const std::size_t wanted = std::max(bytes + bytes / 2, kGrain);
    const std::size_t target = std::min(kReserve, (wanted + kGrain - 1) / kGrain * kGrain);
    if (bytes > target ||
        mprotect(base_ + committed_, target - committed_, PROT_READ | PROT_WRITE) != 0) {
      throw std::bad_alloc();
    }
    const bool locked = cudaHostRegister(base_ + committed_, target - committed_,
                                         cudaHostRegisterDefault) == cudaSuccess;
    if (!locked) {
      (void)cudaGetLastError(); // stays pageable: slower copies, same results
    }
    ranges_.push_back({committed_, target, locked});
    committed_ = target;
  }

  // copy(offset, bytes) for each page-locked range of the first `bytes`.
  template <typename Copy> void copyRanges(std::size_t bytes, Copy &&copy) const {
    for (const Range &range : ranges_) {
      if (range.begin >= bytes) {
        break;
      }
      copy(range.begin, std::min(range.end, bytes) - range.begin);
    }
  }

private:
  static constexpr std::size_t kReserve = std::size_t{1} << 40; // address space only
  struct Range {
    std::size_t begin;
    std::size_t end;
    bool locked;
  };
  char *base_ = nullptr;
  std::size_t size_ = 0;
  std::size_t committed_ = 0;
  std::vector<Range> ranges_;
};

template <typename T> class DeviceArray {
public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  ~DeviceArray() {
    if (data_) {
      cudaFree(data_);
    }
  }

  // With a stream, the old memory is freed and the new allocated in stream
  // order (cudaFree would wait for the whole device, the hand-off copies on
  // another stream too).
  void reserve(std::size_t count, cudaStream_t stream = nullptr) {
    if (count <= capacity_) {
      return;
    }
    if (data_) {
      CUDA_CHECK(stream ? cudaFreeAsync(data_, stream) : cudaFree(data_));
      data_ = nullptr;
    }
    capacity_ = std::max(count, capacity_ + capacity_ / 2);
    CUDA_CHECK(stream ? cudaMallocAsync(&data_, capacity_ * sizeof(T), stream)
                      : cudaMalloc(&data_, capacity_ * sizeof(T)));
  }
  T *get() const { return data_; }
  std::size_t capacity() const { return capacity_; }

private:
  T *data_ = nullptr;
  std::size_t capacity_ = 0;
};

// Optional per-kernel timing (--profile): CUDA events around launches and
// copies. Every span goes to the trace, timed from an origin event recorded at a known host
// time (setOrigin()).
class KernelProfile {
public:
  explicit KernelProfile(bool trace) : trace_(trace) {}
  ~KernelProfile() {
    for (auto &entry : pending_) {
      cudaEventDestroy(entry.start);
      cudaEventDestroy(entry.stop);
    }
    if (origin_) {
      cudaEventDestroy(origin_);
    }
  }
  bool enabled() const { return trace_; }

  // `stream` must be idle.
  void setOrigin(cudaStream_t stream) {
    if (!trace_) {
      return;
    }
    CUDA_CHECK(cudaEventCreate(&origin_));
    CUDA_CHECK(cudaEventRecord(origin_, stream));
    CUDA_CHECK(cudaEventSynchronize(origin_));
    hostOrigin_ = trace::now();
  }

  template <typename Launch>
  void run(const char *name, cudaStream_t stream, Launch &&launch, const char *lane = "compute") {
    if (!enabled()) {
      launch();
      return;
    }
    Pending entry{name, lane, nullptr, nullptr};
    cudaEventCreate(&entry.start);
    cudaEventCreate(&entry.stop);
    cudaEventRecord(entry.start, stream);
    launch();
    cudaEventRecord(entry.stop, stream);
    std::lock_guard lock(mutex_);
    pending_.push_back(entry);
  }

  void report() {
    for (Pending &entry : pending_) {
      float ms = 0.0f;
      cudaEventSynchronize(entry.stop);
      cudaEventElapsedTime(&ms, entry.start, entry.stop);
      if (origin_) {
        float startMs = 0.0f;
        cudaEventElapsedTime(&startMs, origin_, entry.start);
        trace::gpuSpan(entry.name, entry.lane, hostOrigin_ + startMs * 1e-3,
                       hostOrigin_ + (startMs + ms) * 1e-3);
      }
      cudaEventDestroy(entry.start);
      cudaEventDestroy(entry.stop);
    }
    pending_.clear();
  }

private:
  struct Pending {
    std::string name;
    const char *lane;
    cudaEvent_t start;
    cudaEvent_t stop;
  };
  bool trace_;
  cudaEvent_t origin_ = nullptr;
  double hostOrigin_ = 0.0;
  std::mutex mutex_;
  std::vector<Pending> pending_;
};

// The CPUs of the GPU's NUMA node that this process may use; empty if that
// is unknown or the machine has a single node.
std::vector<int> gpuNodeCpus() {
  int device = 0;
  char busId[32] = {};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetPCIBusId(busId, sizeof busId, device) != cudaSuccess) {
    return {};
  }
  std::string id(busId);
  std::transform(id.begin(), id.end(), id.begin(), [](unsigned char c) { return std::tolower(c); });
  int node = -1;
  if (std::FILE *file = std::fopen(("/sys/bus/pci/devices/" + id + "/numa_node").c_str(), "r")) {
    if (std::fscanf(file, "%d", &node) != 1) {
      node = -1;
    }
    std::fclose(file);
  }
  std::FILE *file =
      node < 0 ? nullptr
               : std::fopen(("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist").c_str(), "r");
  if (!file) {
    return {};
  }
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  sched_getaffinity(0, sizeof allowed, &allowed);
  std::vector<int> cpus;
  int first = 0;
  while (std::fscanf(file, "%d", &first) == 1) {
    int last = first;
    if (std::fscanf(file, "-%d", &last) != 1) {
      last = first;
    }
    for (int cpu = first; cpu <= last; ++cpu) {
      if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) {
        cpus.push_back(cpu);
      }
    }
    if (std::fgetc(file) != ',') {
      break;
    }
  }
  std::fclose(file);
  if (static_cast<int>(cpus.size()) == CPU_COUNT(&allowed)) {
    return {}; // one node (or only its CPUs allowed): nothing to choose
  }
  return cpus;
}

// Page-locked staging buffers for copies between the GPU and pageable memory
// (GpuGrower::uploadStaged, downloadStaged). Allocating them takes tens of
// milliseconds, so gpuPrepare() does it while the dataset loads. They live as
// long as the process (freeing them at exit would race with the CUDA
// runtime's teardown).
//
// On a machine with several NUMA nodes, the buffers are on the GPU's node
// and the copies into and out of them are made by threads pinned to its
// CPUs: with the copies spread over both sockets of the A100 node, the
// traffic between the sockets slowed the transfers to half the PCIe speed
// (6 GB uploaded in 0.55-0.75 s instead of 0.3 s).
struct Staging {
  static constexpr int kBuffers = 4;
  static constexpr std::size_t kBytes = std::size_t{32} << 20;
  static constexpr std::size_t kCopyThreads = 32;
  // At least this many copy, even with fewer --threads: the copiers only run
  // during copies. With --gpu-only --threads 1, 1G rows of chess uploaded
  // in 0.67 s on the H100 node and 0.69 s on the A100 node, with 8 in 0.26
  // and 0.49 s (all threads: 0.19 and 0.48 s).
  static constexpr std::size_t kMinCopyThreads = 8;
  char *buffers[kBuffers] = {};
  // The hand-offs' own (GpuGrower::handOff), one per copy stream, each a
  // part of the device's hand-off ring.
  static constexpr int kHandOffBuffers = 2;
  static constexpr std::size_t kHandOffBytes = std::size_t{128} << 20;
  char *handOff[kHandOffBuffers] = {};
  cudaEvent_t done[kBuffers] = {};
  std::unique_ptr<ThreadPool> copiers; // on the GPU's node, if it is known
  std::once_flag once;

  // `threads` (--threads): at most this many threads copy (the caller
  // included), but at least kMinCopyThreads; the first call decides.
  void prepare(unsigned threads = kCopyThreads) {
    std::call_once(once, [this, threads]() {
      const std::vector<int> cpus = gpuNodeCpus();
      auto pin = [cpus]() {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (const int cpu : cpus) {
          CPU_SET(cpu, &set);
        }
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
      };
      // Page-locked memory is placed when it is allocated: on the node of
      // the allocating thread.
      std::thread([&]() {
        if (!cpus.empty()) {
          pin();
        }
        for (int buffer = 0; buffer < kBuffers; ++buffer) {
          void *memory = nullptr;
          CUDA_CHECK(cudaMallocHost(&memory, kBytes));
          buffers[buffer] = static_cast<char *>(memory);
          CUDA_CHECK(cudaEventCreateWithFlags(&done[buffer], cudaEventDisableTiming));
        }
        for (char *&buffer : handOff) {
          void *memory = nullptr;
          CUDA_CHECK(cudaMallocHost(&memory, kHandOffBytes));
          buffer = static_cast<char *>(memory);
        }
      }).join();
      // Without the GPU's node, the caller's pool copies when it has enough
      // threads, else unpinned copiers. Never more than the CPUs this process
      // may run on: under `taskset -c 0` (the benchmarks' one-thread runs),
      // 7 extra copiers on the one CPU doubled the upload time.
      const std::size_t wanted = std::max(kMinCopyThreads, static_cast<std::size_t>(threads));
      cpu_set_t allowed;
      CPU_ZERO(&allowed);
      const std::size_t allowedCount =
          sched_getaffinity(0, sizeof allowed, &allowed) == 0
              ? static_cast<std::size_t>(CPU_COUNT(&allowed))
              : std::max(1u, std::thread::hardware_concurrency());
      const std::size_t available = cpus.empty() ? std::max<std::size_t>(1, allowedCount) : cpus.size();
      const std::size_t copyThreads = std::min({available, kCopyThreads, wanted});
      if (copyThreads > 1 && (!cpus.empty() || threads < copyThreads)) {
        copiers = std::make_unique<ThreadPool>(static_cast<unsigned>(copyThreads - 1),
                                               cpus.empty() ? std::function<void()>{} : pin);
      }
    });
  }

  // The threads for the copies: the caller works too (ThreadPool::parallelFor).
  ThreadPool &copyPool(ThreadPool &fallback) { return copiers ? *copiers : fallback; }
};

Staging &staging() {
  static Staging *instance = new Staging; // never destroyed, see above
  return *instance;
}

} // namespace

} // namespace dt
