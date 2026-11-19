// ./tree's upload to the GPU (GpuGrower::uploadStaged and Staging::prepare in
// src/build/), as a C library for cluster-gpu-vs-others' gpubench.py, so cuML
// and XGBoost get their data to the GPU exactly as ./tree does, inside their
// timed fit: 4 page-locked 32 MB buffers allocated on the GPU's NUMA node,
// copy threads pinned to that node's CPUs (at most 32), each 32 MB chunk
// memcpy'd in parallel into a buffer, then an async host-to-device copy from
// it; a buffer is reused once its last copy is done. Built by
// bench/tools/cluster/build.sh (on hopper) into bench/tools/.tools/libuploader.so:
//   g++ -O2 -shared -fPIC uploader.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -o libuploader.so
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

namespace {
constexpr int kBuffers = 4;
constexpr size_t kBytes = size_t{32} << 20;
constexpr size_t kCopyThreads = 32;

// The CPUs of the GPU's NUMA node that this process may use; empty if that is
// unknown or the machine has a single node. A copy of ./tree's gpuNodeCpus()
// (src/build/gpu_host.cuh).
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
    if (std::fscanf(file, "%d", &node) != 1) node = -1;
    std::fclose(file);
  }
  std::FILE *file =
      node < 0 ? nullptr
               : std::fopen(("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist").c_str(), "r");
  if (!file) return {};
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  sched_getaffinity(0, sizeof allowed, &allowed);
  std::vector<int> cpus;
  int first = 0;
  while (std::fscanf(file, "%d", &first) == 1) {
    int last = first;
    if (std::fscanf(file, "-%d", &last) != 1) last = first;
    for (int cpu = first; cpu <= last; ++cpu)
      if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    if (std::fgetc(file) != ',') break;
  }
  std::fclose(file);
  if (static_cast<int>(cpus.size()) == CPU_COUNT(&allowed)) return {};  // one node: nothing to choose
  return cpus;
}

void pinTo(const std::vector<int> &cpus) {
  if (cpus.empty()) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : cpus) CPU_SET(c, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

struct Pool {  // parts-1 workers + the caller, like ThreadPool::parallelFor
  std::vector<std::thread> threads;
  std::mutex m;
  std::condition_variable start, finished;
  size_t generation = 0, size = 0;
  int pending = 0, parts;
  bool stop = false;
  char *dst = nullptr;
  const char *src = nullptr;
  Pool(int n, std::vector<int> cpus) : parts(n) {
    for (int i = 1; i < n; ++i)
      threads.emplace_back([this, i, cpus] {
        pinTo(cpus);
        size_t seen = 0;
        for (;;) {
          std::unique_lock<std::mutex> l(m);
          start.wait(l, [&] { return stop || generation != seen; });
          if (stop) return;
          seen = generation;
          l.unlock();
          work(i);
          l.lock();
          if (--pending == 0) finished.notify_one();
        }
      });
  }
  void work(int p) {
    const size_t b = size * p / parts, e = size * (p + 1) / parts;
    std::memcpy(dst + b, src + b, e - b);
  }
  void run(char *d, const char *s, size_t n) {
    {
      std::lock_guard<std::mutex> l(m);
      dst = d, src = s, size = n, pending = parts - 1, ++generation;
    }
    start.notify_all();
    work(0);
    std::unique_lock<std::mutex> l(m);
    finished.wait(l, [&] { return pending == 0; });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> l(m);
      stop = true;
    }
    start.notify_all();
    for (auto &t : threads) t.join();
  }
};

struct Staged {
  Pool *pool = nullptr;
  int cpus = 0;  // copy threads pinned to the GPU's node (0: unpinned)
  char *buffers[kBuffers] = {};
  cudaEvent_t done[kBuffers] = {};
  cudaStream_t stream = nullptr;
};
}  // namespace

extern "C" {
// The buffers and copy threads, on the current device's NUMA node (as ./tree's
// Staging::prepare: the buffers allocated by a thread pinned to its CPUs, the
// copy workers pinned to them, the caller copying too).
void *staged_create() {
  const std::vector<int> c = gpuNodeCpus();
  auto *s = new Staged;
  s->cpus = static_cast<int>(c.size());
  std::thread([&] {
    pinTo(c);
    for (int b = 0; b < kBuffers; ++b) {
      cudaMallocHost(reinterpret_cast<void **>(&s->buffers[b]), kBytes);
      cudaEventCreateWithFlags(&s->done[b], cudaEventDisableTiming);
    }
  }).join();
  cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking);
  const size_t n = std::min<size_t>(c.empty() ? kCopyThreads : c.size(), kCopyThreads);
  s->pool = new Pool(static_cast<int>(n), c);
  return s;
}

// How many CPUs the copy threads are pinned to (0: not pinned).
int staged_cpus(void *handle) { return static_cast<Staged *>(handle)->cpus; }

int staged_copy(void *handle, void *target, const void *source, size_t bytes) {
  auto *s = static_cast<Staged *>(handle);
  const char *from = static_cast<const char *>(source);
  char *to = static_cast<char *>(target);
  for (size_t off = 0, chunk = 0; off < bytes; off += kBytes, ++chunk) {
    const int b = static_cast<int>(chunk % kBuffers);
    const size_t size = std::min(kBytes, bytes - off);
    cudaEventSynchronize(s->done[b]);
    s->pool->run(s->buffers[b], from + off, size);
    cudaMemcpyAsync(to + off, s->buffers[b], size, cudaMemcpyHostToDevice, s->stream);
    cudaEventRecord(s->done[b], s->stream);
  }
  return static_cast<int>(cudaStreamSynchronize(s->stream));
}
}
