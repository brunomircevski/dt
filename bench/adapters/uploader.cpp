// ./tree's upload to the GPU (GpuGrower::uploadStaged and Staging::prepare in
// src/build/), as a C library for cluster-gpu-vs-others' gpubench.py, so cuML
// and XGBoost get their data to the GPU exactly as ./tree does, inside their
// timed fit: 4 page-locked 32 MB buffers allocated on the GPU's NUMA node,
// copy threads pinned to that node's CPUs (at most 32), each 32 MB chunk
// memcpy'd in parallel into a buffer, then an async host-to-device copy from
// it; a buffer is reused once its last copy is done. Built by
// bench/cluster/build.sh (on hopper) into bench/.tools/libuploader.so:
//   g++ -O2 -shared -fPIC uploader.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -o libuploader.so
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {
constexpr int kBuffers = 4;
constexpr size_t kBytes = size_t{32} << 20;
constexpr size_t kCopyThreads = 32;

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
  char *buffers[kBuffers] = {};
  cudaEvent_t done[kBuffers] = {};
  cudaStream_t stream = nullptr;
};
}  // namespace

extern "C" {
// cpus: the GPU's NUMA node's CPUs; the caller pins itself to them for the copies too.
void *staged_create(const int *cpus, int ncpus) {
  std::vector<int> c(cpus, cpus + ncpus);
  auto *s = new Staged;
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
