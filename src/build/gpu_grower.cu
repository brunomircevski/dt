// GPU tree builder (Cuda backend).
//
// The same algorithm as CpuTreeBuilder (presorted columns, one threshold sweep
// per feature, stable partition), reorganised for the GPU:
//
// The kernels are in gpu_kernels.cuh, the CUDA helpers (device arrays,
// staging buffers, kernel timing) in gpu_host.cuh; this file is the host side
// that drives them.
//
//  * Presort: every feature column is sorted once with CUB radix sort, each
//    as soon as its values are uploaded.
//  * Memory: one copy of the sorted columns, in a buffer of a slot per column
//    and kSpareSlots more (about 8 bytes per value). The partition moves each
//    column to a free slot, and the hand-offs to the CPU go through a small
//    ring (kRingBytes), so the peak is ~(features + 1) * rows * 8 bytes + 512 MB.
//  * Breadth-first: all large nodes of one tree level are processed together
//    by a handful of kernel launches, so even deep levels with hundreds of
//    nodes keep the whole GPU busy. Work is cut into tiles of kTile entries of
//    one (node, feature) column range (a segment); every kernel runs one block
//    per tile or per segment.
//  * The host knows every node's class counts before the node is processed
//    (from its parent's split), so it decides up front which nodes are leaves;
//    only nodes that search for a split enter a level. Per level:
//      0. segmentConstant  segments whose values are all equal (they are
//                          skipped by every kernel, and dropped below)
//      1. tileHistogram    class counts of every tile (and its cuts); below
//                          the root, the parent level's partition counts them
//      2. segmentPrefix    per segment: exclusive scan over its tiles, giving
//                          each tile the class counts left of it
//      3. tileExact        each tile scores its cuts exactly (the CPU's
//                          double-precision function); with two classes,
//                          tileBound first rules out tiles that cannot hold
//                          the best cut
//      4. segmentBest      best cut per segment and the class counts left of
//                          it -> host picks the split (SplitRules) and the
//                          children
//      5. markGoesLeft / partition:
//                          stable partition of the columns of split nodes, each
//                          into a free slot, in one pass, skipped for nodes
//                          whose children are both leaves
//    so each level needs one round trip to the host (step 4).
//  * Children with fewer than --gpu-min-rows rows are gathered into one
//    block, copied to the host (through the ring) and grown by CpuTreeBuilder tasks
//    on the thread pool while the GPU continues with the next level.

#include "algo/c45_pruning.h"
#include "build/cpu_builder.h"
#include "build/gpu_host.cuh"
#include "build/gpu_kernels.cuh"
#include "build/grower.h"
#include "core/nvtx.h"
#include "core/timing.h"
#include "core/trace.h"

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <functional>
#include <cctype>
#include <optional>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace dt {

namespace {

// One node of the current GPU level, on the host side.
struct LevelNode {
  std::uint32_t node; // NodeStore id
  std::uint32_t begin; // range in the current buffer (all columns)
  std::uint32_t count;
  int depth;
  std::vector<std::uint32_t> features; // features that can still split it
};

class GpuGrower final : public Grower {
public:
  GpuGrower(const Dataset &train, const Options &options, ThreadPool &pool,
            double &setupSeconds)
      : train_(train), options_(options), pool_(pool),
        rows_(train.rowCount), features_(train.featureCount()),
        classCount_(static_cast<int>(train.classCount())),
        codec_(makeEntryCodec(train.classCount(), train.rowCount)),
        hostGoesLeft_(new std::uint8_t[train.rowCount]),
        profile_(trace::on()) {
    if (classCount_ > kMaxGpuClasses) {
      throw std::runtime_error("The Cuda backend supports at most 64 classes.");
    }
    if (features_ > 65535) {
      throw std::runtime_error("The Cuda backend supports at most 65535 features.");
    }
    ScopedTimer timer(setupSeconds);
    setup();
  }

  ~GpuGrower() override {
    if (toucher_.joinable()) {
      toucher_.join();
    }
    for (cudaEvent_t event : {gathered_, copied_}) {
      if (event) {
        cudaEventDestroy(event);
      }
    }
    for (cudaEvent_t event : ringFree_) {
      if (event) {
        cudaEventDestroy(event);
      }
    }
    for (cudaStream_t stream : copyStreams_) {
      if (stream) {
        cudaStreamDestroy(stream);
      }
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  Tree grow(const SplitRules &rules, GrowTimings &timings, DistinctValues *distinct,
            RowLeaves *leafOfRow) override {
    if (grown_) {
      throw std::logic_error("GpuGrower: grows one tree");
    }
    grown_ = true;
    params_.minGap = rules.minValueGap();
    params_.criterion = rules.criterion();
    {
      // setup() presorted the columns during the upload (gpu setup).
      ScopedTimer timer(timings.prepareSeconds);
      trace::Phase phase("presort");
      trace::Doing doing(trace::Work::GpuHost);
      if (distinct) {
        *distinct = std::move(overlapDistinct_);
      }
    }
    Tree tree;
    tree.featureNames = train_.featureNames;
    tree.classNames = train_.classNames;
    ScopedTimer timer(timings.buildSeconds);
    std::optional<trace::Phase> setupPhase(std::in_place, "grow setup");
    NodeStore store(train_.classCount(), 2 * stride_);
    // Leaves of rows: the CPU tasks write theirs into `leafOfRow`, the GPU
    // levels into leafOfRow_ (all ones elsewhere), merged at the end.
    leafJobs_.clear();
    if (leafOfRow) {
      leafOfRow->ids.resize(stride_);
      leafOfRow->map.clear();
      leafOfRow_.reserve(stride_);
      CUDA_CHECK(cudaMemsetAsync(leafOfRow_.get(), 0xff, stride_ * sizeof(std::uint32_t),
                                 stream_));
    }
    recordLeaves_ = leafOfRow != nullptr;
    CpuTreeBuilder cpuBuilder(rules, codec_, features_, store, hostGoesLeft_.get(), &pool_,
                              options_.parallel, {}, leafOfRow ? leafOfRow->ids.data() : nullptr);
    rules_ = &rules;
    store_ = &store;
    cpuBuilder_ = &cpuBuilder;
    arenaUsed_ = 0;

    const std::vector<std::uint32_t> counts = hostRootCounts_; // presortStart()
    std::vector<std::uint32_t> allFeatures(features_);
    std::iota(allFeatures.begin(), allFeatures.end(), 0u);
    LevelNode root{store.add(counts.data()), 0, static_cast<std::uint32_t>(stride_), 0,
                   std::move(allFeatures)};
    minRows_ = gpuMinRows(options_, stride_);
    setupPhase.reset();
    if (stride_ < minRows_) {
      growOnCpu(std::move(root));
    } else {
      trace::Phase phase("gpu levels");
      trace::Doing doing(trace::Work::GpuHost);
      growLevels(std::move(root));
    }
    {
      trace::Phase phase("wait cpu subtrees");
      {
        // Every hand-off has started its CPU tasks once its copy is done.
        trace::Doing doing(trace::Work::GpuWait);
        for (cudaStream_t stream : copyStreams_) {
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
      }
      pool_.waitIdle();
    }
    if (profile_.enabled()) {
      profile_.report();
    }
    trace::Phase phase("assemble tree");
    trace::Doing doing(trace::Work::Assemble);
    if (leafOfRow) {
      mergeLeaves(leafOfRow->ids);
    }
    store.toTree(tree, &pool_, leafOfRow);
    rules_ = nullptr;
    store_ = nullptr;
    cpuBuilder_ = nullptr;
    return tree;
  }

  bool arrangeRows(std::span<const std::uint32_t> slotOfId, std::size_t slots,
                   std::span<const std::uint32_t> ids, float *rowMajor, std::uint32_t *rowOf,
                   std::uint16_t *labels, std::uint32_t *slotCounts) override {
    if (!rawValues_.get() || ids.size() != rows_ || slots == 0) {
      return false;
    }
    // The result in the column buffer: the grow is done with it.
    const std::size_t rows = rows_;
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    unsigned keyBits = 1;
    while ((std::size_t{1} << keyBits) < slots) {
      ++keyBits;
    }
    std::size_t tempBytes = 0;
    cub::DoubleBuffer<std::uint32_t> none; // only the size is asked for
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, none, none,
                                               static_cast<int>(rows), 0,
                                               static_cast<int>(keyBits), stream_));
    // The scratch follows the result in the column buffer if it fits; with
    // few features it does not (chess: 8 bytes per row left, 18 needed) and
    // comes from the stream-ordered pool. The leaves become the sort keys.
    std::uint32_t *nodeSlots, *keys, *keysAlt, *rowIds, *rowIdsAlt, *counts;
    std::uint16_t *sortedLabels;
    void *temp;
    auto layout = [&](char *scratch) { // returns the bytes used
      std::size_t used = 0;
      auto take = [&](std::size_t bytes) {
        void *at = scratch + used;
        used += (bytes + 255) / 256 * 256;
        return at;
      };
      nodeSlots = static_cast<std::uint32_t *>(take(slotOfId.size_bytes()));
      keys = static_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
      keysAlt = static_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
      rowIds = static_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
      rowIdsAlt = static_cast<std::uint32_t *>(take(rows * sizeof(std::uint32_t)));
      sortedLabels = static_cast<std::uint16_t *>(take(rows * sizeof(std::uint16_t)));
      counts = static_cast<std::uint32_t *>(take(slots * classes * sizeof(std::uint32_t)));
      temp = take(tempBytes);
      return used;
    };
    const std::size_t resultBytes = (rows * features_ * sizeof(float) + 255) / 256 * 256;
    const std::size_t scratchBytes = layout(nullptr);
    char *pooled = nullptr;
    if (resultBytes + scratchBytes <= slotCount_ * slotEntries_ * sizeof(Entry)) {
      layout(reinterpret_cast<char *>(columns_.get()) + resultBytes);
    } else if (cudaMallocAsync(reinterpret_cast<void **>(&pooled), scratchBytes, stream_) ==
               cudaSuccess) {
      layout(pooled);
    } else {
      (void)cudaGetLastError(); // out of memory: the CPU arranges them
      return false;
    }
    float *arranged = reinterpret_cast<float *>(columns_.get());
    uploadStaged(nodeSlots, slotOfId.data(), slotOfId.size_bytes(), "uploadArrange");
    uploadStaged(keys, ids.data(), rows * sizeof(std::uint32_t), "uploadArrange");
    cub::DoubleBuffer<std::uint32_t> sortedKeys(keys, keysAlt);
    cub::DoubleBuffer<std::uint32_t> sortedOrder(rowIds, rowIdsAlt);
    profile_.run("arrangeSort", stream_, [&]() {
      leafSlotKernel<<<static_cast<unsigned>((rows + 255) / 256), 256, 0, stream_>>>(
          keys, nodeSlots, rows, keys, rowIds);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp, tempBytes, sortedKeys, sortedOrder,
                                                 static_cast<int>(rows), 0,
                                                 static_cast<int>(keyBits), stream_));
    });
    CUDA_CHECK(cudaMemsetAsync(counts, 0, slots * classes * sizeof(std::uint32_t), stream_));
    const std::size_t width = rowMajor ? features_ : 1;
    profile_.run("arrangeRows", stream_, [&]() {
      arrangeKernel<<<static_cast<unsigned>((rows * width + 255) / 256), 256, 0, stream_>>>(
          rawValues_.get(), rows, static_cast<std::uint32_t>(width), sortedOrder.Current(),
          sortedKeys.Current(), labels_.get(), classCount_, rowMajor ? arranged : nullptr,
          sortedLabels, counts);
    });
    CUDA_CHECK(cudaGetLastError());
    if (rowMajor) {
      downloadStaged(rowMajor, arranged, rows * features_ * sizeof(float), "downloadArranged");
    }
    if (rowOf) {
      downloadStaged(rowOf, sortedOrder.Current(), rows * sizeof(std::uint32_t), "downloadArranged");
    }
    downloadStaged(labels, sortedLabels, rows * sizeof(std::uint16_t), "downloadArranged");
    downloadStaged(slotCounts, counts, slots * classes * sizeof(std::uint32_t), "downloadArranged");
    if (pooled) {
      CUDA_CHECK(cudaFreeAsync(pooled, stream_));
    }
    {
      trace::Doing waiting(trace::Work::GpuWait);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    if (profile_.enabled()) {
      profile_.report();
    }
    return true;
  }

  // The host arena: it only holds the columns of nodes handed to the CPU.
  std::span<std::byte> workspace() override {
    if (!arena_) { // --gpu-only
      arena_.reset(new Entry[features_ * rows_]);
    }
    waitTouched(features_ * rows_ * sizeof(Entry)); // the caller writes all of it
    return {reinterpret_cast<std::byte *>(arena_.get()), features_ * rows_ * sizeof(Entry)};
  }

private:
  // Everything that does not depend on the tree: the device, the device
  // memory, the upload of the data with the presort, the host arena.
  void setup() {
    initDevice();
    const bool keepValues = allocateDevice();
    uploadAndPresort(keepValues);
    prepareHost();
  }

  // The CUDA context, streams and events; the sweep mode.
  void initDevice() {
    double contextSeconds = 0.0;
    {
      ScopedTimer timer(contextSeconds);
      CUDA_CHECK(cudaFree(nullptr)); // creates the CUDA context if needed
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    for (cudaStream_t &stream : copyStreams_) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&gathered_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    // The level arrays grow in stream order (DeviceArray::reserve): keep what
    // the pool frees for reuse instead of returning it at every sync.
    cudaMemPool_t memoryPool = nullptr;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memoryPool, device));
    std::uint64_t keep = UINT64_MAX;
    CUDA_CHECK(cudaMemPoolSetAttribute(memoryPool, cudaMemPoolAttrReleaseThreshold, &keep));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount_, cudaDevAttrMultiProcessorCount, device));
    profile_.setOrigin(stream_);
    if (trace::on()) {
      cudaDeviceProp properties{};
      CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      trace::info("gpu", properties.name);
      trace::add("gpu_context_seconds", contextSeconds);
    }
  }

  // The device buffers. Returns whether a copy of the raw values stays on
  // the GPU for C4.5's pruning.
  bool allocateDevice() {
    const std::size_t total = features_ * rows_;
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const bool leaves = options_.algorithm == Algorithm::C45 && options_.c45.prune;
    // The columns: one slot each, and kSpareSlots to move them to (see
    // PartitionTarget). The hand-offs go through a ring of kRingParts parts.
    slotCount_ = features_ + kSpareSlots;
    ringPartEntries_ = std::max<std::size_t>(
        kTile, std::min(kRingBytes, total * sizeof(Entry)) / kRingParts / sizeof(Entry));
    // Slots start at page boundaries: then a column's old and new place
    // have the same alignment, as in two separate buffers (unaligned slots
    // made the partition 5% slower on airline).
    slotEntries_ = (rows_ + kSlotAlign - 1) / kSlotAlign * kSlotAlign;
    const std::size_t needed = (slotCount_ * slotEntries_ + kRingParts * ringPartEntries_) * sizeof(Entry) +
                               rows_ * 7 +
                               (leaves ? rows_ * sizeof(std::uint32_t) : 0) + (64u << 20);
    if (needed > freeBytes) {
      throw std::runtime_error("Not enough GPU memory: need about " +
                               std::to_string(needed >> 20) + " MiB, " +
                               std::to_string(freeBytes >> 20) +
                               " MiB free. Use --parallel instead.");
    }
    // C4.5's pruning starts by ordering the rows by leaf (arrangeRows()),
    // which needs the raw values again: a copy stays on the GPU if it fits.
    const bool keepValues = leaves && needed + total * sizeof(float) <= freeBytes;
    double allocSeconds = 0.0;
    std::optional<ScopedTimer> allocTimer(std::in_place, allocSeconds);
    columns_.reserve(slotCount_ * slotEntries_);
    ring_.reserve(kRingParts * ringPartEntries_);
    for (cudaEvent_t &event : ringFree_) {
      CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    goesLeft_.reserve(rows_ / 32 + 1);
    labels_.reserve(rows_);

    // The radix sort's workspace (presortFeature())...
    std::size_t tempBytes = 0;
    cub::DoubleBuffer<Entry> buffers(columns_.get(), columns_.get());
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, tempBytes, buffers, static_cast<int>(rows_),
                                              EntryValue{}, 0, 32, stream_));
    // ... and room for C4.5's distinct values (distinctValues()): growing it
    // there would synchronize.
    std::size_t uniqueBytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Unique(
        nullptr, uniqueBytes, thrust::make_transform_iterator(columns_.get(), ValueBits{}),
        reinterpret_cast<std::uint32_t *>(columns_.get()), static_cast<int *>(nullptr),
        static_cast<int>(rows_), stream_));
    sortTemp_.reserve(std::max(tempBytes, uniqueBytes));
    sortTempBytes_ = tempBytes;

    allocTimer.reset();
    trace::add("gpu_alloc_seconds", allocSeconds);
    return keepValues;
  }

  // Upload the labels and the raw values; presort each feature as soon as its
  // values are on the GPU.
  void uploadAndPresort(bool keepValues) {
    const std::size_t total = features_ * rows_;
    // Raw values: uploaded into the end of the column buffer (see
    // presortFeature()), through page-locked staging buffers, without
    // page-locking the source (see uploadStaged()).
    float *target = uploadedValues();
    // Presort each feature as soon as its raw values are uploaded (copies
    // on a copy stream, the presort on stream_ waiting for the chunk that
    // completes the feature): ECBDL 0.5-0.6 s less on an H100. The kept copy
    // of the raw values (C4.5 pruning) is made per feature, before the
    // presort can overwrite them.
    {
      trace::Phase phase("upload+presort");
      uploadStaged(labels_.get(), train_.labels.data(), rows_ * sizeof(std::uint16_t));
      stride_ = rows_;
      const bool distinct = options_.algorithm == Algorithm::C45;
      presortStart(distinct);
      std::size_t next = 0; // the next feature to presort
      const std::size_t featureBytes = rows_ * sizeof(float);
      if (keepValues) {
        rawValues_.reserve(total);
      }
      uploadStaged(target, train_.values.data(), total * sizeof(float), "upload", copyStreams_[0],
                   [&](std::size_t uploaded, cudaEvent_t done) {
                     if (done && next < features_ && (next + 1) * featureBytes <= uploaded) {
                       CUDA_CHECK(cudaStreamWaitEvent(stream_, done, 0));
                     }
                     while (next < features_ && (next + 1) * featureBytes <= uploaded) {
                       if (keepValues) {
                         profile_.run("copyValues", stream_, [&]() {
                           CUDA_CHECK(cudaMemcpyAsync(rawValues_.get() + next * rows_,
                                                      target + next * rows_, featureBytes,
                                                      cudaMemcpyDeviceToDevice, stream_));
                         });
                       }
                       presortFeature(next++, distinct);
                     }
                   });
    }
  }

  // The kernels' tables, the host arena for the hand-offs, the end of the
  // presort.
  void prepareHost() {
    const std::size_t total = features_ * rows_;
    // c * log2(c) tables.
    const std::vector<double> logs = xlog2xTable();
    upload(logTable_, logs);
    params_.codec = codec_;
    params_.classCount = classCount_;
    params_.logs = LogTable{logTable_.get()};

    // Host memory for the columns of nodes handed to the CPU (every row is
    // handed over at most once per grow, so features * rows entries always
    // suffice). Pageable: the hand-offs go through page-locked buffers (see
    // handOff()). --gpu-only hands nothing over (workspace() allocates the
    // arena if the pruning asks for it).
    if (!options_.gpu.only) {
      arena_.reset(new Entry[total]);
    }
    // CART with a fixed alpha stops early and hands few rows to the CPU
    // (airline: 2 of 12 GB): the copies fault in the few pages they need.
    const bool fewHandOffs = options_.gpu.only ||
                             (options_.algorithm == Algorithm::Cart &&
                              options_.cart.prune);
    {
      trace::Phase phase("presort finish");
      presortFinish(options_.algorithm == Algorithm::C45 ? &overlapDistinct_ : nullptr);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    // Only now: touching the arena during presortFinish() made the download
    // of C4.5's distinct values (4 GB on SOREL 4M) 3x slower, both on the
    // GPU's socket (setup 3.4 s instead of 2.2 s on an A100 server; the build
    // 0.05 s longer this way, CART's setup 0.1 s shorter).
    if (fewHandOffs) {
      touchedBytes_ = total * sizeof(Entry);
    } else {
      toucher_ = std::thread([this, bytes = total * sizeof(Entry)]() { touchArena(bytes); });
    }
  }

  // Copy `bytes` from pageable host memory to the device at the speed of
  // page-locked memory, without page-locking the source: the pool's threads
  // copy chunks into a few page-locked staging buffers while the previous
  // chunks are uploaded from them. Page-locking is slow with HMM (Linux's
  // heterogeneous memory management): on an H100 server 6 GB took 0.17-0.22 s
  // this way, 0.32 s page-locked in pipelined chunks and 0.35 s page-locked
  // at once; on an A100 server 0.45-0.55 s either way. Smaller chunks cost more per chunk
  // than they gain, more threads gained nothing.
  // On `stream` (stream_ if null); `uploaded(bytes, done)` is called
  // once the copies of the first `bytes` are enqueued, after
  // staging.done[...] is recorded for the last of them (see setup()).
  void uploadStaged(void *target, const void *source, std::size_t bytes,
                    const char *name = "upload", cudaStream_t stream = nullptr,
                    const std::function<void(std::size_t, cudaEvent_t)> &uploaded = {}) {
    constexpr std::size_t kChunk = Staging::kBytes;
    if (!stream) {
      stream = stream_;
    }
    trace::add("h2d_bytes", static_cast<double>(bytes));
    if (bytes <= kChunk) {
      profile_.run(name, stream, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(target, source, bytes, cudaMemcpyHostToDevice, stream));
      }, "copy");
      {
        trace::Doing waiting(trace::Work::GpuWait);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      if (uploaded) {
        uploaded(bytes, nullptr); // already there: nothing to wait for
      }
      return;
    }
    trace::Doing copying(trace::Work::Staging);
    staging().prepare();
    Staging &staging = dt::staging();
    const char *from = static_cast<const char *>(source);
    char *to = static_cast<char *>(target);
    ThreadPool &copiers = staging.copyPool(pool_);
    const std::size_t parts = std::min<std::size_t>(Staging::kCopyThreads, copiers.workerCount() + 1);
    for (std::size_t offset = 0, chunk = 0; offset < bytes; offset += kChunk, ++chunk) {
      const int buffer = static_cast<int>(chunk % Staging::kBuffers);
      const std::size_t size = std::min(kChunk, bytes - offset);
      {
        trace::Doing waiting(trace::Work::GpuWait);
        CUDA_CHECK(cudaEventSynchronize(staging.done[buffer])); // its last upload is done
      }
      char *stage = staging.buffers[buffer];
      copiers.parallelFor(parts, [&](std::size_t part) {
        const std::size_t begin = size * part / parts;
        const std::size_t end = size * (part + 1) / parts;
        std::memcpy(stage + begin, from + offset + begin, end - begin);
      });
      profile_.run(name, stream, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(to + offset, stage, size, cudaMemcpyHostToDevice, stream));
      }, "copy");
      CUDA_CHECK(cudaEventRecord(staging.done[buffer], stream));
      if (uploaded) {
        uploaded(offset + size, staging.done[buffer]);
      }
    }
    trace::Doing waiting(trace::Work::GpuWait);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // The reverse of uploadStaged(): chunks are downloaded into the staging
  // buffers, and the pool's threads copy each one out while the next ones
  // are downloaded.
  void downloadStaged(void *target, const void *source, std::size_t bytes,
                      const char *name = "download") {
    if (bytes <= Staging::kBytes / 8) {
      trace::add("d2h_bytes", static_cast<double>(bytes));
      profile_.run(name, stream_, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToHost, stream_));
      }, "copy");
      trace::Doing waiting(trace::Work::GpuWait);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      return;
    }
    char *to = static_cast<char *>(target);
    staging().prepare();
    ThreadPool &copiers = staging().copyPool(pool_);
    const std::size_t parts = std::min<std::size_t>(Staging::kCopyThreads, copiers.workerCount() + 1);
    downloadChunks(source, bytes, name, [&](std::size_t offset, const char *stage, std::size_t size) {
      trace::Doing copying(trace::Work::Staging);
      copiers.parallelFor(parts, [&](std::size_t part) {
        const std::size_t begin = size * part / parts;
        const std::size_t end = size * (part + 1) / parts;
        std::memcpy(to + offset + begin, stage + begin, end - begin);
      });
    });
  }

  // Download `bytes` chunk by chunk through the staging buffers; `consume`
  // (offset, chunk, size) takes each chunk in order, while the next ones are
  // downloaded.
  template <typename Consume>
  void downloadChunks(const void *source, std::size_t bytes, const char *name, Consume &&consume) {
    constexpr std::size_t kChunk = Staging::kBytes;
    trace::add("d2h_bytes", static_cast<double>(bytes));
    staging().prepare();
    Staging &staging = dt::staging();
    const char *from = static_cast<const char *>(source);
    const std::size_t chunks = (bytes + kChunk - 1) / kChunk;
    auto sizeOf = [&](std::size_t chunk) { return std::min(kChunk, bytes - chunk * kChunk); };
    auto download = [&](std::size_t chunk) {
      const int buffer = static_cast<int>(chunk % Staging::kBuffers);
      profile_.run(name, stream_, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(staging.buffers[buffer], from + chunk * kChunk, sizeOf(chunk),
                                   cudaMemcpyDeviceToHost, stream_));
      }, "copy");
      CUDA_CHECK(cudaEventRecord(staging.done[buffer], stream_));
    };
    for (std::size_t chunk = 0; chunk < std::min<std::size_t>(chunks, Staging::kBuffers); ++chunk) {
      download(chunk);
    }
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      const int buffer = static_cast<int>(chunk % Staging::kBuffers);
      {
        trace::Doing waiting(trace::Work::GpuWait);
        CUDA_CHECK(cudaEventSynchronize(staging.done[buffer]));
      }
      consume(chunk * kChunk, staging.buffers[buffer], sizeOf(chunk));
      if (chunk + Staging::kBuffers < chunks) {
        download(chunk + Staging::kBuffers);
      }
    }
  }

  // Touch the arena's pages on all threads, in the background (the pool is
  // idle while the GPU grows the first levels), so the hand-offs' copies
  // into it do not fault them in: that made C4.5 on airline 0.2-0.9 s
  // slower on an H100 server. The pages are spread over the NUMA nodes, like the threads that
  // read them. In chunks, in order: a hand-off waits only for the part it
  // fills (waitTouched()). (MADV_POPULATE_WRITE, which leaves the contents
  // alone, holds the process's mmap lock: other threads' allocations and
  // page faults waited for it, and an H100 server's presort went from 0.5 to
  // 4.2 s when its memory was low.)
  void touchArena(std::size_t bytes) {
    constexpr std::size_t kChunks = 256;
    trace::Doing doing(trace::Work::PageLock);
    std::vector<char> done(kChunks, 0);
    std::size_t next = 0; // chunks below it are done (touchMutex_)
    char *data = reinterpret_cast<char *>(arena_.get());
    pool_.parallelFor(kChunks, [&](std::size_t chunk) {
      const std::size_t begin = bytes * chunk / kChunks;
      const std::size_t end = bytes * (chunk + 1) / kChunks;
      std::memset(data + begin, 0, end - begin);
      std::lock_guard<std::mutex> lock(touchMutex_);
      done[chunk] = 1;
      while (next < kChunks && done[next]) {
        ++next;
      }
      touchedBytes_ = bytes * next / kChunks;
      touchCv_.notify_all();
    });
  }

  // Wait until the first `bytes` of the arena are touched.
  void waitTouched(std::size_t bytes) {
    trace::Doing waiting(trace::Work::PageLock);
    std::unique_lock<std::mutex> lock(touchMutex_);
    touchCv_.wait(lock, [&]() { return touchedBytes_ >= bytes; });
  }

  // Where setup() uploads the raw values (column-major, rows_ per feature)
  // when the grower is used once: the end of the column buffer.
  float *uploadedValues() const {
    return reinterpret_cast<float *>(columns_.get() + slotCount_ * slotEntries_) - features_ * rows_;
  }

  // The sorted column of every feature, in a slot each, presorted by setup()
  // while the raw values are uploaded: presortStart(), presortFeature() for
  // each feature as soon as its values are on the GPU, presortFinish(). One
  // feature at a time: its entries are gathered into a free slot and sorted
  // with another one as the radix sort's second buffer; the column stays in
  // the one that holds the result. Uploaded raw values take the last slots,
  // which free up as their features are sorted: a slot is taken only once the
  // raw values in it are no longer needed.
  std::vector<bool> presortTaken_;

  void presortStart(bool distinct) {
    rootCounts_.reserve(classCount_);
    CUDA_CHECK(cudaMemsetAsync(rootCounts_.get(), 0, classCount_ * sizeof(std::uint32_t), stream_));
    profile_.run("countClasses", stream_, [&]() {
      countClassesKernel<<<static_cast<unsigned>(4 * smCount_), 256, 0, stream_>>>(
          labels_.get(), static_cast<std::uint32_t>(stride_), classCount_,
          rootCounts_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    hostRootCounts_.assign(classCount_, 0);
    CUDA_CHECK(cudaMemcpyAsync(hostRootCounts_.data(), rootCounts_.get(),
                               classCount_ * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream_));
    presortTaken_.assign(slotCount_, false);
    if (distinct) {
      distinctCounts_.reserve(features_);
      distinctPlaces_.reserve(features_ + 1);
      CUDA_CHECK(cudaMemsetAsync(distinctPlaces_.get() + features_, 0, sizeof(unsigned long long),
                                 stream_));
    }
    slot_.assign(features_, 0);
  }

  void presortFeature(std::size_t feature, bool distinct) {
    // Whether slot `slot` overlaps the raw values of features `first` onwards.
    const std::size_t rawStart = 2 * slotCount_ * slotEntries_ - features_ * rows_; // in floats
    auto holdsValues = [&](std::size_t slot, std::size_t first) {
      return first < features_ &&
             2 * (slot + 1) * slotEntries_ > rawStart + first * rows_;
    };
    auto freeSlot = [&](std::size_t first, std::size_t other) {
      for (std::size_t slot = 0; slot < slotCount_; ++slot) {
        if (!presortTaken_[slot] && slot != other && !holdsValues(slot, first)) {
          return slot;
        }
      }
      throw std::logic_error("GpuGrower: no free column slot for the presort");
    };
    const std::size_t first = freeSlot(feature, slotCount_);
    const std::size_t second = freeSlot(feature + 1, first); // written after the gather
    Entry *a = columns_.get() + first * slotEntries_;
    Entry *b = columns_.get() + second * slotEntries_;
    profile_.run("presortGather", stream_, [&]() {
      gatherKernel<<<static_cast<unsigned>((stride_ + 255) / 256), 256, 0, stream_>>>(
          uploadedValues() + feature * rows_, static_cast<std::uint32_t>(stride_),
          labels_.get(), codec_, a);
    });
    CUDA_CHECK(cudaGetLastError());
    cub::DoubleBuffer<Entry> buffers(a, b);
    profile_.run("presortSort", stream_, [&]() {
      std::size_t tempBytes = sortTempBytes_;
      CUDA_CHECK(cub::DeviceRadixSort::SortKeys(sortTemp_.get(), tempBytes, buffers,
                                                static_cast<int>(stride_), EntryValue{}, 0, 32,
                                                stream_));
    });
    const bool inFirst = buffers.Current() == a;
    slot_[feature] = static_cast<std::uint32_t>(inFirst ? first : second);
    presortTaken_[slot_[feature]] = true;
    if (distinct) { // the other slot is free
      distinctValues(feature, buffers.Current(), inFirst ? b : a);
    }
  }

  void presortFinish(DistinctValues *distinct) {
    if (distinct) {
      downloadDistinct(*distinct, presortTaken_);
    }
    trace::Doing waiting(trace::Work::GpuWait);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  // The distinct values of a feature, from its sorted column: cub's unique on
  // their bit patterns (into `scratch`, a free slot), appended to the pool (the
  // hand-off ring, free until the build) if they fit.
  void distinctValues(std::size_t feature, const Entry *column, Entry *scratch) {
    const auto bits = thrust::make_transform_iterator(column, ValueBits{});
    auto *unique = reinterpret_cast<std::uint32_t *>(scratch);
    const int count = static_cast<int>(stride_);
    int *counts = distinctCounts_.get();
    std::size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, tempBytes, bits, unique, counts + feature, count,
                                         stream_));
    sortTemp_.reserve(tempBytes);
    const unsigned blocks = static_cast<unsigned>((stride_ + 255) / 256);
    profile_.run("presortDistinct", stream_, [&]() {
      CUDA_CHECK(cub::DeviceSelect::Unique(sortTemp_.get(), tempBytes, bits, unique,
                                           counts + feature, count, stream_));
      placeDistinctKernel<<<1, 1, 0, stream_>>>(counts, static_cast<std::uint32_t>(feature),
                                                distinctPoolCapacity(),
                                                distinctPlaces_.get() + features_,
                                                distinctPlaces_.get());
      copyDistinctKernel<<<blocks, 256, 0, stream_>>>(unique, counts,
                                                      static_cast<std::uint32_t>(feature),
                                                      distinctPlaces_.get(), distinctPool());
    });
    CUDA_CHECK(cudaGetLastError());
  }

  std::uint32_t *distinctPool() const { return reinterpret_cast<std::uint32_t *>(ring_.get()); }
  std::size_t distinctPoolCapacity() const {
    return kRingParts * ringPartEntries_ * sizeof(Entry) / sizeof(std::uint32_t);
  }

  // Download the pooled distinct values; the features that did not fit are
  // done again one at a time from their sorted columns (in a free slot).
  void downloadDistinct(DistinctValues &distinct, const std::vector<bool> &taken) {
    std::vector<int> counts(features_);
    std::vector<unsigned long long> places(features_);
    CUDA_CHECK(cudaMemcpyAsync(counts.data(), distinctCounts_.get(), features_ * sizeof(int),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(places.data(), distinctPlaces_.get(),
                               features_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    distinct.offsets.assign(features_ + 1, 0);
    std::size_t placed = 0; // features 0..placed-1 are in the pool, back to back
    for (std::size_t feature = 0; feature < features_; ++feature) {
      distinct.offsets[feature + 1] = distinct.offsets[feature] + static_cast<std::size_t>(counts[feature]);
      if (places[feature] == kNotPlaced) {
        continue;
      }
      placed = feature + 1;
    }
    distinct.values.resize(distinct.offsets[features_]);
    downloadStaged(distinct.values.data(), distinctPool(), distinct.offsets[placed] * sizeof(float),
                   "downloadDistinct");
    if (placed == features_) {
      return;
    }
    // The rest in batches: as many features as fit, back to back, into the
    // larger of the pool (downloaded, so free again) and a free slot (room for
    // any feature: two rows per slot), then one staged download. One feature
    // at a time was ~2000 synchronous copies on SOREL 4M (1G distinct values,
    // 8x the pool); batched, C4.5's setup there is 0.12-0.15 s shorter on an
    // A100 server.
    const std::size_t free = static_cast<std::size_t>(
        std::find(taken.begin(), taken.end(), false) - taken.begin());
    std::uint32_t *buffer = reinterpret_cast<std::uint32_t *>(columns_.get() + free * slotEntries_);
    std::size_t capacity = slotEntries_ * sizeof(Entry) / sizeof(std::uint32_t);
    if (distinctPoolCapacity() > capacity) {
      buffer = distinctPool();
      capacity = distinctPoolCapacity();
    }
    for (std::size_t first = placed; first < features_;) {
      std::size_t end = first;
      for (; end < features_ && distinct.offsets[end + 1] - distinct.offsets[first] <= capacity;
           ++end) {
        std::size_t tempBytes = sortTemp_.capacity();
        const auto bits = thrust::make_transform_iterator(column(end), ValueBits{});
        CUDA_CHECK(cub::DeviceSelect::Unique(
            sortTemp_.get(), tempBytes, bits,
            buffer + (distinct.offsets[end] - distinct.offsets[first]),
            distinctCounts_.get() + end, static_cast<int>(stride_), stream_));
      }
      downloadStaged(distinct.values.data() + distinct.offsets[first], buffer,
                     (distinct.offsets[end] - distinct.offsets[first]) * sizeof(float),
                     "downloadDistinct");
      first = end;
    }
  }

  // Too small for the GPU: copy the presorted columns and use the CPU.
  void growOnCpu(LevelNode root) {
    waitTouched(features_ * stride_ * sizeof(Entry));
    for (std::size_t feature = 0; feature < features_; ++feature) {
      downloadStaged(arena_.get() + feature * stride_, column(feature), stride_ * sizeof(Entry),
                     "downloadColumns");
    }
    cpuBuilder_->grow(Columns{arena_.get(), stride_},
                      Subtree{root.node, 0, root.count, 0, std::move(root.features), {}, {}});
  }

  void growLevels(LevelNode root) {
    if (rules_->isTerminal(store_->counts(root.node), root.count, 0)) {
      addLeafJob(columnBase(0), root.count, root.node);
      launchLeafJobs();
      return;
    }
    std::vector<LevelNode> level;
    level.push_back(std::move(root));
    histogramReady_ = false;
    for (int depth = 0; !level.empty(); ++depth) {
      double seconds = 0.0;
      std::size_t rows = 0;
      std::size_t segments = 0;
      for (const LevelNode &node : level) {
        rows += node.count;
        segments += node.features.size();
      }
      const std::size_t nodes = level.size();
      {
        ScopedTimer timer(seconds);
        const std::string name = "gpu level " + std::to_string(depth);
        NvtxRange range(name.c_str());
        level = processLevel(level);
      }
      trace::row("gpu_levels", {{"depth", depth}, {"nodes", static_cast<double>(nodes)},
                                {"rows", static_cast<double>(rows)},
                                {"segments", static_cast<double>(segments)},
                                {"end", trace::now()}, {"seconds", seconds}});
    }
  }

  template <int MAXK> void launchScans(std::uint32_t tiles, std::size_t segmentCount) {
    const unsigned segmentBlocks = static_cast<unsigned>(segmentCount);
    const Entry *src = columns_.get();
    // Two classes: bound every tile, so the sweeps skip the hopeless ones.
    const double *tileUpper = nullptr;
    const double *segmentLower = nullptr;
    if (classCount_ == 2) {
      profile_.run("tileBound", stream_, [&]() {
        tileBoundKernel<<<(tiles + 255) / 256, 256, 0, stream_>>>(
            src, segments_.get(), tiles, tileSegment_.get(), constant_.get(), nodes_.get(),
            params_, tileHist_.get(), nodeCounts_.get(), tileUpper_.get(), tileLower_.get());
      });
      CUDA_CHECK(cudaGetLastError());
      profile_.run("segmentLower", stream_, [&]() {
        segmentMaxKernel<double><<<segmentBlocks, kThreads, 0, stream_>>>(
            segments_.get(), tileLower_.get(), segmentLower_.get());
      });
      CUDA_CHECK(cudaGetLastError());
      tileUpper = tileUpper_.get();
      segmentLower = segmentLower_.get();
    }
    profile_.run("tileExact", stream_, [&]() {
      tileExactKernel<MAXK><<<tiles, kThreads, 0, stream_>>>(
          src, segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(), params_,
          tileHist_.get(), nodeCounts_.get(), tileUpper, segmentLower, tileBest_.get(),
          tileTries_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("segmentBest", stream_, [&]() {
      segmentBestKernel<<<segmentBlocks, kThreads, 0, stream_>>>(
          src, segments_.get(), params_, tileHist_.get(), tileBest_.get(), tileTries_.get(),
          segmentBest_.get(), segmentLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
  }

  template <int MAXK> void launchHistogram(std::uint32_t tiles) {
    profile_.run("tileHistogram", stream_, [&]() {
      tileHistogramKernel<MAXK><<<tiles, kThreads, 0, stream_>>>(
          columns_.get(), segments_.get(), tileSegment_.get(), constant_.get(), nodes_.get(),
          params_, tileHist_.get(), tileTries_.get());
    });
    CUDA_CHECK(cudaGetLastError());
  }

  void dispatchHistogram(std::uint32_t tiles) {
    if (classCount_ <= 2) launchHistogram<2>(tiles);
    else if (classCount_ <= 4) launchHistogram<4>(tiles);
    else if (classCount_ <= 8) launchHistogram<8>(tiles);
    else if (classCount_ <= 16) launchHistogram<16>(tiles);
    else if (classCount_ <= 32) launchHistogram<32>(tiles);
    else launchHistogram<64>(tiles);
  }

  void dispatchScans(std::uint32_t tiles, std::size_t segmentCount) {
    if (classCount_ <= 2) launchScans<2>(tiles, segmentCount);
    else if (classCount_ <= 4) launchScans<4>(tiles, segmentCount);
    else if (classCount_ <= 8) launchScans<8>(tiles, segmentCount);
    else if (classCount_ <= 16) launchScans<16>(tiles, segmentCount);
    else if (classCount_ <= 32) launchScans<32>(tiles, segmentCount);
    else launchScans<64>(tiles, segmentCount);
  }

  template <typename T> void upload(DeviceArray<T> &target, const std::vector<T> &source) {
    target.reserve(source.size(), stream_);
    profile_.run("uploadLevel", stream_, [&]() {
      CUDA_CHECK(cudaMemcpyAsync(target.get(), source.data(), source.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_));
    }, "copy");
  }

  template <typename T>
  void download(std::vector<T> &target, const DeviceArray<T> &source, std::size_t count,
                bool wait = true) {
    target.resize(count);
    profile_.run("downloadLevel", stream_, [&]() {
      CUDA_CHECK(cudaMemcpyAsync(target.data(), source.get(), count * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream_));
    }, "copy");
    if (wait) {
      trace::Doing waiting(trace::Work::GpuWait);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
  }

  // A level as the kernels see it: one segment per (node, feature that can
  // still split it), cut into tiles of kTile entries.
  struct LevelPlan {
    std::vector<Segment> segments;
    std::vector<std::size_t> firstSegment; // per node, and the end
    std::vector<NodeDesc> nodes;
    std::uint32_t tiles = 0;
  };

  // What steps 0-4 found, per segment.
  struct SegmentCuts {
    std::vector<CutCandidate> best;     // its best cut
    std::vector<std::uint32_t> lefts;   // segment * classes: class counts left of it
    std::vector<std::uint8_t> constant; // all its values are (nearly) equal
  };

  // The children of a level's splits that are not leaves.
  struct LevelChildren {
    std::vector<LevelNode> next;  // they stay on the GPU
    std::vector<std::pair<std::size_t, int>> nextParent; // (node, side) of each of `next`
    std::vector<LevelNode> toCpu; // too small: handed to the CPU
    bool anyPartition = false;    // some node's columns must be partitioned
  };

  // Split the nodes of one level. Every node in `level` has its class counts
  // set and is not terminal. Returns the children that stay on the GPU.
  std::vector<LevelNode> processLevel(const std::vector<LevelNode> &level) {
    LevelPlan plan = planLevel(level);
    const SegmentCuts found = findBestCuts(plan);
    LevelChildren children = splitNodes(level, plan, found);
    launchLeafJobs(); // reads this level's buffer, before handOff() reuses it
    if (!children.anyPartition) {
      return {};
    }
    partitionLevel(plan, found, children);
    // 6: small children -> CPU.
    if (!children.toCpu.empty()) {
      handOff(children.toCpu);
    }
    return std::move(children.next);
  }

  // The segments and nodes of a level, uploaded, and room for the kernels'
  // per-tile and per-segment results.
  LevelPlan planLevel(const std::vector<LevelNode> &level) {
    const std::size_t nodeCount = level.size();
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    LevelPlan plan;
    // Segments: one per (node, feature that can still split it).
    std::vector<Segment> &segments = plan.segments;
    std::vector<std::size_t> &firstSegment = plan.firstSegment;
    std::vector<NodeDesc> &nodes = plan.nodes;
    firstSegment.assign(nodeCount + 1, 0);
    nodes.resize(nodeCount);
    std::vector<std::uint32_t> tileSegment;
    std::vector<std::uint32_t> nodeCounts(nodeCount * classes);
    for (std::size_t n = 0; n < nodeCount; ++n) {
      const LevelNode &node = level[n];
      const std::uint32_t tilesPerSegment = (node.count + kTile - 1) / kTile;
      firstSegment[n] = segments.size();
      for (const std::uint32_t f : node.features) {
        const std::uint32_t index = static_cast<std::uint32_t>(segments.size());
        segments.push_back({columnBase(f) + node.begin, node.count, static_cast<std::uint32_t>(n),
                            f, static_cast<std::uint32_t>(tileSegment.size()),
                            tilesPerSegment});
        tileSegment.insert(tileSegment.end(), tilesPerSegment, index);
      }
      const std::uint32_t *counts = store_->counts(node.node);
      std::copy_n(counts, classes, nodeCounts.data() + n * classes);
      nodes[n].count = node.count;
      nodes[n].minChild = rules_->minChildRows(node.count);
      nodes[n].parentWeighted = weightedImpurity(counts, classCount_, node.count,
                                                 rules_->criterion(), rules_->logTable());
    }
    firstSegment[nodeCount] = segments.size();
    const std::size_t segmentCount = segments.size();
    const std::uint32_t tiles = static_cast<std::uint32_t>(tileSegment.size());
    plan.tiles = tiles;
    upload(segments_, segments);
    upload(tileSegment_, tileSegment);
    upload(nodes_, nodes);
    upload(nodeCounts_, nodeCounts);
    constant_.reserve(segmentCount, stream_);
    tileHist_.reserve(static_cast<std::size_t>(tiles) * classes, stream_);
    tileBest_.reserve(tiles, stream_);
    tileTries_.reserve(tiles, stream_);
    if (classCount_ == 2) {
      tileUpper_.reserve(tiles, stream_);
      tileLower_.reserve(tiles, stream_);
      segmentLower_.reserve(segmentCount, stream_);
    }
    tileStates_.reserve(static_cast<std::size_t>(tiles) + 1, stream_);
    segmentBest_.reserve(segmentCount, stream_);
    segmentLeft_.reserve(segmentCount * classes, stream_);
    return plan;
  }

  // Steps 0-4: constant segments, class counts left of every tile, then the
  // best cut per segment with its left class counts.
  SegmentCuts findBestCuts(const LevelPlan &plan) {
    const std::size_t segmentCount = plan.segments.size();
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    const std::uint32_t tiles = plan.tiles;
    SegmentCuts found;
    profile_.run("segmentConstant", stream_, [&]() {
      segmentConstantKernel<<<static_cast<unsigned>((segmentCount + 255) / 256), 256, 0,
                              stream_>>>(columns_.get(), segments_.get(), segmentCount,
                                         params_.minGap, constant_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    if (!histogramReady_) { // else counted by the previous level's partition
      dispatchHistogram(tiles);
    }
    histogramReady_ = false;
    profile_.run("segmentPrefix", stream_, [&]() {
      segmentPrefixKernel<<<static_cast<unsigned>(segmentCount), kThreads, 0, stream_>>>(
          tileHist_.get(), classCount_, segments_.get(), constant_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    dispatchScans(tiles, segmentCount);
    download(found.best, segmentBest_, segmentCount, false);
    download(found.constant, constant_, segmentCount, false);
    download(found.lefts, segmentLeft_, segmentCount * classes);
    return found;
  }

  // The split of every node (SplitRules on the host). Children that are
  // leaves are finished right here.
  LevelChildren splitNodes(const std::vector<LevelNode> &level, LevelPlan &plan,
                           const SegmentCuts &found) {
    const std::size_t nodeCount = level.size();
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    LevelChildren children;
    const std::vector<Segment> &segments = plan.segments;
    const std::vector<std::size_t> &firstSegment = plan.firstSegment;
    std::vector<NodeDesc> &nodes = plan.nodes;
    const std::vector<CutCandidate> &best = found.best;
    const std::vector<std::uint32_t> &lefts = found.lefts;
    const std::vector<std::uint8_t> &constant = found.constant;
    bool &anyPartition = children.anyPartition;
    std::vector<LevelNode> &next = children.next;
    std::vector<std::pair<std::size_t, int>> &nextParent = children.nextParent;
    std::vector<LevelNode> &toCpu = children.toCpu;
    thread_local std::vector<CutCandidate> cuts;
    for (std::size_t n = 0; n < nodeCount; ++n) {
      const LevelNode &parent = level[n];
      cuts.assign(features_, CutCandidate{});
      std::vector<std::uint32_t> features; // still useful below this node
      int winnerSegment = -1;
      for (std::size_t s = firstSegment[n]; s < firstSegment[n + 1]; ++s) {
        cuts[segments[s].feature] = best[s];
        if (!constant[s]) {
          features.push_back(segments[s].feature);
        }
      }
      const SplitRules::Decision decision =
          rules_->choose(cuts.data(), features_, parent.count);
      if (decision.feature < 0) {
        addLeafJob(segments[firstSegment[n]].base, parent.count, parent.node);
        continue;
      }
      for (std::size_t s = firstSegment[n]; s < firstSegment[n + 1]; ++s) {
        if (static_cast<int>(segments[s].feature) == decision.feature) {
          winnerSegment = static_cast<int>(s);
        }
      }
      nodes[n].splitFeature = decision.feature;
      nodes[n].leftCount = decision.leftCount;
      nodes[n].marksRight = 2 * std::size_t{decision.leftCount} > parent.count;

      const std::uint32_t *parentCounts = store_->counts(parent.node);
      std::vector<std::uint32_t> childCounts(2 * classes);
      std::uint32_t *leftCounts = childCounts.data();
      std::uint32_t *rightCounts = leftCounts + classes;
      std::copy_n(lefts.data() + static_cast<std::size_t>(winnerSegment) * classes, classes,
                  leftCounts);
      for (std::size_t k = 0; k < classes; ++k) {
        rightCounts[k] = parentCounts[k] - leftCounts[k];
      }
      const std::uint32_t leftId = store_->add(leftCounts);
      const std::uint32_t rightId = store_->add(rightCounts);
      Node &node = store_->node(parent.node);
      node.feature = decision.feature;
      node.threshold = decision.threshold;
      node.left = leftId;
      node.right = rightId;

      for (const bool left : {true, false}) {
        const std::uint32_t begin = left ? parent.begin : parent.begin + decision.leftCount;
        const std::uint32_t count =
            left ? decision.leftCount : parent.count - decision.leftCount;
        const int depth = parent.depth + 1;
        if (rules_->isTerminal(left ? leftCounts : rightCounts, count, depth)) {
          addLeafJob(segments[static_cast<std::size_t>(winnerSegment)].base + (begin - parent.begin),
                     count, left ? leftId : rightId);
          continue;
        }
        nodes[n].partition = 1;
        LevelNode child{left ? leftId : rightId, begin, count, depth, features};
        if (count >= minRows_) {
          next.push_back(std::move(child));
          nextParent.emplace_back(n, left ? 0 : 1);
        } else {
          toCpu.push_back(std::move(child));
        }
      }
      anyPartition = anyPartition || nodes[n].partition;
    }
    return children;
  }

  // Step 5: stable partition of the non-constant columns of split nodes, each
  // into a free slot (see PartitionTarget). It also counts the class
  // histogram and the cuts of every tile of the next level, laid out as the
  // next planLevel() lays it out (children in order, then their features).
  void partitionLevel(const LevelPlan &plan, const SegmentCuts &found,
                      const LevelChildren &children) {
    const std::vector<Segment> &segments = plan.segments;
    const std::vector<std::size_t> &firstSegment = plan.firstSegment;
    const std::vector<NodeDesc> &nodes = plan.nodes;
    const std::vector<std::uint8_t> &constant = found.constant;
    const std::vector<LevelNode> &next = children.next;
    const std::vector<std::pair<std::size_t, int>> &nextParent = children.nextParent;
    const std::size_t nodeCount = nodes.size();
    const std::size_t segmentCount = segments.size();
    const std::size_t classes = static_cast<std::size_t>(classCount_);
    const std::uint32_t tiles = plan.tiles;
    upload(nodes_, nodes);
    std::vector<PartitionTarget> targets(
        segmentCount, PartitionTarget{{kNoTarget, kNoTarget}, {0, 0}, {0, 0}});
    std::size_t nextTiles = 0;
    for (std::size_t c = 0; c < next.size(); ++c) {
      const LevelNode &child = next[c];
      const auto [n, side] = nextParent[c];
      const std::uint32_t tilesPerSegment = (child.count + kTile - 1) / kTile;
      std::size_t s = firstSegment[n];
      for (const std::uint32_t f : child.features) { // in the order of n's segments
        while (segments[s].feature != f) {
          ++s;
        }
        targets[s].firstTile[side] = static_cast<std::uint32_t>(nextTiles);
        targets[s].count[side] = child.count;
        targets[s].minChild[side] = rules_->minChildRows(child.count);
        nextTiles += tilesPerSegment;
      }
    }
    upload(partitionTargets_, targets);
    if (nextTiles > 0) {
      // This level is done with them (the kernels are ordered on the stream).
      tileHist_.reserve(nextTiles * classes, stream_);
      tileTries_.reserve(nextTiles, stream_);
      CUDA_CHECK(cudaMemsetAsync(tileHist_.get(), 0, nextTiles * classes * sizeof(std::uint32_t),
                                 stream_));
      CUDA_CHECK(cudaMemsetAsync(tileTries_.get(), 0, nextTiles * sizeof(std::uint32_t), stream_));
      histogramReady_ = true;
    }
    // The columns that move (those with a segment to partition), in feature
    // order. Each takes the first free slot: at first the slots that hold no
    // column of this level, then those that the moved columns leave.
    auto partitioned = [&](std::size_t s) {
      return !constant[s] && nodes[segments[s].node].partition;
    };
    std::vector<std::uint32_t> moveOf(features_, kNoTarget);
    for (std::size_t s = 0; s < segmentCount; ++s) {
      if (partitioned(s)) {
        moveOf[segments[s].feature] = 0;
      }
    }
    std::vector<bool> occupied(slotCount_, false);
    for (std::uint32_t f = 0; f < features_; ++f) {
      if (moveOf[f] != kNoTarget) {
        occupied[slot_[f]] = true;
      }
    }
    std::deque<std::pair<std::uint32_t, std::uint32_t>> freeSlots; // (slot, column leaving it)
    for (std::uint32_t slot = 0; slot < slotCount_; ++slot) {
      if (!occupied[slot]) {
        freeSlots.emplace_back(slot, kNoTarget);
      }
    }
    std::vector<std::uint32_t> leaving(features_, kNoTarget); // the column leaving f's new slot
    std::vector<long long> shift(features_, 0);
    for (std::uint32_t f = 0; f < features_; ++f) {
      if (moveOf[f] == kNoTarget) {
        continue;
      }
      const auto [slot, left] = freeSlots.front();
      freeSlots.pop_front();
      leaving[f] = left;
      shift[f] = (static_cast<long long>(slot) - static_cast<long long>(slot_[f])) *
                 static_cast<long long>(slotEntries_);
      freeSlots.emplace_back(slot_[f], f);
      slot_[f] = slot;
    }
    // What each segment waits for.
    std::vector<PartitionMove> moves(segmentCount, PartitionMove{0, kNoTarget});
    std::vector<std::uint32_t> segmentOf(features_, kNoTarget); // of the node, by feature
    for (std::size_t n = 0; n < nodeCount; ++n) {
      if (!nodes[n].partition) {
        continue;
      }
      for (std::size_t s = firstSegment[n]; s < firstSegment[n + 1]; ++s) {
        segmentOf[segments[s].feature] = partitioned(s) ? static_cast<std::uint32_t>(s) : kNoTarget;
      }
      for (std::size_t s = firstSegment[n]; s < firstSegment[n + 1]; ++s) {
        if (!partitioned(s)) {
          continue;
        }
        const std::uint32_t f = segments[s].feature;
        moves[s].shift = shift[f];
        const std::uint32_t waited = leaving[f] == kNoTarget ? kNoTarget : segmentOf[leaving[f]];
        if (waited != kNoTarget) {
          moves[s].waitTile = segments[waited].firstTile + segments[waited].tileCount - 1;
        }
      }
    }
    upload(partitionMoves_, moves);
    profile_.run("markGoesLeft", stream_, [&]() {
      CUDA_CHECK(cudaMemsetAsync(goesLeft_.get(), 0, (rows_ / 32 + 1) * sizeof(std::uint32_t),
                                 stream_));
      markGoesLeftKernel<<<tiles, kThreads, 0, stream_>>>(columns_.get(), segments_.get(),
                                                          tileSegment_.get(), nodes_.get(), codec_,
                                                          goesLeft_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    profile_.run("partition", stream_, [&]() {
      // Tile states and the tile counter start at zero.
      CUDA_CHECK(cudaMemsetAsync(tileStates_.get(), 0, (tiles + 1) * sizeof(TileState), stream_));
      partitionKernel<<<tiles, kThreads, 0, stream_>>>(
          columns_.get(), columns_.get(), segments_.get(), tileSegment_.get(), constant_.get(),
          nodes_.get(), params_, goesLeft_.get(), partitionTargets_.get(), partitionMoves_.get(),
          tileStates_.get(),
          reinterpret_cast<unsigned int *>(tileStates_.get() + tiles), tileHist_.get(),
          tileTries_.get());
    });
    CUDA_CHECK(cudaGetLastError());
  }

  // The rows of a leaf made on the GPU: `count` entries of one of its columns
  // at `from` in the current buffer.
  void addLeafJob(std::size_t from, std::uint32_t count, std::uint32_t leaf) {
    if (!recordLeaves_) {
      return;
    }
    constexpr std::uint32_t kPiece = 1u << 14; // entries per block
    for (std::uint32_t done = 0; done < count; done += kPiece) {
      leafJobs_.push_back({from + done, std::min(kPiece, count - done), leaf});
    }
  }

  void launchLeafJobs() {
    if (leafJobs_.empty()) {
      return;
    }
    upload(leafJobsDevice_, leafJobs_);
    profile_.run("leafRows", stream_, [&]() {
      leafKernel<<<static_cast<unsigned>(leafJobs_.size()), kThreads, 0, stream_>>>(
          columns_.get(), leafJobsDevice_.get(), codec_, leafOfRow_.get());
    });
    CUDA_CHECK(cudaGetLastError());
    leafJobs_.clear(); // a copy from pageable memory has read it on return
  }

  // The leaves of the rows that ended on the GPU, into `leafOfRow` (which
  // holds those of the CPU tasks).
  void mergeLeaves(decltype(RowLeaves::ids) &leafOfRow) {
    constexpr std::size_t kPiece = 1 << 16; // rows
    downloadChunks(leafOfRow_.get(), stride_ * sizeof(std::uint32_t), "downloadLeaves",
                   [&](std::size_t offset, const char *stage, std::size_t size) {
                     const auto *gpuLeaves = reinterpret_cast<const std::uint32_t *>(stage);
                     std::uint32_t *leaves = leafOfRow.data() + offset / sizeof(std::uint32_t);
                     const std::size_t count = size / sizeof(std::uint32_t);
                     pool_.parallelFor((count + kPiece - 1) / kPiece, [&](std::size_t piece) {
                       const std::size_t end = std::min(count, (piece + 1) * kPiece);
                       for (std::size_t row = piece * kPiece; row < end; ++row) {
                         if (gpuLeaves[row] != UINT32_MAX) {
                           leaves[row] = gpuLeaves[row];
                         }
                       }
                     });
                   });
  }

  // Copy the columns of `children` to the host arena as one block and start
  // a CPU task per child. Only the columns of features that can still split
  // a child are copied.
  //
  // The block goes in pieces of at most a ring part: each is gathered into
  // the next part of the ring (once that part's last copy is done) and
  // copied to the host on the copy streams, which start the CPU tasks
  // themselves once the last piece is copied (startTasks()). The GPU goes on
  // with the next levels meanwhile; it only waits when it gets a whole ring
  // ahead of the copies.
  //
  // The arena is pageable: a piece is downloaded into a page-locked
  // hand-off buffer, and a host function on the copy stream copies it into
  // the arena (drain()). The two copy streams take turns, so one downloads
  // while the other's buffer is copied out. (Buffers of 32 MB, four times
  // as many host functions, left the GPU levels of C4.5 on airline 0.2-0.4 s
  // slower on both GPU nodes: the host functions run one at a time.)
  // Page-locking the arena instead (75 GB for ECBDL'14) held the CUDA
  // driver's lock 1.2-1.9 s, sometimes 4-6 s on an H100 server, and the GPU
  // loop waited for it.
  void handOff(std::vector<LevelNode> &children) {
    NvtxRange range("hand-off copy");
    std::size_t handOffRows = 0;
    for (const LevelNode &child : children) {
      handOffRows += child.count;
    }
    // Block layout: column f of all children back to back, stride handOffRows,
    // cut into pieces of at most a ring part: copy jobs of up to kTile
    // entries, made in block order, relative to their piece.
    struct Piece {
      std::size_t begin, end; // in the block
      std::size_t firstJob, jobCount;
    };
    std::vector<Piece> pieces;
    std::vector<CopyJob> jobs;
    std::vector<std::size_t> next(children.size(), 0); // into each child's features
    for (std::uint32_t f = 0; f < features_; ++f) {
      std::size_t offset = 0;
      for (std::size_t c = 0; c < children.size(); ++c) {
        const LevelNode &child = children[c];
        if (next[c] < child.features.size() && child.features[next[c]] == f) {
          ++next[c];
          const std::size_t from = columnBase(f) + child.begin;
          const std::size_t to = f * handOffRows + offset;
          for (std::size_t at = to; at < to + child.count;) {
            if (pieces.empty() || at >= pieces.back().begin + ringPartEntries_) {
              pieces.push_back({at, at, jobs.size(), 0});
            }
            Piece &piece = pieces.back();
            const std::size_t end = std::min({to + child.count, piece.begin + ringPartEntries_,
                                              at + kTile});
            jobs.push_back({from + (at - to), at - piece.begin, static_cast<std::uint32_t>(end - at)});
            ++piece.jobCount;
            piece.end = end;
            at = end;
          }
        }
        offset += child.count;
      }
    }
    const std::size_t blockEntries = features_ * handOffRows;
    trace::add("handoff_rows", static_cast<double>(handOffRows));
    trace::add("handoff_nodes", static_cast<double>(children.size()));
    trace::add("handoff_bytes", static_cast<double>(blockEntries * sizeof(Entry)));
    trace::add("d2h_bytes", static_cast<double>(blockEntries * sizeof(Entry)));
    trace::add("handoffs", 1);
    waitTouched((arenaUsed_ + blockEntries) * sizeof(Entry));
    Entry *host = arena_.get() + arenaUsed_;
    arenaUsed_ += blockEntries;
    if (!jobs.empty()) {
      upload(copyJobs_, jobs);
    }
    staging().prepare();
    Staging &staging = dt::staging();
    for (const Piece &piece : pieces) {
      const std::size_t part = ringNext_++ % kRingParts;
      Entry *ring = ring_.get() + part * ringPartEntries_;
      CUDA_CHECK(cudaStreamWaitEvent(stream_, ringFree_[part], 0));
      profile_.run("handOffGather", stream_, [&]() {
        copyKernel<<<static_cast<unsigned>(piece.jobCount), kThreads, 0, stream_>>>(
            columns_.get(), ring, copyJobs_.get() + piece.firstJob);
      });
      CUDA_CHECK(cudaGetLastError());
      // Buffer b is only used on stream b: its next download waits there for
      // its copy into the arena.
      const int buffer = static_cast<int>(handOffNext_++ % Staging::kHandOffBuffers);
      cudaStream_t stream = copyStreams_[buffer];
      CUDA_CHECK(cudaEventRecord(gathered_, stream_));
      CUDA_CHECK(cudaStreamWaitEvent(stream, gathered_, 0));
      const std::size_t bytes = (piece.end - piece.begin) * sizeof(Entry);
      char *stage = staging.handOff[buffer];
      profile_.run("handOffCopy", stream, [&]() {
        CUDA_CHECK(cudaMemcpyAsync(stage, ring, bytes, cudaMemcpyDeviceToHost, stream));
      }, "copy");
      CUDA_CHECK(cudaLaunchHostFunc(stream, drain, new Drain{this, host + piece.begin, stage, bytes}));
      CUDA_CHECK(cudaEventRecord(ringFree_[part], stream));
    }
    // Both streams' pieces are in the arena before the tasks start.
    CUDA_CHECK(cudaEventRecord(copied_, copyStreams_[1]));
    CUDA_CHECK(cudaStreamWaitEvent(copyStreams_[0], copied_, 0));
    auto tasks = std::make_unique<PendingTasks>();
    tasks->grower = this;
    tasks->columns = Columns{host, handOffRows};
    std::size_t offset = 0;
    for (LevelNode &child : children) {
      tasks->subtrees.push_back({child.node, static_cast<std::uint32_t>(offset), child.count,
                                 child.depth, std::move(child.features), {}, {}});
      offset += child.count;
    }
    CUDA_CHECK(cudaLaunchHostFunc(copyStreams_[0], startTasks, tasks.release()));
  }

  // One hand-off buffer to copy into the arena (a stream callback: no CUDA
  // calls). The copier threads are idle during the GPU levels.
  struct Drain {
    GpuGrower *grower;
    void *target;
    const char *stage;
    std::size_t bytes;
  };

  static void CUDART_CB drain(void *data) {
    std::unique_ptr<Drain> drain(static_cast<Drain *>(data));
    constexpr std::size_t kMinPart = std::size_t{1} << 20;
    trace::Doing copying(trace::Work::Staging);
    ThreadPool &copiers = staging().copyPool(drain->grower->pool_);
    const std::size_t parts = std::max<std::size_t>(
        1, std::min({Staging::kCopyThreads, std::size_t{copiers.workerCount()} + 1,
                     drain->bytes / kMinPart}));
    char *to = static_cast<char *>(drain->target);
    copiers.parallelFor(parts, [&](std::size_t part) {
      const std::size_t begin = drain->bytes * part / parts;
      const std::size_t end = drain->bytes * (part + 1) / parts;
      std::memcpy(to + begin, drain->stage + begin, end - begin);
    });
  }

  // The CPU tasks of one hand-off (a stream callback: no CUDA calls).
  struct PendingTasks {
    GpuGrower *grower;
    Columns columns;
    std::vector<Subtree> subtrees;
  };

  static void CUDART_CB startTasks(void *data) {
    std::unique_ptr<PendingTasks> tasks(static_cast<PendingTasks *>(data));
    GpuGrower *grower = tasks->grower;
    for (Subtree &subtree : tasks->subtrees) {
      grower->pool_.submit(
          [grower, columns = tasks->columns, subtree = std::move(subtree)]() mutable {
            NvtxRange range("cpu subtree");
            grower->cpuBuilder_->grow(columns, std::move(subtree));
          });
    }
  }

  const Dataset &train_;
  const Options &options_;
  ThreadPool &pool_;
  bool grown_ = false;
  DistinctValues overlapDistinct_;   // ... with these distinct values (C4.5)
  DeviceArray<std::uint32_t> rootCounts_;
  std::vector<std::uint32_t> hostRootCounts_;
  int smCount_ = 1;
  std::size_t rows_;
  std::size_t features_;
  int classCount_;
  EntryCodec codec_;
  ScanParams params_{};

  // Per grow.
  std::size_t stride_ = 0; // training rows (the column stride)
  std::size_t minRows_ = 0; // smaller nodes go to the CPU (gpu.minRows, or auto)
  const SplitRules *rules_ = nullptr;
  NodeStore *store_ = nullptr;
  CpuTreeBuilder *cpuBuilder_ = nullptr;

  std::unique_ptr<std::uint8_t[]> hostGoesLeft_;
  std::unique_ptr<Entry[]> arena_;
  std::size_t arenaUsed_ = 0;
  std::size_t handOffNext_ = 0; // the next of Staging::handOff
  std::thread toucher_; // touchArena()
  std::mutex touchMutex_;
  std::condition_variable touchCv_;
  std::size_t touchedBytes_ = 0; // the first bytes of the arena are touched (touchMutex_)

  cudaStream_t stream_ = nullptr;
  cudaStream_t copyStreams_[2] = {}; // hand-off copies to the host (handOff())
  cudaEvent_t gathered_ = nullptr;    // the last hand-off piece is gathered
  cudaEvent_t copied_ = nullptr;      // copyStreams_[1] is done with a hand-off

  // The sorted columns, one per slot of slotEntries_ entries; slot_[f] holds
  // feature f (see PartitionTarget).
  // One is enough: a column then waits for the one before it (3-5 ms of 500
  // on airline); every further slot costs 8 bytes per row.
  static constexpr std::size_t kSpareSlots = 1;
  DeviceArray<Entry> columns_;
  std::size_t slotCount_ = 0;
  std::vector<std::uint32_t> slot_;
  Entry *column(std::size_t feature) const { return columns_.get() + columnBase(feature); }
  std::size_t columnBase(std::size_t feature) const {
    return std::size_t{slot_[feature]} * slotEntries_;
  }
  static constexpr std::size_t kSlotAlign = (std::size_t{2} << 20) / sizeof(Entry); // 2 MiB
  std::size_t slotEntries_ = 0; // rows_, rounded up to kSlotAlign
  // Hand-off staging (see handOff()).
  static constexpr std::size_t kRingBytes = std::size_t{512} << 20;
  static constexpr std::size_t kRingParts = 4;
  static_assert(kRingBytes / kRingParts <= Staging::kHandOffBytes);
  DeviceArray<Entry> ring_;
  std::size_t ringPartEntries_ = 0;
  std::size_t ringNext_ = 0;
  cudaEvent_t ringFree_[kRingParts] = {}; // the last copy out of each part is done
  DeviceArray<float> rawValues_; // C4.5 with pruning: for arrangeRows()
  DeviceArray<std::uint16_t> labels_;
  DeviceArray<unsigned char> sortTemp_;
  std::size_t sortTempBytes_ = 0;
  DeviceArray<int> distinctCounts_; // distinctValues(): per feature
  DeviceArray<unsigned long long> distinctPlaces_; // ... where in the pool, then the pool's use
  DeviceArray<std::uint32_t> goesLeft_; // a bit per row: goes left (fits in L2)
  DeviceArray<Segment> segments_;
  DeviceArray<std::uint32_t> tileSegment_;
  DeviceArray<std::uint8_t> constant_;
  DeviceArray<std::uint32_t> tileHist_;
  DeviceArray<std::uint32_t> nodeCounts_; // class counts of the level's nodes
  DeviceArray<NodeDesc> nodes_;
  DeviceArray<BestCut> tileBest_;
  DeviceArray<double> tileUpper_;    // two classes: tileBoundKernel
  DeviceArray<double> tileLower_;
  DeviceArray<double> segmentLower_;
  DeviceArray<std::uint32_t> tileTries_;
  DeviceArray<TileState> tileStates_; // partition: per tile, then the tile counter
  DeviceArray<PartitionTarget> partitionTargets_;
  DeviceArray<PartitionMove> partitionMoves_;
  bool histogramReady_ = false; // tileHist_ / tileTries_ hold the level's step 1
  DeviceArray<CutCandidate> segmentBest_;
  DeviceArray<std::uint32_t> segmentLeft_;
  DeviceArray<CopyJob> copyJobs_;
  DeviceArray<LeafJob> leafJobsDevice_;
  DeviceArray<std::uint32_t> leafOfRow_; // leaves of rows made on the GPU
  std::vector<LeafJob> leafJobs_;
  bool recordLeaves_ = false;
  DeviceArray<double> logTable_;
  KernelProfile profile_;
};

} // namespace

void gpuPrepare(unsigned threads) {
  CUDA_CHECK(cudaFree(nullptr));
  staging().prepare(threads);
}

std::unique_ptr<Grower> makeGpuGrower(const Dataset &train, const Options &options,
                                      ThreadPool &pool, double &setupSeconds) {
  return std::make_unique<GpuGrower>(train, options, pool, setupSeconds);
}

} // namespace dt
