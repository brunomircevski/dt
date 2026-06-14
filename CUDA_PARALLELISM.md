# CUDA parallelism

How the GPU backend (`TreeCuda`) builds a decision tree. Three pieces work
together:

1. **Hybrid routing** — large nodes use the GPU; small nodes use a CPU split
   path inside the same backend.
2. **Inside a GPU node** — every feature and threshold is tested in parallel on
   the device.
3. **Across nodes** — `cudaCpuThreadCount` CPU threads walk the tree; up to
   `cudaGpuWorkerCount` GPU workers run split searches on separate streams.

Files: `tree_cuda.cpp`, `tree_cuda.h`.

---

## The big picture

The tree is built top-down. At each node the backend asks: *which feature and
threshold splits these rows best?*

```
                    rows >= cudaMinRowsForGpu?
                           |
              yes          |          no
               v           |           v
         GPU worker    (most nodes)   CPU split path
    gather/sort/score              evaluateFeatureSplit x F
                                   reduceBestSplitSearch
```

On deep trees most nodes are small. Sending every node through the GPU costs
fixed overhead per node (copy, sort launch, sync) even when the node has only a
few rows. The hybrid path avoids that: tiny nodes stay on the CPU split path
inside `TreeCuda`.

---

## Step 1: upload the data once

At the start of `fit()` the whole dataset is copied to the GPU a single time:

- `d_features` — every feature value of every row (feature-major layout).
- `d_classIds` — the class (label) of every row, as a small integer.

These are **read-only** and shared by all GPU workers. For 5M rows / 18 features
this is ~350 MiB; the copy cost is paid only once. Nodes on the CPU split path
read `dataset_->samples` on the host instead.

---

## Step 2: split search — CPU path (small nodes)

When `nodeRows < cudaMinRowsForGpu` (default **2048**), `findBestSplitAtNode`
runs on the CPU:

1. For each feature: `evaluateFeatureSplit` (sort rows by feature, sweep
   thresholds).
2. `reduceBestSplitSearch` picks the best feature overall.

No GPU worker is checked out; no stream or VRAM scratch is used.

Set `cudaMinRowsForGpu = 0` to send every node to the GPU.

---

## Step 3: split search — GPU path (large nodes)

When `nodeRows >= cudaMinRowsForGpu`, a GPU worker is checked out and the node
runs on that worker's stream:

1. **gather** — copy this node's `(value, rowId)` pairs into scratch memory
   (row indices staged via pinned host memory for async H2D).
2. **sort** — one CUB segmented radix sort per feature slice.
3. **score** — every feature (and, for big nodes, every tile of a feature) is
   scanned in parallel.
4. **copy back** — per-feature winners and the winning feature's sorted row ids
   are copied to pinned host memory; the host picks the overall best split and
   partitions rows.

Small GPU nodes use **one block per feature**. Large nodes split each feature
into **tiles** so more SMs stay busy.

---

## Step 4: GPU workers (`cudaGpuWorkerCount`)

`cudaGpuWorkerCount` (default **4**, clamped 1–32) sets **T = number of GPU
workers**. CPU thread count is `cudaCpuThreadCount` (separate option).

Each worker `i` owns:

- scratch buffers sized for `max(1, N / 2^i)` rows (geometric capacities),
- per-feature output buffers,
- one non-blocking `cudaStream_t`,
- pinned host staging (`h_pinned`) for async copies.

```
Worker 0 → capacity N        (fits any node, including root)
Worker 1 → capacity N/2
Worker 2 → capacity N/4
Worker 3 → capacity N/8
```

Capacities sum to ~**2× root scratch** (not T×), so T=4 fits comfortably in
8 GiB VRAM.

### Picking a worker

For GPU nodes only: check out the **smallest idle worker** with
`maxRows >= nodeRows`. Worker 0 always fits, so a thread never deadlocks waiting
for a slot. The worker is **released immediately after** split + partition,
**before** waiting on children.

At most **T concurrent GPU split searches** can run. Raising
`cudaGpuWorkerCount` adds VRAM and allows more overlapping large-node GPU work;
it does not affect small nodes (CPU split path).

---

## Step 5: walking the tree in parallel

`buildNodeParallel` walks the tree with the same left-pool / right-thread pattern
as a typical parallel tree build:

1. Route node to CPU or GPU split path (see above), run `expandOneNode`, release
   GPU worker if any.
2. Leaf → return.
3. If `nodeRows < minRowsToParallelize` or no task slot → recurse both
   children on the current thread.
4. Else → **submit left child to the thread pool**, build **right child** on the
   current thread, `leftJob.get()`, attach children.

### Two different thread counts

| What | Size | Option |
|------|------|--------|
| CPU tree-walk thread pool | C | `cudaCpuThreadCount` |
| GPU workers (scratch + streams) | T | `cudaGpuWorkerCount` |

`cudaCpuThreadCount` must be >= `cudaGpuWorkerCount`.

The CPU pool builds many small subtrees in parallel. Only nodes on the GPU
path compete for the T workers.

Deadlock avoidance: at most `poolThreads - 1` node tasks run concurrently
(`tryStartNodeTask` / `finishNodeTask`).

```
fit()
  upload shared data (once)
  create T GPU workers (scratch + stream + pinned staging)
  start CPU thread pool (cudaCpuThreadCount)
  buildNodeParallel(root)
      if rows >= cudaMinRowsForGpu:
          check out smallest fitting GPU worker
          GPU: gather -> sort -> score
          release worker
      else:
          CPU split path: evaluateFeatureSplit x F -> reduceBestSplitSearch
      partition rows
      submit LEFT to pool / build RIGHT here / wait
  tear down pool, free workers + shared data
```

---

## Configuration

| Option | Meaning |
|--------|---------|
| `cudaCpuThreadCount` | CPU threads that walk the tree and build subtrees. Default 4. |
| `cudaGpuWorkerCount` | GPU workers (scratch + stream each). Default 4. Clamped 1–32. Must be <= `cudaCpuThreadCount`. |
| `cudaMinRowsForGpu` | Nodes with fewer rows use the CPU split path. 0 = always GPU. Default 2048. |
| `minRowsToParallelize` | Node needs at least this many rows to split across CPU pool threads. |
| `cudaRowsPerTile` | Rows per tile for the large-node GPU scan. |
| `cudaMaxTilesPerFeature` | Cap on tiles per feature (also sizes GPU buffers). |
| `cudaScoreThreadsPerBlock` | Threads per block in the scoring kernels. |
| `cudaGatherBlockSize` | Threads per block in the gather kernel. |

Set `cudaMinRowsForGpu` and `cudaGpuWorkerCount` in `main.cpp` (or wherever
`Options` is configured).

---

## VRAM

- Shared once: `featureCount * N * 4 + N * 2` bytes (+ small buffers).
- Per GPU worker: scratch proportional to its geometric capacity + pinned staging.
- Total ≈ shared + **~2× root scratch** for T workers.

| T | Worker capacities | Rough total (supersymmetry) |
|---|-------------------|------------------------------|
| 1 | N | ~2.4 GiB |
| 2 | N, N/2 | ~3.4 GiB |
| 4 | N, N/2, N/4, N/8 | ~4.2 GiB |

With hybrid routing, GPU workers are idle much of the time on deep trees; peak
VRAM is similar to all-GPU because workers are still allocated at `fit()`.

---

## Performance notes (supersymmetry, 5M rows)

| Config | depth 30 build time | Notes |
|--------|---------------------|-------|
| All GPU (`cudaMinRowsForGpu=0`) | ~15–47 s | Every node pays GPU round-trip |
| Hybrid (default 2048) | **~6.5 s** | ~860k nodes on CPU path, ~few k on GPU |
| All CPU path (`cudaMinRowsForGpu > N`) | ~40+ s | No GPU benefit |

The main speedup comes from **`cudaMinRowsForGpu`**, not from raising
`cudaGpuWorkerCount`. Extra GPU streams help only when many large nodes are
ready at once; on supersymmetry the upper tree already saturates the GPU per
node.

---

## Building

```bash
./build_cuda.sh
./tree --cuda -d 30 datasets/supersymmetry.csv
```
