# CUDA parallelism in `TreeCuda`

How the GPU backend builds a decision tree. Three mechanisms work together:

1. **Hybrid routing** — large nodes use the GPU; small nodes use the exact CPU
   split path (same math as `TreeSerial`).
2. **Inside a GPU node** — every feature and threshold is tested in parallel on
   the device.
3. **Across nodes** — a CPU thread pool walks the tree; up to T GPU workers
   can run concurrent split searches on separate CUDA streams.

Files: `tree_cuda.cpp`, `tree_cuda.h`.

---

## The big picture

The CPU walks the tree top-down. At each node it asks: *"which feature and
threshold splits these rows best?"*

```
                    rows >= cudaMinRowsForGpu?
                           |
              yes          |          no
               v           |           v
         GPU worker    (most nodes)   CPU exact split
    gather/sort/score              evaluateFeatureSplit x F
                                   reduceBestSplitSearch
```

On deep trees most nodes are small (hundreds of rows or less). Sending every
node through the GPU costs ~16 µs of fixed overhead per node (copy, sort launch,
sync) even when the node has only a few rows. The hybrid path avoids that:
**~860k tiny nodes on a depth-30 supersymmetry tree stay on the CPU.**

---

## Step 1: upload the data once

At the start of `fit()` the whole dataset is copied to the GPU a single time:

- `d_features` — every feature value of every row (feature-major layout).
- `d_classIds` — the class (label) of every row, as a small integer.

These are **read-only** and shared by all GPU workers. For 5M rows / 18 features
this is ~350 MiB; we pay the copy cost only once. CPU-only nodes never touch
these buffers during split search (they read `dataset_->samples` on the host).

---

## Step 2: split search — CPU path (small nodes)

When `nodeRows < cudaMinRowsForGpu` (default **2048**), `findBestSplitAtNode` runs
entirely on the CPU — the same code path as `TreeSerial`:

1. For each feature: `evaluateFeatureSplit` (sort rows by feature, sweep
   thresholds).
2. `reduceBestSplitSearch` picks the best feature overall.

No GPU worker is checked out; no mutex, stream, or VRAM scratch is used. This
path is **byte-identical** to `TreeSerial` / `TreeParallel` for that node.

Set `cudaMinRowsForGpu` above the dataset size (e.g. `--cuda-min-gpu-rows
10000000`) to force all nodes onto CPU and match the other backends exactly.
Set `cudaMinRowsForGpu = 0` to send every node to the GPU (old behavior).

---

## Step 3: split search — GPU path (large nodes)

When `nodeRows >= cudaMinRowsForGpu`, a GPU worker is checked out and the node
runs on that worker's stream:

1. **gather** — copy this node's `(value, rowId)` pairs into scratch memory
   (row indices staged via pinned host memory for true async H2D).
2. **sort** — one CUB segmented radix sort per feature slice.
3. **score** — every feature (and, for big nodes, every tile of a feature) is
   scanned in parallel.
4. **copy back** — per-feature winners and the winning feature's sorted row ids
   are copied to pinned host memory; the CPU picks the overall best split and
   partitions rows.

Small GPU nodes use **one block per feature**. Large nodes split each feature
into **tiles** so more SMs stay busy.

---

## Step 4: GPU workers (`cudaGpuWorkerCount`)

`cudaGpuWorkerCount` (default **4**, clamped 1–32) sets **T = number of GPU
workers**, not the size of the tree-walk thread pool.

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
it does not affect small nodes (CPU path).

---

## Step 5: walking the tree in parallel

`buildNodeParallel` mirrors `TreeParallel`:

1. Route node to CPU or GPU (see above), run `expandOneNode`, release GPU
   worker if any.
2. Leaf → return.
3. If `nodeRows < minRowsToParallelize` or no task slot → recurse both
   children on the current thread.
4. Else → **submit left child to the thread pool**, build **right child** on the
   current thread, `leftJob.get()`, attach children.

### Two different thread counts

| What | Size | Controlled by |
|------|------|---------------|
| GPU workers (scratch + streams) | T | `cudaGpuWorkerCount` |
| Tree-walk thread pool | `max(T, hardware_concurrency())` | automatic |

The pool uses all CPU cores so hundreds of thousands of small CPU subtrees build
in parallel. Only nodes on the GPU path compete for the T workers.

Deadlock avoidance: at most `poolThreads - 1` node tasks run concurrently
(`tryStartNodeTask` / `finishNodeTask`).

```
fit()
  upload shared data (once)
  create T GPU workers (scratch + stream + pinned staging)
  start thread pool (max(T, CPU cores))
  buildNodeParallel(root)
      if rows >= cudaMinRowsForGpu:
          check out smallest fitting GPU worker
          GPU: gather -> sort -> score
          release worker
      else:
          CPU: evaluateFeatureSplit x F -> reduceBestSplitSearch
      partition rows
      submit LEFT to pool / build RIGHT here / wait
  tear down pool, free workers + shared data
```

---

## Correctness across backends

| Mode | Matches serial/parallel? |
|------|---------------------------|
| All CPU (`cudaMinRowsForGpu > N`) | **Yes** — identical tree |
| Hybrid (default 2048) | **No** — large nodes use GPU splits, which can differ slightly from CPU |
| All GPU (`cudaMinRowsForGpu = 0`) | **No** — every node on GPU |

The CPU branch reuses `TreeBase::evaluateFeatureSplit` and
`reduceBestSplitSearch` — the same functions `TreeSerial` and `TreeParallel`
call. For strict cross-backend equality, set `--cuda-min-gpu-rows` above your
dataset row count.

Within the CUDA backend, scheduling (T=1 vs T=4, hybrid vs all-GPU) does not
change the tree as long as the same nodes take the same path (CPU vs GPU).

---

## Configuration

| Option | Meaning |
|--------|---------|
| `cudaMinRowsForGpu` | Nodes with fewer rows use the CPU split path. 0 = always GPU. Default 2048. |
| `cudaGpuWorkerCount` | Number of GPU workers / streams. Set in `main.cpp`. Default 4. Clamped 1–32. |
| `minRowsToParallelize` | Node needs at least this many rows to split across pool threads. |
| `cudaRowsPerTile` | Rows per tile for the large-node GPU scan. |
| `cudaMaxTilesPerFeature` | Cap on tiles per feature (also sizes GPU buffers). |
| `cudaScoreThreadsPerBlock` | Threads per block in the scoring kernels. |
| `cudaGatherBlockSize` | Threads per block in the gather kernel. |

CLI: `--cuda-min-gpu-rows N` sets the CPU/GPU threshold (0 = all GPU).
`cudaGpuWorkerCount` is configured in `main.cpp` only, not on the command line.

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

With hybrid routing, GPU workers are idle most of the time on deep trees; peak
VRAM is similar to all-GPU because workers are still allocated at `fit()`.

---

## Performance notes (supersymmetry, 5M rows)

| Config | depth 30 build time | Notes |
|--------|---------------------|-------|
| All GPU (`cudaMinRowsForGpu=0`) | ~15–47 s | Every node pays GPU round-trip |
| Hybrid (default 2048) | **~6.5 s** | ~860k nodes on CPU, ~few k on GPU |
| All CPU (`cudaMinRowsForGpu > N`) | ~40+ s | Matches serial tree; no GPU benefit |

The main speedup comes from **`cudaMinRowsForGpu`**, not from raising
`cudaGpuWorkerCount`. Extra GPU streams help only when many large nodes are
ready at once; on supersymmetry the upper tree already saturates the GPU per
node.

---

## Building

```bash
./build_cuda.sh
./tree --cuda -d 30 datasets/supersymmetry.csv
./tree --cuda --cuda-min-gpu-rows 2048 datasets/supersymmetry.csv
```
