# CUDA node parallelism — design notes

Summary of ideas discussed for adding **node-level parallelism** to `TreeCuda`.
Current code: **one CPU thread** drives tree recursion; **feature parallelism** lives
inside GPU kernels. Node parallelism would overlap CPU bookkeeping with GPU split
search across branches.

---

## Current state

| Level | Parallelism |
|-------|-------------|
| Features (attributes) | Yes — one block per feature (or tiled grid on large nodes) |
| Tree nodes | No — serial `buildNode` recursion on CPU |

Scratch GPU memory is allocated **once at root size** (`featureCount × totalRows`)
and reused per node. Smaller nodes use only a prefix; the rest is unused. Fine with
one thread; does not scale to multiple concurrent nodes without separate buffers.

**Supersymmetry baseline (5M rows, 18 features, maxDepth 7):** ~1.4 s build, ~2.4 GiB VRAM.

---

## Goal

- **T CPU worker threads** (e.g. 2–4, option `cudaNodeThreadCount`)
- Each worker owns **fixed scratch + one `cudaStream_t`**
- Tree walk mirrors `TreeParallel`: submit left subtree to pool, build right on current thread
- **Kernels unchanged** — still take `nodeRows`; only orchestration and memory change

---

## Recommended approach: fixed scratch per thread (geometric caps)

At `fit()`, create **T workers**. Worker `i` gets scratch sized for **`N / 2^i` rows**
(full dataset row count = `N`), plus one CUDA stream.

```text
Thread 0 → capacity N,     stream 0, scratch 0
Thread 1 → capacity N/2,   stream 1, scratch 1
Thread 2 → capacity N/4,   stream 2, scratch 2
Thread 3 → capacity N/8,   stream 3, scratch 3
```

**Job routing:** assign each `buildNode` job to the **smallest idle worker** with
`worker.maxRows >= nodeRows`. If none available, queue until one frees.

**Oversized slots OK:** a 100k-row node may use a worker with N/2 capacity if smaller
workers are busy.

**Release timing:** run GPU split → CPU partition → **release worker for next job**
before waiting on children (maximizes overlap).

Shared read-only GPU data (`d_features`, `d_classIds`) stays **single copy**.

### Why this over alternatives

| Approach | Verdict |
|----------|---------|
| `cudaMalloc` per node | Too slow (thousands of nodes) |
| 4× root scratch (naive) | ~8.5 GiB — does not fit 8 GiB GPU |
| Shared scratch pool (checkout/return) | Same VRAM as fixed-per-thread; slightly more flexible, more code |
| **Fixed scratch per thread (geometric)** | **Best balance** — predictable VRAM, simple ownership, no pool mutex |

Pool vs fixed-per-thread: same total VRAM; fixed-per-thread is simpler (buffers live
in the worker, no acquire/release). Route jobs to fitting idle workers; larger workers
can take smaller nodes when needed.

---

## VRAM estimates (supersymmetry, default options)

Shared once: **~353 MiB** (`d_features`, `d_classIds`, small tile buffers).

Per-slot scratch ≈ linear in row capacity (values, rowIds, sorted buffers, CUB sort temp).

| Threads (T) | Worker capacities | Total VRAM | Fits 8 GiB? |
|-------------|-------------------|------------|-------------|
| 1 (current) | N | **~2.4 GiB** | Yes |
| 2 | N, N/2 | **~3.4 GiB** | Yes |
| 4 | N, N/2, N/4, N/8 | **~4.2 GiB** | Yes |
| 8 | N … N/128 | **~4.4 GiB** | Yes (diminishing slot sizes) |

Geometric series converges to **~2× root scratch** (+ shared), not `T × root`.

Per-slot examples: 5M rows ~2.1 GiB; 2.5M ~1.0 GiB; 1.25M ~0.5 GiB; 625k ~0.26 GiB.

---

## CUDA streams

One stream per worker. Operations on worker `i`'s scratch all use `stream_i`.

- Same stream: ordered (gather → sort → score)
- Different streams: can overlap on one GPU
- Sync with `cudaStreamSynchronize(stream)` before CPU reads results

Avoids serializing concurrent launches on the default stream.

---

## Scheduling (high level)

```text
fit():
  upload shared data
  create T workers (scratch + stream each)
  start thread pool

buildNode(rows, depth):
  assign to idle worker with cap >= rows
  GPU: findBestSplitAtNode (on worker scratch/stream)
  CPU: pick split, partition rows
  if leaf → return
  submit left child to pool
  build right child (or wait per TreeParallel pattern)
  wait for left, attach children
```

**Early tree:** only worker 0 fits root → effectively serial (expected).

**Deeper tree:** many small nodes → several workers busy; GPU pipelines overlap via streams.

**Limitation:** siblings at the same depth often need the same tier; only one worker
per tier exists — borrow larger worker when smaller is busy.

---

## Expected speedup (supersymmetry, one GPU)

Baseline: **~1.42 s**. Theoretical floor (perfect overlap): **~3.2×** (~440 ms).

| CPU threads | Realistic speedup | Est. build time |
|-------------|-------------------|-----------------|
| 1 | 1.0× | ~1.42 s |
| 2 | **1.4–1.7×** | ~0.85–1.0 s |
| 4 | **1.7–2.0×** | ~0.70–0.85 s |
| 8 | **1.8–2.3×** | ~0.62–0.80 s |

Not linear: root and early levels are serial; one GPU saturates around 2–3 concurrent
pipelines. **T=4** is the sweet spot for 8 GiB VRAM; **T=8** adds little over T=4.

---

## Options to add (sketch)

```cpp
int cudaNodeThreadCount = 4;  // TreeCuda only
```

Reuses the spirit of `maxNodeThreadCount` from `TreeParallel` but with GPU worker
routing instead of generic `TaskExecutor` alone.

---

## Implementation checklist

1. Extract per-worker `ScratchBuffers` + `cudaStream_t` from `CudaState`
2. Allocate worker `i` for `N / 2^i` rows at `fit()`
3. Thread pool + job queue with capacity-based routing
4. Pass worker scratch/stream into `findBestSplitAtNode`
5. Use `cudaMemcpyAsync` + stream sync where applicable
6. Parallel `buildNode` (left pool / right current), same as `TreeParallel`

Kernels and split math: **no changes**.
