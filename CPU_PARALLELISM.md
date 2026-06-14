# CPU parallelism

How CPU threading is used to build decision trees. There are three backends:

| Backend | File | CPU parallelism |
|---------|------|-----------------|
| `TreeSerial` | `tree_serial.cpp` | None — one thread, serial recursion |
| `TreeParallel` | `tree_parallel.cpp` | Node + feature pools |
| `TreeCuda` | `tree_cuda.cpp` | Node pool + CPU split path for small nodes |

Shared split math lives in `tree_base.cpp` (`evaluateFeatureSplit`,
`reduceBestSplitSearch`, `scoreAllThresholdsForFeature`, etc.). All three
backends call the same functions for CPU-side split search, so trees match when
every node uses the CPU path.

See also: [`CUDA_PARALLELISM.md`](CUDA_PARALLELISM.md) for the GPU side.

---

## `TreeParallel` — two thread pools

Files: `tree_parallel.cpp`, `tree_parallel.h`, `task_executor.cpp`,
`task_executor.h`.

`TreeParallel` parallelizes on two levels:

1. **Across nodes** — build sibling subtrees on different threads.
2. **Inside one node** — score features in parallel.

### The thread pool (`TaskExecutor`)

A lightweight pool: `submit()` a function, get back a `std::future`.

There are **two separate pools**:

- `nodeExecutor` — builds child subtrees.
- `featureExecutor` — scores features within one node.

Why two pools? A node-building thread calls into feature search and waits for
those results. One shared queue could deadlock (every thread waiting, none free
to run feature jobs). Separate pools avoid that.

---

### Node parallelism (across the tree)

Main walk: `buildNodeParallel(rows, depth)`:

1. `expandOneNode` → leaf, or decision node + row partitions.
2. Leaf → return.
3. If `rows < minRowsToParallelize` or no task slot → build **both** children
   on the current thread.
4. Else → **submit left child to `nodeExecutor`**, build **right child** on the
   current thread, `leftJob.get()`, attach both.

Example: at the root with 1M rows, the left subtree runs on a pool thread while
the current thread builds the right subtree.

**Deadlock avoidance:** at most `maxNodeThreadCount - 1` node tasks run at once
(`tryStartNodeTask` / `finishNodeTask`), leaving one worker for a thread blocked
on `future.get()`.

---

### Feature parallelism (inside one node)

In `findBestSplitAtNode`:

1. Share the node's row list once (`shared_ptr`).
2. Submit each feature to `featureExecutor` → `evaluateFeatureSplit`.
3. Collect results, `reduceBestSplitSearch` picks the best feature.

Only when `featureCount >= minFeaturesToParallelize`; otherwise a simple loop.

Example: 18 features, 8 feature threads → ~8 features scored at once.

---

### Configuration (`TreeParallel`)

| Option | Meaning |
|--------|---------|
| `maxNodeThreadCount` | Threads in `nodeExecutor` (parallel subtrees). |
| `maxFeatureThreadCount` | Threads in `featureExecutor` (parallel features). |
| `minRowsToParallelize` | Min rows in a node to split across threads. |
| `minFeaturesToParallelize` | Min features to parallelize feature search. |

Defaults: 4 / 4 / 32 / 4 (`options.h`).

---

### Diagram (`TreeParallel`)

```
fit()
  setupParallelExecutors()   -> featureExecutor, nodeExecutor, maxNodeTasks
  buildNodeParallel(root)
      expandOneNode
          findBestSplitAtNode -> featureExecutor: evaluateFeatureSplit x F
          partition rows
      [big node + free slot]
          nodeExecutor: buildNodeParallel(left)
          current thread:  buildNodeParallel(right)
          leftJob.get()
      [small node or no slot]
          buildNodeParallel(left), buildNodeParallel(right)  -- same thread
  fitContext_.reset()
  finalizeFit() -> prune (single thread)
```

---

## `TreeCuda` — CPU split path for small nodes

`TreeCuda` reuses the **same CPU split functions** as `TreeSerial` for nodes
with `rows < cudaMinRowsForGpu` (default 2048):

```cpp
// tree_cuda.cpp — CPU branch of findBestSplitAtNode
for each feature:
    evaluateFeatureSplit(rowIndices, featureIndex)
return reduceBestSplitSearch(results, rowIndices);
```

No GPU worker, no stream, no VRAM scratch. This is the exact CART/C4.5 math from
`TreeBase`.

### Tree-walk pool (separate from GPU workers)

`TreeCuda` also uses `buildNodeParallel` with the same left-pool / right-thread
pattern as `TreeParallel`. The pool size is `max(cudaGpuWorkerCount,
hardware_concurrency())` — all CPU cores — so hundreds of thousands of small
subtrees build in parallel.

`cudaGpuWorkerCount` controls **GPU workers only** (see
`CUDA_PARALLELISM.md`). It does not limit how many CPU threads walk the tree.

### When trees match across backends

| `TreeCuda` setting | Same tree as serial/parallel? |
|--------------------|-------------------------------|
| `cudaMinRowsForGpu > dataset size` | **Yes** — every node on CPU |
| Default hybrid (2048) | **No** — large nodes use GPU splits |
| `cudaMinRowsForGpu = 0` | **No** — all nodes on GPU |

---

## `TreeSerial` — no parallelism

One thread, depth-first `buildNode`. Calls the same `evaluateFeatureSplit` /
`reduceBestSplitSearch` as the CPU paths above. Used as the correctness
reference.

---

## Building

CPU only (no CUDA):

```bash
g++ -std=c++20 -O2 -pthread -I. \
  main.cpp tree_base.cpp tree_parallel.cpp tree_serial.cpp \
  task_executor.cpp dataset.cpp node.cpp options.cpp pruning/pruning.cpp \
  -o tree_cpu
```

Full build with `TreeCuda`:

```bash
./build_cuda.sh
```
