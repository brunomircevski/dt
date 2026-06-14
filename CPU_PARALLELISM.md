# CPU parallelism

How the CPU builds decision trees. There are two backends:

| Backend | File | CPU parallelism |
|---------|------|-----------------|
| `TreeSerial` | `tree_serial.cpp` | None — one thread, serial recursion |
| `TreeParallel` | `tree_parallel.cpp` | Node + feature pools |

Split math lives in `tree_base.cpp` (`evaluateFeatureSplit`,
`reduceBestSplitSearch`, `scoreAllThresholdsForFeature`, etc.). Both backends
call the same functions.

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

**Deadlock avoidance:** at most `parallelMaxNodeThreadCount - 1` node tasks run at once
(`tryStartNodeTask` / `finishNodeTask`), leaving one worker for a thread blocked
on `future.get()`.

---

### Feature parallelism (inside one node)

In `findBestSplitAtNode`:

1. Share the node's row list once (`shared_ptr`).
2. Submit each feature to `featureExecutor` → `evaluateFeatureSplit`.
3. Collect results, `reduceBestSplitSearch` picks the best feature.

Only when `featureCount >= parallelMinFeaturesToParallelize`; otherwise a simple loop.

Example: 18 features, 8 feature threads → ~8 features scored at once.

---

### Configuration (`TreeParallel`)

| Option | Meaning |
|--------|---------|
| `parallelMaxNodeThreadCount` | Threads in `nodeExecutor` (parallel subtrees). |
| `parallelMaxFeatureThreadCount` | Threads in `featureExecutor` (parallel features). |
| `minRowsToParallelize` | Min rows in a node to split across threads (also used by TreeCuda). |
| `parallelMinFeaturesToParallelize` | Min features to parallelize feature search. |

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

## `TreeSerial` — no parallelism

One thread, depth-first `buildNode`. Calls the same `evaluateFeatureSplit` /
`reduceBestSplitSearch` as `TreeParallel`. Used as the correctness reference.

---

## Building

```bash
g++ -std=c++20 -O2 -pthread -I. \
  main.cpp tree_base.cpp tree_parallel.cpp tree_serial.cpp \
  task_executor.cpp dataset.cpp node.cpp options.cpp pruning/pruning.cpp \
  -o tree_cpu
```
