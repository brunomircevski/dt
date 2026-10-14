# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS

**Status:** run on 2026-10-14, 3 runs per point, at commit a83b6ec, with ./tree's defaults (the fastest settings found on these servers; [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)).
- **epyc1:** 19:56–20:05.
- **dell1:** 19:57–20:11.
- **Each node ran alone** and was used whole.
- **Same trees:** every thread count built the same tree on both nodes, the same as in [cluster-higgs](../cluster-higgs/BENCHMARK.md).

The cluster's counterpart of the desktop's and laptop's scaling runs ([cpu-pc-susy-scaling](../cpu-pc-susy-scaling/BENCHMARK.md)), on HIGGS and on the two CPU servers only. The GPU servers are not benchmarked on their CPUs.

```bash
bench/cluster/scaling.sh -m 3      # on the login node: one job on epyc1, one on dell1
```

- **Protocols:**
  - `cart_alpha_nodepth`: pruned CART at the fixed α = 1e-5, no depth limit.
  - `c45`: C4.5, CF 0.25, min 2 rows.
- **Threads:** 1, 2, 4, then every 4 up to 48.
  - **1 thread** is the serial backend (`--serial`); more threads use `--parallel --threads N`. Speedup is the serial time divided by the N-thread time, so it includes the parallel backend's own overhead.
  - **Pinning:** N threads are pinned to the first N CPUs in the kernel's order (`--cpus 0-47`). `run.log` records that order (`lscpu`).
- **Measurements:** as in every benchmark here: training time timed inside ./tree (presort, build, pruning; no file reading), and peak RSS up to the end of training, from the same process. Accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).

| Node | CPU | RAM | CPU order (`lscpu`) | Governor |
|---|---|---|---|---|
| epyc1 | 2× EPYC 9255 (Zen 5), 24 cores each, no SMT | 512 GB DDR5-6400 | CPUs 0–23 on socket 0, 24–47 on socket 1 (one NUMA node each) | `performance` |
| dell1 | 2× Xeon Gold 6542Y (Emerald Rapids), 24 cores each, no SMT | 1 TB DDR5-5200 | CPUs alternate between the sockets and their 4 NUMA nodes (0, 2, 1, 3, 0, …) | `powersave` (intel_pstate, boosts) |

So on epyc1 up to 24 threads stay on one socket, and the second socket joins from 28 on. On dell1 every thread count from 2 on is spread over both sockets.

## Results

![Thread scaling on epyc1](chart-epyc.png)

![Thread scaling on dell1](chart-dell.png)

Training time, median of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 15.17 s | 1.0× | 31.95 s | 1.0× | 29.06 s | 1.0× | 55.70 s | 1.0× |
| 2 | 9.43 s | 1.6× | 20.70 s | 1.5× | 14.91 s | 1.9× | 32.95 s | 1.7× |
| 4 | 4.85 s | 3.1× | 10.79 s | 3.0× | 8.79 s | 3.3× | 17.57 s | 3.2× |
| 8 | 2.97 s | 5.1× | 5.94 s | 5.4× | 4.40 s | 6.6× | 9.04 s | 6.2× |
| 12 | 2.18 s | 7.0× | 4.27 s | 7.5× | 3.13 s | 9.3× | 6.29 s | 8.9× |
| 16 | 1.83 s | 8.3× | 3.49 s | 9.2× | 2.42 s | 12.0× | 4.76 s | 11.7× |
| 20 | 1.61 s | 9.4× | 3.01 s | 10.6× | 2.11 s | 13.8× | 4.05 s | 13.8× |
| 24 | 1.41 s | 10.7× | 2.65 s | 12.1× | 1.82 s | 16.0× | 3.50 s | 15.9× |
| 28 | 1.40 s | 10.8× | 2.54 s | 12.6× | 1.71 s | 17.0× | 3.14 s | 17.7× |
| 32 | 1.31 s | 11.6× | 2.35 s | 13.6× | 1.45 s | 20.1× | 2.75 s | 20.3× |
| 36 | 1.27 s | 11.9× | 2.27 s | 14.1× | 1.54 s | 18.8× | 2.59 s | 21.5× |
| 40 | 1.21 s | 12.5× | 2.11 s | 15.1× | 1.32 s | 21.9× | 2.54 s | 21.9× |
| 44 | 1.14 s | 13.3× | 2.02 s | 15.9× | 1.26 s | 23.0× | 2.43 s | 22.9× |
| 48 | **1.05 s** | 14.4× | **1.84 s** | 17.4× | 1.26 s | 23.1× | 2.30 s | 24.2× |

The trees are CART 1,611 nodes (depth 19, training accuracy 71.79%) and C4.5 1,159,295 nodes (depth 71, 87.88%) at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

## Observations

- **With all 48 threads epyc is faster:** CART 1.05 against 1.26 s, C4.5 1.84 against 2.30 s.
- **On one thread dell is much slower:** 29.1 against 15.2 s (CART) and 55.7 against 32.0 s (C4.5), 1.7–1.9× epyc's time. That is why dell's speedups look better (23–24× against 14–17× at 48 threads). The parallel times are closer.
  - Why dell's single thread is this slow was not investigated. Its governor (`powersave` with intel_pstate) still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc stalls where the second socket comes in:** CART takes 1.41 s at 24 threads and 1.40 s at 28. After that it gains again, to 1.05 s at 48. dell, whose threads span both sockets from the start, has no such step; it has a small dip at 36 threads (1.45 → 1.54 s on CART), seen in both runs of this benchmark.
- **Efficiency falls steadily:** at 48 threads it is 30% (CART) and 36% (C4.5) on epyc, and 48% and 50% on dell (each against its own serial time). The big nodes near the root are bound by memory bandwidth, and CART's 1,611-node tree has little else to share out.
- **C4.5 scales better than CART** on epyc (17.4× against 14.4× at 48). Its 1.16-million-node tree has many more subtrees to grow in parallel. Its pruning runs on all threads too: 3.46 s serial, 0.28 s on 48 threads (epyc).
- **Against the run before the node store fix** (17:59, commit 514bdc7): every point is within 5%, most within 3%, faster and slower alike. The fix matters for the many small subtrees of the CUDA backend on airline, not for HIGGS on the CPU servers.
- **Memory:** the serial run needs 3.7 GiB (CART) and 5.0 GiB (C4.5). From 2 threads on, the parallel backend keeps a second copy of the columns for its out-of-place partition, so the peak is a constant 5.7 GiB and 7.1 GiB, the same on both nodes.
