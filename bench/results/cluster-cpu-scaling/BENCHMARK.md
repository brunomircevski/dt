# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS

**Status:** run on 2026-10-15, 3 runs per point, at commit 35cf315 (C4.5 pruning, GPU partition and upload: [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)), with ./tree's defaults (the fastest settings found on these servers). `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **epyc1:** 16:54–17:03.
- **dell1:** 16:55–17:09.
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
| 1 (serial) | 15.30 s | 1.0× | 31.13 s | 1.0× | 29.19 s | 1.0× | 54.75 s | 1.0× |
| 2 | 9.48 s | 1.6× | 19.68 s | 1.6× | 14.90 s | 2.0× | 31.74 s | 1.7× |
| 4 | 4.87 s | 3.1× | 10.00 s | 3.1× | 8.07 s | 3.6× | 16.75 s | 3.3× |
| 8 | 3.00 s | 5.1× | 5.67 s | 5.5× | 4.46 s | 6.6× | 8.65 s | 6.3× |
| 12 | 2.20 s | 7.0× | 4.10 s | 7.6× | 3.15 s | 9.3× | 6.10 s | 9.0× |
| 16 | 1.85 s | 8.3× | 3.33 s | 9.4× | 2.39 s | 12.2× | 4.61 s | 11.9× |
| 20 | 1.61 s | 9.5× | 2.90 s | 10.7× | 2.09 s | 13.9× | 3.93 s | 13.9× |
| 24 | 1.46 s | 10.5× | 2.55 s | 12.2× | 1.80 s | 16.2× | 3.41 s | 16.0× |
| 28 | 1.41 s | 10.8× | 2.40 s | 13.0× | 1.71 s | 17.1× | 2.96 s | 18.5× |
| 32 | 1.41 s | 10.8× | 2.28 s | 13.6× | 1.47 s | 19.9× | 2.74 s | 20.0× |
| 36 | 1.29 s | 11.8× | 2.13 s | 14.6× | 1.53 s | 19.1× | 2.54 s | 21.5× |
| 40 | 1.25 s | 12.3× | 2.02 s | 15.4× | 1.34 s | 21.8× | 2.50 s | 21.9× |
| 44 | 1.15 s | 13.3× | 1.91 s | 16.3× | 1.28 s | 22.9× | 2.43 s | 22.6× |
| 48 | **1.11 s** | 13.8× | **1.77 s** | 17.6× | 1.23 s | 23.8× | 2.20 s | 24.9× |

The trees are CART 1,611 nodes (depth 19, training accuracy 71.79%) and C4.5 1,159,295 nodes (depth 71, 87.88%) at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

## Observations

- **With all 48 threads epyc is faster:** CART 1.11 against 1.23 s, C4.5 1.77 against 2.20 s.
- **On one thread dell is much slower:** 29.2 against 15.3 s (CART) and 54.8 against 31.1 s (C4.5), 1.8–1.9× epyc's time. That is why dell's speedups look better (24–25× against 14–18× at 48 threads). The parallel times are closer.
  - Why dell's single thread is this slow was not investigated. Its governor (`powersave` with intel_pstate) still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc stalls where the second socket comes in:** CART takes 1.46 s at 24 threads and 1.41 s at 28 and 32. After that it gains again, to 1.11 s at 48. dell, whose threads span both sockets from the start, has no such step; it has a small dip at 36 threads (1.47 → 1.53 s on CART), seen in every run of this benchmark.
- **Efficiency falls steadily:** at 48 threads it is 29% (CART) and 37% (C4.5) on epyc, and 50% and 52% on dell (each against its own serial time). The big nodes near the root are bound by memory bandwidth, and CART's 1,611-node tree has little else to share out.
- **C4.5 scales better than CART** on epyc (17.6× against 13.8× at 48). Its 1.16-million-node tree has many more subtrees to grow in parallel. Its pruning runs on all threads too: 2.25 s serial, 0.20 s on 48 threads (epyc).
- **Against the previous run** (2026-10-14, commit a83b6ec):
  - **C4.5 is 2–7% faster at every point** on both nodes (epyc 48 threads 1.84 → 1.77 s, serial 31.95 → 31.13 s). The growers now record each row's leaf, so the pruning does not route every row through the tree again (pruning: epyc 3.46 → 2.25 s serial and 0.28 → 0.20 s on 48 threads, dell 0.35 → 0.28 s on 48).
  - **CART is within 3% at most points.** The CPU builder's CART path is unchanged apart from one test for the leaf recording. epyc is 4–8% slower from 24 threads on (48 threads: 1.05 → 1.11 s); a same-job comparison of both commits on airline found 0.5% ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)), so part of it is the day's variation. Not investigated further.
- **Memory:** the serial run needs 3.7 GiB (CART) and 5.0 GiB (C4.5). From 2 threads on, the parallel backend keeps a second copy of the columns for its out-of-place partition, so the peak is a constant 5.7 GiB and 7.1–7.2 GiB, the same on both nodes.
