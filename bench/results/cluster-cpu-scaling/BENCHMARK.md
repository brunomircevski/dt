# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS and airline

**Status:** run on 2026-10-17, 3 runs per point, at commit cd18717: the columns are kept once, in column slots, with min(features − 1, threads) spare slots (one for `--serial`) instead of a second copy of every column ([docs/CPU.md](../../../docs/CPU.md)); ./tree's defaults otherwise (the fastest settings found on these servers). `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **HIGGS:** epyc1 17:51–18:00, dell1 17:52–18:05.
- **Airline:** epyc1 18:00–18:48, dell1 18:05–19:24.
- **Each node ran alone** and was used whole.
- **Same trees:** every thread count built the same tree on both nodes, the same as in [cluster-higgs](../cluster-higgs/BENCHMARK.md).

The cluster's counterpart of the desktop's and laptop's scaling runs ([cpu-pc-susy-scaling](../cpu-pc-susy-scaling/BENCHMARK.md)), on HIGGS (10.5M rows, 28 features) and airline (115M rows, 13 features), on the two CPU servers only. The GPU servers are not benchmarked on their CPUs.

```bash
bench/cluster/scaling.sh -m 3              # on the login node: one job on epyc1, one on dell1
bench/cluster/scaling.sh -m 3 -d airline   # airline
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

All four charts in one image (HIGGS above airline, epyc left, dell right):

![Thread scaling on epyc1 and dell1, HIGGS and airline](chart.png)

### Against the previous runs

The previous runs (HIGGS 2026-10-15 on 8da0c12, airline 2026-10-17 on 6c7a43b, their one-thread points rerun on 3bd3a46 after `--serial` moved to the parallel backend's out-of-place partition) kept a second copy of every column at every thread count.
- **Memory now grows with the thread count** instead of being constant: a partition needs one free column slot per column it moves at a time, and ./tree keeps min(features − 1, threads) spare slots (1 for `--serial`).
  - HIGGS (28 features): CART 3.6 GiB serial → 5.6 GiB from 28 threads on, C4.5 3.9 → 6.0 GiB (5.7 and 6.0–6.1 GiB at every count before).
  - Airline (13 features): CART 19.6 GiB serial → 27.2–28.1 GiB from 12 threads on, C4.5 23.2 → 31.3–32.3 GiB (29.0 and 32.6–33.2 GiB before).
- **Times are within ±4% at almost every point,** on both nodes and both datasets. Exceptions: epyc's serial C4.5 got faster (HIGGS 37.82 → 36.16 s, airline 178.63 → 170.88 s), and epyc HIGGS C4.5 at 8 threads slower (5.52 → 5.94 s).
- **The same trees** at every point, as before.

## HIGGS

Training time, median of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 18.02 s | 1.0× | 36.16 s | 1.0× | 25.68 s | 1.0× | 53.55 s | 1.0× |
| 2 | 9.52 s | 1.9× | 19.05 s | 1.9× | 15.32 s | 1.7× | 32.61 s | 1.6× |
| 4 | 4.88 s | 3.7× | 9.70 s | 3.7× | 8.31 s | 3.1× | 17.64 s | 3.0× |
| 8 | 3.13 s | 5.8× | 5.94 s | 6.1× | 4.51 s | 5.7× | 8.97 s | 6.0× |
| 12 | 2.19 s | 8.2× | 4.03 s | 9.0× | 3.15 s | 8.1× | 6.19 s | 8.7× |
| 16 | 1.91 s | 9.4× | 3.32 s | 10.9× | 2.47 s | 10.4× | 4.71 s | 11.4× |
| 20 | 1.66 s | 10.9× | 2.89 s | 12.5× | 2.16 s | 11.9× | 3.99 s | 13.4× |
| 24 | 1.47 s | 12.3× | 2.52 s | 14.4× | 1.85 s | 13.8× | 3.57 s | 15.0× |
| 28 | 1.49 s | 12.1× | 2.39 s | 15.1× | 1.77 s | 14.5× | 3.00 s | 17.8× |
| 32 | 1.46 s | 12.4× | 2.30 s | 15.7× | 1.51 s | 17.0× | 2.76 s | 19.4× |
| 36 | 1.33 s | 13.5× | 2.11 s | 17.1× | 1.50 s | 17.1× | 2.54 s | 21.1× |
| 40 | 1.27 s | 14.2× | 1.97 s | 18.4× | 1.35 s | 19.0× | 2.47 s | 21.7× |
| 44 | 1.18 s | 15.3× | 1.89 s | 19.2× | 1.32 s | 19.5× | 2.34 s | 22.8× |
| 48 | **1.15 s** | 15.7× | **1.75 s** | 20.6× | 1.27 s | 20.2× | 2.26 s | 23.7× |

The trees are CART 1,611 nodes (depth 19, training accuracy 71.79%) and C4.5 1,159,295 nodes (depth 71, 87.88%) at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

### Observations

- **With all 48 threads epyc is faster:** CART 1.15 against 1.27 s, C4.5 1.75 against 2.26 s.
- **On one thread dell is slower:** 25.7 against 18.0 s (CART) and 53.6 against 36.2 s (C4.5), 1.4–1.5× epyc's time. That is why dell's speedups look better (20–24× against 16–21× at 48 threads). The parallel times are closer.
  - Why dell's single thread is slower was not investigated. Its governor (`powersave` with intel_pstate) still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc stalls where the second socket comes in:** CART takes 1.47 s at 24 threads, 1.49 s at 28 and 1.46 s at 32. After that it gains again, to 1.15 s at 48. dell, whose threads span both sockets from the start, has no such step.
- **Efficiency falls steadily:** at 48 threads it is 33% (CART) and 43% (C4.5) on epyc, and 42% and 49% on dell (each against its own serial time). The big nodes near the root are bound by memory bandwidth, and CART's 1,611-node tree has little else to share out.
- **C4.5 scales better than CART** on epyc (20.6× against 15.7× at 48). Its 1.16-million-node tree has many more subtrees to grow in parallel. Its pruning runs on all threads too: 2.11 s serial, 0.19 s on 48 threads (epyc).
- **Memory:** 3.6 GiB (CART) and 3.9 GiB (C4.5) serial, rising with the spare column slots to 5.6 and 6.0 GiB from 28 threads on (27 spare slots for 28 features), the same on both nodes.

## Airline

Training time, median of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 75.07 s | 1.0× | 170.88 s | 1.0× | 126.77 s | 1.0× | 263.06 s | 1.0× |
| 2 | 41.08 s | 1.8× | 91.41 s | 1.9× | 105.65 s | 1.2× | 199.18 s | 1.3× |
| 4 | 21.36 s | 3.5× | 47.20 s | 3.6× | 60.55 s | 2.1× | 110.19 s | 2.4× |
| 8 | 17.05 s | 4.4× | 32.33 s | 5.3× | 31.11 s | 4.1× | 56.51 s | 4.7× |
| 12 | 11.92 s | 6.3× | 22.49 s | 7.6× | 21.01 s | 6.0× | 38.63 s | 6.8× |
| 16 | 10.81 s | 6.9× | 19.65 s | 8.7× | 16.26 s | 7.8× | 30.24 s | 8.7× |
| 20 | 9.31 s | 8.1× | 16.82 s | 10.2× | 13.65 s | 9.3× | 25.28 s | 10.4× |
| 24 | 8.10 s | 9.3× | 14.91 s | 11.5× | 11.73 s | 10.8× | 21.88 s | 12.0× |
| 28 | 9.19 s | 8.2× | 16.00 s | 10.7× | 10.49 s | 12.1× | 19.53 s | 13.5× |
| 32 | 9.19 s | 8.2× | 15.59 s | 11.0× | 9.70 s | 13.1× | 17.56 s | 15.0× |
| 36 | 8.46 s | 8.9× | 14.29 s | 12.0× | 9.14 s | 13.9× | 16.49 s | 15.9× |
| 40 | 8.06 s | 9.3× | 13.74 s | 12.4× | 8.39 s | 15.1× | 15.52 s | 16.9× |
| 44 | 7.61 s | 9.9× | 12.74 s | 13.4× | 8.00 s | 15.8× | 14.90 s | 17.7× |
| 48 | **7.17 s** | 10.5× | **11.98 s** | 14.3× | 7.79 s | 16.3× | 14.01 s | 18.8× |

The trees are CART 3,013 nodes (depth 25, training accuracy 71.54%) and C4.5 8,548,725 nodes (depth 72, 87.42%) at every point on both nodes, the same as in [cluster-airlines](../cluster-airlines/BENCHMARK.md), whose 48-thread times these match within 1%, except dell's CART (7.79 against 7.52 s, 3.6%).

### Observations

- **Airline's speedups are lower than HIGGS's:** 10.5× (CART) and 14.3× (C4.5) on epyc at 48 threads against 15.7× and 20.6× on HIGGS; 16.3× and 18.8× on dell against 20.2× and 23.7×.
- **From 1 to 2 threads:** 1.8× (CART, 75.1 → 41.1 s) and 1.9× (C4.5) on epyc; only 1.2× and 1.3× on dell (1.6–1.7× on HIGGS). dell's first two CPUs sit on different sockets, so half of each thread's memory is on the other socket, which may weigh more with airline's 115M rows; this was not investigated.
- **epyc again loses time where its second socket comes in:** CART 8.10 s at 24 threads, 9.19 s at 28 and 32, then down to 7.17 s at 48 (C4.5 14.91 → 16.00 → 15.59 → 11.98 s). dell, whose threads span both sockets from 2 on, keeps improving at every step.
- **At 48 threads the two nodes are close:** CART 7.17 s (epyc) against 7.79 s (dell), C4.5 11.98 against 14.01 s. dell's single thread is 1.7× (CART, 126.8 against 75.1 s) and 1.5× (C4.5) slower than epyc's.
- **Memory:** 19.6 GiB (CART) and 23.2 GiB (C4.5) serial, rising to 27.2–28.1 and 31.3–32.3 GiB from 12 threads on (12 spare slots for 13 features), the same on both nodes.
