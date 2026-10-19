# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS and airline

**Status:** run on 2026-10-19, 3 runs per point, at commit 051949b: cd18717 plus `./tree --profile` and the count tables filled without a zero-fill on one thread. In cd18717 the columns are kept once, in column slots, with min(features − 1, threads) spare slots (one for `--serial`) instead of a second copy of every column ([docs/CPU.md](../../../docs/CPU.md)); ./tree's defaults otherwise (the fastest settings found on these servers). `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **HIGGS:** dell1 15:50–16:03, epyc1 16:04–16:14.
- **Airline:** dell1 16:03–17:23, epyc1 16:14–17:02.
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

**Against cd18717** (2026-10-17): the same trees at every point; every time within −5.2% to +2.8%, the memory unchanged. The count tables no longer zero-fill 2 × (rows + 1) doubles on one thread, which shows most at high thread counts on airline (dell CART at 48 threads 7.79 → 7.38 s, at 32 threads 9.70 → 9.29 s). The rest of this section compares cd18717 with the runs before it.

The runs before cd18717 (HIGGS 2026-10-15 on 8da0c12, airline 2026-10-17 on 6c7a43b, their one-thread points rerun on 3bd3a46 after `--serial` moved to the parallel backend's out-of-place partition) kept a second copy of every column at every thread count.
- **Memory now grows with the thread count** instead of being constant: a partition needs one free column slot per column it moves at a time, and ./tree keeps min(features − 1, threads) spare slots (1 for `--serial`).
  - HIGGS (28 features): CART 3.6 GiB serial → 5.6 GiB from 28 threads on, C4.5 3.9 → 6.0 GiB (5.7 and 6.0–6.1 GiB at every count before).
  - Airline (13 features): CART 19.6 GiB serial → 27.2–28.1 GiB from 12 threads on, C4.5 23.2 → 31.3–32.3 GiB (29.0 and 32.6–33.2 GiB before).
- **Times are within ±4% at almost every point,** on both nodes and both datasets. Exceptions: epyc's serial C4.5 got faster (HIGGS 37.82 → 36.16 s, airline 178.63 → 170.88 s), and epyc HIGGS C4.5 at 8 threads slower (5.52 → 5.94 s).
- **The same trees** at every point, as before.

## HIGGS

Training time, median of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 17.98 s | 1.0× | 36.22 s | 1.0× | 25.57 s | 1.0× | 54.13 s | 1.0× |
| 2 | 9.54 s | 1.9× | 19.07 s | 1.9× | 15.41 s | 1.7× | 32.66 s | 1.7× |
| 4 | 4.88 s | 3.7× | 9.71 s | 3.7× | 8.27 s | 3.1× | 17.73 s | 3.1× |
| 8 | 3.12 s | 5.8× | 5.91 s | 6.1× | 4.46 s | 5.7× | 8.92 s | 6.1× |
| 12 | 2.20 s | 8.2× | 4.04 s | 9.0× | 3.14 s | 8.2× | 6.11 s | 8.9× |
| 16 | 1.89 s | 9.5× | 3.32 s | 10.9× | 2.45 s | 10.4× | 4.74 s | 11.4× |
| 20 | 1.65 s | 10.9× | 2.86 s | 12.7× | 2.11 s | 12.1× | 3.96 s | 13.7× |
| 24 | 1.46 s | 12.3× | 2.54 s | 14.3× | 1.81 s | 14.1× | 3.49 s | 15.5× |
| 28 | 1.47 s | 12.3× | 2.37 s | 15.3× | 1.73 s | 14.8× | 3.00 s | 18.0× |
| 32 | 1.43 s | 12.6× | 2.31 s | 15.7× | 1.50 s | 17.1× | 2.71 s | 20.0× |
| 36 | 1.36 s | 13.3× | 2.11 s | 17.2× | 1.54 s | 16.6× | 2.52 s | 21.5× |
| 40 | 1.26 s | 14.3× | 1.99 s | 18.2× | 1.33 s | 19.2× | 2.43 s | 22.3× |
| 44 | 1.20 s | 15.0× | 1.84 s | 19.7× | 1.28 s | 20.0× | 2.29 s | 23.6× |
| 48 | **1.12 s** | 16.1× | **1.74 s** | 20.8× | 1.25 s | 20.4× | 2.19 s | 24.7× |

The trees are CART 1,611 nodes (depth 19, training accuracy 71.79%) and C4.5 1,159,295 nodes (depth 71, 87.88%) at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

### Observations

- **With all 48 threads epyc is faster:** CART 1.12 against 1.25 s, C4.5 1.74 against 2.19 s.
- **On one thread dell is slower:** 25.6 against 18.0 s (CART) and 54.1 against 36.2 s (C4.5), 1.4–1.5× epyc's time. That is why dell's speedups look better (20–25× against 16–21× at 48 threads). The parallel times are closer.
  - Why dell's single thread is slower was not investigated. Its governor (`powersave` with intel_pstate) still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc stalls where the second socket comes in:** CART takes 1.46 s at 24 threads, 1.47 s at 28 and 1.43 s at 32. After that it gains again, to 1.12 s at 48. dell, whose threads span both sockets from the start, has no such step.
- **Efficiency falls steadily:** at 48 threads it is 34% (CART) and 43% (C4.5) on epyc, and 43% and 51% on dell (each against its own serial time). The big nodes near the root are bound by memory bandwidth, and CART's 1,611-node tree has little else to share out.
- **C4.5 scales better than CART** on epyc (20.8× against 16.1× at 48). Its 1.16-million-node tree has many more subtrees to grow in parallel. Its pruning runs on all threads too: 2.13 s serial, 0.19 s on 48 threads (epyc).
- **Memory:** 3.6 GiB (CART) and 3.9 GiB (C4.5) serial, rising with the spare column slots to 5.6 and 6.0 GiB from 28 threads on (27 spare slots for 28 features), the same on both nodes.

## Airline

Training time, median of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 74.97 s | 1.0× | 170.78 s | 1.0× | 126.71 s | 1.0× | 262.80 s | 1.0× |
| 2 | 40.92 s | 1.8× | 91.81 s | 1.9× | 105.47 s | 1.2× | 200.49 s | 1.3× |
| 4 | 21.12 s | 3.5× | 47.13 s | 3.6× | 60.47 s | 2.1× | 109.29 s | 2.4× |
| 8 | 17.03 s | 4.4× | 32.77 s | 5.2× | 31.13 s | 4.1× | 56.53 s | 4.6× |
| 12 | 11.80 s | 6.4× | 22.55 s | 7.6× | 20.94 s | 6.1× | 38.46 s | 6.8× |
| 16 | 10.67 s | 7.0× | 19.73 s | 8.7× | 16.23 s | 7.8× | 29.62 s | 8.9× |
| 20 | 9.16 s | 8.2× | 16.66 s | 10.3× | 13.44 s | 9.4× | 24.95 s | 10.5× |
| 24 | 8.11 s | 9.2× | 14.80 s | 11.5× | 11.64 s | 10.9× | 21.59 s | 12.2× |
| 28 | 9.31 s | 8.1× | 16.01 s | 10.7× | 10.14 s | 12.5× | 19.12 s | 13.7× |
| 32 | 9.19 s | 8.2× | 15.99 s | 10.7× | 9.29 s | 13.6× | 17.41 s | 15.1× |
| 36 | 8.31 s | 9.0× | 14.25 s | 12.0× | 8.78 s | 14.4× | 16.25 s | 16.2× |
| 40 | 7.93 s | 9.5× | 13.61 s | 12.5× | 8.17 s | 15.5× | 15.15 s | 17.3× |
| 44 | 7.55 s | 9.9× | 12.65 s | 13.5× | 7.77 s | 16.3× | 14.51 s | 18.1× |
| 48 | **7.06 s** | 10.6× | **11.94 s** | 14.3× | 7.38 s | 17.2× | 13.79 s | 19.1× |

The trees are CART 3,013 nodes (depth 25, training accuracy 71.54%) and C4.5 8,548,725 nodes (depth 72, 87.42%) at every point on both nodes, the same as in [cluster-airlines](../cluster-airlines/BENCHMARK.md), whose 48-thread times these match within 2% (dell's CART 7.38 against 7.51 s).

### Observations

- **Airline's speedups are lower than HIGGS's:** 10.6× (CART) and 14.3× (C4.5) on epyc at 48 threads against 16.1× and 20.8× on HIGGS; 17.2× and 19.1× on dell against 20.4× and 24.7×.
- **From 1 to 2 threads:** 1.8× (CART, 75.0 → 40.9 s) and 1.9× (C4.5) on epyc; only 1.2× and 1.3× on dell (1.7× on HIGGS). dell's first two CPUs sit on different sockets, so half of each thread's memory is on the other socket, which may weigh more with airline's 115M rows; this was not investigated.
- **epyc again loses time where its second socket comes in:** CART 8.11 s at 24 threads, 9.31 s at 28 and 9.19 s at 32, then down to 7.06 s at 48 (C4.5 14.80 → 16.01 → 15.99 → 11.94 s). dell, whose threads span both sockets from 2 on, keeps improving at every step.
- **At 48 threads the two nodes are close:** CART 7.06 s (epyc) against 7.38 s (dell), C4.5 11.94 against 13.79 s. dell's single thread is 1.7× (CART, 126.7 against 75.0 s) and 1.5× (C4.5) slower than epyc's.
- **Memory:** 19.6 GiB (CART) and 23.2 GiB (C4.5) serial, rising to 27.2–28.1 and 31.3–32.3 GiB from 12 threads on (12 spare slots for 13 features), the same on both nodes.
