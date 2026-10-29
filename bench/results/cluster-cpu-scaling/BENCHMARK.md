# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS and airline

**Status:** run on 2026-10-20, 3 runs per point: dell1 at commit 8b1ecd1 (051949b plus each thread counting the classes in its own histogram, 6521330, and the GPU changes), epyc1 at 1a2d9a9 (8b1ecd1 plus `run.py --drop-cache`: the datasets leave the page cache before every run, see below). In cd18717 the columns are kept once, in column slots, with min(features − 1, threads) spare slots (one for `--serial`) instead of a second copy of every column ([docs/CPU.md](../../../docs/CPU.md)); ./tree's defaults otherwise (the fastest settings found on these servers). epyc's `machine.json` says `-dirty` only because result files of the cluster's checkout had changed; the code was committed.
- **HIGGS:** dell1 14:41–14:54 (job 1309), epyc1 16:25–16:35 (job 1343).
- **Airline:** dell1 14:55–16:12 (job 1311), epyc1 16:36–17:31 (job 1344).
- **Why epyc was rerun:** its first run at 8b1ecd1 found NUMA node 1 nearly full with our own cached dataset files (236 of 252 GiB), so the 48-thread runs' memory partly landed on the other socket (cluster-ecbdl-old, removed; last in f4668df). With `--drop-cache` the page cache no longer holds the datasets between runs.
- **Each node ran alone** and was used whole.
- **Same trees:** every thread count built the same tree on both nodes, the same as in [cluster-higgs](../cluster-higgs/BENCHMARK.md).

The cluster's counterpart of the desktop's and laptop's scaling runs (`cpu-pc-susy-scaling`, removed in 254d4e8), on HIGGS (10.5M rows, 28 features) and airline (115M rows, 13 features), on the two CPU servers only. The GPU servers are not benchmarked on their CPUs.

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

All four charts in one image (HIGGS above airline, epyc left, dell right). Threads and time are on log axes, so ideal scaling (dotted: the 1-thread time divided by the threads) is a straight line:

![Thread scaling on epyc1 and dell1, HIGGS and airline](chart.png)

### One socket

The diamonds at 24 threads are the one-socket runs of [cluster-higgs](../cluster-higgs/BENCHMARK.md) and cluster-airline-old (removed; last in f4668df) (`socket-*`, 2026-10-20 at 10d125d, `--socket` in `scaling_chart.py`): 24 threads on every CPU of one socket, the memory bound to its NUMA nodes (epyc1's socket 0, dell1's socket 1). Means of 3:

| | epyc: 24 threads / one socket | dell: 24 threads / one socket |
|---|--:|--:|
| HIGGS CART | 1.43 / 1.44 s | 1.84 / 1.78 s |
| HIGGS C4.5 | 2.49 / 2.49 s | 3.47 / 3.58 s |
| airline CART | 8.02 / 7.97 s | 11.63 / **8.90 s** |
| airline C4.5 | 14.71 / 14.71 s | 21.73 / **17.33 s** |

- **epyc:** the same within 1%. Its first 24 CPUs are socket 0, so the 24-thread point already ran on one socket; binding the memory changes nothing.
- **dell:** on HIGGS the same within 4%, but on airline one socket is 1.25–1.31× faster than 24 threads spread over both. dell's CPUs alternate between the sockets, so the 24-thread point uses 12 cores of each and their memory on four NUMA nodes; airline's 115M rows probably move far more data between them than HIGGS's 10.5M (not profiled).
- **Against all 48 threads:** one socket takes 1.1–1.7× as long ([cluster-higgs](../cluster-higgs/BENCHMARK.md), cluster-airline-old (removed; last in f4668df)).

### Against the previous runs

**Against 051949b** (2026-10-19): the same trees and memory at every point, every time within −5.5% to +2.4% (most within ±2%; the largest changes are faster: epyc HIGGS C4.5 at 8 threads 5.91 → 5.58 s and at 32 threads 2.31 → 2.19 s, dell HIGGS C4.5 at 4 threads 17.73 → 16.90 s).

**Against cd18717** (2026-10-17): the same trees at every point; every time within −5.2% to +2.8%, the memory unchanged. The count tables no longer zero-fill 2 × (rows + 1) doubles on one thread, which shows most at high thread counts on airline (dell CART at 48 threads 7.79 → 7.38 s, at 32 threads 9.70 → 9.29 s). The rest of this section compares cd18717 with the runs before it.

The runs before cd18717 (HIGGS 2026-10-15 on 8da0c12, airline 2026-10-17 on 6c7a43b, their one-thread points rerun on 3bd3a46 after `--serial` moved to the parallel backend's out-of-place partition) kept a second copy of every column at every thread count.
- **Memory now grows with the thread count** instead of being constant: a partition needs one free column slot per column it moves at a time, and ./tree keeps min(features − 1, threads) spare slots (1 for `--serial`).
  - HIGGS (28 features): CART 3.6 GiB serial → 5.6 GiB from 28 threads on, C4.5 3.9 → 6.0 GiB (5.7 and 6.0–6.1 GiB at every count before).
  - Airline (13 features): CART 19.6 GiB serial → 27.2–28.1 GiB from 12 threads on, C4.5 23.2 → 31.3–32.3 GiB (29.0 and 32.6–33.2 GiB before).
- **Times are within ±4% at almost every point,** on both nodes and both datasets. Exceptions: epyc's serial C4.5 got faster (HIGGS 37.82 → 36.16 s, airline 178.63 → 170.88 s), and epyc HIGGS C4.5 at 8 threads slower (5.52 → 5.94 s).
- **The same trees** at every point, as before.

## HIGGS

Training time, mean of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 17.92 s | 1.0× | 36.20 s | 1.0× | 25.51 s | 1.0× | 53.81 s | 1.0× |
| 2 | 9.44 s | 1.9× | 18.99 s | 1.9× | 15.43 s | 1.7× | 34.77 s | 1.5× |
| 4 | 4.84 s | 3.7× | 9.67 s | 3.7× | 8.50 s | 3.0× | 17.21 s | 3.1× |
| 8 | 3.04 s | 5.9× | 5.61 s | 6.5× | 4.46 s | 5.7× | 8.84 s | 6.1× |
| 12 | 2.15 s | 8.3× | 3.96 s | 9.1× | 3.22 s | 7.9× | 6.15 s | 8.8× |
| 16 | 1.85 s | 9.7× | 3.32 s | 10.9× | 2.45 s | 10.4× | 4.63 s | 11.6× |
| 20 | 1.62 s | 11.1× | 2.80 s | 12.9× | 2.13 s | 12.0× | 3.96 s | 13.6× |
| 24 | 1.43 s | 12.5× | 2.49 s | 14.5× | 1.84 s | 13.8× | 3.47 s | 15.5× |
| 28 | 1.44 s | 12.4× | 2.36 s | 15.3× | 1.71 s | 14.9× | 2.96 s | 18.2× |
| 32 | 1.36 s | 13.2× | 2.21 s | 16.4× | 1.51 s | 16.9× | 2.70 s | 19.9× |
| 36 | 1.32 s | 13.6× | 2.13 s | 17.0× | 1.48 s | 17.2× | 2.52 s | 21.3× |
| 40 | 1.24 s | 14.4× | 2.01 s | 18.1× | 1.33 s | 19.2× | 2.45 s | 22.0× |
| 44 | 1.16 s | 15.4× | 1.85 s | 19.6× | 1.26 s | 20.2× | 2.31 s | 23.3× |
| 48 | **1.12 s** | 16.0× | **1.76 s** | 20.6× | 1.24 s | 20.6× | 2.18 s | 24.7× |

The trees are CART 1,611 nodes (depth 19, training accuracy 71.79%) and C4.5 1,159,295 nodes (depth 71, 87.88%) at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

### Observations

- **With all 48 threads epyc is faster:** CART 1.12 against 1.24 s, C4.5 1.76 against 2.18 s.
- **On one thread dell is slower:** 25.5 against 17.9 s (CART) and 53.8 against 36.2 s (C4.5), 1.4–1.5× epyc's time. That is why dell's speedups look better (21–25× against 16–21× at 48 threads). The parallel times are closer.
  - Why dell's single thread is slower was not investigated. Its governor (`powersave` with intel_pstate) still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc stalls where the second socket comes in:** CART takes 1.43 s at 24 threads, 1.44 s at 28 and 1.36 s at 32. After that it gains again, to 1.12 s at 48. dell, whose threads span both sockets from the start, has no such step.
- **Efficiency falls steadily:** at 48 threads it is 33% (CART) and 43% (C4.5) on epyc, and 43% and 51% on dell (each against its own serial time). The big nodes near the root are bound by memory bandwidth, and CART's 1,611-node tree has little else to share out.
- **C4.5 scales better than CART** on epyc (20.6× against 16.0× at 48). Its 1.16-million-node tree has many more subtrees to grow in parallel. Its pruning runs on all threads too: 2.12 s serial, 0.19 s on 48 threads (epyc).
- **Memory:** 3.6 GiB (CART) and 3.9 GiB (C4.5) serial, rising with the spare column slots to 5.6 and 6.0 GiB from 28 threads on (27 spare slots for 28 features), the same on both nodes.

## Airline

Training time, mean of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 74.77 s | 1.0× | 170.89 s | 1.0× | 126.19 s | 1.0× | 262.34 s | 1.0× |
| 2 | 40.32 s | 1.9× | 90.94 s | 1.9× | 106.03 s | 1.2× | 200.31 s | 1.3× |
| 4 | 20.99 s | 3.6× | 46.86 s | 3.6× | 60.54 s | 2.1× | 109.49 s | 2.4× |
| 8 | 16.96 s | 4.4× | 32.61 s | 5.2× | 30.98 s | 4.1× | 56.51 s | 4.6× |
| 12 | 11.59 s | 6.5× | 22.18 s | 7.7× | 20.80 s | 6.1× | 38.35 s | 6.8× |
| 16 | 10.49 s | 7.1× | 19.35 s | 8.8× | 16.28 s | 7.8× | 29.90 s | 8.8× |
| 20 | 9.04 s | 8.3× | 16.75 s | 10.2× | 13.48 s | 9.4× | 24.95 s | 10.5× |
| 24 | 8.02 s | 9.3× | 14.71 s | 11.6× | 11.63 s | 10.8× | 21.73 s | 12.1× |
| 28 | 8.92 s | 8.4× | 16.15 s | 10.6× | 10.27 s | 12.3× | 19.04 s | 13.8× |
| 32 | 9.27 s | 8.1× | 15.73 s | 10.9× | 9.30 s | 13.6× | 17.17 s | 15.3× |
| 36 | 8.27 s | 9.0× | 14.31 s | 11.9× | 8.73 s | 14.4× | 16.06 s | 16.3× |
| 40 | 7.94 s | 9.4× | 13.55 s | 12.6× | 8.17 s | 15.5× | 15.12 s | 17.3× |
| 44 | 7.43 s | 10.1× | 12.63 s | 13.5× | 7.80 s | 16.2× | 14.39 s | 18.2× |
| 48 | **7.01 s** | 10.7× | **11.85 s** | 14.4× | 7.36 s | 17.1× | 13.74 s | 19.1× |

The trees are CART 3,013 nodes (depth 25, training accuracy 71.54%) and C4.5 8,548,725 nodes (depth 72, 87.42%) at every point on both nodes, the same as in cluster-airline-old (removed; last in f4668df), whose 48-thread times these match within 1% (dell's CART 7.36 against 7.39 s).

### Observations

- **Airline's speedups are lower than HIGGS's:** 10.7× (CART) and 14.4× (C4.5) on epyc at 48 threads against 16.0× and 20.6× on HIGGS; 17.1× and 19.1× on dell against 20.6× and 24.7×.
- **From 1 to 2 threads:** 1.9× (CART, 74.8 → 40.3 s) and 1.9× (C4.5) on epyc; only 1.2× and 1.3× on dell (1.5–1.7× on HIGGS). dell's first two CPUs sit on different sockets, so half of each thread's memory is on the other socket, which may weigh more with airline's 115M rows; this was not investigated.
- **epyc again loses time where its second socket comes in:** CART 8.02 s at 24 threads, 8.92 s at 28 and 9.27 s at 32, then down to 7.01 s at 48 (C4.5 14.71 → 16.15 → 15.73 → 11.85 s). dell, whose threads span both sockets from 2 on, keeps improving at every step.
- **At 48 threads the two nodes are close:** CART 7.01 s (epyc) against 7.36 s (dell), C4.5 11.85 against 13.74 s. dell's single thread is 1.7× (CART, 126.2 against 74.8 s) and 1.5× (C4.5) slower than epyc's.
- **Memory:** 19.6 GiB (CART) and 23.2 GiB (C4.5) serial, rising to 27.2–28.1 and 31.3–32.4 GiB from 12 threads on (12 spare slots for 13 features), the same on both nodes.
