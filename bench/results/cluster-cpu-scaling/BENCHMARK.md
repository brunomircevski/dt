# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS and airline

**Status:** HIGGS run on 2026-10-15, 3 runs per point, at commit 8da0c12 (GPU hand-off and C4.5 changes, parallel presort: [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)), with ./tree's defaults (the fastest settings found on these servers). `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **epyc1:** 23:26–23:35.
- **dell1:** 23:27–23:41.
- **Airline** (`cpu-epyc-airline`, `cpu-dell-airline`): run on 2026-10-17, 3 runs per point, at commit 6c7a43b (the same ./tree code and binaries as 8da0c12). epyc1 10:11–11:08, dell1 10:11–11:37.
- **One-thread points rerun on 2026-10-17** (`bench/serial_rerun.py`, commit 3bd3a46, 3 runs each): since then `--serial` runs the parallel backend's algorithm, with big nodes partitioned out of place ([docs/CPU.md](../../../docs/CPU.md)). The other points are from the runs above; `machine.json` → `serial_rerun` records it, and the old serial times are in the table below. Logs: `bench/cluster/serial-rerun-{epyc,dell}.log`.
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

All four charts in one image (HIGGS above airline, epyc left, dell right); each is also below in its own section:

![Thread scaling on epyc1 and dell1, HIGGS and airline](chart.png)

### The serial rerun

| Serial (1 thread), median of 3 | before (in place) | now (out of place) | change |
|---|--:|--:|--:|
| epyc HIGGS CART | 15.32 s | 18.16 s | 0.84× |
| epyc HIGGS C4.5 | 30.50 s | 37.82 s | 0.81× |
| dell HIGGS CART | 29.31 s | 25.72 s | 1.14× |
| dell HIGGS C4.5 | 54.43 s | 54.00 s | 1.01× |
| epyc airline CART | 148.54 s | 76.38 s | **1.94×** |
| epyc airline C4.5 | 250.61 s | 178.63 s | **1.40×** |
| dell airline CART | 204.67 s | 128.09 s | **1.60×** |
| dell airline C4.5 | 338.34 s | 264.14 s | **1.28×** |

- **Airline gains most:** with 115M rows the old serial path's byte per row (115 MB) does not fit in the caches; the bitmap (14 MB) does.
- **On epyc HIGGS got slower** (build 13.55 → 16.25 s for CART, the three runs within 0.05 s of each other): out of place on one Zen 5 core costs more than in place for HIGGS's nodes. Why was not investigated; one candidate is the non-temporal stores, tuned for many threads. dell gains or stays level.
- **Memory:** a serial run now holds the second copy of the columns too, like the parallel ones: HIGGS 3.7 → 5.7 GiB (CART), airline 20.4 → 29.0 GiB.
- **Speedups changed with the serial times:** every speedup below is over the new serial time.

## HIGGS

![Thread scaling on epyc1](chart-epyc.png)

![Thread scaling on dell1](chart-dell.png)

Training time, median of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 18.16 s | 1.0× | 37.82 s | 1.0× | 25.72 s | 1.0× | 54.00 s | 1.0× |
| 2 | 9.50 s | 1.9× | 19.01 s | 2.0× | 15.15 s | 1.7× | 31.87 s | 1.7× |
| 4 | 4.91 s | 3.7× | 9.70 s | 3.9× | 8.72 s | 2.9× | 17.77 s | 3.0× |
| 8 | 3.01 s | 6.0× | 5.52 s | 6.9× | 4.48 s | 5.7× | 8.70 s | 6.2× |
| 12 | 2.23 s | 8.1× | 4.03 s | 9.4× | 3.20 s | 8.0× | 6.16 s | 8.8× |
| 16 | 1.89 s | 9.6× | 3.30 s | 11.5× | 2.47 s | 10.4× | 4.75 s | 11.4× |
| 20 | 1.66 s | 10.9× | 2.85 s | 13.2× | 2.13 s | 12.1× | 3.91 s | 13.8× |
| 24 | 1.48 s | 12.3× | 2.52 s | 15.0× | 1.85 s | 13.9× | 3.43 s | 15.7× |
| 28 | 1.47 s | 12.3× | 2.36 s | 16.0× | 1.74 s | 14.8× | 3.05 s | 17.7× |
| 32 | 1.47 s | 12.3× | 2.32 s | 16.3× | 1.56 s | 16.5× | 2.77 s | 19.5× |
| 36 | 1.37 s | 13.2× | 2.13 s | 17.7× | 1.58 s | 16.3× | 2.64 s | 20.5× |
| 40 | 1.29 s | 14.1× | 1.98 s | 19.1× | 1.42 s | 18.1× | 2.54 s | 21.3× |
| 44 | 1.23 s | 14.7× | 1.91 s | 19.8× | 1.36 s | 19.0× | 2.36 s | 22.8× |
| 48 | **1.12 s** | 16.2× | **1.77 s** | 21.3× | 1.26 s | 20.4× | 2.25 s | 24.0× |

The trees are CART 1,611 nodes (depth 19, training accuracy 71.79%) and C4.5 1,159,295 nodes (depth 71, 87.88%) at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

### Observations

- **With all 48 threads epyc is faster:** CART 1.12 against 1.26 s, C4.5 1.77 against 2.25 s.
- **On one thread dell is slower:** 25.7 against 18.2 s (CART) and 54.0 against 37.8 s (C4.5), 1.4× epyc's time (1.8–1.9× before the serial rerun, when epyc's serial path was faster on HIGGS). That is why dell's speedups look better (20–24× against 16–21× at 48 threads). The parallel times are closer.
  - Why dell's single thread is slower was not investigated. Its governor (`powersave` with intel_pstate) still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc stalls where the second socket comes in:** CART takes 1.48 s at 24 threads and 1.47 s at 28 and 32. After that it gains again, to 1.12 s at 48. dell, whose threads span both sockets from the start, has no such step; it has a small dip at 36 threads (1.56 → 1.58 s on CART), seen in every run of this benchmark.
- **Efficiency falls steadily:** at 48 threads it is 34% (CART) and 44% (C4.5) on epyc, and 43% and 50% on dell (each against its own serial time). The big nodes near the root are bound by memory bandwidth, and CART's 1,611-node tree has little else to share out.
- **C4.5 scales better than CART** on epyc (21.3× against 16.2× at 48). Its 1.16-million-node tree has many more subtrees to grow in parallel. Its pruning runs on all threads too: 2.12 s serial, 0.19 s on 48 threads (epyc).
- **Against the previous run** (2026-10-15 at 16:54, commit 35cf315; serial times as before the rerun), little changed: the CPU-side changes of this commit help most with few features and many rows (airline), and HIGGS has 28 features.
  - **C4.5** is 1–3% faster up to 28 threads on epyc (serial 31.13 → 30.50 s) and within 2% from 32 on (48 threads: 1.77 s both). dell is within −3% to +4%, except 4 threads (16.75 → 17.77 s, whose runs spread over 17.0–18.6 s).
  - **CART** is 1–7% slower from 12 threads on, on both nodes (and dell's 4 threads 8%) (epyc 36 threads 1.29 → 1.37 s, 48 threads 1.11 → 1.12 s; dell 48 threads 1.23 → 1.26 s). Its CPU code did not change apart from counting the root's classes in parallel, and the previous run showed swings of the same size between commits with the same CART code; this was not checked with a same-job comparison on HIGGS.
- **Memory:** a constant 5.7 GiB (CART) and 6.0–6.1 GiB (C4.5) at every thread count, the same on both nodes: both backends keep a second copy of the columns for the out-of-place partition (serial needed 3.7 and 4.0 GiB before the rerun). C4.5 no longer keeps a copy of the sorted values (5.0 GiB serial and 7.1–7.2 GiB parallel before).

## Airline

![Thread scaling on epyc1, airline](chart-epyc-airline.png)

![Thread scaling on dell1, airline](chart-dell-airline.png)

Training time, median of 3 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 76.38 s | 1.0× | 178.63 s | 1.0× | 128.09 s | 1.0× | 264.14 s | 1.0× |
| 2 | 41.41 s | 1.8× | 92.39 s | 1.9× | 107.54 s | 1.2× | 201.80 s | 1.3× |
| 4 | 21.54 s | 3.5× | 47.60 s | 3.8× | 60.15 s | 2.1× | 109.09 s | 2.4× |
| 8 | 16.86 s | 4.5× | 32.43 s | 5.5× | 31.19 s | 4.1× | 56.58 s | 4.7× |
| 12 | 11.80 s | 6.5× | 22.43 s | 8.0× | 20.88 s | 6.1× | 38.51 s | 6.9× |
| 16 | 10.75 s | 7.1× | 19.60 s | 9.1× | 16.44 s | 7.8× | 29.95 s | 8.8× |
| 20 | 9.28 s | 8.2× | 16.98 s | 10.5× | 13.73 s | 9.3× | 25.37 s | 10.4× |
| 24 | 8.27 s | 9.2× | 15.01 s | 11.9× | 11.80 s | 10.9× | 21.83 s | 12.1× |
| 28 | 9.34 s | 8.2× | 15.98 s | 11.2× | 10.58 s | 12.1× | 19.47 s | 13.6× |
| 32 | 8.93 s | 8.6× | 15.57 s | 11.5× | 9.53 s | 13.4× | 17.45 s | 15.1× |
| 36 | 8.26 s | 9.2× | 14.12 s | 12.7× | 8.94 s | 14.3× | 16.30 s | 16.2× |
| 40 | 7.98 s | 9.6× | 13.64 s | 13.1× | 8.38 s | 15.3× | 15.29 s | 17.3× |
| 44 | 7.51 s | 10.2× | 12.82 s | 13.9× | 7.90 s | 16.2× | 14.80 s | 17.9× |
| 48 | **7.16 s** | 10.7× | **11.95 s** | 14.9× | 7.59 s | 16.9× | 13.93 s | 19.0× |

The trees are CART 3,013 nodes (depth 25, training accuracy 71.54%) and C4.5 8,548,725 nodes (depth 72, 87.42%) at every point on both nodes, the same as in [cluster-airlines](../cluster-airlines/BENCHMARK.md), whose 48-thread times these match within 1%.

### Observations

- **Airline's speedups are lower than HIGGS's:** 10.7× (CART) and 14.9× (C4.5) on epyc at 48 threads against 16.2× and 21.3× on HIGGS; 16.9× and 19.0× on dell against 20.4× and 24.0×. The serial rerun made airline's serial runs 1.3–1.9× faster and HIGGS's no faster, so the same parallel times now divide a smaller serial time on airline.
- **From 1 to 2 threads:** 1.8× (CART, 76.4 → 41.4 s) and 1.9× (C4.5) on epyc; only 1.2× and 1.3× on dell (1.7× on HIGGS). dell's first two CPUs sit on different sockets, so half of each thread's memory is on the other socket, which may weigh more with airline's 115M rows; this was not investigated. Before the serial rerun epyc's step was 3.6×: the old serial backend partitioned in place through a byte per row, slow on 115M rows, and the parallel backend did not.
- **epyc again loses time where its second socket comes in:** CART 8.27 s at 24 threads, 9.34 s at 28 and 8.93 s at 32, then down to 7.16 s at 48 (C4.5 15.01 → 15.98 → 15.57 → 11.95 s). dell, whose threads span both sockets from 2 on, keeps improving at every step.
- **At 48 threads the two nodes are close:** CART 7.16 s (epyc) against 7.59 s (dell), C4.5 11.95 against 13.93 s. dell's single thread is 1.7× (CART, 128.1 against 76.4 s) and 1.5× (C4.5) slower than epyc's.
- **Memory:** 29.0 GiB (CART) and 32.6–33.1 GiB (C4.5) at every thread count, the same on both nodes: the serial backend now keeps the second copy of the columns too (20.4 and 24.0 GiB before the rerun).
