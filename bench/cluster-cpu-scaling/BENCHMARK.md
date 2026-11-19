# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS and MNIST8M

**Final data (2026-11-15/16):** one build of main (f53347a9, `src/` = d2957ff2; `bench/tools/cluster/final.sh -m 5`), 5 runs per point (the chart's footnote gives the count), run with every other benchmark of the paper. dell's HIGGS runs are a rerun of 2026-11-16 (job 161): the first had a few minutes in which every run of dell1 was slow, presort included. The runs before (f1fe270, 10 runs per point, 2026-11-09/10) are in git at 1e655067. Every tree is grown to depth ≤ 10 without pruning, as in the test-set benchmarks; every thread count built the same tree on both nodes.

**Out of date:** dell's points (⚠, faded on the chart) are from dell1 (1 TB, every memory channel). From 2026-11-17 the dell runs are on a 512 GB node (dell2–4: 8 of 16 channels, 211 GB/s against dell1's 379 GB/s; [bench/tools/cluster/README.md](../tools/cluster/README.md)); the dell scaling is rerun on dell4.

![Speed-up over one thread on epyc1 and dell1, HIGGS and MNIST8M, CART and C4.5](chart.png)

```bash
bench/tools/cluster/final.sh            # queues these runs with all the others (cpu-epyc-<data>, cpu-dell-<data>)
bench/tools/.venv/bin/python bench/tools/scaling_chart.py bench/cluster-cpu-scaling   # chart.png, <run>/scaling.csv
```

- **Protocols:** `cart_depth10` and `c45_depth10_unpruned`.
- **Threads:** 1, 2, 4, then every 4 up to 48.
  - **1 thread** is the serial backend (`--serial`); more threads use `--multicore --threads N`. Speed-up is the serial time divided by the N-thread time, so it includes the parallel backend's own overhead. Efficiency is the speed-up divided by N.
  - **Pinning:** N threads are pinned to the first N CPUs in the kernel's order (`--cpus 0-47`), memory first-touch. `run.log` records that order (`lscpu`).
- **Measurements:** as in every benchmark here: training time timed inside ./tree (presort, build; no file reading), and peak RSS up to the end of training, from the same process. `scaling.csv` in each run directory has every point with its min–max, speed-up, efficiency and peak RSS.

| Node | CPU | RAM | CPU order (`lscpu`) | Governor |
|---|---|---|---|---|
| epyc1 | 2× EPYC 9255 (Zen 5), 24 cores each, no SMT | 512 GB DDR5-6400 | CPUs 0–23 on socket 0, 24–47 on socket 1 (one NUMA node each) | `performance` |
| dell1 | 2× Xeon Gold 6542Y (Emerald Rapids), 24 cores each, no SMT | 1 TB DDR5-5200 | CPUs alternate between the sockets and their 4 NUMA nodes (0, 2, 1, 3, 0, …) | `powersave` (intel_pstate, boosts) |

So on epyc up to 24 threads stay on one socket, and the second socket joins from 28 on (the dotted line in the chart). On dell1 every thread count from 2 on is spread over both sockets.

## HIGGS

10.5M rows, 28 features. Training time, mean of 5 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 7.82 s | 1.0× | 7.34 s | 1.0× | 12.66 s | 1.0× | 12.53 s | 1.0× |
| 2 | 4.00 s | 2.0× | 3.96 s | 1.9× | 6.67 s | 1.9× | 7.42 s | 1.7× |
| 4 | 2.06 s | 3.8× | 2.05 s | 3.6× | 4.21 s | 3.0× | 4.75 s | 2.6× |
| 8 | 1.11 s | 7.0× | 1.10 s | 6.7× | 2.22 s | 5.7× | 2.43 s | 5.1× |
| 12 | 0.77 s | 10.2× | 0.76 s | 9.7× | 1.48 s | 8.5× | 1.59 s | 7.9× |
| 16 | 0.62 s | 12.6× | 0.61 s | 12.0× | 1.17 s | 10.8× | 1.22 s | 10.3× |
| 20 | 0.54 s | 14.4× | 0.56 s | 13.2× | 0.99 s | 12.8× | 1.09 s | 11.5× |
| 24 | 0.47 s | 16.8× | 0.49 s | 14.9× | 0.85 s | 14.9× | 0.94 s | 13.3× |
| 28 | 0.43 s | 18.1× | 0.41 s | 17.7× | 0.75 s | 17.0× | 0.77 s | 16.3× |
| 32 | 0.44 s | 17.8× | 0.40 s | 18.3× | 0.68 s | 18.6× | 0.73 s | 17.1× |
| 36 | 0.38 s | 20.8× | 0.39 s | 18.7× | 0.63 s | 20.0× | 0.71 s | 17.8× |
| 40 | 0.35 s | 22.1× | 0.40 s | 18.2× | 0.60 s | 21.0× | 0.68 s | 18.4× |
| 44 | 0.36 s | 21.9× | 0.40 s | 18.3× | 0.58 s | 21.8× | 0.66 s | 19.1× |
| 48 | 0.33 s | 23.6× | 0.37 s | 20.1× | 0.54 s | 23.6× | 0.67 s | 18.8× |

The trees are CART 2,029 nodes (training accuracy 70.63%) and C4.5 935 nodes (63.76%), depth 10, at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

- **With all 48 threads epyc is faster:** CART 0.33 against 0.54 s, C4.5 0.37 against 0.67 s (the runs before: 0.63 and 0.80 s, 0.63 and 1.11 s).
- **On one thread dell is 1.6–1.7× slower** (CART 12.7 against 7.8 s, C4.5 12.5 against 7.3 s), and with every thread count; the speed-ups at 48 threads are the same on both nodes for CART (23.6×), 20.1× and 18.8× for C4.5. Why dell is slower was not investigated: its governor still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc's step at the second socket is small now:** 16.8× at 24 threads, 18.1× at 28, then about flat to 44 threads (17.8–22.1×) and 23.6× at 48. C4.5 stops gaining at 32 threads (18.3×) until 48 (20.1×).
- **Efficiency at 48 threads:** 49% (CART) and 42% (C4.5) on epyc, 49% and 39% on dell (35–43% before). The big nodes near the root are bound by memory bandwidth, and at 0.3–0.7 s the fixed costs show.
- **Spread:** under 0.1 s everywhere, but 11–33% of the mean from 32 threads on epyc (0.35–0.44 s, a few tens of ms) and 34–49% at 4 and 8 threads on dell (min–max of the 5 runs, `scaling.csv`; [below](#dell1-the-noisy-low-thread-counts)).
- **Memory:** 3.5 GiB serial, rising with the spare column slots to 5.6–5.7 GiB from 28 threads on (27 spare slots for 28 features).

## MNIST8M

8.1M rows, 784 features, 10 classes. Training time, mean of 5 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 114.89 s | 1.0× | 113.15 s | 1.0× | 200.13 s | 1.0× | 203.22 s | 1.0× |
| 2 | 55.93 s | 2.1× | 57.26 s | 2.0× | 94.71 s | 2.1× | 101.26 s | 2.0× |
| 4 | 28.25 s | 4.1× | 28.96 s | 3.9× | 48.63 s | 4.1× | 52.33 s | 3.9× |
| 8 | 14.45 s | 7.9× | 14.86 s | 7.6× | 24.78 s | 8.1× | 26.81 s | 7.6× |
| 12 | 9.92 s | 11.6× | 10.21 s | 11.1× | 16.98 s | 11.8× | 18.29 s | 11.1× |
| 16 | 7.66 s | 15.0× | 7.87 s | 14.4× | 12.93 s | 15.5× | 13.88 s | 14.6× |
| 20 | 6.32 s | 18.2× | 6.58 s | 17.2× | 10.80 s | 18.5× | 11.42 s | 17.8× |
| 24 | 5.55 s | 20.7× | 5.80 s | 19.5× | 9.25 s | 21.6× | 9.94 s | 20.4× |
| 28 | 4.89 s | 23.5× | 5.11 s | 22.2× | 8.19 s | 24.4× | 8.93 s | 22.8× |
| 32 | 4.47 s | 25.7× | 4.64 s | 24.4× | 7.31 s | 27.4× | 7.99 s | 25.4× |
| 36 | 4.04 s | 28.5× | 4.18 s | 27.1× | 6.80 s | 29.4× | 7.30 s | 27.9× |
| 40 | 3.64 s | 31.5× | 3.75 s | 30.2× | 6.32 s | 31.7× | 6.92 s | 29.4× |
| 44 | 3.34 s | 34.4× | 3.52 s | 32.2× | 5.78 s | 34.6× | 6.46 s | 31.4× |
| 48 | 3.11 s | 37.0× | 3.27 s | 34.6× | 5.53 s | 36.2× | 6.04 s | 33.6× |

The trees are CART 2,047 nodes (training accuracy 80.77%) and C4.5 1,453 nodes (70.96%), depth 10, at every point on both nodes.

- **Nearly linear to 48 threads on both nodes:** 37.0× (CART) and 34.6× (C4.5) on epyc, 77% and 72% efficiency; dell 36.2× and 33.6×, 75% and 70% (dell scaled less before: 34.2× and 31.7×). With 784 features there are many columns to share among the threads.
- **dell takes 1.7–1.8× epyc's time** at 1 and at 48 threads (CART 5.53 against 3.11 s at 48).
- **No step at the second socket:** both nodes keep gaining at every thread count. Every point spreads under 10%.
- **Against the runs before (f1fe270):** 2.0× faster serial on epyc (234 → 115 s CART), 1.9× at 48 threads (5.92 → 3.11 s); dell 1.6× and 1.7×.
- **Memory:** 71.1 GiB serial to 74.0 GiB at 48 threads: the spare slots are small against the data.

## dell1: the noisy low thread counts

On dell1 the HIGGS runs at 2 to 16 threads spread more than on epyc. In the first run of 2026-11-15 the spread was 29–46% at 2–8 threads, with a few minutes (13:48–13:55) in which every run was slow, presort included. The rerun of 2026-11-16 (the data above) still spreads 34–49% at 4 and 8 threads, 12–16% at 12 and 16, while 2 threads became steady (4–7%). The slow runs are slow in presort and build alike, so the whole run is slower, not one phase: as on 2026-10-30 (4 threads: 4.8–7.1 s CART), when the cause was not found ([docs/PERFORMANCE.md](../../docs/PERFORMANCE.md)). With few threads on dell1 they land on several of the four NUMA nodes (the CPU order alternates), which may be the cause; compare epyc, whose runs up to 24 threads spread at most 5%. The A/B runs of the earlier check are in git at 3354a8a (`dell-ab/`).
