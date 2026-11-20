# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS and MNIST8M

**Final data (2026-11-19/20):** one build of main (81bf6e6c, the same tree code as the final runs of 2026-11-17; `bench/tools/cluster/final.sh`), 10 runs per point (the chart's footnote gives the count), on epyc2 and dell2 (a 512 GB node, 8 of dell1's 16 memory channels: from 2026-11-17 the dell runs are on dell2–4, [bench/tools/cluster/README.md](../tools/cluster/README.md)). MNIST8M ran in parts of under 4 h each (epyc2: 2, dell2: 3; `bench/tools/cluster/merge_parts.py`, the parts in `machine.json`). The runs before (2026-11-15/16, f53347a9, 5 runs per point, dell on dell1) are in git at 84b4c67e; the ones before them (f1fe270, 10 runs per point, 2026-11-09/10) at 1e655067. Every tree is grown to depth ≤ 10 without pruning, as in the test-set benchmarks; every thread count built the same tree on both nodes.

![Speed-up over one thread on epyc2 and dell2, HIGGS and MNIST8M, CART and C4.5](chart.png)

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
| epyc2 | 2× EPYC 9255 (Zen 5), 24 cores each, no SMT | 512 GB DDR5-6400 | CPUs 0–23 on socket 0, 24–47 on socket 1 (one NUMA node each) | `performance` |
| dell2 | 2× Xeon Gold 6542Y (Emerald Rapids), 24 cores each, no SMT | 512 GB DDR5-5200 (8 × 64 GB) | CPUs alternate between the sockets and their 4 NUMA nodes (0, 2, 1, 3, 0, …) | `powersave` (intel_pstate, boosts) |

So on epyc up to 24 threads stay on one socket, and the second socket joins from 28 on (the dotted line in the chart). On dell every thread count from 2 on is spread over both sockets.

## HIGGS

10.5M rows, 28 features. Training time, mean of 10 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 7.90 s | 1.0× | 7.49 s | 1.0× | 14.81 s | 1.0× | 13.37 s | 1.0× |
| 2 | 4.04 s | 2.0× | 4.03 s | 1.9× | 8.61 s | 1.7× | 9.08 s | 1.5× |
| 4 | 2.08 s | 3.8× | 2.08 s | 3.6× | 5.03 s | 2.9× | 5.57 s | 2.4× |
| 8 | 1.14 s | 7.0× | 1.16 s | 6.5× | 2.43 s | 6.1× | 2.66 s | 5.0× |
| 12 | 0.78 s | 10.1× | 0.78 s | 9.6× | 1.65 s | 9.0× | 1.75 s | 7.6× |
| 16 | 0.65 s | 12.2× | 0.69 s | 10.9× | 1.28 s | 11.5× | 1.39 s | 9.6× |
| 20 | 0.56 s | 14.1× | 0.57 s | 13.0× | 1.09 s | 13.6× | 1.23 s | 10.9× |
| 24 | 0.48 s | 16.5× | 0.51 s | 14.8× | 0.97 s | 15.2× | 1.08 s | 12.4× |
| 28 | 0.45 s | 17.6× | 0.44 s | 17.0× | 0.86 s | 17.2× | 0.94 s | 14.2× |
| 32 | 0.42 s | 18.8× | 0.44 s | 16.9× | 0.81 s | 18.4× | 0.91 s | 14.7× |
| 36 | 0.40 s | 19.6× | 0.42 s | 18.0× | 0.82 s | 18.1× | 0.91 s | 14.8× |
| 40 | 0.40 s | 19.9× | 0.41 s | 18.1× | 0.77 s | 19.3× | 0.86 s | 15.6× |
| 44 | 0.35 s | 22.4× | 0.42 s | 17.8× | 0.73 s | 20.4× | 0.84 s | 15.9× |
| 48 | 0.35 s | 22.8× | 0.40 s | 18.6× | 0.71 s | 20.9× | 0.84 s | 15.8× |

The trees are CART 2,029 nodes (training accuracy 70.63%) and C4.5 935 nodes (63.76%), depth 10, at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

- **With all 48 threads epyc is faster:** CART 0.35 against 0.71 s, C4.5 0.40 against 0.84 s (dell1 took 0.54 and 0.67 s; the runs before f53347a9: 0.63 and 0.80 s, 0.63 and 1.11 s).
- **On one thread dell is 1.8–1.9× slower** (CART 14.8 against 7.9 s, C4.5 13.4 against 7.5 s), and 1.6–2.4× with every thread count. dell's speed-ups at 48 threads are 20.9× (CART) and 15.8× (C4.5), epyc's 22.8× and 18.6×. Why dell is slower was not investigated: its governor still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc's step at the second socket is small:** 16.5× at 24 threads, 17.6× at 28, then 18.8–22.8× to 48. C4.5 stops gaining at 28 threads (17.0×), 18.6× at 48.
- **Efficiency at 48 threads:** 47% (CART) and 39% (C4.5) on epyc, 44% and 33% on dell2 (dell1: 49% and 39%). The big nodes near the root are bound by memory bandwidth, and at 0.35–0.84 s the fixed costs show.
- **Spread** (min–max over the mean, `scaling.csv`): on epyc under 10% up to 24 threads (C4.5 at 8 and 16 threads 17% and 28%: one slow run each) and 13–34% from 28 threads on (0.35–0.45 s, a few tens of ms); on dell 30–69% at 1–12 threads, 8–35% from 16 on ([below](#dell-the-noisy-low-thread-counts)).
- **Memory:** 3.5 GiB serial, rising with the spare column slots to 5.6–5.7 GiB from 28 threads on (27 spare slots for 28 features).

## MNIST8M

8.1M rows, 784 features, 10 classes. Training time, mean of 10 runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 115.11 s | 1.0× | 113.94 s | 1.0× | 202.58 s | 1.0× | 204.45 s | 1.0× |
| 2 | 56.23 s | 2.0× | 57.42 s | 2.0× | 102.12 s | 2.0× | 108.86 s | 1.9× |
| 4 | 28.38 s | 4.1× | 29.08 s | 3.9× | 54.93 s | 3.7× | 58.78 s | 3.5× |
| 8 | 14.50 s | 7.9× | 14.93 s | 7.6× | 29.81 s | 6.8× | 31.54 s | 6.5× |
| 12 | 9.94 s | 11.6× | 10.23 s | 11.1× | 20.38 s | 9.9× | 21.65 s | 9.4× |
| 16 | 7.68 s | 15.0× | 7.90 s | 14.4× | 15.80 s | 12.8× | 17.11 s | 11.9× |
| 20 | 6.36 s | 18.1× | 6.59 s | 17.3× | 12.76 s | 15.9× | 13.89 s | 14.7× |
| 24 | 5.57 s | 20.7× | 5.81 s | 19.6× | 11.13 s | 18.2× | 12.26 s | 16.7× |
| 28 | 5.01 s | 23.0× | 5.08 s | 22.4× | 10.04 s | 20.2× | 11.01 s | 18.6× |
| 32 | 4.50 s | 25.6× | 4.79 s | 23.8× | 9.12 s | 22.2× | 10.27 s | 19.9× |
| 36 | 4.17 s | 27.6× | 4.37 s | 26.1× | 8.52 s | 23.8× | 9.59 s | 21.3× |
| 40 | 3.86 s | 29.8× | 4.09 s | 27.9× | 8.14 s | 24.9× | 9.14 s | 22.4× |
| 44 | 3.47 s | 33.2× | 3.71 s | 30.7× | 7.75 s | 26.1× | 8.76 s | 23.3× |
| 48 | 3.30 s | 34.9× | 3.47 s | 32.8× | 7.56 s | 26.8× | 8.61 s | 23.7× |

The trees are CART 2,047 nodes (training accuracy 80.77%) and C4.5 1,453 nodes (70.96%), depth 10, at every point on both nodes.

- **Nearly linear to 48 threads on epyc:** 34.9× (CART) and 32.8× (C4.5), 73% and 68% efficiency. dell2 reaches 26.8× and 23.7× (56% and 49%), against dell1's 36.2× and 33.6× with twice the memory channels: up to 2 threads dell2 keeps up (2.0×), from 4 threads on it falls behind. With 784 features there are many columns to share among the threads.
- **dell takes 1.8× epyc's time on one thread and 2.3–2.5× at 48 threads** (CART 7.56 against 3.30 s; dell1 took 5.53 s).
- **No step at the second socket:** both nodes keep gaining at every thread count. On dell every point spreads under 12%; on epyc under 2% up to 24 threads and 5–23% from 28 on.
- **Against the runs before (f1fe270):** 2.0× faster serial on epyc (234 → 115 s CART), 1.8× at 48 threads (5.92 → 3.30 s).
- **Memory:** 71.1 GiB serial to 74.0 GiB at 48 threads: the spare slots are small against the data.

## dell: the noisy low thread counts

On dell the HIGGS runs at 1 to 12 threads spread much more than on epyc: on dell2 (2026-11-19) 30–33% on one thread, 44–69% at 2–8 threads and 36–39% at 12, against at most 3% on epyc up to 12 threads (C4.5 at 8: 17%, one slow run). On dell1 it was the same: in the first run of 2026-11-15 the spread was 29–46% at 2–8 threads, with a few minutes (13:48–13:55) in which every run was slow, presort included, and the rerun of 2026-11-16 still spread 34–49% at 4 and 8 threads. The slow runs are slow in presort and build alike, so the whole run is slower, not one phase: as on 2026-10-30 (4 threads: 4.8–7.1 s CART), when the cause was not found ([docs/PERFORMANCE.md](../../docs/PERFORMANCE.md)). With few threads on dell they land on several of the four NUMA nodes (the CPU order alternates), which may be the cause. MNIST8M, whose runs are 10–200 s, spreads under 12% on dell. The A/B runs of the earlier check are in git at 3354a8a (`dell-ab/`).
