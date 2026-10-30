# Thread scaling on the mordor3 CPU servers: ./tree with 1 to 48 threads on HIGGS and MNIST8M

**Status:** run on 2026-10-29/30 at ec77727 (main with the CPU presort of 4fb419c), on epyc1, epyc2 (the same hardware) and dell1, each node alone and used whole. Every tree is grown to depth ≤ 10 without pruning, as in the test-set benchmarks.
- **Runs per point:** 10; MNIST8M at 1 and 2 threads 3 (each over 2 minutes).
- **48 threads and one socket:** the test-set benchmarks' runs ([cluster-higgs](../cluster-higgs/BENCHMARK.md), [cluster-mnist8m](../cluster-mnist8m/BENCHMARK.md)), merged into these folders.
- **Same trees:** every thread count built the same tree on both nodes, the same as in the test-set benchmarks.

The earlier version (HIGGS and airline, pruned trees, 2026-10-20) is in git history.

```bash
s=bench/results/cluster-cpu-scaling
for d in higgs mnist8m; do for n in epyc dell; do
  bench/.venv/bin/python bench/scaling_chart.py $s/cpu-$n-$d --socket bench/results/cluster-$d/socket-$n-$d --out /tmp/$n-$d.png
done; done
bench/.venv/bin/python bench/grid_chart.py /tmp/epyc-higgs.png /tmp/dell-higgs.png /tmp/epyc-mnist8m.png /tmp/dell-mnist8m.png --cols 2 --out $s/chart.png
```

- **Protocols:** `cart_depth10` and `c45_depth10_unpruned`.
- **Threads:** 1, 2, 4, then every 4 up to 48.
  - **1 thread** is the serial backend (`--serial`); more threads use `--parallel --threads N`. Speedup is the serial time divided by the N-thread time, so it includes the parallel backend's own overhead.
  - **Pinning:** N threads are pinned to the first N CPUs in the kernel's order (`--cpus 0-47`). `run.log` records that order (`lscpu`).
- **Measurements:** as in every benchmark here: training time timed inside ./tree (presort, build; no file reading), and peak RSS up to the end of training, from the same process. The legend gives ./tree's training accuracy.

| Node | CPU | RAM | CPU order (`lscpu`) | Governor |
|---|---|---|---|---|
| epyc1, epyc2 | 2× EPYC 9255 (Zen 5), 24 cores each, no SMT | 512 GB DDR5-6400 | CPUs 0–23 on socket 0, 24–47 on socket 1 (one NUMA node each) | `performance` |
| dell1 | 2× Xeon Gold 6542Y (Emerald Rapids), 24 cores each, no SMT | 1 TB DDR5-5200 | CPUs alternate between the sockets and their 4 NUMA nodes (0, 2, 1, 3, 0, …) | `powersave` (intel_pstate, boosts) |

So on epyc up to 24 threads stay on one socket, and the second socket joins from 28 on. On dell1 every thread count from 2 on is spread over both sockets.

All four charts in one image (HIGGS above MNIST8M, epyc left, dell right). Threads and time are on log axes, so ideal scaling (dotted: the 1-thread time divided by the threads) is a straight line:

![Thread scaling on epyc and dell1, HIGGS and MNIST8M](chart.png)

### One socket

The diamonds at 24 threads are the one-socket runs of the test-set benchmarks (`socket-*`): 24 threads on every CPU of one socket, the memory bound to its NUMA nodes.

| | epyc: 24 threads / one socket | dell: 24 threads / one socket |
|---|--:|--:|
| HIGGS CART | 0.83 / 0.83 s | 1.21 / 1.21 s |
| HIGGS C4.5 | 0.94 / 0.93 s | 1.58 / 1.68 s |
| MNIST8M CART | 11.17 / 11.1 s | 15.31 / 16.0 s |
| MNIST8M C4.5 | 10.39 / 10.3 s | 15.18 / 15.5 s |

- **epyc:** the same within 2%. Its first 24 CPUs are one socket, so the 24-thread point already ran on one socket.
- **dell:** within 6%, with one socket no faster, although its 24 threads sit on one socket instead of 12 cores of each.

## HIGGS

10.5M rows, 28 features. Training time, mean of the runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 10.38 s | 1.0× | 11.92 s | 1.0× | 15.56 s | 1.0× | 20.34 s | 1.0× |
| 2 | 5.52 s | 1.9× | 6.32 s | 1.9× | 8.71 s | 1.8× | 12.05 s | 1.7× |
| 4 | 2.84 s | 3.7× | 3.25 s | 3.7× | 5.77 s | 2.7× | 8.31 s | 2.4× |
| 8 | 1.82 s | 5.7× | 1.91 s | 6.2× | 2.81 s | 5.5× | 3.48 s | 5.8× |
| 12 | 1.29 s | 8.1× | 1.41 s | 8.5× | 2.04 s | 7.6× | 2.50 s | 8.1× |
| 16 | 1.08 s | 9.6× | 1.13 s | 10.6× | 1.62 s | 9.6× | 1.93 s | 10.5× |
| 20 | 0.95 s | 11.0× | 1.01 s | 11.8× | 1.38 s | 11.3× | 1.73 s | 11.8× |
| 24 | 0.83 s | 12.5× | 0.94 s | 12.7× | 1.21 s | 12.8× | 1.58 s | 12.9× |
| 28 | 0.82 s | 12.6× | 0.86 s | 13.9× | 1.08 s | 14.5× | 1.27 s | 16.1× |
| 32 | 0.78 s | 13.4× | 0.79 s | 15.1× | 1.00 s | 15.5× | 1.14 s | 17.9× |
| 36 | 0.73 s | 14.2× | 0.75 s | 16.0× | 0.93 s | 16.7× | 1.08 s | 18.8× |
| 40 | 0.70 s | 14.8× | 0.71 s | 16.8× | 0.88 s | 17.7× | 1.04 s | 19.7× |
| 44 | 0.67 s | 15.6× | 0.67 s | 17.9× | 0.83 s | 18.7× | 0.99 s | 20.5× |
| 48 | 0.61 s | 17.1× | 0.59 s | 20.2× | 0.82 s | 18.9× | 0.99 s | 20.5× |

The trees are CART 2,029 nodes (training accuracy 70.63%) and C4.5 935 nodes (63.76%), depth 10, at every point on both nodes. `scaling.csv` in each run directory has every point with its min–max, efficiency and peak RSS.

- **With all 48 threads epyc is faster:** CART 0.61 against 0.82 s, C4.5 0.59 against 0.99 s.
- **On one thread dell is 1.5–1.7× slower** (15.6 against 10.4 s, 20.3 against 11.9 s), so its speedups (19–21×) look as good as epyc's (17–20×) though its times are not. Why was not investigated: its governor still boosts, and a single thread's memory is local to one of four NUMA nodes.
- **epyc gains little where the second socket comes in:** CART 0.83 s at 24 threads, 0.82 s at 28, then down to 0.61 s at 48.
- **Efficiency at 48 threads:** 36% (CART) and 42% (C4.5) on epyc, 39% and 43% on dell. The big nodes near the root are bound by memory bandwidth.
- **Memory:** 3.6 GiB serial, rising with the spare column slots to 5.6 GiB from 28 threads on (27 spare slots for 28 features).

## MNIST8M

8.1M rows, 784 features, 10 classes. Training time, mean of the runs, and speedup over 1 thread:

| Threads | epyc CART | speedup | epyc C4.5 | speedup | dell CART | speedup | dell C4.5 | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 (serial) | 235.16 s | 1.0× | 210.34 s | 1.0× | 314.83 s | 1.0× | 293.61 s | 1.0× |
| 2 | 118.18 s | 2.0× | 106.89 s | 2.0× | 167.23 s | 1.9× | 160.44 s | 1.8× |
| 4 | 59.33 s | 4.0× | 53.61 s | 3.9× | 86.55 s | 3.6× | 84.60 s | 3.5× |
| 8 | 30.18 s | 7.8× | 27.35 s | 7.7× | 43.20 s | 7.3× | 41.89 s | 7.0× |
| 12 | 20.46 s | 11.5× | 18.69 s | 11.3× | 29.07 s | 10.8× | 28.40 s | 10.3× |
| 16 | 15.75 s | 14.9× | 14.39 s | 14.6× | 22.01 s | 14.3× | 21.71 s | 13.5× |
| 20 | 12.95 s | 18.2× | 11.98 s | 17.6× | 18.00 s | 17.5× | 17.72 s | 16.6× |
| 24 | 11.17 s | 21.1× | 10.39 s | 20.2× | 15.31 s | 20.6× | 15.18 s | 19.3× |
| 28 | 9.75 s | 24.1× | 9.17 s | 22.9× | 13.60 s | 23.1× | 13.59 s | 21.6× |
| 32 | 8.95 s | 26.3× | 8.18 s | 25.7× | 12.10 s | 26.0× | 12.19 s | 24.1× |
| 36 | 7.75 s | 30.4× | 7.37 s | 28.6× | 11.05 s | 28.5× | 11.20 s | 26.2× |
| 40 | 7.03 s | 33.5× | 6.67 s | 31.6× | 10.39 s | 30.3× | 10.49 s | 28.0× |
| 44 | 6.45 s | 36.5× | 6.09 s | 34.6× | 9.82 s | 32.0× | 9.83 s | 29.9× |
| 48 | 5.92 s | 39.7× | 5.64 s | 37.3× | 9.28 s | 33.9× | 9.44 s | 31.1× |

The trees are CART 2,047 nodes (training accuracy 80.77%) and C4.5 1,453 nodes (70.96%), depth 10, at every point on both nodes.

- **Nearly linear to 48 threads on epyc:** 39.7× (CART) and 37.3× (C4.5), 83% and 78% efficiency, against 36–42% on HIGGS. With 784 features there are many columns to share among the threads.
- **dell scales a little less:** 33.9× and 31.1×, 71% and 65% efficiency, and takes 1.6–1.7× epyc's time at 48 threads.
- **No step at the second socket:** both nodes keep gaining at every thread count.
- **Memory:** 71.2 GiB serial to 74.0 GiB at 48 threads: the spare slots are small against the data.

## dell1: two oddities, checked on 2026-10-30

**The 4-thread point on HIGGS is noisy.** On dell its runs spread over 4.8–7.1 s (CART) and 6.3–10.0 s (C4.5), against under 10% at 2 and 8 threads, and the mean sits above the line. A repeat on the idle node gave the same spread (4.7–6.9 s, 6.2–8.1 s: `dell-ab/repeat-4-threads`), and so did the previous build (below). Presort and build both vary, the user CPU time does not. The cause was not found.

**One socket looked 27–35% slower than the previous build on HIGGS and SUSY** (HIGGS CART 0.95 → 1.21 s, SUSY CART 0.32 → 0.43 s), on the same socket, and not on airline, ECBDL or MNIST8M. It is not the code: alternating this build with the previous one (`tree_cpu3`, 67495f9 plus the waits of 8bdf641) on dell1, 10 runs of each, twice, this build is 1–3% faster. Both sit between the two batches' numbers (HIGGS CART 0.95 and 1.21 s), so the difference is dell1's state between batches (`dell-ab/`, new-1, old-1, new-2, old-2). The tables keep the scheduled batch's runs.

| Mean of 10, two rounds | ec77727 | previous build |
|---|--:|--:|
| one socket, HIGGS CART | 1.06, 1.07 s | 1.08, 1.09 s |
| one socket, HIGGS C4.5 | 1.45, 1.47 s | 1.47, 1.49 s |
| one socket, SUSY CART | 0.36, 0.36 s | 0.37, 0.37 s |
| one socket, SUSY C4.5 | 0.39, 0.39 s | 0.40, 0.40 s |
| 4 threads, HIGGS CART | 5.48, 5.14 s | 5.69, 5.78 s |
| 4 threads, HIGGS C4.5 | 7.19, 7.10 s | 6.82, 7.96 s |
