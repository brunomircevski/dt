# Chess 3×3 on the mordor3 cluster: 1M to 1000M rows on two GPU and two CPU servers, CART and C4.5 at depth ≤ 10 without pruning

- **Runs:** every case on hopper and ampere (CUDA `--hybrid` and `--gpu`), and on epyc and dell (`--multicore` on both sockets, `--serial`, `--simple`). 5 runs a case, `--simple` 10 (the tables give the count in brackets, the charts on every bar or legend entry).
- **Builds:** ./tree at f53347a9 (`src/` = d2957ff2), one build for every node, run on 2026-11-15/16 by `bench/cluster/final.sh -m 5` ([bench/cluster/README.md](../../cluster/README.md#the-final-rerun)). `--simple` was not rerun (its code has not changed): its runs are from 2026-11-11/12, f1fe270e on epyc and ce907ddf on dell. `cpu-dell-chess_1m` and `cpu-dell-chess_10m` are a rerun of 2026-11-16 on dell1: the first runs, in the middle of dell1's batch, took 1.5–2× as long (chess 10M CART 115–147 ms, presort 62–78 ms), while the old and new builds alternated on dell1 gave 53–88 ms (new) and 81–126 ms (old); the rerun, with the batch's 30 s warm-up, 68–111 ms. The runs before (f1fe270e, 10 runs a case, 2026-11-09/12) are in git at 1e655067.
- **Each node ran alone** and was reserved whole (`--exclusive`). The GPU runs use one GPU and every CPU of the node, the GPU's socket first: hopper's GPU 2 (NUMA node 1, CPUs 16–31, then 0–15), ampere's GPU 0 (NUMA node 0, CPUs 0–47), memory first-touch (`run.log`).
- **Same trees:** every case built the same tree on every node and backend (`--hash-trees`), the same as in the runs before: 17 nodes (9 leaves), depth 4, 100% training accuracy, at every size. That is the exact tree of a 3×3 board.
- **Settings:** ./tree's defaults. `cart_depth10` and `c45_depth10_unpruned`, like every other test-set benchmark: depth ≤ 10 without pruning. The tree stops at depth 4, so the limit never acts. Chess has no test set since f2a4f921: the tree is exact, and the training accuracy is reported.

How the runs are made, and where the data came from: [bench/cluster/README.md](../../cluster/README.md).

The charts:
- `chart-chess_1m.png`, `chart-chess_10m.png`, `chart-chess_100m.png`, `chart-chess_1b.png`: one `testset_chart.py` chart per size. A bar is the mean of the runs, the whisker runs from the fastest to the slowest run, and its label gives the mean ± standard deviation and the number of runs.
- `chart-rows.png`: `rows_chart.py`, means against the rows, with whiskers from the fastest to the slowest run (without `--simple`); the legend gives the runs per line.

**The dataset** is a 3×3 checkerboard: two features, x and y, uniform in [0, 3] with six decimals. The class alternates over the nine unit squares (on five of them: 55.5% of the rows), with no noise. Four sizes:

| Dataset | Training rows | CSV | Note |
|---|--:|--:|---|
| `chess_1m` | 1,000,000 | 20 MB | the first 1M rows of `chess_2b` (`./tree --rows`) |
| `chess_10m` | 10,000,000 | 200 MB | the first 10M rows of `chess_2b` |
| `chess_100m` | 100,000,000 | 2.0 GB | the first 100M rows of `chess_2b` |
| `chess_1b` | 1,000,000,000 | 20.8 GB | its own file: 100M distinct rows, each 10 times |

`chess_2b` (2,000M rows, 41.6 GB) and its other prefixes (50M, 500M) are used by [cluster-profiling](../cluster-profiling/BENCHMARK.md). The 1M, 10M and 100M trees hash as with the earlier copied files.

**What it measures.** The tree is tiny (4 levels, 8 splits), so a run is the presort plus a few full sweeps over all rows: time per row at the top levels, without the many small nodes of a deep tree. With two features, the CPU backend has only two columns to share among 48 threads.

| Run | Node | CPU | RAM | GPU | Backend |
|---|---|---|---|---|---|
| `cuda-hopper-*` | hopper | 2× Xeon Gold 6444Y, 32 cores | 1 TB DDR5-4800 | H100 PCIe 80 GB (GPU 2) | `--hybrid`, 32 CPU threads |
| `cuda-ampere-*` | ampere | 2× EPYC 7402, 48 cores | 1 TB DDR4-3200 | A100 80 GB PCIe (GPU 0) | `--hybrid`, 48 CPU threads |
| `gpuonly-hopper-*`, `gpuonly-ampere-*` | hopper, ampere | as above | | as above | `--gpu` (on the same CPUs) |
| `cpu-epyc-*` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – | `--multicore` (48 threads) |
| `cpu-dell-*` | dell1 | 2× Xeon Gold 6542Y, 48 cores | 1 TB DDR5-5200 (16 × 64 GB) | – | `--multicore` (48 threads) |
| `serial-epyc-*`, `simple-epyc-*` | epyc1 or epyc2 (the same hardware) | | | – | `--serial`, `--simple` (1 thread, the whole node reserved) |
| `serial-dell-*`, `simple-dell-*` | dell1, dell2 or dell3 (the same CPUs; dell2/3 512 GB) | | | – | `--serial`, `--simple` (1 thread, the whole node reserved) |

## Time and memory against the rows

![time, RAM and VRAM against the training rows](chart-rows.png)

Training time (`train total`: presort and build, on CUDA also GPU setup and upload; not reading the CSV), mean of the runs (runs in brackets):

| CART, no pruning | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| CUDA hybrid · hopper (H100 PCIe + 32 threads) | 46 ms (5) | 50 ms (5) | 74 ms (5) | 615 ms (5) |
| CUDA hybrid · ampere (A100 80GB PCIe + 48 threads) | 20 ms (5) | 32 ms (5) | 107 ms (5) | 1.02 s (5) |
| GPU only · hopper | 48 ms (5) | 46 ms (5) | 75 ms (5) | **609 ms** (5) |
| GPU only · ampere | 20 ms (5) | **30 ms** (5) | 102 ms (5) | 1.01 s (5) |
| Multicore · epyc (EPYC 9255, 48 threads) | **15 ms** (5) | 54 ms (5) | 209 ms (5) | 1.58 s (5) |
| Multicore · dell (Xeon 6542Y, 48 threads) | 41 ms (5) | 84 ms (5) | 388 ms (5) | 2.70 s (5) |
| Serial · epyc (1 thread) | 18 ms (5) | 224 ms (5) | 2.07 s (5) | 39.01 s (5) |
| Serial · dell (1 thread) | 37 ms (5) | 395 ms (5) | 4.13 s (5) | 52.15 s (5) |
| Simple · epyc (1 thread) | 481 ms (10) | 6.20 s (10) | 78.70 s (10) | 922 s (10) |
| Simple · dell (1 thread) | 487 ms (10) | 6.23 s (10) | 80.98 s (10) | 965 s (10) |

| C4.5, no pruning | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| CUDA hybrid · hopper | 46 ms (5) | 48 ms (5) | 76 ms (5) | 622 ms (5) |
| CUDA hybrid · ampere | 21 ms (5) | 32 ms (5) | 109 ms (5) | 1.03 s (5) |
| GPU only · hopper | 48 ms (5) | 46 ms (5) | **71 ms** (5) | **618 ms** (5) |
| GPU only · ampere | 21 ms (5) | **31 ms** (5) | 102 ms (5) | 1.02 s (5) |
| Multicore · epyc | **19 ms** (5) | 61 ms (5) | 232 ms (5) | 1.93 s (5) |
| Multicore · dell | 52 ms (5) | 97 ms (5) | 448 ms (5) | 3.23 s (5) |
| Serial · epyc | 26 ms (5) | 267 ms (5) | 2.41 s (5) | 42.17 s (5) |
| Serial · dell | 45 ms (5) | 466 ms (5) | 4.80 s (5) | 58.43 s (5) |
| Simple · epyc | 598 ms (10) | 7.36 s (10) | 90.33 s (10) | 1,039 s (10) |
| Simple · dell | 601 ms (10) | 7.32 s (10) | 91.90 s (10) | 1,075 s (10) |

Peak memory (the same for CART and C4.5 within 0.01 GiB):

| | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| RAM, `--multicore` | 0.05 GiB | 0.42 GiB | 4.14 GiB | 40.5 GiB |
| RAM, `--serial` | 0.05 GiB | 0.40 GiB | 3.93 GiB | 39.3 GiB |
| RAM, `--simple` | 0.03 GiB | 0.25 GiB | 2.19 GiB | 21.4 GiB |
| RAM, CUDA hybrid (hopper / ampere) | 0.65 / 0.64 GiB | 0.88 / 0.86 GiB | 3.06 / 3.04 GiB | 24.9 / 24.8 GiB |
| RAM, GPU only (hopper / ampere) | 0.64 / 0.63 GiB | 0.73 / 0.72 GiB | 1.57 / 1.56 GiB | 9.95 / 9.94 GiB |
| VRAM (hopper / ampere; GPU only the same) | 0.52 / 0.48 GiB | 0.88 / 0.85 GiB | 3.44 / 3.42 GiB | 25.4 / 25.4 GiB |

- **The GPUs pull away with the rows, less than before.** At 1M rows epyc is the fastest (15–19 ms); a GPU run's fixed costs are about 20 ms on ampere and 45 ms on hopper. At 10M the A100 leads (30–32 ms against epyc's 54–61 ms and the H100's 46–50 ms); from 100M on the H100 leads.
  - The H100 is 2.8–3.3× faster than epyc at 100M and **2.6× (CART) to 3.1× (C4.5) at 1000M** (0.61–0.62 s, GPU only and hybrid alike). In the runs before it was 6.9–7.8× at 1000M: epyc's `--multicore` got 2.8× faster there (4.44 → 1.58 s CART, 5.45 → 1.93 s C4.5), the GPUs 3–13%.
  - The A100 is 1.6–1.9× faster than epyc at 1000M.
- **From 100M to 1000M** (10× the rows) the time grows 8.1–8.7× on the H100, 9.4–9.9× on the A100, 7.6–8.3× on epyc and 7.0–7.2× on dell.
- **epyc is 1.7–1.9× faster than dell** from 100M rows on, as on the other datasets; 2.5–2.7× at 1M, where dell's fixed costs (its presort, 24–31 ms against 9–11 ms) dominate.
- **Memory is linear in the rows:** 43 bytes per row with `--multicore` (40.5 GiB for 1000M rows, 2G values: 22 bytes per value; 46.7 GiB before) and 27 bytes per row of VRAM (25.4 GiB of the 80 GB). Every size fits on every GPU; the 40 GB A100s of ampere would hold the 1000M set too. A hybrid run needs about 60% of the host RAM of a `--multicore` run (24.9 GiB at 1000M), GPU only a quarter (9.95 GiB), `--simple` about half (21.4 GiB).
- **Against the runs before (f1fe270e)**, the same trees: `--multicore` 28–43% faster at 1M and 10M on epyc (dell: 5–10% slower at 1M, 11–15% faster at 10M, a few ms either way), 46–57% at 100M and 64–70% at 1000M on both; `--serial` 16–37% faster at 1M–100M and 48–60% at 1000M; the GPUs 3–52%, most at 1M–10M.

## Serial (one thread)

Means of the runs; train total = presort + build (prune ≈ 0).

| | 1M | 10M | 100M | 1000M | 1000M / 100M |
|---|--:|--:|--:|--:|--:|
| epyc CART: presort | 9 ms | 121 ms | 1.21 s | 12.23 s | 10.1× |
| epyc CART: build | 10 ms | 103 ms | 856 ms | 26.78 s | **31.3×** |
| dell CART: presort | 21 ms | 228 ms | 2.25 s | 21.53 s | 9.6× |
| dell CART: build | 15 ms | 168 ms | 1.88 s | 30.62 s | **16.3×** |
| epyc C4.5: build | 15 ms | 138 ms | 1.13 s | 29.22 s | 25.9× |
| dell C4.5: build | 23 ms | 222 ms | 2.33 s | 35.00 s | 15.0× |

| Speed-up of 48 threads over 1 (train total) | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| epyc CART / C4.5 | 1.2× / 1.4× | 4.1× / 4.4× | 9.9× / 10.4× | **24.7× / 21.8×** |
| dell CART / C4.5 | 0.9× / 0.9× | 4.7× / 4.8× | 10.6× / 10.7× | **19.3× / 18.1×** |

- **Up to 100M rows, the serial time grows with the rows** (about 10× per 10× rows), and epyc is 1.7–2.2× faster than dell.
- **At 1000M the serial build is 15–31× its 100M time**, while the presort stays at 10×: one thread moves a column of 1G rows, far beyond the caches ([the section below](#the-1000m-slowdown-and-its-fix-2026-10-19)). The blocked partition that fixed this needs more than one thread. d2957ff2 halved the build at 1000M (epyc CART 86.3 → 26.8 s, dell 77.9 → 30.6 s), and epyc leads again at every size (1.3–1.4× dell at 1000M).
- **Parallelism pays off only with the rows.** At 1M rows, 48 threads are hardly faster than one (0.9–1.4×). The speed-up grows to 10–11× at 100M and 18–25× at 1000M.
- **`--simple`**, the reference implementation, takes 23–27× `--serial`'s time at 1M on epyc, 37–38× at 100M and 24–25× at 1000M; on dell 13–20×. `--simple` itself is the same on epyc and dell within 5%, with about half the RAM of `--serial`.

## GPU only

`--gpu`: every node is grown on the GPU, with no hand-offs to the CPU and no host arena for them, on the same CPUs as the hybrid runs. Means of the runs, CART (C4.5 the same within a few ms):

| | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| hopper, hybrid: GPU setup / build | 41 / 5 ms | 44 / 6 ms | 56 / 18 ms | 0.27 / 0.35 s |
| hopper, GPU only: GPU setup / build | 43 / 5 ms | 41 / 5 ms | 57 / 18 ms | 0.26 / 0.35 s |
| ampere, hybrid: GPU setup / build | 16 / 4 ms | 25 / 7 ms | 76 / 31 ms | 0.55 / 0.47 s |
| ampere, GPU only: GPU setup / build | 17 / 4 ms | 25 / 5 ms | 78 / 24 ms | 0.55 / 0.46 s |

- **GPU only and hybrid take the same time** (within 1–8%; at 1M–10M a few ms either way). The tree has only 17 nodes, all far above the hand-off size, so the hybrid run hands nothing to the CPU either.
- **Still the fastest on the big sets:** 2.6–3.1× (H100) and 1.6–1.9× (A100) faster than epyc's 48 threads at 1000M, and 64–68× and 39–41× faster than one epyc thread.
- **Less than half the host RAM:** 9.95 GiB at 1000M against 24.9 GiB for hybrid (no host arena) and 40.5 GiB for `--multicore`. The VRAM is the same as hybrid's.
- **Reading the CSV is the cost:** a whole 1000M process (`wall_seconds`) takes 38–82 s on the GPU nodes, most of it parsing the 20.8 GB CSV, which every run reads again over NFS (`--drop-cache`). The training time above excludes it.

## The 1000M slowdown and its fix (2026-10-19)

**Before the fix**, the CPU time grew 13–41× from 100M to 1000M rows, 10× the rows. The times before and after, as measured on 2026-10-19 (medians of 3 runs, at 1000M on dell and dell512 of 10; the builds are in git, the run folders were replaced since):

| Before (2a9d64e) → after (d9da007) | 100M CART | 100M C4.5 | 1000M CART | 1000M C4.5 |
|---|--:|--:|--:|--:|
| epyc | 1.15 → 0.69 s | 1.35 → 0.66 s | 47.19 → **6.46 s** (7.3×) | 40.98 → **7.31 s** (5.6×) |
| dell | 2.43 → 1.21 s | 3.47 → 1.51 s | 34.20 → **9.95 s** (3.4×) | 43.33 → **12.32 s** (3.5×) |
| dell512 | 2.75 → 1.32 s | 3.18 → 1.63 s | 36.00 → **11.68 s** (3.1×) | 48.32 → **14.32 s** (3.4×) |

**The cause** was the partition of the big nodes.
- A node of the CPU builder is split by moving every column except the winner's into its children. Normally each column is moved by one thread, fused with the children's sweep.
- With two features, there is one column to move. So one thread moved all 1G rows of the root, and then of its children, while the other 47 waited.
- A single thread moving a column becomes much slower once the column is far beyond the caches: 3.5 ns per row at 100M rows, about 30 ns at 1000M.
- The builder already had a mode that partitions a node in blocks across all threads, but it used that mode only when spare memory ran short.

**The fix:** a node of at least 2^20 rows is also partitioned in blocks when its columns to move would keep at most one thread in 8 busy (48 threads: 1–6 columns).
- In blocks the node loses the fused sweep. Forcing the blocked mode on the other datasets cost airline (12 columns) 0–15%, and SUSY and HIGGS (17 and 27 columns) up to 50%. Those datasets are outside the new rule and keep the fused sweeps.
- The trees are unchanged: the golden tests pass, and so do checks of 3M synthetic rows with 2 and 3 features.

**Ruled out:**
- The TLB: the columns are on huge pages.
- The repeated rows of the 1000M set.
- Prefetching the partition bitmap: no gain at 1000M, and slower on airline.

**At 1M rows** (below 2^20) the code path is unchanged; the differences there (dell CART 70 → 78 ms) are run-to-run noise.

**At 10M rows** the new path is used, with mixed results:
- dell: 1.3–1.4× faster.
- dell512 C4.5: 1.4× faster.
- epyc, and dell512 CART: unchanged.

Some 10M cases had a single slow run (up to 0.77 s on epyc), which the median leaves out.

## One chart per size

![chess 3x3, 1M rows](chart-chess_1m.png)

![chess 3x3, 10M rows](chart-chess_10m.png)

![chess 3x3, 100M rows](chart-chess_100m.png)

![chess 3x3, 1000M rows](chart-chess_1b.png)

## Notes

- **Not timed:** reading the CSV, and the data check before each job (SHA-256 of every prepared file and every CSV row against the binary files). Every run reads the CSV again over NFS (`--drop-cache`). A whole `chess_1b` process (`wall_seconds`) took 38–82 s on the GPU nodes, 37–74 s with `--multicore`, 162–285 s with `--serial` and 980–1,160 s with `--simple`.
- **Hopper's and ampere's 1M and 10M runs** are dominated by fixed costs; their order there says little about the GPUs.
- **dell against dell512** (dell2, half the memory channels: 1.10–1.36× dell's time at 1000M with 58% of its bandwidth) and **one socket against both** were measured on ec77727 (2026-10-29/30); those runs are in git before 6bf6705a.
- **The originals** (C4.5 files from mordor3's `~/chess`) were deleted after the conversion; `datasets/chess/chess_*.csv` on mordor3 are the only copies.
