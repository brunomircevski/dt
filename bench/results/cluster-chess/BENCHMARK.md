# Chess 3×3 on the mordor3 cluster: 1M to 1000M rows on two GPU and three CPU servers, unpruned CART and C4.5

**Status:** three runs per case, on 2026-10-19 and (GPU only) 2026-10-20.
- **GPU runs** (hopper, ampere): rerun 20:30–20:33 with commit 6f6cecb: the GPU's hand-offs go through page-locked buffers instead of a page-locked arena ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)), on top of d9da007. The first GPU runs (17:34–17:42, 2a9d64e = 5058ce1, ./tree as in 051949b) built the same trees; at 1000M the H100 is now 18% (CART) and 15% (C4.5) faster, the A100 unchanged.
- **CPU runs** (epyc, dell, dell512): rerun 19:08–19:12 with commit d9da007, which fixes a slowdown at 1000M rows ([below](#the-1000m-slowdown-and-its-fix)). The fix changes only the CPU builder for nodes of at least 2^20 rows with few columns. The CPU subtrees of a CUDA run have at most 4096 rows, so the fix cannot change the GPU runs.
- **Serial runs** (epyc, dell; one thread): 2026-10-19, 23:01–23:21, with commit 673217f. It has the same CPU code as d9da007; the commit only adds the serial runs to `submit.sh`. [Below](#serial-one-thread).
- **GPU-only runs** (hopper, ampere; `--cuda --gpu-only --threads 1`): 2026-10-20, 08:46–08:59, with commit 0b3a2de. It has the same ./tree as 6f6cecb plus `--gpu-only`. [Below](#gpu-only-one-cpu-thread).
- **dell and dell512 at 1000M** are the 10-run medians of [the check below](#dell-against-dell512-10-runs-at-1000m), run with 41faca9 (the same CPU code as d9da007). Every other case has 3 runs.
- **Each node ran alone** and was used whole (`--exclusive`). The four sizes ran one after another on every node, smallest first.
- **Same trees:** every case built the same tree on every node and backend: 17 nodes (9 leaves), depth 4, 100% training accuracy, at every size. That is the exact tree of a 3×3 board.
- **Settings:** ./tree's defaults. The trees are grown without pruning (`cart_full_nodepth`, `c45_unpruned`), with no depth limit. Only training time and memory are compared; accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).

How the runs are made, and where the data came from: [bench/cluster/README.md](../../cluster/README.md).

```bash
for d in chess_1m chess_10m chess_100m chess_1b; do
  bench/cluster/submit.sh -m 3 -d $d -p cart_full_nodepth,c45_unpruned -a <prepare job> \
      cuda-hopper-$d cuda-ampere-$d cpu-epyc-$d cpu-dell-$d cpu-dell512-$d
done
```

The CPU rerun named only `cpu-epyc-$d cpu-dell-$d cpu-dell512-$d`, without `-a`. The serial runs were submitted with `serial-epyc-$d serial-dell-$d`, after a rebuild job (`-a <build job>`), and the GPU-only runs with `gpuonly-hopper-$d gpuonly-ampere-$d`.

The charts are drawn as shown in the README:
- `chart-1m.png`, `chart-10m.png`, `chart-100m.png`, `chart-1b.png`: one `machines_chart.py --spread` chart per size. The bars are medians, the whiskers run from the fastest to the slowest run, and the dots are the single runs. Both axes are log scale (`--log`), and each bar's label gives ./tree's recorded pruning time: 0 ms everywhere, since both protocols here are unpruned (prune_seconds 0–0.3 ms), so there is no pruning segment to draw.
- `chart-rows.png`: drawn by `rows_chart.py`.

Both use the 10-run results for dell and dell512 at 1000M.

**The dataset** is a 3×3 checkerboard: two features, x and y, uniform in [0, 3] with six decimals. The class (`A`, `B`) alternates over the nine unit squares (`B` on five of them: 55.5% of the rows), with no noise. Four sizes:

| Dataset | Training rows | Test rows (own file) | CSV | Note |
|---|--:|--:|--:|---|
| `chess_1m` | 1,000,000 | 1,000 | 20 MB | the first 1M rows of `chess_10m` |
| `chess_10m` | 10,000,000 | 4,500 | 200 MB | the first 10M rows of `chess_100m` |
| `chess_100m` | 100,000,000 | 4,500 | 2.0 GB | |
| `chess_1b` | 1,000,000,000 | 1,000 | 20.0 GB | `chess_100m` ten times over: 100M distinct rows, each 10 times |

**What it measures.** The tree is tiny (4 levels, 8 splits), so a run is the presort plus a few full sweeps over all rows: time per row at the top levels, without the many small nodes of a deep tree. With two features, the CPU backend has only two columns to share among 48 threads.

| Run | Node | CPU | RAM | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|---|
| `cuda-hopper-*` | hopper | 2× Xeon Gold 6444Y, 32 cores | 1 TB DDR5-4800 | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-*` | ampere | 2× EPYC 7402, 48 cores | 1 TB DDR4-3200 | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-*` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – | parallel (48 threads) |
| `cpu-dell-*` | dell1 | 2× Xeon Gold 6542Y, 48 cores | 1 TB DDR5-5200 (16 × 64 GB) | – | parallel (48 threads) |
| `cpu-dell512-*` | dell2 | the same as dell1 | 512 GB DDR5-5200 (8 × 64 GB) | – | parallel (48 threads) |
| `serial-epyc-*` | epyc1 | as `cpu-epyc-*` | | – | serial (1 thread, the whole node reserved) |
| `serial-dell-*` | dell1 | as `cpu-dell-*` | | – | serial (1 thread, the whole node reserved) |
| `gpuonly-hopper-*` | hopper | as `cuda-hopper-*` | | H100 PCIe 80 GB | CUDA, `--gpu-only` (+ 1 CPU thread, the whole node reserved) |
| `gpuonly-ampere-*` | ampere | as `cuda-ampere-*` | | A100 80 GB PCIe | CUDA, `--gpu-only` (+ 1 CPU thread, the whole node reserved) |

## Time and memory against the rows

![time, RAM and VRAM against the training rows](chart-rows.png)

Training time (`train total`: presort and build, on CUDA also GPU setup and upload; not reading the CSV), median of 3 runs:

| CART, no pruning | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | 51 ms | **88 ms** | **133 ms** | **0.92 s** |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 38 ms | **88 ms** | 190 ms | 1.78 s |
| GPU only · hopper (H100 PCIe + 1 thread) | 43 ms | 72 ms | 226 ms | 2.33 s |
| GPU only · ampere (A100 80GB PCIe + 1 thread) | 33 ms | 56 ms | 253 ms | 2.61 s |
| Parallel · epyc (EPYC 9255, 48 threads) | **23 ms** | 178 ms | 689 ms | 6.46 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 78 ms | 184 ms | 1.21 s | 9.95 s |
| Parallel · dell512 (the same, half the memory channels) | 65 ms | 250 ms | 1.32 s | 11.68 s |
| Serial · epyc (1 thread) | 30 ms | 348 ms | 3.35 s | 99.53 s |
| Serial · dell (1 thread) | 74 ms | 558 ms | 6.46 s | 101.78 s |

| C4.5, no pruning | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| CUDA · hopper | 49 ms | 89 ms | **148 ms** | **0.97 s** |
| CUDA · ampere | 38 ms | **82 ms** | 205 ms | 1.80 s |
| GPU only · hopper | 43 ms | 72 ms | 228 ms | 2.34 s |
| GPU only · ampere | 34 ms | 56 ms | 255 ms | 2.63 s |
| Parallel · epyc | **24 ms** | 190 ms | 660 ms | 7.31 s |
| Parallel · dell | 73 ms | 271 ms | 1.51 s | 12.32 s |
| Parallel · dell512 | 71 ms | 203 ms | 1.63 s | 14.32 s |
| Serial · epyc | 35 ms | 392 ms | 3.75 s | 103.07 s |
| Serial · dell | 85 ms | 625 ms | 7.28 s | 110.77 s |

Peak memory (the same for CART and C4.5 within 0.01 GiB):

| | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| RAM, CPU servers (serial and parallel) | 0.06 GiB | 0.48 GiB | 4.68 GiB | 46.70 GiB |
| RAM, CUDA (hopper / ampere) | 0.66 / 0.65 GiB | 0.88 / 0.87 GiB | 3.06 / 3.05 GiB | 24.87 / 24.86 GiB |
| RAM, GPU only (hopper and ampere) | 0.64 GiB | 0.73 GiB | 1.57 GiB | 9.96 GiB |
| VRAM, GPU 0 (hopper / ampere; GPU only the same) | 0.53 / 0.49 GiB | 0.89 / 0.85 GiB | 3.44 / 3.41 GiB | 25.43 / 25.42 GiB |

- **The GPUs pull away with the rows.** At 1M rows epyc is the fastest (23 ms): a GPU run's fixed costs (setup, transfers, kernel launches) are about 40–50 ms. From 10M rows on, the GPUs lead.
  - The H100 is 2.0–2.1× faster than the best CPU server at 10M, 4.5–5.2× at 100M, and **7.0× (CART) to 7.5× (C4.5) at 1000M**. In every case the best CPU server is epyc.
  - The A100 is 3.6–4.1× faster than epyc at 1000M.
- **The GPU time grows slower than the rows**: 7× from 100M to 1000M on the H100, 9× on the A100. Even 100M rows do not keep a GPU busy; 1000M do.
- **The CPU time grows with the rows**: 8–11× from 100M to 1000M. The presort grows 10× (epyc 0.20 → 1.91 s, dell 0.54 → 4.39 s), and so does the build (epyc CART 0.47 → 4.55 s, dell 0.69 → 5.61 s).
- **epyc is 1.5–2.3× faster than dell** from 100M rows on, as on the other datasets.
- **dell512 against dell:** with 3 runs per case, from 25% less time to 36% more (both at 10M, where the runs are noisy). With 10 runs at 1000M, **dell is faster in every phase, by 14–17%** ([below](#dell-against-dell512-10-runs-at-1000m)).
- **Memory is linear in the rows:** 50 bytes per row on the CPU servers (46.7 GiB for 1000M rows, 2 G values: 25 bytes per value) and 27 bytes per row of VRAM (25.5 GiB of the 80 GB). Every size fits on every GPU; the 40 GB A100s of ampere would hold the 1000M set too. A CUDA run needs about half the host RAM of a CPU run (24.9 GiB at 1000M; 0.25 GiB of it at every size are the page-locked buffers of the uploads and hand-offs).

## Serial (one thread)

Medians of 3 runs; train total = presort + build (prune ≈ 0).

| | 1M | 10M | 100M | 1000M | 1000M / 100M |
|---|--:|--:|--:|--:|--:|
| epyc CART: presort | 9 ms | 127 ms | 1.28 s | 12.85 s | 10.0× |
| epyc CART: build | 20 ms | 221 ms | 2.07 s | 86.67 s | **41.9×** |
| dell CART: presort | 33 ms | 238 ms | 2.40 s | 23.12 s | 9.6× |
| dell CART: build | 41 ms | 322 ms | 4.06 s | 78.60 s | **19.4×** |
| epyc C4.5: build | 24 ms | 256 ms | 2.39 s | 89.41 s | 37.5× |
| dell C4.5: build | 49 ms | 365 ms | 4.69 s | 85.79 s | 18.3× |

| Speed-up of 48 threads over 1 (train total) | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| epyc CART / C4.5 | 1.3× / 1.5× | 2.0× / 2.1× | 4.9× / 5.7× | **15.4× / 14.1×** |
| dell CART / C4.5 (1000M: 10-run medians) | 0.9× / 1.2× | 3.0× / 2.3× | 5.3× / 4.8× | **10.2× / 9.0×** |

- **Up to 100M rows, the serial time grows with the rows** (about 10× per 10× rows), and epyc is 1.6–2.5× faster than dell.
- **At 1000M the serial build is 19–42× its 100M time**, while the presort stays at 10×. It is the slowdown of [the next section](#the-1000m-slowdown-and-its-fix): one thread moves a column of 1G rows. That costs about 3.5 ns per row at 100M and about 30 ns at 1000M, far beyond the caches. The fix (blocked partition by all threads) needs more than one thread, so it does not apply here.
- **At 1000M epyc loses its lead:** epyc's serial build takes 10% longer than dell's (CART 86.7 s against 78.6 s), and the train totals are within 2% (CART) and 7% (C4.5). epyc's presort is still 1.8× faster. The build here waits on memory, row by row, so it measures memory latency more than the core.
- **Parallelism pays off only with the rows.** At 1M rows, 48 threads are hardly faster than one, and dell CART is even slower (0.9×). The speed-up grows to 5× at 100M and 9–15× at 1000M. The 1000M speed-ups include the serial build's slowdown, which the threads avoid.
- **Memory is the same** as with 48 threads (46.70 GiB at 1000M).

## GPU only (one CPU thread)

`--cuda --gpu-only --threads 1`: every node is grown on the GPU, with no hand-offs to the CPU and no host arena for them, and the CPU side is one thread. The whole node was reserved. Medians of 3 runs, CART (C4.5 within 1%):

| | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| hopper, CUDA: GPU setup / build | 46 / 4 ms | 73 / 13 ms | 92 / 31 ms | 0.23 / 0.58 s |
| hopper, GPU only: GPU setup / build | 39 / 3 ms | 56 / 15 ms | 100 / 115 ms | 0.75 / 1.47 s |
| ampere, CUDA: GPU setup / build | 34 / 3 ms | 41 / 42 ms | 89 / 86 ms | 0.52 / 1.13 s |
| ampere, GPU only: GPU setup / build | 30 / 3 ms | 38 / 16 ms | 92 / 147 ms | 0.61 / 1.87 s |
| train total, GPU only / CUDA: hopper | 0.84× | 0.83× | 1.70× | **2.53×** |
| train total, GPU only / CUDA: ampere | 0.87× | 0.64× | 1.33× | **1.47×** |
| whole process (`wall_seconds`, CSV read included), GPU only / CUDA: hopper, ampere | 1.0 / 1.1 s, 0.6 / 0.7 s | 1.7 / 1.1 s, 1.6 / 0.7 s | 7.8 / 1.3 s, 11.0 / 0.9 s | **71.9 / 4.7 s, 106.2 / 6.0 s** |

- **Up to 10M rows GPU only is a little faster** (13–36% less time): one CPU thread starts and syncs faster than 32 or 48, and the fixed costs dominate.
- **From 100M rows on it is slower:** at 1000M 2.5× the CUDA time on the H100 (2.33 s against 0.92 s) and 1.5× on the A100 (2.61 s against 1.78 s). Both the setup with the upload (H100 0.23 → 0.75 s) and the build (0.58 → 1.47 s) take longer; the presort does not change. The tree has only 17 nodes, all of them far above the hand-off size, so the hand-offs are not what the other threads speed up here; what the one thread waits on was not profiled.
- **Still faster than every CPU server:** 2.8× (H100) and 2.5× (A100) faster than epyc's 48 threads at 1000M, and 43× and 38× faster than one epyc thread.
- **Less than half the host RAM:** 9.96 GiB at 1000M against 24.87 GiB with hand-offs (no host arena) and 46.70 GiB on the CPU servers. The VRAM is the same as with hand-offs.
- **Reading the CSV is the cost:** with one thread, the whole 1000M process takes 72 s (hopper) and 106 s (ampere) instead of 4.7 and 6.0 s, nearly all of it parsing the 20 GB CSV. The training time above excludes it.

## The 1000M slowdown and its fix

**Before the fix**, the CPU time grew 13–41× from 100M to 1000M rows, 10× the rows. The times before and after (median; after at 1000M on dell and dell512: 10 runs, otherwise 3):

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

## dell against dell512: 10 runs at 1000M

The 3-run results above had dell512 sometimes faster than dell, although dell has twice the memory channels. To check, `chess_1b` ran 10 times per case on both nodes. That was commit 41faca9, the same CPU code as d9da007, on 2026-10-19 at 20:07–20:13. The results are in `dell-variance/`.

```bash
bench/cluster/submit.sh -m 10 -d chess_1b -p cart_full_nodepth,c45_unpruned -o cluster-chess/dell-variance \
    cpu-dell-chess_1b cpu-dell512-chess_1b
d=bench/results/cluster-chess/dell-variance
bench/.venv/bin/python bench/runs_chart.py $d/cpu-dell-chess_1b $d/cpu-dell512-chess_1b --out $d/chart.png
```

![dell and dell512, chess 1000M: every run of train total, presort and build](dell-variance/chart.png)

| Median of 10 (min–max) | dell (1 TB, 16 DIMMs) | dell512 (512 GB, 8 DIMMs) | dell512 / dell |
|---|--:|--:|--:|
| CART: train total | 9.95 s (9.07–10.49) | 11.68 s (10.45–16.40) | 1.17× |
| CART: presort | 4.39 s | 4.92 s | 1.12× |
| CART: build | 5.61 s | 6.84 s | 1.22× |
| C4.5: train total | 12.32 s (12.10–14.03) | 14.32 s (13.73–16.32) | 1.16× |
| C4.5: presort | 4.37 s | 5.08 s | 1.16× |
| C4.5: build | 8.14 s | 9.34 s | 1.15× |
| Memory bandwidth, 48 threads (triad) | 379 GB/s | 220 GB/s | 0.58× |
| Memory bandwidth, 1 thread (triad) | 17.2 GB/s | 16.3 GB/s | 0.95× |

- **dell is faster in every phase**, 12–22% in the medians, and in train total dell512's fastest run is slower than dell's median.
- **The earlier "dell512 faster" was noise.** The 3-run CART medians were 11.18 s (dell) against 10.74 s (dell512), but the runs of one node spread by 10–15%, and dell512 had two slow CART runs out of 10 (16.4 s; build 11–12 s instead of 6–7 s).
- **Bandwidth matters only partly.** With 48 threads dell512 has 58% of dell's bandwidth, but it takes only about 1.16× the time. So chess is partly limited by memory bandwidth, and partly by other work: the per-thread speed and the synchronisation between threads. One thread alone gets the same 16–17 GB/s on both nodes.
- **The bandwidth was measured** with `bench/cluster/triad.c`, a STREAM-style copy and triad over 3 × 4 GiB arrays, best of 10. It was compiled `gcc -O3 -fopenmp` and run on each whole node. Logs: `dell-variance/triad-dell{1,2}.log`.

## One chart per size

![chess 3x3, 1M rows, on five mordor3 nodes](chart-1m.png)

![chess 3x3, 10M rows, on five mordor3 nodes](chart-10m.png)

![chess 3x3, 100M rows, on five mordor3 nodes](chart-100m.png)

![chess 3x3, 1000M rows, on five mordor3 nodes](chart-1b.png)

## Notes

- **Not timed:** reading the CSV, and the data check before each job (SHA-256 of every prepared file and every CSV row against the binary files: 113 s for `chess_1b` on ampere, 33 GB over NFS). A whole `chess_1b` process (`wall_seconds`, reading the 20 GB CSV included) took 5.2–6.5 s on the GPU nodes and 9–18 s on the CPU nodes (37–52 s before the fix); with one thread 72 s (hopper) and 106 s (ampere) GPU only, and 155–159 s serial on epyc.
- **Hopper's and ampere's 1M and 10M runs** are dominated by fixed costs; their order there says little about the GPUs.
- **The originals** (C4.5 files from mordor3's `~/chess`) were deleted after the conversion; `datasets/chess_*.csv` on mordor3 are the only copies. Their SHA-256 and those of the originals are in [bench/cluster/README.md](../../cluster/README.md) and `bench/cluster/chess-originals.sha256`.
