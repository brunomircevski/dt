# Chess 3×3 on the mordor3 cluster: 1M to 1000M rows on two GPU and three CPU servers, unpruned CART and C4.5

**Status:** every run repeated on 2026-10-20, three runs per case (dell and dell512 at 1000M: ten). The same trees as before at every size and on every node.
- **CPU and serial runs** (epyc, dell, dell512): commit 8b1ecd1, from 13:52. It adds 6521330 to the code of the earlier runs: the class counts are kept per thread in local histograms instead of one shared, false-shared array, and the staged GPU copies use at least 8 threads.
- **epyc's parallel runs:** repeated at 1a2d9a9 (the same ./tree), with the datasets dropped from the page cache before each run (`run.py --drop-cache`). In the 8b1ecd1 run, our own cached dataset files had filled epyc1's NUMA node 1, and the 48-thread runs allocated across the sockets.
- **CUDA and GPU-only runs** (hopper, ampere): commit 235c431, from 18:32, on the GPU's socket only: GPU 0, the CPUs of its NUMA node 0 (16 threads on hopper, 24 on ampere) and the memory bound to that node. GPU-only (`--cuda --gpu-only`) now runs on all of those CPUs, as CUDA does (3ccf4ba): `--gpu-only` keeps the tree's growth off the CPU, not the rest of the run. [Below](#gpu-only).
- **One socket** (`socket-*`, epyc1 and dell1): commit 10d125d, from 18:54: the parallel backend on every CPU of one socket (24 threads), the memory bound to that socket's NUMA nodes (`SOCKET=auto` in job.sh: the socket with the most free memory, epyc1's socket 0 (NUMA node 0) and dell1's socket 1 (NUMA nodes 2, 3)).
- **Before:** the CUDA rows were 8b1ecd1's on the whole node (2026-10-20 from 13:52) and 6f6cecb's before that (2026-10-19), the GPU-only rows 7868bf4's on one CPU thread (2026-10-20 from 15:05, hopper's NUMA node 1 with GPU 2) and 0b3a2de's (that morning, GPU 0, unbound), the CPU rows d9da007's, the serial rows 673217f's, the dell/dell512 10-run check 41faca9's. The git history has them.
- **Each node ran alone** and was reserved whole (`--exclusive`). The four sizes ran one after another on every node, smallest first.
- **Same trees:** every case built the same tree on every node and backend: 17 nodes (9 leaves), depth 4, 100% training accuracy, at every size. That is the exact tree of a 3×3 board.
- **Reruns queued (2026-10-28, jobs 381–545, `bench/cluster/submit-variants-2026-10-28.sh` and `submit-topup-2026-10-28.sh`):** every case on hopper and ampere (CUDA hybrid, GPU only) and epyc (parallel on both sockets and on one socket, --serial) gets 10 runs (each run, reading the CSV included, takes under about 2.5 minutes), and `--simple` on epyc is new here (10 runs; one at 1000M). The 3 runs here count; only the missing ones run (into `cluster-chess-new/` and `-top/` on mordor3), and `bench/merge_runs.py` adds them to the run folders here when done; the dell folders stay as they are. Superseded by the one-build reruns below: every chess folder was of 2026-10-20 builds, so all the ./tree_cpu ones start over. The charts mark the missing `simple-epyc-*` runs as "queued".
- **Spin stall, fixed 2026-10-28 (rerun queued, jobs 550–570, `bench/cluster/submit-spinfix-2026-10-28.sh`):** the parallel backend partitions a big node (≥ 1M rows, few columns to move) in blocks taken in order by all threads, each block waiting for the one before it. It waited by spinning (`_mm_pause`) without end: when the thread of the block before was descheduled, the other threads spun and kept it off a CPU, so some runs took 2–8× as long (epyc chess_10m: user time 5 s instead of 2 s, the same system time; 11 of 80 runs > 2× the median). It now spins 2,048 pauses, then sleeps (`atomic::wait`, woken by the block before). On epyc2, 40 runs each of chess_10m (48 threads): CART 0.155 ± 0.131 s → 0.087 ± 0.005 s, C4.5 0.186 ± 0.163 s → 0.095 ± 0.005 s, no run above 2× the median. Only the parallel backend has those blocks (CUDA hybrid gives the CPU nodes under 4,096 rows; --serial and --simple have no pool), so only the parallel epyc runs (both sockets, one socket) start over, with 10 runs a case (full ECBDL 5), into `cluster-<dataset>-fix/`, replacing the old folders when done. The dell runs of ./tree_cpu run again too (below); until the reruns are in, the bars of old builds are marked (hatched, ⚠, from `problems.json`), as are runs more than twice the mean of the other runs.
- **One build for every epyc ./tree_cpu run (2026-10-28, `bench/cluster/submit-final-2026-10-28.sh`):** after the spin fix, C4.5's list of distinct values (its thresholds are training values) was made allocation-free: counted, then written in place on the pool, and in one pass with --serial (SOREL's C4.5 presort on epyc varied 11–34 s; see cluster-sorel). Same trees as before in every check (SOREL, HIGGS, airline, ECBDL 15M, serial and parallel). So that every epyc run of ./tree_cpu is of that one build, the parallel runs started before it (jobs 550–556) run again, and --serial runs again for every dataset (10 runs a case of about 2 minutes or less, 5 under about 10 minutes, longer ones once), into `cluster-<dataset>-fix/`, replacing the old folders. The long runs (full ECBDL and SOREL parallel, every --serial, then --simple) run one after another on epyc2, the short parallel ones on epyc1. CUDA hybrid, GPU only and --simple do not use that code and are not rerun. Every dell run of ./tree_cpu (parallel on both sockets, on one socket, dell2's 512 GB, --serial) runs again on the same build too (`bench/cluster/submit-dell-chess-2026-10-28.sh`, jobs 592–618: parallel on dell1, then --serial there; dell2 only its 512 GB runs), the same runs as before, replacing the old folders; --simple on dell stays.
- **Every run again on one build each, in batches (2026-10-28 evening, `bench/cluster/submit-batches-2026-10-28.sh`, jobs 636–645):** the pending jobs above were cancelled and replaced by a few batch jobs (`bench/cluster/batch.sh`: one SLURM job per node and group, its runs one after another). CUDA hybrid and GPU only on hopper and ampere run again on the new ./tree (67495f9, the CUDA rework: the smaller side's goes-left bitmap, the split column by position, upload overlapped with the presort, root counts on the GPU), installed by job 636 only after its trees matched the old build's (HIGGS, airline, ECBDL 15M; CART and C4.5, both GPU backends; 10–25 % faster there). On the CPU, dell and epyc for every backend (--simple, --serial, parallel on one socket, on both), on the final ./tree_cpu; the runs already done on it are kept, as are --simple's (that code did not change). Runs per case: about 2 minutes or less 10, under about 10 minutes 3, longer once; long runs on epyc2 (epyc1 stays free for the short ones), every dell run on dell1, short batch first. Test accuracy is now predicted once per distinct tree (`bench/run.py`), not once per run. The chart groups the bars by backend, the nodes under each; it reads `cluster-chess-fix/` too: a rerun done replaces the old bar, one pending marks it ("old build; rerun queued") or shows as "queued".
- **Settings:** ./tree's defaults. The trees are grown without pruning (`cart_full_nodepth`, `c45_unpruned`), with no depth limit. Only training time and memory are compared; accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).

How the runs are made, and where the data came from: [bench/cluster/README.md](../../cluster/README.md).

```bash
for d in chess_1m chess_10m chess_100m chess_1b; do
  bench/cluster/submit.sh -m 3 -d $d -p cart_full_nodepth,c45_unpruned -a <prepare job> \
      cuda-hopper-$d cuda-ampere-$d cpu-epyc-$d cpu-dell-$d cpu-dell512-$d
done
```

The CPU rerun named only `cpu-epyc-$d cpu-dell-$d cpu-dell512-$d`, without `-a`. The serial runs were submitted with `serial-epyc-$d serial-dell-$d`, after a rebuild job (`-a <build job>`). The CUDA and GPU-only reruns named `cuda-hopper-$d cuda-ampere-$d gpuonly-hopper-$d gpuonly-ampere-$d` (job.sh puts every GPU run on its GPU's NUMA node), and the one-socket runs `socket-epyc-$d socket-dell-$d`.

The charts are drawn as shown in the README:
- `chart-chess_1m.png`, `chart-chess_10m.png`, `chart-chess_100m.png`, `chart-chess_1b.png`: one `testset_chart.py` chart per size (2026-10-28; before, `machines_chart.py --spread` charts drew medians). A bar is the mean of the runs, the whisker runs from the fastest to the slowest run, and its label gives the mean ± standard deviation, the number of runs and peak RAM (and VRAM). Log scale.
- `chart-rows.png`: drawn by `rows_chart.py`: means, with whiskers from the fastest to the slowest run.

`chart-rows.png` uses the 10-run results for dell and dell512 at 1000M, the per-size charts the 3-run ones.

**The dataset** is a 3×3 checkerboard: two features, x and y, uniform in [0, 3] with six decimals. The class (`A`, `B`) alternates over the nine unit squares (`B` on five of them: 55.5% of the rows), with no noise. Four sizes:

| Dataset | Training rows | Test rows (own file) | CSV | Note |
|---|--:|--:|--:|---|
| `chess_1m` | 1,000,000 | 1,000 | 20 MB | the first 1M rows of `chess_10m` |
| `chess_10m` | 10,000,000 | 4,500 | 200 MB | the first 10M rows of `chess_100m` |
| `chess_100m` | 100,000,000 | 4,500 | 2.0 GB | |
| `chess_1b` | 1,000,000,000 | 1,000 | 20.0 GB | `chess_100m` ten times over: 100M distinct rows, each 10 times |

**What it measures.** The tree is tiny (4 levels, 8 splits), so a run is the presort plus a few full sweeps over all rows: time per row at the top levels, without the many small nodes of a deep tree. With two features, the CPU backend has only two columns to share among 48 threads.

| Run | Node | CPU | RAM | GPU (GPU 0 used, unless noted) | Backend |
|---|---|---|---|---|---|
| `cuda-hopper-*` | hopper | 2× Xeon Gold 6444Y, 32 cores | 1 TB DDR5-4800 | H100 PCIe 80 GB | CUDA (+ 16 CPU threads of the GPU's socket) |
| `cuda-ampere-*` | ampere | 2× EPYC 7402, 48 cores | 1 TB DDR4-3200 | A100 80 GB PCIe | CUDA (+ 24 CPU threads of the GPU's socket) |
| `cpu-epyc-*` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – | parallel (48 threads) |
| `cpu-dell-*` | dell1 | 2× Xeon Gold 6542Y, 48 cores | 1 TB DDR5-5200 (16 × 64 GB) | – | parallel (48 threads) |
| `cpu-dell512-*` | dell2 | the same as dell1 | 512 GB DDR5-5200 (8 × 64 GB) | – | parallel (48 threads) |
| `serial-epyc-*` | epyc1 | as `cpu-epyc-*` | | – | serial (1 thread, the whole node reserved) |
| `serial-dell-*` | dell1 | as `cpu-dell-*` | | – | serial (1 thread, the whole node reserved) |
| `gpuonly-hopper-*` | hopper | as `cuda-hopper-*` | | H100 PCIe 80 GB | CUDA, `--gpu-only` (+ 16 CPU threads of the GPU's socket) |
| `gpuonly-ampere-*` | ampere | as `cuda-ampere-*` | | A100 80 GB PCIe | CUDA, `--gpu-only` (+ 24 CPU threads of the GPU's socket) |
| `socket-epyc-*` | epyc1 | one EPYC 9255 (socket 0), 24 cores | its NUMA node 0 | – | parallel (24 threads, one socket) |
| `socket-dell-*` | dell1 | one Xeon Gold 6542Y (socket 1), 24 cores | its NUMA nodes 2, 3 | – | parallel (24 threads, one socket) |

## Time and memory against the rows

![time, RAM and VRAM against the training rows](chart-rows.png)

Training time (`train total`: presort and build, on CUDA also GPU setup and upload; not reading the CSV), median of 3 runs (dell and dell512 at 1000M: of 10):

| CART, no pruning | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 16 threads) | 57 ms | 81 ms | **106 ms** | 0.97 s |
| CUDA · ampere (A100 80GB PCIe + 24 threads) | 38 ms | 49 ms | 141 ms | 1.47 s |
| GPU only · hopper (H100 PCIe + 16 threads) | 64 ms | 70 ms | 111 ms | **0.88 s** |
| GPU only · ampere (A100 80GB PCIe + 24 threads) | 35 ms | **46 ms** | 128 ms | 1.33 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 23 ms | 169 ms | 529 ms | 5.17 s |
| Parallel, one socket · epyc (24 threads) | **22 ms** | 74 ms | 598 ms | 6.38 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 72 ms | 189 ms | 1.22 s | 9.66 s |
| Parallel · dell512 (the same, half the memory channels) | 61 ms | 179 ms | 1.13 s | 11.84 s |
| Parallel, one socket · dell (24 threads) | 40 ms | 110 ms | 1.17 s | 13.61 s |
| Serial · epyc (1 thread) | 28 ms | 335 ms | 3.21 s | 98.32 s |
| Serial · dell (1 thread) | 74 ms | 825 ms | 6.25 s | 99.79 s |

| C4.5, no pruning | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| CUDA · hopper | 58 ms | 56 ms | 105 ms | 0.96 s |
| CUDA · ampere | 39 ms | 48 ms | 141 ms | 1.46 s |
| GPU only · hopper | 60 ms | 75 ms | **99 ms** | **0.89 s** |
| GPU only · ampere | 38 ms | **47 ms** | 130 ms | 1.35 s |
| Parallel · epyc | 30 ms | 132 ms | 622 ms | 6.00 s |
| Parallel, one socket · epyc | **25 ms** | 89 ms | 701 ms | 7.07 s |
| Parallel · dell | 71 ms | 211 ms | 1.44 s | 12.27 s |
| Parallel · dell512 | 70 ms | 193 ms | 1.38 s | 14.31 s |
| Parallel, one socket · dell | 44 ms | 124 ms | 1.34 s | 14.98 s |
| Serial · epyc | 33 ms | 377 ms | 3.54 s | 100.86 s |
| Serial · dell | 82 ms | 784 ms | 7.08 s | 109.14 s |

Peak memory (the same for CART and C4.5 within 0.01 GiB, except hopper's VRAM at 1M):

| | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| RAM, CPU servers (serial, parallel and one socket) | 0.06 GiB | 0.48 GiB | 4.68 GiB | 46.70 GiB |
| RAM, CUDA (hopper / ampere) | 0.66 / 0.65 GiB | 0.88 / 0.87 GiB | 3.06 / 3.05 GiB | 24.87 / 24.86 GiB |
| RAM, GPU only (hopper and ampere) | 0.64 GiB | 0.72–0.73 GiB | 1.56–1.57 GiB | 9.96 GiB |
| VRAM (hopper / ampere; GPU only the same within 0.03 GiB) | 0.50–0.53 / 0.46 GiB | 0.89 / 0.85 GiB | 3.44 / 3.41 GiB | 25.43 / 25.41 GiB |

- **The GPUs pull away with the rows.** At 1M rows epyc is the fastest (22–25 ms on one socket, 23–30 ms on both): a GPU run's fixed costs (setup, transfers, kernel launches) are about 30–60 ms. From 10M rows on, the GPUs lead.
  - The H100 is 5.0–6.3× faster than the best CPU server at 100M, and **5.9× (CART) to 6.7× (C4.5) at 1000M** (GPU only, 0.88–0.89 s; with hand-offs 0.96–0.97 s, 5.3–6.2×). In every case the best CPU server is epyc on both sockets.
  - The A100 is 3.5–4.4× faster than epyc at 1000M. Up to 10M it keeps up with the H100 or beats it (46–49 against 56–81 ms), since the fixed costs dominate.
- **The GPU time grows about as the rows** from 100M to 1000M: 8–9× on the H100, 10× on the A100.
- **The CPU time grows with the rows**: 8–10× from 100M to 1000M. The presort grows 9–10× (epyc 0.20 → 1.85 s, dell 0.53 → 4.05 s), and so does the build (epyc CART 0.32 → 3.32 s, dell 0.72 → 5.55 s).
- **epyc is 1.9–2.4× faster than dell and dell512** from 100M rows on, as on the other datasets.
- **dell512 against dell:** with 3 runs per case, from 15% less time (1M) to 5–8% less (10M, 100M). With 10 runs at 1000M, **dell is faster in every phase, by 16–27%** ([below](#dell-against-dell512-10-runs-at-1000m)).
- **Memory is linear in the rows:** 50 bytes per row on the CPU servers (46.7 GiB for 1000M rows, 2 G values: 25 bytes per value) and 27 bytes per row of VRAM (25.4 GiB of the 80 GB). Every size fits on every GPU; the 40 GB A100s of ampere would hold the 1000M set too. A CUDA run needs about half the host RAM of a CPU run (24.9 GiB at 1000M; 0.25 GiB of it at every size are the page-locked buffers of the uploads and hand-offs).
- **Against the earlier runs** (6f6cecb, d9da007, 673217f), with the same trees:
  - **CUDA at 1000M,** 8b1ecd1 on the whole node: the A100 26–27% faster than at 6f6cecb (CART 1.78 → 1.31 s, C4.5 1.80 → 1.33 s), the H100 4–11% (0.92 → 0.88 s, 0.97 → 0.86 s). The class counts of the root's 10^9 rows are no longer one shared array (6521330); at 10M the A100 was 1.7–1.9× faster (88 → 46 ms).
  - **CUDA on the GPU's socket only** (235c431, the table) against the whole node (8b1ecd1): 10–12% slower at 1000M (H100 0.88 → 0.97 s, A100 1.31 → 1.47 s), and the setup with the upload takes most of it on the A100 (0.49 → 0.52 s, build 0.68 → 0.81 s). At 100M the H100 is faster (131 → 106 ms), the A100 slightly slower (131 → 141 ms); at 1M and 10M within the fixed costs' noise.
  - **epyc at 1000M:** 20% (CART 6.46 → 5.17 s) and 18% (C4.5 7.31 → 6.00 s) faster; at 100M CART 0.69 → 0.53 s. Both 6521330 and the page-cache fix came in between; which one gained how much was not measured separately.
  - **dell, dell512 and serial:** within −3% to +1% at 1000M (dell CART 9.95 → 9.66 s, serial epyc CART 99.5 → 98.3 s). At 100M dell512 is 14–15% faster (CART 1.32 → 1.13 s), the rest within 5%.
  - **GPU only:** on the socket's threads instead of one, 1.4–1.9× faster at 1000M ([below](#gpu-only)).
- **One socket against both** (epyc and dell): faster up to 10M rows (0.44–0.96× the time: probably fewer threads to start and sync over two columns, not profiled), slower at 1000M (epyc 1.18–1.23×, dell 1.23–1.44×); at 100M epyc 1.13×, dell 0.93–0.96×. On one socket dell is 2.1× epyc at 1000M (13.6 against 6.4 s).

## Serial (one thread)

Medians of 3 runs; train total = presort + build (prune ≈ 0).

| | 1M | 10M | 100M | 1000M | 1000M / 100M |
|---|--:|--:|--:|--:|--:|
| epyc CART: presort | 9 ms | 121 ms | 1.21 s | 12.17 s | 10.1× |
| epyc CART: build | 19 ms | 214 ms | 2.00 s | 86.15 s | **43.1×** |
| dell CART: presort | 33 ms | 346 ms | 2.26 s | 22.08 s | 9.8× |
| dell CART: build | 41 ms | 479 ms | 3.99 s | 77.73 s | **19.5×** |
| epyc C4.5: build | 23 ms | 246 ms | 2.24 s | 87.76 s | 39.2× |
| dell C4.5: build | 48 ms | 412 ms | 4.64 s | 85.31 s | 18.4× |

| Speed-up of 48 threads over 1 (train total) | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| epyc CART / C4.5 | 1.2× / 1.1× | 2.0× / 2.9× | 6.1× / 5.7× | **19.0× / 16.8×** |
| dell CART / C4.5 (1000M: 10-run medians) | 1.0× / 1.2× | 4.4× / 3.7× | 5.1× / 4.9× | **10.3× / 8.9×** |

- **Up to 100M rows, the serial time grows with the rows** (about 10× per 10× rows), and epyc is 1.9–2.6× faster than dell. dell's 10M runs spread (CART 0.54–0.84 s).
- **At 1000M the serial build is 18–43× its 100M time**, while the presort stays at 10×. It is the slowdown of [the next section](#the-1000m-slowdown-and-its-fix): one thread moves a column of 1G rows. That costs about 3.5 ns per row at 100M and about 30 ns at 1000M, far beyond the caches. The fix (blocked partition by all threads) needs more than one thread, so it does not apply here.
- **At 1000M epyc loses its lead:** epyc's serial build takes 11% longer than dell's (CART 86.2 s against 77.7 s), and the train totals are within 2% (CART) and 8% (C4.5). epyc's presort is still 1.8× faster. The build here waits on memory, row by row, so it measures memory latency more than the core.
- **Parallelism pays off only with the rows.** At 1M rows, 48 threads are hardly faster than one (1.0–1.2×). The speed-up grows to 5–6× at 100M and 9–19× at 1000M. The 1000M speed-ups include the serial build's slowdown, which the threads avoid.
- **Memory is the same** as with 48 threads (46.70 GiB at 1000M).

## GPU only

`--cuda --gpu-only`: every node is grown on the GPU, with no hand-offs to the CPU and no host arena for them. Since 3ccf4ba the GPU-only runs use the same CPUs as the CUDA runs: those of the GPU's NUMA node (16 on hopper, 24 on ampere), with the memory bound to that node and GPU 0. Medians of 3 runs, CART (C4.5 within 11%):

| | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| hopper, CUDA: GPU setup / build | 53 / 4 ms | 69 / 11 ms | 62 / 34 ms | 0.23 / 0.64 s |
| hopper, GPU only: GPU setup / build | 60 / 4 ms | 61 / 7 ms | 76 / 26 ms | 0.23 / 0.55 s |
| ampere, CUDA: GPU setup / build | 34 / 3 ms | 39 / 8 ms | 83 / 44 ms | 0.52 / 0.81 s |
| ampere, GPU only: GPU setup / build | 32 / 3 ms | 39 / 5 ms | 84 / 30 ms | 0.51 / 0.69 s |
| train total, GPU only / CUDA: hopper | 1.12× | 0.86× | 1.05× | **0.91×** |
| train total, GPU only / CUDA: ampere | 0.92× | 0.94× | 0.91× | **0.90×** |
| whole process (`wall_seconds`, CSV read included), GPU only / CUDA: hopper, ampere | 1.1 / 1.1 s, 0.6 / 0.6 s | 1.2 / 1.2 s, 0.7 / 0.6 s | 3.6 / 3.3 s, 3.2 / 2.7 s | 33 / 39 s, 28 / 39 s |

- **GPU only is about 10% faster than CUDA with hand-offs from 100M rows on** (1000M: H100 0.88 against 0.97 s, A100 1.33 against 1.47 s). The setup and presort are the same; the build is 14–15% shorter. The tree has only 17 nodes, all far above the hand-off size, so the CUDA run hands nothing to the CPU either; what the build saves was not profiled. Up to 10M the two are within the fixed costs' noise (0.86–1.12×).
- **One CPU thread was the bottleneck.** On one thread (7868bf4, earlier that day, hopper's NUMA node 1 with GPU 2) GPU only took 1.69 s (H100) and 1.90 s (A100) at 1000M, 1.45–1.9× CUDA on the whole node: setup 0.70 / 0.61 s, build 0.89 / 1.16 s. The staged copies of the upload and the build's host side need more than one thread; with the socket's threads they take 0.23 / 0.51 s and 0.55 / 0.69 s.
- **Still the fastest on the big sets:** 5.9× (H100) and 3.9× (A100) faster than epyc's 48 threads at 1000M, and 112× and 74× faster than one epyc thread.
- **Less than half the host RAM:** 9.96 GiB at 1000M against 24.87 GiB with hand-offs (no host arena) and 46.70 GiB on the CPU servers. The VRAM is the same as with hand-offs.
- **Reading the CSV is the cost:** the whole 1000M process takes 28–39 s, nearly all of it parsing the 20 GB CSV, which every run reads again over NFS (`--drop-cache`); on one thread it took 70 s (hopper) and 106 s (ampere). The training time above excludes it.

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

The 3-run results above had dell512 sometimes faster than dell, although dell has twice the memory channels. To check, `chess_1b` ran 10 times per case on both nodes. That was first commit 41faca9, the same CPU code as d9da007, on 2026-10-19 at 20:07–20:13, and was repeated with 8b1ecd1 on 2026-10-20 from 13:57 (the table). The results are in `dell-variance/`.

```bash
bench/cluster/submit.sh -m 10 -d chess_1b -p cart_full_nodepth,c45_unpruned -o cluster-chess/dell-variance \
    cpu-dell-chess_1b cpu-dell512-chess_1b
d=bench/results/cluster-chess/dell-variance
bench/.venv/bin/python bench/runs_chart.py $d/cpu-dell-chess_1b $d/cpu-dell512-chess_1b --out $d/chart.png
```

![dell and dell512, chess 1000M: every run of train total, presort and build](dell-variance/chart.png)

| Median of 10 (min–max) | dell (1 TB, 16 DIMMs) | dell512 (512 GB, 8 DIMMs) | dell512 / dell |
|---|--:|--:|--:|
| CART: train total | 9.66 s (8.92–11.39) | 11.84 s (10.67–14.59) | 1.23× |
| CART: presort | 4.05 s | 4.94 s | 1.22× |
| CART: build | 5.55 s | 6.77 s | 1.22× |
| C4.5: train total | 12.27 s (11.50–16.19) | 14.31 s (13.34–21.98) | 1.17× |
| C4.5: presort | 3.97 s | 5.06 s | 1.27× |
| C4.5: build | 8.08 s | 9.38 s | 1.16× |
| Memory bandwidth, 48 threads (triad) | 379 GB/s | 220 GB/s | 0.58× |
| Memory bandwidth, 1 thread (triad) | 17.2 GB/s | 16.3 GB/s | 0.95× |

- **dell is faster in every phase**, 16–27% in the medians (12–22% at 41faca9), and in train total dell512's fastest run is slower than dell's median.
- **The earlier "dell512 faster" was noise.** The 3-run CART medians were 11.18 s (dell) against 10.74 s (dell512), but the runs of one node spread by 10–15%, and dell512 had two slow CART runs out of 10 (16.4 s; build 11–12 s instead of 6–7 s). In the 8b1ecd1 repeat dell512 again had the slowest runs (CART up to 14.6 s, C4.5 up to 22.0 s).
- **Bandwidth matters only partly.** With 48 threads dell512 has 58% of dell's bandwidth, but it takes only about 1.2× the time. So chess is partly limited by memory bandwidth, and partly by other work: the per-thread speed and the synchronisation between threads. One thread alone gets the same 16–17 GB/s on both nodes.
- **The bandwidth was measured** with `bench/cluster/triad.c`, a STREAM-style copy and triad over 3 × 4 GiB arrays, best of 10. It was compiled `gcc -O3 -fopenmp` and run on each whole node. Logs: `dell-variance/triad-dell{1,2}.log`.

## One chart per size

![chess 3x3, 1M rows, on five mordor3 nodes](chart-chess_1m.png)

![chess 3x3, 10M rows, on five mordor3 nodes](chart-chess_10m.png)

![chess 3x3, 100M rows, on five mordor3 nodes](chart-chess_100m.png)

![chess 3x3, 1000M rows, on five mordor3 nodes](chart-chess_1b.png)

## Notes

- **Not timed:** reading the CSV, and the data check before each job (SHA-256 of every prepared file and every CSV row against the binary files: 113 s for `chess_1b` on ampere, 33 GB over NFS). A whole `chess_1b` process (`wall_seconds`, reading the 20 GB CSV included) took 28–40 s on the GPU nodes and 41–59 s on one socket of epyc and dell, whose runs read the CSV over NFS each time (`--drop-cache`), 24–25 s on epyc's both sockets, 13–17 s on dell and dell512 (4.8–5.3 s on the GPU nodes and 9–18 s on epyc with the file cached, 37–52 s before the 1000M fix); with one thread 70 s (hopper) and 106 s (ampere) GPU only, and 154–178 s serial.
- **Hopper's and ampere's 1M and 10M runs** are dominated by fixed costs; their order there says little about the GPUs.
- **The originals** (C4.5 files from mordor3's `~/chess`) were deleted after the conversion; `datasets/chess_*.csv` on mordor3 are the only copies. Their SHA-256 and those of the originals are in [bench/cluster/README.md](../../cluster/README.md) and `bench/cluster/chess-originals.sha256`.
