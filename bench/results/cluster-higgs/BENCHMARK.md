# HIGGS on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers

**Status:** every row rerun on 2026-10-20 from 13:52, 3 runs per case, at commit 8b1ecd1: 6521330 (the class counts kept per thread in local histograms instead of a shared, false-shared array; the staged GPU copies use at least 8 threads) plus the cluster scripts' NUMA options (unused here: each run had the whole node). The same trees as before; ampere's CART 13% faster, dell 5–7%, the rest within 3%. `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **Before:** the CPU rows were 051949b's (2026-10-19 from 15:10), the CUDA rows 6f6cecb's (2026-10-19 from 20:15): the GPU's hand-offs through page-locked buffers, the host partition in blocks ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)). Their medians are in the history table below.
- **epyc rerun** on 2026-10-20 at 1a2d9a9, the same code with `run.py --drop-cache`: our own cached dataset files had filled epyc1's NUMA node 1, which made the 48-thread runs allocate across the sockets. HIGGS was hardly affected (CART 1.06 → 1.07 s, C4.5 1.69 s both times); the larger datasets were ([cluster-ecbdl](../cluster-ecbdl/BENCHMARK.md)). The jobs now also drop the datasets from the page cache when they end (99688fa).
- **CUDA rows rerun** on 2026-10-20 from 18:26 at 235c431, on the GPU's socket only: GPU 0, its NUMA node 0's CPUs (16 threads on hopper, 24 on ampere) and the memory bound to that node. The CPU part of a CUDA run (small subtrees, pruning) now has half the threads: CART 7–8% and C4.5 18–24% slower than on the whole node (below), the same trees.
- **Each node ran alone** and was used whole. All four started together.
- **Settings:** ./tree's defaults on every node. They were the best found for each machine; other values of `--task-rows`, `--feature-parallel-rows` and `--gpu-min-rows`, and running on one socket only, were no faster (docs/PERFORMANCE.md).
- **Same trees:** every case built the same tree on every node and backend, as in the earlier runs and on the desktop and laptop.

How the runs are made and what is specific to the cluster: [bench/cluster/README.md](../../cluster/README.md).
- **GPU servers run only CUDA, CPU servers only parallel.** A GPU server's CPU is there to feed the GPU; it grows the small subtrees and prunes, and is not benchmarked on its own.
- **CART has no depth limit** (`cart_alpha_nodepth`), like C4.5. The other benchmarks cap CART at depth 30 only for rpart. Here the cap would make no difference: the pruned tree is depth 19.
- **Accuracy is ./tree's training accuracy.** The test rows are not classified (`run.py --train-accuracy-only`).

| Run | Node | CPU | RAM | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|---|
| `cuda-hopper-higgs` | hopper | 2× Xeon Gold 6444Y, 32 cores | 1 TB DDR5-4800 | H100 PCIe 80 GB | CUDA (+ 16 CPU threads, the GPU's socket) |
| `cuda-ampere-higgs` | ampere | 2× EPYC 7402, 48 cores | 1 TB DDR4-3200 | A100 80 GB PCIe | CUDA (+ 24 CPU threads, the GPU's socket) |
| `cpu-epyc-higgs` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – | parallel (48 threads) |
| `cpu-dell-higgs` | dell1 | 2× Xeon Gold 6542Y, 48 cores | 1 TB DDR5-5200 | – | parallel (48 threads) |
| `socket-epyc-higgs` | epyc1 | one EPYC 9255 (socket 0), 24 cores | its NUMA node 0 | – | parallel (24 threads) |
| `socket-dell-higgs` | dell1 | one Xeon Gold 6542Y (socket 1), 24 cores | its NUMA nodes 2, 3 | – | parallel (24 threads) |

## Results

![HIGGS on four mordor3 nodes](chart.png)

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the CSV). Its darker part is the GPU setup, its lighter part the pruning, each drawn when it is at least 1% of the bar.

Training time, median of 3 runs (min–max):

| | CART | C4.5 | of which C4.5 pruning |
|---|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 16 threads) | **0.42 s** (0.42–0.43) | 1.43 s (1.42–1.45) | 0.45 s |
| CUDA · ampere (A100 80GB PCIe + 24 threads) | 0.49 s (0.49–0.49) | **1.36 s** (1.36–1.37) | 0.36 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 1.07 s (1.04–1.08) | 1.69 s (1.69–1.74) | 0.19 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 1.18 s (1.18–1.22) | 2.14 s (2.13–2.15) | 0.27 s |
| Parallel, one socket · epyc (24 threads) | 1.44 s (1.43–1.46) | 2.49 s (2.48–2.49) | 0.22 s |
| Parallel, one socket · dell (24 threads) | 1.79 s (1.75–1.80) | 3.59 s (3.57–3.59) | 0.36 s |

The trees:
- **CART:** 1,611 nodes, depth 19, training accuracy 71.79%. It prunes in 1–2 ms: with a fixed α it does not grow what pruning would cut back.
- **C4.5:** 1,159,295 nodes, depth 71, training accuracy 87.88%.

### Against the earlier code

Before the tuning for these servers (commit ab8ebdd, the first cluster run, 2026-10-14 at 12:45; its files are no longer in the repository, commit 781f28b still has them in `bench/results/cluster-higgs/before/`), a83b6ec (2026-10-14 at 19:53), 35cf315 (2026-10-15 at 16:51), 8da0c12 (2026-10-15 at 23:23), cd18717 (2026-10-17 at 17:47), 6f6cecb for CUDA and 051949b for the CPUs (2026-10-19), and now (8b1ecd1, 2026-10-20, the whole node). Medians of 3. The CUDA rows on the GPU's socket only (above) are not in this table: they change where the run is, not the code.

| | CART: before → a83b6ec → 35cf315 → 8da0c12 → cd18717 → 6f6cecb/051949b → now | C4.5: before → a83b6ec → 35cf315 → 8da0c12 → cd18717 → 6f6cecb/051949b → now |
|---|--:|--:|
| CUDA · hopper | 1.14 → 0.44 → 0.41 → 0.41 → 0.41 → 0.39 → **0.39 s** | 2.33 → 1.50 → 1.39 → 1.24 → 1.22 → 1.20 → **1.21 s** |
| CUDA · ampere | 1.34 → 0.63 → 0.64 → 0.52 → 0.56 → 0.52 → **0.45 s** | 2.48 → 1.46 → 1.37 → 1.15 → 1.22 → 1.13 → **1.10 s** |
| Parallel · epyc | 1.91 → 1.07 → 1.10 → 1.09 → 1.09 → 1.09 → **1.07 s** | 2.43 → 1.84 → 1.82 → 1.73 → 1.74 → 1.73 → **1.69 s** |
| Parallel · dell | 1.96 → 1.23 → 1.29 → 1.26 → 1.30 → 1.26 → **1.18 s** | 2.85 → 2.32 → 2.30 → 2.30 → 2.26 → 2.24 → **2.14 s** |

What changed is in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md). From 6f6cecb/051949b to 8b1ecd1 the class counts moved to per-thread local histograms (6521330): the same trees, ampere's CART 0.52 → 0.45 s, dell 6–7% faster (CART 1.26 → 1.18 s, C4.5 2.24 → 2.14 s), the others within 3%. From 051949b to 6f6cecb (CUDA rows) the hand-offs stopped page-locking the arena and the host partition works in blocks: the same trees, CART 5–7% and C4.5 2–5% faster. From cd18717 to 051949b only `--profile` (off here) and the count tables' fill changed: the same trees, every time within −3% to +1%. Since 8da0c12 the presorted columns are kept once, not twice, on every backend: a partition moves each column into a free slot (one spare slot on the GPU) instead of into a second copy. The GPU's mirror of the hand-offs to the host became a 512 MB ring.
- **GPU memory halved** with the same trees: 7.1–7.2 → 3.3 GiB (CART), 8.3 → 4.4–4.5 GiB (C4.5), Memory below.
- **The times barely moved,** except ampere (CART 0.52 → 0.56 s, C4.5 1.15 → 1.22 s). A same-job comparison on ampere, the old and the new binary interleaved, 5 runs each (`~/vram-dt/ab.sh`), gave 0.50 → 0.51 s (CART) and 1.19 → 1.16 s (C4.5): most of the gap is the run-to-run drift between days, not the change.
- **C4.5's presort** got faster on both GPUs (hopper 0.07 → 0.03 s, ampere 0.08 → 0.04 s): it sorts straight into the column slots.
- **The CPU nodes** are within 3% (dell CART 1.26 → 1.30 s, C4.5 2.30 → 2.26 s).

## Observations

- **CART:** the H100 takes 0.42 s, 2.6× the fastest CPU node (epyc, 1.07 s), and the A100 0.49 s (2.2×). Only 1.2 GB of data moves to the GPU, so the GPU setup and the per-level round trips are a large part of these times.
- **C4.5:** the GPUs gain less: 1.36 s (A100) and 1.43 s (H100), against 1.69 s on epyc's 48 cores (1.2×). The pruning runs on the host CPU and is 0.36–0.45 s of a CUDA run, 27–31%. On the whole node (both sockets) the GPUs took 1.10 and 1.21 s: the host's half of the work gains from the second socket's cores. The 1.16-million-node tree's small subtrees also grow on the host. Both GPU nodes have slower CPUs than epyc, which prunes in 0.19 s.
- **Hopper against ampere on C4.5:** ampere is slightly ahead (1.36 against 1.43 s; on the whole node 1.10 against 1.21 s). Without the pruning they are even (0.98 against 1.00 s); ampere prunes faster (0.36 against 0.45 s): the pruning runs on the host, and ampere's socket has 24 cores, hopper's 16.
- **One socket against both** (`socket-*`, 2026-10-20 at 10d125d, the same ./tree code: every CPU of the socket with the most free memory, the memory bound to that socket's NUMA nodes, `SOCKET=auto` in job.sh): half the cores take 1.34× (epyc CART) to 1.68× (dell C4.5) as long, not 2×: the second socket adds cores, but probably also traffic between the sockets (not profiled). The same trees; peak RSS 0.2 GiB lower (fewer threads, fewer spare column slots).
- **CPU nodes:** dell takes 10–27% longer than epyc (CART 1.18 against 1.07 s, C4.5 2.14 against 1.69 s). How each scales with the thread count: [cluster-cpu-scaling](../cluster-cpu-scaling/BENCHMARK.md).
- **Against the laptop and desktop** ([`pc-legion-higgs`](../consumer/pc-legion-higgs/BENCHMARK.md)): CART took 1.37 s on the RTX 5070 Laptop GPU, 3.22 s on the laptop's 20 threads and 3.47 s on the desktop's 28. That was older code (49b7f47), so it is not a like-for-like comparison; the current code was not measured there.
- **Memory:** the same on every node of a kind.
  - Parallel: 5.6 GiB (CART) and 6.0 GiB (C4.5) peak RSS (5.7 and 6.1 GiB before). With 48 threads and 28 features the parallel backend keeps min(features − 1, threads) = 27 spare column slots, so it saves little here; with fewer threads it saves more ([cluster-cpu-scaling](../cluster-cpu-scaling/BENCHMARK.md)).
  - CUDA: 3.3–3.4 / 4.3 GiB RSS (3.7 / 4.1 GiB at 051949b: pruned CART no longer touches the whole hand-off arena, and two 128 MB page-locked buffers were added), plus 3.3 / 4.4–4.5 GiB on the GPU, CUDA context included (7.1–7.2 / 8.3 GiB before): one copy of the presorted columns (8 bytes per value) with one spare column, the 512 MB ring and, for pruned C4.5, a copy of the raw values (4 bytes per value) for the pruning's first step, kept only when it fits.
