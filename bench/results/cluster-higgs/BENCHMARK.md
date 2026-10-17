# HIGGS on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers

**Status:** run on 2026-10-17 from 17:47, 3 runs per case, at commit cd18717: one copy of the presorted columns on every backend, the GPU's hand-off through a 512 MB ring ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)). `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **Each node ran alone** and was used whole. All four started together.
- **Settings:** ./tree's defaults on every node. They were the best found for each machine; other values of `--task-rows`, `--feature-parallel-rows` and `--gpu-min-rows`, and running on one socket only, were no faster (docs/PERFORMANCE.md).
- **Same trees:** every case built the same tree on every node and backend, as in the earlier runs and on the desktop and laptop.

How the runs are made and what is specific to the cluster: [bench/cluster/README.md](../../cluster/README.md).
- **GPU servers run only CUDA, CPU servers only parallel.** A GPU server's CPU is there to feed the GPU; it grows the small subtrees and prunes, and is not benchmarked on its own.
- **CART has no depth limit** (`cart_alpha_nodepth`), like C4.5. The other benchmarks cap CART at depth 30 only for rpart. Here the cap would make no difference: the pruned tree is depth 19.
- **Accuracy is ./tree's training accuracy.** The test rows are not classified (`run.py --train-accuracy-only`).

| Run | Node | CPU | RAM | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|---|
| `cuda-hopper-higgs` | hopper | 2× Xeon Gold 6444Y, 32 cores | 1 TB DDR5-4800 | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-higgs` | ampere | 2× EPYC 7402, 48 cores | 1 TB DDR4-3200 | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-higgs` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – | parallel (48 threads) |
| `cpu-dell-higgs` | dell1 | 2× Xeon Gold 6542Y, 48 cores | 1 TB DDR5-5200 | – | parallel (48 threads) |

## Results

![HIGGS on four mordor3 nodes](chart.png)

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the CSV). Its lighter part is the pruning.

Training time, median of 3 runs (min–max):

| | CART | C4.5 | of which C4.5 pruning |
|---|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **0.41 s** (0.39–0.42) | 1.22 s (1.04–1.24) | 0.50 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 0.56 s (0.54–0.59) | **1.22 s** (1.22–1.25) | 0.42 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 1.09 s (1.07–1.10) | 1.74 s (1.71–1.74) | 0.19 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 1.30 s (1.23–1.30) | 2.26 s (2.25–2.27) | 0.27 s |

The trees:
- **CART:** 1,611 nodes, depth 19, training accuracy 71.79%. It prunes in 1–2 ms: with a fixed α it does not grow what pruning would cut back.
- **C4.5:** 1,159,295 nodes, depth 71, training accuracy 87.88%.

### Against the earlier code

Before the tuning for these servers (commit ab8ebdd, the first cluster run, 2026-10-14 at 12:45; its files are no longer in the repository, commit 781f28b still has them in `bench/results/cluster-higgs/before/`), a83b6ec (2026-10-14 at 19:53), 35cf315 (2026-10-15 at 16:51), the previous run (8da0c12, 2026-10-15 at 23:23), and now. Medians of 3.

| | CART: before → a83b6ec → 35cf315 → 8da0c12 → now | C4.5: before → a83b6ec → 35cf315 → 8da0c12 → now |
|---|--:|--:|
| CUDA · hopper | 1.14 → 0.44 → 0.41 → 0.41 → **0.41 s** | 2.33 → 1.50 → 1.39 → 1.24 → **1.22 s** |
| CUDA · ampere | 1.34 → 0.63 → 0.64 → 0.52 → **0.56 s** | 2.48 → 1.46 → 1.37 → 1.15 → **1.22 s** |
| Parallel · epyc | 1.91 → 1.07 → 1.10 → 1.09 → **1.09 s** | 2.43 → 1.84 → 1.82 → 1.73 → **1.74 s** |
| Parallel · dell | 1.96 → 1.23 → 1.29 → 1.26 → **1.30 s** | 2.85 → 2.32 → 2.30 → 2.30 → **2.26 s** |

What changed is in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md). Since 8da0c12 the presorted columns are kept once, not twice, on every backend: a partition moves each column into a free slot (one spare slot on the GPU) instead of into a second copy. The GPU's mirror of the hand-offs to the host became a 512 MB ring.
- **GPU memory halved** with the same trees: 7.1–7.2 → 3.3 GiB (CART), 8.3 → 4.4–4.5 GiB (C4.5), Memory below.
- **The times barely moved,** except ampere (CART 0.52 → 0.56 s, C4.5 1.15 → 1.22 s). A same-job comparison on ampere, the old and the new binary interleaved, 5 runs each (`~/vram-dt/ab.sh`), gave 0.50 → 0.51 s (CART) and 1.19 → 1.16 s (C4.5): most of the gap is the run-to-run drift between days, not the change.
- **C4.5's presort** got faster on both GPUs (hopper 0.07 → 0.03 s, ampere 0.08 → 0.04 s): it sorts straight into the column slots.
- **The CPU nodes** are within 3% (dell CART 1.26 → 1.30 s, C4.5 2.30 → 2.26 s).

## Observations

- **CART:** the H100 takes 0.41 s, 2.7× the fastest CPU node (epyc, 1.09 s), and the A100 0.56 s (1.9×). Only 1.2 GB of data moves to the GPU, so the GPU setup and the per-level round trips are a large part of these times.
- **C4.5:** the GPUs gain less: 1.22 s (A100 and H100), against 1.74 s on epyc's 48 cores (1.4×). The pruning runs on the host CPU and is 0.42–0.50 s of a CUDA run, 34–41%. The 1.16-million-node tree's small subtrees also grow on the host. Both GPU nodes have slower CPUs than epyc, which prunes in 0.19 s.
- **Hopper against ampere on C4.5:** the same median (1.22 s). Without the pruning hopper is faster (0.72 against 0.80 s); ampere prunes faster (0.42 against 0.50 s): the pruning runs on the host, and ampere has 48 cores, hopper 32. hopper's C4.5 also varied the most (1.04–1.24 s).
- **CPU nodes:** dell takes 19–30% longer than epyc (CART 1.30 against 1.09 s, C4.5 2.26 against 1.74 s). How each scales with the thread count: [cluster-cpu-scaling](../cluster-cpu-scaling/BENCHMARK.md).
- **Against the laptop and desktop** ([`pc-legion-higgs`](../pc-legion-higgs/BENCHMARK.md)): CART took 1.37 s on the RTX 5070 Laptop GPU, 3.22 s on the laptop's 20 threads and 3.47 s on the desktop's 28. That was older code (49b7f47), so it is not a like-for-like comparison; the current code was not measured there.
- **Memory:** the same on every node of a kind.
  - Parallel: 5.6 GiB (CART) and 6.0 GiB (C4.5) peak RSS (5.7 and 6.1 GiB before). With 48 threads and 28 features the parallel backend keeps min(features − 1, threads) = 27 spare column slots, so it saves little here; with fewer threads it saves more ([cluster-cpu-scaling](../cluster-cpu-scaling/BENCHMARK.md)).
  - CUDA: 3.7 / 4.1 GiB RSS, as before, plus 3.3 / 4.4–4.5 GiB on the GPU, CUDA context included (7.1–7.2 / 8.3 GiB before): one copy of the presorted columns (8 bytes per value) with one spare column, the 512 MB ring and, for pruned C4.5, a copy of the raw values (4 bytes per value) for the pruning's first step, kept only when it fits.
