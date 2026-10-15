# HIGGS on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers

**Status:** run on 2026-10-15 from 16:51, 3 runs per case, at commit 35cf315: C4.5 pruning from the leaves the growers record, the GPU partition's bitmap, the staged upload and CART's smaller hand-off ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md); on HIGGS CART now hands nodes to the CPU at 1,281 rows instead of 2,563). `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **Each node ran alone** and was used whole. hopper, epyc1 and dell1 started together; ampere at 16:55, when another job on it had ended.
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
| CUDA · hopper (H100 PCIe + 32 threads) | **0.41 s** (0.39–0.42) | 1.39 s (1.38–1.40) | 0.55 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 0.64 s (0.62–0.67) | **1.37 s** (1.37–1.39) | 0.47 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 1.10 s (1.05–1.11) | 1.82 s (1.76–1.84) | 0.19 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 1.29 s (1.23–1.31) | 2.30 s (2.28–2.33) | 0.29 s |

The trees:
- **CART:** 1,611 nodes, depth 19, training accuracy 71.79%. It prunes in 1–2 ms: with a fixed α it does not grow what pruning would cut back.
- **C4.5:** 1,159,295 nodes, depth 71, training accuracy 87.88%.

### Against the earlier code

Before the tuning for these servers (commit ab8ebdd, the first cluster run, 2026-10-14 at 12:45; its files are no longer in the repository, commit 781f28b still has them in `bench/results/cluster-higgs/before/`), the previous run (a83b6ec, 2026-10-14 at 19:53), and now. Medians of 3.

| | CART: before → a83b6ec → now | C4.5: before → a83b6ec → now |
|---|--:|--:|
| CUDA · hopper | 1.14 → 0.44 → **0.41 s** | 2.33 → 1.50 → **1.39 s** |
| CUDA · ampere | 1.34 → 0.63 → **0.64 s** | 2.48 → 1.46 → **1.37 s** |
| Parallel · epyc | 1.91 → 1.07 → **1.10 s** | 2.43 → 1.84 → **1.82 s** |
| Parallel · dell | 1.96 → 1.23 → **1.29 s** | 2.85 → 2.32 → **2.30 s** |

What changed is in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md). Since a83b6ec, C4.5's pruning got faster on every node (hopper 0.72 → 0.55 s, ampere 0.58 → 0.47 s, epyc 0.27 → 0.19 s, dell 0.35 → 0.29 s). CART's CPU code did not change; its 3–5% on the CPU nodes is within the day's variation. On airline, 11× larger, the same changes gain much more ([cluster-airlines](../cluster-airlines/BENCHMARK.md)).

## Observations

- **CART:** the H100 takes 0.41 s, 2.7× the fastest CPU node (epyc, 1.10 s), and the A100 0.64 s. Only 1.2 GB of data moves to the GPU, so the GPU setup and the per-level round trips are a large part of these times.
- **C4.5:** the GPUs gain less: 1.39 s (H100) and 1.37 s (A100), against 1.82 s on epyc's 48 cores (1.3×). The pruning runs on the host CPU and is 0.5–0.55 s of a CUDA run, 34–40%. The 1.16-million-node tree's small subtrees also grow on the host. Both GPU nodes have slower CPUs than epyc, which prunes in 0.19 s.
- **Hopper against ampere on C4.5:** level (1.39 against 1.37 s). Hopper grows the tree a little faster (0.84 against 0.90 s without the pruning) and prunes more slowly (0.55 against 0.47 s): the pruning runs on the host, and hopper has 32 cores, ampere 48.
- **CPU nodes:** dell takes 17–26% longer than epyc (CART 1.29 against 1.10 s, C4.5 2.30 against 1.82 s). How each scales with the thread count: [cluster-cpu-scaling](../cluster-cpu-scaling/BENCHMARK.md).
- **Against the laptop and desktop** ([`pc-legion-higgs`](../pc-legion-higgs/BENCHMARK.md)): CART took 1.37 s on the RTX 5070 Laptop GPU, 3.22 s on the laptop's 20 threads and 3.47 s on the desktop's 28. That was older code (49b7f47), so it is not a like-for-like comparison; the current code was not measured there.
- **Memory:** the same on every node of a kind.
  - Parallel: 5.7 GiB (CART) and 7.2 GiB (C4.5) peak RSS.
  - CUDA: 3.7 / 5.1 GiB RSS plus 4.9–5.0 GiB on the GPU, CUDA context included.
