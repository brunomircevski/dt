# HIGGS on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers

**Status:** run on 2026-10-14 at 19:53, 3 runs per case, at commit a83b6ec: the node store fix and the GPU hand-off of [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md) (on HIGGS the hand-off size is unchanged, 2,563 rows).
- **Each node ran alone** and was used whole; the four runs started together.
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
| CUDA · hopper (H100 PCIe + 32 threads) | **0.44 s** (0.43–0.45) | 1.50 s (1.26–1.53) | 0.72 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 0.63 s (0.62–0.64) | **1.46 s** (1.45–1.46) | 0.58 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 1.07 s (1.04–1.11) | 1.84 s (1.84–1.88) | 0.27 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 1.23 s (1.22–1.27) | 2.32 s (2.31–2.35) | 0.35 s |

The trees:
- **CART:** 1,611 nodes, depth 19, training accuracy 71.79%. It prunes in 1–2 ms: with a fixed α it does not grow what pruning would cut back.
- **C4.5:** 1,159,295 nodes, depth 71, training accuracy 87.88%.

### Against the code before tuning

The first cluster run was made at 12:45, 3 runs per case, with the code before tuning (commit ab8ebdd). Its files are no longer in the repository; commit 781f28b still has them (`bench/results/cluster-higgs/before/`).

| Median of 3 | CART before → now | C4.5 before → now |
|---|--:|--:|
| CUDA · hopper | 1.14 s → **0.44 s** (2.6×) | 2.33 s → **1.50 s** (1.6×) |
| CUDA · ampere | 1.34 s → **0.63 s** (2.1×) | 2.48 s → **1.46 s** (1.7×) |
| Parallel · epyc | 1.91 s → **1.07 s** (1.8×) | 2.43 s → **1.84 s** (1.3×) |
| Parallel · dell | 1.96 s → **1.23 s** (1.6×) | 2.85 s → **2.32 s** (1.2×) |

What changed is in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md): CART's early stop at a fixed α, fewer thread wake-ups, huge pages, a faster GPU upload, the passes over the whole tree on all threads, and the node store. The node store fix alone made C4.5 on the GPUs 4–10% faster here (hopper 1.67 → 1.50 s, ampere 1.52 → 1.46 s, against the previous run of the same day); it matters much more on airline ([cluster-airlines](../cluster-airlines/BENCHMARK.md)).

## Observations

- **CART:** the H100 takes 0.44 s, 2.4× the fastest CPU node (epyc, 1.07 s), and the A100 0.63 s. Only 1.2 GB of data moves to the GPU, so the GPU setup (0.10–0.14 s) and the per-level round trips are a large part of these times.
- **C4.5:** the GPUs gain less: 1.50 s (H100) and 1.46 s (A100), against 1.84 s on epyc's 48 cores. The pruning runs on the host CPU and is 0.6–0.7 s of a CUDA run, 40–48%. The 1.16-million-node tree's small subtrees also grow on the host. Both GPU nodes have slower CPUs than epyc, which prunes in 0.27 s.
- **Hopper against ampere on C4.5:** level (1.50 against 1.46 s). The H100 grows the tree faster (0.57 against 0.61 s), but hopper prunes more slowly (0.72 against 0.58 s): the pruning runs on the host, and hopper has 32 cores, ampere 48. Hopper's CPU timings also drift during the day (bench/cluster/README.md): its fastest C4.5 run took 1.26 s.
- **CPU nodes:** dell takes 15–26% longer than epyc (CART 1.23 against 1.07 s, C4.5 2.32 against 1.84 s). How each scales with the thread count: [cluster-cpu-scaling](../cluster-cpu-scaling/BENCHMARK.md).
- **Against the laptop and desktop** ([`pc-legion-higgs`](../pc-legion-higgs/BENCHMARK.md)): CART took 1.37 s on the RTX 5070 Laptop GPU, 3.22 s on the laptop's 20 threads and 3.47 s on the desktop's 28. That was before this tuning, so it is not the same code. The new defaults were not measured there.
- **Memory:** the same on every node of a kind.
  - Parallel: 5.7 GiB (CART) and 7.1 GiB (C4.5) peak RSS.
  - CUDA: 3.5–3.6 / 5.0 GiB RSS plus 4.9–5.0 GiB on the GPU, CUDA context included.
