# Airline on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers, with and without pruning

**Status:** run on 2026-10-14, 3 runs per case, at commit a83b6ec: the node store fix and the GPU hand-off of [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md).
- **Pruned trees** (the run directories here): all four nodes from 19:53, each right after its HIGGS run ([cluster-higgs](../cluster-higgs/BENCHMARK.md)).
- **Unpruned trees** (`noprunning/`): all four nodes from 19:55.
- **Each node ran alone** and was used whole.
- **Same trees:** every case built the same tree on every node and backend.
- **Settings:** ./tree's defaults on every node. They were the best found for each machine; other values of `--task-rows`, `--feature-parallel-rows` and `--gpu-min-rows`, and running on one socket only, were no faster (docs/PERFORMANCE.md).

How the runs are made, and the dataset: [bench/cluster/README.md](../../cluster/README.md).
- **GPU servers run only CUDA, CPU servers only parallel.**
- **CART has no depth limit**, like C4.5. The other benchmarks cap CART at depth 30 only because rpart cannot grow deeper.
- **Accuracy is ./tree's training accuracy** (`run.py --train-accuracy-only`).

```bash
bench/cluster/submit.sh -m 3 -d airline                       # pruned: cart_alpha_nodepth, c45
bench/cluster/submit.sh -m 3 -d airline -p cart_full_nodepth,c45_unpruned \
    -o cluster-airlines/noprunning                            # unpruned
```

The dataset is gbm-bench's airline: US flights 1987–2008, 114,569,017 training rows, 13 integer features, label "arrival delayed". It has 11× HIGGS's rows. It exists only on the cluster: the CUDA backend needs 23.6 GB of GPU memory for it (two copies of the presorted columns).

| Run | Node | CPU | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|
| `cuda-hopper-airline` | hopper | 2× Xeon Gold 6444Y, 32 cores | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-airline` | ampere | 2× EPYC 7402, 48 cores | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-airline` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | – | parallel (48 threads) |
| `cpu-dell-airline` | dell1 | 2× Xeon Gold 6542Y, 48 cores | – | parallel (48 threads) |

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the 6 GB CSV).

## Pruned trees

`cart_alpha_nodepth` (CART, fixed α = 1e-5) and `c45` (C4.5, CF 0.25). A bar's lighter part is the pruning.

![airline on four mordor3 nodes, pruned trees](chart.png)

Training time, median of 3 runs (min–max):

| | CART | C4.5 | of which C4.5 pruning |
|---|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **2.20 s** (2.08–2.20) | **11.26 s** (11.10–11.94) | 6.89 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 2.84 s (2.84–2.87) | 11.60 s (11.52–11.62) | 6.53 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 7.34 s (7.31–7.40) | 13.91 s (13.76–13.92) | 2.97 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 8.43 s (8.33–8.54) | 16.41 s (15.98–16.45) | 3.93 s |

The trees:
- **CART:** 3,013 nodes, depth 25, training accuracy 71.54%. It grows 0.1 million nodes before pruning.
- **C4.5:** 8,548,725 nodes (19.6 million before pruning), depth 72, training accuracy 87.42%.
- **Accuracy:** `ActualElapsedTime` partly gives the delay away, so the accuracies say little. The dataset is here to compare speed, as in gbm-bench.

Against the earlier code (medians of 3): before the tuning for these servers (a8563b3, measured during the tuning, hopper and epyc only), after it (16cc93b, run at 18:08 the same day), and now.

| | CART: before → tuned → now | C4.5: before → tuned → now |
|---|--:|--:|
| CUDA · hopper | 11.64 → 2.31 → **2.20 s** | 26.03 → 15.83 → **11.26 s** |
| CUDA · ampere | – → 3.37 → **2.84 s** | – → 13.19 → **11.60 s** |
| Parallel · epyc | 15.98 → 7.34 → **7.34 s** | 24.67 → 13.96 → **13.91 s** |
| Parallel · dell | – → 8.36 → **8.43 s** | – → 16.60 → **16.41 s** |

## Unpruned trees

The same trees grown without pruning, so CART and C4.5 do comparable work: each grows its full tree. The bars are not split, because nothing is pruned.
- **CART** (`cart_full_nodepth`, `./tree --cart --no-prune`): without pruning, CART's early stop with a fixed α (docs/ALGORITHMS.md) does not apply. The tree grows until every leaf is pure or cannot be split.
- **C4.5** (`c45_unpruned`, `./tree --c45 --no-prune`): C4.5's collapse of useless splits still runs. It is part of growing, done with or without pruning.

![airline on four mordor3 nodes, unpruned trees](chart-noprunning.png)

Training time, median of 3 runs (min–max):

| | CART, no pruning | C4.5, no pruning |
|---|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **4.06 s** (4.04–4.07) | **4.39 s** (4.36–4.50) |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 4.67 s (4.66–4.68) | 5.39 s (5.37–5.41) |
| Parallel · epyc (EPYC 9255, 48 threads) | 8.76 s (8.74–8.86) | 11.08 s (10.99–11.11) |
| Parallel · dell (Xeon 6542Y, 48 threads) | 10.07 s (10.06–10.08) | 12.58 s (12.58–12.58) |

The trees:
- **CART:** 47,146,877 nodes, depth 65, training accuracy 100.00%: every leaf is pure.
- **C4.5:** 19,579,127 nodes, depth 74, training accuracy 90.60%. C4.5 stops earlier, at leaves of 2 rows and at splits that gain nothing.

Against the earlier code (16cc93b, run at 17:58 the same day), and CART capped at depth 30 (`cart_full`, `-d 30`, 21,600,857 nodes, run at 17:13 on that code):

| Median | CART: depth ≤ 30 (old code) | CART: before → now | C4.5: before → now |
|---|--:|--:|--:|
| CUDA · hopper | 4.66 s | 13.80 → **4.06 s** | 7.17 → **4.39 s** |
| CUDA · ampere | 5.20 s | 7.14 → **4.67 s** | 7.11 → **5.39 s** |
| Parallel · epyc | 8.39 s | 8.89 → **8.76 s** | 11.13 → **11.08 s** |
| Parallel · dell | 9.64 s | 10.51 → **10.07 s** | 12.64 → **12.58 s** |

## Observations

- **Hopper's slow unpruned CART is fixed.** It took 13.80 s, twice ampere's time, because the CPU threads that grow the small subtrees kept waiting for each other in the node store: all of them reached the end of a shared 16K-node block together and waited while one allocated the next. On hopper, with two sockets and the `schedutil` governor, the clocks then fell to about 0.9 GHz, and ./tree ran faster on one socket than on two. Each thread now allocates its own blocks: 4.06 s, the fastest of the four nodes (details in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)).
- **The GPU now hands nodes to the CPU at 4,096 rows** instead of about 28,000. With the CPU threads no longer waiting, keeping more of the tree on the GPU was 2–15% faster on both GPUs (one run each, docs/PERFORMANCE.md).
- **CART, pruned:** the H100 takes 2.20 s, 3.3× the fastest CPU node (epyc, 7.34 s); the A100 2.84 s. With 115M rows the GPU does most of the work. Its setup (uploading 6 GB: 0.36 s on the H100, 0.49 s on the A100) and the per-level round trips matter less than on HIGGS.
- **C4.5, pruned:** the GPUs now beat epyc, by 1.2× (H100 11.26 s, A100 11.60 s, epyc 13.91 s). Before, they only tied with it.
  - **Growing alone,** without the pruning, the GPUs take 4.4 s (H100) and 5.1 s (A100) against epyc's 10.9 s.
  - **Pruning is now 56–61% of a CUDA run** (6.5–6.9 s). It runs on the host CPU, and epyc prunes in 3.0 s with its faster cores. C4.5's pruning on the GPU nodes is the next place to gain.
- **Unpruned trees:** the GPUs are 1.9–2.2× epyc on CART (47.1 million nodes) and 2.1–2.5× on C4.5 (19.6 million). Almost all of these nodes are small and grown by the CPU threads, so the GPU node's CPU matters as much as its GPU.
- **The depth cap hid most of CART's tree.** Without it CART grows 47.1 million nodes, 2.2× as many as up to depth 30 and 2.4× C4.5's. On the CPU nodes the extra 25 million nodes cost little (epyc 8.39 s capped on the old code, 8.76 s uncapped now): they lie deep down, with a few rows each.
- **CART against C4.5 on the CPU nodes:** CART grows its 2.4× larger tree in 79–80% of C4.5's time (epyc 8.76 against 11.08 s).
- **C4.5's pruning is the big cost of a pruned C4.5 run:** 2.8 s on epyc (11.08 → 13.91 s), 6.2–6.9 s on the GPU nodes, whose CPUs are slower.
- **CPU nodes:** dell takes 15–18% longer than epyc on the pruned trees and 14–15% on the unpruned ones. The node store fix changed little there (up to 4%).
- **Memory:** the same on every node of a kind.
  - Parallel: pruned 29.0 GiB (CART) and 38.4–38.5 GiB (C4.5) peak RSS; unpruned 32.5 and 36.9–37.0 GiB.
  - CUDA: pruned 17.2 / 26.3 GiB RSS, unpruned 20.7 / 24.9 GiB, plus 23.9–24.0 GiB on the GPU, CUDA context included.
