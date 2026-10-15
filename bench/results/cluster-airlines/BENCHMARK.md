# Airline on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers, with and without pruning

**Status:** run on 2026-10-15, 3 runs per case, at commit 35cf315: C4.5 pruning from the leaves the growers record, the GPU partition's bitmap, the staged upload and CART's smaller hand-off ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)). `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **Pruned trees** (the run directories here): hopper, epyc1 and dell1 from 16:51, each right after its HIGGS run ([cluster-higgs](../cluster-higgs/BENCHMARK.md)); ampere from 16:55, when another job on it had ended.
- **Unpruned trees** (`noprunning/`): right after the pruned ones, from 16:53 (ampere 16:57). The previous commit's `noprunning/` lacked the tables of dell and ampere (copied before those runs ended); this run has all four.
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
| CUDA · hopper (H100 PCIe + 32 threads) | **1.19 s** (1.18–1.21) | **7.46 s** (7.00–8.05) | 3.38 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 1.99 s (1.95–1.99) | 8.65 s (8.59–8.68) | 3.91 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 7.35 s (7.34–7.36) | 12.34 s (12.33–12.36) | 1.36 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 8.36 s (8.24–8.42) | 14.78 s (14.65–15.09) | 2.41 s |

The trees:
- **CART:** 3,013 nodes, depth 25, training accuracy 71.54%. It grows 0.1 million nodes before pruning.
- **C4.5:** 8,548,725 nodes (19.6 million before pruning), depth 72, training accuracy 87.42%.
- **Accuracy:** `ActualElapsedTime` partly gives the delay away, so the accuracies say little. The dataset is here to compare speed, as in gbm-bench.

Against the earlier code (medians of 3): before the tuning for these servers (a8563b3, measured during the tuning, hopper and epyc only), after it (16cc93b, 2026-10-14 at 18:08), the previous run (a83b6ec: the node store and the GPU hand-off, 2026-10-14 at 19:53), and now.

| | CART: before → tuned → a83b6ec → now | C4.5: before → tuned → a83b6ec → now |
|---|--:|--:|
| CUDA · hopper | 11.64 → 2.31 → 2.20 → **1.19 s** | 26.03 → 15.83 → 11.26 → **7.46 s** |
| CUDA · ampere | – → 3.37 → 2.84 → **1.99 s** | – → 13.19 → 11.60 → **8.65 s** |
| Parallel · epyc | 15.98 → 7.34 → 7.34 → **7.35 s** | 24.67 → 13.96 → 13.91 → **12.34 s** |
| Parallel · dell | – → 8.36 → 8.43 → **8.36 s** | – → 16.60 → 16.41 → **14.78 s** |

## Unpruned trees

The same trees grown without pruning, so CART and C4.5 do comparable work: each grows its full tree. The bars are not split, because nothing is pruned.
- **CART** (`cart_full_nodepth`, `./tree --cart --no-prune`): without pruning, CART's early stop with a fixed α (docs/ALGORITHMS.md) does not apply. The tree grows until every leaf is pure or cannot be split.
- **C4.5** (`c45_unpruned`, `./tree --c45 --no-prune`): C4.5's collapse of useless splits still runs. It is part of growing, done with or without pruning.

![airline on four mordor3 nodes, unpruned trees](chart-noprunning.png)

Training time, median of 3 runs (min–max):

| | CART, no pruning | C4.5, no pruning |
|---|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **3.37 s** (3.29–3.44) | **3.66 s** (3.56–4.06) |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 4.22 s (4.14–4.39) | 4.80 s (4.71–4.81) |
| Parallel · epyc (EPYC 9255, 48 threads) | 8.74 s (8.72–8.81) | 11.10 s (10.95–11.13) |
| Parallel · dell (Xeon 6542Y, 48 threads) | 10.05 s (9.93–10.11) | 12.47 s (12.28–12.75) |

The trees:
- **CART:** 47,146,877 nodes, depth 65, training accuracy 100.00%: every leaf is pure.
- **C4.5:** 19,579,127 nodes, depth 74, training accuracy 90.60%. C4.5 stops earlier, at leaves of 2 rows and at splits that gain nothing.

Against the earlier code: 16cc93b (2026-10-14 at 17:58), the previous run (a83b6ec, 19:55), now, and CART capped at depth 30 (`cart_full`, `-d 30`, 21,600,857 nodes, run at 17:13 on 16cc93b):

| Median | CART: depth ≤ 30 (16cc93b) | CART: 16cc93b → a83b6ec → now | C4.5: 16cc93b → a83b6ec → now |
|---|--:|--:|--:|
| CUDA · hopper | 4.66 s | 13.80 → 4.06 → **3.37 s** | 7.17 → 4.39 → **3.66 s** |
| CUDA · ampere | 5.20 s | 7.14 → 4.67 → **4.22 s** | 7.11 → 5.39 → **4.80 s** |
| Parallel · epyc | 8.39 s | 8.89 → 8.76 → **8.74 s** | 11.13 → 11.08 → **11.10 s** |
| Parallel · dell | 9.64 s | 10.51 → 10.07 → **10.05 s** | 12.64 → 12.58 → **12.47 s** |

## Observations

- **CART, pruned:** the H100 takes 1.19 s, 6.2× the fastest CPU node (epyc, 7.35 s), and the A100 1.99 s (3.7×). CART got 1.8× (H100) and 1.4× (A100) faster than in the previous run:
  - The partition's goes-left flags are now one bit per row (14 MB, kept in L2) instead of one byte (115 MB).
  - Nodes go to the CPU at 2,048 rows instead of 4,096.
  - The hand-off buffer is page-locked only as far as the hand-offs reach (2 of 12 GB), which also cut the peak RSS from 17.2 to 8.1 GiB.
  - The dataset goes up through page-locked staging buffers.
- **C4.5, pruned:** the GPUs now beat epyc by 1.7× (H100 7.46 s) and 1.4× (A100 8.65 s), against 1.2× in the previous run.
  - **Pruning got 1.6–2.2× faster everywhere:** 6.9 → 3.4 s on hopper, 6.5 → 3.9 s on ampere, 3.0 → 1.4 s on epyc, 3.9 → 2.4 s on dell. The growers now record each row's leaf, so the pruning no longer routes 115M rows through the 19.6-million-node tree, and its recursion no longer blocks threads on joins.
  - **Growing alone,** without the pruning, takes 4.1 s on the H100 and 4.7 s on the A100, against epyc's 11.0 s.
  - **Pruning is still 45% of a CUDA run.** It runs on the host CPU, and epyc, with faster cores, prunes in 1.4 s.
  - **Hopper's C4.5 varies the most** (7.00–8.05 s). Its CPU timings drift during the day (bench/cluster/README.md); a same-job comparison of both commits gave 7.11 s ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)).
- **Unpruned trees:** the GPUs are 2.1–2.6× epyc on CART (47.1 million nodes) and 2.3–3.0× on C4.5 (19.6 million). Almost all of these nodes are small and grown by the CPU threads, so the GPU node's CPU matters as much as its GPU. The partition bitmap, the smaller CART hand-off and the staged upload made both GPUs 10–17% faster here.
- **The depth cap hid most of CART's tree.** Without it CART grows 47.1 million nodes, 2.2× as many as up to depth 30 and 2.4× C4.5's. On the CPU nodes the extra 25 million nodes cost little (epyc 8.39 s capped on 16cc93b, 8.74 s uncapped now): they lie deep down, with a few rows each.
- **CART against C4.5 on the CPU nodes:** CART grows its 2.4× larger tree in 79–81% of C4.5's time (epyc 8.74 against 11.10 s).
- **The parallel CPU build is unchanged** (CART and unpruned trees within 1% of the previous run); only C4.5's pruning moved.
- **CPU nodes:** dell takes 14–20% longer than epyc on the pruned trees and 12–15% on the unpruned ones.
- **Memory:** the same on every node of a kind.
  - Parallel: pruned 29.0 GiB (CART) and 38.5 GiB (C4.5) peak RSS; unpruned 32.5 and 37.0 GiB.
  - CUDA: pruned 8.1 / 26.4 GiB RSS, unpruned 20.9 / 25.0 GiB, plus 23.8–24.3 GiB on the GPU, CUDA context included.
