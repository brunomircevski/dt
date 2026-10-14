# Airline on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers, with and without pruning

**Status:** run on 2026-10-14, 3 runs per case. The ./tree source is that of commit 16cc93b, with the defaults tuned for these servers ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)). Later commits changed only `bench/`.
- **Pruned trees** (the run directories here):
  - epyc1 and dell1 ran at 17:56 (commit 7cb9fb9), each right after its HIGGS run ([cluster-higgs](../cluster-higgs/BENCHMARK.md)).
  - hopper and ampere ran at 18:08 (commit 67dc294), CUDA only.
- **Unpruned trees** (`noprunning/`): all four nodes ran from 17:57 to 18:03 (commit 7cb9fb9).
- **Each node ran alone** and was used whole.
- **Same trees:** every case built the same tree on every node and backend.

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
| CUDA · hopper (H100 PCIe + 32 threads) | **2.31 s** (2.27–2.34) | 15.83 s (14.85–16.10) | 7.55 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 3.37 s (3.37–3.41) | **13.19 s** (13.11–13.28) | 6.60 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 7.34 s (7.32–7.38) | 13.96 s (13.86–14.06) | 2.87 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 8.36 s (8.35–8.49) | 16.60 s (16.43–16.61) | 3.90 s |

The trees:
- **CART:** 3,013 nodes, depth 25, training accuracy 71.54%. It grows 0.1 million nodes before pruning, instead of the 21.6 million it grew before the tuning.
- **C4.5:** 8,548,725 nodes (19.6 million before pruning), depth 72, training accuracy 87.42%.
- **Accuracy:** `ActualElapsedTime` partly gives the delay away, so the accuracies say little. The dataset is here to compare speed, as in gbm-bench.

Against the code before tuning, from the tuning's own comparison on hopper and epyc1 (one job each, interleaved, 3 runs; [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)):

| Median of 3 | CART before → after | C4.5 before → after |
|---|--:|--:|
| CUDA · hopper | 11.64 s → 2.32 s (5.0×) | 26.03 s → 14.53 s (1.8×) |
| Parallel · epyc | 15.98 s → 7.34 s (2.2×) | 24.67 s → 13.93 s (1.8×) |

## Unpruned trees

The same trees grown without pruning, so CART and C4.5 do comparable work: each grows its full tree. The bars are not split, because nothing is pruned.
- **CART** (`cart_full_nodepth`, `./tree --cart --no-prune`): without pruning, CART's early stop with a fixed α (docs/ALGORITHMS.md) does not apply. The tree grows until every leaf is pure or cannot be split.
- **C4.5** (`c45_unpruned`, `./tree --c45 --no-prune`): C4.5's collapse of useless splits still runs. It is part of growing, done with or without pruning.

![airline on four mordor3 nodes, unpruned trees](chart-noprunning.png)

Training time, median of 3 runs (min–max):

| | CART, no pruning | C4.5, no pruning |
|---|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | 13.80 s (13.78–14.35) | 7.17 s (6.62–7.18) |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | **7.14 s** (7.08–7.18) | **7.11 s** (7.05–7.14) |
| Parallel · epyc (EPYC 9255, 48 threads) | 8.89 s (8.88–8.93) | 11.13 s (11.11–11.16) |
| Parallel · dell (Xeon 6542Y, 48 threads) | 10.51 s (10.47–10.59) | 12.64 s (12.59–13.00) |

The trees:
- **CART:** 47,146,877 nodes, depth 65, training accuracy 100.00%: every leaf is pure.
- **C4.5:** 19,579,127 nodes, depth 74, training accuracy 90.60%. C4.5 stops earlier, at leaves of 2 rows and at splits that gain nothing.

Pruned against unpruned, and CART against an earlier unpruned run (17:13) capped at depth 30 (`cart_full`, `-d 30`), whose tree had 21,600,857 nodes:

| Median | CART pruned → unpruned | CART unpruned, depth ≤ 30 | C4.5 pruned → unpruned |
|---|--:|--:|--:|
| CUDA · hopper | 2.31 s → 13.80 s | 4.66 s | 15.83 s → 7.17 s |
| CUDA · ampere | 3.37 s → 7.14 s | 5.20 s | 13.19 s → 7.11 s |
| Parallel · epyc | 7.34 s → 8.89 s | 8.39 s | 13.96 s → 11.13 s |
| Parallel · dell | 8.36 s → 10.51 s | 9.64 s | 16.60 s → 12.64 s |

## Observations

Pruned trees:
- **CART:** CUDA wins clearly. The H100 takes 2.31 s, 3.2× the fastest CPU node (epyc, 7.34 s); the A100 takes 3.37 s. With 115M rows the GPU does most of the work. Its setup (uploading 6 GB: 0.35 s on the H100, 0.51 s on the A100) and the per-level round trips matter less than on HIGGS.
- **C4.5:** the GPUs only tie with epyc's 48 cores (A100 13.2 s, epyc 14.0 s, H100 15.8 s).
  - **Pruning is half of a CUDA run** (6.6–7.6 s). It runs on the host CPU, as do the small subtrees of the 19.6-million-node tree. epyc prunes in 2.9 s.
  - **Growing alone,** without the pruning, the GPUs take 6.6 s (A100) and 8.3 s (H100) against epyc's 11.1 s.
  - **Hopper against ampere:** hopper's 32 cores lose to ampere's 48 in pruning (7.55 against 6.60 s) and in the small subtrees. Hopper's CPU timings also drift during the day: at 16:04 the same binary took 14.19 s there.
  - **What would help next:** C4.5's pruning is now the place to gain on the GPU nodes, through more parallelism or work on the GPU.
- **CPU nodes:** dell takes 14–19% longer than epyc (CART 8.36 against 7.34 s, C4.5 16.60 against 13.96 s).

Unpruned trees:
- **The depth cap hid most of CART's tree.** Without it CART grows 47.1 million nodes, 2.2× as many as up to depth 30 and 2.4× C4.5's. On the CPU nodes the extra 25 million nodes cost only 6–9% (epyc 8.39 → 8.89 s). They lie deep down, with a few rows each, and are cheap to split.
- **CART against C4.5 on the CPU nodes:** CART grows its 2.4× larger tree in 80–83% of C4.5's time (epyc 8.89 against 11.13 s). C4.5's time includes its collapse of useless splits and its thresholds (0.1–0.3 s).
- **The GPU's gain shrinks with tiny nodes.** The GPU hands every node below 28 thousand rows to the CPU threads, so almost all of the 47 million nodes are grown by the host. The A100 is 1.2× epyc on CART and 1.6× on C4.5, against 2.2× on pruned CART, where only the first levels are grown and every node has millions of rows.
- **Hopper is slow on unpruned CART:** 13.80 s, against 7.14 s on ampere and 4.66 s with the depth cap. All three runs agree (13.78–14.35 s). C4.5's tree is half the size and is grown as fast on hopper as on ampere (7.17 against 7.11 s). The extra time is in the build (13.4 against 6.5 s on ampere), where hopper's 32 CPU threads grow the small subtrees and ampere has 48. That alone does not explain a factor of 2, and it has not been investigated.
- **C4.5's pruning is the big cost of a pruned C4.5 run:** on epyc it adds 2.8 s (11.13 → 13.96 s); on the GPU nodes, whose CPUs are slower, 6.1–8.7 s, as much as growing the tree.

Both:
- **Memory:** the same on every node of a kind.
  - Parallel: pruned 29.0 GiB (CART) and 38.1 GiB (C4.5) peak RSS; unpruned 32.5 and 36.7 GiB.
  - CUDA: pruned 17.2 / 26.3–26.5 GiB RSS, unpruned 20.7 / 24.9–25.1 GiB, plus 23.9 GiB on the GPU, CUDA context included.
- **Wall time:** with only ./tree's own training accuracy, a whole CPU-node run (3 runs per case and the data check) took 1¼ minutes. The earlier runs predicted the training and test rows in Python and took about 1.5 minutes per C4.5 run alone.
