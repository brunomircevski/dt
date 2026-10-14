# Airline without pruning on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers

**Status:** run on 2026-10-14 from 17:57 to 18:03, 3 runs per case, at commit 7cb9fb9. The ./tree source is that of 16cc93b; later commits changed only `bench/`.
- **Nodes:** each node ran alone and was used whole.
- **Backends:** the GPU nodes ran CUDA only, the CPU nodes parallel only.
- **Same trees:** every case built the same tree on every node and backend.

These are the trees of [cluster-airlines](../cluster-airlines/BENCHMARK.md), grown without pruning and **without a depth limit**, so CART and C4.5 do comparable work: each grows its full tree.
- **CART** (`cart_full_nodepth`): `./tree --cart --no-prune`. Without pruning, CART's early stop with a fixed α (docs/ALGORITHMS.md) does not apply. The tree grows until every leaf is pure or cannot be split.
- **C4.5** (`c45_unpruned`): `./tree --c45 --no-prune`. C4.5's collapse of useless splits still runs: it is part of growing, done with or without pruning.
- **Accuracy is ./tree's training accuracy** (`run.py --train-accuracy-only`).

How the runs are made: [bench/cluster/README.md](../../cluster/README.md):

```bash
bench/cluster/submit.sh -m 3 -d airline -p cart_full_nodepth,c45_unpruned -o cluster-airlines-noprunning
```

| Run | Node | CPU | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|
| `cuda-hopper-airline` | hopper | 2× Xeon Gold 6444Y, 32 cores | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-airline` | ampere | 2× EPYC 7402, 48 cores | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-airline` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | – | parallel (48 threads) |
| `cpu-dell-airline` | dell1 | 2× Xeon Gold 6542Y, 48 cores | – | parallel (48 threads) |

## Results

![airline without pruning on four mordor3 nodes](chart.png)

Each bar is ./tree's training time (`train total`: GPU setup, presort and build, not reading the 6 GB CSV). The bars are not split: nothing is pruned.

Training time, median of 3 runs (min–max):

| | CART, no pruning | C4.5, no pruning |
|---|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | 13.80 s (13.78–14.35) | 7.17 s (6.62–7.18) |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | **7.14 s** (7.08–7.18) | **7.11 s** (7.05–7.14) |
| Parallel · epyc (EPYC 9255, 48 threads) | 8.89 s (8.88–8.93) | 11.13 s (11.11–11.16) |
| Parallel · dell (Xeon 6542Y, 48 threads) | 10.51 s (10.47–10.59) | 12.64 s (12.59–13.00) |

The trees:
- **CART:** 47,146,877 nodes, depth 65, training accuracy 100.00%: every leaf is pure.
- **C4.5:** 19,579,127 nodes, depth 74, training accuracy 90.60%. C4.5 stops earlier, at leaves of 2 rows and splits that gain nothing.

### Against the earlier run with CART capped at depth 30

At 17:13 the same run was made with CART capped at depth 30 (`cart_full`, `-d 30`), the cap that the benchmarks against rpart need. The cap cut CART's tree to 21,600,857 nodes; C4.5 was the same.

| Median | CART depth ≤ 30 → no limit |
|---|--:|
| CUDA · hopper | 4.66 s → 13.80 s |
| CUDA · ampere | 5.20 s → 7.14 s |
| Parallel · epyc | 8.39 s → 8.89 s |
| Parallel · dell | 9.64 s → 10.51 s |

### Against the pruned runs

From [cluster-airlines](../cluster-airlines/BENCHMARK.md); the pruned trees are 3,013 (CART) and 8,548,725 (C4.5) nodes.

| Median | CART pruned → unpruned | C4.5 pruned → unpruned |
|---|--:|--:|
| CUDA · hopper | 2.31 s → 13.80 s | 15.83 s → 7.17 s |
| CUDA · ampere | 3.37 s → 7.14 s | 13.19 s → 7.11 s |
| Parallel · epyc | 7.34 s → 8.89 s | 13.96 s → 11.13 s |
| Parallel · dell | 8.36 s → 10.51 s | 16.60 s → 12.64 s |

## Observations

- **The depth cap hid most of CART's tree.** Without it CART grows 47.1 million nodes, 2.2× as many as up to depth 30 and 2.4× C4.5's. On the CPU nodes the extra 25 million nodes cost only 6–9% (epyc 8.39 → 8.89 s). They lie deep down, with a few rows each, and are cheap to split.
- **CART against C4.5 on the CPU nodes:** CART grows its 2.4× larger tree in 80–83% of C4.5's time (epyc 8.89 against 11.13 s). C4.5's time includes its collapse of useless splits and its thresholds (0.1–0.3 s).
- **The GPU's gain shrinks with tiny nodes.** The GPU hands every node below 28 thousand rows to the CPU threads, so almost all of the 47 million nodes are grown by the host. The A100 is 1.2× epyc on CART and 1.6× on C4.5, against 2.2× on pruned CART, where only the first levels are grown and every node has millions of rows.
- **Hopper is slow on unpruned CART:** 13.80 s, against 7.14 s on ampere and 4.66 s with the depth cap. All three runs agree (13.78–14.35 s). C4.5's tree is half the size and is grown as fast on hopper as on ampere (7.17 against 7.11 s). The extra time is in the build (13.4 against 6.5 s on ampere), where hopper's 32 CPU threads grow the small subtrees and ampere has 48. That alone does not explain a factor of 2, and it has not been investigated.
- **C4.5's pruning is the big cost of a pruned C4.5 run:** on epyc it adds 2.8 s (11.13 → 13.96 s); on the GPU nodes, whose CPUs are slower, 6.1–8.7 s, as much as growing the tree.
- **Memory:** parallel 32.5 GiB (CART) and 36.7 GiB (C4.5) peak RSS; CUDA 20.7 / 24.9–25.1 GiB RSS plus 23.9 GiB on the GPU, CUDA context included.
