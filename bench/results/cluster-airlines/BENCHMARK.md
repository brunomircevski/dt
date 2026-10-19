# Airline on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers, with and without pruning

**Status:** run on 2026-10-19, 3 runs per case, at commit 051949b: cd18717 (one copy of the presorted columns on every backend, the GPU's hand-off through a 512 MB ring, [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)) plus `./tree --profile` and the count tables filled without a zero-fill on one thread. The same trees as on cd18717. `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.

**CUDA rows rerun at 6f6cecb** (2026-10-19 from 20:20, 3 runs per case): the GPU's hand-offs go through page-locked buffers instead of a page-locked arena ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)), and the host threads partition big nodes in blocks (d9da007). The same trees; times −6% to +1%. The CPU rows are still 051949b's.
- **Pruned trees** (the run directories here): all four nodes from 15:11, each right after its HIGGS run ([cluster-higgs](../cluster-higgs/BENCHMARK.md)).
- **Unpruned trees** (`noprunning/`): right after the pruned ones, from 15:12.
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

The dataset is gbm-bench's airline: US flights 1987–2008, 114,569,017 training rows, 13 integer features, label "arrival delayed". It has 11× HIGGS's rows. It exists only on the cluster: the CUDA backend needs 13.3 GiB of GPU memory for it (one copy of the presorted columns, a spare column and a 512 MB ring), 19.3 GiB for pruned C4.5 when the GPU has room for the raw values (see Memory below).

| Run | Node | CPU | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|
| `cuda-hopper-airline` | hopper | 2× Xeon Gold 6444Y, 32 cores | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-airline` | ampere | 2× EPYC 7402, 48 cores | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-airline` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | – | parallel (48 threads) |
| `cpu-dell-airline` | dell1 | 2× Xeon Gold 6542Y, 48 cores | – | parallel (48 threads) |

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the 6 GB CSV).

## Pruned trees

`cart_alpha_nodepth` (CART, fixed α = 1e-5) and `c45` (C4.5, CF 0.25). A bar's lighter part is the pruning.

![airline on four mordor3 nodes: pruned trees (left), unpruned trees (right)](chart.png)

Training time, median of 3 runs (min–max):

| | CART | C4.5 | of which C4.5 pruning |
|---|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **1.13 s** (1.12–1.13) | 5.92 s (5.05–5.98) | 2.53 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 1.63 s (1.62–1.64) | **5.63 s** (5.59–5.66) | 2.09 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 7.05 s (7.05–7.17) | 11.91 s (11.84–11.95) | 1.32 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 7.51 s (7.41–7.57) | 13.80 s (13.70–13.86) | 2.37 s |

The trees:
- **CART:** 3,013 nodes, depth 25, training accuracy 71.54%. It grows 0.1 million nodes before pruning.
- **C4.5:** 8,548,725 nodes (19.6 million before pruning), depth 72, training accuracy 87.42%.
- **Accuracy:** `ActualElapsedTime` partly gives the delay away, so the accuracies say little. The dataset is here to compare speed, as in gbm-bench.

Against the earlier code (medians of 3): before the tuning for these servers (a8563b3, measured during the tuning, hopper and epyc only), after it (16cc93b, 2026-10-14 at 18:08), a83b6ec (the node store and the GPU hand-off, 2026-10-14 at 19:53), 35cf315 (C4.5 pruning from the recorded leaves, 2026-10-15 at 16:51), 8da0c12 (2026-10-15 at 23:23), cd18717 (2026-10-17 at 17:48), and now (CPU: 051949b; CUDA: 6f6cecb, with 051949b's times in brackets).

| | CART: before → tuned → a83b6ec → 35cf315 → 8da0c12 → cd18717 → now | C4.5: before → tuned → a83b6ec → 35cf315 → 8da0c12 → cd18717 → now |
|---|--:|--:|
| CUDA · hopper | 11.64 → 2.31 → 2.20 → 1.19 → 1.08 → 1.12 → **1.13 s** (1.12) | 26.03 → 15.83 → 11.26 → 7.46 → 6.07 → 5.68 → **5.92 s** (5.95) |
| CUDA · ampere | – → 3.37 → 2.84 → 1.99 → 1.59 → 1.64 → **1.63 s** (1.64) | – → 13.19 → 11.60 → 8.65 → 5.90 → 5.94 → **5.63 s** (5.93) |
| Parallel · epyc | 15.98 → 7.34 → 7.34 → 7.35 → 7.10 → 7.14 → **7.05 s** | 24.67 → 13.96 → 13.91 → 12.34 → 11.87 → 11.92 → **11.91 s** |
| Parallel · dell | – → 8.36 → 8.43 → 8.36 → 7.55 → 7.52 → **7.51 s** | – → 16.60 → 16.41 → 14.78 → 13.96 → 14.14 → **13.80 s** |

## Unpruned trees

The same trees grown without pruning, so CART and C4.5 do comparable work: each grows its full tree. The bars are not split, because nothing is pruned.
- **CART** (`cart_full_nodepth`, `./tree --cart --no-prune`): without pruning, CART's early stop with a fixed α (docs/ALGORITHMS.md) does not apply. The tree grows until every leaf is pure or cannot be split.
- **C4.5** (`c45_unpruned`, `./tree --c45 --no-prune`): C4.5's collapse of useless splits still runs. It is part of growing, done with or without pruning.

Training time, median of 3 runs (min–max):

| | CART, no pruning | C4.5, no pruning |
|---|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **2.88 s** (2.88–3.12) | **2.91 s** (2.91–2.92) |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 3.81 s (3.58–3.84) | 3.54 s (3.50–3.54) |
| Parallel · epyc (EPYC 9255, 48 threads) | 8.66 s (8.58–8.67) | 10.64 s (10.58–10.68) |
| Parallel · dell (Xeon 6542Y, 48 threads) | 9.30 s (9.27–9.40) | 11.37 s (11.27–11.39) |

The trees:
- **CART:** 47,146,877 nodes, depth 65, training accuracy 100.00%: every leaf is pure.
- **C4.5:** 19,579,127 nodes, depth 74, training accuracy 90.60%. C4.5 stops earlier, at leaves of 2 rows and at splits that gain nothing.

Against the earlier code: 16cc93b (2026-10-14 at 17:58), a83b6ec (19:55), 35cf315 (2026-10-15 at 16:53), 8da0c12 (2026-10-15 at 23:25), cd18717 (2026-10-17 at 17:49), now (CPU: 051949b; CUDA: 6f6cecb, 051949b in brackets), and CART capped at depth 30 (`cart_full`, `-d 30`, 21,600,857 nodes, run at 17:13 on 16cc93b):

| Median | CART: depth ≤ 30 (16cc93b) | CART: 16cc93b → a83b6ec → 35cf315 → 8da0c12 → cd18717 → now | C4.5: 16cc93b → a83b6ec → 35cf315 → 8da0c12 → cd18717 → now |
|---|--:|--:|--:|
| CUDA · hopper | 4.66 s | 13.80 → 4.06 → 3.37 → 3.24 → 3.13 → **2.88 s** (3.01) | 7.17 → 4.39 → 3.66 → 3.09 → 3.06 → **2.91 s** (2.99) |
| CUDA · ampere | 5.20 s | 7.14 → 4.67 → 4.22 → 3.77 → 3.82 → **3.81 s** (3.81) | 7.11 → 5.39 → 4.80 → 3.67 → 3.72 → **3.54 s** (3.75) |
| Parallel · epyc | 8.39 s | 8.89 → 8.76 → 8.74 → 8.49 → 8.63 → **8.66 s** | 11.13 → 11.08 → 11.10 → 10.58 → 10.71 → **10.64 s** |
| Parallel · dell | 9.64 s | 10.51 → 10.07 → 10.05 → 9.32 → 9.44 → **9.30 s** | 12.64 → 12.58 → 12.47 → 11.75 → 11.95 → **11.37 s** |

## Observations

- **CART, pruned:** the H100 takes 1.13 s, 6.2× the fastest CPU node (epyc, 7.05 s), and the A100 1.63 s (4.3×).
- **C4.5, pruned:** the GPUs beat epyc by 2.0–2.1× (H100 5.92 s, A100 5.63 s, epyc 11.91 s).
  - **Growing alone,** without the pruning, takes 3.4 s on the H100 and 3.5 s on the A100, against epyc's 10.6 s.
  - **Pruning is 37–43% of a CUDA run.** It runs on the host CPU, apart from its first step (115M rows sorted by leaf and transposed, on the GPU).
  - **Hopper's C4.5 varies the most** (5.05–5.98 s; 5.33–6.09 s at 051949b). Its CPU timings drift during the day (bench/cluster/README.md).
- **Unpruned trees:** the GPUs are 2.3–3.0× epyc on CART (47.1 million nodes) and 3.0–3.7× on C4.5 (19.6 million). Almost all of these nodes are small and grown by the CPU threads, so the GPU node's CPU matters as much as its GPU.
- **The depth cap hid most of CART's tree.** Without it CART grows 47.1 million nodes, 2.2× as many as up to depth 30 and 2.4× C4.5's. On the CPU nodes the extra 25 million nodes cost little (epyc 8.39 s capped on 16cc93b, 8.66 s uncapped now): they lie deep down, with a few rows each.
- **CART against C4.5 on the CPU nodes:** CART grows its 2.4× larger tree in 81–82% of C4.5's time (epyc 8.66 against 10.64 s).
- **CPU nodes:** dell takes 7–16% longer than epyc on the pruned trees and 7% on the unpruned ones.
- **CUDA, 6f6cecb against 051949b:** the hand-offs no longer page-lock the 12 GB arena (0.2 s of registration that held up the GPU loop, more on a busy node), and the host partition works in blocks. The same trees. Pruned C4.5 on the A100 5.93 → 5.63 s, unpruned trees on the H100 3.01 → 2.88 s (CART) and 2.99 → 2.91 s (C4.5), on the A100 C4.5 3.75 → 3.54 s; the rest within ±1%.
- **051949b against cd18717 (2026-10-17):** only `--profile` (off here) and the count tables' fill changed. The same trees; times within −5% to +5%. The count tables no longer zero-fill 2 × 115M doubles on one thread, which shows most on dell's C4.5 (pruned 14.14 → 13.80 s, unpruned 11.95 → 11.37 s); hopper's pruned C4.5 (5.68 → 5.95 s) is within its usual spread.
- **cd18717 against 8da0c12:** the presorted columns are kept once instead of twice, and the GPU's mirror of the hand-offs became a 512 MB ring ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)).
  - **GPU memory fell by 53–62%** (below), with the same trees.
  - **GPU times moved by −6% to +4%:** hopper CART 1.08 → 1.12 s and C4.5 6.07 → 5.68 s (its C4.5 varies most); ampere 1.59 → 1.64 s and 5.90 → 5.94 s; unpruned −3% to +1%. A same-job comparison on ampere, the old and the new binary interleaved, 5 runs each, gave CART 1.60 → 1.63 s (+2%) and C4.5 5.91 → 5.90 s. The 2% on CART is probably the ring: the GPU waits for the host copies of earlier hand-offs before it reuses a part, and the A100's link is the slower one (not profiled).
  - **The CPU nodes** are within 2%.
- **Memory:** the same on every node of a kind.
  - **Parallel:** pruned 27.3–28.1 GiB (CART) and 32.2–32.3 GiB (C4.5) peak RSS, unpruned 30.8–31.6 and 29.7–30.4 GiB (29.0, 33.0–33.1, 32.5 and 31.3–31.4 GiB before). With 13 features and 48 threads the parallel backend keeps 12 spare column slots (min(features − 1, threads)), one column fewer than a second copy: about 0.9 GiB saved.
  - **CUDA:** pruned 6.4 / 21.0 GiB RSS, unpruned 21.1 / 19.7 GiB (8.2 / 20.7–20.8 and 20.9 / 19.5 GiB at 051949b: pruned CART no longer touches the whole hand-off arena, and two 128 MB page-locked buffers were added).
  - **GPU:** 13.3 GiB, and 19.3 GiB for pruned C4.5, CUDA context included (34.9–35.0 and 40.9 GiB before). That is one copy of the presorted columns (8 bytes per value) with one spare column and the 512 MB ring; pruned C4.5 adds a copy of the raw values (4 bytes per value) for its pruning's first step, kept only when it fits.
