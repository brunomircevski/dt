# Profiling: where CART and C4.5 spend their growing time

**Status:** run on 2026-10-29/30 at ec77727 (main with the CPU presort of 4fb419c; the hash in `machine.json` and on the chart).
- **Machines:**
  - CUDA: hopper, on the GPU's socket: H100 PCIe, 16 CPU threads, NUMA-bound.
  - Parallel: epyc1, and dell1, 48 threads each (2× EPYC 9255, 2× Xeon Gold 6542Y).
- **Repetition:** every case ran 30 times profiled (`./tree --profile FILE`) on chess and 10 times on HIGGS, Airline and ECBDL. `plain/` holds the runs without `--profile`: for chess 50M, 500M and 2000M their own runs, for the others copies of the test-set benchmarks' runs. The chart and table use the average of the profiled runs (the time and every step); `without --profile s` is the average of the plain runs.
- **Chess only:** `chart-chess.png` is the same chart with the chess sets alone.
- **Earlier runs:** the versions of 2026-10-22 and the previous one (CART with α = 1e-5, C4.5 with CF 0.25 pruning, C4.5 unpruned, no depth limit) are in git; the second at 08c4235.

![chart](chart.png)

Columns: Parallel on epyc, Parallel on dell, CUDA hybrid (the default), CUDA `--gpu-only`. Rows: CART, C4.5.
- **Bars:** each bar is the average of the case's training runs. 100% is `train_seconds`, the total shown above the bar; reading the data is not included. The bar is split into five steps, and hatched segments ran on the GPU.
- **Under each bar:** the dataset, rows × features, the training accuracy, the node count and the depth. A fact that is the same for every bar of a panel would be shown once in the panel's subtitle; none is here, because chess stops at depth 4 and the other sets reach depth 10.
- **Divider:** the dashed line separates the six chess sets from HIGGS, Airline and ECBDL.

## What runs

```bash
bench/cluster/submit-main-2026-10-29.sh   # every case, with the other reruns of 2026-10-29/30
d=bench/results/cluster-profiling
bench/.venv/bin/python bench/profile_share_chart.py $d --out $d/chart.png
bench/.venv/bin/python bench/profile_share_chart.py $d --chess --out $d/chart-chess.png
```

**Datasets:**
- `chess_10m`, `chess_50m`, `chess_100m`, `chess_500m`, `chess_1b`, `chess_2b`: the 3×3 checkerboard, 2 features, 10M–2000M rows.
- `higgs`: 10.5M rows × 28.
- `airline`: 109.3M rows × 13.
- `ecbdl_15m`: the 15M-row ECBDL'14 sample, 631 features.

**Memory:** chess 2000M fits everywhere. On hopper it uses about 51 GB of the H100's 80 GB and about 50 GB of RAM. On epyc1 it uses about 94 GB of 512 GB.

**Protocols:**
- **CART** = `cart_depth10`: `--cart --no-prune -d 10` (Gini).
- **C4.5** = `c45_depth10_unpruned`: `--c45 --no-prune -d 10` (gain ratio). Pruning is off; only C4.5's tidy-up after growing runs, which collapses splits that do not reduce the training errors and moves cuts onto training values. The chart counts the tidy-up as other.
- Every mode grows the same trees.

## The five steps

| Step | What it measures | From which trace data |
|---|---|---|
| **Presort** | Sorting every feature column once, before growing. On the CPU this is a parallel sort of each column; on the GPU it is a gather, a radix sort and the distinct values. After it, every node reads its rows already in order. | GPU kernels `presort*`; CPU work `presort` |
| **Split search** | Finding the best threshold of every feature at every node: walking the sorted values, keeping the class counts on both sides of each cut, scoring it (Gini, or gain then gain ratio), and picking the winning feature. On the GPU: per-tile histograms, prefix sums and cut scans. | kernels `tile*`, `segment*`; CPU `sweep`, `build` |
| **Partition** | Applying the chosen split: marking the rows that go left, then moving every other column so that each child's rows are contiguous and still sorted. On the GPU this also gathers the columns of the small nodes handed to the CPU (hybrid). | kernels `markGoesLeft`, `partition`, `handOffGather`; CPU `mark`, `partition`, `partition_sweep`, `leaves` |
| **CPU↔GPU transfer** | Moving data between RAM and the GPU: the upload of the dataset, the hand-off columns, the per-level lists and cuts, and the host copies through page-locked buffers. | copy-lane spans; CPU `staging`, `page_lock` |
| **Other** | Everything else: GPU context and memory allocation, the host launching kernels or waiting while the GPU is idle, node bookkeeping, assembling the final tree, C4.5's tidy-up after growing, and untracked time. | the rest, including CPU `prune*` |

**How each millisecond is assigned** ([profile_share_chart.py](../../profile_share_chart.py)):
1. Running GPU kernels come first. Time when kernels of two steps overlap is shared between them.
2. Then a host↔GPU copy with no kernel running counts as transfer.
3. Then the training thread's phase decides. While it drives the GPU (gpu setup, upload, grow setup, gpu levels), the time is transfer if threads are staging or page-locking, and other otherwise.
4. Elsewhere, the time is shared among the busy CPU threads by their kind of work, taken from the 5 ms samples. If every thread is idle, it goes to the phase's step.

In hybrid mode, CPU subtree growth that hides behind the GPU is therefore not counted; it shows only when the GPU waits for it.

## Results

| Mode | Algorithm | Dataset | runs | train s | without --profile s | train acc % | nodes | depth | Presort % | Split search % | Partition % | CPU↔GPU transfer % | Other % | GPU share % |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Parallel (CPU only), epyc | CART | chess 10M | 30 | 0.09 | 0.09 | 100.00 | 17 | 4 | 28.3 | 15.2 | 46.9 | 0.0 | 9.6 | 0.0 |
| Parallel (CPU only), epyc | CART | chess 50M | 30 | 0.25 | 0.26 | 100.00 | 17 | 4 | 27.6 | 26.8 | 39.8 | 0.0 | 5.9 | 0.0 |
| Parallel (CPU only), epyc | CART | chess 100M | 30 | 0.44 | 0.45 | 100.00 | 17 | 4 | 25.6 | 30.6 | 37.3 | 0.0 | 6.5 | 0.0 |
| Parallel (CPU only), epyc | CART | chess 500M | 30 | 2.20 | 2.21 | 100.00 | 17 | 4 | 21.5 | 28.4 | 45.0 | 0.0 | 5.1 | 0.0 |
| Parallel (CPU only), epyc | CART | chess 1000M | 30 | 4.44 | 4.44 | 100.00 | 17 | 4 | 21.5 | 29.0 | 44.6 | 0.0 | 4.9 | 0.0 |
| Parallel (CPU only), epyc | CART | chess 2000M | 30 | 9.10 | 9.19 | 100.00 | 17 | 4 | 20.3 | 27.3 | 48.1 | 0.0 | 4.3 | 0.0 |
| Parallel (CPU only), epyc | CART | HIGGS | 10 | 0.61 | 0.61 | 70.63 | 2,029 | 10 | 19.8 | 14.3 | 63.9 | 0.0 | 2.0 | 0.0 |
| Parallel (CPU only), epyc | CART | Airline | 10 | 2.88 | 2.87 | 67.10 | 2,047 | 10 | 9.6 | 14.9 | 74.4 | 0.0 | 1.0 | 0.0 |
| Parallel (CPU only), epyc | CART | ECBDL'14 15M | 10 | 6.39 | 6.32 | 97.92 | 1,863 | 10 | 18.1 | 32.0 | 49.6 | 0.0 | 0.2 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | chess 10M | 30 | 0.10 | 0.10 | 100.00 | 17 | 4 | 28.0 | 24.3 | 39.1 | 0.0 | 8.7 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | chess 50M | 30 | 0.30 | 0.31 | 100.00 | 17 | 4 | 24.9 | 37.6 | 32.0 | 0.0 | 5.5 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | chess 100M | 30 | 0.53 | 0.54 | 100.00 | 17 | 4 | 23.0 | 41.8 | 30.8 | 0.0 | 4.5 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | chess 500M | 30 | 2.71 | 2.67 | 100.00 | 17 | 4 | 18.9 | 38.0 | 38.0 | 0.0 | 5.1 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | chess 1000M | 30 | 5.41 | 5.45 | 100.00 | 17 | 4 | 19.0 | 38.7 | 38.0 | 0.0 | 4.4 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | chess 2000M | 30 | 11.03 | 10.92 | 100.00 | 17 | 4 | 18.0 | 36.1 | 41.1 | 0.0 | 4.8 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | HIGGS | 10 | 0.60 | 0.59 | 63.76 | 935 | 10 | 23.5 | 22.1 | 52.4 | 0.0 | 1.9 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | Airline | 10 | 3.83 | 3.83 | 66.62 | 929 | 10 | 8.6 | 33.7 | 57.1 | 0.0 | 0.6 | 0.0 |
| Parallel (CPU only), epyc | C4.5 | ECBDL'14 15M | 10 | 7.86 | 7.81 | 97.92 | 725 | 10 | 19.4 | 36.9 | 43.5 | 0.0 | 0.2 | 0.0 |
| Parallel (CPU only), dell | CART | chess 10M | 30 | 0.10 | 0.14 | 100.00 | 17 | 4 | 27.6 | 33.7 | 27.5 | 0.0 | 11.2 | 0.0 |
| Parallel (CPU only), dell | CART | chess 50M | 30 | 0.39 | 0.39 | 100.00 | 17 | 4 | 25.6 | 43.8 | 23.0 | 0.0 | 7.6 | 0.0 |
| Parallel (CPU only), dell | CART | chess 100M | 30 | 0.77 | 0.79 | 100.00 | 17 | 4 | 24.9 | 44.5 | 24.1 | 0.0 | 6.5 | 0.0 |
| Parallel (CPU only), dell | CART | chess 500M | 30 | 3.81 | 3.84 | 100.00 | 17 | 4 | 22.5 | 45.6 | 27.2 | 0.0 | 4.7 | 0.0 |
| Parallel (CPU only), dell | CART | chess 1000M | 30 | 7.71 | 7.48 | 100.00 | 17 | 4 | 21.0 | 45.1 | 29.1 | 0.0 | 4.8 | 0.0 |
| Parallel (CPU only), dell | CART | chess 2000M | 30 | 15.87 | 15.74 | 100.00 | 17 | 4 | 20.5 | 41.4 | 34.0 | 0.0 | 4.2 | 0.0 |
| Parallel (CPU only), dell | CART | HIGGS | 10 | 0.79 | 0.82 | 70.63 | 2,029 | 10 | 26.5 | 17.1 | 54.5 | 0.0 | 1.9 | 0.0 |
| Parallel (CPU only), dell | CART | Airline | 10 | 4.13 | 4.21 | 67.10 | 2,047 | 10 | 15.7 | 17.3 | 65.7 | 0.0 | 1.3 | 0.0 |
| Parallel (CPU only), dell | CART | ECBDL'14 15M | 10 | 10.05 | 10.07 | 97.92 | 1,863 | 10 | 20.0 | 26.0 | 53.8 | 0.0 | 0.2 | 0.0 |
| Parallel (CPU only), dell | C4.5 | chess 10M | 30 | 0.12 | 0.17 | 100.00 | 17 | 4 | 26.9 | 41.3 | 23.3 | 0.0 | 8.5 | 0.0 |
| Parallel (CPU only), dell | C4.5 | chess 50M | 30 | 0.51 | 0.51 | 100.00 | 17 | 4 | 21.9 | 53.7 | 18.2 | 0.0 | 6.2 | 0.0 |
| Parallel (CPU only), dell | C4.5 | chess 100M | 30 | 1.01 | 1.03 | 100.00 | 17 | 4 | 20.5 | 56.1 | 18.7 | 0.0 | 4.6 | 0.0 |
| Parallel (CPU only), dell | C4.5 | chess 500M | 30 | 5.23 | 5.23 | 100.00 | 17 | 4 | 18.1 | 57.8 | 20.0 | 0.0 | 4.2 | 0.0 |
| Parallel (CPU only), dell | C4.5 | chess 1000M | 30 | 10.68 | 10.44 | 100.00 | 17 | 4 | 16.8 | 57.7 | 21.6 | 0.0 | 3.9 | 0.0 |
| Parallel (CPU only), dell | C4.5 | chess 2000M | 30 | 22.05 | 21.68 | 100.00 | 17 | 4 | 16.3 | 54.7 | 24.7 | 0.0 | 4.3 | 0.0 |
| Parallel (CPU only), dell | C4.5 | HIGGS | 10 | 0.95 | 0.99 | 63.76 | 935 | 10 | 26.3 | 23.3 | 48.8 | 0.0 | 1.6 | 0.0 |
| Parallel (CPU only), dell | C4.5 | Airline | 10 | 5.70 | 5.68 | 66.62 | 929 | 10 | 13.7 | 33.0 | 52.2 | 0.0 | 1.2 | 0.0 |
| Parallel (CPU only), dell | C4.5 | ECBDL'14 15M | 10 | 12.49 | 12.44 | 97.92 | 725 | 10 | 24.9 | 32.2 | 42.8 | 0.0 | 0.1 | 0.0 |
| CUDA hybrid (default) | CART | chess 10M | 30 | 0.07 | 0.07 | 100.00 | 17 | 4 | 1.5 | 2.2 | 1.6 | 14.9 | 79.8 | 5.4 |
| CUDA hybrid (default) | CART | chess 50M | 30 | 0.07 | 0.07 | 100.00 | 17 | 4 | 6.9 | 3.2 | 7.5 | 28.6 | 53.8 | 17.6 |
| CUDA hybrid (default) | CART | chess 100M | 30 | 0.09 | 0.09 | 100.00 | 17 | 4 | 11.3 | 4.3 | 12.7 | 33.1 | 38.6 | 28.3 |
| CUDA hybrid (default) | CART | chess 500M | 30 | 0.31 | 0.31 | 100.00 | 17 | 4 | 16.3 | 7.1 | 38.7 | 28.5 | 9.4 | 62.0 |
| CUDA hybrid (default) | CART | chess 1000M | 30 | 0.65 | 0.64 | 100.00 | 17 | 4 | 15.9 | 5.1 | 48.5 | 25.8 | 4.7 | 69.5 |
| CUDA hybrid (default) | CART | chess 2000M | 30 | 1.35 | 1.34 | 100.00 | 17 | 4 | 15.3 | 7.0 | 53.3 | 22.3 | 2.1 | 75.6 |
| CUDA hybrid (default) | CART | HIGGS | 10 | 0.16 | 0.15 | 70.63 | 2,029 | 10 | 12.0 | 16.6 | 31.9 | 12.8 | 26.7 | 60.4 |
| CUDA hybrid (default) | CART | Airline | 10 | 0.50 | 0.49 | 67.10 | 2,047 | 10 | 14.7 | 19.9 | 44.2 | 12.0 | 9.1 | 78.8 |
| CUDA hybrid (default) | CART | ECBDL'14 15M | 10 | 2.99 | 2.97 | 97.92 | 1,863 | 10 | 18.2 | 24.6 | 48.5 | 6.6 | 2.0 | 91.3 |
| CUDA hybrid (default) | C4.5 | chess 10M | 30 | 0.08 | 0.07 | 100.00 | 17 | 4 | 1.7 | 2.5 | 1.5 | 14.7 | 79.6 | 5.7 |
| CUDA hybrid (default) | C4.5 | chess 50M | 30 | 0.07 | 0.07 | 100.00 | 17 | 4 | 7.7 | 3.4 | 7.4 | 29.5 | 52.1 | 18.5 |
| CUDA hybrid (default) | C4.5 | chess 100M | 30 | 0.09 | 0.09 | 100.00 | 17 | 4 | 12.3 | 4.3 | 12.2 | 34.8 | 36.4 | 28.7 |
| CUDA hybrid (default) | C4.5 | chess 500M | 30 | 0.34 | 0.34 | 100.00 | 17 | 4 | 16.9 | 6.3 | 35.8 | 32.4 | 8.5 | 59.1 |
| CUDA hybrid (default) | C4.5 | chess 1000M | 30 | 0.70 | 0.70 | 100.00 | 17 | 4 | 16.5 | 4.7 | 44.6 | 29.8 | 4.4 | 65.8 |
| CUDA hybrid (default) | C4.5 | chess 2000M | 30 | 1.44 | 1.44 | 100.00 | 17 | 4 | 16.1 | 6.5 | 49.6 | 25.9 | 1.9 | 72.2 |
| CUDA hybrid (default) | C4.5 | HIGGS | 10 | 0.17 | 0.17 | 63.76 | 935 | 10 | 12.4 | 17.3 | 28.7 | 16.3 | 25.2 | 58.3 |
| CUDA hybrid (default) | C4.5 | Airline | 10 | 0.51 | 0.51 | 66.62 | 929 | 10 | 16.4 | 20.6 | 43.1 | 11.2 | 8.7 | 80.1 |
| CUDA hybrid (default) | C4.5 | ECBDL'14 15M | 10 | 3.01 | 2.99 | 97.92 | 725 | 10 | 20.8 | 24.2 | 49.1 | 3.8 | 2.0 | 94.1 |
| CUDA --gpu-only | CART | chess 10M | 30 | 0.07 | 0.07 | 100.00 | 17 | 4 | 1.6 | 2.2 | 1.6 | 10.5 | 84.2 | 5.4 |
| CUDA --gpu-only | CART | chess 50M | 30 | 0.07 | 0.07 | 100.00 | 17 | 4 | 6.8 | 3.2 | 7.4 | 24.5 | 58.0 | 17.5 |
| CUDA --gpu-only | CART | chess 100M | 30 | 0.09 | 0.09 | 100.00 | 17 | 4 | 11.5 | 4.4 | 13.0 | 29.5 | 41.6 | 28.9 |
| CUDA --gpu-only | CART | chess 500M | 30 | 0.31 | 0.31 | 100.00 | 17 | 4 | 16.3 | 7.0 | 38.7 | 27.3 | 10.7 | 62.0 |
| CUDA --gpu-only | CART | chess 1000M | 30 | 0.65 | 0.64 | 100.00 | 17 | 4 | 15.9 | 5.1 | 48.7 | 25.0 | 5.3 | 69.7 |
| CUDA --gpu-only | CART | chess 2000M | 30 | 1.34 | 1.33 | 100.00 | 17 | 4 | 15.4 | 7.0 | 53.4 | 21.7 | 2.5 | 75.8 |
| CUDA --gpu-only | CART | HIGGS | 10 | 0.16 | 0.15 | 70.63 | 2,029 | 10 | 12.0 | 16.4 | 31.6 | 13.2 | 26.9 | 59.9 |
| CUDA --gpu-only | CART | Airline | 10 | 0.49 | 0.49 | 67.10 | 2,047 | 10 | 14.8 | 19.9 | 44.3 | 11.8 | 9.2 | 79.0 |
| CUDA --gpu-only | CART | ECBDL'14 15M | 10 | 2.99 | 2.97 | 97.92 | 1,863 | 10 | 18.2 | 24.8 | 48.5 | 6.1 | 2.4 | 91.5 |
| CUDA --gpu-only | C4.5 | chess 10M | 30 | 0.07 | 0.07 | 100.00 | 17 | 4 | 1.8 | 2.4 | 1.6 | 11.2 | 83.0 | 5.8 |
| CUDA --gpu-only | C4.5 | chess 50M | 30 | 0.07 | 0.07 | 100.00 | 17 | 4 | 7.7 | 3.4 | 7.4 | 26.3 | 55.2 | 18.5 |
| CUDA --gpu-only | C4.5 | chess 100M | 30 | 0.09 | 0.09 | 100.00 | 17 | 4 | 12.3 | 4.3 | 12.3 | 31.6 | 39.6 | 28.9 |
| CUDA --gpu-only | C4.5 | chess 500M | 30 | 0.34 | 0.34 | 100.00 | 17 | 4 | 17.0 | 6.4 | 36.0 | 31.0 | 9.6 | 59.4 |
| CUDA --gpu-only | C4.5 | chess 1000M | 30 | 0.70 | 0.70 | 100.00 | 17 | 4 | 16.5 | 4.7 | 44.6 | 29.2 | 5.0 | 65.8 |
| CUDA --gpu-only | C4.5 | chess 2000M | 30 | 1.44 | 1.43 | 100.00 | 17 | 4 | 16.1 | 6.5 | 49.7 | 25.3 | 2.4 | 72.4 |
| CUDA --gpu-only | C4.5 | HIGGS | 10 | 0.17 | 0.16 | 63.76 | 935 | 10 | 12.9 | 18.2 | 30.0 | 9.8 | 29.2 | 61.0 |
| CUDA --gpu-only | C4.5 | Airline | 10 | 0.51 | 0.51 | 66.62 | 929 | 10 | 16.4 | 20.5 | 42.9 | 11.0 | 9.3 | 79.8 |
| CUDA --gpu-only | C4.5 | ECBDL'14 15M | 10 | 3.02 | 2.99 | 97.92 | 725 | 10 | 20.8 | 24.4 | 49.0 | 3.2 | 2.6 | 94.2 |

`runs` is the number of profiled runs; each row shows their average. `without --profile s` is the average of the plain runs (`plain/`). GPU share is the share of segments that ran on the GPU.

**Run-to-run noise and profiling overhead.**
- **Above 1 s:** profiled and plain averages agree within ±3%.
- **Small hopper runs (≤ 0.17 s):** they differ by up to 0.01 s.
- **dell chess 10M:** profiled 0.10–0.12 s, plain 0.14–0.17 s. The plain runs are the test-set batch's, a day apart; profiling does not make a run faster.

## Findings

- **CPU growing is partition-bound on epyc, split-search-bound on dell's chess.** On epyc, partition is 37–74% of CART and 31–57% of C4.5.
  - Split search takes more of C4.5 (22–42%) than of CART (14–32%), because of the gain ratio.
  - On dell's chess sets split search is the largest step: 34–46% of CART, 41–58% of C4.5.
  - Presort is 18–28% on epyc's chess sets and HIGGS, down from 30–36% at chess 500M–2000M before the presort speed-up (4fb419c).
  - Other stays under 12%, under 3% outside chess.
- **The GPU's work is mostly partition.** On hopper, chess 1000M/2000M and the non-chess sets spend 29–54% on partition and 3–30% on transfer. The GPU's share of the time grows with size: 5–6% at chess 10M, 59–62% at 500M, 72–76% at 2000M, and 91–94% on ECBDL.
- **Small runs are fixed cost.** CUDA chess 10M–100M take 0.07–0.09 s whatever the size: 36–84% of that is other (context, allocation, launches) and 10–35% is transfer.
- **Time scales linearly with chess size.**
  - CART on epyc: 2.20 s at 500M, 4.44 s at 1000M, 9.10 s at 2000M.
  - CART on hopper hybrid: 0.31, 0.65 and 1.35 s.
  - So the H100 is 6.7× faster for CART on chess 2000M, and 7.7× faster for C4.5.
  - It is also 3.5–3.8× faster on HIGGS and 5.8–7.5× faster on Airline, but only 2.1–2.6× faster on ECBDL.
- **dell is 1.3–2× epyc's time** at 48 threads (chess 2000M CART 15.9 s against 9.10 s, C4.5 22.0 s against 11.0 s; ECBDL 10.1 s against 6.39 s).
- **`--gpu-only` and hybrid take the same time** (within 0.01 s at every size): chess 2000M CART 1.34 and 1.35 s, Airline 0.49 and 0.50 s.
- **C4.5 costs about the same as CART on the GPU** (within 7%). On epyc, C4.5 is 21–33% slower on chess 500M–2000M, Airline and ECBDL, and about the same on HIGGS.
- **The trees differ by algorithm.**
  - On chess, both reach 100% at depth 4 (17 nodes), well before depth 10.
  - On the other sets, C4.5's trees are half the size or smaller: HIGGS 935 nodes against 2,029, ECBDL 725 against 1,863. Its training accuracy is lower on HIGGS (63.8% against 70.6%), about the same on Airline (66.6% against 67.1%), and equal on ECBDL (97.92%).
