# Profiling: where CART and C4.5 spend their growing time

**Status:** rerun on 2026-11-15/16 at f53347a9 (`src/` = d2957ff2: presort of big columns in cache, GPU tiles without a cut skipped, CART bound per node; the hash in `machine.json` and on the chart), by `bench/cluster/final.sh -m 5 -k profiling`. The runs before (ec77727, 2026-10-29/30, 30 or 10 runs) are in git at 1e655067.
- **Machines:**
  - CUDA: hopper: H100 PCIe (GPU 2) and every CPU of the node (32 threads, the GPU's socket first, memory first-touch), as in the test-set benchmarks.
  - Multicore: epyc1 and dell1, 48 threads each (2× EPYC 9255, 2× Xeon Gold 6542Y).
- **Repetition:** every case ran 5 times profiled (`./tree --profile FILE`) and 5 times without (`plain/`), in the same job. The chart and table use the average of the profiled runs (the time and every step); `without --profile s` is the average of the plain runs. The chart's footnote gives the runs.
- **Chess only:** `chart-chess.png` is the same chart with the chess sets alone.
- **Earlier runs:** the versions of 2026-10-22 and the previous one (CART with α = 1e-5, C4.5 with CF 0.25 pruning, C4.5 unpruned, no depth limit) are in git; the second at 08c4235.

![chart](chart.png)

Columns: `--multicore` on epyc, `--multicore` on dell, CUDA `--hybrid`, CUDA `--gpu`. Rows: CART, C4.5.
- **Bars:** each bar is the average of the case's training runs. 100% is `train_seconds`, the total shown above the bar; reading the data is not included. The bar is split into five steps, and hatched segments ran on the GPU.
- **Under each bar:** the dataset, rows × features, the training accuracy, the node count and the depth. A fact that is the same for every bar of a panel would be shown once in the panel's subtitle; none is here, because chess stops at depth 4 and the other sets reach depth 10.
- **Divider:** the dashed line separates the six chess sets from HIGGS, Airline and ECBDL.

## What runs

```bash
bench/cluster/final.sh -m 5 -k profiling     # cluster-profiling/<run> (--profile) and cluster-profiling/plain/<run>, on epyc1, dell1 and hopper
d=bench/results/cluster-profiling
bench/.venv/bin/python bench/profile_share_chart.py $d --out $d/chart.png
bench/.venv/bin/python bench/profile_share_chart.py $d --chess --out $d/chart-chess.png
```

**Datasets:**
- `chess_10m`, `chess_50m`, `chess_100m`, `chess_500m`, `chess_1b`, `chess_2b`: the 3×3 checkerboard, 2 features, 10M–2000M rows (all but `chess_1b` the first rows of `chess_2b`, read with `./tree --rows`).
- `higgs`: 10.5M rows × 28.
- `airline`: 109.3M rows × 13.
- `ecbdl_15m`: the 15M-row ECBDL'14 sample, 631 features.

**Memory:** chess 2000M fits everywhere. On hopper it uses about 51 GB of the H100's 80 GB and about 50 GB of RAM. On epyc1 it uses about 94 GB of 512 GB.

**Protocols:**
- **CART** = `cart_depth10`: `--cart -d 10` (Gini).
- **C4.5** = `c45_depth10_unpruned`: `--c45 -d 10` (gain ratio). Pruning is off; only C4.5's tidy-up after growing runs, which collapses splits that do not reduce the training errors and moves cuts onto training values. The chart counts the tidy-up as other.
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
| Multicore (CPU only), epyc | CART | chess 10M | 5 | 0.05 | 0.05 | 100.00 | 17 | 4 | 35.8 | 7.7 | 39.9 | 0.0 | 16.6 | 0.0 |
| Multicore (CPU only), epyc | CART | chess 50M | 5 | 0.15 | 0.14 | 100.00 | 17 | 4 | 35.2 | 16.4 | 41.7 | 0.0 | 6.7 | 0.0 |
| Multicore (CPU only), epyc | CART | chess 100M | 5 | 0.18 | 0.21 | 100.00 | 17 | 4 | 39.4 | 13.7 | 39.4 | 0.0 | 7.5 | 0.0 |
| Multicore (CPU only), epyc | CART | chess 500M | 5 | 0.80 | 0.78 | 100.00 | 17 | 4 | 33.5 | 13.2 | 49.1 | 0.0 | 4.2 | 0.0 |
| Multicore (CPU only), epyc | CART | chess 1000M | 5 | 1.54 | 1.61 | 100.00 | 17 | 4 | 34.4 | 11.3 | 50.6 | 0.0 | 3.7 | 0.0 |
| Multicore (CPU only), epyc | CART | chess 2000M | 5 | 3.25 | 3.23 | 100.00 | 17 | 4 | 33.6 | 10.9 | 51.9 | 0.0 | 3.6 | 0.0 |
| Multicore (CPU only), epyc | CART | HIGGS | 5 | 0.31 | 0.33 | 70.63 | 2,029 | 10 | 26.3 | 25.4 | 45.6 | 0.0 | 2.8 | 0.0 |
| Multicore (CPU only), epyc | CART | Airline | 5 | 1.64 | 1.59 | 67.10 | 2,047 | 10 | 11.7 | 12.4 | 75.1 | 0.0 | 0.9 | 0.0 |
| Multicore (CPU only), epyc | CART | ECBDL'14 15M | 5 | 4.46 | 4.44 | 97.92 | 1,863 | 10 | 19.4 | 11.6 | 68.8 | 0.0 | 0.2 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | chess 10M | 5 | 0.06 | 0.06 | 100.00 | 17 | 4 | 34.2 | 16.1 | 39.3 | 0.0 | 10.4 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | chess 50M | 5 | 0.13 | 0.16 | 100.00 | 17 | 4 | 40.8 | 10.5 | 36.4 | 0.0 | 12.2 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | chess 100M | 5 | 0.23 | 0.24 | 100.00 | 17 | 4 | 40.0 | 12.8 | 32.6 | 0.0 | 14.7 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | chess 500M | 5 | 1.01 | 0.98 | 100.00 | 17 | 4 | 29.8 | 11.4 | 40.7 | 0.0 | 18.1 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | chess 1000M | 5 | 2.05 | 1.77 | 100.00 | 17 | 4 | 29.6 | 10.0 | 39.5 | 0.0 | 20.9 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | chess 2000M | 5 | 3.88 | 3.86 | 100.00 | 17 | 4 | 31.5 | 12.5 | 43.4 | 0.0 | 12.7 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | HIGGS | 5 | 0.36 | 0.36 | 63.76 | 935 | 10 | 31.2 | 24.0 | 41.7 | 0.0 | 3.1 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | Airline | 5 | 1.70 | 1.69 | 66.62 | 929 | 10 | 14.0 | 14.0 | 69.7 | 0.0 | 2.4 | 0.0 |
| Multicore (CPU only), epyc | C4.5 | ECBDL'14 15M | 5 | 4.69 | 4.72 | 97.92 | 725 | 10 | 26.3 | 10.9 | 62.6 | 0.0 | 0.2 | 0.0 |
| Multicore (CPU only), dell | CART | chess 10M | 5 | 0.11 | 0.10 | 100.00 | 17 | 4 | 55.2 | 14.6 | 21.9 | 0.0 | 8.3 | 0.0 |
| Multicore (CPU only), dell | CART | chess 50M | 5 | 0.25 | 0.24 | 100.00 | 17 | 4 | 52.0 | 17.6 | 23.5 | 0.0 | 6.9 | 0.0 |
| Multicore (CPU only), dell | CART | chess 100M | 5 | 0.36 | 0.32 | 100.00 | 17 | 4 | 51.3 | 14.6 | 29.0 | 0.0 | 5.1 | 0.0 |
| Multicore (CPU only), dell | CART | chess 500M | 5 | 1.31 | 1.30 | 100.00 | 17 | 4 | 39.7 | 14.1 | 41.5 | 0.0 | 4.6 | 0.0 |
| Multicore (CPU only), dell | CART | chess 1000M | 5 | 2.71 | 2.71 | 100.00 | 17 | 4 | 36.4 | 13.4 | 46.0 | 0.0 | 4.1 | 0.0 |
| Multicore (CPU only), dell | CART | chess 2000M | 5 | 5.91 | 6.27 | 100.00 | 17 | 4 | 34.1 | 13.5 | 48.4 | 0.0 | 4.0 | 0.0 |
| Multicore (CPU only), dell | CART | HIGGS | 5 | 0.60 | 0.57 | 70.63 | 2,029 | 10 | 33.4 | 21.5 | 42.8 | 0.0 | 2.3 | 0.0 |
| Multicore (CPU only), dell | CART | Airline | 5 | 3.02 | 2.99 | 67.10 | 2,047 | 10 | 12.7 | 14.6 | 71.9 | 0.0 | 0.7 | 0.0 |
| Multicore (CPU only), dell | CART | ECBDL'14 15M | 5 | 7.69 | 7.75 | 97.92 | 1,863 | 10 | 20.7 | 13.8 | 65.4 | 0.0 | 0.2 | 0.0 |
| Multicore (CPU only), dell | C4.5 | chess 10M | 5 | 0.15 | 0.14 | 100.00 | 17 | 4 | 50.1 | 14.9 | 22.1 | 0.0 | 12.9 | 0.0 |
| Multicore (CPU only), dell | C4.5 | chess 50M | 5 | 0.28 | 0.24 | 100.00 | 17 | 4 | 53.3 | 17.3 | 18.8 | 0.0 | 10.6 | 0.0 |
| Multicore (CPU only), dell | C4.5 | chess 100M | 5 | 0.40 | 0.39 | 100.00 | 17 | 4 | 48.1 | 16.0 | 21.4 | 0.0 | 14.5 | 0.0 |
| Multicore (CPU only), dell | C4.5 | chess 500M | 5 | 1.58 | 1.50 | 100.00 | 17 | 4 | 37.4 | 14.2 | 35.3 | 0.0 | 13.1 | 0.0 |
| Multicore (CPU only), dell | C4.5 | chess 1000M | 5 | 3.14 | 3.17 | 100.00 | 17 | 4 | 36.5 | 14.3 | 39.8 | 0.0 | 9.5 | 0.0 |
| Multicore (CPU only), dell | C4.5 | chess 2000M | 5 | 7.71 | 7.12 | 100.00 | 17 | 4 | 30.3 | 14.3 | 42.3 | 0.0 | 13.1 | 0.0 |
| Multicore (CPU only), dell | C4.5 | HIGGS | 5 | 0.70 | 0.67 | 63.76 | 935 | 10 | 33.8 | 19.0 | 44.8 | 0.0 | 2.3 | 0.0 |
| Multicore (CPU only), dell | C4.5 | Airline | 5 | 3.28 | 3.29 | 66.62 | 929 | 10 | 15.7 | 14.4 | 67.9 | 0.0 | 1.9 | 0.0 |
| Multicore (CPU only), dell | C4.5 | ECBDL'14 15M | 5 | 8.85 | 8.78 | 97.92 | 725 | 10 | 30.9 | 12.4 | 56.5 | 0.0 | 0.2 | 0.0 |
| Hybrid (--hybrid) | CART | chess 10M | 5 | 0.05 | 0.04 | 100.00 | 17 | 4 | 2.8 | 1.3 | 2.5 | 20.5 | 72.9 | 6.6 |
| Hybrid (--hybrid) | CART | chess 50M | 5 | 0.06 | 0.05 | 100.00 | 17 | 4 | 8.5 | 3.0 | 9.1 | 30.9 | 48.4 | 20.6 |
| Hybrid (--hybrid) | CART | chess 100M | 5 | 0.07 | 0.07 | 100.00 | 17 | 4 | 14.3 | 4.7 | 16.1 | 35.2 | 29.7 | 35.1 |
| Hybrid (--hybrid) | CART | chess 500M | 5 | 0.28 | 0.28 | 100.00 | 17 | 4 | 18.2 | 5.5 | 43.0 | 28.6 | 4.7 | 66.7 |
| Hybrid (--hybrid) | CART | chess 1000M | 5 | 0.62 | 0.63 | 100.00 | 17 | 4 | 16.5 | 5.0 | 50.3 | 25.9 | 2.3 | 71.8 |
| Hybrid (--hybrid) | CART | chess 2000M | 5 | 1.31 | 1.31 | 100.00 | 17 | 4 | 15.8 | 4.7 | 54.5 | 23.4 | 1.6 | 75.1 |
| Hybrid (--hybrid) | CART | HIGGS | 5 | 0.13 | 0.13 | 70.63 | 2,029 | 10 | 13.6 | 7.1 | 36.3 | 19.3 | 23.7 | 56.7 |
| Hybrid (--hybrid) | CART | Airline | 5 | 0.43 | 0.42 | 67.10 | 2,047 | 10 | 17.1 | 7.3 | 51.2 | 16.9 | 7.4 | 75.7 |
| Hybrid (--hybrid) | CART | ECBDL'14 15M | 5 | 2.59 | 2.42 | 97.92 | 1,863 | 10 | 21.4 | 8.0 | 56.0 | 12.2 | 2.4 | 85.5 |
| Hybrid (--hybrid) | C4.5 | chess 10M | 5 | 0.05 | 0.04 | 100.00 | 17 | 4 | 2.9 | 2.0 | 2.3 | 20.7 | 72.0 | 7.3 |
| Hybrid (--hybrid) | C4.5 | chess 50M | 5 | 0.06 | 0.06 | 100.00 | 17 | 4 | 10.0 | 4.3 | 9.4 | 32.6 | 43.6 | 23.8 |
| Hybrid (--hybrid) | C4.5 | chess 100M | 5 | 0.08 | 0.07 | 100.00 | 17 | 4 | 14.5 | 4.9 | 14.3 | 35.2 | 31.0 | 33.7 |
| Hybrid (--hybrid) | C4.5 | chess 500M | 5 | 0.29 | 0.28 | 100.00 | 17 | 4 | 20.1 | 5.9 | 41.9 | 27.6 | 4.6 | 67.9 |
| Hybrid (--hybrid) | C4.5 | chess 1000M | 5 | 0.63 | 0.63 | 100.00 | 17 | 4 | 18.6 | 5.0 | 49.6 | 24.1 | 2.8 | 73.1 |
| Hybrid (--hybrid) | C4.5 | chess 2000M | 5 | 1.32 | 1.30 | 100.00 | 17 | 4 | 17.8 | 4.9 | 54.1 | 21.8 | 1.4 | 76.8 |
| Hybrid (--hybrid) | C4.5 | HIGGS | 5 | 0.15 | 0.15 | 63.76 | 935 | 10 | 13.9 | 16.9 | 32.9 | 18.7 | 17.6 | 62.9 |
| Hybrid (--hybrid) | C4.5 | Airline | 5 | 0.45 | 0.46 | 66.62 | 929 | 10 | 18.6 | 9.2 | 48.7 | 15.4 | 8.0 | 76.6 |
| Hybrid (--hybrid) | C4.5 | ECBDL'14 15M | 5 | 2.64 | 2.43 | 97.92 | 725 | 10 | 24.6 | 7.6 | 55.9 | 9.6 | 2.2 | 88.2 |
| GPU (--gpu) | CART | chess 10M | 5 | 0.05 | 0.04 | 100.00 | 17 | 4 | 2.9 | 1.2 | 2.4 | 16.8 | 76.8 | 6.4 |
| GPU (--gpu) | CART | chess 50M | 5 | 0.06 | 0.05 | 100.00 | 17 | 4 | 8.2 | 2.9 | 8.8 | 29.0 | 51.0 | 19.9 |
| GPU (--gpu) | CART | chess 100M | 5 | 0.07 | 0.07 | 100.00 | 17 | 4 | 13.3 | 4.4 | 15.1 | 33.1 | 34.1 | 32.9 |
| GPU (--gpu) | CART | chess 500M | 5 | 0.28 | 0.27 | 100.00 | 17 | 4 | 18.5 | 5.6 | 43.8 | 27.0 | 5.1 | 67.9 |
| GPU (--gpu) | CART | chess 1000M | 5 | 0.63 | 0.61 | 100.00 | 17 | 4 | 16.3 | 4.9 | 49.7 | 26.5 | 2.5 | 71.0 |
| GPU (--gpu) | CART | chess 2000M | 5 | 1.29 | 1.26 | 100.00 | 17 | 4 | 16.1 | 4.8 | 55.6 | 21.9 | 1.6 | 76.5 |
| GPU (--gpu) | CART | HIGGS | 5 | 0.13 | 0.13 | 70.63 | 2,029 | 10 | 13.9 | 7.1 | 37.3 | 19.6 | 22.2 | 58.3 |
| GPU (--gpu) | CART | Airline | 5 | 0.43 | 0.45 | 67.10 | 2,047 | 10 | 17.1 | 7.3 | 51.4 | 16.4 | 7.7 | 75.9 |
| GPU (--gpu) | CART | ECBDL'14 15M | 5 | 2.52 | 2.48 | 97.92 | 1,863 | 10 | 21.8 | 8.4 | 57.6 | 9.4 | 2.8 | 87.8 |
| GPU (--gpu) | C4.5 | chess 10M | 5 | 0.05 | 0.05 | 100.00 | 17 | 4 | 3.0 | 2.1 | 2.4 | 16.2 | 76.2 | 7.5 |
| GPU (--gpu) | C4.5 | chess 50M | 5 | 0.06 | 0.05 | 100.00 | 17 | 4 | 9.4 | 4.1 | 8.8 | 28.6 | 49.1 | 22.3 |
| GPU (--gpu) | C4.5 | chess 100M | 5 | 0.08 | 0.07 | 100.00 | 17 | 4 | 14.8 | 5.0 | 14.7 | 32.9 | 32.5 | 34.5 |
| GPU (--gpu) | C4.5 | chess 500M | 5 | 0.29 | 0.28 | 100.00 | 17 | 4 | 20.3 | 5.9 | 42.3 | 26.5 | 4.9 | 68.5 |
| GPU (--gpu) | C4.5 | chess 1000M | 5 | 0.64 | 0.63 | 100.00 | 17 | 4 | 18.2 | 4.9 | 48.8 | 25.5 | 2.6 | 71.9 |
| GPU (--gpu) | C4.5 | chess 2000M | 5 | 1.33 | 1.28 | 100.00 | 17 | 4 | 17.7 | 4.8 | 53.8 | 21.9 | 1.8 | 76.3 |
| GPU (--gpu) | C4.5 | HIGGS | 5 | 0.15 | 0.15 | 63.76 | 935 | 10 | 13.6 | 15.9 | 32.1 | 18.2 | 20.3 | 61.6 |
| GPU (--gpu) | C4.5 | Airline | 5 | 0.44 | 0.45 | 66.62 | 929 | 10 | 19.0 | 9.4 | 49.9 | 13.8 | 7.9 | 78.3 |
| GPU (--gpu) | C4.5 | ECBDL'14 15M | 5 | 2.65 | 2.44 | 97.92 | 725 | 10 | 24.6 | 7.7 | 55.8 | 9.0 | 2.9 | 88.1 |

`runs` is the number of profiled runs; each row shows their average. `without --profile s` is the average of the plain runs (`plain/`). GPU share is the share of segments that ran on the GPU.

**Run-to-run noise and profiling overhead.**
- **The CPU runs above 0.5 s:** profiled and plain averages agree within ±6%, except C4.5 on chess 1000M and 2000M (epyc 2.05 against 1.77 s, dell 7.71 against 7.12 s): there C4.5's build has two speeds from run to run, in profiled and plain runs alike ([below](#c45-on-big-chess-two-speeds)).
- **hopper:** within ±3% up to Airline; on ECBDL'14 15M the profiled runs take 7–9% longer (2.52–2.65 against 2.42–2.48 s) (the trace costs more there).
- **Small runs (≤ 0.4 s):** they differ by up to 0.04 s.

## Findings

- **CPU growing is partition-bound on the big sets, presort-bound on dell's small chess sets.** On epyc, partition is 39–75% of CART and 33–70% of C4.5 (the most on Airline and ECBDL).
  - Split search takes 8–25% on epyc, the most on HIGGS (24–25%); 13–22% on dell.
  - Presort is 30–41% on epyc's chess sets, 26–31% on HIGGS; on dell's chess sets up to 100M it is the largest step (48–55%), from 500M on partition (35–48%).
  - Other stays under 21%, under 3.1% outside chess; the most is C4.5 on chess 500M–1000M on epyc (18–21%), the two-speed build.
- **The GPU's work is mostly partition.** On hopper, chess 500M–2000M and the non-chess sets spend 32–58% on partition and 9–29% on transfer. The GPU's share of the time grows with size: 6–8% at chess 10M, 66–68% at 500M, 75–77% at 2000M, and 85–88% on ECBDL.
- **Small runs are fixed cost.** CUDA chess 10M–100M take 0.05–0.08 s: 30–77% of that is other (context, allocation, launches) and 16–35% is transfer.
- **Time scales linearly with chess size.**
  - CART on epyc: 0.80 s at 500M, 1.54 s at 1000M, 3.25 s at 2000M (the runs before: 2.20, 4.44, 9.10 s).
  - CART on hopper hybrid: 0.28, 0.62 and 1.31 s (before: 0.31, 0.65, 1.35 s).
  - So the H100 is 2.5× faster for CART on chess 2000M, and 2.9× for C4.5 (6.7× and 7.7× before: the CPU gained most).
  - It is also 2.4× faster on HIGGS, 3.8× on Airline and 1.7–1.8× on ECBDL.
- **dell is 1.6–2.5× epyc's time** at 48 threads (chess 2000M CART 5.91 s against 3.25 s, C4.5 7.71 s against 3.88 s; ECBDL 7.69 s against 4.46 s); the most at chess 10M–100M, where dell's presort is slow.
- **`--gpu` and hybrid take the same time** (within 3%): chess 2000M CART 1.29 and 1.31 s, Airline 0.43 and 0.43 s, ECBDL 2.52 and 2.59 s.
- **C4.5 costs about the same as CART on the GPU** (within 2%, HIGGS 15%). On epyc, C4.5 is 19–33% slower on chess 500M–2000M, 16% on HIGGS, 4–5% on Airline and ECBDL.
- **The trees differ by algorithm.**
  - On chess, both reach 100% at depth 4 (17 nodes), well before depth 10.
  - On the other sets, C4.5's trees are half the size or smaller: HIGGS 935 nodes against 2,029, Airline 929 against 2,047, ECBDL 725 against 1,863. Its training accuracy is lower on HIGGS (63.8% against 70.6%), about the same on Airline (66.6% against 67.1%), and equal on ECBDL (97.92%).

## C4.5 on big chess: two speeds

C4.5 on chess 500M–2000M takes one of two times from run to run on both CPU nodes, while CART on the same data spreads at most 6%. On epyc chess 1000M the build takes 1.13–1.15 s or 1.40–1.47 s, on chess 2000M 2.42 s or 3.0 s; the spread of a case is 20–40% (min to max). Every run grows the same tree with the same sweep and partition work; the extra time is in "other" and idle (the slow chess 2000M run: 31.6 against 7.7 thread-seconds of other, 15.6 against 7.9 idle), and on dell it comes with more system time (121 against 83 s). It is not a measurement error and not the profiler (the plain runs have it too). The runs before (ec77727) spread 5–8% on this case, so the two speeds came with the newer code; even the slow runs are 2.5–3× faster than then. Not fixed yet: [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md).
