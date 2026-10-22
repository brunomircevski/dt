# Profiling: where CART and C4.5 spend their training time

**Status:** run on 2026-10-22 at commit b2aae2c, one run per case. The binaries were built on the server from 9ff3fd1, which has the same source; that hash is the one in `machine.json` and on the chart.
- **CUDA:** hopper (jobs 70–99), on the GPU's socket: H100 PCIe, 16 CPU threads, NUMA-bound.
- **Parallel:** epyc1, 48 threads (2× EPYC 9255). epyc2 was busy with another job.
- **Plain runs:** every case also ran once without `--profile` (`plain/`), to check what profiling costs.
- **C4.5 unpruned:** run later the same day at 76d4964 (same tree source), three runs per case (`-m 3`, jobs 104–133), in `c45-unpruned/` and `c45-unpruned/plain/`. The chart and table show the median run.
- **Reruns:** the epyc chess 10M and 100M cases were rerun at 13:41 (jobs 100–103). Their first profiled runs were 2.2–2.3× slower than the plain ones. The reruns show that this is run-to-run noise of sub-second runs on both sockets, not profiling overhead: now the *plain* runs are the slower ones (see below).

![chart](chart.png)

Rows: Parallel (CPU only), CUDA hybrid (the default), CUDA `--gpu-only`. Columns: CART, C4.5, C4.5 unpruned. Each bar is one training run (for C4.5 unpruned, the median of three) (100% = `train_seconds`, the total above the bar; reading the data is not included), split into six steps. Hatched segments ran on the GPU. Segments of 1% or more are labelled with their share and seconds; thin ones are labelled beside the bar.

## What runs

```bash
bench/cluster/profile.sh -d <dataset> -p cart_alpha_nodepth,c45 -o cluster-profiling \
    cpu-epyc-<dataset> cuda-hopper-<dataset> gpuonly-hopper-<dataset>
bench/cluster/profile.sh -m 3 -d <dataset> -p c45_unpruned -o cluster-profiling/c45-unpruned \
    cpu-epyc-<dataset> cuda-hopper-<dataset> gpuonly-hopper-<dataset>
```

These are the datasets: `chess_10m`, `chess_100m`, `chess_1b` (the 3×3 checkerboard, 2 features), `higgs` (10.5M rows × 28) and `ecbdl_15m` (the 15M-row ECBDL'14 sample, 631 features). `profile.sh` submits one profiled job per run (`./tree --profile FILE`), then the same run without `--profile` into `plain/`.

- **CART** = `cart_alpha_nodepth`: Gini, no depth limit, cost-complexity pruning at a fixed α = 1e-5.
- **C4.5** = `c45`: gain ratio, error-based pruning with CF 0.25 and subtree raising.
- **C4.5 unpruned** = `c45_unpruned` (`--no-prune`): the same tree as C4.5, grown the same way; only C4.5's collapse of useless splits runs, which counts as pruning (0–3%).
- Every mode grows the same trees.

## The six steps

| Step | What it measures | From which trace data |
|---|---|---|
| **Presort** | Sorting every feature column once, before growing. On the CPU this is a parallel sort of each column; on the GPU it is a gather, a radix sort and the distinct values. After it, every node reads its rows already in order. | GPU kernels `presort*`; CPU work `presort` |
| **Split search** | Finding the best threshold of every feature at every node: walking the sorted values, keeping the class counts on both sides of each cut, scoring it (Gini, or gain then gain ratio), and picking the winning feature. On the GPU: per-tile histograms, prefix sums and cut scans. | kernels `tile*`, `segment*`; CPU `sweep`, `build` |
| **Partition** | Applying the chosen split: marking the rows that go left, then moving every other column so that each child's rows are contiguous and still sorted. On the GPU this also gathers the columns of the small nodes handed to the CPU (hybrid). | kernels `markGoesLeft`, `partition`, `handOffGather`; CPU `mark`, `partition`, `partition_sweep`, `leaves` |
| **CPU↔GPU transfer** | Moving data between RAM and the GPU: the upload of the dataset, the hand-off columns, the per-level lists and cuts, and the host copies through page-locked buffers. | copy-lane spans; CPU `staging`, `page_lock` |
| **Pruning** | Cutting the grown tree back. CART: the weakest-link sequence, cut at α. C4.5: rows grouped by leaf, the pessimistic error of every subtree, subtree raising, collapsing and the final thresholds. | kernels `leafRows`, `arrange*`, `copyValues`; CPU `prune*` |
| **Other** | Everything else: GPU context and memory allocation, the host launching kernels or waiting while the GPU is idle, node bookkeeping, assembling the final tree, and untracked time. | the rest |

**How each millisecond is assigned** ([profile_share_chart.py](../../profile_share_chart.py)):
1. Running GPU kernels come first. Time when kernels of two steps overlap is shared between them.
2. Then a host↔GPU copy with no kernel running counts as transfer.
3. Then the training thread's phase decides. While it drives the GPU (gpu setup, upload, grow setup, gpu levels), the time is transfer if threads are staging or page-locking, and other otherwise.
4. Elsewhere, the time is shared among the busy CPU threads by their kind of work, taken from the 5 ms samples. If every thread is idle, it goes to the phase's step.

In hybrid mode, CPU subtree growth that hides behind the GPU is therefore not counted; it shows only when the GPU waits for it.

**Code change (b2aae2c):** in the fused CPU partition (`partitionAndSweep`), the children's sweeps are now traced as `sweep`, so they count as split search instead of partition. When not profiling, this costs one predicted branch per 1024 entries.

## Results

| Mode | Algorithm | Dataset | runs | train s | without --profile s | Presort % | Split search % | Partition % | CPU↔GPU transfer % | Pruning % | Other % | GPU share % |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Parallel (CPU only) | CART | chess 10M | 1 | 0.14 | 0.51 | 27.6 | 5.7 | 62.7 | 0.0 | 0.0 | 4.0 | 0.0 |
| Parallel (CPU only) | CART | chess 100M | 1 | 1.11 | 0.52 | 19.7 | 13.1 | 65.0 | 0.0 | 0.0 | 2.2 | 0.0 |
| Parallel (CPU only) | CART | chess 1000M | 1 | 5.22 | 5.18 | 35.9 | 23.8 | 37.6 | 0.0 | 0.0 | 2.6 | 0.0 |
| Parallel (CPU only) | CART | HIGGS | 1 | 1.11 | 1.07 | 13.2 | 16.1 | 69.8 | 0.0 | 0.0 | 1.0 | 0.0 |
| Parallel (CPU only) | CART | ECBDL'14 15M | 1 | 16.44 | 16.24 | 8.4 | 30.3 | 61.0 | 0.0 | 0.0 | 0.3 | 0.0 |
| Parallel (CPU only) | C4.5 | chess 10M | 1 | 0.12 | 0.25 | 31.9 | 20.2 | 19.4 | 0.0 | 20.2 | 8.4 | 0.0 |
| Parallel (CPU only) | C4.5 | chess 100M | 1 | 0.93 | 1.25 | 22.5 | 23.7 | 23.7 | 0.0 | 22.5 | 7.7 | 0.0 |
| Parallel (CPU only) | C4.5 | chess 1000M | 1 | 8.80 | 8.45 | 21.6 | 22.4 | 22.8 | 0.0 | 26.9 | 6.3 | 0.0 |
| Parallel (CPU only) | C4.5 | HIGGS | 1 | 1.74 | 1.71 | 8.3 | 22.3 | 56.4 | 0.0 | 10.9 | 2.1 | 0.0 |
| Parallel (CPU only) | C4.5 | ECBDL'14 15M | 1 | 77.57 | 77.00 | 2.0 | 31.8 | 52.8 | 0.0 | 13.2 | 0.2 | 0.0 |
| Parallel (CPU only) | C4.5 unpruned | chess 10M | 3 | 0.13 | 0.16 | 38.4 | 22.2 | 35.6 | 0.0 | 0.0 | 3.8 | 0.0 |
| Parallel (CPU only) | C4.5 unpruned | chess 100M | 3 | 0.94 | 0.93 | 22.4 | 21.8 | 52.1 | 0.0 | 0.0 | 3.6 | 0.0 |
| Parallel (CPU only) | C4.5 unpruned | chess 1000M | 3 | 6.04 | 6.03 | 31.6 | 33.5 | 32.7 | 0.0 | 0.0 | 2.2 | 0.0 |
| Parallel (CPU only) | C4.5 unpruned | HIGGS | 3 | 1.56 | 1.54 | 10.5 | 25.0 | 62.0 | 0.0 | 0.7 | 1.8 | 0.0 |
| Parallel (CPU only) | C4.5 unpruned | ECBDL'14 15M | 3 | 67.31 | 66.79 | 2.3 | 36.6 | 60.8 | 0.0 | 0.0 | 0.3 | 0.0 |
| CUDA hybrid (default) | CART | chess 10M | 1 | 0.05 | 0.07 | 2.1 | 2.2 | 2.5 | 12.6 | 0.0 | 80.6 | 6.8 |
| CUDA hybrid (default) | CART | chess 100M | 1 | 0.10 | 0.10 | 10.3 | 3.3 | 14.3 | 31.6 | 0.0 | 40.5 | 28.0 |
| CUDA hybrid (default) | CART | chess 1000M | 1 | 0.88 | 0.88 | 11.6 | 3.2 | 54.0 | 21.5 | 0.0 | 9.6 | 68.8 |
| CUDA hybrid (default) | CART | HIGGS | 1 | 0.43 | 0.42 | 4.0 | 30.3 | 31.1 | 17.7 | 1.2 | 15.6 | 46.6 |
| CUDA hybrid (default) | CART | ECBDL'14 15M | 1 | 7.98 | 7.93 | 6.6 | 25.4 | 55.8 | 9.8 | 0.0 | 2.4 | 87.8 |
| CUDA hybrid (default) | C4.5 | chess 10M | 1 | 0.20 | 0.22 | 0.7 | 0.9 | 0.7 | 8.0 | 45.7 | 44.0 | 2.7 |
| CUDA hybrid (default) | C4.5 | chess 100M | 1 | 0.76 | 0.92 | 1.5 | 0.5 | 1.9 | 5.7 | 75.7 | 14.8 | 5.8 |
| CUDA hybrid (default) | C4.5 | chess 1000M | 1 | 7.59 | 7.78 | 1.5 | 0.4 | 6.3 | 3.8 | 78.8 | 9.1 | 10.2 |
| CUDA hybrid (default) | C4.5 | HIGGS | 1 | 1.47 | 1.43 | 1.4 | 32.8 | 19.7 | 8.5 | 27.8 | 9.8 | 20.6 |
| CUDA hybrid (default) | C4.5 | ECBDL'14 15M | 1 | 49.19 | 48.98 | 1.2 | 15.4 | 33.4 | 3.2 | 45.3 | 1.4 | 49.9 |
| CUDA hybrid (default) | C4.5 unpruned | chess 10M | 3 | 0.10 | 0.08 | 1.4 | 1.9 | 1.4 | 15.0 | 0.0 | 80.2 | 4.7 |
| CUDA hybrid (default) | C4.5 unpruned | chess 100M | 3 | 0.11 | 0.12 | 10.3 | 3.0 | 12.5 | 39.5 | 0.0 | 34.7 | 25.8 |
| CUDA hybrid (default) | C4.5 unpruned | chess 1000M | 3 | 0.97 | 0.96 | 12.0 | 2.9 | 49.1 | 29.5 | 0.0 | 6.5 | 63.9 |
| CUDA hybrid (default) | C4.5 unpruned | HIGGS | 3 | 1.00 | 0.99 | 2.0 | 47.5 | 27.3 | 7.3 | 3.0 | 12.8 | 28.1 |
| CUDA hybrid (default) | C4.5 unpruned | ECBDL'14 15M | 3 | 26.95 | 26.96 | 2.2 | 28.3 | 61.1 | 5.8 | 0.0 | 2.6 | 91.2 |
| CUDA --gpu-only | CART | chess 10M | 1 | 0.08 | 0.10 | 1.6 | 2.1 | 1.9 | 10.6 | 0.0 | 83.8 | 5.6 |
| CUDA --gpu-only | CART | chess 100M | 1 | 0.10 | 0.10 | 10.3 | 3.3 | 14.3 | 31.2 | 0.0 | 40.8 | 28.0 |
| CUDA --gpu-only | CART | chess 1000M | 1 | 0.89 | 0.89 | 11.5 | 3.2 | 53.7 | 22.0 | 0.0 | 9.6 | 68.4 |
| CUDA --gpu-only | CART | HIGGS | 1 | 0.40 | 0.40 | 4.3 | 28.1 | 31.4 | 14.1 | 1.3 | 20.7 | 63.9 |
| CUDA --gpu-only | CART | ECBDL'14 15M | 1 | 7.96 | 7.97 | 6.6 | 25.6 | 55.9 | 9.7 | 0.0 | 2.2 | 88.1 |
| CUDA --gpu-only | C4.5 | chess 10M | 1 | 0.22 | 0.22 | 0.6 | 0.8 | 0.7 | 3.9 | 50.3 | 43.7 | 2.5 |
| CUDA --gpu-only | C4.5 | chess 100M | 1 | 0.97 | 0.94 | 1.2 | 0.4 | 1.5 | 3.2 | 83.0 | 10.8 | 4.5 |
| CUDA --gpu-only | C4.5 | chess 1000M | 1 | 7.19 | 7.77 | 1.6 | 0.4 | 6.6 | 2.7 | 77.7 | 10.9 | 10.7 |
| CUDA --gpu-only | C4.5 | HIGGS | 1 | 3.06 | 3.03 | 0.7 | 22.3 | 15.0 | 12.6 | 14.2 | 35.3 | 38.7 |
| CUDA --gpu-only | C4.5 | ECBDL'14 15M | 1 | 59.87 | 59.70 | 1.0 | 18.9 | 30.9 | 3.3 | 38.4 | 7.5 | 50.9 |
| CUDA --gpu-only | C4.5 unpruned | chess 10M | 3 | 0.09 | 0.09 | 1.5 | 1.8 | 1.5 | 8.3 | 0.0 | 86.9 | 4.7 |
| CUDA --gpu-only | C4.5 unpruned | chess 100M | 3 | 0.10 | 0.10 | 11.2 | 3.3 | 13.7 | 28.9 | 0.0 | 42.8 | 28.2 |
| CUDA --gpu-only | C4.5 unpruned | chess 1000M | 3 | 0.90 | 0.89 | 12.9 | 3.1 | 52.9 | 21.1 | 0.0 | 10.1 | 68.9 |
| CUDA --gpu-only | C4.5 unpruned | HIGGS | 3 | 2.61 | 2.59 | 0.8 | 26.1 | 17.5 | 13.5 | 1.5 | 40.6 | 44.4 |
| CUDA --gpu-only | C4.5 unpruned | ECBDL'14 15M | 3 | 37.33 | 37.26 | 1.6 | 30.4 | 49.7 | 5.7 | 0.0 | 12.5 | 81.8 |

`runs` is the number of profiled runs; the row is the median one. `without --profile s` is the plain run (`plain/`; for C4.5 unpruned, the median of three). GPU share is the share of segments that ran on the GPU.

**Profiling overhead.** Profiled and plain times agree within −8% to +4% everywhere (C4.5 unpruned: within ±3% except hybrid chess 10M, 0.10 / 0.08 s), except the sub-second epyc chess 10M and 100M runs. Those vary 2–4× from run to run, in both directions:

| epyc, CART | first run: profiled / plain | rerun: profiled / plain |
|---|--:|--:|
| chess 10M | 0.27 / 0.12 s | 0.14 / 0.51 s |
| chess 100M | 1.39 / 0.60 s | 1.11 / 0.52 s |

## Findings

- **CPU growing is partition-bound.** On epyc, partition is 38–70% of CART and 19–56% of C4.5, and split search 6–32%. Only chess 1000M has a large presort share (36% CART).
- **On the GPU the trees grow fast, so C4.5's pruning dominates.** C4.5's pruning still runs on the host CPU: 74–83% of a CUDA run on chess 100M/1000M and 38–45% on ECBDL. The same pruning is 11–27% of an epyc run, which has 48 threads instead of 16.
- **CUDA CART on big data is GPU work.** On ECBDL: partition 56%, split search 25%, transfer 10%, 88% of the time on the GPU. Chess 1000M: partition 54%, transfer 22%.
- **Small runs are overhead.** CUDA chess 10M/100M take 0.05–0.10 s; 40–84% of that is "other" (context, allocation, launches) and 13–32% transfer.
- **Hybrid against `--gpu-only`.** Same CART. HIGGS C4.5 takes 1.47 s hybrid and 3.06 s GPU only, where "other" is 35% (1.08 s): launching and waiting on many small nodes the hybrid mode hands to the CPU. ECBDL C4.5: 49.2 s and 59.9 s.
- **What pruning costs, from C4.5 minus C4.5 unpruned.** On hopper, pruning adds 6.6 s to chess 1000M (0.97 s → 7.59 s hybrid, 7.8×) and 22 s to ECBDL (27.0 → 49.2 s hybrid, 37.3 → 59.9 s `--gpu-only`). On epyc it adds 2.8 s and 10.3 s (6.04 → 8.80 s, 67.3 → 77.6 s). So growing on the H100 is 2.5–6× faster than on epyc, but the whole C4.5 run is only 1.2–1.6× faster.
- **Why chess prunes on the CPU even with the GPU path.** Pruning's first step (rows ordered by leaf) can run on the GPU (`arrangeRows`), but its scratch must fit in the column buffer, `(features + 1) · 8` bytes per row. It needs about `4·features + 30` bytes per row plus cub's temporary storage, so it fits only from about 6 features: chess (2 features) needs 287 MiB for 234 MiB at 10M rows and falls back to the CPU (4.9 s of chess 1000M's 6.0 s pruning). HIGGS fits (1434 of 2378 MiB). ECBDL does not keep the raw values on the GPU at all (38 GB next to 73 GB of columns). Checked on ampere with a diagnostic build; not in the code.
- **Without pruning, the GPU rows look like CART.** C4.5 unpruned on hopper is 64–91% GPU time on chess 1000M and ECBDL (partition 49–61%, split search 28–30% on ECBDL), and HIGGS `--gpu-only` again spends 41% on "other".
