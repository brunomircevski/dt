# Benchmark results

One directory per benchmark: `<backend>-<machine>-<dataset>/`.

- **backend:** `cpu` or `cuda`.
- **machine:** a short name for the computer. `pc` is the i7-14700KF desktop (8 performance + 12 efficiency cores, 28 threads); `legion` is the laptop with an i7-13650HX (6 + 8 cores, 20 threads) and an RTX 5070 Laptop GPU (8 GB).
  The desktop's and laptop's HIGGS runs are grouped in `pc-legion-higgs/`. The mordor3 cluster's runs are grouped in `cluster-*/` directories, named after its nodes. Two GPU servers run CUDA only: `hopper` (H100 PCIe, 2× Xeon Gold 6444Y) and `ampere` (A100 80 GB PCIe, 2× EPYC 7402). Two CPU servers run parallel only: `epyc` (2× EPYC 9255) and `dell` (2× Xeon Gold 6542Y). Their scripts are in [bench/cluster/](../cluster/README.md), not in a `run.sh`.
- **dataset:** the `bench/datasets.toml` entry, e.g. `susy` for the full SUSY dataset, `higgs`.
- **`-vs-others`:** the directory also runs other implementations (sklearn, rpart, Weka J48, YaDT) next to ./tree: `cluster-susy-vs-others-cpu`, which has no machine in its name because it holds three, one subdirectory each (`pc`, `epyc`, `dell`). Every other directory is ./tree only.

Each directory holds a `run.sh` that runs that benchmark and writes all of its results into the same directory:

```bash
bench/results/cpu-pc-susy-scaling/run.sh          # each case once
bench/results/cpu-pc-susy-scaling/run.sh -m 5     # each case 5 times
bench/results/cluster-susy-vs-others-cpu/run.sh pc -m 3   # its run.sh takes the machine: pc, epyc or dell
```

`BENCHMARK.md` next to it describes the method, and the results once the run is done. `run.sh` refuses to run where results already exist, so move them away to run again. When only ./tree changed, a run that also holds other tools can keep their rows: `cluster-susy-vs-others-cpu/run.sh <machine> -t` runs ./tree alone into a subdirectory and replaces the ./tree rows of `results.jsonl` with its rows (`machine.json` → `tree_rerun` records it; `report.py` and the charts show the rerun's date).

## The benchmarks

Times are ./tree's training time (presort, build, pruning; not reading the data). CART is pruned at α = 1e-5, C4.5 with CF 0.25.

**Serial (one-thread) results since 2026-10-17:** `--serial` now runs the parallel backend's algorithm (big nodes partitioned out of place, [docs/CPU.md](../../docs/CPU.md)), so the one-thread ./tree cases were rerun with `bench/serial_rerun.py` and their rows replaced, everywhere except on the laptop (`cpu-legion-susy-scaling`, `cuda-legion-susy`: not reachable then, their serial rows are older). Serial airline got 1.3–1.9× faster, SUSY and HIGGS stayed within ±20% (slower on epyc, faster on dell), and a serial run now needs the parallel backend's memory.

**Against other implementations** (2026-10-16, full SUSY, 3 runs per case; the desktop's other tools: one run, 2026-10-09). Every tool builds the same tree on every machine; test accuracies agree within the confidence interval, except scikit-learn's CART, 0.2 points lower. CART is capped at depth 30 in every tool, because rpart cannot grow deeper.

| Directory | What | Main result |
|---|---|---|
| [`cluster-susy-vs-others-cpu`](cluster-susy-vs-others-cpu/BENCHMARK.md) | full SUSY (4.5M training rows): ./tree against scikit-learn, rpart (CART) and Weka J48, YaDT (C4.5), on one thread and on every CPU of the desktop (28 threads), epyc1 and dell1 (48 each); one chart for all three | On one thread ./tree is 80–170× faster than the CART libraries and 22–148× faster than the C4.5 ones (4.4–6.8 s against 412–900 s for CART). On every CPU it builds the trees in 0.49–1.18 s, 28–55× faster than YaDT (31–34 s). It has the lowest memory for C4.5. ./tree's rows rerun on 051949b (epyc, dell, 2026-10-19) and cd18717 (desktop; one copy of the columns: 1.04–1.20 GiB on one thread). |

**./tree alone on the desktop and laptop** (2026-10-10, commit 49b7f47 or e809e97, one run per case):

| Directory | What | Main result |
|---|---|---|
| [`cpu-pc-susy-scaling`](cpu-pc-susy-scaling/BENCHMARK.md) | full SUSY, 1–28 threads on the desktop | 2026-10-10: CART 7.06 → 0.96 s (7.4×), C4.5 6.17 → 1.17 s (5.3×). Flat beyond 12–20 threads: memory bandwidth. The serial points were rerun on 2026-10-17 (4.67 and 5.88 s), the others not. |
| `cpu-legion-susy-scaling` | full SUSY, 1–20 threads on the laptop (`scaling.csv`, `chart.png`) | CART 8.05 → 1.03 s (7.8×), C4.5 6.88 → 1.04 s (6.6×). Serial points from before the serial backend's change. |
| [`cuda-legion-susy`](cuda-legion-susy/BENCHMARK.md) | full SUSY: serial, parallel and CUDA on the laptop | CUDA 0.37 s (CART) and 0.46 s (C4.5): 2.4–2.6× the desktop's 28 threads. |
| [`pc-legion-higgs`](pc-legion-higgs/BENCHMARK.md) | full HIGGS (10.5M rows): `cuda-legion-higgs` (laptop parallel and CUDA) and `cpu-pc-higgs` (desktop parallel) | CUDA 1.37 s (CART) and 2.00 s (C4.5), 2.3–2.6× the CPUs. The laptop's 20 threads match the desktop's 28: its DDR5 is faster. |

**./tree alone on the mordor3 cluster** (2026-10-19, 3 runs per case, commit 051949b: cd18717's one copy of the presorted columns on every backend, plus `--profile` and the count tables filled without a zero-fill on one thread; the same trees as on cd18717, the times within ±5%. The CUDA rows were rerun the same evening at 6f6cecb: hand-offs through page-locked buffers instead of a page-locked arena, plus d9da007's block partition; the same trees). Every machine runs with ./tree's defaults, which were the fastest settings found on each. CART has no depth limit, like C4.5. Accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).

| Directory | What | Main result |
|---|---|---|
| [`cluster-higgs`](cluster-higgs/BENCHMARK.md) | full HIGGS on hopper, ampere (CUDA), epyc, dell (parallel, 48 threads) | CART: H100 0.39 s, A100 0.52 s, epyc 1.09 s, dell 1.26 s. C4.5: A100 1.13 s, H100 1.20 s, epyc 1.73 s, dell 2.24 s. 1.2–2.8× faster than before the tuning for these servers. GPU memory 3.3 GiB (CART) and 4.4–4.5 GiB (C4.5), half of 8da0c12's. |
| [`cluster-airlines`](cluster-airlines/BENCHMARK.md) | airline (115M rows), the same nodes; pruned trees, and in `noprunning/` the trees grown without pruning (a second chart) | Pruned CART: H100 1.13 s, 6.2× epyc (7.05 s). Pruned C4.5: A100 5.6 s, H100 5.9 s, epyc 11.9 s; the pruning, on the host CPU, is 37–43% of a CUDA run. Unpruned, CART grows 47.1 million nodes (depth 65) in 2.9 s (H100) to 9.3 s (dell); C4.5 grows 19.6 million in 2.9 s (H100) to 11.4 s (dell). GPU memory 13.3 GiB (35 GiB before). |
| [`cluster-cpu-scaling`](cluster-cpu-scaling/BENCHMARK.md) | full HIGGS and airline, 1–48 threads on epyc and dell | HIGGS, 48 threads: CART 1.12 s (epyc) and 1.25 s (dell), C4.5 1.74 and 2.19 s. dell's single thread is 1.4–1.5× slower than epyc's. epyc stalls where its second socket comes in (24 → 32 threads). Airline, 48 threads: CART 7.06 s (epyc, 10.6×) and 7.38 s (dell, 17.2×), C4.5 11.94 and 13.79 s. Memory grows with the threads (one spare column slot per thread, up to features − 1): HIGGS 3.6 → 5.6 GiB, airline 19.6 → 28.1 GiB (CART). |
| [`cluster-ecbdl`](cluster-ecbdl/BENCHMARK.md) | ECBDL'14 (631 features, one run per case): a 15M-row sample on all four nodes and dell2 (512 GB), the full 31.9M rows on epyc, dell and dell2 (CUDA would need ~160 GB of VRAM); pruned CART, C4.5 and unpruned CART | 15M: H100 7.96 s (CART), 50.7 s (C4.5, its host pruning slower that evening), 17.4 s (unpruned CART), 1.5–2.1× epyc; VRAM 73.3–73.5 GiB of 80. Full: C4.5 221 s (epyc) and 338 s (dell), unpruned CART 83 and 122 s, 237 GiB of RAM (375 GiB before). Pruned CART is a single leaf (2.15% positives). dell2 has half dell1's memory channels: 1.35–1.55× slower. |
| [`cluster-ecbdl-profile`](cluster-ecbdl-profile/BENCHMARK.md) | where the training time goes (`./tree --profile`), ECBDL'14 15M sample, CUDA on hopper and ampere, parallel on epyc (`gpu-plus-cpu/`: 2026-10-19, 051949b, CUDA rerun at 6f6cecb; one run per case, with and without `--profile`): phases, what every CPU thread does, GPU kernels and copies, timelines | The H100 grows the trees 2.1–2.5× faster than epyc, but prunes C4.5 2.2× slower (on its host CPUs: 47% of a hopper run). On the GPU, moving the columns is 62–69% of the kernel time. The CPU threads of a GPU node are idle 39–94% of the time, epyc's busy 96–97%. 051949b's 1.2–1.9 s page-locking stall is gone. Profiling costs −2.4% to +0.9%. With `--gpu-only` and one CPU thread (2026-10-20, `gpu-only/`): the same trees, unpruned CART 2.45× (H100) and 1.45× (A100) the time, C4.5 7.4× and 3.5× (the one-thread prune), unpruned CART in 36.5 instead of 106.7 GiB of RAM. |
| [`cluster-chess`](cluster-chess/BENCHMARK.md) | a 3×3 checkerboard (2 features) at 1M, 10M, 100M and 1000M rows (2026-10-19/20, 3 runs per case): unpruned CART and C4.5 on hopper, ampere (CUDA, and `--gpu-only` with one CPU thread), epyc, dell and dell2 (parallel), epyc and dell (serial) | The same 17-node tree everywhere. epyc is fastest at 1M (23 ms). At 1000M the H100 takes 0.92 s (CART) and 0.97 s (C4.5), 7.0× and 7.5× faster than the best CPU server (epyc: 6.46 and 7.31 s), with 25.4 GiB of VRAM. The CPU runs are from d9da007, which partitions big nodes with few columns on all threads: before it, 1000M rows took 34–48 s on the CPU servers (13–41× the 100M time). GPU only, one CPU thread: 2.33 s (H100) and 2.61 s (A100) at 1000M, still 2.5–2.8× faster than epyc, with 9.96 GiB of RAM instead of 24.9. |

**Against other GPU libraries on the mordor3 cluster** (2026-10-17, 3 runs per case; the libraries with ./tree at 00ae13f; ECBDL'14 15M added on 2026-10-18 with ./tree on cd18717; ./tree's rows of all three rerun on 6f6cecb on 2026-10-19). One unpruned tree, depth ≤ 19, CART and C4.5, on GPU 0 of hopper and ampere. No public CUDA C4.5 was found, so C4.5 is compared with cuML's entropy tree, its closest. The other libraries' code lives outside this repository.

| Directory | What | Main result |
|---|---|---|
| [`cluster-tree-vs-gpu-libs`](cluster-tree-vs-gpu-libs/BENCHMARK.md) | full HIGGS, airline (115M rows), the ECBDL'14 15M sample (631 features) and chess 3×3 with 1000M rows: ./tree CUDA against cuML (`RandomForestClassifier`, 1 tree, 128 bins, Gini or entropy) and XGBoost (1 round, Gini-equivalent gain, 256 bins, CART only), with peak VRAM | ./tree is 2.4–3.7× faster than cuML and 13–17× faster than XGBoost on HIGGS and airline (airline CART, H100: 1.21 s against 4.23 s and 20.4 s); on ECBDL 2.4–3.5× and 4.0–6.0× (CART, H100: 6.45 s against 21.2 s and 38.6 s); on chess 1000M 7.0–10.3× and 79–118× (CART, H100: 0.90 s against 9.27 s and 106 s; the libraries' binned trees fill all 19 levels where ./tree's exact tree has 17 nodes). CART test accuracy is the same within 0.3 points; ./tree's C4.5 (gain ratio) builds a smaller, less accurate depth-19 tree than cuML's entropy tree. VRAM: ./tree 3.3–3.4 GiB on HIGGS and 13.3 GiB on airline (7.2 and 35 GiB on 00ae13f), cuML 1.9 and 9.5–10 GiB, XGBoost 2.9 and 13 GiB; ECBDL 73.3–73.4, 36 and 71.5 GiB; chess 25.4, 42.4 and 43.5 GiB. |

**Tree visualizations** (2026-10-19, 3 runs per case, ./tree at 5c2e051): every implementation above, on CPU (dell1, dell2) or CUDA (hopper), one unpruned tree with depth ≤ 10 on chess 3×3 with 1M rows split 80/20, drawn as an SVG with the run's time, accuracy, depth and nodes.

| Directory | What | Main result |
|---|---|---|
| [`visualization`](visualization/BENCHMARK.md) | 14 SVGs (`svg/`), the exported trees (`results/`), a log-scale timing chart and `summary.csv` | ./tree, scikit-learn, rpart, J48 and YaDT all find the same 17-node, depth-4 tree with 100% test accuracy. The binned cuML and XGBoost trees fill all 10 levels (389 and 479 nodes) at 98.78% and 99.51%. cuML is fastest at 16 ms; ./tree takes 44–59 ms on CPU and 45 ms on CUDA, scikit-learn 0.69 s, rpart 0.82 s, YaDT 0.18–0.19 s, J48 1.48 s, XGBoost 0.11 s. |

## A new machine

1. Copy a directory and rename it for the machine, e.g. `cpu-server-susy-scaling`. (`cluster-susy-vs-others-cpu` instead takes a new machine as a case in its `run.sh`.)
2. In its `run.sh`, adjust `--cpus` to that machine's CPU numbering:
   - the first CPU listed runs the 1-thread cases and should be a fast core;
   - `--threads 1,all` uses every CPU listed.
3. Run `bench/setup.sh` once, prepare the data (`bench/.venv/bin/python bench/prepare.py susy`), then run `run.sh`.

CUDA benchmarks get `cuda-<machine>-<dataset>` directories the same way (`cuda-legion-susy`): `bench/run.py --impls tree_cuda` runs `./tree --cuda` (built by `make tree`) with the largest thread count and also records the peak GPU memory.
