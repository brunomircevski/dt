# Benchmark results

One directory per benchmark: `<backend>-<machine>-<dataset>/`.

- **backend:** `cpu` or `cuda`.
- **machine:** a short name for the computer. `pc` is the i7-14700KF desktop (8 performance + 12 efficiency cores, 28 threads); `legion` is the laptop with an i7-13650HX (6 + 8 cores, 20 threads) and an RTX 5070 Laptop GPU (8 GB).
  The desktop's and laptop's HIGGS runs are grouped in `pc-legion-higgs/`. The mordor3 cluster's runs are grouped in `cluster-*/` directories, named after its nodes. Two GPU servers run CUDA only: `hopper` (H100 PCIe, 2× Xeon Gold 6444Y) and `ampere` (A100 80 GB PCIe, 2× EPYC 7402). Two CPU servers run parallel only: `epyc` (2× EPYC 9255) and `dell` (2× Xeon Gold 6542Y). Their scripts are in [bench/cluster/](../cluster/README.md), not in a `run.sh`.
- **dataset:** the `bench/datasets.toml` entry, e.g. `susy` for the full SUSY dataset, `higgs`.
- **`-vs-others`:** the directory also runs other implementations (sklearn, rpart, Weka J48, YaDT) next to ./tree: `cpu-susy-vs-others`, which has no machine in its name because it holds three, one subdirectory each (`pc`, `epyc`, `dell`). Every other directory is ./tree only.

Each directory holds a `run.sh` that runs that benchmark and writes all of its results into the same directory:

```bash
bench/results/cpu-pc-susy-scaling/run.sh          # each case once
bench/results/cpu-pc-susy-scaling/run.sh -m 5     # each case 5 times
bench/results/cpu-susy-vs-others/run.sh pc -m 3   # its run.sh takes the machine: pc, epyc or dell
```

`BENCHMARK.md` next to it describes the method, and the results once the run is done. `run.sh` refuses to run where results already exist, so move them away to run again. When only ./tree changed, a run that also holds other tools can keep their rows: `cpu-susy-vs-others/run.sh <machine> -t` runs ./tree alone into a subdirectory and replaces the ./tree rows of `results.jsonl` with its rows (`machine.json` → `tree_rerun` records it; `report.py` and the charts show the rerun's date).

## The benchmarks

Times are ./tree's training time (presort, build, pruning; not reading the data). CART is pruned at α = 1e-5, C4.5 with CF 0.25.

**Against other implementations** (2026-10-16, full SUSY, 3 runs per case; the desktop's other tools: one run, 2026-10-09). Every tool builds the same tree on every machine; test accuracies agree within the confidence interval, except scikit-learn's CART, 0.2 points lower. CART is capped at depth 30 in every tool, because rpart cannot grow deeper.

| Directory | What | Main result |
|---|---|---|
| [`cpu-susy-vs-others`](cpu-susy-vs-others/BENCHMARK.md) | full SUSY (4.5M training rows): ./tree against scikit-learn, rpart (CART) and Weka J48, YaDT (C4.5), on one thread and on every CPU of the desktop (28 threads), epyc1 and dell1 (48 each); one chart for all three | On one thread ./tree is 68–180× faster than the CART libraries and 20–147× faster than the C4.5 ones (4.6 s against 412–900 s for CART). On every CPU it builds the trees in 0.49–1.19 s, 28–52× faster than YaDT (31–34 s). It has the lowest memory for C4.5. |

**./tree alone on the desktop and laptop** (2026-10-10, commit 49b7f47 or e809e97, one run per case):

| Directory | What | Main result |
|---|---|---|
| [`cpu-pc-susy-scaling`](cpu-pc-susy-scaling/BENCHMARK.md) | full SUSY, 1–28 threads on the desktop | CART 7.06 → 0.96 s (7.4×), C4.5 6.17 → 1.17 s (5.3×). Flat beyond 12–20 threads: memory bandwidth. |
| `cpu-legion-susy-scaling` | full SUSY, 1–20 threads on the laptop (`scaling.csv`, `chart.png`) | CART 8.05 → 1.03 s (7.8×), C4.5 6.88 → 1.04 s (6.6×). |
| [`cuda-legion-susy`](cuda-legion-susy/BENCHMARK.md) | full SUSY: serial, parallel and CUDA on the laptop | CUDA 0.37 s (CART) and 0.46 s (C4.5): 2.4–2.6× the desktop's 28 threads. |
| [`pc-legion-higgs`](pc-legion-higgs/BENCHMARK.md) | full HIGGS (10.5M rows): `cuda-legion-higgs` (laptop parallel and CUDA) and `cpu-pc-higgs` (desktop parallel) | CUDA 1.37 s (CART) and 2.00 s (C4.5), 2.3–2.6× the CPUs. The laptop's 20 threads match the desktop's 28: its DDR5 is faster. |

**./tree alone on the mordor3 cluster** (2026-10-15, 3 runs per case, commit 8da0c12). Every machine runs with ./tree's defaults, which were the fastest settings found on each. CART has no depth limit, like C4.5. Accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).

| Directory | What | Main result |
|---|---|---|
| [`cluster-higgs`](cluster-higgs/BENCHMARK.md) | full HIGGS on hopper, ampere (CUDA), epyc, dell (parallel, 48 threads) | CART: H100 0.41 s, A100 0.52 s, epyc 1.09 s, dell 1.26 s. C4.5: A100 1.15 s, H100 1.24 s, epyc 1.73 s, dell 2.30 s. 1.2–2.8× faster than before the tuning for these servers. |
| [`cluster-airlines`](cluster-airlines/BENCHMARK.md) | airline (115M rows), the same nodes; pruned trees, and in `noprunning/` the trees grown without pruning (a second chart) | Pruned CART: H100 1.08 s, 6.6× epyc (7.10 s). Pruned C4.5: A100 5.9 s, H100 6.1 s, epyc 11.9 s; the pruning, on the host CPU, is 36–41% of a CUDA run. Unpruned, CART grows 47.1 million nodes (depth 65) in 3.2 s (H100) to 9.3 s (dell); C4.5 grows 19.6 million in 3.1 s (H100) to 11.7 s (dell). |
| [`cluster-cpu-scaling`](cluster-cpu-scaling/BENCHMARK.md) | full HIGGS, 1–48 threads on epyc and dell | 48 threads: CART 1.12 s (epyc) and 1.26 s (dell), C4.5 1.77 and 2.25 s. dell's single thread is 1.8–1.9× slower than epyc's. epyc stalls where its second socket comes in (24 → 32 threads). |
| [`cluster-ecbdl`](cluster-ecbdl/BENCHMARK.md) | ECBDL'14 (631 features, 2026-10-17, one run per case, commit 49ebb6d): a 6M-row sample on all four nodes, the full 31.9M rows on epyc and dell (CUDA would need ~320 GB of VRAM) | 6M: H100 4.08 s (CART) and 13.92 s (C4.5), 1.8× and 1.6× epyc; VRAM 57 / 71 GiB of 80. Full: CART 52 s on both CPUs, C4.5 210 s (epyc) and 339 s (dell), 375 GiB of RAM. Pruned CART is a single leaf (2.15% positives). |

**Against other GPU libraries on the mordor3 cluster** (2026-10-15, 3 runs per case, commit 35cf315). One unpruned Gini tree, depth ≤ 12 and ≤ 19, on GPU 0 of hopper and ampere. No public CUDA C4.5 was found, so only CART is compared. The other libraries' code lives outside this repository.

| Directory | What | Main result |
|---|---|---|
| [`cluster-higgs-vs-gpu-libs`](cluster-higgs-vs-gpu-libs/BENCHMARK.md) | full HIGGS: ./tree CUDA against cuML (`RandomForestClassifier`, 1 tree) and XGBoost (1 round, Gini-equivalent gain), 128–1024 histogram bins | ./tree is 1.7–2.3× faster than cuML and 8–13× faster than XGBoost (depth 19, H100: 0.42 s against 0.96 s and 5.08 s). Test accuracy is the same within 0.1 points; the histogram libraries' trees classify 83–96% of test rows like ./tree's exact tree. |

## A new machine

1. Copy a directory and rename it for the machine, e.g. `cpu-server-susy-scaling`. (`cpu-susy-vs-others` instead takes a new machine as a case in its `run.sh`.)
2. In its `run.sh`, adjust `--cpus` to that machine's CPU numbering:
   - the first CPU listed runs the 1-thread cases and should be a fast core;
   - `--threads 1,all` uses every CPU listed.
3. Run `bench/setup.sh` once, prepare the data (`bench/.venv/bin/python bench/prepare.py susy`), then run `run.sh`.

CUDA benchmarks get `cuda-<machine>-<dataset>` directories the same way (`cuda-legion-susy`): `bench/run.py --impls tree_cuda` runs `./tree --cuda` (built by `make tree`) with the largest thread count and also records the peak GPU memory.
