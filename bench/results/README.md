# Benchmark results

One directory per benchmark: `<backend>-<machine>-<dataset>/`.

- **backend:** `cpu` or `cuda`.
- **machine:** a short name for the computer. `pc` is the i7-14700KF desktop (8 performance + 12 efficiency cores, 28 threads); `legion` is the laptop with an i7-13650HX (6 + 8 cores, 20 threads) and an RTX 5070 Laptop GPU (8 GB).
  The desktop's and laptop's HIGGS runs are grouped in `pc-legion-higgs/`. The mordor3 cluster's runs are grouped in `cluster-*/` directories, named after its nodes. Two GPU servers run CUDA only: `hopper` (H100 PCIe, 2× Xeon Gold 6444Y) and `ampere` (A100 80 GB PCIe, 2× EPYC 7402). Two CPU servers run parallel only: `epyc` (2× EPYC 9255) and `dell` (2× Xeon Gold 6542Y). Their scripts are in [bench/cluster/](../cluster/README.md), not in a `run.sh`.
- **dataset:** the `bench/datasets.toml` entry, e.g. `susy500k` for `susy_500k`, `susy` for the full dataset.
- **`-vs-others`:** the directory also runs other implementations (sklearn, rpart, Weka J48, YaDT) next to ./tree: `cpu-pc-susy500k-vs-others`, `cpu-pc-susy-vs-others`. Every other directory is ./tree only.

Each directory holds a `run.sh` that runs that benchmark and writes all of its results into the same directory:

```bash
bench/results/cpu-pc-susy500k-vs-others/run.sh          # each case once
bench/results/cpu-pc-susy500k-vs-others/run.sh -m 5     # each case 5 times
```

`BENCHMARK.md` next to it describes the method, and the results once the run is done. `run.sh` refuses to run where results already exist, so move them away to run again. When only ./tree changed, a run that also holds other tools can keep their rows: run `bench/run.py ... --impls tree` into a subdirectory and replace the ./tree rows of `results.jsonl` with its rows (`machine.json` → `tree_rerun` records it; `report.py` and the charts show the rerun's date).

## The benchmarks

Times are ./tree's training time (presort, build, pruning; not reading the data). CART is pruned at α = 1e-5, C4.5 with CF 0.25.

**Against other implementations** (desktop, one run per case; ./tree rerun on 2026-10-10, commit 49b7f47). Every tool builds a comparable tree. Test accuracies agree within the confidence interval; the one exception is scikit-learn's CART on full SUSY, 0.2 points lower. CART is capped at depth 30 in every tool, because rpart cannot grow deeper.

| Directory | What | Main result |
|---|---|---|
| [`cpu-pc-susy500k-vs-others`](cpu-pc-susy500k-vs-others/BENCHMARK.md) | SUSY 500k rows: ./tree against scikit-learn, rpart (CART) and Weka J48, YaDT (C4.5), 1 and 28 threads | On one thread ./tree is 17–30× faster than the CART libraries and 24–59× faster than the C4.5 ones. On 28 threads it is 32× faster than YaDT. |
| [`cpu-pc-susy-vs-others`](cpu-pc-susy-vs-others/BENCHMARK.md) | the same on full SUSY (4.5M training rows) | On one thread ./tree is 59–67× faster than the CART libraries and 21–135× faster than the C4.5 ones (6.2 s against 412 s for scikit-learn and 829 s for J48). On 28 threads it is 28× faster than YaDT. It has the lowest memory for C4.5. |

**./tree alone on the desktop and laptop** (2026-10-10, commit 49b7f47 or e809e97, one run per case):

| Directory | What | Main result |
|---|---|---|
| [`cpu-pc-susy-scaling`](cpu-pc-susy-scaling/BENCHMARK.md) | full SUSY, 1–28 threads on the desktop | CART 7.06 → 0.96 s (7.4×), C4.5 6.17 → 1.17 s (5.3×). Flat beyond 12–20 threads: memory bandwidth. |
| `cpu-legion-susy-scaling` | full SUSY, 1–20 threads on the laptop (`scaling.csv`, `chart.png`) | CART 8.05 → 1.03 s (7.8×), C4.5 6.88 → 1.04 s (6.6×). |
| [`cuda-legion-susy`](cuda-legion-susy/BENCHMARK.md) | full SUSY: serial, parallel and CUDA on the laptop | CUDA 0.37 s (CART) and 0.46 s (C4.5): 2.4–2.6× the desktop's 28 threads. |
| [`pc-legion-higgs`](pc-legion-higgs/BENCHMARK.md) | full HIGGS (10.5M rows): `cuda-legion-higgs` (laptop parallel and CUDA) and `cpu-pc-higgs` (desktop parallel) | CUDA 1.37 s (CART) and 2.00 s (C4.5), 2.3–2.6× the CPUs. The laptop's 20 threads match the desktop's 28: its DDR5 is faster. |

**./tree alone on the mordor3 cluster** (2026-10-14, 3 runs per case, commit a83b6ec). Every machine runs with ./tree's defaults, which were the fastest settings found on each. CART has no depth limit, like C4.5. Accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).

| Directory | What | Main result |
|---|---|---|
| [`cluster-higgs`](cluster-higgs/BENCHMARK.md) | full HIGGS on hopper, ampere (CUDA), epyc, dell (parallel, 48 threads) | CART: H100 0.44 s, A100 0.63 s, epyc 1.07 s, dell 1.23 s. C4.5: A100 1.46 s, H100 1.50 s, epyc 1.84 s, dell 2.32 s. 1.2–2.6× faster than before the tuning. |
| [`cluster-airlines`](cluster-airlines/BENCHMARK.md) | airline (115M rows), the same nodes; pruned trees, and in `noprunning/` the trees grown without pruning (a second chart) | Pruned CART: H100 2.20 s, 3.3× epyc (7.34 s). Pruned C4.5: H100 11.3 s, A100 11.6 s, epyc 13.9 s; over half of a CUDA run is the pruning, on the host CPU. Unpruned, CART grows 47.1 million nodes (depth 65) in 4.1 s (H100) to 10.1 s (dell); C4.5 grows 19.6 million in 4.4 s (H100) to 12.6 s (dell). |
| [`cluster-cpu-scaling`](cluster-cpu-scaling/BENCHMARK.md) | full HIGGS, 1–48 threads on epyc and dell | 48 threads: CART 1.05 s (epyc) and 1.26 s (dell), C4.5 1.84 and 2.30 s. dell's single thread is 1.7–1.9× slower than epyc's. epyc stalls where its second socket comes in (24 → 28 threads). |

## A new machine

1. Copy a directory and rename it for the machine, e.g. `cpu-server-susy500k`.
2. In its `run.sh`, adjust `--cpus` to that machine's CPU numbering:
   - the first CPU listed runs the 1-thread cases and should be a fast core;
   - `--threads 1,all` uses every CPU listed.
3. Run `bench/setup.sh` once, prepare the data (`bench/.venv/bin/python bench/prepare.py susy_500k`), then run `run.sh`.

CUDA benchmarks get `cuda-<machine>-<dataset>` directories the same way (`cuda-legion-susy`): `bench/run.py --impls tree_cuda` runs `./tree --cuda` (built by `make tree`) with the largest thread count and also records the peak GPU memory.
