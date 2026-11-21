# Benchmark results

Every benchmark here ran on the faculty's mordor3 cluster. Each is one directory, `cluster-<what>/`, whose `BENCHMARK.md` gives the method, the results, the commands that ran it and the ones that drew its charts (`chart*.png`). What the probes around them found (noise, placement, depth) is in [docs/PERFORMANCE.md](../docs/PERFORMANCE.md). Older benchmarks (desktop and laptop, other implementations on full SUSY, pruned trees) are in the git history, last at 254d4e8 and 96705a7.

Inside a benchmark, one directory per run: `<backend>-<node>-<dataset>/`.

- **backend:** `cuda` (`--hybrid`, the CPU takes the small nodes), `gpuonly` (`--gpu`), `cpu` (`--multicore`, 48 threads on both sockets), `serial` (`--serial`, one thread), `simple` (`--simple`, the one-thread reference backend).
- **node:** `hopper` (H100 PCIe, 2× Xeon Gold 6444Y) and `ampere` (A100 80 GB PCIe, 2× EPYC 7402) run CUDA only; `epyc` (epyc1, or epyc2 with the same hardware, 2× EPYC 9255) and `dell` (a 512 GB node, dell2, dell3 or dell4, 2× Xeon Gold 6542Y; dell1, 1 TB with every memory channel, is no longer used and its runs are out of date) run on the CPU only.
- **dataset:** the `bench/tools/datasets.toml` entry, e.g. `higgs`, `ecbdl_15m`, `chess_1b`.

A run directory holds `results.jsonl` (one row per run of a case), `plan.json` (the cases and the dataset's rows, features and split), `machine.json` (hardware, commit, date), `run.log`, and `report.md`, `runs.csv`, `summary.csv` from `bench/tools/report.py`. Two benchmarks differ: `cluster-cpu-vs-others` runs ./tree and every library in one `run.py`, `vs-epyc-<dataset>` (./tree's `--simple` bars come from the test-set benchmarks). `cluster-gpu-vs-others` runs ./tree, cuML and XGBoost together in `one-tree/vs-hopper-<dataset>[-depth30]` (the libraries' runner is `bench/tools/cluster/gpubench.py`).

The runs are SLURM jobs: `bench/tools/cluster/final.sh` queues all of them (one chain of jobs per node), `bench/tools/cluster/submit.sh` single ones; each runs `bench/tools/run.py --run-id cluster-<what>/<run>`, which writes into `bench/<run-id>/`. Setup, data and every submit command are in [bench/tools/cluster/README.md](tools/cluster/README.md).

## The benchmarks

**Final data (2026-11-17 to 2026-11-20):** every ./tree result below is from main's tree code as of d2957ff2, in a clone `~/dt-final` on mordor3, 10 runs per case (3 where one run takes over an hour, or over 30 min on one thread), CUDA on one GPU and every CPU of the node, `--multicore` on both sockets. The CPU runs on epyc of 2026-11-17 (8fe0c73d, 82745fb5) stand; the 2026-11-19/20 rerun (81bf6e6c, `bench/tools/cluster/final.sh`) replaced the rest: every CUDA run on hopper (the GPU setup is timed since 17d8b2f3: only the bare CUDA context is untimed), every dell `--multicore` run (dell2–4, 512 GB, half dell1's memory channels), the long one-thread runs, thread scaling, profiling, the comparison with the CPU libraries and the power benchmark. Long runs went in parts of under 4 h (`bench/tools/cluster/merge_parts.py`), so no node was held for a day. `--simple` (code unchanged since 2026-11-02) stands from 2026-11-10/12, except full SOREL-20M (3 runs on epyc and dell, 2026-11-19/20). **Out of date** (faded on the charts): ampere's CUDA runs (2026-11-15/17, GPU setup untimed), not rerun: ampere was unavailable from 2026-11-19 for two weeks. Which node ran what: [bench/tools/cluster/README.md](tools/cluster/README.md#the-2026-11-19-rerun). Every chart shows the runs behind each result.

Times are ./tree's training time (presort, build; not reading the data). Trees are grown to depth ≤ 10 without pruning, CART and C4.5. Depth 10 underfits and holds C4.5 back most ([depth sweep](../docs/PERFORMANCE.md#october-2026-what-the-benchmarks-showed-mordor3-cluster)).

**Trained on the training set, scored on the test set:** ./tree alone, every backend: CUDA `--hybrid` and `--gpu` on hopper and ampere, and on epyc and dell `--multicore` (48 threads), `--serial` and `--simple`.

| Directory | Training / test rows | Main result |
|---|---|---|
| [`cluster-susy`](cluster-susy/BENCHMARK.md) | 4.5M / UCI's 500,000 | H100 GPU only 154–174 ms, most of it GPU setup, 1.06–1.16× epyc's `--multicore` (179–184 ms); hybrid 262–277 ms. `--serial` 2.5–4.0 s, `--simple` 70–80 s. Test acc CART 79.0%, C4.5 78.7%. |
| [`cluster-higgs`](cluster-higgs/BENCHMARK.md) | 10.5M / UCI's 500,000 | H100 GPU only 215–240 ms, 1.5–1.6× epyc (348–366 ms); hybrid 327–353 ms. `--serial` 7.4–12.9 s, `--simple` 237–259 s. Test acc CART 70.5%, C4.5 63.7%. |
| [`cluster-mnist8m`](cluster-mnist8m/BENCHMARK.md) | 8.1M (784 features, 10 classes) / MNIST's 10,000 | H100 2.25–2.37 s, 1.4–1.5× epyc (3.26–3.39 s); dell 7.6–8.6 s. `--serial` 114–207 s, `--simple` 1,896–2,166 s. Test acc CART 84.5%, C4.5 74.7%. |
| [`cluster-ecbdl`](cluster-ecbdl/BENCHMARK.md) | 32.0M (CPU nodes) and the first 15M (every node) / the competition's 2.9M | 15M: H100 2.81–2.98 s, 1.5–1.7× epyc, VRAM 73.5 GB. Full: epyc 9.2–9.7 s, dell 23.6–29.1 s; `--serial` 316–553 s, `--simple` 5,970–7,319 s. Test acc 98.3% is the majority class's. |
| [`cluster-sorel`](cluster-sorel/BENCHMARK.md) | 12.7M (2381 features; CPU nodes) and the first 4M (every node) / 4.2M | Full: epyc 15.2–15.5 s, dell 36.8–42.8 s; `--serial` 591–1,047 s, `--simple` 12,538–13,390 s. 4M: H100 2.93–3.37 s, 1.5–1.7× epyc. Test acc CART 92.7%, C4.5 78.1%. |
| [`cluster-chess`](cluster-chess/BENCHMARK.md) | a 3×3 checkerboard (2 features) at 1M, 10M, 100M and 1000M rows (no test set) | The same 17-node tree everywhere. epyc fastest at 1M and 10M (the H100's GPU setup, 0.1–0.2 s, dominates there), the H100 from 100M on: GPU only 0.71–0.76 s at 1000M, 2.3–2.6× epyc (1.63–1.98 s). `--serial` 39–57 s and `--simple` 922–1,075 s at 1000M. |

**./tree on its own:**

| Directory | What | Main result |
|---|---|---|
| [`cluster-cpu-scaling`](cluster-cpu-scaling/BENCHMARK.md) | HIGGS and MNIST8M, 1–48 threads on epyc and dell; chart: speed-up over one thread | HIGGS, 48 threads: 19–23× one thread on epyc, 16–19× on dell (dell rerun 2026-11-20 with the CPUs in node order: HIGGS at low thread counts now spreads 2–12%, not 30–69%). MNIST8M: 33–35× (epyc), 24–26× (dell; dell1, with twice the memory channels, 34–36×). |
| [`cluster-profiling`](cluster-profiling/BENCHMARK.md) | where the growing time goes (`--profile`), chess 10M–2000M, HIGGS and ECBDL'14 15M, epyc, dell and hopper | Partition is 35–69% on epyc; the GPU's share grows from 2–3% at chess 10M to 59–64% at 2000M and 74–80% on ECBDL. C4.5 on big chess runs at two speeds (open). |
| [`hybrid-vs-gpu-only`](hybrid-vs-gpu-only/BENCHMARK.md) | CUDA `--hybrid` against `--gpu`, depth 5–30 and no limit, HIGGS, MNIST8M and SOREL 4M, ampere and hopper, `--handoff-rows` swept per dataset and node | On hopper, with the GPU setup timed (the hybrid's 0.12 s more for its hand-off buffers), GPU only is faster to depth 10 (HIGGS 0.60–0.68×, MNIST8M and SOREL 4M 0.92–0.97×). Deeper the hybrid leads, on HIGGS from depth 25: with no limit HIGGS 1.2–1.8×, MNIST8M 1.5–1.6×, SOREL 4M 1.08–1.20× (ampere, out of date: 2.1–2.3×, 1.4–1.5×, 1.08–1.16×). |

**Against other implementations** (HIGGS and MNIST8M):

| Directory | What | Main result |
|---|---|---|
| [`cluster-cpu-vs-others`](cluster-cpu-vs-others/BENCHMARK.md) | ./tree `--simple`, `--serial`, `--multicore` on epyc against scikit-learn and rpart (CART) and Weka J48 and YaDT (C4.5), one thread (YaDT also 48), in one `run.py` | HIGGS: `--multicore` 620× scikit-learn and 2,000× J48, `--serial` 27× and 103×; the same trees (J48's C4.5 on HIGGS 923 against 935 nodes). MNIST8M: `--serial` 8.8× scikit-learn and 147× J48. |
| [`cluster-gpu-vs-others`](cluster-gpu-vs-others/BENCHMARK.md) | ./tree `--gpu` and `--hybrid` against cuML (Gini) and XGBoost (Gini-equivalent), CART only (no GPU library grows C4.5 trees), one tree on hopper's H100, every tool uploading through the same code, a timeline of each run's phases; one tree per fresh process | One CART tree (10 runs): ./tree `--gpu` (exact) 2.4–2.5× faster than cuML (128 bins) and 2.4–3.3× than XGBoost (256 bins), on comparable trees; HIGGS at depth ≤ 30: 4.7× and 24.8×, `--hybrid` 1.14× faster than `--gpu`. |

**Energy** (not timed; the whole server at the PSU, from a meter model fitted per node):

| Directory | What | Main result |
|---|---|---|
| [`cluster-power`](cluster-power/BENCHMARK.md) | energy per tree of ./tree's five backends on epyc1 and hopper (ampere and dell4 until 2026-11-16 kept as history), ECBDL'14 15M, depth ≤ 10, CART and C4.5 | Per CART tree the H100 uses the least: 1.99 kJ on the GPU, 27% less than Multicore on epyc1 (2.73 kJ); Hybrid 14% more than GPU (final build, 2026-11-20; GPU trees after a 2 s warm-up, as long as in the time benchmark). |
| [`cluster-power-mnist8m`](cluster-power-mnist8m/BENCHMARK.md) | the same measurement where `--hybrid` gains: MNIST8M without a depth limit, `--multicore`, `--gpu` and `--hybrid` on hopper, CART and C4.5, 6 rounds | `--hybrid` 1.55–1.59× faster than `--gpu` at 1.45–1.51× the power: 2.7% (CART, 7.93 kJ) and 9.0% (C4.5, 9.08 kJ) less energy per tree; `--multicore` 3.1–3.4× more (2 s GPU warm-up since 2026-11-21, as long as in the time benchmark). |

## A new node or machine

- **Another mordor3 node:** add a line to the job table in `bench/tools/cluster/submit.sh` (run name, partition, node, implementation, threads) and name the run after it, e.g. `cpu-<node>-<dataset>`. [bench/tools/cluster/README.md](tools/cluster/README.md) has the settings each node needs.
- **Elsewhere:** run `bench/tools/setup.sh` once, prepare the data (`bench/tools/.venv/bin/python bench/tools/prepare.py <dataset>`), then `bench/tools/run.py <dataset> --protocols <…> --impls tree|tree_cuda --threads all --cpus <list> --reps 3 --run-id <name>`. The first CPU of `--cpus` runs the one-thread cases and should be a fast core; `tree_cuda` runs `./tree --hybrid` and `tree_gpu_only` `./tree --gpu` (built by `make tree`) and also records the peak GPU memory.
