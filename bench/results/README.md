# Benchmark results

Every benchmark here ran on the faculty's mordor3 cluster. Each is one directory, `cluster-<what>/`, whose `BENCHMARK.md` gives the method, the results, the commands that ran it and the ones that drew its charts (`chart*.png`). What the probes around them found (noise, placement, depth) is in [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md). Older benchmarks (desktop and laptop, other implementations on full SUSY, pruned trees) are in the git history, last at 254d4e8 and 96705a7.

Inside a benchmark, one directory per run: `<backend>-<node>-<dataset>/`.

- **backend:** `cuda` (`--hybrid`, the CPU takes the small nodes), `gpuonly` (`--gpu`), `cpu` (`--multicore`, 48 threads on both sockets), `serial` (`--serial`, one thread), `simple` (`--simple`, the one-thread reference backend).
- **node:** `hopper` (H100 PCIe, 2× Xeon Gold 6444Y) and `ampere` (A100 80 GB PCIe, 2× EPYC 7402) run CUDA only; `epyc` (epyc1, or epyc2 with the same hardware, 2× EPYC 9255) and `dell` (a 512 GB node, dell2, dell3 or dell4, 2× Xeon Gold 6542Y; dell1, 1 TB with every memory channel, is no longer used and its runs are out of date) run on the CPU only.
- **dataset:** the `bench/datasets.toml` entry, e.g. `higgs`, `ecbdl_15m`, `chess_1b`.

A run directory holds `results.jsonl` (one row per run of a case), `plan.json` (the cases and the dataset's rows, features and split), `machine.json` (hardware, commit, date), `run.log`, and `report.md`, `runs.csv`, `summary.csv` from `bench/report.py`. Two benchmarks differ: `cluster-cpu-vs-others` names its runs `vs-epyc-<dataset>-<tool>-<algorithm>` and takes ./tree's bars from the test-set benchmarks. `cluster-gpu-vs-others` runs ./tree, cuML and XGBoost together in `one-tree/vs-hopper-<dataset>` (the libraries' runner in `scripts/`).

The runs are SLURM jobs: `bench/cluster/final.sh` queues all of them (one chain of jobs per node), `bench/cluster/submit.sh` single ones; each runs `bench/run.py --run-id cluster-<what>/<run>`, which writes into `bench/results/<run-id>/`. Setup, data and every submit command are in [bench/cluster/README.md](../cluster/README.md).

## The benchmarks

**Final rerun (2026-11-15/16):** every ./tree result below is from one build of main, f53347a9 (`src/` = d2957ff2), run by `bench/cluster/final.sh -m 5` in a clone `~/dt-final` on mordor3: 5 runs per case, CUDA on one GPU and every CPU of the node, `--multicore` on both sockets. Not rerun, because their code has not changed: `--simple` (10 runs per case from 2026-11-10/12, 3 on full ECBDL'14, 1 on full SOREL-20M from 2026-11-02), scikit-learn, J48, cuML and XGBoost (2026-11-09/10). Every chart shows the runs behind each result. The runs before (f1fe270e, 10 runs per case) are in git at 1e655067; the jobs and their order per node are in [bench/cluster/README.md](../cluster/README.md#the-final-rerun).

Times are ./tree's training time (presort, build; not reading the data). Trees are grown to depth ≤ 10 without pruning, CART and C4.5. Depth 10 underfits and holds C4.5 back most ([depth sweep](../../docs/PERFORMANCE.md#october-2026-what-the-benchmarks-showed-mordor3-cluster)).

**Trained on the training set, scored on the test set:** ./tree alone, every backend: CUDA `--hybrid` and `--gpu` on hopper and ampere, and on epyc and dell `--multicore` (48 threads), `--serial` and `--simple`.

| Directory | Training / test rows | Main result (f53347a9, 2026-11-15/16) |
|---|---|---|
| [`cluster-susy`](cluster-susy/BENCHMARK.md) | 4.5M / UCI's 500,000 | Both GPUs 62–85 ms, 2.0–2.6× epyc's `--multicore` (162–172 ms). `--serial` 2.5–4.0 s, `--simple` 70–80 s. Test acc CART 79.0%, C4.5 78.7%. |
| [`cluster-higgs`](cluster-higgs/BENCHMARK.md) | 10.5M / UCI's 500,000 | H100 0.13–0.15 s, 2.2–2.5× epyc (0.33–0.34 s). `--serial` 7.4–12.7 s, `--simple` 237–259 s. Test acc CART 70.5%, C4.5 63.7%. |
| [`cluster-mnist8m`](cluster-mnist8m/BENCHMARK.md) | 8.1M (784 features, 10 classes) / MNIST's 10,000 | H100 2.0–2.1 s, A100 and epyc 3.1–3.3 s. `--serial` 113–205 s, `--simple` 1,896–2,166 s. Test acc CART 84.5%, C4.5 74.7%. |
| [`cluster-ecbdl`](cluster-ecbdl/BENCHMARK.md) | 32.0M (CPU nodes) and the first 15M (every node) / the competition's 2.9M | 15M: H100 2.4–2.6 s, 1.8–1.9× epyc, VRAM 73.5 GB. Full: epyc 9.2–9.7 s, dell 16.4–19.1 s; `--serial` 317–548 s, `--simple` 5,970–7,319 s. Test acc 98.3% is the majority class's. |
| [`cluster-sorel`](cluster-sorel/BENCHMARK.md) | 12.7M (2381 features; CPU nodes) and the first 4M (every node) / 4.2M | Full: epyc 15.3–15.5 s, dell 26.6–29.5 s; `--serial` 599–975 s, `--simple` 12,501–13,355 s. 4M: H100 2.7–3.1 s, 1.6–1.8× epyc. Test acc CART 92.7%, C4.5 78.1%. |
| [`cluster-chess`](cluster-chess/BENCHMARK.md) | a 3×3 checkerboard (2 features) at 1M, 10M, 100M and 1000M rows (no test set) | The same 17-node tree everywhere. epyc fastest at 1M, the A100 at 10M, the H100 from 100M on: 0.61–0.62 s at 1000M, 2.6–3.1× epyc (1.58–1.93 s). `--serial` 39–58 s and `--simple` 922–1,075 s at 1000M. |

**./tree on its own:**

| Directory | What | Main result (f53347a9, 2026-11-15/16) |
|---|---|---|
| [`cluster-cpu-scaling`](cluster-cpu-scaling/BENCHMARK.md) | HIGGS and MNIST8M, 1–48 threads on epyc and dell; chart: speed-up over one thread | HIGGS, 48 threads: 19–24× one thread. MNIST8M: 35–37× (epyc), 34–36× (dell). |
| [`cluster-profiling`](cluster-profiling/BENCHMARK.md) | where the growing time goes (`--profile`), chess 10M–2000M, HIGGS, airline and ECBDL'14 15M, epyc1, dell1 and hopper | Partition is 33–75% on epyc; the GPU's share grows from 6–8% at chess 10M to 75–77% at 2000M and 85–88% on ECBDL. C4.5 on big chess runs at two speeds (open). |
| [`hybrid-vs-gpu-only`](hybrid-vs-gpu-only/BENCHMARK.md) | CUDA `--hybrid` against `--gpu`, depth 5–30 and no limit, HIGGS, MNIST8M and SOREL 4M, ampere and hopper, `--handoff-rows` swept per dataset and node | Equal to depth 15. Deeper the hybrid wins, much less than before as GPU only got faster: with no limit HIGGS 2.1–2.3× on ampere and 1.4–2.1× on hopper (4.3–4.9× before), MNIST8M 1.4–1.6×, SOREL 4M 1.08–1.19×. |

**Against other implementations** (HIGGS and MNIST8M):

| Directory | What | Main result |
|---|---|---|
| [`cluster-cpu-vs-others`](cluster-cpu-vs-others/BENCHMARK.md) | ./tree `--simple`, `--serial`, `--multicore` on epyc against scikit-learn (CART) and Weka J48 (C4.5), one thread | HIGGS: `--multicore` 660× scikit-learn and 2,400× J48, `--serial` 27× and 112×; the same trees (J48's C4.5 on HIGGS 923 against 935 nodes). |
| [`cluster-gpu-vs-others`](cluster-gpu-vs-others/BENCHMARK.md) | ./tree `--gpu` and `--hybrid` against cuML (Gini) and XGBoost (Gini-equivalent), CART only (no GPU library grows C4.5 trees), one tree on hopper's H100, every tool uploading through the same code, a timeline of each run's sections | One CART tree (3 runs): ./tree `--gpu` 2.4–2.8× faster than cuML and 2.3–3.3× than XGBoost; its tree construction 3.1–4.6× faster. |

**Energy** (not timed; the whole server at the PSU, from a meter model fitted per node):

| Directory | What | Main result |
|---|---|---|
| [`cluster-power`](cluster-power/BENCHMARK.md) | energy per tree of ./tree's five backends on epyc1, dell4, ampere and hopper, ECBDL'14 15M, depth ≤ 10, CART and C4.5 | Per CART tree the H100 uses the least: 1.81 kJ on the GPU, 33% less than Multicore on epyc1 (2.70 kJ). |

## A new node or machine

- **Another mordor3 node:** add a line to the job table in `bench/cluster/submit.sh` (run name, partition, node, implementation, threads) and name the run after it, e.g. `cpu-<node>-<dataset>`. [bench/cluster/README.md](../cluster/README.md) has the settings each node needs.
- **Elsewhere:** run `bench/setup.sh` once, prepare the data (`bench/.venv/bin/python bench/prepare.py <dataset>`), then `bench/run.py <dataset> --protocols <…> --impls tree|tree_cuda --threads all --cpus <list> --reps 3 --run-id <name>`. The first CPU of `--cpus` runs the one-thread cases and should be a fast core; `tree_cuda` runs `./tree --hybrid` and `tree_gpu_only` `./tree --gpu` (built by `make tree`) and also records the peak GPU memory.
