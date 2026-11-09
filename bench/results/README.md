# Benchmark results

Every benchmark here ran on the faculty's mordor3 cluster. Each is one directory, `cluster-<what>/`, whose `BENCHMARK.md` gives the method, the results, the commands that ran it and the ones that drew its charts (`chart*.png`). What the probes around them found (noise, placement, depth) is in [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md). Older benchmarks (desktop and laptop, other implementations on full SUSY, pruned trees) are in the git history, last at 254d4e8 and 96705a7.

Inside a benchmark, one directory per run: `<backend>-<node>-<dataset>/`.

- **backend:** `cuda` (`--cuda`, the CPU takes the small nodes), `gpuonly` (`--cuda --gpu-only`), `cpu` (`--parallel`, 48 threads on both sockets), `serial` (`--serial`, one thread), `simple` (`--simple`, the one-thread reference backend).
- **node:** `hopper` (H100 PCIe, 2× Xeon Gold 6444Y) and `ampere` (A100 80 GB PCIe, 2× EPYC 7402) run CUDA only; `epyc` (epyc1, or epyc2 with the same hardware, 2× EPYC 9255) and `dell` (dell1, 2× Xeon Gold 6542Y) run on the CPU only.
- **dataset:** the `bench/datasets.toml` entry, e.g. `higgs`, `ecbdl_15m`, `chess_1b`.

A run directory holds `results.jsonl` (one row per run of a case), `plan.json` (the cases and the dataset's rows, features and split), `machine.json` (hardware, commit, date), `run.log`, and `report.md`, `runs.csv`, `summary.csv` from `bench/report.py`. Two benchmarks differ: `cluster-cpu-vs-others` names its runs `vs-epyc-<dataset>-<tool>-<algorithm>`, and `cluster-gpu-vs-others` keeps one directory per dataset and node (cuML and XGBoost) with the scripts it ran in `scripts/`; it runs outside this repository (`~/gpu-dt-bench`). Both take ./tree's bars from the test-set benchmarks.

The runs are SLURM jobs: `bench/cluster/final.sh` queues all of them (one chain of jobs per node), `bench/cluster/submit.sh` single ones; each runs `bench/run.py --run-id cluster-<what>/<run>`, which writes into `bench/results/<run-id>/`. Setup, data and every submit command are in [bench/cluster/README.md](../cluster/README.md).

## The benchmarks

**Final rerun (2026-11-09):** every benchmark below except `cluster-profiling` and `hybrid-vs-gpu-only` is being rerun on one build of main (`bench/cluster/final.sh`, in a clone `~/dt-final` on mordor3): 10 runs per case, 3 where one process takes over an hour (`--simple` on full ECBDL'14 and SOREL-20M, J48 on MNIST8M); `--parallel` on both sockets only (the one-socket runs, dell2, airline and the vs-others datasets other than HIGGS and MNIST8M are removed, in git at 3354a8a); CUDA runs on one GPU and every CPU of the node. Until the new runs are copied in, each folder holds the runs of 2026-10-29/30 on ec77727 (CUDA on the GPU's socket only), whose numbers the tables below give. Each chart notes its runs per case. The jobs, their order per node and the expected duration (about 3 days on epyc1/epyc2, 5–6 on dell1, half a day per GPU node) are in [bench/cluster/README.md](../cluster/README.md#the-final-rerun).

Times are ./tree's training time (presort, build; not reading the data). Trees are grown to depth ≤ 10 without pruning, CART and C4.5. Depth 10 underfits and holds C4.5 back most ([depth sweep](../../docs/PERFORMANCE.md#october-2026-what-the-benchmarks-showed-mordor3-cluster)).

**Trained on the training set, scored on the test set:** ./tree alone, every backend: CUDA hybrid and GPU only on hopper and ampere, and on epyc and dell `--parallel` (48 threads), `--serial` and `--simple`.

| Directory | Training / test rows | Main result (ec77727, 2026-10) |
|---|---|---|
| [`cluster-susy`](cluster-susy/BENCHMARK.md) | 4.5M / UCI's 500,000 | Both GPUs 82–119 ms, 2.2–2.9× epyc. `--serial` 2.8–4.4 s, `--simple` 71–81 s. Test acc CART 79.0%, C4.5 78.6%. |
| [`cluster-higgs`](cluster-higgs/BENCHMARK.md) | 10.5M / UCI's 500,000 | H100 0.15–0.17 s, 3.5–4.0× epyc. `--serial` 10.4–20.3 s, `--simple` 236–258 s. Test acc CART 70.5%, C4.5 63.7%. |
| [`cluster-mnist8m`](cluster-mnist8m/BENCHMARK.md) | 8.1M (784 features, 10 classes) / MNIST's 10,000 | H100 3.6–3.7 s, epyc 5.6–5.9 s. `--serial` 209–315 s, `--simple` 1,875–2,159 s. Test acc CART 84.5%, C4.5 74.7%. |
| [`cluster-ecbdl`](cluster-ecbdl/BENCHMARK.md) | 32.0M (CPU nodes) and the first 15M (every node) / the competition's 2.9M | 15M: H100 2.97–2.99 s, 2.1–2.6× epyc, VRAM 73.3 GiB. Full: epyc 13.5–17.6 s, dell 21.1–27.9 s; `--serial` 503–980 s, `--simple` 5,857–7,089 s. Test acc 98.3% is the majority class's. |
| [`cluster-sorel`](cluster-sorel/BENCHMARK.md) | 12.7M (2381 features; CPU nodes) and the first 4M (every node) / 4.2M | Full: CART epyc 21.8 s, dell 32.5 s; C4.5 epyc 28.8 s, dell 44.3 s; `--serial` 878–1,825 s, `--simple` 12,501–13,355 s. 4M: H100 2.9–4.3 s, 1.8–2.2× epyc. Test acc CART 92.7%, C4.5 78.1%. |
| [`cluster-chess`](cluster-chess/BENCHMARK.md) | a 3×3 checkerboard (2 features) at 1M, 10M, 100M and 1000M rows | The same 17-node tree everywhere. epyc fastest at 1M, the A100 at 10M, the H100 from 100M on: 0.64–0.70 s at 1000M, 6.9–7.8× epyc. `--serial` 99–109 s and `--simple` 912–1,068 s at 1000M. |

**./tree on its own:**

| Directory | What | Main result (ec77727, 2026-10) |
|---|---|---|
| [`cluster-cpu-scaling`](cluster-cpu-scaling/BENCHMARK.md) | HIGGS and MNIST8M, 1–48 threads on epyc and dell; chart: speed-up over one thread | HIGGS, 48 threads: 17–21× one thread. MNIST8M: 37–40× (epyc), 31–34× (dell). |
| [`cluster-profiling`](cluster-profiling/BENCHMARK.md) | where the growing time goes (`--profile`), chess 10M–2000M, HIGGS, airline and ECBDL'14 15M, epyc1, dell1 and hopper (ec77727; not rerun) | Partition is 31–74% on epyc; the GPU's share grows from 5% at chess 10M to 72–76% at 2000M and 91–94% on ECBDL. |
| [`hybrid-vs-gpu-only`](hybrid-vs-gpu-only/BENCHMARK.md) | CUDA hybrid against `--gpu-only`, depth 5–30 and no limit, HIGGS, MNIST8M and SOREL 4M, ampere and hopper (95b6d77, 2026-11-06 to 08; not rerun) | Equal to depth 10. Deeper the hybrid wins: HIGGS 4.5×/3.1× at depth 30 and 4.9×/4.3× with no limit on ampere. |

**Against other implementations** (HIGGS and MNIST8M; ./tree's bars from the test-set benchmarks):

| Directory | What | Main result (2026-10, HIGGS) |
|---|---|---|
| [`cluster-cpu-vs-others`](cluster-cpu-vs-others/BENCHMARK.md) | ./tree `--simple`, `--serial`, `--parallel` on epyc against scikit-learn (CART) and Weka J48 (C4.5), one thread | `--parallel` 350× scikit-learn and 1,400× J48; the same trees. |
| [`cluster-gpu-vs-others`](cluster-gpu-vs-others/BENCHMARK.md) | ./tree CUDA hybrid and GPU only against cuML (Gini; entropy for C4.5) and XGBoost (Gini-equivalent, CART only), one tree and a default forest of 10, H100 and A100 | One tree: ./tree 2–3× faster than cuML and 19–23× than XGBoost; cuML's default forest is 1.5–2.2× faster per tree. |

## A new node or machine

- **Another mordor3 node:** add a line to the job table in `bench/cluster/submit.sh` (run name, partition, node, implementation, threads) and name the run after it, e.g. `cpu-<node>-<dataset>`. [bench/cluster/README.md](../cluster/README.md) has the settings each node needs.
- **Elsewhere:** run `bench/setup.sh` once, prepare the data (`bench/.venv/bin/python bench/prepare.py <dataset>`), then `bench/run.py <dataset> --protocols <…> --impls tree|tree_cuda --threads all --cpus <list> --reps 3 --run-id <name>`. The first CPU of `--cpus` runs the one-thread cases and should be a fast core; `tree_cuda` runs `./tree --cuda` (built by `make tree`) and also records the peak GPU memory.
