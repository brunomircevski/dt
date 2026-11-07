# Benchmark results

Every benchmark here ran on the faculty's mordor3 cluster. Each is one directory, `cluster-<what>/`, whose `BENCHMARK.md` gives the method, the results, the commands that ran it and the ones that drew its charts (`chart*.png`). What the probes around them found (noise, placement, depth) is in [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md). Older benchmarks (desktop and laptop, other implementations on full SUSY, pruned trees) are in the git history, last at 254d4e8 and 96705a7.

Inside a benchmark, one directory per run: `<backend>-<node>-<dataset>/`.

- **backend:** `cuda` (`--cuda`, the CPU takes the small nodes), `gpuonly` (`--cuda --gpu-only`), `cpu` (parallel, both sockets), `socket` (parallel, one socket's CPUs and memory), `serial` (`--serial`, one thread), `simple` (`--simple`, the one-thread reference backend).
- **node:** `hopper` (H100 PCIe, 2× Xeon Gold 6444Y) and `ampere` (A100 80 GB PCIe, 2× EPYC 7402) run CUDA only; `epyc` (epyc1, 2× EPYC 9255), `dell` (dell1, 2× Xeon Gold 6542Y) and `dell512` (dell2, the same CPUs with 512 GB) run on the CPU only.
- **dataset:** the `bench/datasets.toml` entry, e.g. `higgs`, `ecbdl_15m`, `chess_1b`.

A run directory holds `results.jsonl` (one row per run of a case), `plan.json` (the cases and the dataset's rows, features and split), `machine.json` (hardware, commit, date), `run.log`, and `report.md`, `runs.csv`, `summary.csv` from `bench/report.py`. Two benchmarks differ: `cluster-cpu-vs-others` names its runs `vs-epyc-<dataset>-<backend or tool>-<algorithm>`, and `cluster-gpu-vs-others` keeps one directory per dataset and node (cuML and XGBoost) with the scripts it ran in `scripts/`; it ran outside this repository (`~/gpu-dt-bench`), and takes ./tree's runs from the test-set benchmarks.

The runs are SLURM jobs: `bench/cluster/submit.sh` queues one job per node, and each runs `bench/run.py --run-id cluster-<what>/<run>`, which writes into `bench/results/<run-id>/`. Setup, data and every submit command are in [bench/cluster/README.md](../cluster/README.md).

## The benchmarks

Times are ./tree's training time (presort, build, pruning; not reading the data). Trees are grown to depth ≤ 10 without pruning, on one build of main (ec77727, 2026-10-29/30) unless a row says otherwise. Depth 10 underfits and holds C4.5 back most ([depth sweep](../../docs/PERFORMANCE.md#october-2026-what-the-benchmarks-showed-mordor3-cluster)).

**Against other implementations:**

| Directory | What | Main result |
|---|---|---|
| [`cluster-cpu-vs-others`](cluster-cpu-vs-others/BENCHMARK.md) | chess 10M and 1000M, Covertype, SUSY, HIGGS and the ECBDL'14 15M sample on epyc1 and epyc2 (same hardware): ./tree's `--simple` reference backend against `--serial`, `--parallel` on one socket and on both, scikit-learn and Weka J48, every tree grown to depth ≤ 10 without pruning, test accuracy; ./tree on ec77727 (2026-10-29/30), at least 3 runs where a run takes under 10 minutes | `--serial` is 7–25× faster than `--simple`, `--parallel` on one socket 5–19× faster than `--serial`. On one socket ./tree is 48–308× faster than scikit-learn and 350–1,680× faster than J48. scikit-learn's trees match ./tree's on every dataset but ECBDL's accuracy; J48's nearly; J48 ran out of heap on chess 1000M. |

**Trained on the training set, scored on the test set** (every run again on 2026-10-29/30 on one build of main, ec77727, with the CPU presort of 4fb419c, except `--simple`, which was not rerun then (SOREL's `--simple` was rerun on 32cf657 on 2026-11-02, and `sorel_4m` ran on 4ece22f); [bench/cluster/README.md](../cluster/README.md#which-build-ran)): ./tree alone, depth ≤ 10, CART and C4.5 without pruning (the pruned runs are in git at 96705a7), CUDA hybrid and GPU only on hopper and ampere, and on epyc and dell parallel on both sockets and on one, `--serial` and `--simple`. 10 runs a case of about 2 minutes or less, 3 under about 10 minutes, longer ones once; the charts and tables give means ± standard deviation ([bench/cluster/README.md](../cluster/README.md#test-set-benchmarks)).

| Directory | Training / test rows | Main result |
|---|---|---|
| [`cluster-airline`](cluster-airline/BENCHMARK.md) | 1987–2007 (109.3M) / 2008 (5.8M) | H100 0.49–0.51 s, 5.8–7.6× epyc (2.87–3.83 s); dell 1.5× epyc. `--serial` 35–73 s, `--simple` 770–849 s. Test acc 66.2–66.6%. |
| [`cluster-higgs`](cluster-higgs/BENCHMARK.md) | 10.5M / UCI's 500,000 | H100 0.15–0.17 s, 3.5–4.0× epyc. `--serial` 10.4–20.3 s, `--simple` 236–258 s. Test acc CART 70.5%, C4.5 63.7%. |
| [`cluster-susy`](cluster-susy/BENCHMARK.md) | 4.5M / UCI's 500,000 | Both GPUs 82–119 ms, 2.2–2.9× epyc; one socket 1.1–1.25× two. `--serial` 2.8–4.4 s, `--simple` 71–81 s. Test acc CART 79.0%, C4.5 78.6%. |
| [`cluster-ecbdl`](cluster-ecbdl/BENCHMARK.md) | 32.0M (CPU nodes) and the first 15M (every node) / the competition's 2.9M | 15M: H100 2.97–2.99 s, 2.1–2.6× epyc, VRAM 73.3 GiB. Full: epyc 13.5–17.6 s, dell 21.1–27.9 s, 238 GiB RAM; `--serial` 503–980 s, `--simple` 5,857–7,089 s. Test acc 98.3% is the majority class's. |
| [`cluster-mnist8m`](cluster-mnist8m/BENCHMARK.md) | 8.1M (784 features, 10 classes) / MNIST's 10,000 | H100 3.6–3.7 s, epyc 5.6–5.9 s, one socket 1.6–1.9× both. `--serial` 209–315 s, `--simple` 1,875–2,159 s. Test acc CART 84.5%, C4.5 74.7%. |
| [`cluster-sorel`](cluster-sorel/BENCHMARK.md) | 12.7M (2381 features; CPU nodes) and the first 4M (GPUs and epyc) / 4.2M | Full: CART epyc 21.8 s, dell 32.5 s (343 GiB RAM); C4.5 epyc 28.8 s, dell 44.3 s (352 GiB); `--serial` 878–1,825 s, `--simple` 12,501–13,355 s. 4M: A100 4.8–6.6 s, 1.2–1.4× epyc (6.4–7.9 s), VRAM 76 GiB; H100 2.9–4.3 s, 1.8–2.2× epyc. Test acc CART 92.7%, C4.5 78.1%. |

**./tree on its own:**

| Directory | What | Main result |
|---|---|---|
| [`cluster-cpu-scaling`](cluster-cpu-scaling/BENCHMARK.md) | HIGGS and MNIST8M, depth ≤ 10 without pruning, 1–48 threads on epyc and dell, and one socket (ec77727, 2026-10-29/30, 10 runs per point) | HIGGS, 48 threads: CART 0.61 s (epyc) and 0.82 s (dell), C4.5 0.59 and 0.99 s, 17–21× one thread. MNIST8M, 48 threads: 5.6–5.9 s (epyc, 37–40×) and 9.3–9.4 s (dell, 31–34×). dell's single thread is 1.3–1.7× slower than epyc's. |
| [`cluster-chess`](cluster-chess/BENCHMARK.md) | a 3×3 checkerboard (2 features) at 1M, 10M, 100M and 1000M rows (ec77727, 2026-10-29/30, 30 runs per case where a run is short): CART and C4.5 at depth ≤ 10 without pruning on hopper, ampere (CUDA, and `--gpu-only`), epyc, dell and dell2 (parallel), epyc and dell (one socket, `--serial`, `--simple`) | The same 17-node tree everywhere. epyc is fastest at 1M (21–25 ms), the A100 at 10M (47–49 ms), the H100 from 100M on: 0.64–0.70 s at 1000M, 6.9–7.8× epyc (4.44–5.45 s). dell 1.7–1.9× epyc. `--serial` 99–109 s and `--simple` 912–1,068 s at 1000M. |
| [`cluster-profiling`](cluster-profiling/BENCHMARK.md) | where the growing time goes (`--profile`): CART and C4.5 grown to depth 10 without pruning (ec77727, 2026-10-29/30; 30 runs per chess case, 10 otherwise), chess 10M–2000M, HIGGS, Airline and ECBDL'14 15M, parallel on epyc1 and dell1, CUDA hybrid and `--gpu-only` on hopper; a 4×2 chart split into presort, split search, partition, CPU↔GPU transfer and other, CPU or GPU, with train accuracy, nodes and depth | Partition is 31–74% on epyc; on dell's chess sets split search leads (34–58%). The GPU's share grows from 5% at chess 10M to 72–76% at 2000M and 91–94% on ECBDL; small CUDA runs are fixed cost. The H100 is 6.7–7.7× epyc on chess 2000M, 2.1–2.6× on ECBDL. |
| [`hybrid-vs-gpu-only`](hybrid-vs-gpu-only/BENCHMARK.md) | CUDA hybrid (`--cuda`, at the `--gpu-min-rows` a sweep found fastest) against `--gpu-only`, CART and C4.5 without pruning at depth 5–30 and no limit, airline and MNIST8M, ampere and hopper, 3 runs per case; every CPU of the node from depth 15 (95b6d77, 2026-11-06/07: ampere 48 threads, hopper 32), depth 5 and 10 from the first runs on the GPU's socket (fef9ac0, 2026-11-03/06) | Equal to depth 15 (0.98–1.02×). Deeper the hybrid wins, all in the build: airline 4.9×/3.1× at depth 30 and 7.7×/5.9× with no limit on ampere (CART/C4.5; 45M-node tree), MNIST8M levels off at 1.6–1.9× (CART) from depth 20. hopper the same (airline 6.8×/5.1× at no limit, MNIST8M 1.7–1.9×). The second socket makes the hybrid 1.15–1.37× faster on airline from depth 25 and 1.1–1.2× on MNIST8M CART; tuned `--gpu-min-rows` 1,024–2,048 (airline) and 16,384 (MNIST8M), twice the one-socket values. |

**Against other GPU libraries** (2026-10-27, on GPU 0 of hopper and ampere; ./tree's bars from the test-set benchmarks). No public CUDA C4.5 was found, so C4.5 is compared with cuML's entropy tree, its closest. The harness is in the benchmark's `scripts/`.

| Directory | What | Main result |
|---|---|---|
| [`cluster-gpu-vs-others`](cluster-gpu-vs-others/BENCHMARK.md) | HIGGS, SUSY, airline (115M rows), the ECBDL'14 15M sample (631 features) and chess 3×3 with 1000M rows, 2026-10-27: one tree at depth ≤ 10, no pruning, ./tree CUDA (from the test-set benchmarks) against cuML (`RandomForestClassifier`, 128 bins, Gini or entropy) and XGBoost (Gini-equivalent gain, 256 bins, CART only), each library also as a 10-tree forest with its random-forest defaults (time / 10), with peak VRAM and test accuracy | One tree (./tree at ec77727): ./tree is 1.1–4.3× faster than cuML and 8–31× faster than XGBoost (chess 1000M: 11–14× and 136–168×). Per tree inside a default 10-tree forest cuML is 1.5–2.5× faster than ./tree's single tree on HIGGS and SUSY and 4–6× on ECBDL, 1.1–1.6× slower on airline, 9–13× slower on chess; XGBoost's forest is 1.9–18× slower except on ECBDL on the A100. The identical-tree forests are in git at 8187eb9. CART accuracy within 0.1 point except airline (cuML 1.0–1.3 lower). The depth-19 comparison is in the git history (325039f). |

## A new node or machine

- **Another mordor3 node:** add a line to the job table in `bench/cluster/submit.sh` (run name, partition, node, implementation, threads) and name the run after it, e.g. `cpu-<node>-<dataset>`. [bench/cluster/README.md](../cluster/README.md) has the settings each node needs.
- **Elsewhere:** run `bench/setup.sh` once, prepare the data (`bench/.venv/bin/python bench/prepare.py <dataset>`), then `bench/run.py <dataset> --protocols <…> --impls tree|tree_cuda --threads all --cpus <list> --reps 3 --run-id <name>`. The first CPU of `--cpus` runs the one-thread cases and should be a fast core; `tree_cuda` runs `./tree --cuda` (built by `make tree`) and also records the peak GPU memory.
