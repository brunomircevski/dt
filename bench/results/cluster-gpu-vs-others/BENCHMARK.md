# GPU vs others: ./tree against cuML and XGBoost on HIGGS and MNIST8M (CART)

One tree per run, grown to depth ≤ 10 without pruning on every training row and feature, scored on the dataset's test set, on the H100 PCIe of hopper and the A100 80 GB PCIe of ampere: one GPU and every CPU of the node, as ./tree's CUDA runs. cuML and XGBoost also grow a forest of 10 trees with their own random-forest defaults; its time divided by 10 is the bar "forest of 10 / 10".

**Final data:** ./tree rerun on 2026-11-15/16 (f53347a9, `src/` = d2957ff2; `bench/cluster/final.sh -m 5`, 5 runs per case); cuML and XGBoost from 2026-11-09/10, 10 timed fits per case, not rerun (their code has not changed). The chart gives every bar's runs. XGBoost on MNIST8M first ran out of memory and stalled in `gpubench.py`'s tree walk; it was rerun on 2026-11-10 with the fixes (538fc78; `logs/*-first.log` are the first pass, with every other library run). Until 2026-11-09 this benchmark also had SUSY, airline, ECBDL'14 15M and chess 1000M; those runs are in git at 3354a8a.

![./tree against cuML and XGBoost, HIGGS and MNIST8M, CART and C4.5, H100 and A100](chart.png)

*The chart predates the removal of the cuML entropy runs and still shows them in its C4.5 panels; `vs_chart.py gpu` draws the C4.5 panels with ./tree's bars only. It is redrawn with the rerun.*

```bash
bench/cluster/final.sh            # queues the libraries on hopper and ampere after ./tree's runs
# then copy ~/gpu-dt-bench/results-final/<dataset>/<node>/ to <dataset>/<node>/ here, and:
bench/.venv/bin/python bench/vs_chart.py gpu     # chart.png, summary.csv
```

## Results

Mean of the runs (./tree 5, the libraries 10), H100 (A100 in brackets); the chart has every bar with its spread, runs, memory, test accuracy and nodes.

| | ./tree hybrid | cuML, 1 tree | XGBoost, 1 tree | ./tree faster than cuML / XGBoost |
|---|--:|--:|--:|--:|
| HIGGS CART | 0.131 s (0.175) | 0.450 s (0.507) | 2.62 s (2.72) | 3.4× / 20× (2.9× / 16×) |
| MNIST8M CART | 2.03 s (3.10) | 7.96 s (10.4) | 26.7 s (29.0) | 3.9× / 13× (3.3× / 9.4×) |

- **./tree --gpu** takes the same time as the hybrid (within 2–6%) with a third of its RAM (MNIST8M 24 against 72 GB).
- **Against the runs before (./tree f1fe270):** ./tree is 8–20% faster on HIGGS and 40–48% on MNIST8M, so its lead over cuML grew from 2.0–2.8× to 2.6–3.9×.
- **XGBoost's time is mostly `QuantileDMatrix`:** the binning takes about 2.4 s of 2.6 s on HIGGS and 20–21 s of 27 s on MNIST8M; the tree itself takes 0.2–0.4 s and 6 s. Its one tree is ./tree's CART tree on MNIST8M (2,047 nodes, training 80.77%, test 84.51%) and nearly on HIGGS (2,033 against 2,029 nodes, test 70.51% against 70.52%).
- **XGBoost needs 79 GB of VRAM on MNIST8M:** the multi-output tree's histograms (10 classes × 784 features × 256 bins per node). Its default cache of up to 65,536 nodes' histograms does not fit in 80 GB, nor does 256; `gpubench.py` keeps 128 (`max_cached_hist_node`, the same tree). ./tree uses 53 GB, cuML 27 GB.
- **A forest's tree is cheaper than one full tree:** cuML's default forest takes 0.10 s (HIGGS) and 0.45 s (MNIST8M) per tree on the H100, faster than ./tree's one tree, because each tree sees a bootstrap sample and √features per split; its trees are not the full-data tree (it is a forest, test 90.4% against one tree's 84.5% on MNIST8M).
- **H100 against A100:** ./tree 1.3–1.5× faster on the H100, cuML 1.1–1.3×, XGBoost 1.04–1.09× (its binning runs mostly on the CPUs).

## What is compared

**CART only.** No public CUDA implementation of C4.5 was found (the GPU C4.5 papers publish no code), and an entropy (information-gain) tree is not C4.5, so ./tree's C4.5 has no GPU competitor and no library runs C4.5 here. The cuML entropy runs that stood in for C4.5 until 2026-11-16 were removed from the data and the harness. Every GPU tree library still in use builds binary Gini, entropy or second-order-gain trees on histogram bins: it splits only at the edges of a fixed number of quantile bins per feature (cuML 128, XGBoost 256). ./tree tests every distinct value.

| Implementation | Version | Settings |
|---|---|---|
| **./tree --hybrid** and **--gpu** | the build of the test-set benchmarks | `--cart -d 10` (exact Gini), `--c45 -d 10` (C4.5 gain ratio), no pruning; its bars are `cluster-<data>/cuda-<node>-<data>` and `gpuonly-<node>-<data>` |
| **cuML** `RandomForestClassifier` | 26.10 | 1 tree: `n_estimators=1, bootstrap=False, max_features=1.0, n_bins=128, max_depth=10`, Gini |
| **XGBoost** | 3.4.1, `device="cuda", tree_method="hist"` | `reg:squarederror` with `eta=1, lambda=0`, 256 bins, `max_depth=10`, one round: on 0/1 labels the gain is n/2 × the drop in Gini impurity, so the CART split on bins. MNIST8M (10 classes): one multi-output tree (`multi_strategy="multi_output_tree"`) on the one-hot labels, whose summed squared error is again n/2 × the Gini drop. |

- **Forest of 10 / 10** (`gpubench.py FOREST=default`): each library's own random-forest defaults, only the criterion and `max_depth=10` set. cuML: bootstrap samples, `max_features="sqrt"`, 128 bins, `n_streams=4`. XGBoost (as `XGBRFClassifier`): `binary:logistic` (MNIST8M: `multi:softprob` with multi-output trees), `subsample=0.8`, `colsample_bynode=0.8`, `lambda=1e-5`, `eta=1`, one round of 10 parallel trees. A forest is not one tree: the comparison is per unit of work, not of result.
- **Time:** from host memory to the finished tree. ./tree: its `train total` (GPU setup and upload, presort, build), a fresh process per run; the one-time CUDA context, memory pool and staging buffers are made while the input loads and are not in it (reported as `gpu startup`, see [bench/README.md](../../README.md#how-time-is-measured)). Libraries (since 2026-11-16): one fresh Python process per run, which loads the float32 NumPy arrays, does the start-up (import; the CUDA context alone with `cudaFree(0)`, kernels loaded eagerly with `CUDA_MODULE_LOADING=EAGER`; RMM left at cuML's default, not pre-grown), recorded as `startup_seconds` (`import_seconds` + `gpu_startup_seconds`) and not timed, then exactly one timed fit with no warm-up before it: cuML `fit`, XGBoost `QuantileDMatrix` + `train` (`dmatrix_seconds` and `build_seconds` apart). A fit includes the copy to the GPU, the binning and every allocation it makes. Evaluation (accuracy, predictions, the tree walk) runs after the timer. 10 runs a case: 10 rounds of one process per case (library, dataset, 1 tree or the forest of 10) in a new shuffled order each round ([`scripts/job.sh`](scripts/job.sh)), each under the rusage launcher (`wall_seconds`, peak RSS). Until 2026-11-16 one process per case made an untimed warm-up fit (cuML on every row, XGBoost on 50,000 rows), then the timed fits.
- **VRAM:** peak GPU memory in use above its idle value, sampled with nvidia-smi every 20 ms, for the whole process.
- **Nodes:** each job owns its node (`--exclusive`); the GPU is the first on the NUMA node with the most free memory (hopper GPU 2, ampere GPU 0), nothing is bound (every CPU, memory first-touch), as in ./tree's `bench/cluster/job.sh`.
- **Test sets:** HIGGS the last 500,000 rows (UCI's split), MNIST8M the original MNIST test set (10,000).

## Run it by hand

The harness runs on mordor3 in `~/gpu-dt-bench`, outside this repository (cuML needs its own venv). The scripts are in [`scripts/`](scripts/): `job.sh` (one SLURM job per node: every library, case and dataset), `gpubench.py` (one run: start-up, one timed fit, evaluation unless `EVAL=0`), `build.sh` and `treebench.py` (./tree inside the harness, no longer run by `job.sh`), `requirements-freeze.txt` (pinned versions). The data is the prepared datasets of [bench/cluster](../../cluster/README.md) (`~/datasets/<dataset>`: `train.f32`, `train.y.i32`, `test.*`).

```bash
# on the mordor3 login node, once: the harness and the venv
mkdir ~/gpu-dt-bench && cd ~/gpu-dt-bench && cp ~/dt/bench/results/cluster-gpu-vs-others/scripts/* .
/usr/bin/virtualenv -p python3 venv          # python3-venv is missing on mordor3
venv/bin/pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12 xgboost
# one tree, then the default forests (final.sh submits these)
FORESTS="1:identical 10:default" REPS=10 RUSAGE=~/dt-final/bench/.tools/rusage RESULTS=$PWD/results-final \
  sbatch -p hopper -w hopper --exclusive -t 1-00:00:00 -o results-final/hopper.log job.sh hopper
```

- **Never `scancel` a running `job.sh` started without `timeout --foreground`:** SLURM's kill on mordor3 once left the benchmark running in its own process group. `job.sh` uses `--foreground`.
- **Earlier versions:** the depth-19 comparison and the identical-tree forests are in the git history (325039f, 8187eb9).

## Files

| File | Content |
|---|---|
| `chart.png` | HIGGS and MNIST8M, CART and C4.5: time of one tree per implementation and GPU, with its runs, RAM and VRAM, test accuracy and nodes |
| `summary.csv` | One row per dataset, algorithm, implementation and GPU: mean, standard deviation, runs, min, max, VRAM, test accuracy, nodes |
| `<dataset>/<node>/` | `results.jsonl` (every cuML and XGBoost fit), `machine.json` |
| `logs/` | the SLURM logs: `<node>-trees<1|10>-first.log` every library run but XGBoost on MNIST8M, `<node>-trees<1|10>.log` its rerun |
| `scripts/` | the harness (above) |
