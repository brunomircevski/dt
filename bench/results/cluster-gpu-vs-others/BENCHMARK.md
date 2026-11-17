# GPU vs others: ./tree against cuML and XGBoost on HIGGS and MNIST8M (CART)

One tree per run, grown to depth ≤ 10 without pruning on every training row and feature, scored on the dataset's test set, on the H100 PCIe of hopper and the A100 80 GB PCIe of ampere: one GPU and every CPU of the node, as ./tree's CUDA runs. cuML and XGBoost also grow a forest of 10 trees with their own random-forest defaults; its time divided by 10 is the bar "forest of 10 / 10".

**Final data (2026-11-17, one timing rule for every GPU tool):** ./tree at 713914b8 (`final.sh -D higgs,susy,mnist8m -k gpu`, jobs 298 hopper and 301 ampere; its bars are the test-set benchmarks' `cluster-<data>/{cuda,gpuonly}-<node>-<data>` runs); cuML and XGBoost at 4e9bf9ab (jobs 303 hopper, 304 ampere, `logs/final-<node>.log`; submitted by hand after jobs 299/302 failed on a results folder renamed under them, so their rows got their commit afterwards, the harness unchanged). 10 runs a case, every one a fresh process with no warm-up. Earlier runs (warm-up fits, ./tree's untimed start-up making its pool and buffers, an A100 40 GB on 2026-11-17 morning) are in git history.

![./tree against cuML and XGBoost, HIGGS and MNIST8M, CART and C4.5, H100 and A100](chart.png)

```bash
bench/cluster/final.sh            # queues the libraries on hopper and ampere after ./tree's runs
# then copy ~/gpu-dt-bench/results-final/<dataset>/<node>/ to <dataset>/<node>/ here, and:
bench/cluster/charts.sh   # among them vs_chart.py gpu (chart.png, summary.csv) and gpu-warm (chart-warm.png)
```

## Results

Mean of 10 runs, H100 (A100 80 GB in brackets); the chart has every bar with its spread, memory, test accuracy and nodes.

| | ./tree hybrid | ./tree GPU only | cuML, 1 tree | XGBoost, 1 tree | ./tree (GPU only) faster than cuML / XGBoost |
|---|--:|--:|--:|--:|--:|
| HIGGS CART | 0.326 s (0.392) | 0.221 s (0.247) | 1.43 s (1.67) | 2.87 s (2.81) | 6.5× / 13× (6.8× / 11×) |
| MNIST8M CART | 2.38 s (3.36) | 2.26 s (3.18) | 18.0 s (25.0) | 27.4 s (28.5) | 8.0× / 12× (7.9× / 8.9×) |

Warm variant ([chart-warm.png](chart-warm.png)), the second fit of a process:

| | ./tree hybrid | ./tree GPU only | cuML, 1 tree | XGBoost, 1 tree | best ./tree faster than cuML / XGBoost |
|---|--:|--:|--:|--:|--:|
| HIGGS CART | 0.097 s (0.208) | 0.101 s (0.215) | 0.451 s (0.507) | 2.37 s (2.61) | 4.6× / 24× (2.4× / 13×) |
| MNIST8M CART | 2.29 s (4.29) | 2.53 s (4.46) | 7.96 s (10.8) | 26.6 s (28.7) | 3.5× / 12× (2.5× / 6.7×) |

- **./tree --gpu is faster than --hybrid cold** (HIGGS 0.221 against 0.326 s on the H100): the hybrid's timed setup also makes its 256 MB of page-locked hand-off buffers (about 0.1 s). Warm, with that done, the two are within 3–10%. --gpu needs a third of the RAM (MNIST8M 26 against 77 GB).
- **cuML's first fit of a process is slow:** its cold single tree takes 18.0 s on MNIST8M (H100), its warm one 7.96 s, the time the runs before 2026-11-17 reported after a warm-up fit; on HIGGS 1.43 against 0.451 s. Warm, ./tree's lead over cuML is 2.4–4.6×; cold, 6.5–8.0×.
- **./tree on MNIST8M is not faster warm:** each fit allocates its GPU buffers (about 53 GB) and uploads the data inside its timed setup, and the second fit's setup is slower than the first's (A100: 2.4–2.6 s against 1.3–1.5 s; the build itself takes the same 1.9 s). On HIGGS, where the buffers are 3.6 GB, warm is 1.1–3.3× faster than cold.
- **XGBoost's time is mostly `QuantileDMatrix`:** the binning takes 2.5 s of 2.8–2.9 s on HIGGS and 20–21 s of 27–28 s on MNIST8M; the tree itself 0.2–0.3 s and 6–7 s. Warm saves it little (HIGGS 2.37 against 2.87 s). Its one tree is ./tree's CART tree on MNIST8M (2,047 nodes, test 84.51%) and nearly on HIGGS (2,033 against 2,029 nodes, test 70.51% against 70.52%).
- **XGBoost needs 79 GB of VRAM on MNIST8M:** the multi-output tree's histograms (10 classes × 784 features × 256 bins per node). Its default cache of up to 65,536 nodes' histograms does not fit in 80 GB, nor does 256; `gpubench.py` keeps 128 (`max_cached_hist_node`, the same tree). ./tree uses 53 GB, cuML 27 GB.
- **A forest's tree is cheaper than one full tree:** cuML's default forest takes 0.195 s (HIGGS) and 1.50 s (MNIST8M) per tree on the H100, as fast as or faster than ./tree's one tree, because each tree sees a bootstrap sample and √features per split, and the first fit's one-time cost is spread over 10 trees; its trees are not the full-data tree (it is a forest, test 90.4% against one tree's 84.5% on MNIST8M).
- **H100 against A100:** ./tree 1.1–1.2× faster on the H100 on HIGGS and 1.4× on MNIST8M, cuML 1.2–1.4×, XGBoost 0.98–1.04× (its binning runs mostly on the CPUs).

## What is compared

**CART only.** No public CUDA implementation of C4.5 was found (the GPU C4.5 papers publish no code), and an entropy (information-gain) tree is not C4.5, so ./tree's C4.5 has no GPU competitor and no library runs C4.5 here. The cuML entropy runs that stood in for C4.5 until 2026-11-16 were removed from the data and the harness. Every GPU tree library still in use builds binary Gini, entropy or second-order-gain trees on histogram bins: it splits only at the edges of a fixed number of quantile bins per feature (cuML 128, XGBoost 256). ./tree tests every distinct value.

| Implementation | Version | Settings |
|---|---|---|
| **./tree --hybrid** and **--gpu** | the build of the test-set benchmarks | `--cart -d 10` (exact Gini), `--c45 -d 10` (C4.5 gain ratio), no pruning; its bars are `cluster-<data>/cuda-<node>-<data>` and `gpuonly-<node>-<data>` |
| **cuML** `RandomForestClassifier` | 26.10 | 1 tree: `n_estimators=1, bootstrap=False, max_features=1.0, n_bins=128, max_depth=10`, Gini |
| **XGBoost** | 3.4.1, `device="cuda", tree_method="hist"` | `reg:squarederror` with `eta=1, lambda=0`, 256 bins, `max_depth=10`, one round: on 0/1 labels the gain is n/2 × the drop in Gini impurity, so the CART split on bins. MNIST8M (10 classes): one multi-output tree (`multi_strategy="multi_output_tree"`) on the one-hot labels, whose summed squared error is again n/2 × the Gini drop. |

- **Forest of 10 / 10** (`gpubench.py FOREST=default`): each library's own random-forest defaults, only the criterion and `max_depth=10` set. cuML: bootstrap samples, `max_features="sqrt"`, 128 bins, `n_streams=4`. XGBoost (as `XGBRFClassifier`): `binary:logistic` (MNIST8M: `multi:softprob` with multi-output trees), `subsample=0.8`, `colsample_bynode=0.8`, `lambda=1e-5`, `eta=1`, one round of 10 parallel trees. A forest is not one tree: the comparison is per unit of work, not of result.
- **Time:** from host memory to the finished tree. Every GPU tool by one rule (since 2026-11-17): only the bare CUDA context before the timer. ./tree: its `train total` (GPU setup with kernel loading, memory pool, page-locked staging buffers and copy threads; upload, presort, build), a fresh process per run; the context is made while the input loads and is not in it (reported as `gpu startup`, see [bench/README.md](../../README.md#how-time-is-measured)). Until 2026-11-17 ./tree's untimed start-up also loaded its kernels and made its pool and staging buffers (0.17-0.28 s more than the context on HIGGS); those runs are in git history. Libraries (since 2026-11-16): one fresh Python process per run, which loads the float32 NumPy arrays, does the start-up (import; the CUDA context alone with `cudaFree(0)`, `CUDA_MODULE_LOADING=LAZY`, CUDA's default; RMM left at cuML's default, not pre-grown), recorded as `startup_seconds` (`import_seconds` + `gpu_startup_seconds`) and not timed, then exactly one timed fit with no warm-up before it: cuML `fit`, XGBoost `QuantileDMatrix` + `train` (`dmatrix_seconds` and `build_seconds` apart). A fit includes the copy to the GPU, the binning and every allocation it makes. Evaluation (accuracy, predictions, the tree walk) runs after the timer. 10 runs a case: 10 rounds of one process per case (library, dataset, 1 tree or the forest of 10) in a new shuffled order each round ([`scripts/job.sh`](scripts/job.sh)), each under the rusage launcher (`wall_seconds`, peak RSS). Kernel loading: CUDA_MODULE_LOADING=LAZY, CUDA's default (job.sh, run.py): the libraries link their own CUDA runtime, so cudaFree(0) does not load their kernels and EAGER would load all of them at their first call, inside fit (HIGGS, A100, 2026-11-17, 6 runs each: cuML fit 2.19 s EAGER vs 1.71 s LAZY, XGBoost 3.11 vs 2.88 s); with LAZY the kernels a fit uses still load inside it. ./tree too loads its kernels lazily, inside its timed setup (`--gpu-startup context`). XGBoost's float32 (MNIST8M: one-hot) labels are made before the timer, as the adapter's data preparation (since 2026-11-17; before, inside it). Every process runs under `taskset` with ./tree's CPU list (the GPU's NUMA node first). Until 2026-11-16 one process per case made an untimed warm-up fit (cuML on every row, XGBoost on 50,000 rows), then the timed fits.
- **Warm variant** (`chart-warm.png`, `summary-warm.csv`; since 2026-11-17): one CART tree, timed as a process's second identical fit, after one untimed fit on the same data, for every tool: ./tree `--warm-up` (`bench/run.py --warm-up`, job.sh `WARM_UP=1`; runs in [warm/](warm/)), gpubench.py `WARM=1` (job.sh `WARM_FORESTS=1:identical`, rows with `"warm": true`, the untimed fit's time as `warmup_seconds`). 10 runs a case, interleaved with the cold cases. The cold protocol above is the comparison; the warm one shows the one-time costs of a first fit (cuML's first copy of a NumPy input: MNIST8M on the H100 11.2 s, the next copy 2.1 s).
- **GPU:** one fixed 80 GB GPU per node, by rule: job.sh takes the node's GPU (hopper GPU 2, ampere GPU 0) and stops on any other or below 80 GB (`GPU_MIN_MIB`). hopper: GPU 2, H100 PCIe 80 GB; ampere: GPU 0, A100 80 GB PCIe (ampere's GPUs 2-3 are A100 40 GB; on 2026-11-17 free memory alone picked one of them, so that day's ampere runs were redone).
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
| `summary.csv` | One row per dataset, algorithm, implementation and GPU: mean, standard deviation, runs, min, max, VRAM, test accuracy, nodes, out of date |
| `chart-warm.png`, `summary-warm.csv` | the warm variant (CART, one tree), each bar with its cold mean |
| `<dataset>/<node>/` | `results.jsonl` (every cuML and XGBoost fit, cold and warm), `machine.json` |
| `warm/` | ./tree's warm runs (`--warm-up`): `{cuda,gpuonly}-<node>-<dataset>/` |
| `logs/` | the SLURM logs: `final-<node>.log` the current runs; `<node>-trees<1|10>[-first].log` the runs before 2026-11-16 |
| `scripts/` | the harness (above) |
