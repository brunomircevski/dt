# GPU vs others: ./tree against cuML and XGBoost on HIGGS and MNIST8M (CART)

One CART tree per run, depth ≤ 10, no pruning, on every training row and feature, on hopper's H100 PCIe 80 GB (GPU 2) with every CPU of the node. ./tree `--gpu` and `--hybrid`, cuML and XGBoost run in one `bench/run.py` per dataset, their runs interleaved in a new shuffled order each repetition. Every run is a fresh process that builds exactly one tree with no warm-up. Every tool uploads its data through the same code. Every run records a timeline: when each section of its timed training started and ended.

**Data (2026-11-18, job 395, 3 runs a case, ./tree at d123383a):** the protocol below. A 10-run rerun is `final.sh -k gpulibs`. The earlier protocol (each library uploading in its own way, a warm variant, forests of 10, the A100 node) is in git history up to 8ba60fae.

![One CART tree on the H100: when each section runs, ./tree, cuML and XGBoost, HIGGS and MNIST8M](chart.png)

```bash
bench/cluster/final.sh -k gpulibs   # queues the runs on hopper (needs ~/gpu-dt-bench/venv and bench/.tools/libuploader.so)
bench/cluster/charts.sh             # among them gpu_steps_chart.py: chart.png, summary.csv
```

## Results

Mean of 3 runs on the H100. Each recorded section from its start to its end, in milliseconds since the timer started; sections that overlap ran at the same time. `summary.csv` has every section's start, end and length.

**The chart draws the same bars for every tool:** set-up, upload, prepare, build; `--hybrid` also its hand-off buffers apart (they are the whole difference between `--hybrid` and `--gpu`). ./tree's finer sections are merged into them: set-up = staging buffers + device memory (the libraries allocate their device memory inside their other bars, where it cannot be timed from outside); prepare = the presort + the x·log x table and its copy. `--hybrid` also gets a CPU build bar: its CPU subtree tasks, from the first one's start to the last one's end, during its build. A bar made of pieces that do not touch (`--hybrid`'s set-up, before and after its hand-off buffers) is drawn in pieces, labelled with their sum.

| HIGGS | ./tree --gpu | ./tree --hybrid | cuML 26.10 | XGBoost 3.4.1 |
|---|--:|--:|--:|--:|
| staging buffers, copy threads | 0–61 | 0–58 | 0–58 | 0–61 |
| hand-off buffers | | 58–156 | | |
| device memory | 61–74 | 156–170 | | |
| upload | 74–120 | 170–214 | 58–123 | 61–106 |
| presort (GPU kernels) | 83–121 | 179–215 | | |
| x·log x table, its copy | 120–122, 122–154 | 214–216, 216–249 | | |
| input checks / quantile binning | | | 123–300 | 106–248 |
| build the tree | 154–220 | 250–316 | 300–602 | 248–512 |
| CPU subtrees | | 300–315 (23 ms of work, 52 tasks) | | |
| **total** | **219** | 315 | 602 (2.8×) | 512 (2.3×) |
| test accuracy, nodes, VRAM | 70.52%, 2,029, 3.3 GiB | 70.52%, 2,029, 3.3 GiB | 70.48%, 2,039, 1.9 GiB | 70.53%, 2,031, 2.7 GiB |

| MNIST8M | ./tree --gpu | ./tree --hybrid | cuML 26.10 | XGBoost 3.4.1 |
|---|--:|--:|--:|--:|
| staging buffers, copy threads | 0–61 | 0–62 | 0–52 | 0–50 |
| hand-off buffers | | 62–170 | | |
| device memory | 61–82 | 170–196 | | |
| upload | 82–794 | 196–906 | 52–610 | 50–594 |
| presort (GPU kernels) | 89–795 | 203–907 | | |
| x·log x table, its copy | 794–796, 796–823 | 906–908, 908–935 | | |
| input checks / quantile binning | | | 610–840 | 594–2,939 |
| build the tree | 824–2,245 | 936–2,358 | 840–5,405 | 2,939–7,389 |
| CPU subtrees | | 1,927–2,303 (641 ms of work, 136 tasks) | | |
| **total** | **2,244** | 2,357 | 5,414 (2.4×) | 7,392 (3.3×) |
| test accuracy, nodes, VRAM | 84.51%, 2,047, 49 GiB | 84.51%, 2,047, 49 GiB | 84.55%, 2,047, 25 GiB | 84.51%, 2,047, 67 GiB |

(×: slower than ./tree --gpu.)

- **The build is where ./tree wins:** 65 ms against cuML's 302 ms (4.6×) and XGBoost's 264 ms (4.0×) on HIGGS; 1.42 s against 4.56 s (3.2×) and 4.45 s (3.1×) on MNIST8M. The upload is the same code for every tool (about 47 GB/s for the libraries on MNIST8M; HIGGS's 1.2 GB is too short a copy to reach full speed).
- **The presort hides inside the upload:** its first kernel starts 7–9 ms after the upload starts (the first feature has to arrive), and its last one ends within 1 ms of the upload's end. On MNIST8M ./tree's upload takes 0.71 s against the libraries' 0.55 s for the same 25.4 GB: the presort kernels running at the same time slow the copies. Without the overlap the 0.7 s of presort would come after the upload.
- **The x·log x table's copy costs 27–33 ms** (15% of HIGGS `--gpu`), computing it 1.6–1.8 ms: a 512 KB `cudaMemcpyAsync` from pageable memory, which the host waits for. A copy through the staging buffers, or computing the table on the GPU, would remove most of it (not done yet).
- **--hybrid against --gpu:** its 256 MB of page-locked hand-off buffers take 98–109 ms, the whole difference between the two. Its CPU subtrees run during the last levels (HIGGS: for 15 ms; MNIST8M: for 376 ms) and end with the build, which takes as long as `--gpu`'s: at depth 10 they gain nothing on these datasets.
- **cuML's input checks (176–230 ms)** are its NaN/inf scan, the classes and the lazy loading of the kernels it uses. Its binning is inside its build: `nsys` shows that cuML computes its 128 quantiles from a row sample in about 0.2 ms of GPU time, and 98% of its build's GPU time is `buildHistogramsKernel`. Not split: a 0.2 ms section says nothing.
- **XGBoost's quantile binning** (`QuantileDMatrix`, then freeing the raw device copy, 0.5–8 ms, drawn as part of it): 141 ms on HIGGS, 2.34 s on MNIST8M (784 features).
- **The same tree:** XGBoost's tree on MNIST8M is ./tree's tree (2,047 nodes, the same test accuracy). On HIGGS the libraries' trees differ in a few nodes because of their bins (128 and 256 quantiles; ./tree tests every distinct value). Test accuracy is within 0.05 points.
- **VRAM:** ./tree keeps presorted 8-byte entries for every value (49 GiB on MNIST8M). cuML keeps its bins and needs half as much. XGBoost needs the most (67 GiB) for its multi-output histograms (10 classes × 784 features × 256 bins per node), even with its histogram cache limited to 128 nodes.

## The sections

| Section | ./tree (its `span` lines, `src/core/timeline.h`) | cuML, XGBoost (`scripts/gpubench.py`) |
|---|---|---|
| staging buffers, copy threads | `Staging::prepare` | `uploader.cpp`'s buffers and copy threads |
| hand-off buffers | `--hybrid`: `prepareHandOff` (2 × 128 MB page-locked) | |
| device memory | streams, events, the memory pool, `allocateDevice` | (inside the libraries' later sections) |
| upload | labels and columns through the staging buffers, to the last copy's end | the same, through `uploader.cpp` |
| presort (GPU kernels) | CUDA events: the first presort kernel's start to the last one's end, on the host's clock (an event synchronized before the upload) | |
| x·log x table, its copy | `xlog2xTable()`, then its copy to the GPU | |
| input checks | | cuML `_prepare_fit_inputs` |
| quantile binning | | XGBoost `QuantileDMatrix(max_bin=256)`, then the raw copy freed |
| build the tree | `grow()`: all levels, the hand-offs, waiting for the CPU tasks, the host tree | cuML `_fit_forest` and the rest of `fit`; XGBoost `xgb.train`, one round |
| CPU subtrees | `--hybrid`: the first CPU task's start to the last one's end, with the tasks' summed work | |

Sections of ./tree under 2 ms (the wait for the presort's end, the host arena, the grow's set-up, assembling the tree) are not recorded on their own; they are in the gaps or in the build. Every library section ends with `cudaDeviceSynchronize`, so its GPU work is inside it. ./tree's host sections wait for their own GPU work where the code does (the upload ends with the last copy).

## What is compared

**CART only.** No public GPU implementation of C4.5 exists, and an entropy (information-gain) tree is not C4.5, so ./tree's C4.5 has no GPU competitor here. Both libraries split only at the edges of a fixed number of quantile bins per feature; ./tree tests every distinct value.

| Implementation | Version | Settings |
|---|---|---|
| **./tree --gpu** and **--hybrid** | the commit under test | `--cart -d 10` (exact Gini), no pruning, every CPU of the node (`--hybrid` finishes small subtrees on them) |
| **cuML** `RandomForestClassifier` | 26.10 | `n_estimators=1, bootstrap=False, max_samples=1.0, max_features=1.0, n_bins=128, max_depth=10, split_criterion="gini", n_streams=1` |
| **XGBoost** | 3.4.1, `device="cuda", tree_method="hist"` | `reg:squarederror`, `eta=1, lambda=0, alpha=0, gamma=0, min_child_weight=1`, `max_bin=256`, `max_depth=10`, `grow_policy="depthwise"`, one round. On 0/1 labels the gain is n/2 × the drop in Gini impurity, so this is CART on the bins. MNIST8M (10 classes): one multi-output tree (`multi_strategy="multi_output_tree"`, `max_cached_hist_node=128`) on one-hot labels, whose summed squared error is again n/2 × the Gini drop. |

## The protocol

The same for every tool:

- **Not timed:** reading the files into host RAM as float32, column-major (./tree's loader holds its data column-major; the libraries get `np.asfortranarray`), and the labels (XGBoost's float32 or one-hot labels are made here); importing the library; the CUDA context alone (`cudaFree(0)`; ./tree makes it on a thread while the input loads). `CUDA_MODULE_LOADING=LAZY`, CUDA's default: each tool loads the kernels it uses inside the timed part.
- **Timed:** from the host arrays to the finished tree, in the sections above. ./tree's `train total` and the libraries' `train_seconds` are the totals; run.py also keeps the step sums (./tree's `setup_buffers/allocate/upload/finish_seconds`, `build_seconds`; gpubench's `buffers/upload/prepare/build_seconds`) next to each row's `timeline`.
- **The upload:** 4 page-locked 32 MB buffers on the GPU's NUMA node, up to 32 copy threads pinned to that node's CPUs, each 32 MB chunk copied in parallel into a free buffer, then an asynchronous copy to the GPU. cuML and XGBoost call the same algorithm through `ctypes` (`bench/.tools/libuploader.so`, built by `bench/cluster/build.sh`), so neither library is slowed by its own pageable copy and ./tree gets no advantage from its upload.
- **Not timed, after the tree:** prediction, accuracy, the tree walk (nodes, depth, root split).
- **Runs:** one `bench/run.py` per dataset with `--impls tree_gpu_only,tree_cuda,cuml,xgboost`; each repetition runs every tool once in a new shuffled order. Every process runs on all CPUs of the node, the GPU's NUMA node first; the job owns the node (`--exclusive`).
- **VRAM:** peak GPU memory in use above idle, sampled from outside every 20 ms with `nvidia-smi`, for the whole process, the same for every tool.
- **GPU:** hopper's GPU 2 (H100 PCIe 80 GB), by rule (`bench/cluster/job.sh`); a run on another GPU is marked out of date.
- **Test sets:** HIGGS the last 500,000 rows (UCI's split), MNIST8M the original MNIST test set (10,000).

## Run it by hand

cuML needs its own venv (`GPU_PYTHON`, default `~/gpu-dt-bench/venv/bin/python`; the pinned versions are in [`scripts/requirements-freeze.txt`](scripts/requirements-freeze.txt)):

```bash
# on the mordor3 login node, once
/usr/bin/virtualenv -p python3 ~/gpu-dt-bench/venv       # python3-venv is missing on mordor3
~/gpu-dt-bench/venv/bin/pip install -r bench/results/cluster-gpu-vs-others/scripts/requirements-freeze.txt \
  --extra-index-url=https://pypi.nvidia.com
# a plan line for bench/cluster/batch.sh (final.sh writes these)
- cluster-gpu-vs-others/one-tree vs-hopper-higgs higgs tree_gpu_only,tree_cuda,cuml,xgboost 10 cart_depth10 all 14400 ACCURACY=--test-accuracy
```

## Files

| File | Content |
|---|---|
| `chart.png` | HIGGS and MNIST8M: each tool's bars (set-up, hand-off buffers, upload, prepare, build, CPU build) on a timeline, with total, test accuracy, nodes, VRAM and spread (`bench/gpu_steps_chart.py`) |
| `summary.csv` | one row per dataset, tool and recorded section (with the bar it is drawn in): mean start, end and length (CPU subtrees: work and tasks), and the tool's total (mean, standard deviation, min, max), VRAM, test accuracy, nodes, commit |
| `one-tree/vs-hopper-<dataset>/` | the run folders: `results.jsonl` (every run of every tool, each step's time), `plan.json`, `machine.json`, `run.log`, and `report.md`, `runs.csv`, `summary.csv` from `bench/report.py` |
| `scripts/gpubench.py` | one cuML or XGBoost run (called by `bench/run.py`) |
| `scripts/requirements-freeze.txt` | the venv's pinned versions |
