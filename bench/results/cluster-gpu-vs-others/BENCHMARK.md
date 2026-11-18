# GPU vs others: ./tree against cuML and XGBoost on HIGGS and MNIST8M (CART)

Training time of one depth-limited CART tree (depth ≤ 10, no pruning, every training row and feature) per fresh process, on hopper's H100 PCIe 80 GB (GPU 2) with every CPU of the node: ./tree `--gpu` and `--hybrid` (exact: every distinct value is a candidate split), cuML (128 quantile bins) and XGBoost (256 bins). The trees are comparable, not identical. The four tools run in one `bench/run.py` per dataset, their runs interleaved in a new shuffled order each repetition; every run is a fresh process that trains exactly one tree with no warm-up, so this is cold, one-tree latency, not the time per tree of repeated training. Every tool uploads its data through the same code, pinned to the GPU's NUMA node. Every run records a timeline: when each phase of its timed training started and ended.

**Data (2026-11-18, job 397, 10 runs a case, ./tree at 7c39d6a3):** the protocol below. Until 7c39d6a3 the libraries' copy threads were not pinned to the GPU's NUMA node (gpubench.py's lookup failed silently; jobs 395 and 396); pinned, their HIGGS upload takes 39 ms instead of 45–65 ms. The earlier protocol (each library uploading in its own way, a warm variant, forests of 10, the A100 node) is in git history up to 8ba60fae.

![One CART tree per fresh process on the H100: each tool's phases, ./tree, cuML and XGBoost, HIGGS and MNIST8M](chart.png)

```bash
bench/cluster/final.sh -k gpulibs   # queues the runs on hopper (needs ~/gpu-dt-bench/venv and bench/.tools/libuploader.so)
bench/cluster/charts.sh             # among them gpu_steps_chart.py: chart.png, summary.csv
```

## Results

**The totals are the comparison.** Mean of 10 runs, in seconds (× slower than `./tree --gpu`; ± standard deviation):

| | ./tree --gpu (exact) | ./tree --hybrid (exact) | cuML 26.10 (128 bins) | XGBoost 3.4.1 (256 bins) |
|---|--:|--:|--:|--:|
| HIGGS | **0.216** ± 0.009 | 0.323 ± 0.014 | 0.548 ± 0.031 (2.5×) | 0.508 ± 0.017 (2.4×) |
| MNIST8M | **2.247** ± 0.017 | 2.372 ± 0.013 | 5.371 ± 0.089 (2.4×) | 7.333 ± 0.116 (3.3×) |
| test accuracy, nodes, VRAM: HIGGS | 70.52%, 2,029, 3.3 GiB | 70.52%, 2,029, 3.3 GiB | 70.48%, 2,039, 1.9 GiB | 70.53%, 2,031, 2.7 GiB |
| MNIST8M | 84.51%, 2,047, 49 GiB | 84.51%, 2,047, 49 GiB | 84.55%, 2,047, 25 GiB | 84.51%, 2,047, 67 GiB |

**The phases explain the totals; they are not equal components.** Each tool's phases have its own boundaries: ./tree allocates all its device memory in its set-up and presorts in its prepare phase, the libraries allocate theirs inside their later phases, and cuML computes its bins inside its build. Each recorded section from its start to its end, in milliseconds since the timer started; sections that overlap ran at the same time (`summary.csv` has every section):

| HIGGS | ./tree --gpu | ./tree --hybrid | cuML | XGBoost |
|---|--:|--:|--:|--:|
| staging buffers, copy threads | 0–60 | 0–59 | 0–58 | 0–60 |
| hand-off buffers | | 59–162 | | |
| device memory | 60–74 | 162–176 | | |
| upload | 74–118 | 176–222 | 58–97 | 60–99 |
| presort (GPU kernels) | 84–118 | 186–223 | | |
| x·log x table, its copy | 118–119, 119–151 | 222–224, 224–257 | | |
| input checks / quantile binning | | | 97–246 | 99–249 |
| build | 151–217 | 258–325 | 246–548 | 249–508 |
| CPU build (subtrees) | | 308–323 | | |

| MNIST8M | ./tree --gpu | ./tree --hybrid | cuML | XGBoost |
|---|--:|--:|--:|--:|
| staging buffers, copy threads | 0–63 | 0–63 | 0–53 | 0–49 |
| hand-off buffers | | 63–169 | | |
| device memory | 63–87 | 169–194 | | |
| upload | 87–795 | 194–921 | 53–610 | 49–549 |
| presort (GPU kernels) | 94–796 | 200–922 | | |
| x·log x table, its copy | 795–797, 797–823 | 921–922, 922–950 | | |
| input checks / quantile binning | | | 610–801 | 549–2,874 |
| build | 824–2,248 | 951–2,373 | 801–5,362 | 2,874–7,329 |
| CPU build (subtrees) | | 1,943–2,318 | | |

**The chart** draws the same phase names for every tool, each with that tool's boundaries: set-up, upload, prepare, build; `--hybrid` also its hand-off buffers (the whole difference between `--hybrid` and `--gpu`) and its CPU build. ./tree's sections are merged into them: set-up = staging buffers + device memory; prepare = the presort + the x·log x table and its copy. A phase in pieces that do not touch (`--hybrid`'s set-up, before and after its hand-off buffers) is drawn in pieces, labelled with their sum.

- **Totals:** `./tree --gpu` trains its exact tree 2.4–2.5× faster than cuML and 2.4–3.3× faster than XGBoost, from the same host data through the same upload. The libraries' upload reaches 46–51 GB/s on MNIST8M (HIGGS's 1.2 GB is too short a copy for full speed).
- **Where the time goes:** most of the libraries' time is in their build (cuML 302 ms and 4.56 s, XGBoost 259 ms and 4.45 s, against ./tree's 65 ms and 1.42 s) and in XGBoost's binning. These phases are not like for like: cuML's build also allocates its working memory, and ./tree's build starts from columns already presorted.
- **The presort hides inside the upload:** its first kernel starts 7–10 ms after the upload starts (the first attribute has to arrive), and its last one ends within 1 ms of the upload's end. On MNIST8M ./tree's upload takes 0.71 s against the libraries' 0.50–0.56 s for the same 25.4 GB: the presort kernels running at the same time slow the copies. Without the overlap the 0.7 s of presort would come after the upload.
- **The x·log x table's copy costs 26–33 ms** (14% of HIGGS `--gpu`), computing it 1.6–1.8 ms: a 512 KB `cudaMemcpyAsync` from pageable memory, which the host waits for. A copy through the staging buffers, or computing the table on the GPU, would remove most of it (not done yet).
- **--hybrid against --gpu:** its 256 MB of page-locked hand-off buffers take 103–106 ms, the whole difference between the two. Its CPU build runs during the last levels (HIGGS for 15 ms, MNIST8M for 376 ms) and ends before the GPU build does: at depth 10 it gains nothing on these datasets.
- **cuML's prepare (148–191 ms)** is its input checks: the NaN/inf scan, the classes and the lazy loading of the kernels it uses. Its binning is inside its build: `nsys` shows that cuML computes its 128 quantiles from a row sample in about 0.2 ms of GPU time, and 98% of its build's GPU time is `buildHistogramsKernel`.
- **XGBoost's prepare is its quantile binning** (`QuantileDMatrix`, then freeing the raw device copy): 150 ms on HIGGS, 2.32 s on MNIST8M (784 attributes).
- **Similar trees, not identical:** XGBoost's tree on MNIST8M is ./tree's tree (2,047 nodes, the same test accuracy); cuML's has the same size. On HIGGS the libraries' trees differ in a few nodes because of their bins (2,039 and 2,031 against 2,029). Test accuracy is within 0.05 points.
- **VRAM:** ./tree keeps presorted 8-byte entries for every value (49 GiB on MNIST8M). cuML keeps its bins and needs half as much. XGBoost needs the most (67 GiB) for its multi-output histograms (10 classes × 784 attributes × 256 bins per node), even with its histogram cache limited to 128 nodes.

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
| CPU build (subtrees) | `--hybrid`: the first CPU task's start to the last one's end | |

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
- **The upload:** 4 page-locked 32 MB buffers on the GPU's NUMA node, up to 32 copy threads pinned to that node's CPUs (16 on hopper; the calling thread copies too), each 32 MB chunk copied in parallel into a free buffer, then an asynchronous copy to the GPU. cuML and XGBoost call the same code through `ctypes` (`bench/.tools/libuploader.so`, built by `bench/cluster/build.sh`; it finds the GPU's node with a copy of ./tree's `gpuNodeCpus()`), so neither library is slowed by its own pageable copy and ./tree gets no advantage from its upload. Each library row records `uploader_cpus` (16), and gpubench.py stops if the copies are unpinned on a machine with several NUMA nodes.
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
| `chart.png` | HIGGS and MNIST8M: each tool's phases (set-up, hand-off buffers, upload, prepare, build, CPU build) on a timeline, with total, test accuracy, nodes, VRAM and spread (`bench/gpu_steps_chart.py`) |
| `summary.csv` | one row per dataset, tool and recorded section (with the bar it is drawn in): mean start, end and length and the tool's total (mean, standard deviation, min, max), VRAM, test accuracy, nodes, commit |
| `one-tree/vs-hopper-<dataset>/` | the run folders: `results.jsonl` (every run of every tool, each step's time), `plan.json`, `machine.json`, `run.log`, and `report.md`, `runs.csv`, `summary.csv` from `bench/report.py` |
| `scripts/gpubench.py` | one cuML or XGBoost run (called by `bench/run.py`) |
| `scripts/requirements-freeze.txt` | the venv's pinned versions |
