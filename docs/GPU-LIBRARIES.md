# ./tree against GPU tree libraries

How `./tree --cuda` is compared with cuML and XGBoost, what is timed on each
side and where the comparison is not even. Results, settings and the scripts
are in
[bench/results/cluster-gpu-vs-others](../bench/results/cluster-gpu-vs-others/BENCHMARK.md)
(depth ≤ 10, no pruning; libraries 2026-10-27, ./tree on ec77727).

## The result in one sentence

`./tree` builds an **exact** CART or C4.5 tree (every distinct value tested,
byte-identical to the CPU backends) of depth ≤ 10 on one GPU **1.1–4.3×
faster than cuML** (11–14× on chess 1000M) and **8–31× faster than XGBoost**
(136–168× on chess), which split only at 128 / 256 histogram bins, with the
same CART test accuracy within 0.1 point on HIGGS, SUSY and ECBDL (airline:
1.0–1.3 points above cuML, 0.07 below XGBoost). Per tree inside a default
10-tree forest cuML is faster than ./tree's single tree on HIGGS, SUSY and
ECBDL; a forest tree sees a bootstrap sample and a share of the features.

## Why the design is faster

Each feature is presorted once into 8-byte `(value, row | class)` entries, the
tree grows level by level, and every node's rows stay contiguous in every
column ([CUDA.md](CUDA.md)). A level costs about three streaming passes over
`8·n·F` bytes (sweep: read; partition: read and write), ~24 bytes per value,
with no row-index gathers, no binning and no per-node histograms: the cost
grows with rows, not with nodes × bins.

Histogram libraries, as far as their published design goes, bin first and
then build per-node class × bin × feature histograms through row-index lists:
gathered reads, atomics, and a per-node cost that does not shrink with the
node. **This is an inference from their documentation;** an `nsys` profile of
cuML should confirm it before it is stated as fact.

## What is timed

Both sides: float32 training data in host RAM → finished tree. Not timed on
either side: reading the file, prediction, accuracy.

**./tree:** its `train total`, a fresh process per run, `--cuda --cart|--c45
--no-prune -d 10` with the default threads. `train total = gpu setup + presort
+ build + prune` (printed by `src/app/main.cpp`):

| Phase | Contents |
|---|---|
| gpu setup | streams, device allocations, upload of columns and labels through the staging buffers |
| presort | `(value, row \| class)` entries, CUB radix sort per feature (C4.5: distinct values); overlapped with the upload |
| build | all GPU levels, hand-offs, the CPU threads finishing small subtrees, assembling the host tree |
| prune | C4.5 only: collapse of useless splits and training-value thresholds (part of C4.5 even with `--no-prune`) |

Not in it: CSV parsing; the CUDA context and the pinned staging buffers (4 ×
32 MB for uploads, 2 × 128 MB for the hand-offs), both made on a background
thread while the CSV loads (`src/app/main.cpp`).

**cuML / XGBoost** (`gpubench.py`): one Python process per case loads
row-major float32 NumPy arrays, runs an untimed warm-up fit (cuML on all rows,
XGBoost on 50,000), then 3 timed fits (`time.perf_counter`), reported as means:

* cuML `fit(X, y)`: input conversion, host→device copy, quantile binning,
  growing.
* XGBoost: `QuantileDMatrix(X, y)` (copy and quantile sketch) and
  `xgb.train(..., num_boost_round=1)`.
* `predict` and model export are not timed.

VRAM: peak `nvidia-smi` memory in use above idle, for the libraries over the
whole process (warm-up included).

## Fairness

**Matched:** same rows, features and depth limit, no sampling, no pruning,
same GPUs on exclusive nodes, both start from host RAM, both pay their upload
and preprocessing (presort against binning).

**In ./tree's favour:**

1. **CPU threads** (the biggest): ./tree finishes small subtrees on every CPU
   of the GPU's NUMA node (16 on hopper, 24 on ampere; `--cpus` in the runs'
   `machine.json`) while the GPU goes on; cuML uses about one host thread. XGBoost gets all cores but uses them mainly in the sketch.
2. **Input layout:** ./tree's host data is column-major after loading; the
   libraries get row-major NumPy and any transpose is inside their timed fit.
   Not measured.
3. **Upload path:** ./tree uploads through pinned staging buffers (~55 GB/s
   on the H100 node); cuML copies from pageable memory (typically
   10–25 GB/s). A real engineering win, but it should be shown apart.
4. The pinned buffers and the CUDA context are made during the untimed CSV
   load (small).

**In the libraries' favour:** their timed fits follow a warm-up in the same
process, with a memory pool already grown; ./tree is a cold process every run
and pays its allocations in `gpu setup`.

**Not a timing issue:** C4.5 against cuML's entropy tree compares different
algorithms (gain ratio, MDL threshold cost, MinSplit; ./tree's depth-10 C4.5
tree has half the nodes or fewer). No public GPU C4.5 exists. Lead with CART.

## Caveats to state

* **On the H100 and A100 the mixed-precision filter is not used:** at an
  FP32:FP64 ratio of 4:1 or less the grower runs one exact double sweep
  (`onePass_` in `gpu_grower.cu`). The lead over cuML there comes from the
  data layout, the partition, the two-class tile bound and the CPU/GPU split.
  The float filter is what makes consumer GPUs (64:1, e.g. the laptop's RTX
  5070) work.
* **./tree is a CPU + GPU hybrid:** nodes below `--gpu-min-rows` (CART
  rows / 8192 within [1024, 2048], C4.5 rows / 4096 within [1024, 4096]) go to
  the CPU. Report the GPU / CPU share and a thread sweep.
* **Memory cost of exactness:** ./tree uses more VRAM than the libraries on
  HIGGS, SUSY and ECBDL (73.3 GiB against 36–48 GiB on ECBDL) and less on
  chess (25.4 against 42–44 GiB).
* **Depth:** the GPU's lead shrinks as trees deepen (HIGGS CART against
  epyc: 4× at depth 10, 1.1× unlimited), and depth 10 underfits, C4.5 most
  ([PERFORMANCE.md](PERFORMANCE.md#october-2026-what-the-benchmarks-showed-mordor3-cluster)).
  State the depth with every figure.
* **Single tree:** cuML is designed for forests; the target here is one
  exact tree.

## Experiments still missing

1. **Clean ablation** on one commit and machine: `--gpu-sweep one-pass` against
   `two-pass` (also on the laptop), a `--gpu-min-rows` sweep, a `--threads`
   sweep (1 / 4 / 8 / all); the bitmap, fused next-level histograms, tile bound
   and asynchronous hand-off ring would need compile-time switches. Only the
   2026-10-28 changes were measured on one commit
   ([PERFORMANCE.md](PERFORMANCE.md#october-2026-four-cuda-changes-measured-on-one-commit-mordor3-cluster)).
2. **Per-kernel and per-level profile:** `--profile` (every kernel and copy,
   one row per GPU level) and `nsys` (`make NVTX=1`); achieved bandwidth of
   `partitionKernel` and `tileExactKernel` against peak.
3. **Fairness variants:** the libraries with data already on the GPU (CuPy,
   column-major) and with `np.asfortranarray` input; ./tree's build time
   without `gpu setup` next to them.
4. Optionally: scaling in rows (airline subsamples) for ./tree against cuML,
   and an `nsys` profile of cuML.
