# ./tree against GPU tree libraries

How `./tree --hybrid` is compared with cuML and XGBoost, what is timed on each
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

**./tree:** its `train total`, a fresh process per run, `--hybrid --cart|--c45
-d 10` (unpruned) with the default threads. `train total = gpu setup + presort
+ build + prune` (printed by `src/app/main.cpp`):

| Phase | Contents |
|---|---|
| gpu setup | streams, device allocations, upload of columns and labels through the staging buffers |
| presort | `(value, row \| class)` entries, CUB radix sort per feature (C4.5: distinct values); overlapped with the upload |
| build | all GPU levels, hand-offs, the CPU threads finishing small subtrees, assembling the host tree |
| prune | C4.5 only: collapse of useless splits and training-value thresholds (part of C4.5 even without `--prune`) |

Not in it: reading the input and, in every benchmark run (`--gpu-startup
context`, since 2026-11-17), only the bare CUDA context, made on a background
thread while the input loads (`src/app/main.cpp`) and printed as `gpu
startup`; `train total` starts only after it has finished. Kernel loading
(lazy, CUDA's default), the memory pool and the pinned staging buffers (4 ×
32 MB for uploads, 2 × 128 MB for the hand-offs) and the copy threads are then
made in the timed `gpu setup`, exactly as the libraries make theirs in their
fit. ./tree's default (`--gpu-startup full`) makes all of them before training,
overlapping the load: 0.17-0.28 s more untimed start-up than the bare context
on HIGGS (hopper, ampere), about as long as the timed training itself, so a
comparison with it would not be fair; it is measured end to end instead.

**cuML / XGBoost** (`gpubench.py`, since 2026-11-16): one fresh Python process
per run, no warm-up fit. It loads row-major float32 NumPy arrays, then does the
start-up (import the library, which for cuML imports RMM; then the CUDA context
alone, `cudaFree(0)`, with `CUDA_MODULE_LOADING=LAZY`, CUDA's default: both
libraries link their own CUDA runtime, so EAGER would load *all* their kernels
at their first call, inside `fit` (HIGGS, A100, 2026-11-17, 6 runs each: cuML
2.19 s EAGER vs 1.71 s LAZY, XGBoost 3.11 vs 2.88 s); the kernels a fit uses
load inside it, timed, while ./tree loads its own with the context; RMM stays at cuML's
default memory resource, nothing reconfigured or pre-grown), then exactly one
timed fit (`time.perf_counter`). Ten runs are ten processes, in rounds of one
process per case in a shuffled order (`job.sh`); reported as means:

* cuML `fit(X, y)`: input conversion, host→device copy, quantile binning,
  growing.
* XGBoost: `QuantileDMatrix(X, y)` (copy and quantile sketch) and
  `xgb.train(..., num_boost_round=1)`.
* `predict` and model export are not timed.

VRAM: peak `nvidia-smi` memory in use above idle, for the libraries over the
whole process (start-up included).

## Fairness

**Matched:** same rows, features and depth limit, no sampling, no pruning,
same GPUs on exclusive nodes, both start from host RAM, both pay their upload
and preprocessing (presort against binning).

**In ./tree's favour:**

1. **CPU threads** (the biggest): ./tree's `--hybrid` finishes small subtrees
   on every CPU of the node (32 on hopper, 48 on ampere; `--cpus` in the runs'
   `machine.json`) while the GPU goes on; `--gpu` uses the CPU only to drive
   the GPU; cuML uses about one host thread. XGBoost gets all cores but uses them mainly in the sketch.
2. **Input layout:** ./tree's host data is column-major after loading; the
   libraries get row-major NumPy and any transpose is inside their timed fit.
   Not measured.
3. **Upload path:** ./tree uploads through pinned staging buffers (~55 GB/s
   on the H100 node); cuML copies from pageable memory (typically
   10–25 GB/s). A real engineering win, but it should be shown apart.
4. The pinned buffers and the CUDA context are made during the untimed load
   (small); the libraries' context and imports are untimed too, but not
   overlapped with anything.

**Neither has a warm-up:** since 2026-11-16 every library fit is a process's
first and only fit, as ./tree's tree is: memory pools grow inside the timed
fit for both (./tree in `gpu setup`, cuML's allocations in `fit`), and any JIT or
first-call cost is timed. Until then the libraries' timed fits followed an
untimed warm-up fit in the same process (cuML on every row, XGBoost on 50,000
rows), which favoured them.

**The warm variant, apart** (since 2026-11-17): every GPU tool, ./tree too,
also gets one CART tree timed as a process's second identical fit, after an
untimed one (./tree `--warm-up`, gpubench.py `WARM=1`), on its own chart
(cluster-gpu-vs-others/chart-warm.png) beside the cold time. It shows the
one-time costs of a first fit: cuML's first copy of the NumPy input is the
largest (MNIST8M on the H100: 11.2 s, a second copy 2.1 s).

**C4.5 has no GPU competitor.** No public GPU C4.5 exists, and an entropy
(information-gain) tree is a different algorithm (no gain ratio, MDL threshold
cost or MinSplit), so the libraries run CART only. cuML's entropy tree is not
used anywhere as a C4.5 stand-in (removed 2026-11-16).

## Caveats to state

* **One exact sweep, no mixed precision:** the grower scores every cut the
  boundary-point rule and the two-class tile bound leave in double precision
  (the H100 and A100 run FP64 at half the FP32 rate). The lead over cuML comes
  from the data layout, the partition, the two-class tile bound and the
  CPU/GPU split. Consumer GPUs (FP64 about 64× slower) are not a target.
* **./tree is a CPU + GPU hybrid:** nodes below `--handoff-rows` (10 ×
  features, at most 2^depth limit, within [512, 16384]) go to the CPU. Report the GPU / CPU share and a thread sweep.
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

1. **Clean ablation** on one commit and machine: a `--handoff-rows` sweep, a `--threads`
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
4. Optionally: scaling in rows (chess_1m to chess_500m) for ./tree against cuML,
   and an `nsys` profile of cuML.
