# ./tree against GPU tree libraries

How `./tree --gpu` and `--hybrid` are compared with cuML and XGBoost, what is
timed on each side and where the comparison is still not even. Results,
settings and the scripts are in
[bench/results/cluster-gpu-vs-others](../bench/results/cluster-gpu-vs-others/BENCHMARK.md)
(one CART tree, depth ≤ 10, no pruning, hopper's H100; draft of 2026-11-18,
3 runs a case, until the rerun).

## The result in one sentence

`./tree --gpu` builds an **exact** CART tree (every distinct value tested,
byte-identical to the CPU backends) of depth ≤ 10 on one H100 **2.4–2.7×
faster than cuML 26.10** and **2.4–3.3× faster than XGBoost 3.4.1** on HIGGS
and MNIST8M, from the same host data through the same upload code. Its tree
construction alone is 3.2–4.6× faster. cuML and XGBoost split only at 128 and
256 histogram bins, and their test accuracy is within 0.05 points.

## Why the design is faster

Each feature is presorted once into 8-byte `(value, row | class)` entries, the
tree grows level by level, and every node's rows stay contiguous in every
column ([CUDA.md](CUDA.md)). A level costs about three streaming passes over
`8·n·F` bytes (sweep: read; partition: read and write), ~24 bytes per value,
with no row-index gathers, no binning and no per-node histograms: the cost
grows with rows, not with nodes × bins.

Histogram libraries bin first and then build per-node class × bin × feature
histograms. XGBoost's binning is a separate, measured step
(`QuantileDMatrix`: 0.15 s on HIGGS, 2.3 s on MNIST8M). cuML computes its
128 quantiles from a row sample inside its fit (about 0.2 ms of GPU time,
`nsys`), so its time is almost all tree construction. A per-kernel comparison
of the libraries' tree construction with ./tree's has not been made.

## What is timed

The same for every tool (`scripts/gpubench.py` for the libraries, ./tree's
`train total`):

- **Not timed:** reading the data into host RAM as column-major float32 (and
  the labels; XGBoost's float32 or one-hot labels are made here), importing
  the library, and the bare CUDA context (`cudaFree(0)`; ./tree makes it on a
  thread while the input loads, printed as `gpu startup`).
  `CUDA_MODULE_LOADING=LAZY`, CUDA's default, so each tool loads the kernels
  it uses inside the timed part.
- **Timed:** from the host arrays to the finished tree, in four steps, each
  ending with `cudaDeviceSynchronize`:

| Step | ./tree | cuML | XGBoost |
|---|---|---|---|
| set-up | staging buffers, copy threads (`--hybrid`: also its 256 MB of hand-off buffers), device memory | staging buffers, copy threads | the same |
| upload | labels and columns through the staging buffers; the presort of each feature runs on the GPU as soon as it arrives | labels and values through the same code (`bench/adapters/uploader.cpp`) | the same |
| prepare | the x·log x tables and host arena (`setup finish`) | input checks: NaN/inf scan, classes, lazy kernel loading | `QuantileDMatrix(max_bin=256)`, then the raw device copy is freed |
| build | all levels, the hand-offs, the CPU subtrees, the host tree | `_fit_forest`: row-sample quantiles, then the tree | `xgb.train`, one round |

- One fresh process per run, no warm-up, exactly one tree. The tools'
  runs are interleaved in a new shuffled order each repetition (one
  `bench/run.py` per dataset).
- VRAM: peak `nvidia-smi` memory above idle, sampled from outside for every
  tool and the whole process.
- Prediction, accuracy and the tree walk run after the timer.

## Fairness

**Matched:** the same rows, features and depth limit, no sampling, no pruning;
the same GPU (hopper's H100, GPU 2) on an exclusive node; the same host data
in the same layout (column-major float32); the same upload code (4
page-locked 32 MB buffers on the GPU's NUMA node, up to 32 pinned copy
threads, asynchronous copies); the same untimed start-up (import and bare
context) and the same lazy kernel loading; every tool pays its own
preprocessing (presort against binning) and allocations inside the timer.

**Still in ./tree's favour:**

1. **CPU threads:** `--hybrid` finishes small subtrees on every CPU of the
   node (32 on hopper) while the GPU goes on. `--gpu`, the bar the ratios are
   taken against, uses the CPUs only to drive the GPU and for the upload, as
   the libraries do with `uploader.cpp`. cuML's tree uses about one host
   thread; XGBoost's binning runs on the GPU.
2. **Overlap of upload and presort:** ./tree presorts each feature while the
   next ones upload. The libraries cannot start before their whole input is on
   the GPU. This is part of ./tree's design and is drawn as two lanes on the
   chart.

**In the libraries' favour:**

1. **Bins:** 128 (cuML) and 256 (XGBoost) candidate splits per feature against
   every distinct value. Their trees differ from the exact tree in a few nodes
   on HIGGS (2,039 and 2,031 against 2,029) and are the same size on MNIST8M.
2. **VRAM:** cuML needs half of ./tree's memory (25 against 49 GiB on
   MNIST8M). XGBoost needs more (67 GiB).

**What changed on 2026-11-18:** until then each library uploaded its input its
own way inside its fit (cuML from pageable row-major NumPy, which on MNIST8M
made its first fit 18 s), ./tree had the only staged upload, and the bench
also ran a warm variant (a process's second fit), forests of 10 and the A100
node. Those results are in git history up to 8ba60fae.

**C4.5 has no GPU competitor.** No public GPU C4.5 exists, and an entropy
(information-gain) tree is a different algorithm (no gain ratio, MDL threshold
cost or MinSplit), so the libraries run CART only.

## Caveats to state

* **One exact sweep, no mixed precision:** the grower scores every cut the
  boundary-point rule and the two-class tile bound leave in double precision
  (the H100 runs FP64 at half the FP32 rate). Consumer GPUs (FP64 about 64×
  slower) are not a target.
* **./tree --hybrid is a CPU + GPU hybrid:** nodes below `--handoff-rows`
  (10 × features, at most 2^depth limit, within [512, 16384]) go to the CPU.
  At depth 10 on these datasets its build is the same as `--gpu`'s and its
  set-up 0.1 s longer.
* **Memory cost of exactness:** ./tree keeps presorted 8-byte entries for every
  value: more VRAM than cuML on both datasets, less than XGBoost on MNIST8M.
* **Depth:** the GPU's lead shrinks as trees deepen (HIGGS CART against
  epyc: 4× at depth 10, 1.1× unlimited), and depth 10 underfits
  ([PERFORMANCE.md](PERFORMANCE.md#october-2026-what-the-benchmarks-showed-mordor3-cluster)).
  State the depth with every figure.
* **Single tree:** cuML is designed for forests, XGBoost for boosting; the
  target here is one exact tree.

## Experiments still missing

1. **The rerun:** 10 runs a case on a committed build (the draft is 3 runs on
   uncommitted code).
2. **./tree's `setup finish`** (22–35 ms, about 16% of HIGGS `--gpu`): the
   x·log x table and a pageable copy of it; the cause of its time is not
   confirmed.
3. **Per-kernel profile of tree construction** for all three tools (`nsys`),
   to compare the libraries' histogram kernels with ./tree's sweep and
   partition.
4. **Clean ablation** on one commit: a `--handoff-rows` sweep and a
   `--threads` sweep; the bitmap, fused next-level histograms, tile bound
   and asynchronous hand-off ring would need compile-time switches.
