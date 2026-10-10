# Growing a tree on the GPU

File: `src/build/gpu_grower.cu` (the CPU side it hands work to is
`src/build/cpu_builder.cpp`).

Read [CPU.md](CPU.md) first: the GPU runs the same algorithm — presorted
columns, one sweep per feature, stable partition — but organised so that
thousands of GPU threads have work at the same time.

## CUDA in five words

*Host* = CPU and its RAM, *device* = GPU and its memory; data must be copied
between them. A *kernel* is a function the GPU runs in many *threads* at once;
threads come in *blocks* (here 256 threads) that share a small fast memory, and
32 neighbouring threads form a *warp* that executes in lockstep and can
exchange registers directly (`__shfl_*`, `__ballot_sync`).

## Breadth-first instead of depth-first

The CPU builds one node at a time. The GPU instead processes **all large nodes
of one tree level together**: a level with 300 nodes × 18 features is one set
of kernel launches, not 300. Work is cut into **tiles**: 2048 consecutive
entries of one (node, feature) column range — a *segment*. Every kernel runs
one block per tile (or per segment).

Two copies of all columns live on the device. Level `d` reads buffer `d % 2`
and writes the partitioned children into the other one (ping-pong), so a
partition never overwrites data it still has to read.

## Setup

1. Upload the feature columns (through temporarily page-locked memory: 4×
   faster copies) and labels. All device buffers are allocated once per run.
2. Per grown tree: build `(value, packed row/class)` pairs of the selected
   rows and sort each feature with CUB's radix sort (stable, like the CPU's).
   For C4.5 the sorted values are copied back to the host (its thresholds
   need them) into a buffer that was touched by all threads and page-locked
   in the background during setup, so the copy runs at full speed and costs
   no page faults.

With cross-validation (CART `--cv K`) the raw values stay on the device in
their own buffer, so each fold tree only uploads its row list and
sorts on the device; nothing is reallocated or re-pinned between folds.

The CUDA context itself is created on a background thread while the CSV is
being parsed (`gpuPrepare`).

## One level

Every node that enters a level already has its class counts (the root's are
counted on the host, the others come from their parent's split) and is known
to need a split search, so the host prepares the whole level before launching
anything.

| # | Kernel | What |
|---|--------|------|
| 0 | `segmentConstantKernel` | Segments whose values are all equal (first vs last entry). Every later kernel skips their tiles, they are not partitioned, and the feature is dropped for the node's whole subtree (see CPU.md). |
| 1 | `tileHistogramKernel` | Class counts of each tile (one `__ballot_sync` per class per 32 entries), and C4.5's cut counts. Only at the root: below it, the parent level's partition (5) counts them while it writes the children. |
| 2 | `segmentPrefixKernel` | Per segment, exclusive scan over its tiles: each tile learns the class counts *before* it. |
| 3a | `tileEstimateKernel` | Sweep all cuts of the tile in **single precision**; keep the tile's best estimate and count C4.5's `tries`. With at most 8 classes it also keeps the tile's *candidates*: the cuts within the float error bound of the tile's own best estimate (with their class counts and values), if there are at most 8. |
| 3b | `segmentMaxKernel` | Best estimate per segment. |
| 3c | `tileCandidateKernel`, `tileExactKernel` | Compute the exact **double-precision** gain (`dt::cutGain`, the CPU's function) only for cuts whose estimate is within a proven float error bound of the segment's best. `tileCandidateKernel` (one warp per tile) scores the candidates 3a kept; `tileExactKernel` sweeps again only the tiles that had more (with more than 8 classes: every tile that can hold the best cut). On GPUs with fast double precision (data-center parts: A100, H100, B200…, detected from the FP32:FP64 ratio) 3a and 3b are skipped and `tileExactKernel` scores every cut once (`--gpu-sweep` overrides). |
| 4 | `segmentBestKernel` | Best cut per (node, feature), and the class counts left of it. **Host** (the level's only round trip): `SplitRules::choose` picks each node's split exactly like the CPU; the winning feature's left counts give both children's class counts, so children that are leaves (pure, too small, depth limit) are finished on the spot. |
| 5 | `markGoesLeftKernel`, `partitionKernel` | Stable partition of every non-constant column of every split node into the other buffer (skipped for nodes whose children are both leaves), in **one pass**: each tile learns how many left and right entries come before it by decoupled look-back on its predecessors' published counts, then writes its entries coalesced. While writing, it counts the class histograms (and C4.5 cut counts) of the next level's tiles, so step 1 is not needed below the root. |
| 6 | `copyKernel` | Children with fewer than `--gpu-min-rows` rows are gathered into one block (in the buffer that was just read, now free) and copied to the host in **one** transfer; only the columns that can still split them. |

Small children become `CpuTreeBuilder` tasks on the thread pool, which grow them
while the GPU continues with the next level. The host buffer they read from is
page-locked once, in the background, so these copies run at full speed.

### Inside the sweep (kernels 3a / 3c)

Each warp handles 256 consecutive entries of the tile in 8 steps of 32. In a
step, lane `i` holds entry `i`; the class counts left of its entry are

```
left[k] = (count before this step) + popc(ballot(class == k) & lanes below i)
```

and the previous / next values come from warp shuffles. No thread ever walks
entries one by one.

### Why two passes

Consumer GPUs run double precision about 64× slower than single precision. The
float pass rules out almost all cuts; the exact pass keeps the result
bit-identical to the CPU, so `tests/check.sh` can require identical
trees. (Comparing against the *tile's* best estimate was not enough: in a node
with millions of rows the gain hardly changes within 2048 rows, so most cuts
survived. The segment-wide best fixes that.)

The tighter the float error bound, the fewer cuts survive. The Gini estimate
is computed as `Σ_k (l_k·n − t_k·n_L)² / (n² · n_L · n_R)` (`l` left, `t` node
class counts): the differences are exact in 64-bit integers and no term
cancels another, so its error is a small fraction of the gain itself, not of
`n` (the bound is relative: `cutoff = best − best·scale − margin`). With the
former formula, which subtracts terms of size `n`, hundreds of cuts per
segment survived near the optimum in big nodes. Entropy is summed as
`c·log2(m/c)` terms, which are accurate already.

The candidates of 3a are kept against the tile's own best estimate, which is
at most the segment's, so they include every cut 3c has to score
(`tileCandidateKernel` checks that, and sweeps the tile again otherwise).
Few tiles have more than 8 of them, so the second sweep, which read every
surviving tile again, mostly disappears: on HIGGS (test laptop) the exact
pass went from 260–290 ms to about 15 ms (CART) and 55 ms (C4.5).

## Tuning

* `--gpu-min-rows N` (default 512): nodes smaller than this are finished on
  the CPU. Lower values keep more of the tree on the GPU (more, smaller levels);
  higher values hand over earlier and keep the CPU pool busier. The best value
  depends on the balance between the GPU and the CPU: on the test laptop
  (power-limited GPU) full CART trees on SUSY were fastest around 512,
  covertype (54 features, 7 classes) around 4096, and C4.5 barely cared.
* `./tree --cuda --gpu-profile ...` prints per-level times, a per-kernel
  profile and the setup phases.

## Memory

Device: two copies of all columns (`2 × 8 × rows × features` bytes, 1.4 GB for
5M × 18), with cross-validation also the raw values (`4 × rows × features`),
plus small per-level arrays (the largest: 3a's candidates, 8 × (16 + 4·K)
bytes per tile with K the class count rounded up to 2, 4 or 8, e.g. 28 MB at
HIGGS's root level). The builder checks free memory first and
suggests `--parallel` if it does not fit.

## Building for other GPUs

`make` compiles for the GPU of the build machine (`-arch=native`). For a
cluster, name the targets, e.g.

```bash
make CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90"
```
