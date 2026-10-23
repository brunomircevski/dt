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

One copy of the columns lives on the device, in a buffer of **slots**: one per
feature and one more (each `8 × rows` bytes, rounded up to 2 MiB). The
partition moves every column it writes to a free slot: first the slot that
holds no column of the level, then the ones the moved columns leave, so column
`f` usually goes to the slot column `f − 1` just left. A node's rows keep their
index range in every column, so in that slot a segment overwrites only the same
node's segment of the column that left; a tile writes once that segment's last
tile has published its running total (which every tile does right after
loading its entries), and within a node the segments are in column order, so
the tiles it waits for have started before it. Until 2026-10-17 two full
copies ping-ponged instead (and a third mirrored the host arena, see Memory).

## Setup

1. Upload the feature columns and labels at the speed of page-locked memory
   (55 GB/s on the H100 PCIe, against 10 GB/s pageable) without page-locking
   them, which is slow with HMM (both cluster GPU nodes): the pool's threads
   copy 32 MB chunks into four page-locked staging buffers while the previous
   chunks upload. Airline's 6 GB take 0.2 s on the H100 node, against 0.35 s
   page-locked. The staging buffers, and the threads that copy through them,
   are on the GPU's NUMA node: on the A100 node (two sockets) copies from the
   other socket ran at 21 GB/s up and 16 GB/s down instead of 25 and 24, and
   the upload went from 0.5 to 0.3 s. The staging buffers are allocated while
   the CSV loads. All device buffers are allocated once per run.
2. Per grown tree: build `(value, packed row/class)` pairs of the selected
   rows and sort each feature with CUB's radix sort (stable, like the CPU's).
   For C4.5 the device keeps each feature's distinct values (CUB's
   `DeviceSelect::Unique` on the sorted values' bits, into free device
   memory) and copies only those to the host for the thresholds, instead of
   all sorted values (as big as the columns), and the presort went from
   0.54 to 0.09 s on the H100 node.

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
| 5 | `markGoesLeftKernel`, `partitionKernel` | Stable partition of every non-constant column of every split node into a free slot (skipped for nodes whose children are both leaves), in **one pass**: each tile learns how many left and right entries come before it by decoupled look-back on its predecessors' published counts, then writes its entries coalesced. While writing, it counts the class histograms (and C4.5 cut counts) of the next level's tiles, so step 1 is not needed below the root. |
| 6 | `copyKernel` | Children with fewer than `--gpu-min-rows` rows are gathered into one block (in the buffer that was just read, now free) and copied to the host in **one** transfer; only the columns that can still split them. |

Small children become `CpuTreeBuilder` tasks on the thread pool, which grow them
while the GPU continues with the next level. The host buffer they read from
(the arena, room for every column of every row: 75 GB for ECBDL'14's 15M
sample) is pageable. The copies go through two page-locked 128 MB buffers,
one per copy stream, like the upload: a host function on the copy stream
copies the buffer into the arena on the staging threads, and the two
streams take turns, so that one downloads while the other's buffer is
copied out. The arena's pages are touched on all threads in the background
during the presort, so the copies do not fault them in; a hand-off waits
only for the part of the arena it fills.
Page-locking the arena instead took the CUDA driver's lock for 1.2-1.9 s
(sometimes 4-6 s on the H100 node) and stalled the level loop's CUDA calls.

The goes-left flags of step 5 are a bitmap (one bit per row, set with
`atomicOr` for the left rows only, cleared every level): 14 MB for airline,
which stays in L2, against 115 MB of bytes that did not. The H100's CART build
on airline went from 1.73 to 1.09 s.

The columns are gathered into a 512 MB ring of four parts on the device and
copied to the host on the copy streams, and a host callback there
(`cudaLaunchHostFunc`) starts the CPU tasks when the copy is done: the level
loop goes on with the next level instead of waiting for each hand-off to
arrive. Device buffers that change
size between levels use stream-ordered allocation (`cudaMallocAsync`, with a
pool that keeps its memory), so a resize does not synchronize the device.

For C4.5 with pruning the grower also records the leaf of every row: a
kernel writes them for the nodes finished on the GPU, the CPU tasks for
theirs, and they are merged at the end, streamed from the staging buffers
straight into the ids. Pruning then does not route the rows down the tree
again (see CPU.md). When there is room, the device also keeps a copy of the
raw values, and the pruning's first step (rows sorted by leaf, their features
transposed to rows, their labels and the class counts of every leaf) runs on
the GPU (`arrangeRows`): a stable radix sort by leaf and one gather kernel
instead of two CPU passes over 115M rows.

Subtree raising can also send its big row slices through the tree on the GPU
(`--gpu-prune-rows N`, `--gpu-prune` for all of them): the CPU keeps every
decision, the GPU only routes and counts (kernels `pruneCount`,
`pruneGather`, `pruneArrange`, with the changed nodes sent before each call),
so the tree is the same to the bit. It is off by default, with `--gpu-only`
too: each call costs about 0.45 ms on the host side, the columns the tree uses
have to be uploaded first unless the raw values were kept (ECBDL: 631
columns, 2.5 s), and the CPU was faster on every dataset and threshold tried
(see [cluster-gpu-prune](../bench/results/cluster-gpu-prune/BENCHMARK.md)).

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

* `--gpu-min-rows N` (default: CART training rows / 8192, between 1024 and
  2048; C4.5 rows / 4096, between 1024 and 4096): nodes smaller than this are
  finished on the CPU. Lower values keep more of the tree on the GPU (more,
  smaller levels); higher values hand over earlier and keep the CPU pool
  busier. A level costs the GPU about as much for thousands of mid-size nodes
  as for a few big ones, so the best value grows with the dataset, but
  slowly: on the H100 (32 CPU cores) and the A100 (48), HIGGS (10.5M rows) was
  fastest at 1-2.5K and airline (115M rows) at 2K for CART and 2-4K for C4.5,
  10-15% faster than at 28K. (Airline's best was 16-64K while the CPU threads
  still waited on the node store; see docs/PERFORMANCE.md.) On the test laptop
  (power-limited GPU) full CART trees on SUSY were fastest around 512,
  covertype (54 features, 7 classes) around 4096.
* `--gpu-only`: no node goes to the CPU; the GPU grows every level down to
  the leaves, and the host arena for the hand-offs (8 bytes per value) is not
  allocated. Everything else stays as without it: the CPU threads
  (`--threads`, by default every CPU the process may run on) count the
  classes, stage the uploads and run C4.5's pruning. With `--threads 1` the
  CPU side is a single thread, except for the staged copies: they use at
  least 8 threads on the GPU's node, which only run during copies (1G rows
  of chess uploaded in 0.26 s instead of 0.67 s with one thread on the H100
  node, 0.49 instead of 0.69 s on the A100 node). Never more than the CPUs
  the process may run on. For comparing the GPU alone with the GPU + CPU split
  ([cluster-gpu-vs-others](../bench/results/cluster-gpu-vs-others/BENCHMARK.md)).
* `./tree --cuda --gpu-profile ...` prints per-level times, a per-kernel
  profile and the setup phases (allocation, upload).

## Memory

Device: one copy of the columns in `features + 1` slots (`8 × rows ×
(features + 1)` bytes, 0.76 GB for 5M × 18), a 512 MB ring for the hand-offs to
the CPU, with cross-validation also the raw values (`4 × rows × features`),
with C4.5 pruning a copy of the raw values if it fits (for `arrangeRows`),
plus small per-level arrays (the largest: 3a's candidates, 8 × (16 + 4·K)
bytes per tile with K the class count rounded up to 2, 4 or 8, e.g. 28 MB at
HIGGS's root level). The builder checks free memory first and
suggests `--parallel` if it does not fit.

* **Presort** in the slots too: the raw values are uploaded into the last
  slots; each feature is gathered into a free slot and radix-sorted (CUB, by
  value only, stable) with another free slot as the second buffer, and stays
  in whichever holds the result. A slot is taken once the raw values in it
  are sorted. C4.5's distinct values are collected in the ring (free until the
  build) and downloaded once.
* **Hand-offs** to the CPU go through the ring, in pieces of a quarter of it:
  each piece is gathered into the next part once that part's last copy to the
  host is done, so the GPU waits only when it is a whole ring ahead of PCIe.
  Before, a mirror of the whole host arena (another `8 × rows × features`)
  took the hand-offs.
* **Cost:** on the H100, HIGGS (10.5M × 28) went from 7.2 to 3.4 GiB at the
  same speed, airline (115M × 13) from 35 to 13.3 GiB, 2-4% slower (the
  partition waits a little for the column before it, and the ring for PCIe);
  see [PERFORMANCE.md](PERFORMANCE.md).

## Building for other GPUs

`make` compiles for the GPU of the build machine (`-arch=native`). For a
cluster, name the targets, e.g.

```bash
make CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90"
```
