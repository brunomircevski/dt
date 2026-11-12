# Growing a tree on the GPU

Files: `src/build/gpu_grower.cu` (the grower: setup, levels, hand-offs),
`src/build/gpu_kernels.cuh` (the kernels), `src/build/gpu_host.cuh` (device
arrays, pinned staging, CPU pinning); the CPU side it hands work to is
`src/build/cpu_builder.cpp`.

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
2. Build `(value, packed row/class)` pairs of all
   rows and sort each feature with CUB's radix sort (stable, like the CPU's).
   The root's class counts come from a kernel in the same stream (a shared
   histogram per block), not from a pass over the labels on the host.
   Each feature is presorted as soon
   as its raw values are uploaded: the copies run on a copy stream, and the
   presort waits only for the staging event of the chunk that completes the
   feature, so the upload and the presort overlap (ECBDL 0.5-0.6 s less on the
   H100 node). With C4.5 pruning the kept copy of the raw values is made per
   feature, just before its presort.
   For C4.5 the device keeps each feature's distinct values (CUB's
   `DeviceSelect::Unique` on the sorted values' bits, into free device
   memory) and copies only those to the host for the thresholds, instead of
   all sorted values (as big as the columns), and the presort went from
   0.54 to 0.09 s on the H100 node.

The CUDA context itself is created on a background thread while the CSV is
being parsed (`gpuPrepare`).

## One level

Every node that enters a level already has its class counts (the root's come
from the presort's kernel, the others from their parent's split) and is known
to need a split search, so the host prepares the whole level before launching
anything.

| # | Kernel | What |
|---|--------|------|
| 0 | `segmentConstantKernel` | Segments whose values are all equal (first vs last entry). Every later kernel skips their tiles, they are not partitioned, and the feature is dropped for the node's whole subtree (see CPU.md). |
| 1 | `tileHistogramKernel` | Class counts of each tile (one `__ballot_sync` per class per 32 entries), and C4.5's cut counts. Only at the root: below it, the parent level's partition (5) counts them while it writes the children. |
| 2 | `segmentPrefixKernel` | Per segment, exclusive scan over its tiles: each tile learns the class counts *before* it. |
| 2b | `tileBoundKernel`, `segmentMaxKernel` | Two classes only: an upper bound of every tile's gains and the exact gain of one of its cuts. The gain is convex in the left class counts, and a tile's cuts lie in the box between the counts before and after it, so none beats the largest of the box's four corners. The sweep (3) skips a tile whose bound is more than `kTileBoundGap = 1e-8` below the best of those exact gains in its segment: most tiles of a big node. |
| 3 | `tileExactKernel` | Score every cut of the tile that the boundary-point rule leaves with the exact **double-precision** gain (`dt::cutGain`, the CPU's function), and count C4.5's `tries` of single-tile segments. One sweep, on every GPU: the H100 and the A100 run double precision at half the single-precision rate, so a single-precision pre-filter does not pay (see [One exact pass](#one-exact-pass)). |
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
`atomicOr` for the rows of the smaller side only, cleared every level): 14 MB
for airline, which stays in L2, against 115 MB of bytes that did not. The
H100's CART build on airline went from 1.73 to 1.09 s. A node with more left
than right rows marks its right rows and the partition flips the bit
(`NodeDesc::marksRight`), so at most half of a node's rows take an atomic. In
the split feature's own column the partition does not read the bitmap at
all: the left rows are the first `leftCount` entries. On chess_1b (H100,
CART depth 10, total 0.97 s) each alone saved 0.07 and 0.12 s.

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

Sending subtree raising's big row slices through the tree on the GPU (the
CPU keeping every decision, the GPU only routing and counting) was tried and
removed in November 2026: each call cost about 0.45 ms on the host side, the
columns the tree uses had to be uploaded first unless the raw values were kept
(ECBDL: 631 columns, 2.5 s), and the CPU was faster on every dataset and
threshold tried (`cluster-gpu-prune`, a benchmark that was never committed).
It is in the git history before the commit that removed `--gpu-prune`.

### Inside the sweep (kernel 3)

Each warp handles 256 consecutive entries of the tile in 8 steps of 32. In a
step, lane `i` holds entry `i`; the class counts left of its entry are

```
left[k] = (count before this step) + popc(ballot(class == k) & lanes below i)
```

and the previous / next values come from warp shuffles. No thread ever walks
entries one by one, and there is no shared-memory scan and no atomic. A cut
the boundary-point rule rules out (`isSkippableCut`, shared with the CPU) is
not scored.

### One exact pass

Until November 2026 the grower could also sweep in two passes: a
single-precision estimate of every cut (with a proven error bound), then the
exact gain only for the cuts within that bound of the node's best. That paid
on consumer GPUs, whose double precision is about 64× slower than single
precision, and was picked from the device's FP32:FP64 ratio. ./tree now
targets data-center GPUs only (the cluster's H100 and A100, FP32:FP64 = 2:1),
where one exact sweep was as fast or faster
([PERFORMANCE.md](PERFORMANCE.md#november-2026-one-exact-sweep-on-every-gpu)),
so the two-pass kernels and the `--gpu-sweep` option were removed; they are in
the git history before that commit. Every GPU now runs the same kernels.

### Why the result is exact

* Every cut that the boundary-point rule and the two-class tile bound leave
  is scored with the CPU's `cutGain` in double precision.
* Near-ties (gains within `kTieEps = 1e-12`) go to the earlier cut, then the
  lower feature, in a fixed reduction order; `c·log2 c` comes from the table
  the CPU uses ([ALGORITHMS.md](ALGORITHMS.md#ties-and-cpugpu-agreement)).
* So the trees are byte-identical across `--simple`, `--serial`,
  `--parallel` and `--cuda`, and across the H100 and the A100
  (`tests/check.sh`).

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
  (the depth-19 cluster-gpu-vs-others run of 2026-10-20, in the git history
  before 325039f; the gpuonly-* runs of the cluster-* benchmarks).
* `./tree --cuda --profile FILE ...` records per-level rows, every kernel and
  copy, and the setup phases (context, allocation, upload);
  `bench/profile_chart.py` draws it.

### What did not help

* Keeping a tile's values in registers after the sweep: 52 → 74 registers,
  3 instead of 4 blocks per SM, 15% slower; reloading them from cache is
  faster.
* 4 or 16 entries per thread instead of 8: slower.
* More GPU levels (a smaller `--gpu-min-rows`): the CPU threads finish small
  subtrees faster than a GPU level of tiny nodes.
* The partition's traps (tile order, local memory, move records) are in
  [PERFORMANCE.md](PERFORMANCE.md#october-2026-one-copy-of-the-columns-cuda-serial-parallel).

## Memory

Device: one copy of the columns in `features + 1` slots (`8 × rows ×
(features + 1)` bytes, 0.76 GB for 5M × 18), a 512 MB ring for the hand-offs to
the CPU, with C4.5 pruning a copy of the raw values if it fits (for `arrangeRows`),
plus small per-level arrays (the largest: the tile histograms, 4·K bytes per
tile with K the class count, e.g. about 124 MB at MNIST8M's root level). The builder checks free memory first and
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

## Pseudo code

Notation: `n` rows, `F` features, `K` classes, `T = 2048` entries per tile
(256 threads × 8), `C_f` the presorted column of feature `f`, a segment the
range of one node in one column. Kernel names in comments.

### Algorithm 1: level-synchronous growth with CPU hand-off

```
Input: X ∈ R^{n×F}, labels y ∈ {0..K-1}^n, depth limit D, hand-off size m
 1  upload y; upload X through 4 pinned 32 MB staging buffers on a copy stream
 2  for f ← 1..F, as soon as column f is uploaded:       ▷ overlapped with step 1
 3      C_f ← StableRadixSort{(X[i,f], pack(i, y_i))}    ▷ 8 B per entry, into a slot
 4  L ← {root};  counts(root) ← Histogram(y)              ▷ on the GPU, during the presort
 5  while L ≠ ∅:
 6      S ← {(v, f) : v ∈ L, f not constant at v}         ▷ segments, cut into tiles of T
 7      if L = {root}: TileHistogram(S)                   ▷ below the root, step 13 did it
 8      SegmentPrefix(S)                                  ▷ class counts before each tile
 9      if K = 2: TileBound(S)                            ▷ Alg. 2b
10      SweepTiles(S); SegmentBest(S)                     ▷ Alg. 2
11      copy the best cut of every (v, f) to the host     ▷ the level's only round trip
        host: choose each v's split (CART / C4.5 rule); children's counts from
              the winner's left counts; leaves (pure, < minChild, depth D) finished
12      MarkGoesLeft                                      ▷ Alg. 4: 1-bit bitmap in L2
13      Partition(S) → children's columns in free slots,
                       with the next level's tile histograms   ▷ Alg. 3
14      children with < m rows: gather → device ring → host (copy stream);
              a host callback starts CPU subtree tasks; the GPU continues
15      L ← the other children
16  wait for the CPU tasks; assemble the tree
```

### Algorithm 2: warp-ballot sweep of one tile

(`loadTile`, `sweepTile`, `tileExactKernel`)

```
warp w owns entries [256w, 256w + 256) of the tile; lane ℓ ∈ [0, 32)
before[k] ← tilePrefix[k] + Σ_{w' < w} warpCount[w'][k]     ▷ shared memory, one barrier
best ← none
for s ← 0..7:
    e ← C[base + 256w + 32s + ℓ]                             ▷ coalesced 8-byte load
    for k ← 0..K-1:
        b ← ballot(e.class = k)
        left[k] ← before[k] + popc(b & lanesBelow(ℓ))
        before[k] ← before[k] + popc(b)
    prev ← shfl_up(e.value, 1);  next ← shfl_down(e.value, 1)
    if prev < e.value ∧ minChild ≤ pos ≤ n − minChild ∧ ¬isSkippableCut(...):
        best ← max(best, (Gain(total, left, pos), pos))
tileBest ← BlockReduce(best)                     ▷ fixed order; ties → lowest position
```

### Algorithm 2b: tile bound for two classes

(`tileBoundKernel`)

```
for each tile t of segment (v, f):
    a ← class counts before t;  z ← class counts after t
    U_t ← max over the corners c ∈ {a₀, z₀} × {a₁, z₁} of Gain(total, c)   ▷ convex gain
    ℓ_t ← exact gain of one allowed cut of t                                ▷ a lower bound
lower(v, f) ← max_t ℓ_t
the sweep skips tile t if U_t < lower(v, f) − 1e-8
```

### Algorithm 3: single-pass stable partition with next-level counts

(`partitionKernel`)

```
t ← atomicAdd(tileCounter, 1)          ▷ dynamic tile ids: predecessors are already running
load T entries (coalesced)
g_j ← (pos_j < nL(v))        if f is v's split feature     ▷ its left rows come first
      B[row_j] ⊕ flip(v)     otherwise                     ▷ flip(v): B marks the right rows
(prefix_j, ℓ_t) ← BlockScan(g)
publish state[t] ← (PARTIAL, ℓ_t), or (TOTAL, ℓ_t) for the segment's first tile
thread 0: walk back t−1, t−2, … summing PARTIAL counts until a TOTAL;
          publish state[t] ← (TOTAL, before + ℓ_t)        ▷ self-validating 64-bit words
          wait until the column that held the target slot has been read
stage in shared memory: left entries first, then right entries
write the left run to dst[before + i], the right run to dst[nLeft + (start − before) + i]
while writing: class histogram (match_any + shared atomics) and cut count (ballot)
               of each next-level tile, so the next level skips TileHistogram
```

### Algorithm 4: goes-left bitmap

(`markGoesLeftKernel`)

```
for each split node v with winning feature f, n_v rows, nL of them left:
    flip(v) ← (2·nL > n_v)                       ▷ mark the smaller side
    I ← flip(v) ? [nL, n_v) : [0, nL)
    for i ∈ I in parallel:
        r ← row(C_f[base_v + i])
        atomicOr(B[r / 32], 1 << (r mod 32))     ▷ ≤ n_v / 2 atomics; B cleared every level
```

## Building for other GPUs

`make` compiles for the GPU of the build machine (`-arch=native`). For a
cluster, name the targets, e.g.

```bash
make CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90"
```
