# Growing a tree on the GPU

File: `gpu_builder.cu` (the CPU side it hands work to is `cpu_builder.cpp`).

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
   faster copies) and labels.
2. Build `(value, packed row/class)` pairs and sort each feature with CUB's
   radix sort (stable, like the CPU's).
3. For C4.5, copy the sorted values back; the CPU pool extracts each feature's
   distinct values in the background (needed for C4.5's thresholds).

The CUDA context itself is created on a background thread while the CSV is
being parsed (`gpuPrepare`).

## One level

Every node that enters a level already has its class counts (the root's are
counted on the host, the others come from their parent's split) and is known
to need a split search, so the host prepares the whole level before launching
anything.

| # | Kernel | What |
|---|--------|------|
| 1 | `tileHistogramKernel` | Class counts of each tile (one `__ballot_sync` per class per 32 entries). |
| 2 | `segmentPrefixKernel` | Per segment, exclusive scan over its tiles: each tile learns the class counts *before* it. |
| 3a | `tileEstimateKernel` | Sweep all cuts of the tile in **single precision**; keep the tile's best estimate and count C4.5's `tries`. |
| 3b | `segmentMaxKernel` | Best estimate per segment. |
| 3c | `tileExactKernel` | Sweep again, but compute the exact **double-precision** gain (`dt::cutGain`, the CPU's function) only for cuts whose estimate is within a proven float error bound of the segment's best. |
| 4 | `segmentBestKernel` | Best cut per (node, feature), and the class counts left of it. **Host** (the level's only round trip): `SplitRules::choose` picks each node's split exactly like the CPU; the winning feature's left counts give both children's class counts, so children that are leaves (pure, too small, depth limit) are finished on the spot. |
| 5 | `markGoesLeftKernel`, `tileLeftCountKernel`, `segmentPrefixKernel`, `scatterKernel` | Stable partition of every column of every split node into the other buffer (skipped for nodes whose children are both leaves). |
| 6 | `copyKernel` | Children with fewer than `--gpu-min-rows` rows are gathered into one block (in the buffer that was just read, now free) and copied to the host in **one** transfer. |

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
bit-identical to the CPU, so `tests/check_backends.sh` can require identical
trees. (Comparing against the *tile's* best estimate was not enough: in a node
with millions of rows the gain hardly changes within 2048 rows, so most cuts
survived. The segment-wide best fixes that.)

## Tuning

* `--gpu-min-rows N` (default 512): nodes smaller than this are finished on
  the CPU. Lower values keep more of the tree on the GPU (more, smaller levels);
  higher values hand over earlier and keep the CPU pool busier. The best value
  depends on the balance between the GPU and the CPU: on the test laptop
  (power-limited GPU) full CART trees on SUSY were fastest around 512,
  covertype (54 features, 7 classes) around 4096, and C4.5 barely cared.
* `DT_GPU_VERBOSE=1 ./tree --cuda ...` prints per-level times, a per-kernel
  profile and the setup phases.

## Memory

Device: two copies of all columns (`2 × 8 × rows × features` bytes, 1.4 GB for
5M × 18) plus small per-level arrays. The builder checks free memory first and
suggests `--parallel` if it does not fit.
