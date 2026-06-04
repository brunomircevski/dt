# How the CUDA decision tree works

This document explains `tree_cuda.cpp` for someone who has never written GPU
code before. It assumes you understand the basic idea of a decision tree but
nothing about CUDA. Read it top to bottom.

---

## 1. The 30-second picture

A decision tree is built by repeatedly asking: **"for this group of rows, what
is the single best yes/no question to split them into two cleaner groups?"** A
question looks like `feature_7 <= 0.42`.

- The **CPU** owns the tree structure. It starts with all rows at the root,
  finds a split, divides the rows into a left group and a right group, and then
  repeats on each group. This recursion lives in the shared base class
  (`TreeBase::buildNode`), not in the CUDA file.
- The **GPU** does the one expensive job: given a group of rows, **try every
  possible split and report the best one**. That is the function
  `TreeCuda::findBestSplitAtNode`.

So the CPU keeps asking "best split for these rows?" and the GPU answers, over
and over, once per node in the tree.

```
CPU: "best split for these 5,000,000 rows?"  ->  GPU: "feature 11 <= 0.83"
CPU splits rows into left (3.1M) and right (1.9M)
CPU: "best split for these 3,100,000 rows?"  ->  GPU: "feature 4 <= -0.10"
... and so on, down the tree
```

---

## 2. A tiny bit of CUDA vocabulary

You only need five words:

- **Host** = the CPU and its normal RAM. **Device** = the GPU and its own
  memory (VRAM). They are separate; you must *copy* data between them.
- **Kernel** = a function that runs on the GPU. We mark it `__global__`. When we
  "launch" a kernel we ask the GPU to run that function many thousands of times
  at once, each copy on a different piece of data.
- **Thread** = one of those copies. Each thread knows its own number.
- **Block** = a team of threads (e.g. 256 of them) that runs together on one of
  the GPU's processors and can share a small, very fast scratchpad of memory.
- **Grid** = all the blocks of one kernel launch.

The launch syntax `myKernel<<<numBlocks, threadsPerBlock>>>(args)` means "run
`myKernel` using this many blocks, each with this many threads."

Inside a kernel a thread figures out which data it owns from built-in variables:
`blockIdx` (which block am I in), `threadIdx` (which thread am I inside my
block), and `blockDim` (how big is my block).

A GPU is fast only when thousands of threads are busy at the same time. A lot of
this code is about keeping the whole GPU busy instead of a few threads.

---

## 3. Why sorting is the key idea

For one numeric feature, the best threshold always sits *between two values that
are next to each other once you line the rows up in order*. So if we **sort the
rows by that feature's value**, we can walk left to right and, at each gap,
instantly know how many rows of each class are on the left vs the right. That
"walk" is what finds the best threshold.

We do this for every feature, then pick the feature+threshold with the best
score overall.

To score a split we need, at each gap, the **class counts** on each side.
Keeping a running count as we walk is cheap. The "impurity" math (Gini or
entropy) just turns those counts into a number that says how mixed the groups
are; a good split makes the two sides much less mixed than the parent.

---

## 4. What is stored on the GPU

Set up once at the start of `fit()` and kept for the whole training run:

| Buffer | What it holds |
|---|---|
| `d_features` | Every feature value of every row, stored **feature by feature** (all of feature 0, then all of feature 1, ...). |
| `d_classIds`  | The class (label as a small integer) of every row. |

Two important memory choices:

- **`d_features` uses 32-bit `float`, not 64-bit `double`** (see the `GpuValue`
  alias). This halves the memory and makes sorting about twice as fast. For data
  like SUSY it does not change the resulting tree. (One comment in the code says
  how to switch back to `double` if you ever need an exact match with the CPU.)
- The full feature matrix lives on the GPU the entire time, so we never re-upload
  it. Each node only sends the GPU a small list of which rows it contains.

Then there is **scratch** memory that is reused by every node (it is sized for
the biggest node, the root, and never shrinks, so we don't keep re-allocating):

| Buffer | What it holds |
|---|---|
| `d_currentRows`  | The row numbers in the current node (`uint32`, since datasets have far fewer than 4 billion rows). |
| `d_values` / `d_valuesSorted`   | The current node's feature values, before / after sorting. |
| `d_rowIds` / `d_rowIdsSorted`   | The matching row numbers, before / after sorting. |
| `d_temp`         | Temporary space the sort library needs. |
| `d_candidates`   | The best split found for each feature. |
| `d_tileHist`, `d_featureTotals`, `d_blockBest` | Small helpers for the "large node" path (Section 6). |

---

## 5. What happens for ONE node (`findBestSplitAtNode`)

This is the heart of the file. Given the node's list of row numbers, it does
four steps.

### Step 1 — Gather (`gatherNodeFeaturesKernel`)

We copy the node's rows out of the big feature matrix into the compact scratch
arrays, laid out one feature after another. Each GPU thread handles one
`(feature, row)` pair: it writes that row's feature **value** (the thing we will
sort by) and the **row number** (so we remember where each value came from).

We deliberately do **not** copy the class here. Instead, later code looks up a
row's class from the small `d_classIds` array using the row number. Why? Because
it keeps the sort dealing with just one value + one row-number per element, which
lets us use the fastest kind of sort.

### Step 2 — Sort (one call to CUB's segmented radix sort)

`CUB` is NVIDIA's library of ready-made fast GPU building blocks. We ask it to
sort, **in a single launch**, every feature's slice by value. The trick is
"segmented" sorting: we tell CUB "treat each feature's chunk as its own segment,"
so all 18 features get sorted independently but in one go. The row numbers ride
along with their values, so afterwards `d_rowIdsSorted` tells us the sorted order
of rows for each feature.

(We use a `DoubleBuffer`, which just lets the sort flip-flop between the two
buffers we already own instead of allocating a big extra one — that saved a lot
of VRAM.)

### Step 3 — Score every split (the interesting kernel)

Now each feature's rows are sorted. We sweep through them to find the best
threshold. To know the class counts on the left side at any point, we keep a
running tally as we move right.

The simple version (used for small nodes) is **one block per feature** and works
in three phases inside the block. Suppose the block has 256 threads:

1. **Histogram.** Split the feature's sorted rows into 256 equal chunks, one per
   thread. Each thread counts how many rows of each class are in its chunk.
2. **Prefix sum.** Add up those per-chunk counts so each thread learns how many
   rows of each class come *before* its chunk. That is exactly the "left side"
   class counts at the start of its chunk.
3. **Sweep.** Each thread walks its own chunk. At every gap where the value
   changes *and* the class changes, it computes the split's score (using the
   running left/right counts) and remembers the best one it has seen. Finally the
   block compares all 256 thread-bests and keeps the single best for that feature.

The result — the best split per feature — is written to `d_candidates`.

> Why the three phases? A single thread walking 5 million rows would be painfully
> slow and waste the GPU. Splitting the work across 256 threads needs each thread
> to know its starting left-side counts, and the histogram + prefix sum is how
> they get that.

### Step 4 — Copy back and choose

We copy the per-feature best splits (`d_candidates`) to the CPU — a tiny copy,
just one entry per feature. The CPU then picks the overall winner using the exact
same tie-breaking rules as the pure-CPU backend (`chooseBestSplit`), so the GPU
and CPU agree on what "best" means.

Finally we copy back just the **sorted row numbers of the winning feature** so
the CPU can split the node into its left and right groups without doing any work
over again. (We copy only the row numbers — not the values, and we don't re-look
up classes — because that host-side bookkeeping used to be the slowest part of
the whole program.)

---

## 6. The "large node" speed-up (tiling)

Step 3's simple version uses one block per feature. With ~18 features that is
only 18 blocks. But the GPU has 36 processors (SMs), so half of it would sit
idle on the big nodes near the top of the tree — and those nodes have the most
rows, so they dominate the running time.

For large nodes we instead cut each feature's sorted slice into many **tiles**
(about 32k rows each) and give each tile its own block. Now there are hundreds
of blocks and the whole GPU is busy. The catch is the same as before: a tile
needs to know the class counts of every row *before* it. So the large-node path
runs four small kernels:

1. `tileHistogramKernel` — each tile counts its own classes.
2. `tilePrefixKernel` — add up tile counts so each tile learns the counts before
   it (its starting "left" side) plus the totals for the whole feature.
3. `tileScanKernel` — each tile sweeps its rows (same three-phase idea as above,
   just starting from its tile's counts) and reports its best split.
4. `reduceTilesKernel` — combine each feature's tile-winners into one best split.

Small nodes skip all this and use the single-block kernel from Step 3, because
launching four kernels is not worth it when there is little data. The code picks
the path automatically based on how many rows the node has.

---

## 7. How the pieces map to the code

| Concept | Where to look in `tree_cuda.cpp` |
|---|---|
| Per-node entry point | `TreeCuda::findBestSplitAtNode` |
| One-time setup / upload | `TreeCuda::fit` |
| Scratch allocation (grows as needed) | `TreeCuda::ensureNodeScratch` |
| Step 1: gather | `gatherNodeFeaturesKernel` |
| Step 2: sort | the `cub::DeviceSegmentedRadixSort` calls |
| Step 3: score (small nodes) | `scoreSplitsKernel` |
| Step 3+: score (large nodes) | `tileHistogramKernel`, `tilePrefixKernel`, `tileScanKernel`, `reduceTilesKernel` |
| Impurity / scoring math | `deviceImpurity`, `deviceScoreCandidate` |
| Tie-breaking (which split is "better") | `deviceIsBetterMaxGain`, `deviceIsBetterC45` |
| Cleanup of GPU memory | `TreeCuda::releaseCudaState` |

---

## 8. The few rules that keep GPU and CPU in agreement

1. The GPU scoring math mirrors the CPU functions exactly (`deviceScoreCandidate`
   matches `TreeBase::scoreCandidateFromCounts`).
2. The "which split wins" comparison matches the CPU's rules, so ties are broken
   the same way.
3. The CPU still drives the recursion and partitions the rows, so the GPU only
   ever answers the narrow question "best split for this list of rows?".

The one intentional difference: feature values are `float` on the GPU. On
continuous data this produces the same tree as the CPU; on data with many exact
ties it can produce slightly different (but equally accurate) trees.

---

## 9. Building it

The GPU code must be compiled with NVIDIA's compiler, `nvcc` (a normal C++
compiler does not understand `__global__`, `<<<...>>>`, or the CUB headers).
See `build_cuda.sh` (or the "Build tree" task in `.vscode/tasks.json`). Your code
editor's linter may underline CUDA keywords as errors — that is expected and
harmless; only the `nvcc` build matters.
```
./build_cuda.sh        # produces the ./tree executable
./tree
```
