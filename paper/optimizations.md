# Why it is fast: design and optimizations

Code: `src/build/gpu_grower.cu`. Design doc: [docs/CUDA.md](../docs/CUDA.md).
Pseudo code: [pseudocode.md](pseudocode.md).

## The core idea

Each feature is **presorted once** into 8-byte entries `(float value,
row << b | class)`, so the label travels with the value and no kernel gathers
labels. The tree grows **level-synchronously**: all large nodes of a level are
processed by one set of kernel launches, and every node's rows stay
**contiguous in every column** (the partition is stable). Work is cut into
**tiles** of 2048 consecutive entries of one (node, feature) range, a
*segment*; one 256-thread block per tile.

A level therefore costs about **three streaming passes over `8·n·F` bytes**:
the sweep (read), the partition (read + write), i.e. ~24 bytes per value per
level. There are no row-index gathers, no histograms, no binning step, and the
cost grows with rows, not with nodes × bins.

Histogram libraries (cuML RF, XGBoost `hist`), as far as their published
design goes, quantize first and then build per-node class × bin × feature
histograms through row-index lists: gathered reads, atomics, and a per-node
cost that does not shrink with the node. That hurts at depth 19, where trees
have 300K–650K nodes. **This description of cuML's internals is an inference;
confirm it with an `nsys` profile of cuML before stating it as fact.**

## Optimizations, roughly by importance

The measured effects come from different commits and machines (sources in
[docs/PERFORMANCE.md](../docs/PERFORMANCE.md) and [docs/CUDA.md](../docs/CUDA.md));
fine for a narrative, but the paper needs a clean ablation.

| # | Optimization | Measured effect |
|---|---|---|
| 1 | **Presorted, label-packed columns, level-synchronous growth.** Every access coalesced; all nodes of a level share one set of launches; ~24 B per value per level. | The core design |
| 2 | **Single-pass stable partition with decoupled look-back** (as in CUB's scans; self-validating 64-bit tile states, no fences; tiles taken from an atomic counter so a tile only waits for tiles already running). It also counts the **next level's tile class histograms and cut counts** while writing, so below the root the histogram pass disappears. | Partition at ~1.2 TB/s on the H100 PCIe (peak ~2 TB/s) |
| 3 | **Goes-left flags as a bitmap**: one bit per row, set by `atomicOr` for left rows only; 14 MB for 115M rows stays in L2 (a byte per row, 115 MB, did not) | H100 airline CART build 1.73 → 1.09 s |
| 4 | **Warp-ballot class prefix counts**: `left[k] = before[k] + popc(ballot(cls == k) & lanesBelow)`, neighbours via `__shfl_up/down_sync`. No thread walks entries one by one; no shared-memory scan; no atomics. | Sweep is bandwidth-bound |
| 5 | **Convexity bound per tile (2 classes)**: the gain is convex in the left class counts, a tile's cuts lie in the box between the counts before and after it, so their gains are at most the max over the box's 4 corners. Tiles below the exact gain of another cut of the segment (minus 1e-8) are skipped. (`tileBoundKernel`) | Skips most tiles in big nodes; used on HIGGS and airline (binary) |
| 6 | **Boundary-point rule** (Fayyad–Irani style): a cut between two same-class entries whose neighbours are singleton values cannot be optimal and is not scored (`isSkippableCut`) | Fewer gain evaluations |
| 7 | **One host round trip per level.** A node's class counts come from its parent's split (winning feature's left counts), so leaves (pure, too small, depth limit) are decided on the host before any launch; only nodes that search a split enter a level. | Fewer syncs |
| 8 | **CPU/GPU hybrid**: children below `--gpu-min-rows` (CART rows/8192 in [1024, 2048]; C4.5 rows/4096 in [1024, 4096]) are gathered into one block, copied through a 512 MB device ring on a second stream, and a host callback (`cudaLaunchHostFunc`) starts CPU subtree tasks; the GPU goes on with the next level. A level of thousands of small nodes costs the GPU about as much as one of big nodes. | Ampere C4.5 build 3.3 → 5.9 s if forced to 512 rows |
| 9 | **Mixed precision with a proven error bound** (consumer GPUs only, FP32:FP64 > 4:1): float estimate of every cut, exact double `cutGain` (the CPU's function) only for cuts within a proven bound of the segment's best; ≤ 8 candidates kept per tile and scored by one warp. Closed-form Gini `Σ_k (l_k·n − t_k·n_L)² / (n²·n_L·n_R)` with differences exact in int64, so the error is relative to the gain, not to n. Entropy summed as `c·log2(m/c)` terms. | Laptop HIGGS exact pass 260–290 → 15 ms (CART), 55 ms (C4.5); CART overflow sweeps 130 → 0.2 ms |
| 10 | **Setup**: upload through four pinned 32 MB staging buffers on the GPU's NUMA node instead of page-locking the data (slow under HMM); CUDA context created while the CSV loads; C4.5 distinct values via `cub::DeviceSelect::Unique` on the device instead of copying all sorted values back | 6 GB in 0.2 s vs 0.35 s pinned (H100 node); A100 node upload 0.5 → 0.3 s with NUMA placement; C4.5 presort 0.54 → 0.09 s |
| 11 | **Column slots** (F + 1) instead of two ping-pong copies: the partition moves each column into a free slot and waits per segment for the column that left it | Airline VRAM 35 → 13.3 GiB, HIGGS 7.2 → 3.4 GiB, 0–3% slower |
| 12 | **Stream-ordered allocation** (`cudaMallocAsync` with a pool that keeps its memory) for per-level arrays that change size | No device syncs on resize |

## Correctness / determinism (worth a paragraph)

- Trees are **byte-identical** across the serial CPU, parallel CPU and CUDA
  backends, and across H100, A100 and the laptop GPU (`tests/check.sh`).
- Near-ties (gains closer than `kTieEps = 1e-12`) go to the earlier cut / lower
  feature index; `c·log2(c)` comes from a shared table, so CPU and GPU agree
  bit for bit.
- The float filter is conservative by construction: cutoff
  `best − (best·σ + μ)` with σ, μ covering 2B + ε_tie, B a proven bound on
  |float estimate − exact gain| (derivation in the comment above
  `floatGainFilter` in `gpu_grower.cu`).

## Engineering traps (good for a "lessons" section)

- Keeping tile values live in registers after the sweep: 52 → 74 registers,
  3 instead of 4 blocks/SM, +15% time; reloading from cache is faster.
- Tiles ordered column by column instead of node by node: partition −10%.
- A 64-bit field in a struct indexed by side pushed it to local memory: +5%.
- A move record read by every thread (instead of thread 0): +3%.
- More GPU levels (smaller `--gpu-min-rows`) is slower: the CPU threads finish
  small subtrees faster than a GPU level of tiny nodes.
- Tile sizes of 4 or 16 entries per thread instead of 8: slower.
