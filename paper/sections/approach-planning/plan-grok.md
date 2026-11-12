# Section 3 “Our Approach” — writing plan (Grok)

Plan only. Do not draft `approach.tex` from this note without checking the cited files again. Names in the paper are the macros in `paper/main.tex`:

| Macro | Paper name | Flag |
|---|---|---|
| `\ImplSerialBase` | Serial Simple | `--simple` |
| `\ImplSerialOpt` | Serial Optimized | `--serial` |
| `\ImplParallel` | Multicore CPU | `--parallel` |
| `\ImplGpu` | GPU-Based | `--cuda --gpu-only` |
| `\ImplHybrid` | Hybrid | `--cuda` (hand-offs on) |

`\ImplGpu` and `\ImplHybrid` are two modes of one CUDA grower (`src/build/gpu_grower.cu`). Say that once. The source enum calls the whole thing `Backend::Cuda`; the paper splits it into the two macros above.

Target when written: **about 1,550–1,700 words**, which is the middle of the requested 1,200–1,800 and about **2.5–3 LNCS pages** of prose. One optional schematic (column ranges, or the hybrid schedule) is worth it only if the prose is cut by ~100–150 words so the section still fits in three pages. No second figure.

Tone: design choices and why they exist. Not a file-by-file commentary, not a kernel catalogue, not a results section. CART/C4.5 rules stay at the level already in the introduction and related work.

---

## 1. Proposed structure

Keep the five implementation subsections. Add a short shared subsection so layout and scoring are not repeated five times. Titles below are suggestions; the macro names should appear in the implementation subsections.

1. **Shared search and column layout** (new; before the skeleton’s five)
2. **`\ImplSerialBase`** — the reference
3. **`\ImplSerialOpt`** — presort, sweep, one-time allocation
4. **`\ImplParallel`** — same algorithm, two schedules
5. **`\ImplGpu`** — level-wise growth on the device
6. **`\ImplHybrid`** — size-based hand-off, copies, and the only NUMA placement in the source

A section-opening sentence can name the five implementations and the identical-tree claim, then yield to (1). Do not give (1) a long “what is CART” lecture.

---

## 2. Per subsection

### 2.1 Shared search and column layout — 180 words

What to say:

- Scope in one or two sentences: numeric features, binary trees, exact cuts between consecutive distinct values. Stopping, impurity, and pruning already differ between CART and C4.5 and are not re-derived here. Missing values and multi-way categorical splits are out of scope (one clause).
- All five call the same host-side choice (`SplitRules`) and the same double-precision gain (`cutGain` in `split_math.h`, compiled for CPU and GPU). Gains within \(10^{-12}\) tie; the earlier cut, then the lower feature index, wins. Entropy of small counts uses one \(c\log_2 c\) table. That is why the backends can grow the same tree. Do not repeat the scikit-learn / Quinlan case analysis from the introduction.
- Input storage, once: column-major `float` features, labels as small integer ids (`dataset.h`).
- Optimized builders (everything except `\ImplSerialBase`) then keep, per feature, a stable value order of 8-byte entries `(value, row and class packed in 32 bits)`. A node is the same index range in every column. A stable partition of that range yields two sorted child ranges, so each feature is sorted once (SLIQ/SPRINT idea, already named in related work — cite, do not re-explain the papers).
- The grown tree is flat preorder arrays. Parallel creation uses a chunked store and a final renumbering so creation order does not change the tree (`node_store`, `tree`).
- Post-processing is shared (`trainer.cpp`): CART cost-complexity at a fixed alpha; C4.5 collapse, training-value thresholds, error-based pruning. Optimized growers can record each row’s leaf and the distinct values while building; say that here in one sentence and do not describe raising.

Files: `src/algo/split_math.h`, `src/algo/split_rules.*`, `src/core/dataset.h`, `src/core/tree.h`, `src/build/node_store.h`, `src/app/trainer.cpp`, `docs/ALGORITHMS.md` (ties; “not supported”), `README.md` (backend table).

### 2.2 `\ImplSerialBase` — 150 words

What to say:

- Textbook grower, one thread: at each node, for each feature, copy the node’s row ids, `std::sort` by that feature, score every admissible cut with `cutGain`, choose the feature, then build two new id vectors by comparing values with the threshold. Recurses in preorder.
- Same rules and the same pruners as the others. It does not presort, pack entries, estimate gains, skip boundary cuts, or drop constant features for a subtree. Its cost is repeated sorting and gathers through row ids, plus a heap vector per node and feature.
- Why it is in the paper: it is the sequential baseline the later implementations change, not a second algorithm. C4.5 distinct values and leaf routing are computed later, outside the grower (`simple_grower.cpp` leaves those outputs empty).
- One caveat, not a performance claim: recursion depth follows the tree. Not observed as a problem; do not dwell.

Files: `src/build/simple_grower.cpp`, `docs/SIMPLE.md` (rows 1–3, 15, 19–21, 23–24 only).

### 2.3 `\ImplSerialOpt` — 400 words

This is where the optimized sequential design is explained. `\ImplParallel` and the GPU modes should point back instead of repeating it.

What to say, as a few design choices:

- **Sort once.** LSD radix presort of every column (stable, float bits). Histograms first so expensive columns start early; a column whose keys share a digit skips that pass; large columns can be counted and scattered by several threads even in the one-thread *build*, because presort may use the pool — **check before writing**. `--serial` passes a null pool into the builder (`trainer.cpp`: serial and simple get `buildPool = nullptr`). Presort is called from `cpu_grower.cpp` with that same pool, so **`--serial` presort is single-threaded**. Do not say the serial presort is parallel. (Parallel presort belongs in 2.4.)
- **Why the tree is unchanged by the sort.** Cuts depend on values and class counts, not on order inside a run of equal values. Both sorts are ascending. The loader rejects NaN. The pack is lossless. State this in a clause, not a proof.
- **Sweep.** One left-to-right pass per feature. Class counts move with the entry, so the scan does not gather labels. Exact `cutGain` is not evaluated at every cut: a cheap incremental estimate (integer sums of squares for Gini; running \(c\log_2 c\) for entropy, refreshed on a stride) is within a small proven margin, and only candidates that can beat the best exact gain by the tie tolerance are settled, in order, by `cutGain`. Two-class data has a specialised sweep. On binary nodes of at least 64 rows, blocks of 16 cuts are skipped when a convex corner bound cannot win. Runs of equal values only update counts. A feature constant on the node (for C4.5, within \(10^{-5}\)) is dropped for the whole subtree. A restricted same-class boundary test can skip a cut; see risks before describing it as “Fayyad & Irani”.
- **Partition.** Children are the first `leftCount` entries of the winning column (the threshold is the double midpoint of two floats that are strictly apart, so a value comparison is unnecessary). Other columns are stably partitioned with a goes-left mark. Nodes of at least \(2^{15}\) rows move out of place into one spare slot, each entry once, non-temporal stores, a bit per row rather than a byte. Smaller nodes partition in place through a reused buffer. If both children are already leaves, the partition is skipped; if one is, only the other side moves. Child class counts come from the winning sweep.
- **Memory.** Columns live in \(F+1\) slots of \(N\) entries (8 bytes), allocated once: \((F+1)\cdot 8N\) plus the bitmap and, for \(N \ge 65536\), full-length \(c\log_2 c\) and \(1/c\) tables (\(16N\) bytes). Blocks of at least 2 MiB ask for transparent huge pages (`memory.cpp`) because the columns are large and later touched at arbitrary rows. No per-node heap traffic in the hot path. An explicit stack replaces recursion.
- **Why, in one breath:** the asymptotic win is not sorting again; the constant-factor win is sequential traffic, fewer exact gains, and not moving columns that will not be read.

Do **not** quote per-item speed-ups. `docs/SIMPLE.md` says none of the 29 diffs was measured alone.

Files: `docs/CPU.md` (presort, sweep, constant features, memory), `docs/SIMPLE.md` (the diff table and “why the tree is the same”), `src/build/presort.h`, `src/build/cpu_builder.h` (the `Columns` comment), `src/build/cpu_grower.cpp` (`spareSlots`: serial returns 1), `src/build/cpu_sweep.h` (bounds; read the comment if the shortcut paragraph is kept), `src/core/memory.cpp`, `src/app/trainer.cpp` (null pool).

### 2.4 `\ImplParallel` — 240 words

Deltas from `\ImplSerialOpt` only. Same columns, same sweep, same out-of-place rule, same one-thread algorithm when the pool is null.

What to say:

- One pool, two kinds of work, chosen by node size (the mix related work already attributes to FastFlow-style schedulers — one sentence of continuity, then our thresholds).
- **Data parallel** on nodes with at least 32,768 rows: features of the sweep in parallel, and one column per thread in the partition. Below that, waking the pool costs more than the scan. The calling thread takes part of a `parallelFor` and only waits for work already running, so a parallel scan nested inside a subtree task cannot deadlock the pool (`thread_pool.h`).
- **Task parallel** after a split of a node with at least 4,096 rows: the left child is a task, the current thread continues with the right child on an explicit stack. No join until the grower waits once at the end.
- **Slots.** Spare slots are \(\min(F-1, \text{threads})\) so a large node can move every non-winning column at once. Fewer slots save \(8N\) bytes each and leave threads idle; serial keeps one spare. Nodes of at least \(2^{20}\) rows, when there are fewer free slots than threads, are partitioned in blocks by all threads instead of one column per thread. Two-class children of at least 4,096 rows can be swept while the parent’s out-of-place partition writes them, so those columns are read once at that level. These nodes are bandwidth-bound; that is the reason for streaming stores and for dividing instead of consulting the big reciprocal table on large Gini sweeps.
- **Presort** uses the same pool: per-feature parallelism, and several threads inside one wide column when there are fewer features than threads.
- **NUMA — what is actually implemented.** Nothing that binds threads or pages to a socket. The pool inherits the process affinity mask (`taskset`, `numactl`, a job scheduler); thread count 0 means “CPUs this process may run on”, not “all CPUs in the machine” (`options.cpp`). Large arrays are first-touched by whoever writes them. Huge pages are a TLB hint, not a placement policy. **Do not describe a NUMA scheduler, interleaved allocation, or socket-aware stealing.** One-socket versus two-socket speed belongs in the experiments.

Files: `docs/CPU.md` (“Parallelism”, “Memory”), `src/core/options.h` (`nodeTaskRows` 4096, `featureParallelRows` 32768), `src/core/thread_pool.h`, `src/core/options.cpp` (`threadCount`), `src/build/cpu_grower.cpp` (`spareSlots`), `src/build/cpu_builder.cpp` (`kOutOfPlaceRows` \(2^{15}\), `kBlockedRows` \(2^{20}\)).

### 2.5 `\ImplGpu` — 340 words

Same logical algorithm as `\ImplSerialOpt`: presorted entries, one sweep per feature, stable partition, constant features dropped, exact `cutGain` for any cut that can win. Organisation changes so one launch covers many nodes.

What to say:

- **Breadth-first levels** of every node still on the device. The CPU builder is depth-first, one node at a time. A level is one set of launches, not one launch per node.
- **Tiles.** Work is 2,048 consecutive entries of one (node, feature) range (256 threads, 8 entries each). A warp takes 256 of those with ballot/popcount class counts and shuffles for neighbouring values; threads do not walk entries one by one.
- **Device memory.** One copy of the columns in \(F+1\) slots (8 bytes per entry, slots rounded up to 2 MiB in the CUDA path). The partition writes a column into a free slot; the node’s range in that slot is only that node’s. A 512 MB device ring exists for hand-offs; under `\ImplGpu` it is unused for subtree copies because nothing is handed off and the host arena is not allocated. Say “same slot idea as the CPU, on the device” and point back to 2.3.
- **Presort on the device** with CUB’s stable radix sort into those slots, overlapped with the upload: a column starts sorting once its raw values have been staged, on a copy stream. Root class counts come from a device histogram, not a host pass over the labels.
- **Upload (both GPU modes).** Pageable source, four page-locked 32 MB staging buffers, threads copying the next chunk while the previous one is in flight. Do not page-lock the whole matrix. The NUMA placement of those buffers is specified in 2.6; mention here only that the upload path is shared, so the reader does not think pinning is hybrid-only.
- **Exactness of the sweep.** On GPUs whose double precision is not much slower than single precision (data-center parts; detected from the FP32:FP64 ratio, overridable with `--gpu-sweep`), every allowed cut is scored in double precision with the CPU’s `cutGain`. Otherwise a single-precision pass drops cuts that cannot win under a bound that covers the tie tolerance, and only the survivors are scored in double precision. Two-class tiles can also be skipped from a convex corner bound. The host receives, per level, each feature’s best cut and the class counts to its left — the level’s only decision round trip — and runs the same `SplitRules::choose` as the CPU. Leaves are closed without a partition.
- **Partition.** Stable, one pass, by position on the winning feature and by a one-bit goes-left map elsewhere (the smaller side is marked). While writing, the kernel counts the next level’s tile histograms, so that histogram kernel runs only at the root.
- **`\ImplGpu` specifically.** `--gpu-only` grows every remaining node on the device, down to leaves. CPU threads still stage copies and, for C4.5, prune. They do not grow subtrees. With `--threads 1` the build-side CPU work is one thread **except** the staged copies, which still use at least eight threads on the GPU’s node when that node is known (`gpu_host.cuh`, `kMinCopyThreads`). The comment in `options.h` that says the staged copies also collapse to one thread is **stale**; follow `gpu_host.cuh` and `docs/CUDA.md`.

Files: `docs/CUDA.md` (level loop, two-pass, memory, `--gpu-only`), `src/build/gpu_kernels.cuh` (`kTile`, `kTileBoundGap`), `src/build/gpu_host.cuh` (staging), `src/build/gpu_grower.cu` (setup, level, arena skipped when `gpu.only`), `src/core/options.h` / `options.cpp` (`gpuMinRows` returns 1 when `only`).

### 2.6 `\ImplHybrid` — 280 words

Deltas from `\ImplGpu`. This is the heterogeneous scheme the introduction claims.

What to say:

- After a level’s partition, children with fewer than \(m\) rows are gathered on the device (only columns that can still split) and copied to the host. The GPU continues with the large children. A host callback starts CPU subtree tasks when the copy completes; the level loop does not wait for those tasks. The CPU builder is the same one as `\ImplParallel`. The grower waits for the pool once, then assembles one tree.
- **\(m\).** `--gpu-min-rows`, or automatic: CART clamps \(N/8192\) to \([1024, 2048]\); C4.5 clamps \(N/4096\) to \([1024, 4096]\) (`gpuMinRows`). Say why in design language: a GPU level of many tiny nodes costs about as much as a level of a few large ones, so small subtrees are cheaper on the CPU pool, which can run while the next large level runs. Do not retune \(m\) with benchmark tables here; the experimental section owns the sensitivity.
- **Copies.** Host arena is pageable, sized for every column of every row (each row is handed over at most once). Downloads use two page-locked 128 MB buffers and two copy streams, alternating so one transfers while the other is copied into the arena. The device side is a 512 MB ring of four parts; the GPU waits only when it is a full ring ahead of the bus. The arena is touched in the background during the early levels so the copy does not fault pages; a hand-off waits only for the prefix it fills. Page-locking the arena itself was rejected because registering it stalled the driver (state as a design choice, no timing table).
- **NUMA, only what the source does.** If the GPU’s PCI NUMA node is known and is not the only node the process may use, the page-locked staging buffers are allocated on that node (placement follows the allocating thread) and the copy threads are pinned to that node’s allowed CPUs (`gpuNodeCpus`, `Staging::prepare` in `gpu_host.cuh`). Reason, from the comment and `docs/CUDA.md`: copies that ran on both sockets of the A100 machine moved data across the interconnect and cut transfer rate toward half of PCIe. If the node is unknown or the process is already confined to it, there is nothing to pin. This applies to **uploads and downloads of both `\ImplGpu` and `\ImplHybrid`**, not only to hand-offs. The hand-off arena is first-touched by the CPU pool and is **not** bound to the GPU node; the comment says those pages follow the threads that touch them. Do not call that interleaving or a mempolicy.
- **What the CPU still does under both modes:** staging, C4.5 pruning, and (hybrid only) the small subtrees. Optional one-clause: when device memory remains, C4.5 can sort rows by leaf on the GPU before pruning. Skip the removed `--gpu-prune` experiment.

Files: `docs/CUDA.md` (hand-off, ring, NUMA paragraph in Setup, `--gpu-min-rows`), `src/build/gpu_host.cuh` (lines 148–281), `src/build/gpu_grower.cu` (`prepareHost`, `uploadStaged`, `touchArena`, hand-off), `src/core/options.cpp` (`gpuMinRows`).

---

## 3. Shared prelude versus per-backend deltas

**Say once, in 2.1 (and not again):**

- Numeric, exact, binary CART/C4.5; shared `cutGain` and tie rule; shared pruning entry points.
- Column-major inputs; 8-byte packed entries; a node as one index range; stable partition instead of re-sorting.
- Preorder tree; chunked store so parallel allocation does not change node ids.
- Leaf-id recording and distinct values as an interface to pruning, not the pruning algorithm.

**`\ImplSerialBase` delta:** re-sort row-id vectors per node and feature; exact gain at every cut; no column slots; recursion; pruning reconstructs distinct values and routes rows.

**`\ImplSerialOpt` delta:** the whole optimized sequential machine (presort, estimates, bounds, slots, huge pages, explicit stack). This is the reference the later sections modify.

**`\ImplParallel` delta:** thresholds 32,768 (feature parallel, and the out-of-place width already used by serial) and 4,096 (subtree tasks); more spare slots; blocked partition when slots are scarce; parallel presort; nested `parallelFor`. No NUMA policy.

**`\ImplGpu` delta:** level-synchronous tiles, device slots, CUB presort overlapped with upload, float filter only when double precision is slow, one host round trip per level for the split decision, no CPU subtrees, no hand-off arena.

**`\ImplHybrid` delta:** size cutoff, asynchronous hand-off through the ring and pinned buffers, CPU pool concurrent with the next GPU level, arena pre-touch. NUMA pinning of **staging copies** is implemented here in the source but serves both GPU modes — write it in this subsection and cross-refer from 2.5.

---

## 4. What to omit

- Re-teaching Gini, entropy, gain ratio, MDL, cost-complexity, or subtree raising. Point at the introduction and related work.
- The 29-row diff in `docs/SIMPLE.md` as a list. Group into the choices in 2.3.
- Per-optimization timings, ablation gaps, and any number from `docs/PERFORMANCE.md` or `bench/`. The section may say *why* a threshold exists (“waking threads dominated the scan below this size”) without quoting milliseconds.
- Kernel names (`tileEstimateKernel`, and so on). “One block per tile”, “warp ballot”, “single-pass partition” are enough. The CUDA pseudocode in `docs/CUDA.md` is a source map for the authors, not paper prose.
- Micro-experiments that were tried and removed (register-cached tiles, page-locking the arena in chunks, GPU subtree raising, packed integer upload, code-alignment flags, `atomic::wait` hangs).
- `--profile`, `--repeat`, energy, holdout seeds, CSV dialect, except the column-major layout.
- Two-socket speed-up curves, first-touch accidents on SOREL, dell thread-placement noise. Those are experimental facts. The loader’s dropping of the CSV from the page cache (`dataset.cpp`) is a benchmark-hygiene fix, not part of induction.
- Byte counts of particular datasets (HIGGS 3.4 GiB, airline 13 GiB) unless a single formula example is needed. Prefer \((F+1)\cdot 8N\).
- Proofs of the rounding margin, convexity, and `maxGainStep`. One sentence: a cut is discarded only when the bound says it cannot change the winner under the tie rule; the accepted cut is always an exact `cutGain`.
- The five covertype nodes where double precision disagrees with Quinlan’s single precision (`docs/ALGORITHMS.md`). The introduction already states the reference check. This section’s claim is agreement **across our backends**.

---

## 5. Open questions and risks

1. **Do not invent CPU NUMA features.** Grep of `src/` shows socket placement only in `gpu_host.cuh` (GPU staging buffers and copy-thread affinity) plus comments about first-touch. `memory.cpp` is huge pages. `dataset.cpp` mentions a NUMA node only as a reason to drop page cache. Multi-socket runs in `bench/` are job-script bindings, not builder logic.
2. **Arena versus staging.** The arena is first-touched by the pool and may lie on several nodes. Only the small pinned staging buffers are placed on the GPU node. A sentence that says “we allocate the training columns on the GPU’s NUMA node” would be false.
3. **Stale comment.** `options.h` says `--gpu-only --threads 1` makes staged copies single-threaded. `Staging::kMinCopyThreads` is 8, and `docs/CUDA.md` matches the code. Write the code.
4. **Boundary-point shortcut (Risk A in `docs/SIMPLE.md`).** The current test skips a cut between two singleton equal-class values when neighbouring cuts are allowed. An earlier, broader skip did change trees. The remaining argument is stronger than a strict `>` but is not a formal guarantee against a three-way tie inside \(10^{-12}\). No golden mismatch is known. In the paper, either omit the shortcut’s name or say the scan may skip a cut only when that test says it cannot be the selected one, and that the implementations were checked against each other. Do not cite Fayyad & Irani as a theorem we fully apply; related work already states the classical result, and our test is narrower. `\ImplSerialBase` does not use the shortcut.
5. **Estimate bounds (Risk B).** Comments in `cpu_sweep.h` argue the margin. There is no adversarial unit test. The safe paper sentence is the one in section 4 (exact `cutGain` decides; estimates only reject). Do not write “provably bit-identical” as a theorem.
6. **Serial presort parallelism.** Easy to over-claim from `docs/CPU.md`’s “all threads” paragraphs. Those paths run when a pool is passed. `--serial` does not pass one (`trainer.cpp`, `cpu_grower.cpp`).
7. **“CPU counts the classes”** in the README’s `--gpu-only` sentence. Root counts on the CUDA path are a device histogram (`docs/CUDA.md`). The CPU counts classes while loading, and inside CPU subtrees. Do not say the host histograms the root for `\ImplGpu`.
8. **Identical to reference implementations.** Across `--simple`, `--serial`, `--parallel`, and `--cuda`, `tests/check.sh` requires the same trees. Agreement with scikit-learn and Quinlan has documented tie and rounding exceptions in `docs/ALGORITHMS.md`. This section should claim identical trees among our five implementations. Leave the external caveats where the introduction put the external claim.
9. **Figure or not.** The hybrid schedule is the contribution and is already summarised at the end of related work. A figure helps if it shows level-wise GPU nodes plus a size cutoff to the CPU pool. Without a page budget for it, prose in 2.6 is enough. Do not draw a kernel pipeline.
10. **Depth of the sweep paragraph.** The failure mode is an annotated walk through blocks of 16, lanes of 4, and the 256-cut refresh. If 2.3 exceeds ~400 words, cut the block size and the refresh stride before cutting presort or the partition rule.
11. **C4.5 `tries` on skipped CART blocks** (`docs/SIMPLE.md` Risk C) is harmless because CART does not use `tries`. Not worth a sentence.
12. **Shared \(c\log_2 c\) table versus full-length CPU tables.** GPU uses the 65,536-entry table plus `log2` above it; large CPU sweeps use tables up to \(N\). Both still finish with `cutGain` for the winner. Do not claim every intermediate entropy estimate is bit-identical; claim the selected cut is.

---

## 6. Estimated length

| Subsection | Words |
|---|---:|
| 2.1 Shared search and column layout | 180 |
| 2.2 `\ImplSerialBase` | 150 |
| 2.3 `\ImplSerialOpt` | 400 |
| 2.4 `\ImplParallel` | 240 |
| 2.5 `\ImplGpu` | 340 |
| 2.6 `\ImplHybrid` | 280 |
| **Total** | **1,590** |

Range if the sweep paragraph is tightened or a figure caption is added: **1,450–1,700**. That stays inside 1,200–1,800 and should set as roughly 2.5–3 LNCS pages. If a schematic is included, hold the body near 1,500 words.

Order of writing later, if useful: 2.1 and 2.3 first (they define the vocabulary), then 2.4–2.6 as diffs, and 2.2 last so the baseline is described against that vocabulary without turning into the 29-row table.
