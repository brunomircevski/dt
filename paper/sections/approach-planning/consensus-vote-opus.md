# Section 3 — consensus vote (Opus)

Plans reviewed: `plan-opus.md` (mine), `plan-gptsol.md`, `plan-muse.md`,
`plan-grok.md`. I re-checked the disputed facts in the source before
voting (see "Facts re-checked" at the end).

---

## 1. Scores of the peer plans (1–5)

| Plan | Structure | Fact caution (esp. NUMA) | Completeness vs request | Length realism | Mean |
|---|:-:|:-:|:-:|:-:|:-:|
| GPT-Sol | 5 | 4 | 4 | 5 | 4.5 |
| Muse | 5 | 4 | 5 | 3 | 4.25 |
| Grok | 4 | 5 | 4 | 4 | 4.25 |

**GPT-Sol.** It has the cleanest structure: a shared prelude, then five
delta subsections with descriptive titles. Its NUMA boundary is drawn
correctly. GPU staging buffers and copy threads are placed on the GPU's
node. The CPU uses the affinity mask and first-touch, with no pinning. It
also correctly says that the hybrid arena is first-touched in parallel.
The budget (1,450–1,650 words) is realistic.

It is thin in two places. It does not mention that the upload overlaps
the presort, or that the root's class counts come from a device kernel.
It also does not say that the hand-off threshold under `--gpu-only` is
effectively 1 row.

It puts a little too much C4.5 post-processing into §3.1 (collapse and
training-value thresholds by name). The experiments run without pruning,
so one sentence is enough.

**Muse.** It is the most complete plan. Its allocation map is excellent.
Its NUMA scoping statement ("present as a scoping statement, not a
feature") is exactly the right framing. Muse and Grok both correctly note
that `--gpu-only` gives a threshold of about 1.

It has three weaknesses:
* The bullets would overrun the stated 1,600 words when written. They
  include the ring and buffer sizes, the overlap of CUDA context creation
  with loading, and the GPU `arrangeRows` clause. Realistically this is
  about 2,000 words.
* It uses "proven float-error bound" in the §3.5 bullets while flagging
  the same wording as a risk elsewhere.
* Context-creation overlap is tooling, not design.

**Grok.** It has the best fact caution of the four. It caught three
things the others missed:
* `--serial`'s presort is single-threaded, because the pool is null.
* The `options.h` comment about `--gpu-only --threads 1` copies is stale
  (`kMinCopyThreads = 8`).
* Shared vs full-length \(c\log c\) tables: the claim is that the
  *selected cut* is identical, not every intermediate estimate.

It also correctly says the arena is *not* bound to the GPU's node.

Its structure is weaker on one point. It places the GPU-node staging
(NUMA) in the Hybrid subsection, although that mechanism serves both GPU
modes. Grok acknowledges this and adds a cross-reference, but placing it
in §3.5 is simpler. §3.3 at 400 words is dense with constants (blocks of
16, 64 rows, \(2^{15}\), \(16N\) tables) and risks becoming the annotated
walkthrough the author asked to avoid.

**Self-correction to my plan.** I wrote that \ImplGpu{} is "threshold 0".
The code returns **1** (`gpuMinRows` when `gpu.only`). The consensus below
uses the correct value.

---

## 2. Consensus structure I accept

All four plans agree on the backbone: a short shared subsection, then the
skeleton's five subsections as deltas. Each subsection covers what
changes and why, and why the tree stays the same. The CART and C4.5 rules
are not taught again.

**Target: about 1,600 words of prose plus one figure, about 2.8–3 LNCS
pages.** If the figure is dropped, the target is about 1,500 words.

### Lead-in (2–3 sentences, counted in 3.1)

* Five implementations, one tree. They differ on three axes (Muse's
  framing):
  1. how a node's sorted order is obtained;
  2. how the best cut is found;
  3. where (which thread or device) and in what order nodes are grown.
* There are four code paths. \ImplGpu{} and \ImplHybrid{} are two modes of
  one CUDA grower. Say this once.

### 3.1 Common Design: Exact Splits on Presorted Columns — ~230 words

* Data in memory:
  * column-major float32 features and small integer class ids;
  * for every implementation except \ImplSerialBase{}, per-attribute
    sorted columns of 8-byte entries (value plus row and class packed in
    32 bits), so a sweep never gathers labels;
  * a node owns the same index range in every column, and a stable
    partition keeps both children sorted. This is the SLIQ/SPRINT idea:
    cite §2, do not re-explain it.
* One double-precision scoring routine, compiled for CPU and GPU, and one
  tie rule:
  * gains within \(10^{-12}\) are equal; the earlier cut, then the lower
    attribute, wins;
  * a shared \(c\log_2 c\) table for counts below \(2^{16}\);
  * the same CART/C4.5 rule layer chooses the attribute on CPU and GPU.
* Every optimization later in the section may only *discard* cuts that
  cannot win under this rule. The cut that is selected is always scored
  by the exact routine. This is the identical-tree mechanism.
* The tree is stored as flat arrays in preorder, built from a concurrent
  node store and renumbered at the end, so creation order does not matter.
* Verification in one sentence: tests require byte-identical trees
  across our implementations. External agreement and its documented tie
  exceptions stay in the Introduction.
* Pruning in one sentence: it is shared CPU post-processing after growth,
  not used in the experiments. Optimized growers record each row's leaf
  and the distinct values for C4.5. Do not describe pruning internals.

### 3.2 \ImplSerialBase{}: Textbook Reference — ~140 words

* At each node and for each attribute: copy the row ids, comparison-sort
  them by value (gathering values through the ids), score *every*
  admissible cut exactly, apply the rule, and build two new id vectors.
  Recursion is depth-first.
* Cost: \(O(F\,n\log n)\) per node, random gathers, and heap allocation
  per node. Describe these as design properties, without measurements.
* Role: it is the baseline and the executable specification, and it
  shares the rules and pruning with the others. Say "straightforward"
  once and do not apologize for it.

### 3.3 \ImplSerialOpt{}: Presorting and a Filtered Sweep — ~380 words

Organize as three moves. For each one, say what changes, why, and why the
tree is unchanged (one clause each).

1. **Sort once, partition by position.**
   * Stable LSD radix presort of each attribute, once; single-threaded in
     this implementation.
   * At a split, the winning column's left rows are its first \(n_L\)
     entries. The other columns are stably partitioned by a goes-left
     mark.
   * The cost per level becomes \(O(F\,n)\).
   * Large nodes are partitioned out of place into a spare slot, with a
     bitmap and streaming stores. Small nodes are partitioned in place.
   * If both children are leaves, nothing moves.
   * An attribute that is constant in a node is dropped for its whole
     subtree.
2. **Exact search with a cheap filter.**
   * An incremental gain estimate with a bounded error. Only cuts that
     could still win are scored exactly, in order.
   * A two-class specialization with a convex block bound.
   * Runs of equal values are only counted.
   * The children's class counts come from the winning sweep.
   * Name these as groups, not 29 items. Do not mention the boundary-point
     test by name (see D4).
3. **Memory discipline.**
   * \(F+1\) column slots of \(8N\) bytes, allocated once.
   * Huge pages for large blocks.
   * An explicit stack instead of recursion.
   * The footprint is about \(8(F+1)\) bytes per row, compared with row-id
     lists in \ImplSerialBase{}.
* Close by saying that \ImplSerialOpt{} *is* the \ImplParallel{} builder
  without a pool.
* No speed-up attributions per item: `SIMPLE.md` says none was ablated.

### 3.4 \ImplParallel{}: Two Levels of Parallelism — ~260 words

* One thread pool. The node size decides the kind of parallelism (tie to
  Aldinucci's NAP in §2):
  * **≥ 32,768 rows:** the sweeps run in parallel over attributes, and the
    partition moves one column per thread. Below this size, waking the
    pool costs more than the work.
  * **≥ 4,096 rows:** after a split, the left child becomes a task and the
    thread continues with the right child. There is one join at the end.
  * Nested fork–join is deadlock-free: the caller works too and waits
    only for items already running.
* Spare slots \(=\min(F-1,\text{threads})\), so a big node can move all
  its columns at once. Very large nodes with too few free slots are
  partitioned in row blocks by all threads. Big nodes are limited by
  memory bandwidth, which is why the code uses streaming stores and sweeps
  two-class children while the parent is partitioned.
* The presort is parallel: per attribute, and also inside one column when
  there are fewer attributes than threads.
* The tree is byte-identical to \ImplSerialOpt{} because of preorder
  renumbering (§3.1).
* **Multiple sockets: a scoping statement, not a feature.**
  * There is no NUMA policy: no pinning, no `mbind` or interleaving, no
    per-socket data.
  * The thread count follows the process affinity mask, so
    `numactl`/`taskset` placement is respected.
  * Page placement follows Linux first-touch by the threads that write the
    arrays.
  * Huge pages reduce TLB and page-fault cost; they are not a placement
    policy.
  * Point forward to the scaling results.

### 3.5 \ImplGpu{}: Level-Synchronous Growth — ~340 words

* The same logical algorithm, organized breadth-first. All nodes of a
  level are processed by one fixed set of launches. Work is cut into
  2,048-entry tiles of (node, attribute) segments, with one block per
  tile. Contrast with CUDT's node-at-a-time approach.
* Device memory: one copy of the columns in \(F+1\) slots (about
  \(8N(F+1)\) bytes). This is why the wide datasets are subsampled
  (pointer to §4).
* Presort: CUB's stable radix sort, started per attribute as soon as its
  upload completes. The root's class counts come from a device kernel.
* The level pipeline as stages, without kernel names:
  1. constant segments are skipped;
  2. class counts before each tile;
  3. a two-class tile bound;
  4. a warp-level sweep using ballot/popcount and shuffles, with no
     serial scan;
  5. the best cut per segment, then **one host round trip per level**,
     where the host applies the same CART/C4.5 rule;
  6. a bitmap marks the smaller side;
  7. a single-pass stable partition that also produces the next level's
     tile counts.
* Exactness: data-center GPUs score every cut in double precision with
  the shared routine. GPUs with slow FP64 use a conservative float filter
  plus an exact rescore. One sentence, and note that the evaluated runs
  use the double-precision path.
* **Host–device communication and the only explicit NUMA placement**
  (applies to both GPU modes):
  * the pageable data is uploaded through four page-locked staging
    buffers, filled by CPU threads while earlier chunks transfer;
  * the staging buffers are allocated on the GPU's NUMA node (read from
    sysfs), and the copy threads are pinned to its CPUs;
  * nothing else is NUMA-bound.
* In \ImplGpu{} the hand-off threshold is 1 row, so every node stays on
  the GPU and no host arena is allocated. CPU threads still stage copies
  (at least 8 copy threads) and run C4.5 pruning.

### 3.6 \ImplHybrid{}: Assigning Nodes by Size — ~250 words

* The delta from §3.5: children below \(m\) rows leave the GPU and are
  grown by the \ImplParallel{} builder on the pool, while the GPU
  continues with the next level. Both devices add nodes to one tree.
* Why: a GPU level costs about the same for many tiny nodes as for a few
  large ones, while CPU threads finish small subtrees independently. This
  ties back to the two difficulties named in the Introduction.
* \(m\) is configurable. The automatic default scales with \(N\), clamped:
  CART \(N/8192\) to [1024, 2048], C4.5 \(N/4096\) to [1024, 4096]. Size
  is the only criterion (there is no cost model). The defaults were chosen
  empirically.
* The hand-off path:
  * a gather kernel copies only the still-splittable columns into a
    bounded device ring;
  * asynchronous downloads go through two page-locked buffers into a
    pageable host arena;
  * a host callback on the copy stream starts the CPU tasks;
  * the GPU stalls only when it is a full ring ahead of PCIe.
* The arena is pre-touched by all threads in the background, so copies do
  not page-fault. Its pages follow the touching threads; it is **not**
  placed on the GPU's node.
* One sentence: the host arena costs \(8NF\) bytes.

### Figure (one, about 1/3 page)

A hybrid level step. On the GPU: large nodes in tiles, sweep, host
decision, partition. Small children go through the ring, the pinned
buffers and the arena to CPU pool tasks, while the GPU proceeds to the
next level. An optional left mini-panel shows a node's range across two
presorted columns. There is no separate overview table: the lead-in's
three axes do that job.

### Omit (consensus of all four plans)

* Benchmark numbers and per-optimization speed-ups.
* History and failed or removed attempts (GPU subtree raising, register
  caching, page-locking the arena).
* Kernel names.
* Small constants (256-cut refresh, 4 lanes, skip lengths, K=3..8).
* Pruning internals.
* `--profile`, `--repeat`, loader page-cache handling, and CUDA context
  overlap.
* PERFORMANCE.md's NUMA probes and placement experiments.

---

## 3. Disagreements still to resolve (max 5)

**D1. Where GPU-node staging (the NUMA mechanism) is described.**
* Grok: in \ImplHybrid{}. GPT-Sol, Muse and I: in \ImplGpu{}.
* **Resolution:** in §3.5. It serves both GPU modes, including uploads,
  so it belongs where the upload is described. §3.6 then mentions only
  that the arena is first-touched, not GPU-bound.

**D2. Overview table, figure, or algorithm box.**
* Mine: table plus an algorithm box. Muse: a two-panel figure. Grok: an
  optional schematic. GPT-Sol: none.
* **Resolution:** one figure for the hybrid level step, with an optional
  column-range mini-panel. No table and no pseudocode box: the stage list
  in §3.5 replaces the algorithm box, and the three-axis lead-in replaces
  the table. If space allows only text, drop the figure and keep about
  1,500 words.

**D3. Total length.**
* Proposals ranged from 1,450 (GPT-Sol) to 1,900 (mine).
* **Resolution:** about 1,600 words with the figure (the budgets above add
  up to about 1,600). §3.3 is capped at 380 words. If it runs over, cut
  sweep constants before the presort or partition content (Grok's rule).

**D4. The boundary-point shortcut and "proven" bounds.**
* All four flag the risk. Muse still writes "proven" in its bullets.
* **Resolution:** do not name Fayyad–Irani as applied in our code. Write
  "conservative bounds that only discard cuts unable to win under the tie
  rule", and never write "provably identical". Claim only that the
  *selected* cut is scored exactly and that trees are verified identical
  across our implementations.

**D5. How much C4.5 post-processing goes into §3.1.**
* GPT-Sol names collapse, thresholds, and raising. Muse and Grok give one
  sentence plus the grower-to-pruner data flow.
* **Resolution:** one sentence (pruning is shared CPU post-processing and
  the experiments do not prune) plus one clause saying that growers record
  each row's leaf and the distinct values. Revisit only if the Results
  report pruned timings.

Settled without dispute: the 8-byte entry goes in §3.1 (three of four;
I withdraw my preference for §3.3). State the four-backends/five-paper-names
mapping once. Use only the "conservative bound" wording for the GPU float
filter.

---

## Facts re-checked for this vote

* `gpuMinRows` returns **1** when `gpu.only` (`options.cpp`). Grok and Muse
  are right; my "threshold 0" was wrong.
* `--serial` passes a null pool to `CpuGrower`, and `presortColumns`
  receives that pool, so the serial presort is single-threaded
  (`trainer.cpp`, `cpu_grower.cpp`). Grok is right.
* The `options.h` comment ("With --threads 1 the CPU side is one thread
  (the staged copies too)") contradicts `gpu_host.cuh`'s
  `kMinCopyThreads = 8`, which is used whenever the CPUs allow it. Grok is
  right that the comment is stale; follow the code. This is worth fixing
  in the source.
* The thread count comes from `sched_getaffinity` (`options.cpp`). The only
  `pthread_setaffinity_np` is in `gpu_host.cuh` (the staging copiers).
  There is no CPU NUMA policy, which all four plans agree on.
