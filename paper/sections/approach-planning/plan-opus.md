# Section 3 "Our Approach" — writing plan (Opus)

Status: plan only. Nothing in `approach.tex` was changed.
Sources read: `README.md`, `docs/{ALGORITHMS,SIMPLE,CPU,CUDA}.md`, the NUMA
parts of `docs/PERFORMANCE.md`, `src/build/{simple_grower,cpu_builder.h,
cpu_grower,gpu_grower.cu,gpu_host.cuh}`, `src/algo/split_math.h`,
`src/core/{dataset.h,thread_pool.h,options.*,memory.cpp}`,
`src/app/trainer.cpp`, and the Introduction, Related Work and Experiment
Design sections of the paper.

---

## 0. Guiding decisions

* **Narrative:** one design that is gradually optimized and parallelized.
  Each subsection says what changes compared with the previous version and
  why. Readers who want code can follow the repository link. No function
  names in the prose, except perhaps `cutGain` once (the shared scoring
  function). No file paths.
* **Shared material goes first, once.** The column layout, the scoring
  code, the tie rule and the identical-tree guarantee are common to four
  of the five implementations (\ImplSerialBase{} shares only the scoring).
  Stating them in a short common subsection keeps the per-implementation
  subsections to what is new in each.
* **Do not re-teach CART/C4.5.** Related Work §2.1 already defines Gini,
  gain ratio, cost-complexity pruning, error-based pruning and the
  Fayyad–Irani boundary points. Section 3 only *refers* to them.
* **Scope matches the experiments:** the protocol is depth ≤ 10 without
  pruning (experiment-design.tex, results.tex captions). Pruning gets one or
  two sentences, not a subsection.
* **No numbers from benchmarks** in Section 3 (they belong in Section 5).
  The only numbers are design parameters: tile size, thresholds and
  entry size.
* Use the macros `\ImplSerialBase{}`, `\ImplSerialOpt{}`, `\ImplParallel{}`,
  `\ImplGpu{}`, `\ImplHybrid{}`. Mention the command-line flags once, in the
  overview table or a footnote, not in running prose.

---

## 1. Proposed structure

```
3   Our Approach                                   (lead-in, ~110 words + Table)
3.1 Common Design                                  (~280 words)
3.2 \ImplSerialBase                                (~130 words)
3.3 \ImplSerialOpt: Presorted Columns and a Pruned Sweep (~400 words)
3.4 \ImplParallel                                  (~300 words)
3.5 \ImplGpu: Level-Wise Growth                    (~420 words + Fig./Alg.)
3.6 \ImplHybrid: Assigning Nodes by Size           (~280 words)
                                                   total ≈ 1,900 words
```

Change from the skeleton: one new subsection, **3.1 Common Design**, placed
before the five implementation subsections. The skeleton's order is
kept. 3.3, 3.5 and 3.6 have descriptive subtitles, so that the subsection
titles tell the story. If the author prefers bare titles, the skeleton's
names work too.

Alternative (not recommended): merge 3.5 and 3.6 into "GPU and Hybrid",
since \ImplGpu{} is the hybrid grower with the hand-off turned off
(`--gpu-only` sets the hand-off size to 0). Keeping them apart matches the
results tables, which have separate \ImplGpu{} and \ImplHybrid{} columns.
It also puts the hybrid, the main contribution, in its own subsection.

---

## 2. Per-subsection content

### 3 (lead-in) — ~110 words + Table "Implementations"

* There are five implementations. Each one builds on the previous one, and
  all of them grow the same tree.
* A compact table can carry most of the differences, which saves prose:

  | Impl. | Sorting | Split search | Parallelism | Where nodes grow |
  |---|---|---|---|---|
  | \ImplSerialBase | per node, per feature (comparison sort of row ids) | every cut scored exactly | none | CPU, 1 thread |
  | \ImplSerialOpt | once (radix presort), then stable partition | estimate + exact check on candidates | none | CPU, 1 thread |
  | \ImplParallel | same, presort parallel | same | features of big nodes + subtrees as tasks | CPU, all threads |
  | \ImplGpu | same, GPU radix sort | warp-level sweep, all nodes of a level at once | level × feature × tile | GPU (CPU decides splits per level) |
  | \ImplHybrid | same | GPU for big nodes, CPU sweep for small | GPU levels + CPU task pool, concurrently | GPU ≥ threshold, CPU below |

  The table is optional if space is short. It is worth about 150 words of
  prose.
* Sources: README table; `src/app/trainer.cpp` (backend dispatch).

### 3.1 Common Design — ~280 words

Shared prelude. What to say:

1. **Data in memory.** The dataset is column-major float32, one array per
   attribute. Class labels are mapped to small integer ids at load time.
   Source: `src/core/dataset.h` (`values[feature * rowCount + row]`,
   `uint16` labels).
2. **Presorted entries** (all implementations except \ImplSerialBase): each
   entry is 8 bytes, a float value plus a 32-bit word that packs the row id
   and the class (`row << classBits | class`). A sweep therefore reads the
   class without an indirect lookup. A node owns the *same index range* in
   every attribute's column. This is SLIQ/SPRINT's idea (cite back to §2)
   with the class inside the entry.
   Source: `split_math.h` (`Entry`, `EntryCodec`), CPU.md "Sort once".
   *Alternatively, introduce entries in 3.3 instead of here; see Open
   question Q1.*
3. **One scoring routine for all.** Gain is computed from weighted
   impurities. The scoring code sits in one header that both the C++ compiler
   and nvcc compile, so the CPU and the GPU score a cut with the same code.
   Thresholds are double-precision midpoints. Entropy uses a shared table of
   \(c\log_2 c\) for counts below \(2^{16}\).
   Source: `split_math.h`, ALGORITHMS.md "Ties and CPU/GPU agreement".
4. **Deterministic tie rule.** Gains within \(10^{-12}\) count as equal; the
   earlier cut wins, then the lower attribute index. Every optimization
   below is only allowed to *skip* cuts that provably cannot win under this
   rule. This sentence is the frame for the claim that the trees are
   identical.
5. **Tree output.** Nodes are stored in flat arrays in preorder, so the
   result does not depend on which thread or device created a node. Source:
   `core/tree.h`, CPU.md "Memory" last paragraph.
6. **Verification (one or two sentences, without re-teaching).** Every
   implementation is checked byte-for-byte against trees verified node by
   node against scikit-learn and Quinlan's C4.5. The Introduction already
   says this, so add only the mechanism: shared code plus the tie rule plus
   golden-tree tests. Source: ALGORITHMS.md, `tests/check.sh`.
7. **Pruning (one sentence).** CART cost-complexity pruning and C4.5
   error-based pruning run after growth with shared CPU code for all
   implementations. The experiments do not prune. Source: `trainer.cpp`.
   Optional clause: C4.5 pruning reuses each row's leaf, which the grower
   records. Probably omit.

Not here: the definitions of CART and C4.5, C4.5's MinSplit/MDL/1e-5 rules
(at most one clause pointing to the original program).

### 3.2 \ImplSerialBase — ~130 words

* This is the textbook algorithm and serves as the reference and baseline.
  At each node, for each attribute, it copies the node's row ids, sorts them
  by value with a comparison sort, scores *every* allowed cut with the
  shared routine, picks the attribute with the CART or C4.5 rule, and splits
  the row list into two new lists. Recursion is depth-first.
* Cost: \(O(F\,n\log n)\) per node. The sort reads values through row ids,
  that is, by random gathers into each column. This makes it bound by memory
  latency rather than bandwidth. The latency claim is supported by
  PERFORMANCE.md (SOREL --simple and NUMA); state it as a qualitative design
  property, not a measured one.
* It shares the scoring and stopping rules with the others, so it grows the
  same tree.
* Source: `simple_grower.cpp` (≈150 lines), SIMPLE.md intro.

### 3.3 \ImplSerialOpt — ~400 words

Present this as three groups of changes, following the "Differences" table in
SIMPLE.md. Do not reproduce all 29 items.

**(a) Sort once, then partition (largest change).**
* An LSD radix presort on the float bits orders each attribute once
  (stable; 11-bit digits, 3 passes; passes whose digit is constant are
  skipped). This makes each tree level cost \(O(F\,n)\) instead of
  \(O(F\,n\log n)\).
* Split: mark the rows of the winning attribute's left part, then
  stable-partition every other column of the node. The winning column
  needs no comparison, because its left rows are the first \(n_L\) entries.
  Both children again own one sorted range in every column.
* Large nodes (≥ \(2^{15}\) rows) are partitioned *out of place* into a
  spare column slot. Rows are marked in a bitmap that fits in cache, and
  the column is written with streaming (non-temporal) stores. Small nodes
  are partitioned in place through a per-thread buffer. If both children
  will be leaves, nothing is moved.
* Attributes that are constant within a node (first entry = last entry) are
  dropped for the node's whole subtree. This matters for one-hot/binary
  attributes.

**(b) Sweep without scoring every cut.**
* A cheap incremental estimate of the gain at every cut, computed without a
  loop over classes or divisions, with a bound on its error. Only cuts
  whose estimate could beat the best by the tie tolerance are scored
  exactly with the shared routine, in order. So the exact routine still
  decides every result.
* Two-class specialization. For nodes of ≥ 64 rows, blocks of 16 cuts are
  skipped when a bound at the corners of their class-count box (the gain
  is convex in the left class counts) cannot beat the best so far.
* A step bound skips cuts that cannot catch up (Gini gain rises by
  < 3/n per row moved).
* The Fayyad–Irani boundary-point rule (already in §2) is applied only
  where it is valid under the tie rule.
* The children's class counts come from the winning sweep. For two classes
  and children of ≥ 4096 rows, the children are swept *during* the
  parent's partition, so each column is read once per level instead of
  twice.

**(c) Memory management.**
* Column slots: \(F+1\) slots of \(8N\) bytes each, allocated once. There
  is no allocation in the hot loop. Large blocks use transparent huge pages
  (the global `operator new` is replaced). There is an explicit stack instead
  of recursion.
* Trade-off to state: \ImplSerialOpt{} needs about \(8F\) bytes per row
  more than \ImplSerialBase{}.

**Why the tree is unchanged** (one sentence): every shortcut only discards
cuts that provably cannot win or tie, and the exact routine decides in cut
order.

**Caveat for the writer:** SIMPLE.md states that *none* of these items was
measured on its own. The prose must not attribute speed-up shares to
individual items. Say "expected" or describe the effect in terms of
complexity. The intro's "up to 25×" is a total.

Sources: SIMPLE.md table and "Why the tree is the same"; CPU.md "Sort once",
"The sweep", "Constant features"; `presort.cpp`, `cpu_sweep.h`,
`cpu_builder.h`, `memory.cpp`.

### 3.4 \ImplParallel — ~300 words

* **Same algorithm on a thread pool.** \ImplSerialOpt{} is this code
  without a pool, so the two cannot diverge. Source: `trainer.cpp`
  (`buildPool = nullptr` for serial).
* **Two kinds of parallelism by node size**, which connects back to
  Aldinucci's NAP in §2:
  * Nodes of ≥ 32,768 rows: data parallelism over attributes for the sweep,
    and one column per thread for the partition. Below that size, waking
    threads costs more than it saves (state it qualitatively).
  * Nodes of ≥ 4,096 rows: after a split, the left child becomes a pool
    task and the thread continues with the right child. No thread waits for
    a task; there is one barrier at the end.
  * Nested fork–join inside tasks without deadlock: the caller works too
    and only ever waits for items that are already running.
  Source: `options.h` (`featureParallelRows = 32768`,
  `nodeTaskRows = 4096`), `thread_pool.h`, CPU.md "Parallelism".
* **Spare column slots** = min(F − 1, threads). A big node can then move
  all its columns at once, one per thread, each to a free slot. For very
  large nodes (≥ \(2^{20}\) rows) with fewer free slots than threads, all
  threads partition a column in blocks of 16K entries with a chained
  prefix of left counts. This is the main memory/speed trade-off of the
  CPU code. State it qualitatively; the numbers belong in Results or
  Discussion.
* **Parallel presort.** Attributes needing more passes are sorted first, to
  balance the threads. With fewer attributes than threads (wide-row
  datasets are the opposite case), each column is sorted by several
  threads with a stable parallel scatter.
* **Memory bandwidth.** Big nodes are bandwidth-bound. Streaming stores,
  fused child sweeps, and for Gini sweeps of ≥ 65,536 rows a division
  instead of a reciprocal table lookup (the table is as large as the
  training set).
* **Node storage and determinism.** Nodes are created in a concurrent store
  (chunks taken whole by a thread) and renumbered in preorder at the end,
  in parallel. So serial and parallel produce byte-identical trees.
* **Multiple sockets (state exactly what is there and no more):**
  * The CPU backends have **no explicit NUMA policy**: no `libnuma`, no
    `mbind`, no interleaving, no thread pinning in the pool. Placement
    relies on Linux first-touch. The large arrays (parsed columns, sorted
    columns, node chunks) are first written by the threads that later
    work on them, so pages end up spread over the sockets roughly as the
    work is. Huge pages reduce TLB misses and page-fault cost on these
    arrays.
  * The thread count defaults to the CPUs the process may run on, so that
    `numactl`/`taskset` can restrict a run to one socket. Source:
    `options.cpp` (sched_getaffinity).
  * Suggested sentence: "We rely on first-touch placement rather than an
    explicit NUMA policy" plus a forward pointer to the scaling results.
    Do **not** claim NUMA-aware scheduling.

Sources: CPU.md "Parallelism", "Memory"; `thread_pool.h`; `options.cpp`;
`memory.cpp`; PERFORMANCE.md (node store chunk taken "on its own socket",
no NUMA effect from `--membind` on dell).

### 3.5 \ImplGpu: Level-Wise Growth — ~420 words + one figure or algorithm

* **Breadth-first instead of depth-first.** All nodes of one level that
  need a split search are processed together by a fixed sequence of kernel
  launches. Work is cut into *tiles* of 2,048 consecutive entries of one
  (node, attribute) segment, and one thread block handles one tile. The
  launch count therefore does not grow with the number of nodes in a
  level. Contrast with CUDT (one node at a time, §2) and
  Mitchell–Frank/Strnad (level- or batch-wise).
* **Data in device memory.** The same 8-byte entries in \(F+1\) column
  slots, one copy. The presort uses CUB's stable radix sort, per attribute,
  as soon as that attribute's upload is complete (upload and presort
  overlap). Device memory needed ≈ \(8N(F+1)\) bytes, which is why the two
  widest datasets are subsampled (point to §4). The root's class counts are
  computed by a kernel.
* **Per-level kernel pipeline** (summarize in about 5 lines, or as
  Algorithm 1 adapted from CUDA.md):
  1. detect constant segments (they are skipped and dropped for the subtree);
  2. class counts before each tile (a prefix over per-tile histograms);
  3. two classes: a per-tile upper bound (the convex corner bound, now per
     tile) lets the sweep skip most tiles of big nodes;
  4. sweep: each warp takes 32 entries per step; the class counts left of
     each entry come from `__ballot_sync` + `popc`, and the neighbouring
     values from warp shuffles, so there is no serial scan and no atomics;
  5. best cut per segment; **one device-to-host round trip per level**: the
     host applies the CART/C4.5 rule (the same code as the CPU) and finishes
     children that are leaves;
  6. mark rows of the *smaller* side in a bitmap (≤ n/2 atomics), then a
     **single-pass stable partition** with decoupled look-back into a free
     slot, which also computes the next level's tile histograms (so step 2
     needs no extra pass below the root).
* **Exactness on the GPU.** Data-center GPUs (A100/H100, the ones
  evaluated) score every cut in double precision with the shared routine.
  On GPUs with slow FP64, a float pass with a conservative error bound
  filters the cuts, and only the survivors are scored in double. The mode is
  chosen from the device's FP32:FP64 ratio. One or two sentences; flag that
  the evaluated runs use the one-pass mode (auto). Ties are reduced in a
  fixed order. Result: the same tree as on the CPU.
* **Host–device communication and NUMA** (the one place with explicit
  NUMA handling):
  * Upload of pageable data at pinned-memory speed without pinning the
    dataset: CPU threads copy 32 MB chunks into 4 page-locked staging
    buffers while earlier chunks transfer on a separate copy stream.
  * On multi-socket hosts the staging buffers are allocated, and the copy
    threads pinned, on the **GPU's NUMA node**. The node is read from the
    PCI device's `numa_node` in sysfs; at least 8 copy threads, never more
    than the allowed CPUs. Source: `gpu_host.cuh` (`gpuNodeCpus`,
    `Staging`), CUDA.md Setup step 1. The cross-socket penalty is a
    measured fact (A100 node), so leave the number to Results or omit it.
  * During growth the only per-level traffic is the per-segment best cuts
    (device→host) and the level description (host→device).
* **What the CPU still does in \ImplGpu{}:** it decides splits once per
  level, stages the copies, and runs post-processing/pruning. There are no
  CPU subtrees. Source: README `--gpu-only`, options.cpp.

Figure suggestion (pick one; space is tight):
* **Algorithm box** "Level-synchronous growth with CPU hand-off" (CUDA.md
  Algorithm 1, cut to ~12 lines). This covers both 3.5 and 3.6, with
  the hand-off line marked as \ImplHybrid{} only. Probably the best use of
  space.
* Or a **diagram** of column slots and a node's range across columns before
  and after a partition. This is more useful for 3.1/3.3, but costs a
  quarter page.

Sources: CUDA.md (all sections except "What did not help"); `gpu_grower.cu`
header comment (lines 1–60); `gpu_kernels.cuh`; `gpu_host.cuh`.

### 3.6 \ImplHybrid: Assigning Nodes by Size — ~280 words

* **Idea** (the contribution; tie to Intro and §2.4): the GPU grows the
  large nodes level by level. Any child below a size threshold *m* is handed
  to the CPU pool, which grows the whole subtree with the \ImplParallel{}
  builder while the GPU continues with the next level. Both devices work on
  the same tree at the same time.
* **Why by size:** a GPU level costs about as much for thousands of small
  nodes as for a few large ones (launch and synchronization overheads, too
  little work per segment), while CPU threads finish small subtrees quickly
  and in parallel as independent tasks. This links back to the two
  difficulties named in the Introduction.
* **Threshold:** default *m* scales with the training-set size. CART:
  N/8192 clamped to [1024, 2048]; C4.5: N/4096 clamped to [1024, 4096]. It
  can be set by the user. Source: `options.h`/`options.cpp` (`gpuMinRows`).
  Flag: the defaults were tuned on the evaluation hardware (CUDA.md
  "Tuning"). Say so honestly ("chosen empirically").
* **Hand-off mechanism (communication):**
  * A gather kernel packs the small children's columns into a 512 MB
    device ring of four parts, taking only the attributes that can still
    split them (not constant).
  * Asynchronous device→host copies on copy streams go through two 128 MB
    page-locked buffers. Staging threads (on the GPU's NUMA node) copy
    them into a pageable host *arena* with room for all columns.
  * A host callback enqueued on the copy stream (`cudaLaunchHostFunc`)
    submits the CPU subtree tasks when their data has arrived. The level
    loop never waits for a hand-off; the GPU only stalls when it is a whole
    ring ahead of PCIe.
  * The arena's pages are touched by all threads in the background while
    the GPU grows the first levels, so copies do not page-fault and the
    pages are spread over the sockets like the threads that later read
    them. This is the second (first-touch based) NUMA-related design point.
  * GPU levels and CPU tasks create nodes in the same concurrent node store;
    the tree is renumbered in preorder at the end, giving byte-identical
    trees to \ImplGpu{}.
* **Degenerate cases:** \ImplGpu{} = threshold 0, with no arena allocated.
  A training set smaller than *m* is grown entirely on the CPU. One clause.
* **Memory cost of the hybrid:** a host arena of \(8NF\) bytes in addition
  to the dataset; the device ring is fixed at 512 MB. Keep it to one
  sentence, or omit.

Sources: CUDA.md "One level" step 6 and the paragraphs after it, "Memory",
"Tuning"; `gpu_grower.cu` (`handOff`, `growOnCpu`, `cudaLaunchHostFunc`
lines ~1494–1536, arena touch ~647–660); `options.cpp`.

---

## 3. Shared prelude vs per-backend deltas (summary)

| Topic | Where | Notes |
|---|---|---|
| Column-major float32 dataset, integer labels | 3.1 | all |
| 8-byte entry, class packed; node owns same range in every column | 3.1 (or 3.3) | all except \ImplSerialBase |
| Shared scoring header (g++ + nvcc), tie rule, \(c\log c\) table | 3.1 | all |
| Preorder flat tree → determinism | 3.1 | all; concurrency aspect revisited in 3.4/3.6 |
| Golden-tree verification | 3.1 (2 sentences) | already in Intro |
| Pruning shared, not in experiments | 3.1 (1 sentence) | |
| Per-node comparison sort, random gathers | 3.2 | delta |
| Radix presort, stable partition, slots, sweep shortcuts, huge pages | 3.3 | delta |
| Thread pool, node-size thresholds, spare slots, first-touch NUMA | 3.4 | delta |
| Level-wise tiles, warp sweep, look-back partition, 1 round trip/level, pinned staging on GPU NUMA node | 3.5 | delta |
| Size threshold, ring → pinned → arena, host-callback tasks, shared node store | 3.6 | delta |

---

## 4. What to omit

* All historical notes ("until 2026-10-17", "what was wrong before",
  `--gpu-prune` removal), and CUDA.md "What did not help" (registers,
  entries per thread). Possibly one sentence in Discussion instead.
* Benchmark numbers of any kind (speed-ups, GB/s, GiB, seconds). These go to
  Results/Discussion. Exception: none.
* Low-level constants that do not explain a design choice: 256-cut settle
  interval, skip-length thresholds 8/32, 4-lane counters, K = 3..8
  templating, 12-bit digits for 1M+ rows, `int(v)−min` keys, kTileBoundGap,
  candidate buffer of 8, 52→74 registers.
* C4.5 pruning engineering: leaf-ordered row sort, row-major copy below 32
  attributes, raise routing of the smaller branch, GPU `arrangeRows`,
  distinct values via `DeviceSelect::Unique`. These are irrelevant to the
  unpruned experiments.
* CART α early stop (nodes with < α·n errors not grown). This only matters
  with pruning on, so omit it, or give it one clause in 3.1 if the author
  wants it.
* Tooling: `--profile`/trace, `--repeat`/warm-up, CUDA context created
  during CSV load, stream-ordered allocation, HMM remarks, CSV loader page
  cache dropping.
* Float-filter derivation (σ, μ, error bound formulas). One sentence is
  enough.
* Full kernel names. At most a descriptive phrase per kernel; the
  repository has the names.

---

## 5. Open questions / risks

1. **Q1 — where to introduce the 8-byte entry.** In 3.1 (shared by 4/5) or in
   3.3 (where it first appears)? Recommendation: 3.3, with 3.1 saying only
   "column-major float32". This keeps the progression story clean: the
   layout change *is* one of \ImplSerialOpt{}'s optimizations.
2. **NUMA claims for CPU backends.** The main source has **no explicit NUMA
   placement** for \ImplParallel{} (no libnuma, mbind, interleave or pool
   pinning; the only `pthread_setaffinity_np` is for the GPU staging
   copiers). "Pages spread over sockets" is a first-touch consequence
   stated in comments (`gpu_grower.cu` arena touch; PERFORMANCE.md node
   store "allocates … on its own socket"). The writer must phrase this as
   reliance on first-touch, not as a NUMA-aware design.
3. **Thread binding in the GPU experiments.** experiment-design.tex says GPU
   runs are bound to the GPU's NUMA node (16 CPUs on hopper, 24 on ampere).
   But a chart caption says "all 48 CPU threads of the node" for the A100
   hybrid-vs-GPU-only runs, and results.tex says "\ImplHybrid{} with 16 CPU
   threads". Section 3 should not state the binding; that belongs in §4.
   The inconsistency should be resolved with the author.
4. **README vs code on `--gpu-only`.** The README says the CPU threads
   "count the classes" in GPU-only mode, but `gpu_grower.cu` computes the
   root's class counts with `countClassesKernel` on the GPU (`main.cpp` may
   still count on the CPU for reporting). Recommendation: describe the GPU
   kernel and do not repeat the README phrase.
5. **One-pass vs two-pass sweep.** On A100/H100 `auto` picks the
   double-precision one-pass sweep, so the float filter is never exercised
   in the evaluated runs. Recommendation: one sentence for generality, and
   flag in the text that the experiments use the one-pass mode. The
   author should confirm that no runs forced `--gpu-sweep two-pass`.
6. **"Provably" wording.** SIMPLE.md Risk B: the CPU rounding bounds are
   argued in code comments but not tested directly. Use "conservative bound"
   or "bounded error" rather than "proven", or "argued in the source" if the
   claim must be precise. Trees match on every test and benchmark (golden
   tests, `tree_sha256`).
7. **Boundary-point rule.** The Fayyad–Irani shortcut is used in a
   restricted form (single rows with distinct values, neighbouring cuts
   allowed). SIMPLE.md Risk A says this is theoretically not airtight against
   the 1e-12 tie rule; no case is known. Either omit the shortcut from the
   text (safest), or describe it as restricted.
8. **Thresholds in the hybrid are empirical** and tuned on the evaluation
   machines (HIGGS, airline). Reviewers may ask about portability. Mention
   that the threshold is a parameter and defaults scale with N.
9. **Naming.** The code calls the hybrid backend "Cuda" (`--cuda`) and
   \ImplGpu{} is `--cuda --gpu-only`. The paper should never say "CUDA
   backend" for the hybrid, to avoid confusion with \ImplGpu{}.
10. **Length risk.** About 1,900 words plus a table and an algorithm box is
   ≈ 3.2 LNCS pages. If the budget is a hard 3 pages: drop the overview
   table (−150) or trim 3.3(b) to the estimate+exact idea and the convex
   bound (−100), and shorten 3.4's presort paragraph (−60).

---

## 6. Estimated length

| Part | Words |
|---|---:|
| Lead-in (+ table) | 110 |
| 3.1 Common Design | 280 |
| 3.2 \ImplSerialBase | 130 |
| 3.3 \ImplSerialOpt | 400 |
| 3.4 \ImplParallel | 300 |
| 3.5 \ImplGpu | 420 |
| 3.6 \ImplHybrid | 280 |
| **Total prose** | **≈ 1,920** |

Floats: one table (≈ 0.25 page) and one algorithm box (≈ 0.35 page).
Estimated total ≈ 3.0–3.3 LNCS pages. A tight version (≈ 1,550 words, no
table) fits in ≈ 2.5 pages including the algorithm box.
