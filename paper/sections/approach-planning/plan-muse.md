# Section 3 "Our Approach" — Planning Document (Muse Wonder)

> **Status:** Plan only. No prose for `approach.tex` is drafted here; bullets below are
> content specifications for the later writing pass. All factual claims are traceable to
> the files cited per subsection. Items marked **[VERIFY]** or **[FLAG]** must be resolved
> before or during writing — do not smooth them over in prose.
>
> **Naming (paper macros, `paper/main.tex`):**
> `\ImplSerialBase` = Serial Simple (`--simple`) ·
> `\ImplSerialOpt` = Serial Optimized (`--serial`) ·
> `\ImplParallel` = Multicore CPU (`--parallel`) ·
> `\ImplGpu` = GPU-Based (`--gpu-only`) ·
> `\ImplHybrid` = Hybrid (`--cuda`)
>
> **Critical naming fact for the writer:** there are only *four* code backends
> (`Backend::{Simple, Serial, Parallel, Cuda}`, `src/core/options.h`).
> `\ImplGpu` and `\ImplHybrid` are two configurations of the *same* CUDA grower
> (`makeGpuGrower`): `--gpu-only` sets `GpuOptions.only = true`, which forces the
> hand-off threshold to effectively 1 row and skips the host arena
> (`src/core/options.cpp: gpuMinRows`, `src/build/gpu_grower.cu`). Section 3 must state
> this once, explicitly, or reviewers will assume two separate GPU implementations.

---

## 1. Proposed subsection structure

The current skeleton (`paper/sections/approach.tex`) has five bare subsections.
Recommendation: **add one shared-design subsection at the front and keep the five
implementation subsections**, giving six subsections. Rationale: roughly a third of
what a reader needs (data layout, scoring, identical-tree invariant) is common to all
five configurations; repeating it per backend — or scattering it — is the main
structural risk in the skeleton as-is. The five implementation subsections then become
pure *delta* descriptions, which matches the author's "design-choice narrative over
annotated source walkthrough" instruction and the delta style already used in the
Introduction ("from a straightforward sequential baseline to a hybrid one").

| # | Proposed title | Macro focus | Words (target) |
|---|---|---|---|
| 3.1 | Common Design: Exact Splits on Presorted Columns | all | ~300 |
| 3.2 | Serial Simple: A Textbook Baseline | `\ImplSerialBase` | ~150 |
| 3.3 | Serial Optimized: Presorting and a Filtered Sweep | `\ImplSerialOpt` | ~330 |
| 3.4 | Multicore CPU: Two Levels of Parallelism | `\ImplParallel` | ~280 |
| 3.5 | GPU-Based: Level-Synchronous Growth on One Device | `\ImplGpu` | ~300 |
| 3.6 | Hybrid: Splitting One Tree Between GPU and CPU | `\ImplHybrid` | ~240 |
| — | **Total prose** | | **~1,600** |

Plus: one overview figure (see §7) and optionally one small comparison table.
If the figure is included at column or page width, cut prose to **~1,350–1,450 words**
to stay within 3 pages (see §6 for the page arithmetic).

Alternative if the author prefers to keep exactly five subsections: fold 3.1 into an
unnumbered opening passage of ~250 words before `\ImplSerialBase`. This is workable
but weaker — the shared layout is substantial enough to deserve a heading, and
Related Work already primes the reader for the SLIQ/SPRINT presorting idea, so 3.1
can cross-reference it instead of re-teaching it.

**Section opening (before 3.1, 2–3 sentences, not budgeted separately — take from 3.1):**
state the organising principle of the whole section: all five configurations execute
the *same* exact search and produce the *same* tree; they differ only in (a) how the
sorted order of a node's rows is obtained and maintained, (b) how the best split is
found, and (c) where (thread / device) and in what order nodes are processed.
This sentence does heavy lifting: it tells the reader that every later subsection is
a variation on three axes, and it pre-empts the "five unrelated programs" reading.

---

## 2. Per-subsection content plan

### 3.1 Common Design: Exact Splits on Presorted Columns (~300 words)

**Purpose:** everything that is *not* a differentiator. Write this once, precisely,
so later subsections can say "as in §3.1" instead of repeating.

Bullets — what to say:

- **Pipeline position.** Training is: grow the (maximal) tree with a backend, then
  apply shared post-processing (CART cost-complexity pruning at fixed α; C4.5
  collapse + training-value thresholds + error-based pruning). Post-processing is
  identical for all backends and is *not* described per backend; mention in one
  sentence, with pruning internals deferred (they are algorithm semantics, already
  framed in Intro/Related Work; see §4 on omissions).
  *Support:* `src/app/trainer.cpp: trainTree` (grow → `c45CollapseUselessSplits` /
  `c45UseTrainingValueThresholds` / `c45PessimisticPrune`, or
  `cartCostComplexityPrune`); `docs/ALGORITHMS.md` overview table.
- **Shared scoring code — the identical-tree mechanism.** All backends score cuts
  with literally the same function (`cutGain` on weighted Gini/entropy impurities),
  compiled for both host and device from one header; ties within `1e-12` go to the
  earlier cut, then the lower feature index; entropy uses a shared
  `c·log₂c` table for counts < 65,536 so CPU and GPU agree bit-for-bit.
  CART vs C4.5 differ only in the rule layer on top (criterion; C4.5's per-feature
  gain maximisation, MDL threshold cost `log₂(tries)/n`, average-gain filter, gain
  ratio choice; algorithm-specific `minChildRows` / stopping). **Do not re-derive
  CART/C4.5 here** — one sentence pointing back to §2.1 suffices; the only new
  information is *that the rule layer is shared and backend-independent*.
  *Support:* `src/algo/split_math.h` (`cutGain`, `weightedImpurity`, `kTieEps`,
  `LogTable`, `isSkippableCut`, `isBetterCut`); `src/algo/split_rules.h`
  (`SplitRules::choose`, `minChildRows`, `isTerminal`); `docs/ALGORITHMS.md`
  ("Ties and CPU/GPU agreement").
- **Verification claim (brief).** Because of the shared scoring, all backends are
  required — and checked — to produce byte-identical trees, against each other and
  against scikit-learn (CART) and Quinlan's C4.5. Two sentences maximum; the
  experimental setup section owns the protocol detail. This claim is load-bearing
  for the whole paper ("speed comparisons always compare runs that build the same
  tree", Intro), so it belongs here as a *design* statement.
  *Support:* `docs/ALGORITHMS.md` (verification paragraphs); `tests/check.sh` /
  `tests/golden/` (per `README.md`); `docs/SIMPLE.md` preamble.
- **In-memory representation (the part every backend shares).** Dataset is
  column-major `float` values with integer class ids (`values[feature * rows + row]`).
  The working representation for the optimised backends is, per feature, a presorted
  column of 8-byte entries `{float value; uint32 packed = row << classBits | class}`
  — carrying the class inside the entry is what makes a threshold sweep a pure
  sequential scan with no row-id lookups. A tree node owns the *same* index range
  `[begin, end)` in every column; splitting stably partitions every column's range
  so both children again own one sorted range per column (the SLIQ/SPRINT idea,
  cross-ref §2.2). The finished tree is flat arrays of nodes in preorder plus one
  class histogram per node, so pruning/prediction need no further pass over data.
  *Support:* `src/core/dataset.h`; `src/algo/split_math.h` (`Entry`, `EntryCodec`);
  `docs/CPU.md` ("Sort once"); `src/core/tree.h` (`Tree`, preorder invariant);
  `src/build/cpu_grower.cpp`.
- **What "exact" means operationally (one sentence).** Every cut between two
  consecutive distinct values (C4.5: values ≥ 1e-5 apart) that respects the minimum
  child size is a candidate; no binning anywhere. Already in the Intro — here just
  tie it to the entry/partition representation above so the reader sees *how*
  exactness is afforded.
  *Support:* `docs/ALGORITHMS.md` notation section; `split_math.h: isCut`.

**Explicitly not here:** presort algorithm details (→ 3.3, where the radix sort is
introduced as \ImplSerialOpt's first optimisation), partition mechanics beyond the
one-sentence invariant above, any kernel or thread-pool detail.

---

### 3.2 Serial Simple: A Textbook Baseline (~150 words)

**Purpose:** establish the reference point. This subsection should be *short* — its
brevity is itself the message ("nothing clever happens here"), and it motivates
every optimisation in 3.3 by contrast. Frame it as a deliberate methodological
choice (an executable specification / correctness oracle), not as a straw man.

Bullets — what to say:

- **Algorithm as written in textbooks.** Single thread, recursion in preorder.
  At each node, for every feature: copy the node's row ids, `std::sort` them by
  that feature's values (values and labels gathered from the column-major dataset
  by row id), sweep all allowed cuts scoring *each one* with the exact shared
  `cutGain`, keep the best; let the shared CART/C4.5 rule choose the feature;
  split the row list into two freshly allocated lists and recurse.
  *Support:* `src/build/simple_grower.cpp` (whole file, ~150 lines; header comment
  states the design intent verbatim); `docs/SIMPLE.md` preamble.
- **Cost characterisation (complexity only, no measured numbers).** Sorting is
  repeated at every node: `O(F · n log n)` per node with random-access gathers;
  per-node heap allocation throughout. This is the quantity 3.3 eliminates.
  *Support:* `docs/SIMPLE.md` differences table, item 1 & "Data layout" group.
- **Role in the paper.** (i) Performance baseline for the speed-up chain in the
  results; (ii) semantic reference — it shares `SplitRules`/`split_math`/pruning
  with the optimised backends, so agreement with it (checked against golden trees)
  is evidence the optimisations preserve the tree. Note its one structural
  divergence for C4.5: it fills neither distinct values nor leaf-of-row, so the
  trainer recomputes distinct values and routes rows for pruning — a sentence that
  sets up the "growers also feed the pruner" point in 3.3/3.5.
  *Support:* `src/build/grower.h` (`Grower` interface contract);
  `simple_grower.cpp: SimpleGrower::grow`; `src/app/trainer.cpp` (fallbacks
  `distinctValues(train)` when the grower provides none); `docs/SIMPLE.md` items
  23–24.
- **Known limitation worth one clause (optional, author decision):** unbounded
  recursion depth (`growNode` recurses per level; no explicit stack), unlike the
  optimised builder. Only include if the discussion section picks it up; otherwise
  omit — it invites a question the experiments never exercise.
  *Support:* `docs/SIMPLE.md` Risk D. **[FLAG — author to decide; default: omit.]**

**Tone note:** do not apologise for this backend or call it "naïve" repeatedly;
"straightforward/textbook" (the Intro's wording) once is enough.

---

### 3.3 Serial Optimized: Presorting and a Filtered Sweep (~330 words)

**Purpose:** the single largest conceptual step in the paper (and, per the Intro,
worth up to 25×). This is the subsection to spend words on. Organise as three
moves — *sort once*, *score cheaply but exactly*, *partition without copying
chaos* — each stated as: what changes, why it is faster, why the tree is unchanged.
That "why the tree is unchanged" clause per move is what distinguishes this from a
changelog; keep it to one sentence each, citing the invariant, not the proof.

Bullets — what to say:

- **Move 1 — presort once, partition stably (replaces per-node sorting).**
  Each feature column is sorted once up front by a stable LSD radix sort on the
  float bits (3 passes of 11 bits, skipping passes whose digit is constant;
  adaptive plans for large/integer-ranged columns). A node is then split by
  *position*: exactly the first `leftCount` entries of the winning column go left,
  so every other column's range is stably partitioned by a goes-left marking of
  the winning column's rows — no threshold comparisons during partition, no
  re-sorting ever; each tree level costs `O(F · n)`. Why identical: a cut depends
  only on values and class counts per side, not on order within equal values, and
  the partition preserves sorted order in both children.
  *Support:* `docs/CPU.md` ("Sort once: presorted columns", incl. `radixPlan` and
  skipped passes); `src/build/presort.{h,cpp}`; `docs/SIMPLE.md` items 1–2, 15, 18
  and "Why the tree is the same" (Sort order). Keep radix-pass bit-width detail to
  at most a parenthetical; the *stability + once-only* properties are the point.
- **Move 2 — exact split search with a cheap filter (the sweep).** The naive loop
  would call the exact gain (loop over classes, divisions/log table) at every cut.
  Instead: an incremental estimate with a proven rounding-error bound is maintained
  per cut (integer-exact sum-of-squares updates for Gini; running `c·log₂c` sums
  refreshed periodically for entropy; specialised two-class form), and the exact
  shared `cutGain` is evaluated only for cuts whose estimate can still beat the
  incumbent by more than the tie tolerance, settled in cut order. Three further
  *result-preserving* prunings of the search, to be named in one breath each:
  (a) convexity bound over blocks of 16 cuts (binary case, nodes ≥ 64 rows) —
  gain is convex in the left class counts, so a block whose corner bound cannot
  win is counted but not scored; (b) a step bound (`maxGainStep`) that skips cuts
  provably unable to catch up; (c) the Fayyad–Irani boundary-point skip for cuts
  between two singleton same-class values. Also: runs of equal values are absorbed
  into class counts without per-cut work, and features constant over the node's
  range (first vs last sorted entry) are dropped for the whole subtree — a large
  effect on one-hot-heavy data; say *why* it matters (covertype-like binary
  features) without citing the benchmark numbers here.
  *Support:* `docs/CPU.md` ("The sweep", "Constant features are dropped");
  `src/build/cpu_sweep.{h,cpp}`; `docs/SIMPLE.md` items 4–13;
  `split_math.h: isSkippableCut` (with its convexity justification in the comment).
  **[FLAG — honesty constraint from `docs/SIMPLE.md`: "none of the items was
  measured on its own; an ablation is needed before citing numbers per item."
  Section 3 must therefore describe these optimisations qualitatively and must
  NOT attribute individual speed-ups to them. Any per-item factor belongs in
  Results only if an ablation is actually run.]**
- **Move 3 — memory discipline.** All column storage is allocated once as
  `F + 1` slots of `N` entries (one spare slot enables out-of-place partition of
  large nodes; small nodes partition in place through a reused per-thread buffer);
  no allocation in the hot loop; children class counts are taken from the winning
  sweep instead of a recount; splits whose children are both leaves skip the
  partition entirely; large allocations (≥ 2 MiB) are 2 MiB-aligned and marked
  for transparent huge pages. Footprint statement (one clause, it matters for the
  memory results later): the optimised layout costs roughly `8(F+1)` bytes per row
  for columns (plus count tables on large training sets), vs. row-id lists in
  \ImplSerialBase.
  *Support:* `docs/CPU.md` ("Memory", big-node slot scheme, terminal-children
  skip); `src/build/cpu_grower.cpp` (`spareSlots`, `work_`, `countTables`);
  `src/core/memory.cpp`; `docs/SIMPLE.md` items 16–22.
- **Grower/pruner interface (one sentence, bridges to 3.5).** Unlike
  \ImplSerialBase, this grower also returns each feature's distinct values (read
  off the presorted columns) and each row's leaf, which C4.5's post-processing
  consumes directly instead of re-deriving them.
  *Support:* `docs/CPU.md` ("Post-processing" preamble); `docs/SIMPLE.md` items
  23–26; `trainer.cpp`.
- **Serial = the parallel algorithm with the pool switched off.** End the
  subsection by stating that \ImplSerialOpt is exactly the builder of §3.4 run on
  one thread (same code path, `pool == nullptr`), including the out-of-place
  partition of large nodes. This makes 3.4 a pure "what parallelism adds" delta
  and pre-empts "is Serial Optimized a different algorithm?" — it is not.
  *Support:* `src/app/trainer.cpp` (`buildPool = nullptr` for Serial/Simple);
  `docs/SIMPLE.md` ("`--serial` is the `--parallel` algorithm on one thread");
  `cpu_grower.cpp` comment ("serial and parallel builds run the same algorithm").

---

### 3.4 Multicore CPU: Two Levels of Parallelism (~280 words)

**Purpose:** show that parallelism is *added around* the unchanged serial
algorithm at exactly the two granularities where tree induction offers work —
inside a large node, and between subtrees — with size thresholds deciding which
applies. This mirrors the mixed strategy the Related Work section attributes to
Aldinucci et al.; one cross-reference sentence makes that continuity explicit.

Bullets — what to say:

- **Intra-node (data) parallelism for large nodes.** At/above
  `featureParallelRows = 32,768` rows, the per-feature sweeps of a node run as a
  `parallelFor` over features, and its partition runs one column per thread
  (requires the spare-slot scheme: `--parallel` keeps up to `min(F−1, threads)`
  spare slots so a large node can move all its columns concurrently; this is also
  where most parallel time goes because these nodes are memory-bandwidth-bound).
  Below the threshold the wake-up/synchronisation cost of the pool exceeds the
  work — state the reason, not the futex war story (see §4).
  *Support:* `docs/CPU.md` ("Parallelism", big-node bullets);
  `src/core/options.h: ParallelOptions` (constants + tuning rationale in
  comments); `cpu_grower.cpp: spareSlots`.
- **Inter-node (task) parallelism for subtrees.** After splitting a node of at
  least `nodeTaskRows = 4,096` rows, the left child becomes a pool task and the
  current thread continues with the right child on an explicit stack; nobody
  blocks on a task — the grower waits for idleness once at the end. Combined
  with intra-node parallelism this is the "data-parallel near the root,
  task-parallel deeper" schedule. Determinism is preserved because nodes are
  created in a thread-safe chunked store and renumbered into preorder at the
  end: the resulting tree is byte-identical to the serial one regardless of
  task completion order. That last sentence is essential — it is the parallel
  analogue of the identical-tree invariant.
  *Support:* `docs/CPU.md` ("Subtrees", `parallelFor` nesting note, NodeStore
  paragraph); `src/core/thread_pool.h` (deadlock-free nested `parallelFor`:
  caller participates, only waits for already-running items);
  `src/build/node_store.*`; `docs/CPU.md` ("Memory", preorder copy).
- **Presort and assembly are parallel too (one sentence each, no detail).**
  Radix presort: feature histograms first so expensive features start first;
  columns ≥ 1M rows are counted in chunks across threads; when features are
  fewer than threads a single column's sort is itself split across threads.
  Final preorder assembly copies disjoint subtrees as parallel tasks.
  *Support:* `docs/CPU.md` (presort paragraph; "Memory", parallel tree copy);
  `src/build/presort.cpp`; `src/core/tree.h: assignPreorder`.
- **NUMA / multi-socket — what is true, stated carefully (2–3 sentences).**
  The pool is deliberately *not* NUMA-aware: workers are created unpinned
  (`main.cpp` constructs `ThreadPool(threadCount−1)` with no affinity callback),
  the presorted columns are a single shared allocation (not replicated or
  partitioned per socket), and there is no socket-local work assignment in the
  CPU builder. What the implementation *does* do: the thread count respects the
  process affinity mask (`sched_getaffinity`), so external placement with
  `taskset`/`numactl` is honoured rather than fought; and large blocks get
  transparent huge pages, reducing TLB/page-fault cost on multi-socket working
  sets. Cross-socket traffic is thus left to the OS scheduler and first-touch
  behaviour of the parallel presort/partition writes. **Present this as a
  scoping statement, not a feature.** If the Results/Discussion report socket
  scaling (e.g. "second socket starts at 25 threads" in the scaling figure),
  this subsection is where the reader learns there is no NUMA machinery behind
  it — which makes the scaling curves *more* informative, not less.
  *Support:* `src/app/main.cpp` (pool construction, no `onStart`);
  `src/core/options.cpp: threadCount` (affinity-mask comment);
  `src/core/memory.cpp`; absence of any pinning/NUMA call in
  `src/build/cpu_*` and `src/core/thread_pool.cpp` (verified by search).
  **[FLAG — the single most likely reviewer trap in this section. Do NOT write
  "NUMA-aware", "socket-local", or "pinned" anywhere near the CPU backend.
  `docs/PERFORMANCE.md`'s NUMA material (membind/pinning probes on dell) is
  benchmark-harness investigation, not shipped behaviour, and must not leak
  into Approach. See §5.]**
- **Thresholds are heuristics, stated as such.** 32,768 / 4,096 are fixed
  compile-time constants tuned on 48-core servers; note in one clause that
  C4.5's lopsided splits can favour lower values on smaller machines (the docs
  record exactly such a case) — this inoculates against "magic constants"
  criticism without a tuning study in this section.
  *Support:* `src/core/options.h: ParallelOptions` comments; `docs/CPU.md`
  (featureParallelRows paragraph).

---

### 3.5 GPU-Based: Level-Synchronous Growth on One Device (~300 words)

**Purpose:** explain the one genuinely different execution model in the paper.
The organising contrast to state up front: the CPU builders are depth-first and
node-at-a-time; the GPU grower is **breadth-first, one whole tree level at a
time**, because a GPU needs thousands of concurrent work items and a single
small node cannot supply them. `\ImplGpu` (`--cuda --gpu-only`) is the pure form
of this model — every node stays on the device — which is why it is presented
before the hybrid that relaxes it.

Bullets — what to say:

- **Level structure and work decomposition.** All nodes of a level that need a
  split are processed together. Work is cut into *segments* (one node's range in
  one feature's column) and *tiles* of 2,048 consecutive entries; each kernel
  launch runs one block per tile/segment, so a level of, say, 300 nodes × 18
  features is a handful of launches, not 300 separate computations. Device
  memory mirrors the CPU layout: one copy of the presorted columns in `F + 1`
  slots, partition moving columns into the slot just vacated, same
  "node owns one range in every column" invariant as §3.1.
  *Support:* `docs/CUDA.md` ("Breadth-first instead of depth-first", slot
  scheme); `src/build/gpu_grower.cu` (level loop); `src/build/gpu_kernels.cuh`.
- **The per-level pipeline, at kernel-*stage* granularity (not kernel names).**
  Five stages are enough for the reader: (1) per-tile class histograms and
  per-segment prefix counts (below the root these are produced *by the parent
  level's partition*, so the histogram stage is root-only); (2) split search
  (next bullet); (3) per-(node, feature) best cut reduced on device, then **one
  host round-trip per level**: the host applies the shared `SplitRules::choose`
  — the same code as on CPU — and derives both children's class counts from the
  winning sweep, finishing leaf children on the spot; (4) goes-left marking as a
  1-bit-per-row bitmap (marking only the smaller side; the winning column needs
  no marks — its left rows are by construction the first `leftCount`); (5)
  single-pass stable partition of all columns into free slots, which also emits
  the next level's tile histograms. Constant segments are detected first and
  excluded from all later stages, as on CPU.
  *Support:* `docs/CUDA.md` ("One level" table, goes-left bitmap paragraph);
  `gpu_kernels.cuh` (stage kernels); `split_rules` reuse in `gpu_grower.cu`.
  Kernel *names* (`tileEstimateKernel`, …) should appear at most in a figure
  caption or not at all — see §4.
- **Split search on device: warp-cooperative, still exact.** One warp sweeps
  256 entries in 8 steps of 32; class counts left of a lane's entry come from
  warp ballots + popcount, neighbour values from shuffles — no shared-memory
  scan, no atomics in the sweep. Precision strategy is the subtle part and must
  be stated accurately: on GPUs with fast double precision (detected from the
  FP32:FP64 ratio; `--gpu-sweep` can override) every cut is scored once in
  double precision with the shared `cutGain`. Otherwise a two-pass scheme: a
  single-precision filter estimates every cut and keeps only candidates within
  a *proven* float-error bound of the segment best (per-tile candidate lists of
  ≤ 8 cover almost all tiles), and only those are re-scored exactly in double
  precision. Either way the result is bit-identical to the CPU — the filter is
  conservative by construction (cutoff `best − (best·σ + μ)` covering the bound
  plus the tie tolerance), never an approximation of the tree. For the binary
  case an additional tile-level convexity bound skips whole tiles, mirroring
  the CPU sweep's block bound.
  *Support:* `docs/CUDA.md` ("Inside the sweep", "Why two passes", "Why the
  result is exact"); `src/core/options.h: GpuSweep`; `gpu_grower.cu`
  (`onePass_` selection); `gpu_kernels.cuh` (`floatGainFilter` comment).
- **Setup and host↔device communication (GPU-Based still has some).** CUDA
  context creation overlaps CSV loading; uploads go through four 32 MiB
  page-locked staging buffers with pool threads copying chunks (page-locked
  speed without page-locking the dataset), and each feature is radix-sorted on
  device (CUB, stable) as soon as its upload completes, overlapping transfer
  and presort. Per level, communication is only the small best-cut array
  host-ward (stage 3 above) — in `--gpu-only` mode there are *no* subtree
  downloads and no host arena. The CPU is not idle by definition here: its
  threads run the staged copies and, for C4.5, the shared pruning (and can
  accelerate pruning's row arrangement on device when the raw values fit —
  optional clause, author may cut).
  *Support:* `docs/CUDA.md` ("Setup", "Memory", `--gpu-only` tuning bullet);
  `src/build/gpu_host.cuh` (staging design); `src/build/grower.h`
  (`gpuPrepare` contract); `README.md` (`--gpu-only` row).
- **NUMA where it actually exists in this codebase (2 sentences).** The staging
  buffers are allocated on, and the copy threads pinned to, the GPU's own NUMA
  node (read from sysfs via the PCI bus id), because copies staged from the
  other socket of a two-socket node ran at roughly half the PCIe rate. This is
  the *only* explicit NUMA placement in the system; say so, and cross-reference
  the §3.4 scoping statement so the contrast (GPU comms: NUMA-placed; CPU
  compute: OS-managed) is explicit rather than implied.
  *Support:* `src/build/gpu_host.cuh` (`gpuNodeCpus`, `Staging` pinning,
  rationale comment); `docs/CUDA.md` Setup item 1; `docs/PERFORMANCE.md`
  ("Copies on the GPU's NUMA node").
- **Capacity boundary (one sentence).** Device memory must hold `(F+1)·8N`
  bytes of columns plus per-level workspace; the builder checks free memory
  and suggests `--parallel` when it does not fit. This sets up both the
  hybrid's existence and the "does not fit in 80 GB" cells in the results
  tables.
  *Support:* `docs/CUDA.md` ("Memory"); `gpu_grower.cu` (free-memory check);
  `paper/sections/results.tex` table captions.

---

### 3.6 Hybrid: Splitting One Tree Between GPU and CPU (~240 words)

**Purpose:** the paper's headline mechanism (Intro: "no previous work divides
the nodes of a single tree between CPU and GPU by size"). Keep it tight and
mechanical: threshold → hand-off path → concurrency → why this wins. Related
Work already argued novelty; here only the mechanism is owed.

Bullets — what to say:

- **Assignment rule.** Identical level-synchronous GPU growth as §3.5, except:
  children with fewer than `--gpu-min-rows` rows leave the GPU. Default
  threshold is automatic — CART: `rows/8192` clamped to [1024, 2048];
  C4.5: `rows/4096` clamped to [1024, 4096] — overridable on the command line.
  Rationale in one sentence: a GPU level costs roughly the same for a few large
  nodes as for thousands of small ones (launch + synchronisation dominated),
  while a CPU thread finishes a small subtree depth-first with no transfers;
  the threshold puts each node where its marginal cost is lower. Size is the
  *only* assignment criterion — no cost model, no lookahead; say this plainly,
  it is a defensible simplicity claim.
  *Support:* `src/core/options.h: GpuOptions` (+ tuning comment);
  `src/core/options.cpp: gpuMinRows`; `docs/CUDA.md` ("Tuning",
  `--gpu-min-rows` bullet); `README.md` (performance-tuning table).
- **Hand-off path (the communication design).** Handed-off children are
  gathered on device into a 512 MiB ring buffer (four parts) and downloaded
  through two 128 MiB page-locked buffers on two copy streams, alternating so
  one downloads while the other is copied out into a pageable host arena
  (room for every column of every row; pages pre-touched in the background
  during presort so hand-offs don't fault them in; a hand-off waits only for
  the arena part it fills). A host callback on the copy stream starts the CPU
  subtree tasks the moment its copy lands — the GPU level loop does **not**
  wait for hand-offs; it proceeds to the next level, and the CPU pool grows
  the handed subtrees (the same `CpuTreeBuilder` as §3.3/3.4, in-place
  partition) concurrently. Only the still-splittable columns are gathered.
  Final assembly waits for the CPU tasks to drain. For C4.5, leaf-of-row
  records from GPU-finished and CPU-finished parts are merged so the shared
  pruning sees one complete mapping.
  *Support:* `docs/CUDA.md` ("One level" item 6 + following paragraphs, ring /
  callback / arena discussion); `src/build/gpu_grower.cu` (`handOff`,
  `touchArena`); `src/build/gpu_host.cuh` (hand-off buffers).
- **What Hybrid is not (one short paragraph or two sentences — pre-empts
  misreading).** Not two implementations racing, not whole-subtree static
  partitioning decided at the root: assignment is per child node, decided at
  each level as sizes become known, and both devices contribute nodes to the
  *same* tree whose identity with the other backends' trees is still required
  and checked. `\ImplGpu` is the degenerate case (threshold ≈ 1) of the same
  mechanism — which is precisely why the experimental comparison between them
  isolates the value of the CPU's share.
  *Support:* `gpu_grower.cu` (`only` branches); `docs/CUDA.md` (hand-off
  design); Intro/Related Work positioning (cross-ref, don't re-argue).
- **Costs/risks to acknowledge in one sentence (credibility).** The host arena
  costs `8·F·N` bytes of host memory even though each row is handed over at
  most once, and the threshold is a tuned heuristic whose optimum drifts with
  dataset size and CPU:GPU speed ratio — the sensitivity data belong to
  Results; here just flag that the default is a clamp of a size-proportional
  rule, not a fitted model.
  *Support:* `docs/CUDA.md` (arena sizing "room for every column of every
  row", tuning paragraph); `options.h` comment ("the best value grows with
  the dataset, but slowly").

---

## 3. Shared prelude vs per-backend deltas (allocation map)

Use this table when writing, to guarantee nothing is said twice or nowhere.

| Topic | Where it lives | Notes |
|---|---|---|
| Column-major dataset, integer class ids | 3.1 | One sentence; loader itself not described |
| `Entry{value, packed row/class}`, 8 B | 3.1 (define) | Reused silently in 3.3/3.5 |
| Node = same `[begin,end)` range in every column; stable partition invariant | 3.1 (invariant) | Mechanics: 3.3 (CPU), 3.5 stage 5 (GPU) |
| Shared `cutGain`, weighted impurities, `1e-12` tie rule, log table | 3.1 | GPU precision filter in 3.5 must reference this, not redefine it |
| CART vs C4.5 rule differences | 3.1, one sentence + §2.1 cross-ref | **Not** re-taught per backend |
| Identical-tree requirement + reference verification | 3.1 | Protocol detail → Experimental Setup |
| Flat preorder tree + per-node class histograms | 3.1 | Determinism-under-parallelism consequence → 3.4 |
| Growers returning distinct values / leaf-of-row for C4.5 pruning | 3.1 (contract, one line) | Simple lacking it: 3.2; CPU providing it: 3.3; merge across devices: 3.6 |
| Per-node `std::sort`, exact gain at every cut, fresh allocations | **3.2 only** | Baseline cost model |
| Radix presort (stable, once) | **3.3** (CPU) | GPU presort (CUB, overlapped) is a *delta* in 3.5, one clause |
| Estimate-then-exact sweep, convexity/step/boundary skips, equal-run absorption, constant-feature dropping | **3.3 only** | GPU has analogues (tile bound, two-pass filter) — present as analogues in 3.5, don't re-explain the CPU versions |
| Slot-based partition (`F+1` / spare slots), terminal-child skip, huge pages | 3.3 (scheme) | Spare-slot count becomes parallelism mechanism in 3.4; device slots in 3.5 |
| Feature-parallel vs subtree-task parallelism, thresholds 32768/4096 | **3.4 only** | — |
| CPU NUMA scoping statement (no pinning; affinity mask respected) | **3.4 only** | Contrast pointer from 3.5 |
| Level-synchronous BFS, tiles/segments, 5-stage level pipeline | **3.5 only** | Assumed known in 3.6 |
| Warp ballot/shuffle sweep; one-pass vs two-pass precision | **3.5 only** | Correctness argument: 3.5, referencing 3.1 |
| Staged pinned uploads, GPU-node NUMA pinning of staging/copiers | **3.5 only** | The paper's only NUMA mechanism |
| Hand-off threshold + auto rule, ring download, callback-started CPU tasks, host arena | **3.6 only** | — |
| Pruning algorithms (CCP, error-based, raising, thresholds-from-training-values) | **Nowhere in §3** beyond the 3.1 pipeline sentence | Semantics in §2.1/Intro; cost results in Results. Exception: backends *feeding* the pruner (above row) |

---

## 4. What to omit (and why)

- **Kernel-by-kernel enumeration.** `docs/CUDA.md`'s 0–6 kernel table, names like
  `tileCandidateKernel`, decoupled look-back self-validating words, shared-memory
  staging details, register counts. *Why:* annotated-walkthrough smell; the
  five-stage pipeline conveys the design. Kernel names may survive in the
  overview figure's caption if the figure is pipeline-style.
- **Per-optimisation speed attributions.** `docs/SIMPLE.md` explicitly warns no
  ablation exists. Section 3 says "expected effect from complexity alone" at most;
  all factors (25×, 40×, …) live in Results.
- **Tuning archaeology and failed experiments.** Old thresholds (4096-row futex
  episode), register-caching regression, the removed GPU subtree-raising pruner,
  `MADV_POPULATE_WRITE` episode, arena page-locking stalls (`docs/CUDA.md`
  "What did not help", PERFORMANCE.md passim). *Why:* valuable lab notes, wrong
  genre; at most one "alternatives considered" sentence in Discussion if wanted.
- **Benchmark-derived constants as findings.** Staging bandwidths (55 GB/s etc.),
  buffer sizes beyond the two that explain the design (32 MiB staging, 512 MiB
  ring — even these are candidates for the figure caption rather than prose),
  thread-count minutiae of the copier pool (≥ 8 threads rule). Keep: tile size
  2,048 and thresholds 32,768 / 4,096 / auto `gpu-min-rows` clamps, because a
  reader needs them to reproduce behaviour.
- **Pruning internals, CSV loading, `--profile`/`--repeat`/energy harness,
  CLI validation.** None is a backend differentiator; the trainer pipeline
  sentence in 3.1 covers existence.
- **NUMA performance probes.** membind/pinning noise studies, dell thread-placement
  curves (`docs/PERFORMANCE.md`) are experimental-setup/results material and, if
  cited in Approach, would falsely imply shipped NUMA features.
- **C4.5 post-processing optimisations in the CPU pruner** (row-major transposition,
  leaf-order radix sort, raise-merge routing in `docs/CPU.md` "Post-processing").
  Real and substantial, but they accelerate *shared post-processing*, not a backend
  delta; including them unbalances §3 toward C4.5 plumbing. One clause in 3.1 at
  most. **[FLAG — author decision: if the paper's C4.5 timings lean heavily on
  pruning time, this may deserve 2–3 sentences in 3.1. Default: omit.]**
- **Multi-GPU, distributed, missing-value / categorical handling.** Not in the
  code (numeric-only, single device); do not speculate. The numeric-only scope
  is already set by the dataset description in Experimental Setup.

---

## 5. Open questions / risks

1. **[RISK — NUMA expectation vs reality]** The author asked for "NUMA / multiple
   sockets … where applicable". The accurate answer is asymmetric: *explicit*
   NUMA placement exists only for GPU staging copies (GPU's node, from sysfs);
   the multicore CPU builder has **no** pinning, partitioning, or socket-local
   data structures in `src/` (pool built without `onStart`; single shared column
   allocation; only the affinity mask and THP are used). The plan handles this
   as an explicit scoping statement in 3.4. *Author must confirm* this framing
   is acceptable vs. e.g. moving a shorter version to Discussion/Limitations.
   Writing anything stronger would be inventing features.
2. **[VERIFY — `--gpu-only` thread semantics]** In \ImplGpu mode the staged
   copies still use ≥ 8 copier threads (bounded by allowed CPUs) even with
   `--threads 1` (`gpu_host.cuh: kMinCopyThreads`). If the paper presents
   \ImplGpu as "the GPU alone", a footnote-level clause is needed so the
   comparison with \ImplHybrid is not misread as "CPU does nothing". Results
   captions already imply CPU threads exist; §3.5's bullet covers it, but the
   author should confirm the intended framing of the \ImplGpu/\ImplHybrid
   comparison (device-isolation vs realistic-configuration).
3. **[VERIFY — threshold constants in Approach vs Experimental Setup]** The
   constants 32,768 / 4,096 rows and the `gpu-min-rows` auto-clamps are
   compile-time defaults (`options.h`). If Experimental Setup already tabulates
   them, §3 should give them once and cross-reference, not duplicate. Decide
   which section owns the numbers before writing to avoid drift between the two.
4. **[RISK — sweep-filter correctness argument]** Both the CPU estimate filter
   (3.3) and the GPU two-pass filter (3.5) rest on proven error bounds that are
   argued in code comments, not in any citable document, and `docs/SIMPLE.md`
   Risks A/B record that near-tie adversarial cases are argued, not directly
   tested. In the paper, phrase as "the bound guarantees every cut that could
   win or tie is scored exactly" (a design property) — do not upgrade to a
   formal proof claim, and do not mention Risks A/B unless Discussion has a
   limitations paragraph ready to receive them.
5. **[QUESTION — figure]** Is there an existing pipeline figure to reuse, or
   must one be drawn? §3 without a figure is feasible but the Hybrid mechanism
   (3.6) is markedly easier with one. If drawn new, budget author time; the
   plan's word counts assume the figure carries the level-pipeline detail
   (allowing 3.5's stage list to shrink by ~60 words).
6. **[QUESTION — pruning placement]** The plan assumes pruning *semantics* stay
   in §2/Intro and §3 mentions only the grower→pruner data flow. If the paper
   instead promises pruning as part of "our approach" (the trainer does run it
   inside training time), a short shared paragraph in 3.1 (~80 words, taken
   from 3.3/3.4 budgets) is the cheapest fix — decide before writing, since
   timings in Results include prune time and a reader will look for its
   description somewhere.
7. **[VERIFY — C4.5-on-GPU distinct values]** §3.5 mentions device-side distinct
   values (`DeviceSelect::Unique`) only via the setup bullet's "for C4.5"
   machinery. Confirm whether Experimental Setup or §3.1's grower-contract line
   is the right home; do not let it appear in both.
8. **[STYLE RISK]** `docs/` are written in a distinctive plain style ("CUDA in
   five words", lab-note asides). The paper's Intro/Related Work voice is
   formal LNCS. The writer must translate, not transplant, doc phrasing —
   several doc sentences would read as AI/changelog pastiche if lifted.

---

## 6. Estimated length

- **Prose target: ~1,600 words** (per-subsection budgets in §1), plus the
  section's opening 2–3 sentences absorbed into 3.1's budget.
- **Page arithmetic (LNCS, ~800–850 words/page of dense text, less with
  headings):** 1,600 words ≈ 2 pages of pure text; six subsection headings,
  paragraph breaks, and cross-references push toward 2.3 pages. Adding the
  recommended figure (≈ ⅓ page incl. caption) and/or the optional table
  (≈ ¼ page) lands at **≈ 2.6–2.9 pages** — inside the 2–3 page envelope but
  with no slack. If the figure *and* table are both wanted, trim to
  **~1,350 words** (easiest cuts: 3.5 stage list if the figure shows it,
  3.3 Move-3 detail, 3.4 presort/assembly sentences).
- **Hard floor:** below ~1,200 words the section cannot cover all five
  configurations + shared design at the promised "reader understands how the
  implementations differ" level; if page pressure comes from elsewhere in the
  paper, prefer cutting the optional table over prose.

## 7. Suggested visual (optional but recommended)

**One figure, two panels, placed after 3.1 or in 3.6:**
(a) *Shared representation* — dataset → presorted 8-byte entry columns → node
range `[begin,end)` splitting into two ranges (schematic, ~6 entries per column,
2 features). (b) *Hybrid level step* — GPU processes the level's large nodes;
small children flow device-ring → pinned buffers → host arena → CPU pool tasks,
while the GPU starts the next level. This single figure carries: the layout
(3.1), the level model (3.5), and the hand-off (3.6), and is the cheapest way
to make the Hybrid contribution legible. A small comparison table
(rows: configuration; columns: where nodes are processed / split search /
order of growth / extra memory) is a viable substitute but duplicates §3.1
prose; prefer the figure.

---

*Sources consulted for this plan: `README.md`; `docs/{ALGORITHMS,SIMPLE,CPU,CUDA}.md`
(+ `docs/PERFORMANCE.md` for NUMA/GPU-communication facts only);
`src/build/{simple_grower.cpp,cpu_grower.cpp,presort.h,gpu_grower.cu,gpu_host.cuh}`,
`src/algo/{split_math.h,split_rules.h}`, `src/core/{dataset.h,thread_pool.h,options.h,
tree.h,memory.cpp}`, `src/app/{trainer.cpp,main.cpp}`;
`paper/sections/{introduction,related-work,approach}.tex`, `paper/main.tex`.*
