# Section 3, “Our Approach”: writing plan

## Recommended organization

The current five-backend skeleton is useful, but it should be preceded by a
short shared-design subsection. Otherwise the same exact split semantics,
column representation, and verification argument would have to be repeated
for every implementation.

Proposed structure:

1. **Shared Algorithm and Data Representation**
2. **\ImplSerialBase: Reference Implementation**
3. **\ImplSerialOpt: Presorted Sequential Induction**
4. **\ImplParallel: Multicore Execution**
5. **\ImplGpu: Level-Synchronous GPU Execution**
6. **\ImplHybrid: Size-Aware CPU--GPU Execution**

Use the paper macros for every implementation name. The last two are two
configurations of the same CUDA backend in the source: \ImplGpu{} is
`--cuda --gpu-only`, whereas \ImplHybrid{} is `--cuda` with CPU hand-offs.
This distinction should be stated explicitly.

## 3.1 Shared Algorithm and Data Representation (220--280 words)

### What to say

- Establish the scope without re-teaching CART or C4.5: both algorithms grow
  binary trees over numeric features, evaluate exact thresholds between
  consecutive distinct values, and differ only in their scoring, stopping,
  and pruning rules.
- State the invariant that all five implementations use the same split
  semantics. The CPU and CUDA code compile the same double-precision
  `cutGain` routine; gains within \(10^{-12}\) tie, after which the earlier
  cut and lower feature index win. Small entropy counts use the same
  \(c\log_2c\) table. This is the basis for deterministic backend agreement.
- Describe the input layout once: feature values are 32-bit floats stored
  column-major; labels are mapped to compact integer identifiers. Optimized
  builders transform each feature into sorted 8-byte entries
  `(float value, packed row/class)`. Carrying the class with the value turns
  split search into a sequential scan rather than a row-index gather.
- Introduce the central representation invariant used by all optimized
  backends: a node occupies the same index interval in every presorted feature
  column. A stable partition creates two sorted child intervals, so each
  feature is sorted only once.
- Briefly separate induction from shared post-processing. CART uses fixed-
  alpha cost-complexity pruning; C4.5 performs collapse, converts midpoints to
  training-value thresholds, and applies error-based pruning with subtree
  raising. The growers record class counts and, when needed, each row's leaf
  to avoid reconstructing this information later.
- End with verification, not performance: golden-tree tests compare every
  implementation byte-for-byte; CART was checked node-by-node against
  scikit-learn (apart from documented exact ties), and C4.5 against Quinlan's
  Release 8. Phrase this as semantic validation, leaving timing evidence to
  the evaluation section.

### Supporting files

- `src/core/dataset.h`, `src/core/dataset.cpp`
- `src/algo/split_math.h`
- `src/app/trainer.cpp`
- `src/build/presort.h`, `src/build/presort.cpp`
- `docs/ALGORITHMS.md`
- `README.md`

## 3.2 \ImplSerialBase: Reference Implementation (130--170 words)

### What to say

- Present `--simple` as the transparent reference, not as a competitive
  system: for every node and feature, copy the node's row identifiers, sort
  them by the feature, sweep every admissible threshold, choose the split, and
  recursively materialize left and right row vectors.
- Explain why it is useful: it directly expresses the algorithm and invokes
  the same split rules and pruning code as all optimized versions, isolating
  implementation changes from model changes.
- Identify its costs at a design level: repeated \(O(Fn\log n)\) node-local
  sorting, indirect value/label gathers, per-node vector allocation, and
  recursive traversal. Do not list every micro-optimization absent from it.
- Note that its working representation is row identifiers plus temporary
  sorted copies, unlike the persistent presorted columns below.

### Supporting files

- `src/build/simple_grower.cpp`
- `docs/SIMPLE.md`
- `src/app/trainer.cpp`

## 3.3 \ImplSerialOpt: Presorted Sequential Induction (300--380 words)

### What to say

- Make the main design transition the organizing idea: replace repeated
  comparison sorting by one stable LSD radix presort per feature, followed by
  linear sweeps and stable partitions. Attribute this representation to the
  SLIQ/SPRINT tradition already introduced in Related Work, without repeating
  that history.
- Explain storage concretely but compactly. Each feature column consists of
  8-byte entries and resides in a memory “slot”; a serial run keeps one spare
  slot. Large nodes are partitioned out of place into free slots, while small
  nodes use reusable thread-local storage. A one-bit-per-row bitmap marks the
  selected side for streaming partition. Features constant within a node are
  removed for the whole subtree.
- Group split-search optimizations by purpose rather than enumerating source
  code:
  - incremental gain estimates identify the small set of cuts that need the
    exact shared scoring routine;
  - binary-class specializations and convex upper bounds skip blocks that
    cannot win;
  - runs of equal values are accumulated in bulk, count-dependent logarithms
    and reciprocals are tabled, and children reuse class counts from the
    winning sweep;
  - for sufficiently large binary children, sweeping is fused with the
    parent's partition.
- State the semantic safeguard: estimates and bounds only reject candidates
  that cannot beat the exact incumbent under the shared tie tolerance; the
  final decision remains double precision. Avoid claiming a formal proof
  beyond what the implementation comments establish.
- Mention memory behavior relevant to later NUMA discussion: large allocations
  request transparent 2 MiB huge pages, and large arrays are reused rather
  than repeatedly allocated. Do not claim explicit CPU NUMA binding or
  socket-local allocation; the main CPU builder has no such policy.
- Briefly mention that nodes are first placed in chunked `NodeStore` storage
  and finally copied into deterministic preorder arrays, decoupling tree
  identity from construction order.

### Supporting files

- `src/build/presort.cpp`, `src/build/presort.h`
- `src/build/cpu_builder.cpp`, `src/build/cpu_builder.h`
- `src/build/cpu_grower.cpp`
- `src/algo/split_math.h`
- `src/core/memory.cpp`
- `docs/CPU.md`, `docs/SIMPLE.md`

## 3.4 \ImplParallel: Multicore Execution (210--270 words)

### What to say

- Emphasize that `--parallel` executes the same optimized algorithm and data
  structures as \ImplSerialOpt{}; it is not a separate split algorithm.
- Explain the two-level schedule:
  - for nodes with at least 32,768 rows, feature sweeps, marking, and column
    partitions use data parallelism;
  - after splitting a node with at least 4,096 rows, one child becomes a pool
    task while the current worker continues with the other, exposing subtree
    parallelism as nodes shrink.
- Explain why both levels are necessary: large upper nodes have few subtrees
  but abundant per-feature work, whereas lower levels offer many independent
  subtrees but insufficient work to justify repeatedly waking all threads.
- Mention nested execution safety: the caller participates in `parallelFor`,
  and nested operations wait only for already-running helpers; subtree tasks
  are joined once at the end.
- Describe the memory/performance trade-off: the multicore version reserves up
  to `min(F-1, threads)` spare column slots so several columns can move
  concurrently; exceptionally large nodes with too few movable columns are
  partitioned in row blocks to avoid idle cores.
- Treat multiple sockets cautiously. The pool uses all CPUs allowed by the
  process (`taskset`/`numactl` restrictions are respected), but the main CPU
  path neither detects sockets nor pins workers. Parallel initialization
  first-touches large arrays and huge pages reduce TLB overhead, but explicit
  NUMA placement should not be claimed.

### Supporting files

- `src/build/cpu_builder.cpp`, `src/build/cpu_grower.cpp`
- `src/core/thread_pool.h`, `src/core/thread_pool.cpp`
- `src/core/options.h`, `src/core/options.cpp`
- `src/app/main.cpp`, `src/app/trainer.cpp`
- `docs/CPU.md`
- `docs/PERFORMANCE.md` only for the caution about measured socket placement

## 3.5 \ImplGpu: Level-Synchronous GPU Execution (280--350 words)

### What to say

- Define \ImplGpu{} as `--cuda --gpu-only`: all nonterminal nodes remain on the
  GPU, although host threads still stage transfers, choose among per-feature
  winners, assemble the tree, and perform C4.5 pruning.
- Contrast its schedule with the CPU: process all active nodes of a depth
  together. Each `(node, feature)` sorted interval is divided into 2,048-entry
  tiles handled by 256-thread CUDA blocks, providing enough parallel work even
  when a level contains many medium-sized nodes.
- Summarize one level as a short pipeline rather than listing every kernel:
  identify constant segments; compute tile histograms and prefix counts;
  evaluate cuts; return one best cut per feature to the host for the shared
  CART/C4.5 feature-choice rule; then stably partition all surviving columns.
  The partition also accumulates the next level's histograms, eliminating a
  separate pass after the root.
- Explain exact GPU scoring. Datacenter GPUs with sufficiently fast FP64 score
  cuts once in double precision. Otherwise a conservative single-precision
  pass filters candidates and only possible winners are rescored with the
  shared double-precision routine. Binary-class tile bounds can omit tiles
  that cannot contain the winner.
- Describe device storage at a useful level: presorted entries occupy one slot
  per feature plus one spare; partitioning moves columns through that spare
  rather than keeping two full copies. A bitmap marks at most the smaller side
  of each split, and the split feature is partitioned by position.
- Cover communication and NUMA precisely. Host/device transfers pass through
  pinned staging buffers. On multi-socket systems, the code reads the GPU's
  PCI NUMA node from sysfs, allocates these buffers there, and pins only the
  dedicated copy threads to CPUs of that node. This does not imply that the
  general training pool or all host memory is NUMA-bound.

### Supporting files

- `src/build/gpu_grower.cu`
- `src/build/gpu_kernels.cuh`
- `src/build/gpu_host.cuh`
- `src/algo/split_math.h`
- `src/core/options.h`, `src/core/options.cpp`
- `docs/CUDA.md`
- `docs/PERFORMANCE.md` only for the measured motivation for GPU-local staging

## 3.6 \ImplHybrid: Size-Aware CPU--GPU Execution (220--290 words)

### What to say

- Start from the delta relative to \ImplGpu{}, avoiding a second GPU
  walkthrough: `--cuda` keeps large nodes in the same level-synchronous GPU
  pipeline but transfers smaller child subtrees to the multicore CPU builder.
- Give the policy accurately. A configurable row threshold controls the
  hand-off; the automatic default scales slowly with training-set size and is
  bounded (CART: 1,024--2,048 rows; C4.5: 1,024--4,096 rows). Explain the
  rationale qualitatively: launch and level-wide coordination costs dominate
  for small nodes, while the CPU efficiently exposes them as subtree tasks.
- Explain overlap: small children from a level are gathered into a device
  ring, downloaded through two pinned staging buffers/copy streams, copied
  into a pageable host arena, and launched as CPU tasks by host callbacks.
  The GPU proceeds to the next level while the CPU grows the transferred
  subtrees.
- State the memory/communication choices: every row is handed over at most
  once; only columns that can still split a subtree are gathered; the bounded
  512 MiB device ring replaces a full device mirror of the host arena. The
  arena is first-touched in parallel before use so transfer callbacks do not
  incur page faults.
- Clarify assembly and semantic equivalence: GPU and CPU branches allocate
  from the shared thread-safe node store and are renumbered into preorder at
  the end. Both builders use the same presorted-entry representation and split
  rules, so the device boundary changes scheduling, not the tree.
- Do not describe removed GPU-assisted C4.5 subtree-raising code; the current
  main source performs that pruning on the CPU.

### Supporting files

- `src/build/gpu_grower.cu`
- `src/build/gpu_host.cuh`
- `src/build/cpu_builder.cpp`, `src/build/cpu_builder.h`
- `src/build/node_store.*`
- `src/core/options.h`, `src/core/options.cpp`
- `docs/CUDA.md`

## Shared prelude versus backend deltas

### Explain once in the shared prelude

- CART/C4.5 split and pruning semantics.
- Column-major input and compact class identifiers.
- The optimized 8-byte presorted entry and stable-partition invariant.
- Shared exact scoring, deterministic tie-breaking, and preorder output.
- Cross-backend and external-reference verification.

### Reserve for the individual implementations

- \ImplSerialBase{}: repeated sorting and row-vector recursion.
- \ImplSerialOpt{}: radix presort, optimized sweep, slot-based partition, and
  storage reuse.
- \ImplParallel{}: feature/data parallelism, subtree tasks, and spare-slot
  trade-off.
- \ImplGpu{}: breadth-first tiled kernels, exact/two-pass scoring, device
  slots, and staged PCIe transfers.
- \ImplHybrid{}: row-count hand-off policy, asynchronous communication, and
  concurrent CPU/GPU subtree growth.

## What to omit

- Benchmark numbers, speed-ups, energy, accuracy, and claims that an individual
  optimization caused a measured improvement; these belong in Results, and
  `docs/SIMPLE.md` explicitly says the serial optimizations were not ablated
  one by one.
- A full derivation of Gini, entropy, gain ratio, MDL correction, CART
  cost-complexity pruning, or C4.5 error-based pruning; Introduction and
  Related Work already establish the algorithms.
- An annotated kernel-by-kernel or function-by-function source tour. Kernel
  names are unnecessary unless one concise algorithm box is later desired.
- Fixed microarchitectural constants that do not help distinguish designs
  (warp intrinsics, register counts, exact staging chunk sizes, allocator
  growth factors, and detailed radix digit planning).
- Historical implementations and removed experiments, especially GPU-assisted
  C4.5 subtree raising and obsolete page-locking strategies.
- Hardware-specific bandwidth figures or statements that a given threshold is
  universally optimal.
- Unsupported NUMA claims: no general CPU socket pinning, socket-aware work
  stealing, or explicit per-socket column replication exists in the main
  source.

## Open questions and risks

1. **Five paper implementations versus four source backends.** The source has
   four `Backend` enum values; \ImplGpu{} and \ImplHybrid{} are configurations
   of `Backend::Cuda`. Section 3 should say this directly so that “five
   implementations” is not read as five independent code paths.
2. **Meaning of “same tree.”** Internal tests enforce byte-identical trees
   across local backends. External CART comparison has documented differences
   only when scikit-learn chooses another exactly tied feature; C4.5 has a few
   documented single- versus double-precision edge cases on full covertype.
   Avoid saying every external tree is unconditionally byte-identical.
3. **Boundary-point shortcut.** The optimized CPU/GPU builders skip a narrow
   class-boundary case under a convexity argument, but `docs/SIMPLE.md` records
   a theoretical interaction with the \(10^{-12}\) tie rule and no known
   counterexample. It is safer not to feature this optimization in the main
   prose, or to describe it only as a conservative candidate-elimination rule.
4. **NUMA scope.** Explicit NUMA detection and pinning apply to GPU staging
   buffers and dedicated copy threads. CPU training uses the process affinity
   mask but is not socket-aware. Keep this boundary clear.
5. **CART early stopping under fixed alpha.** The implementation can avoid
   growing a node that fixed-alpha pruning must remove. This is semantically
   important but may distract from the backend comparison. Include it only as
   a shared algorithm-preserving optimization if space permits.
6. **Pruning detail versus page limit.** C4.5 post-processing is sophisticated
   and shared across backends. The section should mention recorded leaf IDs
   and shared CPU pruning, but leave row grouping, row-major routing, and
   subtree-raising internals to documentation unless pruning is a claimed
   contribution.
7. **Terminology.** Use “exact split search” to mean no histogram/binning and
   all admissible cuts considered semantically; candidate bounds may avoid
   explicitly evaluating cuts proven unable to win.

## Estimated length

Target **1,450--1,650 words**, comfortably within approximately 2--3 LNCS
pages depending on figures and pseudocode:

- Shared design: 220--280
- \ImplSerialBase{}: 130--170
- \ImplSerialOpt{}: 300--380
- \ImplParallel{}: 210--270
- \ImplGpu{}: 280--350
- \ImplHybrid{}: 220--290

If space is tight, combine the two serial subsections under “Sequential
Implementations” while retaining separate paragraphs and macros; cut sweep
micro-optimizations before cutting the shared representation or the
GPU/hybrid distinction.
