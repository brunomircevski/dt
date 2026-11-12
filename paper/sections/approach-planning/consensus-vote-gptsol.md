# Round 2 consensus vote — GPT Sol

## 1. Scores

Scale: 1 = poor, 3 = usable with revision, 5 = strong.

| Plan | Structure | Fact caution, especially NUMA | Completeness vs. request | Length realism |
|---|---:|---:|---:|---:|
| Opus | 5 | 4 | 5 | 2 |
| GPT Sol | 5 | 5 | 5 | 5 |
| Muse | 5 | 5 | 5 | 4 |
| Grok | 5 | 5 | 5 | 5 |

### Rationale

**Opus.** The cumulative design narrative and shared prelude are excellent,
and the plan covers every requested backend and design concern. Its NUMA
discussion correctly distinguishes OS first-touch from explicit placement,
but it occasionally leans too far into claims about pages being spread across
sockets and about bounds being proven. The proposed 1,920 words plus a table
and algorithm box exceeds the requested 1,200--1,800-word range; the later
1,550-word fallback is realistic, but the primary budget is not.

**GPT Sol.** The structure is compact, source-oriented, and directly allocates
shared invariants versus backend deltas. It explicitly limits NUMA claims to
GPU staging and dedicated copy threads, treats CPU affinity as a scope
boundary rather than a NUMA feature, and keeps the target at 1,450--1,650
words. Its main weakness is that it does not suggest a visual, although no
visual was required and omitting one protects the page budget.

**Muse.** This is the most comprehensive audit trail and is especially strong
on the four-source-backend/five-paper-configuration distinction, CPU NUMA
non-features, and the GPU staging scope. The six-subsection delta structure is
sound. Its 1,600-word body is realistic by itself, but adding the recommended
figure and optional table leaves little room within three LNCS pages; some
low-level details would have to move to a caption or be cut.

**Grok.** This is the tightest source-critical plan. It catches several easy
errors: serial presort is not parallel, the hand-off arena is not bound to the
GPU's NUMA node, GPU-only still uses copy threads, and external-reference
agreement has caveats. Its 1,590-word allocation is credible. The main
editorial risk is that the serial and GPU subsections still contain enough
microarchitectural detail to become an implementation catalogue if the later
writer follows every bullet.

## 2. Preferred consensus structure

Target: **about 1,500--1,600 words**, with no overview table. If a hybrid
schematic is added later, hold the prose near 1,450 words.

### Section opening: design progression (40--60 words, included in 3.1)

- Name the five paper configurations and state that they differ in data
  maintenance, split-search execution, and node placement rather than learning
  semantics.
- State once that \ImplGpu{} and \ImplHybrid{} are configurations of the same
  CUDA grower: `--cuda --gpu-only` and `--cuda`, respectively.

### 3.1 Shared Semantics and Data Model (200--230 words)

- Numeric binary-tree scope and exact, non-binned split search; refer back to
  the CART/C4.5 treatment rather than re-deriving it.
- Column-major float input and compact integer labels.
- Shared backend-independent split rules, double-precision final scoring,
  \(10^{-12}\) tie handling, and deterministic feature/cut ordering.
- Define the optimized 8-byte `(value, packed row/class)` entry and the
  invariant that a node occupies the same range in every presorted column.
  Leave the radix-sort and partition mechanics to 3.3.
- Flat preorder output and final renumbering after concurrent construction.
- One compact paragraph on shared post-processing and verification:
  backend-internal trees are byte-identical; external CART/C4.5 checks have
  the documented tie/rounding qualifications.

### 3.2 \ImplSerialBase: Textbook Reference (120--140 words)

- Per node and feature: copy row identifiers, comparison-sort by value, score
  every admissible cut exactly, partition into new row vectors, recurse.
- Explain repeated sorting, indirect gathers, allocation, and recursion as the
  costs that motivate 3.3.
- Present it as an executable semantic reference sharing rules and pruning,
  not as a competing algorithm or a straw baseline.

### 3.3 \ImplSerialOpt: Presort, Filter, and Partition (340--380 words)

- Stable LSD radix presort once per feature; stable child partition preserves
  sorted order and changes repeated \(O(Fn\log n)\) sorting into linear
  per-level scans.
- Sequential split search over packed entries; incremental estimates,
  binary-class specialization, conservative block/step bounds, equal-value
  runs, constant-feature removal, and exact scoring of every possible winner.
  Omit the boundary-point shortcut or mention it only generically because of
  its documented near-tie risk.
- Slot-based storage: one spare slot in serial, out-of-place partition for
  large nodes, reusable buffer for small nodes, one-bit marking, terminal-child
  partition avoidance, and reuse of child counts.
- Reused allocations, transparent-huge-page hint, explicit traversal stack,
  and deterministic node-store assembly.
- State that this is the same CPU builder used by \ImplParallel{}, with no
  pool; do not imply that serial presort is parallel.

### 3.4 \ImplParallel: Intra-node and Subtree Parallelism (220--250 words)

- Same optimized CPU algorithm and representation as 3.3.
- Feature/column parallelism for nodes at or above 32,768 rows; subtree tasks
  after splits at or above 4,096 rows.
- Explain the granularity shift: data parallelism near the root and task
  parallelism deeper in the tree; nested `parallelFor` remains deadlock-free
  because the caller participates.
- Up to `min(F-1, threads)` spare slots permit concurrent column movement;
  exceptionally large/narrow cases use blocked row partitioning.
- Parallel presort and final preorder assembly, briefly.
- Multi-socket scope: no CPU worker pinning, socket-aware scheduler, NUMA
  allocation policy, or per-socket replication in the main source. The pool
  respects the process affinity mask, and huge pages address TLB pressure, not
  memory placement.

### 3.5 \ImplGpu: Level-Synchronous Device Growth (310--340 words)

- Define this subsection specifically as `--cuda --gpu-only`: no CPU-grown
  subtrees and no hand-off arena, although the host still stages data, chooses
  among per-feature winners, assembles output, and performs shared
  post-processing.
- Breadth-first processing of all active nodes at a depth; decompose
  `(node, feature)` ranges into 2,048-entry tiles.
- Summarize the level pipeline by stages, not kernel names: constant detection,
  histograms/prefixes, exact split search, one best-cut round trip per level,
  shared host-side feature choice, stable partition, and next-level histogram
  production.
- Device storage uses the same packed entries in one slot per feature plus one
  spare; CUB performs stable radix presort as uploads arrive.
- Precision policy: direct double-precision scoring on fast-FP64 devices;
  otherwise a conservative single-precision filter followed by exact
  double-precision scoring of all candidates that could win or tie. Avoid a
  formal-proof claim.
- Communication/NUMA: pageable data passes through pinned staging buffers.
  Only these buffers and dedicated copy threads are explicitly placed on the
  GPU's PCI NUMA node; the training pool and general host memory are not.

### 3.6 \ImplHybrid: Size-Aware CPU--GPU Growth (240--270 words)

- Delta from 3.5: large nodes remain in the level-synchronous GPU pipeline;
  children below the configurable threshold become \ImplParallel{} subtree
  tasks.
- Give the automatic policy accurately: CART \(N/8192\), clamped to
  1,024--2,048; C4.5 \(N/4096\), clamped to 1,024--4,096. Describe these as
  empirical defaults, not universal optima.
- Explain overlap: gather still-splittable columns into the bounded device
  ring, transfer through two pinned buffers/copy streams into a pageable host
  arena, and launch CPU tasks from host callbacks while the GPU advances.
- Clarify that the host arena is parallel-first-touched, not explicitly bound
  or interleaved by a NUMA policy.
- Shared node storage and final preorder renumbering merge CPU and GPU work
  without changing tree semantics.
- End with the design rationale: GPU throughput for large levels, CPU task
  efficiency for small subtrees, both operating concurrently on one tree.

## 3. Disagreements to resolve

1. **Where to introduce packed presorted entries.**  
   Preferred resolution: define the entry and cross-column range invariant in
   3.1 because four configurations share them; introduce radix presort, slots,
   and partition mechanics only in 3.3.

2. **Whether to include a table, algorithm, or figure.**  
   Preferred resolution: no overview table and no kernel pseudocode. If space
   and author time permit, use one compact hybrid-schedule schematic and cut
   100--150 words from 3.5--3.6.

3. **Strength of correctness wording for candidate filters.**  
   Preferred resolution: say the estimates and bounds are conservative and
   that final winners are scored by the shared exact routine; do not claim a
   published formal proof or feature the restricted boundary-point shortcut.

4. **Where NUMA communication belongs.**  
   Preferred resolution: explain GPU-local staging once in 3.5 because it
   serves both GPU modes; in 3.6 describe the hand-off path and explicitly say
   the arena is not GPU-node-bound. In 3.4 state the absence of a CPU NUMA
   policy.

5. **How much pruning to describe.**  
   Preferred resolution: two or three shared sentences in 3.1 covering the
   grow-then-post-process pipeline and leaf/distinct-value outputs. Omit C4.5
   row arrangement, subtree-raising engineering, and removed GPU-pruning work
   unless pruning becomes a stated contribution.

## 4. Consensus judgment

All four plans converge on the same defensible architecture: one shared
subsection followed by five implementation deltas. The strongest combined
version uses Grok's source cautions, Muse's allocation discipline, Opus's
progressive narrative, and the 1,500--1,600-word budget from the tighter
plans.
