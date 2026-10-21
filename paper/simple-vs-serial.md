# What `--serial` does that `--simple` does not

Code: `src/build/simple_grower.cpp` (`--simple`, 148 lines) against
`src/build/cpu_grower.cpp` + `src/build/cpu_builder.cpp` (`--serial`, about
1,900 lines). Both use the same scoring and selection code (`src/algo/split_math.h`,
`src/algo/split_rules.cpp`) and the same pruning (`src/algo/cart_pruning.cpp`,
`src/algo/c45_pruning.cpp`). Design doc: [docs/CPU.md](../docs/CPU.md).

`--simple` is the textbook algorithm. At every node, for every feature, it sorts
the node's row ids by value, tries every cut, scores every allowed cut with
`cutGain`, and then splits the row list into two new lists. `--serial` runs the
same algorithm as `--parallel` (since 4b1ced0, [docs/PERFORMANCE.md](../docs/PERFORMANCE.md)),
on one thread. This page lists every difference between the two, why each one
is faster, and why the tree is still the same. `tests/check.sh:46` runs
`--simple --serial --parallel` against `tests/golden/`, and all three match.
The *effect* column gives the expected impact from the complexity alone; none of
the items was measured on its own. **The paper needs an ablation before it can
cite numbers per item.**

Notation: `N` training rows, `F` features, `K` classes, `n` rows of a node,
`D` tree depth.

## Summary

| # | Item | `--simple` | `--serial` | Expected effect |
|---|---|---|---|---|
| **Data layout** |||||
| 1 | Sorting | `std::sort` of each node's rows, per feature | radix presort once, then a stable partition keeps order | O(F·n log n) per node → O(F·N) once: the largest item |
| 2 | Column entries | row ids; value and label gathered from the dataset | 8-byte `Entry{value, row<<b \| class}` streamed | random gathers → sequential reads |
| 3 | Column slots | none (per-node vectors) | F+1 slots of N entries, one spare for out-of-place | lets the partition write out of place without copying back |
| **Split search** |||||
| 4 | Gain per cut | exact `cutGain` at every allowed cut | incremental estimate with a proven error bound; `cutGain` only for candidates | about 1 exact gain per sweep block instead of per cut |
| 5 | Two classes | general loop | `TwoClassSweep`, specialised | constant factor |
| 6 | Convexity bound | none | blocks of 16 cuts skipped when the bound is below the best | skips most of a big binary node |
| 7 | `maxGainStep` skip | none | skips cuts that cannot catch up with the best | fewer estimates |
| 8 | Runs of equal values | one cut test per row | runs skipped; counts moved in 4 lanes | constant factor on low-cardinality features |
| 9 | Boundary points | every cut tried | cuts between two same-class rows skipped | fewer estimates; see Risk A |
| 10 | Log tables | `LogTable` < 65,536, `log2` above | `c·log₂c` and `1/c` for every count up to N | no `log2` calls in big nodes |
| 11 | Class count K | runtime `K` | templated for K = 3..8 | constant factor |
| 12 | Constant features | sorted and scanned again in every node | dropped for the whole subtree | fewer features deeper down |
| 13 | Class counts of the children | counted again from the labels at the child | taken from the winner's sweep | O(n) per node saved |
| 14 | Fused child sweeps | none | small children of 2-class nodes are swept during the parent's partition | one less pass over the child |
| **Partitioning** |||||
| 15 | What is partitioned | one row list (4 B/row) | F columns (8 B/row each), stable | simple moves less here but must sort again (item 1) |
| 16 | How | new left/right vectors by `value <= threshold` | ≥ 2^15 rows: out of place into the spare slot, `leftBits` bitmap, non-temporal stores; smaller: in place with `goesLeft_` bytes | no allocation, streaming writes |
| 17 | Terminal children | partitioned anyway | partition skipped if both are leaves; only one side moved if one is | saves the last partition of every branch |
| 18 | Partition key | value vs threshold | position (`leftCount`) | no comparisons |
| **Memory** |||||
| 19 | Allocation | heap vectors per node and per feature | everything allocated once; `thread_local` buffers reused | no allocator in the hot loop |
| 20 | Node storage | `push_back` into `Tree` in preorder | `NodeStore` chunks, preorder renumbering at the end | no reallocation |
| 21 | Recursion | recursive `growNode` | explicit stack | no stack overflow (Risk D) |
| 22 | Footprint | O(N) row ids + per-node sorted copies | (F+1)·N·8 B work, 8N scratch, N goesLeft, N/8 bitmap, 16N tables, 4N leaves | serial uses ~F·8 B per row more |
| **Pruning and C4.5 post-processing** |||||
| 23 | Distinct values (C4.5) | `distinctValues(train)`: copy + `std::sort` per feature, timed inside pruning | read off the presorted columns, timed as presort | O(F·N log N) → O(F·N) |
| 24 | Leaf of every row (C4.5) | none: pruning routes all N rows through the tree | `leafOfRow` filled by the grower; routing skipped | one gathering pass over the dataset saved |
| 25 | Workspace for pruning (C4.5) | pruner allocates 2·F·N floats | borrows the grower's column slots | no allocation, no page faults |
| 26 | Collapse (C4.5) | `compact` only | also remaps `leafOfRow` | small extra cost for serial |
| 27 | CART pruning | same | same | none |
| **Miscellaneous** |||||
| 28 | Threads | one | one (pool is `nullptr`) | none |
| 29 | Cross-validation folds | sort each fold from scratch | filter the presorted columns by fold | O(F·N log N) → O(F·N) per fold |
| 30 | Root class counts | plain loop | 4-lane `countLabels` | negligible |

---

## Data layout

### 1. Presort once vs sort at every node

* **`--simple`:** `growNode` copies the node's row ids and calls `std::sort`
  by `train.value(row, feature)` once per feature
  (`simple_grower.cpp:49-56`). Every node of the tree sorts all of its rows
  again for every feature: O(F · n log n) per node and O(F · N log N · D) per
  tree in the worst case. Every comparison gathers two values from a
  column-major dataset at random row ids.
* **`--serial`:** `presortColumns` sorts each feature once with an LSD radix
  sort (11-bit digits, three passes, passes whose digit is constant are skipped)
  on a key that orders floats (`cpu_builder.cpp:76-79, 94-109, 117-195,
  199-270`), into the slots of `work_` (`cpu_grower.cpp:146-189`). From then on
  the partition is stable (items 15, 16), so every node's rows stay sorted in
  every column and no node sorts again.
* **Faster:** O(F·N) once instead of O(F·n log n) at every node, and sequential
  instead of gathering. This is the main difference between the two backends.
* **Same tree:** a cut depends only on the multiset of values and the class
  counts on each side, not on the order of the rows within a run of equal
  values. Both orders are ascending in value. `sortColumn` adds `+0.0f`
  (`cpu_builder.cpp:122`) so −0 and +0 share a key; `std::sort` on floats also
  treats them as equal, and a midpoint is computed from the values either way
  (`split_rules.cpp:15-18`), so the threshold is the same. NaN is rejected by
  the loader (`dataset.cpp:221`), so `operator<` is a strict weak order and both
  sorts agree (Risk E).

### 2. Label-packed entries vs row ids

* **`--simple`:** works on `std::vector<uint32_t>` row ids. The sweep reads
  `labels[rows[cut-1]]` (`simple_grower.cpp:90-91`) and `train.value(...)` for
  the gap test: two dependent gathers per cut.
* **`--serial`:** columns hold `Entry{float value; uint32 packed}` with
  `packed = row << classBits | class` (`split_math.h:28-43`); the sweep reads
  the class from the entry it already streams.
* **Faster:** no random access in the sweep; 8 B per cut, sequential, which the
  hardware prefetcher handles.
* **Same tree:** the codec is lossless (`classBits` is chosen to hold every
  class), so the sweep sees the same values and classes.

### 3. Column slots and a spare slot

* **`--simple`:** no columns; each node owns a vector of row ids.
* **`--serial`:** `slots_ = F + spareSlots` (1 spare with no pool,
  `cpu_grower.cpp:25, 89-91`), `work_` holds `slots_ · N` entries
  (`cpu_grower.cpp:30`). A node writes a feature's partitioned column into a
  free slot and the slot map is updated (`cpu_builder.cpp:553-643`), so nothing
  is copied back.
* **Faster:** an out-of-place partition costs one read and one write instead of
  a write plus a copy back.
* **Same tree:** only where the entries live changes; their order is the same
  stable order.

## Split search

### 4. Incremental estimates and candidates vs exact gain at every cut

* **`--simple`:** `bestCut` evaluates `cutGain` (`split_math.h:96-124`) at
  every allowed cut and keeps it if `gain > best.gain + kTieEps`
  (`simple_grower.cpp:84-111`).
* **`--serial`:** `scanFeatureK` updates the left impurity incrementally
  (Gini squares, entropy `c·log₂c` terms, with periodic exact recomputation,
  `cpu_builder.cpp:1431-1456`) and keeps a proven bound `margin` on the
  estimate's rounding error (`1418-1425`). A cut whose estimate is within the
  margin of the best goes into a candidate buffer (`1549-1566`); every
  `kSweepBlock = 256` cuts (`952-957`) `settleCandidates` computes the exact
  `cutGain` of the buffered cuts in order and applies the same `kTieEps` rule
  (`1023-1054`). The two-class sweep does the same (`1299-1315`).
* **Faster:** an incremental update is a few adds; `cutGain` recomputes both
  sides over K classes. Most cuts are never scored exactly.
* **Same tree:** the decision is made only by the exact `cutGain`, in cut
  order, with the same tie rule. An estimate only rules a cut out when its
  estimate plus the margin cannot beat the best exact gain plus `kTieEps`
  (the clear-lead test uses `2·margin + kTieEps`). This holds only if the
  margin is a true bound; that is argued in the code comments, not tested in
  isolation (Risk B).

### 5. Two-class sweep

* **`--simple`:** one loop for any K.
* **`--serial`:** two classes and `n < tables_.size()` dispatch to
  `TwoClassSweep` (`cpu_builder.cpp:913-947, 1083-1346, 1348-1356`): one counter,
  the right side is `total − left`.
* **Faster:** constant factor; the common case on the benchmark datasets
  (HIGGS, airline, ECBDL, SUSY are binary).
* **Same tree:** same cut order, same exact `cutGain` for the decision.

### 6. Convexity bound over blocks of cuts (two classes)

* **`--simple`:** none.
* **`--serial`:** the gain is convex in the left class counts, so for a block of
  `kBoundBlock = 16` cuts it is at most the maximum over the corners of the box
  between the counts before and after the block. A block whose bound (with a
  rounding slack) is below the best is skipped without estimates
  (`cpu_builder.cpp:1003-1012, 1197-1242`). Used in nodes of at least
  `kBoundedSweepRows = 64` rows.
* **Faster:** in a big binary node most blocks are far below the best cut.
* **Same tree:** a skipped block contains no cut that beats the best by
  `kTieEps`, so the first best cut is the same. Rests on the bound slack being
  correct (Risk B). Skipped blocks are not counted in `tries` for CART, which
  does not use it (Risk C).

### 7. `maxGainStep` skipping

* **`--simple`:** none.
* **`--serial`:** moving one row changes the gain by at most `maxGainStep`
  (`cpu_builder.cpp:967-973`); when the current estimate is far below the
  best, the sweep jumps the number of cuts it would take to catch up
  (`983-991, 1567-1576`).
* **Faster:** fewer estimates in long stretches of bad cuts.
* **Same tree:** the skipped cuts provably cannot exceed the best; the bound
  includes the rounding margin (Risk B).

### 8. Runs of equal values

* **`--simple`:** tests `isCut(lower, upper, minValueGap)` at every cut
  (`simple_grower.cpp:95-97`, `split_math.h:166-168`).
* **`--serial`:** a run of equal (or, for C4.5, within `1e-5`) values is passed
  over as a whole and its class counts are added with four interleaved lanes
  (`cpu_builder.cpp:1494-1511, 1521-1528`).
* **Faster:** on low-cardinality features (airline, chess, ECBDL) most rows are
  inside runs.
* **Same tree:** no cut is allowed inside a run, in either backend; the counts
  after the run are the same sum.

### 9. Boundary-point shortcut

* **`--simple`:** every allowed cut is scored.
* **`--serial`:** `isSkippableCut` (`split_math.h:170-183`) skips a cut between
  two single rows of the same class whose neighbouring cuts are both allowed
  (Fayyad & Irani): the gain is convex along such a stretch, so the cut is
  never better than both neighbours (`cpu_builder.cpp:1252-1256, 1530-1533`).
* **Faster:** fewer estimates; large on data with long same-class stretches.
* **Same tree:** the comment states the skipped cut "can never be the first
  best cut". Convexity proves it is not above both neighbours; it does not
  prove it under the `kTieEps` rule in every case. See Risk A. An earlier,
  broader version of this shortcut did change trees
  ([docs/ALGORITHMS.md](../docs/ALGORITHMS.md), "What was wrong before").

### 10. Count tables

* **`--simple`:** `weightedImpurity` uses `LogTable` (`kLogTableSize = 65536`,
  `split_math.h:58-66, 74-92`) and calls `log2` for larger counts.
* **`--serial`:** with N ≥ 65,536, `fillCountTables(N+1)` fills `c·log₂c` and
  `1/c` for every count up to N (`cpu_grower.cpp:102-110`,
  `cpu_builder.cpp:272-294`).
* **Faster:** no `log2` or division in the sweep of big nodes.
* **Same tree:** the estimates use the tables; the decision uses `cutGain`,
  identical in both. The tables only change which cuts become candidates, and
  the margin covers them. *(The CPU/GPU agreement in ALGORITHMS.md relies on
  `cutGain` being the decider; that the table values equal `xlog2x` bit for bit
  was not checked.)*

### 11. Templated class count

* **`--simple`:** `std::vector<uint32_t>` counts of runtime size.
* **`--serial`:** `scanFeatureK<K>` for K = 3..8, dynamic beyond
  (`cpu_builder.cpp:913-947, 1381-1587`).
* **Faster:** fixed-size arrays in registers, unrolled loops.
* **Same tree:** same arithmetic.

### 12. Constant features dropped for the subtree

* **`--simple`:** every node sorts and scans every feature.
* **`--serial`:** a feature whose first and last value in the node are within
  the gap (or which the sweep found to have no allowed cut) is removed from the
  node's feature list, and the children inherit the list
  (`cpu_builder.cpp:361-375, 1714`).
* **Faster:** deep nodes scan far fewer features.
* **Same tree:** if first and last are within `minValueGap`, every adjacent pair
  is too, and a subset of the node's rows stays within that range, so no
  descendant has an allowed cut on that feature.

### 13. Child class counts from the sweep

* **`--simple`:** each child counts `train.labels[row]` again
  (`simple_grower.cpp:32-35`).
* **`--serial`:** the winner's left counts come from the sweep, right =
  parent − left (`cpu_builder.cpp:421-430`).
* **Faster:** O(n) per node saved.
* **Same tree:** the counts are equal by construction.

### 14. Fused child sweeps

* **`--simple`:** none.
* **`--serial`:** for two classes and children below `kSweptChildRows = 2^12`
  rows, the parent's partition stages each child's entries through a
  1024-entry buffer and sweeps them while they are in L1
  (`cpu_builder.cpp:72-73, 499-517, 1596-1716`); the child reuses the result
  (`394-408`).
* **Faster:** removes one read of the child's columns.
* **Same tree:** the same sweep code on the same entries in the same order.

## Partitioning

### 15. What is moved

* **`--simple`:** one list of 4-byte row ids per node
  (`simple_grower.cpp:62-66`).
* **`--serial`:** all F live columns, 8 B per entry, stable.
* **Faster:** per node, simple moves less (4n vs 8Fn bytes), but it pays for it
  with the sort of item 1. The partition is the price of never sorting again.
* **Same tree:** stability keeps each child's columns sorted.

### 16. Out of place with a bitmap and streaming stores vs new vectors

* **`--simple`:** allocates `left` and `right` vectors, pushes each row by
  `value <= threshold`, frees the parent's list (`simple_grower.cpp:62-67`).
* **`--serial`:** nodes of at least `kOutOfPlaceRows = 2^15` rows mark left rows
  in `leftBits_` (plain stores without a pool, `cpu_builder.cpp:465-471`) and
  write each column into the spare slot with `_mm_stream_si64`, then `sfence`
  (`700-739`). Smaller nodes mark `goesLeft_` bytes (`480`) and partition in
  place through a `thread_local` buffer (`682-692, 824-871`). The blocked and
  in-place big-node branches need a pool and are dead code in serial
  (`496-498, 652-679, 748-817, 876-911`).
* **Faster:** a bitmap of N/8 bytes stays in cache; streaming stores avoid
  reading the destination; no allocation.
* **Same tree:** both send exactly the rows of the left count to the left (item
  18), in sorted order.

### 17. Skipping the partition of terminal children

* **`--simple`:** always partitions, then each child discovers it is a leaf.
* **`--serial`:** child terminality is checked before the partition; if both
  are leaves it records their rows and returns; if one is, `Keep` moves only the
  other side (`cpu_builder.cpp:443-454, 485`).
* **Faster:** the last level of every branch is not partitioned.
* **Same tree:** `isTerminal` is the same function on the same counts and depth
  (`split_rules.cpp:69-84`).

### 18. Partition by position vs by value

* **`--simple`:** `value <= threshold`.
* **`--serial`:** the first `leftCount` entries of the winning column go left.
* **Same tree:** the threshold is the double midpoint of two floats that are
  strictly apart (`isCut` requires `lower < upper − gap`), so exactly the first
  `leftCount` values are `<=` it.

## Memory

### 19. Allocation in the hot path

* **`--simple`:** per node: a class-count vector, a cut vector per feature, a
  sorted copy of the row ids per feature, two child vectors.
* **`--serial`:** all buffers are allocated in the constructor
  (`cpu_grower.cpp:21-35`); `cuts`/`cutLefts` and the partition buffer are
  `thread_local` and reused (`cpu_builder.cpp:388-393, 682-692`).
* Both go through the global `operator new` in `src/core/memory.cpp`, which
  aligns blocks of 2 MiB and more to huge pages; it applies equally.

### 20. Node storage

* **`--simple`:** appends to `tree.nodes` and `classCounts` in preorder as it
  recurses (`simple_grower.cpp:36-41`), with vector reallocation.
* **`--serial`:** `NodeStore` in chunks of 4,096 nodes (`node_store.cpp:58-70`),
  then `toTree` renumbers in preorder and remaps `leafOfRow`
  (`node_store.cpp:72-87`, `tree.h:161-249`).
* **Same tree:** the preorder numbering is unique for a given shape, and
  `majorityClass` breaks ties toward the lowest class in both
  (`tree.cpp:13-22`).

### 21. Recursion vs explicit stack

* **`--simple`:** `growNode` recurses (`simple_grower.cpp:70-75`); the depth
  equals the tree depth, unlimited by default (`options.h:120`).
* **`--serial`:** an explicit stack (`cpu_builder.cpp:339-348, 526-544`).
* Not a speed item; see Risk D.

### 22. Footprint

* **`--serial`:** `work_` (F+1)·N·8 B; `scratch_` 8N B (allocated,
  `cpu_grower.cpp:25`, but not touched in serial as far as I can tell: it is
  used only by the pool branches); `goesLeft_` N B; `leftBits_` N/8 B; count
  tables 16N B for N ≥ 65,536; `leafOfRow` 4N B; and for `--cv` another F·N·8 B
  of presorted columns (`cpu_grower.cpp:31-32`).
* **`--simple`:** row ids, at most about 2·N·4 B live along the recursion plus
  one sorted copy per feature of the current node.
* `--serial` needs about F·8 B per row more. For the benchmark sizes it fits;
  it is the reason serial lends this memory to the pruner (item 25).

## Pruning and C4.5 post-processing

### 23. Distinct training values

* **`--simple`:** `grow` does not fill `distinct`, so `trainer.cpp:207-209`
  calls `distinctValues(train)`: a copy of each feature, `+0.0f`, `std::sort`,
  `std::unique` (`dataset.cpp:473-488`). This runs inside the pruning timer
  (`trainer.cpp:200`), so `--simple` reports it as pruning time.
* **`--serial`:** the last value of each run, read off the presorted columns
  with a bit-pattern compare (`cpu_grower.cpp:114-142`), timed with the presort
  (`43-51`).
* **Faster:** O(F·N) instead of O(F·N log N).
* **Same tree:** the same sorted set of values (−0 normalised in both; no NaN).
  The bit compare and `std::unique`'s `==` agree once −0 is gone.
* **Benchmark note:** the build/prune split differs between the backends for
  this reason; a comparison of prune times alone is not like for like.

### 24. Leaf of every training row

* **`--simple`:** no `leafOfRow`. `arrange` (`c45_pruning.cpp:312-397`) routes
  all N rows through the tree with `routeRange` (16 lanes, column-major reads,
  `222-285`) to find each row's leaf.
* **`--serial`:** `recordLeaf` writes each row's leaf as nodes become leaves
  (`cpu_builder.cpp:308-328`); the pruner maps the ids to its slots and skips
  routing (`c45_pruning.cpp:332-344, 408-418`).
* **Faster:** saves a pass of N · depth gathers.
* **Same tree:** routing with the float threshold rounded down (`Hop`,
  `c45_pruning.cpp:198-214`) sends every training row to the leaf the grower put
  it in (item 18). The arrangement is then the same counting sort by leaf, in
  row order (`433-446`).

### 25. Workspace for the pruner

* **`--simple`:** `workspace()` returns an empty span
  (`simple_grower.cpp:136`); `c45PessimisticPrune` allocates 2·F·N floats
  (`c45_pruning.cpp:1041-1069`).
* **`--serial`:** lends `work_` ((F+1)·N·8 B ≥ 2·F·N·4 B,
  `cpu_grower.cpp:83-85`).
* **Faster:** no allocation and no first-touch page faults.
* **Same tree:** scratch memory only.

### 26. Collapse

* **`--simple`:** `c45CollapseUselessSplits` only compacts the tree
  (`c45_pruning.cpp:985-988`).
* **`--serial`:** also builds the owner/remap tables to keep `leafOfRow` valid
  (`963-1012`). A small cost that pays for item 24.

### 27. CART pruning and the row arranger

* `cartCostComplexityPrune` (`trainer.cpp:236`, `cart_pruning.cpp`) is the same
  code for both. The pruner's arranger hook (`c45_pruning.cpp:346-351`) is not
  overridden by `CpuGrower`, so it has no effect for serial, and simple never
  reaches it.

## Miscellaneous

### 28. Both are single-threaded

* `buildPool` is `nullptr` for `Serial` and `Simple` (`trainer.cpp:161-163`).
  `parallelFor` with a null pool is a plain loop (`thread_pool.h:118-126`);
  the dividing sweep needs a pool (`divide = pool_ != nullptr`,
  `cpu_builder.cpp:1080`), as do the blocked partition (`496-498`) and the
  atomic bitmap marking (`465-471`). `options.cpp:194-196` rejects
  `--task-rows` and `--feature-parallel-rows` for both.
* Loading and evaluation still use `ThreadPool pool(threadCount - 1)`
  (`main.cpp:77`) with every backend, so a wall-clock time of the whole run is
  not single-threaded; the build and prune times are.

### 29. Cross-validation folds

* **`--simple`:** each fold tree starts from scratch, with its own sorts.
* **`--serial`:** the main tree's presorted columns are kept in `sorted_` and
  each fold copies them with a keep-filter on the fold's rows
  (`cpu_grower.cpp:156-188`), which keeps the order.
* **Same tree:** a filtered sorted column is the sorted column of the fold.

### 30. Root class counts

* **`--simple`:** a loop in `growNode`.
* **`--serial`:** `countClasses` with 4 interleaved lanes
  (`dataset.cpp:411-471`). Negligible next to the rest.

---

## Risks

**A. Boundary-point shortcut against the tie rule (theoretical).**
Simple evaluates every cut and moves the best only on
`gain > best + kTieEps` (`simple_grower.cpp:105-109`). Serial skips a
boundary cut `i` (`split_math.h:170-183`, `cpu_builder.cpp:1252-1263, 1530-1563`).
Convexity along the stretch gives `g(i) ≤ max(g(i−1), g(i+1))`, which is enough
for strict `>`, not for `> + 1e-12`. A divergence needs, with `B` the best gain
before cut `i−1`:

* `g(i−1) ≤ B + ε` (cut `i−1` not taken), `g(i) > B + ε` (simple takes `i`),
* `g(i+1) ≤ g(i) + ε` (simple keeps `i`), `g(i+1) > B + ε` (serial takes `i+1`).

Simple then ends at `i` and serial at `i+1`: a different threshold and left
count. Convexity forces `g(i) − g(i−1) ≤ g(i+1) − g(i) ≤ ε = 1e-12`, so it needs
three consecutive near-equal gains within 1e-12 and `B` in a window of width
below 1e-12. Rounding that breaks convexity by ~1e-16 opens a smaller window of
the same kind. No case is known: `tests/check.sh` and the golden trees agree,
and the bench runs record `tree_sha256` (`bench/results/cluster-simple/`).
**Uncertain; worth a sentence in the paper or an extra check that a skipped cut
is never within `kTieEps` above the best.**

**B. Correctness depends on proven rounding bounds.** Items 4, 6, 7 and 9
replace exact gains with estimates plus a bound (`margin`, the convexity
`boundSlack`, `maxGainStep`). If any bound were too small, serial could miss a
cut that simple finds. The bounds are argued in comments
(`cpu_builder.cpp:952-1012, 1418-1425`), and the trees match on every test and
benchmark checked; I found no error, but there is no test that targets the
bounds directly (e.g. adversarial near-ties on large nodes).

**C. `tries` differs for CART (harmless).** `TwoClassSweep` counts tries only
when `HasGap` (C4.5); skipped blocks leave CART's count off. `chooseCart`
ignores `tries` (`split_rules.cpp:92-113`); `chooseC45` uses it
(`115-189`) and serial counts every allowed cut there, so C4.5's MDL cost is the
same.

**D. Stack overflow in `--simple` (crash, not a different tree).**
`growNode` recurses once per level (`simple_grower.cpp:70-75`) and `--max-depth`
defaults to unlimited (`options.h:120`). Each frame holds vectors and a few
locals, so with the default 8 MB stack a degenerate tree a few tens of thousands
of levels deep could crash where serial would not. Not seen on the benchmark
datasets; *the frame size was not measured*.

**E. Float order of `std::sort` vs the radix key (uncertain, believed safe).**
Simple sorts with `operator<` on raw floats, serial with the radix key. They
agree when there is no NaN: the loader rejects NaN (`dataset.cpp:221`),
`--multiply` only scales finite values (infinity stays infinity, no `inf·0`),
and ±0 tie in `operator<` and share a key after `+0.0f`. Only the order inside
equal-value runs can differ, which does not change any cut (item 1).

**F. Timing attribution.** Item 23 moves the distinct-value sort into
`--simple`'s prune time, and `scratch_` (item 22) is allocated by serial but
unused. Neither changes a tree; both matter when the paper reports build and
prune times apart.
