# `--simple` against `--serial`

Files: `src/build/simple_grower.cpp` (`--simple`, about 150 lines) against
`src/build/cpu_grower.cpp` + `src/build/cpu_builder.cpp` (`--serial`, about
2,200 lines). Both score and choose cuts with the same code
(`src/algo/split_math.h`, `src/algo/split_rules.cpp`) and prune with the same
code (`src/algo/cart_pruning.cpp`, `src/algo/c45_pruning.cpp`).

`--simple` is the textbook algorithm: at every node, for every feature, it
sorts the node's row ids by value (`std::sort`), scores every allowed cut with
`cutGain`, keeps the best by the `kTieEps` rule and splits the row list into
two new lists, recursively. `--serial` is the `--parallel` algorithm
([CPU.md](CPU.md)) on one thread. `tests/check.sh` runs `--simple`,
`--serial` and `--parallel` against `tests/golden/`, and all match.

This page lists every difference, why it is faster and why the tree is still
the same. The effects are expected from the complexity alone: **none of the
items was measured on its own**; an ablation is needed before citing numbers
per item. The measured totals are in
[bench/results/cluster-cpu-vs-others](../bench/results/cluster-cpu-vs-others/BENCHMARK.md).

Notation: `N` training rows, `F` features, `K` classes, `n` rows of a node.

## The differences

| # | Item | `--simple` | `--serial` | Expected effect |
|---|---|---|---|---|
| **Data layout** |||||
| 1 | Sorting | `std::sort` of each node's rows, per feature | LSD radix presort once, then a stable partition keeps the order | O(F·n log n) per node → O(F·N) once: the largest item |
| 2 | Column entries | row ids; value and label gathered from the dataset | 8-byte `Entry{value, row << b \| class}` streamed | random gathers → sequential reads |
| 3 | Column slots | none (per-node vectors) | F + 1 slots of N entries, one spare | out-of-place partition without a copy back |
| **Split search** |||||
| 4 | Gain per cut | exact `cutGain` at every allowed cut | incremental estimate with a proven error bound; `cutGain` only for candidates, settled every 256 cuts | about one exact gain per block of cuts instead of one per cut |
| 5 | Two classes | general loop | specialised `TwoClassSweep` | constant factor; HIGGS, SUSY, airline, ECBDL are binary |
| 6 | Convexity bound | none | blocks of 16 cuts skipped when their corner bound is below the best (nodes of 64+ rows) | skips most of a big binary node |
| 7 | `maxGainStep` skip | none | skips the cuts that cannot catch up with the best | fewer estimates |
| 8 | Runs of equal values | one cut test per row | runs passed over, counts added in 4 lanes | constant factor on low-cardinality features |
| 9 | Boundary points | every cut scored | `isSkippableCut`: a cut between two single same-class rows is skipped | fewer estimates (Risk A) |
| 10 | Log tables | `LogTable` below 65,536, `log2` above | `c·log₂c` and `1/c` tabled for every count up to N | no `log2` or division in big nodes |
| 11 | Class count | runtime `K` | templated for K = 3..8 | constant factor |
| 12 | Constant features | sorted and scanned again in every node | dropped for the whole subtree | fewer features deeper down |
| 13 | Children's class counts | counted again from the labels | taken from the winner's sweep | O(n) per node |
| 14 | Fused child sweeps | none | two classes: children of 4096+ rows are swept during the parent's partition | one pass less over the child |
| **Partitioning** |||||
| 15 | What moves | one row-id list (4 B/row) | F columns (8 B/row each), stable | simple moves less but sorts again (item 1) |
| 16 | How | new left/right vectors by `value <= threshold` | 2^15+ rows: out of place into the spare slot, bitmap, non-temporal stores; smaller: in place through a `thread_local` buffer | no allocation, streaming writes |
| 17 | Terminal children | partitioned anyway | skipped if both are leaves; only the other side moved if one is | saves the last partition of every branch |
| 18 | Partition key | value against threshold | position (first `leftCount` entries) | no comparisons |
| **Memory** |||||
| 19 | Allocation | heap vectors per node and feature | allocated once; `thread_local` buffers reused | no allocator in the hot loop |
| 20 | Node storage | `push_back` in preorder | `NodeStore` chunks, renumbered in preorder at the end | no reallocation |
| 21 | Recursion | recursive `growNode` | explicit stack | no stack overflow (Risk D) |
| 22 | Footprint | row ids + one sorted copy per feature of the current node | (F+1)·8N columns, 8N scratch, N goes-left bytes, N/8 bitmap, 16N tables, 4N leaf ids | serial needs ~8F bytes per row more |
| **C4.5 post-processing** |||||
| 23 | Distinct values | `distinctValues(train)`: copy and `std::sort` per feature, timed as pruning | read off the presorted columns, timed as presort | O(F·N log N) → O(F·N) |
| 24 | Leaf of every row | pruning routes all N rows through the tree | recorded by the grower | one pass of N·depth gathers saved |
| 25 | Pruning workspace | pruner allocates 2·F·N floats | borrows the grower's column slots | no allocation, no page faults |
| 26 | Collapse | `compact` only | also remaps the leaf ids | small extra cost |
| **Other** |||||
| 27 | CART pruning | same | same | none |
| 28 | Threads | one | one (no pool) | none |
| 29 | CV folds | each fold sorts from scratch | filters the presorted columns | O(F·N log N) → O(F·N) per fold |
| 30 | Root class counts | plain loop | 4-lane `countClasses` | negligible |

## Why the tree is the same

* **Sort order (1, 2).** A cut depends only on the values and the class counts
  on each side, not on the order of rows inside a run of equal values. Both
  sorts are ascending; `+0.0f` makes −0 and +0 share a radix key, as they
  compare equal for `std::sort`; NaN is rejected by the loader, so
  `operator<` is a strict weak order (Risk E). The entry codec is lossless.
* **Estimates (4, 6, 7, 10).** Only the exact `cutGain` decides, in cut order,
  with the same `kTieEps` rule. An estimate, a block bound or a skip only rules
  a cut out when it provably cannot beat the best exact gain by `kTieEps`;
  this holds if the rounding bounds are true bounds (Risk B).
* **Runs and constant features (8, 12).** No cut is allowed inside a run in
  either backend. If a node's first and last value are within the gap, every
  descendant's are too.
* **Counts and terminal children (13, 17).** Equal by construction;
  `isTerminal` is the same function on the same counts.
* **Partition by position (18).** The threshold is the double midpoint of two
  floats that are strictly apart, so exactly the first `leftCount` values are
  `<=` it.
* **Node order (20).** Preorder numbering is unique for a given shape;
  `majorityClass` breaks ties to the lowest class in both.
* **C4.5 post-processing (23, 24).** The same sorted set of distinct values.
  Routing with float thresholds rounded down sends every training row to the
  leaf the grower put it in.

## Risks

**A. Boundary-point shortcut against the tie rule (theoretical).** Convexity
along a same-class stretch gives `g(i) ≤ max(g(i−1), g(i+1))`, enough for a
strict `>` but not for `> best + 1e-12`. A different cut would need three
consecutive gains within 1e-12 of each other and the best so far in a window
narrower than 1e-12. No case is known: every golden tree and every benchmark
run (`tree_sha256` in the results) agree. An earlier, broader version of the
shortcut did change trees ([ALGORITHMS.md](ALGORITHMS.md), "What was wrong
before").

**B. Rounding bounds.** Items 4, 6, 7 and 9 rely on `margin`, the convexity
slack and `maxGainStep` being true bounds. They are argued in the comments of
`cpu_builder.cpp`, and trees match on every test and benchmark, but no test
targets the bounds directly (e.g. adversarial near-ties in large nodes).

**C. `tries` for CART (harmless).** Skipped blocks are not counted in CART's
`tries`, which CART does not use; C4.5's MDL cost counts every allowed cut.

**D. Stack depth of `--simple`.** `growNode` recurses once per level and
`--max-depth` is unlimited by default, so a degenerate tree tens of thousands
of levels deep could overflow the 8 MB stack. Not seen on any dataset; the
frame size was not measured.

**E. Float order.** `operator<` on floats and the radix key agree without NaN;
`--multiply` only repeats the rows.

**F. Timing attribution.** Item 23 puts the distinct-value sort into
`--simple`'s prune time, and serial allocates an 8N scratch array it does not
use (the pool branches need it). Neither changes a tree; both matter when
build and prune times are reported apart.
