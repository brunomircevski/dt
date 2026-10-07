# Growing a tree on the CPU

Files: `src/build/cpu_builder.*` (the builder), `src/build/cpu_grower.cpp`
(presort, cross-validation folds), `src/build/node_store.*`,
`src/core/thread_pool.*`, `src/algo/split_math.h`, `src/algo/split_rules.cpp`.

## The expensive part

To find a node's split, every feature's rows must be in sorted order: then one
left-to-right sweep can try every threshold while updating the class counts of
the left side (`left[k]`) one row at a time. The previous version sorted every
feature again at every node — `O(F · n log n)` per node.

## Sort once: presorted columns

Each feature is sorted once at the start (`presortColumns`, an LSD radix sort
on the float bits, stable, 3 passes of 11 bits; a pass in which all keys share
the digit is skipped). The histograms of all features are computed first, so
that the features needing the most passes (continuous ones need three, binary
ones one) are sorted first, which balances the threads. The first pass reads
the raw values directly and the last one writes the final column (and, for
C4.5, the sorted values alone, which its thresholds need). For feature `f`
the builder keeps a column of entries

```
Entry { float value; uint32 packed = row << classBits | class }
```

in value order. Carrying the class inside the entry means the sweep never looks
anything up by row id.

A node owns the **same index range `[begin, end)` in every column**. When the
node splits at "feature w, first L entries go left":

1. mark the rows of the first `L` entries of column `w` in a byte array
   `goesLeft[row]`;
2. *stable*-partition every other column's range by `goesLeft`: left rows to the
   front, right rows to the back, each keeping its order.

Both children again own one range in every column, still sorted. Nothing is
ever sorted twice, and each level of the tree costs `O(F · n)` (the SLIQ /
SPRINT idea). The partition is done in place with a small per-thread scratch
buffer for the right rows. The children's class counts come from the sweep of
column `w` (it returns the counts left of its best cut); when both children
are leaves anyway (pure, too small, at the depth limit — common at the bottom
of the tree), steps 1–2 are skipped, and when one of them is, only the other
child's rows are moved (in place, no buffer).

## The sweep (`scanFeatureK`, `scanTwoClasses`)

The result is defined by the simple loop

```
for cut in [minChild, n - minChild]:          # cut = rows going left
    left[class(entry[cut-1])] += 1
    if value[cut-1] and value[cut] are (nearly) equal: continue
    tries += 1                                # C4.5's MDL cost needs it
    if boundary-point shortcut applies: continue
    gain = cutGain(total, left, ...)          # shared with the GPU
    keep it if gain > best + 1e-12            # ties: earliest cut
```

but `cutGain` (a loop over the classes and up to three divisions) is only
called for a few cuts. The sweep works in two steps:

1. **Estimate.** Moving one row left changes the gain's ingredients by one
   term, so at every cut with a threshold a cheap estimate is computed with
   no loop over the classes and no division: Gini from the sums of squared
   class counts (they change by an exact integer) and tabled reciprocals
   `1/n`, entropy from running sums of `c·log2 c` (recomputed after runs and
   every 256 updates, so rounding errors stay small). With two classes the
   counts follow from the number of class-1 rows, and the estimate needs no
   running sums at all. The estimate is provably within a tiny `margin` of
   `cutGain` (a bound on the rounding errors, ~1e-14).
2. **Exact chain on candidates.** A cut whose estimate cannot beat the best
   cut by the tie tolerance cannot change the result. The others are
   buffered with their class counts and settled in order: `cutGain` decides,
   exactly as in the loop above. Before that, every buffered cut before the
   last estimate that beats all earlier buffered ones by more than
   `2·margin + 1e-12` is dropped: that cut's exact gain beats them all by
   more than the tolerance, so whatever they did to the best cut, the result
   is the same once it is reached (this keeps rising stretches cheap).

Two more shortcuts never change the result:

* **Runs of equal values** (most rows of discrete features, e.g. covertype)
  only update the class counts, in a tight loop with four interleaved sets of
  counters, so rows of the same class in a row do not wait for each other.
* **Skipping cuts that cannot win.** One more row moving left raises the
  Gini gain by less than `3/n` and the entropy gain by at most
  `(log2 n + log2 e)/n`. So after a cut whose estimate is far below the best,
  the next `(best - estimate)·n / bound` cuts are skipped: their rows are only
  counted (and, for C4.5, their thresholds). Only skips of at least 8 cuts are
  taken; shorter ones cost more than they save.

The boundary-point shortcut (Fayyad & Irani): if entries `cut-1` and `cut`
have the same class and each is the only entry with its value, and the cuts
on both sides are allowed, then this cut can never be the strict best, so it
is not a candidate. On continuous features this is a coin flip per cut, so it
is decided without a branch.

The sweep also returns the class counts left of its best cut, which gives the
children's class counts without another pass over the winning column.

## Constant features are dropped

A feature whose values are all equal in a node (for C4.5: within `1e-5`) has
no cut there, nor in any node below. The range is sorted, so comparing its
first and last entry tells in O(1). Such a feature is dropped from the node's
feature list (`Subtree::features`): it is neither scanned nor partitioned again
in the whole subtree. This matters a lot for one-hot / binary features: on
covertype (44 of 54 features binary) about half of all scanned entries used to
belong to such columns.

## Parallelism

One thread pool (`ThreadPool`), two kinds of work:

* **Big nodes** (≥ `featureParallelRows` = 65536 rows, e.g. the top levels):
  `parallelFor` over features for the sweep. The partition runs one column per
  thread, or, above 2^18 rows, column by column with all threads working on
  blocks of the column. Marking `goesLeft` is split into blocks too.
* **Subtrees**: after a split of a node with ≥ `nodeTaskRows` = 4096 rows, the
  left child becomes a pool task, and the current thread continues with the
  right child (an explicit stack, no recursion). Nobody waits for a task; the
  grower calls `waitIdle()` once at the end.

`parallelFor` lets the calling thread work too and only ever waits for items
that are already running on another thread, so it can be nested inside pool
tasks without deadlocks.

## Memory

The columns take `8 · F · n` bytes. Partitioning needs a buffer for the rows
going right: a node uses its own range of one shared `n`-entry scratch array,
and per-thread buffers are capped at 2^18 entries, so extra memory is about
`8 · n` bytes plus 2 MB per thread, however many threads run. The sweep's
tables of `c·log2 c` and `1/c` take another `16 · n` bytes for training sets
of 65536 rows or more (smaller counts use shared tables).

Nodes are created in a `NodeStore` (fixed-size chunks, an atomic counter, safe
for concurrent use). At the end the tree is copied into a `Tree`: flat arrays
of nodes and class counts in preorder, so the result does not depend on which
thread created which node, and serial and parallel builds give byte-identical
trees.

## Cross-validation

CART's K-fold cross-validation (`--cv K`) grows K more trees on (K-1)/K subsets. The
grower keeps the presorted columns of the full training set; a fold's columns
are obtained by filtering them (one pass, the order stays sorted) instead of
copying the data and sorting again.

## Post-processing

`src/algo/c45_pruning.cpp`: C4.5 pruning routes every training row once
through the tree (using a row-major copy of the data, so one row's values share
a cache line), groups rows by leaf in preorder — every subtree then owns one
contiguous slice of rows — and only re-routes rows for subtree raising. C4.5's
training-value thresholds are found by binary-searching each threshold among
its feature's sorted training values (a by-product of the presort). CART
cost-complexity pruning only needs the per-node class counts of the tree.
