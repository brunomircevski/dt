# Growing a tree on the CPU

Files: `cpu_builder.h/.cpp`, `thread_pool.h/.cpp`, `split_math.h`,
`split_rules.cpp`.

## The expensive part

To find a node's split, every feature's rows must be in sorted order: then one
left-to-right sweep can try every threshold while updating the class counts of
the left side (`left[k]`) one row at a time. The previous version sorted every
feature again at every node — `O(F · n log n)` per node.

## Sort once: presorted columns

Each feature is sorted once at the start (`presortColumns`, an LSD radix sort
on the float bits, stable). For feature `f` the builder keeps a column of
entries

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
buffer for the right rows. The children's class counts come from the first `L`
entries of column `w`; when both children are leaves anyway (pure, too small,
at the depth limit — common at the bottom of the tree), steps 1–2 are skipped.

## The sweep (`scanFeatureK`)

```
for cut in [minChild, n - minChild]:          # cut = rows going left
    left[class(entry[cut-1])] += 1
    if value[cut-1] and value[cut] are (nearly) equal: continue
    tries += 1                                # C4.5's MDL cost needs it
    if boundary-point shortcut applies: continue
    gain = cutGain(total, left, ...)          # shared with the GPU
    keep the best (ties: earliest cut)
```

The function is a template on the class count (2…8) so the class loops unroll.
For entropy, `c · log₂ c` comes from a 65536-entry table.

The boundary-point shortcut (Fayyad & Irani): if entries `cut-1` and `cut` have
the same class and each is the only entry with its value, and the cuts on both
sides are allowed, then this cut can never be the strict best, so its gain is
not computed.

## Parallelism

One thread pool (`ThreadPool`), two kinds of work:

* **Big nodes** (≥ `minRowsForFeatureParallel` rows, e.g. the top levels):
  `parallelFor` over features, for the sweep and again for the partition.
* **Subtrees**: after a split of a node with ≥ `minRowsForNodeTask` rows, the
  left child becomes a pool task, and the current thread continues with the
  right child. A task writes its result straight into the parent's
  `left`/`right` pointer, so nobody waits for it; the trainer calls
  `waitIdle()` once at the end.

`parallelFor` lets the calling thread work too and only ever waits for items
that are already running on another thread, so it can be nested inside pool
tasks without the deadlocks the old two-pool design had to work around.

All backends make the same decisions in the same order per node, so serial and
parallel produce byte-identical trees.

## Post-processing

`pruning.cpp`: C4.5 pruning routes every training row once through the tree
(using a row-major copy of the data, so one row's values share a cache line),
groups rows by leaf in DFS order — every subtree then owns one contiguous slice
of rows — and only re-routes rows for subtree raising. CART cost-complexity
pruning only needs the per-node class counts kept in every `Node`.
