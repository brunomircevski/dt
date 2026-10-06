# CART and C4.5 as implemented here

Both algorithms grow a binary tree on numeric features top-down. At every node
they look at every feature, sort the node's rows by that feature, try every
threshold between two neighbouring values, and keep the best one. They differ in
*how* "best" is measured, in *when* to stop, and in how the grown tree is
*pruned*. All of that lives in three small files:

| File | What |
|------|------|
| `split_math.h` | Impurity, gain, tie rules. Compiled by both g++ and nvcc, so the CPU and the GPU score cuts with literally the same code. |
| `split_rules.cpp` | Stopping rules, minimum child size, how the winning feature is chosen (CART vs C4.5). |
| `pruning.cpp` | C4.5 post-processing and pruning, CART cost-complexity pruning. |

Not supported (on purpose): missing values (C4.5's fractional weights) and
multi-way / subset splits on categorical features.

---

## Notation

A node has `n` rows and class counts `c_1 .. c_K`. A threshold `t` on feature
`f` sends rows with `x_f <= t` left (`n_L` rows, counts `l_k`) and the rest
right (`n_R`, `r_k`).

The code works with **weighted impurities** (impurity times row count), which
need no division per class and make gains exact sums:

* Gini: `G(c) = n - Σ c_k² / n`  (= n · (1 − Σ p_k²))
* Entropy: `E(c) = n log₂ n − Σ c_k log₂ c_k`  (= n · H; C4.5 calls it `TotalInfo`)

`gain = (W(parent) − W(left) − W(right)) / n` is then the usual impurity
decrease (Gini) or information gain (entropy).

---

## CART (Breiman, Friedman, Olshen, Stone 1984)

**Split selection.** Every feature's best threshold is the one with the largest
impurity decrease (Gini by default, entropy optional). Across features the
largest decrease wins. Thresholds are midpoints between the two neighbouring
values, computed in `double` (the midpoint of two floats is exact in double),
like scikit-learn.

**Stopping.** A node becomes a leaf when it is pure, has fewer than
`--min-split` rows, is at `--max-depth`, or no threshold leaves `--min-leaf`
rows on both sides. A split with zero gain is still accepted (CART grows the
maximal tree and lets pruning decide; scikit-learn does the same — it matters
for XOR-like data). `--min-decrease` is scikit-learn's `min_impurity_decrease`.

**Pruning: minimal cost-complexity.** For a subtree `T` let `R(T)` be its
training misclassification rate and `|T|` its number of leaves. For a given
`α`, `T(α)` is the smallest subtree minimising `R(T) + α|T|`
(`cartCostComplexityPrune`, a bottom-up dynamic program).

* `--alpha X` prunes with a fixed `α` (a rate, so it does not depend on the
  dataset size).
* `--cv K` does what Breiman describes: compute the weakest-link sequence
  `α_1 = 0 < α_2 < …` of the full tree, grow a tree on each of K folds'
  complement, measure each fold tree's error at the geometric midpoints
  `√(α_k α_{k+1})`, and pick the simplest tree within one standard error of
  the best (1-SE rule). The sequence is computed in one bottom-up pass by
  representing each subtree's optimal cost as a concave piecewise-linear
  function of `α` (`cartPruningSequence`).

**Verified against** scikit-learn (`python/compare_reference.py cart`): trees
are identical except where two splits have exactly the same impurity decrease
(scikit-learn visits features in random order, this code takes the lowest
feature index); the script recomputes both gains at every such node to prove it
is a tie. The pruning sequence was checked against a direct textbook
weakest-link implementation.

---

## C4.5 (Quinlan 1993, Release 8 — the last C4.5)

This follows the original C source (`contin.c`, `build.c`, `info.c`,
`prune.c`, `stats.c`) rule by rule.

**Threshold of a feature (`EvalContinuousAtt`).**
* Only thresholds leaving at least `MinSplit` rows on each side are tried, with
  `MinSplit = 0.10 · n / K` clamped to `[m, 25]` (`m` = `--min-objs`, default 2).
  This is what stops C4.5 from cutting off single outliers.
* Two values closer than `1e-5` are treated as equal (no threshold between
  them) — a C4.5 detail that matters for real-valued data.
* The best threshold is the one with the largest **information gain** (not gain
  ratio!). The gain is then reduced by the MDL cost of choosing a threshold:
  `gain − log₂(tries) / n`, where `tries` is the number of thresholds that were
  allowed. A feature is usable only if this is > 0.

**Feature choice (`FormTree` + `Worth`).** Among usable features, only those
whose (corrected) gain is at least the average gain minus 0.001 are candidates;
of those the highest **gain ratio** = gain / split information wins (split
information = entropy of the left/right row counts).

**Stopping.** Pure node, fewer than `2m` rows, or no usable feature. After the
children are built, a subtree that makes as many training errors as one leaf
would is collapsed (`c45CollapseUselessSplits`).

**Thresholds in the final tree (`ContinTest`).** C4.5 replaces the midpoint by
the largest training value (of the whole training set) not above it. This does
not change how training rows are split, only unseen values in the gap.

**Pruning: error-based pruning (`prune.c`).** Every leaf's error count `e` out
of `n` is replaced by the upper limit of a binomial confidence interval at
confidence `CF` (`--cf`, default 0.25; normal deviate interpolated from C4.5's
table, `AddErrs`). Bottom-up, a subtree is replaced by

* a leaf, if the leaf's estimate is no worse (+0.1) than the subtree's and than
  its largest branch's, or
* its **largest branch** ("subtree raising"), if that branch — receiving all of
  the node's rows — is no worse (+0.1) than the subtree. The raised branch is
  then pruned again with all rows.

**Verified against** Quinlan's own `c4.5` binary (`python/compare_reference.py
c45` and `c45prune`): iris, diabetes, and covertype / supersymmetry samples
give identical unpruned and pruned trees for several `--cf` and `--min-objs`
values. The only differences are nodes where two features have exactly the same
gain ratio: C4.5 computes in single precision, so the rounding of mathematically
equal values decides there, while this code takes the first feature (what the
C4.5 source intends with its strict `>`). Feeding C4.5's own unpruned tree to
this code's pruning (`--load-tree`) reproduces C4.5's pruned tree exactly.

---

## Ties and CPU/GPU agreement

Two gains within `1e-12` are treated as equal; the earlier threshold and then
the lower feature index win. Gini gains are computed with exact integer sums,
entropy uses a shared table of `c · log₂ c` for counts below 65536, so the CPU
and GPU produce bit-identical trees (checked by `tests/check_backends.sh`).

## What was wrong before

* Thresholds between two rows of the same class were skipped. That shortcut
  (Fayyad & Irani's boundary points) is only valid between single rows with
  distinct values *and* when the neighbouring thresholds are allowed; next to a
  run of equal values with mixed classes it can skip the best cut. The CPU and
  GPU even skipped different cuts, so they grew different trees.
* "C4.5" chose thresholds by gain ratio, had no MinSplit, no MDL threshold
  cost, no 1e-5 value tolerance, no subtree collapse, and pruning had no subtree
  raising; it was not C4.5.
* "CART" rejected zero-gain splits, measured `α` in raw sample counts, and any
  impurity / selection / pruning combination could be mixed.
* The CSV loader always dropped the first column (diabetes lost `Pregnancies`).
