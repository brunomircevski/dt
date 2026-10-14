# CART and C4.5 as implemented here

Both algorithms grow a binary tree on numeric features top-down. At every node
they look at every feature, sort the node's rows by that feature, try every
threshold between two neighbouring values, and keep the best one. They differ in
*how* "best" is measured, in *when* to stop, and in how the grown tree is
*pruned*. All of that lives in three small files:

| File | What |
|------|------|
| `src/algo/split_math.h` | Impurity, gain, tie rules. Compiled by both g++ and nvcc, so the CPU and the GPU score cuts with literally the same code. |
| `src/algo/split_rules.cpp` | Stopping rules, minimum child size, how the winning feature is chosen (CART vs C4.5). |
| `src/algo/cart_pruning.cpp` | CART cost-complexity pruning and its weakest-link sequence. |
| `src/algo/c45_pruning.cpp` | C4.5 post-processing: collapse, training-value thresholds, error-based pruning. |
| `src/app/trainer.cpp` | Puts it together; Breiman's cross-validation for CART's alpha. |

Not supported (on purpose): missing values (C4.5's fractional weights) and
multi-way / subset splits on categorical features.

---

## Notation

A node has `n` rows and class counts `c_1 .. c_K`. A threshold `t` on feature
`f` sends rows with `x_f <= t` left (`n_L` rows, counts `l_k`) and the rest
right (`n_R`, `r_k`).

The code works with **weighted impurities** (impurity times row count), which
need no division per class and make gains exact sums:

* Gini (CART): `G(c) = n - Σ c_k² / n`  (= n · (1 − Σ p_k²))
* Entropy (C4.5): `E(c) = n log₂ n − Σ c_k log₂ c_k`  (= n · H; C4.5 calls it `TotalInfo`)

`gain = (W(parent) − W(left) − W(right)) / n` is then the usual impurity
decrease (Gini) or information gain (entropy).

---

## CART (Breiman, Friedman, Olshen, Stone 1984)

**Split selection.** Every feature's best threshold is the one with the largest
decrease of the Gini index, CART's standard impurity (Breiman et al.
found the choice of impurity matters little; entropy is C4.5's). Across features the
largest decrease wins. Thresholds are midpoints between the two neighbouring
values, computed in `double` (the midpoint of two floats is exact in double),
like scikit-learn.

**Stopping.** A node becomes a leaf when it is pure, is at `--max-depth`, or
no threshold leaves `--min-leaf` rows (default 1) on both sides. A split with
zero gain is still accepted (CART grows the maximal tree and lets pruning
decide; scikit-learn does the same — it matters for XOR-like data).

**Pruning: minimal cost-complexity.** For a subtree `T` let `R(T)` be its
training misclassification rate and `|T|` its number of leaves. For a given
`α`, `T(α)` is the smallest subtree minimising `R(T) + α|T|`
(`cartCostComplexityPrune`, a bottom-up dynamic program).

* By default `α` is fixed (`--alpha X`, default 0.00001), as in scikit-learn
  and rpart: one tree, grown on all rows. `α` is a rate, so it does not
  depend on the dataset size. `--no-prune` keeps the maximal tree. The values
  are not interchangeable with theirs: scikit-learn's `ccp_alpha` weighs the
  leaves' Gini impurity instead of `R(T)`, and rpart's `cp` is `α / R(root)`.
* With a fixed `α` a node that misclassifies fewer than `α · n` training rows
  (`n` = all training rows) is not split: in `T(α)` it is a leaf anyway. As a
  leaf it costs its errors `e < α · n` plus `α · n`; any subtree under it
  costs at least `2 α · n` (two leaves), so pruning would cut it back. The
  pruned tree is the same, but most of the maximal tree is never grown: on
  HIGGS (`-d 30`) 116 thousand nodes instead of 2.2 million, and 0.1 million
  instead of 21.6 million on airline. `--cv` grows the maximal tree, since it
  needs the whole pruning sequence.
* `--cv K` chooses `α` by Breiman's K-fold cross-validation instead: the tree
  is grown on all rows, a tree is grown on each of K folds' complement, and
  each fold tree is measured on its fold along the full tree's weakest-link
  sequence `α_1 = 0 < α_2 < …`, at the geometric midpoints `√(α_k α_{k+1})`.
  The simplest tree within one standard error of the best is kept (1-SE
  rule). The sequence is computed in one bottom-up pass by representing each
  subtree's optimal cost as a concave piecewise-linear function of `α`
  (`cartPruningSequence`). The fold trees are grown by the same backend as
  the main tree, reusing its presorted columns (CPU) or device buffers (GPU).
  Breiman's other estimate, a held-back test sample, is not offered: it grows
  the tree on part of the rows only.

**Verified against** scikit-learn (`tools/compare_reference.py cart`): trees
are identical except where two splits have exactly the same impurity decrease
(scikit-learn visits features in random order, this code takes the lowest
feature index); the script recomputes both gains at every such node to prove it
is a tie. An exhaustive search at every node of the grown tree (largest Gini
decrease over all features and thresholds, lowest feature on ties) agrees on
all 24,048 decision nodes of the full covertype tree, and every subtree of the
pruning sequence matches a direct textbook weakest-link implementation.

---

## C4.5 (Quinlan 1993, Release 8 — the last C4.5)

This follows the original C source (`contin.c`, `build.c`, `info.c`,
`prune.c`, `stats.c`) rule by rule.

**Threshold of a feature (`EvalContinuousAtt`).**
* Only thresholds leaving at least `MinSplit` rows on each side are tried, with
  `MinSplit = 0.10 · n / K` clamped to `[m, 25]` (`m` = `--min-leaf`, default 2).
  This is what stops C4.5 from cutting off single outliers.
* Two values closer than `1e-5` are treated as equal (no threshold between
  them) — a C4.5 detail that matters for real-valued data.
* The best threshold is the one with the largest **information gain** (not gain
  ratio!). The gain is then reduced by the MDL cost of choosing a threshold:
  `gain − log₂(tries) / n`, where `tries` is the number of thresholds that were
  allowed. A feature is usable only if this is > 0 — by more than the `1e-12`
  tie tolerance: a cut that keeps the node's class proportions (24:3 into 16:2
  and 8:1) has a gain of exactly 0, which rounding can make +4e-16, and with a
  single allowed cut the MDL cost `log₂ 1` is 0 as well.

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

**Verified against** Quinlan's own `c4.5` binary (`tools/compare_reference.py
c45`): iris, diabetes, covertype (up to its full 465k-row training set) and
supersymmetry samples, `--min-leaf` 1 to 300 and `--cf` 0.05 to 0.5. The trees
are identical node by node (thresholds, row counts, classes) except where
C4.5's single-precision arithmetic decides:

* Two features, or two thresholds of one feature, with exactly the same gain
  ratio (gain): the rounding of mathematically equal values picks one in C4.5,
  while this code takes the first (what the C4.5 source intends with its
  strict `>`). This is the only kind of difference on the samples.
* On the full covertype tree, 5 more of about 15,000 nodes: C4.5 picks a
  threshold whose gain is up to 4e-7 below the best one (3 nodes), rounds a
  gain of 3e-8 to 0 and so leaves three features out of the average gain, or
  rounds a gain of exactly 0 up to +1e-7 and splits on it. This code computes
  in double precision and treats a gain within `1e-12` of 0 as 0.

Pruning is the same: a direct port of `prune.c` turns C4.5's unpruned tree
into exactly C4.5's pruned tree and this code's unpruned tree into exactly
this code's pruned tree, so pruned trees differ only below the nodes above.
One known difference has never changed a tree in these checks: with
`--min-leaf` above 25, C4.5 counts `tries` (the MDL cost) of a big node from
25 rows on each side, this code from `--min-leaf` rows. `tests/golden/` keeps
the verified trees.

---

## Ties and CPU/GPU agreement

Two gains within `1e-12` are treated as equal; the earlier threshold and then
the lower feature index win; a C4.5 gain within `1e-12` of 0 is 0. Gini gains
are computed with exact integer sums, entropy uses a shared table of
`c · log₂ c` for counts below 65536, so the CPU and GPU produce bit-identical
trees (checked by `tests/check.sh`, which also compares every backend with the
verified trees in `tests/golden/`).

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
* C4.5 accepted a feature whose only allowed cut keeps the class proportions:
  the gain is exactly 0 but was computed as +4e-16 (2 of 15,318 splits of the
  full covertype tree; `tests/data/zero_gain.csv` is one of those nodes).
