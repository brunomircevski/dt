# ./tree vs Python CART and C4.5 libraries

How fast and how accurate this project's CART and C4.5 are compared with the
Python implementations people actually use. CPU only (`tree_cpu`), single core
and all cores. Generated with `bench/compare_python.py`; raw numbers below.

## Summary

* **CART, same tree, much faster.** scikit-learn's `DecisionTreeClassifier`
  grows the same unpruned Gini tree as `./tree --cart --no-prune` (checked node
  by node with `tools/compare_reference.py`: the only differences are exact
  ties, which the two break differently). On one core `./tree` is **4–6×
  faster on covertype and 14× faster on SUSY**. scikit-learn cannot use more
  than one core for one tree; with 20 threads `./tree` is **14× (covertype) to
  111× (SUSY 1M) faster**.
* **CART with pruning: 14–400× faster, and the gap grows with the data.** scikit-learn cannot prune a tree it has
  already grown. To choose `ccp_alpha` you refit the whole tree once per
  candidate alpha (17 fits here), or K × candidates with cross-validation (161
  fits). `./tree` grows one tree and evaluates the whole weakest-link sequence.
  Even when scikit-learn spreads its fits over all 20 threads, `./tree` is
  **83× faster on covertype and 405× faster on SUSY 1M** for test-sample
  pruning, and **48× faster on covertype** for 10-fold CV (14–18× on the 20k
  sample). Accuracy is the same
  or slightly better with smaller trees, because `./tree` uses Breiman's
  misclassification cost and 1-SE rule.
* **C4.5: no real Python C4.5 exists.** scikit-learn has no C4.5. Its closest
  setting, `criterion="entropy"`, uses information gain, not gain ratio, and has
  no MinSplit, MDL threshold correction or error-based pruning. It is 4–22×
  slower than `./tree --c45` on one core and less accurate (SUSY: 72.0% vs
  79.3%, because the unpruned entropy tree overfits). The pure-Python C4.5
  packages are **thousands of times slower** (about 4,000× on covertype_20k)
  and much less accurate. ChefBoost tries
  only about 10 thresholds per numeric feature. c4dot5 stops growing after a
  few dozen nodes.
* **Multi-core scaling of `./tree`:** 7–8× on SUSY 1M (continuous features)
  and 2.4–5× on covertype, over 14 physical cores (6 P + 8 E). On
  tiny data (diabetes, 614 rows) there is nothing to parallelise.

## Headline numbers

Training time. "1 thr" is one core, "20 thr" is all cores. Speed-up = Python
time / `./tree` time with the same number of threads. scikit-learn has no
multi-core single tree, so "20 thr" compares against its one-core time.

| Dataset | Task | ./tree 1 thr | ./tree 20 thr | Python 1 thr | Python 20 thr | Speed-up 1 thr | Speed-up 20 thr |
|---|---|---:|---:|---:|---:|---:|---:|
| covertype | CART full tree (sklearn) | 0.88 s | 0.37 s | 5.05 s | — | 5.7× | 13.8× |
| covertype | CART + test-sample pruning (sklearn) | 0.66 s | 0.14 s | 68.8 s | 11.3 s | 104× | 83× |
| covertype | CART + 10-fold CV (sklearn GridSearchCV) | 8.48 s | 2.25 s | (not run, ~15 min) | 108 s | — | 48× |
| covertype | CART depth 12 (sklearn) | 0.59 s | 0.14 s | 3.50 s | — | 5.9× | 25× |
| covertype | C4.5 unpruned vs sklearn entropy | 1.21 s | 0.25 s | 4.69 s | — | 3.9× | 18.5× |
| SUSY 1M | CART full tree (sklearn) | 2.80 s | 0.34 s | 38.0 s | — | 13.6× | 111× |
| SUSY 1M | CART + test-sample pruning (sklearn) | 1.95 s | 0.27 s | 653 s | 110 s | 334× | 405× |
| SUSY 1M | CART depth 12 (sklearn) | 1.39 s | 0.21 s | 21.2 s | — | 15.3× | 103× |
| SUSY 1M | C4.5 unpruned vs sklearn entropy | 1.82 s | 0.26 s | 39.1 s | — | 21.5× | 150× |
| covertype 20k | C4.5 vs ChefBoost C4.5 | 0.056 s | 0.036 s | 237 s | > 300 s | 4,200× | — |
| covertype 20k | C4.5 vs c4dot5 | 0.056 s | — | 229 s | — | 4,100× | — |

## Setup

* **Machine:** laptop, i7-13650HX (14 cores = 6 P + 8 E, 20 threads), 32 GB.
  It throttles under sustained load, so expect ±10–15% noise. Each case runs in
  its own process, one after another, so cases never compete for cores.
* **Data:** one fixed train/test split per dataset, used by every
  implementation. The test set is 20% of the rows and never used for training
  or pruning.
  * `diabetes`: 614 train / 154 test, 8 features, 2 classes (shuffled, seed 1).
  * `covertype_20k`: 20,000 train / 5,000 test, sampled from the covertype split.
  * `covertype`: 464,810 train / 116,202 test, 54 features (44 binary), 7 classes.
  * `SUSY 1M`: first 1,000,000 rows train / last 250,000 rows test, 18
    continuous features, 2 classes. The full 5M SUSY and 11M HIGGS were left
    out to keep the run short. The scikit-learn pruning search alone would
    take hours there. `./tree` numbers for them are in the README.
* **Timing:** training only. For `./tree` that is its `train total` (presort,
  build, cross-validation, pruning). For Python it is the `fit` call, timed
  with `perf_counter`, after an untimed warm-up fit. For the scikit-learn
  pruning searches it is the whole search: path fit, candidate fits and
  scoring. Loading the CSV and predicting the test set are excluded
  everywhere. Best of 3 runs (small data), 2 (covertype, most SUSY cases) or 1
  (SUSY pruning searches, c4dot5, ChefBoost multi-core).
* **Accuracy:** `./tree` writes its tree with `--dump`; the script parses it and
  predicts the test rows. Python models use their own `predict`.
* **Versions:** Python 3.14, scikit-learn 1.9.1, numpy 2.5.3, ChefBoost 0.0.19,
  c4dot5-decision-tree 1.2.1 (`bench/requirements-python.txt`).

### What each implementation runs

| Task | ./tree | Python |
|---|---|---|
| CART full tree | `--cart --no-prune` | `DecisionTreeClassifier(criterion="gini")` |
| CART + test-sample pruning | `--cart` (grow on 2/3, choose alpha on 1/3, 1-SE rule) | Grow on 2/3, take 16 `ccp_alpha` candidates from `cost_complexity_pruning_path` (0 plus 15 giving geometrically spaced tree sizes), refit one tree per candidate, score on 1/3, apply the 1-SE rule. Candidate fits run in parallel with joblib threads. |
| CART + 10-fold CV | `--cart --cv 10` | `GridSearchCV(DecisionTreeClassifier(), {"ccp_alpha": same 16 candidates}, cv=10, n_jobs=…)`: 1 + 160 + 1 fits |
| CART depth 12 | `--cart --no-prune -d 12` | `DecisionTreeClassifier(max_depth=12)` |
| C4.5 | `--c45` (and `--c45 --no-prune`) | sklearn `criterion="entropy", min_samples_leaf=2`; ChefBoost `algorithm="C4.5"` (`max_depth` raised from its default 5 to 1000); c4dot5 `DecisionTreeClassifier(max_depth=10000, node_purity=1.0)` |

The two pruning methods differ in their cost. scikit-learn's `ccp_alpha`
measures a subtree by its leaves' Gini impurity. Breiman's CART, which
`./tree` implements, measures it by misclassification. So the pruned trees are
not expected to be identical, only comparable.

## Results

### CART

| Dataset | Task | Implementation | Threads | Time | Nodes | Test acc. |
|---|---|---|---:|---:|---:|---:|
| diabetes | full tree | ./tree | 1 / 20 | 0.0015 s / 0.0017 s | 227 | 69.48% |
| diabetes | full tree | scikit-learn | 1 | 0.0020 s | 225 | 71.43% |
| diabetes | test-sample pruning | ./tree | 1 / 20 | 0.0014 s / 0.0017 s | 3 | 73.38% |
| diabetes | test-sample pruning | scikit-learn | 1 / 20 | 0.022 s / 0.027 s | 3 | 74.03% |
| diabetes | 10-fold CV | ./tree | 1 / 20 | 0.0053 s / 0.0059 s | 15 | 77.27% |
| diabetes | 10-fold CV | scikit-learn | 1 / 20 | 0.345 s / 0.101 s | 17 | 74.03% |
| diabetes | depth 12 | ./tree | 1 / 20 | 0.0015 s / 0.0015 s | 207 | 70.13% |
| diabetes | depth 12 | scikit-learn | 1 | 0.0020 s | 205 | 70.78% |
| covertype_20k | full tree | ./tree | 1 / 20 | 0.042 s / 0.017 s | 6,005 | 76.76% |
| covertype_20k | full tree | scikit-learn | 1 | 0.161 s | 6,001 | 76.90% |
| covertype_20k | test-sample pruning | ./tree | 1 / 20 | 0.032 s / 0.017 s | 417 | 74.52% |
| covertype_20k | test-sample pruning | scikit-learn | 1 / 20 | 1.72 s / 0.30 s | 947 | 75.52% |
| covertype_20k | 10-fold CV | ./tree | 1 / 20 | 0.40 s / 0.17 s | 1,037 | 77.42% |
| covertype_20k | 10-fold CV | scikit-learn | 1 / 20 | 23.4 s / 2.37 s | 1,301 | 77.44% |
| covertype_20k | depth 12 | ./tree | 1 / 20 | 0.030 s / 0.013 s | 1,789 | 76.42% |
| covertype_20k | depth 12 | scikit-learn | 1 | 0.119 s | 1,789 | 76.42% |
| covertype | full tree | ./tree | 1 / 20 | 0.88 s / 0.37 s | 48,095 | 93.88% |
| covertype | full tree | scikit-learn | 1 | 5.05 s | 48,085 | 93.89% |
| covertype | test-sample pruning | ./tree | 1 / 20 | 0.66 s / 0.14 s | 28,355 | 92.80% |
| covertype | test-sample pruning | scikit-learn | 1 / 20 | 68.8 s / 11.3 s | 38,179 | 92.58% |
| covertype | 10-fold CV | ./tree | 1 / 20 | 8.48 s / 2.25 s | 24,825 | 93.65% |
| covertype | 10-fold CV | scikit-learn | 20 | 108 s | 48,085 (alpha 0) | 93.89% |
| covertype | depth 12 | ./tree | 1 / 20 | 0.59 s / 0.14 s | 3,715 | 80.80% |
| covertype | depth 12 | scikit-learn | 1 | 3.50 s | 3,711 | 80.80% |
| SUSY 1M | full tree | ./tree | 1 / 20 | 2.80 s / 0.34 s | 266,607 | 71.63% |
| SUSY 1M | full tree | scikit-learn | 1 | 38.0 s | 266,545 | 71.68% |
| SUSY 1M | test-sample pruning | ./tree | 1 / 20 | 1.95 s / 0.27 s | 677 | 79.43% |
| SUSY 1M | test-sample pruning | scikit-learn | 1 / 20 | 653 s / 110 s | 1,153 | 79.31% |
| SUSY 1M | 10-fold CV | ./tree | 1 / 20 | 27.3 s / 3.27 s | 677 | 79.45% |
| SUSY 1M | 10-fold CV | scikit-learn | | not run (161 fits × ~38 s ≈ 1.7 h on one core) | | |
| SUSY 1M | depth 12 | ./tree | 1 / 20 | 1.39 s / 0.21 s | 5,827 | 79.11% |
| SUSY 1M | depth 12 | scikit-learn | 1 | 21.2 s | 5,829 | 79.12% |

Notes:

* **Full and depth-limited trees** are the same algorithm. Node counts differ
  by a few nodes, and accuracy by up to 0.2 points. Both come from exact ties:
  scikit-learn visits features in random order, while `./tree` takes the lowest
  feature index. `tools/compare_reference.py cart` confirms every differing
  node on diabetes and covertype_20k is a tie.
* **Pruned trees:** `./tree` picks smaller trees (covertype 28k vs 38k nodes,
  SUSY 677 vs 1,153) with equal or better accuracy. On full covertype,
  `GridSearchCV` chose alpha 0, the unpruned tree: it maximises mean CV
  accuracy and has no 1-SE rule. That gives 0.24 points more test accuracy
  than `./tree --cv 10` with twice the nodes.
* **Why the gap is largest on SUSY:** scikit-learn sorts the rows of every
  node again for every feature. `./tree` sorts each column once (presort) and
  keeps it sorted while splitting. With 18 continuous features and 1M rows
  that re-sorting dominates. On covertype, 44 of 54 features are binary and
  cheap to sort.
* **scikit-learn and multiple cores:** a single `DecisionTreeClassifier.fit`
  always runs on one core (`n_jobs` exists only for ensembles and model
  selection). The only parallelism available is over the pruning candidates
  or CV folds. Even that scaled only 6–10× on 20 threads here: memory
  bandwidth limits it, and the candidate fits run at different speeds.

### C4.5

| Dataset | Implementation | Threads | Time | Nodes | Test acc. |
|---|---|---:|---:|---:|---:|
| diabetes | ./tree `--c45` | 1 / 20 | 0.0014 s / 0.0014 s | 63 | 77.92% |
| diabetes | ./tree `--c45 --no-prune` | 1 / 20 | 0.0013 s / 0.0014 s | 63 | 77.92% |
| diabetes | scikit-learn entropy | 1 | 0.0021 s | 191 | 72.08% |
| diabetes | ChefBoost C4.5 | 1 / 20 | 1.23 s / 3.03 s | 83 / 41 | 75.97% |
| diabetes | c4dot5 | 1 | 15.3 s | 15 | 72.08% |
| covertype_20k | ./tree `--c45` | 1 / 20 | 0.068 s / 0.045 s | 2,775 | 79.26% |
| covertype_20k | ./tree `--c45 --no-prune` | 1 / 20 | 0.056 s / 0.036 s | 3,459 | 78.94% |
| covertype_20k | scikit-learn entropy | 1 | 0.168 s | 5,047 | 78.26% |
| covertype_20k | ChefBoost C4.5 | 1 | 237 s | 9,753 | 66.64% |
| covertype_20k | ChefBoost C4.5 | 20 | > 300 s per run (3 runs did not finish in 900 s) | | |
| covertype_20k | c4dot5 | 1 | 229 s | 55 | 54.02% |
| covertype | ./tree `--c45` | 1 / 20 | 1.41 s / 0.35 s | 26,377 | 94.46% |
| covertype | ./tree `--c45 --no-prune` | 1 / 20 | 1.21 s / 0.25 s | 30,321 | 94.45% |
| covertype | scikit-learn entropy | 1 | 4.69 s | 40,123 | 94.21% |
| SUSY 1M | ./tree `--c45` | 1 / 20 | 2.02 s / 0.29 s | 5,757 | 79.32% |
| SUSY 1M | ./tree `--c45 --no-prune` | 1 / 20 | 1.82 s / 0.26 s | 8,217 | 79.20% |
| SUSY 1M | scikit-learn entropy | 1 | 39.1 s | 227,613 | 72.01% |

Notes:

* **scikit-learn `entropy` is not C4.5.** It grows an ID3/CART hybrid: binary
  splits by information gain. It has no gain ratio, no MinSplit, no MDL
  correction of the threshold gain, and no error-based pruning. On SUSY that
  means a 228k-node tree at 72.0%, against `./tree --c45`'s 5.8k nodes at
  79.3%. `./tree --c45 --no-prune` is still small, because C4.5's own
  stopping rules (MinSplit, the MDL penalty, collapsing useless splits) apply
  without pruning.
* **ChefBoost** (pure Python and pandas) is not C4.5 on numeric data either.
  For a feature with more than 20 distinct values, it tries only the min, max,
  mean and mean ± 1, 2, 3 standard deviations as thresholds. That is why it
  loses 13 points on covertype_20k. Its default `max_depth` is 5. Its
  parallel mode spawns processes per branch and was slower than one core on
  both datasets.
* **c4dot5** (pure Python and pandas) stops after very few nodes on this data
  (15 on diabetes, 55 on covertype_20k) and takes minutes for 20k rows.
* **Not tested:** Weka's J48 (Java C4.5, usable from Python through
  `python-weka-wrapper3`) needs a Java runtime, which this machine does not
  have. Quinlan's original `c4.5` binary is the correctness reference
  (`tools/compare_reference.py c45`, see `ALGORITHMS.md`), but it is not a
  Python library.

### Multi-core scaling of ./tree (1 thread → 20 threads)

| Dataset | CART full | CART test-sample | CART 10-fold CV | C4.5 |
|---|---:|---:|---:|---:|
| covertype_20k | 2.5× | 1.9× | 2.4× | 1.5× |
| covertype | 2.4× | 4.9× | 3.8× | 4.0× |
| SUSY 1M | 8.2× | 7.2× | 8.4× | 7.1× |

## About the earlier Python code

* `tools/compare_reference.py` is sound for what it does: a node-by-node
  correctness check against scikit-learn and Quinlan's `c4.5`. It recomputes
  the gain wherever the trees disagree, so it tells ties apart from real
  differences. It is not a benchmark, and its CSV loader is pure Python, so
  it is slow on big files.
* The removed `python/benchmark_comparison.py` (in git history) ran the
  implementations at the same time in a process pool. They competed for cores,
  so its timings were not reliable. It also tested only depth-5 trees on a
  covertype subset, and c4dot5 with its default `max_depth=10`.
  `bench/compare_python.py` replaces it.

## Reproduce

```bash
python3 -m venv .bench-venv && .bench-venv/bin/pip install -r bench/requirements-python.txt
make cpu
.bench-venv/bin/python bench/compare_python.py prep               # splits into bench/work/
.bench-venv/bin/python bench/compare_python.py run diabetes covertype_20k --repeat 3
.bench-venv/bin/python bench/compare_python.py run covertype susy_1m --repeat 2 \
    --skip 'CV pruning \| scikit-learn \| 1 thr'
.bench-venv/bin/python bench/compare_python.py table              # Markdown table of all results
```

`--only REGEX` / `--skip REGEX` select cases by `dataset | task |
implementation | N thr`. `--timeout` limits each case (default 1800 s).
Results are appended to `bench/work/results.jsonl`.
