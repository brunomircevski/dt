# CPU vs others: ./tree's backends against scikit-learn and Weka J48, depth 10, no pruning

**Status:** ./tree's --serial and --parallel bars are the test-set benchmarks' runs on ec77727 (main with the CPU presort of 4fb419c), 2026-10-29/30: [cluster-higgs](../cluster-higgs/BENCHMARK.md), [cluster-susy](../cluster-susy/BENCHMARK.md), [cluster-ecbdl](../cluster-ecbdl/BENCHMARK.md), [cluster-chess](../cluster-chess/BENCHMARK.md); Covertype's are the `*-epyc-covertype` folders here. --simple: the same folders (2026-10-27/29, its code has not changed). scikit-learn and Weka J48 (`vs-epyc-<data>-<tool>-<protocol>`): rerun on 2026-10-29/30 with 3 runs where a run takes under 10 minutes; the longer ones (J48 on HIGGS and ECBDL, scikit-learn on ECBDL and chess 1000M) are the single runs of 2026-10-27/28. All on epyc1 and epyc2, which have the same hardware (2× EPYC 9255).

Every tree is grown to depth ≤ 10 without pruning, and its accuracy is measured on the dataset's test set. Every job finished but one: J48 on chess 1000M ran out of Java heap (251 GB, half the node's RAM) after 2 h 36 min.

The previous version of this benchmark (every tree pruned, no depth limit, training accuracy, 2026-10-21/22) was `bench/results/cluster-cpu-vs-others-old`, removed since; it is in the git history, commit 8ce4940.

![chart](chart.png)

## What runs

| Bar | Command | Threads | CPUs and memory |
|---|---|--:|---|
| ./tree --simple | `tree_cpu_main --simple --cart --no-prune -d 10` / `--simple --c45 --no-prune -d 10` | 1 | one socket |
| ./tree --serial | `tree_cpu_main --serial ...` | 1 | one socket |
| ./tree --parallel, one socket | `tree_cpu_main --parallel --threads 24 ...` | 24 | one socket |
| ./tree --parallel, both sockets | `tree_cpu_main --parallel --threads 48 ...` | 48 | the whole node |
| scikit-learn 1.9.1 | `DecisionTreeClassifier(max_depth=10)`, gini, CART only | 1 | one socket |
| Weka J48 3.8.7 | `-U -M 2` with the adapter's depth limit 10, C4.5 only | 1 | one socket |

- **Protocols:** `cart_depth10` and `c45_depth10_unpruned` from [benchlib.py](../../benchlib.py), the same ones as the test-set benchmarks (cluster-higgs, cluster-susy).
- **scikit-learn has no C4.5**, so it is in the CART comparison only. Its entropy tree (information gain) is a different algorithm from C4.5's gain ratio; the jobs also ran it (`criterion="entropy", min_samples_leaf=2`, in the run dirs' results.jsonl), but it is not in the chart or the tables.
- **J48 has no depth limit.** [J48Fit](../../adapters/J48Fit.java) adds one: `DepthLimitedTree` grows J48's own `C45PruneableClassifierTree`, and at depth 10 it makes a leaf (`NoSplit`) instead of choosing a split. `-U` turns off J48's pruning and collapsing. Weka has no CART in its core (SimpleCart is a separate package that prunes by cross-validation), so CART has no Weka bar.
- **Test sets:** HIGGS and SUSY use UCI's official test rows (the last 500,000). chess 10M and 1000M use their own test files (4,500 and 1,000 rows), Covertype its prepared test split (116,202 rows), and ECBDL'14 the competition's labelled test set (2,897,917 rows). Every tool predicts the same test rows (`run.py --test-accuracy`).
- **One socket:** every job owned its node (`--exclusive`).
  - Every run except the dual-socket bar was pinned to one socket (24 CPUs), with its memory bound to that socket's NUMA node (job.sh `SOCKET=auto`). The one-thread runs use that socket's first CPU.
  - The two-socket runs use all 48 CPUs and both NUMA nodes.
- **Time:** the training time, which does not include reading the data. Python and Java first get one untimed warm-up fit on 50,000 rows.
- **Memory:** the peak resident memory (RSS) right after training: loading, warm-up and training.

## Same trees

`run.py --hash-trees` records the SHA-256 of every ./tree tree (`--dump`). For each dataset and protocol, every ./tree backend (--simple, --serial, and --parallel on one socket and on both) grew the same tree, on the new runs too:

| Dataset | Protocol | Tree SHA-256 (first 16) | Nodes | Depth | Test accuracy |
|---|---|---|--:|--:|--:|
| chess_10m | CART | a537d8753d217de7 | 17 | 4 | 99.98% |
| chess_10m | C4.5 | a7a65727f69d848a | 17 | 4 | 100.00% |
| covertype | CART | 474aeff3e34e323d | 1,499 | 10 | 77.66% |
| covertype | C4.5 | 7d7954ba222d8f5b | 723 | 10 | 59.93% |
| susy | CART | b4b927530b8265bb | 1,959 | 10 | 79.01% |
| susy | C4.5 | c4948bc23841a3fc | 743 | 10 | 78.65% |
| higgs | CART | 4c01bec85a4914cd | 2,029 | 10 | 70.52% |
| higgs | C4.5 | 0978d478e2f50bb7 | 935 | 10 | 63.73% |
| ecbdl_15m | CART | f45a02bf62a6ebb3 | 1,863 | 10 | 98.31% |
| ecbdl_15m | C4.5 | be533fee73ea3a4a | 725 | 10 | 98.32% |
| chess_1b | CART | 56bb1724556db415 | 17 | 4 | 100.00% |
| chess_1b | C4.5 | 927b90dc84a0b268 | 17 | 4 | 100.00% |

The other tools:

| Dataset | scikit-learn (CART) | Weka J48 (C4.5) |
|---|--:|--:|
| chess_10m | 17 nodes, 99.98% | 17 nodes, 100.00% |
| covertype | 1,499 nodes, 77.66% | 723 nodes, 59.93% |
| susy | 1,959 nodes, 79.01% | 741 nodes, 78.64% |
| higgs | 2,029 nodes, 70.52% | 923 nodes, 63.75% |
| ecbdl_15m | 1,863 nodes, 98.24% | 711 nodes, 98.32% |
| chess_1b | 17 nodes, 100.00% | out of memory |

- **CART:** scikit-learn's gini tree has the same node count and the same test accuracy as ./tree's on every dataset except ECBDL'14: the same 1,863 nodes, but 98.24% against 98.31%, so some splits differ (likely ties between features or thresholds; not checked).
- **C4.5:** Weka J48 grows the same tree size and accuracy as ./tree on chess and Covertype, and nearly the same on SUSY (741 against 743 nodes), HIGGS (923 against 935) and ECBDL'14 (711 against 725). ./tree follows C4.5 Release 8; J48 differs from R8 in small details.
- **C4.5 is less accurate here:** on Covertype 59.9% against CART's 77.7%, and on HIGGS 63.7% against 70.5%. Both C4.5 implementations agree, so this comes from the algorithm, not from ./tree. scikit-learn's entropy tree (plain information gain, not charted) is as accurate as CART (77.2% on Covertype, 70.5% on HIGGS), so the cause is gain ratio. A likely explanation (not checked node by node): gain ratio favours splits that cut off a small group of rows, whose split information is low. At a depth limit of 10 those splits use up the levels, and C4.5's trees have half as many nodes as CART's. Without a depth limit, C4.5 grows on to depth 38–71 (the previous version, in the git history).

## Results (training time, mean of the runs; peak memory; number of runs)

| Dataset | Protocol | --simple | --serial | --parallel, 24 (one socket) | --parallel, 48 (both) | scikit-learn | Weka J48 |
|---|---|--:|--:|--:|--:|--:|--:|
| chess_10m | CART | 6.14 s, 261 MiB (30) | 334 ms, 490 MiB (30) | 68.9 ms, 491 MiB (30) | 91.1 ms, 493 MiB (30) | 6.24 s, 586 MiB (3) | – |
| chess_10m | C4.5 | 7.3 s, 260 MiB (30) | 382 ms, 491 MiB (30) | 76.7 ms, 492 MiB (30) | 103 ms, 494 MiB (30) | – | 27.1 s, 3.78 GiB (3) |
| covertype | CART | 3.98 s, 112 MiB (10) | 567 ms, 307 MiB (10) | 48.6 ms, 392 MiB (10) | 61.2 ms, 481 MiB (10) | 2.31 s, 233 MiB (3) | – |
| covertype | C4.5 | 4.31 s, 126 MiB (10) | 397 ms, 307 MiB (10) | 36.6 ms, 391 MiB (10) | 40.6 ms, 478 MiB (10) | – | 36.4 s, 1.14 GiB (3) |
| susy | CART | 70.8 s, 436 MiB (3) | 2.84 s, 1.03 GiB (10) | 281 ms, 1.57 GiB (10) | 239 ms, 1.58 GiB (10) | 86.5 s, 608 MiB (3) | – |
| susy | C4.5 | 77.2 s, 543 MiB (3) | 3.1 s, 1.12 GiB (10) | 277 ms, 1.66 GiB (10) | 249 ms, 1.66 GiB (10) | – | 230 s, 4.81 GiB (3) |
| higgs | CART | 236 s, 1.32 GiB (3) | 10.4 s, 3.57 GiB (10) | 829 ms, 5.37 GiB (10) | 607 ms, 5.61 GiB (10) | 215 s, 1.61 GiB (3) | – |
| higgs | C4.5 | 257 s, 1.43 GiB (3) | 11.9 s, 3.6 GiB (10) | 925 ms, 5.4 GiB (10) | 590 ms, 5.64 GiB (10) | – | 830 s, 12.3 GiB (1) |
| ecbdl_15m | CART | 2,273 s, 35.5 GiB (1) | 225 s, 106 GiB (3) | 11.8 s, 109 GiB (10) | 6.32 s, 111 GiB (10) | 2,198 s, 35.9 GiB (1) | – |
| ecbdl_15m | C4.5 | 2,587 s, 35.6 GiB (1) | 277 s, 106 GiB (3) | 14.4 s, 109 GiB (10) | 7.81 s, 111 GiB (10) | – | 24,151 s, 129 GiB (1) |
| chess_1b | CART | 912 s, 21.3 GiB (1) | 98.6 s, 46.7 GiB (3) | 5.53 s, 46.7 GiB (30) | 4.44 s, 46.7 GiB (30) | 642 s, 45.8 GiB (1) | – |
| chess_1b | C4.5 | 1,031 s, 21.4 GiB (1) | 101 s, 46.7 GiB (3) | 6.22 s, 46.7 GiB (30) | 5.45 s, 46.7 GiB (30) | – | out of memory (heap 251 GB, after 2 h 36 min) |

- **--serial against --simple:** --serial is faster by:
  - 18–19× on chess 10M and 7–11× on Covertype;
  - 25× on SUSY and 22–23× on HIGGS;
  - only 9–10× on ECBDL'14 15M and chess 1000M (not investigated).

  --simple sorts each node's rows again for every feature; --serial sorts once and keeps the order while it partitions ([paper/simple-vs-serial.md](../../../paper/simple-vs-serial.md)).
- **--parallel on one socket against --serial:** 5× faster on chess 10M, which has only two features to share among 24 threads. 11–12× faster on Covertype, 10–11× on SUSY, 12–13× on HIGGS, 19× on ECBDL'14 15M and 16–18× on chess 1000M.
- **Both sockets against one:** the second socket helps on the big datasets: 1.8–1.9× faster on ECBDL'14 15M, 1.4–1.6× on HIGGS, 1.1–1.2× on chess 1000M and on SUSY. It is 1.1–1.3× slower on Covertype and on chess 10M, where the threads on the other socket mostly cost cross-socket memory traffic and synchronisation.
- **--simple against the libraries:**
  - CART: as fast as scikit-learn on chess (6.14 s against 6.24 s), 1.7× slower on Covertype, 1.2× faster on SUSY, 1.1× slower on HIGGS, as fast on ECBDL'14 (2,273 s against 2,198 s) and 1.4× slower on chess 1000M. Without pruning, the two grow at about the same speed: the old benchmark's 6–13× lead on SUSY and HIGGS came from scikit-learn's quadratic pruning (the previous version, in the git history).
  - 3.0–3.7× faster than J48 on chess, SUSY and HIGGS, 8.4× faster on Covertype and 9.3× on ECBDL'14 (2,587 s against 24,151 s, 6.7 hours).
- **--serial against the libraries:** CART 19× faster than scikit-learn on chess 10M, 4× on Covertype, 30× on SUSY, 21× on HIGGS, 10× on ECBDL'14 and 6.5× on chess 1000M; C4.5 70–92× faster than J48 (87× on ECBDL'14).
- **--parallel on one socket against the libraries:** CART 91× faster than scikit-learn on chess 10M, 48× on Covertype, 308× on SUSY, 259× on HIGGS, 186× on ECBDL'14 and 116× on chess 1000M; C4.5 350–1,680× faster than J48 (1,680× on ECBDL'14).
- **Memory:**
  - --simple uses the least of the ./tree backends, as it keeps no presorted orders: 0.1–1.4 GiB, 21–36 GiB on the two big datasets (a third to a half of --serial's).
  - --serial uses 2–3× as much (sorted feature orders), and --parallel up to 1.5× more than --serial (per-thread buffers; about the same on chess).
  - scikit-learn uses 0.2–1.6 GiB.
  - J48 uses the most: 1.1–12.3 GiB, 129 GiB on ECBDL'14. Weka keeps every row as a Java object of doubles: chess 1000M (1 billion row objects) did not fit a 251 GB heap. With a heap of about 450 GB on both sockets it might; that was not tried.

### Why J48 on HIGGS took 19 hours before and 14 minutes now

In the old benchmark J48 grew HIGGS without a depth limit: 1,152,747 nodes, depth 71, 68,662 s of growing. Here it stops at depth 10: 923 nodes, 830 s. The rows per level are the same (all 10.5M at every level down to the limit), so the extra time came from the deep levels' half a million internal nodes. After choosing a numeric split, J48's `C45ModelSelection` calls `setSplitPoint` on the whole training set, to move the threshold to a value that occurs in the data. That is one scan of all 10.5M rows per internal node: 576k scans without a depth limit, 461 at depth 10. This is the likely cause; it was not profiled.

## Reproduce

```bash
# on mordor3: the libraries and Covertype's ./tree runs, with every other rerun of 2026-10-29/30
bench/cluster/submit-main-2026-10-29.sh
bench/.venv/bin/python bench/simple_chart.py bench/results/cluster-cpu-vs-others
```

- Each `vs-epyc-<data>-<tool>-<protocol>` job runs one tool on one socket (`SOCKET=auto`).
- `simple_chart.py` refuses to draw if the ./tree backends' tree hashes differ.
