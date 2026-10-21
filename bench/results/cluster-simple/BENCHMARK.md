# Simple: ./tree's reference backend against --serial, --parallel, scikit-learn and Weka J48

**Status:** all runs are on one socket of epyc1.
- **chess 10M and Covertype:** run on 2026-10-21 at commit e418f65, three runs per case.
- **SUSY:** run on 2026-10-21 at commit 85ea702, one run per case, one SLURM job per tool and protocol (`simple-epyc-susy-<tool>-<protocol>`). Weka J48 (job 45) is still running.
- **HIGGS:** queued the same way (jobs 46–54). Its panel completes the 2×2 chart once it is done.

This is a quick general benchmark, so one run per case is enough for the big datasets. Every tree is pruned, with no depth limit. The first version of this benchmark (chess 1M, epyc1 and dell1, scikit-learn unpruned) is in the git history (d4d4c1d).

![chart](chart.png)

## What runs

| Bar | Command | Threads |
|---|---|--:|
| ./tree --simple | `tree_cpu --simple --cart --alpha 1e-05` / `--simple --c45` | 1 |
| ./tree --serial | `tree_cpu --serial ...` | 1 |
| ./tree --parallel | `tree_cpu --parallel --threads 24 ...` | 24 |
| scikit-learn 1.9.1 | `DecisionTreeClassifier(ccp_alpha=1e-5)`: gini for CART, `criterion="entropy"` for C4.5 | 1 |
| Weka J48 3.8.7 | `-C 0.25 -M 2` (its defaults), C4.5 only | 1 |

- **Pruning:** every tool prunes.
  - ./tree prunes CART by cost-complexity at α = 1e-5, and C4.5 the way C4.5 does: collapse, CF 0.25 and subtree raising.
  - scikit-learn has only cost-complexity pruning (`ccp_alpha`), so it prunes both of its trees at the same α. It has no C4.5; its entropy tree is the nearest stand-in.
  - Weka has no CART in its core. SimpleCart is a separate package that prunes by cross-validation, so CART has no Weka bar.
- **α is not the same measure:** ./tree's cost-complexity measure is Breiman's, where R(T) is the misclassified fraction of the training rows ([cart_pruning.h](../../../src/algo/cart_pruning.h)). scikit-learn's R(T) is the impurity weighted by row share (gini or entropy). The same α = 1e-5 therefore prunes scikit-learn's trees less, so its trees are bigger (Covertype CART: 14,737 nodes against 9,851).
- **One socket:** the job owned epyc1 (`--exclusive`). Every run was pinned to socket 0 (CPUs 0–23), with the memory bound to NUMA node 0 (job.sh `SOCKET=auto`). The one-thread runs use CPU 0.
- **epyc1 and epyc2:** HIGGS's J48 job ran on epyc2, which has the same hardware as epyc1 (2× EPYC 9255, governor performance), to finish sooner. First, five short SUSY cases (one run each, `submit.sh -w epyc2`) were rerun on epyc2 to check this. They grew the same trees (same SHA-256), with the same times within 1%:

  | SUSY, one run | epyc1 | epyc2 |
  |---|--:|--:|
  | CART --simple | 125.4 s | 125.6 s |
  | CART --serial | 5.245 s | 5.245 s |
  | C4.5 --serial | 6.379 s | 6.437 s |
  | CART --parallel | 0.513 s | 0.515 s |
  | C4.5 --parallel | 0.572 s | 0.590 s |
- **Time:** the training time (not reading the data): the median of 3 runs (chess, Covertype) or a single run (SUSY, HIGGS). Each bar is split into growing (dark) and pruning (light):
  - ./tree: its `prune` phase is the pruning part. For C4.5 this also includes --simple's sort of each feature's distinct values; --serial gets those from its presort, timed as growing ([paper/simple-vs-serial.md](../../../paper/simple-vs-serial.md), risk F).
  - scikit-learn: [sklearn_fit.py](../../adapters/sklearn_fit.py) fits with `ccp_alpha=0` (growing), then calls `_prune_tree()` at α (pruning), which are the two steps `fit` takes.
  - Weka J48: [J48Fit](../../adapters/J48Fit.java) times `buildTree` (growing) apart from `collapse` + `prune` + `cleanup` (pruning). Its `check=1` mode confirmed that this grows the same tree as `J48.buildClassifier`.
  - The managed runtimes get one untimed warm-up fit on 50,000 rows first.
- **Single-threaded pruning:** --simple and --serial get no thread pool (`buildPool = nullptr` in trainer.cpp), so they prune on one thread too.

## Same trees

`run.py --hash-trees` records the SHA-256 of every ./tree tree (`--dump`). For each dataset and protocol, every ./tree run (3 backends × 3 runs, or × 1 on SUSY) produced the same tree:

| Dataset | Protocol | Tree SHA-256 (first 16) | Nodes | Depth | Train accuracy |
|---|---|---|--:|--:|--:|
| chess_10m | CART | a537d8753d217de7 | 17 | 4 | 100.00% |
| chess_10m | C4.5 | a7a65727f69d848a | 17 | 4 | 100.00% |
| covertype | CART | d18c31438ee3de2b | 9,851 | 35 | 94.18% |
| covertype | C4.5 | bf01ecc0f4651b88 | 26,377 | 59 | 98.66% |
| susy | CART | 94d1983c618f53cd | 723 | 19 | 79.68% |
| susy | C4.5 | 8afe20ed2579def1 | 15,113 | 38 | 80.67% |

The other tools on Covertype:
- **Weka J48:** 26,219 nodes against ./tree's 26,377, both of depth 59, with 98.65% training accuracy. ./tree follows C4.5 Release 8, which was verified node for node earlier (2026-10-13), and J48 differs from R8 in small details.
- **scikit-learn:** 14,737 nodes (gini) and 31,839 nodes (entropy, depth 36). It prunes less, as explained above.

On chess every tool grows the exact 17-node tree. On SUSY scikit-learn's trees are smaller than ./tree's at the same α (1,059 and 2,067 nodes), unlike on Covertype, so the two measures do not differ in one direction.

## Results (median training time; pruning part in brackets)

| Dataset | Protocol | --simple | --serial | --parallel (24) | scikit-learn | Weka J48 |
|---|---|--:|--:|--:|--:|--:|
| chess_10m | CART | 6.10 s | 340 ms | 80.9 ms | 6.23 s (0.2 ms) | – |
| chess_10m | C4.5 | 7.56 s (1.40 s) | 495 ms (93 ms) | 108 ms (27 ms) | 6.25 s (0.2 ms) | 36.5 s (10.1 s) |
| covertype | CART | 7.60 s (0.7 ms) | 858 ms (0.2 ms) | 82.4 ms (0.4 ms) | 4.74 s (513 ms) | – |
| covertype | C4.5 | 12.6 s (438 ms) | 1.13 s (78 ms) | 124 ms (17 ms) | 4.11 s (271 ms) | 141 s (16.3 s) |
| susy | CART | 125 s (0.4 ms) | 5.24 s (0.6 ms) | 513 ms (0.7 ms) | 853 s (666 s) | – |
| susy | C4.5 | 148 s (6.52 s) | 6.38 s (381 ms) | 572 ms (60 ms) | 847 s (648 s) | running |

- **--simple against --serial:** --serial is 15–18× faster on chess 10M, 9–11× faster on Covertype and 23–24× faster on SUSY. --simple sorts each node's rows again for every feature; --serial sorts once and keeps the order while it partitions (paper/simple-vs-serial.md lists all 30 differences).
- **--parallel on one socket:** 4.2–4.6× faster than --serial on chess, which has only two features to share among 24 threads, 9–10× faster on Covertype and 10–11× faster on SUSY.
- **--simple against the libraries:**
  - About as fast as scikit-learn on chess, and 1.6–3.1× slower on Covertype.
  - 5.7–6.8× faster than scikit-learn on SUSY. Most of scikit-learn's time there is pruning (650–670 s): `_prune_tree` works on the fully grown tree, which on SUSY is huge. Even growing alone, scikit-learn (187–198 s) is slower than --simple (125–141 s).
  - 4.8× faster than J48 on chess and 11× faster on Covertype.
  - --serial is 13–18× faster than scikit-learn on chess and 3.6–5.5× faster on Covertype. It is 74–125× faster than J48.
- **Pruning:**
  - CART at a fixed α is under 1 ms in ./tree, 0.2–513 ms in scikit-learn.
  - C4.5's pruning takes --simple 1.40 s on chess and 438 ms on Covertype, against --serial's 93 and 78 ms. Part of that is the distinct-value sort counted there.
  - J48's C4.5 pruning takes 10–16 s.
  - scikit-learn's cost-complexity pruning grows with the unpruned tree: 0.2 ms on chess, 0.3–0.5 s on Covertype, 11 minutes on SUSY.

## Reproduce

```bash
# on mordor3, after build.sh and prepare.sh covertype (bench/cluster/README.md)
for d in chess_10m covertype; do
  bench/cluster/submit.sh -m 3 -d $d -o cluster-simple simple-epyc-$d
done
# SUSY and HIGGS: one run, one job per tool and protocol (J48 has no CART)
for d in susy higgs; do
  bench/cluster/submit.sh -m 1 -d $d -o cluster-simple $(for t in parallel serial sklearn simple; do
    echo simple-epyc-$d-$t-cart simple-epyc-$d-$t-c45; done) simple-epyc-$d-j48-c45
done
bench/.venv/bin/python bench/simple_chart.py bench/results/cluster-simple
```

`simple-epyc-<data>` runs use impls `tree,tree_simple,sklearn,j48` with threads `1,all`; `simple-epyc-<data>-<tool>-<protocol>` runs one of them, with a time limit that fits the tool. All of them and pass `SOCKET=auto` and `RUN_EXTRA=--hash-trees` to job.sh. `simple_chart.py` refuses to draw if the ./tree backends' tree hashes differ.
