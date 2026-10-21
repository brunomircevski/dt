# Simple: ./tree's reference backend against --serial, --parallel, scikit-learn and Weka J48

**Status:** run on 2026-10-21, commit abd4bc7 (`-dirty` only because the build rewrote the tracked `bench/cluster/build.log`), three runs per case, on epyc1 and dell1. Chess 1M and Covertype, CART and C4.5, no depth limit, pruning on.

![chart](chart.png)

## What runs

| Bar | Command | Threads |
|---|---|--:|
| ./tree --simple | `tree_cpu --simple --cart --alpha 1e-05` / `--simple --c45` | 1 |
| ./tree --serial | `tree_cpu --serial ...` | 1 |
| ./tree --parallel | `tree_cpu --parallel --threads 24 ...` | 24 |
| scikit-learn 1.9.1 | `DecisionTreeClassifier()` (defaults: gini), and `criterion="entropy"` for the C4.5 panels | 1 |
| Weka J48 3.8.7 | `-C 0.25 -M 2` (its defaults), C4.5 panels only | 1 |

- **Options:** each tool runs with its default options. ./tree's `--cart` defaults are α = 1e-5 and no depth limit, and its `--c45` defaults are CF 0.25, 2 rows per leaf and subtree raising. scikit-learn prunes only when given `ccp_alpha`, so with defaults it grows the full tree; it has no C4.5, so its entropy tree is only the nearest stand-in. Weka has no CART in its core: SimpleCart is a separate package that prunes by cross-validation, not at a fixed α, so the CART panels have no Weka bar.
- **One socket:** each job owned its whole node (`--exclusive`), and every run was pinned to one socket with the memory bound to its NUMA nodes (job.sh `SOCKET=auto`). That is epyc1 socket 0 (CPUs 0–23, NUMA node 0) and dell1 socket 1 (the odd CPUs, NUMA nodes 2 and 3). The one-thread runs take that socket's first CPU.
- **Time:** the training time (not reading the data), median of 3 runs. Each bar is split into growing (dark) and pruning (light):
  - ./tree: `prune` is its pruning phase, and growing is the rest, presort included.
  - Weka J48: [J48Fit](../../adapters/J48Fit.java) times `buildTree` (growing) apart from `collapse` + `prune` + `cleanup` (pruning), following C45PruneableClassifierTree.buildClassifier step by step. Its `check=1` mode confirmed it grows the same tree as plain `J48.buildClassifier` (covertype).
  - scikit-learn: it does not prune, so it has one part only.
  - The managed runtimes (Python, JVM) get one untimed warm-up fit on 50,000 rows first.
- **Single-threaded pruning:** `--simple` and `--serial` get no thread pool (`buildPool = nullptr` in trainer.cpp), so their pruning (CART cost-complexity, C4.5 collapse, thresholds and pessimistic pruning) runs on one thread as well. Only `--parallel` prunes C4.5 on its pool.

## Same trees

`run.py --hash-trees` dumps every ./tree tree (`--dump`) and records its SHA-256. For each dataset and protocol, all 18 ./tree runs (3 backends × 2 machines × 3 runs) produced the same dump:

| Dataset | Protocol | Tree SHA-256 (first 16) | Nodes | Depth | Train accuracy |
|---|---|---|--:|--:|--:|
| chess_1m | CART | 5090a0938506c19c | 17 | 4 | 100.00% |
| chess_1m | C4.5 | 48f05fc3dd115c13 | 17 | 4 | 100.00% |
| covertype | CART | d18c31438ee3de2b | 9,851 | 35 | 94.18% |
| covertype | C4.5 | bf01ecc0f4651b88 | 26,377 | 59 | 98.66% |

How the other tools' trees compare:
- **Weka J48:** the exact chess tree. On covertype, 26,219 nodes against ./tree's 26,377, depth 59 for both, training accuracy 98.65% against 98.66%. ./tree follows C4.5 Release 8, which was verified node for node earlier (2026-10-13), and J48 differs from R8 in small details.
- **scikit-learn:** unpruned, so 48,085 (gini) and 42,773 (entropy) nodes with 100% training accuracy on covertype.

## Results (median training time)

| Dataset | Protocol | Machine | --simple | --serial | --parallel (24) | scikit-learn | Weka J48 |
|---|---|---|--:|--:|--:|--:|--:|
| chess_1m | CART | epyc1 | 474 ms | 29.5 ms | 21.6 ms | 544 ms | – |
| chess_1m | CART | dell1 | 493 ms | 46 ms | 54.3 ms | 568 ms | – |
| chess_1m | C4.5 | epyc1 | 623 ms (prune 143) | 45.4 ms (9.9) | 30.1 ms (5.8) | 548 ms | 2.69 s (0.78 s) |
| chess_1m | C4.5 | dell1 | 640 ms (141) | 72.5 ms (15.9) | 73.3 ms (13.3) | 577 ms | 2.71 s (0.79 s) |
| covertype | CART | epyc1 | 7.59 s | 857 ms | 82.3 ms | 4.23 s | – |
| covertype | CART | dell1 | 8.62 s | 1.10 s | 103 ms | 4.31 s | – |
| covertype | C4.5 | epyc1 | 12.5 s (0.45 s) | 1.13 s (77.6 ms) | 122 ms (17.1 ms) | 3.84 s | 142 s (17 s) |
| covertype | C4.5 | dell1 | 14.2 s (0.51 s) | 1.67 s (136 ms) | 180 ms (35.3 ms) | 3.96 s | 187 s (17.7 s) |

CART pruning at a fixed α takes under 1 ms in every backend.

- **--simple against --serial:** --serial is 7.8–16× faster. On covertype it is 7.8–11× faster: --simple sorts every node's rows by all 54 features, while --serial sorts once (the presort) and partitions the sorted orders. On chess it is 8.8–16× faster, where the per-node sort of a million rows dominates.
- **Pruning:** --simple's C4.5 pruning is 3.7–14× slower than --serial's. It routes the rows through the tree itself and allocates its own memory, while --serial reuses the grower's leaf of each row and lends its workspace.
- **Against the libraries:**
  - --simple is about as fast as scikit-learn on chess, and 1.8–3.6× slower on covertype.
  - It is 11–13× faster than Weka J48 on covertype C4.5 and 4.2–4.3× faster on chess.
  - --serial is 2.4–18× faster than scikit-learn on one thread, and 37–126× faster than J48 on C4.5 (112–126× on covertype).
- **--parallel on one socket:**
  - Covertype: 9.3–11× faster than --serial (CART 857 → 82 ms on epyc1).
  - Chess 1M: too small to gain. 17 nodes and two features leave 24 threads nothing to share. On dell1 the parallel runs are as slow as serial, or 1.2× slower. dell1's governor is `powersave` (epyc1's is `performance`), and its one-thread times are 1.03–1.6× epyc1's.

## Reproduce

```bash
# on mordor3, after build.sh and prepare.sh covertype (bench/cluster/README.md)
for d in chess_1m covertype; do
  bench/cluster/submit.sh -m 3 -d $d -o cluster-simple simple-epyc-$d simple-dell-$d
done
bench/.venv/bin/python bench/simple_chart.py bench/results/cluster-simple
```

`simple-*` runs use impls `tree,tree_simple,sklearn,j48` with threads `1,all`, and pass `SOCKET=auto` and `RUN_EXTRA=--hash-trees` to job.sh. `simple_chart.py` refuses to draw if the ./tree backends' tree hashes differ.
