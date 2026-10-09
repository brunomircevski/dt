# SUSY 500k benchmark: ./tree vs CART and C4.5 implementations

**Status:** run on 2026-10-09 (one run per case). Results at the end, files in this directory.

## Run it

```bash
bench/results/cpu-pc-susy500k/run.sh          # each case once
bench/results/cpu-pc-susy500k/run.sh -m 5     # each case 5 times
```

- **Time:** about 6 min with one run per case (setup and the warm-up check take ~4 min of it), about 12 min with `-m 5`. Per run: J48 ~27 s, scikit-learn ~18 s, YaDT ~14 s (1 thread) or ~5 s (28 threads), rpart ~10 s, ./tree ~1 s.
- **To run again:** move this directory's results away first; `run.sh` refuses to mix two runs.
- **Progress:** every process prints `[n/total]`, what it runs, and when it ends its time, peak memory, tree size, accuracy and the elapsed time. Ctrl+C stops the run and the tool it is running.
- **Before:**
  - commit the code (`machine.json` records the ./tree version);
  - set the `performance` governor (already set);
  - close browsers and chat apps.
- **What `run.sh` does:**
  1. Refuses to overwrite existing results.
  2. Builds the tools (`bench/setup.sh`).
  3. Runs the J48 warm-up pilot (see Training time).
  4. Runs the benchmark:
     `bench/run.py susy_500k --protocols cart_alpha,c45 --threads 1,all --cpus 2,0,1,3-27 --reps <-m> --timeout 3600`
  5. Writes the outputs.
- **Outputs:**

  | File | Content |
  |---|---|
  | `report.md` | Tables |
  | `summary.csv` | One row per case |
  | `runs.csv` | One row per process |
  | `chart.png` | Time and peak memory per tool |
  | `results.jsonl` | Raw rows with the exact commands |
  | `machine.json` | Hardware and versions |
  | `plan.json` | Settings and SHA-256 of every data file |
  | `warmup-check/` | The pilot's runs and decision |
  | `run.log` | Everything `run.sh` printed |

## Method

### Implementations

| Algorithm | Implementation | Threads |
|---|---|---|
| CART | **./tree** (this repository) | 1 (`--serial`), 28 (`--parallel`) |
| CART | scikit-learn 1.9.1 `DecisionTreeClassifier` | 1 |
| CART | rpart 4.1.27 (R 4.6.1) | 1 |
| C4.5 | **./tree** | 1, 28 |
| C4.5 | Weka 3.8.7 J48 (OpenJDK 17.0.20.1) | 1 |
| C4.5 | YaDT 2.3.0 | 1, 28 (`-tt`) |

- **Single-threaded tools:** scikit-learn, rpart and J48 build a tree on one thread only, so they have no 28-thread case.
- **Separate tables:** single-thread and 28-thread results are never compared in one table.
- **No parallel CART competitor.** No other library builds the same pruned CART tree in parallel, so the 28-thread CART table holds only ./tree.
  - XGBoost `exact` grows the same unpruned tree (115,607 vs 115,583 nodes on 500k SUSY rows). Its pruning isn't cost-complexity: at the same α it keeps 9,751 nodes against scikit-learn's 3,473.
  - Intel oneDAL does not parallelise a single tree (6.2 s on 1 thread, 5.7 s on 28).
  - Spark MLlib and H2O split on histogram bins.
- **C4.5:** YaDT is the published multicore C4.5 (Aldinucci, Ruggieri, Torquati 2010).

### Data and evaluation

- **Dataset:** SUSY (Baldi, Sadowski, Whiteson 2014; UCI): 5,000,000 rows, 18 numeric features, 2 classes. Its published split puts the last 500,000 rows in the test set.
- **Subset used here** (`susy_500k` in `bench/datasets.toml`): the first 500,000 rows of the published training part, and the first 100,000 rows of the published test part. Both are taken in file order; SUSY's rows are already in random order.

  | Part | Rows | Signal |
  |---|--:|--:|
  | training | 500,000 | 45.7% |
  | test | 100,000 | 45.7% |

- **Same data everywhere.** The split was written once (`bench/prepare.py`); every tool and every run reads those rows:
  - float32 binary for scikit-learn, rpart and J48;
  - CSV for ./tree and YaDT, printing each float32 value exactly.
- **Checked on every run:**
  - every CSV row is compared with the binary file;
  - every data file's SHA-256 is recorded;
  - every process reports the rows and features it loaded, and a mismatch is an error.
- **Held-out test set, not training accuracy:**
  - Training accuracy rises with every leaf (unpruned CART on 500k SUSY rows: 98.05% on training rows, 72.6% on test rows), so it would reward pruning less.
  - Training accuracy is still reported, as a consistency check.
- **Precision:** with 100,000 test rows, the 95% interval of an accuracy near 80% is ±0.25 points; smaller differences are noise. The majority class gives 54.3%.
- **No cross-validation:** every case grows one tree on the training rows.

### Settings

| | Settings |
|---|---|
| **CART** (`cart_alpha`) | Gini, binary splits, depth ≤ 30 (rpart's limit), one tree pruned by cost-complexity at a fixed **α = 1e-5**, the same number for every tool, no cross-validation. |
| ./tree | `--cart -d 30 --alpha 1e-05` |
| scikit-learn | `criterion="gini", max_depth=30, ccp_alpha=1e-5` |
| rpart | `split="gini", minsplit=2, minbucket=1, cp=0, maxcompete=0, maxsurrogate=0, xval=0, maxdepth=30`, then `prune(cp = α·n / root risk)` |
| **C4.5** (`c45`) | Gain ratio, error-based pruning with CF 0.25, subtree raising, at least 2 rows per leaf. |
| ./tree | `--c45` |
| J48 | `-C 0.25 -M 2` |
| YaDT | `-ebpg -c 0.25 -m 2` |

No tool is tuned to match another tool's tree size; each reports the tree it builds.

### Training time

**Wall time measured inside each tool**, from "training data in memory, in the tool's own format" to "final pruned tree":

| Tool | Timed | Inside the timer |
|---|---|---|
| ./tree | `train total` | presort, build, prune |
| scikit-learn | `fit()` | build, pruning (incl. its internal sorting) |
| rpart | `rpart()` + `prune()` | data-frame conversion, build, prune |
| J48 | `buildClassifier()` | data copy, build, pruning |
| YaDT | its log: indexing + build + prune | indexing, build, prune |

- **Left out for every tool:** process start-up, reading the file, building the input structure, and prediction. Including them would measure parsers and language runtimes.
- **Preparation counts as training:** presort and indexing are part of the algorithm.
- **Warm-up:**
  - scikit-learn, rpart and J48 fit one untimed tree on the first 50,000 rows in the same process (code loading, JVM JIT).
  - ./tree and YaDT need none, because loading isn't timed.
- **Pilot check of the warm-up:**
  - The pilot times J48 on these 500k rows, 3 runs after a 50k-row warm-up and 3 after a full-size warm-up.
  - If the 50k warm-up is more than 3% slower, the main run warms up on all rows.
- **Reported numbers:**
  - median and min–max over the runs (one run per case unless `-m` asks for more);
  - the ratio to ./tree with the same thread count.

### Peak memory

**Peak resident memory (RSS) up to the end of training**, the kernel's exact high-water mark. It counts runtime, input data, parser, warm-up and training structures, but not evaluation.

- **Same process as the time.** Each tool reads its peak (`VmHWM`) right after training, before it loads the test set or predicts. So the peak covers runtime, data, the 50k-row warm-up and training.
  - YaDT, a closed binary, can't read its own peak. Its measured process therefore only trains and saves the tree, and its whole-process peak is used. A second, unmeasured YaDT process loads the saved tree and classifies the test rows.
- **Runtime footprint:** each tool's peak RSS on a 200-row dataset. The report also gives peak RSS minus this footprint (data + training).
- **The launcher:** every process is started by a tiny launcher (`bench/adapters/rusage.c`). Linux gives a child the peak RSS of the process it was forked from, so starting tools from the Python harness inflated them (./tree on 200 rows: 316 MiB instead of 7 MiB).
- **Input reading:**
  - scikit-learn, rpart and J48 read a 36 MB binary file.
  - ./tree reads its CSV through a 64 KiB buffer per thread.
  - YaDT reads its CSV with its own parser.
  - No tool keeps a copy of the file in memory.
- **J48's peak is an upper bound:** the JVM grows its heap (`-Xmx` 15g) rather than collect early.

### Tree size and accuracy

- **Every run reports**, after its timer stops: nodes, leaves, depth, training and test accuracy. With `-m` above 1 they must be identical across the runs; otherwise the report marks the case.
- **Nodes** = internal nodes + leaves. All splits are binary, so leaves = (nodes + 1) / 2.
- **Depth** = edges from the root to the deepest leaf; a root-only tree has depth 0. The same in every tool.
- **Accuracy:** each tool predicts with its own code. YaDT's comes from its confusion matrices (its printed error is rounded).

### Machine and procedure

- **CPU:** Intel Core i7-14700KF.
  - 8 performance cores with two threads each: CPUs 0–15, up to 5.6 GHz.
  - 12 efficiency cores: CPUs 16–27, up to 4.3 GHz.
- **System:** 31 GiB RAM, Arch Linux, kernel 7.2.9.
- **Settings:** `performance` governor, turbo on.
- **Builds:**
  - ./tree: g++ 16.2.1, `-O3`, no `-march=native` (the other tools are generic builds too).
  - Python 3.14.7, NumPy 2.5.3.
- **1 thread:** every tool is pinned with `taskset` to CPU 2, a performance core. The JVM's and R's helper threads share that core.
- **28 threads:** CPUs 0–27. On this hybrid CPU the ideal speedup is well below 28×.
- **No hidden threads:** `OMP/OPENBLAS/MKL_NUM_THREADS=1` for every process.
- **Per case:** one run, or `-m N` runs, each a fresh process measuring time and memory together. Each repetition runs the cases in a new random order.
- **Timeout:** 1 hour per process; a timeout is reported, not dropped.

### Caveats for the paper

- **α units:** scikit-learn measures α in Gini impurity, ./tree and rpart in misclassified rows. The same α therefore prunes scikit-learn's tree to a different size.
- **Depth cap:** a CART tree at depth 30 has reached the cap, which is the same for every tool.
- **YaDT** is a closed binary: its times come from its own log.
- **R's timer** (`Sys.time`) is a wall clock, which is accurate at the scale of seconds.
- **One split:** accuracy differences carry the ±0.25-point uncertainty above.
- **One run per case:** no spread over repetitions; rerun with `-m 5` for medians.

## Results

Run on 2026-10-09: one run per case, governor `performance`, load average 1.4 at the start.
- **Name:** run as `susy500k-20261009`, then renamed; `machine.json`'s command line still shows the old run id.
- **Code:** commit `582251c`. `machine.json` says `-dirty` only because an old results folder (`bench/results/susy500k-20261008-v2/`) had been deleted in the working tree; the code was unchanged.
- **Warm-up check:** J48 took 25.86 s after the 50k-row warm-up and 25.82 s after the full-size one (×1.002), so the 50k rule was kept.
- **Runtime footprints** (peak RSS on 200 rows): ./tree 7 MiB, YaDT 5 MiB, Weka J48 55 MiB, rpart 72 MiB, scikit-learn 126 MiB.

**CART, 1 thread** (α = 1e-5)

| Implementation | Train time | × ./tree | Peak RSS | Nodes | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 0.85 s | 1.0× | 138 MiB | 3,269 | 30 | 81.98% | 79.26% |
| scikit-learn | 16.9 s | 19.8× | 188 MiB | 3,473 | 25 | 81.36% | 78.94% |
| rpart | 9.48 s | 11.1× | 425 MiB | 2,863 | 29 | 81.75% | 79.32% |

**C4.5, 1 thread** (CF 0.25)

| Implementation | Train time | × ./tree | Peak RSS | Nodes | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 0.64 s | 1.0× | 203 MiB | 3,579 | 27 | 80.96% | 79.26% |
| Weka J48 | 25.9 s | 40.6× | 749 MiB | 3,635 | 27 | 80.97% | 79.27% |
| YaDT | 10.4 s | 16.3× | 758 MiB | 3,533 | 27 | 80.94% | 79.29% |

**28 threads**

| Algorithm | Implementation | Train time | × ./tree | Peak RSS | Nodes | Depth | Test acc. |
|---|---|--:|--:|--:|--:|--:|--:|
| CART | ./tree | 0.114 s | 1.0× | 248 MiB | 3,269 | 30 | 79.26% |
| C4.5 | ./tree | 0.137 s | 1.0× | 314 MiB | 3,579 | 27 | 79.26% |
| C4.5 | YaDT | 2.87 s | 20.9× | 1.10 GiB | 3,533 | 27 | 79.29% |

**Reading the results**
- **Accuracy:** every tool reaches the same test accuracy within the ±0.25-point interval (78.94–79.32%).
- **Speed:** ./tree is 11–20× faster than the CART libraries and 16–41× faster than the C4.5 ones on one thread. On 28 threads it is 21× faster than YaDT.
- **Speedup on 28 threads:** ./tree 7.5× (CART) and 4.7× (C4.5); YaDT 3.6×.
- **Memory:** ./tree has the lowest peak RSS in every table. Without each tool's runtime footprint, scikit-learn's data and training take less (62 vs 131 MiB). On 28 threads both ./tree and YaDT need more memory than on 1 thread.

Full tables: `report.md`; every number: `summary.csv` (per case) and `runs.csv` (per process); chart: `chart.png`.
