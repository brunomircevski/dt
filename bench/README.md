# Benchmark harness: ./tree vs reference CART and C4.5 implementations

This directory holds the benchmarks for a paper. Results are in [results/](results/README.md), the runs on the mordor3 cluster in [cluster/](cluster/README.md). It compares `./tree` on the CPU with:
- **CART:** scikit-learn and R rpart.
- **C4.5:** Weka J48 and YaDT.

It measures training time and peak memory, and records tree size and training and test accuracy so you can check that the tools built comparable trees.

**Two rules hold for every benchmark here:**
- **No cross-validation.** Every case grows exactly one tree.
- **No forced tree size.** Each tool gets the same settings, including one fixed α for all CART tools, and reports whatever tree it builds. No tool's parameters are tuned to reproduce another tool's node or leaf count.

## Files

| File | Purpose |
|---|---|
| `setup.sh` | Builds `tree_cpu`, creates `.venv`, and downloads Weka and YaDT into `.tools/`. It also compiles the J48 adapter and the `rusage` launcher. |
| `datasets.toml` | Dataset registry: source CSV, split, subsampling. Add new datasets here. |
| `prepare.py` | Writes `data/<name>/`: the split as float32 binary (`train.f32`, `train.y.i32`, `meta.json`: read by the adapters and by `./tree`), a CSV with the same values (for users of `./tree`), and CSV plus `.names` for YaDT. Every CSV row is checked to read back as the binary values. Also writes the tiny `_baseline` dataset. A `prefix_of` set (another set's first rows, e.g. `sorel_4m`) gets only `meta.json`: `./tree` reads the parent's `train.f32` with `--rows N`, the adapters the first N rows of the parent's binary files. A `split = "none"` set (the synthetic chess sets) has no test set. |
| `benchlib.py` | The protocols (the exact settings for every tool), the implementations, the data layout and the data check. |
| `adapters/` | One program per library (`sklearn_fit.py`, `rpart_fit.R`, `J48Fit.java`). Each loads the binary data, times only the fit, then measures accuracy and prints one JSON line. `./tree` and YaDT are run directly. `rusage.c` is the launcher every measured process runs under. |
| `run.py` | Checks the data, then runs every case in fresh processes. It measures time inside each tool and memory from outside, and writes `results/<run-id>/`. |
| `report.py` | Turns results into Markdown tables (single- and multi-thread separate), a per-case CSV and a per-process CSV. It can merge runs from several machines. |
| `chart.py` | One figure per dataset: training time and peak RSS per tool, single- and multi-thread in separate panels. |
| `testset_chart.py`, `rows_chart.py`, `runs_chart.py`, `machines_chart.py` | The test-set benchmarks' charts: bars per backend and node, time and memory against rows, every run, several machines, charts in a grid. |
| `scaling_chart.py` | The thread-scaling chart of `cluster-cpu-scaling`: speed-up over one thread against threads, epyc and dell, HIGGS and MNIST8M, CART and C4.5, plus each run's `scaling.csv` (time, speed-up, efficiency, peak RSS). |
| `vs_chart.py` | The charts of `cluster-cpu-vs-others` (./tree against scikit-learn and J48) and `cluster-gpu-vs-others` (against cuML and XGBoost). |
| `profile_chart.py`, `profile_share_chart.py` | Where the training time goes, from `run.py --profile`. |
| `merge_runs.py` | Appends extra runs of a case to its run folder. |

## Usage

```bash
bench/setup.sh                                   # once per machine; needs python3, R, JDK 11+, taskset, unzip, curl
bench/.venv/bin/python bench/prepare.py          # every dataset in datasets.toml whose source exists
bench/.venv/bin/python bench/run.py --dry-run    # print the plan
bench/.venv/bin/python bench/run.py diabetes covertype_10k --cpus 0,2,4,6,8,10 --threads 1,2,4,6
bench/.venv/bin/python bench/report.py --md report.md --csv summary.csv --runs-csv runs.csv
bench/.venv/bin/python bench/chart.py bench/results/<run-id>
```

Finished benchmarks, one directory per backend, machine and dataset: see [results/README.md](results/README.md).

`run.py` options:
- Selection: `--protocols`, `--impls`.
- CART pruning: `--alpha` sets the fixed α of `cart_alpha` (default `CART_ALPHA` = 1e-5 in `benchlib.py`); `./tree` gets it as `--prune α`.
- Threads: `--threads 1,2,4,all` sets the thread counts tried for the multi-threaded tools.
- Pinning: `--cpus` sets the order of logical CPUs to pin to.
- Repetitions and limits: `--reps` (runs per case, default 1), `--timeout`.
- Warm-up: `--warmup-rows` sets the rows of the managed runtimes' untimed warm-up fit (default `WARMUP_ROWS` = 50,000; `all` = the whole training set).
- Thermal rest: `--cooldown` sets seconds of rest between processes.
- JVM: `--java-heap` sets its maximum heap.
- `--no-baseline` leaves out the runtime-footprint runs.
- Output: `--run-id` names the directory in `bench/results/` (default host-date); `--dry-run` prints the plan and exits; `--seed` seeds the shuffled order of the cases.
- NUMA and cache: `--membind NODES` runs every process under `numactl --membind`; `--drop-cache` drops the datasets from the page cache before each process.
- `./tree` only: `--train-accuracy-only`, `--test-accuracy` (skip or limit predicting the dumped tree), `--hash-trees` (SHA-256 of every dumped tree), `--profile` (write `--profile` JSON per run), `--handoff-rows` (a number, or `default` for ./tree's own; without it the value measured best for the node, dataset and depth limit in [`handoff_rows.json`](handoff_rows.json), see below).

## Protocols (what is compared)

Each protocol is one well-defined piece of work, done with equivalent settings in every tool. The exact flags are in `benchlib.py`.

| Protocol | ./tree | scikit-learn | rpart | J48 | YaDT |
|---|---|---|---|---|---|
| `cart_full`: grown to pure leaves, depth ≤ 30 | `--cart -d 30` | `max_depth=30` | `cp=0, minsplit=2, minbucket=1, maxdepth=30` | | |
| `cart_depth12` | `-d 12` | `max_depth=12` | `maxdepth=12` | | |
| `cart_alpha`: one tree, pruned at a fixed α | `--cart -d 30 --prune α` | `ccp_alpha=α` | `xval=0`, `prune(cp = α·n / root risk)` | | |
| `cart_alpha_nodepth`: `cart_alpha` without the depth cap, for ./tree-only benchmarks | `--cart --prune α` | | | | |
| `c45`: error-based pruning, CF 0.25, subtree raising | `--c45 --prune 0.25` | | | `-C 0.25 -M 2` | `-ebpg -c 0.25 -m 2` |
| `c45_unpruned` | `--c45` | | | `-U -M 2` | `-np -m 2` |
| `cart_depth10`, `c45_depth10_unpruned`: depth ≤ 10, no pruning, every current benchmark | `-d 10` | `max_depth=10` | | `-U -M 2`, its depth limit | |

**Settings that keep the comparison fair:**
- **Same depth cap.** Every CART tool stops at depth 30, because rpart cannot grow deeper.
- **No extra rpart work.** rpart's competitor and surrogate splits are off (`maxcompete=0, maxsurrogate=0`), since no other tool computes them.
- **One fixed α for all CART tools in `cart_alpha`.** Every tool gets the same number: `CART_ALPHA`, or `run.py --alpha`.
  - Nothing chooses or tunes it per tool or per dataset.
  - There is no cross-validation, and no tool is matched to another's tree size.
  - scikit-learn's `ccp_alpha` measures Gini impurity, while `./tree` and rpart count misclassified rows. So the same α prunes scikit-learn's tree to a somewhat different size. Report the size each tool builds; don't adjust for it.
- **C4.5 needs no α.** Error-based pruning is a single pass over the tree, so every C4.5 protocol grows exactly one tree.

## The hybrid's hand-off size

`./tree --hybrid` hands nodes below `--handoff-rows` rows to the CPU threads. Each implementation runs in its best measured setting, so every `--hybrid` benchmark run takes the value measured fastest for its GPU node, dataset and depth limit, stored in [`handoff_rows.json`](handoff_rows.json) (`{node: {dataset: {"depth10" | "nodepth": rows}}}`; `run.py` takes the shallowest measured limit at least as deep as the run's: `-d 5` and `-d 10` take `depth10`, `-d 15` to `-d 30` and no limit `nodepth`, as the hand-off acts on the small nodes of the deep levels; with no such entry, ./tree's default). The values come from [`cluster/handoff_sweep.sh`](cluster/handoff_sweep.sh) (a sweep of 256–32,768 and ./tree's default, CART and C4.5, the fastest mean over both, timed without the GPU setup, which does not depend on the value; results in [`results/handoff-sweep/`](results/handoff-sweep/)) and are reused, not swept again for each benchmark: rerun the sweep only when a dataset, a GPU node or the hybrid's code changes. Chess has no entry: its depth-10 tree has no node small enough to hand off. ./tree's own default (10 × features, between 512 and 16,384) is reported next to the tuned value.

## How time is measured

The reported time is **training time**: from the data being in memory to the finished, pruned model. Each tool measures it itself, with a monotonic clock:

| Tool | What is timed |
|---|---|
| ./tree | `train total`: GPU setup and upload (`--gpu`, `--hybrid`), presort, build, pruning |
| YaDT | the indexing part of its read time, plus build and prune (from its log) |
| scikit-learn | `fit` |
| rpart | `rpart()` + `prune()` |
| J48 | `buildClassifier` |

**What is excluded, and why:**
- **Excluded:** reading files, process start-up, and prediction.
- **Why:** these measure the CSV parser and the language runtime, not the algorithm.
- **Where to find them:** the whole-process wall time is recorded too (`wall_seconds`) for an appendix.

**The GPU's one-time start-up is excluded, and reported apart.** Creating the CUDA context, the device memory pool and the page-locked staging buffers costs 0.1–0.7 s once per process (0.6–0.7 s on the RTX 2080 Ti node), whatever the data. ./tree does it on a background thread while the data loads (`gpuPrepare`) and prints it as `gpu startup` (`gpu_startup_seconds`), not in `train total`. Everything that depends on the data stays in: allocating and uploading it, the GPU presort, the build. This matches how the GPU libraries are timed (cuML, XGBoost: an untimed warm-up fit in the same process first, see [results/cluster-gpu-vs-others/BENCHMARK.md](results/cluster-gpu-vs-others/BENCHMARK.md)), i.e. a process that already uses the GPU. Do not add the start-up to the training time: it overlaps the load. For a single cold run, use `wall_seconds`: on small sets (SUSY) the start-up is larger than the whole training, and a CPU backend wins end to end.

**Preparation counts as training.** Each tool's own data preparation is included: `./tree`'s presort, YaDT's indexing, and the sorting the other libraries do inside their fit call. It is part of the algorithm, and leaving it out would favour some tools over others.

**Repetitions and run order:**
- **One kind of run.** `--reps` fresh processes per case; each one gives training time, peak memory, tree size and accuracy. Report the mean and the spread.
- **Tree size and accuracy come from every run.** After its timer stops, each process reports nodes, leaves, depth, and training and test accuracy (each tool predicts with its own code; ./tree's dumped tree is evaluated by `benchlib.predict_tree`, checked against ./tree's own training accuracy; with `run.py --train-accuracy-only`, used on the mordor3 cluster, ./tree reports only its own training accuracy). They must be the same in every repetition; `report.py` marks a case where they are not.
- **One warm-up rule for the managed runtimes** (scikit-learn, rpart, J48): one untimed fit on the first `--warmup-rows` training rows, in the same process, before the timed fit. It loads code paths, and the JVM's JIT compiles the same methods on 50,000 rows as on millions. The native tools (./tree, YaDT) need none: loading is not timed, so the page cache does not matter.
- **Order:** shuffled anew on each repetition, so slow drift (heat, background load) is spread over all cases.

**Machine hygiene (record it; `run.py` writes `machine.json`):**
- **Idle machine.** Set the `performance` governor, and either fix turbo or record whether it is on. Laptops throttle, so use `--cooldown` and look at the min–max spread.
- **Pin with `--cpus`.** On hybrid Intel CPUs, list P-cores first (`/sys/devices/cpu_core/cpus`); otherwise a single-thread run may land on an E-core. For thread-scaling curves, list one logical CPU per physical core before the SMT siblings (`lscpu --extended`).
- **No hidden threads.** `OMP/OPENBLAS/MKL_NUM_THREADS=1` is set for every process.
- **Generic build.** Build `./tree` with the repository's flags (`-O3`, no `-march=native`), because the other tools are generic builds too.

## How memory is measured (and why from outside)

**Memory is measured by the operating system, outside the process**, and no tool's code is changed. In-process counters (Python `tracemalloc`, JVM heap statistics, R `gc()`, malloc hooks) can't be compared across languages:
- each counter sees only its own allocator: `tracemalloc` misses the C/C++ memory of scikit-learn's tree builder, and JVM heap statistics miss memory outside the heap;
- YaDT is a closed binary, so it can't be instrumented at all.

`run.py` records, per process:

| Metric | Source | Includes | Excludes |
|---|---|---|---|
| **`peak_rss_train_bytes` (main metric)** | the kernel's high-water mark of resident memory (`VmHWM`), read by the tool right after training. YaDT, a closed binary, can't: its measured process only trains and saves the tree (a second, unmeasured process classifies the test rows), so its number is `ru_maxrss`. Exact, no sampling. | everything up to the end of training: runtime, libraries, input data, the loader's temporary memory, the 50k-row warm-up, training structures | evaluation (test set, predictions) |
| **`peak_rss_bytes`** | `ru_maxrss` from `wait4()` in the `rusage` launcher (GNU time's `%M`): the whole process | everything, evaluation included | |
| **runtime footprint** | `peak_rss_bytes` of the same tool on the tiny `_baseline` dataset (200 rows) | interpreter, JVM or R, libraries | |

**Report three numbers** for each case:
- peak RSS;
- peak RSS minus the runtime footprint, which is memory for the data plus training;
- the input size (n × F × 4 bytes) for scale.

**Why the launcher:** Linux starts a new process's `ru_maxrss` at the peak RSS of the process it was forked from, and `exec` keeps it. Started straight from `run.py` (a Python process that has just checked hundreds of MiB of data), every tool would report at least `run.py`'s own peak: ./tree on 200 rows showed 316 MiB instead of 7 MiB. So every measured process runs under `.tools/rusage` (`adapters/rusage.c`), a tiny static program that forks the tool and reports its `wait4()` usage; `taskset` pins the launcher, and the tool inherits the pinning.

**How the measurement is kept clean:**
- **Measured in the timed process, before evaluation.** Time and peak memory come from the same run; the peak is read right after training, so loading the test set and predicting never count.
- **Binary input for the adapters and ./tree.** scikit-learn, rpart and J48 read the float32 binary files directly into their own data structures, so no CSV parser's temporary objects inflate the peak. ./tree is given the same files (`train.f32`; it picks the format by the extension): it reads them through a 4 MiB buffer per thread (no memory mapping of the file), so it never holds more than a few MiB of the file, and its load is 3–5× faster than parsing the CSV of the same values (the trees are identical). Runs before this change (2026-11-16) loaded `train.csv`; loading is not timed, so their training times stand.
- **CSV for YaDT.** It only reads CSV, so its peak includes the parser.
- **Single process per run.** Every tool runs as one process, so `ru_maxrss` covers all of its work. A tool that spawned worker processes would need cgroup v2 `memory.peak` instead. That counts the whole process tree, but also charges it with page cache from file reads.

**Caveats to state in the paper:**
- **JVM.** Its peak RSS depends on its heap sizing: the JVM grows the heap rather than collect early, so J48's number is an upper bound. Use the same `--java-heap` on every machine and report it. To find J48's minimum heap, lower `-Xmx` until it fails (an optional extra experiment).
- **Not the same as allocation profiles.** Peak RSS isn't an allocation profile. To explain where `./tree`'s memory goes, use heaptrack or valgrind massif on `./tree` alone; they are too slow and too language-specific for the cross-tool comparison.

**Cross-checking by hand:**

```bash
bench/.tools/rusage /dev/stdout ./tree_cpu --serial --c45 bench/data/susy_50k/train.f32   # first number: peak RSS in KiB
/usr/bin/time -v ./tree_cpu --serial --c45 bench/data/susy_50k/train.f32                  # "Maximum resident set size"
```

GNU time is the `time` package (forked from a small process, so it is exact too). Its "Maximum resident set size" must match `peak_rss_bytes`.

## Same data for every run

- `prepare.py` writes one split per dataset, in every tool's format, and checks every CSV row against the binary files.
- At the start of every run, `run.py` repeats that check and records the SHA-256 of every data file in `plan.json`.
- Every process reports the rows and features it loaded (and YaDT the rows it classified); a mismatch with the dataset makes the run an error.

## Datasets

Edit `datasets.toml`; [cluster/README.md](cluster/README.md#data) describes each set.

Only numeric features are supported so far. To add one, put the CSV under `datasets/`, add an entry, and run `prepare.py <name>`. Use small variants (`train_rows`) to check correctness and the full datasets for timing and scaling. Keep cases that time out in the report as timeouts rather than dropping them.
