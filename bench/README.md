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
- Long cases: a case whose first run takes over `--long-seconds` (3600) runs `--long-runs` (3) times in all.
- End to end: `--no-eval` (no evaluation, ./tree `--no-accuracy`), `--prefetch` (the input into the page cache right before each process, dropped after it).
- GPU libraries: `--impls cuml,xgboost` runs `results/cluster-gpu-vs-others/scripts/gpubench.py` in `GPU_PYTHON` (default `~/gpu-dt-bench/venv/bin/python`); never by default.
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

**One run = one fresh process** that starts, loads the data, does the tool's
start-up (if it has any), trains **exactly one** tree, reports its numbers and
exits. No tool fits anything before the timed fit, and no process trains
several timed trees: ten runs are ten executions of the same command. This is
the rule of every timing benchmark since 2026-11-16
([BENCHMARK-CHANGES.MD](../BENCHMARK-CHANGES.MD), section 1).

**Start-up versus warm-up.**

| | Start-up: allowed, before the timer, reported apart | Warm-up: not allowed |
|---|---|---|
| What | preparation that does not depend on the data | running real training code before the timed fit |
| Examples | process launch, interpreter / JVM / R start, imports, CUDA context, eager loading of the GPU kernels (`CUDA_MODULE_LOADING=EAGER`), **empty** memory pools and staging buffers | a fit on a few rows, an untimed full-size fit, growing a memory pool to the data's size, running the split kernels once |
| Why | every real program pays it once, before its first tree, whatever the data | it leaves the JIT compiled, the pools grown and the caches filled: the timed fit would be a process's second fit, not running the tool once |

**What is timed: the training time**, all the work that depends on the data,
from the data in host memory as the tool's input arrays to the finished tree.
Each tool measures it with its own monotonic clock:

| Tool | Start-up (not timed) | Timed |
|---|---|---|
| ./tree `--serial`, `--multicore`, `--simple` | none | `train total`: presort and build |
| ./tree `--gpu`, `--hybrid` | CUDA context, a 2 MiB memory pool, pinned staging buffers (`gpuPrepare`), printed as `gpu startup`; it runs while the data loads, and `train total` starts only after it has finished (`gpu startup wait`: the part still running when the load ended) | `train total`: GPU setup and upload (allocation and pool growth included), presort, build |
| scikit-learn | interpreter, imports | `DecisionTreeClassifier.fit` |
| rpart | R, `library(rpart)` | `rpart()` |
| Weka J48 | JVM, class loading | `buildClassifier` (HotSpot's JIT compilation during it stays in: a user who trains one tree pays it too) |
| YaDT (1 and 48 threads) | process start | its `total time` minus its `load time` (parsing the CSV), i.e. its indexing, plus the sum of its per-tree `time` fields (build) |
| cuML | interpreter, imports (RMM), CUDA context with eager kernels; RMM left at cuML's default, not configured or pre-grown | `fit` from host float32 arrays: copy, binning, build, and every allocation it makes |
| XGBoost | interpreter, imports, CUDA context with eager kernels | `QuantileDMatrix` from host arrays plus `train` (both recorded) |

**Not timed:** reading the file into memory; start-up (the GPU's is recorded as
`gpu_startup_seconds`, the rest is only in the whole process's
`wall_seconds`, broken down by the end-to-end benchmark,
[results/cluster-end-to-end](results/cluster-end-to-end/BENCHMARK.md));
prediction and evaluation; process exit.

**Trees:** unpruned, depth ≤ 10 (protocols `cart_depth10` and
`c45_depth10_unpruned`) for every timing number. Accuracy on pruned trees
(CART at a fixed alpha, C4.5 at CF 0.25) will be compared separately, untimed;
it is not part of these benchmarks yet.

**Runs, order, placement:**
- **10 runs per case**, 3 where one run takes over an hour (`--simple` on full
  ECBDL'14 and SOREL-20M; `run.py` stops any case at `--long-runs` 3 whose
  first run took over `--long-seconds` 3600, e.g. J48 on MNIST8M). Every
  3-run case is named on its chart.
- **Means only** in text and tables; charts may show the spread (min–max).
- **Shuffled order:** `run.py` runs every case once per repetition, in a new
  order each time, so slow drift spreads over all cases. The libraries of
  `cluster-cpu-vs-others` share one `run.py`; the GPU libraries run in rounds
  of one process per case, shuffled (their `job.sh`).
- **Fixed placement:** every run of a case uses the same CPUs (`taskset`), the
  same memory policy and the same GPU (`--exclusive` nodes; the GPU on the NUMA
  node `bench/cluster/job.sh` picks).
- **The page cache:** `--drop-cache` drops the datasets before every process
  (reading is not timed; this keeps cached files from filling one NUMA node).
- **Every row records** the commit, the node, the CPU list, the GPU,
  `wall_seconds` and, for GPU tools, `gpu_startup_seconds`.

**Preparation counts as training.** Each tool's own data preparation is
included: ./tree's presort, YaDT's indexing, XGBoost's `QuantileDMatrix`, and
the sorting or binning the other libraries do inside their fit call. It is part
of the algorithm, and leaving it out would favour some tools over others.

**Tree size and accuracy come from every run.** After its timer stops, each
process reports nodes, leaves, depth, and training and test accuracy (each tool
predicts with its own code; ./tree's dumped tree is evaluated by
`benchlib.predict_tree`, checked against ./tree's own training accuracy). They
must be the same in every repetition; `report.py` marks a case where they are
not. `run.py --no-eval` skips all of it (the end-to-end runs).

**`./tree --repeat` is not used for timing.** It serves only the power
benchmark (`bench/power`, whose `--repeat-warmup` is part of its energy method)
and A/B tests of code changes (`bench/cluster/ab.sh`).

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
| **`peak_rss_train_bytes` (main metric)** | the kernel's high-water mark of resident memory (`VmHWM`), read by the tool right after training. YaDT, a closed binary, can't: its measured process only trains and saves the tree (a second, unmeasured process classifies the test rows), so its number is `ru_maxrss`. Exact, no sampling. | everything up to the end of training: runtime, libraries, input data, the loader's temporary memory, training structures | evaluation (test set, predictions) |
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

Only numeric features are supported so far. To add one, put the CSV in `datasets/<name>/` (each dataset's folder holds its originals next to its prepared files; on mordor3 `datasets/` and `bench/data` both link to `~/datasets`), add an entry, and run `prepare.py <name>`. Use small variants (`train_rows`) to check correctness and the full datasets for timing and scaling. Keep cases that time out in the report as timeouts rather than dropping them.
