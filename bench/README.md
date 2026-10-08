# CPU benchmark: ./tree vs reference CART and C4.5 implementations

This directory is a benchmark for a paper. It compares `./tree` on the CPU with:
- **CART:** scikit-learn and R rpart.
- **C4.5:** Weka J48 and YaDT.

It measures training time and peak memory, and records tree size and test accuracy so you can check that the tools built comparable trees. Nothing has been run yet.

## Files

| File | Purpose |
|---|---|
| `setup.sh` | Builds `tree_cpu`, creates `.venv`, and downloads Weka and YaDT into `.tools/`. It also compiles the J48 adapter. |
| `datasets.toml` | Dataset registry: source CSV, split, subsampling. Add new datasets here. |
| `prepare.py` | Writes `data/<name>/`: the split as float32 binary (for the adapters), a CSV for `./tree`, and CSV plus `.names` for YaDT. Also writes the tiny `_baseline` dataset. |
| `benchlib.py` | The protocols (the exact settings for every tool), the implementations, and the data layout. |
| `adapters/` | One program per library (`sklearn_fit.py`, `rpart_fit.R`, `J48Fit.java`). Each loads the binary data, times only the fit, and prints one JSON line. `./tree` and YaDT are run directly. |
| `run.py` | Runs every case in fresh processes. It measures time inside each tool and memory from outside, and writes `results/<run-id>/`. |
| `report.py` | Turns results into Markdown tables and a CSV. It can merge runs from several machines. |

## Usage

```bash
bench/setup.sh                                   # once per machine; needs python3, R, JDK 11+, taskset, unzip, curl
bench/.venv/bin/python bench/prepare.py          # every dataset in datasets.toml whose source exists
bench/.venv/bin/python bench/run.py --dry-run    # print the plan
bench/.venv/bin/python bench/run.py diabetes covertype_10k --cpus 0,2,4,6,8,10 --threads 1,2,4,6
bench/.venv/bin/python bench/report.py --md report.md --csv summary.csv
```

`run.py` options:
- Selection: `--protocols`, `--impls`.
- Threads: `--threads 1,2,4,all` sets the thread counts tried for the multi-threaded tools.
- Pinning: `--cpus` sets the order of logical CPUs to pin to.
- Repetitions and limits: `--reps` (default 5), `--mem-reps` (default 3), `--timeout`.
- Thermal rest: `--cooldown` sets seconds of rest between processes.
- JVM: `--java-heap` sets its maximum heap.
- `--skip check,time,memory,baseline` leaves out phases.

## Protocols (what is compared)

Each protocol is one well-defined piece of work, done with equivalent settings in every tool. The exact flags are in `benchlib.py`.

| Protocol | ./tree | scikit-learn | rpart | J48 | YaDT |
|---|---|---|---|---|---|
| `cart_full`: grown to pure leaves, depth ≤ 30 | `--cart --no-prune -d 30` | `max_depth=30` | `cp=0, minsplit=2, minbucket=1, maxdepth=30` | | |
| `cart_depth12` | `-d 12` | `max_depth=12` | `maxdepth=12` | | |
| `cart_alpha`: one tree, pruned at a fixed α | `--alpha α` | `ccp_alpha` with the same leaf count | `prune(cp = α·n / root risk)` | | |
| `cart_cv10`: α chosen by 10-fold CV with the 1-SE rule (whole procedure) | `--cv 10` | GridSearchCV over 16 α, 1-SE refit, threads | `xval=10`, 1-SE | | |
| `c45`: error-based pruning, CF 0.25, subtree raising | `--c45` | | | `-C 0.25 -M 2` | `-ebpg -c 0.25 -m 2` |
| `c45_unpruned` | `--c45 --no-prune` | | | `-U -M 2` | `-np -m 2` |

**Settings that keep the comparison fair:**
- **Same depth cap.** Every CART tool stops at depth 30, because rpart cannot grow deeper.
- **No extra rpart work.** rpart's competitor and surrogate splits are off (`maxcompete=0, maxsurrogate=0`), since no other tool computes them.
- **One α for all CART tools in `cart_alpha`.** α is computed once per dataset and cached in `data/<name>/alpha.json`. It is `./tree`'s 10-fold CV choice, which is deterministic, so it is the same on every machine.
- **scikit-learn's α is matched by size.** scikit-learn's `ccp_alpha` measures Gini impurity, not misclassified rows, so it gets the value from its own pruning path whose tree has the same number of leaves.
- **`cart_cv10` is a whole procedure, not one tree.** Report it separately. scikit-learn has no built-in CV pruning, so it fits 162 trees there instead of 11.
- **C4.5 needs no α search.** Error-based pruning is a single pass over the tree, so every C4.5 protocol grows exactly one tree.

## How time is measured

The reported time is **training time**: from the data being in memory to the finished, pruned model. Each tool measures it itself, with a monotonic clock:

| Tool | What is timed |
|---|---|
| ./tree | `train total`: presort, build, pruning or CV |
| YaDT | the indexing part of its read time, plus build and prune (from its log) |
| scikit-learn | `fit`, or the whole search |
| rpart | `rpart()` + `prune()` |
| J48 | `buildClassifier` |

**What is excluded, and why:**
- **Excluded:** reading files, process start-up, and prediction.
- **Why:** these measure the CSV parser and the language runtime, not the algorithm.
- **Where to find them:** the whole-process wall time is recorded too (`wall_seconds`) for an appendix.

**Preparation counts as training.** Each tool's own data preparation is included: `./tree`'s presort, YaDT's indexing, and the sorting the other libraries do inside their fit call. It is part of the algorithm, and leaving it out would favour some tools over others.

**Repetitions and run order:**
- **Check run:** one untimed process per case records tree size and test accuracy. For the native tools (./tree, YaDT) it is also the warm-up: it puts the file in the page cache and raises CPU clocks.
- **Timed runs:** `--reps` fresh processes per case. Report the median and the min–max range.
- **In-process warm-up:** the adapters warm up before the timed fit. J48 builds once on all rows so the JIT can compile; scikit-learn and rpart fit once on 2,000 rows to load code paths.
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
| **`peak_rss_bytes` (main metric)** | the kernel's high-water mark of resident memory, `ru_maxrss` from `wait4()`. This is the same number GNU time's `%M` prints. Exact, no sampling, no overhead. | everything the process touched: runtime, libraries, input data, the loader's temporary memory, training structures, and memory-mapped files | |
| **`peak_anon_bytes`** | the largest sum of `RssAnon` over the process tree, sampled from `/proc` every 5 ms (`--sample-ms`) | memory the program allocated | file-backed pages: memory-mapped input (./tree maps its CSV), program and library code. Can miss a spike shorter than the sampling interval. |
| **runtime footprint** | `peak_rss_bytes` of the same tool on the tiny `_baseline` dataset (200 rows) | interpreter, JVM or R, libraries | |

**Report three numbers** for each case:
- peak RSS;
- peak RSS minus the runtime footprint, which is memory for the data plus training;
- the input size (n × F × 4 bytes) for scale.

**How the measurement is kept clean:**
- **Separate memory runs.** Memory runs are fresh processes with no warm-up and no evaluation (no test set loaded), separate from the timing runs.
- **Binary input for the adapters.** scikit-learn, rpart and J48 read float32 binary files directly into their own data structures, so no CSV parser's temporary objects inflate the peak.
- **CSV for ./tree and YaDT.** They only read CSV, so their peak includes the parser. Two possible improvements for `./tree`: a binary input format, or a `--load-only` mode that would allow subtracting the load peak exactly.
- **Single process per run.** Every tool runs as one process (scikit-learn's search uses threads, not worker processes), so `ru_maxrss` covers all of its work. A tool that spawned worker processes would need cgroup v2 `memory.peak` instead. That counts the whole process tree, but also charges it with page cache from file reads.

**Caveats to state in the paper:**
- **JVM.** Its peak RSS depends on its heap sizing: the JVM grows the heap rather than collect early, so J48's number is an upper bound. Use the same `--java-heap` on every machine and report it. To find J48's minimum heap, lower `-Xmx` until it fails (an optional extra experiment).
- **Not the same as allocation profiles.** Peak RSS isn't an allocation profile. To explain where `./tree`'s memory goes, use heaptrack or valgrind massif on `./tree` alone; they are too slow and too language-specific for the cross-tool comparison.

**Cross-checking by hand:**

```bash
/usr/bin/time -v ./tree_cpu --serial --c45 bench/data/covertype_10k/train.csv   # "Maximum resident set size"
```

GNU time is the `time` package. Its "Maximum resident set size" must match `peak_rss_bytes`.

## Datasets

Edit `datasets.toml`. The entries so far:
- diabetes;
- covertype: 10k and full;
- SUSY: 50k and full, test set = its last 500k rows;
- HIGGS: full, test set = its last 500k rows.

Only numeric features are supported so far. To add one, put the CSV under `datasets/`, add an entry, and run `prepare.py <name>`. Use small variants (`train_rows`) to check correctness and the full datasets for timing and scaling. Keep cases that time out in the report as timeouts rather than dropping them.
