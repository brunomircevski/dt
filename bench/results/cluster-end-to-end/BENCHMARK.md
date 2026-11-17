# End to end: from launch to finished tree, step by step (HIGGS)

**Status: implemented on 2026-11-16 (bench/run.py --no-eval --prefetch through
bench/cluster/job.sh E2E=1, final.sh kind `e2e`, chart bench/e2e_chart.py), not
run on 2026-11-17.** Data: `e2e-epyc-higgs` (epyc1, CPU tools, commit
8fe0c73d) and `e2e-hopper-higgs` (hopper, H100, commit 001a56c6), 10
fresh-process runs per tool and algorithm; `chart.png`, `summary.csv`
(`bench/e2e_chart.py`). This file is the specification it was built from. Read
[BENCHMARK-CHANGES.MD](../../../BENCHMARK-CHANGES.MD) first: its timing rule,
run counts and problem handling apply here too. Replace this status paragraph
with the Data line once it has run.

## Purpose

The other benchmarks time only training: the input is already in memory and
start-up is excluded (`BENCHMARK-CHANGES.MD` §1). This benchmark shows
**everything a user waits for**: starting the program, reading the data, GPU
start-up, preparing the data, building the tree. It shows each step for every
tool on one dataset.

It answers two questions:
- **What does a user really wait for?** For fast backends, the steps around
  training dominate: on HIGGS, ./tree's GPU training takes 0.13 s, but CSV
  loading took about 7.7 s.
- **Where does each library spend its time?** In start-up, data conversion
  (Weka's Instances, YaDT's parsing and indexing, XGBoost's `QuantileDMatrix`)
  or the tree itself?

It is a small, descriptive benchmark, not a replacement for the training-time
tables.

## What runs

**Dataset:** HIGGS only: 10.5M training rows, 28 features, 2 classes, 1.2 GB
as float32. It is small enough that the slowest tool (J48, about 14 min a run)
still finishes 10 runs in a few hours.

**Trees:** the same as the timing benchmarks: one tree, depth <= 10, no
pruning; protocols `cart_depth10` and `c45_depth10_unpruned`.

| Algorithm | Tool | Threads | Node |
|---|---|---|---|
| CART | ./tree `--simple` | 1 | epyc |
| CART | ./tree `--serial` | 1 | epyc |
| CART | ./tree `--multicore` | 48 | epyc |
| CART | ./tree `--gpu`, `--hybrid` | all 32 CPUs, 1 GPU | hopper (H100) |
| CART | scikit-learn | 1 | epyc |
| CART | rpart | 1 | epyc |
| CART | cuML (1 tree) | all CPUs, 1 GPU | hopper |
| CART | XGBoost (1 tree) | all CPUs, 1 GPU | hopper |
| C4.5 | ./tree `--simple`, `--serial`, `--multicore` | 1, 1, 48 | epyc |
| C4.5 | ./tree `--gpu`, `--hybrid` | all 32 CPUs, 1 GPU | hopper |
| C4.5 | Weka J48 | 1 | epyc |
| C4.5 | YaDT | 1 and 48 | epyc |

Settings per tool are those of `cluster-cpu-vs-others` and
`cluster-gpu-vs-others`: the same adapters, flags and versions. The CPU tools
run on epyc1 or epyc2 (the same hardware). The GPU tools run on hopper, with
the GPU and CPU placement of `cluster-gpu-vs-others/scripts/job.sh`.

**Input:** the float32 binary files `bench/data/higgs/train.f32` and
`train.y.i32` for every tool. That is what the adapters read today, and what
./tree reads once its binary input lands.
- **YaDT is the exception.** It only reads its own text format (`yadt.names`
  plus `train.yadt.csv`), so its "read" step includes parsing text. The chart
  says so.
- **Not CSV for everyone.** A CSV comparison would mostly compare CSV parsers
  (pandas, R, Weka's CSVLoader), not tree learners.
- **The page cache:** /home is NFS, and the timing benchmarks evict the data
  before every run (`run.py --drop-cache`, so cached files do not pile up on
  one NUMA node).
  - Here, **prefetch the input right before every process**:
    `cat` the files to `/dev/null`, pinned to the same CPUs and memory policy
    as the run, outside the run's wall timer.
  - Evict the cache again after the run, as `--drop-cache` does.
  - "Read" then measures getting the bytes into the tool's data structures,
    not NFS speed. Say so in the chart footnote.

**Runs:** 10 per tool and algorithm (none takes an hour). Each run is a fresh
process, in shuffled order over the tools on a node, with no warm-up fit,
exactly as `BENCHMARK-CHANGES.MD` §1–2.

**No prediction or evaluation in these runs.** The process should exit right
after the tree is built, so the wall time ends with training.
- **Adapters:** `eval=0` already skips their evaluation.
- **./tree** always computes training (and test) accuracy after training
  ([src/app/main.cpp:225](../../../src/app/main.cpp)). Add an option that
  skips it (e.g. `--no-accuracy`) and use it here. Otherwise the evaluation
  lands silently in the remainder (step 1).
- **YaDT** writes its tree to finish: keep that write and name it in the
  footnote.
- Tree size and accuracy for these settings come from the timing
  benchmarks.

## The steps

Every run reports these steps in seconds. Steps 2–5 are measured directly by
the tool.
- **Step 1 is the remainder:** wall time minus the non-overlapping measured
  steps. It equals the wall time by construction, so it is no validation.
- **Validate instead that:**
  - step 1 is never negative;
  - it is about the same across runs of a tool;
  - steps 4 + 5 agree with the timing benchmark (below).

| Step | What it covers | How it is measured |
|---|---|---|
| **1. Runtime start, exit and other** | exec, dynamic linking, interpreter/JVM/R start, imports, `library()`, class loading, process exit, and any work not covered by steps 2–5 | **the remainder:** total wall time (from the launcher, below) minus the non-overlapping time of steps 2–5. No clock comparison across processes is needed. |
| **2. Read data** | file → the tool's in-memory input: NumPy arrays, R data.frame, Weka `Instances`, YaDT's table, ./tree's columns | the tool's monotonic clock around its reading code |
| **3. GPU start-up** | ./tree (`--gpu-startup full`, its default): CUDA context, eager kernel loading, empty memory pool and staging buffers; cuML/XGBoost: the CUDA context (their kernels load lazily inside the fit) | ./tree: `gpu startup`; cuML/XGBoost: the new start-up phase of `gpubench.py` |
| **4. Prepare** | data-dependent preparation before growing: ./tree's GPU upload (`gpu setup`) and presort; XGBoost's `QuantileDMatrix`; YaDT's indexing | the tool's clock or log |
| **5. Build** | growing the tree | the tool's clock or log |

Steps 4 and 5 together are the training time of the timing benchmarks. Check
that their mean is within noise of that benchmark's mean for the same case,
and note it if not.

**Tools that cannot split steps 4 and 5** report one combined step,
**"Prepare + build"**, drawn as one segment with its own legend entry:
- scikit-learn's `fit`;
- rpart's `rpart()`;
- J48's `buildClassifier`;
- cuML's `fit`, which copies, bins and builds in one call.

Do not split a library call with tricks: what is not measurable stays
combined.

**./tree's GPU start-up overlaps the read** (`gpuPrepare` runs on a background
thread while the data loads). Draw what the process actually waits for:
- the read;
- then only the part of the start-up that is still running after the read
  ends ("GPU start-up, not overlapped").

Report the full start-up time in the table. The end-to-end bar must not count
the overlap twice. If ./tree does not print its load time and the remaining
wait, add those two lines to its output; that is a change to its printing
only, not to how it works.

### Per tool: where the step boundaries are

| Tool | 2 Read | 3 GPU start-up | 4 Prepare | 5 Build |
|---|---|---|---|---|
| ./tree CPU | `load` (new output line) | – | `presort` | `build` |
| ./tree GPU, hybrid | `load` | `gpu startup` (the non-overlapped part in the bar) | `gpu setup`, `presort` | `build` |
| scikit-learn | `np.fromfile` + reshape | – | `fit` (combined with build) | |
| rpart | `readBin` + `data.frame` | – | `rpart()` (combined) | |
| J48 | binary → `Instances` | – | `buildClassifier` (combined) | |
| YaDT | `load time` from its log (text parse) | – | indexing: `total time` − `load time` | build: the sum of its per-tree `time` fields (as `run.py` reads them today) |
| cuML | `np.fromfile` | context (kernels lazily, in `fit`) | `fit` (combined: copy, binning, build) | |
| XGBoost | `np.fromfile` | context (kernels lazily, in the fit) | `QuantileDMatrix` | `train` |

**Total wall time:** from the `rusage` launcher (fork to `wait4`), as
`run.py`'s `wall_seconds` is today. For `gpubench.py`, launch it the same way:
- under `bench/.tools/rusage`;
- or with a timestamp taken by `job.sh` right before `exec` and right after
  exit.

Use the same method for every tool.

Also record, per run:
- peak RSS (`peak_rss_bytes`);
- peak VRAM above idle (GPU tools);
- the commit, the node, the CPU list and the GPU.

## Implementation

Built as `bench/run.py` cases: `--no-eval` (./tree `--no-accuracy`, adapters and
`gpubench.py` `eval=0`), `--prefetch` (the input read into the page cache by a
pinned `cat` right before each process, dropped after it), `--no-baseline`;
`bench/cluster/job.sh` with `E2E=1`; `final.sh` plan lines `e2e-epyc-higgs`
(epyc1) and `e2e-hopper-higgs` (hopper); `bench/e2e_chart.py` draws `chart.png`
and writes `summary.csv`. The design notes below are kept as written.

### Design (2026-11-16)

- **One scheduler per node** (epyc, hopper).
  - It builds 10 rounds; each round holds every case of that node, in
    shuffled order.
  - It launches exactly one fresh process per entry (prefetch, run, evict),
    under the same `rusage` launcher.
  - It appends to one result stream per node: `e2e-epyc-higgs/` and
    `e2e-hopper-higgs/`, each with `results.jsonl` holding one row per
    process, with the case's name and its step fields.
  - The chart picks the cases from those streams.
- **Implementation:** either extend `bench/run.py` (it already has the
  launcher, pinning, shuffling and `results.jsonl`) so it can also launch
  `gpubench.py`, or write a small scheduler script that calls the adapters,
  ./tree and `gpubench.py`. Choose whichever is simpler.
- **The adapters and `gpubench.py`** print their step times; ./tree prints
  `load` and runs with the no-accuracy option.
- **`final.sh`:** a kind `e2e` that queues the epyc and hopper parts after the
  main runs.
- **Chart script:** `bench/e2e_chart.py`, writing `chart.png` and
  `summary.csv` here.

## Chart

One figure, two panels: CART (top) and C4.5 (bottom).
- **One horizontal bar per tool and backend**, sorted by total time.
- **Each bar shows the share of each step**, as a 100% stacked bar. Times
  range from about 1 s to about 14 min, so absolute stacked bars on a linear
  axis would make the fast tools invisible, and log axes cannot stack.
- **The mean total wall time in seconds** is printed at the end of each bar.
  The mean seconds of each step go in `summary.csv` and the table below.
- **The same step colours in both panels.** "Prepare + build" has its own
  colour, close to Build's, so combined and split tools can be told apart.
- **Footnote:**
  - HIGGS;
  - 10 fresh-process runs each, no warm-up;
  - input float32 binary, in the page cache; YaDT reads its own text format;
  - no prediction;
  - nodes epyc and hopper (H100);
  - commit;
  - anything unusual found while running (`BENCHMARK-CHANGES.MD` §6).

## Results

*(to fill: a table of the mean seconds per step, the total and the wall time
for every tool, and 3–5 bullets on what stands out)*

## Files

| File | Content |
|---|---|
| `chart.png` | the figure above |
| `summary.csv` | one row per tool, backend and algorithm: mean seconds per step, total, wall time, runs, peak RSS, VRAM |
| `e2e-epyc-higgs/`, `e2e-hopper-higgs/` | `results.jsonl` (every run of every case on that node), `machine.json`, `run.log` |
