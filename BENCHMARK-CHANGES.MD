# Benchmark changes before the final rerun

This file is a work order for the agent that implements and runs the final
benchmarks on the mordor3 cluster. It records every decision about how the
paper's benchmarks are measured, taken in the review session of 2026-11-16.
Implement everything below, rerun the benchmarks, draw the charts, update the
docs, then commit and push to GitHub.

The author's goal: **one simple, fair, professional rule for every tool,
defensible to a paper's reviewers. Ten runs must equal ten times running the
tool once.**

Contents:
1. The timing rule
2. Run counts, run order, placement
3. What to change in the code
4. Which benchmarks to rerun
5. Charts and documentation
6. Problems while running
7. Decisions already taken (do not reopen)
8. mordor3 practicalities
9. Order of work and checklist

---

## 1. The timing rule (all tools)

**One run = one fresh process** that does the following, in order:
1. starts;
2. loads the data;
3. does the tool's **start-up**, if it has any;
4. trains **exactly one** tree;
5. reports its numbers;
6. exits.

Rules:
- No tool fits anything before the timed fit.
- No tool trains several timed trees in one process.
- Ten runs are ten executions of the same command.

### Start-up versus warm-up

| | Start-up (allowed, before the timer, reported separately) | Warm-up (forbidden) |
|---|---|---|
| What it is | Preparation that does not depend on the data | Running real training code before the timed fit |
| Examples | process launch, interpreter/JVM/R start, imports, CUDA context creation, eager loading of GPU kernels (`CUDA_MODULE_LOADING=EAGER`), creating **empty** memory pools and staging buffers | a fit on a few rows, a full-size untimed fit, pre-growing a memory pool to the data's size, running the split kernels once |
| Why | The tool needs it once per process, whatever the data, and every real program pays it before its first tree | It leaves the JIT compiled, memory pools grown and caches filled, so the timed fit is "the second fit in a process", not "running the tool once" |

### What is timed: the training time

The training time is **all work that depends on the data**, from the data
being in host memory as the tool's input arrays to the finished tree:
- copying the data to the GPU and allocating its memory, **including growing
  memory pools**;
- presorting, binning or indexing (`QuantileDMatrix`, YaDT's indexing,
  ./tree's presort);
- building the tree.

**Not timed:**
- reading the file into memory;
- start-up (see above). GPU start-up is reported separately as
  `gpu_startup_seconds`. Process and runtime start-up is only in the whole
  process's `wall_seconds`, and is broken down in the end-to-end benchmark;
- prediction and evaluation;
- process exit.

**Unpruned trees, depth <= 10** (protocols `cart_depth10` and
`c45_depth10_unpruned`) for every timing number. The author will compare
accuracy on pruned trees (CART with a fixed alpha, C4.5 with CF 0.25), untimed.
Those runs are **not part of this work order**: there is no plan for them
yet, so do not invent one. Mention in the docs that they come later.

### Per tool

| Tool | Start-up (not timed, reported) | Timed | Status |
|---|---|---|---|
| ./tree `--serial`, `--multicore`, `--simple` | none | `train total`: presort and build | already correct |
| ./tree `--gpu`, `--hybrid` | CUDA context, 2 MiB memory pool, pinned staging buffers (`gpuPrepare`, [src/build/gpu_grower.cu:1863](src/build/gpu_grower.cu)), printed as `gpu startup` | `train total`: GPU setup and upload, presort, build | correct; check the binary-input change (§3.1) |
| scikit-learn | interpreter start and imports | `DecisionTreeClassifier.fit` | warm-up removed in e3362b96; remove its code (§3.2) |
| rpart | R start, `library(rpart)` | `rpart()` | same |
| Weka J48 | JVM start, class loading of the adapter | `buildClassifier` (JIT compilation during it stays in) | same |
| YaDT 1 and 48 threads | process start | dTcmd's `total time` minus its `load time` (CSV parse), which is its indexing, plus the sum of its per-tree `time` fields (build) ([bench/run.py:305](bench/run.py)) | correct |
| cuML | Python start, imports, CUDA context, eager kernels; RMM left as cuML's default, not reconfigured or pre-grown | `fit` from host float32 arrays: copy, binning, build, and any allocation `fit` makes | **to change** (§3.3) |
| XGBoost | Python start, imports, CUDA context, eager kernels | `QuantileDMatrix` from host arrays plus `train` | **to change** (§3.3) |

**The JIT question for J48.** Not warming up the JVM is deliberate. HotSpot
compiles a method after a few thousand calls or loop iterations, which takes
seconds inside the first fit. Measured against J48's fits (about 830 s on
HIGGS, about 5.3 h on MNIST8M), that is well under 1%, and a real user who
trains one tree pays it too.
- **Optional check, not for the paper:** run one J48 HIGGS fit with
  `-XX:+PrintCompilation` and note in `cluster-cpu-vs-others/BENCHMARK.md`
  when compilation settles.

**`--repeat` is not used for timing.** It stays in ./tree for two purposes
only, and neither feeds the timing tables:
- the power benchmark (`bench/power/power_bench.sh`), which needs a long,
  steady window. Its `--repeat-warmup` spins the threads before each timer so
  that the power meter sees a steady state; it is part of the energy method,
  documented in `bench/power/README.md`;
- A/B tests of code changes (`bench/cluster/ab.sh`).

### Paper text (Section 4, Protocol)

Replace the `\plan{...}` of "Protocol and Measurement" in
`paper/sections/experiment-design.tex` with this text (adapt only the numbers):

> Each run is a fresh process that loads the data and trains one unpruned tree
> of depth at most 10; no warm-up fit is performed and every run is an
> independent execution of the same command. We time all data-dependent
> training work (host-to-device transfer, memory allocation including pool
> growth, presorting or binning, tree construction) and exclude reading the
> input file and runtime start-up (process launch, interpreter or JVM start,
> CUDA context creation, kernel loading, creation of empty memory pools); the
> GPU start-up is reported separately. Each case is run 10 times (3 times where a run
> takes over an hour) in shuffled order on an exclusively allocated node, and
> we report the mean.

---

## 2. Run counts, run order, placement

- **Runs per case: 10.** If one run of a case takes **more than about an
  hour**, run that case **3 times** instead.
  - Known 3-run cases: `--simple` on full ECBDL'14 (1.7–2 h) and full
    SOREL-20M (3.6–3.9 h), and J48 on MNIST8M (about 5.3 h).
  - Estimate the new tools from their first run (rpart on MNIST8M may take
    over an hour) and extend `runs()` in `bench/cluster/final.sh`
    accordingly.
  - **Every 3-run case is named on its chart** (e.g. "3 runs: one run takes
    over an hour").
- **Report means only.** No medians, no bootstrap intervals. The charts may
  show the spread (min–max, or each run as a dot), but every number in text
  and tables is a mean.
- **Placement is fixed.** Every run of a case uses the same CPUs, the same
  memory policy and the same GPU (`--exclusive` nodes, `taskset`, the GPU on
  the NUMA node `job.sh` picks). Do not let the OS choose: earlier, where the
  data landed made small epyc runs up to 8× slower.
- **Order is shuffled.** Slow drift (temperature, other jobs, the shared file
  system) must spread over all cases, not land on one tool.
  - `run.py` already shuffles the order of its cases on each repetition.
  - Where one plan line runs one tool's ten runs back to back (the `vs` lines
    and `gpubench.py`), interleave instead: either one `run.py` invocation
    per dataset with all tools, or ten rounds of one run per tool in shuffled
    order.
  - Choose the simplest of these that the code supports and describe it in
    the benchmark's `BENCHMARK.md`.
- **Keep `run.py --drop-cache`** (set by `bench/cluster/job.sh`) for every
  training-time benchmark.
  - The dataset leaves the page cache before every run. Reading is not timed,
    so this only lengthens the wall time.
  - It prevents cached files from earlier runs filling one NUMA node and
    pushing a run's memory to the other socket (`bench/cluster/README.md`,
    "The datasets leave the page cache").
  - The end-to-end benchmark handles the cache differently (see its
    `BENCHMARK.md`).

---

## 3. What to change in the code

### 3.1 ./tree (only after the binary-input work lands)

Another agent is adding **binary float32 input** to ./tree (every dataset
already has `train.f32` / `train.y.i32` / `test.*` in
`bench/data/<dataset>/`). Merged 2026-11-16 (a `.f32` path is read as the binary dump, by extension;
CSV stays for users). Checks:
- [x] `bench/benchlib.py tree_input()` passes the binary files (`train.f32`),
  so ./tree reads the same bytes as the libraries.
- [x] **Start-up never leaks into `train total`.** `main.cpp` waits for
  `gpuReady.get()` before `trainTree` starts its timer, so a start-up longer
  than the load waits outside `train total`.
  - `gpu startup` runs on a background thread while the data loads.
  - With binary input the load becomes much shorter (HIGGS about 1 s, MNIST8M
    a few seconds) and may finish **before** the start-up does.
  - The training timer must start only after the start-up thread has
    finished. Otherwise waiting for it becomes part of `train total`.
  - Verify in `src/app/main.cpp` / `trainer.h`, and print the wait if there is
    one.
- [x] ./tree prints its load time (`load NNN ms`), for the end-to-end
  benchmark.
- [x] ./tree gets an option that skips its built-in accuracy computation
  (e.g. `--no-accuracy`; today it always computes training and test accuracy
  after training, [src/app/main.cpp:225](src/app/main.cpp)), for the
  end-to-end benchmark. This is the same switch the adapters already have
  (`eval=0`).
- [ ] Trees are bit-identical to the CSV input: compare `tree_sha256` with the
  2026-11-15 runs for at least HIGGS and MNIST8M. (Already checked when it
  was built: dumped trees identical from CSV and `.f32` on covertype, SUSY,
  HIGGS, chess, ECBDL'14 and MNIST8M; tests/check.sh runs every case from
  both.)

### 3.2 CPU libraries (scikit-learn, rpart, J48, YaDT)

- [x] **Remove the warm-up code entirely**, not just set it to 0, so nobody
  can turn it on again:
  - `WARMUP_ROWS` and the `warmup` field of the tools table in
    `bench/benchlib.py`;
  - `--warmup-rows` in `bench/run.py`;
  - the `warmup=` argument of `bench/adapters/sklearn_fit.py`,
    `rpart_fit.R` and `J48Fit.java`;
  - their comments.

  Keep the results' `warmup_rows` column only if old rows need it to parse.
- [x] The rpart and YaDT harness from e3362b96 stays:
  - rpart: CART, one thread, `maxdepth=10`;
  - YaDT: C4.5, `-np -m 2 -depth 10`, one thread and all 48 CPUs (run name
    `vs-epyc-<data>-yadt-c45-multi`).

  YaDT needs `yadt.names`, `train.yadt.csv` and `test.yadt.csv` for HIGGS and
  MNIST8M:
  - prepare them with `bench/prepare.py` (`datasets.toml` no longer sets
    `yadt = false` for MNIST8M);
  - on mordor3, HIGGS had only `yadt.names` on 2026-11-16;
  - record the sha256 of the `.f32`/`.i32` files first and check they are
    unchanged afterwards.
- [x] `bench/setup.sh` downloads Weka and YaDT. R 4.6.1 with rpart 4.1.27 and
  OpenJDK 17 are in the micromamba env `~/micromamba/envs/dtbench` on mordor3.

### 3.3 GPU libraries: `bench/results/cluster-gpu-vs-others/scripts/gpubench.py` and `job.sh`

Today one process does a warm-up fit (cuML on all rows with
`CUML_WARMUP_ROWS=0`: 20.9 s on MNIST8M, then 7.96 s timed fits), then 10
timed fits. That breaks the rule twice.

Change it to:
- [x] **One timed fit per process.**
  - `job.sh` launches a new `gpubench.py` process for every run (REPS = 1),
    10 per case, interleaved over the cases (shuffled order, §2).
  - `results.jsonl` gets one row per process.
- [x] **No warm-up fit.**
  - Delete `WARMUP_ROWS`, `CUML_WARMUP_ROWS` and the warm-up block.
  - Also delete the 50,000-row forest agreement check, or move it into a
    separate, untimed checking process.
- [x] **An explicit start-up phase before the timer, recorded as
  `startup_seconds`:**
  1. export `CUDA_MODULE_LOADING=EAGER` in `job.sh` before the process
     starts;
  2. import the library (cuML imports RMM);
  3. create only the CUDA context on the chosen GPU (e.g. `cudaFree(0)`
     through `cuda-python`, or the smallest call that creates it without
     setting up an allocator pool of its own; avoid a CuPy allocation, which
     creates CuPy's pool, not cuML's);
  4. **do not reconfigure or pre-grow RMM:** leave cuML's default memory
     resource as it is. Any allocation `fit` makes stays timed, as ./tree's
     pool growth does;
  5. for XGBoost, nothing beyond the context.

  Done 2026-11-16. Caveat (review, 2026-11-17): libcuml and libxgboost link
  their own CUDA runtime statically, so `cudaFree(0)` through cuda-python
  creates the shared context but does not load *their* kernels; with EAGER
  each library loads all of its kernels at its first CUDA call, inside `fit`.
  `CUDA_MODULE_LOADING` can be overridden (`job.sh`, `run.py`) to measure
  EAGER against LAZY; see `cluster-gpu-vs-others/BENCHMARK.md`.
- [x] **Timed:** exactly what it is today, from host float32 NumPy arrays to
  the fitted model.
  - cuML: `fit` (copy, binning, build).
  - XGBoost: `QuantileDMatrix` and `train`.
  - Keep recording XGBoost's `QuantileDMatrix` part separately, as now.
- [x] An evaluation switch like the adapters' `eval=0|1`: today the last fit
  always computes accuracy, predictions and the tree walk. The timing runs
  keep evaluation on, after the timer; the end-to-end benchmark turns it off.
- [x] VRAM: peak above idle per process, sampled as now.
- [x] The "forest of 10 / 10" bars (library default forests) follow the same
  rule: one fit per process, no warm-up.
- [x] Document the new procedure in the docstring and in
  `cluster-gpu-vs-others/BENCHMARK.md` ("Time" bullet).

### 3.4 Harness

- [x] `bench/cluster/final.sh`:
  - default runs 10, with 3 for the long cases (§2);
  - remove `airline` from `PROFILE_DATA` (airline is dropped from the paper;
    chess stays for profiling);
  - add the new kind `e2e` for the end-to-end benchmark (§4) and the
    interleaving (§2);
  - submit the GPU libraries' job as part of the run (`gpulibs`), with the new
    `gpubench.py`.
- [x] Every result row records: the commit (`git describe`), the node, the CPU
  list, the GPU, `wall_seconds`, and `gpu_startup_seconds` for GPU tools
  (./tree already does; `gpubench.py` adds it).

---

## 4. Which benchmarks to rerun (all of them)

All of them, with the build that includes binary input, at 10 runs per case
(3 for the long ones):

| Benchmark | Tools | Notes |
|---|---|---|
| `cluster-susy`, `cluster-higgs`, `cluster-mnist8m`, `cluster-ecbdl`, `cluster-sorel`, `cluster-chess` | ./tree all five backends | epyc and dell for CPU, hopper and ampere for GPU |
| `cluster-cpu-scaling` | ./tree 1–48 threads | HIGGS and MNIST8M |
| `cluster-profiling` | ./tree `--profile` | chess and HIGGS, ECBDL'14 15M; **no airline** |
| `hybrid-vs-gpu-only` | ./tree `--hybrid` vs `--gpu` | as before; `--handoff-rows` from `bench/handoff_rows.json` (below) |
| `cluster-cpu-vs-others` | scikit-learn, rpart (CART, 1 thread), YaDT (C4.5, 1 and 48 threads), Weka J48 (C4.5, 1 thread) | HIGGS and MNIST8M only; ./tree's bars from the same rerun |
| `cluster-gpu-vs-others` | cuML, XGBoost, CART only | HIGGS and MNIST8M only; hopper and ampere |
| `cluster-end-to-end` (**new**) | everything, end to end with every step | HIGGS only; design in [bench/results/cluster-end-to-end/BENCHMARK.md](bench/results/cluster-end-to-end/BENCHMARK.md) |
| `bench/power` | ./tree, energy | keeps its `--repeat` method (§1); rerun with the new build |

Every `--hybrid` run takes its `--handoff-rows` from `bench/handoff_rows.json`
(`run.py`; the fastest of a sweep per GPU node, dataset and depth limit, made
once by `bench/cluster/handoff_sweep.sh`, results in
`bench/results/handoff-sweep/`). Do not sweep again in the rerun; a missing
entry falls back to ./tree's default.

The stored values (swept 2026-11-16 at 291a736f, binary input, 5 runs per
value of 256–32,768 and ./tree's default, CART and C4.5 at depth 10). The pick
is the smallest mean, over CART and C4.5, of the time over that protocol's
fastest, timed without the GPU setup (`train_seconds - gpu_setup_seconds`).
The allocation and upload do not depend on the value, and on the H100 they
take one of two speeds from run to run, which would otherwise decide a close
sweep (ECBDL'14 15M). `nodepth` is the earlier no-limit sweep
(`bench/results/hybrid-vs-gpu-only/sweep/`, f53347a9), re-picked the same way
with unchanged values. `run.py` uses `depth10` for `-d 5` and `-d 10`, and
`nodepth` for `-d 15` to `-d 30` and for no limit. Chess has no entry: its
depth-10 tree has no node small enough to hand off, so the value cannot
matter, and it runs at ./tree's default.

| Dataset | Node | `depth10` | ./tree default | CART best / default (s, presort + build) | C4.5 best / default (s, presort + build) | `nodepth` (f53347a9) |
|---|---|--:|--:|--:|--:|--:|
| susy | hopper | 1,024 | 512 | 0.024 / 0.024 | 0.029 / 0.030 | – |
| susy | ampere | 1,024 | 512 | 0.034 / 0.035 | 0.039 / 0.040 | – |
| higgs | hopper | 512 | 512 | 0.066 / 0.066 | 0.083 / 0.082 | 256 |
| higgs | ampere | 1,024 | 512 | 0.095 / 0.096 | 0.115 / 0.116 | 512 |
| ecbdl_15m | hopper | 4,096 | 6,310 | 1.691 / 1.700 | 1.714 / 1.711 | – |
| ecbdl_15m | ampere | 4,096 | 6,310 | 2.409 / 2.433 | 2.438 / 2.447 | – |
| mnist8m | hopper | 2,048 | 7,840 | 1.418 / 1.474 | 1.395 / 1.386 | 8,192 |
| mnist8m | ampere | 1,024 | 7,840 | 1.885 / 2.047 | 1.887 / 1.875 | 8,192 |
| sorel_4m | hopper | 1,024 | 16,384 | 1.650 / 1.790 | 1.676 / 1.659 | 16,384 |
| sorel_4m | ampere | 1,024 | 16,384 | 2.367 / 2.636 | 2.377 / 2.352 | 16,384 |

Before the full run, run `final.sh -t` (one run of each kind into
`cluster-test/`) and check every kind works. Then move the old run folders out
of the way (`git rm`; they stay in git history) so `final.sh`'s clash check
passes, and submit.

---

## 5. Charts and documentation

- **Every chart states**:
  - the runs behind each bar, and which cases have 3 runs and why;
  - the commit;
  - the node and GPU;
  - "mean of N fresh-process runs, no warm-up";
  - anything unusual that a reader needs to interpret the bar (see §6).
- Redraw every chart:
  - `vs_chart.py cpu` and `vs_chart.py gpu`;
  - each `cluster-*` chart script named in its `BENCHMARK.md`;
  - `paper/tools/charts.py`.

  The GPU chart must no longer show cuML entropy bars.
- Update the docs:
  - every `BENCHMARK.md` (Data line: dates, commit, runs, rule);
  - `bench/results/README.md`;
  - `bench/README.md`, section "How time is measured". It still describes
    the old warm-up rule and says "pruned model". Rewrite it to §1 of this
    file;
  - `bench/cluster/README.md`;
  - `docs/GPU-LIBRARIES.md`.
- **Paper:**
  - put the protocol paragraph (§1) into `experiment-design.tex`;
  - replace `\tbd` cells in `results.tex` with the new means;
  - recompute the introduction's ratios by the rules in §7;
  - the run-count sentences in the table captions must match what ran.
- Only means in text and tables.

---

## 6. Problems while running

The agent running the benchmarks resolves problems itself and records them:
- **Infrastructure failures** (PATH, missing files, a killed job, a node
  reboot): fix the cause and rerun the affected case. Note it in that
  benchmark's `BENCHMARK.md`.
- **Out of memory, timeouts, or a tool growing a clearly different tree**:
  - do not change the protocol or settings to make it pass;
  - record what happened;
  - leave the bar out or mark it on the chart, and explain it in the
    `BENCHMARK.md`.
- **Outliers or bimodal times** (e.g. the H100's two upload speeds seen
  before):
  - keep all runs and report the mean;
  - mention it on the chart if it moves the mean by more than a few percent;
  - investigate if it is new.
- **Anything a reader needs to interpret a number goes on the chart** (a
  short footnote line) and in the `BENCHMARK.md`.

---

## 7. Decisions already taken (do not reopen)

- **No cuML entropy anywhere.** An entropy tree is not C4.5 and no GPU library
  grows C4.5 trees, so the GPU libraries are compared on **CART only**.
- **Library comparisons on HIGGS and MNIST8M only.** Comparison with existing
  libraries is not the paper's main goal.
- **Datasets:**
  - airline is dropped for good;
  - chess is used only for timing and profiling (no test set);
  - ECBDL'14 and SOREL-20M run in full on the CPU, and as 15M / 4M
    subsamples on the GPU.
- **Introduction ratios:**
  - libraries against Serial Optimized only (one thread against one thread:
    scikit-learn for CART, J48 for C4.5);
  - Multicore against Serial Optimized;
  - GPU against Multicore;
  - no Serial Simple, Hybrid, cuML or XGBoost ratios in the introduction.
- **Tables:** YaDT on 48 threads is the parallel C4.5 baseline, shown under
  the Multicore column of `tab:time-c45` (paper commit 7da01487).
- **48-thread CPU baseline:** YaDT 48 threads, HIGGS and MNIST8M.
- **Paper scope:** written up to Section 3; Sections 4 and later are short
  `\plan` notes. Keep them short. `\tbd` marks numbers awaiting the rerun.

---

## 8. mordor3 practicalities

- **Login node and SLURM:**
  - the login node only submits jobs; everything that computes (setup,
    prepare, benchmarks, chart drawing) runs in SLURM jobs;
  - SLURM is on PATH only in a login shell (`bash -lc "..."`);
  - hidden partitions: `squeue -a`, `sinfo -a`;
  - benchmark jobs use `--exclusive`; no `--mem` or `--gres`.
- **Nodes:**
  - never CPU benchmarks on hopper or ampere;
  - epyc1 and epyc2 have the same hardware; dell2 and dell3 only for
    one-thread runs.
- **Clone:**
  - work in a **fresh clone** (e.g. `~/dt-final`);
  - `~/dt` is the author's working checkout: do not touch it;
  - share the prepared data with `ln -s ~/dt/bench/data bench/data`;
  - GitHub over SSH works on mordor3, so commit and push from there.
- **Never `scancel` a running job without a reason**, and never one started
  without `timeout --foreground` (SLURM's kill once left a benchmark running).
- **Before submitting:** check jobs every 30–60 minutes; no busy polling.
- `bench/cluster/README.md` has the node table, settings and the existing
  final-rerun procedure.

---

## 9. Order of work and checklist

1. Wait for the binary-input change to ./tree to be on main; pull.
2. Implement §3.2, §3.3, §3.4 and the end-to-end harness. Commit.
3. Build (`bench/cluster/build.sh` or as in `bench/cluster/README.md`) and run
   `bench/setup.sh`, each in a SLURM job. Prepare the YaDT files (§3.2).
4. Check §3.1 on a short job.
5. `final.sh -t`: one run of each kind; fix what fails.
6. Remove the old run folders (`git rm`), commit, submit the full plan.
7. Monitor; handle problems by §6.
8. Charts, docs, paper tables (§5).
9. Commit and push to GitHub. Report the commits, every case's mean and runs,
   and every problem met.
