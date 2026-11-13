# Handoff: CPU optimization hunt (started 2026-11-13)

This file is for the agent that continues this work **directly on mordor3**. The
work was started from a desktop session (worktree `cart-c45-optimization`) that stopped on a usage
limit. Nothing from it is merged. This file is the only thing pushed to main.

## The task (from the user)

> Analyze the codebase and last optimizations and performance changes. Think of improvements
> we can make to make this the fastest, the best optimized CART/C45 implementation possible.
> Dont change simple backend, but try to optimize all other. [...] propose to keep only the
> ones that improve performance. Try not to overcomplicate code for a negligible gains. When
> finish present results, what you tried and how much it helped. I will decide what to merge
> into main. [...] Run jobs on mordor3 (epyc, dell, hopper, ampere. GPU machines for cuda, CPU
> machines for cpu backends). [...] the goal is to find new optimizations and improve
> performance for hardware on cluster.

The rules:

* **The user decides what merges.** Never merge into main, and never push feature branches to
  GitHub: only `main` may exist there. Keep experiments on local branches in your clone on mordor3.
* **Trees must stay byte-identical.** Every backend must give the same `--dump` as base.
  `TREE_BIN=./tree_cpu tests/check.sh` must pass, and so must `cmpdumps.sh` (below) over every
  A/B case.
* **Do not touch the simple backend** (`src/build/simple_grower.cpp`).
* **Keep the code simple.** Under about 2% gain is not worth extra complexity. No new user-facing
  flags in a final version, and no dead code. Comments say why, with the measured number. Follow
  the house style of the file you edit.
* **Commit with explicit paths** (`git commit -m ... -- <paths>`). End messages with the
  attribution line the harness gives you.
* **Never run builds or `./tree` on the login node.** Everything goes through `sbatch`. Other
  people use the cluster, so jobs may queue.
* **Measure on whole nodes with interleaved A/B:**
  * epyc1/epyc2: 2× EPYC 9255, 48 cores.
  * dell1: 2× Xeon 6542Y, 48 cores.
  * dell2–4 have half the memory channels. They are fine for iterating, but quote final numbers
    from epyc and dell1.
  * Use 5 or more reps and medians.
  * Noise: epyc ~1% on one thread and a few % at 48 threads; dell at 48 threads 5–10% per run.

## Where things are on mordor3

| What | Path |
|---|---|
| Kit: scripts, binaries, results | `~/opt-dt/` |
| Patches of the four experiments | `~/opt-dt/patches/{wc,pb,kf,cs}/*.patch` (`git am`-able onto `55d83a48`) |
| Sources each binary was built from | `~/opt-dt/wt/<variant>/{src,Makefile,tests}` |
| Built binaries | `~/opt-dt/bin/tree_cpu_<variant>`: base, base+v3, base+native, wc1, wc1p, wc2, pb1, pb2, pb3, kf0, kf1, kf2 |
| Raw results (CSV from tune.sh) | `~/opt-dt/out/*.csv` |
| `--profile` JSONs | `~/opt-dt/prof/*.json` |
| Dumps of every case run | `~/opt-dt/dumps/<case>-<variant>.dump` |
| Datasets | `~/datasets/bench/<ds>/train.csv` (higgs, susy, mnist8m, covertype, chess_1b, ...); ecbdl_15m, sorel_4m and chess_100m: see below |
| SLURM | `/opt/slurm-25.05.8/bin`. It is not in PATH for non-login shells, so run `export PATH=/opt/slurm-25.05.8/bin:$PATH` first. |

Datasets (2026-11-13): every dataset lives in `~/datasets` (raw CSVs) and `~/datasets/bench/<ds>/`
(prepared). A checkout links `datasets -> ~/datasets` and `bench/data -> ~/datasets/bench`.

**Subsets are no longer copies** (main since this commit): `./tree --rows N` reads only the first N
rows of a CSV, and `datasets.toml`'s `prefix_of` sets (ecbdl_15m, sorel_4m, chess_1m to chess_500m)
hold only `meta.json`; `bench/run.py` passes the parent's `train.csv` with `--rows`. The kit's
binaries (base `55d83a48` and the variants) have no `--rows`, so the kit reads copies kept only for
it: `~/datasets/ecbdl/ecbdl_15m.csv`, `~/datasets/sorel/sorel_4m.csv` and
`~/datasets/chess/chess_100m.csv` (the same bytes as before). `gen.sh` and the case files in
`cases/` already point there. The chess sets have no test set any more.

`~/dt` is a clone of main (with both links): make the experiment branches there, or in a worktree
of it, instead of another clone:

```bash
cd ~/dt
for x in wc pb kf cs; do git switch -c opt/$x 55d83a48 && git am ~/opt-dt/patches/$x/*.patch; done
git switch main
```

`base` is `55d83a48`, the main from before this handoff commit. That commit only adds this file.

### The kit (`~/opt-dt`; read `AGENT.md` there, which was the brief for each experiment agent)

`AGENT.md` and `push.sh` assume the desktop. `push.sh` tars a desktop worktree to
`~/opt-dt/wt/<variant>`. On mordor3, copy the sources yourself:
`rm -rf ~/opt-dt/wt/<v> && mkdir -p ~/opt-dt/wt/<v> && git -C ~/dt archive opt/<x> src Makefile tests | tar x -C ~/opt-dt/wt/<v>`.

* `build_cpu.sh "<v>[+tag:CXXFLAGS]"` builds `bin/tree_cpu_<v>` (g++ 13.3) from `wt/<v>`. Run it as a job:
  `sbatch --wait -p dell -w dell4 -c 16 -t 20 -o out/build-<v>.out build_cpu.sh <v>`
* `build_cuda.sh <v>` builds `bin/tree_<v>` for sm_80+90. Run it on hopper. It is not used yet.
* `gen.sh "<variants>" algo:ds:mode[:extra args] ... > cases/x.txt` writes a case list.
  * algo:
    * `cart` and `c45`: `--no-prune -d 10`, the paper's protocol.
    * `cartfull`: alpha-pruned, no depth limit.
    * `c45full`: full depth, pruned.
  * mode:
    * `s`: `taskset -c 0 --serial`.
    * `p`: `--parallel`, 48 threads.
    * `pN`: N threads.
    * `g`, `go`: GPU.
  * extra example: `cart:higgs:p:--profile /home/bmircevski/opt-dt/prof/x.json`.
* `tune.sh cases/x.txt N` runs every case N times, interleaved, and writes CSV:
  `sbatch --wait -p epyc --exclusive -t 120 -o out/x-epyc.csv tune.sh cases/x.txt 5`
  (dell: `-p dell -w dell1 --exclusive`). Submit epyc and dell together.
* `python3 summ.py out/x.csv base` prints medians and % against base, with the spread.
* `./cmpdumps.sh base` checks that every dump equals base's. The last run printed "all dumps equal to base" (122 dumps).
* `python3 profsum.py prof/x.json` prints thread-seconds per kind of work (mark, partition_sweep,
  sweep, presort, idle, ...).
* `pbmem.sh "variants" reps` gives the max RSS of cart higgs/chess_100m. `out/pb-mem-epyc.csv` came out **empty**:
  rerun it as a job if pb's memory matters. pb adds one bitmap of rows/8 bytes per thread.

Load times: HIGGS ~8 s/run, SUSY ~3 s, mnist8m and ecbdl_15m minutes per run. Use the large
datasets (mode p, 3 reps) only for final A/Bs.

## Base profile findings (main 55d83a48, `--no-prune -d 10` unless noted)

Times are ms; "thread-s" is `--profile` thread-seconds. Per-node detail is in `~/opt-dt/prof/`.

### HIGGS CART, 48 threads

| | train | presort | build | partition_sweep | presort | idle | sweep | mark |
|---|---|---|---|---|---|---|---|---|
| epyc | ~496 | ~130 | ~366 | 9.95 thread-s | 5.24 | 4.92 | 2.46 | 2.27 |
| dell | ~722 | | | 11.29 thread-s | | | | 5.09 |

On one thread, epyc runs mark in 0.19, partition_sweep in 3.63 and presort in 1.84 (train
8130); dell runs mark in 0.29 and partition_sweep in 6.25.

**Mark costs 12–17× its one-thread cost.** That is the shared atomic `Columns::leftBits`
bitmap bouncing between cores.

Per-depth profiles (`prof/*higgs-cartd{1..8}-{p,s}.json`) show:

* Mark costs about 0.24 thread-s per level at every level, the root included.
* partition_sweep inflates from level 4 on: 2.1 thread-s per level at levels 5–6, against 0.41
  on one thread.

### Other datasets

**SUSY CART, 48 threads, epyc:** train ~212, idle 32–44%, mark 18× its one-thread cost.

**MNIST8M (10 classes):** bandwidth-bound, and the children's sweeps are not fused into the
partition.

| | train | presort | build | partition | sweep | presort |
|---|---|---|---|---|---|---|
| epyc CART | 4274 | 755 | 3529 | 105 thread-s | 54 | 35 |
| dell CART | 7452 | | | 159 thread-s | 120 | |
| epyc one thread | wall 115 s | | | 56.8 thread-s | 29.0 | 29.3 |

C4.5 trains in 4441 on epyc and 7822 on dell.

**ECBDL 15M, 48 threads:**

| | CART | presort | C4.5 | presort |
|---|---|---|---|---|
| epyc | 4.85 s | 1.19 | 4.99 s | 1.55 |
| dell | 8.31 s | 2.09 | 9.50 s | 3.25 |

**Full-depth HIGGS, epyc:**

| | train | partition (in place) | mark | dell mark |
|---|---|---|---|---|
| cartfull | 859 | 10.1 thread-s | 3.25 | 7.4 |
| c45full | 1423 | 10.4 thread-s | 5.1 | 10.1 |

**Presort timeline (HIGGS, epyc):** 40–46 threads are busy until 115 ms, then only 2–4 threads
work for the last 20–30 ms of 148 ms. This is a tail of the last columns. The root sweep also
ends on 1–2 threads.

**chess_100m (2 features), 48 threads:**

| | CART train | presort | build | C4.5 train |
|---|---|---|---|---|
| epyc | 357 | 123 | 234 | 456 |
| dell | 640 | | | 822 |

A stored October chess_1b profile (epyc, C4.5, 5.3 s) shows about 1.9 s of sweeps on only 2–4
threads, 1.2 s of mark, and 1.06 s of presort.

**NUMA (epyc):** at 24 threads, remote memory costs +27% on HIGGS and +54% on MNIST8M against
local memory. Interleaving at 48 threads changes nothing. On dell, remote and local are about the
same. Raw data: `out/numa-*.csv`.

**Compiler flags, dropped:** `-march=x86-64-v3` and `-march=native` (with `-ffp-contract=off`)
make one-thread C4.5 5–7% slower on epyc and 2–5% faster on dell. At 48 threads they are within
noise. Raw data: `out/flags-*.csv`.

## The four experiments

Each is on a local branch in the desktop repo and in `~/opt-dt/patches/<x>/`. Every measured
variant's dumps equal base's.

### pb: per-thread left bitmap. **Best result so far, finish it.**

**Commits:**

* pb1 (`65b9cd66`): each splitting thread marks its big nodes' rows in its own thread_local
  bitmap (`threadLeftBits(rows)`, an `UninitializedAllocator` vector).
  * Marks are plain stores on a single thread for nodes under 2^20 rows
    (`kParallelMarkRows`); above that they are atomic and parallel.
  * `partition()` takes `const uint64_t *leftBits` instead of `bool outOfPlace`.
  * `Columns::outOfPlace` replaces the use of the shared `leftBits`.
  * Touches `cpu_builder.cpp/.h` and `cpu_grower.cpp`.
* pb2 (WIP `97b4acf4`): the biggest nodes are marked from the raw column
  (`raw[winner][row] <= leftValue`) instead of from the winner's sorted entries. K comes from
  env `PB_K` and was tuned in variants pb2k2–pb2k32.
* pb3 (WIP `d3429a33`): in-place (small) nodes are marked in the thread's bitmap too, not in the
  `goesLeft_` bytes. This is the branch tip.

**Results, train time against base** (summ.py medians, 3 reps):

| case | epyc | dell1 |
|---|---|---|
| higgs cart p | pb1 −27%, pb2k4 −27%, pb2k16 −34% | pb1 −19%, pb2k4 −22% |
| higgs c45 p | pb1 −23%, pb2 −22..−25% | pb1 −18%, pb2 −16..−18% |
| susy cart p | pb1 −21%, pb2k2 −36% | pb1 −16%, pb2k16 −19% |
| susy c45 p | pb1 −31%, pb2 −27..−36% | pb1 −17%, pb2 −12..−16% |
| chess_100m cart p | pb1 −10%, pb2k4 −42%, pb2k32 −37% | pb1 −6%, pb2k8 −21%, pb2k2 −1% |
| higgs cart s (1 thread) | pb1 +3.1%, pb2 −2% | pb1 −2.6%, pb2 −1..−3% |
| **full depth** higgs cartfull p | pb2 −14%, **pb3 −32%** | pb2 −16%, **pb3 −21%** |
| higgs c45full p | pb2 −13%, **pb3 −23%** | pb2 −10%, **pb3 −12%** |
| susy cartfull p | pb2 −13%, **pb3 −41%** | pb2 −17%, **pb3 −33%** |
| susy c45full p | pb2 −17%, **pb3 −22%** | pb2 −10%, **pb3 −11%** |
| higgs cartfull s | pb3 +1.7% | pb3 +3.1% (build +3.6%) |
| susy c45full s | pb3 −1.3% | pb3 −0.4% |

Raw files: `out/pb-k-{epyc,dell}.csv` (pb1, pb2k*, `--no-prune -d 10`) and
`out/pb-full-{epyc,dell}.csv` (pb2, pb3).

**To do:**

1. Fix K. It must not stay an env var. K = 4 to 16 is fine everywhere except dell chess, where
   K = 2 is bad. Pick one constant, for example 8, and justify it in a comment.
2. Check pb3's one-thread regression on HIGGS cartfull (+1.7% epyc, +3.1% dell). If it is real,
   keep `goesLeft_` for `--serial` or for small nodes.
3. Clean up the WIP commits into one or two commits. Run `tests/check.sh`.
4. Measure peak RSS with `pbmem.sh` as a job.
5. Do a final A/B against base on epyc and dell1, 5 reps: higgs, susy, chess_100m, mnist8m,
   ecbdl_15m; cart/c45 `-d 10` and full; p and s.

### kf: K-class resumable sweep fused into the out-of-place partition

**What it does:** `KClassSweep` in `cpu_sweep.h` has the same piece-wise interface as
`TwoClassSweep`. With it, multi-class nodes (MNIST8M, covertype) get their children swept during
the partition, like two-class nodes already do. `scanFeatureK` now uses it, and `cpu_sweep.cpp`
shrank.

**Variants:**

* kf0 and kf1: built and measured.
* kf2: the branch tip, built (`bin/tree_cpu_kf2`) but **not measured**. It is kf1 plus skipping
  the sweep of a child whose column is constant (checked from the child's first and last
  entries).

**Results:**

| case | variant | epyc | dell1 | note |
|---|---|---|---|---|
| mnist8m cart p | kf1 | **−20%** train (4268→3412), build −24% | **−18%** (7478→6142), build −22% | 1 rep each |
| covertype p | kf1 | cart −9%, c45 −16% | cart −20%, c45 +3% | noisy, 30–50% spread |
| covertype s | kf0 / kf1 | kf0 cart +3.0%, c45 +0.7%; kf1 cart +1.9%, c45 +3.6% | kf1 cart +2.0%, c45 −1.2% | **one-thread regression** |

**To do:**

1. Find the one-thread covertype regression in `scanFeatureK` through `KClassSweep`. Compare it
   with base's old loop and get it to ≤0.
2. Measure kf2.
3. Do a final A/B: mnist8m cart/c45 p (3 reps) and s (1 rep, ~2 min each), covertype s/p (5 reps).
   Do it on top of pb if pb is kept: both edit `partitionAndSweep`, so expect conflicts.

### wc: presort write-combining. **Recommend drop.**

**Variants:**

* wc1 (`f6bdcfa3`): software write-combining in the radix scatter for columns of ≥1M rows.
* wc1p: wc1, but write-combining only when there is a pool.
* wc2 (the WIP tip): wc1p plus `kSharedSortParts = 4` parts per thread, so idle threads help the
  last columns. That targets the presort tail.

**Results, train time against base** (5 reps, `out/wc-1-{epyc,dell}.csv`):

| case | node | wc1 | wc1p | wc2 |
|---|---|---|---|---|
| higgs cart p | epyc | −2.9% | −5.9% | −4.7% |
| | dell | −1.8% | +2.0% | −0.8% |
| susy cart p | epyc | +2.8% | +8.8% | +6.7% |
| | dell | | −1.3% | −4.1% |
| one thread | epyc | +10% | ~0 | ~0 |

Everything is within noise or mixed, so it is not worth the code. The presort tail (2–4 threads
for the last 20–30 ms) is still open. Another idea: start the root's sweeps per column as soon as
that column is sorted.

### cs: chunked parallel sweep of big two-class nodes with few features. **Unbuilt, untested.**

**Target:** the root and other unfused sweeps run one feature per thread, so with
features < threads (chess: 2, SUSY: 18, HIGGS: 28) most threads idle. Examples: the chess_1b
sweeps on 2–4 threads, and the end of the HIGGS root sweep.

**Design** (WIP commit on `opt/cs`): `CpuTreeBuilder::sweepInChunks()` plus
`TwoClassSweep<Crit, HasGap, InChunks=true>` with `restrict()` and a `Pick` record. Each column is
split into chunks of `kSweepChunk` cuts, and all threads work on the chunks.

* **Phase A:** each chunk is swept as if it started there (best = −inf). It counts its class-1
  entries and takes the count before it from the previous chunk (a chain, as in
  `partitionInBlocks`). It records its picks, whose gains grow by more than `kTieEps` each.
* **Phase B:** each column's chunks are joined in order, keeping the chain's best gain b:
  * If all of a chunk's picks are ≤ b, the chunk cannot beat b by `kTieEps`.
  * Otherwise, let p be the chunk's first pick above b. If gain(p) > b + `kTieEps`, the
    sequential sweep picks p too and from there on both agree, so the chunk's own best stands.
  * Otherwise (b < gain(p) ≤ b + `kTieEps`), sweep the cuts after p again from b, until the sweep
    picks a cut the chunk also picked, or to the chunk's end.

This reproduces the sequential tie rule `gain > best + kTieEps` exactly. C4.5 also needs `tries`
counted once per cut that has a threshold.

**Status:** the agent stopped mid-edit. It was never built, so it may not compile, and a `CS_FEATURES` tuning `#ifndef` is
still in it.

**To do:** build it, run `tests/check.sh` and dumps (ties matter), then A/B chess_100m, susy and
higgs, p and s.

**Expected gain:** SUSY build ~20% and HIGGS ~8% by an earlier estimate. chess is larger.

## Ideas not tried

* **Two threads per column on the top levels when features < threads.** The partition of the
  parent is already split for ≥2^20-row nodes (`partitionInBlocks`), but the fused children's
  sweeps are not. cs covers only the unfused sweeps.
* **A CART node-wide bound across features.** Skip a feature's remaining cuts when its maximum
  possible Gini gain cannot beat the best gain of the other features + `kTieEps`. It must keep
  the per-feature best: C4.5 needs every feature's gain, and CART only needs the winner. The GPU
  has a `tileBound` analogue.
* **A NUMA-aware pool or placement for epyc** (+27–54% remote penalty at 24 threads). This
  overlaps with the `numa-gpu` work below; coordinate before touching `thread_pool.*`.
* **GPU (CUDA hybrid):** not started here, left to the other session.

## The other session: "Hybrid/GPU socket communication analysis"

That session runs on the desktop, in worktree `numa-gpu`, branch `worktree-numa-gpu` (at
`5dcaae59` when this was written, unmerged).

* It changes the CUDA files and `src/core/thread_pool.*`, which is why the experiments here left
  those files alone.
* Its A/B jobs were `ab-p7-ampere` (667) and `ab-p7-hopper` (668).
* It never answered a coordination message.

If you change `thread_pool.*` or any `gpu_*` file, check first with the user whether that branch
was merged.

## Suggested order

1. Finish **pb**: the biggest and broadest win.
2. Rebase **kf** onto pb, fix the one-thread regression, and measure.
3. Get **cs** compiling and measure it.
4. Try the CART bound and the presort tail.
5. Write the final report for the user: a table per experiment (case, node, base, variant, %,
   reps), what was dropped and why, and a keep/drop recommendation for each. Add the kept results
   to `docs/PERFORMANCE.md` on the experiment branch. The user merges.

No cluster jobs of this session were running at handoff (`squeue` empty).

## Cleanup when this work is merged (or dropped)

Delete, after the user decides and nothing of the kit runs any more:

* `~/opt-dt` (binaries, worktree copies, dumps; its `patches/` are the only copy of unmerged work).
* The copies kept for the kit, if the numa-gpu A/Bs are done too: `~/datasets/ecbdl/ecbdl_15m.csv`
  (28 GB), `~/datasets/sorel/sorel_4m.csv` (43 GB), `~/datasets/chess/chess_100m.csv` (2 GB, a prefix
  of `chess_2b.csv`).
* The `opt/*` branches in `~/dt`.
