# Handoff: CART / C4.5 performance work (2026-10-28/29)

This note is for the next agent that picks up the optimisation work, on any machine. Nothing here is merged into `main` yet, and **the user decides what gets merged**. Do not merge without their approval.

## The task and its rules

The user's brief: make the C++ CART / C4.5 builder faster or leaner, without changing a single tree.

Rules the user set:
- **Experiments:** unpruned trees, `--no-prune -d 10`.
- **Correctness:** trees must stay identical. Compare every run's `--dump` with the base binary's.
- **Run limits:** keep each run under about 10 minutes. Don't run the largest datasets with `--serial` or `--simple`.
- **`--simple`:** never modify it; it is the reference implementation.
- **Nothing heavy locally, and nothing on the mordor3 login node.** Use cluster compute nodes:
  - CPU work on `dell` or `epyc`.
  - CUDA work on `hopper` or `ampere`, never CPU benchmarks there.
- **Shared cluster:** other users' jobs run there too, so behave well.
- **Branches:** work on a branch; don't touch `main` until the user approves.
- **Commits:** end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Branches (on GitHub)

| Branch | What |
|---|---|
| `exp/perf-d10` | All the experiments, behind environment switches (`src/core/experiment.h`). It also holds the job scripts in `bench/exp/` and the full write-up in **`bench/exp/FINDINGS.md`**. Read FINDINGS.md first. |
| `merge/perf-d10` | The clean merge candidate: the recommended CPU changes with no switches. 2 commits on top of `main` 67495f9 (details below). |

**`exp/perf-d10`**
- Its base is 0c1c21d: main 67495f9 plus main's uncommitted cpu_builder/cpu_grower changes as they were on 2026-10-28.
- How the switches work:
  - `DT_EXP_<NAME>=1` turns one on.
  - `DT_EXP_ALL=1` turns all of them on.
  - `DT_EXP_<NAME>=0` turns one off again.

**`merge/perf-d10`**
- **9731c36:** main's *uncommitted* working-tree changes as of 2026-10-29 08:40, copied unchanged: `src/build/cpu_builder.cpp`, `src/build/cpu_grower.cpp` and `Makefile`.
  - They include **another session's fix for the `--parallel` hang**: `waitNonZero()` spins, then yields, then sleeps, and never uses `atomic::wait`.
  - They also add `-falign-functions=64 -falign-loops=32` to the Makefile.
  - That other session copied the code to `~/dt-final*` on mordor3 and kept working on it. If main's working tree has moved on since, rebase the second commit onto the newer code. It only touches the presort part of `cpu_builder.cpp`.
- **20250b6:** HISTPAR + RADIX as permanent code. The old `radixHistogram()` is removed.
- **Merging:** git refuses to merge while those three files have uncommitted changes on main, even identical ones. The user must first commit or stash their working copy.

## Results (all trees identical, `--parallel` on dell, 48 threads)

| Change | Verdict | Effect |
|---|---|---|
| **HISTPAR**: presort digit histograms of big columns counted in chunks on all threads, not one thread per column | merge (in merge/perf-d10) | presort −40 to −58% on chess, −17 to −26% on SUSY, −15 to −22% on airline (CART / C4.5) |
| **RADIX**: radix sort only over the key bits that vary; integer columns keyed by `int(v) − min`; 12-bit digits; columns of ≥ 1M rows only | merge (in merge/perf-d10) | presort −25% / −19% on ECBDL 15M; together with HISTPAR −37% / −29% on airline |
| **Hang fix**: lost wake-up in libstdc++ 13 `atomic::notify` after a release store (see below) | fixed on main's working tree by the other session, so not needed from exp/perf-d10 (ece66ab) | about 1 in 40 big `--parallel` runs on dell hung forever |
| **NOWRITE**: fused child sweeps at depth max−1 skip writing columns that are never read again | skip | build −5% to +3%, noise |
| **PREFETCH**: partition loops prefetch the left/right flag 32 rows ahead | drop | within ±3% |
| **PACK** (CUDA): upload integer columns as uint8/16 and unpack them in a kernel | **pending** | trees identical (job 751); not timed yet |
| **SORTBITS** (CUDA): cub radix over the varying bits only | **pending** | trees identical (job 751); not timed yet |

End-to-end train time with all CPU switches on:
- CART: chess_100m −27%, chess_1b −20%, airline −14%, chess_10m −13%, ECBDL −8%, SUSY −7%.
- C4.5: chess_1b −16%, chess_10m −9%, ECBDL −5%.
- Build time is unchanged; the whole gain is in presort.

**Root cause of the hang**
- libstdc++ 13 `notify_*` makes the futex wake call only if it sees a waiter in the bucket's waiter count.
- After a `memory_order_release` store (a plain `mov` on x86), the load of that count can run before the store is visible. The waiter then sleeps on the old value for good.
- It was caught under gdb in job 739: the main thread was in `atomic::wait` in `partitionInBlocks`, and all 47 pool workers were idle.
- The waits involved were `lefts[]` in `partitionInBlocks` and `done[]` in the column partition.
- Two possible fixes:
  - A `seq_cst` store, as in ece66ab. Job 749 ran it 160 times on dell with 0 hangs.
  - Avoiding `atomic::wait` altogether, as main's working tree now does.

## Check of merge/perf-d10 against main's code (jobs 775 / 776, 2026-10-29)

`bench/exp/merge.sh` compared the base (9731c36, main's code) with the merge build (20250b6):
- **Nodes:** dell2 for CART, dell3 for C4.5. Median of 3 interleaved reps.
- **Trees:** `tests/check.sh` passes, and all 24 trees are identical.
- **Raw data:** `~/dt-perf/bench/exp/out/exp-merge-{cart-775,c45-776}/ab.csv` on mordor3.

| dataset | presort CART | train CART | presort C4.5 | train C4.5 |
|---|---|---|---|---|
| chess_1b par | −62% | **−32%** | −49% | **−20%** |
| chess_100m par | −52% | −24% | −49% | −19% |
| chess_10m par | −37% | −17% | −42% | −6% |
| airline par | −34% | −7% | −29% | −2% |
| ecbdl_15m par | −25% | −4% | −19% | −6% |
| susy par | −17% | −0% | −15% | −12% |
| covertype par | −9% | −4% | −7% | −7% |
| mnist8m par | −3% | −2% | −8% | +3% (build noise) |
| higgs par | +3% | +1% | +3% | −2% |
| covertype / susy / higgs serial | −2 / −9 / −4% | −0 / −2 / −1% | −2 / −5 / −3% | −0 / −2 / −0% |

- Build time is unchanged within the dell run-to-run noise (±8%; chess C4.5 up to ±18%).
- The higgs +3% presort is noise: 28 columns of 11M rows keep all threads busy without chunks, and RADIX saves no pass on its float columns.

## What is left

1. **CUDA timing (PACK, SORTBITS).**
   - Jobs **721** (hopper) and **722** (ampere) are queued on mordor3 with `--exclusive`. They wait behind another user's open-ended GPU jobs (asawicki, 11+ h so far).
   - They run `bench/exp/cuda.sh` from `~/dt-perf` with the binaries in `~/dt-perf/bench/exp/bin/{base,exp}/tree`. Those are built from the tarballs in `bench/exp/src` (0c1c21d vs 82dafdb; the CUDA code at 82dafdb is the same as the branch head).
   - Results go to `~/dt-perf/bench/exp/out/exp-cuda-{hopper-721,ampere-722}/`:
     - `ab.csv`: base / off / pack / sortbits / all, 3 reps each.
     - `dumps.txt`: tree comparisons.
     - `profile.csv`: `--gpu-profile` steps.
   - Check with `squeue -a`. If the jobs have been cancelled, resubmit them; the command is in the header of `bench/exp/cuda.sh`.
   - **Decision rule:**
     - If PACK and/or SORTBITS make upload + presort clearly faster on both GPUs and the trees are the same, port them to `merge/perf-d10` without switches. The code is in `src/build/gpu_grower.cu`; see `git diff 0c1c21d exp/perf-d10 -- src/build/gpu_grower.cu`. Then rebuild and run `tests/check.sh` (CUDA backends) on hopper.
     - Otherwise leave them out.
   - Update §6 of `bench/exp/FINDINGS.md`.
2. **Report to the user.** Give the CUDA verdict and confirm that `merge/perf-d10` is ready. The user merges.
3. **Clean up mordor3 once 721/722 are done.** Delete `~/dt-perf/bench/exp/src`, `srcm` (219 MB each) and `bin`.

## The cluster (mordor3)

- **Login:** `ssh mordor3`. To avoid quoting problems, send commands as a heredoc: `ssh mordor3 bash -l <<'EOF' ... EOF`. The login node is submit-only.
- **Partitions:**
  - `dell`: dell1–4, Xeon, 48 threads.
  - `epyc`: epyc1–2.
  - `hopper`: H100.
  - `ampere`: A100.
- **Quirks:**
  - Use `--exclusive` for timing.
  - `--mem` fails on hopper/ampere.
  - Slurm accounting is off; list jobs with `squeue -a`.
- **Locations:**
  - Datasets: `~/dt/bench/data/<name>/train.csv`.
  - Server clone of the branches: `~/dt-perf`. Update it from a laptop with `git push mordor3:dt-perf <branch>`, or pull from GitHub there.
- **A/B harness** (`bench/exp/`, on exp/perf-d10):
  - `pack.sh <base> <exp> [folder]` writes git-archive tarballs.
  - A job (`cpu.sh`, `cuda.sh`, `merge.sh`, …) builds both versions in `/tmp`, runs `tests/check.sh`, then runs interleaved reps through `bench/cluster/tune.sh`. Each run has a 600 s timeout and its `--dump` is compared with base.
  - Output goes to `bench/exp/out/<job>-<id>/`.
