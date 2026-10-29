# Handoff: cluster-dataset benchmark reruns (2026-10-29 09:00 CET)

Work in progress, for the agent that continues it on another machine. The
benchmarks run on mordor3 (`ssh mordor3`, user `bmircevski`, checkout `~/dt`);
this repository is the copy the results, charts and docs are committed from.
Delete this file when the work is done.

## What was asked

Rerun the `bench/results/cluster-<dataset>/` benchmarks (airline, higgs, susy,
ecbdl, mnist8m, sorel, chess; not cpu-scaling, profiling or the vs-others
folders):

- **GPU:** every CUDA run again on ./tree at 67495f9, the CUDA rework.
- **CPU:** dell and epyc for every backend.
- **Runs per case:** the slow ones (over about 10 minutes) once, on one node, so
  the other node stays free; the faster ones at least 3 times (about 2 minutes
  or less: 10). Existing runs of the same build count.
- **ECBDL:** `ecbdl_15m` on every backend; the full set on the CPU only (it does
  not fit in VRAM).
- **SOREL:** CPU only.
- **Charts:** bars grouped by backend (--simple, --serial, parallel on one
  socket, parallel on both sockets, CUDA hybrid, CUDA GPU only), the machines
  under each. Each bar shows its **mean** (never medians), sd and number of
  runs. Problems are marked on the chart, but the goal is **no problems in the
  end**. Tall charts go side by side.

## Where it stands

**Done, results here (rsynced 09:00):**
- every GPU run (hopper, ampere; CUDA hybrid and GPU only) on 67495f9;
- every epyc run: --simple, --serial, and parallel on one socket and on both;
- every dell --serial run;
- dell parallel up to `cpu-dell-mnist8m`;
- dell2's 512 GB chess runs.

**Still running on mordor3** (plans and logs in `bench/cluster/plans/`):

| Job | Node | What | Writes to | Expected |
|---|---|---|---|---|
| 771 `dt-batch-dell1-par2` | dell1 | rest of `dell1-par2.txt`: socket-dell-mnist8m, cpu/socket-dell-ecbdl (3 runs), cpu/socket-dell-sorel (1 run) | `cluster-<x>-par2/` | until about 10:30 |
| 644 `dt-batch-dell1-simple` | dell1, after 771 | `dell1-simple.txt`: --simple on dell, 9 cases, incl. full ECBDL once (epyc took 5,857 s + 6,890 s) | `cluster-<x>-fix/` | many hours |
| 645 `dt-batch-dell1-simple-ecbdl` | dell1, after 644 | `dell1-simple-ecbdl.txt` | `cluster-ecbdl-fix/` | hours |

Not ours, do not touch: 721/722 `exp-cuda-*`, 775/776 `exp-merge-*` (another
session's experiments).

Check with `ssh mordor3 'bash -lc "squeue -a -u bmircevski"'` (always `-a`:
without it squeue hides jobs; always `bash -lc` for the SLURM environment).
Wait in a background loop, e.g.
`while r=$(ssh mordor3 'bash -lc "squeue -a -h -j 771,644,645"'); [ -n "$r" ]; do sleep 300; done`.

## Builds on mordor3

| Binary | What | Used by |
|---|---|---|
| `~/dt/tree` | 67495f9 with CUDA (job 636 checked: 24 tree pairs identical to the old build) | CUDA hybrid, GPU only |
| `~/dt/tree_cpu` | the "final" CPU build of 2026-10-28: spin fix (2,048 pauses, then `atomic::wait`) + two-pass `distinctValues` | --serial, --simple (these never wait, so unaffected) |
| `~/dt/tree_cpu3` | **the source committed with this file**: `waitNonZero()` in `src/build/cpu_builder.cpp` + `-falign-functions=64 -falign-loops=32` (Makefile); sha256 f33f9547… | every parallel run in `-par2/` (via `TREE_BIN`, git describe `67495f9-dirty-poll` in machine.json) |
| `~/dt/tree_cpu2` | dropped yield + seq_cst build | nothing (delete) |

After the jobs finish you may rename `tree_cpu3` → `tree_cpu` (and rebuild
./tree from the committed source on a GPU node if needed: the CUDA path doesn't
use the changed waits). `~/dt` on mordor3 is a git checkout at 8187eb9 with
these changes copied in by scp. Don't `git pull` there while batches run:
`batch.sh` and `job.sh` are read during the runs.

## Finishing (once 771, 644, 645 are done)

1. **Sync:** `for x in airline higgs susy ecbdl mnist8m sorel chess; do rsync -a mordor3:dt/bench/results/cluster-$x-fix mordor3:dt/bench/results/cluster-$x-par2 bench/results/; done`
2. **Check:** `bench/.venv/bin/python bench/cluster/outliers.py bench/results par2`
   (and `fix`). There should be no `OUTLIER` and no failed run. Check that the
   batch logs end in `done` with no `failed:`, and that SOREL has an ok run on
   cpu/socket dell. A run that failed or is an outlier gets rerun with
   `bench/cluster/submit.sh -P` plus `batch.sh` (see `submit-poll-2026-10-29.sh`
   for an example: plan lines, `TREE_BIN`, `TREE_GIT_DESCRIBE`).
3. **Swap:** in `bench/results`, on the server and locally, run the loop in
   `bench/cluster/README.md`. Every `-fix`, then `-par2`, folder replaces the run
   folder of the same name (no merging). Then delete the `-fix`/`-par2` dirs and
   mordor3's `~/dt/bench/results-old-build/` (old builds, 5.6 MB; not needed
   after the swap).
4. **Redraw:** `bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-$x`
   for each dataset (local drawing is fine; never build or benchmark locally).
   No "queued" or ⚠ should be left. `problems.json` must stay `{}` everywhere
   except sorel's `socket-epyc-sorel` "not run" note (one epyc socket's 256 GB
   is too little).
5. **Docs:** rewrite each `bench/results/cluster-<x>/BENCHMARK.md`:
   - Header bullets: condense the history ("reruns queued", "spin stall",
     "one build", "batches") into a short note on which builds ran what.
   - Results section: dates, builds and run counts; tables from
     `testset_chart.py --table` (means); update the prose numbers.
   - Remove all "queued"/"still running" wording.
   - `cluster-chess/BENCHMARK.md` still has median tables and a dell-variance
     table to redo with means.
   - Also update `bench/results/README.md` (index rows) and the
     `bench/cluster/README.md` paragraph on the reruns.
6. **Commit** and push (the user asked for this handoff commit; ask again
   before the next push).

## Findings to write into the docs

- **Spin stall (epyc):** waiting by unbounded `_mm_pause` in `partitionInBlocks`
  stalled when the previous block's thread was descheduled. chess_10m: 13/50
  runs over 2× the median.
- **Spin fix, then `atomic::wait`:** dell chess about 2× slower in some node
  states (dell2 chess_100m 2.8 s vs 1.25 s).
- **libstdc++ 13 lost wake-up:** `notify` skips the futex wake when it sees no
  waiter, and that check can pass the release store. About 1 in 40 big dell
  runs hung; this explains the hang of job 747 (dt-spin-var). The other
  session's fix was seq_cst stores (ece66ab, branch exp/perf-d10).
- **Final wait (`waitNonZero`):** 2,048 pauses, then `yield` for up to 2 ms,
  then 50 µs sleeps; release stores, no notify, so no lost wake-up is possible.
- **Code alignment, not seq_cst:** the 8% slowdown of epyc's ECBDL 15M C4.5
  (7.80 → 8.41 s) came from C4.5's sweep (`TwoClassSweep::advance`) moving to
  another offset mod 64. `-falign-functions=64 -falign-loops=32` restores it
  (A/B jobs 763, 764, 766; `~/dt-final3,4,5` on mordor3).
- **Final vs old, epyc:** ECBDL C4.5 1.02–1.03× (15M 7.82 → 7.99 s, full
  17.7 → 18.1 s), CART 0.99×, mnist8m C4.5 0.95×, chess_100m 0.93–0.95×.
- **Final vs old, dell:** chess 100M/1B about 2× faster (chess_1b CART
  23.6 → 10.8 s); elsewhere up to 14% faster (susy CART) or within 3%.
- **Same trees everywhere:** E and G dumps identical to the old build (20 pairs
  plus ECBDL 15M and chess_10m).
- **dell one-socket chess 1M/10M looked 15–50% slower** in the reruns, but
  interleaved old/new on dell2 (job 774, `~/dt-final5/sock.sh`) gives equal or
  faster: chess_1m CART 0.061 → 0.060 s, chess_10m 0.178 → 0.163 s. The old
  fast numbers were dell1's node state at that time (presort, which has no
  waits, was slower by the same factor). Say so in the chess doc.

## Rules from the user

- Build, run and benchmark only on mordor3 compute nodes, never locally.
- No CPU benchmarks on hopper or ampere.
- Don't `srun` into busy nodes.
- Charts show means only.
- Commit or push only when asked; a push needs explicit permission each time.
- Scripts that submitted the jobs: `bench/cluster/submit-*-2026-10-2{8,9}.sh`,
  each with an as-run header; `submit-poll-2026-10-29.sh` is the current one.
