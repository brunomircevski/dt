# Handoff: SOREL `--simple` rerun and SOREL GPU sample (2026-11-02, 15:40 CET)

Two pieces of work, both queued on mordor3 and meant to finish overnight
(hopper excepted, see part B):

- **A.** Rerun of the two SOREL `--simple` folders (an anomaly, explained below).
- **B.** New dataset `sorel_4m` (SOREL's first 4M training rows, its own test
  set) so SOREL runs on the GPUs, as `ecbdl_15m` does for ECBDL.

Work in progress, for the agent that continues it on another PC. The jobs run
on mordor3 (`ssh mordor3`, user `bmircevski`, checkout `~/dt`); this repository
is the copy the results, charts and docs are committed from. Delete this file
when the work is done.

# Part A: SOREL `--simple` rerun

## Why

In `bench/results/cluster-sorel/`, `--simple` was the only backend where epyc
was slower than dell (CART 14,124 s vs 12,414 s, C4.5 14,240 s vs 12,712 s;
1.12–1.14×), while on every other dataset epyc's `--simple` is 0.89–1.00× dell
and every other backend 0.35–0.77×. Every other cluster result was checked
(spread, two-mode runs, epyc/dell ratios) and is fine; only these two folders
are rerun.

The cause, most likely (not proven: the old runs recorded no page placement):

- The old `--simple` runs (2026-10-28) ran before 8bdf641, while `run.py`'s
  data check still kept about 135 GiB of pyarrow memory in the harness process
  for the whole job. `results.jsonl`'s `mem_free_node*_kb` (free memory just
  before ./tree started) shows epyc's node 0 at **159 GB** free (every other
  epyc run: ~255–263 GB); dell's node 0 was at 212 GB (normal ~220).
- ./tree runs `taskset -c 0` (node 0). SOREL is 121 GB of values, and the
  loader then kept the 136 GB CSV in the page cache too: node 0 overflowed and
  part of every column probably went to node 1, the other socket on epyc
  (on dell, with SNC2, node 1 is likely the same socket).
- `--simple` sorts row indices by `train.value(row, feature)`: a chain of
  random reads into a 50 MB column, latency-bound, so remote memory hurts it
  most. `--serial` streams presorted columns (prefetch hides latency) and ran
  on 10-30 with the fixed harness: epyc 1.47× faster, as on other data.

Fixes already on main: the harness check runs in a child process (8bdf641,
2026-10-29), and the loader now drops the CSV from the page cache as pass 2
parses it (da1f100), so a one-thread SOREL load needs ~max(136, 121) GB on its
node instead of ~257.

## What is running (submitted 2026-11-02 14:07)

Build: `~/dt-32cf657` (`git archive 32cf657`; its `src/` is the same as
da1f100), `make cpu` in job 13 on epyc1 (done), installed as
`~/dt/tree_cpu_32cf657` (sha256 `eb58ed87…`), given to the runs as `TREE_BIN`,
`TREE_GIT_DESCRIBE=32cf657`. One run per case (runs over 10 minutes run once,
as before), `--test-accuracy --hash-trees`, one thread pinned to CPU 0
(`--cpus 0-47`, as the old runs).

| Job | Node | Plan (`~/dt/bench/cluster/plans/`) | Writes to `~/dt/bench/results/cluster-sorel-rerun/` | Expected end |
|---|---|---|---|---|
| 14 `dt-sorel-simple-epyc1` | epyc1 | `epyc1-sorel-simple.txt` | `simple-epyc-sorel-cart/` | ~19:00–20:00 |
| 15 `dt-sorel-simple-epyc2` | epyc2 | `epyc2-sorel-simple.txt` | `simple-epyc-sorel-c45/` | ~19:00–20:00 |
| 16 `dt-sorel-simple-dell1` | dell1 (`dell-long`) | `dell1-sorel-simple.txt` (CART, then C4.5) | `simple-dell-sorel-cart/`, `simple-dell-sorel-c45/` | ~00:00–01:00 |

Each job: ~50 min of data check (SHA-256 of the 136 GB CSV), ~13 min load,
then 3.5–4 h of training per case. Time limits 10 h (epyc), 20 h (dell1);
per-process timeout 6 h. At the start, free memory on node 0 was 248–250 GiB
on epyc1/epyc2 and 211 GiB on dell1 (as usual there).

Check:

```bash
ssh mordor3 'bash -lc "squeue -a -u bmircevski"'   # always -a and bash -lc
ssh mordor3 'cd ~/dt; cat bench/cluster/plans/*-sorel-simple.log; tail -3 bench/results/cluster-sorel-rerun/*/run.log'
# part B:
ssh mordor3 'cd ~/dt; tail -3 bench/cluster/plans/prepare-sorel_4m.log bench/results/cluster-sorel/*sorel_4m/run.log bench/cluster/plans/epyc2-sorel_4m-simple.log'
```

A batch log ends in `done` (or `done, failed: …`).

# Part B: `sorel_4m`, SOREL on the GPUs (submitted 2026-11-02 15:30)

Asked by the user: a SOREL sample that fits in 80 GB of VRAM, like ECBDL's,
on ampere and hopper, 10 runs each. **4,000,000 rows** × 2381 features = 9.52G
values, about ecbdl_15m's 9.47G (its peak was 73.3 GiB on both GPUs: 8 bytes
per value plus one spare column, a 512 MB ring and ~2 GB more). If a GPU run
fails for memory, lower `train_rows` in `bench/datasets.toml` (3.5M) and
prepare again. Also queued, so the paper's table row has measured numbers for
this work's CPU columns: parallel on epyc1 (10 runs), `--serial` (3 runs) and
`--simple` (1 run) on epyc2. Not run: dell, the libraries (the paper keeps
estimates for those).

Committed for it (4ece22f): `[sorel_4m]` in `bench/datasets.toml`, its case
in `bench/cluster/submit.sh` (results in `cluster-sorel/`), and in
`paper/main.tex` the dataset table's "GPU sample" row and a "4M sample" row in
both time tables, **all estimates (`\est`)** until the runs are in.

Build: `~/dt/tree` and `~/dt/tree_cpu` at 4ece22f (`src/` = da1f100), built in
job 17 on ampere (`bench/cluster/build.sh`, sm_75/80/90; log
`bench/cluster/build.log`). Prepared data: `bench/data/sorel_4m/` by job 18.

| Job | Node | What | Writes to `~/dt/bench/results/` | Expected end |
|---|---|---|---|---|
| 18 `dt-prepare-sorel_4m` | ampere | `bench/prepare.py sorel_4m` (log `bench/cluster/plans/prepare-sorel_4m.log`) | `bench/data/sorel_4m/` | ~16:30 |
| 20 `dt-cuda-ampere-sorel_4m` | ampere, after 18 | CUDA hybrid, 10 runs | `cluster-sorel/cuda-ampere-sorel_4m/` | ~18:30 |
| 23 `dt-gpuonly-ampere-sorel_4m` | ampere | `--gpu-only`, 10 runs | `cluster-sorel/gpuonly-ampere-sorel_4m/` | ~20:30 |
| 19 `dt-cuda-hopper-sorel_4m` | hopper | CUDA hybrid, 10 runs | `cluster-sorel/cuda-hopper-sorel_4m/` | **when hopper is free** |
| 22 `dt-gpuonly-hopper-sorel_4m` | hopper | `--gpu-only`, 10 runs | `cluster-sorel/gpuonly-hopper-sorel_4m/` | **when hopper is free** |
| 21 `dt-cpu-epyc-sorel_4m` | epyc1, after job 14 | parallel, 48 threads, 10 runs | `cluster-sorel/cpu-epyc-sorel_4m/` | ~21:30 |
| 24 `dt-serial-epyc-sorel_4m` | epyc2, after job 15 | `--serial`, 3 runs | `cluster-sorel/serial-epyc-sorel_4m/` | ~21:00 |
| 25 `dt-sorel_4m-simple-epyc2` | epyc2 | `--simple`, 1 run each; plan `bench/cluster/plans/epyc2-sorel_4m-simple.txt` | `cluster-sorel-rerun/simple-epyc-sorel_4m-{cart,c45}/` | ~01:00 |

All with `--test-accuracy` (scored on SOREL's 4,195,042 test rows) and
protocols `cart_depth10,c45_depth10_unpruned`; every job waits for job 18
(`afterok`: if the prepare fails, they never start; `squeue` shows
`DependencyNeverSatisfied`).

**Hopper is busy:** another user (ibalyka) holds all 4 GPUs with jobs 9–12
(5-day limit, started 2026-11-01 ~13:00, so they may run until ~2026-11-06).
Ours (19, 22) wait in the queue with `--exclusive` and start on their own
when hopper frees up. Don't cancel the other user's jobs or `srun` there.
Until then the sorel_4m chart has ampere only; draw it anyway and say hopper
is pending (testset_chart marks queued folders).

# Finishing (both parts)

1. **Sync** (part A once jobs 14–16 are done; part B as its jobs finish):
   `rsync -a mordor3:dt/bench/results/cluster-sorel-rerun mordor3:dt/bench/results/cluster-sorel bench/results/`
   (the second brings the new `*-sorel_4m` folders; the existing SOREL folders
   are the same on both sides).
2. **Check** part A's 4 folders: `status` ok, nodes 1,767 (CART) / 379
   (C4.5), depth 10, test accuracy 92.65% / 78.11%, the same `tree_sha256` as
   the other SOREL runs (`--hash-trees`; the serial, cpu and socket SOREL runs have it), and
   `mem_free_node0_kb` before the run ≈ 250 GB on epyc. Look at the result:
   epyc should now be about 0.9–1.0× dell, as `--simple` is on ECBDL, MNIST8M
   and chess 1000M. If epyc is still slower, the explanation above is wrong:
   say so, and investigate (e.g. a run with `numactl --membind=0` vs
   `--membind=1`, `taskset -c 0`, `-d 1`, recording `/proc/<pid>/numa_maps`).
   Part B: every run `ok`, the same `tree_sha256` for every backend and node
   per protocol (CPU and GPU grow the same tree), VRAM peak (`gpu_peak_bytes`)
   ≈ 73–74 GiB, no run over twice the mean of the others. Write the sorel_4m
   trees' nodes, depth and test accuracy into the doc.
3. **Merge** each node's two folders into the run folder, replacing the old one
   (the chart wants one folder per node with `run-cart.log`/`run-c45.log`);
   the same for part B's `simple-epyc-sorel_4m` (new, nothing to replace):

   ```bash
   cd bench/results
   for run in simple-epyc-sorel simple-dell-sorel simple-epyc-sorel_4m; do
     src=cluster-sorel-rerun/$run new=cluster-sorel/$run
     rm -rf $new && mkdir $new
     cat $src-cart/results.jsonl $src-c45/results.jsonl |
       sed -E "s#cluster-sorel-rerun/$run-(cart|c45)#cluster-sorel/$run#g" > $new/results.jsonl
     cp $src-c45/machine.json $src-c45/plan.json $new/
     cp $src-cart/run.log $new/run-cart.log && cp $src-c45/run.log $new/run-c45.log
     ../.venv/bin/python ../report.py $new --md $new/report.md --csv $new/summary.csv --runs-csv $new/runs.csv
   done
   rm -rf cluster-sorel-rerun
   ```

   Then on mordor3: `rm -rf ~/dt/bench/results/cluster-sorel-rerun` after the
   sync, `rm -rf ~/dt-32cf657`; keep `~/dt/tree_cpu_32cf657`. `submit.sh` made
   empty folders for queued runs: an empty one left in the sync is a run that
   has not started (hopper).
4. **Redraw:** `bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-sorel`
   (and `--table` for the doc). With two datasets in the folder it writes
   `chart-sorel.png` and `chart-sorel_4m.png` (as cluster-ecbdl): delete the
   old `chart.png` and point BENCHMARK.md at the two. Drawing locally is fine;
   never build or benchmark locally.
5. **Docs** (new numbers, no "rerun"/"queued" wording left):
   - `bench/results/cluster-sorel/BENCHMARK.md`: the **Builds** bullet
     (`--simple` now 32cf657 on 2026-11-02, the rest ec77727), the **Simple**
     results bullet (drop "epyc is 1.1× slower here, unlike every other run";
     give the new times and the ratio to `--serial`), the results table, and a
     short note on why the old `--simple` numbers were off (harness memory on
     node 0 → data on the other socket; the loader fix).
   - `bench/results/README.md`: the cluster-sorel row (`--simple` range) and
     the "except `--simple` where it was not rerun" sentence.
   - `bench/cluster/README.md` "Which build ran": mention the SOREL `--simple`
     rerun on 32cf657.
   - Part B in `cluster-sorel/BENCHMARK.md`: the sample (first 4M training
     rows, why 4M: VRAM), a GPU paragraph and table (as cluster-ecbdl's 15M
     part), and the GPUs' speed-up over epyc; the "No GPU runs: the data does
     not fit" sentence becomes "the GPUs get sorel_4m". Also
     `bench/cluster/README.md` (test-set table, "Not run" sentence) and
     `bench/results/README.md` (cluster-sorel row).
   - `paper/main.tex`: replace the `\est` cells of both "4M sample" rows in
     the time tables with the measured means and min–max (\ImplSerialBase =
     --simple, \ImplSerialOpt = --serial, \ImplMulticore = epyc parallel,
     \ImplGpu = gpuonly-hopper, \ImplHybrid = cuda-hopper; until hopper has
     run, keep the estimates for the GPU cells or use ampere and say so in the
     caption); also part A's new `--simple` numbers in the SOREL-20M rows
     (14,100 / 14,200 now). The library cells stay estimates.
   - `docs/PERFORMANCE.md`: a short section on the finding (one-thread runs
     and NUMA: the old harness and the page cache pushed SOREL's values to the
     other socket; `--simple`'s random reads are latency-bound), with the old
     and new numbers.
6. **Commit** the results, chart and docs (explicit paths: peer sessions may
   share the checkout, and another session edits `paper/main.tex`). Delete
   this file once hopper's runs are in too (until then keep part B's hopper
   rows). Push only if the user asks.

## Rules from the user

- Build, run and benchmark only on mordor3 compute nodes, never locally.
- No CPU benchmarks on hopper or ampere; don't `srun` into busy nodes.
- Charts show means only.
- Commit or push only when asked (this handoff commit was asked for).
