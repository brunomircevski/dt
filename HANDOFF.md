# Handoff: SOREL `--simple` rerun (2026-11-02 14:10 CET)

Work in progress, for the agent that continues it on another PC. The jobs run
on mordor3 (`ssh mordor3`, user `bmircevski`, checkout `~/dt`); this repository
is the copy the results, charts and docs are committed from. Delete this file
when the work is done.

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
```

A batch log ends in `done` (or `done, failed: …`). Hopper is taken by another
user's 5-day jobs (not ours, not needed here). Nothing else of ours runs.

## Finishing

1. **Sync** (once all three jobs are done):
   `rsync -a mordor3:dt/bench/results/cluster-sorel-rerun bench/results/`
2. **Check** each of the 4 folders: `status` ok, nodes 1,767 (CART) / 379
   (C4.5), depth 10, test accuracy 92.65% / 78.11%, the same `tree_sha256` as
   the other SOREL runs (`--hash-trees`; the serial, cpu and socket SOREL runs have it), and
   `mem_free_node0_kb` before the run ≈ 250 GB on epyc. Look at the result:
   epyc should now be about 0.9–1.0× dell, as `--simple` is on ECBDL, MNIST8M
   and chess 1000M. If epyc is still slower, the explanation above is wrong:
   say so, and investigate (e.g. a run with `numactl --membind=0` vs
   `--membind=1`, `taskset -c 0`, `-d 1`, recording `/proc/<pid>/numa_maps`).
3. **Merge** each node's two folders into the run folder, replacing the old one
   (the chart wants one folder per node with `run-cart.log`/`run-c45.log`):

   ```bash
   cd bench/results
   for m in epyc dell; do
     src=cluster-sorel-rerun/simple-$m-sorel new=cluster-sorel/simple-$m-sorel
     rm -rf $new && mkdir $new
     cat $src-cart/results.jsonl $src-c45/results.jsonl |
       sed -E "s#cluster-sorel-rerun/simple-$m-sorel-(cart|c45)#cluster-sorel/simple-$m-sorel#g" > $new/results.jsonl
     cp $src-c45/machine.json $src-c45/plan.json $new/
     cp $src-cart/run.log $new/run-cart.log && cp $src-c45/run.log $new/run-c45.log
     ../.venv/bin/python ../report.py $new --md $new/report.md --csv $new/summary.csv --runs-csv $new/runs.csv
   done
   rm -rf cluster-sorel-rerun
   ```

   Then on mordor3: the same merge (or just `rm -rf ~/dt/bench/results/cluster-sorel-rerun`
   after the sync), `rm -rf ~/dt-32cf657`; keep `~/dt/tree_cpu_32cf657`.
4. **Redraw:** `bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-sorel`
   (and `--table` for the doc). Drawing locally is fine; never build or
   benchmark locally.
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
   - `docs/PERFORMANCE.md`: a short section on the finding (one-thread runs
     and NUMA: the old harness and the page cache pushed SOREL's values to the
     other socket; `--simple`'s random reads are latency-bound), with the old
     and new numbers.
6. **Commit** the results, chart and docs (explicit paths: peer sessions may
   share the checkout) and delete this file in the same commit. Push only if
   the user asks.

## Rules from the user

- Build, run and benchmark only on mordor3 compute nodes, never locally.
- No CPU benchmarks on hopper or ampere; don't `srun` into busy nodes.
- Charts show means only.
- Commit or push only when asked (this handoff commit was asked for).
