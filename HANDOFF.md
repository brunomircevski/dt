# Handoff (updated 2026-11-06 10:40)

One piece of work is open: job 60 on hopper. It needs mordor3 (`ssh mordor3`,
user `bmircevski`). If SSH times out, the PC is off the network mordor3
accepts: ask the user to connect (VPN or campus), don't keep retrying.

mordor3 basics: the login node is for submitting only. SLURM is on `PATH` only
in a login shell, so wrap remote commands in `bash -lc "..."`. The partitions
are hidden, so use `squeue -a` and `sinfo -a`. SLURM's default is 1 CPU per
job, so pass `-c` or `--exclusive` (in an `srun`, `--exclusive` alone still
gives 1 CPU). `/home` is NFS, shared by all nodes. ssh into compute nodes
fails, so use `srun`. Python for the analysis: `~/dt/bench/.venv/bin/python`,
run on a compute node (e.g. `srun -p epyc -w epyc2 -c 4 -t 0:10:00 ...`).
Never build, run or analyse locally (user's rule).

## hybrid-vs-gpu-only on hopper (job 60)

### Background (done, pushed)

The user asked for a benchmark of ./tree's CUDA **hybrid** (`--cuda`: nodes
below `--gpu-min-rows` go to the CPU threads) against **GPU only**
(`--cuda --gpu-only`). It covers CART and C4.5 without pruning, depth 5, 10,
15, 20, 25, 30 and no limit, 3 runs a case. The datasets are airline (biggest
gap) and MNIST8M (smallest gap). The user set the name "hybrid-vs-gpu-only"
and wants **linear y-axes** on its charts. Each node first gets a sweep of the
hybrid's `--gpu-min-rows`, and the depth runs use the fastest value.

- **ampere is done and pushed.** The depth runs are in c992b84 (best
  `--gpu-min-rows` 512 for airline, 8192 for MNIST8M). The profiled MNIST8M
  runs and the "who is working" charts are in c63fbac1. All of it is in
  `bench/results/hybrid-vs-gpu-only/` with `BENCHMARK.md`, and it has a row in
  `bench/results/README.md`.
- **The scripts:**
  - `bench/cluster/hybrid_vs_gpu_only.sh <node>`: one SLURM job that runs the
    sweep, picks the value, then runs the depth runs for both modes;
  - `bench/hybrid_chart.py` (`--node`): the depth chart;
  - `bench/hybrid_profile_chart.py`: the profile charts.

### The waiting job

| | |
|---|---|
| Job | 60 `dt-hvg-hopper`, hopper, `--exclusive`, 8 h limit, submitted 2026-11-03 |
| State on 2026-11-06 | running since 10:05 (started when jobs 19 and 22 ended), in the airline `--gpu-min-rows` sweep at 10:12. Expect 3–5 h of run time |
| Checkout | `~/dt-hvg` on mordor3: a clone at c992b84. Its binaries were built on 2026-11-03 from fef9ac0's `src/`, for sm_80 and sm_90. That predates 50d29cd's C4.5 setup change, but setup is the same in both modes on these datasets, so it doesn't matter here. `bench/data` and `.venv` are as in `~/dt`. The untracked `profile/` folder there holds ampere's profile runs, which are already committed |
| Script | the c992b84 version of `hybrid_vs_gpu_only.sh`, without main's `PROFILE_DATASETS` stage, so **hopper gets no profiled runs**. Don't update `~/dt-hvg` while job 60 is queued: the job reads its script, `run.py` and the binaries from there |
| Output | `~/dt-hvg/bench/results/hybrid-vs-gpu-only/`: `hopper.log` (the driver's log, ends with `done` or `done, failed: ...`); `sweep/cuda-hopper-<dataset>-mr<N>/` and `sweep/hopper-<dataset>.json` (the value taken); `cuda-hopper-<dataset>/` and `gpuonly-hopper-<dataset>/` (the depth runs, each with `run.log`) |

job.sh takes the GPU on the NUMA node with the most free memory. On hopper
that should be GPU 2, NUMA node 1, CPUs 16–31, so 16 threads; check
`run.log` and `machine.json` (`gpu_index`).

```bash
ssh mordor3 'bash -lc "squeue -a -p hopper"; tail -n 20 ~/dt-hvg/bench/results/hybrid-vs-gpu-only/hopper.log'
```

### Finishing (once job 60 is done)

1. **Check the run.**
   - `hopper.log` ends with `done` and has no `failed`.
   - Every `results.jsonl` has 42 `"kind": "run"` rows, all `"status": "ok"`
     (14 protocols × 3 runs).
   - Hybrid and GPU only have the same `nodes` in every case. These should
     also be ampere's node counts (BENCHMARK.md's tables, chart labels).
   - In `sweep/hopper-<dataset>.json`, the chosen `min_rows` is not at the
     edge of the values tried (256 or 32768). If it is, ask the user about
     extending the sweep. On ampere, airline needed 128 to confirm 512.
2. **Copy only hopper's files to the PC.** The ampere ones are already
   committed.
   ```bash
   d=bench/results/hybrid-vs-gpu-only
   rsync -a mordor3:dt-hvg/$d/hopper.log mordor3:dt-hvg/$d/cuda-hopper-airline mordor3:dt-hvg/$d/gpuonly-hopper-airline \
     mordor3:dt-hvg/$d/cuda-hopper-mnist8m mordor3:dt-hvg/$d/gpuonly-hopper-mnist8m $d/
   rsync -a 'mordor3:dt-hvg/'$d'/sweep/*hopper*' $d/sweep/
   ```
3. **Draw the chart.** It is light work, and earlier charts of this benchmark
   were drawn with the local `bench/.venv`. If the new PC has no venv, draw it
   on a compute node as the header says.
   ```bash
   bench/.venv/bin/python bench/hybrid_chart.py bench/results/hybrid-vs-gpu-only --node hopper --datasets airline,mnist8m
   ```
   This writes `chart-hopper.png` with linear axes. Look at it before
   reporting.
4. **Update the docs** in `bench/results/hybrid-vs-gpu-only/BENCHMARK.md`:
   - Replace the line "hopper: queued ..." under the ampere chart with
     `![hopper](chart-hopper.png)`.
   - Add a **Results (hopper)** table like ampere's (hybrid / GPU only, ratio)
     and hopper's setup to **Where**: H100 PCIe, GPU index and threads from
     `machine.json`, build fef9ac0 `src/` (`tree_git` c992b84).
   - Add hopper's sweep tables, with times from `sweep/hopper-*.json`.
   - Compare with ampere, for example whether the H100 widens the gap.

   In `bench/results/README.md`, change the row's "hopper queued" to the
   result.
5. **Commit with explicit paths:** `bench/results/hybrid-vs-gpu-only`,
   `bench/results/README.md`, `HANDOFF.md`. Never use `git add -A`: other
   sessions leave uncommitted files (on 2026-11-05: `bench/cluster/README.md`,
   `bench/cluster/submit.sh`, `bench/results/cluster-chess/BENCHMARK.md`,
   `src/build/gpu_grower.cu`). Ask the user before pushing.
6. **Clean up on mordor3** after the commit:
   `rm -rf ~/dt-hvg ~/dt-hvg-profile.sh`. Then delete this part.

Possible follow-up, only if the user asks: the profiled MNIST8M runs on
hopper. Wait until job 60 is done, then copy `~/dt-hvg-profile.sh`, which is
the ampere version: set `-p hopper -w hopper`, `*-hopper-mnist8m`, and
`--gpu-min-rows` to hopper's `sweep/hopper-mnist8m.json` value. Draw the
charts with `bench/hybrid_profile_chart.py ... --node hopper --dataset mnist8m`
(see that BENCHMARK.md's Commands).

## Rules from the user

- Build, run and benchmark only on mordor3 compute nodes, never locally.
- No CPU benchmarks on hopper or ampere (the one exception, power job 99, is
  done); don't `srun` into busy nodes.
- Charts show means only.
- Commit or push only when asked.
- One `main` branch on GitHub; every checkout up to date at the end of a task.
