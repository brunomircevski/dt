# Handoff (updated 2026-11-05 16:30)

Two independent pieces of work are open, both waiting for hopper. Both need mordor3 (`ssh mordor3`,
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

## Part A: `sorel_4m` on hopper

Only this is left of the SOREL work of 2026-11-02. Done and pushed (75793c0,
b140fb0): the `--simple` rerun of SOREL (epyc now 0.94–0.97× dell, explained
in `docs/PERFORMANCE.md`), and `sorel_4m` on ampere and epyc, with the charts,
`bench/results/cluster-sorel/BENCHMARK.md`, the READMEs and the paper's CPU
cells of the two "4M sample" rows. Delete this part when hopper's runs are in
(or the user drops them).

The jobs run on mordor3 (`ssh mordor3`, user `bmircevski`, checkout `~/dt`).
If SSH times out, the PC is off the network mordor3 accepts: ask the user to
connect (VPN or campus), don't keep retrying.

### Waiting

| Job | Node | What | Writes to `~/dt/bench/results/` |
|---|---|---|---|
| 19 `dt-cuda-hopper-sorel_4m` | hopper | CUDA hybrid, 10 runs | `cluster-sorel/cuda-hopper-sorel_4m/` |
| 22 `dt-gpuonly-hopper-sorel_4m` | hopper | `--gpu-only`, 10 runs | `cluster-sorel/gpuonly-hopper-sorel_4m/` |

Both submitted 2026-11-02 with `--exclusive`; they start on their own. On
2026-11-05 09:30 another user (ibalyka) still held hopper with jobs 11 and 12
(9 and 10 had ended; 5-day limit from 2026-11-01 ~13:00). SLURM's estimate
for job 19: start 2026-11-06 14:26. Still so on 2026-11-05 16:16: 19 pending
(Resources), 22 and 60 behind it, ibalyka's 11 and 12 running for 4 days 2 h. Job 60 (`dt-hvg-h`, another session's
hybrid-vs-GPU-only benchmark, see its memory/BENCHMARK) also waits for hopper.
Don't cancel the other user's jobs or `srun` there.

```bash
ssh mordor3 'bash -lc "squeue -a -u bmircevski; squeue -a -w hopper"'
ssh mordor3 'cd ~/dt; tail -n 3 bench/results/cluster-sorel/*hopper-sorel_4m/run.log'
```

The two empty `*-hopper-sorel_4m` folders (made by `submit.sh`, also in the
local checkout, untracked) are what makes `testset_chart.py` draw "queued"
bars: leave them until the runs land.

### Decide with the user before they start: old build, GPU 0

Jobs 19 and 22 will run what was current on 2026-11-02, not today's main:

- **Binary:** `~/dt/tree` built 2026-11-02 at 4ece22f (`src/` = da1f100), the
  same build as the ampere and epyc `sorel_4m` runs. It predates 50d29cd
  (2026-11-04: the arena toucher now starts after C4.5's distinct download),
  which made SOREL 4M C4.5 hybrid on the A100 6.38 → 5.33 s (job 61); the
  committed ampere hybrid C4.5 number (6.57 s) is from the old build.
- **GPU and socket:** mordor3's `~/dt` checkout is still at ad7e06a, whose
  `job.sh` uses GPU 0 and every CPU. Main's docs (50d29cd) now say hopper runs
  on GPU 2, NUMA node 1, CPUs 16–31, because another user's /dev/shm fills
  node 0. So these runs would differ from how the other hopper results are
  now described.
- Don't `git pull` in mordor3's `~/dt` while they are queued unless that is
  the choice: the jobs read `run.py` and the binaries from there.

The user said (2026-11-03) all benchmarks will be redesigned, so ask:

1. **Let them run as they are** (consistent with the committed ampere/epyc
   `sorel_4m` runs). Then say in BENCHMARK.md that hopper ran on GPU 0 with
   the 4ece22f build.
2. **Cancel 19 and 22 and resubmit on current main** (pull and rebuild
   `~/dt` on mordor3 with `bench/cluster/build.sh` on ampere, then
   `bench/cluster/submit.sh -m 10 -e -d sorel_4m -p cart_depth10,c45_depth10_unpruned cuda-hopper-sorel_4m gpuonly-hopper-sorel_4m`;
   check `submit.sh`'s header for the current options). For consistency,
   ampere's two `sorel_4m` runs should then be rerun too.
3. **Drop them** and leave the paper's GPU cells as estimates until the
   redesign.

### Finishing (once the runs are in)

1. `rsync -a mordor3:dt/bench/results/cluster-sorel bench/results/`
2. Check: every run `ok`, nodes 1,619 (CART) / 299 (C4.5), depth 10, test
   accuracy 92.57% / 78.30% (as ampere and epyc), VRAM peak ≈ 76 GiB, no run
   over twice the mean of the others, `machine.json` names the build.
3. `bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-sorel`
   (and `--table`). Make sure no empty `cluster-*-fix`/`-par2` folders exist
   locally: the script reads them as pending reruns ("old build; rerun
   queued").
4. Docs: in `bench/results/cluster-sorel/BENCHMARK.md` the two hopper rows of
   the sorel_4m table ("queued" now), the "4M sample on the GPUs" bullet (H100
   speed-up over epyc's 6.43 s / 7.85 s), the "Hopper's runs are queued"
   sentence in **Runs** and the build in **Builds**;
   `bench/results/README.md` (cluster-sorel row, "hopper queued").
   `paper/main.tex`: the `\est` GPU cells of both "4M sample" rows of "This
   work" (lines ~177 and ~210; `\ImplGpu` = gpuonly-hopper, `\ImplHybrid` =
   cuda-hopper; mean and min–max of `train_seconds` from `runs.csv`, as the
   other cells), then rebuild the PDF (`latexmk -pdf main.tex` in `paper/`).
   Other sessions edit `paper/main.tex` too: pull first, change only those
   cells. The library rows' estimates stay.
5. Commit with explicit paths (other sessions share the checkout: e.g.
   `bench/power/` had uncommitted changes of another session on 2026-11-05)
   and delete this part. Push only if the user asks.

## Part B: hybrid-vs-gpu-only on hopper (job 60)

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
| State on 2026-11-05 | pending. ibalyka's jobs 11 and 12 hold hopper until 2026-11-06 14:26 at the latest (5-day limit). Jobs 19 and 22 (Part A) are ahead of it. Expect 3–5 h of run time |
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
