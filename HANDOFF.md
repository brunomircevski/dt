# Handoff (updated 2026-11-05)

Three independent pieces of work are open. All three need mordor3 (`ssh mordor3`,
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

## Part 1: power benchmark, all five implementations on ampere (job 99)

### Background (done, pushed)

The user asked for the time and energy of the five implementations:

| Paper name | Option |
|---|---|
| Serial-Simple | `--simple` |
| Serial-Opt | `--serial` |
| Parallel (renamed "Multicore" in the paper, a0e53395) | `--parallel` |
| GPU | `--cuda --gpu-only` |
| Hybrid | `--cuda` |

What is done and where:
- **The first round** is pushed in f941d7d9 and written up in
  `bench/power/results/BENCHMARK.md`: ECBDL'14 15M, depth ≤ 10, no pruning,
  CART and C4.5. The CPU implementations ran on dell4 and epyc1, measured as
  whole-server power at the PSU. GPU and Hybrid ran on ampere, measured as CPU
  packages through RAPL plus every GPU through NVML.
- **The tooling:**
  - `bench/power/pwrlog.py` samples every 0.1 s;
  - `bench/power/power_bench.sh` is the SLURM job;
  - `bench/power/analyze.py` makes the tables and charts;
  - `./tree --repeat N` trains N trees on data loaded once and prints each
    tree's Unix start and end.
- **The sensor survey** is `bench/power/README.md`. Without root, only ampere
  lets users read CPU power: `perf stat -e power/energy-pkg/` works because
  `perf_event_paranoid` is 0 there. dell, epyc, hopper and turing are at 4,
  and their RAPL files are root-only. dell and epyc have a user-readable PSU
  meter; ampere has none. `bench/power/ADMIN-REQUEST.md` is the request to
  the admins for root-free CPU and PSU access (sent or not: ask the user).

The user then wanted one chart with **energy above idle split into CPU and
GPU, time, and max power** for every implementation. Only ampere can give the
CPU/GPU split, so the user allowed **an exception to the rule "no CPU
benchmarks on ampere" for this one power job** (2026-11-05). The exception
covers only this job: don't run other CPU work on ampere.

### The running job

| | |
|---|---|
| Job | 99 `pwr-all`, ampere, `--exclusive -c 48`, 9 h limit, started 2026-11-05 09:20 |
| Expected end | about 13:30 (3.5–5 h). Serial-Simple runs 1–1.3 h per tree on ampere's Zen 2 (an estimate) |
| Checkout | `~/dt-power`: a clone of GitHub main at f941d7d9, plus this commit's `bench/power` changes, copied there before the commit (so it shows as dirty). `bench/data` and `bench/.venv` link to `~/dt`'s. Binaries were built on ampere on 2026-11-05 with `bench/cluster/build.sh` |
| Submitted with | `~/dt-power/submit-power-ampere.sh`: `power_bench.sh all bench/results/power-ampere/ampere` with `TARGET=30 REPS=5 REPS_BY="simple=1 serial=3"` |
| Output | `~/dt-power/bench/results/power-ampere/ampere/`: `machine.txt`, `markers.txt`, `samples.csv`, `perf_rapl.csv`, `out/*.txt`, `job.log` |

Protocol:
1. Warm-ups of Parallel, GPU and Hybrid, two trees each. They pick K, the
   number of trees per process, so that each process trains for at least
   30 s.
2. 60 s idle.
3. Rounds of the cases: Serial-Simple runs once, Serial-Opt 3 times, and
   Parallel, GPU and Hybrid 5 times each, with 20 s of idle before each
   process.
4. 60 s idle at the end.

On ampere, analyze.py measures only the first tree of each process. That is
the tree the time benchmarks time. The later trees, on the GPU, are slower
because the data is uploaded again, and they appear in their own table. The
warm-ups gave second-tree times of:

| Implementation | CART | C4.5 |
|---|---|---|
| Parallel | 15.5 s | 27.9 s |
| GPU | 7.2 s | 7.2 s |
| Hybrid | 6.9 s | 7.0 s |

```bash
ssh mordor3 'bash -lc "squeue -a -u bmircevski -p ampere"; cut -c19- ~/dt-power/bench/results/power-ampere/ampere/markers.txt | tail -5'
```

The job ends with the marker `job_end`. If it was killed instead (time limit,
node failure), the folder lacks `idle2_*`. analyze.py still works with the
first idle window alone, and it skips runs that have no `end` marker.

### Finishing (once job 99 is done)

1. **Check the run.**
   - `machine.txt` says `cpus available 48`.
   - Every `end ... rc=0` in `markers.txt`.
   - Every `out/rep*.txt` shows the same tree for its algorithm: the same
     `nodes`, depth 10.
   - Parallel is much faster than Serial-Opt. The aborted test had Parallel
     on 1 CPU, which made it no faster than Serial-Opt.
2. **Run the analysis on a compute node:**
   ```bash
   ssh mordor3 'cd ~/dt-power && bash -lc "srun -p epyc -w epyc2 -c 4 -t 0:20:00 bench/.venv/bin/python bench/power/analyze.py bench/results/power-ampere"'
   ```
   It writes `summary.md`, `summary.csv`, `runs.csv`, the six usual charts
   and the new **`chart-cpu-gpu-ampere.png`**. That chart has rows for CART
   and C4.5 and three columns:
   - energy above idle per tree, stacked CPU (sockets 0 + 1) and GPU (all
     four), log scale, labelled with the total and the GPU share;
   - time per tree, log scale;
   - peak power of CPU + GPUs, with the idle level shown lighter.

   This is the chart the user asked for: look at it before reporting. CPU
   runs show a few joules of GPU "above idle", which is noise in the idle
   GPUs.
3. **Copy the results to the PC** into a new folder, and gzip the large raw
   files (analyze.py reads `.gz`):
   ```bash
   mkdir -p bench/power/results-ampere
   rsync -a mordor3:dt-power/bench/results/power-ampere/ bench/power/results-ampere/
   gzip -9 bench/power/results-ampere/ampere/samples.csv bench/power/results-ampere/ampere/perf_rapl.csv
   ```
4. **Write `bench/power/results-ampere/BENCHMARK.md`** in the style of
   `bench/power/results/BENCHMARK.md` (protocol, sensors, tables from
   `summary.md`, each chart described, conclusions, limitations, how to
   reproduce). Points to make:
   - All five implementations share one node, one set of idle readings and
     the same sensors, so this is the fair CPU-vs-GPU energy comparison.
   - It measures CPU packages + GPUs only: no RAM, fans, board or PSU
     losses.
   - Compare with dell and epyc (`results/`): ampere's Zen 2 CPUs are older
     and slower.
   - Hybrid's extra CPU energy against GPU-only.
   - The three unused GPUs' share of the total energy.

   Link the new folder from `bench/power/README.md`, which lists the results,
   and from the top of `results/BENCHMARK.md`.
5. **Commit** with explicit paths: `bench/power/results-ampere`,
   `bench/power/README.md`, `bench/power/results/BENCHMARK.md`. Other
   sessions share checkouts, so don't use `git add -A`. Ask the user before
   pushing. End the commit message with
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
6. **Clean up on mordor3 once the results are committed:**
   `rm -rf ~/dt-power` (this includes the submit script). Then delete this
   part of HANDOFF.md.

Possible follow-ups, to do only if the user asks:
- the paper's energy table (`paper/main.tex`, `tab:energy`);
- rerunning dell and epyc with RAPL once the admins apply `ADMIN-REQUEST.md`.

## Part 2: `sorel_4m` on hopper

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
for job 19: start 2026-11-06 14:26. Job 60 (`dt-hvg-h`, another session's
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

## Part 3: hybrid-vs-gpu-only on hopper (job 60)

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
| State on 2026-11-05 | pending. ibalyka's jobs 11 and 12 hold hopper until 2026-11-06 14:26 at the latest (5-day limit). Jobs 19 and 22 (Part 2) are ahead of it. Expect 3–5 h of run time |
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
- No CPU benchmarks on hopper or ampere (one exception: power job 99 above);
  don't `srun` into busy nodes.
- Charts show means only.
- Commit or push only when asked.
- One `main` branch on GitHub; every checkout up to date at the end of a task.
