# Handoff: `sorel_4m` on hopper (updated 2026-11-05)

Only this is left of the SOREL work of 2026-11-02. Done and pushed (75793c0,
b140fb0): the `--simple` rerun of SOREL (epyc now 0.94–0.97× dell, explained
in `docs/PERFORMANCE.md`), and `sorel_4m` on ampere and epyc, with the charts,
`bench/results/cluster-sorel/BENCHMARK.md`, the READMEs and the paper's CPU
cells of the two "4M sample" rows. Delete this file when hopper's runs are in
(or the user drops them).

The jobs run on mordor3 (`ssh mordor3`, user `bmircevski`, checkout `~/dt`).
If SSH times out, the PC is off the network mordor3 accepts: ask the user to
connect (VPN or campus), don't keep retrying.

## Waiting

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

## Decide with the user before they start: old build, GPU 0

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

## Finishing (once the runs are in)

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
   and delete this file. Push only if the user asks.

## Rules from the user

- Build, run and benchmark only on mordor3 compute nodes, never locally.
- No CPU benchmarks on hopper or ampere; don't `srun` into busy nodes.
- Charts show means only.
- Commit or push only when asked.
