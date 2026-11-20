# Energy per tree: ./tree's five backends on epyc1 and hopper

ECBDL'14 15M (the first 15M rows of ECBDL'14's train.csv, 631 features),
depth ≤ 10, no pruning, CART and C4.5. Code 81bf6e6c (the final build of the
2026-11-19 time benchmark rerun, `~/dt-final`; GPU set-up timed). Jobs 519
(hopper, 2026-11-20 02:24–08:47) and 528 (epyc1, 2026-11-20 11:04–13:30);
the warm-up check: job 171 (hopper, 2026-11-16, f53347a9) and the 2 s and 15 s
checks inside job 519.
Method: [Method](#method). Every table, with more columns:
[summary.md](summary.md); every process: [runs.csv](runs.csv); the meter
models: [meter.csv](meter.csv).
The same measurement without a depth limit, where `--hybrid` gains (MNIST8M,
Multicore, GPU and Hybrid on hopper): [cluster-power-mnist8m](../cluster-power-mnist8m/BENCHMARK.md).

**Two servers: epyc1 and hopper.** The power benchmark needs one CPU server
and one GPU server, and uses our best of each: epyc1 (2× EPYC 9255) and
hopper (H100 PCIe + 2× Xeon Gold 6444Y). Since 2026-11-19 it no longer runs
on ampere or dell ([why](#why-not-ampere-and-dell)); their last results
(2026-11-16, jobs 170, 172, 173) are kept in
[ampere-dell-2026-11-16](ampere-dell-2026-11-16/BENCHMARK.md).

**Energy** is the whole server's, at the PSU, over the training windows only
(the CSV load, the gaps between trees and the evaluation are left out), from
each node's meter modelled from the fast sensors. **Above idle** subtracts
what the idle server would have used in the same time.

## Results

Energy per tree (± one standard deviation over the processes):

| Backend | Node | CART: t | E | above idle | C4.5: t | E | above idle |
|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc1 | 2,291 s | 587 kJ | 302 kJ | 2,596 s | 666 kJ | 343 kJ |
| Serial-Simple | hopper | 2,551 s | 1.38 MJ | 341 kJ | 2,793 s | 1.51 MJ | 373 kJ |
| Serial-Opt | epyc1 | 147 s | 38.5 kJ ± 0.9% | 20.3 kJ | 150 s | 39.3 kJ ± 1.2% | 20.7 kJ |
| Serial-Opt | hopper | 251 s | 136 kJ ± 0.7% | 33.5 kJ | 258 s | 139 kJ ± 0.7% | 34.3 kJ |
| Multicore | epyc1 | 4.56 s | 2.73 kJ ± 0.3% | 2.16 kJ | 4.79 s | 2.87 kJ ± 0.6% | 2.27 kJ |
| Multicore | hopper | 9.12 s | 8.26 kJ ± 0.3% | 4.54 kJ | 10.1 s | 9.08 kJ ± 0.3% | 4.95 kJ |
| GPU | hopper (H100) | 2.54 s | **1.88 kJ** ± 1.1% | 843 J | 2.55 s | **1.88 kJ** ± 0.5% | 840 J |
| Hybrid | hopper (H100) | 2.62 s | 2.12 kJ ± 0.6% | 1.05 kJ | 2.63 s | 2.14 kJ ± 0.4% | 1.07 kJ |

Runs × trees: Serial-Simple 1 × 1, Serial-Opt 3 × 1, the others 3 processes of
13–49 trees (about 120 s of training each).

![energy per tree](chart-energy.png)

What it shows:

- **The H100 server uses the least energy per tree**: 1.88 kJ (GPU), 31%
  (CART) and 34% (C4.5) less than Multicore on epyc1 (2.73 and 2.87 kJ).
  Above idle the gap is wider: 0.84 kJ against 2.16–2.27 kJ (2.6–2.7×).
- **Hybrid costs 13–14% more energy than GPU** and is 3% slower: with the
  GPU set-up timed (since 81bf6e6c), its page-locked hand-off buffers cost
  about 0.1 s per tree, and its CPU threads work while the GPU does (CPU
  packages + DRAM 0.92 kJ against 0.69 kJ per tree).
- **Serial-Opt uses 10–17× less energy than Serial-Simple**, and Multicore
  another 14–16× less than Serial-Opt.
- **Idle power weighs on hopper's single-thread runs**: hopper idles at
  406 W (four H100s, three of them unused by any run), epyc1 at 119 W, so a
  Serial run costs 2.3–3.5× as much on hopper as on epyc1.
- **Against the 2026-11-16 run (f53347a9)**: GPU and Hybrid 3–5% more energy
  per tree (2.46 → 2.54 s and 2.46 → 2.62 s: the GPU set-up is now inside the
  timer); epyc1 idled 19 W lower (119 against 138 W at the meter), so its
  Serial runs used 4–5% less energy at the same speed; everything else
  within 3%.

![power while training](chart-power.png)

Per node, CPU, GPU and PSU energy and power side by side:
[epyc1](chart-cpu-gpu-epyc.png), [hopper](chart-cpu-gpu-hopper.png)
(CART above C4.5); time against energy: [chart-time-energy.png](chart-time-energy.png); power
over time of each case's first process: [chart-traces.png](chart-traces.png).

## The meter model

![meter model](chart-meter.png)

| Node | meter refresh | readings | delay D, window W | model (W) | rms of a reading |
|---|---|---|---|---|---|
| epyc1 | 6.3 s | 893 | 1.5 s, 2 s | 23 + 1.39·CPU | 12 W |
| hopper | 1.0 s | 16,113 | 0.5 s, 1 s | 58 + 154 + 1.01·CPU + GPU | 12 W |

CPU = RAPL packages + DRAM (epyc1: packages only), GPU = the GPU the GPU
backends use (NVML), counted as measured (× 1); 154 W = G_idle, the mean
NVML power of hopper's three unused GPUs over the job (measured, not fitted),
so that the constant (58 W) is the server without them. The model's idle
power matches the meter's within 2 W on hopper and 5 W on epyc1 (124 against
119 W). The model fitted on either half of a job gives the same energy per
tree within 0.7% (Multicore on epyc1: 1.4%). The MNIST8M job's fit: 53 + 155 + 1.03·CPU + GPU, rms 19 W.

**Model against the meter** ([chart-methods.png](chart-methods.png)): on
windows much longer than the meter's refresh (the Serial runs, Multicore on
hopper, every MNIST8M case) the meter's own energy agrees with the model
within 1.8% (here Serial-Opt on epyc1), and the readings inside the training
windows have a mean residual within ±0.6% of the power on every hopper case,
the 2.5 s H100 trees included. On epyc1 the model is 0.9–1.8% low on the
single-thread runs (mean residual inside +1.7% on Serial-Opt, +0.9% on
Serial-Simple; −0.9% on Multicore): the bias the old busy term removed
([below](#the-simpler-model-2026-11-19)), still within 2%. Leaving a case
out moves its energy by ≤ 0.5%. The meter's
own reading of a short window is what the model replaces: on the November 8
hopper job it was 3–5% low on the 3 s GPU trees.

## Time check

The time per tree against the time benchmark
([../cluster-ecbdl](../cluster-ecbdl), single runs, 10 a case, same build):

- **epyc1: within 2.2%** on every backend (Multicore 4.56 s against 4.47 s).
- **hopper's GPU and Hybrid trees are 10–12% faster than the time
  benchmark's** (GPU 2.54 s against 2.81 s, Hybrid 2.62 s against 2.94 s).
  This is the warm-up: the measured trees spin their CPU threads for 15 s
  before the timer, while a time benchmark run's threads are only as warm as
  its input load leaves them. The 2 s warm-up check of the same job matches
  the time benchmark (GPU 2.82 s, Hybrid 2.91 s, trees 3–10); see
  [Warm-up check](#warm-up-check). The energy per GPU or Hybrid tree here is
  that of a warmed-up tree; a cold single run takes about 10% longer and, from
  the 2026-11-16 check, costs up to ~8% more energy. No conclusion changes
  (the H100 tree would still need about 1.3× less energy than epyc1's
  Multicore).

Multicore on hopper (not in the time benchmark) varied 0.5% between
processes; its first tree of each process was 1–2% slower than the rest.

## Warm-up check

GPU and Hybrid trees need their CPU threads busy before the timer starts
(`--repeat-warmup`): without it their CPU clocks are low (hopper schedutil)
and the trees set up slower.

![warm-up](chart-warmup.png)

Mean time of trees 3– of one process per warm-up, CART:

| Job | warm-up | GPU | Hybrid | time benchmark (GPU / Hybrid) |
|---|---|---|---|---|
| 171 (2026-11-16, f53347a9) | 0 s | 2.77 s | 2.81 s | 2.47 / 2.42 s |
| 171 | 2 s | 2.70 s | 2.57 s | |
| 171 | 5 s | 2.48 s | 2.45 s | |
| 171 | **15 s** | 2.46 s | 2.47 s | |
| 519 (2026-11-20, 81bf6e6c) | 2 s | 2.82 s | 2.91 s | 2.81 / 2.94 s |
| 519 | **15 s** | 2.56 s | 2.68 s | |

The measured runs used 15 s (bold). On f53347a9 the time benchmark matched
the 15 s warm-up; on the final build it matches the 2 s one
([Time check](#time-check)). `chart-warmup.png` and `summary.md` pool both
jobs.

## Notes

- Hybrid uses the default `--handoff-rows` of the final build
  (max(512, min(10 × features, 2^depth, 16384)) = 1024 at depth 10 for
  ECBDL'14), like the time benchmark.
- Serial-Simple: one process per case (38–47 min each).
- epyc1 has no DRAM sensor (AMD): its memory is in "rest of the server".
  hopper's three unused H100s (~51 W each, 154 W together: G_idle in the
  model) are their own component,
  "unused GPUs", not part of the rest.

## Why not ampere and dell

Until 2026-11-16 the benchmark also ran on ampere (A100 80 GB + 2× EPYC 7402)
and dell4 (2× Xeon Gold 6542Y). Both were dropped on 2026-11-19:

- **ampere's meter is too slow to keep.** Its PSU meter is the BMC over IPMI
  DCMI (`/run/power/psu_W`), which gives one point reading (a ~2.5 s
  average) only every ~59 s. The model still worked (rms 23 W, leave-one-case-out
  ≤ 1.1%), but only with special handling: 240 s processes, 240 s staircase
  steps and 600 s idle periods (16 h jobs), a separate "slow meter" path in
  `analyze.py` (readings spread over the refreshes, a 70 s delay grid, the
  meter check from 5–9 point readings per case). We did not want that in the
  paper, and the code is gone.
- **One CPU server and one GPU server are enough.** The comparison the paper
  needs is the best CPU server against the best GPU server: epyc1 and hopper
  (the H100). On 2026-11-16 (f53347a9), dell4 (Multicore 7.18 kJ against
  epyc1's 2.70 kJ) and the A100 (GPU 2.94 kJ against the H100's 1.81 kJ) were the slower of each kind
  and added no conclusion; dell1, where dell's Multicore time benchmark ran,
  is retired.

Their last results are in [ampere-dell-2026-11-16](ampere-dell-2026-11-16/BENCHMARK.md).

## Method

### The problem: the whole-server meters are slow

The number that compares a CPU server with a GPU server is the energy of the
whole server, measured at the power supply (PSU). Each node has such a
meter, but neither is fast enough for a tree that takes 2.4 s (H100)
or 4.5 s (Multicore on epyc1):

| Node | PSU meter | refresh | what a reading is |
|---|---|---|---|
| hopper | ACPI power meter (Intel Node Manager) | 1 s | an average over ~1.25 s |
| epyc1 | ACPI power meter (iDRAC) | 4–10 s (median 6.3 s) | an average over ~2.5 s |

(The averaging windows are fitted, see below.) The fast sensors, RAPL
(CPU packages and, on hopper, DRAM) and NVML (each GPU's energy
counter), cover only parts of the server: not the fans, disks, network,
board, the other GPUs or the PSU's own losses (on hopper about 30% of the
energy of a GPU tree).

### The solution: model the meter from the fast sensors

`pwrlog.py` reads every sensor every 0.1 s for the whole job. For each node,
`analyze.py` then fits the meter itself:

    reading(t) = mean over [t − D − W, t − D] of  a + G_idle + b·CPU + GPU

where CPU = RAPL packages + DRAM (W), GPU = the NVML power of the GPU in use
(W, as measured), G_idle = the mean NVML power of the node's other GPUs over
the whole job (hopper's three idle H100s, ~154 W; none on epyc1; measured,
not fitted, so a is the server without them), D is the meter's delay and W
its averaging window. a and b come from least squares over **every reading
of the job** (idle, a staircase of CPU load, every run, the gaps); D and W
from a grid search (D from 0 to 12 s in 0.1 s steps; W from 0.1 to 10 s).
The PSU energy of a tree is then the integral of a + G_idle + b·CPU + GPU
over its training window, at 0.1 s resolution, whatever the meter's refresh.
Why this form, and the other models tried: [The simpler model](#the-simpler-model-2026-11-19).
"Unused GPUs" = the NVML energy of the node's other GPUs (all four for a
CPU backend); "rest of the server" = PSU − CPU packages − DRAM − every GPU.

Why the model is needed even on epyc1: its meter (~6 s) is slower than its
Multicore trees (~4.5 s), and the meter's own readings over short windows
include the gaps between trees (earlier meter-only runs were 3–5% low on
short GPU trees).

**Without the unused GPUs.** hopper holds four GPUs; a backend uses one
(CPU backends none), and the other cards idle at ~50 W each, so ~150 W of
every GPU tree's power (~20% of its energy) and ~200 W of a CPU backend's
are cards it does not use. `summary.md` and `chart-energy.png` (dashed)
also give the energy minus the unused GPUs' NVML energy: the same server with only the GPU in use. It is
a counterfactual (a smaller server has other fans and PSUs), for comparing
a GPU server with a CPU server; the whole-server number stays the measured
one.

The job gives the fit the range of loads it needs: idle; a staircase of
1, 2, 4, …, all CPUs busy (60 s per step); all five backends (three on
epyc1); and, on hopper, GPU and Hybrid runs with their CPU warm-ups.

**Checks** (`summary.md`, `chart-meter.png`, `chart-methods.png`):

1. The coefficients fitted on the first and on the second half of the job
   alone, and the energy each half's model gives.
2. The mean residual (reading − model) of the readings whose averaging
   window lies inside a case's training windows: a bias of the model during
   that backend's trees would show here.
3. The meter's own energy over the same windows (readings held, shifted back
   by D + W/2). It needs windows much longer than the meter's refresh: it
   agrees with the model within ~1.5% on every long window, and on 2–3 s
   GPU trees it is 3–5% low (re-analysis of the 2026-11-08 hopper job), which
   is the error the model removes.
4. **Leave the case out**: the model fitted without any reading of the
   case's own processes (its warm-ups and checks too), so it predicts a
   load it has never seen; and the held-out readings' mean residual. On the
   2026-11-16 jobs the energy per tree moved by ≤ 0.5%, so no case's number
   depends on its own readings.

### The simpler model (2026-11-19)

Until 2026-11-19 the model was a + b·CPU + c·GPU + d·busy (busy = the number
of busy CPUs, `/proc/stat`), with hopper's three unused GPUs inside a:
epyc1 39 + 1.36·CPU − 0.10·busy, hopper 209 + 1.03·CPU + 0.99·GPU − 0.12·busy.
The busy term had been added for the 2026-11-07 epyc job, where the model
without it was 1.5–2.3% low on single-thread runs; on the 2026-11-16 jobs
it no longer mattered. Candidates, each fitted (with its own D and W) on the
three jobs (epyc1 169, hopper 174, MNIST8M hopper 415); the ranges are over
all 22 cases:

| model | rms epyc1 / hopper / MNIST8M | E vs the old model | meter vs model | mean residual inside | leave the case out |
|---|---|---|---|---|---|
| old: a + b·CPU + c·GPU + d·busy | 13.2 / 12.3 / 18.7 W | – | ≤ 1.6% | ≤ 0.9% | ≤ 0.7% |
| no busy: a + b·CPU + c·GPU | 13.2 / 12.3 / 18.9 W | −0.35 … +0.09% | ≤ 1.4% | ≤ 1.0% | ≤ 0.5% |
| **no busy, c = 1: a + G_idle + b·CPU + GPU** | 13.2 / 12.3 / 18.9 W | −0.32 … +0.18% | ≤ 1.3% | ≤ 0.9% | ≤ 0.5% |
| no busy, c = 1, every GPU's NVML power as measured | 13.2 / 12.3 / 18.8 W | −0.36 … +0.26% | ≤ 1.3% | ≤ 0.9% | ≤ 0.5% |
| no busy, one c for all four GPUs | 13.2 / 12.3 / 18.8 W | −0.35 … +0.09% | ≤ 1.4% | ≤ 1.0% | ≤ 0.5% |
| b = c = 1: a + CPU + every GPU | 36.0 / 12.4 / 19.5 W | −13.4 … +1.3% | ≤ 14% | ≤ 13% | ≤ 1.2% |

The chosen model (bold) drops the busy term, counts the GPU in use at its
NVML power (fitted, c was 0.99 on both hopper jobs) and makes the unused
GPUs an explicit measured constant: G_idle, their mean NVML power over the job
(154.3 W in job 174, 154.7 W in job 415; 154.1 W in the idle windows alone, so
nearly constant). As a constant it only splits a, so its fit is the same as
a + b·CPU + GPU. Its rms is within 0.2 W of the old model's, every other
check is as good or better, and no energy per tree moved by more than 0.32%. b cannot be fixed to 1: epyc1 has no
DRAM sensor, so b (1.34) carries its memory and more, and without it the
model is 13% off; on hopper alone b = 1 fits (rms 12.4 W) but moves the MNIST8M
energies by up to 1.1%. (The comparison was a one-off script, not kept.)

### The runs

Each measured process is `./tree --repeat K`: it loads the CSV once, then
trains K trees, each in its own forked child, so that every tree is a fresh
process's only tree, and prints each tree's training window as Unix times.
K = ceil(TARGET / t) + 1 (t from a warm-up run): about 120 s of training per
process; 3 processes per case, round-robin, 20 s of idle before each
(Serial-Simple: one process of one tree, 39–47 min). The mean energy per
tree is reported.

**Warm-up of the GPU backends.** A GPU or Hybrid tree whose CPU threads were
idle before it sets up slower (the CPU clocks are low: hopper schedutil):
trees 3–12 of a process took 2.8 s instead of 2.5 s. A single run's tree
follows the data load (every thread reads its part of the .f32 file), so
with `--repeat-warmup S` each child spins its threads for S s before its
timer: hopper needs 5 s (with 2 s its trees were 3–10% slower) and the
measured runs used 15 s.
The warm-up is outside the training window, so it is never counted, and it
does not inflate the window: in the warm-up checks (2026-11-16) the CPU
packages + GPU used 1–4% *less* energy per tree after 15 s of warm-up than
after none (hopper GPU: 1,165 against 1,217 J), because the tree is faster.
What it must match is the time benchmark's single runs (the time check):
with no warm-up, hopper's GPU trees took 13% longer and cost ~8% more.

**Placement**, as in the time benchmark (`bench/tools/cluster/job.sh`,
`bench/tools/run.py`): Serial-Simple and Serial-Opt on CPU 0; Multicore on every
CPU; GPU and Hybrid on hopper's GPU 2 (NUMA node 1), with every CPU of that
node first and first-touch memory.

**Time check**: every case's time per tree against the time benchmark
(`bench/cluster-ecbdl`, single runs).

Run it (from the repository root, on the login node, with ./tree and
./tree_cpu built by `bench/tools/cluster/build.sh`; about 2.5 h on epyc1 and
6.5 h on hopper):

    bench/tools/power/submit.sh -c hopper           # the warm-up check (optional)
    bench/tools/power/submit.sh epyc1 hopper        # one job per node
    bench/tools/.venv/bin/python bench/tools/power/analyze.py bench/cluster-power

### Earlier runs

Jobs 84–86 (2026-11-05) and 217, 218, 223, 300 (2026-11-07/08) measured the
code of that week with the meters' own readings over the tree windows (long
windows of many trees, readings shifted by a step-measured delay). They are in
the git history, as is the 2026-11-16 run of this benchmark on f53347a9 (jobs
169 epyc1, 174 hopper). The 2026-11-16 run on four servers (with ampere and dell4):
[ampere-dell-2026-11-16](ampere-dell-2026-11-16/BENCHMARK.md).
