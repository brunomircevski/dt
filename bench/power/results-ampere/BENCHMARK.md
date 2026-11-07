# Power and energy of the five implementations on ampere

This experiment measures the time and energy that ./tree's five implementations
need to train one tree on ECBDL'14 15M (15 M rows, 631 features). Settings:
depth ≤ 10, no pruning, CART (gini) and C4.5 (gain ratio). All five ran in one
SLURM job on ampere (job 223, 2026-11-07, 11:31–20:00).

Since 2026-11-06 a user can read ampere's CPU packages (RAPL) and its PSU
(set up by the cluster's admins), and NVML gives the GPU. So every
implementation is measured three ways at once: CPU packages, the GPU it used,
and the whole server.

This run replaces job 136 (2026-11-06). Job 136 trained a process's trees one
after another in the same process, and the later GPU trees were 40% slower than
a single run (7.1 s against 5.0 s). Since 2026-11-07, `--repeat` trains every
tree in its own forked child, and a GPU child keeps its CPU threads busy for
2 s before its timer starts (`--repeat-warmup 2`), as the CSV load does before
a single run's tree. Each tree is now the only tree of a fresh process, every
tree is measured, and the times match the time benchmark (see Time check).
The first ampere round (job 99, 2026-11-05) had no PSU reading; its files are
in the git history before commit 0ff754a.

| Implementation | Option | Resources | Runs × trees measured |
|---|---|---|---|
| Serial-Simple | `--simple` | 1 core | 1 × 1 |
| Serial-Opt | `--serial` | 1 core | 1 × 2 (two trees, each in its own child) |
| Parallel ("Multicore" in the paper) | `--parallel` | all 48 cores | 2 × 47 (CART), 2 × 27 (C4.5) |
| GPU | `--cuda --gpu-only` | one A100 80 GB (GPU 0) + the 24 cores of its NUMA node (CPUs 0–23, memory node 0) | 2 × 144 (CART), 2 × 143 (C4.5) |
| Hybrid | `--cuda` | same | 2 × 144 (CART), 2 × 141 (C4.5) |

Code: main at a2891e7 (forking `--repeat`, `--repeat-warmup`), built on hopper
on 2026-11-07 for sm_80 and sm_90 (`~/dt-wt/tree`, `~/dt-wt/tree_cpu`). Every
run built the same tree: CART 1,863 nodes, C4.5 725 nodes, depth 10.

## What was measured

| Sensor | What it covers | Readout | Refresh |
|---|---|---|---|
| BMC PSU reading (`/run/power/psu_W`, the admins' IPMI DCMI poller) | whole server, PSU input | 0.1 s | **a point reading every ~59 s** |
| RAPL `package-0`, `package-1` (powercap `energy_uj`) | each CPU socket, including the I/O die | 0.1 s | ~1 ms (hardware) |
| NVML energy counter of GPU 0 | the GPU the run used; GPU and Hybrid runs only | 0.1 s | ~100 ms |

"Rest of the server" in the charts = PSU − both CPU packages − the GPU counted:
memory, fans, disks, network, the board, PSU losses and **the three other A100s,
which stay idle** (141 W together). AMD has no DRAM sensor, so memory is not
separated.

Supermicro AS-4124GS-TNR, 2× EPYC 7402 (Zen 2, 24 cores each), 4× A100 (two
80 GB, two 40 GB), 1 TB RAM. CPU frequency governor: `acpi-cpufreq ondemand`.
Idle power (mean of three windows: 900 s at the start, 900 s after the step
load, 900 s at the end): **PSU 467 W**, CPU packages 119 W (59 + 59 W), GPU 0
53 W.

## How the slow PSU meter is handled

ampere's BMC gives a point reading about once a minute, not an average, so a
short run cannot be measured from it. The protocol makes the measured windows
of every case add up to at least 720 s:

- Parallel, GPU and Hybrid train K = ceil(720 s / t) + 1 trees per process,
  each in its own child. Serial-Opt trains two trees (1,220 s and 1,420 s in
  all); Serial-Simple's one tree lasts 82–98 min.
- The PSU energy of a case is the mean of the readings that fall in any of its
  tree windows × the total window length; the gaps between children (fork,
  CUDA setup, warm-up) are left out. The readings are found as the times the
  value changes, so a reading equal to the previous one is lost.
- Every run has **at least 8 readings** (Parallel 12–13, GPU 11–13, Hybrid
  8–12, Serial-Opt 21–24, Serial-Simple 83–99).
- No lag correction: a point reading has no averaging delay.

The CPU packages and the GPU are counters, so they are exact over any window.

## Protocol

[../power_bench.sh](../power_bench.sh) `all` with `TARGET=720 IDLE=900
STEP=600 REPS=2 REPS_BY="simple=1 serial=1" REPEAT_BY="serial=2"` (and the
default `WARMUP=2`). [../pwrlog.py](../pwrlog.py) samples every sensor every
0.1 s for the whole job.

1. **Warm-up.** Parallel, GPU and Hybrid train two trees per algorithm. The
   second tree's time sets K: Parallel 47 (CART) and 27 (C4.5), GPU 144 and
   143, Hybrid 144 and 141.
2. **Idle, step, idle.** 900 s idle, 600 s with every core busy, 900 s idle.
3. **Runs.** Round-robin over algorithm × implementation, 20 s of idle before
   each process. Serial-Simple and Serial-Opt ran once, Parallel, GPU and
   Hybrid twice. A process loads the data once, then forks one child per tree.
4. **Idle.** 900 s at the end.

Every number is per tree, over every tree. A tree's window runs from the start
of its training to the end of pruning, before its memory is freed.

## Time check

The time per tree matches the time benchmark
([../../results/cluster-ecbdl](../../results/cluster-ecbdl), single runs on the
same node, data and settings) within 1.4%, and the first tree of each process
matches the rest. GPU and Hybrid trees stay at 5.0–5.2 s through all 140+
trees of a process.

| Implementation | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|
| Parallel | CART | 15.5 s | 15.7 s | – |
| Parallel | C4.5 | 28.6 s | 28.3 s | – |
| GPU | CART | 5.05 s | 4.98 s | 5.02 s (+0.6%) |
| GPU | C4.5 | 5.09 s | 5.05 s | 5.05 s (+0.8%) |
| Hybrid | CART | 5.09 s | 5.04 s | 5.07 s (+0.4%) |
| Hybrid | C4.5 | 5.13 s | 5.13 s | 5.10 s (+0.7%) |

## Results

### CART (gini)

| Implementation | t per tree | PSU power (above idle) | CPU packages | GPU | PSU energy per tree | above idle | CPU packages | GPU | rest of the server |
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | 4,929 s | 481 W (+14) | 125 W | – | 2.37 MJ | 68.6 kJ | 617 kJ | – | 1.75 MJ |
| Serial-Opt | 611 s | 480 W (+13) | 126 W | – | 293 kJ | 8.16 kJ | 77.1 kJ | – | 216 kJ |
| Parallel | 15.5 s ± 0.1% | 707 W (+241) | 312 W | – | 11.0 kJ ± 0.4% | 3.73 kJ | 4.84 kJ | – | 6.14 kJ |
| GPU | 5.05 s ± 0.1% | 615 W (+148) | 138 W | 221 W | 3.11 kJ ± 0.8% | 0.75 kJ | 0.70 kJ | 1.12 kJ | 1.29 kJ |
| Hybrid | 5.09 s ± 0.1% | 646 W (+180) | 153 W | 219 W | 3.29 kJ ± 1.3% | 0.91 kJ | 0.78 kJ | 1.12 kJ | 1.39 kJ |

### C4.5 (gain ratio)

| Implementation | t per tree | PSU power (above idle) | CPU packages | GPU | PSU energy per tree | above idle | CPU packages | GPU | rest of the server |
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | 5,855 s | 483 W (+16) | 125 W | – | 2.83 MJ | 93.5 kJ | 734 kJ | – | 2.09 MJ |
| Serial-Opt | 709 s | 480 W (+13) | 126 W | – | 340 kJ | 9.50 kJ | 89.4 kJ | – | 251 kJ |
| Parallel | 28.6 s ± 0.7% | 704 W (+238) | 295 W | – | 20.1 kJ ± 0.7% | 6.79 kJ | 8.43 kJ | – | 11.7 kJ |
| GPU | 5.09 s ± 0.0% | 620 W (+153) | 138 W | 221 W | 3.16 kJ ± 3.0% | 0.78 kJ | 0.70 kJ | 1.13 kJ | 1.33 kJ |
| Hybrid | 5.13 s ± 0.0% | 634 W (+167) | 153 W | 220 W | 3.25 kJ ± 2.7% | 0.86 kJ | 0.78 kJ | 1.13 kJ | 1.34 kJ |

± is one standard deviation over the runs (none for the single serial runs).
Full tables, with peaks and both sockets: [summary.md](summary.md),
[summary.csv](summary.csv), every run in [runs.csv](runs.csv).

## Charts

- **Energy, power and time in one chart:** [CART](chart-cpu-gpu-ampere-cart.png),
  [C4.5](chart-cpu-gpu-ampere-c45.png). Per implementation: energy per tree (CPU
  packages, GPU, PSU), mean power (CPU packages, GPU, PSU) and time, one bar
  each, on log axes. The light part of a bar ends at the idle value (power: the
  node's idle power, the same for every run; energy: idle power × time).
- [chart-power.png](chart-power.png): mean power, stacked socket 0, socket 1,
  GPU, rest of the server.
- [chart-energy.png](chart-energy.png), [chart-energy-split.png](chart-energy-split.png):
  energy per tree, and the share of each component.
- [chart-time.png](chart-time.png), [chart-time-energy.png](chart-time-energy.png):
  time per tree, and time against energy.
- [chart-traces.png](chart-traces.png): power over time of each case's first run.

![C4.5](chart-cpu-gpu-ampere-c45.png)

## Findings

- **GPU needs the least energy per tree:** 3.11–3.16 kJ at the PSU, 3.5× (CART)
  and 6.4× (C4.5) less than Parallel, 94–108× less than Serial-Opt and
  760–890× less than Serial-Simple.
- **Hybrid takes the same time as GPU** (within 1%) and 3–6% more energy: its
  CPU threads add 15 W at the CPU packages and 14–31 W at the PSU. The two
  runs of a case differ by 1–4% at the PSU (Hybrid CART 3.32 and 3.26 kJ),
  while their CPU packages agree within 0.5%.
- **The idle server dominates.** Idle is 467 W, so a serial run adds only
  13–16 W (3%) to it, and even a GPU run only 148–180 W. In a GPU tree the GPU
  is 36% of the PSU energy, the CPU packages 22% and the rest of the server
  42%; the three idle A100s alone are about 0.7 kJ (23%).
- **Against job 136:** GPU and Hybrid trees are now 26–29% faster (5.05 s
  against 7.09 s) and need a third less energy (GPU CART 3.11 against
  4.67 kJ), because job 136 measured slowed-down later trees. Parallel and the
  serial runs agree within 2–3% (Parallel CART 11.0 kJ in both).

## Reproduce

On mordor3, from `~/dt`:

```bash
sbatch -p ampere -w ampere --exclusive -c 48 -t 12:00:00 -J pwr-psu-ampere \
  --export=ALL,TARGET=720,IDLE=900,STEP=600,REPS=2,REPS_BY="simple=1 serial=1",REPEAT_BY="serial=2" \
  -o bench/results/power-psu-ampere/ampere/job.log bench/power/power_bench.sh all bench/results/power-psu-ampere/ampere
```

Then, on the PC, after copying the job folder:

```bash
bench/.venv/bin/python bench/power/analyze.py bench/power/results-ampere
```

The raw files are in [ampere/](ampere/): `markers.txt`, `machine.txt`,
`samples.csv.gz` (pwrlog.py, which also has the RAPL counters),
`perf_rapl.csv.gz` (`perf stat` RAPL, logged too but not used) and `out/`
(./tree's output per process). `job.log` is empty.
