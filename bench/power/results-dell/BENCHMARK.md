# Power and energy of the three CPU implementations on dell4

This experiment measures the time and energy that ./tree's three CPU
implementations need to train one tree on ECBDL'14 15M (15 M rows, 631
features). Settings: depth ≤ 10, no pruning, CART (gini) and C4.5 (gain ratio).
It ran in one SLURM job on dell4 (job 218, 2026-11-07, 10:21–14:07).

It replaces job 148 (2026-11-06), which trained a process's trees one after
another in the same process. Since 2026-11-07, `--repeat` trains every tree in
its own forked child, so each tree is the only tree of a fresh process, like a
single run, and every tree is measured. The first round on dell4
([../results/BENCHMARK.md](../results/BENCHMARK.md), job 85) could read only
the PSU; since 2026-11-06 the RAPL counters are readable by users
([../ADMIN-REQUEST.md](../ADMIN-REQUEST.md)), so this run measures the CPU
packages and the whole server at the same time.

| Implementation | Option | Resources | Runs × trees measured |
|---|---|---|---|
| Serial-Simple | `--simple` | 1 core | 1 × 1 |
| Serial-Opt | `--serial` | 1 core | 3 × 1 |
| Parallel ("Multicore" in the paper) | `--parallel` | all 48 cores | 10 × 11 (CART), 10 × 9 (C4.5) |

Code: main at b486bbb (forking `--repeat`), built on hopper on 2026-11-07
(`~/dt/tree_cpu`; machine.txt says 9af6148, the checkout's label). Every run
built the same tree: CART 1,863 nodes, C4.5 725 nodes, depth 10.

## What was measured

| Sensor | What it covers | Readout | Refresh |
|---|---|---|---|
| ACPI power meter (`hwmon power_meter`, the iDRAC) | whole server, PSU input | 0.1 s | a 2 s average, refreshed every 2.1 s |
| RAPL `package-0`, `package-1` (powercap `energy_uj`) | each CPU socket | 0.1 s | ~1 ms (hardware) |
| RAPL `dram` ×2, `psys` | memory per socket, platform | 0.1 s | logged in `samples.csv.gz`, not in the charts |

"Rest of the server" in the charts = PSU − both CPU packages: memory, fans,
disks, network, the board and PSU losses.

Dell PowerEdge R660, 2× Xeon Gold 6542Y (24 cores each), 512 GB RAM. CPU
frequency governor: `intel_pstate powersave`. Idle power (mean of three
windows: 300 s at the start, 300 s after the step load, 300 s at the end):
**PSU 155 W**, CPU packages 68 W (37 + 31 W). The PSU meter lags the load by
**2.6 s**, measured on a 300 s all-core step.

## Protocol

[../power_bench.sh](../power_bench.sh) `cpu` with `TARGET=120 IDLE=300
STEP=300 REPS=10 REPS_BY="simple=1 serial=3"`. [../pwrlog.py](../pwrlog.py)
samples every sensor every 0.1 s for the whole job.

1. **Warm-up.** Parallel trains two trees per algorithm. The second tree's time
   sets K = ceil(120 s / t) + 1 trees per process: 11 (CART) and 9 (C4.5).
2. **Idle, step, idle.** 300 s idle, 300 s with every core busy (to measure the
   meter's delay), 300 s idle.
3. **Runs.** Round-robin over algorithm × implementation, 20 s of idle before
   each process. Serial-Simple ran once, Serial-Opt 3 times, Parallel 10 times.
   A Parallel process loads the data once, then forks one child per tree.
4. **Idle.** 300 s at the end.

Every number is per tree, over every tree. A tree's window runs from the start
of its training to the end of pruning (before its memory is freed); the gaps
between children (fork, thread start) are left out. The PSU energy is its
readings held until the next one, shifted back by the meter's delay. Every
measured case has at least 62 PSU changes (Parallel), 110 (Serial-Opt) and 810
(Serial-Simple).

## Time check

The time benchmark ran on dell1, not dell4, and dell1's runs vary by 13–35%
between batches, so the times here are not compared with it. Within this run,
each Parallel process's first tree is 1–2% slower than its mean tree:

| Implementation | Algorithm | t per tree | first tree |
|---|---|---|---|
| Serial-Simple | CART | 2,576 s | — |
| Serial-Simple | C4.5 | 2,865 s | — |
| Serial-Opt | CART | 359 s | — |
| Serial-Opt | C4.5 | 441 s | — |
| Parallel | CART | 12.6 s | 12.9 s |
| Parallel | C4.5 | 15.9 s | 16.2 s |

## Results

### CART (gini)

| Implementation | t per tree | PSU power (above idle) | CPU packages | PSU energy per tree | above idle | CPU packages | rest of the server |
|---|---|---|---|---|---|---|---|
| Serial-Simple | 2,576 s | 307 W (+152) | 198 W | 792 kJ | 392 kJ | 509 kJ | 282 kJ |
| Serial-Opt | 359 s ± 1.2% | 309 W (+153) | 199 W | 111 kJ ± 1.3% | 55.1 kJ | 71.4 kJ | 39.4 kJ |
| Parallel | 12.6 s ± 0.7% | 676 W (+521) | 479 W | 8.54 kJ ± 1.5% | 6.58 kJ | 6.04 kJ | 2.49 kJ |

### C4.5 (gain ratio)

| Implementation | t per tree | PSU power (above idle) | CPU packages | PSU energy per tree | above idle | CPU packages | rest of the server |
|---|---|---|---|---|---|---|---|
| Serial-Simple | 2,865 s | 308 W (+153) | 197 W | 883 kJ | 438 kJ | 564 kJ | 318 kJ |
| Serial-Opt | 441 s ± 0.2% | 309 W (+153) | 199 W | 136 kJ ± 0.3% | 67.6 kJ | 87.6 kJ | 48.4 kJ |
| Parallel | 15.9 s ± 1.1% | 663 W (+508) | 469 W | 10.6 kJ ± 1.9% | 8.10 kJ | 7.47 kJ | 3.10 kJ |

± is one standard deviation over the runs. Full tables, with peaks and both
sockets: [summary.md](summary.md), [summary.csv](summary.csv), every run in
[runs.csv](runs.csv).

## Charts

- **Energy, power and time in one chart:** [CART](chart-cpu-gpu-dell-cart.png),
  [C4.5](chart-cpu-gpu-dell-c45.png). Per implementation: energy per tree (CPU
  packages, PSU), mean power (CPU packages, PSU) and time, on log axes; the
  lighter part of each bar is idle power × time.
- [chart-power.png](chart-power.png): mean power, stacked socket 0, socket 1,
  rest of the server.
- [chart-energy.png](chart-energy.png), [chart-energy-split.png](chart-energy-split.png):
  energy per tree, and the share of each component.
- [chart-time.png](chart-time.png), [chart-time-energy.png](chart-time-energy.png):
  time per tree, and time against energy.
- [chart-traces.png](chart-traces.png): power over time of each case's first run.

![CART](chart-cpu-gpu-dell-cart.png)

## Findings

- **Parallel needs the least energy per tree by far:** 8.54 kJ (CART) and
  10.6 kJ (C4.5), 13× less than Serial-Opt and 83–93× less than Serial-Simple.
- **The CPU packages are 64% of the server's energy** in the serial runs and
  70–71% in Parallel.
- **One busy core raises the server by ~153 W**, the CPU packages by only
  ~130 W.
- **dell4 needs 2.2× epyc1's energy per tree in Parallel** (8.54 vs 3.83 kJ
  CART, [../results-epyc/BENCHMARK.md](../results-epyc/BENCHMARK.md)): it is
  2× slower and draws 14% more power. For the serial runs the gap is
  1.3–1.9×.
- **Agrees with the earlier rounds:** per-tree energy within 2% of job 148
  (Parallel CART 8.54 vs 8.55 kJ, Serial-Simple CART 792 vs 798 kJ) and of
  round 1 (job 85, PSU only: 8.46 and 786 kJ). Parallel trees are 2–3% slower
  than job 148's (12.6 vs 12.3 s), whose first tree of each process was left
  out.

## Reproduce

On mordor3, from `~/dt`:

```bash
sbatch -p dell -w dell4 --exclusive -c 48 -t 6:00:00 -J pwr-psu-dell \
  --export=ALL,TARGET=120,IDLE=300,STEP=300,REPS=10,REPS_BY="simple=1 serial=3" \
  -o bench/results/power-psu-dell/dell/job.log bench/power/power_bench.sh cpu bench/results/power-psu-dell/dell
```

```bash
bench/.venv/bin/python bench/power/analyze.py bench/results/power-psu-dell
```

The raw files are in [dell/](dell/): `markers.txt`, `machine.txt`,
`samples.csv.gz` (pwrlog.py) and `out/` (./tree's output per process).
