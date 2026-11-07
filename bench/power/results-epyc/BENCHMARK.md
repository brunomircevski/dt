# Power and energy of the three CPU implementations on epyc1

This experiment measures the time and energy that ./tree's three CPU
implementations need to train one tree on ECBDL'14 15M (15 M rows, 631
features). Settings: depth ≤ 10, no pruning, CART (gini) and C4.5 (gain ratio).
It ran in one SLURM job on epyc1 (job 217, 2026-11-07, 10:21–13:38).

It replaces job 144 (2026-11-06), which trained a process's trees one after
another in the same process. Since 2026-11-07, `--repeat` trains every tree in
its own forked child, so each tree is the only tree of a fresh process, like a
single run, and every tree is measured. The first round on epyc1
([../results/BENCHMARK.md](../results/BENCHMARK.md), job 86) could read only
the PSU; since 2026-11-06 the RAPL counters are readable by users
(set up by the cluster's admins), so this run measures the CPU
packages and the whole server at the same time.

| Implementation | Option | Resources | Runs × trees measured |
|---|---|---|---|
| Serial-Simple | `--simple` | 1 core | 1 × 1 |
| Serial-Opt | `--serial` | 1 core | 3 × 1 |
| Parallel ("Multicore" in the paper) | `--parallel` | all 48 cores | 10 × 20 (CART), 10 × 16 (C4.5) |

Code: main at b486bbb (forking `--repeat`), built on hopper on 2026-11-07
(`~/dt/tree_cpu`; machine.txt says 9af6148, the checkout's label). Every run
built the same tree: CART 1,863 nodes, C4.5 725 nodes, depth 10.

## What was measured

| Sensor | What it covers | Readout | Refresh |
|---|---|---|---|
| ACPI power meter (`hwmon power_meter`, the iDRAC) | whole server, PSU input | 0.1 s | a 2 s average, refreshed every 4–10 s (median 6.3 s) |
| RAPL `package-0`, `package-1` (powercap `energy_uj`) | each CPU socket, including the I/O die | 0.1 s | ~1 ms (hardware) |

"Rest of the server" in the charts = PSU − both CPU packages: memory, fans,
disks, network, the board and PSU losses. AMD has no DRAM sensor, so memory is
not separated.

Dell PowerEdge R6725, 2× EPYC 9255 (Zen 5, 24 cores each), 512 GB RAM. CPU
frequency governor: `amd-pstate-epp performance`. Idle power (mean of three
windows: 300 s at the start, 300 s after the step load, 300 s at the end):
**PSU 116 W**, CPU packages 72 W (39 + 33 W). The PSU meter lags the load by
**7.1 s**, measured on a 300 s all-core step.

## Protocol

[../power_bench.sh](../power_bench.sh) `cpu` with `TARGET=120 IDLE=300
STEP=300 REPS=10 REPS_BY="simple=1 serial=3"`. [../pwrlog.py](../pwrlog.py)
samples both sensors every 0.1 s for the whole job.

1. **Warm-up.** Parallel trains two trees per algorithm. The second tree's time
   sets K = ceil(120 s / t) + 1 trees per process: 20 (CART) and 16 (C4.5).
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
measured case has at least 21 PSU changes (Parallel), 20 (Serial-Opt) and 142
(Serial-Simple).

## Time check

The time per tree matches the time benchmark
([../../results/cluster-ecbdl](../../results/cluster-ecbdl), the same node,
data and settings), and the first tree of each process matches the rest:

| Implementation | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|
| Serial-Simple | CART | 2,332 s | — | 2,273 s (+2.6%) |
| Serial-Simple | C4.5 | 2,639 s | — | 2,587 s (+2.0%) |
| Serial-Opt | CART | 225 s | — | 225 s (+0.2%) |
| Serial-Opt | C4.5 | 278 s | — | 277 s (+0.2%) |
| Parallel | CART | 6.44 s | 6.44 s | 6.32 s (+1.9%) |
| Parallel | C4.5 | 7.91 s | 7.88 s | 7.81 s (+1.3%) |

## Results

### CART (gini)

| Implementation | t per tree | PSU power (above idle) | CPU packages | PSU energy per tree | above idle | CPU packages | rest of the server |
|---|---|---|---|---|---|---|---|
| Serial-Simple | 2,332 s | 256 W (+140) | 167 W | 598 kJ | 327 kJ | 390 kJ | 208 kJ |
| Serial-Opt | 225 s ± 0.5% | 264 W (+147) | 172 W | 59.4 kJ ± 1.3% | 33.2 kJ | 38.7 kJ | 20.6 kJ |
| Parallel | 6.44 s ± 0.4% | 595 W (+479) | 424 W | 3.83 kJ ± 1.3% | 3.08 kJ | 2.73 kJ | 1.10 kJ |

### C4.5 (gain ratio)

| Implementation | t per tree | PSU power (above idle) | CPU packages | PSU energy per tree | above idle | CPU packages | rest of the server |
|---|---|---|---|---|---|---|---|
| Serial-Simple | 2,639 s | 257 W (+141) | 167 W | 679 kJ | 372 kJ | 441 kJ | 238 kJ |
| Serial-Opt | 278 s ± 0.2% | 263 W (+147) | 171 W | 73.0 kJ ± 1.1% | 40.8 kJ | 47.4 kJ | 25.6 kJ |
| Parallel | 7.91 s ± 0.3% | 601 W (+485) | 429 W | 4.75 kJ ± 1.4% | 3.83 kJ | 3.39 kJ | 1.36 kJ |

± is one standard deviation over the runs. Full tables, with peaks and both
sockets: [summary.md](summary.md), [summary.csv](summary.csv), every run in
[runs.csv](runs.csv).

## Charts

- **Energy, power and time in one chart:** [CART](chart-cpu-gpu-epyc-cart.png),
  [C4.5](chart-cpu-gpu-epyc-c45.png). Per implementation: energy per tree (CPU
  packages, PSU), mean power (CPU packages, PSU) and time, on log axes; the
  lighter part of each bar is idle power × time.
- [chart-power.png](chart-power.png): mean power, stacked socket 0, socket 1,
  rest of the server.
- [chart-energy.png](chart-energy.png), [chart-energy-split.png](chart-energy-split.png):
  energy per tree, and the share of each component.
- [chart-time.png](chart-time.png), [chart-time-energy.png](chart-time-energy.png):
  time per tree, and time against energy.
- [chart-traces.png](chart-traces.png): power over time of each case's first run.

![C4.5](chart-cpu-gpu-epyc-c45.png)

## Findings

- **Parallel needs the least energy per tree by far:** 3.83 kJ (CART) and
  4.75 kJ (C4.5), 15× less than Serial-Opt and 140–155× less than
  Serial-Simple. It draws 2.3× the power of a serial run but finishes 35–360×
  sooner.
- **The CPU packages are 65% of the server's energy** in the serial runs and
  71% in Parallel. The rest (116 − 72 = 44 W at idle) is mostly a fixed cost,
  so it weighs more in long runs.
- **One busy core raises the server by ~140–147 W**, but the CPU packages by
  only ~95–100 W: the remaining ~45 W is memory, fans and the rest. Serial runs
  pay this for their whole length.
- **Agrees with the earlier rounds:** per-tree energy within 2% of job 144
  (Parallel CART 3.83 vs 3.88 kJ, Serial-Simple CART 598 vs 594 kJ) and of
  round 1 (job 86, PSU only: 3.80 and 593 kJ). This run's Parallel trees are
  2% slower than job 144's (6.44 vs 6.29 s) and its PSU idle is 8 W lower
  (116 vs 124 W); both are within the node's run-to-run spread.

## Reproduce

On mordor3, from `~/dt`:

```bash
sbatch -p epyc -w epyc1 --exclusive -c 48 -t 6:00:00 -J pwr-psu-epyc \
  --export=ALL,TARGET=120,IDLE=300,STEP=300,REPS=10,REPS_BY="simple=1 serial=3" \
  -o bench/results/power-psu-epyc/epyc/job.log bench/power/power_bench.sh cpu bench/results/power-psu-epyc/epyc
```

```bash
bench/.venv/bin/python bench/power/analyze.py bench/results/power-psu-epyc
```

The raw files are in [epyc/](epyc/): `markers.txt`, `machine.txt`,
`samples.csv.gz` (pwrlog.py) and `out/` (./tree's output per process).
