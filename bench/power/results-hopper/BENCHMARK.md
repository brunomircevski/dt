# Power and energy of the five implementations on hopper

This experiment measures the time and energy that ./tree's five implementations
need to train one tree on ECBDL'14 15M (15 M rows, 631 features). Settings:
depth ≤ 10, no pruning, CART (gini) and C4.5 (gain ratio). All five ran in one
SLURM job on hopper (job 300, 2026-11-08, 14:34–19:56).

Since 2026-11-06 a user can read hopper's CPU packages (RAPL) and its PSU
(set up by the cluster's admins), and NVML gives the GPU. So every
implementation is measured three ways at once: CPU packages, the GPU it used,
and the whole server.

This run replaces job 137 (2026-11-06). Job 137 trained a process's trees one
after another in the same process, and its later GPU trees were slower than a
single run (GPU 3.6 s against 2.97 s). Since 2026-11-07, `--repeat` trains
every tree in its own forked child, and a GPU child keeps its CPU threads busy
for 15 s before its timer starts (`--repeat-warmup 15`), as the CSV load does
before a single run's tree. Each tree is now the only tree of a fresh process,
every tree is measured, and the times match the time benchmark (see Time
check).

A first rerun (job 224, 2026-11-07) used a 2 s warm-up, which is enough on
ampere. On hopper it was not: in every GPU and Hybrid process the first 2–11
trees took 3.0 s and the rest 3.2–3.3 s (+8%). A test (job 228, 20 GPU trees
in a row) gave 3.21 s from the fifth tree on with 2 s, and 2.965–2.972 s for
every tree with 15 s and 40 s. nvidia-smi showed no heat throttling (≤ 57 °C,
SM clock within 1% of 1755 MHz), so the cause is on the host side.

| Implementation | Option | Resources | Runs × trees measured |
|---|---|---|---|
| Serial-Simple | `--simple` | 1 core | 1 × 1 |
| Serial-Opt | `--serial` | 1 core | 1 × 1 |
| Parallel ("Multicore" in the paper) | `--parallel` | all 32 cores | 3 × 11 (CART), 3 × 10 (C4.5) |
| GPU | `--cuda --gpu-only` | one H100 (GPU 2) + the 16 cores of its NUMA node (CPUs 16–31, memory node 1) | 3 × 42 |
| Hybrid | `--cuda` | same | 3 × 42 (CART), 3 × 41 (C4.5) |

Code: main at a2891e7 (forking `--repeat`, `--repeat-warmup`), built on hopper
on 2026-11-07 for sm_80 and sm_90 (`~/dt-wt/tree`, `~/dt-wt/tree_cpu`). Every
run built the same tree: CART 1,863 nodes, C4.5 725 nodes, depth 10.

## What was measured

| Sensor | What it covers | Readout | Refresh |
|---|---|---|---|
| ACPI power meter (`hwmon power_meter`) | whole server, PSU input | 0.1 s | ~1 s |
| RAPL `package-0`, `package-1` (powercap `energy_uj`) | each CPU socket | 0.1 s | ~1 ms (hardware) |
| NVML energy counter of GPU 2 | the GPU the run used; GPU and Hybrid runs only | 0.1 s | ~100 ms |

"Rest of the server" in the charts = PSU − both CPU packages − the GPU counted:
memory, fans, disks, network, the board, PSU losses and the rest of the
hardware. `/run/power/psu_W` (the BMC reading) is logged too; it agrees with
the ACPI meter, which the analysis uses.

Supermicro SYS-741GE-TNRT, 2× Xeon Gold 6444Y (16 cores each), H100 PCIe 80 GB,
1 TB RAM. CPU frequency governor: `intel_cpufreq schedutil`. Idle power (mean
of three windows: 300 s at the start, 300 s after the step load, 300 s at the
end): **PSU 403 W**, CPU packages 116 W (63 + 53 W), GPU 2 49 W. The PSU meter
lags the load by **1.8 s**, measured on a 300 s all-core step.

## Protocol

[../power_bench.sh](../power_bench.sh) `all` with `GPU=2 GPU_CPUS=16-31
GPU_MEM=1 GPU_THREADS=16 TARGET=120 IDLE=300 STEP=300 REPS=3
REPS_BY="simple=1 serial=1" WARMUP=15`. [../pwrlog.py](../pwrlog.py) samples
every sensor every 0.1 s for the whole job.

1. **Warm-up.** Parallel, GPU and Hybrid train two trees per algorithm. The
   second tree's time sets K = ceil(120 s / t) + 1 trees per process: Parallel
   11 (CART) and 10 (C4.5), GPU 42, Hybrid 42 and 41.
2. **Idle, step, idle.** 300 s idle, 300 s with every core busy (to measure the
   meter's delay), 300 s idle.
3. **Runs.** Round-robin over algorithm × implementation, 20 s of idle before
   each process. Serial-Simple and Serial-Opt ran once, Parallel, GPU and
   Hybrid 3 times. A process loads the data once, then forks one child per tree.
4. **Idle.** 300 s at the end.

Every number is per tree, over every tree. A tree's window runs from the start
of its training to the end of pruning, before its memory is freed; the gaps
between children (fork, CUDA setup, warm-up) are left out. The PSU energy is
its readings held until the next one, shifted back by the meter's delay. Every
measured case has at least 103 PSU changes.

## Time check

The time per tree matches the time benchmark
([../../results/cluster-ecbdl](../../results/cluster-ecbdl), single runs on the
same node, GPU and settings) within 1.4%, and the first tree of each process
matches the rest:

| Implementation | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|
| Parallel | CART | 12.4 s | 12.4 s | – |
| Parallel | C4.5 | 14.9 s | 14.9 s | – |
| GPU | CART | 2.99 s | 3.00 s | 2.97 s (+0.9%) |
| GPU | C4.5 | 3.03 s | 3.01 s | 2.99 s (+1.2%) |
| Hybrid | CART | 3.01 s | 3.01 s | 2.97 s (+1.4%) |
| Hybrid | C4.5 | 3.02 s | 3.02 s | 2.99 s (+1.0%) |

## Results

### CART (gini)

| Implementation | t per tree | PSU power (above idle) | CPU packages | GPU | PSU energy per tree | above idle | CPU packages | GPU | rest of the server |
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | 2,520 s | 540 W (+137) | 238 W | – | 1.36 MJ | 346 kJ | 600 kJ | – | 761 kJ |
| Serial-Opt | 341 s | 541 W (+139) | 238 W | – | 185 kJ | 47.3 kJ | 81.0 kJ | – | 104 kJ |
| Parallel | 12.4 s ± 0.3% | 901 W (+499) | 534 W | – | 11.2 kJ ± 0.3% | 6.18 kJ | 6.61 kJ | – | 4.56 kJ |
| GPU | 2.99 s ± 0.1% | 718 W (+316) | 202 W | 284 W | 2.15 kJ ± 0.4% | 0.95 kJ | 0.60 kJ | 0.85 kJ | 0.70 kJ |
| Hybrid | 3.01 s ± 0.4% | 790 W (+387) | 260 W | 284 W | 2.38 kJ ± 0.6% | 1.16 kJ | 0.78 kJ | 0.85 kJ | 0.74 kJ |

### C4.5 (gain ratio)

| Implementation | t per tree | PSU power (above idle) | CPU packages | GPU | PSU energy per tree | above idle | CPU packages | GPU | rest of the server |
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | 2,757 s | 540 W (+137) | 238 W | – | 1.49 MJ | 378 kJ | 655 kJ | – | 833 kJ |
| Serial-Opt | 415 s | 538 W (+135) | 237 W | – | 223 kJ | 56.0 kJ | 98.4 kJ | – | 124 kJ |
| Parallel | 14.9 s ± 0.1% | 898 W (+496) | 531 W | – | 13.4 kJ ± 0.1% | 7.40 kJ | 7.93 kJ | – | 5.48 kJ |
| GPU | 3.03 s ± 0.3% | 720 W (+317) | 202 W | 286 W | 2.18 kJ ± 0.3% | 0.96 kJ | 0.61 kJ | 0.87 kJ | 0.70 kJ |
| Hybrid | 3.02 s ± 0.4% | 797 W (+394) | 264 W | 286 W | 2.41 kJ ± 0.4% | 1.19 kJ | 0.80 kJ | 0.86 kJ | 0.75 kJ |

± is one standard deviation over the runs (none for the single serial runs).
Full tables, with peaks and both sockets: [summary.md](summary.md),
[summary.csv](summary.csv), every run in [runs.csv](runs.csv).

## Charts

- **Energy, power and time in one chart:** [CART](chart-cpu-gpu-hopper-cart.png),
  [C4.5](chart-cpu-gpu-hopper-c45.png). Per implementation: energy per tree (CPU
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

![C4.5](chart-cpu-gpu-hopper-c45.png)

## Findings

- **GPU needs the least energy per tree:** 2.15–2.18 kJ at the PSU, 5.2×
  (CART) and 6.1× (C4.5) less than Parallel, 86–102× less than Serial-Opt and
  630–680× less than Serial-Simple.
- **Hybrid takes the same time as GPU (within 1%) and 11% more energy:** its
  16 CPU threads add ~60 W on the CPU packages (260 vs 202 W) and ~75 W at the
  PSU for no gain in time.
- **The whole server matters more than the GPU.** In a GPU tree the GPU is
  40% of the PSU energy (0.85 of 2.15 kJ); the CPU packages are 28% and the
  rest of the server 32%.
- **hopper is an expensive host for CPU-only work:** its 403 W idle is 2.6×
  dell4's and 3.5× epyc1's. Parallel needs 11.2 kJ per CART tree here against
  8.54 kJ on dell4 and 3.83 kJ on epyc1
  ([../results-dell/BENCHMARK.md](../results-dell/BENCHMARK.md),
  [../results-epyc/BENCHMARK.md](../results-epyc/BENCHMARK.md)); for the serial
  runs the idle cost dominates (56% of the energy is the rest of the server).
- **The H100 beats ampere's A100:** a GPU tree takes 2.99 s and 2.15 kJ here,
  against 5.05 s and 3.11 kJ on ampere
  ([../results-ampere/BENCHMARK.md](../results-ampere/BENCHMARK.md)).
- **Against job 137:** GPU trees are 17% faster (2.99 against 3.61 s) and need
  19% less energy (2.15 against 2.64 kJ), Hybrid 7% faster and 13% less;
  job 137 measured slowed-down later trees. Parallel and the serial runs agree
  within 2–3%.

## Reproduce

On mordor3, from `~/dt`:

```bash
sbatch -p hopper -w hopper --exclusive -c 32 -t 14:00:00 -J pwr-psu-hopper \
  --export=ALL,GPU=2,GPU_CPUS=16-31,GPU_MEM=1,GPU_THREADS=16,TARGET=120,IDLE=300,STEP=300,REPS=3,REPS_BY="simple=1 serial=1",WARMUP=15 \
  -o bench/results/power-psu-hopper/hopper/job.log bench/power/power_bench.sh all bench/results/power-psu-hopper/hopper
```

Then, on the PC, after copying the job folder:

```bash
bench/.venv/bin/python bench/power/analyze.py bench/power/results-hopper
```

The raw files are in [hopper/](hopper/): `markers.txt`, `machine.txt`,
`samples.csv.gz` (pwrlog.py) and `out/` (./tree's output per process).
`job.log` is empty.
