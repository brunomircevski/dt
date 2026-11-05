# Power and energy of the five implementations, all on ampere

This experiment measures the time and energy that ./tree's five implementations
need to train one tree on ECBDL'14 15M (15 M rows, 631 features). Settings:
depth ≤ 10, no pruning, CART (gini) and C4.5 (gain ratio). All five ran on **one
node**, ampere, in one SLURM job (job 99, 2026-11-05, 09:20–14:23). ampere is the
only mordor3 node where a user can read both CPU power (RAPL) and GPU power
(NVML), so here every implementation is measured with the same sensors and the
same idle reading. The first round
([../results/BENCHMARK.md](../results/BENCHMARK.md)) ran the CPU
implementations on dell4 and epyc1 instead, where only the PSU is readable.

The user allowed this one job as an exception to the rule "no CPU benchmarks on
ampere" (2026-11-05).

| Implementation | Option | Resources | What is measured |
|---|---|---|---|
| Serial-Simple | `--simple` | 1 core | both CPU packages |
| Serial-Opt | `--serial` | 1 core | both CPU packages |
| Parallel ("Multicore" in the paper) | `--parallel` | all 48 cores (`-c 48`) | both CPU packages |
| GPU | `--cuda --gpu-only` | the GPU (A100 80 GB) + the 24 cores of its NUMA node | both CPU packages + the GPU |
| Hybrid | `--cuda` | same | both CPU packages + the GPU |

Code: a clone of GitHub main at f941d7d9 (`src/` as on main that day), built on
ampere on 2026-11-05 with `bench/cluster/build.sh`. Every run built the same
tree: CART 1,863 nodes, C4.5 725 nodes, depth 10.

## What was measured

| Node | Hardware | Sensors | Readout |
|---|---|---|---|
| ampere | Supermicro AS-4124GS-TNR, 2× EPYC 7402 (Zen 2, 24 cores each), A100 80 GB, 1 TB RAM | both CPU packages (RAPL `power/energy-pkg/` via `perf stat`) + the GPU the run used (NVML energy counter) | 0.1 s |

- **CPU implementations:** the two CPU packages only. No GPU is counted, because
  these runs do not use one.
- **GPU and Hybrid:** the two CPU packages plus the GPU they ran on.
- **Not measured:** RAM, fans, the board, disks and PSU losses, because ampere
  has no PSU meter a user can read. On AMD, the package includes the I/O die,
  which explains most of the 59 W idle per socket.

CPU frequency governor: `acpi-cpufreq ondemand`, as the admins left it. Idle
power (two 60 s windows, at the start and the end of the job): **CPU packages
118 W** (59 + 59 W) and **GPU 54.5 W**. So the idle baseline is 118 W for the
CPU implementations and 172 W for GPU and Hybrid.

## Protocol

[../power_bench.sh](../power_bench.sh) `all` with `TARGET=30 REPS=5
REPS_BY="simple=1 serial=3"`. [../pwrlog.py](../pwrlog.py) samples NVML every
0.1 s for the whole job, and `perf stat -I 100` does the same for RAPL.

1. **Warm-up.** Parallel, GPU and Hybrid each train two trees per algorithm. The
   second tree's time sets K, the number of trees per process, so that a
   process trains for at least 30 s: K = 2 for Parallel and K = 5 for GPU and
   Hybrid.
2. **Idle.** 60 s at the start and 60 s at the end. The idle power is their
   mean.
3. **Runs.** Round-robin rounds over algorithm × implementation, with 20 s of
   idle before each process. Serial-Simple ran once (82 min CART, 94 min
   C4.5), Serial-Opt 3 times, and Parallel, GPU and Hybrid 5 times each.
4. **Which tree counts.** Only the **first** tree of each process: that is the
   tree the time benchmarks time. On the GPU, later trees in the same process
   are slower (about 7 s instead of 5 s), because `gpu setup` uploads the data
   again. They are listed separately below.
5. **Integration** ([../analyze.py](../analyze.py)) over each tree's printed
   window: the GPU's NVML energy counter is interpolated at the window's edges;
   RAPL joules are weighted by how much of each `perf` interval falls inside the
   window. The peak is the highest 0.1 s sample inside the window.

The spread between runs is 0.3–0.5% for time and 0.2–1.5% for energy on the fast
implementations. For Serial-Opt CART it is 4% (617–665 s).

### A sampler bug found and fixed in this run

NVML calls sometimes block for about 2 s. On ampere this happens every time a
GPU tree ends. pwrlog.py took the timestamp **before** its reads, so the
counter of the sample taken right after a GPU tree was read about 2 s later
than its timestamp. Every GPU window ends just before such a sample, so each
GPU and Hybrid tree got about 2 s of extra GPU energy: the GPU's energy for a
CART tree read 1.24 kJ instead of 1.10 kJ (+13%). The fix:

- `analyze.py` now times the counters with a corrected read time: a sample
  followed by a gap of more than 0.3 s was read just before the next sample.
- `pwrlog.py` now also writes `t_end`, the time after the reads, which
  analyze.py uses when it is there.

With the fix, the energy spread between runs fell from 1.4–4.1% to 0.2–1.5%. The
first round on ampere had the same error. It is reanalysed with the fix and
corrected in [../results/BENCHMARK.md](../results/BENCHMARK.md). Both rounds now
agree within 1% (GPU CART 1.81 kJ there, 1.80 kJ here). The dell and epyc PSU
numbers did not change.

## Results

`± x%` is the standard deviation over the runs, as a percentage of the mean.
"Above idle" subtracts the idle power of the measured components over the same
window. The full tables are in [summary.md](summary.md) and
[summary.csv](summary.csv), and every run is in [runs.csv](runs.csv).

### CART (gini)

| Implementation | Runs | Time per tree | Mean power | Above idle | Peak | Energy per tree | Energy above idle |
|---|---|---|---|---|---|---|---|
| Serial-Simple | 1 | 4,941 s | 125 W | 7 W | 153 W | 618 kJ | 35.6 kJ |
| Serial-Opt | 3 | 636 s ± 4.0% | 126 W | 9 W | 183 W | 80.4 kJ ± 5.0% | 5.45 kJ |
| Parallel | 5 | 15.4 s ± 0.3% | 307 W | 189 W | 320 W | 4.82 kJ ± 0.2% | 2.97 kJ |
| GPU | 5 | 5.01 s ± 0.3% | 354 W | 182 W | 462 W | 1.80 kJ ± 0.7% | 924 J |
| Hybrid | 5 | 5.09 s ± 0.3% | 358 W | 186 W | 512 W | 1.94 kJ ± 0.4% | 1.01 kJ |

### C4.5 (gain ratio)

| Implementation | Runs | Time per tree | Mean power | Above idle | Peak | Energy per tree | Energy above idle |
|---|---|---|---|---|---|---|---|
| Serial-Simple | 1 | 5,656 s | 125 W | 7 W | 173 W | 706 kJ | 39.2 kJ |
| Serial-Opt | 3 | 717 s ± 0.3% | 125 W | 8 W | 157 W | 90.0 kJ ± 0.2% | 5.43 kJ |
| Parallel | 5 | 27.7 s ± 0.3% | 293 W | 175 W | 320 W | 8.22 kJ ± 0.3% | 4.92 kJ |
| GPU | 5 | 5.06 s ± 0.5% | 352 W | 180 W | 464 W | 1.81 kJ ± 1.5% | 926 J |
| Hybrid | 5 | 5.12 s ± 0.3% | 358 W | 186 W | 521 W | 1.95 kJ ± 0.6% | 1.01 kJ |

### CPU and GPU parts of the GPU runs (per tree)

| Implementation | Algorithm | CPU packages: mean power | CPU energy (above idle) | GPU: mean power | GPU energy (above idle) |
|---|---|---|---|---|---|
| GPU | CART | 138 W | 699 J (101 J) | 217 W | 1.10 kJ (823 J) |
| Hybrid | CART | 152 W | 822 J (183 J) | 206 W | 1.12 kJ (823 J) |
| GPU | C4.5 | 138 W | 708 J (102 J) | 215 W | 1.11 kJ (824 J) |
| Hybrid | C4.5 | 152 W | 826 J (184 J) | 207 W | 1.13 kJ (829 J) |

### Later trees in the same process

| Implementation | Algorithm | First tree | Later trees (mean) |
|---|---|---|---|
| Parallel | CART | 15.4 s, 4.82 kJ | 15.4 s, 4.83 kJ |
| Parallel | C4.5 | 27.7 s, 8.22 kJ | 27.9 s, 8.29 kJ |
| GPU | CART | 5.01 s, 1.80 kJ | 7.13 s, 2.20 kJ |
| GPU | C4.5 | 5.06 s, 1.81 kJ | 7.17 s, 2.22 kJ |
| Hybrid | CART | 5.09 s, 1.94 kJ | 6.92 s, 2.28 kJ |
| Hybrid | C4.5 | 5.12 s, 1.95 kJ | 6.95 s, 2.29 kJ |

## Charts

### Energy, power and time in one chart — [CART](chart-cpu-gpu-ampere-cart.png), [C4.5](chart-cpu-gpu-ampere-c45.png)

![CART: energy, power, time](chart-cpu-gpu-ampere-cart.png)

![C4.5: energy, power, time](chart-cpu-gpu-ampere-c45.png)

The summary charts, one per algorithm. Each implementation has three bars,
labelled under the axis, and all three axes are log scale:

- **energy** (left bar, left axis): energy per tree in J;
- **power** (middle bar, first right axis): mean power while training in W.
  The dotted line is the peak, the highest 0.1 s sample;
- **time** (right bar, outer right axis): time per tree in s.

The energy and power bars are stacked in the same order, from the bottom:
1. CPU packages, idle part;
2. GPU, idle part;
3. CPU packages, above idle;
4. GPU, above idle.

The CPU implementations have no GPU parts. On a log axis, a plain stack would
squash the upper parts, so each part's height is drawn as its share of the bar.
The labels give the totals: energy and energy above idle; mean power, its CPU
and GPU parts, and the peak.

What it shows:
- **The CPU on GPU runs:** the CPU packages draw 138 W with GPU-only (20 W above
  idle) and 152 W with Hybrid (34 W above idle), against 293–307 W with
  Parallel. The GPU draws 206–217 W, up from 54.5 W idle.
- **Serial runs** add only 7–9 W to the CPU packages' 118 W idle draw: their
  energy is mostly idle power × time.

### Mean power while training — [chart-power.png](chart-power.png)

![Mean power](chart-power.png)

Mean power per component on a linear axis, each component's idle part lighter,
the peak dotted. It shows the same numbers as the tables, with the two CPU
sockets apart: Parallel loads both sockets equally (about 150 W each). On GPU
and Hybrid, only socket 0 (the GPU's NUMA node) rises, to 78 W and 92 W.

### Energy per tree — [chart-energy.png](chart-energy.png)

![Energy per tree](chart-energy.png)

Total energy (outer bar, ± sd) on a log scale, with the part above idle as the
inner bar. Each step saves energy (CART / C4.5):
- Serial-Simple → Serial-Opt: 7.7× / 7.8× less.
- Serial-Opt → Parallel: 17× / 11× less.
- Parallel → GPU: 2.7× / 4.5× less.
- Overall, GPU trains a tree with 343× / 389× less energy than Serial-Simple.

### Where the energy goes — [chart-energy-split.png](chart-energy-split.png)

![Energy split](chart-energy-split.png)

Each component's share of each run's energy, with the idle and above-idle
parts shown separately. On GPU runs, the CPU packages take 39% (GPU-only) and
42% (Hybrid) of the energy, most of it their idle draw.

### Training time — [chart-time.png](chart-time.png)

![Training time](chart-time.png)

Time per tree on a log scale. Speed-ups on ampere (CART / C4.5):
Serial-Simple → Serial-Opt 7.8× / 7.9×, Serial-Opt → Parallel 41× / 26×,
Parallel → GPU 3.1× / 5.5×.

### Time against energy — [chart-time-energy.png](chart-time-energy.png)

![Time vs energy](chart-time-energy.png)

Each case as a point on linear axes, with the fast implementations zoomed in
the inset. Grey lines are constant mean power. The serial runs sit at 125 W,
Parallel at about 300 W and GPU and Hybrid at about 355 W. Energy is almost
proportional to time.

### Power over time — [chart-traces.png](chart-traces.png)

![Power traces](chart-traces.png)

The first run of each case, from process start to end, with the measured tree
shaded. The GPU line appears only on GPU and Hybrid runs. It shows the CSV load
(both sockets at about 70 W), Parallel holding both sockets at 150–160 W, and
the GPU's bursts of up to about 330 W with CPU socket 0 following it.

## Conclusions

1. **Faster always used less energy.** Every speed-up lowered the energy per
   tree, from 706 kJ (Serial-Simple C4.5) to 1.81 kJ (GPU C4.5). Mean power only
   rises from 125 W (serial) to 300 W (Parallel) and 355 W (GPU), while time
   falls by up to 1,100×, so time decides the energy.
2. **Serial runs pay mostly for idle.** One busy Zen 2 core adds 7–9 W to the
   118 W the CPU packages draw idle.
3. **GPU vs Parallel on one node.** The GPU is 3.1× (CART) and 5.5× (C4.5)
   faster than Parallel on ampere's 48 Zen 2 cores, at 15–20% more power (CPU
   packages + GPU against CPU packages alone). So it uses 2.7× / 4.5× less
   energy, and 3.2× / 5.3× less above idle.
4. **Hybrid costs 8% more energy than GPU-only.** Hybrid's CPU work raises the
   CPU packages from 138 to 152 W, about 120 J more per tree, and it does not
   make the tree faster at this size (5.09 s against 5.01 s).
5. **The CPU is a large part of a GPU run.** The CPU packages take 39–42% of a
   GPU tree's energy, mostly their idle draw.
6. **ampere's CPUs are older.** ampere's 2019 Zen 2 CPUs are slower than the
   CPU nodes of the first round, so its CPU numbers here are not the best CPU
   result. The same tree takes:
   - Parallel CART: 15.4 s, against 6.3 s on epyc1 and 12.2 s on dell4;
   - Serial-Opt CART: 636 s, against 225 s and 356 s;
   - Serial-Simple CART: 4,941 s, against 2,309 s and 2,556 s.

## Limitations

- **CPU packages and the GPU only.** No RAM, fans, board, disks or PSU losses:
  the wall-plug energy is higher, and the missing part grows with time. So the
  serial implementations are flattered the most, being the slowest. The admin
  request ([../ADMIN-REQUEST.md](../ADMIN-REQUEST.md)) would add the IPMI PSU
  reading.
- **Different scopes.** The CPU implementations count no GPU, while GPU and
  Hybrid count their GPU's 54.5 W of idle draw too. This adds about 0.28 kJ to
  a GPU tree.
- **Serial-Simple ran once** (82 + 94 min), so it has no spread.
- **One dataset, depth 10, one node.** The GPU numbers are for the A100 80 GB.
  Hopper's H100 was not measured, because its node has no readable CPU power.
- **Governor `ondemand`.** It was left as the admins set it.

## Reproduce

From the repository root on the mordor3 login node (`./tree` built on ampere
with `bench/cluster/build.sh`; data `bench/data/ecbdl_15m/train.csv`):

```bash
mkdir -p bench/results/power-ampere/ampere
sbatch -p ampere -w ampere --exclusive -c 48 -t 9:00:00 -J pwr-all \
  --export=ALL,TARGET=30,REPS=5,REPS_BY="simple=1 serial=3" \
  -o bench/results/power-ampere/ampere/job.log bench/power/power_bench.sh all bench/results/power-ampere/ampere
```

`-c 48` matters: without it, `--exclusive` still gives the job 1 CPU, and
Parallel would run on one core. The analysis runs on a compute node:

```bash
srun -p epyc -w epyc2 -c 4 -t 0:20:00 bench/.venv/bin/python bench/power/analyze.py bench/results/power-ampere
```

`ampere/` holds:
- `machine.txt`: node, commit and `cpus available 48`;
- `markers.txt`: every phase as Unix times;
- `samples.csv.gz`: NVML and CPU-busy samples every 0.1 s;
- `perf_rapl.csv.gz`: `perf stat` energy per socket every 0.1 s;
- `out/`: ./tree's output, with each tree's training window.

`job.log` is empty. analyze.py counts only the GPU the run used (GPU 0, as
`power_bench.sh` assigns it), and only for GPU and Hybrid.
