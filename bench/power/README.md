# Power and energy measurement on mordor3

How much energy each of ./tree's five backends uses to train one tree
(ECBDL'14 15M, depth ≤ 10, no pruning), on the four kinds of server:
epyc1, dell4 (CPU only), ampere (A100) and hopper (H100).

- **`results/BENCHMARK.md`: the results** (2026-11-16, code f53347a9, jobs
  169, 170, 173, 174): time, power and energy per tree, split into CPU
  packages, DRAM, the GPU in use and the rest of the server, with the checks
  and charts. `results/summary.csv` feeds the paper's energy figure
  (`paper/tools/charts.py energy`).
- `power_bench.sh`: one SLURM job per node; `submit.sh` submits them;
  `analyze.py`: the analysis, tables and charts; `pwrlog.py`: the sampler.
- `probe-2026-11-04/`: the survey of the power sensors (jobs 65–75).

The method is below ([Method](#method)); the sensors each node offers first.

## What each node offers

| Source | What it measures | ampere (Supermicro, 2× EPYC 7402, 4× A100) | epyc1/2 (Dell R6725, 2× EPYC 9255) | dell1–4 (Dell R660, 2× Xeon 6542Y) | User access on 2026-11-04 (see Access below for today) |
|---|---|---|---|---|---|
| ACPI power meter (`hwmon power_meter`, which reads the iDRAC) | **Whole server, PSU input (AC) power** | not present | yes, but it changes only every **4–6 s** | yes, it changes every **2.1 s** | **yes**, nothing to install |
| IPMI / BMC (`ipmitool dcmi power reading`, PSU sensors) | Whole server; per-PSU input on most BMCs | `/dev/ipmi0`, ipmitool installed | `/dev/ipmi0`, no ipmitool | `/dev/ipmi0`, no ipmitool | no: `/dev/ipmi0` is root-only (0600) |
| RAPL package (powercap or `perf power/energy-pkg/`) | CPU socket: cores + uncore (+ I/O die on AMD) | yes, per socket | yes, per socket | yes, per socket | **ampere only** (perf_event_paranoid=0). Elsewhere paranoid=4 and `energy_uj` is mode 0400 root |
| RAPL core (`power_core` PMU / `core` zone) | Cores only | yes | yes | no | ampere only |
| RAPL DRAM | DIMMs | **no** (AMD has no DRAM domain) | **no** | yes, per socket | no (root) |
| RAPL psys | Platform | no | no | yes (zone `intel-rapl:1`) | no (root) |
| NVML (`nvmlDeviceGetPowerUsage`, `nvmlDeviceGetTotalEnergyConsumption`) | GPU board power and an energy counter in mJ | yes; it changes every ~100 ms | — | — | **yes** |
| turbostat, cpupower | turbostat reads RAPL through MSRs | installed | installed | installed | turbostat needs root |
| likwid, DCGM, AMD uProf, scaphandre | — | not installed | not installed | not installed | — |

Notes:

- **The PSU total is already readable on all Dell nodes.** On dell1–4 and
  epyc1–2, `/sys/class/hwmon/hwmon*/power1_average` (name `power_meter`) is
  world-readable. It is the iDRAC's system input power, read through ACPI-IPMI.
  It reports "accuracy 90%" and an averaging interval of 2000 ms. On the R660
  (dell) the value changes every 2.1 s. On the R6725 (epyc) it changes only
  every 4.2–10.5 s (median 6.3 s), even though it also claims 2000 ms.
  Ampere (Supermicro) has no ACPI power meter, so its PSU power is only
  available through IPMI, which needs root.
- **Ampere's GPUs:** at idle each A100 draws 42–54 W, so the four together draw
  about 195 W. A node-level number for a run that uses one GPU therefore
  contains about 140 W from the three idle GPUs. The NVML energy counter
  updates every ~100 ms. Caveat for the paper: on A100/H100, NVML samples the
  power sensor only part of the time (it averages windows). Measure runs of at
  least several seconds and repeat them; never use a single short kernel.
- **AMD has no DRAM sensor.** On ampere and epyc, memory power can only be
  estimated as the remainder: node − CPU packages − GPUs. That remainder also
  contains fans, PSU losses, NICs and disks.
- The governors differ between nodes: dell is intel_pstate/powersave, epyc is
  amd-pstate-epp/performance and ampere is acpi-cpufreq/ondemand. This changes
  both idle power and active power. Users cannot change it, so report the
  governor next to each number.

## What the short experiments showed

These were HIGGS runs with `-m` for longer training. Each phase includes the
CSV load. Idle power was measured in the same job just before the run. All
values are ACPI node power unless marked otherwise.

| Node | Run | Phase length | Mean node power | Above idle | Energy (total / above idle) |
|---|---|---|---|---|---|
| dell4 (idle 150 W) | parallel CART `-m 10` | 13.2 s | 497 W | 347 W | 6.5 / 4.6 kJ |
| dell4 | parallel C4.5 `-m 10` | 22.0 s | 455 W | 305 W | 10.0 / 6.7 kJ |
| dell4 | serial CART `-m 2` | 39.6 s | 303 W | 153 W | 12.0 / 6.1 kJ |
| epyc1 (idle 110 W) | parallel CART `-m 10` | 11.1 s | 261 W | 151 W | 2.9 / 1.7 kJ |
| epyc1 | parallel C4.5 `-m 10` | 10.6 s | 266 W | 157 W | 2.8 / 1.7 kJ |
| epyc1 | serial CART `-m 2` | 24.5 s | 221 W | 111 W | 5.4 / 2.7 kJ |
| ampere, GPU0 (NVML) | cuda CART `-m 20` | 26.4 s | GPU0 138 W, RAPL S0 104 W / S1 61 W | GPU0 +86 W | GPU0 3.6 kJ |
| ampere, GPU0 (NVML) | gpu-only C4.5 `-m 20` | 47.5 s | GPU0 177 W (peak 322 W), RAPL S0 85 W / S1 60 W | GPU0 +125 W | GPU0 8.4 kJ |

Ampere's idle readings were GPU0 52 W and RAPL 59 + 58 W.

Takeaways:

- **Serial runs are not cheap per joule.** One busy thread on dell raises node
  power by about 150 W, because the package, uncore and fans wake up. That is
  half of what 48 threads add. Fast parallel or GPU runs should therefore win
  clearly on energy to solution, and this is the argument for the paper.
- **The epyc meter is too coarse for windows of a few seconds.** In the first
  experiment, a 7 s C4.5 run on epyc2 registered only +28 W, because the reading
  had not caught up. Every measured window must be long (see below).

## Access

Since 2026-11-06 a normal user can read every source above: the admins made
the RAPL counters (CPU power, all nodes) readable and publish ampere's and
hopper's PSU power (IPMI) in a world-readable file. GPU power (NVML) and the
Dell nodes' PSU power needed nothing. RAM power is out of scope (only the
Intel dell nodes have a sensor for it).

## Method

### The problem: the whole-server meters are slow

The number that compares a CPU server with a GPU server is the energy of the
whole server, measured at the power supply (PSU). Every node has such a
meter, but none is fast enough for a tree that takes 2.4 s (H100), 4.3 s
(A100) or 4.5 s (Multicore on epyc1):

| Node | PSU meter | refresh | what a reading is |
|---|---|---|---|
| hopper | ACPI power meter (Intel Node Manager) | 1 s | an average over ~1.25 s |
| dell4 | ACPI power meter (iDRAC) | 2.1 s | an average over ~2.5 s |
| epyc1 | ACPI power meter (iDRAC) | 4–10 s (median 6.3 s) | an average over ~2.5 s |
| ampere | BMC (IPMI DCMI, `/run/power/psu_W`) | 59 s | one point reading (~2.5 s average) |

(The averaging windows are fitted, see below.) The fast sensors, RAPL
(CPU packages and, on the Intel nodes, DRAM) and NVML (each GPU's energy
counter), cover only parts of the server: not the fans, disks, network,
board, the other GPUs or the PSU's own losses (on hopper about 30% of the
energy of a GPU tree).

### The solution: model the meter from the fast sensors

`pwrlog.py` reads every sensor every 0.1 s for the whole job. For each node,
`analyze.py` then fits the meter itself:

    reading(t) = mean over [t − D − W, t − D] of  a + b·CPU + c·GPUs + d·busy

where CPU = RAPL packages + DRAM (W), GPUs = the NVML power of every GPU
(W), busy = the number of busy CPUs (`/proc/stat`), D is the meter's delay
and W its averaging window. a, b, c, d come from least squares over **every
reading of the job** (idle, a staircase of CPU load, every run, the gaps); D
and W from a grid search (D in 0.1 s steps; W from 0.1 to 10 s). The PSU
energy of a tree is then the integral of a + b·CPU + c·GPUs + d·busy over
its training window, at 0.1 s resolution, whatever the meter's refresh.
"Rest of the server" = PSU − CPU packages − DRAM − the GPU in use.

The job gives the fit the range of loads it needs: idle; a staircase of
1, 2, 4, …, all CPUs busy (60 s per step; 240 s on ampere, so that its
once-a-minute meter samples each step); all five backends; and, on the GPU
nodes, GPU and Hybrid runs with their CPU warm-ups.

**Checks** (`results/BENCHMARK.md`, `chart-meter.png`, `chart-methods.png`):

1. The coefficients fitted on the first and on the second half of the job
   alone, and the energy each half's model gives.
2. The mean residual (reading − model) of the readings whose averaging
   window lies inside a case's training windows: a bias of the model during
   that backend's trees would show here.
3. The meter's own energy over the same windows (readings held, shifted back
   by D + W/2; ampere: the mean of its point readings in the windows). It
   needs windows much longer than the meter's refresh: it agrees with the
   model within ~1.5% on every long window, and on 2–3 s GPU trees it is
   3–5% low (re-analysis of the 2026-11-08 hopper job), which is the error the
   model removes.

Why busy CPUs: on epyc1 (AMD) the PSU rises more with one busy core than
RAPL accounts for. Without the term the model was 1.5–2.3% low on the
single-thread runs (re-analysis of the 2026-11-07 epyc job); with it, ≤ 0.8%.

### The runs

Each measured process is `./tree --repeat K`: it loads the CSV once, then
trains K trees, each in its own forked child, so that every tree is a fresh
process's only tree, and prints each tree's training window as Unix times.
K = ceil(TARGET / t) + 1 (t from a warm-up run): about 120 s of training per
process (240 s on ampere); 3 processes per case, round-robin, 20 s of idle
before each (Serial-Simple: one process of one tree, 40–90 min).

**Warm-up of the GPU backends.** A GPU or Hybrid tree whose CPU threads were
idle before it sets up slower: on ampere (current code, the warm-up check in
`results/`) trees 3–12 of a process took 5.1–5.5 s instead of 4.3 s (hopper: 2.8 s instead of 2.5 s). A single
run's tree follows the CSV load, which keeps every core busy, so with
`--repeat-warmup S` each child spins its threads for S s before its timer: 2 s
is enough on ampere; hopper needs 5 s (with 2 s its trees were 3–10% slower)
and the measured runs used 15 s.
The warm-up is outside the training window, so it is never counted.

**Placement**, as in the time benchmark (`bench/cluster/job.sh`,
`bench/run.py`): Serial-Simple and Serial-Opt on CPU 0; Multicore on every
CPU; GPU and Hybrid on the GPU of the NUMA node with the most free memory,
with every CPU of the node (that node's first) and first-touch memory.

**Time check**: every case's time per tree against the time benchmark
(`bench/results/cluster-ecbdl`, single runs).

Run it (from the repository root, on the login node, with ./tree and
./tree_cpu built by `bench/cluster/build.sh`):

    bench/power/submit.sh -c ampere hopper            # the warm-up check (optional)
    bench/power/submit.sh epyc1 dell4 ampere hopper   # one job per node: 4-8 h each
    bench/.venv/bin/python bench/power/analyze.py bench/power/results

### Earlier runs

Jobs 84–86 (2026-11-05) and 217, 218, 223, 300 (2026-11-07/08) measured the
code of that week with the meters' own readings over the tree windows (long
windows of many trees, readings shifted by a step-measured delay). They are in
the git history.
