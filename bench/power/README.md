# Power and energy measurement on mordor3

This folder holds a survey of the power sensors on the mordor3 nodes, done on
2026-11-04 on the free nodes ampere, epyc1, epyc2, dell3 and dell4 (dell1, dell2
and hopper were busy). It also holds a plan for the paper's energy measurements.

- `pwrlog.py`: the sampler. It reads every power source a normal user can read,
  at a fixed interval, and writes CSV.
- `probe-2026-11-04/`: the probe script with its output for each node, and two
  short experiments (`exp.sh`, `exp2.sh`) with their raw samples. Jobs 65–75.
- `power_bench.sh` and `analyze.py`: the energy benchmark of the five
  implementations (SLURM job per node, then the analysis and charts).
- **`results/BENCHMARK.md`: the results** (ECBDL'14 15M, depth 10, jobs 84–86):
  time, power and energy per tree on ampere, dell4 and epyc1, with charts.
- **`results-ampere/BENCHMARK.md`: all five implementations on ampere** (job 99,
  2026-11-05): the same node and sensors for every implementation; one chart
  per algorithm with energy, power (CPU and GPU parts) and time.
- `ADMIN-REQUEST.md`: what the admins need to set up so that users can read
  CPU and PSU power.

## What each node offers

| Source | What it measures | ampere (Supermicro, 2× EPYC 7402, 4× A100) | epyc1/2 (Dell R6725, 2× EPYC 9255) | dell1–4 (Dell R660, 2× Xeon 6542Y) | User access now |
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

## What needs sudo (request for the admins)

[ADMIN-REQUEST.md](ADMIN-REQUEST.md) is the request to send: step-by-step
commands that make CPU power (RAPL, all nodes) readable by users, and a small
root service that publishes ampere's and hopper's PSU power (IPMI) in a
world-readable file. Root is needed once, for the setup; measuring then needs
no sudo. GPU power (NVML) and the Dell nodes' PSU power already work. RAM
power is out of scope (only the Intel dell nodes have a sensor for it).

## Plan for the paper's measurements

**What to report, for each implementation × dataset × node:**

1. **Node energy to train one tree (J), measured at the PSU.** This is the
   primary metric because it includes everything the machine burns. On the Dell
   nodes, use the ACPI meter. On ampere, use IPMI once item 2 is granted. Until
   then, report ampere as the sum GPU (NVML) + CPU (RAPL).
2. **Components:** CPU packages (RAPL), DRAM (RAPL; Intel/dell only), GPU
   (NVML energy counter, only the GPU the run uses, and only in GPU runs), and
   "rest" = node − the sum. On AMD, "rest" includes DRAM.
3. Both **total** energy and energy **above idle**. Idle is measured in the same
   job for 30 s just before the run. Total is the honest "energy to solution".
   Above idle shows what the algorithm itself costs. Also report mean power and
   train time, so that energy = power × time can be checked.

**How:**

- **Same harness for everyone.** `bench/run.py` already runs ./tree, sklearn,
  rpart and Weka J48 through adapters that time only the fit. Two changes:
  (1) each adapter prints the fit's start and end as Unix timestamps
  (`time.time()`, `Sys.time()`, `System.currentTimeMillis()`; for ./tree,
  either a timestamp line in its output or a timestamp taken after `load`);
  (2) `measure()` runs `pwrlog.py` during each run and integrates each source
  over the fit window. This replaces the nvidia-smi memory sampler, or runs
  next to it. The result is energy for training only, the same window as
  `train_seconds`, so data loading in Python, R or Java does not count.
- **Long windows.** The ACPI meter averages over 2 s (dell) or about 6 s (epyc).
  Each measured window must therefore be at least about 60 s, which keeps the
  edge error to a few percent. A fit shorter than that runs K times inside one
  process (an adapter `repeat=K` argument, the same mechanism as `warmup`), and
  the energy is divided by K. RAPL and NVML are fine down to ~1 s, but use the
  same windows for consistency.
- **Fixed conditions.** Exclusive node; a warm node, i.e. one untimed warm-up
  run so that fans and temperature are steady; data already in the page cache;
  the same thread count as in the time benchmarks; NUMA pinning as in
  `submit.sh -n`. Interleave implementations (A B C A B C …) for at least 5
  repetitions, and report the median with an interquartile or 95% interval.
- **Nodes:** follow the existing rule. CPU implementations (./tree
  serial/parallel, sklearn, rpart, Weka) run on epyc1 and dell1. ./tree CUDA
  and gpu-only run on ampere (and hopper). For a cross-node "energy per tree"
  chart, node-level PSU energy is the only number that is comparable between
  CPU-only and GPU servers. Mention that ampere's PSU number includes three
  idle A100s (about 140 W); a "GPU in use + CPU" number can be shown next to it.
- **Library choice:** use our own sampler instead of CodeCarbon or similar.
  CodeCarbon uses the same RAPL and NVML data, but it models RAM as a fixed
  W/GB and cannot read the iDRAC. A transparent sampler is easier to defend in
  the paper.
