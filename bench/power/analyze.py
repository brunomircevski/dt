#!/usr/bin/env python3
"""Energy of ./tree's backends from bench/power/power_bench.sh jobs: tables and charts.

  bench/.venv/bin/python bench/power/analyze.py bench/results/power  # every job folder in it

Every job folder (ampere/, dell/, epyc/, hopper/) holds markers.txt, samples.csv
(pwrlog.py), perf_rapl.csv (ampere: RAPL per socket through perf) and out/<run>.txt
(./tree's output with --repeat K training windows). RAPL comes from the sampled
powercap counters when samples.csv has them (ampere and hopper since 2026-11-06),
else from perf.

Per run (one process), over the span from the first tree's start to the last
tree's end, divided by the number of trees (energy and time per tree); with
K >= 3 trees the first one is left out (it is slower: CUDA context, first
allocations):
  - PSU (whole server): the ACPI meter (Dell nodes, hopper) or /run/power/psu_W
    (ampere: the BMC's reading). The readings are held until the next one and
    shifted back by the meter's delay. The delay is measured per job: from the
    all-core step (power_bench.sh STEP) when there is one, else the median time
    from a process's start to the reading half-way up its first step (the CSV
    load lifts power from idle within a fraction of a second). A slow meter
    (ampere's BMC takes a point reading about every 59 s) gets the mean of the
    readings it took inside the window, times the window's length.
    With CPU and GPU readings too, the PSU part not explained by them is the
    component "rest" (memory, fans, disks, network, the other GPUs, PSU losses).
  - GPU i: the NVML energy counter, interpolated at the span's ends.
  - CPU socket s: RAPL package energy from perf's 100 ms intervals, the part of
    each interval inside the span.
Idle power of each source: the job's two idle windows (before the first run and
after the last, skipping each window's first 5 s). "Above idle" = energy - idle
power x time.

Writes into the given folder: runs.csv (every run), summary.csv (mean and
standard deviation per case), summary.md (tables) and charts chart-*.png.
"""
import bisect
import csv
import gzip
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
IMPLS = {"simple": "Serial-Simple", "serial": "Serial-Opt", "parallel": "Parallel",
         "gpuonly": "GPU", "cuda": "Hybrid"}
IMPL_ORDER = list(IMPLS)
ALGOS = {"cart": "CART (gini)", "c45": "C4.5 (gain ratio)"}
NODE_ORDER = ["epyc", "dell", "ampere", "hopper"]
NODE_HW = {"epyc": "epyc1: 2× EPYC 9255", "dell": "dell4: 2× Xeon Gold 6542Y",
           "ampere": "ampere: A100 80 GB + 2× EPYC 7402",
           "hopper": "hopper: H100 PCIe + 2× Xeon Gold 6444Y"}
IMPL_COLORS = {"simple": "#d14f8e", "serial": "#2a78d6", "parallel": "#eb6834",
               "cuda": "#1baf7a", "gpu_only": "#eda100", "gpuonly": "#eda100"}
NODE_COLORS = {"epyc": "#8e5bd0", "dell": "#2a78d6", "ampere": "#1baf7a", "hopper": "#eb6834"}
MARKERS = {"simple": "s", "serial": "D", "parallel": "o", "gpuonly": "^", "cuda": "v"}
# Power components, bottom to top of a stacked bar.
# "psu" alone when the PSU is the only reading (Dell nodes); with CPU and GPU
# readings too, the PSU is split into cpu0 + cpu1 + gpu + rest.
COMPONENTS = {"psu": ("Whole server (PSU)", "#5b6770"),
              "cpu0": ("CPU socket 0 (RAPL)", "#2a78d6"),
              "cpu1": ("CPU socket 1 (RAPL)", "#8e5bd0"),
              "gpu": ("GPU (NVML)", "#1baf7a"),
              "rest": ("Rest of the server (PSU − CPU − GPU)", "#8a8780")}
# The implementations that use a GPU. They run on one GPU with the CPUs of its
# NUMA node (power_bench.sh: GPU 0 unless machine.txt says "used_gpu N"). Only
# that GPU is counted, and only in their runs: the CPU implementations count the
# CPU packages alone.
GPU_IMPLS = {"cuda", "gpuonly"}
IDLE_SKIP = 5.0
# s: a PSU meter refreshing less often than this is "slow" (ampere's BMC, 59 s
# point readings); epyc's ACPI meter (2 s averages, refreshed every 4-10 s) is not
SLOW_METER = 20.0


def read_csv(path):
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        rows = list(csv.reader(f))
    head = rows[0]
    return {c: [float(r[i]) for r in rows[1:] if len(r) == len(head)] for i, c in enumerate(head)}


def read_markers(path):
    events = []
    for line in open(path):
        t, rest = line.split(None, 1)
        words = rest.split()
        events.append((float(t), words[0], words[1] if len(words) > 1 else "",
                       dict(w.split("=", 1) for w in words[2:] if "=" in w)))
    return events


def read_tree_output(path):
    text = path.read_text(errors="replace")
    windows = [(float(a), float(b), float(ms) / 1000) for ms, a, b in
               re.findall(r"repeat \d+/\d+: train ([\d.]+) ms, unix ([\d.]+) ([\d.]+)", text)]
    load = re.search(r"load\s+([\d.]+) ms", text)
    nodes = re.search(r"nodes (\d+), leaves (\d+), depth (\d+)", text)
    return {"windows": windows, "load_s": float(load.group(1)) / 1000 if load else None,
            "nodes": int(nodes.group(1)) if nodes else None,
            "depth": int(nodes.group(3)) if nodes else None, "text": text}


class Job:
    def __init__(self, folder):
        self.folder = folder
        self.node = folder.name
        self.samples = read_csv(folder / "samples.csv")
        self.t = self.samples["t"]
        self.t_read = self.read_times()
        self.events = read_markers(folder / "markers.txt")
        self.machine = (folder / "machine.txt").read_text()
        m = re.search(r"^used_gpu (\d+)", self.machine, re.M)
        self.used_gpu = int(m.group(1)) if m else 0
        self.psu_col = next((c for c in ("sys_W", "psu_W") if c in self.samples
                             and any(not math.isnan(v) for v in self.samples[c])), None)
        self.has_psu = self.psu_col is not None
        self.gpus = sorted(int(c[3:-3]) for c in self.samples if re.fullmatch(r"gpu\d+_mJ", c))
        self.rapl = self.read_powercap() or self.read_rapl()
        self.idle_windows = self.find_idle()
        self.psu_period = self.measure_period() if self.has_psu else 0.0
        self.slow = self.psu_period > SLOW_METER
        self.refreshes = self.find_refreshes() if self.slow else []
        self.lag = self.measure_lag() if self.has_psu and not self.slow else 0.0
        self.idle = {k: self.mean_power(k, self.idle_windows) for k in self.sources()}

    def read_times(self):
        """When each sample's counters were read. pwrlog.py stamps `t` before its
        reads, and an NVML call can block for ~2 s (on ampere, every time a GPU
        tree ends): the counters are then read ~2 s after `t`, and a window ending
        just before that sample would get ~2 s of extra GPU energy. Newer files
        have `t_end`, the time after the reads. For older ones, a sample followed
        by a gap of more than 0.3 s was read just before the next sample (once
        the reads return, pwrlog.py takes the next sample at once: its loop
        takes ~0.03 s)."""
        if "t_end" in self.samples:
            return self.samples["t_end"]
        t = self.t
        return [t[i + 1] - 0.02 if i + 1 < len(t) and t[i + 1] - t[i] > 0.3 else t[i]
                for i in range(len(t))]

    def read_powercap(self):
        """RAPL package energy per socket from the sampled powercap counters:
        {socket: [(read time, joules since the previous sample, seconds)]}, the
        format of read_rapl. A counter wraps at its max_energy_range_uj."""
        ranges = dict(re.findall(r"^rapl_max (\S+) \S+ (\d+)", self.machine, re.M))
        out = {}
        for col in self.samples:
            m = re.fullmatch(r"rapl_(intel-rapl:\d+)_package-(\d+)_uJ", col)
            if not m:
                continue
            wrap = int(ranges.get(m.group(1), 2 ** 32))
            uj, t = self.samples[col], self.t_read
            series = []
            for i in range(1, len(uj)):
                d = uj[i] - uj[i - 1]
                series.append((t[i], (d + wrap if d < 0 else d) / 1e6, t[i] - t[i - 1]))
            out[int(m.group(2))] = series
        return out

    def read_rapl(self):
        path = self.folder / "perf_rapl.csv"
        if not path.exists():
            path = self.folder / "perf_rapl.csv.gz"
        start = [t for t, kind, *_ in self.events if kind == "perf_start"]
        if not path.exists() or not start:
            return {}
        out = {}
        for line in (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)):
            p = line.strip().split(",")
            if len(p) > 4 and p[1].startswith("S") and p[3] not in ("", "<not counted>"):
                dur = float(p[6]) / 1e9 if p[6] else 0.1  # the interval's length (the last one is short)
                out.setdefault(int(p[1][1:]), []).append((start[0] + float(p[0]), float(p[3]), dur))
        return out

    def find_idle(self):
        starts = {kind: t for t, kind, *_ in self.events}
        windows = []
        for a, b in (("idle_start", "idle_end"), ("step_end", "step_idle_end"), ("idle2_start", "idle2_end")):
            if a in starts and b in starts:
                windows.append((starts[a] + IDLE_SKIP, starts[b]))
        return windows

    def sources(self):
        s = (["psu"] if self.has_psu else []) + ["cpu%d" % k for k in sorted(self.rapl)]
        return s + ["gpu%d" % g for g in self.gpus]

    def processes(self):
        """[(id, attrs, start, end, rc)] in order."""
        open_runs, out = {}, []
        for t, kind, run, attrs in self.events:
            if kind == "start":
                open_runs[run] = (t, attrs)
            elif kind == "end" and run in open_runs:
                s, attrs = open_runs.pop(run)
                out.append((run, attrs, s, t, attrs.get("rc", "0")))
        return out

    def psu_readings(self):
        return [(t, v) for t, v in zip(self.t, self.samples[self.psu_col]) if not math.isnan(v)]

    def measure_period(self):
        """Median time between changes of the PSU reading: the meter's refresh
        period, or longer where consecutive refreshes give the same value."""
        r = self.psu_readings()
        changes = [r[i][0] for i in range(1, len(r)) if r[i][1] != r[i - 1][1]]
        gaps = [b - a for a, b in zip(changes, changes[1:])]
        return statistics.median(gaps) if gaps else 0.0

    def find_refreshes(self):
        """A slow meter's refreshes, [(time, watts)]. ampere's BMC takes a point
        reading about every 59 s (a 90 s all-core step shows full power 42 s in,
        and idle again 11 s after its end), and /run/power/psu_W shows it within
        a second. A refresh that repeats the previous value leaves no change in
        the file, so the refreshes between two changes are spread evenly at the
        refresh period."""
        r = self.psu_readings()
        ch = [r[0]] + [r[i] for i in range(1, len(r)) if r[i][1] != r[i - 1][1]]
        gaps = [b[0] - a[0] for a, b in zip(ch[1:], ch[2:])]
        med = statistics.median(gaps)
        period = statistics.median([g for g in gaps if g < 1.5 * med])
        out = []
        t = ch[1][0] - period  # ch[1] is the first change, a refresh; ch[0] only the first sample
        while t >= r[0][0]:
            out.append((t, ch[0][1]))
            t -= period
        out.reverse()
        for (t0, v), (t1, _) in zip(ch[1:], ch[2:]):
            n = max(1, round((t1 - t0) / period))
            out += [(t0 + j * (t1 - t0) / n, v) for j in range(n)]
        t, v = ch[-1]
        while t <= r[-1][0]:
            out.append((t, v))
            t += period
        self.psu_period = period
        return out

    def refreshes_in(self, a, b):
        return [v for t, v in self.refreshes if a <= t <= b]

    def changes(self, a, b):
        """Readings that stand for [a, b]: a slow meter's refreshes in it, or the
        number of times a fast meter's reading changed."""
        if self.slow:
            return len(self.refreshes_in(a, b))
        r = [v for t, v in self.psu_readings() if a <= t <= b]
        return sum(1 for x, y in zip(r, r[1:]) if x != y)

    def measure_lag(self):
        """Delay from a load step to the PSU reading half-way up the step: the
        all-core step when the job has one, else the median over the processes."""
        col = self.psu_col
        idle = statistics.mean(self.values_between(col, *self.idle_windows[0])) \
            if self.idle_windows else min(self.samples[col])
        marks = {kind: t for t, kind, *_ in self.events}
        if "step_start" in marks and "step_end" in marks:
            s0, s1 = marks["step_start"], marks["step_end"]
            top = statistics.median(self.values_between(col, (s0 + s1) / 2, s1))
            if top - idle >= 30:
                return next(t for t, v in self.psu_readings() if t >= s0 and v >= idle + 0.5 * (top - idle)) - s0
        reach = max(20.0, 3 * self.psu_period)
        lags = []
        for _, _, start, end, _ in self.processes():
            after = [(t, v) for t, v in self.psu_readings() if start <= t <= start + reach]
            if not after or end - start < reach - 5:
                continue
            top = max(v for _, v in after)
            if top - idle < 30:
                continue
            first = next(t for t, v in after if v >= idle + 0.5 * (top - idle))
            lags.append(first - start)
        return statistics.median(lags) if lags else 2.0

    def values_between(self, col, a, b):
        return [v for t, v in zip(self.t, self.samples[col]) if a <= t <= b and not math.isnan(v)]

    def psu_window(self, a, b):
        """The meter-time span that stands for [a, b]: shifted by the meter's
        delay (a slow meter's refreshes are point readings: no shift), and
        clipped to the samples."""
        return max(a + self.lag, self.t[0]), min(b + self.lag, self.t[-1])

    def energy(self, source, a, b):
        """Joules of `source` from a to b (Unix times)."""
        if source == "psu":
            lo, hi = self.psu_window(a, b)
            if self.slow:  # mean of the point readings in the window x its length
                vals = self.refreshes_in(lo, hi)
                if vals:
                    return statistics.mean(vals) * (b - a)
                # shorter than a refresh period (test runs): the reading held over it
            return self.held_integral(self.samples[self.psu_col], lo, hi) / (hi - lo) * (b - a)
        if source.startswith("gpu"):
            col = self.samples["gpu%s_mJ" % source[3:]]
            return (self.interpolate(col, b, self.t_read) - self.interpolate(col, a, self.t_read)) / 1000
        total = 0.0
        for t, joules, dur in self.rapl[int(source[3:])]:
            overlap = min(b, t) - max(a, t - dur)
            if overlap > 0 and dur > 0:
                total += joules * overlap / dur
        return total

    def held_integral(self, values, a, b):
        """Integral of the readings, each held until the next sample, from a to b."""
        total, last = 0.0, float("nan")
        for i in range(len(self.t) - 1):
            if not math.isnan(values[i]):
                last = values[i]
            lo, hi = max(a, self.t[i]), min(b, self.t[i + 1])
            if hi > lo:
                total += last * (hi - lo)
        return total

    def interpolate(self, col, x, t):
        lo, hi = 0, len(t) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if t[mid] <= x:
                lo = mid
            else:
                hi = mid
        if t[hi] == t[lo]:
            return col[lo]
        return col[lo] + (col[hi] - col[lo]) * (x - t[lo]) / (t[hi] - t[lo])

    def mean_power(self, source, windows):
        if not windows:
            return float("nan")
        return sum(self.energy(source, a, b) for a, b in windows) / sum(b - a for a, b in windows)

    def peak_psu(self, a, b):
        # Skip the first 10 s of a long window: the meter's first reading there
        # still averages over the end of the (all-core) CSV load.
        lo, hi = self.psu_window(a, b)
        if self.slow:
            vals = self.refreshes_in(lo, hi)
            return max(vals) if vals else float("nan")
        if b - a > 60:  # and the last 5 s: evaluation starts on every core at the window's end
            lo, hi = lo + 10.0, hi - 5.0
        vals = self.values_between(self.psu_col, lo, hi)
        return max(vals) if vals else float("nan")

    def peak_gpu(self, a, b):
        i0, i1 = bisect.bisect_left(self.t, a), bisect.bisect_right(self.t, b)
        vals = self.samples["gpu%d_W" % self.used_gpu][i0:i1]
        return max(vals) if vals else float("nan")

    def peak(self, a, b, gpu=False, cpu_only=False):
        """Highest sampled total power in [a, b]: PSU, or GPU + CPU sockets
        (cpu_only: the CPU sockets alone)."""
        if self.has_psu and not cpu_only:
            return self.peak_psu(a, b)
        best = 0.0
        # per socket: interval ends and watts; a sample at t falls in the first interval ending >= t
        rapl = [([t for t, _, d in v if d > 0], [j / d for _, j, d in v if d > 0])
                for v in self.rapl.values()]
        # widen by one sample period so that a window shorter than it still has samples
        pad = 0.1 if b - a < 0.5 else 0.0
        for i in range(bisect.bisect_left(self.t, a - pad), bisect.bisect_right(self.t, b + pad)):
            t = self.t[i]
            power = self.samples["gpu%d_W" % self.used_gpu][i] if gpu and self.gpus and not cpu_only else 0.0
            for ends, watts in rapl:
                if ends:
                    power += watts[min(bisect.bisect_left(ends, t), len(ends) - 1)]
            best = max(best, power)
        return best


def components(job, per_source, impl):
    """Map source values to chart components: the CPU packages plus, for the GPU
    implementations, the GPU they ran on; with a PSU reading also the rest of the
    PSU's power ("rest"), or the PSU alone when there are no CPU readings."""
    out = {}
    for k in sorted(job.rapl):
        out["cpu%d" % k] = per_source["cpu%d" % k]
    if job.gpus and impl in GPU_IMPLS:
        out["gpu"] = per_source["gpu%d" % job.used_gpu]
    if job.has_psu:
        if out:
            out["rest"] = per_source["psu"] - sum(out.values())
        else:
            out["psu"] = per_source["psu"]
    return out


def analyse(job):
    rows = []
    for run, attrs, start, end, rc in job.processes():
        if run.startswith("warmup"):
            continue
        tree = read_tree_output(job.folder / "out" / f"{run}.txt")
        if rc != "0" or not tree["windows"]:
            print(f"  {job.node} {run}: failed (rc {rc}), skipped", file=sys.stderr)
            continue
        # Which trees are measured. Dell nodes (PSU meter, 2-6 s readings): the
        # steady state, every tree but the first (with K >= 3), whose window is
        # >= 120 s. ampere (NVML and RAPL, 100 ms): the first tree, which is the
        # tree the time benchmarks time; the later ones are slower on the GPU
        # (their "gpu setup" uploads the data again without overlapping the
        # load) and are reported as "steady_*".
        steady = tree["windows"][1:] if len(tree["windows"]) >= 3 else tree["windows"]
        windows = steady if job.has_psu else tree["windows"][:1]
        k = len(windows)
        a, b = windows[0][0], windows[-1][1]
        span = b - a
        energy = {s: job.energy(s, a, b) / k for s in job.sources()}
        idle = {s: job.idle[s] * span / k for s in job.sources()}
        comp_e = components(job, energy, attrs["impl"])
        comp_idle = components(job, idle, attrs["impl"])
        total = sum(comp_e.values())
        sa, sb = steady[0][0], steady[-1][1]
        steady_e = sum(components(job, {s: job.energy(s, sa, sb) for s in job.sources()},
                                  attrs["impl"]).values()) / len(steady)
        rows.append({
            "node": job.node, "impl": attrs["impl"], "algo": attrs["algo"],
            "rep": int(run.split("-")[0][3:]), "repeat": k, "trees_in_process": len(tree["windows"]),
            "train_s": statistics.mean(w[2] for w in windows), "first_tree_s": tree["windows"][0][2],
            "steady_train_s": statistics.mean(w[2] for w in steady), "steady_energy_J": steady_e,
            "span_s": span / k, "load_s": tree["load_s"], "nodes": tree["nodes"],
            "depth": tree["depth"], "energy_J": total,
            "above_idle_J": total - sum(comp_idle.values()),
            "power_W": total / (span / k), "idle_W": sum(comp_idle.values()) / (span / k),
            "peak_W": job.peak(a, b, attrs["impl"] in GPU_IMPLS), "lag_s": job.lag,
            "peak_cpu_W": job.peak(a, b, cpu_only=True) if job.rapl else float("nan"),
            "peak_gpu_W": job.peak_gpu(a, b) if job.gpus and attrs["impl"] in GPU_IMPLS else float("nan"),
            "peak_psu_W": job.peak_psu(a, b) if job.has_psu else float("nan"),
            "psu_J": energy["psu"] if job.has_psu else float("nan"),
            "psu_idle_J": idle["psu"] if job.has_psu else float("nan"),
            "psu_changes": job.changes(*job.psu_window(a, b)) if job.has_psu else 0,
            "psu_period_s": job.psu_period,
            **{f"E_{c}": v for c, v in comp_e.items()},
            **{f"Eidle_{c}": v for c, v in comp_idle.items()},
            "_window": (a, b, start, end),
        })
    return rows


def mean_sd(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarise(rows):
    cases = {}
    for r in rows:
        cases.setdefault((r["node"], r["impl"], r["algo"]), []).append(r)
    summary = []
    for (node, impl, algo), rs in cases.items():
        s = {"node": node, "impl": impl, "algo": algo, "runs": len(rs), "repeat": rs[0]["repeat"],
             "nodes": sorted({r["nodes"] for r in rs}), "lag_s": rs[0]["lag_s"]}
        s["trees_in_process"] = rs[0]["trees_in_process"]
        s["psu_changes_min"] = min(r["psu_changes"] for r in rs)
        s["psu_period_s"] = rs[0]["psu_period_s"]
        for key in ("train_s", "load_s", "energy_J", "above_idle_J", "power_W", "idle_W", "peak_W",
                    "steady_train_s", "steady_energy_J", "span_s", "peak_cpu_W", "peak_gpu_W",
                    "peak_psu_W", "psu_J", "psu_idle_J"):
            s[key], s[key + "_sd"] = mean_sd([r[key] for r in rs])
        for c in COMPONENTS:
            if f"E_{c}" in rs[0]:
                s[f"E_{c}"] = statistics.mean(r[f"E_{c}"] for r in rs)
                s[f"Eidle_{c}"] = statistics.mean(r[f"Eidle_{c}"] for r in rs)
                s[f"P_{c}"] = s[f"E_{c}"] / s["span_s"]
        summary.append(s)
    key = lambda s: (NODE_ORDER.index(s["node"]) if s["node"] in NODE_ORDER else 9,
                     IMPL_ORDER.index(s["impl"]), list(ALGOS).index(s["algo"]))
    return sorted(summary, key=key)


def fmt_s(v):
    return f"{v:,.0f} s" if v >= 100 else f"{v:.3g} s"


def fmt_j(v):
    if v >= 1e6:
        return f"{v / 1e6:.3g} MJ"
    return f"{v / 1e3:.3g} kJ" if v >= 1e3 else f"{v:.3g} J"


def label(s):
    return f"{IMPLS[s['impl']]} · {s['node']}"


def trees(s):
    """Which trees were measured, for the tables."""
    if s["repeat"] == 1 and s["trees_in_process"] > 1:
        return f"{s['runs']} × 1st of {s['trees_in_process']}"
    return f"{s['runs']} × {s['repeat']}"


def write_tables(folder, rows, summary):
    keys = [k for k in rows[0] if not k.startswith("_")] if rows else []
    with open(folder / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    skeys = sorted({k for s in summary for k in s}, key=lambda k: (k not in summary[0], k))
    with open(folder / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, skeys)
        w.writeheader()
        w.writerows(summary)
    lines = []
    for algo, title in ALGOS.items():
        lines += [f"### {title}, depth ≤ 10, no pruning", "",
                  "| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | "
                  "E per tree | E above idle | split of E | ",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for s in summary:
            if s["algo"] != algo:
                continue
            split = ", ".join(f"{COMPONENTS[c][0].split(' (')[0]} {fmt_j(s['E_' + c])}"
                              for c in COMPONENTS if f"E_{c}" in s)
            lines.append(
                f"| {IMPLS[s['impl']]} | {s['node']} | {trees(s)} | "
                f"{fmt_s(s['train_s'])} ± {s['train_s_sd'] / s['train_s'] * 100:.1f}% | "
                f"{s['power_W']:.0f} W | {s['power_W'] - s['idle_W']:.0f} W | {s['peak_W']:.0f} W | "
                f"{fmt_j(s['energy_J'])} ± {s['energy_J_sd'] / s['energy_J'] * 100:.1f}% | "
                f"{fmt_j(s['above_idle_J'])} | {split} |")
        lines.append("")
    later = [s for s in summary if s["repeat"] == 1 and s["trees_in_process"] > 1]
    if later:
        lines += ["### Later trees in the same process (ampere)", "",
                  "| Implementation | Algorithm | first tree: t, E | later trees: t, E (mean) |",
                  "|---|---|---|---|"]
        for s in later:
            lines.append(f"| {IMPLS[s['impl']]} | {ALGOS[s['algo']].split(' (')[0]} | "
                         f"{fmt_s(s['train_s'])}, {fmt_j(s['energy_J'])} | "
                         f"{fmt_s(s['steady_train_s'])}, {fmt_j(s['steady_energy_J'])} |")
        lines.append("")
    (folder / "summary.md").write_text("\n".join(lines))


def unit_ticks(ax, axis, unit):
    """Tick labels with their unit ("100 W", "10 kJ", "50 %")."""
    from matplotlib.ticker import FuncFormatter
    sep = "" if unit == "%" else " "
    f = FuncFormatter(lambda v, _: f"{v:,g}{sep}{unit}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_axisbelow(True)


def ordered(summary, algo):
    return [s for s in summary if s["algo"] == algo]


def chart_power(folder, summary):
    """Mean power while training, stacked by component; the light part of each is its idle power."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), sharey=True)
    used = set()
    for ax, (algo, title) in zip(axes, ALGOS.items()):
        cases = ordered(summary, algo)
        for i, s in enumerate(cases):
            bottom = 0.0
            for c, (_, color) in COMPONENTS.items():
                if f"P_{c}" not in s:
                    continue
                used.add(c)
                idle = s[f"Eidle_{c}"] / s["span_s"]
                total = s[f"P_{c}"]
                ax.bar(i, min(idle, total), bottom=bottom, color=color, alpha=0.38, width=0.7,
                       edgecolor="white", linewidth=0.5)
                ax.bar(i, max(total - idle, 0), bottom=bottom + min(idle, total), color=color,
                       width=0.7, edgecolor="white", linewidth=0.5)
                bottom += total
            ax.text(i, bottom + 6, f"{s['power_W']:.0f} W", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(cases)), [label(s).replace(" · ", "\n") for s in cases], fontsize=8)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID)
        style(ax)
    axes[0].set_ylabel("mean power while training (W)", color=INK)
    for ax in axes:
        unit_ticks(ax, "y", "W")
    handles = [Patch(color=COMPONENTS[c][1], label=COMPONENTS[c][0]) for c in COMPONENTS if c in used]
    handles += [Patch(facecolor="#999", alpha=0.38, label="lighter part: idle power of that component")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("ECBDL'14 15M: mean power while training one tree (with a PSU reading: the whole server; "
                 "without one: CPU packages + the GPU in use)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(folder / "chart-power.png", dpi=150)
    plt.close(fig)


def chart_energy(folder, summary):
    """Energy per tree on a log scale: the whole bar is total, the dark inner bar above idle."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), sharey=True)
    for ax, (algo, title) in zip(axes, ALGOS.items()):
        cases = ordered(summary, algo)
        for i, s in enumerate(cases):
            color = IMPL_COLORS[s["impl"]]
            ax.bar(i, s["energy_J"] / 1e3, color=color, alpha=0.45, width=0.75)
            ax.bar(i, max(s["above_idle_J"], 1) / 1e3, color=color, width=0.4)
            ax.errorbar(i, s["energy_J"] / 1e3, yerr=s["energy_J_sd"] / 1e3, color=INK, capsize=2, lw=0.8)
            ax.text(i, s["energy_J"] / 1e3 * 1.15, fmt_j(s["energy_J"]), ha="center", va="bottom", fontsize=7.5,
                    color=INK)
            ax.text(i, max(s["above_idle_J"], 1) / 1e3 * 1.06, fmt_j(s["above_idle_J"]), ha="center", va="bottom",
                    fontsize=7, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(range(len(cases)), [label(s).replace(" · ", "\n") for s in cases], fontsize=8)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, which="both")
        style(ax)
    axes[0].set_ylabel("energy per tree (kJ, log scale)", color=INK)
    for ax in axes:
        unit_ticks(ax, "y", "kJ")
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 4)
    fig.legend(handles=[Patch(facecolor="#888", alpha=0.45, label="wide bar: total energy while training"),
                        Patch(facecolor="#888", label="narrow bar: above idle, what the run adds to the idle server")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("ECBDL'14 15M: energy to train one tree (mean of the runs; whisker ± 1 sd; "
                 "with a PSU reading: whole server; without: CPU packages + the GPU in use)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(folder / "chart-energy.png", dpi=150)
    plt.close(fig)


def chart_energy_split(folder, summary):
    """Where the energy goes: each component's share, split into idle and above idle."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    used = set()
    for ax, (algo, title) in zip(axes, ALGOS.items()):
        cases = ordered(summary, algo)
        for i, s in enumerate(cases):
            left = 0.0
            for c, (_, color) in COMPONENTS.items():
                if f"E_{c}" not in s:
                    continue
                used.add(c)
                idle = min(s[f"Eidle_{c}"], s[f"E_{c}"]) / s["energy_J"] * 100
                dyn = max(s[f"E_{c}"] - s[f"Eidle_{c}"], 0) / s["energy_J"] * 100
                ax.barh(i, idle, left=left, color=color, alpha=0.38, height=0.7, edgecolor="white", lw=0.5)
                ax.barh(i, dyn, left=left + idle, color=color, height=0.7, edgecolor="white", lw=0.5)
                if dyn > 7:
                    ax.text(left + idle + dyn / 2, i, f"{dyn:.0f}%", ha="center", va="center",
                            fontsize=7.5, color="white")
                if idle > 7:
                    ax.text(left + idle / 2, i, f"{idle:.0f}%", ha="center", va="center", fontsize=7.5,
                            color=INK)
                left += idle + dyn
        ax.set_yticks(range(len(cases)), [label(s) for s in cases], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of the energy per tree (%)", color=INK)
        unit_ticks(ax, "x", "%")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        style(ax)
    handles = [Patch(color=COMPONENTS[c][1], label=COMPONENTS[c][0]) for c in COMPONENTS if c in used]
    handles.append(Patch(facecolor="#999", alpha=0.38, label="lighter part: what idle power would have used"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("ECBDL'14 15M: where the training energy goes", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(folder / "chart-energy-split.png", dpi=150)
    plt.close(fig)


def chart_time(folder, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (algo, title) in zip(axes, ALGOS.items()):
        cases = ordered(summary, algo)
        for i, s in enumerate(cases):
            ax.barh(i, s["train_s"], color=IMPL_COLORS[s["impl"]], height=0.7)
            ax.text(s["train_s"] * 1.08, i, f"{fmt_s(s['train_s'])} ± {s['train_s_sd'] / s['train_s'] * 100:.1f}%"
                    f"  ({trees(s)})", va="center", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_yticks(range(len(cases)), [label(s) for s in cases], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(right=ax.get_xlim()[1] * 12)
        ax.set_xlabel("training time per tree (s, log scale)", color=INK)
        unit_ticks(ax, "x", "s")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, which="both")
        style(ax)
    fig.suptitle("ECBDL'14 15M: training time per tree during the power runs", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(folder / "chart-time.png", dpi=150)
    plt.close(fig)


def chart_scatter(folder, summary):
    """Time against energy, linear axes, with lines of constant power; an inset zooms
    in on the fast implementations, which sit in the corner at this scale."""
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    ts = [s["train_s"] for s in summary]
    es = [s["energy_J"] / 1e3 for s in summary]
    fast = [s for s in summary if s["train_s"] < 60]
    fx = max(s["train_s"] for s in fast) * 1.15
    fy = max(s["energy_J"] for s in fast) / 1e3 * 1.15

    def draw(a, xmax, ymax, size, hidden=None):
        for watts in (100, 200, 500, 1000):
            a.plot([0, xmax], [0, xmax * watts / 1e3], color=GRID, lw=1, zorder=0)
            # label just inside the plot, where the line leaves it
            x = min(xmax, ymax * 1e3 / watts)
            y = x * watts / 1e3
            if hidden and hidden(x, y):
                continue
            if x < xmax:
                a.text(x, ymax, f" {watts} W", fontsize=7.5, color=MUTED, ha="left", va="top")
            else:
                a.text(xmax, y, f"{watts} W ", fontsize=7.5, color=MUTED, ha="right", va="bottom")
        for s in summary:
            color = NODE_COLORS.get(s["node"], INK)
            t, e, ea = s["train_s"], s["energy_J"] / 1e3, s["above_idle_J"] / 1e3
            a.scatter(t, e, marker=MARKERS[s["impl"]], s=size,
                      facecolors=color if s["algo"] == "cart" else "white", edgecolors=color,
                      linewidths=1.5, zorder=3, clip_on=True)
            a.scatter(t, ea, marker=MARKERS[s["impl"]], s=size * 0.3, color=color, alpha=0.45, zorder=2)
            a.plot([t, t], [ea, e], color=color, alpha=0.35, lw=0.8, zorder=1)
        a.set_xlim(0, xmax)
        a.set_ylim(0, ymax)
        a.grid(color=GRID, which="major")
        style(a)

    xmax, ymax = max(ts) * 1.08, max(es) * 1.1
    box = (0.07, 0.42, 0.42, 0.52)  # inset, in axes fractions
    draw(ax, xmax, ymax, 60, lambda x, y: x < (box[0] + box[2] + 0.02) * xmax and y > box[1] * ymax)
    ax.set_xlabel("training time per tree (s)", color=INK)
    ax.set_ylabel("energy per tree (kJ)", color=INK)
    ins = ax.inset_axes(list(box))
    ins.set_facecolor("white")
    ins.patch.set_alpha(1)
    ins.set_zorder(5)
    draw(ins, fx, fy, 50)
    ins.tick_params(labelsize=7)
    for a in (ax, ins):
        unit_ticks(a, "x", "s")
        unit_ticks(a, "y", "kJ")
    ins.set_title("zoom: Parallel, GPU, Hybrid", fontsize=8, color=MUTED, loc="left")
    ax.indicate_inset_zoom(ins, edgecolor=MUTED, alpha=0.6)
    handles = [plt.Line2D([], [], marker=MARKERS[i], ls="", color=INK, label=IMPLS[i])
               for i in IMPL_ORDER if any(s["impl"] == i for s in summary)]
    handles += [Patch(color=NODE_COLORS[n], label=NODE_HW[n]) for n in NODE_ORDER
                if any(s["node"] == n for s in summary)]
    handles += [plt.Line2D([], [], marker="o", ls="", markerfacecolor=INK, color=INK, label="CART"),
                plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", color=INK, label="C4.5"),
                plt.Line2D([], [], marker="o", ls="", color=INK, alpha=0.45, markersize=4,
                           label="small dot: energy above idle")]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("ECBDL'14 15M: time and energy per tree (grey lines: constant power)",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(folder / "chart-time-energy.png", dpi=150)
    plt.close(fig)


def chart_cpu_gpu(folder, summary):
    """Nodes with CPU (RAPL) and GPU (NVML) sensors: one chart per node and
    algorithm. Each implementation gets up to six thin bars on three log axes:
    energy per tree (CPU + GPU, stacked; and the PSU, the whole server), mean
    power while training (CPU, GPU and PSU, one bar each) and time per tree.
    Every bar's lighter bottom part ends at the idle value (power: the node's
    idle power, the same for every run; energy: idle power x time), the darker
    top part is what the run adds above idle.
    The GPU is counted only for the GPU implementations, and only the GPU they
    ran on. The PSU bars appear when the job had a PSU reading."""
    from matplotlib.colors import to_rgba
    from matplotlib.ticker import FuncFormatter
    cases = [s for s in summary if "E_cpu0" in s]
    nodes = sorted({s["node"] for s in cases})
    if not cases or len({s["impl"] for s in cases}) < 3:
        return
    for old in folder.glob("chart-cpu-gpu-*.png"):
        old.unlink()
    CPU, GPU, PSU, TIME = "#2a78d6", "#1baf7a", "#8e5bd0", "#3d3c3a"
    cpus = [c for c in ("cpu0", "cpu1") if f"E_{c}" in cases[0]]
    light = lambda c: to_rgba(c, 0.35)  # noqa: E731

    def finite(v):
        return v is not None and not math.isnan(v)

    def cpu(s, div):  # (idle, above idle)
        i = sum(s[f"Eidle_{c}"] for c in cpus) / div
        return i, max(sum(s[f"E_{c}"] for c in cpus) / div - i, 0.0)

    def gpu(s, div):
        i = s.get("Eidle_gpu", 0.0) / div
        return i, max(s.get("E_gpu", 0.0) / div - i, 0.0)

    def psu(s, div):
        i = s["psu_idle_J"] / div
        return i, max(s["psu_J"] / div - i, 0.0)

    def stack(ax, x, parts, floor):
        """parts: [(value, color)] bottom to top, each drawn at its real value on
        the axis: the first part from the floor up to its value, the next from
        there up to the sum, and so on. So the idle part of every power bar of a
        node ends at the same height: the node's idle power."""
        total = sum(v for v, _ in parts)
        cum = 0.0
        for v, col in parts:
            if v <= 0:
                continue
            lo = max(cum, floor)
            cum += v
            if cum > lo:
                ax.bar(x, cum - lo, bottom=lo, width=W, color=col, edgecolor="white", lw=0.3)
        return total

    def joules(v, _):
        return fmt_j(v)

    def watts(v, _):
        return f"{v:,.0f} W"

    def seconds(v, _):
        return f"{v:,.0f} s" if v >= 1 else f"{v:g} s"

    def short_j(v):
        return fmt_j(v).replace(" ", "\u2009")

    def short_w(v):
        return f"{v:,.0f}\u2009W"

    def numbers(ax, x, idle, total, fmt, floor):
        """The bar's value right above it; the idle value inside the light
        part, just under the line where the light and dark parts meet."""
        ax.text(x, total * 1.06, fmt(total), ha="center", va="bottom", fontsize=fs, color=INK)
        if idle > floor * 1.6:
            ax.text(x, idle / 1.06, fmt(idle), ha="center", va="top", fontsize=fs, color=INK)

    W = 0.066  # bar width
    NAMES = {"e_cpu": "CPU", "e_gpu": "GPU", "e_psu": "PSU", "p_cpu": "CPU", "p_gpu": "GPU", "p_psu": "PSU",
             "t": "time"}
    fs = 6.3
    for node in nodes:
        for algo in [a for a in ALGOS if any(s["node"] == node and s["algo"] == a for s in cases)]:
            ss = sorted([s for s in cases if s["node"] == node and s["algo"] == algo],
                        key=lambda s: IMPL_ORDER.index(s["impl"]))
            has_psu = all(finite(s.get("psu_J")) for s in ss)
            has_gpu = any("E_gpu" in s for s in ss)
            # one bar per sensor and quantity: energy (CPU, GPU, PSU), power (CPU, GPU, PSU), time;
            # an implementation that does not use the GPU gets no GPU bars (and no space for them)
            def sensors_of(s):
                used_gpu = has_gpu and s["impl"] in GPU_IMPLS
                return ["cpu"] + (["gpu"] if used_gpu else []) + (["psu"] if has_psu else [])

            def layout(s):
                sens = sensors_of(s)
                groups = [[f"e_{x}" for x in sens], [f"p_{x}" for x in sens], ["t"]]
                off, x0 = {}, 0.0
                for g in groups:
                    for k in g:
                        off[k] = x0
                        x0 += 0.088
                    x0 += 0.04
                mid = (x0 - 0.04 - 0.088) / 2
                return groups, {k: v - mid for k, v in off.items()}

            lay = [layout(s) for s in ss]
            fig, ax_e = plt.subplots(figsize=(22 if has_gpu else 15, 7.6))
            ax_p = ax_e.twinx()
            ax_t = ax_e.twinx()
            ax_t.spines["right"].set_position(("axes", 1.055 if has_gpu else 1.085))
            parts_of = {"cpu": (cpu, CPU), "gpu": (gpu, GPU), "psu": (psu, PSU)}

            def bars_of(s, div):
                """[(sensor, (idle, above idle), color)]"""
                return [(x, parts_of[x][0](s, div), parts_of[x][1]) for x in sensors_of(s)]

            # energy per tree (log): one bar per sensor
            tops = [sum(v) for s in ss for _, v, _ in bars_of(s, 1.0) if sum(v) > 0]
            fe = min(tops) / 8
            for i, s in enumerate(ss):
                for x, v, color in bars_of(s, 1.0):
                    pos = i + lay[i][1][f"e_{x}"]
                    tot = stack(ax_e, pos, [(v[0], light(color)), (v[1], color)], fe)
                    numbers(ax_e, pos, v[0], tot, short_j, fe)
            ax_e.set_yscale("log")
            ax_e.set_ylim(fe, max(tops) * 25)

            # mean power while training (W, log): CPU, GPU and PSU bars
            fp = 20.0
            pmax = 0.0
            for i, s in enumerate(ss):
                for x, parts, color in bars_of(s, s["span_s"]):
                    pos = i + lay[i][1][f"p_{x}"]
                    tot = stack(ax_p, pos, [(parts[0], light(color)), (parts[1], color)], fp)
                    pmax = max(pmax, tot)
                    numbers(ax_p, pos, parts[0], tot, short_w, fp)
            ax_p.set_yscale("log")
            ax_p.set_ylim(fp, max(pmax * 25, 3000))

            # time per tree (s, log)
            t = [s["train_s"] for s in ss]
            ft = min(t) / 8
            ax_t.bar([i + lay[i][1]["t"] for i in range(len(ss))], [v - ft for v in t], bottom=ft, width=W,
                     color=TIME)
            for i, v in enumerate(t):
                ax_t.text(i + lay[i][1]["t"], v * 1.1, fmt_s(v).replace(" ", " "), ha="center", va="bottom",
                          fontsize=fs, color=INK)
            ax_t.set_yscale("log")
            ax_t.set_ylim(ft, max(t) * 25)

            ax_e.set_ylabel("ENERGY per tree (log scale): left bars", color=INK)
            ax_p.set_ylabel("mean POWER while training (log scale): middle bars", color=INK)
            ax_t.set_ylabel("TIME per tree (log scale): right bar", color=INK)
            for a, f in ((ax_e, joules), (ax_p, watts), (ax_t, seconds)):
                a.yaxis.set_major_formatter(FuncFormatter(f))
                a.spines["top"].set_visible(False)
                a.tick_params(colors=INK, labelsize=8)
            ax_e.grid(axis="y", color=GRID, which="major")
            ax_e.set_axisbelow(True)
            # two-level x labels: what each bar is, then the implementation
            ax_e.set_xticks([i + v for i in range(len(ss)) for v in lay[i][1].values()])
            ax_e.set_xticklabels([NAMES[k] for i in range(len(ss)) for k in lay[i][1]], fontsize=6.5, color=MUTED)
            ax_e.tick_params(axis="x", length=2)
            ax_e.set_xlim(-0.55, len(ss) - 0.45)
            for i, s in enumerate(ss):
                groups, off = lay[i]
                for g, name in zip(groups[:2], ("energy", "power")):
                    ax_e.text(i + (off[g[0]] + off[g[-1]]) / 2, -0.045, name, transform=ax_e.get_xaxis_transform(),
                              ha="center", va="top", fontsize=7, color=MUTED)
                ax_e.text(i, -0.08, IMPLS[s["impl"]], transform=ax_e.get_xaxis_transform(),
                          ha="center", va="top", fontsize=10.5, color=INK, fontweight="bold")
            handles = [Patch(color=CPU, label="CPU packages (RAPL)")]
            if has_gpu:
                handles.append(Patch(color=GPU, label="GPU in use (NVML; GPU and Hybrid only)"))
            if has_psu:
                handles.append(Patch(color=PSU, label="whole server at the PSU"))
            handles += [Patch(color=light("#777"), label="light part: idle (number under its top edge)"),
                        Patch(color="#777", label="dark part: what the run adds above idle"),
                        Patch(color=TIME, label="time per tree")]
            ax_e.legend(handles=handles, fontsize=8, frameon=False, loc="upper right", ncol=3,
                        title="Number above a bar: its mean value. Idle energy = idle power × time.",
                        title_fontsize=8)
            n = ", ".join(f"{IMPLS[s['impl']]} {s['runs']}×{s['repeat']}" for s in ss)
            hw = NODE_HW.get(node, node)
            what = ("both CPU packages (RAPL) and the whole server at the PSU (no GPU)" if has_psu and not has_gpu
                    else "both CPU packages (RAPL), the GPU in use (NVML; GPU and Hybrid only) and the whole server "
                    "at the PSU" if has_psu else "both CPU packages (RAPL) + the GPU in use (NVML; GPU and Hybrid "
                    "runs only). No PSU, RAM or fans")
            psu_note = ""
            if has_psu:
                period = ss[0]["psu_period_s"]
                psu_note = (f" PSU reading refreshed every ~{period:.0f} s; "
                            f"≥ {min(s['psu_changes_min'] for s in ss)} {'readings' if period > SLOW_METER else 'changes'} "
                            f"in every measured window.")
            ax_e.set_title(f"{ALGOS[algo]}: energy, power and time to train one tree on ECBDL'14 15M "
                           f"(depth ≤ 10), {hw}\nMeasured: {what}.{psu_note}\n"
                           f"Means of runs × trees: {n} (the first tree of a process with ≥ 3 trees is "
                           f"left out)", fontsize=9.5, color=INK, loc="left")
            fig.tight_layout()
            fig.savefig(folder / f"chart-cpu-gpu-{node}-{algo}.png", dpi=150)
            plt.close(fig)


def chart_traces(folder, jobs, rows):
    """Power over time of every case's first run: process start to end, training shaded."""
    first = {}
    for r in rows:
        key = (r["node"], r["impl"], r["algo"])
        if key not in first or r["rep"] < first[key]["rep"]:
            first[key] = r
    keys = sorted(first, key=lambda k: (NODE_ORDER.index(k[0]) if k[0] in NODE_ORDER else 9,
                                        IMPL_ORDER.index(k[1]), list(ALGOS).index(k[2])))
    cols = 4
    nrows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(16, 2.9 * nrows), squeeze=False)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax, key in zip(axes.flat, keys):
        r = first[key]
        job = jobs[key[0]]
        a, b, start, end = r["_window"]
        lo, hi = start - 10, end + 10
        idx = [i for i, t in enumerate(job.t) if lo <= t <= hi]
        x = [job.t[i] - start for i in idx]
        if job.has_psu:
            ax.plot([job.t[i] - start - job.lag for i in idx], [job.samples[job.psu_col][i] for i in idx],
                    color=COMPONENTS["psu"][1], lw=1.2, drawstyle="steps-post", label="PSU (shifted by delay)")
        if job.gpus and key[1] in GPU_IMPLS:
            ax.plot(x, [job.samples["gpu%d_W" % job.used_gpu][i] for i in idx], lw=1,
                    color=COMPONENTS["gpu"][1], label="GPU")
        for k, series in sorted(job.rapl.items()):
            pts = [(t - start, j / d) for t, j, d in series if lo <= t <= hi and d > 0]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=0.7, color=COMPONENTS["cpu%d" % k][1],
                    label=f"CPU socket {k}")
        ax.axvspan(a - start, b - start, color=IMPL_COLORS[key[1]], alpha=0.12, lw=0)
        busy = ax.twinx()
        busy.plot(x, [job.samples["cpu_busy"][i] * 100 for i in idx], color=MUTED, lw=0.5, alpha=0.6)
        busy.set_ylim(0, 105)
        unit_ticks(busy, "y", "%")
        unit_ticks(ax, "y", "W")
        unit_ticks(ax, "x", "s")
        busy.tick_params(labelsize=6.5, colors=MUTED)
        busy.spines["top"].set_visible(False)
        ax.set_title(f"{IMPLS[key[1]]}, {key[2].upper().replace('C45', 'C4.5')} · {key[0]}  "
                     f"({r['trees_in_process']} tree{'s' if r['trees_in_process'] > 1 else ''})", fontsize=8.5,
                     loc="left", color=INK)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(color=GRID, lw=0.5)
        style(ax)
    handles = [plt.Line2D([], [], color=COMPONENTS["psu"][1], label="whole server, PSU (shifted back by the meter delay)"),
               plt.Line2D([], [], color=COMPONENTS["gpu"][1], label="GPU (NVML; GPU runs only)"),
               plt.Line2D([], [], color=COMPONENTS["cpu0"][1], label="CPU socket 0 (RAPL)"),
               plt.Line2D([], [], color=COMPONENTS["cpu1"][1], label="CPU socket 1 (RAPL)"),
               plt.Line2D([], [], color=MUTED, lw=0.5, label="CPU busy % (right axis)"),
               Patch(color="#888", alpha=0.2, label="measured trees (shaded); before them: loading the CSV")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Power over time, first run of each case (W against seconds since the process started)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(folder / "chart-traces.png", dpi=130)
    plt.close(fig)


def main():
    folder = Path(sys.argv[1])
    jobs = {}
    for sub in sorted(folder.iterdir()):
        if (sub / "markers.txt").exists() and (sub / "samples.csv").exists():
            jobs[sub.name] = Job(sub)
        elif (sub / "markers.txt").exists() and (sub / "samples.csv.gz").exists():
            jobs[sub.name] = Job(sub)
    rows = []
    for name, job in jobs.items():
        print(f"{name}: sources {job.sources()}, PSU delay {job.lag:.2f} s, idle "
              + ", ".join(f"{k} {v:.0f} W" for k, v in job.idle.items()))
        rows += analyse(job)
    summary = summarise(rows)
    write_tables(folder, rows, summary)
    chart_power(folder, summary)
    chart_energy(folder, summary)
    chart_energy_split(folder, summary)
    chart_time(folder, summary)
    chart_scatter(folder, summary)
    chart_traces(folder, jobs, rows)
    chart_cpu_gpu(folder, summary)
    print((folder / "summary.md").read_text())


if __name__ == "__main__":
    main()
