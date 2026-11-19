#!/usr/bin/env python3
"""Energy of ./tree's backends from bench/tools/power/power_bench.sh jobs: tables and charts.

  bench/tools/.venv/bin/python bench/tools/power/analyze.py bench/cluster-power

The folder holds one job folder per node (epyc1/, dell4/, ampere/, hopper/),
each with markers.txt, machine.txt, samples.csv.gz (pwrlog.py, every 0.1 s)
and out/<run>.txt (./tree's output, with the --repeat K training windows).
Writes into the folder: runs.csv (every process), summary.csv (mean and
standard deviation per case), meter.csv (the meter model of each node),
warmup.csv (the warm-up check), summary.md (tables) and charts chart-*.png.
POWER_TIME_BENCH (environment): the time benchmark's cluster-ecbdl folder for
the time check (default bench/cluster-ecbdl; only for ECBDL'14 15M at
depth <= 10). The dataset and depth limit in titles and tables come from the
jobs' machine.txt (the first job's "data" and "protocol" lines).

Per process (one ./tree --repeat K), over its K training windows (each one a
fresh process's only tree; the gaps between them, fork, CUDA setup and the
warm-up, are left out), summed and divided by K:

  - CPU socket s: RAPL package energy (powercap counter, read every 0.1 s),
    linear between reads. DRAM: RAPL DRAM energy of both sockets (Intel
    nodes; AMD has no DRAM counter).
  - GPU: the NVML energy counter of the GPU in use (GPU and Hybrid only),
    linear between reads.
  - PSU (whole server): the meters refresh every 1 s (hopper), 2 s (dell),
    4-10 s (epyc) or take a point reading once a minute (ampere), too slowly
    for 2-5 s trees. So each node's meter is modelled from the fast sensors:
        meter reading at t = mean over [t - D - W, t - D] of
                             a + b * (CPU packages + DRAM) + c * (all GPUs) + d * (busy CPUs)
    with a, b, c, d fitted by least squares over every reading of the job (idle,
    the all-CPU staircase, every run) and the delay D and averaging window W
    by a grid search. The PSU energy of a window is the integral of
    the model over it. Checks (summary.md, chart-meter.png):
    the fit's residual, the coefficients fitted on each half of the job, the
    mean residual of the readings that fall inside each case's training
    windows (a bias there would show), and the meter's own energy over the
    same windows (readings held and shifted by D + W/2; ampere: the mean of
    its point readings in the windows), which needs long windows to be exact.
  - unused GPUs: the NVML energy of the node's other GPUs (all of them for a
    CPU backend), idle but powered: ~150 W on hopper and ampere.
  - rest: PSU - CPU packages - DRAM - every GPU: fans, disks, network, the
    board, PSU losses, and on AMD the memory.

Checks of the meter model, per case (summary.md "PSU energy: the model against
the meter"): the model fitted on each half of the job; fitted without any
reading of the case's own processes (leave the case out: what the model
predicts for a load it has not seen, and the mean residual of those held-out
readings); fitted without the busy-CPUs term.

"E without the unused GPUs": the energy minus c x the unused GPUs' NVML
energy (c: the model's GPU coefficient, what a GPU watt costs at the PSU),
the server as if it held only the GPU in use (CPU backends: no GPU). A
counterfactual for comparing servers, not a measurement.

Idle power of each source: the job's idle windows (before and after the
staircase, at the end; the first 5 s of each skipped). "Above idle" =
energy - idle power x time.
"""
import bisect
import csv
import gzip
import math
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/tools/
import benchlib as B  # noqa: E402
from chart import out_of_date_banner  # noqa: E402

# Jobs measured before the final protocol (benchlib.out_of_date: each job's machine.txt commit),
# all jobs; main() sets it, save() puts a banner on every chart while any is out of date.
STALE = [0, 0]


def save(fig, path, **kwargs):
    fig.savefig(path, **{**kwargs, **out_of_date_banner(fig, *STALE, unit="power jobs", faded=False)})

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
IMPLS = {"simple": "Serial-Simple", "serial": "Serial-Opt", "parallel": "Multicore",
         "gpuonly": "GPU", "cuda": "Hybrid"}
IMPL_ORDER = list(IMPLS)
ALGOS = {"cart": "CART (gini)", "c45": "C4.5 (gain ratio)"}
NODE_ORDER = ["epyc", "dell", "ampere", "hopper"]
NODE_HW = {"epyc": "epyc1: 2× EPYC 9255", "dell": "dell4: 2× Xeon Gold 6542Y",
           "ampere": "ampere: A100 80 GB + 2× EPYC 7402",
           "hopper": "hopper: H100 PCIe + 2× Xeon Gold 6444Y"}
IMPL_COLORS = {"simple": "#d14f8e", "serial": "#2a78d6", "parallel": "#eb6834",
               "cuda": "#1baf7a", "gpuonly": "#eda100"}
NODE_COLORS = {"epyc": "#8e5bd0", "dell": "#2a78d6", "ampere": "#1baf7a", "hopper": "#eb6834"}
MARKERS = {"simple": "s", "serial": "D", "parallel": "o", "gpuonly": "^", "cuda": "v"}
# Power components, bottom to top of a stacked bar; the PSU is split into
# them ("rest" = PSU - the others).
COMPONENTS = {"cpu0": ("CPU socket 0 (RAPL)", "#2a78d6"),
              "cpu1": ("CPU socket 1 (RAPL)", "#8e5bd0"),
              "dram": ("Memory (RAPL DRAM)", "#c98a1a"),
              "gpu": ("GPU in use (NVML)", "#1baf7a"),
              "gpus_other": ("Unused GPUs (NVML)", "#a8d8c2"),
              "rest": ("Rest of the server (PSU − the others)", "#8a8780")}
PSU_COLOR = "#5b6770"
# The implementations that use a GPU: only they count the GPU in use.
GPU_IMPLS = {"cuda", "gpuonly"}
IDLE_SKIP = 5.0
# s: a meter refreshing less often than this is "slow" (ampere's BMC, point
# readings every 59 s): its readings are spread over the refresh times.
SLOW_METER = 20.0
# the meter model's grid: delay D and averaging window W (s)
GRID_D_FAST = np.round(np.arange(0.0, 12.01, 0.1), 2)
GRID_D_SLOW = np.arange(-5.0, 70.1, 0.5)
GRID_W = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
SKIP_START = 60.0
MODEL_INPUTS = ["x_cpu", "x_gpu", "x_busy"]  # s of the job before the first reading the fit uses


# What the jobs trained (set by main() from the first job's machine.txt): the dataset and the depth
# limit, for titles and tables. The time check needs the time benchmark's case (ECBDL'14 15M, depth 10).
WHAT = "ECBDL'14 15M"
DEPTH_TXT = "depth ≤ 10"
DATA_NAMES = {"ecbdl": "ECBDL'14", "mnist8m": "MNIST8M", "higgs": "HIGGS", "susy": "SUSY",
              "sorel": "SOREL-20M"}


def read_case(machine):
    """machine.txt -> (dataset label, depth label)."""
    global WHAT, DEPTH_TXT
    data = re.search(r"^data (\S+)(?:.*\(first (\d+) rows\))?", machine, re.M)
    depth = re.search(r"^protocol -d (-?\d+)", machine, re.M)
    if data:
        name = Path(data.group(1)).parent.name
        WHAT = DATA_NAMES.get(name, name) + (f" {int(data.group(2)) / 1e6:g}M" if data.group(2) else "")
    if depth:
        DEPTH_TXT = "no depth limit" if int(depth.group(1)) < 0 else f"depth ≤ {depth.group(1)}"


def node_of(folder):
    """epyc1 -> epyc, dell4 -> dell, hopper -> hopper."""
    return re.sub(r"\d+$", "", folder.name)


def read_csv(path):
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        rows = list(csv.reader(f))
    head = rows[0]
    rows = [r for r in rows[1:] if len(r) == len(head)]
    return {c: np.array([float(r[i]) for r in rows]) for i, c in enumerate(head)}


def read_markers(path):
    events = []
    for line in open(path):
        t, rest = line.split(None, 1)
        words = rest.split()
        events.append((float(t), words[0], words[1] if len(words) > 1 else "",
                       dict(w.split("=", 1) for w in words[1:] if "=" in w)))
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


class Cumulative:
    """Energy of a counter between two times: the counter (J) is linear between reads."""

    def __init__(self, t, joules):
        self.t, self.j = np.asarray(t, float), np.asarray(joules, float)

    def between(self, a, b):
        return np.interp(b, self.t, self.j) - np.interp(a, self.t, self.j)

    def power(self):
        """W per sample interval, at the interval's end."""
        dt = np.diff(self.t)
        return np.concatenate([[np.nan], np.where(dt > 0, np.diff(self.j) / np.where(dt > 0, dt, 1), np.nan)])


class Job:
    def __init__(self, folder):
        self.folder = folder
        self.node = node_of(folder)
        self.samples = read_csv(folder / "samples.csv")
        self.t = self.samples["t"]
        # When each sample's NVML counters were read: pwrlog.py stamps t before
        # its reads, and an NVML call can block for ~2 s (ampere, when a GPU
        # tree ends); t_end is the time after the reads. Powercap is read right
        # after t.
        self.t_read = self.samples.get("t_end", self.t)
        self.events = read_markers(folder / "markers.txt")
        self.machine = (folder / "machine.txt").read_text()
        m = re.search(r"^job (\S+) on (\S+),", self.machine, re.M)
        self.job_id, self.host = (m.group(1), m.group(2)) if m else ("?", self.node)
        m = re.search(r"^used_gpu (\d+)", self.machine, re.M)
        self.used_gpu = int(m.group(1)) if m else None
        self.counters = self.read_counters()
        self.rapl = sorted(k for k in self.counters if k.startswith("cpu"))
        self.has_dram = "dram" in self.counters
        self.gpus = sorted(int(k[3:]) for k in self.counters if re.fullmatch(r"gpu\d+", k))
        self.psu_col = next((c for c in ("sys_W", "psu_W") if c in self.samples
                             and np.any(~np.isnan(self.samples[c]))), None)
        self.has_psu = self.psu_col is not None
        self.idle_windows = self.find_idle()
        if self.has_psu:
            self.psu_period = self.measure_period()
            self.slow = self.psu_period > SLOW_METER
            self.readings = self.find_readings()
            self.fit_meter()
        self.idle = {k: self.mean_power(k, self.idle_windows) for k in self.sources()}

    # ------------------------------------------------------------------ sensors
    def read_counters(self):
        """{name: Cumulative}: cpu<s> (RAPL package of socket s), dram (both
        sockets' DRAM), gpu<i> (NVML energy of GPU i), and the model's inputs
        x_cpu (packages + DRAM) and x_gpu (every GPU)."""
        ranges = dict(re.findall(r"^rapl_max (\S+) \S+ (\d+)", self.machine, re.M))
        out, dram = {}, []
        for col, v in self.samples.items():
            m = re.fullmatch(r"rapl_(intel-rapl:[\d:]+)_(package|dram)(?:-(\d+))?_uJ", col)
            if m:
                wrap = float(ranges.get(m.group(1), 2 ** 32))
                d = np.diff(v)
                d = np.where(d < 0, d + wrap, d)
                j = np.concatenate([[0.0], np.cumsum(d)]) / 1e6
                if m.group(2) == "package":
                    out["cpu%s" % m.group(3)] = Cumulative(self.t, j)
                else:
                    dram.append(j)
                continue
            m = re.fullmatch(r"gpu(\d+)_mJ", col)
            if m:
                out["gpu" + m.group(1)] = Cumulative(self.t_read, (v - v[0]) / 1e3)
        if dram:
            out["dram"] = Cumulative(self.t, np.sum(dram, axis=0))
        cpu = [out[k].j for k in out if k.startswith("cpu")] + ([out["dram"].j] if dram else [])
        out["x_cpu"] = Cumulative(self.t, np.sum(cpu, axis=0))
        gpus = [k for k in out if re.fullmatch(r"gpu\d+", k)]
        if gpus:  # on one time base: the sample times, the counters interpolated
            out["x_gpu"] = Cumulative(self.t, np.sum([np.interp(self.t, out[k].t, out[k].j) for k in gpus], axis=0))
        if "cpu_busy" in self.samples:
            # busy CPUs (/proc/stat, the fraction since the previous sample) x the CPU count: CPU-seconds
            n = int(re.search(r"^cpus available (\d+)", self.machine, re.M).group(1))
            b = self.samples["cpu_busy"] * n
            out["x_busy"] = Cumulative(self.t, np.concatenate([[0.0], np.cumsum(b[1:] * np.diff(self.t))]))
        return out

    def find_idle(self):
        starts = {kind: t for t, kind, *_ in self.events}
        windows = []
        for a, b in (("idle_start", "idle_end"), ("step_end", "step_idle_end"), ("idle2_start", "idle2_end")):
            if a in starts and b in starts:
                windows.append((starts[a] + IDLE_SKIP, starts[b]))
        return windows

    def sources(self):
        s = (["psu", "meter"] if self.has_psu else []) + self.rapl + (["dram"] if self.has_dram else [])
        return s + ["gpu%d" % g for g in self.gpus]

    def processes(self):
        """[(id, attrs, start, end, rc)] in order."""
        open_runs, out = {}, []
        for t, kind, run, attrs in self.events:
            if kind == "start":
                open_runs[run] = (t, attrs)
            elif kind == "end" and run in open_runs:
                s, a = open_runs.pop(run)
                out.append((run, a, s, t, attrs.get("rc", "0")))
        return out

    # ------------------------------------------------------------------ the meter
    def psu_values(self):
        v = self.samples[self.psu_col]
        ok = ~np.isnan(v)
        return self.t[ok], v[ok]

    def measure_period(self):
        """Median time between changes of the PSU reading: the meter's refresh
        period, or longer where consecutive refreshes give the same value."""
        t, v = self.psu_values()
        changes = t[1:][v[1:] != v[:-1]]
        return float(np.median(np.diff(changes))) if len(changes) > 2 else 0.0

    def find_readings(self):
        """The meter's readings [(time, watts)]: where a fast meter's value
        changes; a slow meter's refreshes. ampere's BMC takes a point reading
        about every 59 s and /run/power/psu_W shows it within a second; a
        refresh that repeats the previous value leaves no change in the file,
        so the refreshes between two changes are spread evenly at the period."""
        t, v = self.psu_values()
        ch = np.concatenate([[True], v[1:] != v[:-1]])
        ct, cv = t[ch], v[ch]
        if not self.slow:
            return ct[1:], cv[1:]
        gaps = np.diff(ct[1:])
        med = float(np.median(gaps))
        period = float(np.median(gaps[gaps < 1.5 * med]))
        rt, rv = [], []
        for (t0, x), t1 in zip(zip(ct[1:], cv[1:]), list(ct[2:]) + [t[-1] + period]):
            n = max(1, round((t1 - t0) / period))
            rt += [t0 + j * (t1 - t0) / n for j in range(n)]
            rv += [x] * n
        self.psu_period = period
        return np.array(rt), np.array(rv)

    def design(self, times, D, W):
        """The model's inputs for readings at `times`: [1, cpu, gpu] mean power
        over [t - D - W, t - D]."""
        hi, lo = times - D, times - D - W
        cols = [np.ones(len(times))]
        for k in MODEL_INPUTS:
            if k in self.counters:
                cols.append(self.counters[k].between(lo, hi) / W)
        return np.column_stack(cols)

    def fit_meter(self):
        rt, rv = self.readings
        keep = (rt > self.t[0] + SKIP_START + 12) & (rt < self.t[-1])
        rt, rv = rt[keep], rv[keep]
        best = None
        for W in GRID_W:
            for D in (GRID_D_SLOW if self.slow else GRID_D_FAST):
                X = self.design(rt, D, W)
                coef, *_ = np.linalg.lstsq(X, rv, rcond=None)
                rms = float(np.sqrt(np.mean((rv - X @ coef) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, W, D, coef)
        self.fit_rms, self.W, self.D, self.coef = best
        X = self.design(rt, self.D, self.W)
        self.fit_t, self.fit_y, self.fit_pred = rt, rv, X @ self.coef
        half = len(rt) // 2
        self.coef_halves = [np.linalg.lstsq(X[s], rv[s], rcond=None)[0] for s in (slice(0, half), slice(half, None))]
        # the meter's delay to the middle of its averaging window: what the
        # direct estimate shifts its readings by
        self.lag = self.D + self.W / 2

    def inputs(self):
        return [k for k in MODEL_INPUTS if k in self.counters]

    def fit_coef(self, keep=None, drop=()):
        """The model's coefficients fitted at the job's D and W on the readings
        in `keep` (a mask over fit_t; default all), without the inputs in
        `drop` (their coefficient 0)."""
        names = ["a"] + self.inputs()
        cols = [i for i, n in enumerate(names) if n not in drop]
        keep = np.ones(len(self.fit_y), bool) if keep is None else keep
        X = self.design(self.fit_t[keep], self.D, self.W)
        coef = np.zeros(len(names))
        coef[cols] = np.linalg.lstsq(X[:, cols], self.fit_y[keep], rcond=None)[0]
        return coef

    def readings_during(self, spans):
        """Mask over fit_t: the readings whose averaging window overlaps one of the spans."""
        lo, hi = self.fit_t - self.D - self.W, self.fit_t - self.D
        mask = np.zeros(len(self.fit_t), bool)
        for a, b in spans:
            mask |= (hi >= a) & (lo <= b)
        return mask

    def coef_by_name(self, coef=None):
        """{"a": W, "x_cpu": ..., "x_gpu": ..., "x_busy": ...}"""
        c = self.coef if coef is None else coef
        return dict(zip(["a"] + self.inputs(), c))

    def model_energy(self, a, b, coef=None):
        """PSU joules over [a, b] from the model."""
        c = self.coef if coef is None else coef
        e = c[0] * (b - a)
        for ci, k in zip(c[1:], self.inputs()):
            e += ci * self.counters[k].between(a, b)
        return float(e)

    def model_energy_in(self, windows, coef=None):
        return sum(self.model_energy(a, b, coef) for a, b in windows)

    def meter_energy(self, windows):
        """The meter's own joules over the windows, without the model: a fast
        meter's readings held until the next one, shifted back by D + W/2; a
        slow meter's point readings: the mean of those in the (shifted)
        windows x their total length. None when no reading falls in them."""
        length = sum(b - a for a, b in windows)
        if self.slow:
            rt, rv = self.readings
            vals = [v for a, b in windows for v in rv[(rt >= a + self.lag) & (rt < b + self.lag)]]
            return statistics.mean(vals) * length if vals else None
        t, v = self.psu_values()
        cum = np.concatenate([[0.0], np.cumsum(v[:-1] * np.diff(t))])
        return float(sum(np.interp(b + self.lag, t, cum) - np.interp(a + self.lag, t, cum) for a, b in windows))

    def residuals_in(self, windows):
        """Residuals (reading - model) of the readings whose averaging window
        lies inside one of the windows."""
        lo, hi = self.fit_t - self.D - self.W, self.fit_t - self.D
        res = self.fit_y - self.fit_pred
        inside = np.zeros(len(res), bool)
        for a, b in windows:
            inside |= (lo >= a) & (hi <= b)
        return res[inside]

    # ------------------------------------------------------------------ energy
    def energy(self, source, a, b):
        """Joules of `source` from a to b (Unix times)."""
        if source == "psu":
            return self.model_energy(a, b)
        if source == "meter":
            return self.meter_energy([(a, b)]) or float("nan")
        return float(self.counters[source].between(a, b))

    def energy_in(self, source, windows):
        if source == "meter":
            e = self.meter_energy(windows)
            return float("nan") if e is None else e
        return sum(self.energy(source, a, b) for a, b in windows)

    def mean_power(self, source, windows):
        if not windows:
            return float("nan")
        return self.energy_in(source, windows) / sum(b - a for a, b in windows)

    def power_series(self, source, a, b):
        """(times, watts) of a counter between a and b (per 0.1 s sample)."""
        c = self.counters[source]
        i0, i1 = np.searchsorted(c.t, a), np.searchsorted(c.t, b)
        return c.t[i0:i1], c.power()[i0:i1]

    def model_series(self, a, b):
        """(times, model W) per sample between a and b."""
        t, _ = self.power_series("x_cpu", a, b)
        p = np.full(len(t), self.coef[0])
        for ci, k in zip(self.coef[1:], self.inputs()):
            p = p + ci * self.power_series(k, a, b)[1]
        return t, p

    def peak(self, source, a, b):
        """Highest 0.1 s power of a counter (or the model) in [a, b]."""
        if source == "psu":
            _, p = self.model_series(a, b)
        else:
            _, p = self.power_series(source, a, b)
        p = p[~np.isnan(p)]
        return float(p.max()) if len(p) else float("nan")


def components(job, per_source, impl):
    """Map source values to chart components: CPU packages, DRAM, the GPU in
    use (GPU implementations only) and the rest of the PSU."""
    out = {k: per_source[k] for k in job.rapl}
    if job.has_dram:
        out["dram"] = per_source["dram"]
    if job.gpus and impl in GPU_IMPLS:
        out["gpu"] = per_source["gpu%d" % job.used_gpu]
    others = [g for g in job.gpus if not (impl in GPU_IMPLS and g == job.used_gpu)]
    if others:
        out["gpus_other"] = sum(per_source["gpu%d" % g] for g in others)
    if job.has_psu:
        out["rest"] = per_source["psu"] - sum(out.values())
    return out


def analyse(job):
    rows = []
    for run, attrs, start, end, rc in job.processes():
        if not run.startswith("rep"):
            continue
        tree = read_tree_output(job.folder / "out" / f"{run}.txt")
        if rc != "0" or not tree["windows"]:
            print(f"  {job.node} {run}: failed (rc {rc}), skipped", file=sys.stderr)
            continue
        windows = tree["windows"]
        k = len(windows)
        spans = [(w[0], w[1]) for w in windows]
        a, b = windows[0][0], windows[-1][1]
        span = sum(w1 - w0 for w0, w1 in spans)  # training time only, gaps left out
        energy = {s: job.energy_in(s, spans) / k for s in job.sources()}
        idle = {s: job.idle[s] * span / k for s in job.sources()}
        comp_e = components(job, energy, attrs["impl"])
        comp_idle = components(job, idle, attrs["impl"])
        total = energy["psu"] if job.has_psu else sum(comp_e.values())
        res = job.residuals_in(spans) if job.has_psu else np.array([])
        steady = windows[1:] if k >= 3 else windows
        rows.append({
            "node": job.node, "host": job.host, "impl": attrs["impl"], "algo": attrs["algo"],
            "rep": int(run.split("-")[0][3:]), "repeat": k,
            "train_s": statistics.mean(w[2] for w in windows), "first_tree_s": windows[0][2],
            "steady_train_s": statistics.mean(w[2] for w in steady),
            "span_s": span / k, "load_s": tree["load_s"], "nodes": tree["nodes"], "depth": tree["depth"],
            "energy_J": total, "above_idle_J": total - (idle["psu"] if job.has_psu else sum(comp_idle.values())),
            "power_W": total / (span / k),
            "idle_W": (idle["psu"] if job.has_psu else sum(comp_idle.values())) / (span / k),
            "psu_J": energy.get("psu", float("nan")), "psu_idle_J": idle.get("psu", float("nan")),
            "meter_J": energy.get("meter", float("nan")),
            "model_half1_J": job.model_energy_in(spans, job.coef_halves[0]) / k if job.has_psu else float("nan"),
            "model_half2_J": job.model_energy_in(spans, job.coef_halves[1]) / k if job.has_psu else float("nan"),
            "res_in_n": len(res), "res_in_sum": float(res.sum()), "res_in_sq": float((res ** 2).sum()),
            "peak_psu_W": job.peak("psu", a, b) if job.has_psu else float("nan"),
            "peak_cpu_W": sum(job.peak(c, a, b) for c in job.rapl),
            "peak_gpu_W": job.peak("gpu%d" % job.used_gpu, a, b) if job.gpus and attrs["impl"] in GPU_IMPLS
            else float("nan"),
            **{f"E_{c}": v for c, v in comp_e.items()},
            **{f"Eidle_{c}": v for c, v in comp_idle.items()},
            "_window": (a, b, start, end), "_spans": spans,
        })
    if job.has_psu:
        model_checks(job, rows)
    return rows


def model_checks(job, rows):
    """Per row: the energy with the model fitted without the case's readings
    (every process of the same backend and algorithm: warm-ups and checks too),
    the held-out readings' residuals in that process, the energy with the model
    fitted without the busy-CPUs term, and the energy without the unused GPUs."""
    nobusy = job.fit_coef(drop=("x_busy",))
    c_gpu = job.coef_by_name().get("x_gpu", 0.0)
    procs = job.processes()
    for impl, algo in {(r["impl"], r["algo"]) for r in rows}:
        held = job.readings_during([(s, e) for _, at, s, e, _ in procs
                                    if at.get("impl") == impl and at.get("algo") == algo])
        coef = job.fit_coef(keep=~held)
        for r in rows:
            if (r["impl"], r["algo"]) != (impl, algo):
                continue
            k = r["repeat"]
            mine = held & job.readings_during([r["_window"][2:4]])
            res = job.fit_y[mine] - job.design(job.fit_t[mine], job.D, job.W) @ coef
            r.update({"model_loco_J": job.model_energy_in(r["_spans"], coef) / k,
                      "model_nobusy_J": job.model_energy_in(r["_spans"], nobusy) / k,
                      "loco_n": int(mine.sum()), "loco_sum": float(res.sum()),
                      "loco_sq": float((res ** 2).sum())})
    for r in rows:
        r["energy_wo_unused_gpus_J"] = r["energy_J"] - c_gpu * r.get("E_gpus_other", 0.0)


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
        s = {"node": node, "host": rs[0]["host"], "impl": impl, "algo": algo, "runs": len(rs),
             "repeat": rs[0]["repeat"], "nodes": sorted({r["nodes"] for r in rs})}
        for key in ("train_s", "load_s", "energy_J", "above_idle_J", "power_W", "idle_W", "steady_train_s",
                    "first_tree_s", "span_s", "peak_cpu_W", "peak_gpu_W", "peak_psu_W", "psu_J", "psu_idle_J",
                    "meter_J", "model_half1_J", "model_half2_J", "model_loco_J", "model_nobusy_J",
                    "energy_wo_unused_gpus_J"):
            s[key], s[key + "_sd"] = mean_sd([r.get(key) for r in rs])
        n = sum(r["res_in_n"] for r in rs)
        s["res_in_n"] = n
        if n:
            mean = sum(r["res_in_sum"] for r in rs) / n
            sd = math.sqrt(max(sum(r["res_in_sq"] for r in rs) / n - mean ** 2, 0.0))
            s["res_in_W"], s["res_in_se_W"] = mean, sd / math.sqrt(n)
        else:
            s["res_in_W"] = s["res_in_se_W"] = float("nan")
        n = sum(r.get("loco_n", 0) for r in rs)
        s["loco_n"] = n
        if n:
            mean = sum(r["loco_sum"] for r in rs) / n
            sd = math.sqrt(max(sum(r["loco_sq"] for r in rs) / n - mean ** 2, 0.0))
            s["loco_res_W"], s["loco_res_se_W"] = mean, sd / math.sqrt(n)
        else:
            s["loco_res_W"] = s["loco_res_se_W"] = float("nan")
        for c in COMPONENTS:
            if f"E_{c}" in rs[0]:
                s[f"E_{c}"] = statistics.mean(r[f"E_{c}"] for r in rs)
                s[f"Eidle_{c}"] = statistics.mean(r[f"Eidle_{c}"] for r in rs)
                s[f"P_{c}"] = s[f"E_{c}"] / s["span_s"]
        summary.append(s)
    key = lambda s: (NODE_ORDER.index(s["node"]) if s["node"] in NODE_ORDER else 9,  # noqa: E731
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


SHORT = {"simple": "Simple", "serial": "Opt", "parallel": "Multi", "gpuonly": "GPU", "cuda": "Hybrid"}


def short_label(s):
    """Two lines for a narrow bar: the backend, short, and the node."""
    return f"{SHORT[s['impl']]}\n{s['node']}"


def trees(s):
    """Which trees were measured, for the tables."""
    return f"{s['runs']} × {s['repeat']}"


def pct(a, b):
    return f"{(a / b - 1) * 100:+.1f}%"


def write_tables(folder, jobs, rows, summary, warmup):
    keys = list(dict.fromkeys(k for r in rows for k in r if not k.startswith("_")))
    with open(folder / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    skeys = sorted({k for s in summary for k in s}, key=lambda k: (k not in summary[0], k))
    with open(folder / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, skeys)
        w.writeheader()
        w.writerows(summary)
    meters = meter_rows(jobs)
    with open(folder / "meter.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(meters[0]) if meters else [])
        w.writeheader()
        w.writerows(meters)
    lines = []
    for algo, title in ALGOS.items():
        lines += [f"### {title}, {WHAT}, {DEPTH_TXT}, no pruning", "",
                  "| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | "
                  "E per tree | E above idle | E without the unused GPUs | split of E |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in summary:
            if s["algo"] != algo:
                continue
            split = ", ".join(f"{COMPONENTS[c][0].split(' (')[0]} {fmt_j(s['E_' + c])}"
                              for c in COMPONENTS if f"E_{c}" in s)
            lines.append(
                f"| {IMPLS[s['impl']]} | {s['node']} | {trees(s)} | "
                f"{fmt_s(s['train_s'])} ± {s['train_s_sd'] / s['train_s'] * 100:.1f}% | "
                f"{s['power_W']:.0f} W | {s['power_W'] - s['idle_W']:.0f} W | {s['peak_psu_W']:.0f} W | "
                f"{fmt_j(s['energy_J'])} ± {s['energy_J_sd'] / s['energy_J'] * 100:.1f}% | "
                f"{fmt_j(s['above_idle_J'])} | {wo_unused(s)} | {split} |")
        lines.append("")
    lines += meter_table(jobs)
    lines += method_check(summary)
    lines += time_check(summary)
    lines += warmup_table(warmup)
    (folder / "summary.md").write_text("\n".join(lines))


def wo_unused(s):
    """E without the unused GPUs, or – on a node without GPUs."""
    if "E_gpus_other" not in s:
        return "–"
    return f"{fmt_j(s['energy_wo_unused_gpus_J'])} ({pct(s['energy_wo_unused_gpus_J'], s['energy_J'])})"


def meter_rows(jobs):
    out = []
    nan = float("nan")
    for job in jobs.values():
        if not job.has_psu:
            continue
        row = {"node": job.node, "host": job.host, "job": job.job_id, "meter": job.psu_col,
               "period_s": job.psu_period, "slow": job.slow, "readings": len(job.fit_y),
               "D_s": job.D, "W_s": job.W}
        for tag, coef in (("", None), ("_half1", job.coef_halves[0]), ("_half2", job.coef_halves[1])):
            c = job.coef_by_name(coef)
            row.update({f"a_W{tag}": c["a"], f"b_cpu{tag}": c.get("x_cpu", nan), f"c_gpu{tag}": c.get("x_gpu", nan),
                        f"d_busy{tag}": c.get("x_busy", nan)})
        row.update({"rms_W": job.fit_rms, "idle_model_W": job.idle["psu"],
                    "idle_meter_W": job.mean_power("meter", job.idle_windows)})
        out.append(row)
    return out


def model_text(m, tag=""):
    """The model of a meter_rows() row as text: "59 W + 1.13·CPU + 0.98·GPUs − 0.4 W·busy CPUs"."""
    t = f"{m['a_W' + tag]:.0f} W + {m['b_cpu' + tag]:.2f}·CPU"
    if not math.isnan(m["c_gpu" + tag]):
        t += f" + {m['c_gpu' + tag]:.2f}·GPUs"
    if not math.isnan(m["d_busy" + tag]):
        d = m["d_busy" + tag]
        t += f" {'+' if d >= 0 else '−'} {abs(d):.2f} W·busy CPUs"
    return t


def meter_table(jobs):
    lines = ["### The meter model", "",
             "Each node's meter, fitted over every reading of its job: reading(t) = mean over "
             "[t − D − W, t − D] of a + b × CPU + c × GPUs + d × busy CPUs, where CPU = RAPL packages + DRAM "
             "(W), GPUs = every GPU's NVML power (W) and busy CPUs = how many CPUs are busy (/proc/stat). "
             "Halves: the same fit on the first and on the second half of the job's readings.", "",
             "| Node | meter | refresh | readings | D | W | model | first half | second half | rms | "
             "idle: model / meter |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in meter_rows(jobs):
        lines.append(f"| {m['host']} | {m['meter']} | {m['period_s']:.1f} s{' (point)' if m['slow'] else ''} | "
                     f"{m['readings']} | {m['D_s']:.1f} s | {m['W_s']:g} s | {model_text(m)} | "
                     f"{model_text(m, '_half1')} | {model_text(m, '_half2')} | {m['rms_W']:.1f} W | "
                     f"{m['idle_model_W']:.0f} / {m['idle_meter_W']:.0f} W |")
    return lines + [""]


def method_check(summary):
    """Model against the meter itself, per case."""
    lines = ["### PSU energy: the model against the meter", "",
             "E per tree from the model (used everywhere above) against: the model with each half's "
             "coefficients; the model fitted without any reading of this case's processes (held out: "
             "a load the model has not seen) and those held-out readings' mean residual ± its standard "
             "error; the model without the busy-CPUs term; the meter's own readings over the same windows (held, shifted by D + W/2; "
             "ampere: the mean of its point readings in the windows), which is exact only for windows "
             "much longer than the meter's refresh; and the mean residual (reading − model) of the "
             "readings whose averaging window lies inside the training windows, ± its standard error, "
             "relative to the mean power.", "",
             "| Implementation | Node | Algorithm | E per tree (model) | halves | held out | held-out readings: "
             "mean residual | no busy term | meter | readings inside | mean residual inside |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        if math.isnan(s["psu_J"]):
            continue
        e = s["psu_J"]
        halves = f"{pct(s['model_half1_J'], e)} / {pct(s['model_half2_J'], e)}"
        meter = "–" if math.isnan(s["meter_J"]) else f"{fmt_j(s['meter_J'])} ({pct(s['meter_J'], e)})"
        res = "–" if not s["res_in_n"] else (f"{s['res_in_W']:+.1f} ± {s['res_in_se_W']:.1f} W "
                                             f"({s['res_in_W'] / s['power_W'] * 100:+.1f}%)")
        loco = "–" if math.isnan(s["model_loco_J"]) else pct(s["model_loco_J"], e)
        loco_res = "–" if not s["loco_n"] else (f"{s['loco_n']}: {s['loco_res_W']:+.1f} ± "
                                                f"{s['loco_res_se_W']:.1f} W")
        nobusy = "–" if math.isnan(s["model_nobusy_J"]) else pct(s["model_nobusy_J"], e)
        lines.append(f"| {IMPLS[s['impl']]} | {s['node']} | {ALGOS[s['algo']].split(' (')[0]} | {fmt_j(e)} | "
                     f"{halves} | {loco} | {loco_res} | {nobusy} | {meter} | {s['res_in_n']} | {res} |")
    return lines + [""]


# The time benchmark of the same cases: <TIME_BENCH>/<prefix>-<node>-ecbdl_15m.
# Which node ran the time benchmark (bench/tools/cluster/final.sh): epyc's Multicore
# and Serial-Simple on epyc1, Serial-Opt on epyc2 (the same hardware); dell's
# Serial-Opt and Serial-Simple on dell2, which has dell4's half memory
# channels, its Multicore on dell1 (all channels: not comparable).
TIME_BENCH = Path(os.environ.get("POWER_TIME_BENCH",
                                 Path(__file__).resolve().parents[2] / "cluster-ecbdl"))
TIME_PREFIX = {"simple": "simple", "serial": "serial", "parallel": "cpu", "gpuonly": "gpuonly", "cuda": "cuda"}
TIME_SKIP = {("dell", "parallel")}
TIME_TOLERANCE = 0.03


def time_bench(node, impl, algo):
    """Mean training time of the case in the time benchmark, or None."""
    path = TIME_BENCH / f"{TIME_PREFIX[impl]}-{node}-ecbdl_15m" / "runs.csv"
    if (WHAT, DEPTH_TXT) != ("ECBDL'14 15M", "depth ≤ 10") or (node, impl) in TIME_SKIP or not path.exists():
        return None
    protocol = "cart_depth10" if algo == "cart" else "c45_depth10_unpruned"
    t = [float(r["train_seconds"]) for r in read_csv_rows(path)
         if r["protocol"] == protocol and r.get("status", "ok") == "ok"]
    return statistics.mean(t) if t else None


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def time_check(summary):
    """Markdown lines: each case's time per tree against its first tree and the
    time benchmark. Prints a warning for every difference above TIME_TOLERANCE."""
    ref = (f"the time benchmark ({TIME_BENCH.name}, ecbdl_15m; dell's Multicore ran on dell1, not comparable)"
           if (WHAT, DEPTH_TXT) == ("ECBDL'14 15M", "depth ≤ 10") else f"the time benchmark (none for {WHAT}, {DEPTH_TXT})")
    lines = ["### Time check", "",
             f"Time per tree (./tree's train time) against the first tree of each process and {ref}. "
             f"Flagged (!) above {TIME_TOLERANCE:.0%}.", "",
             "| Implementation | Node | Algorithm | t per tree | first tree | time benchmark |",
             "|---|---|---|---|---|---|"]
    for s in summary:
        t = s["train_s"]
        bench = time_bench(s["node"], s["impl"], s["algo"])
        cells = []
        for name, ref in (("first tree", s["first_tree_s"]), ("time benchmark", bench)):
            if ref is None or math.isnan(ref):
                cells.append("–")
                continue
            d = t / ref - 1
            bad = abs(d) > TIME_TOLERANCE
            cells.append(f"{fmt_s(ref)} ({d:+.1%}){' !' if bad else ''}")
            if bad:
                print(f"WARNING: {s['node']} {s['impl']} {s['algo']}: {fmt_s(t)} per tree, "
                      f"{name} {fmt_s(ref)} ({d:+.1%})", file=sys.stderr)
        lines.append(f"| {IMPLS[s['impl']]} | {s['node']} | {ALGOS[s['algo']].split(' (')[0]} | "
                     f"{fmt_s(t)} | " + " | ".join(cells) + " |")
    return lines + [""]


def read_warmup(folder):
    """The warm-up check of the GPU nodes: check-<impl>-w<S> processes, in the
    job folder or in a check-<node> folder next to it. -> [row per tree]."""
    out = []
    for sub in sorted(folder.iterdir()):
        for f in sorted((sub / "out").glob("check-*.txt")) if (sub / "out").is_dir() else []:
            m = re.fullmatch(r"check-(\w+)-w([\d.]+)", f.stem)
            if not m:
                continue
            tree = read_tree_output(f)
            for i, (_, _, s) in enumerate(tree["windows"]):
                out.append({"node": node_of(sub).replace("check-", ""), "impl": m.group(1),
                            "warmup_s": float(m.group(2)), "tree": i + 1, "train_s": s})
    return out


def warmup_table(warmup):
    if not warmup:
        return []
    lines = ["### Warm-up check", "",
             "GPU and Hybrid, CART, one process per warm-up (--repeat-warmup S: each tree's child keeps its "
             "CPU threads busy S s before its timer). Mean of trees 1–2 and of the rest.", "",
             "| Node | Implementation | warm-up | trees | trees 1–2 | trees 3– | slowest |", "|---|---|---|---|---|---|---|"]
    cases = sorted({(w["node"], w["impl"], w["warmup_s"]) for w in warmup},
                   key=lambda c: (NODE_ORDER.index(c[0]) if c[0] in NODE_ORDER else 9, IMPL_ORDER.index(c[1]), c[2]))
    for node, impl, ws in cases:
        t = [w["train_s"] for w in sorted(warmup, key=lambda w: w["tree"])
             if (w["node"], w["impl"], w["warmup_s"]) == (node, impl, ws)]
        lines.append(f"| {node} | {IMPLS[impl]} | {ws:g} s | {len(t)} | {statistics.mean(t[:2]):.3f} s | "
                     f"{statistics.mean(t[2:]) if len(t) > 2 else float('nan'):.3f} s | {max(t):.3f} s |")
    return lines + [""]


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
    fig, axes = plt.subplots(1, 2, figsize=(21, 5.8), sharey=True)
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
        ax.set_xticks(range(len(cases)), [short_label(s) for s in cases], fontsize=7.5)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID)
        style(ax)
    axes[0].set_ylabel("mean power while training (W)", color=INK)
    for ax in axes:
        unit_ticks(ax, "y", "W")
    handles = [Patch(color=COMPONENTS[c][1], label=COMPONENTS[c][0]) for c in COMPONENTS if c in used]
    handles += [Patch(facecolor="#999", alpha=0.38, label="lighter part: idle power of that component")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(f"{WHAT}, {DEPTH_TXT}: mean power of the whole server (PSU, meter model) while training one tree, "
                 "split by sensor", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.text(0.99, 0.01, "Simple = Serial-Simple, Opt = Serial-Opt, Multi = Multicore", ha="right", va="bottom",
             fontsize=8, color=MUTED)
    save(fig, folder / "chart-power.png", dpi=150)
    plt.close(fig)


def chart_energy(folder, summary):
    """Energy per tree on a log scale: the whole bar is total, the dark inner bar above idle."""
    fig, axes = plt.subplots(1, 2, figsize=(21, 5.8), sharey=True)
    for ax, (algo, title) in zip(axes, ALGOS.items()):
        cases = ordered(summary, algo)
        for i, s in enumerate(cases):
            color = IMPL_COLORS[s["impl"]]
            ax.bar(i, s["energy_J"] / 1e3, color=color, alpha=0.45, width=0.75)
            ax.bar(i, max(s["above_idle_J"], 1) / 1e3, color=color, width=0.4)
            ax.errorbar(i, s["energy_J"] / 1e3, yerr=s["energy_J_sd"] / 1e3, color=INK, capsize=2, lw=0.8)
            if "E_gpus_other" in s:  # the server without its unused GPUs
                ax.plot([i - 0.375, i + 0.375], [s["energy_wo_unused_gpus_J"] / 1e3] * 2, color=INK, lw=1.4,
                        ls=(0, (2, 1.5)))
            ax.text(i, s["energy_J"] / 1e3 * 1.15, fmt_j(s["energy_J"]), ha="center", va="bottom", fontsize=7.5,
                    color=INK)
            ax.text(i, max(s["above_idle_J"], 1) / 1e3 * 1.06, fmt_j(s["above_idle_J"]), ha="center", va="bottom",
                    fontsize=7, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(range(len(cases)), [short_label(s) for s in cases], fontsize=7.5)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, which="both")
        style(ax)
    axes[0].set_ylabel("energy per tree (kJ, log scale)", color=INK)
    for ax in axes:
        unit_ticks(ax, "y", "kJ")
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 4)
    fig.legend(handles=[Patch(facecolor="#888", alpha=0.45, label="wide bar: total energy while training"),
                        Patch(facecolor="#888", label="narrow bar: above idle, what the run adds to the idle server"),
                        plt.Line2D([], [], color=INK, lw=1.4, ls=(0, (2, 1.5)),
                                   label="dashed: GPU nodes without their unused GPUs (model's GPU coefficient × their NVML energy)")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(f"{WHAT}, {DEPTH_TXT}: whole-server energy (PSU, meter model) to train one tree "
                 "(mean of the runs; whisker ± 1 sd)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.99, 0.01, "Simple = Serial-Simple, Opt = Serial-Opt, Multi = Multicore", ha="right", va="bottom",
             fontsize=8, color=MUTED)
    save(fig, folder / "chart-energy.png", dpi=150)
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
    ax.set_title(f"{WHAT}, {DEPTH_TXT}: time and energy per tree (grey lines: constant power)",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    save(fig, folder / "chart-time-energy.png", dpi=150)
    plt.close(fig)


def chart_cpu_gpu(folder, summary):
    """Nodes with CPU (RAPL) and GPU (NVML) sensors: one chart per node, one
    panel per algorithm (CART above C4.5). Each implementation gets up to six thin bars on three log axes:
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
        algos = [a for a in ALGOS if any(s["node"] == node and s["algo"] == a for s in cases)]
        node_gpu = any("E_gpu" in s for s in cases if s["node"] == node)
        fig, axs = plt.subplots(len(algos), 1, figsize=(22 if node_gpu else 15, 6.4 * len(algos)), squeeze=False)
        for row, (algo, ax_e) in enumerate(zip(algos, axs[:, 0])):
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
            if row == 0:
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
            if has_psu and node in METERS:
                m = METERS[node]
                psu_note = (f" PSU: the meter (refresh {m['period_s']:.0f} s) modelled from the fast sensors, "
                            f"fit rms {m['rms_W']:.0f} W.")
            ax_e.set_title(f"{ALGOS[algo]}. Means of runs × trees: {n}", fontsize=10.5, color=INK, loc="left",
                           fontweight="bold")
            if row == 0:
                fig.suptitle(f"Energy, power and time to train one tree on {WHAT} ({DEPTH_TXT}), {hw}\n"
                             f"Measured: {what}.{psu_note} Every tree a fresh process's only tree.",
                             fontsize=9.5, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        save(fig, folder / f"chart-cpu-gpu-{node}.png", dpi=150)
        plt.close(fig)




METERS = {}  # node -> meter_rows() row, for chart titles


def phase_of(job, times):
    """Label every time: idle, staircase, a run's implementation, or other."""
    labels = np.array(["other"] * len(times), dtype=object)
    for a, b in job.idle_windows:
        labels[(times >= a) & (times <= b)] = "idle"
    marks = [(t, kind) for t, kind, *_ in job.events]
    starts = [t for t, k in marks if k == "cal_start"]
    ends = [t for t, k in marks if k == "cal_end"]
    for a, b in zip(starts, ends):
        labels[(times >= a) & (times <= b)] = "staircase"
    for run, attrs, s, e, _ in job.processes():
        if run.startswith("rep"):
            labels[(times >= s) & (times <= e)] = attrs["impl"]
    return labels


PHASE_COLORS = {"idle": "#b9b8b3", "staircase": "#3d3c3a", "other": "#e4e3df", **IMPL_COLORS}


def chart_meter(folder, jobs):
    """Per node: every meter reading against the model (top), and a stretch of
    one GPU (or Multicore) process: the model at 0.1 s, the model as the meter
    would show it, and the readings (bottom)."""
    jobs = [j for j in sorted(jobs.values(), key=lambda j: NODE_ORDER.index(j.node)
                              if j.node in NODE_ORDER else 9) if j.has_psu]
    if not jobs:
        return
    fig, axes = plt.subplots(2, len(jobs), figsize=(4.6 * len(jobs), 8.4), squeeze=False,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    for col, job in enumerate(jobs):
        ax = axes[0][col]
        ph = phase_of(job, job.fit_t - job.D - job.W / 2)
        for p in ["other", "idle", "staircase"] + IMPL_ORDER:
            m = ph == p
            if m.any():
                ax.scatter(job.fit_pred[m], job.fit_y[m], s=4 if not job.slow else 10, color=PHASE_COLORS[p],
                           alpha=0.5 if not job.slow else 0.8, lw=0,
                           label={"other": "warm-up, gaps", "idle": "idle", "staircase": "CPU staircase"}.get(p, IMPLS.get(p, p)))
        lo = min(job.fit_pred.min(), job.fit_y.min()) - 20
        hi = max(job.fit_pred.max(), job.fit_y.max()) + 20
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"{job.host}: {len(job.fit_y)} readings, refresh {job.psu_period:.1f} s"
                     f"{' (point)' if job.slow else ''}\n{model_text(METERS[job.node])}\n"
                     f"D {job.D:.1f} s, W {job.W:g} s; rms {job.fit_rms:.1f} W", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("model, as the meter would read it (W)", fontsize=8, color=INK)
        if col == 0:
            ax.set_ylabel("meter reading (W)", fontsize=8, color=INK)
        ax.tick_params(labelsize=7)
        ax.grid(color=GRID, lw=0.5)
        style(ax)
        ax.legend(fontsize=6.5, frameon=False, markerscale=2, loc="upper left")
        # a stretch of a run
        ax = axes[1][col]
        procs = [p for p in job.processes() if p[0].startswith("rep")]
        pick = next((p for impl in ("gpuonly", "cuda", "parallel") for p in procs if p[1]["impl"] == impl), None)
        if pick is None:
            ax.axis("off")
            continue
        run, attrs, s, e, _ = pick
        tree = read_tree_output(job.folder / "out" / f"{run}.txt")
        w0 = tree["windows"][min(2, len(tree["windows"]) - 1)][0] - 3
        span = 300 if job.slow else 60
        a, b = w0, min(w0 + span, e)
        t, p = job.model_series(a - job.D - job.W, b)
        ax.plot(t - a, p, color=MUTED, lw=0.6, label="model, every 0.1 s")
        tt = np.linspace(a, b, int((b - a) * 10))
        ax.plot(tt - a, job.design(tt, job.D, job.W) @ job.coef, color=PSU_COLOR, lw=1.3,
                label=f"model as the meter reads it (D {job.D:.1f} s, W {job.W:g} s)")
        m = (job.fit_t >= a) & (job.fit_t <= b)
        ax.scatter(job.fit_t[m] - a, job.fit_y[m], s=12 if not job.slow else 30, color="#d14f8e", zorder=3,
                   label="meter readings")
        for w in tree["windows"]:
            if w[1] > a and w[0] < b:
                ax.axvspan(max(w[0], a) - a, min(w[1], b) - a, color=IMPL_COLORS[attrs["impl"]], alpha=0.15, lw=0)
        ax.set_xlim(0, b - a)
        ax.set_title(f"{IMPLS[attrs['impl']]}, {attrs['algo'].upper().replace('C45', 'C4.5')}: "
                     f"trees (shaded) and the gaps between them", fontsize=8.5, loc="left", color=INK)
        ax.set_xlabel("s", fontsize=8, color=INK)
        if col == 0:
            ax.set_ylabel("whole-server power (W)", fontsize=8, color=INK)
        ax.tick_params(labelsize=7)
        ax.grid(color=GRID, lw=0.5)
        style(ax)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("The PSU meter modelled from the fast sensors (RAPL CPU packages + DRAM, NVML GPUs, every 0.1 s)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, folder / "chart-meter.png", dpi=140)
    plt.close(fig)


def chart_methods(folder, summary):
    """Per case: the meter's own energy and the readings inside the training
    windows, relative to the model's energy."""
    cases = [s for s in summary if not math.isnan(s["psu_J"])]
    if not cases:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(cases) + 1.6), sharey=True)
    y = np.arange(len(cases))
    for ax, (name, val, err) in zip(axes, (
            ("meter's own energy over the windows vs model (%)",
             [(s["meter_J"] / s["psu_J"] - 1) * 100 for s in cases], None),
            ("mean residual of the readings inside the windows (% of power, ± 2 se)",
             [s["res_in_W"] / s["power_W"] * 100 if s["res_in_n"] else float("nan") for s in cases],
             [2 * s["res_in_se_W"] / s["power_W"] * 100 if s["res_in_n"] else 0 for s in cases]))):
        ax.barh(y, val, color=[NODE_COLORS.get(s["node"], INK) for s in cases], height=0.6,
                xerr=err, error_kw={"lw": 0.8, "capsize": 2, "color": INK})
        ax.axvline(0, color=INK, lw=0.7)
        for v in (-3, 3):
            ax.axvline(v, color=GRID, lw=0.8, ls="--")
        ax.set_title(name, fontsize=9, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.5)
        unit_ticks(ax, "x", "%")
        style(ax)
    axes[0].set_yticks(y, [f"{label(s)} · {s['algo'].upper().replace('C45', 'C4.5')} "
                           f"({fmt_s(s['train_s'])}/tree)" for s in cases], fontsize=7.5)
    axes[0].invert_yaxis()
    fig.suptitle("PSU energy per tree: the meter model against the meter (trees of a few seconds: "
                 "only the model resolves them)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, folder / "chart-methods.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def chart_warmup(folder, warmup):
    if not warmup:
        return
    nodes = [n for n in NODE_ORDER if any(w["node"] == n for w in warmup)]
    fig, axes = plt.subplots(1, len(nodes), figsize=(6.5 * len(nodes), 3.8), squeeze=False)
    for ax, node in zip(axes[0], nodes):
        for impl in ("gpuonly", "cuda"):
            for ws in sorted({w["warmup_s"] for w in warmup if w["node"] == node}):
                pts = sorted((w["tree"], w["train_s"]) for w in warmup
                             if (w["node"], w["impl"], w["warmup_s"]) == (node, impl, ws))
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[impl], ms=4,
                            color=IMPL_COLORS[impl], alpha=0.35 + 0.65 * (ws == max(w["warmup_s"] for w in warmup)),
                            label=f"{IMPLS[impl]}, warm-up {ws:g} s")
        ax.set_title(f"{node}: CART tree time by tree of the process", fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("tree (each in its own forked child)", fontsize=8, color=INK)
        ax.set_ylabel("training time", fontsize=8, color=INK)
        unit_ticks(ax, "y", "s")
        ax.grid(color=GRID, lw=0.5)
        style(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, folder / "chart-warmup.png", dpi=140)
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
        job = next(j for j in jobs.values() if j.node == key[0])
        a, b, start, end = r["_window"]
        lo, hi = start - 10, end + 10
        if job.has_psu:
            t, p = job.model_series(lo, hi)
            ax.plot(t - start, p, color=PSU_COLOR, lw=0.6, label="PSU model")
            m = (job.readings[0] >= lo) & (job.readings[0] <= hi)
            ax.plot(job.readings[0][m] - start - job.lag, job.readings[1][m], color="#d14f8e", lw=0.9,
                    drawstyle="steps-post", label="meter (shifted by its delay)")
        if job.gpus and key[1] in GPU_IMPLS:
            t, p = job.power_series("gpu%d" % job.used_gpu, lo, hi)
            ax.plot(t - start, p, lw=0.7, color=COMPONENTS["gpu"][1])
        for k in job.rapl:
            t, p = job.power_series(k, lo, hi)
            ax.plot(t - start, p, lw=0.6, color=COMPONENTS[k][1])
        for w0, w1 in ((a, b),):
            ax.axvspan(w0 - start, w1 - start, color=IMPL_COLORS[key[1]], alpha=0.12, lw=0)
        unit_ticks(ax, "y", "W")
        unit_ticks(ax, "x", "s")
        ax.set_title(f"{IMPLS[key[1]]}, {key[2].upper().replace('C45', 'C4.5')} · {key[0]}  "
                     f"({r['repeat']} tree{'s' if r['repeat'] > 1 else ''})", fontsize=8.5, loc="left", color=INK)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(color=GRID, lw=0.5)
        style(ax)
    handles = [plt.Line2D([], [], color=PSU_COLOR, label="whole server: the meter model"),
               plt.Line2D([], [], color="#d14f8e", label="whole server: the meter's readings (shifted by its delay)"),
               plt.Line2D([], [], color=COMPONENTS["gpu"][1], label="GPU in use (NVML; GPU runs only)"),
               plt.Line2D([], [], color=COMPONENTS["cpu0"][1], label="CPU socket 0 (RAPL)"),
               plt.Line2D([], [], color=COMPONENTS["cpu1"][1], label="CPU socket 1 (RAPL)"),
               Patch(color="#888", alpha=0.2, label="from the first to the last tree's training (shaded); "
                                                    "before: loading the CSV")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Power over time, first run of each case (W against seconds since the process started)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save(fig, folder / "chart-traces.png", dpi=120)
    plt.close(fig)


def main():
    folder = Path(sys.argv[1])
    jobs = {}
    for sub in sorted(folder.iterdir()):
        if sub.name.startswith("check-") or not (sub / "markers.txt").exists():
            continue
        if (sub / "samples.csv").exists() or (sub / "samples.csv.gz").exists():
            jobs[sub.name] = Job(sub)
    if jobs:
        read_case(next(iter(jobs.values())).machine)
    commits = {name: re.search(r"^commit (\S+)", job.machine, re.M) for name, job in jobs.items()}
    STALE[:] = [sum(B.out_of_date(m and m.group(1), jobs[name].host) for name, m in commits.items()), len(jobs)]
    rows = []
    for name, job in jobs.items():
        if job.has_psu:
            print(f"{name}: meter {job.psu_col} every {job.psu_period:.1f} s, D {job.D:.1f} s W {job.W:g} s, "
                  f"coef {np.round(job.coef, 3)}, rms {job.fit_rms:.1f} W; idle "
                  + ", ".join(f"{k} {v:.0f} W" for k, v in job.idle.items()))
        rows += analyse(job)
    for m in meter_rows(jobs):
        METERS[m["node"]] = m
    summary = summarise(rows)
    warmup = read_warmup(folder)
    if warmup:
        with open(folder / "warmup.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(warmup[0]))
            w.writeheader()
            w.writerows(warmup)
    write_tables(folder, jobs, rows, summary, warmup)
    chart_power(folder, summary)
    chart_energy(folder, summary)
    chart_scatter(folder, summary)
    chart_traces(folder, jobs, rows)
    chart_cpu_gpu(folder, summary)
    chart_meter(folder, jobs)
    chart_methods(folder, summary)
    chart_warmup(folder, warmup)
    print((folder / "summary.md").read_text())


if __name__ == "__main__":
    main()
