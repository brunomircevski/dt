#!/usr/bin/env python3
"""Energy of ./tree's backends from bench/power/power_bench.sh jobs: tables and charts.

  bench/.venv/bin/python bench/power/analyze.py bench/results/power  # every job folder in it

Every job folder (ampere/, dell/, epyc/) holds markers.txt, samples.csv
(pwrlog.py), perf_rapl.csv (ampere: RAPL per socket through perf) and out/<run>.txt
(./tree's output with --repeat K training windows).

Per run (one process), over the span from the first tree's start to the last
tree's end, divided by the number of trees (energy and time per tree); with
K >= 3 trees the first one is left out (it is slower: CUDA context, first
allocations):
  - PSU (whole server, Dell nodes): the ACPI meter's readings held until the
    next one, shifted back by the meter's delay. The delay is measured per job:
    the median time from a process's start to the reading half-way up its
    first step (the CSV load lifts power from idle within a fraction of a second).
  - GPU i: the NVML energy counter, interpolated at the span's ends.
  - CPU socket s: RAPL package energy from perf's 100 ms intervals, the part of
    each interval inside the span.
Idle power of each source: the job's two idle windows (before the first run and
after the last, skipping each window's first 5 s). "Above idle" = energy - idle
power x time.

Writes into the given folder: runs.csv (every run), summary.csv (mean and
standard deviation per case), summary.md (tables) and charts chart-*.png.
"""
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
NODE_ORDER = ["epyc", "dell", "ampere"]
NODE_HW = {"epyc": "epyc1: 2× EPYC 9255", "dell": "dell4: 2× Xeon Gold 6542Y",
           "ampere": "ampere: A100 80 GB + 2× EPYC 7402 (GPUs + CPU packages, no PSU)"}
IMPL_COLORS = {"simple": "#d14f8e", "serial": "#2a78d6", "parallel": "#eb6834",
               "cuda": "#1baf7a", "gpu_only": "#eda100", "gpuonly": "#eda100"}
NODE_COLORS = {"epyc": "#8e5bd0", "dell": "#2a78d6", "ampere": "#1baf7a"}
MARKERS = {"simple": "s", "serial": "D", "parallel": "o", "gpuonly": "^", "cuda": "v"}
# Power components, bottom to top of a stacked bar.
COMPONENTS = {"psu": ("Whole server (PSU)", "#5b6770"),
              "cpu0": ("CPU socket 0 (RAPL)", "#2a78d6"),
              "cpu1": ("CPU socket 1 (RAPL)", "#8e5bd0"),
              "gpu_used": ("GPU 0, in use (NVML)", "#1baf7a"),
              "gpu_idle": ("GPUs 1–3, unused (NVML)", "#a08a2c")}
IDLE_SKIP = 5.0


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
        self.events = read_markers(folder / "markers.txt")
        self.machine = (folder / "machine.txt").read_text()
        self.has_psu = "sys_W" in self.samples
        self.gpus = sorted(int(c[3:-3]) for c in self.samples if re.fullmatch(r"gpu\d+_mJ", c))
        self.rapl = self.read_rapl()
        self.idle_windows = self.find_idle()
        self.lag = self.measure_lag() if self.has_psu else 0.0
        self.idle = {k: self.mean_power(k, self.idle_windows) for k in self.sources()}

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
        for a, b in (("idle_start", "idle_end"), ("idle2_start", "idle2_end")):
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

    def measure_lag(self):
        """Median delay from a process's start to the PSU reading half-way up its step."""
        idle = statistics.mean(self.values_between("sys_W", *self.idle_windows[0])) \
            if self.idle_windows else min(self.samples["sys_W"])
        lags = []
        for _, _, start, end, _ in self.processes():
            after = [(t, v) for t, v in zip(self.t, self.samples["sys_W"]) if start <= t <= start + 20]
            if not after or end - start < 15:
                continue
            top = max(v for _, v in after)
            if top - idle < 30:
                continue
            first = next(t for t, v in after if v >= idle + 0.5 * (top - idle))
            lags.append(first - start)
        return statistics.median(lags) if lags else 2.0

    def values_between(self, col, a, b):
        return [v for t, v in zip(self.t, self.samples[col]) if a <= t <= b]

    def energy(self, source, a, b):
        """Joules of `source` from a to b (Unix times)."""
        if source == "psu":
            return self.held_integral(self.samples["sys_W"], a + self.lag, b + self.lag)
        if source.startswith("gpu"):
            col = self.samples["gpu%s_mJ" % source[3:]]
            return (self.interpolate(col, b) - self.interpolate(col, a)) / 1000
        total = 0.0
        for t, joules, dur in self.rapl[int(source[3:])]:
            overlap = min(b, t) - max(a, t - dur)
            if overlap > 0 and dur > 0:
                total += joules * overlap / dur
        return total

    def held_integral(self, values, a, b):
        """Integral of the readings, each held until the next sample, from a to b."""
        total = 0.0
        for i in range(len(self.t) - 1):
            lo, hi = max(a, self.t[i]), min(b, self.t[i + 1])
            if hi > lo:
                total += values[i] * (hi - lo)
        return total

    def interpolate(self, col, x):
        t = self.t
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

    def peak(self, a, b):
        """Highest sampled total power in [a, b]: PSU, or GPUs + CPU sockets."""
        if self.has_psu:
            # Skip the first 10 s of a long window: the meter's first reading
            # there still averages over the end of the (all-core) CSV load.
            skip = 10.0 if b - a > 60 else 0.0
            vals = self.values_between("sys_W", a + self.lag + skip, b + self.lag)
            return max(vals) if vals else float("nan")
        best = 0.0
        rapl = {k: dict((round(t, 1), j / d) for t, j, d in v if d > 0) for k, v in self.rapl.items()}
        for i, t in enumerate(self.t):
            if a <= t <= b:
                gpu = sum(self.samples["gpu%d_W" % g][i] for g in self.gpus)
                cpu = sum(min(v.items(), key=lambda kv: abs(kv[0] - t))[1] for v in rapl.values()) \
                    if rapl else 0
                best = max(best, gpu + cpu)
        return best


def components(job, per_source):
    """Map source values to chart components."""
    out = {}
    if job.has_psu:
        out["psu"] = per_source["psu"]
    for k in sorted(job.rapl):
        out["cpu%d" % k] = per_source["cpu%d" % k]
    if job.gpus:
        out["gpu_used"] = per_source["gpu0"]
        out["gpu_idle"] = sum(per_source["gpu%d" % g] for g in job.gpus if g != 0)
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
        comp_e = components(job, energy)
        comp_idle = components(job, idle)
        total = sum(comp_e.values())
        sa, sb = steady[0][0], steady[-1][1]
        steady_e = sum(job.energy(s, sa, sb) for s in job.sources()) / len(steady)
        rows.append({
            "node": job.node, "impl": attrs["impl"], "algo": attrs["algo"],
            "rep": int(run.split("-")[0][3:]), "repeat": k, "trees_in_process": len(tree["windows"]),
            "train_s": statistics.mean(w[2] for w in windows), "first_tree_s": tree["windows"][0][2],
            "steady_train_s": statistics.mean(w[2] for w in steady), "steady_energy_J": steady_e,
            "span_s": span / k, "load_s": tree["load_s"], "nodes": tree["nodes"],
            "depth": tree["depth"], "energy_J": total,
            "above_idle_J": total - sum(comp_idle.values()),
            "power_W": total / (span / k), "idle_W": sum(comp_idle.values()) / (span / k),
            "peak_W": job.peak(a, b), "lag_s": job.lag,
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
        for key in ("train_s", "load_s", "energy_J", "above_idle_J", "power_W", "idle_W", "peak_W",
                    "steady_train_s", "steady_energy_J", "span_s"):
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
            ax.text(i, max(bottom, s["peak_W"]) + 6, f"{s['power_W']:.0f} W\n(+{s['power_W'] - s['idle_W']:.0f})",
                    ha="center", va="bottom", fontsize=8, color=INK)
            ax.plot([i - 0.38, i + 0.38], [s["peak_W"]] * 2, color=INK, lw=1, ls=":")
        ax.set_xticks(range(len(cases)), [label(s).replace(" · ", "\n") for s in cases], fontsize=8)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID)
        style(ax)
    axes[0].set_ylabel("mean power while training (W)", color=INK)
    handles = [Patch(color=COMPONENTS[c][1], label=COMPONENTS[c][0]) for c in COMPONENTS if c in used]
    handles += [Patch(facecolor="#999", alpha=0.38, label="lighter part: idle power of that component"),
                plt.Line2D([], [], color=INK, ls=":", label="highest sample (peak)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("ECBDL'14 15M: mean power while training one tree (Dell nodes: whole server at the PSU; "
                 "ampere: GPUs + CPU packages, no PSU reading)", fontsize=11, color=INK, x=0.01, ha="left")
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
            ax.text(i, s["energy_J"] / 1e3 * 1.15, f"{fmt_j(s['energy_J'])}\n({fmt_j(s['above_idle_J'])})",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(range(len(cases)), [label(s).replace(" · ", "\n") for s in cases], fontsize=8)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, which="both")
        style(ax)
    axes[0].set_ylabel("energy per tree (kJ, log scale)", color=INK)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 4)
    fig.legend(handles=[Patch(facecolor="#888", alpha=0.45, label="total energy while training (label: total)"),
                        Patch(facecolor="#888", label="above idle power (label in brackets)")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("ECBDL'14 15M: energy to train one tree (mean of the runs; whisker ± 1 sd; "
                 "ampere = GPUs + CPU packages)", fontsize=11, color=INK, x=0.01, ha="left")
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
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, which="both")
        style(ax)
    fig.suptitle("ECBDL'14 15M: training time per tree during the power runs", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(folder / "chart-time.png", dpi=150)
    plt.close(fig)


def chart_scatter(folder, summary):
    """Time against energy, log-log, with lines of constant power."""
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ts = [s["train_s"] for s in summary]
    lo, hi = min(ts) / 2, max(ts) * 2
    for watts in (100, 200, 500, 1000):
        ax.plot([lo, hi], [lo * watts / 1e3, hi * watts / 1e3], color=GRID, lw=1, zorder=0)
        ax.text(hi, hi * watts / 1e3, f" {watts} W", fontsize=7.5, color=MUTED, va="center")
    for s in summary:
        filled = s["algo"] == "cart"
        color = NODE_COLORS.get(s["node"], INK)
        ax.scatter(s["train_s"], s["energy_J"] / 1e3, marker=MARKERS[s["impl"]], s=60,
                   facecolors=color if filled else "white", edgecolors=color, linewidths=1.5, zorder=3)
        ax.scatter(s["train_s"], s["above_idle_J"] / 1e3, marker=MARKERS[s["impl"]], s=18,
                   color=color, alpha=0.45, zorder=2)
        ax.plot([s["train_s"]] * 2, [s["above_idle_J"] / 1e3, s["energy_J"] / 1e3], color=color,
                alpha=0.35, lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi * 3)
    ax.set_xlabel("training time per tree (s)", color=INK)
    ax.set_ylabel("energy per tree (kJ)", color=INK)
    ax.grid(color=GRID, which="major")
    style(ax)
    handles = [plt.Line2D([], [], marker=MARKERS[i], ls="", color=INK, label=IMPLS[i])
               for i in IMPL_ORDER if any(s["impl"] == i for s in summary)]
    handles += [Patch(color=NODE_COLORS[n], label=NODE_HW[n]) for n in NODE_ORDER
                if any(s["node"] == n for s in summary)]
    handles += [plt.Line2D([], [], marker="o", ls="", markerfacecolor=INK, color=INK, label="CART"),
                plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", color=INK, label="C4.5"),
                plt.Line2D([], [], marker="o", ls="", color=INK, alpha=0.45, markersize=4,
                           label="small dot: energy above idle")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    ax.set_title("ECBDL'14 15M: time and energy per tree (grey lines: constant power)",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(folder / "chart-time-energy.png", dpi=150)
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
            ax.plot([job.t[i] - start - job.lag for i in idx], [job.samples["sys_W"][i] for i in idx],
                    color=COMPONENTS["psu"][1], lw=1.2, drawstyle="steps-post", label="PSU (shifted by delay)")
        for g in job.gpus:
            ax.plot(x, [job.samples["gpu%d_W" % g][i] for i in idx], lw=1 if g == 0 else 0.6,
                    color=COMPONENTS["gpu_used" if g == 0 else "gpu_idle"][1],
                    label="GPU 0" if g == 0 else ("GPUs 1–3" if g == 1 else None))
        for k, series in sorted(job.rapl.items()):
            pts = [(t - start, j / d) for t, j, d in series if lo <= t <= hi and d > 0]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=0.7, color=COMPONENTS["cpu%d" % k][1],
                    label=f"CPU socket {k}")
        ax.axvspan(a - start, b - start, color=IMPL_COLORS[key[1]], alpha=0.12, lw=0)
        busy = ax.twinx()
        busy.plot(x, [job.samples["cpu_busy"][i] * 100 for i in idx], color=MUTED, lw=0.5, alpha=0.6)
        busy.set_ylim(0, 105)
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
               plt.Line2D([], [], color=COMPONENTS["gpu_used"][1], label="GPU 0 (in use)"),
               plt.Line2D([], [], color=COMPONENTS["gpu_idle"][1], label="GPUs 1–3 (each)"),
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
    print((folder / "summary.md").read_text())


if __name__ == "__main__":
    main()
