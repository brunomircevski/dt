#!/usr/bin/env python3
"""Charts of where ./tree's training time goes, from runs made with
`run.py --profile` (bench/cluster/profile.sh): every run directory holds
results.jsonl and profiles/*.json (./tree --profile, src/core/trace.h).

  d=bench/results/cluster-ecbdl-profile/gpu-plus-cpu
  bench/.venv/bin/python bench/profile_chart.py $d/cuda-hopper-ecbdl_15m $d/cuda-ampere-ecbdl_15m \\
      $d/cpu-epyc-ecbdl_15m --out-dir $d

writes into --out-dir:
  chart-phases.png     the training thread's phases, in seconds (the critical path)
  chart-threads.png    what the CPU threads did: thread-seconds by kind of work, idle included
  chart-gpu.png        CUDA runs: what the GPU ran (kernels by step, copies by kind), idle included
  timeline-<protocol>.png  per run, over the training time: the phases, the CPU threads
                       at work by kind (every 5 ms) and how busy the GPU's kernels and copies were
  profile.md           the numbers behind the charts, and what profiling cost (--plain, or each run's
                       sibling plain/: the same cases run without --profile)

One run per case is drawn (the first profiled run of each case).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES  # noqa: E402
from machines_chart import TITLES, short_cpu, short_gpu  # noqa: E402

# The reference palette's categorical slots, always in this order (validated on
# adjacent pairs for stacked bars); slots 3-5 are below 3:1 on white, so the
# segments carry labels and profile.md has the numbers. Idle is a neutral gray.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
IDLE = "#dcdbd6"
BACKENDS = {"tree_cuda": "CUDA", "tree_gpu_only": "GPU only", "tree": "Parallel"}
GPU = {"tree_cuda", "tree_gpu_only"}  # the runs that use the GPU

# The training thread's phases (trace phase names -> group), in stacking order.
PHASES = [
    ("setup", "GPU setup: allocation, waiting for page-locked host memory"),
    ("upload", "upload: data to the GPU"),
    ("presort", "presort"),
    ("gpu", "grow: GPU levels (CPU grows small subtrees meanwhile)"),
    ("cpu", "grow on the CPU (CUDA: the subtrees left after the GPU)"),
    ("arrange", "prune: rows ordered by leaf"),
    ("prune", "prune: the rest"),
    ("other", "other: tree assembly, bookkeeping"),
]
PHASE_OF = {"gpu setup": "setup", "grow setup": "setup", "upload": "upload", "presort": "presort", "gpu levels": "gpu",
            "build": "cpu", "wait cpu subtrees": "cpu", "assemble tree": "other",
            "c45 collapse, thresholds": "prune", "prune setup": "prune",
            "prune arrange rows": "arrange", "prune subtrees": "prune",
            "prune compact": "prune", "cart prune": "prune"}

# What the CPU threads did (trace Work names -> group), in stacking order; idle last.
WORKS = [
    ("sweep", "sweep: best cut of each column"),
    ("partition", "partition the columns"),
    ("fused", "partition + sweeps of the children"),
    ("presort", "presort"),
    ("grow", "other growing: marking rows, leaves, nodes, assembly"),
    ("feed", "driving the GPU: launches, waits, staging, page-locking"),
    ("prune", "pruning"),
    ("other", "other"),
]
WORK_OF = {"sweep": "sweep", "partition": "partition", "partition_sweep": "fused",
           "presort": "presort", "build": "grow", "mark": "grow", "leaves": "grow",
           "assemble": "grow", "gpu_host": "feed", "gpu_wait": "feed", "staging": "feed",
           "page_lock": "feed", "prune_arrange": "prune", "prune": "prune",
           "prune_other": "prune", "other": "other"}

# GPU kernels (KernelProfile names -> step) and copies (-> kind): slots 1-5
# and 6-8, so that one legend holds both.
KERNELS = [
    ("presort", "presort: gather, sort, distinct values"),
    ("search", "split search: histograms, prefix sums, cut scans"),
    ("partition", "partition: mark rows, move the columns"),
    ("handoff", "hand-off: gather small nodes' columns"),
    ("leaves", "C4.5: leaf of every row, rows ordered by leaf"),
]
KERNEL_OF = {"presortGather": "presort", "presortSort": "presort", "presortDistinct": "presort",
             "tileBound": "search", "segmentLower": "search", "tileExact": "search",
             "tileEstimate": "search", "segmentMax": "search", "tileCandidate": "search",
             "segmentBest": "search", "tileHistogram": "search", "segmentConstant": "search",
             "segmentPrefix": "search", "markGoesLeft": "partition", "partition": "partition",
             "handOffGather": "handoff", "leafRows": "leaves", "arrangeSort": "leaves",
             "arrangeRows": "leaves", "copyValues": "leaves"}
COPIES = [
    ("upload", "upload: data to the GPU"),
    ("handoff", "hand-off: small nodes' columns to the host"),
    ("other", "other: per-level lists and cuts, leaves"),
]
COPY_OF = {"upload": "upload", "handOffCopy": "handoff"}  # everything else: other
# Short names of the phases for the timeline's band.
SHORT = {"gpu setup": "setup", "grow setup": "wait", "upload": "upload", "presort": "presort", "gpu levels": "GPU levels",
         "build": "grow", "wait cpu subtrees": "CPU subtrees", "assemble tree": "",
         "c45 collapse, thresholds": "collapse", "prune setup": "", "prune arrange rows": "arrange",
         "prune subtrees": "prune", "prune compact": "", "cart prune": "prune"}
# Protocol order of the panels (as in cluster-ecbdl's charts).
ORDER = ["cart_alpha_nodepth", "cart_full_nodepth", "c45", "cart_alpha", "cart_full",
         "c45_unpruned"]


def colors(groups, first=0):
    return {key: SLOTS[first + i] for i, (key, _) in enumerate(groups)}


def seconds_text(value):
    return f"{value:.1f} s" if value >= 10 else f"{value:.2f} s"


def load_run(run):
    """-> {protocol: case dict} of the run's first profiled ./tree run per protocol."""
    machine = json.load(open(run / "machine.json"))
    info = B.hardware(machine)
    cases = {}
    for line in open(run / "results.jsonl"):
        row = json.loads(line)
        if row["kind"] != "run" or row["status"] != "ok" or "profile" not in row:
            continue
        if row["protocol"] in cases:
            continue
        trace = json.load(open(run / row["profile"]))
        node = run.name.split("-")[1]
        plural = "s" if row["threads"] > 1 else ""
        hardware = (f"{short_gpu(info)} + {row['threads']} CPU thread{plural}" if row["impl"] in GPU
                    else f"{short_cpu(info)}, {row['threads']} thread{plural}")
        cases[row["protocol"]] = {"run": run, "row": row, "trace": trace, "node": node,
                                  "impl": row["impl"], "hardware": hardware,
                                  "machine": machine, "threads": row["threads"]}
    return cases


def phase_seconds(case):
    """-> {phase group: seconds}: the depth-0 phases (gpu setup without its
    upload), the rest of the training time as `other`."""
    out = defaultdict(float)
    for phase in case["trace"]["phases"]:
        group = PHASE_OF.get(phase["name"])
        span = phase["end"] - phase["start"]
        if group is None:
            continue
        if phase["name"] == "upload":
            out["setup"] -= span
        if phase["depth"] == 0 or phase["name"] == "upload":
            out[group] += span
    total = case["row"]["train_seconds"]
    out["other"] += max(0.0, total - sum(out.values()))
    return dict(out)


def work_seconds(case):
    """-> {work group: thread-seconds}, idle = threads x training time - the rest."""
    out = defaultdict(float)
    for name, value in case["trace"]["work_seconds"].items():
        if name in WORK_OF:
            out[WORK_OF[name]] += value
    capacity = case["threads"] * case["trace"]["seconds"]
    out["idle"] = max(0.0, capacity - sum(out.values()))
    return dict(out), capacity


def busy_by_kind(spans, kinds):
    """Spans [(kind, start, end)] -> {kind: seconds} with overlaps counted once,
    for the kind earliest in `kinds`, and the union's length."""
    events = []
    for kind, start, end in spans:
        if end > start:
            events += [(start, 1, kind), (end, -1, kind)]
    events.sort(key=lambda e: (e[0], e[1]))
    active = defaultdict(int)
    out = defaultdict(float)
    union = 0.0
    last = None
    for time, step, kind in events:
        if last is not None and time > last:
            live = [k for k in kinds if active[k] > 0]
            if live:
                out[live[0]] += time - last
                union += time - last
        active[kind] += step
        last = time
    return dict(out), union


def gpu_lanes(case):
    """-> (kernel seconds by step, copy seconds by kind, wall)."""
    kernels, copies = [], []
    for name, lane, start, end in case["trace"]["gpu"]:
        if lane == "compute":
            kernels.append((KERNEL_OF.get(name, "search"), start, end))
        else:
            copies.append((COPY_OF.get(name, "other"), start, end))
    k, _ = busy_by_kind(kernels, [key for key, _ in KERNELS])
    c, _ = busy_by_kind(copies, [key for key, _ in COPIES])
    return k, c, case["trace"]["seconds"]


def style_axis(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks([])


def stacked_bar(ax, y, parts, order, palette, height, scale, label_min, fmt, idle=None):
    """Draw one stacked bar; label segments wider than label_min (axis units)."""
    left = 0.0
    keys = [key for key, _ in order] + (["idle"] if idle else [])
    for key in keys:
        value = parts.get(key, 0.0)
        if value <= 0:
            continue
        color = IDLE if key == "idle" else palette[key]
        ax.barh(y, value, left=left, height=height, color=color, edgecolor="white",
                linewidth=1.0, zorder=2)
        if value >= label_min:
            ax.text(left + value / 2, y, fmt(value), ha="center", va="center", fontsize=7.6,
                    color=INK if key == "idle" else "white", fontweight="bold", zorder=3)
        left += value
    return left


def row_labels(ax, y, case):
    name = f"{BACKENDS[case['impl']]} · {case['node']}"
    ax.text(-0.015, y - 0.14, name, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(-0.015, y + 0.24, case["hardware"], transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7.8, color=MUTED)


def legend(fig, groups, palette, y, idle=False, ncol=None):
    handles = [Patch(color=palette[key], label=label) for key, label in groups]
    if idle:
        handles.append(Patch(color=IDLE, label="idle"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.215, y),
               ncol=ncol or 2, frameon=False, fontsize=8.4, handletextpad=0.4,
               columnspacing=1.3)


def figure(rows_per_panel, panels, legend_rows, footer_lines, width=13):
    header = 0.62 + 0.21 * legend_rows
    inches = header + 0.55 + footer_lines * 0.19 + 0.35 + panels * (0.85 + 0.62 * max(rows_per_panel, 2))
    fig, axes = plt.subplots(panels, 1, figsize=(width, inches), squeeze=False)
    gap = 0.85
    space = inches - header - 0.55 - footer_lines * 0.19 - 0.35
    fig.subplots_adjust(left=0.215, right=0.97, top=1 - (header + 0.45) / inches,
                        bottom=(footer_lines * 0.19 + 0.35) / inches,
                        hspace=gap / ((space - (panels - 1) * gap) / panels))
    return fig, [a[0] for a in axes], inches, header


def title(fig, inches, text):
    fig.suptitle(text, x=0.012, y=1 - 0.09 / inches, ha="left", fontsize=12, color=INK,
                 fontweight="bold")


def footer(fig, inches, lines):
    fig.text(0.215, 0.12 / inches, "\n".join(lines), fontsize=8, color=MUTED, linespacing=1.5)


def chart_phases(cases, protocols, meta, out, overhead):
    rows = max(len(cases[p]) for p in protocols)
    lines = ["Bars: ./tree's 'train total' of one profiled run (presort, build, prune; CUDA also "
             "setup and upload), not reading the CSV;",
             "split by the training thread's phases (--profile). Labels: seconds, shown on "
             "segments of 4% of the axis or more.",
             f"Profiling cost: {overhead}."]
    fig, axes, inches, header = figure(rows, len(protocols), 3, len(lines))
    palette = colors(PHASES)
    top = max(c["row"]["train_seconds"] for p in protocols for c in cases[p])
    for ax, protocol in zip(axes, protocols):
        style_axis(ax)
        mine = cases[protocol]
        tree = mine[0]["row"]
        ax.set_title(f"{TITLES[protocol]}: where the training time goes", loc="left",
                     fontsize=10.5, color=INK, fontweight="bold", pad=20)
        shape = ("a single leaf" if tree["nodes"] == 1
                 else f"{tree['nodes']:,} nodes, depth {tree['depth']}")
        same = len({(c["row"]["nodes"], c["row"]["depth"]) for c in mine}) == 1
        ax.text(0, 1.02, shape + (" (the same tree on every machine and backend)" if same
                                  else " (on the first machine; the trees differ)"),
                transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
        for y, case in enumerate(mine):
            parts = phase_seconds(case)
            total = stacked_bar(ax, y, parts, PHASES, palette, 0.62, top, top * 0.04, seconds_text)
            ax.text(total + top * 0.012, y, seconds_text(case["row"]["train_seconds"]),
                    va="center", fontsize=9, color=INK, fontweight="bold")
            row_labels(ax, y, case)
        ax.set_ylim(len(mine) - 0.35, -0.65)
        ax.set_xlim(0, top * 1.12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    title(fig, inches, f"./tree on {NAMES.get(meta[0], meta[0])}, {meta[1]:,} training rows, "
          f"{meta[2]} features: the training thread's phases")
    legend(fig, PHASES, palette, 1 - 0.5 / inches, ncol=3)
    footer(fig, inches, lines)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


def chart_threads(cases, protocols, meta, out):
    rows = max(len(cases[p]) for p in protocols)
    lines = ["Bars: all CPU threads together over the training time (threads × 'train total' "
             "= 100%), by what each thread was doing (--profile);",
             "idle: waiting for work or for other threads. Pool tasks and parallel loops count "
             "as the work of the thread that started them.",
             "CUDA: the pool grows the small subtrees the GPU hands over; the training thread "
             "drives the GPU. Labels: share of the thread time."]
    fig, axes, inches, header = figure(rows, len(protocols), 3, len(lines))
    palette = colors(WORKS)
    for ax, protocol in zip(axes, protocols):
        style_axis(ax)
        mine = cases[protocol]
        ax.set_title(f"{TITLES[protocol]}: what the CPU threads did", loc="left", fontsize=10.5,
                     color=INK, fontweight="bold", pad=20)
        ax.text(0, 1.02, "share of threads × training time", transform=ax.transAxes,
                fontsize=8, color=MUTED, va="bottom")
        for y, case in enumerate(mine):
            parts, capacity = work_seconds(case)
            shares = {k: v / capacity for k, v in parts.items()}
            stacked_bar(ax, y, shares, WORKS, palette, 0.62, 1, 0.045,
                        lambda v: f"{100 * v:.0f}%", idle=True)
            busy = 1 - shares.get("idle", 0)
            ax.text(1.012, y, f"{100 * busy:.0f}% busy", va="center", fontsize=9, color=INK,
                    fontweight="bold")
            ax.text(1.012, y + 0.26, f"{capacity:,.0f} thread-s", va="center", fontsize=7.6,
                    color=MUTED)
            row_labels(ax, y, case)
        ax.set_ylim(len(mine) - 0.35, -0.65)
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    title(fig, inches, f"./tree on {NAMES.get(meta[0], meta[0])}: what the CPU threads did "
          "during training")
    legend(fig, WORKS, palette, 1 - 0.5 / inches, idle=True, ncol=3)
    footer(fig, inches, lines)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


def chart_gpu(cases, protocols, meta, out):
    gpu = {p: [c for c in cases[p] if c["impl"] in GPU] for p in protocols}
    protocols = [p for p in protocols if gpu[p]]
    if not protocols:
        return
    rows = max(2 * len(v) for v in gpu.values())
    lines = ["Each run has two bars over its training time ('train total', from the start of "
             "GPU setup): the kernels (one CUDA stream)",
             "and the copies between host and GPU (both directions, both streams; overlapping "
             "copies counted once). CUDA events around every launch and copy.",
             "Idle: no kernel (or no copy) running, e.g. while the host decides the splits, "
             "prunes on the CPU or waits for CPU subtrees. Labels: seconds."]
    fig, axes, inches, header = figure(rows / 1.6, len(protocols), 4, len(lines))
    kernel_palette, copy_palette = colors(KERNELS), colors(COPIES, len(KERNELS))
    top = max(c["row"]["train_seconds"] for p in protocols for c in gpu[p])
    for ax, protocol in zip(axes, protocols):
        style_axis(ax)
        mine = gpu[protocol]
        ax.set_title(f"{TITLES[protocol]}: what the GPU ran", loc="left", fontsize=10.5,
                     color=INK, fontweight="bold", pad=20)
        y = 0.0
        for case in mine:
            kernels, copies, wall = gpu_lanes(case)
            wall = case["row"]["train_seconds"]
            for lane, parts, groups, palette in (("kernels", kernels, KERNELS, kernel_palette),
                                                 ("copies", copies, COPIES, copy_palette)):
                parts = dict(parts)
                parts["idle"] = max(0.0, wall - sum(parts.values()))
                stacked_bar(ax, y, parts, groups, palette, 0.36, top, top * 0.045,
                            seconds_text, idle=True)
                busy = wall - parts["idle"]
                ax.text(wall + top * 0.012, y, f"{lane}: {seconds_text(busy)} busy "
                        f"({100 * busy / wall:.0f}%)", va="center", fontsize=8.2, color=INK)
                y += 0.42
            name = f"CUDA · {case['node']}"
            mid = y - 0.63
            ax.text(-0.015, mid - 0.14, name, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=9.5, color=INK, fontweight="bold")
            ax.text(-0.015, mid + 0.2, case["hardware"], transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7.8, color=MUTED)
            y += 0.45
        ax.set_ylim(y - 0.6, -0.4)
        ax.set_xlim(0, top * 1.3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    title(fig, inches, f"./tree --cuda on {NAMES.get(meta[0], meta[0])}: GPU kernels and copies "
          "during training")
    handles = ([Patch(color=kernel_palette[k], label=f"kernel · {label}") for k, label in KERNELS]
               + [Patch(color=copy_palette[k], label=f"copy · {label}") for k, label in COPIES]
               + [Patch(color=IDLE, label="idle")])
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.215, 1 - 0.5 / inches),
               ncol=3, frameon=False, fontsize=8.2, handletextpad=0.4, columnspacing=1.3)
    footer(fig, inches, lines)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


def busy_fraction(spans, edges):
    """Share of each bin [edges[i], edges[i+1]) covered by the union of spans."""
    import numpy as np
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out = np.zeros(len(edges) - 1)
    width = edges[1] - edges[0]
    for start, end in merged:
        first = max(0, int(start // width))
        last = min(len(out) - 1, int(end // width))
        for b in range(first, last + 1):
            lo, hi = edges[b], edges[b + 1]
            out[b] += max(0.0, min(hi, end) - max(lo, start))
    return out / width


def chart_timeline(cases, protocol, meta, out):
    import numpy as np
    mine = cases[protocol]
    gpu_rows = sum(c["impl"] in GPU for c in mine)
    heights = []
    for case in mine:
        heights += [0.32, 1.9] + ([0.28, 0.28] if case["impl"] in GPU else []) + [0.5]
    lines = ["Phases: the training thread's (top band; see chart-phases.png). Threads at work: "
             "how many CPU threads did each kind of work, sampled every 5 ms",
             "(colours as in chart-threads.png; the dashed line is the thread count). GPU: share "
             "of each 1% of the time with a kernel or a copy running."]
    inches = 1.25 + sum(heights) + len(lines) * 0.19 + 0.45
    fig = plt.figure(figsize=(13, inches))
    grid = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.12,
                            left=0.12, right=0.97, top=1 - 1.25 / inches,
                            bottom=(len(lines) * 0.19 + 0.45) / inches)
    span = max(c["trace"]["seconds"] for c in mine)
    work_palette = colors(WORKS)
    row = 0
    axes_x = []
    lowest = []  # the last axis of every run: the only one with time labels
    for case in mine:
        if axes_x:
            lowest.append(axes_x[-1])
        trace = case["trace"]
        # Phases band.
        ax = fig.add_subplot(grid[row])
        row += 1
        axes_x.append(ax)
        shown = [p for p in trace["phases"] if p["name"] in PHASE_OF
                 and (p["depth"] == 0 or p["name"] == "upload")]
        shown.sort(key=lambda p: (p["start"], -p["depth"]))
        for index, phase in enumerate(shown):
            width = phase["end"] - phase["start"]
            ax.barh(0, width, left=phase["start"], height=1,
                    color="#e9e8e4" if index % 2 else "#d3d2cc", edgecolor="white",
                    linewidth=0.8, zorder=2 + phase["depth"])
            label = SHORT.get(phase["name"], "")
            # Label the longest part that no nested phase covers (gpu setup: around the upload).
            free, left = [], phase["start"]
            for inner in sorted((q for q in shown if q["depth"] > phase["depth"]
                                 and phase["start"] <= q["start"] < phase["end"]),
                                key=lambda q: q["start"]):
                free.append((left, inner["start"]))
                left = max(left, inner["end"])
            free.append((left, phase["end"]))
            begin, end = max(free, key=lambda part: part[1] - part[0])
            if label and end - begin >= span * 0.012 * (len(label) * 0.6 + 1):
                ax.text((begin + end) / 2, 0, label, ha="center", va="center",
                        fontsize=7.4, color=INK, zorder=5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.text(-0.01, 0, "phases", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=7.8, color=MUTED)
        ax.set_title(f"{BACKENDS[case['impl']]} · {case['node']}  ({case['hardware']}): "
                     f"{seconds_text(case['row']['train_seconds'])}", loc="left", fontsize=10,
                     color=INK, fontweight="bold", pad=5)
        # Threads at work.
        ax = fig.add_subplot(grid[row])
        row += 1
        axes_x.append(ax)
        samples = trace["samples"]
        t = np.array(samples["time"])
        stacks = []
        for key, _ in WORKS:
            series = np.zeros(len(t))
            for name, group in WORK_OF.items():
                if group == key and name in samples["threads"]:
                    series += np.array(samples["threads"][name])
            stacks.append(series)
        ax.stackplot(t, stacks, colors=[work_palette[k] for k, _ in WORKS], linewidth=0)
        ax.axhline(case["threads"], color=MUTED, linestyle="--", linewidth=0.8)
        ax.set_ylim(0, case["threads"] * 1.12)
        ax.set_ylabel("threads\nat work", fontsize=7.8, color=MUTED)
        ax.tick_params(axis="y", colors=MUTED, labelsize=7.5, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        # GPU.
        if case["impl"] in GPU:
            bins = 100
            edges = np.linspace(0, trace["seconds"], bins + 1)
            for lane, label in (("compute", "GPU\nkernels"), ("copy", "GPU\ncopies")):
                ax = fig.add_subplot(grid[row])
                row += 1
                axes_x.append(ax)
                spans = [(s, e) for _, l, s, e in trace["gpu"] if (l == "compute") == (lane == "compute")]
                share = busy_fraction(spans, edges)
                ax.bar(edges[:-1], share, width=edges[1] - edges[0], align="edge",
                       color=INK if lane == "compute" else MUTED, linewidth=0)
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.text(-0.01, 0.5, label, transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=7.6, color=MUTED, linespacing=1.0)
        ax_gap = fig.add_subplot(grid[row])
        row += 1
        ax_gap.axis("off")
    lowest.append(axes_x[-1])
    for ax in axes_x:
        ax.tick_params(axis="x", labelbottom=ax in lowest)
        ax.set_xlim(0, span * 1.01)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(axis="x", colors=MUTED, labelsize=7.5, length=0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    fig.suptitle(f"./tree on {NAMES.get(meta[0], meta[0])}, {TITLES[protocol]}: timeline of "
                 "the training", x=0.012, y=1 - 0.09 / inches, ha="left", fontsize=12,
                 color=INK, fontweight="bold")
    handles = [Patch(color=work_palette[k], label=label) for k, label in WORKS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.12, 1 - 0.42 / inches),
               ncol=4, frameon=False, fontsize=7.8, handletextpad=0.4, columnspacing=1.1)
    fig.text(0.12, 0.12 / inches, "\n".join(lines), fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


def plain_times(plain_dir, run):
    """-> {protocol: train seconds} of the same case without --profile."""
    path = plain_dir / run.name / "results.jsonl" if plain_dir else None
    if not path or not path.exists():
        return {}
    out = {}
    for line in open(path):
        row = json.loads(line)
        if row["kind"] == "run" and row["status"] == "ok":
            out.setdefault(row["protocol"], (row["train_seconds"], row["nodes"]))
    return out


def report(cases, protocols, plain, out):
    lines = ["# Profile numbers", "",
             "Seconds of one profiled run per case (`train total`), from `profiles/*.json`.", ""]
    lines += ["## Phases of the training thread (s)", "",
              "| Case | Run | " + " | ".join(key for key, _ in PHASES)
              + " | train total | without --profile |",
              "|---|---|" + "--:|" * (len(PHASES) + 2)]
    for protocol in protocols:
        for case in cases[protocol]:
            parts = phase_seconds(case)
            reference = plain.get(case["run"].name, {}).get(protocol)
            lines.append(f"| {protocol} | {case['run'].name} | "
                         + " | ".join(f"{parts.get(k, 0):.2f}" for k, _ in PHASES)
                         + f" | {case['row']['train_seconds']:.2f} | "
                         + (f"{reference[0]:.2f}" if reference else "–") + " |")
    lines += ["", "## CPU threads (thread-seconds)", "",
              "| Case | Run | threads | " + " | ".join(k for k, _ in WORKS) + " | idle |",
              "|---|---|--:|" + "--:|" * (len(WORKS) + 1)]
    for protocol in protocols:
        for case in cases[protocol]:
            parts, _ = work_seconds(case)
            lines.append(f"| {protocol} | {case['run'].name} | {case['threads']} | "
                         + " | ".join(f"{parts.get(k, 0):.1f}" for k, _ in WORKS + [("idle", "")])
                         + " |")
    gpu_cases = [(p, c) for p in protocols for c in cases[p] if c["impl"] in GPU]
    if gpu_cases:
        lines += ["", "## GPU (s busy)", "",
                  "| Case | Run | " + " | ".join(f"kernel {k}" for k, _ in KERNELS) + " | "
                  + " | ".join(f"copy {k}" for k, _ in COPIES) + " | GPU levels | hand-offs "
                  "(nodes, rows, GB) | host→GPU GB | GPU→host GB |",
                  "|---|---|" + "--:|" * (len(KERNELS) + len(COPIES) + 4)]
        for protocol, case in gpu_cases:
            kernels, copies, _ = gpu_lanes(case)
            facts = case["trace"]["facts"]
            levels = len(case["trace"]["tables"].get("gpu_levels", []))
            lines.append(
                f"| {protocol} | {case['run'].name} | "
                + " | ".join(f"{kernels.get(k, 0):.2f}" for k, _ in KERNELS) + " | "
                + " | ".join(f"{copies.get(k, 0):.2f}" for k, _ in COPIES)
                + f" | {levels} | {facts.get('handoff_nodes', 0):,.0f}, "
                f"{facts.get('handoff_rows', 0):,.0f}, {facts.get('handoff_bytes', 0) / 1e9:.1f} | "
                f"{facts.get('h2d_bytes', 0) / 1e9:.1f} | {facts.get('d2h_bytes', 0) / 1e9:.1f} |")
    out.write_text("\n".join(lines) + "\n")
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--plain", type=Path, help="directory with the same runs without --profile "
                        "(default: each run's sibling plain/, if there is one)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    loaded = [load_run(run) for run in args.runs]
    protocols = [p for p in ORDER if any(p in cases for cases in loaded)]
    cases = {p: [c[p] for c in loaded if p in c] for p in protocols}
    for protocol in protocols:
        trees = {(c["row"]["nodes"], c["row"]["depth"]) for c in cases[protocol]}
        if len(trees) > 1:
            raise SystemExit(f"{protocol}: different trees {sorted(trees)}")
    first = cases[protocols[0]][0]
    plan = json.load(open(first["run"] / "plan.json"))
    dataset = first["row"]["dataset"]
    meta = (dataset, plan["datasets"][dataset]["n_train"], len(plan["datasets"][dataset]["features"]))
    plain = {run.name: plain_times(args.plain or run.parent / "plain", run) for run in args.runs}
    ratios = [c["row"]["train_seconds"] / plain[c["run"].name][p][0] - 1
              for p in protocols for c in cases[p] if p in plain.get(c["run"].name, {})]
    overhead = (f"the profiled runs took {100 * min(ratios):+.0f}% to {100 * max(ratios):+.0f}% "
                "of the same cases without --profile (one run each)" if ratios
                else "not measured")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chart_phases(cases, protocols, meta, args.out_dir / "chart-phases.png", overhead)
    chart_threads(cases, protocols, meta, args.out_dir / "chart-threads.png")
    chart_gpu(cases, protocols, meta, args.out_dir / "chart-gpu.png")
    for protocol in protocols:
        chart_timeline(cases, protocol, meta, args.out_dir / f"timeline-{protocol}.png")
    report(cases, protocols, plain, args.out_dir / "profile.md")


if __name__ == "__main__":
    main()
