#!/usr/bin/env python3
"""Where ./tree's training time goes, as shares of the training wall time: one
3×3 grid (rows: Parallel, CUDA, CUDA --gpu-only; columns: CART, C4.5, C4.5
unpruned), one stacked bar per dataset, from runs made with
bench/cluster/profile.sh (results.jsonl + profiles/*.json, ./tree --profile,
src/core/trace.h) in <root> and its subdirectories in SUBDIRS.

  d=bench/results/cluster-profiling
  bench/.venv/bin/python bench/profile_share_chart.py $d --out $d/chart.png

Every millisecond of the training time goes to one step (attribute()): first
to the GPU kernels running then, then to copies between host and GPU, and the
rest by what the CPU threads were doing (the 5 ms samples) or, while the
training thread drives the GPU, to staging copies or to the host's driving.
Writes the PNG and prints the numbers as a Markdown table (for BENCHMARK.md).
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES  # noqa: E402

DATASETS = ["chess_10m", "chess_100m", "chess_1b", "higgs", "ecbdl_15m"]
LABELS = {"chess_10m": "chess 10M", "chess_100m": "chess 100M", "chess_1b": "chess 1000M",
          "higgs": "HIGGS", "ecbdl_15m": "ECBDL'14 15M"}
PROTOCOLS = [("cart_alpha_nodepth", "CART"), ("c45", "C4.5"), ("c45_unpruned", "C4.5 unpruned")]
SUBDIRS = ["", "c45-unpruned"]  # under <root>: profile.sh -o directories, one per batch of runs
MODES = [("cpu-epyc", "Parallel (CPU only)"),
         ("cuda-hopper", "CUDA hybrid (default)"),
         ("gpuonly-hopper", "CUDA --gpu-only")]

# The steps, in stacking order (bottom up), with the reference palette's
# categorical slots in order; "other" is a neutral gray.
STEPS = [
    ("presort", "Presort", "#2a78d6"),
    ("search", "Split search", "#eb6834"),
    ("partition", "Partition", "#1baf7a"),
    ("transfer", "CPU↔GPU transfer", "#eda100"),
    ("prune", "Pruning", "#e87ba4"),
    ("other", "Other", "#a3a29d"),
]
COLOR = {key: color for key, _, color in STEPS}
NAME = {key: name for key, name, _ in STEPS}
EXPLAIN = {
    "presort": "Sorting every feature column once, before growing starts (CPU: parallel sort of "
               "each column; GPU: gather + radix sort + distinct values). Every node after that "
               "reads its rows already in order, so no node sorts again.",
    "search": "Finding the best threshold of every feature at every node: walking the sorted "
              "values, keeping the class counts left/right of each cut and scoring it (CART: Gini; "
              "C4.5: information gain, then gain ratio), then choosing the winning feature. "
              "GPU: per-tile histograms, prefix sums and cut scans.",
    "partition": "Applying the chosen split: marking which rows go left, then moving every other "
                 "column so each child's rows are contiguous and still sorted. GPU: also gathering "
                 "the columns of small nodes it hands to the CPU (hybrid mode).",
    "transfer": "Moving data between RAM and the GPU: the upload of the dataset, the columns of "
                "small nodes sent back to the CPU (hybrid), per-level lists and cuts, and the CPU "
                "copies through page-locked buffers that feed them.",
    "prune": "Cutting the grown tree back. CART: cost-complexity pruning (the weakest-link "
             "sequence of subtrees, cut at the fixed α = 1e-5). C4.5: rows grouped by leaf, the "
             "pessimistic error estimate of every subtree (CF 0.25) with subtree raising, "
             "collapsing branches, final thresholds.",
    "other": "Everything else: GPU context and memory allocation, the host preparing and "
             "launching kernels or waiting while the GPU is idle, node bookkeeping, assembling "
             "the final tree, and time no thread was tracked in.",
}

KERNEL_STEP = {"presortGather": "presort", "presortSort": "presort", "presortDistinct": "presort",
               "markGoesLeft": "partition", "partition": "partition", "handOffGather": "partition",
               "leafRows": "prune", "arrangeSort": "prune", "arrangeRows": "prune",
               "copyValues": "prune"}  # everything else (tile*, segment*): search
# What a CPU thread does (trace Work names) -> step.
WORK_STEP = {"presort": "presort", "sweep": "search", "mark": "partition",
             "partition": "partition", "partition_sweep": "partition", "staging": "transfer",
             "page_lock": "transfer", "prune_arrange": "prune", "prune": "prune",
             "prune_other": "prune", "build": "other", "leaves": "other", "assemble": "other",
             "gpu_host": "other", "gpu_wait": "other", "other": "other"}
# Phases in which the training thread drives the GPU: there the CPU's part is
# staging copies (transfer) or driving (other), whatever the pool does meanwhile.
DRIVING = {"gpu setup", "upload", "grow setup", "gpu levels"}
PHASE_STEP = {"presort": "presort", "c45 collapse, thresholds": "prune", "prune setup": "prune",
              "prune arrange rows": "prune", "prune subtrees": "prune", "prune compact": "prune",
              "cart prune": "prune"}
BIN = 0.001  # seconds


def coverage(spans, edges):
    """Seconds of each bin [edges[i], edges[i+1]) covered by the union of spans."""
    spans = sorted((s, e) for s, e in spans if e > s)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    if not merged:
        return np.zeros(len(edges) - 1)
    xs, ys, total = [], [], 0.0
    for s, e in merged:
        xs += [s, e]
        ys += [total, total + e - s]
        total += e - s
    cumulative = np.interp(edges, xs, ys, left=0.0, right=total)
    return np.diff(cumulative)


def attribute(trace):
    """-> {(step, device): seconds} over the training time; device GPU, CPU or ''."""
    total = trace["seconds"]
    edges = np.append(np.arange(0.0, total, BIN), total)
    width = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    out = {}

    def add(key, values):
        out[key] = out.get(key, 0.0) + float(values.sum())

    # GPU: kernels by step, overlaps shared; copies where no kernel runs.
    kernels = {}
    for name, lane, start, end in trace.get("gpu", []):
        if lane == "compute":
            kernels.setdefault(KERNEL_STEP.get(name, "search"), []).append((start, end))
    copies = [(s, e) for _, lane, s, e in trace.get("gpu", []) if lane != "compute"]
    computing = coverage([span for spans in kernels.values() for span in spans], edges)
    busy = coverage([span for spans in kernels.values() for span in spans] + copies, edges)
    per_step = {step: coverage(spans, edges) for step, spans in kernels.items()}
    summed = sum(per_step.values()) if per_step else np.zeros_like(width)
    scale = np.divide(computing, summed, out=np.zeros_like(width), where=summed > 0)
    for step, seconds in per_step.items():
        add((step, "GPU"), seconds * scale)
    add(("transfer", ""), busy - computing)
    rest = np.clip(width - busy, 0.0, None)

    # The deepest phase at every bin (nested phases come later in depth order).
    phase = np.full(len(centers), "", dtype=object)
    for p in sorted(trace["phases"], key=lambda p: p["depth"]):
        phase[(centers >= p["start"]) & (centers < p["end"])] = p["name"]

    # The CPU threads' work at every bin: the nearest earlier sample.
    samples = trace["samples"]
    times = np.array(samples["time"])
    at = np.clip(np.searchsorted(times, centers, side="right") - 1, 0, len(times) - 1)
    counts = {name: np.array(values, dtype=float)[at]
              for name, values in samples["threads"].items() if name != "idle"}
    working = sum(counts.values())
    driving = np.isin(phase, list(DRIVING))
    staging = counts["staging"] + counts["page_lock"]
    add(("transfer", "CPU"), np.where(driving & (staging > 0), rest, 0.0))
    add(("other", ""), np.where(driving & (staging == 0), rest, 0.0))
    cpu = ~driving
    shared = np.where(cpu & (working > 0), rest / np.maximum(working, 1), 0.0)
    for name, threads in counts.items():
        step = WORK_STEP[name]
        add((step, "" if step in ("other", "transfer") else "CPU"), shared * threads)
    idle = cpu & (working == 0)
    for name in set(phase[idle]):
        step = PHASE_STEP.get(name, "other")
        add((step, "" if step == "other" else "CPU"), np.where(idle & (phase == name), rest, 0.0))
    # Transfer: one segment whichever side does the copying.
    out[("transfer", "")] = out.get(("transfer", ""), 0.0) + out.pop(("transfer", "CPU"), 0.0)
    return {key: value for key, value in out.items() if value > 0}


def runs(path):
    """-> {protocol: [rows of successful runs]} of a results.jsonl."""
    rows = {}
    if path.exists():
        for line in open(path):
            row = json.loads(line)
            if row["kind"] == "run" and row["status"] == "ok":
                rows.setdefault(row["protocol"], []).append(row)
    return rows


def median(rows):
    return sorted(rows, key=lambda row: row["train_seconds"])[(len(rows) - 1) // 2]


def load(root):
    """-> {(mode, dataset, protocol): case} of the median profiled run per case."""
    cases = {}
    for sub in SUBDIRS:
        for mode, _ in MODES:
            for dataset in DATASETS:
                run = root / sub / f"{mode}-{dataset}"
                if not (run / "results.jsonl").exists():
                    continue
                machine = json.load(open(run / "machine.json"))
                plain = runs(root / sub / "plain" / run.name / "results.jsonl")
                for protocol, rows in runs(run / "results.jsonl").items():
                    key = (mode, dataset, protocol)
                    rows = [row for row in rows if "profile" in row]
                    if not rows or key in cases:
                        continue
                    row = median(rows)
                    trace = json.load(open(run / row["profile"]))
                    cases[key] = {"row": row, "trace": trace, "machine": machine, "runs": len(rows),
                                  "parts": attribute(trace),
                                  "plain": median(plain[protocol]) if protocol in plain else None}
    return cases


def ordered(parts):
    keys = []
    for step, _, _ in STEPS:
        keys += [(step, device) for device in ("GPU", "CPU", "") if (step, device) in parts]
    return keys


def text_color(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def seconds_text(value):
    if value >= 100:
        return f"{value:.0f} s"
    if value >= 10:
        return f"{value:.1f} s"
    return f"{value:.2f} s"


# Inside the bars: shorter names, so labels stay within their bar.
SHORT = {"search": "Search", "transfer": "Transfer"}


def segment_label(key, share, seconds, short):
    step, device = key
    name = SHORT.get(step, NAME[step]) + (f" · {device}" if device else "")
    numbers = f"{share:.0f}% · {seconds_text(seconds)}" if share >= 9.5 else \
        f"{share:.1f}% · {seconds_text(seconds)}"
    return numbers if short else f"{name}\n{numbers}"


def draw_panel(ax, cases, mode, protocol):
    width = 0.46
    outside = []  # (x, y, text) of segments too thin to hold their label
    ax.set_xlim(-0.5, len(DATASETS) - 0.02)
    ax.set_ylim(0, 108)
    for x, dataset in enumerate(DATASETS):
        case = cases.get((mode, dataset, protocol))
        if case is None:
            ax.text(x, 50, "no run", ha="center", va="center", color=MUTED, fontsize=11)
            continue
        parts = case["parts"]
        total = sum(parts.values())
        bottom = 0.0
        for key in ordered(parts):
            share = 100 * parts[key] / total
            color = COLOR[key[0]]
            ax.add_patch(Rectangle((x - width / 2, bottom), width, share, facecolor=color,
                                   edgecolor="white", linewidth=1.6, zorder=2))
            if key[1] == "GPU":  # GPU work: hatched
                ax.add_patch(Rectangle((x - width / 2, bottom), width, share, facecolor="none",
                                       edgecolor=(1, 1, 1, 0.45), hatch="///", linewidth=0,
                                       zorder=3))
            if share >= 1:
                middle = bottom + share / 2
                if share >= 7:
                    ax.text(x, middle, segment_label(key, share, parts[key], False), ha="center",
                            va="center", fontsize=9.6, color=text_color(color),
                            fontweight="bold", linespacing=1.15, zorder=4,
                            bbox=dict(boxstyle="round,pad=0.12", facecolor=color, alpha=0.8,
                                      linewidth=0))  # also where the text is wider than the bar
                elif share >= 3.2:
                    ax.text(x, middle, segment_label(key, share, parts[key], True), ha="center",
                            va="center", fontsize=8.2, color=text_color(color),
                            fontweight="bold", zorder=4,
                            bbox=dict(boxstyle="round,pad=0.08", facecolor=color, alpha=0.8,
                                      linewidth=0))
                else:
                    device = f"{key[1]} " if key[1] else ""
                    outside.append((x, middle, f"{device}{share:.1f}%\n{seconds_text(parts[key])}",
                                    COLOR[key[0]]))
            bottom += share
        ax.text(x, 101.2, seconds_text(case["row"]["train_seconds"]), ha="center", va="bottom",
                fontsize=11.5, color=INK, fontweight="bold")
    # Thin segments: labels right of their bar (in the step's color), pushed apart vertically.
    by_bar = {}
    for item in outside:
        by_bar.setdefault(item[0], []).append(item)
    for x, items in by_bar.items():
        items.sort(key=lambda item: item[1])
        placed = []
        for _, y, text, color in items:
            y_text = max(y, placed[-1] + 5.4) if placed else max(y, 2.6)
            placed.append(y_text)
            ax.annotate(text, xy=(x + width / 2, y), xytext=(x + width / 2 + 0.12, y_text),
                        fontsize=7.4, linespacing=1.1, color=text_color(color), fontweight="bold", va="center",
                        ha="left", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor=color, linewidth=0),
                        arrowprops=dict(arrowstyle="-", color=color, linewidth=1.2,
                                        shrinkA=0, shrinkB=0))
    ax.set_xticks(range(len(DATASETS)))
    ticks = []
    for dataset in DATASETS:
        trace = next((c["trace"] for (m, d, _), c in cases.items() if d == dataset), None)
        shape = f"\n{trace['rows'] / 1e6:g}M rows × {trace['features']}" if trace else ""
        ticks.append(LABELS[dataset] + shape)
    ax.set_xticklabels(ticks, fontsize=10.5, color=INK)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=10, color=MUTED)
    ax.set_ylabel("share of training time", fontsize=10.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def hardware_text(cases, mode):
    case = next((c for (m, _, _), c in cases.items() if m == mode), None)
    if case is None:
        return ""
    info = B.hardware(case["machine"])
    node = case["machine"].get("host", "").split(".")[0]
    threads = case["row"]["threads"]
    if mode == "cpu-epyc":
        return f"{node}: {info['cpu']}, all used"
    gpu = info["gpu"].split(",")[0] if info["gpu"] else "GPU"
    return f"{node}: {gpu} + {threads} CPU threads ({info['cpu'].split(',')[0]})"


def chart(cases, out):
    fig = plt.figure(figsize=(32, 32))
    grid = fig.add_gridspec(len(MODES), len(PROTOCOLS), left=0.032, right=0.993, top=0.93,
                            bottom=0.19, hspace=0.3, wspace=0.1)
    for r, (mode, mode_title) in enumerate(MODES):
        for c, (protocol, algorithm) in enumerate(PROTOCOLS):
            ax = fig.add_subplot(grid[r, c])
            draw_panel(ax, cases, mode, protocol)
            ax.set_title(f"{mode_title} · {algorithm}", loc="left", fontsize=15,
                         fontweight="bold", color=INK, pad=30)
            ax.text(0, 1.025, hardware_text(cases, mode), transform=ax.transAxes, fontsize=10.5,
                    color=MUTED, va="bottom")
    commits = sorted({c["machine"]["versions"].get("tree_git", "?") for c in cases.values()})
    fig.suptitle("Where ./tree's training time goes: share of each step, per dataset",
                 x=0.012, y=0.992, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.012, 0.981, f"The median of each case's profiled runs (one run for CART and C4.5, three for C4.5 unpruned; "
             f"./tree --profile, commit {', '.join(commits)}). Bar "
             "height = training wall time (100%), the total above each bar; reading the CSV is "
             "not included.\nLabels: share and seconds, for segments of 1% or more (thin ones beside the bar, GPU/CPU = where it ran).",
             fontsize=11.5, color=MUTED, va="top", linespacing=1.4)
    handles = [Patch(facecolor=color, label=name) for _, name, color in STEPS]
    handles.append(Patch(facecolor="#7d7c78", hatch="///", edgecolor="white", label="hatched: on the GPU"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.966), ncol=7,
               frameon=False, fontsize=12.5, handlelength=2.2, handleheight=1.3,
               columnspacing=2.0)

    # The steps, explained, below the charts.
    top = 0.158
    fig.text(0.012, top + 0.006, "What each segment measures", fontsize=15, fontweight="bold",
             color=INK)
    y = top - 0.008
    for key, name, color in STEPS:
        fig.patches.append(Rectangle((0.012, y - 0.0085), 0.0062, 0.0075, transform=fig.transFigure,
                                     facecolor=color, edgecolor="none"))
        lines = textwrap.wrap(EXPLAIN[key], 290)
        fig.text(0.023, y, name, fontsize=12, fontweight="bold", color=INK, va="top")
        fig.text(0.085, y, "\n".join(lines), fontsize=11.2, color=INK, va="top", linespacing=1.35)
        y -= 0.0068 * len(lines) + 0.0075
    notes = [
        "How time is assigned: every millisecond of training goes to the GPU kernels running then "
        "(by kernel), else to a host↔GPU copy, else to what the CPU threads were doing (sampled "
        "every 5 ms, idle threads ignored, the time shared among the busy ones). While the "
        "training thread drives the GPU, its time is transfer (staging copies) or other (launching, "
        "waiting). · CPU / GPU tells which processor did the step; transfer is between them.",
        "Parallel: CPU only, every thread of both sockets. CUDA hybrid (default): the GPU grows "
        "the big nodes level by level and hands small nodes (< --gpu-min-rows) to the CPU threads, "
        "which grow those subtrees meanwhile; that CPU work is hidden behind the GPU and only "
        "shows when the GPU has finished and waits for it. CUDA --gpu-only: the GPU grows every "
        "node; the CPU only drives it and prunes.",
        "CART = cart_alpha_nodepth (Gini, no depth limit, cost-complexity pruning at α = 1e-5); C4.5 = c45 "
        "(gain ratio, CF 0.25 error-based pruning with subtree raising); C4.5 unpruned = c45_unpruned (the "
        "same tree grown, only C4.5's collapse of useless splits, no pruning). Same trees in every mode.",
    ]
    y -= 0.004
    for note in notes:
        lines = textwrap.wrap(note, 345)
        fig.text(0.012, y, "\n".join(lines), fontsize=10.8, color=MUTED, va="top", linespacing=1.35)
        y -= 0.0066 * len(lines) + 0.005
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


def table(cases):
    print("| Mode | Algorithm | Dataset | runs | train s | without --profile s | "
          + " | ".join(f"{NAME[s]} %" for s, _, _ in STEPS) + " | GPU share % |")
    print("|---|---|---|--:|--:|--:|" + "--:|" * len(STEPS) + "--:|")
    for mode, mode_title in MODES:
        for protocol, algorithm in PROTOCOLS:
            for dataset in DATASETS:
                case = cases.get((mode, dataset, protocol))
                if case is None:
                    continue
                parts = case["parts"]
                total = sum(parts.values())
                steps = [sum(v for (s, _), v in parts.items() if s == step) for step, _, _ in STEPS]
                gpu = sum(v for (_, d), v in parts.items() if d == "GPU")
                plain = case["plain"]["train_seconds"] if case["plain"] else float("nan")
                print(f"| {mode_title} | {algorithm} | {LABELS[dataset]} | "
                      f"{case['runs']} | {case['row']['train_seconds']:.2f} | {plain:.2f} | "
                      + " | ".join(f"{100 * v / total:.1f}" for v in steps)
                      + f" | {100 * gpu / total:.1f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="results directory (e.g. cluster-profiling)")
    parser.add_argument("--out", type=Path, help="PNG (default: <root>/chart.png)")
    args = parser.parse_args()
    cases = load(args.root)
    chart(cases, args.out or args.root / "chart.png")
    table(cases)


if __name__ == "__main__":
    main()
