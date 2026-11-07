#!/usr/bin/env python3
"""Who is working during hybrid-vs-gpu-only's training: ./tree --cuda (hybrid)
and --cuda --gpu-only, by depth limit, as shares of the training wall time.
One panel per algorithm (CART, C4.5, both unpruned), one group per depth (5 ..
30, no limit), two stacked bars per group: hybrid, GPU only. Each bar is the
average of the case's profiled runs (./tree --profile, src/core/trace.h), from
bench/cluster/hybrid_vs_gpu_only.sh (PROFILE_DATASETS) in
<dir>/profile/{cuda,gpuonly}-<node>-<dataset>/.

Every millisecond of the training time goes to one of (split()):
  setup         the phases before growing: GPU context and memory, upload, presort
  GPU alone     a GPU kernel runs, no CPU thread grows a subtree
  GPU + CPU     a GPU kernel runs and CPU threads grow subtrees (hybrid only)
  CPU alone     no kernel or copy runs, CPU threads grow subtrees
  transfer      a host<->GPU copy runs with no kernel, or the host stages copies
  host / other  none of these: the host launching kernels, waiting, assembling
                the tree, C4.5's tidy-up
CPU subtree work = threads in the trace's sweep, partition, mark, build or
leaves work (5 ms samples); under each bar its thread-seconds.

  d=bench/results/hybrid-vs-gpu-only
  bench/.venv/bin/python bench/hybrid_profile_chart.py $d --node ampere --dataset mnist8m

  bench/.venv/bin/python bench/hybrid_profile_chart.py $d --hybrid-only --seconds
  bench/.venv/bin/python bench/hybrid_profile_chart.py $d --hybrid-only

Writes <dir>/chart-profile-<node>-<dataset>.png (--hybrid-only:
chart-profile-hybrid-{seconds,percent}-<node>-<dataset>.png) and prints a Markdown table.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from chart import GRID, INK, MUTED, NAMES  # noqa: E402
from hybrid_chart import DEPTHS, protocol  # noqa: E402
from profile_share_chart import coverage, mean_row, runs, seconds_text, text_color  # noqa: E402

ALGORITHMS = [("cart", "CART"), ("c45", "C4.5")]
MODES = [("cuda", "hybrid"), ("gpuonly", "GPU only")]
# Bottom up. "both" is drawn GPU blue with CPU green stripes.
PARTS = [
    ("setup", "Setup (GPU init, upload, presort)", "#7e7d78"),
    ("gpu", "GPU working alone", "#2a78d6"),
    ("both", "GPU + CPU working at once", "#2a78d6"),
    ("cpu", "CPU working alone", "#1baf7a"),
    ("transfer", "CPU↔GPU transfer", "#eda100"),
    ("other", "Host: launching, waiting, finishing", "#cfcec9"),
]
COLOR = {key: color for key, _, color in PARTS}
NAME = {key: name for key, name, _ in PARTS}
CPU_GREEN = "#1baf7a"
SUBTREE = ["sweep", "partition", "partition_sweep", "mark", "build", "leaves"]
SETUP = {"gpu setup", "presort", "grow setup"}  # top-level phases before growing
BIN = 0.001


def split(trace):
    """-> ({part: seconds}, CPU subtree thread-seconds) over the training time."""
    total = trace["seconds"]
    edges = np.append(np.arange(0.0, total, BIN), total)
    width = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    spans = trace.get("gpu", [])
    kernel = coverage([(s, e) for _, lane, s, e in spans if lane == "compute"], edges)
    busy = coverage([(s, e) for _, _, s, e in spans], edges)
    copy = busy - kernel
    rest = np.clip(width - busy, 0.0, None)

    setup = np.zeros(len(centers), dtype=bool)
    for p in trace["phases"]:
        if p["depth"] == 0 and p["name"] in SETUP:
            setup |= (centers >= p["start"]) & (centers < p["end"])
    samples = trace["samples"]
    times = np.array(samples["time"])
    at = np.clip(np.searchsorted(times, centers, side="right") - 1, 0, len(times) - 1)
    threads = samples["threads"]
    count = lambda names: sum(np.array(threads[n], dtype=float) for n in names if n in threads)  # noqa: E731
    workers_samples = count(SUBTREE)
    workers = workers_samples[at] if np.ndim(workers_samples) else np.zeros(len(centers))
    staging_samples = count(["staging", "page_lock"])
    staging = staging_samples[at] if np.ndim(staging_samples) else np.zeros(len(centers))

    grow = ~setup
    cpu_on = workers > 0
    parts = {
        "setup": width[setup].sum(),
        "gpu": (kernel * grow * ~cpu_on).sum(),
        "both": (kernel * grow * cpu_on).sum(),
        "cpu": (rest * grow * cpu_on).sum(),
        "transfer": (copy * grow).sum() + (rest * grow * ~cpu_on * (staging > 0)).sum(),
        "other": (rest * grow * ~cpu_on * (staging == 0)).sum(),
    }
    thread_seconds = float(np.sum(workers_samples)) * samples["interval"] if np.ndim(workers_samples) else 0.0
    return {k: float(v) for k, v in parts.items()}, thread_seconds


def load(folder, node, dataset):
    """-> ({(mode, protocol): case}, machine, shallow): the average of each case's profiled
    runs; shallow: machine-depth5-10.json (depth 5 and 10 from an earlier job) or None."""
    cases, machine, shallow = {}, None, None
    for mode, _ in MODES:
        run = folder / "profile" / f"{mode}-{node}-{dataset}"
        if not (run / "results.jsonl").exists():
            continue
        machine = json.load(open(run / "machine.json"))
        if (run / "machine-depth5-10.json").exists():
            shallow = json.load(open(run / "machine-depth5-10.json"))
        plain = runs(folder / f"{mode}-{node}-{dataset}" / "results.jsonl")
        for proto, rows in runs(run / "results.jsonl").items():
            rows = [row for row in rows if "profile" in row]
            if not rows:
                continue
            splits = [split(json.load(open(run / row["profile"]))) for row in rows]
            parts = {key: sum(s[0][key] for s in splits) / len(splits) for key, _, _ in PARTS}
            cases[(mode, proto)] = {
                "row": mean_row(rows), "runs": len(rows), "parts": parts,
                "cpu_thread_seconds": sum(s[1] for s in splits) / len(splits),
                "plain": mean_row(plain[proto]) if proto in plain else None}
    return cases, machine, shallow


def draw_segment(ax, x, width, bottom, share, key):
    color = COLOR[key]
    ax.add_patch(Rectangle((x - width / 2, bottom), width, share, facecolor=color,
                           edgecolor="white", linewidth=1.2, zorder=2))
    if key == "both":  # CPU stripes over the GPU's blue
        ax.add_patch(Rectangle((x - width / 2, bottom), width, share, facecolor="none",
                               edgecolor=CPU_GREEN, hatch="//", linewidth=0, zorder=3))
        ax.add_patch(Rectangle((x - width / 2, bottom), width, share, facecolor="none",
                               edgecolor="white", linewidth=1.2, zorder=3))


def bar(ax, x, width, case, seconds, top):
    """One stacked bar: shares of 100% or (seconds) the parts' seconds; top = the axis' top."""
    parts = case["parts"]
    total = sum(parts.values())
    bottom = 0.0
    for key, _, color in PARTS:
        share = 100 * parts[key] / total
        height = parts[key] if seconds else share
        if height <= 0:
            continue
        draw_segment(ax, x, width, bottom, height, key)
        of_axis = 100 * height / top
        if of_axis >= 4.5:
            if seconds:
                text = f"{seconds_text(parts[key])}\n{share:.0f}%" if of_axis >= 9 else seconds_text(parts[key])
            else:
                text = f"{share:.0f}%\n{seconds_text(parts[key])}" if share >= 9 else f"{share:.0f}%"
            ax.text(x, bottom + height / 2, text, ha="center", va="center", fontsize=8.2,
                    color=text_color(color), fontweight="bold", zorder=4, linespacing=1.1,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.85, linewidth=0))
        bottom += height
    ax.text(x, (total if seconds else 100) + top * 0.008, seconds_text(case["row"]["train_seconds"]),
            ha="center", va="bottom", fontsize=8.8, color=INK, fontweight="bold")


def panel(ax, cases, algorithm, title, modes, seconds, top):
    single = len(modes) == 1
    width, gap = (0.6, 0) if single else (0.36, 0.42)
    ax.set_xlim(-0.6, len(DEPTHS) - 0.4)
    ax.set_ylim(0, top * 1.07)
    ticks, labels = [], []
    for g, depth in enumerate(DEPTHS):
        proto = protocol(algorithm, depth)
        nodes = None
        for m, (mode, short) in enumerate(modes):
            x = g + (m - (len(modes) - 1) / 2) * gap
            case = cases.get((mode, proto))
            ticks.append(x)
            head = "" if single else f"{short}\n"
            if case is None:
                labels.append(head)
                ax.text(x, top / 2, "no run", ha="center", va="center", rotation=90, color=MUTED,
                        fontsize=9)
                continue
            bar(ax, x, width, case, seconds, top)
            nodes = case["row"]["nodes"]
            cpu = case["cpu_thread_seconds"]
            labels.append(head + (f"CPU {cpu:.0f} thr·s" if cpu >= 0.5 else "CPU –"))
        name = "no limit" if depth is None else f"depth {depth}"
        ax.text(g, -0.155 if not single else -0.11, name + (f"\n{nodes:,} nodes" if nodes else ""),
                ha="center", va="top", fontsize=10, color=INK, fontweight="bold", clip_on=False,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(ticks, labels, fontsize=8.2, color=MUTED, linespacing=1.2)
    if seconds:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g} s")
        ax.tick_params(axis="y", labelsize=9, labelcolor=MUTED)
        ax.set_ylabel("training time (seconds)", fontsize=10, color=MUTED)
    else:
        ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], fontsize=9, color=MUTED)
        ax.set_ylabel("share of training time", fontsize=10, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=18)


def chart(cases, machine, shallow, node, dataset, out, modes=MODES, seconds=False):
    single = len(modes) == 1
    fig, axes = plt.subplots(len(ALGORITHMS), 1, figsize=(15, 13.5))
    top = max(case["row"]["train_seconds"] for (mode, _), case in cases.items()
              if mode in dict(modes)) if seconds else 100
    for ax, (algorithm, name) in zip(axes, ALGORITHMS):
        panel(ax, cases, algorithm, f"{name} (unpruned)", modes, seconds, top)
    versions = machine.get("versions", {}) if machine else {}
    gpus = (versions.get("gpu") or "GPU").splitlines()
    gpu = gpus[int(machine.get("gpu_index", 0))].split(",")[0] if machine else "GPU"
    row = next(iter(cases.values()))["row"]
    # One job, or depth 5 and 10 from an earlier one (load(): machine-depth5-10.json).
    groups = [("depth 15 to no limit", DEPTHS[2:]), ("depth 5 and 10", DEPTHS[:2])] if shallow else [("", DEPTHS)]
    parts = []
    for label, depths in groups:
        some = [c["row"] for (mode, p), c in cases.items()
                if p in {protocol(a, d) for a, _ in ALGORITHMS for d in depths}]
        hybrid = [c["row"] for (mode, p), c in cases.items()
                  if mode == "cuda" and p in {protocol(a, d) for a, _ in ALGORITHMS for d in depths}]
        threads = sorted({r["threads"] for r in some})
        m = shallow if label == "depth 5 and 10" else machine
        where = "the GPU's socket" if label == "depth 5 and 10" else "both sockets"
        rows_used = sorted({v for r in hybrid for v in re.findall(r"--gpu-min-rows (\d+)", r.get("command", ""))}, key=int)
        parts.append((f"{label}: " if label else "") + f"{'/'.join(map(str, threads))} CPU threads ({where}), ./tree "
                     f"{(m or {}).get('versions', {}).get('tree_git', '?')}, {(m or {}).get('started', '?')[:10]}"
                     + (f", hybrid at --gpu-min-rows {', '.join(f'{int(v):,}' for v in rows_used)}" if rows_used else ""))
    placement = ";\n".join(parts)
    counts = sorted({c["runs"] for c in cases.values()})
    what = "CUDA hybrid" if single else "CUDA hybrid vs GPU only"
    fig.suptitle(f"Who is working: {what}, {NAMES.get(dataset, dataset)} on "
                 f"{node}, by depth limit", x=0.01, y=0.99, ha="left", fontsize=16,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.958,
             f"{row.get('n_train_loaded', 0):,} rows × {row.get('n_features_loaded', 0):,} features · "
             f"{gpu}, --profile, {'/'.join(map(str, counts))} runs per bar (averaged). CPU threads (each group):\n" + placement + ".\n" + ("Bar = training wall time (linear), total above; labels: seconds "
             "(and share) for segments taller than 4.5% of the axis." if seconds else
             "Bar = training wall time (100%), seconds above; labels: share (and seconds) for "
             "segments of 4.5% or more.") + " Under each bar: CPU threads' subtree work "
             "in thread-seconds (48 threads × 1 s = 48 thr·s).",
             fontsize=10, color=MUTED, va="top")
    handles = []
    for key, name, color in PARTS:
        if key == "both":
            handles.append(Patch(facecolor=color, hatch="//", edgecolor=CPU_GREEN, label=name))
        else:
            handles.append(Patch(facecolor=color, label=name))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.905), ncol=6,
               frameon=False, fontsize=10, handlelength=2, handleheight=1.2, columnspacing=1.4)
    fig.text(0.01, 0.012,
             "Each millisecond goes to one segment: setup if before growing; else GPU (alone, or + CPU "
             "when CPU threads grow subtrees then) if a kernel runs; else transfer if a host↔GPU copy "
             "runs;\nelse CPU alone if CPU threads grow subtrees; else transfer if the host stages "
             "copies; else host (launching kernels, waiting, assembling the tree, C4.5's tidy-up). "
             "CPU threads are sampled every 5 ms.\nHybrid: nodes with fewer rows than --gpu-min-rows "
             "are copied to the host and grown by the CPU threads while the GPU goes on with the "
             "next level." + ("" if single else " GPU only: every node on the GPU; the CPU only drives it."),
             fontsize=9.2, color=MUTED, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89), h_pad=6)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


def table(cases):
    print("| Algorithm | Depth | Mode | runs | train s | without --profile s | nodes | "
          + " | ".join(f"{NAME[k]} s (%)" for k, _, _ in PARTS) + " | CPU subtree thr·s |")
    print("|---|---|---|--:|--:|--:|--:|" + "--:|" * len(PARTS) + "--:|")
    for algorithm, name in ALGORITHMS:
        for depth in DEPTHS:
            for mode, short in MODES:
                case = cases.get((mode, protocol(algorithm, depth)))
                if case is None:
                    continue
                parts = case["parts"]
                total = sum(parts.values())
                plain = case["plain"]["train_seconds"] if case["plain"] else float("nan")
                print(f"| {name} | {'none' if depth is None else depth} | {short} | {case['runs']} | "
                      f"{case['row']['train_seconds']:.2f} | {plain:.2f} | {case['row']['nodes']:,} | "
                      + " | ".join(f"{parts[k]:.2f} ({100 * parts[k] / total:.0f})" for k, _, _ in PARTS)
                      + f" | {case['cpu_thread_seconds']:.0f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="e.g. bench/results/hybrid-vs-gpu-only")
    parser.add_argument("--node", default="ampere")
    parser.add_argument("--dataset", default="mnist8m")
    parser.add_argument("--hybrid-only", action="store_true", help="only the hybrid's bars")
    parser.add_argument("--seconds", action="store_true", help="bars in seconds (linear), not shares")
    args = parser.parse_args()
    cases, machine, shallow = load(args.folder, args.node, args.dataset)
    if not cases:
        raise SystemExit(f"no profiled runs in {args.folder}/profile/*-{args.node}-{args.dataset}")
    modes = [m for m in MODES if m[0] == "cuda"] if args.hybrid_only else MODES
    unit = "seconds" if args.seconds else "percent"
    name = (f"chart-profile-hybrid-{unit}-{args.node}-{args.dataset}.png" if args.hybrid_only else
            f"chart-profile{'-seconds' if args.seconds else ''}-{args.node}-{args.dataset}.png")
    chart(cases, machine, shallow, args.node, args.dataset, args.folder / name, modes, args.seconds)
    table(cases)


if __name__ == "__main__":
    main()
