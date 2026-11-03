#!/usr/bin/env python3
"""The hybrid-vs-gpu-only chart: ./tree --cuda (hybrid: nodes below
--gpu-min-rows go to the CPU threads) against ./tree --cuda --gpu-only (every
node on the GPU), on one GPU node, grown to depth 5, 10, 15, 20, 25, 30 and
without a depth limit, no pruning. One panel per dataset, one group of bars per
depth: CART hybrid, CART GPU only, C4.5 hybrid, C4.5 GPU only. A bar is the
mean training time of its runs (log scale), the whisker from the fastest to
the slowest; above each pair, GPU only / hybrid; under each group, the trees'
nodes. The --gpu-min-rows each dataset ran with comes from <dir>/sweep.json
({dataset: {"min_rows": N, "default": {"cart": N, "c45": N}, "note": "..."}}),
written after the sweep (bench/results/hybrid-vs-gpu-only/BENCHMARK.md).

  bench/.venv/bin/python bench/hybrid_chart.py bench/results/hybrid-vs-gpu-only

Writes <dir>/chart.png.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from chart import GRID, INK, MUTED, NAMES, seconds_tick  # noqa: E402
from machines_chart import COLORS  # noqa: E402  (the validated backend colours)

DEPTHS = [5, 10, 15, 20, 25, 30, None]  # None: no depth limit
ALGORITHMS = {"cart": "CART", "c45": "C4.5"}
BACKENDS = {"cuda": ("cuda", "hybrid"), "gpuonly": ("gpu_only", "GPU only")}
LIGHT = {"cart": 1.0, "c45": 0.55}  # C4.5's bars are lighter


def protocol(algorithm, depth):
    if depth is None:
        return "cart_full_nodepth" if algorithm == "cart" else "c45_unpruned"
    return f"cart_depth{depth}" if algorithm == "cart" else f"c45_depth{depth}_unpruned"


def tint(color, alpha):
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(1 - (1 - c) * alpha for c in rgb)


def load(folder):
    """{dataset: {"machine": ..., "rows": {(backend, protocol): [result rows]}}}"""
    data = {}
    for run in sorted(folder.iterdir()):
        results = run / "results.jsonl"
        if not run.is_dir() or not results.exists():
            continue
        backend, _, dataset = run.name.split("-", 2)
        entry = data.setdefault(dataset, {"machine": None, "rows": {}})
        if (run / "machine.json").exists():
            entry["machine"] = json.loads((run / "machine.json").read_text())
        for line in results.read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") == "run" and row.get("status") == "ok":
                entry["rows"].setdefault((backend, row["protocol"]), []).append(row)
    return data


def duration(value):
    return f"{value:.3g} s" if value >= 1 else f"{value * 1e3:.3g} ms"


def panel(ax, dataset, entry, sweep):
    width = 0.2
    labels = []
    for g, depth in enumerate(DEPTHS):
        nodes = {}
        for a, algorithm in enumerate(ALGORITHMS):
            means = {}
            for b, (backend, (color_key, _)) in enumerate(BACKENDS.items()):
                rows = entry["rows"].get((backend, protocol(algorithm, depth)), [])
                if not rows:
                    continue
                times = [row["train_seconds"] for row in rows]
                mean = statistics.mean(times)
                means[backend] = mean
                nodes.setdefault(algorithm, set()).add(rows[0]["nodes"])
                x = g + (a * 2 + b - 1.5) * width
                ax.bar(x, mean, width * 0.92, color=tint(COLORS[color_key], LIGHT[algorithm]),
                       edgecolor=COLORS[color_key], linewidth=0.6, zorder=2)
                ax.plot([x, x], [min(times), max(times)], color=INK, linewidth=0.8, zorder=3)
            if len(means) == 2:
                ratio = means["gpuonly"] / means["cuda"]
                x = g + (a * 2 - 0.5) * width
                ax.text(x, max(means.values()) * 1.12, f"{ratio:.2f}×", ha="center", va="bottom",
                        fontsize=7.5, color=INK if abs(ratio - 1) >= 0.05 else MUTED,
                        fontweight="bold" if ratio >= 1.5 else "normal")
        name = "no limit" if depth is None else f"depth {depth}"
        counts = "\n".join(f"{ALGORITHMS[a]} {', '.join(f'{n:,}' for n in sorted(v))}"
                           for a, v in nodes.items())
        labels.append(f"{name}\n{counts}")
    ax.set_xticks(range(len(DEPTHS)), labels, fontsize=7.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(seconds_tick)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("training time (log scale)", color=MUTED, fontsize=9)
    m = sweep.get(dataset, {})
    rows = entry["rows"]
    any_row = next(iter(next(iter(rows.values()))), {}) if rows else {}
    shape = (f"{any_row.get('n_train_loaded', 0):,} rows × {any_row.get('n_features_loaded', 0):,} features"
             if any_row else "")
    title = f"{NAMES.get(dataset, dataset)}: {shape}"
    if m:
        title += f"; hybrid at --gpu-min-rows {m['min_rows']:,}"
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    if m.get("note"):
        ax.text(0, 1.0, m["note"], transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--datasets", help="comma list, in panel order (default: every one found)")
    args = parser.parse_args()
    data = load(args.folder)
    sweep_file = args.folder / "sweep.json"
    sweep = json.loads(sweep_file.read_text()) if sweep_file.exists() else {}
    datasets = args.datasets.split(",") if args.datasets else sorted(data)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4.6 * len(datasets) + 1.6), squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        panel(ax, dataset, data[dataset], sweep)

    machine = next((d["machine"] for d in data.values() if d["machine"]), None) or {}
    rows = [r for d in data.values() for v in d["rows"].values() for r in v]
    reps = sorted({len(v) for d in data.values() for v in d["rows"].values()})
    gpu = machine.get("gpu") or "GPU"
    if isinstance(gpu, list):
        gpu = gpu[0]
    if isinstance(gpu, dict):
        gpu = gpu.get("name", "GPU")
    threads = sorted({r["threads"] for r in rows})
    commit = machine.get("tree_git", "?")
    date = min((r["time"][:10] for r in rows), default="?")
    fig.suptitle("./tree CUDA: hybrid (GPU + CPU) vs GPU only, by depth limit, no pruning",
                 x=0.06, ha="left", fontsize=13, color=INK)
    legend = [Patch(facecolor=tint(COLORS[c], LIGHT[a]), edgecolor=COLORS[c],
                    label=f"{ALGORITHMS[a]}, {label}")
              for a in ALGORITHMS for c, label in BACKENDS.values()]
    fig.legend(handles=legend, loc="upper right", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.98, 0.985))
    details = (
        f"ampere node: {gpu}, {threads[0] if threads else '?'} CPU threads (the GPU's socket, memory bound "
        f"to its NUMA node), ./tree {commit}, {date}. Runs per case: "
        f"{'/'.join(map(str, reps))}; bar = mean training time (load excluded), whisker = fastest to slowest.\n"
        "Hybrid (--cuda): nodes with fewer rows than --gpu-min-rows are copied to the host and grown by the CPU "
        "threads while the GPU goes on with the next level. GPU only (--cuda --gpu-only): every node, down to the "
        "leaves, on the GPU.\n"
        "Above each pair: GPU-only time / hybrid time (above 1: hybrid faster). Under each group: the tree's nodes "
        "(the same in both modes). --gpu-min-rows: the fastest value of a sweep on this node (BENCHMARK.md).")
    fig.text(0.06, 0.012, details, fontsize=7.8, color=MUTED, va="bottom", wrap=True)
    fig.tight_layout(rect=(0.02, 0.09, 1, 0.95))
    out = args.folder / "chart.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
