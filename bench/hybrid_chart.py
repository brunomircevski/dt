#!/usr/bin/env python3
"""The hybrid-vs-gpu-only chart: ./tree --hybrid (nodes below
--handoff-rows go to the CPU threads) against ./tree --gpu (every
node on the GPU), on one GPU node, grown to depth 5, 10, 15, 20, 25, 30 and
without a depth limit, no pruning. One panel per dataset, one group of bars per
depth: CART hybrid, CART GPU only, C4.5 hybrid, C4.5 GPU only. A bar is the
mean training time of its runs, the whisker from the fastest to
the slowest; above each pair, GPU only / hybrid; under each group, the trees'
nodes. The --handoff-rows each dataset ran with comes from
<dir>/sweep/<node>-<dataset>.json ({"min_rows": N, ...}), written by the sweep
(bench/cluster/hybrid_vs_gpu_only.sh).

  bench/.venv/bin/python bench/hybrid_chart.py bench/results/hybrid-vs-gpu-only --node ampere

Writes <dir>/chart-<node>.png.
"""

import argparse
import json
import re
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


def min_rows(rows):
    """The --handoff-rows (before: --gpu-min-rows) values in these result rows' commands
    (empty: ./tree's default)."""
    return sorted({int(m.group(1)) for r in rows if (m := re.search(r"--(?:handoff-rows|gpu-min-rows) (\d+)", r.get("command", "")))})


def tint(color, alpha):
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(1 - (1 - c) * alpha for c in rgb)


def load(folder, node):
    """{dataset: {"machine": ..., "rows": {(backend, protocol): [result rows]}}}"""
    data = {}
    for run in sorted(folder.iterdir()):
        results = run / "results.jsonl"
        if not run.is_dir() or not results.exists():
            continue
        backend, machine, dataset = run.name.split("-", 2)
        if machine != node:
            continue
        entry = data.setdefault(dataset, {"machine": None, "rows": {}})
        if (run / "machine.json").exists():
            entry["machine"] = json.loads((run / "machine.json").read_text())
        if (run / "machine-depth5-10.json").exists():  # depth 5 and 10 from an earlier job
            entry["machine_shallow"] = json.loads((run / "machine-depth5-10.json").read_text())
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
    top = max(row["train_seconds"] for rows in entry["rows"].values() for row in rows)
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
                ax.text(x, max(means.values()) + top * 0.015, f"{ratio:.2f}×", ha="center", va="bottom",
                        fontsize=7.5, color=INK if abs(ratio - 1) >= 0.05 else MUTED,
                        fontweight="bold" if ratio >= 1.5 else "normal")
        name = "no limit" if depth is None else f"depth {depth}"
        counts = "\n".join(f"{ALGORITHMS[a]} {', '.join(f'{n:,}' for n in sorted(v))}"
                           for a, v in nodes.items())
        labels.append(f"{name}\n{counts}")
    ax.set_xticks(range(len(DEPTHS)), labels, fontsize=7.5)
    ax.yaxis.set_major_formatter(lambda v, _: "0 s" if v == 0 else seconds_tick(v))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, top * 1.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("training time", color=MUTED, fontsize=9)
    m = sweep.get(dataset, {})
    rows = entry["rows"]
    any_row = next(iter(next(iter(rows.values()))), {}) if rows else {}
    shape = (f"{any_row.get('n_train_loaded', 0):,} rows × {any_row.get('n_features_loaded', 0):,} features"
             if any_row else "")
    title = f"{NAMES.get(dataset, dataset)}: {shape}"
    if m:
        title += f"; hybrid at --handoff-rows {m['min_rows']:,}"
        low = min_rows([r for (b, p), v in rows.items() if b == "cuda"
                        for r in v if p in {protocol(a, d) for a in ALGORITHMS for d in DEPTHS[:2]}])
        if low and low != [m["min_rows"]]:
            title += f" (depth 5 and 10: {', '.join(f'{v:,}' for v in low)})"
    ax.text(0, 1.075, title, transform=ax.transAxes, fontsize=10.5, color=INK, va="bottom")
    times = m.get("times", {})
    chosen, default = times.get(str(m.get("min_rows"))), times.get("default")
    if chosen and default:
        def pair(t):
            return f"CART {duration(t['cart_full_nodepth'])}, C4.5 {duration(t['c45_unpruned'])}"
        tried = [int(v) for v in times if v != "default"]
        rows_in = any_row.get("n_train_loaded", 0)  # ./tree's default: rows/8192 (CART), rows/4096 (C4.5), clamped
        own = f"CART {min(max(rows_in // 8192, 1024), 2048):,}, C4.5 {min(max(rows_in // 4096, 1024), 4096):,}"
        ax.text(0, 1.02, f"sweep (no depth limit, {len(tried)} values {min(tried):,}–{max(tried):,}): {m['min_rows']:,}: {pair(chosen)}; "
                f"./tree's default ({own}): {pair(default)}", transform=ax.transAxes,
                fontsize=7.5, color=MUTED, va="bottom")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--node", default="ampere", help="the GPU node (run folders <backend>-<node>-<dataset>)")
    parser.add_argument("--datasets", help="comma list, in panel order (default: every one found)")
    args = parser.parse_args()
    data = load(args.folder, args.node)
    sweep = {f.name[len(args.node) + 1:-5]: json.loads(f.read_text())
             for f in (args.folder / "sweep").glob(f"{args.node}-*.json")}
    datasets = args.datasets.split(",") if args.datasets else sorted(data)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4.6 * len(datasets) + 1.6), squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        panel(ax, dataset, data[dataset], sweep)

    machine = next((d["machine"] for d in data.values() if d["machine"]), None) or {}
    rows = [r for d in data.values() for v in d["rows"].values() for r in v]
    reps = sorted({len(v) for d in data.values() for v in d["rows"].values()})
    versions = machine.get("versions", {})
    gpus = (versions.get("gpu") or "GPU").splitlines()
    gpu = gpus[int(machine.get("gpu_index", 0))].split(",")[0]
    commit = versions.get("tree_git", "?")
    shallow_sets = [ds for ds, d in data.items() if d.get("machine_shallow")]
    shallow = data[shallow_sets[0]]["machine_shallow"] if shallow_sets else None

    def is_shallow(r):
        return r["dataset"] in shallow_sets and any(r["protocol"] == protocol(a, d) for a in ALGORITHMS for d in DEPTHS[:2])

    def placement(some):
        threads = sorted({r["threads"] for r in some})
        dates = sorted({r["time"][:10] for r in some})
        return f"{'/'.join(map(str, threads))} CPU threads", f"{dates[0]}" + (f" to {dates[-1]}" if len(dates) > 1 else "")
    fig.suptitle("./tree CUDA: hybrid (GPU + CPU) vs GPU only, by depth limit, no pruning",
                 x=0.06, y=0.99, ha="left", fontsize=13, color=INK)
    legend = [Patch(facecolor=tint(COLORS[c], LIGHT[a]), edgecolor=COLORS[c],
                    label=f"{ALGORITHMS[a]}, {label}")
              for a in ALGORITHMS for c, label in BACKENDS.values()]
    fig.legend(handles=legend, loc="upper left", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.055, 0.965))
    every = placement([r for r in rows if not is_shallow(r)])
    where = f"{args.node} node: {gpu}, {every[0]} on both sockets (the GPU's first, memory not bound), ./tree {commit}, {every[1]}"
    if shallow:  # depth 5 and 10 of these datasets from an earlier one-socket job
        low = placement([r for r in rows if is_shallow(r)])
        names = ", ".join(NAMES.get(ds, ds) for ds in shallow_sets)
        where += (f"; {names} at depth 5 and 10: {low[0]} on the GPU's socket, "
                  f"./tree {shallow.get('versions', {}).get('tree_git', '?')}, {low[1]}")
    where += "."
    details = (
        f"{where} Runs per case: "
        f"{'/'.join(map(str, reps))}; bar = mean training time (load excluded), whisker = fastest to slowest.\n"
        "Hybrid (--hybrid): nodes with fewer rows than --handoff-rows are copied to the host and grown by the CPU "
        "threads while the GPU goes on with the next level. GPU only (--gpu): every node, down to the "
        "leaves, on the GPU.\n"
        "Above each pair: GPU-only time / hybrid time (above 1: hybrid faster). Under each group: the tree's nodes "
        "(the same in both modes). --handoff-rows: the fastest value of a sweep on this node (BENCHMARK.md).")
    fig.text(0.06, 0.012, details, fontsize=7.8, color=MUTED, va="bottom", wrap=True)
    fig.tight_layout(rect=(0.02, 0.09, 1, 0.94), h_pad=3)
    out = args.folder / f"chart-{args.node}.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
