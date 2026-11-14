#!/usr/bin/env python3
"""Thread-scaling chart of ./tree (cluster-cpu-scaling): speed-up over one
thread against the number of threads, one panel per node (epyc, dell), one
colour per dataset, CART solid and C4.5 dashed, with the ideal line, the start
of the second socket and the speed-up and parallel efficiency at the most
threads. 1 thread is the serial backend (--serial), more the parallel one.

Also writes scaling.csv in every run folder: one row per (protocol, threads)
with the mean time, speed-up over 1 thread, parallel efficiency and peak RSS.

  bench/.venv/bin/python bench/scaling_chart.py bench/results/cluster-cpu-scaling [--out chart.png]
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES  # noqa: E402

# Categorical slots 1-5 of the reference palette (slots 1-2, CART and C4.5,
# validated: CVD dE 24.7); both CART variants are CART, so they share slot 1.
COLORS = {"cart_alpha_nodepth": "#2a78d6", "cart_alpha": "#2a78d6", "c45": "#eb6834",
          "cart_depth10": "#2a78d6", "c45_depth10_unpruned": "#eb6834",
          "cart_full": "#1baf7a", "cart_depth12": "#eda100", "c45_unpruned": "#e87ba4"}
LABELS = {"cart_alpha_nodepth": "CART", "cart_alpha": "CART, depth ≤ 30", "c45": "C4.5",
          "cart_depth10": "CART, depth ≤ 10, no pruning",
          "c45_depth10_unpruned": "C4.5, depth ≤ 10, no pruning",
          "cart_full": "CART unpruned", "cart_depth12": "CART depth 12",
          "c45_unpruned": "C4.5 unpruned"}


def collect(run):
    """-> {(protocol, threads): [rows]} of the ./tree runs that went well, the dataset."""
    rows = [json.loads(line) for line in open(run / "results.jsonl")]
    runs = [r for r in rows if r["kind"] == "run" and r["impl"] == "tree"]
    bad = [r for r in runs if r["status"] != "ok"]
    for r in bad:
        print(f"warning: {r['protocol']}, {r['threads']} threads: {r['status']} (left out)")
    datasets = {r["dataset"] for r in runs}
    if len(datasets) != 1:
        raise SystemExit(f"expected one dataset in {run}, found {sorted(datasets)}")
    cases = defaultdict(list)
    for r in runs:
        if r["status"] == "ok":
            cases[(r["protocol"], r["threads"])].append(r)
    return cases, datasets.pop()


def summarise(cases):
    """-> {protocol: [row per thread count, ascending]} with means and speedup."""
    out = defaultdict(list)
    for (protocol, threads), rows in sorted(cases.items(), key=lambda kv: kv[0][1]):
        times = [r["train_seconds"] for r in rows]
        rss = [r["peak_rss_train_bytes"] for r in rows]
        out[protocol].append({
            "protocol": protocol, "threads": threads, "runs": len(rows),
            "time_mean": statistics.mean(times), "time_min": min(times),
            "time_max": max(times), "peak_rss_mean": statistics.mean(rss),
            "nodes": rows[0]["nodes"], "test_accuracy": rows[0].get("test_accuracy"),
            "train_accuracy": rows[0].get("train_accuracy"),
            "same_tree": len({(r["nodes"], r["leaves"], r["depth"]) for r in rows}) == 1,
            "rows": rows})
    for protocol, points in out.items():
        serial = next((p["time_mean"] for p in points if p["threads"] == 1), None)
        for p in points:
            p["speedup"] = serial / p["time_mean"] if serial else None
            p["efficiency"] = p["speedup"] / p["threads"] if serial else None
        # Every thread count must grow the same tree (the backends are exact).
        if len({p["nodes"] for p in points}) > 1:
            print(f"warning: {protocol}: the tree size differs between thread counts")
    return [p for p in B.PROTOCOLS if p in out], out


NODES = [("epyc", "epyc1"), ("dell", "dell1")]
DATASETS = [("higgs", "#2a78d6", "o"), ("mnist8m", "#eb6834", "s")]
STYLE = {"cart_depth10": ("CART", "-", True), "c45_depth10_unpruned": ("C4.5", "--", False)}
FIELDS = ["protocol", "threads", "runs", "time_mean", "time_min", "time_max", "speedup",
          "efficiency", "peak_rss_mean", "nodes", "test_accuracy", "train_accuracy", "same_tree"]


def load(base):
    """-> {(node, dataset): (series, machine)}, writing each run folder's scaling.csv."""
    out = {}
    for node, _ in NODES:
        for dataset, _, _ in DATASETS:
            run = base / f"cpu-{node}-{dataset}"
            if not (run / "results.jsonl").exists():
                print(f"missing: {run}")
                continue
            cases, found = collect(run)
            protocols, series = summarise(cases)
            with open(run / "scaling.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, FIELDS)
                writer.writeheader()
                for protocol in protocols:
                    writer.writerows({k: p[k] for k in FIELDS} for p in series[protocol])
            out[(node, dataset)] = (series, json.load(open(run / "machine.json")))
    return out


def runs_note(data):
    """'10 runs per point' or the counts that differ, for the footnote."""
    counts = defaultdict(set)
    for (node, dataset), (series, _) in data.items():
        for protocol, points in series.items():
            for p in points:
                counts[p["runs"]].add(f"{NAMES.get(dataset, dataset)} {p['threads']}")
    if len(counts) == 1:
        n = next(iter(counts))
        return f"mean of {n} runs per point" if n > 1 else "one run per point"
    common = max(counts, key=lambda n: len(counts[n]))
    rest = "; ".join(f"{n} runs: " + ", ".join(sorted(v)) for n, v in sorted(counts.items()) if n != common)
    return f"mean of {common} runs per point ({rest} threads)"


def figure(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    top = max(p["threads"] for series, _ in data.values() for pts in series.values() for p in pts)
    for ax, (node, host) in zip(axes, NODES):
        ax.plot([1, top], [1, top], color=GRID, linewidth=1.5, zorder=0)
        ax.text(top * 0.3, top * 0.3 + 2.5, "ideal", fontsize=9, color=MUTED, rotation=34)
        if node == "epyc":  # CPUs 0-23 socket 0, 24-47 socket 1; dell1's alternate between the sockets
            half = top // 2
            ax.axvline(half + 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (1, 3)), zorder=0)
            ax.text(half + 1.5, 1.5, "2nd socket \u2192", fontsize=8.5, color=MUTED, va="bottom")
        else:
            ax.text(top * 0.55, 1.5, "threads alternate between the sockets", fontsize=8.5, color=MUTED,
                    va="bottom")
        machine = None
        ends = []
        lines = []
        for dataset, color, marker in DATASETS:
            if (node, dataset) not in data:
                continue
            series, machine = data[(node, dataset)]
            one = []
            for protocol, (algo, ls, filled) in STYLE.items():
                points = series.get(protocol)
                if not points or points[0]["threads"] != 1:
                    continue
                t = [p["threads"] for p in points]
                sp = [p["speedup"] for p in points]
                ax.plot(t, sp, linestyle=ls, color=color, linewidth=1.8, marker=marker, markersize=5,
                        markerfacecolor=color if filled else "white", markeredgewidth=1.2, zorder=3)
                ends.append((sp[-1], t[-1], color, f"{sp[-1]:.0f}\u00d7 ({100 * points[-1]['efficiency']:.0f}%)"))
                one.append(f"{algo} {points[0]['time_mean']:.3g} s \u2192 {points[-1]['time_mean']:.3g} s")
            lines.append(f"{NAMES.get(dataset, dataset)}: " + ", ".join(one))
        # end labels, pushed apart where they would overlap
        placed = []
        for y, x, color, text in sorted(ends):
            yy = y if not placed or y - placed[-1] > 1.6 else placed[-1] + 1.6
            placed.append(yy)
            ax.annotate(text, (x, y), xytext=(x + 1, yy), fontsize=8.5, color=color, va="center")
        ax.set_xlim(0, top * 1.27)
        ax.set_ylim(0, top)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
        ax.set_xlabel("threads", fontsize=9.5, color=MUTED)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        cpu = B.hardware(machine)["cpu"] if machine else "?"
        ax.set_title(f"{host}: {cpu}", loc="left", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(0.01, 0.99, "1 thread \u2192 48 threads:\n" + "\n".join(lines), transform=ax.transAxes,
                fontsize=8, color=INK, va="top", linespacing=1.4,
                bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.4"))
    axes[0].set_ylabel("speed-up over 1 thread", fontsize=9.5, color=MUTED)
    handles = [Line2D([], [], color=c, linewidth=1.8, marker=m, markersize=5, label=NAMES.get(d, d))
               for d, c, m in DATASETS]
    handles += [Line2D([], [], color=MUTED, linewidth=1.8, linestyle=ls, marker="o", markersize=5,
                       markerfacecolor=MUTED if filled else "white", label=algo)
                for algo, ls, filled in STYLE.values()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), ncol=4, frameon=False,
               fontsize=9.5)
    fig.suptitle("./tree thread scaling: depth \u2264 10, no pruning", x=0.06, y=0.975, ha="left",
                 fontsize=12.5, color=INK, fontweight="bold")
    commits = sorted({m.get("versions", {}).get("tree_git", "?") for _, m in data.values()})
    days = sorted({r["time"][:10] for series, _ in data.values() for pts in series.values()
                   for p in pts for r in p["rows"]})
    dates = days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}"
    fig.text(0.06, 0.012,
             f"Speed-up: the 1-thread time (--serial) over the time with n threads (--multicore), {runs_note(data)}.\n"
             "End label: speed-up (parallel efficiency) at 48 threads. Threads take the CPUs in the kernel's order: on "
             "epyc1 socket 0's 24 cores, then socket 1's; on dell1 the sockets alternate. Memory first-touch.\nEvery thread count grows the same tree. Training time excludes "
             f"reading the data. ./tree {', '.join(commits)}, run {dates}.",
             fontsize=8.3, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2, wspace=0.08)
    fig.savefig(out, dpi=150, facecolor="white")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="bench/results/cluster-cpu-scaling")
    parser.add_argument("--out", help="output file (default: <base>/chart.png)")
    args = parser.parse_args()
    base = Path(args.base)
    print(figure(load(base), Path(args.out) if args.out else base / "chart.png"))


if __name__ == "__main__":
    main()
