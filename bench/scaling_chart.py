#!/usr/bin/env python3
"""Thread-scaling chart of ./tree: training time (solid lines, left axis) and peak
memory (dashed lines, right axis) against the number of threads, one colour per
protocol (CART, C4.5); dot = median over the runs, bar = min-max of the time.

Also writes scaling.csv: one row per (protocol, threads) with the median time,
speedup over 1 thread (serial backend), parallel efficiency and peak RSS.

  bench/.venv/bin/python bench/scaling_chart.py bench/results/<run> [--out chart.png]
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
from matplotlib.ticker import FuncFormatter  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES  # noqa: E402

# Categorical slots 1-5 of the reference palette (slots 1-2, CART and C4.5,
# validated: CVD dE 24.7); both CART variants are CART, so they share slot 1.
COLORS = {"cart_alpha_nodepth": "#2a78d6", "cart_alpha": "#2a78d6", "c45": "#eb6834",
          "cart_full": "#1baf7a", "cart_depth12": "#eda100", "c45_unpruned": "#e87ba4"}
LABELS = {"cart_alpha_nodepth": "CART", "cart_alpha": "CART, depth ≤ 30", "c45": "C4.5",
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
    """-> {protocol: [row per thread count, ascending]} with medians and speedup."""
    out = defaultdict(list)
    for (protocol, threads), rows in sorted(cases.items(), key=lambda kv: kv[0][1]):
        times = [r["train_seconds"] for r in rows]
        rss = [r["peak_rss_train_bytes"] for r in rows]
        out[protocol].append({
            "protocol": protocol, "threads": threads, "runs": len(rows),
            "time_median": statistics.median(times), "time_min": min(times),
            "time_max": max(times), "peak_rss_median": statistics.median(rss),
            "nodes": rows[0]["nodes"], "test_accuracy": rows[0].get("test_accuracy"),
            "same_tree": len({(r["nodes"], r["leaves"], r["depth"]) for r in rows}) == 1})
    for protocol, points in out.items():
        serial = next((p["time_median"] for p in points if p["threads"] == 1), None)
        for p in points:
            p["speedup"] = serial / p["time_median"] if serial else None
            p["efficiency"] = p["speedup"] / p["threads"] if serial else None
        # Every thread count must grow the same tree (the backends are exact).
        if len({p["nodes"] for p in points}) > 1:
            print(f"warning: {protocol}: the tree size differs between thread counts")
    return [p for p in B.PROTOCOLS if p in out], out


def figure(run, dataset, protocols, series, settings, out):
    meta = json.load(open(run / "plan.json"))["datasets"][dataset]
    threads = sorted({p["threads"] for points in series.values() for p in points})
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.set_xlim(0, max(threads) * 1.13)  # room for the end labels
    ax.set_xticks(threads)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("threads", fontsize=9, color=MUTED)
    ax.set_title("Training time (solid, left) and peak memory (dashed, right)", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold")
    mem = ax.twinx()
    mem.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ("top", "left", "bottom"):
        mem.spines[side].set_visible(False)
    mem.spines["right"].set_color(GRID)

    for protocol in protocols:
        points, color = series[protocol], COLORS.get(protocol, INK)
        xs = [p["threads"] for p in points]
        ax.plot(xs, [p["time_median"] for p in points], color=color, linewidth=2, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.vlines(xs, [p["time_min"] for p in points], [p["time_max"] for p in points],
                  color=color, linewidth=2, zorder=2)
        mem.plot(xs, [p["peak_rss_median"] / 1e9 for p in points], color=color, linewidth=1.6,
                 linestyle=(0, (4, 2.5)), marker="s", markersize=5, markerfacecolor="white",
                 markeredgecolor=color, markeredgewidth=1.4, zorder=3)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    mem.set_ylim(0, 1.15 * max(p["peak_rss_median"] for points in series.values()
                               for p in points) / 1e9)
    mem.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} GB"))

    # Value at the last thread count, at the end of each line, pushed apart
    # vertically where two lines end close together.
    fig.canvas.draw()
    gap = 13  # pixels between label centres
    for axis, key, scale, unit in ((ax, "time_median", 1, "s"),
                                   (mem, "peak_rss_median", 1e9, "GB")):
        placed = []
        ends = [(series[p][-1]["threads"], series[p][-1][key] / scale) for p in protocols]
        for x, y in sorted(ends, key=lambda end: end[1]):
            pixel = wanted = axis.transData.transform((x, y))[1]
            if placed and pixel - placed[-1] < gap:
                pixel = placed[-1] + gap
            placed.append(pixel)
            axis.annotate(f"{y:.3g} {unit}", (x, y), xytext=(7, (pixel - wanted) * 72 / fig.dpi),
                          textcoords="offset points", va="center", fontsize=9, color=INK)

    handles = [Line2D([], [], color=COLORS.get(p, INK), linewidth=2, marker="o", markersize=7,
                      markeredgecolor="white", label=f"{LABELS.get(p, p)}: "
                      f"{series[p][0]['nodes']:,} nodes, test "
                      f"{100 * series[p][0]['test_accuracy']:.2f}%")
               for p in protocols if series[p][0].get("test_accuracy") is not None]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.925),
               ncol=len(handles), frameon=False, fontsize=9, handletextpad=0.4,
               columnspacing=1.6)
    fig.suptitle(f"./tree thread scaling on {NAMES.get(dataset, dataset)}: "
                 f"{meta['n_train']:,} training rows, {len(meta['features'])} features",
                 x=0.08, y=0.995, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    reps = settings.get("reps")
    runs = f"median of {reps} runs, bar: min–max" if reps and reps > 1 else "one run per point"
    fig.text(0.08, 0.015,
             f"Dot: {runs}. 1 thread = the serial backend, more = the parallel backend;\n"
             "every thread count builds the same tree. Training time excludes reading the data;\n"
             "peak memory is the peak RSS up to the end of training, the loaded data included.",
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.8, bottom=0.2)
    fig.savefig(out, dpi=150, facecolor="white")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--out", help="output file (default: <run>/chart.png)")
    args = parser.parse_args()
    run = Path(args.run)
    settings = json.load(open(run / "machine.json"))["settings"]
    cases, dataset = collect(run)
    protocols, series = summarise(cases)
    fields = ["protocol", "threads", "runs", "time_median", "time_min", "time_max", "speedup",
              "efficiency", "peak_rss_median", "nodes", "test_accuracy", "same_tree"]
    with open(run / "scaling.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for protocol in protocols:
            writer.writerows(series[protocol])
    print(run / "scaling.csv")
    print(figure(run, dataset, protocols, series, settings,
                 Path(args.out) if args.out else run / "chart.png"))


if __name__ == "__main__":
    main()
