#!/usr/bin/env python3
"""Chart of one benchmark run, one figure per dataset: training time (median,
line = min-max over the runs, log scale) and peak RSS up to the end of
training (median of the same runs) per implementation. Single-thread and multi-thread cases are in
separate rows of panels, never in one comparison. Each row is labelled with the
tool and the tree it built (nodes, depth, test accuracy).

  bench/.venv/bin/python bench/chart.py bench/results/<run-id> [--out chart.png]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

import benchlib as B  # noqa: E402

# Categorical slots 1-5 of the validated reference palette, in its fixed order
# (bench/README.md: colour follows the implementation in every panel).
COLORS = {"tree": "#2a78d6", "sklearn": "#eb6834", "rpart": "#1baf7a", "j48": "#eda100",
          "yadt": "#e87ba4", "tree_cuda": "#008300"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"susy": "SUSY", "higgs": "HIGGS", "covertype": "Covertype", "diabetes": "Diabetes"}
GROUP = {"cart_alpha": "CART, fixed α", "cart_full": "CART, unpruned",
         "cart_depth12": "CART, depth 12", "cart_alpha_nodepth": "CART, fixed α, no depth limit", "c45": "C4.5, CF 0.25", "c45_unpruned": "C4.5, unpruned"}


def seconds_tick(value, _=None):
    return f"{value:g} s" if value >= 1 else f"{value * 1e3:g} ms"


def seconds_label(value):
    return f"{value:.3g} s" if value >= 1 else f"{value * 1e3:.3g} ms"


def collect(run):
    rows = [json.loads(line) for line in open(run / "results.jsonl")]
    times, memory, tree, status = defaultdict(list), defaultdict(list), {}, {}
    for r in rows:
        key = (r["dataset"], r["protocol"], r["impl"], r["threads"])
        if r["kind"] == "baseline":
            continue
        if r["status"] != "ok":
            status.setdefault(key, r["status"])
            continue
        if r["kind"] == "run":
            times[key].append(r["train_seconds"])
            memory[key].append(r["peak_rss_train_bytes"])
            tree.setdefault(key, r)
    return times, memory, tree, status


def entries(keys, protocols):
    """Rows of one panel, top to bottom: (y, key) with a gap row before each group."""
    order = list(B.IMPLS)
    keys = sorted(keys, key=lambda k: (protocols.index(k[1]), order.index(k[2])))
    out, groups, y, previous = [], [], 0.0, None
    for key in keys:
        if key[1] != previous:
            if previous is not None:
                y += 0.5
            groups.append((y, GROUP.get(key[1], key[1])))
            y += 0.75
            previous = key[1]
        out.append((y, key))
        y += 1
    return out, groups, y


def figure(run, dataset, times, memory, tree, status, meta, settings, out):
    keys = {k for k in set(times) | set(status) if k[0] == dataset}
    protocols = [p for p in B.PROTOCOLS if any(k[1] == p for k in keys)]
    thread_rows = sorted({k[3] for k in keys})
    layouts = [entries([k for k in keys if k[3] == t], protocols) for t in thread_rows]
    fig, axes = plt.subplots(
        len(thread_rows), 2, figsize=(12.5, 1.6 + 0.62 * sum(l[2] for l in layouts)),
        gridspec_kw={"width_ratios": [1.45, 1], "height_ratios": [l[2] for l in layouts],
                     "wspace": 0.14, "hspace": 0.32}, squeeze=False)
    every_time = [t for k in keys for t in times.get(k, [])]
    every_rss = [m for k in keys for m in memory.get(k, [])]
    time_limits = (min(every_time) / 2.5, max(every_time) * 4) if every_time else (0.1, 10)
    rss_limit = max(every_rss) / 2**30 * 1.28 if every_rss else 1

    for row, (threads, (rows, groups, height)) in enumerate(zip(thread_rows, layouts)):
        ax_time, ax_rss = axes[row]
        for ax in (ax_time, ax_rss):
            ax.set_ylim(height - 0.25, -0.35)
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="x", which="both", colors=MUTED, labelsize=8.5, length=0)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
        title = "1 thread" if threads == 1 else f"{threads} threads"
        ax_time.set_title(f"{title}: training time (log scale)", loc="left", fontsize=10.5,
                          color=INK, fontweight="bold")
        ax_rss.set_title(f"{title}: peak memory (RSS)", loc="left",
                         fontsize=10.5, color=INK, fontweight="bold")
        for y, text in groups:
            ax_time.text(-0.02, y, text, transform=ax_time.get_yaxis_transform(), ha="right",
                         va="center", fontsize=9.5, color=INK, fontweight="bold")
        for y, key in rows:
            _, protocol, impl, _ = key
            color, label = COLORS[impl], B.IMPLS[impl]["label"]
            info = tree.get(key)
            stats = (f"{info['nodes']:,} nodes · depth {info['depth']} · test "
                     f"{100 * info['test_accuracy']:.2f}%"
                     if info and info.get("test_accuracy") is not None else "")
            weight = "bold" if impl in ("tree", "tree_cuda") else "normal"
            ax_time.text(-0.02, y - 0.13, label, transform=ax_time.get_yaxis_transform(),
                         ha="right", va="center", fontsize=9.5, color=INK, fontweight=weight)
            ax_time.text(-0.02, y + 0.24, stats, transform=ax_time.get_yaxis_transform(),
                         ha="right", va="center", fontsize=7.8, color=MUTED)
            if key in times:
                values = times[key]
                mid = statistics.median(values)
                ax_time.plot([min(values), max(values)], [y, y], color=color, linewidth=2,
                             solid_capstyle="round", zorder=2)
                ax_time.scatter([mid], [y], s=70, color=color, edgecolor="white",
                                linewidth=1.5, zorder=3)
                ax_time.text(max(values) * 1.18, y, seconds_label(mid), va="center",
                             fontsize=9, color=INK)
            else:
                ax_time.text(time_limits[0] * 1.2, y, status.get(key, "no data"), va="center",
                             fontsize=9, color=MUTED, style="italic")
            if key in memory:
                gib = statistics.median(memory[key]) / 2**30
                ax_rss.barh(y, gib, height=0.42, color=color, zorder=2)
                ax_rss.text(gib + rss_limit * 0.012, y, f"{gib:.2f} GiB", va="center",
                            fontsize=9, color=INK)
            else:
                ax_rss.text(rss_limit * 0.01, y, status.get(key, "no data"), va="center",
                            fontsize=9, color=MUTED, style="italic")
        ax_time.set_xscale("log")
        ax_time.set_xlim(*time_limits)
        ax_time.xaxis.set_major_locator(LogLocator(base=10))
        ax_time.xaxis.set_major_formatter(FuncFormatter(seconds_tick))
        ax_time.xaxis.set_minor_formatter(NullFormatter())
        ax_rss.set_xlim(0, rss_limit)
        ax_rss.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} GiB"))

    shown = [impl for impl in B.IMPLS if any(k[2] == impl for k in keys)]
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=COLORS[impl],
                      label=B.IMPLS[impl]["label"]) for impl in shown]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.21, 0.962),
               ncol=len(shown), frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.4)
    reps = settings.get("reps")
    fig.suptitle(f"{NAMES.get(dataset, dataset)}: {meta['n_train']:,} training rows, {meta['n_test']:,} test rows, "
                 f"{len(meta['features'])} features", x=0.215, y=0.99, ha="left", fontsize=12.5,
                 color=INK, fontweight="bold")
    runs = f"median of {reps} runs; line: min–max" if reps and reps > 1 else "one run"
    fig.text(0.215, 0.012, f"Dot: {runs}. Training time excludes reading the data. Same data "
             "and the same CPUs for every tool.\nPeak RSS: highest resident memory of the same "
             "process up to the end of training (runtime, data, warm-up, training).",
             fontsize=8,
             color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.215, right=0.985, top=0.87, bottom=0.08)
    fig.savefig(out, dpi=150, facecolor="white")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--out", help="output file (default: <run>/chart.png, or "
                        "chart_<dataset>.png for several datasets)")
    args = parser.parse_args()
    run = Path(args.run)
    plan = json.load(open(run / "plan.json"))
    settings = json.load(open(run / "machine.json"))["settings"]
    times, memory, tree, status = collect(run)
    datasets = list(dict.fromkeys(k[0] for k in list(times) + list(status)))
    for dataset in datasets:
        out = Path(args.out) if args.out else (
            run / ("chart.png" if len(datasets) == 1 else f"chart_{dataset}.png"))
        print(figure(run, dataset, times, memory, tree, status, plan["datasets"][dataset],
                     settings, out))


if __name__ == "__main__":
    main()
