#!/usr/bin/env python3
"""How much ./tree's times vary from run to run: one row per machine and phase
(train total, presort, build), the bar at the median of the runs, a whisker
from the fastest to the slowest run and a dot for every run. One panel per
protocol. Colour follows the machine; each bar is labelled with its median and
min–max. Meant for runs with many repetitions (submit.sh -m 10).

Refuses to draw when the runs did not read the same data, as machines_chart.py.

  d=bench/results/cluster-chess/dell-variance
  bench/.venv/bin/python bench/runs_chart.py --protocols cart_full_nodepth,c45_unpruned \\
      $d/cpu-dell-chess_1b $d/cpu-dell512-chess_1b --out $d/chart.png
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import benchlib as B  # noqa: E402
import machines_chart as M  # noqa: E402
from chart import GRID, INK, MUTED, NAMES  # noqa: E402

# Categorical slots of the reference palette, in order, one per machine.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"]
PHASES = (("train_seconds", "train total"), ("presort_seconds", "presort"),
          ("build_seconds", "build"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--protocols", default="cart_full_nodepth,c45_unpruned")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocols = args.protocols.split(",")
    M.PROTOCOLS = {p: M.TITLES[p] for p in protocols}
    dataset, _ = M.check_same(args.runs)

    runs = []
    for i, run in enumerate(args.runs):
        rows = [json.loads(line) for line in open(run / "results.jsonl")]
        info = B.hardware(json.load(open(run / "machine.json")))
        runs.append({"name": M.where(run), "info": info, "color": COLORS[i % len(COLORS)],
                     "rows": [r for r in rows if r["kind"] == "run" and r["status"] == "ok"
                              and r["impl"] == "tree"]})

    rows_per_panel = len(runs) * len(PHASES)
    fig, axes = plt.subplots(len(protocols), 1, squeeze=False,
                             figsize=(11, 1.9 + len(protocols) * (0.75 + 0.42 * rows_per_panel)))
    top = 1 - 1.55 / fig.get_figheight()
    fig.subplots_adjust(left=0.2, right=0.97, top=top, bottom=0.55 / fig.get_figheight(),
                        hspace=0.45)
    reps = set()
    for ax, protocol in zip(axes[:, 0], protocols):
        y = 0
        ticks, labels = [], []
        longest = max(r[f] for run in runs for r in run["rows"] if r["protocol"] == protocol
                      for f, _ in PHASES)
        for phase, (field, phase_name) in enumerate(PHASES):
            for run in runs:
                times = [r[field] for r in run["rows"] if r["protocol"] == protocol]
                reps.add(len(times))
                median, low, high = statistics.median(times), min(times), max(times)
                strong = phase == 0
                ax.barh(y, median, height=0.62, color=run["color"],
                        alpha=1 if strong else 0.55, zorder=2)
                ax.plot([low, high], [y, y], color=INK, linewidth=1.3, zorder=3)
                for end in (low, high):
                    ax.plot([end, end], [y - 0.2, y + 0.2], color=INK, linewidth=1.3, zorder=3)
                # The runs, spread a little vertically in run order so equal times stay visible.
                n = len(times)
                ax.scatter(times, [y - 0.22 + 0.44 * k / max(n - 1, 1) for k in range(n)],
                           s=14, color="white", edgecolor=INK, linewidth=0.8, zorder=4)
                ax.text(high + longest * 0.012, y,
                        f"{median:.2f} s  (min {low:.2f}, max {high:.2f})",
                        va="center", fontsize=8.8, color=INK,
                        fontweight="bold" if strong else "normal")
                ticks.append(y)
                labels.append(f"{run['name']} · {phase_name}")
                y += 1
            y += 0.5
        ax.set_yticks(ticks, labels, fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, longest * 1.42)
        ax.set_title(M.TITLES[protocol], loc="left", fontsize=11, color=INK,
                     fontweight="bold")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g} s")

    name = NAMES.get(dataset, dataset)
    fig.text(0.02, 1 - 0.3 / fig.get_figheight(),
             f"./tree on {name}: time of each run, "
             f"{'/'.join(map(str, sorted(reps)))} runs per case",
             fontsize=12, fontweight="bold", color=INK)
    for i, run in enumerate(runs):
        info = run["info"]
        fig.text(0.02, 1 - (0.55 + 0.19 * i) / fig.get_figheight(),
                 f"{run['name']}: CPU {info['cpu']} · RAM {info['ram']}",
                 fontsize=8.8, color=MUTED)
    fig.legend(handles=[Patch(color=run["color"], label=run["name"]) for run in runs]
               + [Patch(color="white", label="bar: median · whisker: min to max · dots: every run;"
                                              " darker bar: train total = presort + build")],
               loc="upper left", bbox_to_anchor=(0.012, 1 - (0.5 + 0.19 * len(runs)) / fig.get_figheight()),
               frameon=False, fontsize=8.8, ncols=len(runs) + 1, handlelength=1.4)
    fig.savefig(args.out, dpi=160)
    print(args.out)


if __name__ == "__main__":
    main()
