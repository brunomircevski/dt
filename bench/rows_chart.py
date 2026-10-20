#!/usr/bin/env python3
"""How ./tree scales with the number of rows: one line per machine and backend
through runs of the same data at several sizes (e.g. cluster-chess's 1M to
1000M rows), on log-log axes. One row of panels per protocol: training time,
peak RAM (host) and peak VRAM (GPU, CUDA runs only). Colour follows the
backend as in machines_chart.py; each machine has its own marker and is named
at the end of its line. A dotted grey line marks time proportional to the
rows, from the fastest machine's smallest run.

Runs are named <backend>-<machine>-<dataset> (submit.sh). Every dataset's runs
must have read the same data files and built the same trees, as
machines_chart.py checks.

  d=bench/results/cluster-chess
  bench/.venv/bin/python bench/rows_chart.py --protocols cart_full_nodepth,c45_unpruned \\
      $d/{cuda-hopper,cuda-ampere,cpu-epyc,cpu-dell,cpu-dell512}-chess_* --out $d/chart-rows.png
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

import benchlib as B  # noqa: E402
import machines_chart as M  # noqa: E402
from chart import GRID, INK, MUTED, NAMES  # noqa: E402

MARKERS = ["o", "s", "^", "D", "v"]  # per machine of a backend, in the order given


def rows_label(value, _=None):
    for size, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= size:
            return f"{value / size:.3g}{suffix}"
    return f"{value:g}"


def seconds_label(value, _=None):
    return f"{value:g} s" if value >= 1 else f"{value * 1e3:g} ms"


def gib_label(value, _=None):
    return f"{value:g} GiB"


def value_text(field, value):
    """A point's value for its direct label, to 3 significant digits."""
    if field == "time":
        return f"{value:.3g} s" if value >= 1 else f"{value * 1e3:.3g} ms"
    return f"{value:.3g} GiB" if value >= 1 else f"{value * 1024:.3g} MiB"


def spread(values, gap):
    """Label positions (log10) near `values`, at least `gap` apart, in order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = {}
    previous = None
    for i in order:
        y = values[i] if previous is None else max(values[i], previous + gap)
        placed[i] = previous = y
    # Centre the block that moved on the values it labels.
    shift = (sum(placed.values()) - sum(values)) / len(values)
    return [placed[i] - shift for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--protocols", default="cart_full_nodepth,c45_unpruned",
                        help=f"one row of panels each, of: {', '.join(M.TITLES)}")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocols = args.protocols.split(",")
    M.PROTOCOLS.clear()
    M.PROTOCOLS.update((p, M.TITLES[p]) for p in protocols)

    # Runs by dataset; each dataset checked as machines_chart does.
    by_dataset = defaultdict(list)
    for run in args.runs:
        by_dataset[run.name.split("-", 2)[2]].append(run)
    meta, entries = {}, []
    for dataset, runs in by_dataset.items():
        name, meta[dataset] = M.check_same(runs)
        assert name == dataset, (name, dataset)
        mine = [e for run in runs for e in M.bars(run, False)]
        for protocol in protocols:
            trees = {e["tree"][:2] for e in mine if e["protocol"] == protocol}
            if len(trees) > 1:
                raise SystemExit(f"{dataset} {protocol}: different trees {sorted(trees)}")
        for e in mine:
            e["rows"] = meta[dataset]["n_train"]
            e["machine"] = M.where(e["run"])
        entries += mine
    lines = defaultdict(list)  # (backend, machine) -> entries in row order
    for e in sorted(entries, key=lambda e: e["rows"]):
        lines[(e["backend"], e["machine"], e["protocol"])].append(e)
    order = list(dict.fromkeys(M.where(run) for run in args.runs))
    keys = sorted({(b, m) for b, m, _ in lines},
                  key=lambda k: (list(M.BACKENDS).index(k[0]), order.index(k[1])))
    marker = {}
    for backend in M.BACKENDS:
        for index, key in enumerate(k for k in keys if k[0] == backend):
            marker[key] = MARKERS[index % len(MARKERS)]

    columns = [("time", "training time", seconds_label),
               ("ram", "peak RAM (host)", gib_label),
               ("vram", "peak VRAM (GPU 0)", gib_label)]
    fig, axes = plt.subplots(len(protocols), 3, figsize=(16, 1.9 + 4.4 * len(protocols)),
                             squeeze=False, gridspec_kw={"wspace": 0.42, "hspace": 0.75})
    all_rows = sorted({e["rows"] for e in entries})
    headers = []  # (protocol title, its trees) above each row of panels
    for row, protocol in enumerate(protocols):
        for col, (field, title, label) in enumerate(columns):
            ax = axes[row][col]
            ax.set_xscale("log")
            ax.set_yscale("log")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
            ax.grid(which="major", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xticks(all_rows)
            ax.xaxis.set_major_formatter(FuncFormatter(rows_label))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(label))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("training rows", fontsize=8.5, color=MUTED)
            ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=8)
            ends = []  # (key, last row count, last value) for the direct labels
            for key in keys:
                points = [e for e in lines.get((*key, protocol), []) if e[field] is not None]
                if not points:
                    continue
                color = M.COLORS[key[0]]
                ax.plot([e["rows"] for e in points], [e[field] for e in points], color=color,
                        linewidth=2, marker=marker[key], markersize=8, markeredgecolor="white",
                        markeredgewidth=1.5, zorder=3)
                ends.append((key, points[-1]["rows"], points[-1][field]))
            if not ends:
                ax.text(0.5, 0.5, "no GPU runs", transform=ax.transAxes, ha="center",
                        color=MUTED, fontsize=9)
                continue
            if field == "time":
                # Proportional to the rows, from the fastest smallest run.
                first = min(all_rows)
                start = min(e["time"] for e in entries
                            if e["protocol"] == protocol and e["rows"] == first)
                ax.plot(all_rows, [start * r / first for r in all_rows], color=MUTED,
                        linewidth=1, linestyle=":", zorder=2)
            # At least a factor 4 between the axis ends: a flat line (VRAM) is drawn flat.
            low, high = ax.get_ylim()
            if high / low < 4:
                middle = math.sqrt(low * high)
                low, high = middle / 2, middle * 2
                ax.set_ylim(low, high)
            # Lines ending on the same value share one label, when there are too many
            # labels to spread (a merged label is wider: few labels stay apart).
            merged = {}
            for key, x, value in (ends if len(ends) > 4 else []):
                text = value_text(field, value)
                if (x, text) in merged:
                    merged[(x, text)][0].append(key[1])
                else:
                    merged[(x, text)] = ([key[1]], value)
            ends = ([((", ".join(dict.fromkeys(names)),), x, value)
                     for (x, _), (names, value) in merged.items()] if merged
                    else [((key[1],), x, value) for key, x, value in ends])
            # Direct labels right of the last points, spread so they do not overlap.
            span = math.log10(high) - math.log10(low)
            ys = spread([math.log10(v) for _, _, v in ends], gap=span * 0.075)
            ax.set_xlim(right=max(all_rows) * 3.2)
            for (key, x, value), y in zip(ends, ys):
                ax.annotate(f"{key[0]} {value_text(field, value)}",
                            (x, value), xytext=(x * 1.25, 10 ** y), fontsize=8.3, color=INK,
                            va="center", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8,
                                            shrinkA=0, shrinkB=3))
        trees = []
        for dataset in sorted(meta, key=lambda d: meta[d]["n_train"]):
            tree = next((e["tree"] for e in entries
                         if e["protocol"] == protocol and e["rows"] == meta[dataset]["n_train"]),
                        None)
            if tree:
                trees.append(f"{rows_label(meta[dataset]['n_train'])}: {tree[0]:,} nodes, "
                             f"depth {tree[1]}, {100 * tree[2]:.2f}%")
        headers.append((M.TITLES[protocol], "the same tree on every machine and backend: "
                        + " · ".join(trees) + f" ({entries[0]['tree'][3]} accuracy)"))

    datasets = sorted(meta, key=lambda d: meta[d]["n_train"])
    features = {len(meta[d]["features"]) for d in datasets}
    family = NAMES.get(datasets[0], datasets[0]).split(" (")[0]
    fig.suptitle(f"./tree on {family}: training time and peak memory against the rows "
                 f"({rows_label(meta[datasets[0]]['n_train'])} to "
                 f"{rows_label(meta[datasets[-1]]['n_train'])} training rows, "
                 f"{'/'.join(map(str, sorted(features)))} features)",
                 x=0.04, y=0.985, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    # Machines: one line each, as in machines_chart's header.
    machines = {}
    for run in args.runs:
        machines.setdefault(M.where(run), run)
    inches = fig.get_figheight()
    for index, (name, run) in enumerate(machines.items()):
        info = json.load(open(run / "machine.json"))
        used_gpu = run.name.startswith("cuda-")
        fig.text(0.04, 0.985 - (0.36 + 0.2 * index) / inches, name.capitalize(), fontsize=9,
                 color=INK, fontweight="bold", va="top")
        fig.text(0.1, 0.985 - (0.36 + 0.2 * index) / inches,
                 B.hardware_line(info, used_gpu) + f"   (run {info['started'][:10]})",
                 fontsize=8.8, color=INK, va="top")
    handles = [Line2D([], [], color=M.COLORS[b], marker=marker[k], markersize=8,
                      markeredgecolor="white", linewidth=2,
                      label=f"{M.BACKENDS[b]} · {m}") for k in keys for b, m in [k]]
    handles.append(Line2D([], [], color=MUTED, linestyle=":", linewidth=1,
                          label="time ∝ rows"))
    header = 0.36 + 0.2 * len(machines) + 0.1
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               fontsize=8.8, bbox_to_anchor=(0.035, 0.985 - header / inches),
               handletextpad=0.4, columnspacing=1.4)
    reps = {json.load(open(run / "machine.json"))["settings"]["reps"] for run in args.runs}
    fig.text(0.04, 0.12 / inches,
             ("Points: median over the runs of each case. " if reps != {1} else
              "Points: one run per case. ")
             + "Time: ./tree's 'train total' (presort, build, prune; CUDA also upload and "
             "device sort), not reading the CSV. Log-log axes.\nRAM: peak RSS of the process "
             "up to the end of training. VRAM: peak GPU memory during the process above idle, "
             "CUDA context included (nvidia-smi, every 10 ms).",
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.985 - (header + 1.25) / inches,
                        bottom=1.0 / inches)
    for row, (title, trees) in enumerate(headers):
        box = axes[row][0].get_position()
        fig.text(box.x0, box.y1 + 0.62 / inches, title, fontsize=11.5, color=INK,
                 fontweight="bold", va="bottom")
        fig.text(box.x0, box.y1 + 0.42 / inches, trees, fontsize=8, color=MUTED, va="bottom")
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
