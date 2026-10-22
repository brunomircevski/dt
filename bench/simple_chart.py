#!/usr/bin/env python3
"""Chart of the "simple" benchmark (bench/results/cluster-simple): ./tree's
--simple reference backend against --serial, --parallel (on one socket and on
both), scikit-learn and Weka J48, on epyc1 and epyc2 (the same hardware).

One panel per dataset (one column for two datasets, a 2×2 grid for three or
four), CART and C4.5 in it, one under the other. Every bar is the median training time
(log scale; 3 runs, or 1 for SUSY and HIGGS), split into growing (dark) and pruning (light) where the tool
prunes; the end label gives both. The trees are in one table under each
panel, not next to the bars: every ./tree backend must grow the same tree
(the same --dump, byte for byte: run.py --hash-trees), and the script stops
if they do not.

  bench/.venv/bin/python bench/simple_chart.py bench/results/cluster-simple [--out chart.png]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chart import GRID, INK, MUTED, NAMES, seconds_tick  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

MACHINE = "epyc"
DATASETS = ("chess_10m", "covertype", "susy", "higgs")
PROTOCOLS = {"cart_alpha_nodepth": "CART: gini, pruned at α = 1e-5",
             "c45": "C4.5: CF 0.25, ≥ 2 rows per leaf"}
# (impl, threads) -> bar label, colour; in this order, top to bottom.
BARS = {("tree_simple", 1): ("./tree --simple", "#8e5bd0"),
        ("tree", 1): ("./tree --serial", "#2a78d6"),
        ("tree", 24): ("./tree --parallel, 24 threads (one socket)", "#eb6834"),
        ("tree", 48): ("./tree --parallel, 48 threads (both sockets)", "#b8401a"),
        ("sklearn", 1): ("scikit-learn", "#1baf7a"),
        ("j48", 1): ("Weka J48", "#eda100")}
TREE = {"tree", "tree_simple"}
SHORT = {"cart_alpha_nodepth": "CART", "c45": "C4.5"}
SKLEARN = {"cart_alpha_nodepth": "gini", "c45": "entropy, as it has no C4.5"}


def lighter(color, amount=0.55):
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)


def duration(seconds):
    if seconds < 0.0001:
        return "< 0.1 ms"  # ./tree prints tenths of a millisecond
    if seconds >= 1000:
        return f"{seconds:,.0f} s"
    return f"{seconds * 1000:.3g} ms" if seconds < 1 else f"{seconds:.3g} s"


def load(root):
    """-> ({(machine, dataset, protocol, impl, threads): [ok run records]},
    {same key: seconds a run timed out after})."""
    cases, timeouts = defaultdict(list), {}
    for path in sorted(root.glob("simple-*/results.jsonl")):
        machine = path.parent.name.split("-")[1]
        for line in open(path):
            r = json.loads(line)
            if r["kind"] != "run":
                continue
            key = (machine, r["dataset"], r["protocol"], r["impl"], r["threads"])
            if r["status"] == "timeout":
                timeouts[key] = r["wall_seconds"]
                continue
            if r["status"] != "ok":
                raise SystemExit(f"{path}: {r['impl']} {r['protocol']} {r['dataset']}: {r['status']}")
            cases[key].append(r)
    return cases, timeouts


def check_same(cases):
    """Every ./tree backend on every machine: one tree per dataset and protocol."""
    hashes = defaultdict(set)
    for (machine, dataset, protocol, impl, _), runs in cases.items():
        if impl in TREE:
            for r in runs:
                hashes[(dataset, protocol)].add(r["tree_sha256"])
    for key, found in sorted(hashes.items()):
        if len(found) != 1:
            raise SystemExit(f"{key}: the ./tree backends grew {len(found)} different trees")
        print(f"{key[0]} {key[1]}: one tree, sha256 {next(iter(found))[:16]}")


def parts(runs, impl):
    """-> (growing, pruning or None): medians over the runs."""
    total = statistics.median(r["train_seconds"] for r in runs)
    if "prune_seconds" not in runs[0]:
        return total, None
    prune = statistics.median(r["prune_seconds"] for r in runs)
    return total - prune, prune


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases, timeouts = load(args.root)
    check_same(cases)

    datasets = [d for d in DATASETS if any(key[:2] == (MACHINE, d) for key in cases)]
    columns = 1 if len(datasets) <= 2 else 2
    rows = -(-len(datasets) // columns)
    fig = plt.figure(figsize=(12 * columns, 9.2 * rows))
    # Per dataset a panel of bars, and under it its table of the trees.
    grid = fig.add_gridspec(2 * rows, columns, height_ratios=[6, 1.6] * rows,
                            left=0.27 / columns, right=0.985, top=1 - 1.25 / fig.get_figheight(),
                            bottom=0.01, hspace=0.3, wspace=0.32)
    for index, dataset in enumerate(datasets):
        row, column = divmod(index, columns)
        ax = fig.add_subplot(grid[2 * row, column])
        box = fig.add_subplot(grid[2 * row + 1, column])
        box.axis("off")
        medians = [statistics.median(r["train_seconds"] for r in runs)
                   for key, runs in cases.items() if key[:2] == (MACHINE, dataset)]
        medians += [s for key, s in timeouts.items() if key[:2] == (MACHINE, dataset)]
        low, high = min(medians) / 3, max(medians)
        labels, y, lines = [], 0, []
        for protocol, title in PROTOCOLS.items():
            ax.text(0, y - 0.25, title, transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)
            y += 0.55
            trees = {}
            for (impl, threads), (label, color) in BARS.items():
                runs = cases.get((MACHINE, dataset, protocol, impl, threads))
                stopped = timeouts.get((MACHINE, dataset, protocol, impl, threads))
                if not runs and stopped:
                    ax.barh(y, stopped - low, left=low, color="none", edgecolor=color,
                            hatch="///", height=0.72)
                    ax.text(stopped * 1.08, y, f"timed out after {stopped / 3600:.0f} h, no tree",
                            va="center", fontsize=9, color=INK)
                    labels.append((y, label))
                    y += 1
                    continue
                if not runs:
                    continue
                grow, prune = parts(runs, impl)
                total = grow + (prune or 0)
                # On the log axis each part's length is its share of the bar.
                split = low * (total / low) ** (grow / total) if prune else total
                ax.barh(y, split - low, left=low, color=color, height=0.72)
                if prune:
                    ax.barh(y, total - split, left=split, color=lighter(color), height=0.72)
                text = duration(total) + (f"  (grow {duration(grow)} + prune {duration(prune)})"
                                          if prune is not None else "  (no pruning)")
                ax.text(total * 1.08, y, text, va="center", fontsize=9, color=INK)
                labels.append((y, label))
                first = runs[0]
                name = "./tree (every backend)" if impl in TREE else label
                if impl == "sklearn":
                    name += f" ({SKLEARN[protocol]})"
                trees.setdefault(name, (first["nodes"], first["depth"], first.get("train_accuracy")))
                y += 1
            y += 0.6
            lines.append(f"{SHORT[protocol]}:  " + ";  ".join(
                f"{name} {nodes:,} nodes, depth {depth}"
                + (f", train acc. {accuracy:.2%}" if accuracy is not None else "")
                for name, (nodes, depth, accuracy) in trees.items()))
        ax.set_xscale("log")
        ax.set_xlim(low, high * (high / low) ** 0.45)  # room for the end labels
        ax.set_xticks([10.0 ** e for e in range(-4, 5) if low <= 10.0 ** e <= high * 3])
        ax.xaxis.set_major_formatter(FuncFormatter(seconds_tick))
        ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=9.5)
        ax.set_ylim(y - 0.4, -0.75)  # top to bottom, room for the first group's name
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(NAMES.get(dataset, dataset), fontsize=12, color=INK, loc="left",
                     fontweight="bold")
        reps = {len(runs) for key, runs in cases.items() if key[:2] == (MACHINE, dataset)}
        runs = "median of 3" if reps == {3} else "1 run" if reps == {1} else "median of up to 3 runs"
        ax.set_xlabel(f"training time, {runs} (log scale): dark = growing, light = pruning",
                      fontsize=9, color=MUTED)
        # One tree line per protocol and tool.
        text = "\n".join(line.replace(";  ", ";\n      ") for line in lines)
        box.text(0, 0.95, text, transform=box.transAxes, va="top", fontsize=9, color=INK,
                 linespacing=1.5,
                 bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f5f2", "edgecolor": GRID})
    fig.suptitle("./tree --simple (the reference) against --serial, --parallel, scikit-learn and "
                 "Weka J48, every tree pruned\non epyc1 and epyc2 (2× EPYC 9255 each); every run on "
                 "one socket, except --parallel on both sockets",
                 fontsize=12.5, color=INK, x=0.02, y=1 - 0.2 / fig.get_figheight(), ha="left",
                 va="top")
    out = args.out or args.root / "chart.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
