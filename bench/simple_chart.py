#!/usr/bin/env python3
"""Chart of the "simple" benchmark (bench/results/cluster-simple): ./tree's
--simple reference backend against --serial and --parallel (one socket), and
scikit-learn and Weka J48 with their default options, on epyc1 and dell1.

One panel per dataset and protocol. Every bar is the median training time
(log scale), split into growing (dark) and pruning (light) where the tool
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

MACHINES = {"epyc": "epyc1, 1× EPYC 9255", "dell": "dell1, 1× Xeon Gold 6542Y"}
DATASETS = ("chess_1m", "covertype")
PROTOCOLS = {"cart_alpha_nodepth": "CART (gini, α = 1e-5)",
             "c45": "C4.5 (CF 0.25, ≥ 2 rows per leaf)"}
# (impl, single thread?) -> bar label, colour; in this order, top to bottom.
BARS = {("tree_simple", True): ("./tree --simple", "#8e5bd0"),
        ("tree", True): ("./tree --serial", "#2a78d6"),
        ("tree", False): ("./tree --parallel", "#eb6834"),
        ("sklearn", True): ("scikit-learn", "#1baf7a"),
        ("j48", True): ("Weka J48", "#eda100")}
TREE = {"tree", "tree_simple"}
STAND_IN = {"cart_alpha_nodepth": "scikit-learn: defaults (gini), so unpruned",
            "c45": "scikit-learn: defaults with entropy, unpruned (it has no C4.5)"}


def lighter(color, amount=0.55):
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)


def duration(seconds):
    if seconds < 0.0001:
        return "< 0.1 ms"  # ./tree prints tenths of a millisecond
    return f"{seconds * 1000:.3g} ms" if seconds < 1 else f"{seconds:.3g} s"


def load(root):
    """-> {(machine, dataset, protocol, impl, single): [ok run records]}."""
    cases = defaultdict(list)
    for path in sorted(root.glob("simple-*/results.jsonl")):
        machine = path.parent.name.split("-")[1]
        for line in open(path):
            r = json.loads(line)
            if r["kind"] != "run":
                continue
            if r["status"] != "ok":
                raise SystemExit(f"{path}: {r['impl']} {r['protocol']} {r['dataset']}: {r['status']}")
            cases[(machine, r["dataset"], r["protocol"], r["impl"], r["threads"] == 1)].append(r)
    return cases


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
    cases = load(args.root)
    check_same(cases)

    datasets = [d for d in DATASETS if any(key[1] == d for key in cases)]
    fig = plt.figure(figsize=(16, 7.6 * len(datasets)))
    # Per dataset a row of bar panels, and under each its table of the trees.
    grid = fig.add_gridspec(2 * len(datasets), len(PROTOCOLS), height_ratios=[5, 1.25] * len(datasets),
                            left=0.12, right=0.985, top=1 - 0.75 / fig.get_figheight(),
                            bottom=0.01, hspace=0.32, wspace=0.36)
    for row, dataset in enumerate(datasets):
        for column, (protocol, title) in enumerate(PROTOCOLS.items()):
            ax = fig.add_subplot(grid[2 * row, column])
            box = fig.add_subplot(grid[2 * row + 1, column])
            box.axis("off")
            labels, y, trees = [], 0, {}
            low = min(statistics.median(r["train_seconds"] for r in runs)
                      for key, runs in cases.items() if key[1:3] == (dataset, protocol)) / 3
            for machine, machine_name in MACHINES.items():
                ax.text(0, y - 0.25, machine_name, transform=ax.get_yaxis_transform(),
                        ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
                y += 0.55
                for (impl, single), (label, color) in BARS.items():
                    runs = cases.get((machine, dataset, protocol, impl, single))
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
                    ax.text(total * 1.08, y, text, va="center", fontsize=8.5, color=INK)
                    threads = "" if single else f", {runs[0]['threads']} threads"
                    labels.append((y, f"{label}{threads}"))
                    first = runs[0]
                    name = "./tree" if impl in TREE else label
                    trees.setdefault(name, (first["nodes"], first["depth"], first.get("train_accuracy")))
                    y += 1
                y += 0.6
            ax.set_xscale("log")
            high = max(statistics.median(r["train_seconds"] for r in runs)
                       for key, runs in cases.items() if key[1:3] == (dataset, protocol))
            ax.set_xlim(low, high * (high / low) ** 0.75)  # room for the end labels
            ax.set_xticks([10.0 ** e for e in range(-4, 5) if low <= 10.0 ** e <= high * 3])
            ax.xaxis.set_major_formatter(FuncFormatter(seconds_tick))
            ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=9)
            ax.set_ylim(y - 0.4, -0.75)  # top to bottom, room for the first machine's name
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{NAMES.get(dataset, dataset)}: {title}", fontsize=11, color=INK, loc="left")
            ax.set_xlabel("training time, median (log scale): dark = growing, light = pruning",
                          fontsize=8.5, color=MUTED)
            table = "\n".join(
                f"{name + (' (every backend: the same tree)' if name == './tree' else '')}: "
                f"{nodes:,} nodes, depth {depth}"
                + (f", train accuracy {accuracy:.2%}" if accuracy is not None else "")
                for name, (nodes, depth, accuracy) in trees.items())
            box.text(0, 1, table + "\n" + STAND_IN[protocol], transform=box.transAxes,
                     va="top", fontsize=8.5, color=INK, linespacing=1.5,
                    bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f5f2", "edgecolor": GRID})
    fig.suptitle("./tree --simple (the reference) against --serial, --parallel on one socket, "
                 "scikit-learn and Weka J48 (default options)", fontsize=12.5, color=INK, x=0.02,
                 y=1 - 0.25 / fig.get_figheight(), ha="left")
    out = args.out or args.root / "chart.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
