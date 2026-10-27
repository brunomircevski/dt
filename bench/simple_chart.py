#!/usr/bin/env python3
"""Chart of bench/results/cluster-cpu-vs-others: ./tree's --simple reference
backend against --serial, --parallel (on one socket and on both), scikit-learn
and Weka J48, every tree grown to depth <= 10 without pruning, on epyc1 and
epyc2 (the same hardware), one run per case.

One PNG, a 2x2 grid with one panel per dataset; CART and C4.5 one under the
other in each panel. Every bar is the training time (log scale), its end label
the time and the peak memory. The trees are in one table under each panel:
every ./tree backend must grow the same tree (the same --dump, byte for byte:
run.py --hash-trees), and the script stops if they do not.

  bench/.venv/bin/python bench/simple_chart.py bench/results/cluster-cpu-vs-others [--out chart.png]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chart import GRID, INK, MUTED, NAMES, seconds_tick  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATASETS = ("chess_10m", "covertype", "susy", "higgs")
PROTOCOLS = {"cart_depth10": "CART: gini, depth ≤ 10, no pruning",
             "c45_depth10_unpruned": "C4.5: gain ratio, ≥ 2 rows per leaf, depth ≤ 10, no pruning"}
# (impl, threads) -> bar label, colour; in this order, top to bottom.
BARS = {("tree_simple", 1): ("./tree --simple", "#8e5bd0"),
        ("tree", 1): ("./tree --serial", "#2a78d6"),
        ("tree", 24): ("./tree --parallel, 24 threads (one socket)", "#eb6834"),
        ("tree", 48): ("./tree --parallel, 48 threads (both sockets)", "#b8401a"),
        ("sklearn", 1): ("scikit-learn", "#1baf7a"),
        ("j48", 1): ("Weka J48", "#eda100")}
TREE = {"tree", "tree_simple"}
SHORT = {"cart_depth10": "CART", "c45_depth10_unpruned": "C4.5"}
SKLEARN = {"cart_depth10": "gini", "c45_depth10_unpruned": "entropy, as it has no C4.5"}


def duration(seconds):
    if seconds < 0.0001:
        return "< 0.1 ms"  # ./tree prints tenths of a millisecond
    if seconds >= 1000:
        return f"{seconds:,.0f} s"
    return f"{seconds * 1000:.3g} ms" if seconds < 1 else f"{seconds:.3g} s"


def memory(nbytes):
    gib = nbytes / 2 ** 30
    return f"{nbytes / 2 ** 20:.0f} MiB" if gib < 1 else f"{gib:.3g} GiB"


def load(root):
    """-> {(dataset, protocol, impl, threads): the run's record}; one run per case."""
    cases = {}
    for path in sorted(root.glob("vs-*/results.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r["kind"] != "run":
                continue
            if r["status"] != "ok":
                raise SystemExit(f"{path}: {r['impl']} {r['protocol']} {r['dataset']}: {r['status']}")
            key = (r["dataset"], r["protocol"], r["impl"], r["threads"])
            if key in cases:
                raise SystemExit(f"{path}: {key} ran twice")
            cases[key] = r
    return cases


def check_same(cases):
    """Every ./tree backend: one tree per dataset and protocol."""
    hashes = defaultdict(set)
    for (dataset, protocol, impl, _), r in cases.items():
        if impl in TREE:
            hashes[(dataset, protocol)].add(r["tree_sha256"])
    for key, found in sorted(hashes.items()):
        if len(found) != 1:
            raise SystemExit(f"{key}: the ./tree backends grew {len(found)} different trees")
        print(f"{key[0]} {key[1]}: one tree, sha256 {next(iter(found))[:16]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = load(args.root)
    check_same(cases)

    datasets = [d for d in DATASETS if any(key[0] == d for key in cases)]
    columns = 1 if len(datasets) <= 2 else 2
    rows = -(-len(datasets) // columns)
    fig = plt.figure(figsize=(12 * columns, 8.6 * rows))
    # Per dataset a panel of bars, and under it its table of the trees.
    grid = fig.add_gridspec(2 * rows, columns, height_ratios=[6, 1.6] * rows,
                            left=0.27 / columns, right=0.985, top=1 - 1.25 / fig.get_figheight(),
                            bottom=0.01, hspace=0.3, wspace=0.32)
    for index, dataset in enumerate(datasets):
        row, column = divmod(index, columns)
        ax = fig.add_subplot(grid[2 * row, column])
        box = fig.add_subplot(grid[2 * row + 1, column])
        box.axis("off")
        times = [r["train_seconds"] for key, r in cases.items() if key[0] == dataset]
        low, high = min(times) / 3, max(times)
        labels, y, lines = [], 0, []
        for protocol, title in PROTOCOLS.items():
            ax.text(0, y - 0.25, title, transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)
            y += 0.55
            trees = {}
            for (impl, threads), (label, color) in BARS.items():
                r = cases.get((dataset, protocol, impl, threads))
                if not r:
                    continue
                total = r["train_seconds"]
                ax.barh(y, total - low, left=low, color=color, height=0.72)
                ax.text(total * 1.08, y, f"{duration(total)}   {memory(r['peak_rss_train_bytes'])}",
                        va="center", fontsize=9, color=INK)
                labels.append((y, label))
                name = "./tree (every backend)" if impl in TREE else label
                if impl == "sklearn":
                    name += f" ({SKLEARN[protocol]})"
                trees.setdefault(name, (r["nodes"], r["depth"], r["test_accuracy"]))
                y += 1
            y += 0.6
            lines.append(f"{SHORT[protocol]}:  " + ";  ".join(
                f"{name} {nodes:,} nodes, depth {depth}, test acc. {accuracy:.2%}"
                for name, (nodes, depth, accuracy) in trees.items()))
        ax.set_xscale("log")
        ax.set_xlim(low, high * (high / low) ** 0.3)  # room for the end labels
        ax.set_xticks([10.0 ** e for e in range(-4, 5) if low <= 10.0 ** e <= high * 3])
        ax.xaxis.set_major_formatter(FuncFormatter(seconds_tick))
        ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=9.5)
        ax.set_ylim(y - 0.4, -0.75)  # top to bottom, room for the first group's name
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        first = next(r for key, r in cases.items() if key[0] == dataset)
        ax.set_title(f"{NAMES.get(dataset, dataset)}   ({first['n_train_loaded']:,} training rows, "
                     f"{first['n_features_loaded']} features)", fontsize=12, color=INK, loc="left",
                     fontweight="bold")
        ax.set_xlabel("training time, 1 run (log scale); label: time, peak memory (RSS)",
                      fontsize=9, color=MUTED)
        # One tree line per protocol and tool.
        text = "\n".join(line.replace(";  ", ";\n      ") for line in lines)
        box.text(0, 0.95, text, transform=box.transAxes, va="top", fontsize=9, color=INK,
                 linespacing=1.5,
                 bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f5f2", "edgecolor": GRID})
    fig.suptitle("./tree --simple (the reference) against --serial, --parallel, scikit-learn and "
                 "Weka J48: depth ≤ 10, no pruning, accuracy on the test set\non epyc1 and epyc2 "
                 "(2× EPYC 9255 each, one job per tool and algorithm); every run on one socket, "
                 "except --parallel on both sockets",
                 fontsize=12.5, color=INK, x=0.02, y=1 - 0.2 / fig.get_figheight(), ha="left",
                 va="top")
    out = args.out or args.root / "chart.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
