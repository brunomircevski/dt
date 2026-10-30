#!/usr/bin/env python3
"""Chart of bench/results/cluster-cpu-vs-others: ./tree's --simple reference
backend against --serial, --parallel (on one socket and on both), scikit-learn
and Weka J48, every tree grown to depth <= 10 without pruning, on epyc1 and
epyc2 (the same hardware).

./tree's runs: HIGGS, SUSY, ECBDL 15M and chess from the test-set benchmarks
(cluster-<dataset>/{simple,serial,socket,cpu}-epyc-<dataset>, chess in
cluster-chess), Covertype from the same-named folders in <root>; the
libraries' runs from <root>/vs-epyc-<dataset>-<tool>-<protocol>/ (3 when a run
takes under 10 minutes, else one).

One PNG, a grid with one panel per dataset; CART and C4.5 one under the other
in each panel. Every bar is the mean training time (log scale), its whisker
the fastest to the slowest run, its end label the mean ± standard deviation,
the number of runs and the peak memory. The trees are in one table under each
panel: every ./tree run must grow the same tree (nodes, depth, test accuracy,
and the --dump's SHA-256 where run.py --hash-trees recorded it), and the
script stops if they do not.

  bench/.venv/bin/python bench/simple_chart.py bench/results/cluster-cpu-vs-others [--out chart.png]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES, seconds_tick  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATASETS = ("chess_10m", "covertype", "susy", "higgs", "ecbdl_15m", "chess_1b")
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
# scikit-learn has no C4.5 (its entropy tree is another algorithm): CART only.
SKIP = {("c45_depth10_unpruned", "sklearn")}


def duration(seconds):
    if seconds < 0.0001:
        return "< 0.1 ms"  # ./tree prints tenths of a millisecond
    if seconds >= 1000:
        return f"{seconds:,.0f} s"
    return f"{seconds * 1000:.3g} ms" if seconds < 1 else f"{seconds:.3g} s"


def memory(nbytes):
    gib = nbytes / 2 ** 30
    return f"{nbytes / 2 ** 20:.0f} MiB" if gib < 1 else f"{gib:.3g} GiB"


# ./tree's runs of these datasets are the test-set benchmarks' (cluster-<name>/).
TESTSET = {"higgs": "cluster-higgs", "susy": "cluster-susy", "ecbdl_15m": "cluster-ecbdl",
           "chess_10m": "cluster-chess", "chess_1b": "cluster-chess"}
BACKENDS = ("simple", "serial", "socket", "cpu")


def load(root):
    """-> {(dataset, protocol, impl, threads): [run records]}: ./tree's runs and
    the libraries'. A case that did not finish ok keeps its records."""
    cases = defaultdict(list)
    paths = [p for p in sorted(root.glob("vs-*/results.jsonl")) if p.parent.name.split("-")[-2]
             in ("sklearn", "j48")]
    for dataset in DATASETS:
        folder = root.parent / TESTSET[dataset] if dataset in TESTSET else root
        paths += [folder / f"{b}-epyc-{dataset}" / "results.jsonl" for b in BACKENDS
                  if (folder / f"{b}-epyc-{dataset}" / "results.jsonl").exists()]
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r["kind"] == "run" and r["protocol"] in PROTOCOLS:
                cases[(r["dataset"], r["protocol"], r["impl"], r["threads"])].append(r)
    return cases


def check_same(cases):
    """Every ./tree run: one tree per dataset and protocol."""
    trees, hashes = defaultdict(set), defaultdict(set)
    for (dataset, protocol, impl, _), runs in cases.items():
        for r in runs:
            if impl in TREE and r["status"] == "ok":
                trees[(dataset, protocol)].add((r["nodes"], r["depth"], round(r["test_accuracy"], 6)))
                if r.get("tree_sha256"):
                    hashes[(dataset, protocol)].add(r["tree_sha256"])
    for key, found in sorted(trees.items()):
        if len(found) != 1 or len(hashes[key]) > 1:
            raise SystemExit(f"{key}: the ./tree runs grew different trees: {sorted(found)}")
        print(f"{key[0]} {key[1]}: one tree, {next(iter(found))}")


def ok(runs):
    return [r for r in runs if r["status"] == "ok"]


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
        times = [r["train_seconds"] for key, runs in cases.items()
                 if key[0] == dataset and (key[1], key[2]) not in SKIP for r in ok(runs)]
        low, high = min(times) / 3, max(times)
        labels, y, lines = [], 0, []
        for protocol, title in PROTOCOLS.items():
            ax.text(0, y - 0.25, title, transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)
            y += 0.55
            trees = {}
            for (impl, threads), (label, color) in BARS.items():
                if (protocol, impl) in SKIP or (protocol == "cart_depth10" and impl == "j48"):
                    continue
                runs = cases.get((dataset, protocol, impl, threads), [])
                good = ok(runs)
                if not good:  # still queued or running, or failed
                    note = f"failed: {B.failure(runs[0])}" if runs else "not finished yet"
                    ax.text(low * 1.08, y, note, va="center", fontsize=9, color=MUTED,
                            style="italic")
                    labels.append((y, label))
                    y += 1
                    continue
                r = good[0]
                spread = [x["train_seconds"] for x in good]
                total = statistics.mean(spread)
                ax.barh(y, total - low, left=low, color=color, height=0.72)
                end = duration(total)
                if len(good) > 1:
                    ax.hlines(y, min(spread), max(spread), color=INK, linewidth=1.2)
                    end += f" ± {duration(statistics.stdev(spread))} ({len(good)} runs)"
                else:
                    end += " (1 run)"
                rss = statistics.mean(x["peak_rss_train_bytes"] for x in good)
                ax.text(max(spread) * 1.08, y, f"{end}   {memory(rss)}",
                        va="center", fontsize=9, color=INK)
                labels.append((y, label))
                name = "./tree (every backend)" if impl in TREE else label
                if impl == "sklearn":
                    name += " (gini)"
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
        first = next(r for key, runs in cases.items() if key[0] == dataset for r in ok(runs))
        ax.set_title(f"{NAMES.get(dataset, dataset)}   ({first['n_train_loaded']:,} training rows, "
                     f"{first['n_features_loaded']} features)", fontsize=12, color=INK, loc="left",
                     fontweight="bold")
        ax.set_xlabel("training time (log scale): mean of the runs, whisker: fastest to slowest; "
                      "label: mean ± sd (runs), peak memory (RSS)",
                      fontsize=9, color=MUTED)
        # One tree line per protocol and tool.
        text = "\n".join(line.replace(";  ", ";\n      ") for line in lines)
        box.text(0, 0.95, text, transform=box.transAxes, va="top", fontsize=9, color=INK,
                 linespacing=1.5,
                 bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f5f2", "edgecolor": GRID})
    builds = defaultdict(set)  # ./tree or the libraries -> (build, day)
    for (_, _, impl, _), runs in cases.items():
        for r in ok(runs):
            builds["./tree" if impl in TREE else "libraries"].add(r["time"][:10])
    when = "; ".join(f"{who} {min(days)}" + (f" to {max(days)}" if len(days) > 1 else "")
                     for who, days in sorted(builds.items()))
    fig.suptitle("./tree --simple (the reference) against --serial, --parallel, scikit-learn and "
                 "Weka J48: depth ≤ 10, no pruning, accuracy on the test set\non epyc1 and epyc2 "
                 "(2× EPYC 9255 each); one thread on one socket's CPU, --parallel on one socket "
                 f"(24 threads) and on both (48).  Run {when}",
                 fontsize=12.5, color=INK, x=0.02, y=1 - 0.2 / fig.get_figheight(), ha="left",
                 va="top")
    out = args.out or args.root / "chart.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
