#!/usr/bin/env python3
"""Charts and summary.csv of the GPU comparison: one tree, depth <= 10, no
pruning, on the H100 (hopper) and the A100 (ampere).

  summarize.py ROOT [--tree-results bench/results]
    ROOT  holds <dataset>/<node>/results.jsonl (cuML and XGBoost, from
          submit-forest.sh) and gets chart-<dataset>.png and summary.csv

The libraries ran 1 tree and 10 trees. All 10 trees see every row and every
feature (no bootstrap, max_features 1.0), so they are the same tree, and the
job checks that the forest predicts like the single tree; "10 trees / 10" is
then the time of one tree inside a forest.

./tree is not run here: its bars are the depth-10 unpruned CUDA runs of the
test-set benchmarks, cluster-<dataset>/cuda-<node>-<dataset> (mean of 3).
"""
import argparse, csv, json, statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATASETS = {"higgs": "HIGGS: 10,500,000 training rows, 28 features; test: 500,000 rows",
            "susy": "SUSY: 4,500,000 training rows, 18 features; test: 500,000 rows",
            "airline": "airline: 114,569,017 training rows (1987–2007), 13 features; "
                       "test: 2008 (7,009,728 rows)",
            "ecbdl_15m": "ECBDL'14, 15M sample: 15,000,000 training rows, 631 features; "
                         "test: the official 2,897,917 rows",
            "chess_1b": "chess 3×3: 1,000,000,000 training rows, 2 features; test: 1,000 rows"}
TREE_DIRS = {"higgs": "cluster-higgs", "susy": "cluster-susy", "airline": "cluster-airline",
             "ecbdl_15m": "cluster-ecbdl", "chess_1b": "cluster-chess"}
# chess ran without a depth limit; its exact tree has depth 4.
TREE_PROTOCOLS = {"cart": ("cart_depth10", "cart_full_nodepth"),
                  "c45": ("c45_depth10_unpruned", "c45_unpruned")}
NODES = {"hopper": "H100", "ampere": "A100"}
COLORS = {"hopper": "#2a78d6", "ampere": "#eb6834"}
CASES = {"cart": "CART", "c45": "C4.5"}
# Per case, top to bottom: (impl, trees) -> label.
BARS = {"cart": {("tree", 1): "./tree, exact Gini",
                 ("cuml", 1): "cuML, Gini, 1 tree",
                 ("cuml", 10): "cuML, Gini, 10 trees / 10",
                 ("xgboost", 1): "XGBoost, 1 tree",
                 ("xgboost", 10): "XGBoost, 10 trees / 10"},
        "c45": {("tree", 1): "./tree, exact gain ratio",
                ("cuml", 1): "cuML, entropy, 1 tree",
                ("cuml", 10): "cuML, entropy, 10 trees / 10"}}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def seconds(value):
    if value >= 100:
        return f"{value:,.0f} s"
    return f"{value * 1000:.3g} ms" if value < 1 else f"{value:.3g} s"


def load_libraries(root, dataset):
    """-> {(node, impl, case, trees): summary}; the last rep has the tree's facts."""
    out = {}
    for node in NODES:
        path = root / dataset / node / "results.jsonl"
        if not path.exists():
            continue
        groups = {}
        for r in map(json.loads, open(path)):
            groups.setdefault((r["impl"], r["case"], r["trees"]), []).append(r)
        for (impl, case, trees), rs in groups.items():
            last = rs[-1]
            if last["depth_limit"] != 10:
                raise SystemExit(f"{path}: {impl} {case}: depth limit {last['depth_limit']}")
            if trees > 1 and last.get("warmup_agreement_one_tree") != 1.0:
                raise SystemExit(f"{path}: {impl} {case} {trees} trees: the forest does not "
                                 f"predict like one tree")
            out[(node, impl, case, trees)] = {
                "runs": len(rs), "seconds": statistics.mean(r["train_seconds"] for r in rs),
                "vram_gib": last["gpu_peak_mib"] / 1024, "nodes": last["nodes"],
                "depth": last["depth"], "test_accuracy": last["test_accuracy"],
                "version": last["version"]}
    return out


def load_tree(results, dataset):
    """-> {(node, "tree", case, 1): summary} of ./tree's CUDA runs."""
    out = {}
    for node in NODES:
        path = results / TREE_DIRS[dataset] / f"cuda-{node}-{dataset}" / "results.jsonl"
        rs = [r for r in map(json.loads, open(path)) if r.get("kind") == "run"]
        for case, protocols in TREE_PROTOCOLS.items():
            runs = [r for r in rs if r["protocol"] in protocols and r["status"] == "ok"]
            last = runs[-1]
            accuracy = last.get("test_accuracy")
            out[(node, "tree", case, 1)] = {
                "runs": len(runs), "seconds": statistics.mean(r["train_seconds"] for r in runs),
                "vram_gib": last["gpu_peak_bytes"] / 2 ** 30, "nodes": last["nodes"],
                "depth": last["depth"], "test_accuracy": accuracy,
                "train_accuracy": last.get("train_accuracy"), "version": last["protocol"]}
    return out


def chart(dataset, runs, out):
    fig = plt.figure(figsize=(13, 9.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[6, 1.7], left=0.21, right=0.97,
                            top=0.885, bottom=0.01, hspace=0.2)
    ax = fig.add_subplot(grid[0])
    box = fig.add_subplot(grid[1])
    box.axis("off")

    def per_tree(r, trees):
        return r["seconds"] / trees

    times = [per_tree(r, key[3]) for key, r in runs.items()]
    low, high = min(times) / 3, max(times)
    y, ticks, lines = 0, [], []
    for case, bars in BARS.items():
        ax.text(0.004, y - 0.35, f"{CASES[case]}", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
        y += 0.45
        for (impl, trees), label in bars.items():
            if not any((node, impl, case, trees) in runs for node in NODES):
                continue
            for offset, node in zip((-0.2, 0.2), NODES):
                r = runs.get((node, impl, case, trees))
                if not r:
                    continue
                t = per_tree(r, trees)
                ax.barh(y + offset, t - low, left=low, height=0.38, color=COLORS[node],
                        linewidth=0)
                text = f"{seconds(t)}   {r['vram_gib']:.3g} GiB"
                if trees > 1:
                    text = f"{seconds(t)}  (10 trees: {seconds(r['seconds'])})   {r['vram_gib']:.3g} GiB"
                ax.text(t * 1.06, y + offset, text, va="center", fontsize=8.5, color=INK)
            ticks.append((y, label))
            r = runs.get(("hopper", impl, case, trees)) or runs.get(("ampere", impl, case, trees))
            if trees == 1:
                accuracy = (f"test acc. {r['test_accuracy']:.2%}" if r["test_accuracy"] is not None
                            else f"training acc. {r['train_accuracy']:.0%} (no test accuracy recorded)")
                lines.append(f"{label.replace(', 1 tree', '')}:  {r['nodes']:,} nodes, depth "
                             f"{r['depth']}, {accuracy}")
            y += 1
        y += 0.5
    ax.set_xscale("log")
    ax.set_xlim(low, high * (high / low) ** 0.45)
    ax.set_xticks([10.0 ** e for e in range(-3, 5) if low <= 10.0 ** e <= high * 3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: seconds(v)))
    ax.set_yticks([p for p, _ in ticks], [t for _, t in ticks], fontsize=10)
    ax.set_ylim(y - 0.4, -0.85)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("training time of one tree, mean of 3 runs (log scale); label: time, peak VRAM",
                  fontsize=9.5, color=MUTED)
    ax.set_title(DATASETS[dataset], fontsize=12, color=INK, loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=COLORS[n], label=f"{NODES[n]} ({n})") for n in NODES],
              loc="lower right", fontsize=9, frameon=False)
    box.text(0, 0.95, "\n".join(lines), transform=box.transAxes, va="top", fontsize=9,
             color=INK, linespacing=1.5,
             bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f5f2", "edgecolor": GRID})

    versions = sorted({f"{impl} {r['version']}" for (_, impl, _, _), r in runs.items()
                       if impl != "tree"})
    fig.suptitle(
        "./tree against cuML and XGBoost on the GPU: one tree, depth ≤ 10, no pruning, all rows "
        "and features\n"
        "cuML RandomForest (128 bins); XGBoost: squared error with lambda 0, i.e. the Gini split "
        "(256 bins, CART only), binning inside the time.\n"
        "10 trees / 10: a forest of 10 identical trees (no bootstrap), its time divided by 10.  "
        f"{', '.join(versions)}; ./tree: the {TREE_DIRS[dataset]} CUDA runs (depth 10, unpruned).",
        fontsize=10.5, color=INK, x=0.015, y=0.995, ha="left", va="top", linespacing=1.5)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--tree-results", type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    rows = []
    for dataset in DATASETS:
        runs = load_libraries(args.root, dataset)
        if not runs:
            continue
        runs.update(load_tree(args.tree_results, dataset))
        chart(dataset, runs, args.root / f"chart-{dataset}.png")
        for (node, impl, case, trees), r in sorted(runs.items()):
            rows.append({"dataset": dataset, "node": node, "impl": impl, "case": case,
                         "trees": trees, "runs": r["runs"], "mean_s": round(r["seconds"], 4),
                         "per_tree_s": round(r["seconds"] / trees, 4),
                         "vram_gib": round(r["vram_gib"], 2), "nodes": r["nodes"],
                         "depth": r["depth"], "test_accuracy": r["test_accuracy"],
                         "version": r["version"]})
    with open(args.root / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.root / 'summary.csv'}")


if __name__ == "__main__":
    main()
