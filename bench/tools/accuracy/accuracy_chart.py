#!/usr/bin/env python3
"""Charts of the pruned-accuracy benchmark from report.py's CSV.

  accuracy_chart.py REPORT_CSV OUT_DIR

chart-accuracy.png: test accuracy against the depth limit, one row per
dataset, CART and C4.5 side by side; every tool's pruned tree (solid), ./tree's
unpruned tree (dashed) and the 95% interval of ./tree's pruned accuracy (band).
chart-nodes.png: the same layout, nodes of each tree (log scale).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# ./tree, scikit-learn, rpart, J48 and YaDT as in vs_chart.py
TOOLS = {"tree": ("./tree", "#2a78d6", "o"), "sklearn": ("scikit-learn", "#1baf7a", "s"),
         "rpart": ("rpart", "#0f6e4c", "D"), "cuml": ("cuML (128 bins)", "#7f4fc9", "^"),
         "xgboost": ("XGBoost (256 bins)", "#d6455a", "v"), "j48": ("Weka J48", "#eda100", "s"),
         "yadt": ("YaDT", "#e87ba4", "D")}
DATASETS = {"susy": "SUSY (4.5M rows, 18 features; test 500k)",
            "higgs": "HIGGS (10.5M rows, 28 features; test 500k)",
            "mnist8m": "MNIST8M (8.1M rows, 784 features, 10 classes; test 10k)",
            "ecbdl_15m": "ECBDL'14 15M (15M rows, 631 features; test 2.9M)",
            "sorel_4m": "SOREL 4M (4M rows, 2381 features; test 4.2M)"}
ALGOS = {"cart": "CART, cost-complexity α = 1e-5", "c45": "C4.5, error-based pruning CF 0.25"}


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    out = Path(sys.argv[2])
    data = defaultdict(list)  # (dataset, algo, tool, pruned) -> [(depth, row)]
    for r in rows:
        data[(r["dataset"], r["algo"], r["tool"], r["pruned"] == "True")].append((int(r["depth_limit"]), r))
    datasets = [d for d in DATASETS if any(k[0] == d for k in data)]
    for what in ("accuracy", "nodes"):
        fig, axes = plt.subplots(len(datasets), 2, figsize=(13, 3.3 * len(datasets) + 0.9), squeeze=False)
        for i, dataset in enumerate(datasets):
            for j, algo in enumerate(ALGOS):
                ax = axes[i][j]
                ax.set_title(f"{DATASETS[dataset]}\n{ALGOS[algo]}", fontsize=9, color=INK, loc="left")
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_color(MUTED)
                ax.grid(axis="y", color=GRID, lw=0.8)
                ax.set_axisbelow(True)
                ax.tick_params(colors=MUTED, labelsize=8)
                ax.set_xticks([5, 10, 15, 20])
                ax.set_xlim(4, 21)
                if i == len(datasets) - 1:
                    ax.set_xlabel("depth limit", fontsize=9, color=MUTED)
                if what == "nodes":
                    ax.set_yscale("log")
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
                    ax.set_ylabel("nodes (log)", fontsize=8, color=MUTED)
                else:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
                    ax.set_ylabel("test accuracy", fontsize=8, color=MUTED)
                key = lambda t, p: sorted(data.get((dataset, algo, t, p), []))  # noqa: E731
                tree = key("tree", True)
                if what == "accuracy" and tree:
                    ax.fill_between([d for d, _ in tree], [100 * float(r["ci_low"]) for _, r in tree],
                                    [100 * float(r["ci_high"]) for _, r in tree], color="#2a78d6", alpha=0.12,
                                    lw=0, label="./tree pruned: 95% interval")
                for tool, (label, color, marker) in TOOLS.items():
                    for pruned in (True, False):
                        points = key(tool, pruned)
                        if not points or (not pruned and tool != "tree" and what == "accuracy"):
                            continue
                        if not pruned and what == "nodes" and tool != "tree":
                            continue
                        x = [d for d, _ in points]
                        y = [100 * float(r["test_accuracy"]) if what == "accuracy" else float(r["nodes"])
                             for _, r in points]
                        ax.plot(x, y, color=color, lw=2.6 if tool == "tree" else 1.4,
                                ls="-" if pruned else "--", marker=marker if pruned else None,
                                ms=7 if tool == "tree" else 5, mfc=color if tool == "tree" else "white",
                                mec=color, mew=1.4, zorder=2 if tool == "tree" else 3,
                                label=f"{label}{'' if pruned else ', unpruned'}")
        if what == "accuracy":  # enough decimals that neighbouring ticks differ
            for ax in axes.flat:
                lo, hi = ax.get_ylim()
                places = 2 if hi - lo < 1 else 1 if hi - lo < 10 else 0
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, k=places: f"{v:.{k}f}%"))
        handles, labels = {}, []
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in handles:
                    handles[l] = h
                    labels.append(l)
        fig.legend([handles[l] for l in labels], labels, loc="upper center", ncol=5, fontsize=8.5,
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle("Test accuracy of one pruned tree per tool" if what == "accuracy"
                     else "Tree size: nodes of each pruned tree (./tree unpruned dashed)",
                     y=1.0 + 0.35 / fig.get_figheight(), fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
        fig.savefig(out / f"chart-{what}.png", dpi=140, facecolor="white", bbox_inches="tight")
        print(out / f"chart-{what}.png")


if __name__ == "__main__":
    main()
