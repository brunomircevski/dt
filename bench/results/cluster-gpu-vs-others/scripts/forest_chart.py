#!/usr/bin/env python3
"""Charts of the GPU forest runs: cuML and XGBoost growing 1 tree and 10 trees,
depth <= 10, no pruning, on the H100 (hopper) and the A100 (ampere).

  forest_chart.py FOREST [--tree-results bench/results]
    FOREST  holds <dataset>/<node>/results.jsonl (submit-forest.sh), and gets
            chart-<dataset>.png and summary.csv

Every forest tree sees every row and every feature (no bootstrap,
max_features 1.0), so all 10 trees are the same tree; the job checks that the
forest predicts like the single tree (warmup_agreement_one_tree). The ratio
T10 / T1 therefore shows what a library pays per tree and what it pays once:
per tree = (T10 - T1) / 9, setup = T1 - per tree.

./tree is not rerun: its bars are the one-tree depth-10 unpruned runs of the
same test-set benchmarks (cluster-<dataset>/cuda-<node>-<dataset>, mean of 3).
"""
import argparse, csv, json, statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATASETS = {"higgs": "HIGGS: 10,500,000 training rows, 28 features, 500,000 test rows",
            "susy": "SUSY: 4,500,000 training rows, 18 features, 500,000 test rows",
            "airline": "airline: 114,569,017 training rows (1987–2007), 13 features, "
                       "test: 2008 (7,009,728 rows)",
            "ecbdl_15m": "ECBDL'14, 15M sample: 15,000,000 training rows, 631 features, "
                         "official test set (2,897,917 rows)",
            "chess_1b": "chess 3×3: 1,000,000,000 training rows, 2 features, 1,000 test rows"}
# ./tree's one-tree runs: (results dir, dataset name there) and protocol per case.
TREE_DIRS = {"higgs": "cluster-higgs", "susy": "cluster-susy", "airline": "cluster-airline",
             "ecbdl_15m": "cluster-ecbdl", "chess_1b": "cluster-chess"}
TREE_PROTOCOLS = {"cart": ("cart_depth10", "cart_full_nodepth"),
                  "c45": ("c45_depth10_unpruned", "c45_unpruned")}
NODES = {"hopper": "H100 PCIe 80 GB", "ampere": "A100 80 GB PCIe"}
COLORS = {"hopper": "#2a78d6", "ampere": "#eb6834"}
LIGHT = {"hopper": "#a9c9ef", "ampere": "#f6c3a6"}
# Groups top to bottom: (impl, case) -> label.
GROUPS = {("tree", "cart"): "./tree CART (exact Gini), 1 tree",
          ("cuml", "cart"): "cuML RandomForest, Gini, 128 bins",
          ("xgboost", "cart"): "XGBoost, squared error (= Gini), 256 bins",
          ("tree", "c45"): "./tree C4.5 (exact gain ratio), 1 tree",
          ("cuml", "c45"): "cuML RandomForest, entropy, 128 bins"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def seconds(value):
    if value >= 100:
        return f"{value:,.0f} s"
    return f"{value * 1000:.3g} ms" if value < 1 else f"{value:.3g} s"


def mean_of(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def load_forest(root, dataset):
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
            if "test_accuracy" not in last:
                print(f"{path}: {impl} {case} {trees} trees: unfinished, skipped")
                continue
            if trees > 1 and last.get("warmup_agreement_one_tree") != 1.0:
                raise SystemExit(f"{path}: {impl} {case} {trees} trees: the forest does not "
                                 f"predict like one tree ({last.get('warmup_agreement_one_tree')})")
            out[(node, impl, case, trees)] = {
                "runs": len(rs), "seconds": mean_of(rs, "train_seconds"),
                "dmatrix": mean_of(rs, "dmatrix_seconds"),
                "n_streams": last.get("n_streams"),
                **{k: last.get(k) for k in ("nodes", "depth", "test_accuracy", "gpu_peak_mib",
                                            "bins", "version")}}
    return out


def load_tree(results, dataset):
    """-> {(node, case): summary} of ./tree's one-tree CUDA runs."""
    out = {}
    for node in NODES:
        path = results / TREE_DIRS[dataset] / f"cuda-{node}-{dataset}" / "results.jsonl"
        if not path.exists():
            continue
        rs = [r for r in map(json.loads, open(path)) if r.get("kind") == "run"]
        for case, protocols in TREE_PROTOCOLS.items():
            runs = [r for r in rs if r["protocol"] in protocols and r["status"] == "ok"]
            if not runs:
                continue
            last = runs[-1]
            out[(node, case)] = {"runs": len(runs), "seconds": mean_of(runs, "train_seconds"),
                                 "protocol": last["protocol"], "nodes": last.get("nodes"),
                                 "depth": last.get("depth"),
                                 "test_accuracy": last.get("test_accuracy"),
                                 "train_accuracy": last.get("train_accuracy")}
    return out


def split(one, ten):
    """-> (setup, per tree) from the 1-tree and 10-tree times."""
    per_tree = (ten - one) / 9
    return one - per_tree, per_tree


def chart(dataset, forest, tree, out):
    fig = plt.figure(figsize=(15, 12))
    grid = fig.add_gridspec(2, 1, height_ratios=[6.6, 2.2], left=0.25, right=0.97,
                            top=0.885, bottom=0.01, hspace=0.2)
    ax = fig.add_subplot(grid[0])
    box = fig.add_subplot(grid[1])
    box.axis("off")

    times = [v["seconds"] for v in forest.values()] + [v["seconds"] for v in tree.values()]
    low, high = min(times) / 3, max(times)
    y, ticks = 0, []
    for (impl, case), title in GROUPS.items():
        bars = [(node, trees) for trees in (1, 10) for node in NODES
                if (impl == "tree" and trees == 1 and (node, case) in tree)
                or (impl != "tree" and (node, impl, case, trees) in forest)]
        if not bars:
            continue
        ax.text(0.004, y - 0.3, title, transform=ax.get_yaxis_transform(), ha="left",
                va="bottom", fontsize=10, fontweight="bold", color=INK)
        y += 0.5
        for node, trees in bars:
            r = tree[(node, case)] if impl == "tree" else forest[(node, impl, case, trees)]
            total = r["seconds"]
            hatch = "//" if impl == "tree" else None
            ax.barh(y, total - low, left=low, height=0.72, color=COLORS[node], hatch=hatch,
                    edgecolor="white", linewidth=0)
            if r.get("dmatrix"):  # XGBoost: the QuantileDMatrix binning share, from the left
                ax.barh(y, r["dmatrix"] - low, left=low, height=0.72, color=LIGHT[node],
                        linewidth=0)
            label = seconds(total)
            if trees == 10:
                one = forest[(node, impl, case, 1)]["seconds"]
                label += f"   ×{total / one:.2f} of 1 tree"
            if r.get("dmatrix"):
                label += f"   (binning {seconds(r['dmatrix'])})"
            ax.text(total * 1.07, y, label, va="center", fontsize=9, color=INK)
            what = "1 tree" if trees == 1 else "10 trees"
            ticks.append((y, f"{NODES[node]}, {what}"))
            y += 1
        y += 0.65
    ax.set_xscale("log")
    ax.set_xlim(low, high * (high / low) ** 0.32)
    ax.set_xticks([10.0 ** e for e in range(-3, 5) if low <= 10.0 ** e <= high * 3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: seconds(v)))
    ax.set_yticks([p for p, _ in ticks], [t for _, t in ticks], fontsize=9.5)
    ax.set_ylim(y - 0.5, -0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("training time, mean of 3 runs (log scale)", fontsize=9.5, color=MUTED)
    ax.set_title(DATASETS[dataset], fontsize=12, color=INK, loc="left", fontweight="bold")
    legend = [Patch(color=COLORS[n], label=NODES[n]) for n in NODES]
    legend += [Patch(facecolor=LIGHT["hopper"], label="XGBoost: QuantileDMatrix binning "
                     "(inside the bar, from its left end)"),
               Patch(facecolor="#bbbbbb", hatch="//", edgecolor="white",
                     label="./tree: earlier one-tree run (not a forest)")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, -0.07), ncol=4,
              fontsize=8.5, frameon=False)

    # The table: per library and GPU, setup / per tree, VRAM, the tree, accuracy.
    lines = []
    for (impl, case), title in GROUPS.items():
        for node in NODES:
            if impl == "tree":
                r = tree.get((node, case))
                if not r:
                    continue
                accuracy = (f"test acc. {r['test_accuracy']:.2%}" if r["test_accuracy"] is not None
                            else f"train acc. {r['train_accuracy']:.0%} (exact 3×3 board)")
                lines.append(f"{title.split(',')[0]}, {NODES[node]}:  {r['nodes']:,} nodes, "
                             f"depth {r['depth']}, {accuracy}  [{r['protocol']}]")
                continue
            one, ten = forest.get((node, impl, case, 1)), forest.get((node, impl, case, 10))
            if not one or not ten:
                continue
            setup, per_tree = split(one["seconds"], ten["seconds"])
            name = title.split(",")[0] + (" Gini" if case == "cart" and impl == "cuml" else
                                          " entropy" if impl == "cuml" else "")
            text = (f"{name}, {NODES[node]}:  setup {seconds(max(setup, 0))}, per tree "
                    f"{seconds(max(per_tree, 0))};  VRAM {one['gpu_peak_mib'] / 1024:.1f} → "
                    f"{ten['gpu_peak_mib'] / 1024:.1f} GiB;  {one['nodes']:,} nodes, depth "
                    f"{one['depth']}, test acc. {one['test_accuracy']:.2%}")
            if ten.get("n_streams") and ten["n_streams"] != 4:
                text += f"  (n_streams {ten['n_streams']})"
            lines.append(text)
    box.text(0, 0.97, "\n".join(lines), transform=box.transAxes, va="top", fontsize=8.8,
             color=INK, linespacing=1.45, family="DejaVu Sans",
             bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f5f2", "edgecolor": GRID})

    versions = sorted({f"{impl} {v['version']}" for (_, impl, _, _), v in forest.items()})
    fig.suptitle(
        "cuML and XGBoost on the GPU: 1 tree against a forest of 10 trees, depth ≤ 10, no pruning\n"
        "Every forest tree sees all rows and all features (no bootstrap, max_features 1.0), so the "
        "10 trees are the same tree (checked: the forest predicts like the single tree).\n"
        "setup = T1 − per tree, per tree = (T10 − T1) / 9.  cuML: one fit of a RandomForest, "
        "n_streams 4 for 10 trees, after a full-size untimed warm-up fit.  XGBoost: "
        "num_parallel_tree 10, one round,\nreg:squarederror with lambda 0 (the Gini split), "
        "binning (QuantileDMatrix) inside the time.  Accuracy on the dataset's test set.  "
        f"VRAM: peak.  {', '.join(versions)}.",
        fontsize=10.5, color=INK, x=0.015, y=0.995, ha="left", va="top", linespacing=1.5)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("forest", type=Path)
    parser.add_argument("--tree-results", type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    rows = []
    for dataset in DATASETS:
        forest = load_forest(args.forest, dataset)
        if not forest:
            continue
        tree = load_tree(args.tree_results, dataset)
        chart(dataset, forest, tree, args.forest / f"chart-{dataset}.png")
        for (node, impl, case, trees), v in sorted(forest.items()):
            rows.append({"dataset": dataset, "node": node, "impl": impl, "case": case,
                         "trees": trees, "runs": v["runs"], "mean_s": round(v["seconds"], 4),
                         "dmatrix_s": v["dmatrix"] and round(v["dmatrix"], 4),
                         "gpu_peak_mib": v["gpu_peak_mib"], "nodes": v["nodes"],
                         "depth": v["depth"], "test_accuracy": v["test_accuracy"],
                         "n_streams": v["n_streams"], "version": v["version"]})
    with open(args.forest / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.forest / 'summary.csv'}")


if __name__ == "__main__":
    main()
