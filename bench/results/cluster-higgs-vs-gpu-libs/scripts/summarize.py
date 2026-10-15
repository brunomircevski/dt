#!/usr/bin/env python3
"""Tables and chart of the GPU comparison.

  summarize.py RESULTS OUT
    RESULTS  holds <node>/results.jsonl, <node>/machine.json, <node>/pred/*.npy
    OUT      gets summary.csv, tables.md, chart.png

Agreement: share of the 500,000 test rows that a tree classifies like ./tree's
tree of the same case on the same node.
"""
import csv, json, statistics, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

NODES = {"hopper": "H100 PCIe 80 GB", "ampere": "A100 80 GB PCIe"}
CASES = {"d12": "CART, depth ≤ 12", "d19": "CART, depth ≤ 19"}
LABELS = {("tree", "exact"): "./tree (exact)",
          ("cuml", "default"): "cuML RF, 128 bins", ("cuml", "1024"): "cuML RF, 1024 bins",
          ("xgboost", "default"): "XGBoost, 256 bins", ("xgboost", "1024"): "XGBoost, 1024 bins"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"hopper": "#2a78d6", "ampere": "#eb6834"}


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for node in NODES:
        runs = [json.loads(l) for l in open(root / node / "results.jsonl")]
        groups = {}
        for r in runs:
            groups.setdefault((r["impl"], r["case"], r["bins"]), []).append(r)
        for (impl, case, bins), rs in groups.items():
            last = rs[-1]
            times = [r["train_seconds"] for r in rs]
            row = {"node": node, "impl": impl, "case": case, "bins": bins,
                   "label": LABELS[(impl, bins)], "runs": len(rs),
                   "median_s": statistics.median(times), "min_s": min(times), "max_s": max(times),
                   **{k: last.get(k) for k in ("nodes", "depth", "train_accuracy",
                                               "test_accuracy", "gpu_peak_mib", "root",
                                               "version")}}
            pred = root / node / "pred" / f"{impl}-{case}-{bins}.npy"
            ref = root / node / "pred" / f"tree-{case}-exact.npy"
            if impl != "tree" and pred.exists() and ref.exists():
                row["agreement"] = float(np.mean(np.load(pred) == np.load(ref)))
            rows.append(row)
    order = list(LABELS)
    rows.sort(key=lambda r: (list(NODES).index(r["node"]), list(CASES).index(r["case"]),
                             order.index((r["impl"], r["bins"]))))
    fields = ["node", "case", "impl", "bins", "runs", "median_s", "min_s", "max_s", "nodes", "depth",
              "train_accuracy", "test_accuracy", "agreement", "gpu_peak_mib", "root", "version"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Markdown: one table per case, the nodes side by side.
    md = []
    for case, title in CASES.items():
        cr = [r for r in rows if r["case"] == case]
        if not cr:
            continue
        md += [f"### {title}", "",
               "| Implementation | H100 time | A100 time | Nodes | Depth | Train acc. | Test acc. "
               "| Same as ./tree | GPU mem. (H100) | Root split |",
               "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
        for key in order:
            by = {r["node"]: r for r in cr if (r["impl"], r["bins"]) == key}
            if not by:
                continue
            h, a = by.get("hopper"), by.get("ampere")
            ref = h or a

            def t(r):
                return "–" if r is None else f"{r['median_s']:.2f} s ({r['min_s']:.2f}–{r['max_s']:.2f})"
            agree = ref.get("agreement")
            md.append(f"| {LABELS[key]} | {t(h)} | {t(a)} | {ref['nodes']:,} | {ref['depth']} | "
                      f"{100 * ref['train_accuracy']:.2f}% | {100 * ref['test_accuracy']:.2f}% | "
                      f"{'–' if agree is None else f'{100 * agree:.2f}%'} | "
                      f"{(h or {}).get('gpu_peak_mib') or 0:,} MiB | `{ref['root']}` |")
        md.append("")
    (out / "tables.md").write_text("\n".join(md))

    # Chart: one row of panels per case: training time on both GPUs (bars) and
    # the tree each implementation built (text column; identical on both GPUs).
    # Only implementations with results in every case are drawn.
    cases = [c for c in CASES if any(r["case"] == c for r in rows)]
    keys = [k for k in order if all(any((r["impl"], r["bins"]) == k and r["case"] == c
                                        and r["node"] == n for r in rows)
                                    for c in cases for n in NODES)][::-1]
    fig = plt.figure(figsize=(15, 2.3 + 2.9 * len(cases)))
    grid = fig.add_gridspec(len(cases), 2, width_ratios=(1.15, 1), wspace=0.02, hspace=0.45)
    y = np.arange(len(keys))
    for row_index, case in enumerate(cases):
        ax = fig.add_subplot(grid[row_index, 0])
        info = fig.add_subplot(grid[row_index, 1], sharey=ax)
        tmax = max(r["median_s"] for r in rows if r["case"] == case and (r["impl"], r["bins"]) in keys)
        for i, node in enumerate(NODES):
            got = {(r["impl"], r["bins"]): r for r in rows if r["node"] == node and r["case"] == case}
            vals = [got[k]["median_s"] for k in keys]
            pos = y + (0.2 if i == 0 else -0.2)
            ax.barh(pos, vals, height=0.38, color=COLORS[node], label=f"{node}: {NODES[node]}")
            for p, v in zip(pos, vals):
                ax.text(v + tmax * 0.012, p, f"{v:.2f} s", va="center", fontsize=8.3, color=INK)
        ax.set_xlim(0, tmax * 1.12)
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_yticks(y, [LABELS[k] for k in keys], fontsize=9.5, color=INK)
        ax.set_title(f"{CASES[case]}: training time", loc="left", fontsize=11, color=INK,
                     fontweight="bold", pad=16)
        ax.set_xlabel("seconds (median of 3 runs)", fontsize=8.5, color=MUTED)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)

        # The trees: the same on both GPUs (checked on the test predictions),
        # so hopper's.
        got = {(r["impl"], r["bins"]): r for r in rows if r["node"] == "hopper" and r["case"] == case}
        columns = [(0.02, "nodes", lambda r: f"{r['nodes']:,}"),
                   (0.26, "depth", lambda r: f"{r['depth']}"),
                   (0.40, "test accuracy", lambda r: f"{100 * r['test_accuracy']:.2f}%"),
                   (0.66, "same as ./tree", lambda r: "–" if r.get("agreement") is None
                    else f"{100 * r['agreement']:.1f}%")]
        for x, head, cell in columns:
            info.text(x, len(keys) - 0.45, head, fontsize=8.5, color=MUTED, ha="left", va="bottom",
                      transform=info.get_yaxis_transform())
            for k, p in zip(keys, y):
                info.text(x, p, cell(got[k]), fontsize=9.5, color=INK, ha="left", va="center",
                          transform=info.get_yaxis_transform(),
                          fontweight="bold" if k == ("tree", "exact") else "normal")
        info.set_title("the tree (500,000 test rows)", loc="left", fontsize=11, color=INK,
                       fontweight="bold", pad=16)
        info.axis("off")
        if row_index == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("HIGGS: one unpruned Gini (CART) tree on one GPU — 10,500,000 training rows, "
                 "28 features", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.012, "Bins: the libraries split only at the edges of per-feature quantile bins; "
             "./tree tests every distinct value. Same as ./tree: share of test rows classified as by "
             "./tree's tree of the same depth.", fontsize=8.5, color=MUTED, style="italic", va="bottom")
    fig.subplots_adjust(left=0.13, right=0.99, top=1 - 1.05 / fig.get_figheight(),
                        bottom=0.95 / fig.get_figheight())
    fig.savefig(out / "chart.png", dpi=150)


if __name__ == "__main__":
    main()
