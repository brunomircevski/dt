#!/usr/bin/env python3
"""Tables and charts of the GPU comparison.

  summarize.py RESULTS OUT
    RESULTS  holds <dataset>/<node>/results.jsonl, machine.json, pred/*.npy
    OUT      gets summary.csv, tables.md, chart-<dataset>.png

Agreement: share of the test rows (TEST_ROWS) that a tree classifies like
./tree's tree of the same case on the same node.
"""
import csv, json, statistics, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATASETS = {"higgs": "HIGGS: 10,500,000 training rows, 28 features",
            "airline": "airline: 114,569,017 training rows, 13 features",
            "ecbdl_15m": "ECBDL'14, 15M sample: 15,000,000 training rows, 631 features"}
TEST_ROWS = {"higgs": 500_000, "airline": 500_000, "ecbdl_15m": 100_000}
NODES = {"hopper": "H100 PCIe 80 GB", "ampere": "A100 80 GB PCIe"}
CASES = {"cart": "CART, depth ≤ 19", "c45": "C4.5, depth ≤ 19"}
LABELS = {("tree", "cart"): "./tree CART (exact Gini)",
          ("cuml", "cart"): "cuML RF, Gini, 128 bins",
          ("xgboost", "cart"): "XGBoost, squared error, 256 bins",
          ("tree", "c45"): "./tree C4.5 (exact gain ratio)",
          ("cuml", "c45"): "cuML RF, entropy, 128 bins"}
IMPLS = ["tree", "cuml", "xgboost"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"hopper": "#2a78d6", "ampere": "#eb6834"}


def load_rows(root):
    rows = []
    for dataset in DATASETS:
        for node in NODES:
            path = root / dataset / node / "results.jsonl"
            if not path.exists():
                continue
            groups = {}
            for r in map(json.loads, open(path)):
                groups.setdefault((r["impl"], r["case"]), []).append(r)
            for (impl, case), rs in groups.items():
                last = rs[-1]
                times = [r["train_seconds"] for r in rs]
                row = {"dataset": dataset, "node": node, "impl": impl, "case": case,
                       "bins": last["bins"], "label": LABELS[(impl, case)], "runs": len(rs),
                       "median_s": statistics.median(times), "min_s": min(times),
                       "max_s": max(times),
                       **{k: last.get(k) for k in ("nodes", "depth", "train_accuracy",
                                                   "test_accuracy", "gpu_peak_mib", "root",
                                                   "version")}}
                pred = root / dataset / node / "pred" / f"{impl}-{case}.npy"
                ref = root / dataset / node / "pred" / f"tree-{case}.npy"
                if impl != "tree" and pred.exists() and ref.exists():
                    row["agreement"] = float(np.mean(np.load(pred) == np.load(ref)))
                rows.append(row)
    rows.sort(key=lambda r: (list(DATASETS).index(r["dataset"]), list(NODES).index(r["node"]),
                             list(CASES).index(r["case"]), IMPLS.index(r["impl"])))
    return rows


def tables(rows):
    md = []
    for dataset in DATASETS:
        for case, title in CASES.items():
            cr = [r for r in rows if r["dataset"] == dataset and r["case"] == case]
            if not cr:
                continue
            md += [f"### {dataset}: {title}", "",
                   "| Implementation | H100 time | A100 time | Nodes | Depth | Train acc. | Test acc. "
                   "| Same as ./tree | VRAM H100 | VRAM A100 | Root split |",
                   "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
            for impl in IMPLS:
                by = {r["node"]: r for r in cr if r["impl"] == impl}
                if not by:
                    continue
                h, a = by.get("hopper"), by.get("ampere")
                ref = h or a

                def t(r):
                    return "–" if r is None else f"{r['median_s']:.2f} s ({r['min_s']:.2f}–{r['max_s']:.2f})"

                def mem(r):
                    return "–" if r is None else f"{r['gpu_peak_mib'] / 1024:.2f} GiB"
                agree = ref.get("agreement")
                md.append(f"| {LABELS[(impl, case)]} | {t(h)} | {t(a)} | {ref['nodes']:,} | "
                          f"{ref['depth']} | {100 * ref['train_accuracy']:.2f}% | "
                          f"{100 * ref['test_accuracy']:.2f}% | "
                          f"{'–' if agree is None else f'{100 * agree:.2f}%'} | "
                          f"{mem(h)} | {mem(a)} | `{ref['root']}` |")
            md.append("")
    return "\n".join(md)


def style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def chart(rows, dataset, path):
    """One row of panels per case: training time on both GPUs (filled bars),
    peak VRAM (outlined bars) and the tree each implementation built (text;
    the same on both GPUs)."""
    cases = [c for c in CASES if any(r["dataset"] == dataset and r["case"] == c for r in rows)]
    heights = [len([i for i in IMPLS if (i, c) in LABELS]) for c in cases]
    fig = plt.figure(figsize=(17, 2.2 + 0.95 * sum(heights) + 1.3 * len(cases)))
    grid = fig.add_gridspec(len(cases), 3, width_ratios=(1.15, 0.6, 0.75), wspace=0.04,
                            hspace=0.42, height_ratios=heights)
    dr = [r for r in rows if r["dataset"] == dataset]
    runs = sorted({r["runs"] for r in dr})
    for row_index, case in enumerate(cases):
        cr = [r for r in dr if r["case"] == case]
        keys = [i for i in IMPLS if any(r["impl"] == i for r in cr)][::-1]
        y = np.arange(len(keys))
        ax = fig.add_subplot(grid[row_index, 0])
        mem = fig.add_subplot(grid[row_index, 1], sharey=ax)
        info = fig.add_subplot(grid[row_index, 2], sharey=ax)
        tmax = max(r["median_s"] for r in cr)
        mmax = max(r["gpu_peak_mib"] for r in cr) / 1024
        for i, node in enumerate(NODES):
            got = {r["impl"]: r for r in cr if r["node"] == node}
            pos = y + (0.2 if i == 0 else -0.2)
            for k, p in zip(keys, pos):
                if k not in got:
                    continue
                v = got[k]["median_s"]
                ax.barh(p, v, height=0.38, color=COLORS[node])
                ax.text(v + tmax * 0.012, p, f"{v:.2f} s", va="center", fontsize=8.5, color=INK,
                        fontweight="bold" if k == "tree" else "normal")
                g = got[k]["gpu_peak_mib"] / 1024
                mem.barh(p, g, height=0.34, facecolor="white", edgecolor=COLORS[node],
                         linewidth=1.6)
                mem.text(g + mmax * 0.02, p, f"{g:.2f} GiB", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, tmax * 1.12)
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
        style(ax)
        ax.set_yticks(y, [LABELS[(k, case)] for k in keys], fontsize=9.5, color=INK)
        for label, k in zip(ax.get_yticklabels(), keys):
            label.set_fontweight("bold" if k == "tree" else "normal")
            label.set_color(INK)
        ax.set_title(f"{CASES[case]}: training time", loc="left", fontsize=11, color=INK,
                     fontweight="bold", pad=10)
        mem.set_xlim(0, mmax * 1.3)
        mem.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} GiB"))
        mem.set_title("peak VRAM", loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
        mem.tick_params(axis="y", labelleft=False)
        style(mem)

        # The trees are the same on both GPUs (checked on the test predictions),
        # so hopper's (ampere's where hopper has none).
        got = {r["impl"]: r for r in sorted(cr, key=lambda r: r["node"] == "hopper")}
        columns = [(0.04, "nodes", lambda r: f"{r['nodes']:,}"),
                   (0.30, "depth", lambda r: f"{r['depth']}"),
                   (0.44, "test acc.", lambda r: f"{100 * r['test_accuracy']:.2f}%"),
                   (0.68, "same as ./tree", lambda r: "–" if r.get("agreement") is None
                    else f"{100 * r['agreement']:.1f}%")]
        for x, head, cell in columns:
            info.text(x, len(keys) - 0.5, head, fontsize=8.5, color=MUTED, ha="left", va="bottom",
                      transform=info.get_yaxis_transform())
            for k, p in zip(keys, y):
                info.text(x, p, cell(got[k]), fontsize=9.5, color=INK, ha="left", va="center",
                          transform=info.get_yaxis_transform(),
                          fontweight="bold" if k == "tree" else "normal")
        info.set_title(f"the tree ({TEST_ROWS[dataset]:,} test rows)", loc="left", fontsize=11, color=INK,
                       fontweight="bold", pad=10)
        info.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[n]) for n in NODES]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, linewidth=1.6))
    fig.legend(handles, [f"{n}: {NODES[n]}" for n in NODES] + ["VRAM: outlined bar"],
               loc="upper left", bbox_to_anchor=(0.01, 1 - 0.62 / fig.get_figheight()), ncol=3,
               frameon=False, fontsize=9.5)
    fig.suptitle(f"{DATASETS[dataset]} — one unpruned tree on one GPU, ./tree against the "
                 "GPU libraries", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    run_text = " or ".join(map(str, runs))
    fig.text(0.01, 0.012,
             f"Time: median of {run_text} runs, from host memory to the finished tree (copy to the "
             "GPU, binning or presort, growing). VRAM: peak GPU memory above idle, CUDA context and "
             "memory pools included (nvidia-smi, every 20 ms).\n"
             "Bins: the libraries split only at the edges of per-feature quantile bins; ./tree tests "
             "every distinct value. Same as ./tree: share of test rows classified as by ./tree's tree.\n"
             "C4.5: ./tree follows Quinlan's C4.5 (gain ratio, MDL-corrected thresholds, at least 2 rows "
             "per side); cuML's closest is plain entropy (information gain). XGBoost has no entropy "
             "criterion, so it is in CART only.",
             fontsize=8.5, color=MUTED, style="italic", va="bottom")
    fig.subplots_adjust(left=0.15, right=0.99, top=1 - 1.25 / fig.get_figheight(),
                        bottom=1.05 / fig.get_figheight())
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    fields = ["dataset", "node", "case", "impl", "bins", "runs", "median_s", "min_s", "max_s",
              "nodes", "depth", "train_accuracy", "test_accuracy", "agreement", "gpu_peak_mib",
              "root", "version"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "tables.md").write_text(tables(rows))
    for dataset in DATASETS:
        if any(r["dataset"] == dataset for r in rows):
            chart(rows, dataset, out / f"chart-{dataset}.png")


if __name__ == "__main__":
    main()
