#!/usr/bin/env python3
"""One chart of every run's training time (log scale) with the tree's depth,
node count and test accuracy, plus summary.csv.

  chart.py RESULTS_DIR OUT_DIR    writes chart.png, chart.svg, summary.csv
"""
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HOSTS = {"dell1": ("#2a78d6", "dell1 · 2× Xeon Gold 6542Y (48 cores)"),
         "dell2": ("#1baf7a", "dell2 · 2× Xeon Gold 6542Y (48 cores)"),
         "hopper": ("#eb6834", "hopper · H100 PCIe 80 GB")}
CASES = {"cart": "CART (Gini)", "c45": "C4.5 (cuML: entropy)"}
ORDER = ["tree-serial", "tree-parallel", "tree-cuda", "sklearn", "rpart", "j48", "yadt-1t",
         "yadt-48t", "cuml", "xgboost"]


def label(m):
    if m["impl"] == "tree":
        return {"serial": "./tree, 1 thread", "parallel": f"./tree, {m['threads']} threads",
                "cuda": "./tree --cuda"}[m["backend"]]
    extra = {"cuml": ", 128 bins", "xgboost": ", 256 bins"}.get(m["impl"], "")
    if m["impl"] == "yadt":
        extra = f", {m['threads']} thread{'s' if m['threads'] > 1 else ''}"
    if m["impl"] == "cuml" and m["case"] == "c45":
        extra = ", entropy, 128 bins"
    return m["label"] + extra


def key(m):
    k = m["name"].rsplit("-", 1)[0]
    return ORDER.index(k) if k in ORDER else 99


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    metas = [json.load(open(p))["meta"] for p in sorted(src.glob("*.json"))]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "implementation", "case", "host", "hardware", "threads", "median_s", "min_s",
                    "max_s", "runs", "depth", "nodes", "leaves", "train_accuracy", "test_accuracy",
                    "version", "command"])
        for m in sorted(metas, key=lambda m: (m["case"], key(m))):
            w.writerow([m["name"], label(m), m["case"], m["host"], m["gpu"] or m["cpu"], m["threads"],
                        f"{m['median_seconds']:.6f}", f"{m['min_seconds']:.6f}", f"{m['max_seconds']:.6f}",
                        len(m["times"]), m["depth"], m["nodes"], m["leaves"],
                        f"{m['train_accuracy']:.6f}", f"{m['test_accuracy']:.6f}", m["version"],
                        m["command"]])

    groups = [(c, sorted([m for m in metas if m["case"] == c], key=key)) for c in CASES]
    groups = [(c, ms) for c, ms in groups if ms]
    heights = [len(ms) for _, ms in groups]
    fig = plt.figure(figsize=(14, 1.9 + 0.42 * sum(heights) + 0.9 * len(groups)))
    grid = fig.add_gridspec(len(groups), 2, width_ratios=(1.0, 0.62), wspace=0.02, hspace=0.38,
                            height_ratios=heights)
    lo = min(m["min_seconds"] for m in metas) / 2.5
    hi = max(m["max_seconds"] for m in metas) * 4
    for row, (case, ms) in enumerate(groups):
        ax = fig.add_subplot(grid[row, 0])
        info = fig.add_subplot(grid[row, 1], sharey=ax)
        y = list(range(len(ms)))[::-1]
        for m, p in zip(ms, y):
            color = HOSTS.get(m["host"], (MUTED, m["host"]))[0]
            ax.barh(p, m["median_seconds"] - lo, left=lo, height=0.62, color=color)
            ax.plot([m["min_seconds"], m["max_seconds"]], [p, p], color=INK, lw=1, alpha=0.55)
            s = m["median_seconds"]
            text = f"{s * 1e3:.1f} ms" if s < 1 else f"{s:.2f} s"
            ax.text(m["max_seconds"] * 1.12, p, text, va="center", fontsize=9, color=INK,
                    fontweight="bold" if m["impl"] == "tree" else "normal")
            info.text(0.02, p, f"{m['depth']}", va="center", fontsize=9.5, color=INK,
                      transform=info.get_yaxis_transform())
            info.text(0.17, p, f"{m['nodes']:,}", va="center", fontsize=9.5, color=INK,
                      transform=info.get_yaxis_transform())
            info.text(0.40, p, f"{100 * m['test_accuracy']:.2f}%", va="center", fontsize=9.5, color=INK,
                      transform=info.get_yaxis_transform())
            info.text(0.63, p, f"{100 * m['train_accuracy']:.2f}%", va="center", fontsize=9.5,
                      color=MUTED, transform=info.get_yaxis_transform())
            info.text(0.86, p, m["host"], va="center", fontsize=9.5, color=MUTED,
                      transform=info.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_yticks(y, [label(m) for m in ms], fontsize=10, color=INK)
        for t, m in zip(ax.get_yticklabels(), ms):
            t.set_fontweight("bold" if m["impl"] == "tree" else "normal")
        ax.set_ylim(-0.6, len(ms) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.8, which="major")
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_title(f"{CASES[case]}: training time, log scale",
                     loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
        info.axis("off")
        for x, h in ((0.02, "depth"), (0.17, "nodes"), (0.40, "test acc."), (0.63, "train acc."),
                     (0.86, "server")):
            info.text(x, len(ms) - 0.35, h, fontsize=9, color=MUTED, transform=info.get_yaxis_transform(),
                      va="bottom")
        if row == len(groups) - 1:
            ax.set_xlabel("seconds", color=MUTED, fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in HOSTS.values()]
    fig.legend(handles, [t for _, t in HOSTS.values()], loc="upper left", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.01, 1 - 0.62 / fig.get_figheight()))
    fig.suptitle("Chess 3×3, 1M rows (800,000 train / 200,000 test): one tree, depth ≤ 10, no pruning",
                 x=0.01, ha="left", y=1 - 0.18 / fig.get_figheight(), fontsize=13, fontweight="bold",
                 color=INK)
    fig.text(0.01, 0.1 / fig.get_figheight(),
             "Bars: median of 3 runs, whiskers: min–max. Times are each tool's own training time (data loading not timed).\n"
             "Accuracies, depth and nodes come from the exported tree. cuML and XGBoost bin each feature first (128 / 256 bins); "
             "the others test every distinct value.", fontsize=8.5, color=MUTED, style="italic")
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 1.4 / fig.get_figheight(),
                        bottom=1.0 / fig.get_figheight())
    for ext in ("png", "svg"):
        fig.savefig(out / f"chart.{ext}", dpi=150)
    print(out / "chart.png")


if __name__ == "__main__":
    main()
