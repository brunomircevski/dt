#!/usr/bin/env python3
"""Candidate chart styles for the paper, as vector PDFs sized for LNCS
(text width 122 mm = 4.80 in; fonts rendered by LaTeX, 7-8 pt, so they match
the text). Every chart is built from the benchmark results in bench/results/.

  bench/.venv/bin/python paper/charts.py          # writes paper/figs/test-*.pdf

Styles:
  ladder    speed-up staircase: each dataset's time through the four steps of
            this work (log scale), with the step's speed-up range above it
  dots      Cleveland dot plot: every implementation per dataset on one log
            axis, this work filled and the existing ones hollow, run range as
            whiskers, the advantage over the best existing one at the right
  heatmap   hardware matrix: time per dataset (rows) on every node and backend
            (columns), colour = how many times slower than the row's best
  scaling   thread scaling: speed-up against threads, two nodes, two datasets,
            CART and C4.5, ideal line and the 48-thread efficiency
  pareto    time-accuracy-memory bubbles for one dataset: every implementation
  profile   100 % stacked bars: where the training time goes, per backend
  bars      grouped bars, log scale: the five implementations on every dataset
  energy    time, mean power and total energy of one C4.5 tree on ECBDL'14 15M,
            epyc and hopper, from bench/power/results-*/
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "bench" / "results"
OUT = ROOT / "paper" / "figs"
sys.path.insert(0, str(ROOT / "bench"))

FULL, HALF = 4.80, 2.35  # inches: LNCS \textwidth and a bit under half of it

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Implementation colours (reference palette slots, as in bench/chart.py), and a
# marker each so the charts survive grayscale printing.
STYLE = {
    "simple":  ("Serial Simple", "#d6455a", "v"),
    "serial":  ("Serial Optimized", "#7f4fc9", "s"),
    "socket":  ("Multicore CPU 24T", "#6fa8e8", "D"),
    "cpu":     ("Multicore CPU", "#2a78d6", "D"),
    "gpuonly": ("GPU-Based", "#1baf7a", "^"),
    "cuda":    ("Hybrid", "#008300", "o"),
    "sklearn": ("scikit-learn", "#eb6834", "o"),
    "j48":     ("Weka J48", "#eda100", "s"),
    "cuml":    ("cuML", "#e87ba4", "^"),
    "xgboost": ("XGBoost", "#9a3412", "D"),
}
DATA = {  # key -> (label, results dir, run suffix)
    "susy": ("SUSY", "cluster-susy", "susy"),
    "higgs": ("HIGGS", "cluster-higgs", "higgs"),
    "mnist8m": ("MNIST8M", "cluster-mnist8m", "mnist8m"),
    "ecbdl": ("ECBDL'14", "cluster-ecbdl", "ecbdl"),
    "ecbdl_15m": ("ECBDL'14 15M", "cluster-ecbdl", "ecbdl_15m"),
    "sorel": ("SOREL-20M", "cluster-sorel", "sorel"),
    "sorel_4m": ("SOREL 4M", "cluster-sorel", "sorel_4m"),
}
PROTO = {"cart": "cart_depth10", "c45": "c45_depth10_unpruned"}
ALGO = {"cart": "CART", "c45": "C4.5"}


def tex(s):
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def summary(path):
    """-> {protocol: row} of a run directory's summary.csv (numbers as floats)."""
    rows = {}
    f = path / "summary.csv"
    if not f.exists():
        return rows
    for r in csv.DictReader(open(f)):
        if r["status"] != "ok":
            continue
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
        rows[r["protocol"]] = r
    return rows


def ours(dataset, mode, node, algo):
    """This work's summary row for a dataset/backend/node (mode: simple, serial, socket, cpu,
    cuda, gpuonly), or None."""
    _, d, suffix = DATA[dataset]
    return summary(RES / d / f"{mode}-{node}-{suffix}").get(PROTO[algo])


def h100(dataset, mode, algo):
    """(row, estimated): the H100 run, or, where there is none (SOREL 4M), the A100 run
    scaled by the H100/A100 ratio of ECBDL'14 15M (a rough demo estimate)."""
    r = ours(dataset, mode, "hopper", algo)
    if r:
        return r, False
    a = ours(dataset, mode, "ampere", algo)
    if not a:
        return None, False
    k = ours("ecbdl_15m", mode, "hopper", algo)["time_mean"] / ours("ecbdl_15m", mode, "ampere", algo)["time_mean"]
    return dict(a, time_mean=a["time_mean"] * k), True


def cpu_other(dataset, impl, algo):
    return summary(RES / "cluster-cpu-vs-others" / f"vs-epyc-{dataset}-{impl}-{algo}").get(PROTO[algo])


def gpu_other(dataset, impl, algo, host="hopper", trees=1):
    f = RES / "cluster-gpu-vs-others" / dataset / host / "results.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r["impl"] == impl and r["case"] == algo and r.get("trees") == trees]
    if not rows:
        return None
    t = [r["train_seconds"] for r in rows]
    last = rows[-1]
    return {"time_mean": statistics.mean(t), "time_min": min(t), "time_max": max(t),
            "test_accuracy": last.get("test_accuracy"), "nodes": last.get("nodes"),
            "gpu_peak": last.get("gpu_peak_mib", 0) * 2**20}


def sec(v, _=None):
    if v >= 1000:
        return f"{v / 1000:g}k"
    if v >= 1:
        return f"{v:g}"
    return f"{v:g}".lstrip("0") if v >= 0.01 else f"{v:g}"


def tsec(v):
    """Compact cell text for seconds."""
    if v >= 1000:
        return f"{v / 1000:.1f}k" if v < 10000 else f"{v / 1000:.0f}k"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}" if v < 20 else f"{v:.0f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}".lstrip("0")


def log_axis(ax, axis="y", lo=None, hi=None):
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    a.set_major_formatter(FuncFormatter(sec))
    a.set_minor_formatter(NullFormatter())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"test-{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"test-{name}.pdf")


# ---------------------------------------------------------------- 1. ladder
def ladder():
    datasets = ["susy", "higgs", "ecbdl_15m", "mnist8m", "sorel_4m"]
    stages = [("simple", "epyc"), ("serial", "epyc"), ("cpu", "epyc"), ("gpuonly", "hopper"), ("cuda", "hopper")]
    names = ["Serial\nSimple", "Serial\nOptimized", "Multicore\nCPU", "GPU-\nBased", "Hybrid"]
    n = len(stages)
    shades = ["#2a78d6", "#eb6834", "#1baf7a", "#7f4fc9", "#eda100"]
    marks = ["o", "s", "D", "^", "v"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1), sharey=True)
    for ax, algo in zip(axes, ["cart", "c45"]):
        times = {}
        for ds, c, m in zip(datasets, shades, marks):
            ys = []
            for mode, node in stages:
                r = h100(ds, mode, algo)[0] if node == "hopper" else ours(ds, mode, node, algo)
                ys.append(r["time_mean"] if r else np.nan)
            times[ds] = ys
            ax.plot(range(n), ys, color=c, marker=m, ms=3.2, lw=0.9, label=tex(DATA[ds][0]), zorder=3)
        # step speed-ups: range over datasets, at the top between stages
        for i in range(n - 1):
            f = [times[d][i] / times[d][i + 1] for d in datasets
                 if np.isfinite(times[d][i]) and np.isfinite(times[d][i + 1])]
            fmt = (lambda v: f"{v:.0f}") if max(f) >= 10 else (lambda v: f"{v:.1f}")
            rng = fmt(min(f)) if fmt(min(f)) == fmt(max(f)) else f"{fmt(min(f))}--{fmt(max(f))}"
            ax.text(i + 0.5, 2.6e4 if i % 2 == 0 else 7e3, rf"$\times${rng}",
                    ha="center", va="center", fontsize=5.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=GRID, lw=0.4))
        total = [times[d][0] / times[d][-1] for d in datasets]
        ax.set_title(rf"\textbf{{{ALGO[algo]}}}: overall $\times${min(total):.0f}--{max(total):.0f}",
                     loc="left", pad=3)
        log_axis(ax, "y")
        ax.set_ylim(0.05, 6e4)
        ax.axvspan(2.6, n - 0.7, color="#f3f3f3", zorder=-1, lw=0)
        ax.text(n - 1.5, 2.6e4, "H100", ha="center", va="center", fontsize=6, color=MUTED)
        ax.set_xticks(range(n), names)
        ax.set_xlim(-0.3, n - 0.7)
        ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        ax.tick_params(axis="x", length=0, labelsize=6.2)
    axes[0].set_ylabel("training time (s)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=6.5,
               handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "ladder")


# ---------------------------------------------------------------- 2. dots
def dots():
    datasets = ["susy", "higgs", "ecbdl_15m"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.45), sharey=True)
    for ax, algo in zip(axes, ["cart", "c45"]):
        impls = [("serial", lambda d: ours(d, "serial", "epyc", algo)),
                 ("cpu", lambda d: ours(d, "cpu", "epyc", algo)),
                 ("cuda", lambda d: ours(d, "cuda", "hopper", algo)),
                 ("sklearn" if algo == "cart" else "j48",
                  lambda d: cpu_other(d, "sklearn" if algo == "cart" else "j48", algo)),
                 ("cuml", lambda d: gpu_other(d, "cuml", algo))]
        if algo == "cart":
            impls.append(("xgboost", lambda d: gpu_other(d, "xgboost", algo)))
        mine = {"serial", "cpu", "cuda"}
        for y, ds in enumerate(datasets):
            ax.axhline(y, color=GRID, lw=0.4, zorder=0)
            best_mine, best_other = np.inf, (np.inf, "")
            for k, fn in impls:
                r = fn(ds)
                if not r:
                    continue
                label, c, m = STYLE[k]
                ax.plot([r["time_min"], r["time_max"]], [y, y], color=c, lw=1.6, zorder=2,
                        solid_capstyle="butt")
                ax.plot(r["time_mean"], y, m, color=c, mfc=c if k in mine else "white",
                        mew=0.8, ms=4.2, zorder=3)
                if k in mine:
                    best_mine = min(best_mine, r["time_mean"])
                elif r["time_mean"] < best_other[0]:
                    best_other = (r["time_mean"], STYLE[k][0])
            ax.text(1.02, y, rf"$\times${best_other[0] / best_mine:.1f}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=6.5, color=INK)
        log_axis(ax, "x")
        ax.set_xlim(0.05, 4e4)
        ax.set_ylim(len(datasets) - 0.5, -0.6)
        ax.set_yticks(range(len(datasets)), [tex(DATA[d][0]) for d in datasets])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(rf"\textbf{{{ALGO[algo]}}}", loc="left", pad=3)
    fig.supxlabel("training time (s), log scale", fontsize=7, y=-0.1)
    handles = [Line2D([], [], ls="", marker=STYLE[k][2], color=STYLE[k][1],
                      mfc=STYLE[k][1] if k in {"serial", "cpu", "cuda"} else "white", mew=0.8,
                      label=STYLE[k][0]) for k in ["serial", "cpu", "cuda", "sklearn", "j48", "cuml", "xgboost"]]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.3),
               columnspacing=0.9, handletextpad=0.2)
    fig.text(0.995, 0.93, r"\textit{vs best existing}", ha="right", fontsize=6, color=MUTED)
    fig.subplots_adjust(wspace=0.22)
    save(fig, "dots")


# ---------------------------------------------------------------- 3. heatmap
def heatmap():
    rows = ["susy", "higgs", "mnist8m", "ecbdl_15m", "ecbdl", "sorel_4m", "sorel"]
    cols = [("serial", "1 thread\nEPYC"), ("cpu", "2 sockets\nEPYC"), ("gpuonly", "GPU-Based\nH100"),
            ("cuda", "Hybrid\nH100")]
    cmap = LinearSegmentedColormap.from_list("slow", ["#1a6a3a", "#7cc49a", "#f4f1e6", "#eda100", "#b23a12"])
    cmap.set_bad("#f3f2ee")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.0), sharey=True)
    for ax, algo in zip(axes, ["cart", "c45"]):
        M = np.full((len(rows), len(cols)), np.nan)
        E = np.zeros_like(M, dtype=bool)
        for i, ds in enumerate(rows):
            for j, (mode, _) in enumerate(cols):
                if mode in ("cuda", "gpuonly"):
                    r, E[i, j] = h100(ds, mode, algo)
                else:
                    r = ours(ds, mode, "epyc", algo)
                if r:
                    M[i, j] = r["time_mean"]
        rel = np.log10(M / np.nanmin(M, axis=1, keepdims=True))
        ax.imshow(rel, cmap=cmap, vmin=0, vmax=2.5, aspect="auto", interpolation="nearest")
        for i in range(len(rows)):
            for j in range(len(cols)):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "--", ha="center", va="center", fontsize=6, color=MUTED)
                    continue
                txt = tsec(M[i, j]) + ("*" if E[i, j] else "")
                ax.text(j, i, (r"\textbf{%s}" if rel[i, j] < 1e-9 else "%s") % txt, ha="center",
                        va="center", fontsize=6.3,
                        color="white" if rel[i, j] > 1.9 or rel[i, j] < 0.15 else INK)
        ax.set_yticks(range(len(rows)), [tex(DATA[d][0]) for d in rows], fontsize=6.5)
        ax.set_xticks(range(len(cols)), [c for _, c in cols], fontsize=6)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=0.8)
        ax.tick_params(which="minor", length=0)
        ax.axvline(1.5, color=INK, lw=0.7)
        ax.set_xlabel(rf"\textbf{{{ALGO[algo]}}}", fontsize=7.5, labelpad=3)
    fig.subplots_adjust(wspace=0.05, right=0.88)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2.5))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.9, 0.11, 0.018, 0.77]))
    cb.set_ticks([0, 0.5, 1, 1.5, 2, 2.5], labels=["1", "3", "10", "30", "100", "300"], fontsize=6)
    cb.set_label(r"$\times$ slower than the row's fastest (bold)", fontsize=6.5)
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(length=1.5, width=0.3)
    save(fig, "heatmap")


# ---------------------------------------------------------------- 4. scaling
def scaling():
    sets = [("higgs", "o", "HIGGS", "#2a78d6"), ("mnist8m", "s", "MNIST8M", "#eb6834")]
    for node, title in [("epyc", "epyc1: 2 × EPYC 9255"), ("dell", "dell1: 2 × Xeon Gold 6542Y")]:
        fig, ax = plt.subplots(figsize=(HALF, 1.85))
        ax.plot([1, 48], [1, 48], color=GRID, lw=0.8, zorder=0)
        ax.text(10, 13, "ideal", fontsize=6, color=MUTED, rotation=37)
        ax.axvline(24, color=MUTED, lw=0.5, ls=(0, (1, 2)), zorder=0)
        ax.text(25, 1.5, "2nd socket $\\rightarrow$", fontsize=5.5, color=MUTED, va="bottom")
        for ds, m, dl, c in sets:
            f = RES / "cluster-cpu-scaling" / f"cpu-{node}-{ds}" / "scaling.csv"
            end = {}
            for algo in ("cart", "c45"):
                rr = [x for x in csv.DictReader(open(f)) if x["protocol"] == PROTO[algo]]
                end[algo] = float(rr[-1]["speedup"])
            for algo, ls in [("cart", "-"), ("c45", "--")]:
                other = end["c45" if algo == "cart" else "cart"]
                r = [x for x in csv.DictReader(open(f)) if x["protocol"] == PROTO[algo]]
                t = [int(x["threads"]) for x in r]
                sp = [float(x["speedup"]) for x in r]
                ax.plot(t, sp, ls=ls, color=c, lw=0.9, marker=m, ms=2.4, mfc="white" if algo == "c45" else c,
                        mew=0.6, markevery=[0, 3, 7, 13])
                t1 = float(r[0]["time_mean"])
                ax.annotate(rf"{sp[-1]:.0f}$\times$ ({100 * sp[-1] / 48:.0f}\%)", (t[-1], sp[-1]),
                            xytext=(2, 3.5 if end[algo] >= other else -3.5), textcoords="offset points",
                            fontsize=5.3, color=c, va="center")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 48)
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48])
        ax.set_xlabel("threads")
        ax.set_ylabel("speed-up over 1 thread")
        ax.set_title(title, loc="left", pad=3, fontsize=7)
        ax.grid(color=GRID, lw=0.3, zorder=0)
        h = [Line2D([], [], color=c, lw=0.9, marker=m, ms=2.5, label=dl) for _, m, dl, c in sets]
        h += [Line2D([], [], color=MUTED, lw=0.9, ls="-", label="CART"),
              Line2D([], [], color=MUTED, lw=0.9, ls="--", label="C4.5")]
        ax.legend(handles=h, loc="upper left", fontsize=5.8, labelspacing=0.2, handlelength=1.8)
        save(fig, f"scaling-{node}")


# ---------------------------------------------------------------- 7. depth
def depth():
    """Hybrid against GPU only by depth limit (A100, hybrid-vs-gpu-only): grouped bars, linear."""
    root = RES / "hybrid-vs-gpu-only"
    depths = ["5", "10", "15", "20", "25", "30", "none"]
    proto = {"cart": [f"cart_depth{d}" for d in depths[:-1]] + ["cart_full_nodepth"],
             "c45": [f"c45_depth{d}_unpruned" for d in depths[:-1]] + ["c45_unpruned"]}
    c_hy, c_go = STYLE["cuda"][1], "#b9c2bc"
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 2.75), sharex=True,
                             gridspec_kw={"hspace": 0.32, "wspace": 0.12})
    x = np.arange(len(depths))
    w = 0.38
    for row, ds, name in [(0, "airline", "Airline"), (1, "mnist8m", "MNIST8M")]:
        hy, go = summary(root / f"cuda-ampere-{ds}"), summary(root / f"gpuonly-ampere-{ds}")
        top = max(go[p]["time_mean"] for a in proto for p in proto[a])
        for col, algo in enumerate(["cart", "c45"]):
            ax = axes[row, col]
            th = np.array([hy[p]["time_mean"] for p in proto[algo]])
            tg = np.array([go[p]["time_mean"] for p in proto[algo]])
            ax.bar(x - w / 2, th, w, color=c_hy, lw=0, label="Hybrid", zorder=2)
            ax.bar(x + w / 2, tg, w, color=c_go, lw=0, label="GPU-Based", zorder=2)
            for i in range(len(x)):
                r = tg[i] / th[i]
                ax.text(x[i], max(th[i], tg[i]) + 0.02 * top, f"{r:.1f}" + r"$\times$",
                        ha="center", va="bottom", fontsize=5.3, color=INK)
            ax.set_ylim(0, top * 1.15)
            ax.grid(axis="y", color=GRID, lw=0.3, zorder=0)
            ax.tick_params(axis="x", length=0)
            ax.set_xticks(x, depths)
            ax.set_title(rf"\textbf{{{name}}}, {ALGO[algo]}", loc="left", pad=2, fontsize=7)
            if col:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_ylabel("time (s)")
        axes[row, 1].sharey(axes[row, 0])
    fig.supxlabel("depth limit", fontsize=7, y=0.0)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, bbox_to_anchor=(0.91, 1.02), frameon=False)
    save(fig, "depth")


# ---------------------------------------------------------------- 5. pareto
def pareto(ds="higgs"):
    # broken y axis: the CART trees (top) and the C4.5/entropy trees (bottom) are
    # 7 points apart; each band is zoomed so the small accuracy gaps show
    fig, (top, bot) = plt.subplots(2, 1, figsize=(HALF, 2.05), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})
    pts = []
    for algo in ["cart", "c45"]:
        for k, node in [("serial", "epyc"), ("cpu", "epyc"), ("cuda", "hopper")]:
            r = ours(ds, k, node, algo)
            mem = r["gpu_peak"] if k == "cuda" else r["peak_rss"]
            pts.append((k, algo, r["time_mean"], r["test_accuracy"], mem, True))
        r = cpu_other(ds, "sklearn", algo)  # sklearn's C4.5 stand-in: entropy tree
        if r:
            pts.append(("sklearn", algo, r["time_mean"], r["test_accuracy"], r["peak_rss"], False))
        if algo == "c45":
            r = cpu_other(ds, "j48", algo)
            pts.append(("j48", algo, r["time_mean"], r["test_accuracy"], r["peak_rss"], False))
        for k in (["cuml", "xgboost"] if algo == "cart" else ["cuml"]):
            r = gpu_other(ds, k, algo)
            if r:
                pts.append((k, algo, r["time_mean"], r["test_accuracy"], r["gpu_peak"], False))
    for k, algo, t, acc, mem, mine in pts:
        label, c, m = STYLE[k]
        ax = top if acc > 0.68 else bot
        size = 5 + 7 * (mem / 2**30)  # marker area grows with peak memory (GB)
        ax.scatter(t, 100 * acc, s=size, marker="o" if algo == "cart" else "s",
                   color=c if algo == "cart" else "white", edgecolors=c, linewidths=0.8,
                   alpha=0.9, zorder=3 if mine else 2)
    top.set_ylim(70.43, 70.56)
    bot.set_ylim(63.69, 63.78)
    top.set_yticks([70.45, 70.5, 70.55])
    bot.set_yticks([63.7, 63.75])
    for ax in (top, bot):
        log_axis(ax, "x")
        ax.grid(color=GRID, lw=0.3, zorder=0)
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", length=0)
    d = dict(transform=top.transAxes, color=INK, clip_on=False, lw=0.5)
    top.plot([-0.03, 0.03], [-0.02, 0.02], **d)
    d["transform"] = bot.transAxes
    bot.plot([-0.03, 0.03], [0.98, 1.02], **d)
    bot.set_xlim(0.08, 2000)
    bot.set_xlabel("training time (s)")
    fig.supylabel("test accuracy (\\%)", fontsize=8, x=-0.1)
    top.text(0.02, 0.06, "CART + sklearn/cuML entropy", transform=top.transAxes, fontsize=5.5, color=MUTED)
    bot.text(0.02, 0.06, "C4.5 (gain ratio, exact)", transform=bot.transAxes, fontsize=5.5, color=MUTED)
    h = [Line2D([], [], ls="", marker="o", color=STYLE[k][1], ms=3.5, label=STYLE[k][0])
         for k in ["serial", "cpu", "cuda", "sklearn", "j48", "cuml", "xgboost"]]
    h += [Line2D([], [], ls="", marker="o", color=MUTED, ms=3.5, label="CART"),
          Line2D([], [], ls="", marker="s", mfc="white", color=MUTED, ms=3.5, label="C4.5/entr.")]
    top.legend(handles=h, loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=5.8,
               labelspacing=0.25, handletextpad=0.1)
    bot.text(1.04, 0.0, DATA[ds][0] + "\nmarker area\n" + r"$\propto$ peak RAM" + "\nor VRAM",
             transform=bot.transAxes, fontsize=5.5, color=MUTED, va="bottom", linespacing=1.1)
    save(fig, "pareto")


# ---------------------------------------------------------------- 6. profile
def profile():
    import profile_share_chart as P
    datasets = ["higgs", "airline", "ecbdl_15m", "chess_1b"]
    labels = {"higgs": "HIGGS", "airline": "Airline", "ecbdl_15m": "ECBDL", "chess_1b": "Chess"}
    modes = [("cpu-epyc", "EPYC"), ("cpu-dell", "Xeon"), ("gpuonly-hopper", "GPU-Based"),
             ("cuda-hopper", "Hybrid")]
    cases = P.load(RES / "cluster-profiling", datasets)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.25), sharey=True)
    for ax, (proto, algo) in zip(axes, P.PROTOCOLS):
        y, ticks = 0, []
        for ds in datasets:
            for mode, ml in modes:
                case = cases.get((mode, ds, proto))
                if case:
                    parts = case["parts"]
                    total = sum(parts.values())
                    left = 0
                    for step, _, color in P.STEPS:
                        v = sum(val for (s, _), val in parts.items() if s == step) / total
                        ax.barh(y, v, left=left, color=color, height=0.78, lw=0)
                        if v > 0.12:
                            ax.text(left + v / 2, y, f"{100 * v:.0f}", ha="center", va="center",
                                    fontsize=5, color=P.text_color(color))
                        left += v
                    gpu = sum(val for (_, d), val in parts.items() if d == "GPU") / total
                    if gpu > 0:
                        ax.plot([gpu], [y - 0.39], marker="v", color=INK, ms=2.2, clip_on=False)
                    ax.text(1.015, y, tsec(case["row"]["train_seconds"]) + " s", va="center",
                            fontsize=5.3, transform=ax.get_yaxis_transform())
                ticks.append((y, ml))
                y += 1
            y += 0.45
        ax.set_yticks([t for t, _ in ticks], [l for _, l in ticks], fontsize=5.5)
        # dataset labels as group headers on the left panel
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100\\%"])
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.spines["left"].set_visible(False)
        ax.set_title(rf"\textbf{{{algo}}}", loc="left", pad=3)
    axes[0].invert_yaxis()
    fig.supxlabel("share of training time", fontsize=7, y=0.0)
    for i, ds in enumerate(datasets):
        yc = i * (len(modes) + 0.45) + 1.5
        axes[0].text(-0.25, yc, labels[ds], rotation=90, ha="center", va="center", fontsize=6,
                     transform=axes[0].get_yaxis_transform())
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=tex(n).replace("↔", r"$\leftrightarrow$"))
         for _, n, c in P.STEPS]
    h.append(Line2D([], [], ls="", marker="v", color=INK, ms=3, label="GPU-busy share"))
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12),
               handlelength=1.0, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "profile")

# ---------------------------------------------------------------- 8. bars
def bars():
    """Grouped bars, log scale: this work's five implementations on every dataset (CPU on
    epyc1, GPU on the H100), value above each bar, no bar where there is no run."""
    datasets = ["susy", "higgs", "mnist8m", "ecbdl", "ecbdl_15m", "sorel", "sorel_4m"]
    impls = [("simple", "epyc"), ("serial", "epyc"), ("cpu", "epyc"), ("gpuonly", "hopper"),
             ("cuda", "hopper")]
    labels = {"simple": "Serial Simple, 1 thread", "serial": "Serial Optimized, 1 thread",
              "cpu": "Multicore CPU, 48 threads", "gpuonly": "GPU-Based, H100", "cuda": "Hybrid, H100"}
    n = len(impls)
    w = 0.72 / n
    x = np.arange(len(datasets))
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.6), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax, algo in zip(axes, ["cart", "c45"]):
        for j, (mode, node) in enumerate(impls):
            _, c, _ = STYLE[mode]
            for i, ds in enumerate(datasets):
                xi = x[i] + (j - (n - 1) / 2) * w
                r = ours(ds, mode, node, algo)
                if not r:
                    continue
                t = r["time_mean"]
                ax.bar(xi, t, w * 0.9, color=c, ec=INK, lw=0.3, zorder=2,
                       label=labels[mode] if i == 0 else None)
                ax.text(xi, t * 1.12, tsec(t), rotation=90, ha="center", va="bottom", fontsize=4.6,
                        color=INK)
        log_axis(ax, "y")
        ax.set_ylim(0.03, 8e4)
        ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
        ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        ax.set_ylabel(rf"\textbf{{{ALGO[algo]}}}: time (s)")
        ax.tick_params(axis="x", length=0)
        for i in range(1, len(datasets)):
            ax.axvline(i - 0.5, color=GRID, lw=0.4, zorder=0)
    names = {"ecbdl_15m": "ECBDL'14\n15M samples", "sorel_4m": "SOREL\n4M samples"}
    axes[1].set_xticks(x, [tex(names.get(d, DATA[d][0])) for d in datasets], fontsize=5.4)
    axes[1].set_xlim(-0.5, len(datasets) - 0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.885), fontsize=5.8,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    save(fig, "bars")


# ---------------------------------------------------------------- power (energy per tree)
# bench/power/results-<node>/summary.csv, ECBDL'14 15M, depth <= 10, unpruned. Whole-server
# energy at the PSU; components: CPU packages (RAPL), the GPU in use (NVML), the rest.
POWER = ROOT / "bench" / "power"
P_NODES = [("epyc1", "epyc"), ("dell4", "dell"), ("ampere", "ampere"), ("hopper", "hopper")]
P_IMPLS = [("simple", "simple"), ("serial", "serial"), ("parallel", "cpu"), ("gpuonly", "gpuonly"),
           ("cuda", "cuda")]  # (power_bench name, STYLE key)
P_MARK = {"epyc": "o", "dell": "s", "ampere": "^", "hopper": "D"}
P_GPU = {"ampere": "A100", "hopper": "H100"}


def power_rows():
    """{(STYLE key, node short name, algo): row with floats} from every power result."""
    out = {}
    for full, short in P_NODES:
        f = POWER / f"results-{short}" / "summary.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            v = {}
            for k, x in r.items():
                try:
                    v[k] = float(x)
                except (TypeError, ValueError):
                    v[k] = x
            key = dict(P_IMPLS)[r["impl"]]
            out[(key, short, r["algo"])] = v
    return out


def p_cases(rows):
    """[(STYLE key, node)] in chart order: implementation, then node."""
    return [(k, n) for _, k in P_IMPLS for _, n in P_NODES if (k, n, "cart") in rows]


def kj(v):
    """Energy in J -> TeX text."""
    if v >= 1e6:
        return rf"{v / 1e6:.2f}\,MJ"
    if v >= 1e5:
        return rf"{v / 1e3:.0f}\,kJ"
    if v >= 1e4:
        return rf"{v / 1e3:.1f}\,kJ"
    return rf"{v / 1e3:.2f}\,kJ"


def joule_axis(ax, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    names = {1e3: "1\\,kJ", 1e4: "10\\,kJ", 1e5: "100\\,kJ", 1e6: "1\\,MJ", 1e7: "10\\,MJ"}
    a.set_major_locator(FixedLocator(list(names)))
    a.set_major_formatter(FuncFormatter(lambda v, _: names.get(v, "")))
    a.set_minor_formatter(NullFormatter())


def energy(algo="c45", cpu_nodes=("epyc", "dell"), gpu_nodes=("ampere", "hopper")):
    """Time, power and total energy of one tree, one row per implementation x node (CPU
    implementations on the CPU nodes, GPU ones on the GPU nodes), four aligned panels: time
    per tree (log); mean power of the whole server (linear; the lighter part is the idle
    server's power); mean power of the CPU packages (blue stripes) + the GPU in use while
    training (linear; no idle split); total energy per tree at the PSU (log;
    idle included). The number at the end of each bar is its total."""
    from matplotlib.colors import to_rgba
    rows = power_rows()
    cases = [(k, n) for k, n in p_cases(rows)
             if n in (gpu_nodes if k in ("gpuonly", "cuda") else cpu_nodes)]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.2), sharey=True,
                             gridspec_kw={"wspace": 0.16, "width_ratios": [1, 1, 0.95, 0.9]})
    ax_t, ax_p, ax_c, ax_e = axes
    # CPU+GPU panel: the CPU packages in subtle blue stripes, the GPU in the implementation's colour
    CPU_BLUE = "#2a78d6"
    stripes = dict(color=to_rgba(CPU_BLUE, 0.12), ec=CPU_BLUE, lw=0, hatch="//////")
    plt.rcParams["hatch.linewidth"] = 0.4  # thin stripes (read when the bars are made)
    y = np.arange(len(cases))[::-1].astype(float)
    hgt, fs = 0.62, 4.9
    for yi, (k, node) in zip(y, cases):
        r = rows[(k, node, algo)]
        c, lc = STYLE[k][1], to_rgba(STYLE[k][1], 0.35)
        t, p, e = r["train_s"], r["power_W"], r["psu_J"]
        ax_t.barh(yi, t - 1, left=1, height=hgt, color=c, lw=0)
        ax_t.text(t * 1.2, yi, (f"{t:,.0f}".replace(",", "{,}") if t >= 100 else tsec(t)) + r"\,s",
                  va="center", fontsize=fs)
        idle_w = r["idle_W"]
        ax_p.barh(yi, idle_w, height=hgt, color=lc, lw=0)
        ax_p.barh(yi, p - idle_w, left=idle_w, height=hgt, color=c, lw=0)
        ax_p.text(p + 15, yi, rf"{p:.0f}\,W", va="center", fontsize=fs)
        cpu = r["P_cpu0"] + r["P_cpu1"]
        gpu = r["P_gpu"] if k in ("gpuonly", "cuda") else 0.0
        ax_c.barh(yi, cpu, height=hgt, **stripes)
        if gpu:
            ax_c.barh(yi, gpu, left=cpu, height=hgt, color=c, lw=0)
        ax_c.text(cpu + gpu + 12, yi, rf"{cpu + gpu:.0f}\,W", va="center", fontsize=fs)
        ax_e.barh(yi, e - 1e3, left=1e3, height=hgt, color=c, lw=0)
        ax_e.text(e * 1.2, yi, kj(e), va="center", fontsize=fs)
    ax_t.set_xscale("log")
    ax_t.set_xlim(1, 8e4)
    ax_t.set_xticks([1, 100, 1e4], [r"1\,s", r"100\,s", r"10k\,s"])
    ax_t.xaxis.set_minor_formatter(NullFormatter())
    ax_t.set_title("time per tree", loc="left", pad=3, fontsize=7)
    ax_p.set_xlim(0, 1100)
    ax_p.set_xticks([0, 400, 800], [r"0\,W", r"400\,W", r"800\,W"])
    ax_p.set_title("server power", loc="left", pad=3, fontsize=7)
    ax_c.set_xlim(0, 720)
    ax_c.set_xticks([0, 400], [r"0\,W", r"400\,W"])
    ax_c.set_title("CPU+GPU power", loc="left", pad=3, fontsize=7)
    joule_axis(ax_e, "x")
    ax_e.set_xlim(1e3, 1.5e7)
    ax_e.set_xticks([1e3, 1e5, 1e7])
    ax_e.set_title("total energy per tree", loc="left", pad=3, fontsize=7)
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=5.2)
    ax_t.set_yticks(y, [f"{STYLE[k][0]}, {P_GPU[n] if k in ('gpuonly', 'cuda') else n}"
                        for k, n in cases], fontsize=6.2)
    for i in range(1, len(cases)):
        if cases[i][0] != cases[i - 1][0]:
            for ax in axes:
                ax.axhline(y[i] + 0.5, color=GRID, lw=0.5)
    h = [plt.Rectangle((0, 0), 1, 1, color=to_rgba("#777", 0.35), label="idle server"),
         plt.Rectangle((0, 0), 1, 1, label="CPU", **stripes),
         plt.Rectangle((0, 0), 1, 1, color=STYLE["gpuonly"][1], label="GPU")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=6)
    save(fig, "energy")

if __name__ == "__main__":
    which = sys.argv[1:] or ["ladder", "scaling", "profile", "depth", "bars", "energy"]
    for name in which:
        globals()[name]()
