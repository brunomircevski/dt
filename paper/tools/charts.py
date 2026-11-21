#!/usr/bin/env python3
"""Candidate chart styles for the paper, as vector PDFs sized for LNCS
(text width 122 mm = 4.80 in; fonts rendered by LaTeX, 7-8 pt, so they match
the text). Every chart is built from the benchmark results in bench/.

  bench/tools/.venv/bin/python paper/tools/charts.py          # writes paper/figs/test-*.pdf

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
            epyc and hopper, from bench/cluster-power/ecbdl_15m-depth10/summary.csv
"""

import csv
import json
import re
import shutil
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

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "bench"
OUT = ROOT / "paper" / "figs"
sys.path.insert(0, str(ROOT / "bench" / "tools"))
import benchlib as B  # noqa: E402

# Whether each result a figure reads is out of date (benchlib.out_of_date: measured before the
# final protocol); save() puts a banner on the figure while any is, and starts the next figure.
USED = []

FULL, HALF = 4.80, 2.35  # inches: LNCS \textwidth and a bit under half of it
# One type scale for every chart (pt, at the printed size: save() makes each figure exactly
# FULL wide, and the paper includes it at that width, so nothing is scaled).
FS_TEXT, FS_TICK, FS_ANN = 7, 6, 5.5  # titles/axis labels/legend, tick labels, numbers on marks

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": FS_TEXT, "axes.titlesize": FS_TEXT, "axes.labelsize": FS_TEXT,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TEXT - 0.5,
    "axes.titlelocation": "left", "axes.titlepad": 3, "axes.labelpad": 2,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "legend.columnspacing": 1.2, "legend.handletextpad": 0.4, "legend.handlelength": 1.2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.borderaxespad": 0.2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Implementation colours (reference palette slots, as in bench/tools/chart.py), and a
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
    table = list(csv.DictReader(open(f)))
    USED.append(B.rows_out_of_date([{"impl": i} for i in {r["impl"] for r in table}] or [{}],
                                   json.load(open(path / "machine.json"))))  # impl: --simple's own rule
    for r in table:
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


def gpu_other(dataset, impl, algo, host="hopper"):
    """cuML / XGBoost: one tree, cold, the same staged upload as ./tree
    (bench/tools/gpu_steps_chart.py). CART only: no GPU library grows C4.5 trees."""
    f = RES / "cluster-gpu-vs-others" / "one-tree" / f"vs-{host}-{dataset}" / "results.jsonl"
    if algo != "cart" or not f.exists():
        return None
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r["impl"] == impl and r.get("kind", "run") == "run" and r.get("status") == "ok"
            and r.get("protocol", PROTO[algo]) == PROTO[algo]]
    if not rows:
        return None
    USED.append(B.rows_out_of_date(rows))
    t = [r["train_seconds"] for r in rows]
    last = rows[-1]
    return {"time_mean": statistics.mean(t), "time_min": min(t), "time_max": max(t),
            "test_accuracy": last.get("test_accuracy"), "nodes": last.get("nodes"),
            "gpu_peak": max(r.get("gpu_peak_bytes") or 0 for r in rows)}


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


def ink_box(fig):
    """The figure's drawn extent (tight bbox), in inches."""
    fig.draw_without_rendering()
    return fig.get_tightbbox(fig._get_renderer())


def fit_width(fig, width):
    """Resize the figure until what it draws is exactly `width` inches wide (the text stays the
    same size, the axes take up the difference); -> the save box, that wide, centred."""
    from matplotlib.transforms import Bbox
    for _ in range(6):
        b = ink_box(fig)
        if abs(b.width - width) < 0.005:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + width - b.width, h)
    b = ink_box(fig)
    x0 = b.x0 - (width - b.width) / 2
    return Bbox([[x0, b.y0 - 0.01], [x0 + width, b.y1 + 0.01]])


def axes_mid(fig):
    """Horizontal middle of the panels, as a figure fraction."""
    return (min(a.get_position().x0 for a in fig.axes) + max(a.get_position().x1 for a in fig.axes)) / 2


def legend_below(fig, handles, gap=0.04, **kw):
    """One legend row centred under everything drawn so far (axes, tick labels, x labels)."""
    b = ink_box(fig)
    h = fig.get_size_inches()[1]
    kw = dict(dict(ncol=len(handles)), **kw)
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(axes_mid(fig), (b.y0 - gap) / h),
                      bbox_transform=fig.transFigure, **kw)


def legend_above(fig, handles, gap=0.02, **kw):
    """One legend row centred over everything drawn so far (panel titles)."""
    b = ink_box(fig)
    h = fig.get_size_inches()[1]
    kw = dict(dict(ncol=len(handles)), **kw)
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(axes_mid(fig), (b.y1 + gap) / h),
                      bbox_transform=fig.transFigure, **kw)


def save(fig, name, copy=None, width=FULL):
    """paper/figs/test-<name>.pdf; copy: also <copy>.pdf (the same file) and <copy>.png (for the
    benchmark's BENCHMARK.md), the figure the paper shows kept with its benchmark's data."""
    OUT.mkdir(exist_ok=True)
    stale = sum(USED)
    if stale:
        what = f"all {len(USED)}" if stale == len(USED) else f"{stale} of {len(USED)}"
        fig.text(0.0, 1.0, rf"\textbf{{OUT OF DATE}}: {what} results not from the final protocol; "
                 "a placeholder until the rerun", ha="left", va="bottom", fontsize=6.5, color=INK,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fab219", ec="none"))
    USED.clear()
    box = fit_width(fig, width)
    fig.savefig(OUT / f"test-{name}.pdf", bbox_inches=box, metadata={"CreationDate": None})  # no date (the font subset tags still vary)
    print("wrote", OUT / f"test-{name}.pdf")
    if copy:
        shutil.copyfile(OUT / f"test-{name}.pdf", copy.with_suffix(".pdf"))
        fig.savefig(copy.with_suffix(".png"), bbox_inches=box, dpi=300, facecolor="white")
        print("wrote", copy.with_suffix(".pdf"), copy.with_suffix(".png"))
    plt.close(fig)


# ---------------------------------------------------------------- 1. ladder
def ladder():
    datasets = ["susy", "higgs", "ecbdl_15m", "mnist8m", "sorel_4m"]
    stages = [("simple", "epyc"), ("serial", "epyc"), ("cpu", "epyc"), ("gpuonly", "hopper"), ("cuda", "hopper")]
    names = ["Serial\nSimple", "Serial\nOptimized", "Multicore\nCPU", "GPU-\nBased", "Hybrid"]
    n = len(stages)
    shades = ["#2a78d6", "#eb6834", "#1baf7a", "#7f4fc9", "#eda100"]
    marks = ["o", "s", "D", "^", "v"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.75), sharey=True, gridspec_kw={"wspace": 0.06})
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
                    ha="center", va="center", fontsize=FS_ANN, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=GRID, lw=0.4))
        total = [times[d][0] / times[d][-1] for d in datasets]
        ax.set_title(rf"\textbf{{{ALGO[algo]}}}: overall $\times${min(total):.0f}--{max(total):.0f}")
        log_axis(ax, "y")
        ax.set_ylim(0.05, 6e4)
        ax.axvspan(2.6, n - 0.7, color="#f3f3f3", zorder=-1, lw=0)
        ax.text(n - 1.5, 2.6e4, "H100", ha="center", va="center", fontsize=FS_ANN, color=MUTED)
        ax.set_xticks(range(n), names, linespacing=0.95)
        ax.set_xlim(-0.35, n - 0.65)
        ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("training time (s)")
    h, _ = axes[0].get_legend_handles_labels()
    legend_below(fig, h, handlelength=1.6)
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
                  lambda d: cpu_other(d, "sklearn" if algo == "cart" else "j48", algo))]
        if algo == "cart":  # no GPU library grows C4.5 trees (an entropy tree is not C4.5)
            impls += [("cuml", lambda d: gpu_other(d, "cuml", algo)),
                      ("xgboost", lambda d: gpu_other(d, "xgboost", algo))]
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
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (node, title) in zip(axes, [("epyc", r"\textbf{epyc1}: 2 $\times$ EPYC 9255"),
                                        ("dell", r"\textbf{dell}: 2 $\times$ Xeon Gold 6542Y")]):
        ax.plot([1, 48], [1, 48], color=GRID, lw=0.8, zorder=0)
        ax.text(12, 15.5, "ideal", fontsize=FS_ANN, color=MUTED, rotation=33, ha="center")
        ax.axvline(24, color=MUTED, lw=0.5, ls=(0, (1, 2)), zorder=1)
        ax.text(25, 1.5, "2nd socket $\\rightarrow$", fontsize=FS_ANN, color=MUTED, va="bottom")
        labels = []  # [speed-up at 48 threads, text, colour]: placed together below, apart
        for ds, m, dl, c in sets:
            f = RES / "cluster-cpu-scaling" / f"cpu-{node}-{ds}" / "scaling.csv"
            USED.append(B.rows_out_of_date([], json.load(open(f.parent / "machine.json"))))
            for algo, ls in [("cart", "-"), ("c45", "--")]:
                r = [x for x in csv.DictReader(open(f)) if x["protocol"] == PROTO[algo]]
                t = [int(x["threads"]) for x in r]
                sp = [float(x["speedup"]) for x in r]
                ax.plot(t, sp, ls=ls, color=c, lw=0.9, marker=m, ms=2.4, mfc="white" if algo == "c45" else c,
                        mew=0.6, markevery=[0, 3, 7, 13])
                labels.append([sp[-1], rf"{sp[-1]:.0f}$\times$ ({100 * sp[-1] / 48:.0f}\%)", c])
        labels.sort()
        for i in range(1, len(labels)):  # at least one label's height apart
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 3.2)
        for y, text, c in labels:
            ax.annotate(text, (49.5, y), fontsize=FS_ANN, color=c, va="center")
        ax.set_xlim(0, 62)
        ax.set_ylim(0, 48)
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48])
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_xlabel("threads")
        ax.set_title(title)
        ax.grid(color=GRID, lw=0.3, zorder=0)
    axes[0].set_ylabel("speed-up over 1 thread")
    h = [Line2D([], [], color=c, lw=0.9, marker=m, ms=2.5, label=dl) for _, m, dl, c in sets]
    h += [Line2D([], [], color=MUTED, lw=0.9, ls="-", label="CART"),
          Line2D([], [], color=MUTED, lw=0.9, ls="--", label="C4.5")]
    legend_below(fig, h, handlelength=1.8)
    save(fig, "scaling")


# ---------------------------------------------------------------- 7. depth
def depth():
    """Hybrid against GPU only by depth limit (H100, hybrid-vs-gpu-only, 10 runs a case): grouped bars, linear."""
    root = RES / "hybrid-vs-gpu-only"
    depths = ["5", "10", "15", "20", "25", "30", "none"]
    proto = {"cart": [f"cart_depth{d}" for d in depths[:-1]] + ["cart_full_nodepth"],
             "c45": [f"c45_depth{d}_unpruned" for d in depths[:-1]] + ["c45_unpruned"]}
    c_hy, c_go = STYLE["cuda"][1], "#b9c2bc"
    sets = [("higgs", "HIGGS"), ("mnist8m", "MNIST8M"), ("sorel_4m", "SOREL 4M")]
    fig, axes = plt.subplots(len(sets), 2, figsize=(FULL, 0.95 * len(sets)), sharex=True,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.1})
    x = np.arange(len(depths))
    w = 0.38
    for row, (ds, name) in enumerate(sets):
        hy, go = summary(root / f"cuda-hopper-{ds}"), summary(root / f"gpuonly-hopper-{ds}")
        top = max(go[p]["time_mean"] for a in proto for p in proto[a])
        for col, algo in enumerate(["cart", "c45"]):
            ax = axes[row, col]
            th = np.array([hy[p]["time_mean"] for p in proto[algo]])
            tg = np.array([go[p]["time_mean"] for p in proto[algo]])
            ax.bar(x - w / 2, th, w, color=c_hy, lw=0, label="Hybrid", zorder=2)
            ax.bar(x + w / 2, tg, w, color=c_go, lw=0, label="GPU-Based", zorder=2)
            for i in range(len(x)):
                r = tg[i] / th[i]
                ax.text(x[i], max(th[i], tg[i]) + 0.03 * top, f"{r:.1f}" + r"$\times$",
                        ha="center", va="bottom", fontsize=FS_ANN, color=INK)
            ax.set_ylim(0, top * 1.22)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, steps=[1, 2, 5, 10]))
            ax.grid(axis="y", color=GRID, lw=0.3, zorder=0)
            ax.tick_params(axis="x", length=0)
            ax.set_xticks(x, depths)
            ax.set_title(rf"\textbf{{{name}}}, {ALGO[algo]}", pad=2)
            if not col:
                ax.set_ylabel("time (s)")
        axes[row, 1].sharey(axes[row, 0])  # one scale a row, its numbers on both panels
    for ax in axes[-1]:
        ax.set_xlabel("depth limit")
    h, _ = axes[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], ls="", label=r"1.2$\times$ above the bars: GPU-Based time $\div$ Hybrid time"))
    legend_below(fig, h)
    save(fig, "depth")


# ---------------------------------------------------------------- 5. pareto
def pareto(ds="higgs"):
    # broken y axis: the CART trees (top) and the C4.5 trees (bottom) are
    # 7 points apart; each band is zoomed so the small accuracy gaps show
    fig, (top, bot) = plt.subplots(2, 1, figsize=(HALF, 2.05), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})
    pts = []
    for algo in ["cart", "c45"]:
        for k, node in [("serial", "epyc"), ("cpu", "epyc"), ("cuda", "hopper")]:
            r = ours(ds, k, node, algo)
            mem = r["gpu_peak"] if k == "cuda" else r["peak_rss"]
            pts.append((k, algo, r["time_mean"], r["test_accuracy"], mem, True))
        r = cpu_other(ds, "sklearn", algo) if algo == "cart" else None  # no C4.5 stand-ins
        if r:
            pts.append(("sklearn", algo, r["time_mean"], r["test_accuracy"], r["peak_rss"], False))
        if algo == "c45":
            r = cpu_other(ds, "j48", algo)
            pts.append(("j48", algo, r["time_mean"], r["test_accuracy"], r["peak_rss"], False))
        for k in (["cuml", "xgboost"] if algo == "cart" else []):  # no GPU C4.5
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
    top.text(0.02, 0.06, "CART (Gini)", transform=top.transAxes, fontsize=5.5, color=MUTED)
    bot.text(0.02, 0.06, "C4.5 (gain ratio, exact)", transform=bot.transAxes, fontsize=5.5, color=MUTED)
    h = [Line2D([], [], ls="", marker="o", color=STYLE[k][1], ms=3.5, label=STYLE[k][0])
         for k in ["serial", "cpu", "cuda", "sklearn", "j48", "cuml", "xgboost"]]
    h += [Line2D([], [], ls="", marker="o", color=MUTED, ms=3.5, label="CART"),
          Line2D([], [], ls="", marker="s", mfc="white", color=MUTED, ms=3.5, label="C4.5")]
    top.legend(handles=h, loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=5.8,
               labelspacing=0.25, handletextpad=0.1)
    bot.text(1.04, 0.0, DATA[ds][0] + "\nmarker area\n" + r"$\propto$ peak RAM" + "\nor VRAM",
             transform=bot.transAxes, fontsize=5.5, color=MUTED, va="bottom", linespacing=1.1)
    save(fig, "pareto")


# ---------------------------------------------------------------- 6. profile
def profile():
    import profile_share_chart as P
    datasets = ["higgs", "ecbdl_15m", "chess_1b"]
    labels = {"higgs": "HIGGS", "ecbdl_15m": "ECBDL", "chess_1b": "Chess"}
    modes = [("cpu-epyc", "EPYC"), ("cpu-dell", "Xeon"), ("gpuonly-hopper", "GPU-Based"),
             ("cuda-hopper", "Hybrid")]
    cases = P.load(RES / "cluster-profiling", datasets)
    USED.extend(c["stale"] for c in cases.values())
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.0), sharey=True, gridspec_kw={"wspace": 0.2})
    gap = 0.8  # between dataset groups: room for the group's name
    heads = []
    for ax, (proto, algo) in zip(axes, P.PROTOCOLS):
        y, ticks = 0, []
        for ds in datasets:
            heads.append((y - 0.62, labels[ds]))
            for mode, ml in modes:
                case = cases.get((mode, ds, proto))
                if case:
                    parts = case["parts"]
                    total = sum(parts.values())
                    left = 0
                    for step, _, color in P.STEPS:
                        v = sum(val for (s, _), val in parts.items() if s == step) / total
                        ax.barh(y, v, left=left, color=color, height=0.78, lw=0)
                        if v > 0.1:
                            ax.text(left + v / 2, y, f"{100 * v:.0f}", ha="center", va="center",
                                    fontsize=FS_ANN - 0.5, color=P.text_color(color))
                        left += v
                    gpu = sum(val for (_, d), val in parts.items() if d == "GPU") / total
                    if gpu > 0:
                        ax.plot([gpu], [y - 0.39], marker="v", color=INK, ms=2.2, clip_on=False)
                    ax.text(1.015, y, tsec(case["row"]["train_seconds"]) + r"\,s", va="center",
                            fontsize=FS_ANN, transform=ax.get_yaxis_transform())
                ticks.append((y, ml))
                y += 1
            y += gap
        ax.set_yticks([t for t, _ in ticks], [l for _, l in ticks])
        ax.set_xlim(0, 1)
        ax.set_ylim(y - gap - 0.5, -1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
        ax.set_xlabel(r"share of training time (\%)")
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.spines["left"].set_visible(False)
        ax.set_title(rf"\textbf{{{algo}}}", pad=1)
        for yh, name in heads[-len(datasets):]:  # dataset name over its group, in each panel
            ax.text(0, yh, rf"\textit{{{name}}}", ha="left", va="baseline", fontsize=FS_ANN, color=MUTED)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=tex(n).replace("↔", r"$\leftrightarrow$"))
         for _, n, c in P.STEPS]
    h.append(Line2D([], [], ls="", marker="v", color=INK, ms=3, label="GPU-busy share"))
    legend_below(fig, h, handlelength=1.0)
    save(fig, "profile")

# ---------------------------------------------------------------- 8. bars
def bars():
    """Grouped bars, log scale: this work's five implementations on every dataset (CPU on
    epyc1, GPU on the H100), value above each bar, no bar where there is no run."""
    datasets = ["susy", "higgs", "mnist8m", "ecbdl", "ecbdl_15m", "sorel", "sorel_4m"]
    impls = [("simple", "epyc"), ("serial", "epyc"), ("cpu", "epyc"), ("gpuonly", "hopper"),
             ("cuda", "hopper")]
    labels = {"simple": "Serial Simple", "serial": "Serial Optimized",
              "cpu": "Multicore CPU", "gpuonly": "GPU-Based", "cuda": "Hybrid"}
    n = len(impls)
    w = 0.72 / n
    x = np.arange(len(datasets))
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 2.9), sharex=True, gridspec_kw={"hspace": 0.1})
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
                ax.text(xi, t * 1.15, tsec(t), rotation=90, ha="center", va="bottom", fontsize=FS_ANN - 0.5,
                        color=INK)
        log_axis(ax, "y")
        ax.set_ylim(0.03, 1.5e5)
        ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
        ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        ax.set_ylabel(rf"\textbf{{{ALGO[algo]}}}: time (s)")
        ax.tick_params(axis="x", length=0)
        for i in range(1, len(datasets)):
            ax.axvline(i - 0.5, color=GRID, lw=0.4, zorder=0)
    names = {"ecbdl_15m": "ECBDL'14\n15M samples", "sorel_4m": "SOREL\n4M samples"}
    axes[1].set_xticks(x, [tex(names.get(d, DATA[d][0])) for d in datasets], linespacing=0.95)
    axes[1].set_xlim(-0.5, len(datasets) - 0.5)
    h, _ = axes[0].get_legend_handles_labels()
    legend_below(fig, h, handlelength=1.0)
    save(fig, "bars")


# ---------------------------------------------------------------- power (energy per tree)
# bench/cluster-power/ecbdl_15m-depth10/summary.csv, ECBDL'14 15M, depth <= 10, unpruned. Whole-server
# energy at the PSU (each node's meter modelled from the fast sensors, bench/cluster-power/BENCHMARK.md);
# components: CPU packages (RAPL), DRAM (RAPL, Intel nodes), the GPU in use (NVML), the rest.
POWER = ROOT / "bench" / "cluster-power" / "ecbdl_15m-depth10"
P_NODES = [("epyc1", "epyc"), ("hopper", "hopper")]  # the power benchmark's servers
P_IMPLS = [("simple", "simple"), ("serial", "serial"), ("parallel", "cpu"), ("gpuonly", "gpuonly"),
           ("cuda", "cuda")]  # (power_bench name, STYLE key)
P_MARK = {"epyc": "o", "hopper": "D"}
P_GPU = {"hopper": "H100"}


def power_rows():
    """{(STYLE key, node short name, algo): row with floats} from every power result."""
    out = {}
    f = POWER / "summary.csv"
    shorts = {short for _, short in P_NODES}
    if f.exists():
        for r in csv.DictReader(open(f)):
            if r["node"] not in shorts:
                continue
            v = {}
            for k, x in r.items():
                try:
                    v[k] = float(x)
                except (TypeError, ValueError):
                    v[k] = x
            key = dict(P_IMPLS)[r["impl"]]
            out[(key, r["node"], r["algo"])] = v
        for host, _ in P_NODES:  # each node's power job: its machine.txt's commit
            machine = POWER / host / "machine.txt"
            commit = re.search(r"^commit (\S+)", machine.read_text(), re.M) if machine.exists() else None
            USED.append(B.out_of_date(commit and commit.group(1), host))
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


def energy(algo="c45", cpu_nodes=("epyc",), gpu_nodes=("hopper",)):
    """Time, power and total energy of one tree, one row per implementation x node (CPU
    implementations on the CPU nodes, GPU ones on the GPU nodes), four aligned panels: time
    per tree (log); mean power of the whole server (linear; the lighter part is the idle
    server's power); mean power of the CPU packages + the GPU in use (on GPU runs the CPU
    part is striped) while
    training (linear; no idle split); total energy per tree at the PSU (log;
    idle included). The number at the end of each bar is its total."""
    from matplotlib.colors import to_rgba
    rows = power_rows()
    cases = [(k, n) for k, n in p_cases(rows)
             if n in (gpu_nodes if k in ("gpuonly", "cuda") else cpu_nodes)]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.35), sharey=True,
                             gridspec_kw={"wspace": 0.14})
    ax_t, ax_p, ax_c, ax_e = axes
    # CPU+GPU panel: bars in the implementation's colour; on the GPU runs the CPU packages'
    # part (the start of the bar) carries blue stripes
    # stripes as wide as the gaps between them (about half the area), a blue that stands
    # out from each green: dark on the light green, light on the dark green
    STRIPE = {"gpuonly": "#1a4d99", "cuda": "#8cc4ff"}
    HATCH = "/////"  # about 5 pt apart along the bar in the PDF
    plt.rcParams["hatch.linewidth"] = 1.8  # about half the gap; read when the bars are made
    y = np.arange(len(cases))[::-1].astype(float)
    hgt, fs = 0.7, FS_ANN
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
        if gpu:
            ax_c.barh(yi, cpu, height=hgt, fc=c, ec=STRIPE[k], lw=0, hatch=HATCH)
            ax_c.barh(yi, gpu, left=cpu, height=hgt, color=c, lw=0)
        else:
            ax_c.barh(yi, cpu, height=hgt, color=c, lw=0)
        ax_c.text(cpu + gpu + 12, yi, rf"{cpu + gpu:.0f}\,W", va="center", fontsize=fs)
        ax_e.barh(yi, e - 1e3, left=1e3, height=hgt, color=c, lw=0)
        ax_e.text(e * 1.2, yi, kj(e), va="center", fontsize=fs)
    ax_t.set_xscale("log")
    ax_t.set_xlim(1, 8e4)
    ax_t.set_xticks([1, 100, 1e4], [r"1\,s", r"100\,s", r"10k\,s"])
    ax_t.xaxis.set_minor_formatter(NullFormatter())
    ax_t.set_title("time per tree")
    ax_p.set_xlim(0, 1100)
    ax_p.set_xticks([0, 400, 800], [r"0\,W", r"400\,W", r"800\,W"])
    ax_p.set_title("server power")
    ax_c.set_xlim(0, 760)
    ax_c.set_xticks([0, 400], [r"0\,W", r"400\,W"])
    ax_c.set_title("CPU+GPU power")
    joule_axis(ax_e, "x")
    ax_e.set_xlim(1e3, 3e7)
    ax_e.set_xticks([1e3, 1e5, 1e7])
    ax_e.set_title("energy per tree")
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    ax_t.set_yticks(y, [STYLE[k][0] for k, n in cases])
    ax_t.set_ylim(y[-1] - 0.55, y[0] + 0.55)
    for i in range(1, len(cases)):
        if cases[i][0] != cases[i - 1][0]:
            for ax in axes:
                ax.axhline(y[i] + 0.5, color=GRID, lw=0.5)
    h = [plt.Rectangle((0, 0), 1, 1, color=to_rgba("#777", 0.35), label="idle server"),
         plt.Rectangle((0, 0), 1, 1, fc=STYLE["gpuonly"][1], ec=STRIPE["gpuonly"], lw=0, hatch=HATCH,
                       label="CPU part of a GPU run")]
    legend_below(fig, h, handlelength=2.4, handleheight=1.0)
    save(fig, "energy", copy=POWER.parent / "paper-fig-energy-ecbdl_15m")


# bench/cluster-power/mnist8m-nodepth: the same measurement on MNIST8M without a depth limit, the
# three fast implementations on one server (hopper, the H100 PCIe server), CART and C4.5.
POWER_DEEP = RES / "cluster-power" / "mnist8m-nodepth"


def energy_deep():
    """Time, power and energy of one tree on MNIST8M without a depth limit, Multicore CPU, GPU-Based
    and Hybrid on the same H100 server; one group of rows per algorithm, four aligned linear
    panels as in energy(): time per tree; mean power of the whole server (the lighter part is the
    idle server's power); mean power of the CPU packages + the GPU in use (the CPU part striped;
    none on the Multicore run); energy per tree at the PSU (the lighter part is what
    the idle server would have used in the same time)."""
    from matplotlib.colors import to_rgba
    rows = {}
    for r in csv.DictReader(open(POWER_DEEP / "summary.csv")):
        rows[(r["impl"], r["algo"])] = {k: (float(v) if re.fullmatch(r"[-+0-9.eE]+|nan", v or "") else v)
                                        for k, v in r.items()}
    for machine in sorted(POWER_DEEP.glob("*/machine.txt")):
        commit = re.search(r"^commit (\S+)", machine.read_text(), re.M)
        USED.append(B.out_of_date(commit and commit.group(1), machine.parent.name.rstrip("0123456789")))
    impls = [("parallel", "cpu"), ("gpuonly", "gpuonly"), ("cuda", "cuda")]
    cases = [(algo, impl, key) for algo in ("c45",) for impl, key in impls if (impl, algo) in rows]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 0.2 * len(cases) + 0.35), sharey=True,
                             gridspec_kw={"wspace": 0.14})
    ax_t, ax_p, ax_c, ax_e = axes
    STRIPE = {"cpu": "#123f7a", "gpuonly": "#1a4d99", "cuda": "#8cc4ff"}
    HATCH = "/////"
    plt.rcParams["hatch.linewidth"] = 1.8
    y = np.arange(len(cases))[::-1].astype(float)
    hgt, fs = 0.7, FS_ANN
    for yi, (algo, impl, k) in zip(y, cases):
        r = rows[(impl, algo)]
        c, lc = STYLE[k][1], to_rgba(STYLE[k][1], 0.35)
        t, p, e, idle_w = r["train_s"], r["power_W"], r["psu_J"], r["idle_W"]
        ax_t.barh(yi, t, height=hgt, color=c, lw=0)
        ax_t.text(t + 0.6, yi, rf"{t:.1f}\,s", va="center", fontsize=fs)
        ax_p.barh(yi, idle_w, height=hgt, color=lc, lw=0)
        ax_p.barh(yi, p - idle_w, left=idle_w, height=hgt, color=c, lw=0)
        ax_p.text(p + 15, yi, rf"{p:.0f}\,W", va="center", fontsize=fs)
        cpu = r["P_cpu0"] + r["P_cpu1"]
        gpu = r["P_gpu"] if impl != "parallel" else 0.0  # Multicore: no GPU in use (all four idle)
        ax_c.barh(yi, cpu, height=hgt, fc=c, ec=STRIPE[k], lw=0, hatch=HATCH)
        ax_c.barh(yi, gpu, left=cpu, height=hgt, color=c, lw=0)
        ax_c.text(cpu + gpu + 12, yi, rf"{cpu + gpu:.0f}\,W", va="center", fontsize=fs)
        e_idle = r["psu_idle_J"]
        ax_e.barh(yi, e_idle / 1e3, height=hgt, color=lc, lw=0)
        ax_e.barh(yi, (e - e_idle) / 1e3, left=e_idle / 1e3, height=hgt, color=c, lw=0)
        ax_e.text(e / 1e3 + 0.4, yi, kj(e), va="center", fontsize=fs)
    t_max = max(rows[(i, a)]["train_s"] for a, i, _ in cases)
    p_max = max(rows[(i, a)]["power_W"] for a, i, _ in cases)
    c_max = max(rows[(i, a)]["P_cpu0"] + rows[(i, a)]["P_cpu1"] + (rows[(i, a)]["P_gpu"] if i != "parallel" else 0)
                for a, i, _ in cases)
    e_max = max(rows[(i, a)]["psu_J"] for a, i, _ in cases) / 1e3
    for ax, top, unit, title in ((ax_t, t_max, "s", "time per tree"), (ax_p, p_max, "W", "server power"),
                                 (ax_c, c_max, "W", "CPU+GPU power"),
                                 (ax_e, e_max, "kJ", "energy per tree")):
        ax.set_xlim(0, top * 1.45)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _, u=unit: rf"{v / 1e3:g}\,k{u}" if u == "W" and v >= 1e3 else rf"{v:g}\,{u}"))
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    ax_t.set_yticks(y, [STYLE[k][0] for a, _, k in cases])
    ax_t.set_ylim(y[-1] - 0.55, y[0] + 0.55)
    for i in range(1, len(cases)):
        if cases[i][0] != cases[i - 1][0]:
            for ax in axes:
                ax.axhline(y[i] + 0.5, color=MUTED, lw=0.5)
    h = [plt.Rectangle((0, 0), 1, 1, color=to_rgba("#777", 0.35), label="idle server"),
         plt.Rectangle((0, 0), 1, 1, fc=STYLE["gpuonly"][1], ec=STRIPE["gpuonly"], lw=0, hatch=HATCH,
                       label="CPU packages"),
         plt.Rectangle((0, 0), 1, 1, color=STYLE["gpuonly"][1], label="GPU")]
    legend_below(fig, h, handlelength=2.4, handleheight=1.0)
    save(fig, "energy-mnist8m", copy=POWER_DEEP.parent / "paper-fig-energy-mnist8m")


if __name__ == "__main__":
    which = sys.argv[1:] or ["ladder", "scaling", "profile", "depth", "bars", "energy", "energy_deep"]
    for name in which:
        globals()[name]()
