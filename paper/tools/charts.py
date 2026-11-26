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
    "chess_1b": ("Chess 1B", "cluster-chess", "chess_1b"),
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
    return f"{v:g}"


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
    return f"{v:.3f}"


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


def save(fig, name, copy=None, width=FULL, final=False):
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
    out = OUT / (f"{name}.pdf" if final else f"test-{name}.pdf")
    fig.savefig(out, bbox_inches=box, metadata={"CreationDate": None})  # no date (the font subset tags still vary)
    print("wrote", out)
    if copy:
        shutil.copyfile(out, copy.with_suffix(".pdf"))
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
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.42), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (node, title) in zip(axes, [("epyc", r"\textbf{epyc}: 2 $\times$ EPYC 9255"),
                                        ("dell", r"\textbf{dell}: 2 $\times$ Xeon Gold 6542Y")]):
        ax.plot([1, 48], [1, 48], color=GRID, lw=0.8, zorder=0)
        ax.text(12, 15.5, "ideal", fontsize=FS_ANN, color=MUTED, rotation=30, ha="center")
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
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 3.6)
        for y, text, c in labels:
            ax.annotate(text, (49.5, y), fontsize=FS_ANN, color=c, va="center")
        ax.set_xlim(0, 62)
        ax.set_ylim(0, 48)
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48])
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_title(title)
        ax.grid(color=GRID, lw=0.3, zorder=0)
    axes[0].set_ylabel("speedup over 1 thread")
    fig.supxlabel("threads", fontsize=FS_TEXT, y=-0.04)
    h = [Line2D([], [], color=c, lw=0.9, marker=m, ms=2.5, label=dl) for _, m, dl, c in sets]
    h += [Line2D([], [], color=MUTED, lw=0.9, ls="-", label="CART"),
          Line2D([], [], color=MUTED, lw=0.9, ls="--", label="C4.5")]
    legend_below(fig, h, handlelength=1.8, gap=0.03)
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
    epyc, GPU on the H100), no bar where there is no run. Exact values are in Table 6."""
    datasets = ["susy", "higgs", "ecbdl", "ecbdl_15m", "mnist8m", "sorel", "sorel_4m", "chess_1b"]
    impls = [("simple", "epyc"), ("serial", "epyc"), ("cpu", "epyc"), ("gpuonly", "hopper"),
             ("cuda", "hopper")]
    labels = {"simple": "Serial Simple", "serial": "Serial Optimized",
              "cpu": "Multicore CPU", "gpuonly": "GPU-Based", "cuda": "Hybrid"}
    n = len(impls)
    w = 0.72 / n
    x = np.arange(len(datasets))
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 1.95), sharex=True, gridspec_kw={"hspace": 0.24})
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
        log_axis(ax, "y")
        ax.set_ylim(0.05, 3e4)
        ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
        ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        ax.set_title(rf"\textbf{{{ALGO[algo]}}}", pad=2)
        ax.tick_params(axis="x", length=0)
        for i in range(1, len(datasets)):
            ax.axvline(i - 0.5, color=GRID, lw=0.4, zorder=0)
    fig.supylabel("training time (s)", fontsize=FS_TEXT, x=0.01)
    names = {"ecbdl_15m": "ECBDL'14\n15M", "sorel": "SOREL-\n20M", "sorel_4m": "SOREL\n4M", "chess_1b": "Chess\n$10^9$"}
    axes[1].set_xticks(x, [names[d] if d in names else tex(DATA[d][0]) for d in datasets], linespacing=0.95)
    axes[1].set_xlim(-0.5, len(datasets) - 0.5)
    h, _ = axes[0].get_legend_handles_labels()
    legend_below(fig, h, handlelength=1.0, gap=0.03)
    save(fig, "results-bars", final=True)


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
    """Unified energy figure: (a) ECBDL'14 15M depth <= 10 (5 implementations) and
    (b) MNIST8M no depth limit (3 implementations on hopper), sharing four aligned panels:
    time (log), server power (linear with idle tint), CPU+GPU power (linear with CPU hatch
    on GPU runs), and energy (log), plus one legend."""
    from matplotlib.colors import to_rgba
    from matplotlib.transforms import offset_copy

    rows_ec = power_rows()
    cases_ec = [(k, n) for k, n in p_cases(rows_ec)
                if n in (gpu_nodes if k in ("gpuonly", "cuda") else cpu_nodes)]

    rows_mn = {}
    for r in csv.DictReader(open(POWER_DEEP / "summary.csv")):
        rows_mn[(r["impl"], r["algo"])] = {k: (float(v) if re.fullmatch(r"[-+0-9.eE]+|nan", v or "") else v)
                                           for k, v in r.items()}
    for machine in sorted(POWER_DEEP.glob("*/machine.txt")):
        commit = re.search(r"^commit (\S+)", machine.read_text(), re.M)
        USED.append(B.out_of_date(commit and commit.group(1), machine.parent.name.rstrip("0123456789")))
    cases_mn = [("cpu", "parallel"), ("gpuonly", "gpuonly"), ("cuda", "cuda")]

    # Build top-to-bottom y positions (ECBDL top, thin gap, MNIST8M bottom)
    entries = []
    y = 0.0
    for k, node in cases_ec:
        entries.append((y, k, rows_ec[(k, node, algo)]))
        y += 1.0
    ec_end = y - 1.0
    gap_y = y - 0.15
    y += 0.7
    mn_start = y
    for k, impl in cases_mn:
        entries.append((y, k, rows_mn[(impl, algo)]))
        y += 1.0
    mn_end = y - 1.0

    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.78), sharey=True, gridspec_kw={"wspace": 0.14})
    ax_t, ax_p, ax_c, ax_e = axes
    STRIPE = {"gpuonly": "#1a4d99", "cuda": "#8cc4ff"}
    HATCH = "/////"
    plt.rcParams["hatch.linewidth"] = 1.8
    hgt, fs = 0.68, FS_ANN

    for yi, k, r in entries:
        c, lc = STYLE[k][1], to_rgba(STYLE[k][1], 0.35)
        t, p, e = r["train_s"], r["power_W"], r["psu_J"]
        ax_t.barh(yi, t - 1, left=1, height=hgt, color=c, lw=0, zorder=2)
        t_txt = f"{t:,.0f}".replace(",", "{,}") if t >= 100 else (f"{t:.1f}" if t >= 10 else f"{t:.2f}")
        ax_t.text(t * 1.25, yi, t_txt + r"\,s", va="center", fontsize=fs)

        idle_w = r["idle_W"]
        ax_p.barh(yi, idle_w, height=hgt, color=lc, lw=0, zorder=2)
        ax_p.barh(yi, p - idle_w, left=idle_w, height=hgt, color=c, lw=0, zorder=2)
        ax_p.text(p + 20, yi, rf"{p:.0f}\,W", va="center", fontsize=fs)

        cpu = r["P_cpu0"] + r["P_cpu1"]
        gpu = r["P_gpu"] if k in ("gpuonly", "cuda") else 0.0
        if gpu:
            ax_c.barh(yi, cpu, height=hgt, fc=c, ec=STRIPE[k], lw=0, hatch=HATCH, zorder=2)
            ax_c.barh(yi, gpu, left=cpu, height=hgt, color=c, lw=0, zorder=2)
        else:
            ax_c.barh(yi, cpu, height=hgt, color=c, lw=0, zorder=2)
        ax_c.text(cpu + gpu + 18, yi, rf"{cpu + gpu:.0f}\,W", va="center", fontsize=fs)

        ax_e.barh(yi, e - 1e3, left=1e3, height=hgt, color=c, lw=0, zorder=2)
        ax_e.text(e * 1.25, yi, kj(e), va="center", fontsize=fs)

    ax_t.set_xscale("log")
    ax_t.set_xlim(1, 1.2e5)
    ax_t.set_xticks([1, 100, 1e4], [r"1\,s", r"100\,s", r"10k\,s"])
    ax_t.xaxis.set_minor_formatter(NullFormatter())
    ax_t.set_title("time per tree")

    ax_p.set_xlim(0, 1450)
    ax_p.set_xticks([0, 500, 1000], [r"0", r"500", r"1k\,W"])
    ax_p.set_title("server power")

    ax_c.set_xlim(0, 1050)
    ax_c.set_xticks([0, 400, 800], [r"0", r"400", r"800\,W"])
    ax_c.set_title("CPU+GPU power")

    joule_axis(ax_e, "x")
    ax_e.set_xlim(1e3, 4e7)
    ax_e.set_xticks([1e3, 1e5, 1e7])
    ax_e.set_title("energy per tree")

    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.axhline(gap_y, color=MUTED, lw=0.6, zorder=1)

    ax_t.set_yticks([yi for yi, _, _ in entries], [STYLE[k][0] for _, k, _ in entries], fontsize=FS_TICK)
    ax_t.set_ylim(mn_end + 0.55, -0.55)

    fig.canvas.draw()
    renderer = fig._get_renderer()
    max_lbl_pt = max(t.get_window_extent(renderer).width for t in ax_t.get_yticklabels()) * 72.0 / fig.dpi
    line_off = -(max_lbl_pt + 5.0)
    lbl_off = line_off - 8.5
    t_line = offset_copy(ax_t.get_yaxis_transform(), fig=fig, x=line_off, y=0, units="points")
    t_lbl = offset_copy(ax_t.get_yaxis_transform(), fig=fig, x=lbl_off, y=0, units="points")
    for grp_lbl, y0, y1 in [(r"\textbf{(a)} ECBDL'14" + "\n" + r"15M, $d \le 10$", 0.0, ec_end),
                            (r"\textbf{(b)} MNIST8M" + "\n" + r"no limit", mn_start, mn_end)]:
        ymid = 0.5 * (y0 + y1)
        ax_t.text(0, ymid, grp_lbl, transform=t_lbl,
                  ha="center", va="center", rotation=90, fontsize=FS_ANN, color=INK, linespacing=0.95)
        ax_t.plot([0, 0], [y0 - 0.25, y1 + 0.25], transform=t_line, color=MUTED, lw=0.6, clip_on=False)

    h = [plt.Rectangle((0, 0), 1, 1, color=to_rgba("#777", 0.35), label="idle server"),
         plt.Rectangle((0, 0), 1, 1, fc=STYLE["gpuonly"][1], ec=STRIPE["gpuonly"], lw=0, hatch=HATCH,
                       label="CPU part of a GPU run")]
    legend_below(fig, h, handlelength=2.4, handleheight=1.0, gap=0.03)
    save(fig, "results-energy", final=True, copy=POWER.parent / "paper-fig-energy-ecbdl_15m")


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
        # stripes mark the CPU part of a GPU run, as in energy(); the Multicore bar is all CPU
        ax_c.barh(yi, cpu, height=hgt, fc=c, ec=STRIPE[k], lw=0, hatch=HATCH if gpu else None)
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
                       label="CPU part of a GPU run")]
    legend_below(fig, h, handlelength=2.4, handleheight=1.0)
    save(fig, "results-energy-mnist8m", final=True, copy=POWER_DEEP.parent / "paper-fig-energy-mnist8m")

# ================================================================ Section 5 figures (final names)
STYLE.update({
    "rpart":  ("rpart", "#8c6d31", "v"),
    "yadt":   ("YaDT", "#5b6b7a", "P"),
    "yadt48": ("YaDT 48T", "#2f3e4d", "P"),
})


# Section 5 unified phase palette (distinct from implementation colors in STYLE).
GPU_C, CPU_C = STYLE["gpuonly"][1], STYLE["cpu"][1]
PHASE_C = {
    "setup": "#a3a29d",
    "other": "#cfcec9",
    "transfer": "#eda100",
    "presort": "#a8bedb",
    "search": "#6f8ab0",
    "partition": "#3c5270",
    "gpu": GPU_C,
    "cpu": CPU_C,
}
PHASE = [  # gpu_steps_chart bar -> (colour, legend label, lane offset: -0.19 host / +0.19 GPU)
    ("set-up", PHASE_C["setup"], "setup", -0.19),
    ("upload", PHASE_C["transfer"], "upload", -0.19),
    ("prepare", PHASE_C["presort"], "presort / binning (GPU)", 0.19),
    ("build", PHASE_C["gpu"], "build (GPU)", 0.19),
    ("CPU build", PHASE_C["cpu"], "small subtrees (CPU)", -0.19),
]


def gpulibs():
    """One CART tree on the H100: 2+1 layout with HIGGS d<=10 and MNIST8M d<=10 on top and
    HIGGS d<=30 full-width on bottom using a true broken axis (0..1.15 s | 1.15..38 s).
    Each tool is a single horizontal bar; concurrent phases are hatched (matching Figs. 6b & 9)."""
    import gpu_steps_chart as G
    from matplotlib.patches import Patch

    tool_order = [("tree_gpu_only", "GPU-Based"), ("tree_cuda", "Hybrid"),
                  ("cuml", "cuML"), ("xgboost", "XGBoost")]
    order = {k: i for i, (k, _) in enumerate(tool_order)}
    sec_color = {
        "set-up": PHASE_C["setup"],
        "upload": PHASE_C["transfer"],
        "prepare": PHASE_C["presort"],
        "build": PHASE_C["gpu"],
        "CPU build": PHASE_C["cpu"],
    }

    def load_panel(p):
        tools = sorted(G.load("hopper", p), key=lambda t: order[t["impl"]])
        USED.extend(False for _ in tools)
        return tools

    def fmt_total(t, ref):
        m = t["mean"]
        tt = f"{m:.2f}" if m < 10 else f"{m:.1f}"
        if t["impl"] == "tree_gpu_only":
            return rf"{tt}\,s"
        r = m / ref
        rs = f"{r:.2f}" if r < 1 else (f"{r:.0f}" if r >= 10 else f"{r:.1f}")
        return rf"{tt}\,s ({rs}$\times$)"

    def split_intervals(tool):
        segs, pts = [], set()
        for s in tool["sections"]:
            for a, b in s["segments"]:
                segs.append((a, b, s["name"]))
                pts.add(a)
                pts.add(b)
        pts = sorted(pts)
        out = []
        for a, b in zip(pts, pts[1:]):
            if b - a < 1e-6:
                continue
            mid = 0.5 * (a + b)
            active = {name for sa, sb, name in segs if sa <= mid <= sb}
            if active:
                out.append((a, b, active))
        return out

    def draw_bar(ax, yi, tool, h=0.58, hatch="////"):
        plt.rcParams["hatch.linewidth"] = 1.0
        max_x = 0.0
        for a, b, active in split_intervals(tool):
            max_x = max(max_x, b)
            w = b - a
            if len(active) == 1:
                name = next(iter(active))
                ax.barh(yi, w, left=a, height=h, color=sec_color[name], lw=0, zorder=2)
            elif "build" in active and "CPU build" in active:
                ax.barh(yi, w, left=a, height=h, fc=PHASE_C["gpu"], ec=PHASE_C["cpu"],
                        hatch=hatch, lw=0, zorder=2)
            elif "upload" in active and "prepare" in active:
                ax.barh(yi, w, left=a, height=h, fc=PHASE_C["presort"], ec=PHASE_C["transfer"],
                        hatch=hatch, lw=0, zorder=2)
            else:
                name = next(iter(active))
                ax.barh(yi, w, left=a, height=h, color=sec_color[name], lw=0, zorder=2)
        return max_x

    fig = plt.figure(figsize=(FULL, 2.02))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.52)
    top_gs = outer[0].subgridspec(1, 2, wspace=0.14)
    bot_gs = outer[1].subgridspec(1, 2, width_ratios=[0.66, 0.34], wspace=0.05)

    ax_h10 = fig.add_subplot(top_gs[0, 0])
    ax_m10 = fig.add_subplot(top_gs[0, 1], sharey=ax_h10)
    ax_bl = fig.add_subplot(bot_gs[0, 0])
    ax_br = fig.add_subplot(bot_gs[0, 1], sharey=ax_bl)

    for ax, p, title, xhi, ticks in [
        (ax_h10, "higgs", r"\textbf{(a) HIGGS, depth $\le$ 10}", 0.82, [0, 0.2, 0.4, 0.6]),
        (ax_m10, "mnist8m", r"\textbf{(b) MNIST8M, depth $\le$ 10}", 10.4, [0, 2, 4, 6, 8]),
    ]:
        tools = load_panel(p)
        ref = next(t["mean"] for t in tools if t["impl"] == "tree_gpu_only")
        ax.set_xlim(0, xhi)
        for yi, t in enumerate(tools):
            end_x = draw_bar(ax, yi, t, h=0.58, hatch="////")
            ax.text(end_x + 0.018 * xhi, yi, fmt_total(t, ref), va="center", ha="left",
                    fontsize=FS_ANN - 0.2, color=INK)
            if yi > 0:
                ax.axhline(yi - 0.5, color=GRID, lw=0.4, zorder=0)
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.set_ylim(len(tool_order) - 0.5, -0.5)
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, pad=2)
    ax_h10.set_yticks(range(len(tool_order)), [l for _, l in tool_order], fontsize=FS_TICK)
    ax_m10.tick_params(labelleft=False)

    tools = load_panel("higgs-depth30")
    ref = next(t["mean"] for t in tools if t["impl"] == "tree_gpu_only")
    split = 1.15
    ax_bl.set_xlim(0, split)
    ax_br.set_xlim(split, 38.0)
    for yi, t in enumerate(tools):
        draw_bar(ax_bl, yi, t, h=0.58, hatch="///")
        draw_bar(ax_br, yi, t, h=0.58, hatch="///")
        if t["mean"] <= split:
            ax_bl.text(t["mean"] + 0.02, yi, fmt_total(t, ref), va="center", ha="left",
                       fontsize=FS_ANN - 0.2, color=INK)
        else:
            ax_br.text(t["mean"] + 0.8, yi, fmt_total(t, ref), va="center", ha="left",
                       fontsize=FS_ANN - 0.2, color=INK)
        if yi > 0:
            ax_bl.axhline(yi - 0.5, color=GRID, lw=0.4, zorder=0)
            ax_br.axhline(yi - 0.5, color=GRID, lw=0.4, zorder=0)
    ax_bl.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax_br.set_xticks([5, 15, 25], ["5", "15", "25"])
    ax_bl.set_yticks(range(len(tool_order)), [l for _, l in tool_order], fontsize=FS_TICK)
    ax_bl.set_ylim(len(tool_order) - 0.5, -0.5)
    ax_br.tick_params(axis="y", length=0, labelleft=False)
    ax_br.spines["left"].set_visible(False)
    for ax in (ax_bl, ax_br):
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    d = 0.015
    ax_bl.plot([1 - d, 1 + d], [-0.05, 0.05], transform=ax_bl.transAxes, color=MUTED, lw=0.7, clip_on=False)
    ax_br.plot([-d * 2, d * 2], [-0.05, 0.05], transform=ax_br.transAxes, color=MUTED, lw=0.7, clip_on=False)
    ax_bl.set_title(r"\textbf{(c) HIGGS, depth $\le$ 30}", pad=2)
    fig.supxlabel("seconds since timer started", fontsize=FS_TEXT, y=-0.02)

    plt.rcParams["hatch.linewidth"] = 1.0
    h = [
        Patch(fc=PHASE_C["setup"], lw=0, label="setup"),
        Patch(fc=PHASE_C["transfer"], lw=0, label="upload"),
        Patch(fc=PHASE_C["presort"], lw=0, label="presort / binning"),
        Patch(fc=PHASE_C["gpu"], lw=0, label="build (GPU)"),
        Patch(fc=PHASE_C["cpu"], lw=0, label="subtrees (CPU)"),
        Patch(fc="white", ec=MUTED, hatch="////", lw=0.4, label="concurrent"),
    ]
    legend_below(fig, h, handlelength=1.1, gap=0.03, ncol=3)
    save(fig, "results-gpulibs", final=True)


def libs():
    """Every implementation, depth <= 10, CART (top) and C4.5 (bottom), four aligned horizontal
    panels: HIGGS time (log) and peak RAM/VRAM (GiB), MNIST8M time (log) and peak RAM/VRAM (GiB).
    Sources: cluster-cpu-vs-others/vs-epyc-<ds>, cluster-gpu-vs-others/one-tree (H100),
    cluster-<ds> for C4.5 GPU runs and Serial Simple."""
    C, Q = "cart_depth10", "c45_depth10_unpruned"

    def runs(path):
        return [r for r in (json.loads(l) for l in open(path / "results.jsonl"))
                if r.get("status") == "ok" and r.get("kind", "run") == "run"]

    def pick(rows, impl, proto, threads=None):
        rs = [r for r in rows if r["impl"] == impl and r.get("protocol") == proto
              and (threads is None or r.get("threads") == threads)]
        if not rs:
            return None
        USED.append(B.rows_out_of_date(rs))
        vram = [r["gpu_peak_bytes"] for r in rs if r.get("gpu_peak_bytes")]
        return (statistics.mean(r["train_seconds"] for r in rs),
                statistics.mean(r["peak_rss_train_bytes"] for r in rs) / 2**30,
                statistics.mean(vram) / 2**30 if vram else None)

    def fmt_mem(v, ds):
        return f"{v:.1f}" if ds == "higgs" else f"{v:.0f}"

    cart_groups = [
        ("Serial Simple", "simple", "tree_simple", C, None),
        ("Serial Opt.", "cpu", "tree", C, 1),
        ("Multicore", "cpu", "tree", C, 48),
        ("GPU-Based", "gpu", "tree_gpu_only", C, None),
        ("Hybrid", "gpu", "tree_cuda", C, None),
        ("scikit-learn", "cpu", "sklearn", C, None),
        ("rpart", "cpu", "rpart", C, None),
        ("cuML", "gpu", "cuml", C, None),
        ("XGBoost", "gpu", "xgboost", C, None),
    ]
    c45_groups = [
        ("Serial Simple", "simple", "tree_simple", Q, None),
        ("Serial Opt.", "cpu", "tree", Q, 1),
        ("Multicore", "cpu", "tree", Q, 48),
        ("GPU-Based", "gpuonly", "tree_gpu_only", Q, None),
        ("Hybrid", "cuda", "tree_cuda", Q, None),
        ("Weka J48", "cpu", "j48", Q, None),
        ("YaDT", "cpu", "yadt", Q, 1),
        ("YaDT 48T", "cpu", "yadt", Q, 48),
    ]
    # y positions (top to bottom): CART rows, small gap, C4.5 rows
    rows_meta = []
    y = 0.0
    for g in cart_groups:
        rows_meta.append((y, g))
        y += 1.0
    cart_end = y - 1.0
    gap_y = y - 0.15
    y += 0.7
    c45_start = y
    for g in c45_groups:
        rows_meta.append((y, g))
        y += 1.0
    c45_end = y - 1.0

    TIMEC, RAMC, VRAMC = "#7e7d78", CPU_C, GPU_C

    fig = plt.figure(figsize=(FULL, 2.35))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.14, 1.0, 1.0], wspace=0.10)
    ax_ht = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1], sharey=ax_ht)
    ax_mt = fig.add_subplot(gs[0, 3], sharey=ax_ht)
    ax_mm = fig.add_subplot(gs[0, 4], sharey=ax_ht)

    panels = [
        ("higgs", "HIGGS", ax_ht, ax_hm, 0.1, 7e3, [0.1, 10, 1000], 18.5, [0, 5, 10, 15]),
        ("mnist8m", "MNIST8M", ax_mt, ax_mm, 1.0, 2e5, [1, 100, 10000], 106, [0, 40, 80]),
    ]
    fs_val = FS_ANN - 0.3

    for ds, ds_title, ax_t, ax_m, t_lo, t_hi, t_ticks, m_hi, m_ticks in panels:
        d = DATA[ds][1]
        src = {"cpu": runs(RES / "cluster-cpu-vs-others" / f"vs-epyc-{ds}"),
               "gpu": runs(RES / "cluster-gpu-vs-others" / "one-tree" / f"vs-hopper-{ds}"),
               "simple": runs(RES / d / f"simple-epyc-{ds}"),
               "gpuonly": runs(RES / d / f"gpuonly-hopper-{ds}"),
               "cuda": runs(RES / d / f"cuda-hopper-{ds}")}
        for yi, (label, s, impl, proto, th) in rows_meta:
            v = pick(src[s], impl, proto, th)
            if v is None:
                ax_t.text(t_lo * 1.2, yi, r"\textit{fails}", va="center", ha="left",
                          fontsize=fs_val, color=INK)
                ax_m.text(0.03 * m_hi, yi, r"--", va="center", ha="left",
                          fontsize=fs_val, color=INK)
                continue
            t, ram, vram = v
            ax_t.barh(yi, t - t_lo, left=t_lo, height=0.68, color=TIMEC, lw=0, zorder=2)
            ax_t.text(t * 1.28, yi, tsec(t), va="center", ha="left", fontsize=fs_val, color=INK)
            if vram is None:
                ax_m.barh(yi, ram, height=0.68, color=RAMC, lw=0, zorder=2)
                ax_m.text(ram + 0.025 * m_hi, yi, fmt_mem(ram, ds), va="center", ha="left",
                          fontsize=fs_val, color=INK)
            else:
                ax_m.barh(yi - 0.18, ram, height=0.34, color=RAMC, lw=0, zorder=2)
                ax_m.barh(yi + 0.18, vram, height=0.34, color=VRAMC, lw=0, zorder=2)
                m_end = max(ram, vram)
                lbl = f"{fmt_mem(ram, ds)}/{fmt_mem(vram, ds)}"
                ax_m.text(m_end + 0.025 * m_hi, yi, lbl, va="center", ha="left",
                          fontsize=fs_val, color=INK)

        log_axis(ax_t, "x")
        ax_t.set_xlim(t_lo, t_hi)
        ax_t.set_xticks(t_ticks)
        ax_t.set_title("time (s)", loc="left", fontsize=FS_TEXT - 0.5, pad=2)

        ax_m.set_xlim(0, m_hi)
        ax_m.set_xticks(m_ticks)
        ax_m.set_title("memory (GiB)", loc="left", fontsize=FS_TEXT - 0.5, pad=2)

        for ax in (ax_t, ax_m):
            ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.axhline(4.5, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
            ax.axhline(c45_start + 4.5, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
            ax.axhline(gap_y, color=MUTED, lw=0.6, zorder=1)

    ax_ht.set_yticks([yi for yi, _ in rows_meta], [tex(g[0]) for _, g in rows_meta], fontsize=FS_TICK - 0.2)
    ax_ht.set_ylim(c45_end + 0.55, -0.55)
    for ax in (ax_hm, ax_mt, ax_mm):
        ax.tick_params(labelleft=False)

    # Dataset group titles centered over each (time, memory) pair
    fig.canvas.draw()
    for ds_title, a0, a1 in [("HIGGS", ax_ht, ax_hm), ("MNIST8M", ax_mt, ax_mm)]:
        xmid = 0.5 * (a0.get_position().x0 + a1.get_position().x1)
        ytop = a0.get_position().y1
        fig.text(xmid, ytop + 0.055, rf"\textbf{{{ds_title}}}", ha="center", va="bottom", fontsize=FS_TEXT)

    # Algorithm group labels computed dynamically from actual y-tick label widths
    from matplotlib.transforms import offset_copy
    renderer = fig._get_renderer()
    max_lbl_pt = max(t.get_window_extent(renderer).width for t in ax_ht.get_yticklabels()) * 72.0 / fig.dpi
    line_off = -(max_lbl_pt + 5.0)
    lbl_off = line_off - 7.0
    t_line = offset_copy(ax_ht.get_yaxis_transform(), fig=fig, x=line_off, y=0, units="points")
    t_lbl = offset_copy(ax_ht.get_yaxis_transform(), fig=fig, x=lbl_off, y=0, units="points")
    for grp_lbl, y0, y1 in [("CART", 0.0, cart_end), ("C4.5", c45_start, c45_end)]:
        ymid = 0.5 * (y0 + y1)
        ax_ht.text(0, ymid, rf"\textbf{{{grp_lbl}}}", transform=t_lbl,
                   ha="center", va="center", rotation=90, fontsize=FS_TEXT - 0.5, color=INK)
        ax_ht.plot([0, 0], [y0 - 0.25, y1 + 0.25], transform=t_line,
                   color=MUTED, lw=0.6, clip_on=False)

    h = [plt.Rectangle((0, 0), 1, 1, color=TIMEC, label="training time (log)"),
         plt.Rectangle((0, 0), 1, 1, color=RAMC, label="peak RAM"),
         plt.Rectangle((0, 0), 1, 1, color=VRAMC, label="peak VRAM")]
    legend_below(fig, h, handlelength=1.0, gap=0.03)
    save(fig, "results-libs", final=True)


def depth_ratio():
    """Hybrid against GPU-Based by depth limit: GPU-Based time / Hybrid time (above 1: Hybrid
    faster), HIGGS, MNIST8M, SOREL 4M, H100 (filled) and A100 (hollow), CART solid, C4.5 dashed;
    below, active-device breakdown during unrestricted MNIST8M CART on the H100."""
    import hybrid_profile_chart as HP
    PARTS = [(k, n, {"setup": PHASE_C["setup"], "gpu": PHASE_C["gpu"], "both": PHASE_C["gpu"],
                     "cpu": PHASE_C["cpu"], "transfer": PHASE_C["transfer"],
                     "host": PHASE_C["other"]}.get(k, c)) for k, n, c in HP.PARTS]
    root = RES / "hybrid-vs-gpu-only"
    depths = ["5", "10", "15", "20", "25", "30", "none"]
    proto = {"cart": [f"cart_depth{d}" for d in depths[:-1]] + ["cart_full_nodepth"],
             "c45": [f"c45_depth{d}_unpruned" for d in depths[:-1]] + ["c45_unpruned"]}
    sets = [("higgs", r"(a) HIGGS"), ("mnist8m", "MNIST8M"), ("sorel_4m", "SOREL 4M")]
    fig = plt.figure(figsize=(FULL, 1.82))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.36], hspace=0.72, wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    x = np.arange(len(depths))
    c_node = {"hopper": STYLE["cuda"][1], "ampere": "#7fcf9f"}
    for ax, (ds, name) in zip(axes, sets):
        ax.axhline(1, color=MUTED, lw=0.6, ls="--", zorder=1)
        for node in ("hopper", "ampere"):
            hy, go = summary(root / f"cuda-{node}-{ds}"), summary(root / f"gpuonly-{node}-{ds}")
            for algo, ls in (("cart", "-"), ("c45", "--")):
                r = [go[p]["time_mean"] / hy[p]["time_mean"] for p in proto[algo]]
                ax.plot(x, r, ls=ls, color=c_node[node], marker="o" if node == "hopper" else "s",
                        mfc=c_node[node] if node == "hopper" else "white", ms=2.6, mew=0.6, lw=0.8, zorder=3)
                if node == "hopper" and algo == "cart":
                    ax.text(x[-1] + 0.25, r[-1], rf"{r[-1]:.2f}$\times$", fontsize=FS_ANN, va="center",
                            color=c_node[node])
        ax.set_xticks(x, ["5", "10", "15", "20", "25", "30", r"$\infty$"])
        ax.set_xlim(-0.4, len(depths) + 0.15)
        ax.set_ylim(0.45, 2.05)
        ax.set_yticks([0.5, 1, 1.5, 2])
        ax.grid(axis="y", color=GRID, lw=0.3, zorder=0)
        ax.set_title(rf"\textbf{{{name}}}", pad=2)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    axes[1].set_xlabel("depth limit", labelpad=1.5)
    axes[0].set_ylabel("Hybrid speedup")
    h = [Line2D([], [], color=c_node["hopper"], marker="o", ms=2.6, lw=0.8, label="H100"),
         Line2D([], [], color=c_node["ampere"], marker="s", mfc="white", ms=2.6, lw=0.8, label="A100"),
         Line2D([], [], color=MUTED, lw=0.8, ls="-", label="CART"),
         Line2D([], [], color=MUTED, lw=0.8, ls="--", label="C4.5")]
    axes[1].legend(handles=h, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.14), handlelength=1.8)

    cases, machine, _ = HP.load(root, "hopper", "mnist8m")
    axp = fig.add_subplot(gs[1, :])
    rows = [("cuda", "Hybrid"), ("gpuonly", "GPU-Based")]
    for yi, (mode, lbl) in enumerate(rows):
        case = cases[(mode, "cart_full_nodepth")]
        USED.append(case["stale"])
        parts = case["parts"]
        total = sum(parts.values())
        left = 0.0
        for key, _, color in PARTS:
            v = parts[key] / total
            if key == "both":
                axp.barh(yi, v, left=left, height=0.72, fc=color, ec=CPU_C, lw=0, hatch="////")
            else:
                axp.barh(yi, v, left=left, height=0.72, color=color, lw=0)
            if v > 0.07:
                axp.text(left + v / 2, yi, f"{100 * v:.0f}", ha="center", va="center", fontsize=FS_ANN - 0.3,
                         color=HP.text_color(color) if key != "both" else "white")
            left += v
        axp.text(1.01, yi, rf"{case['row']['train_seconds']:.1f}\,s", va="center", fontsize=FS_ANN,
                 transform=axp.get_yaxis_transform())
    axp.set_yticks(range(len(rows)), [l for _, l in rows])
    axp.set_ylim(len(rows) - 0.5, -0.5)
    axp.set_xlim(0, 1)
    axp.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    axp.set_xlabel(r"share of training time (\%)", labelpad=1.5)
    axp.tick_params(axis="y", length=0)
    axp.spines["left"].set_visible(False)
    axp.set_title(r"\textbf{(b) MNIST8M, CART, no depth limit, H100}", pad=2)
    plt.rcParams["hatch.linewidth"] = 1.0
    hp = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=CPU_C if k == "both" else c, lw=0,
                        hatch="////" if k == "both" else None,
                        label=tex(n).replace("↔", r"$\leftrightarrow$").replace("Setup (GPU init, upload, presort)", "setup"))
          for k, n, c in PARTS]
    legend_below(fig, hp, handlelength=1.0, gap=0.03, ncol=3)
    save(fig, "results-depth", final=True)


def chess():
    """Chess 10^7 to 2*10^9 samples, CART, depth <= 10: three panels (Multicore CPU on epyc,
    GPU-Based on the H100, Hybrid on the H100), a 100 % bar per sample count split by step,
    the total time above. Source cluster-profiling."""
    import profile_share_chart as P
    datasets = ["chess_10m", "chess_100m", "chess_1b", "chess_2b"]
    labels = [r"$10^7$", r"$10^8$", r"$10^9$", r"$2 \cdot 10^9$"]
    modes = [("cpu-epyc", "Multicore CPU (epyc)"), ("gpuonly-hopper", "GPU-Based (H100)"),
             ("cuda-hopper", "Hybrid (H100)")]
    steps = [("presort", "presort", PHASE_C["presort"]),
             ("search", "split search", PHASE_C["search"]),
             ("partition", "partition", PHASE_C["partition"]),
             ("transfer", r"CPU$\leftrightarrow$GPU transfer", PHASE_C["transfer"]),
             ("other", "setup and other", PHASE_C["other"])]
    cases = P.load(RES / "cluster-profiling", datasets)
    USED.extend(c["stale"] for c in cases.values())
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.42), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (mode, title) in zip(axes, modes):
        for x, ds in enumerate(datasets):
            case = cases[(mode, ds, "cart_depth10")]
            parts = case["parts"]
            total = sum(parts.values())
            bottom = 0.0
            for step, _, c in steps:
                v = 100 * sum(val for (s, _), val in parts.items() if s == step) / total
                ax.bar(x, v, 0.54, bottom=bottom, color=c, lw=0, zorder=2)
                if v >= 10:
                    ax.text(x, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=FS_ANN - 0.3,
                            color=P.text_color(c))
                bottom += v
            t = case["row"]["train_seconds"]
            tlbl = f"{t:.2f}" if t < 1 else tsec(t)
            ax.text(x, 102, tlbl, ha="center", va="bottom", fontsize=FS_ANN)
        ax.set_xticks(range(len(datasets)), labels)
        ax.set_xlim(-0.55, len(datasets) - 0.45)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", length=0)
        ax.set_title(rf"\textbf{{{title}}}", pad=10)
    axes[0].set_ylabel(r"share of time (\%)")
    fig.supxlabel("samples", fontsize=FS_TEXT, y=-0.04)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=n) for _, n, c in steps]
    legend_below(fig, h, handlelength=1.0, gap=0.03)
    save(fig, "results-chess", final=True)


def scaling_final():
    """scaling() with the paper's server names, saved as results-scaling."""
    global save
    orig = save
    save = lambda fig, name, **kw: orig(fig, "results-scaling", final=True)  # noqa: E731
    try:
        scaling()
    finally:
        save = orig



if __name__ == "__main__":
    which = sys.argv[1:] or ["bars", "gpulibs", "libs", "depth_ratio", "chess", "scaling_final", "energy",
                             "energy_deep"]
    for name in which:
        globals()[name]()
