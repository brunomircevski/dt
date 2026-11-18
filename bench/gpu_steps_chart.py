#!/usr/bin/env python3
"""cluster-gpu-vs-others: one CART tree (depth <= 10; HIGGS also <= 30) per tool on one GPU, as a
timeline with the same bars for every tool (set-up, upload, prepare, build;
--hybrid also its hand-off buffers and CPU build), each a thin bar on its own lane from the
moment it started to the moment it ended, so bars that ran at the same time
(./tree's presort during its upload) overlap. ./tree --gpu and --hybrid, cuML,
XGBoost; every run a fresh process with no warm-up, all uploading through
./tree's staged upload. Reads cluster-gpu-vs-others/one-tree/vs-<node>-<panel>/results.jsonl
(bench/run.py; each row's "timeline"), writes chart.png and summary.csv there. A
panel is a dataset at depth <= 10, or <dataset>-depth<D> at depth <= D.

  bench/.venv/bin/python bench/gpu_steps_chart.py [--node hopper] [--panels higgs,mnist8m,higgs-depth30]

Where the times come from (all since the timer started, the mean of the runs):
  ./tree    its "span" lines: host clock around each section; the presort's
            span on the GPU from CUDA events (its first kernel's start, its last
            one's end); --hybrid's CPU subtrees from the first task's start to
            the last one's end, with the tasks' summed work
  cuML,     gpubench.py: the host clock at each section's end, after
  XGBoost   cudaDeviceSynchronize, so its GPU work is inside it
"""
import argparse, csv, json, statistics as st, textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

RESULTS = Path(__file__).resolve().parent / "results" / "cluster-gpu-vs-others"
TOOLS = [("tree_gpu_only", "./tree --gpu"), ("tree_cuda", "./tree --hybrid"),
         ("cuml", "cuML"), ("xgboost", "XGBoost")]
SPLITS = {"tree_gpu_only": "exact", "tree_cuda": "exact", "cuml": "128 bins", "xgboost": "256 bins"}
INK, MUTED, GRID, WHITE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# The same sections for every tool; each run's detailed timeline (results.jsonl,
# summary.csv) is merged into them. category -> (colour, legend); the reference
# palette's slots 1-4; hand-off buffers and CPU build lighter steps of set-up's and build's
CATEGORIES = {
    "set-up": ("#2a78d6", "Set-up: staging buffers, copy threads; ./tree also all its device memory at once "
                          "(the libraries allocate theirs inside their other bars)"),
    "hand-off buffers": ("#86b6ef", "Hand-off buffers: ./tree --hybrid's 256 MB page-locked buffers for handing nodes to the CPUs"),
    "upload": ("#eb6834", "Upload to the GPU: the same code for every tool"),
    "prepare": ("#1baf7a", "Prepare: ./tree's presort (during the upload) and x·log x table · cuML's input checks · "
                           "XGBoost's quantile binning"),
    "build": ("#eda100", "Build the tree"),
    "CPU build": ("#f5c96a", "CPU build: ./tree --hybrid finishes small subtrees on the node's CPUs during its build "
                             "(from the first subtree's start to the last one's end)"),
}
# a recorded section -> the bar it is drawn in
BAR_OF = {
    "staging buffers": "set-up", "device memory": "set-up", "hand-off buffers": "hand-off buffers",
    "upload": "upload",
    "presort (GPU)": "prepare", "x log x table": "prepare", "x log x table copy": "prepare",
    "input checks": "prepare", "quantile binning": "prepare", "free raw copy": "prepare",
    "build": "build", "build tree": "build", "fit's end": "build",
    "cpu subtrees": "CPU build",
}
TITLES = {"higgs": "HIGGS (10.5M rows × 28 features, 2 classes)",
          "mnist8m": "MNIST8M (8.1M rows × 784 features, 10 classes)"}
LANE = 1.0       # one lane per bar
BAR = 0.62       # the bar's thickness in a lane
GAP = 0.6        # space between tools
HEAD = 1.1       # the tool's lane (its total, then its facts below it) to its first bar
INCH = 0.15      # one lane's height in inches


def panel_of(panel):
    """"higgs" -> ("higgs", 10), "higgs-depth30" -> ("higgs", 30)."""
    dataset, _, depth = panel.partition("-depth")
    return dataset, int(depth or 10)


def load(node, panel, merged=True):
    """Each tool's runs: its bars (merged) or its recorded sections, mean start and end."""
    dataset, depth = panel_of(panel)
    path = RESULTS / "one-tree" / f"vs-{node}-{panel}" / "results.jsonl"
    rows = [json.loads(line) for line in open(path)] if path.exists() else []
    rows = [r for r in rows if r.get("status") == "ok" and r.get("kind", "run") == "run"
            and r.get("protocol", "cart_depth10") == f"cart_depth{depth}" and r.get("timeline")]
    tools = []
    for impl, label in TOOLS:
        mine = [r for r in rows if r["impl"] == impl]
        if not mine:
            continue
        spans = defaultdict(list)  # name -> [(start, end, busy, tasks)] over the runs
        order = []
        for r in mine:
            timeline = {}
            for s in r["timeline"]:
                name = BAR_OF.get(s["name"], "build") if merged else s["name"]
                if name is None:
                    continue
                timeline.setdefault(name, []).append((s["start"], s["end"], s.get("busy", 0.0), s.get("tasks", 0)))
            for name, pieces in timeline.items():
                if name not in spans:
                    order.append(name)
                spans[name].append(pieces)
        sections = []
        for name in order:
            runs = [joined(pieces) for pieces in spans[name]]  # each run: [(start, end)], touching pieces joined
            if len({len(r) for r in runs}) > 1:  # the runs disagree: one segment from first start to last end
                runs = [[(r[0][0], r[-1][1])] for r in runs]
            segments = [(st.mean(r[i][0] for r in runs), st.mean(r[i][1] for r in runs)) for i in range(len(runs[0]))]
            raw = spans[name]
            sections.append({"name": name, "start": segments[0][0], "end": segments[-1][1], "segments": segments,
                             "seconds": st.mean(sum(e - b for b, e in r) for r in runs), "runs": len(runs),
                             "busy": st.mean(sum(p[2] for p in pieces) for pieces in raw),
                             "tasks": st.mean(sum(p[3] for p in pieces) for pieces in raw)})
        sections.sort(key=lambda s: (s["start"], s["end"]))
        times = [r["train_seconds"] for r in mine]
        vram = [r["gpu_peak_bytes"] for r in mine if r.get("gpu_peak_bytes")]
        last = mine[-1]
        tools.append({"impl": impl, "label": label + ("" if impl.startswith("tree") else f" {last.get('version', '')}")
                      + f" · {SPLITS[impl]}",
                      "n": len(mine), "mean": st.mean(times), "sd": st.stdev(times) if len(times) > 1 else 0.0,
                      "min": min(times), "max": max(times), "sections": sections,
                      "vram": st.mean(vram) if vram else None, "test": last.get("test_accuracy"),
                      "nodes": last.get("nodes"), "gpu": last.get("gpu"), "commit": last.get("commit")})
    return tools


def joined(pieces, slack=0.002):
    """Pieces of one bar in one run as segments: pieces that touch or overlap
    (within `slack` seconds) are one segment."""
    out = []
    for start, end, *_ in sorted(pieces):
        if out and start <= out[-1][1] + slack:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def duration(seconds):
    if seconds < 0.0095:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds * 1000:.0f} ms" if seconds < 0.9995 else f"{seconds:.2f} s"


def lanes_of(tools):
    """The y of every lane, top to bottom: [(y, tool, section or None for the header)]."""
    y, out = 0.0, []
    for t in tools:
        out.append((y, t, None))
        y += HEAD
        for s in t["sections"]:
            out.append((y, t, s))
            y += LANE
        y += GAP
    return out, y


def windows_of(tools):
    """The x ranges the axis shows, at one scale: everything if the slowest tool
    is less than 4x ./tree's slowest; else ./tree's whole run, and around every
    later moment something ends (a library's last phases), with the empty
    stretches between them cut out of the axis."""
    lead = max((t["mean"] for t in tools if t["impl"].startswith("tree")), default=0)
    right = max(t["mean"] for t in tools)
    if not lead or right <= 4 * lead:
        return [(0.0, right * 1.1)]
    events = sorted({x for t in tools for s in t["sections"] for seg in s["segments"] for x in seg}
                    | {t["mean"] for t in tools})
    clusters = [[0.0, lead]]
    for x in events:
        if x <= clusters[-1][1] + lead:  # close to the last one: the same window
            clusters[-1][1] = max(clusters[-1][1], x)
        else:
            clusters.append([x, x])
    out = [(0.0, clusters[0][1] + 0.12 * lead)]
    out += [(lo - 0.12 * lead, hi + 0.3 * lead) for lo, hi in clusters[1:]]  # room for the label after the end
    return out


def axis_scale(ax, windows, gap):
    """Draw the windows side by side, each cut-out stretch as `gap` seconds of axis."""
    data, shown, x = [], [], 0.0
    for i, (lo, hi) in enumerate(windows):
        if i:
            x += gap
        data += [lo, hi]
        shown += [x, x + hi - lo]
        x += hi - lo
    data = [data[0] - 1e6] + data + [data[-1] + 1e6]  # linear beyond the ends
    shown = [shown[0] - 1e6] + shown + [shown[-1] + 1e6]
    ax.set_xscale("function", functions=(lambda v: np.interp(v, data, shown), lambda v: np.interp(v, shown, data)))
    ticks = []
    for i, (lo, hi) in enumerate(windows):
        ticks += [v for v in MaxNLocator(6 if i == 0 else 2).tick_values(lo, hi) if lo <= v <= hi]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    for (_, hi), (lo, _) in zip(windows, windows[1:]):  # the cut: a pale band and a break mark on the axis
        ax.axvspan(hi, lo, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.text((hi + lo) / 2, 0, "//", transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=9, color=MUTED, fontweight="bold", bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.5})
    ax.set_xlim(windows[0][0], windows[-1][1])
    ax.set_xlabel("seconds since the timer started (mean of the runs; // the axis skips a stretch where nothing ends)")


def draw(ax, tools):
    lanes, height = lanes_of(tools)
    windows = windows_of(tools)
    span = windows[0][1]  # the first window's width sets the smallest bar
    ref = next((t["mean"] for t in tools if t["impl"] == "tree_gpu_only"), None)
    min_width = span * 0.0025 / 1.1  # a section too short to see is drawn this wide
    ticks, names = [], []
    for y, t, s in lanes:
        if s is None:  # the tool's header lane: its whole training, and its total
            note = f"total {t['mean']:.3g} s"
            if ref and t["impl"] != "tree_gpu_only":
                ratio = t["mean"] / ref
                note += f" ({ratio:.2f}× ./tree --gpu)" if ratio < 1 else f" ({ratio:.1f}× ./tree --gpu)"
            total = ax.text(0, y, note, va="center", fontsize=8.5, color=INK,
                            fontweight="bold")
            ax.annotate(facts(t), xy=(1, 0.5), xycoords=total, xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=7, color=MUTED)
            ticks.append(y)
            names.append(t["label"])
            continue
        colour = CATEGORIES[s["name"]][0]
        label = s["name"]
        for start, end in s["segments"]:
            ax.barh(y, max(end - start, min_width), left=start, height=BAR, color=colour, linewidth=0)
        ax.annotate(duration(s["seconds"]), xy=(max(end, start + min_width), y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7, color=MUTED)
        ticks.append(y)
        names.append(label)
    ax.set_yticks(ticks, names)
    for tick, (_, _, s) in zip(ax.get_yticklabels(), lanes):
        tick.set_fontsize(7.5 if s else 9)
        tick.set_color(MUTED if s else INK)
        tick.set_fontweight("normal" if s else "bold")
    ax.set_ylim(height - GAP + 0.2, -0.8)
    if len(windows) > 1:
        axis_scale(ax, windows, gap=0.08 * span)
    else:
        ax.set_xlim(*windows[0])


def facts(t):
    out = []
    if t["test"] is not None:
        out.append(f"test {t['test'] * 100:.2f}%")
    if t["nodes"]:
        out.append(f"{t['nodes']:,} nodes")
    if t["vram"]:
        out.append(f"{t['vram'] / 2**30:.1f} GiB VRAM")
    out.append(f"{t['n']} runs, ±{t['sd'] * 1000:.0f} ms")
    return " · ".join(out)


def style(ax, title):
    ax.set_xlabel("seconds since the timer started (mean of the runs)", fontsize=8.5, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)


def write_summary(grid, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "depth_limit", "implementation", "gpu", "section", "bar", "start_mean", "end_mean",
                    "seconds_mean", "cpu_work_seconds", "cpu_tasks", "runs", "time_mean", "time_sd", "time_min",
                    "time_max", "vram_bytes", "test_accuracy", "nodes", "commit"])
        for panel, tools in grid:
            for t in tools:
                for s in [None] + t["sections"]:
                    w.writerow([*panel_of(panel), t["label"], t["gpu"],
                                "total" if s is None else s["name"],
                                "" if s is None else BAR_OF.get(s["name"], "build") or "(inside build)",
                                f"{0.0 if s is None else s['start']:.5f}",
                                f"{t['mean'] if s is None else s['end']:.5f}",
                                f"{t['mean'] if s is None else s['seconds']:.5f}",
                                f"{s['busy']:.4f}" if s and s["tasks"] else "",
                                f"{s['tasks']:.0f}" if s and s["tasks"] else "",
                                t["n"], f"{t['mean']:.4f}", f"{t['sd']:.4f}", f"{t['min']:.4f}",
                                f"{t['max']:.4f}", t["vram"], t["test"], t["nodes"], t["commit"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node", default="hopper")
    parser.add_argument("--panels", default="higgs,mnist8m,higgs-depth30")
    args = parser.parse_args()
    grid = [(p, tools) for p in args.panels.split(",") if (tools := load(args.node, p))]
    if not grid:
        raise SystemExit("no results with a timeline")
    heights = [lanes_of(tools)[1] for _, tools in grid]
    fig, axes = plt.subplots(len(grid), 1, figsize=(12, 2.0 + INCH * sum(heights)), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, (panel, tools) in zip(axes[:, 0], grid):
        dataset, depth = panel_of(panel)
        style(ax, f"{TITLES.get(dataset, dataset)}, depth ≤ {depth}")
        draw(ax, tools)
    gpu = grid[0][1][0]["gpu"]
    fig.suptitle(f"One CART tree per fresh process on one {gpu} ({args.node}): each tool's phases",
                 x=0.01, y=1 - 0.12 / fig.get_figheight(), ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c[0]) for c in CATEGORIES.values()],
               labels=[c[1] for c in CATEGORIES.values()], loc="upper left", bbox_to_anchor=(0.01, 1 - 0.36 / fig.get_figheight()),
               ncol=1, frameon=False, fontsize=8, handlelength=1.6, labelspacing=0.3, labelcolor=INK)
    note = ("Every run a fresh process that trains one tree, no warm-up; the totals are the comparison. "
            "The bars are each tool's own phases, with its own boundaries, from their start to their end (mean of "
            "the runs); they explain a total, they are not equal components: ./tree allocates its device memory "
            "in set-up, the libraries inside their later phases; cuML computes its bins inside its build (from a "
            "row sample, < 1 ms of GPU time, nsys). Bars that overlap ran at the same time (./tree presorts each "
            "attribute on the GPU while the next ones upload, measured with CUDA events). Not timed: reading the "
            "data into host RAM (column-major float32 for every tool), importing the library, the CUDA context. "
            "Every tool uploads through ./tree's staged upload, pinned to the GPU's NUMA node "
            "(bench/adapters/uploader.cpp). Exact: every distinct value is a candidate split; bins: only quantile "
            "bin edges (XGBoost: squared error with λ = 0 on 0/1 or one-hot labels = Gini). The trees are "
            "comparable, not identical. ./tree --hybrid hands nodes under 512 rows to the CPUs "
            "(at depth ≤ 10 the value measured best, at depth ≤ 30 its default).")
    fig.text(0.01, 0.005, textwrap.fill(note, 190), fontsize=7.5, color=MUTED, va="bottom")
    top = 1 - 1.2 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0.75 / fig.get_figheight(), 1, top))
    fig.savefig(RESULTS / "chart.png", dpi=160)
    write_summary([(p, load(args.node, p, merged=False)) for p, _ in grid], RESULTS / "summary.csv")
    print(RESULTS / "chart.png")


if __name__ == "__main__":
    main()
