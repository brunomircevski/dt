#!/usr/bin/env python3
"""cluster-gpu-vs-others: one CART tree (depth <= 10) per tool on one GPU, as a
timeline: every section of the timed training is a thin bar from the moment it
started to the moment it ended, one lane per section, so sections that run at
the same time (./tree's presort during its upload, --hybrid's CPU subtrees
during its GPU levels) overlap. ./tree --gpu and --hybrid, cuML, XGBoost; every
run a fresh process with no warm-up, all uploading through ./tree's staged
upload. Reads cluster-gpu-vs-others/one-tree/vs-<node>-<dataset>/results.jsonl
(bench/run.py; each row's "timeline"), writes chart.png and summary.csv there.

  bench/.venv/bin/python bench/gpu_steps_chart.py [--node hopper] [--datasets higgs,mnist8m]

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

RESULTS = Path(__file__).resolve().parent / "results" / "cluster-gpu-vs-others"
TOOLS = [("tree_gpu_only", "./tree --gpu"), ("tree_cuda", "./tree --hybrid"),
         ("cuml", "cuML"), ("xgboost", "XGBoost")]
INK, MUTED, GRID, WHITE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# category -> (colour, legend); the reference palette's slots 1-5, in order
CATEGORIES = {
    "set-up": ("#2a78d6", "Set-up: buffers, device memory, tables"),
    "upload": ("#eb6834", "Upload to the GPU (the same code for every tool)"),
    "prepare": ("#1baf7a", "Presort (./tree) · input checks (cuML) · quantile binning (XGBoost)"),
    "build": ("#eda100", "Build the tree"),
    "cpu": ("#e87ba4", "CPU subtrees (./tree --hybrid, on the node's CPUs)"),
}
# a section's category and its label on the chart (unlisted: build)
SECTIONS = {
    "staging buffers": ("set-up", "staging buffers, copy threads"),
    "hand-off buffers": ("set-up", "hand-off buffers"),
    "device memory": ("set-up", "device memory"),
    "upload": ("upload", "upload"),
    "presort (GPU)": ("prepare", "presort (GPU kernels)"),
    "x log x table": ("set-up", "x·log x table"),
    "x log x table copy": ("set-up", "x·log x table copy"),
    "input checks": ("prepare", "input checks"),
    "quantile binning": ("prepare", "quantile binning"),
    "build": ("build", "build the tree"),
    "cpu subtrees": ("cpu", "CPU subtrees"),
    "build tree": ("build", "build the tree"),
}
# sections too short to tell anything on their own (< 1% of the run), drawn as
# part of the one they follow
MERGE = {"free raw copy": "quantile binning", "fit's end": "build tree"}
TITLES = {"higgs": "HIGGS (10.5M rows × 28 features, 2 classes)",
          "mnist8m": "MNIST8M (8.1M rows × 784 features, 10 classes)"}
LANE = 1.0       # one lane per section
BAR = 0.42       # the bar's thickness in a lane
GAP = 0.9        # space between tools
HEAD = 1.5       # the tool's lane (its total, then its facts below it) to its first section


def load(node, dataset):
    path = RESULTS / "one-tree" / f"vs-{node}-{dataset}" / "results.jsonl"
    rows = [json.loads(line) for line in open(path)] if path.exists() else []
    rows = [r for r in rows if r.get("status") == "ok" and r.get("kind", "run") == "run"
            and r.get("protocol", "cart_depth10") == "cart_depth10" and r.get("timeline")]
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
                name = MERGE.get(s["name"], s["name"])
                if name in timeline:
                    timeline[name]["start"] = min(timeline[name]["start"], s["start"])
                    timeline[name]["end"] = max(timeline[name]["end"], s["end"])
                else:
                    timeline[name] = dict(s, name=name)
            for s in timeline.values():
                if s["name"] not in spans:
                    order.append(s["name"])
                spans[s["name"]].append((s["start"], s["end"], s.get("busy", 0.0), s.get("tasks", 0)))
        sections = []
        for name in order:
            runs = spans[name]
            start, end = st.mean(x[0] for x in runs), st.mean(x[1] for x in runs)
            sections.append({"name": name, "start": start, "end": end,
                             "seconds": st.mean(x[1] - x[0] for x in runs), "runs": len(runs),
                             "busy": st.mean(x[2] for x in runs), "tasks": st.mean(x[3] for x in runs)})
        sections.sort(key=lambda s: (s["start"], s["end"]))
        times = [r["train_seconds"] for r in mine]
        vram = [r["gpu_peak_bytes"] for r in mine if r.get("gpu_peak_bytes")]
        last = mine[-1]
        tools.append({"impl": impl, "label": label + ("" if impl.startswith("tree") else f" {last.get('version', '')}"),
                      "n": len(mine), "mean": st.mean(times), "sd": st.stdev(times) if len(times) > 1 else 0.0,
                      "min": min(times), "max": max(times), "sections": sections,
                      "vram": st.mean(vram) if vram else None, "test": last.get("test_accuracy"),
                      "nodes": last.get("nodes"), "gpu": last.get("gpu"), "commit": last.get("commit")})
    return tools


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


def draw(ax, tools):
    lanes, height = lanes_of(tools)
    right = max(t["mean"] for t in tools)
    ref = next((t["mean"] for t in tools if t["impl"] == "tree_gpu_only"), None)
    min_width = right * 0.0025  # a section too short to see is drawn this wide
    ticks, names = [], []
    for y, t, s in lanes:
        if s is None:  # the tool's header lane: its whole training, and its total
            ax.barh(y, t["mean"], left=0, height=BAR * 0.35, color=MUTED, alpha=0.35, linewidth=0)
            note = f"total {t['mean']:.3g} s"
            if ref and not t["impl"].startswith("tree"):
                note += f" ({t['mean'] / ref:.1f}× ./tree --gpu)"
            ax.text(t["mean"] + right * 0.01, y, note, va="center", fontsize=8.5, color=INK,
                    fontweight="bold")
            ax.text(t["mean"] + right * 0.01, y + 0.62, facts(t), va="center", fontsize=7, color=MUTED)
            ticks.append(y)
            names.append(t["label"])
            continue
        category, label = SECTIONS.get(s["name"], ("build", s["name"]))
        colour = CATEGORIES[category][0]
        width = max(s["end"] - s["start"], min_width)
        ax.barh(y, width, left=s["start"], height=BAR, color=colour, linewidth=0)
        text = duration(s["seconds"])
        if s["tasks"]:
            text += f"; {duration(s['busy'])} of CPU work in {s['tasks']:.0f} tasks"
        ax.text(s["start"] + width + right * 0.008, y, text, va="center", fontsize=7, color=MUTED)
        ticks.append(y)
        names.append(label)
    ax.set_yticks(ticks, names)
    for tick, (_, _, s) in zip(ax.get_yticklabels(), lanes):
        tick.set_fontsize(7.5 if s else 9.5)
        tick.set_color(MUTED if s else INK)
        tick.set_fontweight("normal" if s else "bold")
    ax.set_ylim(height - GAP + 0.2, -0.8)
    ax.set_xlim(0, right * 1.3)


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
        w.writerow(["dataset", "implementation", "gpu", "section", "category", "start_mean", "end_mean",
                    "seconds_mean", "cpu_work_seconds", "cpu_tasks", "runs", "time_mean", "time_sd", "time_min",
                    "time_max", "vram_bytes", "test_accuracy", "nodes", "commit"])
        for dataset, tools in grid:
            for t in tools:
                for s in [None] + t["sections"]:
                    w.writerow([dataset, t["label"], t["gpu"],
                                "total" if s is None else s["name"],
                                "" if s is None else SECTIONS.get(s["name"], ("build",))[0],
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
    parser.add_argument("--datasets", default="higgs,mnist8m")
    args = parser.parse_args()
    grid = [(d, tools) for d in args.datasets.split(",") if (tools := load(args.node, d))]
    if not grid:
        raise SystemExit("no results with a timeline")
    heights = [lanes_of(tools)[1] for _, tools in grid]
    fig, axes = plt.subplots(len(grid), 1, figsize=(12, 2.2 + 0.2 * sum(heights)), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, (dataset, tools) in zip(axes[:, 0], grid):
        draw(ax, tools)
        style(ax, TITLES.get(dataset, dataset))
    gpu = grid[0][1][0]["gpu"]
    fig.suptitle(f"One CART tree, depth ≤ 10, on one {gpu} ({args.node}): when each section runs",
                 x=0.01, y=1 - 0.12 / fig.get_figheight(), ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c[0]) for c in CATEGORIES.values()],
               labels=[c[1] for c in CATEGORIES.values()], loc="upper left", bbox_to_anchor=(0.01, 1 - 0.38 / fig.get_figheight()),
               ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    note = ("Every run a fresh process, no warm-up; one lane per section, from its start to its end (mean of the "
            "runs), so sections that run at the same time overlap; the grey bar is the whole timed training. "
            "Not timed: reading the data into host RAM (column-major float32 for every tool), importing the "
            "library, the CUDA context. cuML and XGBoost upload through ./tree's own staged upload "
            "(bench/adapters/uploader.cpp). ./tree's presort span is measured on the GPU (CUDA events): it "
            "sorts each feature as soon as it is uploaded. cuML's 128 bins are computed inside its build from a "
            "row sample (< 1 ms of GPU time, nsys). ./tree is exact (every distinct value); cuML splits on 128 "
            "quantile bins, XGBoost on 256 (squared error with λ = 0 on 0/1 or one-hot labels = Gini).")
    fig.text(0.01, 0.005, textwrap.fill(note, 190), fontsize=7.5, color=MUTED, va="bottom")
    top = 1 - 0.95 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0.75 / fig.get_figheight(), 1, top))
    fig.savefig(RESULTS / "chart.png", dpi=160)
    write_summary(grid, RESULTS / "summary.csv")
    print(RESULTS / "chart.png")


if __name__ == "__main__":
    main()
