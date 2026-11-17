#!/usr/bin/env python3
"""The end-to-end benchmark's chart and table (bench/results/cluster-end-to-end):
from launch to finished tree, step by step, every tool on one dataset.

  bench/.venv/bin/python bench/e2e_chart.py [--dir bench/results/cluster-end-to-end]

Reads <dir>/e2e-*/results.jsonl (one row per process, bench/run.py --no-eval
--prefetch) and writes <dir>/chart.png and <dir>/summary.csv. The steps of a
run, in seconds (BENCHMARK.md there):

  read             the tool's own clock around reading the input (read_seconds)
  gpu startup      the part of the GPU start-up the process waited for: ./tree's
                   "gpu startup wait" (its start-up overlaps the read), the
                   libraries' CUDA context (their import is in "other")
  prepare          ./tree: gpu setup + presort; XGBoost: QuantileDMatrix;
                   YaDT: its indexing (total time - load time)
  build            growing the tree (./tree: + C4.5's collapse)
  prepare + build  one call that does both (scikit-learn, rpart, J48, cuML)
  other            wall time minus the steps above: process and runtime start,
                   imports, exit, anything not measured; never negative
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import benchlib as B  # noqa: E402
from chart import faded, out_of_date_banner  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
STEPS = ["other", "read", "gpu_startup", "prepare", "build", "prepare_build"]
STEP_NAMES = {"other": "runtime start, exit, other", "read": "read data",
              "gpu_startup": "GPU start-up (not overlapped)", "prepare": "prepare",
              "build": "build", "prepare_build": "prepare + build (one call)"}
STEP_HELP = {"other": "wall time minus the steps below: process and runtime start, imports, exit",
             "read": "reading the input file into memory (the tool's own clock)",
             "gpu_startup": "CUDA context and device set-up the process waited for",
             "prepare": "./tree GPU setup + presort, XGBoost QuantileDMatrix, YaDT indexing",
             "build": "growing the tree (./tree: + C4.5's collapse)",
             "prepare_build": "one fit call that does both: scikit-learn, rpart, J48, cuML"}
TIMED = {"prepare", "build", "prepare_build"}  # the training time, what the standard benchmarks time
TIMED_COLOR = INK
STEP_COLORS = {"other": "#b4b2a9", "read": "#2a78d6", "gpu_startup": "#6250d6", "prepare": "#eda100",
               "build": "#1baf7a", "prepare_build": "#008300"}
COMBINED = {"sklearn", "rpart", "j48", "cuml"}  # one call prepares and builds
ALGOS = {"cart_depth10": "CART (Gini), depth ≤ 10", "c45_depth10_unpruned": "C4.5 (gain ratio), depth ≤ 10, unpruned"}


def tool_name(row):
    impl, threads = row["impl"], row["threads"]
    gpu = row.get("gpu", "") or ""
    gpu = "H100" if "H100" in gpu else "A100" if "A100" in gpu else gpu
    return {"tree": "./tree --serial" if threads == 1 else f"./tree --multicore, {threads} threads",
            "tree_simple": "./tree --simple",
            "tree_cuda": f"./tree --hybrid ({gpu})", "tree_gpu_only": f"./tree --gpu ({gpu})",
            "sklearn": "scikit-learn", "rpart": "rpart", "j48": "Weka J48",
            "yadt": "YaDT" + ("" if threads == 1 else f", {threads} threads"),
            "cuml": f"cuML, 1 tree ({gpu})", "xgboost": f"XGBoost, 1 tree ({gpu})"}[impl]


def steps(row):
    """One run's steps in seconds; 'other' is the remainder of the wall time."""
    impl = row["impl"]
    s = dict.fromkeys(STEPS, 0.0)
    s["read"] = row.get("read_seconds") or 0.0
    if impl in ("tree", "tree_simple", "tree_cuda", "tree_gpu_only"):
        s["gpu_startup"] = row.get("gpu_startup_wait_seconds") or 0.0
        s["prepare"] = (row.get("gpu_setup_seconds") or 0.0) + (row.get("presort_seconds") or 0.0)
        s["build"] = (row.get("build_seconds") or 0.0) + (row.get("prune_seconds") or 0.0)
    elif impl == "yadt":
        s["prepare"], s["build"] = row["index_seconds"], row["build_seconds"]
    elif impl == "xgboost":
        s["gpu_startup"] = row["gpu_startup_seconds"]
        s["prepare"], s["build"] = row["dmatrix_seconds"], row["build_seconds"]
    else:
        s["gpu_startup"] = row.get("gpu_startup_seconds") or 0.0  # cuML; 0 for the CPU libraries
        s["prepare_build"] = row["train_seconds"]
    s["other"] = row["wall_seconds"] - sum(v for k, v in s.items() if k != "other")
    return s


def overlap(row):
    """./tree's GPU start-up that ran during the read (on its background thread): full start-up minus the
    part still running after the read, at most the read itself. 0 for the other tools."""
    if row.get("gpu_startup_wait_seconds") is None or row.get("gpu_startup_seconds") is None:
        return 0.0
    return max(0.0, min(row.get("read_seconds") or 0.0, row["gpu_startup_seconds"] - row["gpu_startup_wait_seconds"]))


def collect(folder):
    cases = defaultdict(list)
    for f in sorted(folder.glob("e2e-*/results.jsonl")):
        for line in open(f):
            row = json.loads(line)
            if row.get("kind") == "run" and row.get("status") == "ok":
                cases[(row["protocol"], tool_name(row))].append(row)
    out = []
    for (protocol, name), rows in cases.items():
        per = [steps(r) for r in rows]
        mean = {k: statistics.mean(p[k] for p in per) for k in STEPS}
        out.append({"protocol": protocol, "tool": name, "runs": len(rows), "dataset": rows[0]["dataset"],
                    "steps": mean, "overlap": statistics.mean(overlap(r) for r in rows), "other_min": min(p["other"] for p in per),
                    "other_max": max(p["other"] for p in per),
                    "wall": statistics.mean(r["wall_seconds"] for r in rows),
                    "train": statistics.mean(r["train_seconds"] for r in rows),
                    "gpu_startup_full": statistics.mean(r["gpu_startup_seconds"] for r in rows)
                    if "gpu_startup_seconds" in rows[0] else None,
                    "peak_rss": statistics.mean(r["peak_rss_bytes"] for r in rows),
                    "vram": max((r.get("gpu_peak_bytes") or 0) for r in rows) or None,
                    "commits": {r.get("commit") for r in rows}, "hosts": {r.get("host") for r in rows},
                    "stale": B.rows_out_of_date(rows)})
    return out


def seconds(v):
    return f"{v:,.0f} s" if v >= 100 else f"{v:.3g} s" if v >= 1 else f"{v * 1000:.3g} ms"


def draw(cases, out, notes):
    protocols = [p for p in ALGOS if any(c["protocol"] == p for c in cases)]
    sizes = [sum(c["protocol"] == p for c in cases) for p in protocols]
    walls = [c["wall"] for c in cases]
    lo = 10 ** math.floor(math.log10(min(walls)))
    hi = 10 ** math.ceil(math.log10(max(walls)))
    fig, axes = plt.subplots(len(protocols), 1, figsize=(15, 2.4 + 0.5 * sum(sizes) + 1.2 * len(protocols)),
                             gridspec_kw={"height_ratios": sizes}, squeeze=False)
    labels = []  # (axes, x0, x1, y, text, step, stale): written after the layout, where they fit
    for tx, protocol in zip(axes[:, 0], protocols):
        rows = sorted((c for c in cases if c["protocol"] == protocol), key=lambda c: c["wall"])
        for y, c in enumerate(rows):
            col = (lambda k: faded(STEP_COLORS[k])) if c["stale"] else (lambda k: STEP_COLORS[k])
            # the bar's length is the wall time on the log axis; inside it the steps split the drawn
            # length linearly by their share
            at = lambda share: lo * (c["wall"] / lo) ** share  # noqa: E731
            done = 0.0
            for k in STEPS:
                share = c["steps"][k] / c["wall"]
                if share > 0:
                    # a 2px-looking surface gap between segments
                    tx.barh(y, at(done + share) - at(done), left=at(done), height=0.62, color=col(k),
                            edgecolor=SURFACE, linewidth=1.2, zorder=2)
                    if share >= 0.01:
                        labels.append((tx, at(done), at(done + share), y, f"{100 * share:.0f}%", k, c["stale"]))
                done += share
            if c["overlap"] > 0:  # GPU start-up during the read: hatched over the read segment
                r0 = c["steps"]["other"]
                h0, h1 = at(r0 / c["wall"]), at((r0 + c["overlap"]) / c["wall"])
                tx.barh(y, h1 - h0, left=h0, height=0.62, facecolor="none",
                        edgecolor=col("gpu_startup"), hatch="////", linewidth=0, zorder=2.5)
            # what the standard benchmarks time (train_seconds = prepare + build), thin, under the bar
            tx.barh(y + 0.41, c["train"] - lo, left=lo, height=0.11, color=faded(TIMED_COLOR) if c["stale"]
                    else TIMED_COLOR, zorder=2)
            tx.text(max(c["train"], lo) * 1.08, y + 0.41, f"timed {seconds(c['train'])} ({100 * c['train'] / c['wall']:.0f}% of end to end)", va="center", fontsize=7.5,
                    color=MUTED)
            tx.text(c["wall"] * 1.08, y, f"{seconds(c['wall'])}  ({c['runs']} runs)" + ("\n⚠ out of date" if c["stale"] else ""),
                    va="center", fontsize=9, color=MUTED if c["stale"] else INK)
        tx.set_xscale("log")
        tx.set_xlim(lo, hi * 2.5)  # room for the labels
        tx.set_xticks([10.0 ** e for e in range(round(math.log10(lo)), round(math.log10(hi)) + 1)])
        tx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: seconds(v)))
        tx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        tx.set_yticks(range(len(rows)), [c["tool"] for c in rows], fontsize=9)
        tx.set_title(ALGOS[protocol] + ": wall time (log scale), split by share of each step", loc="left",
                     fontsize=10, color=INK, fontweight="bold")
        tx.set_ylim(-0.6, len(rows) - 0.4 + 0.15)
        tx.invert_yaxis()
        tx.tick_params(colors=MUTED, labelsize=8.5, length=0)
        tx.grid(axis="x", color=GRID, linewidth=0.6)
        tx.set_axisbelow(True)
        for side in ("top", "right", "left"):
            tx.spines[side].set_visible(False)
        tx.spines["bottom"].set_color(GRID)
    dataset = cases[0]["dataset"].upper() if cases else "?"
    fig.suptitle(f"End to end on {dataset}: where a run's wall time goes, launch to finished tree "
                 "(share of each step, and the part the standard benchmarks time)", x=0.01, y=0.995, ha="left",
                 fontsize=11, color=INK, fontweight="bold")
    handles = [Patch(color=STEP_COLORS[k], label=f"{STEP_NAMES[k]} ({'TIMED' if k in TIMED else 'not timed'}): "
                     f"{STEP_HELP[k]}") for k in STEPS]
    if any(c["overlap"] > 0 for c in cases):
        handles.append(Patch(facecolor=STEP_COLORS["read"], edgecolor=STEP_COLORS["gpu_startup"], hatch="////",
                             linewidth=0, label="GPU start-up during the read (./tree, background thread): "
                             "overlapped, so counted once, in the read"))
    handles.append(Patch(color=TIMED_COLOR, label="thin bar: what the standard benchmarks report, the TIMED steps (training time)"))
    fig.legend(handles=handles,
               loc="upper left", bbox_to_anchor=(0.01, 0.975), ncol=2, frameon=False, fontsize=8.5,
               columnspacing=3)
    fig.text(0.01, 0.005, "\n".join(notes), fontsize=7.6, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.03 + 0.012 * len(notes), 1, 0.91), h_pad=2.5, w_pad=3)
    renderer = fig.canvas.get_renderer()
    for a, x0, x1, y, text, k, stale in labels:
        (px0, _), (px1, _) = a.transData.transform([(x0, y), (x1, y)])
        t = a.text((x0 * x1) ** 0.5 if a.get_xscale() == "log" else (x0 + x1) / 2, y, text, ha="center",
                   va="center", fontsize=7, color="white" if k not in ("other", "prepare") and not stale else INK, zorder=3)
        if t.get_window_extent(renderer).width + 4 > px1 - px0:  # does not fit inside its segment
            t.remove()
    fig.savefig(out, dpi=140, facecolor="white",
                **out_of_date_banner(fig, sum(c["stale"] for c in cases), len(cases)))
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parent / "results" / "cluster-end-to-end"))
    args = parser.parse_args()
    folder = Path(args.dir)
    cases = collect(folder)
    if not cases:
        raise SystemExit(f"no runs in {folder}/e2e-*/results.jsonl")
    with open(folder / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "tool", "runs", "wall_mean"] + [f"{k}_mean" for k in STEPS]
                   + ["other_min", "other_max", "train_mean", "gpu_startup_full_mean", "peak_rss_mean",
                      "vram_peak"])
        for c in sorted(cases, key=lambda c: (c["protocol"], c["wall"])):
            w.writerow([c["protocol"], c["tool"], c["runs"], c["wall"]] + [c["steps"][k] for k in STEPS]
                       + [c["other_min"], c["other_max"], c["train"], c["gpu_startup_full"], c["peak_rss"],
                          c["vram"]])
    print(folder / "summary.csv")
    negative = [c["tool"] for c in cases if c["other_min"] < 0]
    if negative:
        print("WARNING: 'other' negative in some run of:", ", ".join(sorted(set(negative))))
    runs = sorted({c["runs"] for c in cases})
    commits = sorted({x for c in cases for x in c["commits"] if x})
    hosts = sorted({x for c in cases for x in c["hosts"] if x})
    notes = [
        f"Mean of {'/'.join(map(str, runs))} fresh-process runs per tool, no warm-up, interleaved in a shuffled "
        "order per node; no prediction or evaluation (the process ends with its tree; YaDT still saves its tree).",
        "Input: the float32 binary files, read into the page cache right before each process (pinned as the run) "
        "and dropped after it, so 'read' is not NFS speed; YaDT reads its own text format (parsing included).",
        "./tree's GPU start-up runs during the read: only the part still running after it is drawn. "
        "Prepare: ./tree GPU setup + presort, XGBoost QuantileDMatrix, YaDT indexing; one call does both for "
        "scikit-learn, rpart, J48, cuML.",
        f"Nodes {', '.join(hosts)} (CPU tools on epyc"
        + "".join(f", GPU tools on {n}, {g}" for n, g in (("hopper", "H100"), ("ampere", "A100 80 GB"))
                  if any(h.startswith(n) for h in hosts))
        + f"); commit {', '.join(commits)}.",
        "A bar's length is the mean wall time on a log axis; inside it the steps split that length linearly by "
        "their share (a segment's length is not its seconds on the axis).",
        "Timed by the standard benchmarks (bench/README.md): only the training, from the data in memory to the "
        "finished tree; the read, start-up (CPU and GPU) and process exit are not timed.",
    ] + ([B.OUT_OF_DATE_NOTE + "."] if any(c["stale"] for c in cases) else [])
    draw(cases, folder / "chart.png", notes)


if __name__ == "__main__":
    main()
