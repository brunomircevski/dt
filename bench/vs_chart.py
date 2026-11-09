#!/usr/bin/env python3
"""The charts of the comparisons with other implementations, HIGGS and MNIST8M
(rows) by CART and C4.5 (columns), one tree at depth <= 10 without pruning:

  cpu  cluster-cpu-vs-others/chart.png: ./tree --simple, --serial and
       --parallel (48 threads) on epyc against scikit-learn (CART) and Weka J48
       (C4.5), one thread each
  gpu  cluster-gpu-vs-others/chart.png: ./tree CUDA hybrid and --gpu-only
       against cuML (1 tree, and a default forest of 10 / 10) and XGBoost (the
       same, CART only), on the H100 (hopper) and the A100 (ampere); also
       cluster-gpu-vs-others/summary.csv

./tree's bars are the test-set benchmarks' runs (cluster-<dataset>/<backend>-<node>-<dataset>),
the libraries' the comparison's own: cluster-cpu-vs-others/vs-epyc-<dataset>-<tool>-<algorithm>/
(bench/run.py) and cluster-gpu-vs-others/<dataset>/<node>/results.jsonl (scripts/gpubench.py).

  bench/.venv/bin/python bench/vs_chart.py cpu|gpu [--results bench/results]
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATASETS = ["higgs", "mnist8m"]
TITLES = {"higgs": "HIGGS: 10,500,000 training rows, 28 features, 2 classes; test 500,000 rows",
          "mnist8m": "MNIST8M: 8,100,000 training rows, 784 features, 10 classes; test 10,000 rows"}
ALGOS = {"cart": ("CART (Gini)", "cart_depth10"), "c45": ("C4.5 (gain ratio)", "c45_depth10_unpruned")}
CPU_COLORS = {"simple": "#d6455a", "serial": "#7f4fc9", "cpu": "#2a78d6", "sklearn": "#1baf7a",
              "j48": "#eda100"}
GPU_COLORS = {"hopper": "#2a78d6", "ampere": "#eb6834"}
GPU_NAMES = {"hopper": "H100 (hopper)", "ampere": "A100 80 GB (ampere)"}


def summary(run, protocol):
    """A run folder's summary.csv row for the protocol, as a bar's numbers, or None."""
    f = run / "summary.csv"
    if not f.exists():
        return None
    for r in csv.DictReader(open(f)):
        if r["protocol"] == protocol and r["status"] == "ok":
            num = lambda k: float(r[k]) if r.get(k) not in (None, "") else None  # noqa: E731
            machine = json.load(open(run / "machine.json"))
            return {"mean": num("time_mean"), "sd": num("time_stdev") or 0.0, "n": int(r["time_reps"]),
                    "min": num("time_min"), "max": num("time_max"), "ram": num("peak_rss"),
                    "vram": num("gpu_peak"), "acc": num("test_accuracy"), "nodes": num("nodes"),
                    "host": machine.get("host"), "commit": machine.get("versions", {}).get("tree_git"),
                    "date": machine.get("started", "")[:10]}
    return None


def library(root, dataset, node, impl, case, trees):
    """cuML or XGBoost rows of cluster-gpu-vs-others: one bar (a forest's time / trees)."""
    f = root / dataset / node / "results.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(line) for line in open(f)]
    rows = [r for r in rows if r["impl"] == impl and r["case"] == case and r.get("trees") == trees
            and r.get("depth_limit", 10) == 10]
    if not rows:
        return None
    t = [r["train_seconds"] / trees for r in rows]
    last = rows[-1]
    return {"mean": statistics.mean(t), "sd": statistics.stdev(t) if len(t) > 1 else 0.0, "n": len(t),
            "min": min(t), "max": max(t), "vram": last.get("gpu_peak_mib", 0) * 2**20,
            "acc": last.get("test_accuracy"), "nodes": last.get("nodes"), "host": node,
            "version": last.get("version"), "date": last.get("time", "")[:10]}


def seconds(v):
    if v >= 100:
        return f"{v:,.0f} s"
    if v >= 1:
        return f"{v:.3g} s"
    return f"{v * 1000:.3g} ms"


def label(b):
    text = f"{seconds(b['mean'])}"
    if b["n"] > 1:
        text += f" ± {seconds(b['sd'])}"
    text += f" ({b['n']} run{'s' if b['n'] > 1 else ''})"
    if b.get("ram"):
        text += f"   RAM {b['ram'] / 1e9:.3g} GB"
    if b.get("vram"):
        text += f"   VRAM {b['vram'] / 1e9:.3g} GB"
    tail = []
    if b.get("acc") is not None:
        tail.append(f"test {100 * b['acc']:.2f}%")
    if b.get("nodes"):
        tail.append(f"{b['nodes']:,.0f} nodes")
    return text + ("   ·  " + ", ".join(tail) if tail else "")


def panel(ax, bars, title):
    """bars: [(name, colour, numbers or None)], top to bottom."""
    y = list(range(len(bars)))[::-1]
    present = [b for _, _, b in bars if b]
    lo = min((b["min"] for b in present), default=0.1) / 3
    hi = max((b["max"] for b in present), default=10) * 300  # room for the labels
    for yy, (name, color, b) in zip(y, bars):
        if not b:
            ax.text(lo * 1.15, yy, "no run yet", va="center", fontsize=8, color=MUTED, style="italic")
            continue
        ax.barh(yy, b["mean"], left=lo, height=0.68, color=color, zorder=2)
        ax.plot([b["min"], b["max"]], [yy, yy], color=INK, linewidth=1, zorder=3)
        ax.text(b["max"] * 1.12, yy, label(b), va="center", fontsize=7.8, color=INK)
    ax.set_yticks(y, [name for name, _, _ in bars], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(bars) - 0.4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: seconds(v) if v >= 1e-3 else ""))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")


def cpu_bars(res, dataset, algo):
    protocol = ALGOS[algo][1]
    d = res / f"cluster-{dataset}"
    other = (("scikit-learn, one thread", "sklearn", f"vs-epyc-{dataset}-sklearn-cart") if algo == "cart"
             else ("Weka J48, one thread", "j48", f"vs-epyc-{dataset}-j48-c45"))
    return [("./tree --simple, one thread", CPU_COLORS["simple"], summary(d / f"simple-epyc-{dataset}", protocol)),
            ("./tree --serial, one thread", CPU_COLORS["serial"], summary(d / f"serial-epyc-{dataset}", protocol)),
            ("./tree --parallel, 48 threads", CPU_COLORS["cpu"], summary(d / f"cpu-epyc-{dataset}", protocol)),
            (other[0], CPU_COLORS[other[1]], summary(res / "cluster-cpu-vs-others" / other[2], protocol))]


def gpu_bars(res, dataset, algo):
    protocol = ALGOS[algo][1]
    d, g = res / f"cluster-{dataset}", res / "cluster-gpu-vs-others"
    impls = [("./tree hybrid", lambda n: summary(d / f"cuda-{n}-{dataset}", protocol)),
             ("./tree GPU only", lambda n: summary(d / f"gpuonly-{n}-{dataset}", protocol)),
             ("cuML, 1 tree", lambda n: library(g, dataset, n, "cuml", algo, 1)),
             ("cuML, forest of 10 / 10", lambda n: library(g, dataset, n, "cuml", algo, 10))]
    if algo == "cart":
        impls += [("XGBoost, 1 tree", lambda n: library(g, dataset, n, "xgboost", algo, 1)),
                  ("XGBoost, forest of 10 / 10", lambda n: library(g, dataset, n, "xgboost", algo, 10))]
    return [(f"{name} — {GPU_NAMES[n].split(' (')[0]}", GPU_COLORS[n], fn(n)) for name, fn in impls
            for n in GPU_COLORS]


def provenance(grid):
    bars = [b for row in grid for bs in row for _, _, b in bs if b]
    commits = sorted({b["commit"] for b in bars if b.get("commit")})
    hosts = sorted({b["host"] for b in bars if b.get("host")})
    dates = sorted({b["date"] for b in bars if b.get("date")})
    versions = sorted({b["version"] for b in bars if b.get("version")})
    when = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else (dates[0] if dates else "?")
    return (f"./tree {', '.join(commits) or '?'}" + (f"; libraries {', '.join(versions)}" if versions else "")
            + f"; nodes {', '.join(hosts)}; run {when}")


def draw(kind, res, out):
    builder = cpu_bars if kind == "cpu" else gpu_bars
    grid = [[builder(res, d, a) for a in ALGOS] for d in DATASETS]
    per_panel = max(len(bs) for row in grid for bs in row)
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(17, 1.9 + 2 * 0.36 * per_panel * len(DATASETS)))
    for i, d in enumerate(DATASETS):
        for j, a in enumerate(ALGOS):
            panel(axes[i][j], grid[i][j], f"{TITLES[d].split(':')[0]}, {ALGOS[a][0]}")
        axes[i][0].annotate(TITLES[d], (0, 1), xycoords="axes fraction", xytext=(0, 22),
                            textcoords="offset points", fontsize=10.5, fontweight="bold", color=INK)
    if kind == "cpu":
        head = ("./tree against scikit-learn and Weka J48 on the CPU: one tree, depth ≤ 10, no pruning, "
                "all rows and features; epyc (2× AMD EPYC 9255, 48 cores, DDR5-6400)")
        note = ("scikit-learn DecisionTreeClassifier (gini, max_depth 10); Weka J48 (-U -M 2, its adapter's "
                "depth limit 10); every tool one tree on the same float32 data. ./tree --parallel uses both sockets.")
    else:
        head = ("./tree against cuML and XGBoost on the GPU: one tree, depth ≤ 10, no pruning, all rows and features; "
                "one GPU and every CPU of the node")
        note = ("cuML RandomForestClassifier, 1 tree on every row and feature (128 bins; C4.5: entropy); XGBoost hist, "
                "squared error, λ 0 (256 bins; = Gini; MNIST8M: one multi-output tree on the one-hot labels).\n"
                "Forest of 10 / 10: the library's random-forest defaults (cuML bootstrap and √features; XGBoost RF 0.8 "
                "of rows and features), its time divided by 10. ./tree hybrid: --cuda, GPU only: --cuda --gpu-only.")
        fig.legend(handles=[Patch(color=c, label=GPU_NAMES[n]) for n, c in GPU_COLORS.items()],
                   loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(head, x=0.01, y=0.995, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, "Training time (log scale), excluding reading the data: bar = mean of the runs, whisker = "
             "fastest to slowest; label: mean ± standard deviation (runs), peak memory, test accuracy, tree nodes.\n"
             + note + "\n" + provenance(grid), fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.075 if kind == "gpu" else 0.06, 1, 0.965), h_pad=3.5, w_pad=2)
    fig.savefig(out, dpi=140, facecolor="white")
    print(out)
    return grid


def write_summary(grid, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "algorithm", "implementation", "gpu", "time_mean", "time_sd", "runs", "time_min",
                    "time_max", "vram_bytes", "test_accuracy", "nodes"])
        for d, row in zip(DATASETS, grid):
            for a, bars in zip(ALGOS, row):
                for name, color, b in bars:
                    impl, gpu = name.split(" — ")
                    if b:
                        w.writerow([d, a, impl, gpu, b["mean"], b["sd"], b["n"], b["min"], b["max"],
                                    b.get("vram"), b.get("acc"), b.get("nodes")])
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["cpu", "gpu"])
    parser.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    args = parser.parse_args()
    res = Path(args.results)
    target = res / ("cluster-cpu-vs-others" if args.kind == "cpu" else "cluster-gpu-vs-others")
    grid = draw(args.kind, res, target / "chart.png")
    if args.kind == "gpu":
        write_summary(grid, target / "summary.csv")


if __name__ == "__main__":
    main()
