#!/usr/bin/env python3
"""The charts of the comparisons with other implementations, HIGGS and MNIST8M
(rows) by CART and C4.5 (columns), one tree at depth <= 10 without pruning:

  cpu  cluster-cpu-vs-others/chart.png: ./tree --simple, --serial and
       --multicore (48 threads) on epyc against scikit-learn and rpart (CART) and
       Weka J48 and YaDT (C4.5), one thread each, and YaDT on 48 threads
  gpu  cluster-gpu-vs-others/chart.png: ./tree --hybrid and --gpu
       against cuML (1 tree, and a default forest of 10 / 10) and XGBoost (the
       same, CART only), on the H100 (hopper) and the A100 (ampere); also
       cluster-gpu-vs-others/summary.csv
  gpu-warm  cluster-gpu-vs-others/chart-warm.png (and summary-warm.csv): the warm
       variant, CART, one tree: every tool's timed fit is the second identical
       fit in one process (./tree --warm-up, gpubench.py WARM=1); each label
       gives the cold time (chart.png's) beside it

./tree's bars are the test-set benchmarks' runs (cluster-<dataset>/<backend>-<node>-<dataset>),
the libraries' the comparison's own: cluster-cpu-vs-others/vs-epyc-<dataset>/ (bench/run.py,
every library in one run, interleaved) and cluster-gpu-vs-others/<dataset>/<node>/results.jsonl
(scripts/gpubench.py, one process per run). Every bar: the mean of its runs, each a fresh
process with no warm-up; the whisker spans the fastest to the slowest run.

  bench/.venv/bin/python bench/vs_chart.py cpu|gpu|gpu-warm [--results bench/results]
"""

import argparse
import csv
import json
import statistics
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

import benchlib as B  # noqa: E402
from chart import faded, out_of_date_banner  # noqa: E402

NOTE_CHARS = 215  # footnote characters per line at 8 pt on the 17-inch-wide figure
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATASETS = ["higgs", "mnist8m"]
TITLES = {"higgs": "HIGGS: 10,500,000 training rows, 28 features, 2 classes; test 500,000 rows",
          "mnist8m": "MNIST8M: 8,100,000 training rows, 784 features, 10 classes; test 10,000 rows"}
ALGOS = {"cart": ("CART (Gini)", "cart_depth10"), "c45": ("C4.5 (gain ratio)", "c45_depth10_unpruned")}
CPU_COLORS = {"simple": "#d6455a", "serial": "#7f4fc9", "cpu": "#2a78d6", "sklearn": "#1baf7a",
              "rpart": "#0f6e4c", "j48": "#eda100", "yadt": "#e87ba4", "yadt_multi": "#a3326b"}
GPU_COLORS = {"hopper": "#2a78d6", "ampere": "#eb6834"}
GPU_NAMES = {"hopper": "H100 (hopper)", "ampere": "A100 80 GB (ampere)"}


def summary(run, protocol, impl=None, threads=None):
    """A run folder's summary.csv row for the protocol (and implementation and
    thread count, in a folder of several), as a bar's numbers, or None."""
    f = run / "summary.csv"
    if not f.exists():
        return None
    for r in csv.DictReader(open(f)):
        if impl and (r["impl"] != impl or (threads is not None and int(r["threads"]) != threads)):
            continue
        if r["protocol"] == protocol and r["status"] == "ok":
            num = lambda k: float(r[k]) if r.get(k) not in (None, "") else None  # noqa: E731
            machine = json.load(open(run / "machine.json"))
            return {"mean": num("time_mean"), "sd": num("time_stdev") or 0.0, "n": int(r["time_reps"]),
                    "min": num("time_min"), "max": num("time_max"), "ram": num("peak_rss"),
                    "vram": num("gpu_peak"), "acc": num("test_accuracy"), "nodes": num("nodes"),
                    "host": machine.get("host"), "commit": machine.get("versions", {}).get("tree_git"),
                    "stale": B.rows_out_of_date([], machine),
                    "date": machine.get("started", "")[:10]}
    return None


def library(root, dataset, node, impl, case, trees, warm=False):
    """cuML or XGBoost rows of cluster-gpu-vs-others: one bar (a forest's time / trees);
    warm: the warm variant's rows (gpubench.py WARM=1), else the cold ones."""
    f = root / dataset / node / "results.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(line) for line in open(f)]
    rows = [r for r in rows if r["impl"] == impl and r["case"] == case and r.get("trees") == trees
            and r.get("depth_limit", 10) == 10 and bool(r.get("warm")) == warm]
    if any("startup_seconds" in r for r in rows):
        # one fit per process (since 2026-11-16): never mixed with the older
        # warm-up-era rows (10 fits per process) a results.jsonl may still hold
        rows = [r for r in rows if "startup_seconds" in r]
    if not rows:
        return None
    t = [r["train_seconds"] / trees for r in rows]
    last = rows[-1]
    return {"mean": statistics.mean(t), "sd": statistics.stdev(t) if len(t) > 1 else 0.0, "n": len(t),
            "min": min(t), "max": max(t), "vram": max(r.get("gpu_peak_mib") or 0 for r in rows) * 2**20,
            "acc": last.get("test_accuracy"), "nodes": last.get("nodes"), "host": last.get("host", node),
            "commit": last.get("commit"), "gpu": last.get("gpu"), "stale": B.rows_out_of_date(rows),
            "startup": statistics.mean(r["startup_seconds"] for r in rows) if "startup_seconds" in last else None,
            "module_loading": last.get("cuda_module_loading"),
            "version": last.get("version"), "date": last.get("time", "")[:10]}


def seconds(v):
    if v >= 100:
        return f"{v:,.0f} s"
    if v >= 1:
        return f"{v:.3g} s"
    return f"{v * 1000:.3g} ms"


def label(b):
    text = f"{seconds(b['mean'])} ({b['n']} run{'s' if b['n'] > 1 else ''})"
    if b.get("ram"):
        text += f"   RAM {b['ram'] / 1e9:.3g} GB"
    if b.get("vram"):
        text += f"   VRAM {b['vram'] / 1e9:.3g} GB"
    tail = []
    if b.get("acc") is not None:
        tail.append(f"test {100 * b['acc']:.2f}%")
    if b.get("nodes"):
        tail.append(f"{b['nodes']:,.0f} nodes")
    if b.get("cold"):
        tail.append(f"cold {seconds(b['cold'])} ({b['cold'] / b['mean']:.1f}× the warm time)")
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
        ax.barh(yy, b["mean"], left=lo, height=0.68, color=faded(color) if b.get("stale") else color, zorder=2)
        ax.plot([b["min"], b["max"]], [yy, yy], color=INK, linewidth=1, zorder=3)
        ax.text(b["max"] * 1.12, yy, ("⚠ out of date · " if b.get("stale") else "") + label(b), va="center",
                fontsize=7.8, color=MUTED if b.get("stale") else INK)
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


def first(*bars):
    """The first bar there is: the interleaved run (./tree with the libraries, since
    2026-11-17), else the older separate run, a placeholder until the rerun."""
    return next((b for b in bars if b), None)


def cpu_bars(res, dataset, algo):
    protocol = ALGOS[algo][1]
    d, v = res / f"cluster-{dataset}", res / "cluster-cpu-vs-others" / f"vs-epyc-{dataset}"
    others = ([("scikit-learn, one thread", "sklearn", "sklearn", 1),
               ("rpart, one thread", "rpart", "rpart", 1)] if algo == "cart"
              else [("Weka J48, one thread", "j48", "j48", 1),
                    ("YaDT, one thread", "yadt", "yadt", 1),
                    ("YaDT, 48 threads", "yadt_multi", "yadt", 48)])
    return [("./tree --simple, one thread", CPU_COLORS["simple"],
             first(summary(v, protocol, "tree_simple", 1), summary(d / f"simple-epyc-{dataset}", protocol))),
            ("./tree --serial, one thread", CPU_COLORS["serial"],
             first(summary(v, protocol, "tree", 1), summary(d / f"serial-epyc-{dataset}", protocol))),
            ("./tree --multicore, 48 threads", CPU_COLORS["cpu"],
             first(summary(v, protocol, "tree", 48), summary(d / f"cpu-epyc-{dataset}", protocol)))] + \
        [(label, CPU_COLORS[key], summary(v, protocol, impl, threads)) for label, key, impl, threads in others]


def gpu_bars(res, dataset, algo):
    protocol = ALGOS[algo][1]
    d, g = res / f"cluster-{dataset}", res / "cluster-gpu-vs-others"
    one = lambda n: g / "one-tree" / f"vs-{n}-{dataset}"  # noqa: E731  (the interleaved run)
    impls = [("./tree hybrid", lambda n: first(summary(one(n), protocol, "tree_cuda"),
                                               summary(d / f"cuda-{n}-{dataset}", protocol))),
             ("./tree GPU only", lambda n: first(summary(one(n), protocol, "tree_gpu_only"),
                                                 summary(d / f"gpuonly-{n}-{dataset}", protocol)))]
    if algo == "cart":  # no GPU library grows C4.5 trees (an entropy tree is not C4.5)
        impls += [("cuML, 1 tree", lambda n: first(summary(one(n), protocol, "cuml"),
                                                   library(g, dataset, n, "cuml", algo, 1))),
                  ("cuML, forest of 10 / 10", lambda n: library(g, dataset, n, "cuml", algo, 10)),
                  ("XGBoost, 1 tree", lambda n: first(summary(one(n), protocol, "xgboost"),
                                                      library(g, dataset, n, "xgboost", algo, 1))),
                  ("XGBoost, forest of 10 / 10", lambda n: library(g, dataset, n, "xgboost", algo, 10))]
    return [(f"{name} — {GPU_NAMES[n].split(' (')[0]}", GPU_COLORS[n], fn(n)) for name, fn in impls
            for n in GPU_COLORS]


def gpu_warm_bars(res, dataset, algo):
    """The warm variant (CART, one tree): each tool's warm bar, with its cold mean."""
    protocol = ALGOS[algo][1]
    d, g = res / f"cluster-{dataset}", res / "cluster-gpu-vs-others"
    one = lambda n: g / "one-tree" / f"vs-{n}-{dataset}"  # noqa: E731  (interleaved, cold)
    hot = lambda n: g / "warm" / f"vs-{n}-{dataset}"  # noqa: E731  (interleaved, warm)
    impls = [("./tree hybrid", lambda n: first(summary(hot(n), protocol, "tree_cuda"),
                                               summary(g / "warm" / f"cuda-{n}-{dataset}", protocol)),
              lambda n: first(summary(one(n), protocol, "tree_cuda"), summary(d / f"cuda-{n}-{dataset}", protocol))),
             ("./tree GPU only", lambda n: first(summary(hot(n), protocol, "tree_gpu_only"),
                                                 summary(g / "warm" / f"gpuonly-{n}-{dataset}", protocol)),
              lambda n: first(summary(one(n), protocol, "tree_gpu_only"),
                              summary(d / f"gpuonly-{n}-{dataset}", protocol))),
             ("cuML, 1 tree", lambda n: first(summary(hot(n), protocol, "cuml"),
                                              library(g, dataset, n, "cuml", algo, 1, warm=True)),
              lambda n: first(summary(one(n), protocol, "cuml"), library(g, dataset, n, "cuml", algo, 1))),
             ("XGBoost, 1 tree", lambda n: first(summary(hot(n), protocol, "xgboost"),
                                                 library(g, dataset, n, "xgboost", algo, 1, warm=True)),
              lambda n: first(summary(one(n), protocol, "xgboost"), library(g, dataset, n, "xgboost", algo, 1)))]
    bars = []
    for name, warm, cold in impls:
        for n in GPU_COLORS:
            b, c = warm(n), cold(n)
            if b and c:
                b = {**b, "cold": c["mean"], "stale": b["stale"] or c["stale"]}
            bars.append((f"{name}, warm — {GPU_NAMES[n].split(' (')[0]}", GPU_COLORS[n], b))
    return bars


def few_runs(grid, algos=ALGOS):
    """The bars with fewer runs than the most, named: with 3 runs, a run takes over an hour."""
    bars = [(d, a, name, b) for d, row in zip(DATASETS, grid) for a, bs in zip(algos, row)
            for name, _, b in bs if b]
    most = max((b["n"] for *_, b in bars), default=0)
    few = list(dict.fromkeys(  # one entry per tool and case, not per node
        f"{name.split(' — ')[0].split(',')[0]} {ALGOS[a][0].split(' ')[0]} on {TITLES[d].split(':')[0]}"
        f" ({b['n']})" for d, a, name, b in bars if b["n"] < most))
    runs = "fresh-process runs, no warm-up" if not WARM else "runs, each a fresh process's second fit"
    if not few:
        return f"Every bar: the mean of {most} {runs}."
    return (f"Every bar: the mean of {most} {runs}; fewer runs (3: one run takes over "
            f"an hour; 5: the 2026-11-15 rerun, not yet redone): {'; '.join(few)}.")


def provenance(grid):
    bars = [b for row in grid for bs in row for _, _, b in bs if b]
    commits = sorted({b["commit"] for b in bars if b.get("commit")})
    hosts = sorted({b["host"] for b in bars if b.get("host")})
    dates = sorted({b["date"] for b in bars if b.get("date")})
    versions = sorted({b["version"] for b in bars if b.get("version")})
    when = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else (dates[0] if dates else "?")
    return (f"./tree {', '.join(commits) or '?'}" + (f"; libraries {', '.join(versions)}" if versions else "")
            + f"; nodes {', '.join(hosts)}; run {when}")


def stale(grid):
    """The number of out-of-date bars."""
    return sum(1 for row in grid for bs in row for *_, b in bs if b and b.get("stale"))


WARM = False  # set by draw(): the warm variant's chart


def draw(kind, res, out):
    global WARM
    WARM = kind == "gpu-warm"
    builder = {"cpu": cpu_bars, "gpu": gpu_bars, "gpu-warm": gpu_warm_bars}[kind]
    algos = ["cart"] if WARM else list(ALGOS)
    grid = [[builder(res, d, a) for a in algos] for d in DATASETS]
    per_panel = max(len(bs) for row in grid for bs in row)
    fig, axes = plt.subplots(len(DATASETS), len(algos), squeeze=False,
                             figsize=(17 if len(algos) == 2 else 12, 1.9 + 2 * 0.36 * per_panel * len(DATASETS)))
    for i, d in enumerate(DATASETS):
        for j, a in enumerate(algos):
            panel(axes[i][j], grid[i][j], f"{TITLES[d].split(':')[0]}, {ALGOS[a][0]}")
        axes[i][0].annotate(TITLES[d], (0, 1), xycoords="axes fraction", xytext=(0, 22),
                            textcoords="offset points", fontsize=10.5, fontweight="bold", color=INK)
    if kind == "cpu":
        head = ("./tree against scikit-learn, rpart, Weka J48 and YaDT on the CPU: one tree, depth ≤ 10, no pruning, "
                "all rows and features; epyc (2× AMD EPYC 9255, 48 cores, DDR5-6400)")
        note = ("scikit-learn DecisionTreeClassifier (gini, max_depth 10); rpart (gini, maxdepth 10, cp 0, no "
                "competitor or surrogate splits); Weka J48 (-U -M 2, its adapter's depth limit 10); YaDT (-np -m 2 "
                "-depth 10). The libraries' runs interleaved in a shuffled order. ./tree --multicore and YaDT 48 threads "
                "use both sockets.")
    elif WARM:
        head = ("Warm: ./tree against cuML and XGBoost on the GPU, each timed fit the second identical fit "
                "in one process;\none CART tree, depth ≤ 10, all rows and features")
        note = ("Warm: one untimed fit, the same as the timed one on the same data, first in the same process "
                "(./tree --warm-up, gpubench.py WARM=1): kernels loaded, memory pools, staging buffers and the "
                "driver's mappings of the input made. Cold (chart.png, the default protocol): the one fit of a "
                "fresh process. Settings as chart.png: cuML 128 bins, Gini; XGBoost hist 256 bins, squared error, "
                "λ 0 (MNIST8M: one multi-output tree on the one-hot labels); ./tree hybrid: --hybrid, GPU only: "
                "--gpu. ./tree on MNIST8M is slower warm than cold: each fit allocates its GPU buffers (about 53 GB) "
                "and uploads the data in its timed setup, and the second fit's setup takes longer than the "
                "first's (A100: 2.4-2.6 s against 1.3-1.5 s; its build time is the same).")
    else:
        head = ("./tree against cuML and XGBoost on the GPU: one tree, depth ≤ 10, no pruning, all rows and features; "
                "one GPU and every CPU of the node")
        note = ("cuML RandomForestClassifier, 1 tree on every row and feature (128 bins, Gini); XGBoost hist, "
                "squared error, λ 0 (256 bins; = Gini; MNIST8M: one multi-output tree on the one-hot labels).\n"
                "Forest of 10 / 10: the library's random-forest defaults (cuML bootstrap and √features; XGBoost RF 0.8 "
                "of rows and features), its time divided by 10. ./tree hybrid: --hybrid, GPU only: --gpu. Not timed: "
                "the imports and the bare CUDA context, for every tool alike; timed: from the host arrays to the "
                "fitted model, kernel loading (lazy), memory pools and staging buffers included.")
    fig.suptitle(head, x=0.01, y=0.995, ha="left", va="top", fontsize=11.5, color=INK, fontweight="bold")
    width = NOTE_CHARS if len(algos) == 2 else NOTE_CHARS * 12 // 17  # characters per footnote line
    text = "\n".join(textwrap.fill(part, width) for part in (
        "Training time (log scale), excluding reading the data and start-up: bar = mean of the runs, whisker = "
        "fastest to slowest; label: mean (runs), peak memory, test accuracy, tree nodes"
        + (", the cold mean" if WARM else "") + ". " + few_runs(grid, algos),
        *note.split("\n"), provenance(grid), *([B.OUT_OF_DATE_NOTE + "."] if stale(grid) else [])))
    fig.text(0.01, 0.005, text, fontsize=8, color=MUTED, linespacing=1.5)
    # the footnote's and the title's heights (8 pt and 11.5 pt lines), as fractions of the figure
    bottom = 0.01 + (text.count("\n") + 1) * 8 * 1.5 / 72 / fig.get_figheight()
    top = 1 - (0.005 + ((head.count("\n") + 1) * 11.5 * 1.3 + 14) / 72 / fig.get_figheight())
    if kind != "cpu":  # under the title, at the right
        fig.legend(handles=[Patch(color=c, label=GPU_NAMES[n]) for n, c in GPU_COLORS.items()],
                   loc="upper right", bbox_to_anchor=(0.995, top + 4 / 72 / fig.get_figheight()),
                   ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, bottom, 1, top), h_pad=3.5, w_pad=2)
    fig.savefig(out, dpi=140, facecolor="white",
                **out_of_date_banner(fig, stale(grid), sum(1 for row in grid for bs in row for *_, b in bs if b)))
    print(out)
    return grid, algos


def write_summary(grid, out, algos=ALGOS):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "algorithm", "implementation", "gpu", "time_mean", "time_sd", "runs", "time_min",
                    "time_max", "vram_bytes", "test_accuracy", "nodes", "out_of_date"])
        for d, row in zip(DATASETS, grid):
            for a, bars in zip(algos, row):
                for name, color, b in bars:
                    impl, gpu = name.split(" — ")
                    if b:
                        w.writerow([d, a, impl, gpu, b["mean"], b["sd"], b["n"], b["min"], b["max"],
                                    b.get("vram"), b.get("acc"), b.get("nodes"), int(bool(b.get("stale")))])
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["cpu", "gpu", "gpu-warm"])
    parser.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    args = parser.parse_args()
    res = Path(args.results)
    target = res / ("cluster-cpu-vs-others" if args.kind == "cpu" else "cluster-gpu-vs-others")
    suffix = "-warm" if args.kind == "gpu-warm" else ""
    grid, algos = draw(args.kind, res, target / f"chart{suffix}.png")
    if args.kind != "cpu":
        write_summary(grid, target / f"summary{suffix}.csv", algos)


if __name__ == "__main__":
    main()
