#!/usr/bin/env python3
"""The chart of the comparison with other implementations on the CPU, HIGGS and
MNIST8M (rows) by CART and C4.5 (columns), one tree at depth <= 10 without pruning:

  cpu  cluster-cpu-vs-others/chart.png: ./tree --simple, --serial and
       --multicore (48 threads) on epyc against scikit-learn and rpart (CART) and
       Weka J48 and YaDT (C4.5), one thread each, and YaDT on 48 threads

./tree's bars are the comparison's own run (cluster-cpu-vs-others/vs-epyc-<dataset>/,
bench/tools/run.py, every tool in one run, interleaved), else the test-set benchmarks' runs
(cluster-<dataset>/<backend>-epyc-<dataset>). Every bar: the mean of its runs, each a fresh
process with no warm-up; the whisker spans the fastest to the slowest run. The GPU
comparison has its own chart: bench/tools/gpu_steps_chart.py.

  bench/tools/.venv/bin/python bench/tools/vs_chart.py cpu [--results bench]
"""

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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


FAILED = "failed"  # a bar whose tool ran and failed on every run (not a missing run)


def failed(run, protocol, impl, threads):
    """FAILED if the folder's summary.csv has this case and none of its runs is ok, else None."""
    f = run / "summary.csv"
    if not f.exists():
        return None
    rows = [r for r in csv.DictReader(open(f)) if r["impl"] == impl and int(r["threads"]) == threads
            and r["protocol"] == protocol]
    return FAILED if rows and all(r["status"] != "ok" for r in rows) else None


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
    return text + ("   ·  " + ", ".join(tail) if tail else "")


def panel(ax, bars, title):
    """bars: [(name, colour, numbers or None)], top to bottom."""
    y = list(range(len(bars)))[::-1]
    present = [b for _, _, b in bars if isinstance(b, dict)]
    lo = min((b["min"] for b in present), default=0.1) / 3
    hi = max((b["max"] for b in present), default=10) * 300  # room for the labels
    for yy, (name, color, b) in zip(y, bars):
        if not isinstance(b, dict):
            ax.text(lo * 1.15, yy, "failed: cannot fit this dataset (BENCHMARK.md)" if b == FAILED
                    else "no run yet", va="center", fontsize=8, color=MUTED, style="italic")
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
    """The first bar there is: the interleaved run (./tree with the libraries), else
    the test-set benchmark's run."""
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
        [(label, CPU_COLORS[key], summary(v, protocol, impl, threads) or failed(v, protocol, impl, threads))
         for label, key, impl, threads in others]


def few_runs(grid, algos=ALGOS):
    """The bars with fewer runs than the most, named: with 3 runs, a run takes over an hour."""
    bars = [(d, a, name, b) for d, row in zip(DATASETS, grid) for a, bs in zip(algos, row)
            for name, _, b in bs if isinstance(b, dict)]
    most = max((b["n"] for *_, b in bars), default=0)
    few = list(dict.fromkeys(  # one entry per tool and case, not per node
        f"{name.split(' — ')[0].split(',')[0]} {ALGOS[a][0].split(' ')[0]} on {TITLES[d].split(':')[0]}"
        f" ({b['n']})" for d, a, name, b in bars if b["n"] < most))
    runs = "fresh-process runs, no warm-up"
    if not few:
        return f"Every bar: the mean of {most} {runs}."
    return (f"Every bar: the mean of {most} {runs}; fewer runs (3: one run takes over "
            f"an hour; 5: the 2026-11-15 rerun, not yet redone): {'; '.join(few)}.")


def provenance(grid):
    bars = [b for row in grid for bs in row for _, _, b in bs if isinstance(b, dict)]
    commits = sorted({b["commit"] for b in bars if b.get("commit")})
    hosts = sorted({b["host"] for b in bars if b.get("host")})
    dates = sorted({b["date"] for b in bars if b.get("date")})
    versions = sorted({b["version"] for b in bars if b.get("version")})
    when = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else (dates[0] if dates else "?")
    return (f"./tree {', '.join(commits) or '?'}" + (f"; libraries {', '.join(versions)}" if versions else "")
            + f"; nodes {', '.join(hosts)}; run {when}")


def stale(grid):
    """The number of out-of-date bars."""
    return sum(1 for row in grid for bs in row for *_, b in bs if isinstance(b, dict) and b.get("stale"))


def draw(res, out):
    algos = list(ALGOS)
    grid = [[cpu_bars(res, d, a) for a in algos] for d in DATASETS]
    per_panel = max(len(bs) for row in grid for bs in row)
    fig, axes = plt.subplots(len(DATASETS), len(algos), squeeze=False,
                             figsize=(17 if len(algos) == 2 else 12, 1.9 + 2 * 0.36 * per_panel * len(DATASETS)))
    for i, d in enumerate(DATASETS):
        for j, a in enumerate(algos):
            panel(axes[i][j], grid[i][j], f"{TITLES[d].split(':')[0]}, {ALGOS[a][0]}")
        axes[i][0].annotate(TITLES[d], (0, 1), xycoords="axes fraction", xytext=(0, 22),
                            textcoords="offset points", fontsize=10.5, fontweight="bold", color=INK)
    head = ("./tree against scikit-learn, rpart, Weka J48 and YaDT on the CPU: one tree, depth ≤ 10, no pruning, "
            "all rows and features; epyc (2× AMD EPYC 9255, 48 cores, DDR5-6400)")
    note = ("scikit-learn DecisionTreeClassifier (gini, max_depth 10); rpart (gini, maxdepth 10, cp 0, no "
            "competitor or surrogate splits); Weka J48 (-U -M 2, its adapter's depth limit 10); YaDT (-np -m 2 "
            "-depth 10). The libraries' runs interleaved in a shuffled order. ./tree --multicore and YaDT 48 threads "
            "use both sockets.")
    fig.suptitle(head, x=0.01, y=0.995, ha="left", va="top", fontsize=11.5, color=INK, fontweight="bold")
    width = NOTE_CHARS if len(algos) == 2 else NOTE_CHARS * 12 // 17  # characters per footnote line
    text = "\n".join(textwrap.fill(part, width) for part in (
        "Training time (log scale), excluding reading the data and start-up: bar = mean of the runs, whisker = "
        "fastest to slowest; label: mean (runs), peak memory, test accuracy, tree nodes"
        + ". " + few_runs(grid, algos),
        *note.split("\n"), provenance(grid), *([B.OUT_OF_DATE_NOTE + "."] if stale(grid) else [])))
    fig.text(0.01, 0.005, text, fontsize=8, color=MUTED, linespacing=1.5)
    # the footnote's and the title's heights (8 pt and 11.5 pt lines), as fractions of the figure
    bottom = 0.01 + (text.count("\n") + 1) * 8 * 1.5 / 72 / fig.get_figheight()
    top = 1 - (0.005 + ((head.count("\n") + 1) * 11.5 * 1.3 + 14) / 72 / fig.get_figheight())
    fig.tight_layout(rect=(0, bottom, 1, top), h_pad=3.5, w_pad=2)
    fig.savefig(out, dpi=140, facecolor="white",
                **out_of_date_banner(fig, stale(grid), sum(1 for row in grid for bs in row for *_, b in bs if isinstance(b, dict))))
    print(out)
    return grid, algos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["cpu"])
    parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    res = Path(args.results)
    draw(res, res / "cluster-cpu-vs-others" / "chart.png")


if __name__ == "__main__":
    main()
