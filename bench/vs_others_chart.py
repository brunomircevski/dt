#!/usr/bin/env python3
"""One chart of a benchmark against other implementations run on several
machines (bench/results/cpu-susy-vs-others/<machine>/): for CART and C4.5,
training time and peak RSS of every implementation on every machine (log
scales; one thread solid, every CPU hatched), then the trees they built
(nodes, depth, test accuracy with its 95% interval). Also writes summary.csv
with every machine's cases.

  bench/.venv/bin/python bench/vs_others_chart.py bench/results/cpu-susy-vs-others \
      [--machines pc,epyc,dell] [--out chart.png]
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

import benchlib as B  # noqa: E402
from chart import COLORS, GRID, INK, MUTED, NAMES, collect, seconds_tick  # noqa: E402

PROTOCOLS = {"cart_alpha": "CART  ·  Gini, depth ≤ 30, cost-complexity pruning at α = 1e-5",
             "c45": "C4.5  ·  gain ratio, error-based pruning with CF 0.25"}
SHORT = {"cart_alpha": "CART", "c45": "C4.5"}
HATCH = "////"
# Figure x of the row labels, left of every panel: section, machine, its bracket, implementation.
X_SECTION, X_MACHINE, X_BRACKET, X_IMPL = 0.015, 0.015, 0.07, 0.077
plt.rcParams["hatch.linewidth"] = 1.6


def load(top, names):
    machines = {}
    for name in names:
        run = top / name
        if not (run / "results.jsonl").exists():
            print(f"{run}: no results yet, left out")
            continue
        times, memory, tree, status, _ = collect(run)
        machines[name] = {"times": times, "memory": memory, "tree": tree, "status": status,
                          "machine": json.load(open(run / "machine.json")),
                          "plan": json.load(open(run / "plan.json"))}
    return machines


def all_threads(data):
    return max(k[3] for k in list(data["times"]) + list(data["status"]))


def seconds_label(value):
    return f"{value:.0f} s" if value >= 100 else f"{value:.3g} s" if value >= 1 \
        else f"{value * 1e3:.3g} ms"


def gib_label(value):
    return f"{value:.2f} GiB" if value < 10 else f"{value:.1f} GiB"


def style(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", which="both", colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis=grid_axis, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks([])


def layout(machines, protocol):
    """Rows of one algorithm, top to bottom: section headers, then one bar per
    (machine, implementation): (y, kind, payload)."""
    impls = [i for i in B.PROTOCOLS[protocol]["impls"] if i in COLORS and i != "tree_cuda"]
    rows, y = [], 0.0
    for parallel in (False, True):
        rows.append((y, "section", "Every CPU of the machine (parallel)" if parallel
                     else "One thread"))
        y += 1.0
        for name, data in machines.items():
            threads = all_threads(data) if parallel else 1
            keys = [("susy", protocol, impl, threads) for impl in impls
                    if ("susy", protocol, impl, threads) in data["times"]
                    or ("susy", protocol, impl, threads) in data["status"]]
            if not keys:
                continue
            rows.append((y + (len(keys) - 1) / 2, "machine", (name, threads, len(keys))))
            for key in keys:
                rows.append((y, "bar", (name, key)))
                y += 1
            y += 0.45
        y += 0.35
    return rows, y - 0.8


def bars(fig, spec, machines, protocol, limits):
    rows, height = layout(machines, protocol)
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.08, width_ratios=[1.35, 1])
    ax_time, ax_rss = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    for ax in (ax_time, ax_rss):
        style(ax)
        ax.set_xscale("log")
        ax.set_ylim(height + 0.1, -0.7)
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax_time.set_xlim(*limits["time"])
    ax_rss.set_xlim(*limits["rss"])
    ax_time.xaxis.set_major_locator(LogLocator(base=10))
    ax_time.xaxis.set_major_formatter(FuncFormatter(seconds_tick))
    ax_rss.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax_rss.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} GiB"))
    ax_time.set_title(f"{SHORT[protocol]}: training time (log scale)", loc="left",
                      fontsize=11, color=INK, fontweight="bold", pad=8)
    ax_rss.set_title(f"{SHORT[protocol]}: peak memory, RAM (RSS, log scale)", loc="left",
                     fontsize=11, color=INK, fontweight="bold", pad=8)
    trans = blended_transform_factory(fig.transFigure, ax_time.transData)
    for y, kind, payload in rows:
        if kind == "section":
            ax_time.text(X_SECTION, y, payload, transform=trans, ha="left", va="center",
                         fontsize=9.5, color=INK, fontweight="bold", style="italic")
            for ax in (ax_time, ax_rss):
                ax.axhline(y + 0.5, color=GRID, linewidth=0.8, zorder=0)
            continue
        if kind == "machine":
            name, threads, count = payload
            ax_time.text(X_MACHINE, y - (0.2 if threads > 1 else 0), name, transform=trans,
                         ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")
            if threads > 1:
                ax_time.text(X_MACHINE, y + 0.2, f"{threads} threads", transform=trans,
                             ha="left", va="top", fontsize=7.8, color=MUTED)
            if count > 1:
                ax_time.plot([X_BRACKET, X_BRACKET], [y - (count - 1) / 2 - 0.3, y + (count - 1) / 2 + 0.3],
                             transform=trans, color=GRID, linewidth=1.2, clip_on=False)
            continue
        name, key = payload
        data = machines[name]
        _, _, impl, threads = key
        color = COLORS[impl]
        ax_time.text(X_IMPL, y, B.IMPLS[impl]["label"], transform=trans, ha="left", va="center",
                     fontsize=9, color=INK, fontweight="bold" if impl == "tree" else "normal")
        hatch = HATCH if threads > 1 else None
        if key not in data["times"]:
            for ax, low in ((ax_time, limits["time"][0]), (ax_rss, limits["rss"][0])):
                ax.text(low * 1.15, y, data["status"].get(key, "no data"), va="center",
                        fontsize=8.5, color=MUTED, style="italic")
            continue
        values = data["times"][key]
        mid = statistics.median(values)
        ax_time.barh(y, mid - limits["time"][0], left=limits["time"][0], height=0.66,
                     color=color, hatch=hatch, edgecolor="white", linewidth=0, zorder=2)
        if len(values) > 1:
            ax_time.plot([min(values), max(values)], [y, y], color=INK, linewidth=1, zorder=3)
            ax_time.plot([min(values)] * 2 + [None] + [max(values)] * 2,
                         [y - 0.17, y + 0.17, None, y - 0.17, y + 0.17], color=INK,
                         linewidth=1, zorder=3)
        tree_key = (key[0], key[1], "tree", threads)
        note = []
        if impl != "tree" and tree_key in data["times"]:
            note.append(f"{mid / statistics.median(data['times'][tree_key]):.0f}× ./tree")
        single = (key[0], key[1], impl, 1)
        if threads > 1 and single in data["times"]:
            note.append(f"speedup {statistics.median(data['times'][single]) / mid:.1f}× "
                        "over 1 thread")
        x = max(values) * 1.12
        label = ax_time.text(x, y, seconds_label(mid), va="center", fontsize=8.8, color=INK)
        if note:
            ax_time.annotate("   " + "  ·  ".join(note), xycoords=label, xy=(1, 0.5),
                             va="center", fontsize=8, color=MUTED)
        gib = statistics.median(data["memory"][key]) / 2**30
        ax_rss.barh(y, gib - limits["rss"][0], left=limits["rss"][0], height=0.66, color=color,
                    hatch=hatch, edgecolor="white", linewidth=0, zorder=2)
        ax_rss.text(gib * 1.07, y, gib_label(gib), va="center", fontsize=8.8, color=INK)


def trees(fig, spec, machines):
    """Nodes, depth and test accuracy per implementation. Every machine and
    thread count must build the same tree; any that differs gets its own row."""
    groups = {}
    for name, data in machines.items():
        for key, row in data["tree"].items():
            if key[1] not in PROTOCOLS:
                continue
            tree = (row["nodes"], row["depth"], row["test_accuracy"], row["train_accuracy"])
            groups.setdefault((key[1], key[2], tree), []).append(
                f"{name} {key[3]} thread" + ("s" if key[3] > 1 else ""))
    order = list(B.IMPLS)
    entries = sorted(groups.items(), key=lambda e: (list(PROTOCOLS).index(e[0][0]),
                                                    order.index(e[0][1])))
    per_impl = {}
    for (protocol, impl, _), _ in entries:
        per_impl[(protocol, impl)] = per_impl.get((protocol, impl), 0) + 1
    rows, y, previous = [], 0.0, None
    for (protocol, impl, tree), where in entries:
        if protocol != previous:
            if previous is not None:
                y += 0.4
            rows.append((y, "section", SHORT[protocol]))
            y += 1
            previous = protocol
        label = B.IMPLS[impl]["label"]
        if per_impl[(protocol, impl)] > 1:  # not the same tree everywhere: say where
            label += f" ({', '.join(where)})"
        rows.append((y, "row", (impl, tree, label)))
        y += 1
    height = y - 0.5
    same = all(n == 1 for n in per_impl.values())

    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.1, width_ratios=[1, 0.8, 1.3])
    ax_nodes, ax_depth, ax_acc = (fig.add_subplot(sub[i]) for i in range(3))
    for ax in (ax_nodes, ax_depth, ax_acc):
        style(ax)
        ax.set_ylim(height, -0.7)
    ax_nodes.set_xscale("log")
    ax_nodes.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_nodes.xaxis.set_minor_formatter(NullFormatter())
    where = "the same tree on every machine and thread count" if same else \
        "rows name the runs where they differ"
    ax_nodes.set_title("nodes (log scale)", loc="left", fontsize=11, color=INK,
                       fontweight="bold", pad=8)
    ax_depth.set_title("depth (edges, root to deepest leaf)", loc="left", fontsize=11,
                       color=INK, fontweight="bold", pad=8)
    ax_acc.set_title("test accuracy, 95% interval (500,000 rows)", loc="left", fontsize=11,
                     color=INK, fontweight="bold", pad=8)
    trans = blended_transform_factory(fig.transFigure, ax_nodes.transData)
    ax_nodes.text(X_SECTION, 1.0, "Trees built:\n" + where.replace(" on every", "\non every"),
                  transform=blended_transform_factory(fig.transFigure, ax_nodes.transAxes),
                  fontsize=8, color=MUTED, va="bottom", linespacing=1.4)
    nodes = [r[2][1][0] for r in rows if r[1] == "row"]
    accuracy = [r[2][1][2] for r in rows if r[1] == "row"]
    ax_nodes.set_xlim(min(nodes) / 2.2, max(nodes) * 9)
    ax_depth.set_xlim(0, max(r[2][1][1] for r in rows if r[1] == "row") * 1.3)
    span = max(accuracy) - min(accuracy)
    ax_acc.set_xlim(100 * (min(accuracy) - max(span, 0.004) * 0.6),
                    100 * (max(accuracy) + max(span, 0.004) * 0.9))
    ax_acc.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
    n_test = next(iter(machines.values()))["plan"]["datasets"]["susy"]["n_test"]
    for y, kind, payload in rows:
        if kind == "section":
            ax_nodes.text(X_SECTION, y, payload, transform=trans, ha="left", va="center",
                          fontsize=9.5, color=INK, fontweight="bold", style="italic")
            continue
        impl, (n, depth, test, train), label = payload
        color = COLORS[impl]
        ax_nodes.text(X_IMPL, y, label, transform=trans, ha="left", va="center", fontsize=9,
                      color=INK, fontweight="bold" if impl == "tree" else "normal")
        low = ax_nodes.get_xlim()[0]
        ax_nodes.barh(y, n - low, left=low, height=0.62, color=color, zorder=2)
        ax_nodes.text(n * 1.1, y, f"{n:,}", va="center", fontsize=8.8, color=INK)
        ax_depth.barh(y, depth, height=0.62, color=color, zorder=2)
        ax_depth.text(depth + 0.6, y, f"{depth}", va="center", fontsize=8.8, color=INK)
        ci = 100 * 1.96 * (test * (1 - test) / n_test) ** 0.5
        ax_acc.plot([100 * test - ci, 100 * test + ci], [y, y], color=color, linewidth=2,
                    solid_capstyle="round", zorder=2)
        ax_acc.scatter([100 * test], [y], s=60, color=color, edgecolor="white", linewidth=1.5,
                       zorder=3)
        ax_acc.text(100 * test + ci + 0.02, y, f"{100 * test:.2f}%", va="center", fontsize=8.8,
                    color=INK)
        ax_acc.annotate(f"   training {100 * train:.2f}%", xycoords=ax_acc.texts[-1],
                        xy=(1, 0.5), va="center", fontsize=8, color=MUTED)


def write_summary(top, machines, out):
    rows = []
    for name in machines:
        for row in csv.DictReader(open(top / name / "summary.csv")):
            rows.append({"machine": name, **row})
    if not rows:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("top", help="directory with one subdirectory per machine")
    parser.add_argument("--machines", default="pc,epyc,dell")
    parser.add_argument("--out", help="default: <top>/chart.png")
    args = parser.parse_args()
    top = Path(args.top)
    machines = load(top, args.machines.split(","))
    if not machines:
        raise SystemExit("no results")

    every_time = [t for d in machines.values() for v in d["times"].values() for t in v]
    every_rss = [m / 2**30 for d in machines.values() for v in d["memory"].values() for m in v]
    limits = {"time": (min(every_time) / 2.5, max(every_time) * 30),
              "rss": (min(every_rss) / 1.8, max(every_rss) * 3.2)}
    heights = [layout(machines, p)[1] for p in PROTOCOLS]
    tree_height = 2 + sum(len(B.PROTOCOLS[p]["impls"]) for p in PROTOCOLS)
    pitch = 0.235
    header, footer = 2.2, 0.85
    body = [h * pitch + 0.55 for h in heights] + [tree_height * pitch + 0.55]
    fig = plt.figure(figsize=(16, header + sum(body) + 0.9 + footer))
    inches = fig.get_figheight()
    grid = GridSpec(3, 1, figure=fig, height_ratios=body, hspace=0.9 / (sum(body) / 3),
                    left=0.15, right=0.985, top=1 - header / inches, bottom=footer / inches)
    for i, protocol in enumerate(PROTOCOLS):
        bars(fig, grid[i], machines, protocol, limits)
    trees(fig, grid[2], machines)

    first = next(iter(machines.values()))
    meta = first["plan"]["datasets"]["susy"]
    x0 = 0.015
    fig.text(x0, 1 - 0.15 / inches, f"{NAMES['susy']}: ./tree against other CART and C4.5 "
             f"implementations on {len(machines)} machines", fontsize=15, color=INK,
             fontweight="bold", va="top")
    fig.text(x0, 1 - 0.47 / inches, f"{meta['n_train']:,} training rows, {meta['n_test']:,} test "
             f"rows, {len(meta['features'])} features.  {PROTOCOLS['cart_alpha']}.  "
             f"{PROTOCOLS['c45']}.", fontsize=9.5, color=MUTED, va="top")
    for i, (name, data) in enumerate(machines.items()):
        y = 1 - (0.78 + 0.22 * i) / inches
        fig.text(x0, y, name, fontsize=9.5, color=INK, fontweight="bold", va="top")
        fig.text(x0 + 0.04, y, B.hardware_line(data["machine"], False).replace(
            "  ·  no GPU used", ""), fontsize=9.5, color=INK, va="top")
    shown = [impl for impl in B.IMPLS if impl in COLORS and impl != "tree_cuda"
             and any(k[2] == impl for d in machines.values() for k in d["times"])]
    handles = [Patch(facecolor=COLORS[impl], label=B.IMPLS[impl]["label"]) for impl in shown]
    handles += [Patch(facecolor="#9b9a96", label="one thread"),
                Patch(facecolor="#9b9a96", hatch=HATCH, edgecolor="white", linewidth=0,
                      label="every CPU (parallel)")]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               fontsize=9.5, bbox_to_anchor=(x0 - 0.004, 1 - (0.82 + 0.22 * len(machines)) / inches),
               handlelength=1.6, handleheight=1.1, columnspacing=1.6)
    reps = sorted({d["machine"]["settings"]["reps"] for d in machines.values()})
    runs = (f"median of {reps[0]} runs, black line: min–max" if reps == [reps[0]] and reps[0] > 1
            else "one run" if reps == [1] else "median of the runs, black line: min–max")
    fig.text(x0, 0.6 / inches, f"Bars: {runs}. Training time excludes reading the data; "
             "\"× ./tree\": how many times longer than ./tree on the same machine and thread "
             "count.\nPeak RSS: the highest resident memory of the training process up to the "
             "end of training (runtime, data, warm-up, training). scikit-learn, rpart and "
             "Weka J48 build a tree on one thread only, so they have no parallel case. One "
             "thread is pinned to one core;\nthe parallel case uses every logical CPU of the "
             "machine. Same data files (SHA-256) everywhere.", fontsize=8.2, color=MUTED,
             va="top", linespacing=1.55)
    out = Path(args.out) if args.out else top / "chart.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)
    if all((top / name / "summary.csv").exists() for name in machines):
        write_summary(top, machines, top / "summary.csv")


if __name__ == "__main__":
    main()
