#!/usr/bin/env python3
"""The test-set benchmark's chart (cluster-airline, -higgs, -susy, -ecbdl): one
dataset, ./tree on every node, trained on the training set and scored on the
test set. One panel per algorithm (CART, C4.5; depth <= 10), two bars per run
(node and backend): no pruning on top, pruned at fixed values right below it
(striped). A bar is the mean training time of its runs, a whisker from the
fastest to the slowest. Next to each bar:
the time, and peak RAM (and VRAM on a GPU) as numbers, the means of the runs.
The panel title holds the trees: test and training accuracy, nodes, depth (when
the runs grew different trees, each bar says its own).

  bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-higgs
  bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-ecbdl  # one chart per dataset

Writes <dir>/chart.png, or <dir>/chart-<dataset>.png when the runs hold more
than one dataset (ECBDL: the full set on the CPU nodes, the 15M sample on all).
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES, seconds_tick  # noqa: E402
from machines_chart import COLORS  # noqa: E402  (the validated backend colours)

# One panel per algorithm: the unpruned and the pruned bar of a run next to each other.
PANELS = {"CART": [("cart_depth10", "no pruning"), ("cart_depth10_alpha", "pruned, fixed α = 1e-5")],
          "C4.5": [("c45_depth10_unpruned", "no pruning"), ("c45_depth10", "pruned, CF 0.25")]}
PROTOCOLS = [protocol for variants in PANELS.values() for protocol, _ in variants]
# Pruned bars: the backend colour with thin light stripes.
PRUNED = {"hatch": "////", "edgecolor": (1, 1, 1, 0.55), "linewidth": 0}
matplotlib.rcParams["hatch.linewidth"] = 0.6
# Run directories are <backend>-<machine>-<dataset>.
BACKENDS = {"cuda": ("cuda", "CUDA hybrid"), "gpuonly": ("gpu_only", "CUDA, GPU only"),
            "cpu": ("parallel", "Parallel, both sockets")}
ORDER = ["cuda-hopper", "gpuonly-hopper", "cuda-ampere", "gpuonly-ampere", "cpu-epyc", "cpu-dell"]


def duration(value):
    return f"{value:.3g} s" if value >= 1 else f"{value * 1e3:.3g} ms"


def gib(value):
    return f"{value / 2**30:.1f} GB" if value >= 2**30 else f"{value / 2**20:.0f} MB"


def load(folder):
    """{dataset: {run name: {"machine", "meta", "cases": {protocol: [rows]}}}}"""
    runs = {}
    for results in sorted(folder.glob("*/results.jsonl")):
        name = results.parent.name
        plan = json.loads((results.parent / "plan.json").read_text())
        machine = json.loads((results.parent / "machine.json").read_text())
        for line in results.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] != "run" or row["dataset"] == B.BASELINE or row["protocol"] not in PROTOCOLS:
                continue
            run = runs.setdefault(row["dataset"], {}).setdefault(
                name, {"machine": machine, "meta": plan["datasets"][row["dataset"]], "cases": {}})
            run["cases"].setdefault(row["protocol"], []).append(row)
    return runs


def label(name, machine):
    backend, node = name.split("-")[:2]
    hw = B.hardware(machine)
    device = hw["gpu"] if backend in ("cuda", "gpuonly") and hw["gpu"] else hw["cpu"].split(",")[0]
    return backend, f"{node}: {BACKENDS[backend][1]}\n{device}"


def tree_text(rows):
    """'test 74.12 % · train 75.03 % · 1,023 nodes · depth 10' (means of the runs)."""
    test = [r["test_accuracy"] for r in rows if r.get("test_accuracy") is not None]
    train = [r["train_accuracy"] for r in rows if r.get("train_accuracy") is not None]
    parts = []
    if test:
        parts.append(f"test acc {100 * statistics.mean(test):.2f} %")
    if train:
        parts.append(f"train acc {100 * statistics.mean(train):.2f} %")
    parts.append(f"{statistics.mean(r['nodes'] for r in rows):,.0f} nodes")
    parts.append(f"depth {max(r['depth'] for r in rows)}")
    return " · ".join(parts)


def bar_note(rows, same, text):
    times = [r["train_seconds"] for r in rows]
    memory = f"RAM {gib(statistics.mean(r['peak_rss_train_bytes'] for r in rows))}"
    gpu = [r["gpu_peak_bytes"] for r in rows if r.get("gpu_peak_bytes") is not None]
    if gpu:
        memory += f" · VRAM {gib(statistics.mean(gpu))}"
    return f"{duration(statistics.mean(times))}   {memory}" + ("" if same else f"   {text}")


def chart(dataset, runs, out):
    names = sorted(runs, key=lambda n: (ORDER.index("-".join(n.split("-")[:2]))
                                        if "-".join(n.split("-")[:2]) in ORDER else 99, n))
    meta = next(iter(runs.values()))["meta"]
    reps = max(len(rows) for run in runs.values() for rows in run["cases"].values())
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 1.4 + len(PANELS) * (0.95 * len(names) + 1.2)),
                             sharex=True)
    largest = max(r["train_seconds"] for run in runs.values() for rows in run["cases"].values()
                  for r in rows)
    wide = False
    for ax, (algorithm, variants) in zip(axes, PANELS.items()):
        lines = [algorithm]
        for v, (protocol, title) in enumerate(variants):
            ok = {n: [r for r in runs[n]["cases"].get(protocol, []) if r["status"] == "ok"] for n in names}
            texts = {n: tree_text(rows) for n, rows in ok.items() if rows}
            same = len(set(texts.values())) == 1
            wide |= not same
            lines.append(f"{title}" + (f":  {next(iter(texts.values()))}" if same and texts else ""))
            for i, name in enumerate(names):
                backend, _ = label(name, runs[name]["machine"])
                rows = ok[name]
                # Unpruned on top, pruned right below it.
                y = len(names) - 1 - i + (0.21 if v == 0 else -0.21)
                if not rows:
                    failed = runs[name]["cases"].get(protocol, [])
                    ax.text(0, y, "  " + (failed[0]["status"] if failed else "not run"),
                            va="center", fontsize=8, color=MUTED)
                    continue
                times = [r["train_seconds"] for r in rows]
                ax.barh(y, statistics.mean(times), height=0.4, color=COLORS[BACKENDS[backend][0]],
                        **(PRUNED if v else {}))
                if len(times) > 1:
                    ax.plot([min(times), max(times)], [y, y], color=INK, lw=1)
                ax.text(max(times) + largest * 0.012, y, bar_note(rows, same, texts[name]),
                        va="center", fontsize=8, color=INK)
        ax.set_title("\n".join(lines), loc="left", fontsize=9.5, color=INK, linespacing=1.5)
        ax.title.set_fontweight("bold")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([label(n, runs[n]["machine"])[1] for n in reversed(names)], fontsize=8)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[-1].set_xlim(0, largest * (2.4 if wide else 1.55))
    axes[-1].xaxis.set_major_formatter(lambda v, _: seconds_tick(v) if v else "0")
    axes[-1].set_xlabel(f"training time, mean of {reps} runs (whisker: fastest to slowest); "
                        "RAM, VRAM: peak while training, mean", color=MUTED, fontsize=9)
    split = ("the official test set" if meta.get("spec", {}).get("test_source") or
             meta.get("spec", {}).get("split") == "tail" else "held-out rows")
    if "test_from" in meta.get("spec", {}):
        column, value = meta["spec"]["test_from"]
        split = f"every row with {column} ≥ {value}"
    height = fig.get_figheight()
    fig.suptitle(f"{NAMES.get(dataset, dataset)}: {meta['n_train']:,} training rows, "
                 f"{meta['n_test']:,} test rows ({split}), {len(meta['features'])} features, "
                 f"{len(meta['classes'])} classes", x=0.01, y=1 - 0.12 / height, ha="left", va="top",
                 fontsize=12, color=INK,
                 fontweight="bold")
    fig.text(0.01, 1 - 0.45 / height,
             "./tree, depth ≤ 10; trained on the training rows, accuracy on the test rows",
             fontsize=9, color=MUTED)
    used = sorted({BACKENDS[n.split("-")[0]][0] for n in names}, key=list(COLORS).index)
    handles = [Patch(color=COLORS[b], label=dict(BACKENDS.values())[b]) for b in used]
    handles += [Patch(facecolor=MUTED, label="upper bar: no pruning"),
                Patch(facecolor=MUTED, label="lower, striped: pruned", **PRUNED)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.62 / height),
               ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / height))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    runs = load(args.folder)
    if not runs:
        raise SystemExit(f"no runs of {', '.join(PROTOCOLS)} in {args.folder}")
    for dataset, by_name in runs.items():
        chart(dataset, by_name, args.folder / ("chart.png" if len(runs) == 1 else f"chart-{dataset}.png"))


if __name__ == "__main__":
    main()
