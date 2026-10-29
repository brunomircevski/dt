#!/usr/bin/env python3
"""The test-set benchmark's chart (cluster-airline, -higgs, -susy, -ecbdl, -mnist8m, -sorel): one
dataset, ./tree on every node, trained on the training set and scored on the
test set. One panel per algorithm (CART, C4.5; depth <= 10, no pruning), one
bar per run (node and backend), on a log scale, the bars grouped by backend (--simple,
--serial, parallel on one socket, on both, CUDA hybrid, GPU only), the nodes under each. Tall panels go
side by side, sharing the bars' labels. A bar is the mean training time
of its runs, a whisker from the fastest to the slowest. Next to each bar:
the time ± its standard deviation, the number of runs, and peak RAM (and VRAM
on a GPU) as numbers, the means of the runs. A run folder still in the queue
or running is drawn as "queued" or "still running". cluster-chess (no depth
limit, training accuracy only) gets the same chart, one per size.
The panel title holds the trees: test and training accuracy, nodes, depth (when
the runs grew different trees, each bar says its own).
A potential problem is marked on its bar (hatched, "⚠" and a note, in red):
runs more than twice the mean of the other runs (found here), and the notes of <dir>/problems.json,
{"<run>": "note"} or {"<run>": {"<protocol>": "note"}}, e.g. a cause found or
suspected (BENCHMARK.md explains them). A problems.json run with no folder (a run
not done, e.g. cancelled) gets a row with its note in place of a bar.
Reruns in progress: the runs of <dir>-fix/, then <dir>-par2/ (bench/cluster/README.md),
replace those of <dir> when done; one still queued or running marks the old run's bar
("old build; rerun queued"), or gets a "queued" row if there is none (only the
problems.json notes of the folder a run is taken from apply to it).

  bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-higgs
  bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-ecbdl  # one chart per dataset
  bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-chess

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
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES, seconds_tick  # noqa: E402
from machines_chart import COLORS  # noqa: E402  (the validated backend colours)

# One panel per algorithm, both without pruning: the test-set protocols (depth 10), or
# cluster-chess's (no depth limit).
PANELS = {"cart_depth10": "CART, gini, no pruning", "c45_depth10_unpruned": "C4.5, gain ratio, no pruning",
          "cart_full_nodepth": "CART, gini, no depth limit, no pruning",
          "c45_unpruned": "C4.5, gain ratio, no depth limit, no pruning"}
PROTOCOLS = list(PANELS)
COLORS = {**COLORS, "simple": "#d14f8e"}
# Run directories are <backend>-<machine>-<dataset>.
BACKENDS = {"cuda": ("cuda", "CUDA hybrid"), "gpuonly": ("gpu_only", "CUDA, GPU only"),
            "cpu": ("parallel", "Parallel, both sockets"), "socket": ("socket", "Parallel, one socket"),
            "serial": ("serial", "--serial, one thread"),
            "simple": ("simple", "--simple, one thread")}
WARN = "#c0392b"
GROUPS = ["simple", "serial", "socket", "cpu", "cuda", "gpuonly"]  # the backends, top to bottom
NODES = ["epyc", "dell", "dell512", "hopper", "ampere"]  # the nodes in each
GAP = 0.6  # the room of a backend's heading, in bars
BAR = 0.42  # inches a bar takes (its slot)
TALL = 5.0  # inches of a panel above which the panels go side by side


def duration(value):
    if value >= 1000:
        return f"{value:,.0f} s"
    return f"{value:.3g} s" if value >= 1 else f"{value * 1e3:.3g} ms"


def gib(value):
    return f"{value / 2**30:.1f} GB" if value >= 2**30 else f"{value / 2**20:.0f} MB"


def load(folder, datasets=()):
    """{dataset: {run name: {"machine", "meta", "cases": {protocol: [rows]}}}}"""
    runs = {dataset: {} for dataset in datasets}
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
    # Folders still in the queue (no machine.json yet) or running (no report.md, or a --simple
    # job's run-<protocol>.log not done): their missing cases are labelled so. A folder with
    # no results yet goes to every dataset whose name it ends in (or to all).
    for folder_ in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.count("-") >= 2):
        logs = list(folder_.glob("run-*.log"))
        done = (folder_ / "report.md").exists() and all("\ndone " in log.read_text() for log in logs)
        if done:
            continue
        machine = folder_ / "machine.json"
        state = "running" if machine.exists() else "queued"
        if not machine.exists():
            machine = node_machine(folder, folder_.name)
            if machine is None:
                continue
        for dataset, by_name in runs.items():
            if folder_.name in by_name or "-".join(folder_.name.split("-")[2:]) in (dataset, ""):
                by_name.setdefault(folder_.name, {"machine": json.loads(machine.read_text()), "meta": None,
                                                  "cases": {}})["state"] = state
    return runs


def overlay(runs, fix):
    """The reruns of <dir>-fix (or -par2) over the runs: the done ones replace them (marked
    with the folder's suffix as "from"), an old run with one pending gets its state as "rerun"."""
    suffix = fix.name[fix.name.rindex("-"):]
    for dataset, by_name in load(fix, runs).items():
        for name, run in by_name.items():
            old = runs.setdefault(dataset, {}).get(name)
            if "state" not in run or old is None or old.get("state"):  # (an old folder never finished)
                runs[dataset][name] = {**run, "from": suffix}
            elif old.get("state") is None:
                old["rerun"] = run["state"]


def node_machine(folder, name):
    """The machine.json of a finished folder of the run's node, here or in another benchmark."""
    node = name.split("-")[1]
    return next(iter(sorted(folder.glob(f"*-{node}-*/machine.json"))
                     + sorted(folder.parent.glob(f"cluster-*/*-{node}-*/machine.json"))), None)


def add_unrun(folder, runs, notes):
    """A run of problems.json with no folder (e.g. cancelled) gets a row of its own, labelled
    with its note, in the dataset its name ends in."""
    for name in notes:
        machine = node_machine(folder, name)
        for dataset, by_name in runs.items():
            if name not in by_name and name.endswith("-" + dataset) and machine is not None:
                by_name[name] = {"machine": json.loads(machine.read_text()), "meta": None, "cases": {},
                                 "state": "unrun"}


def label(name, machine):
    backend, node = name.split("-")[:2]
    hw = B.hardware(machine)
    device = hw["gpu"] if backend in ("cuda", "gpuonly") and hw["gpu"] else hw["cpu"].split(",")[0]
    return backend, f"{node}\n{device}"


def layout(names):
    """The runs in order (by backend, then node), their y (top down) and the backends' headings."""
    def key(name):
        backend, node = name.split("-")[:2]
        return (GROUPS.index(backend) if backend in GROUPS else 99, NODES.index(node) if node in NODES else 99,
                name)
    names = sorted(names, key=key)
    ys, headings, y = {}, [], 0.0
    for i, name in enumerate(names):
        backend = name.split("-")[0]
        if i == 0 or names[i - 1].split("-")[0] != backend:
            y -= GAP if i == 0 else GAP + 0.1
            headings.append((y + GAP / 2 + 0.12, backend))
        ys[name] = y
        y -= 1
    return names, ys, headings, -y - 0.5


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
    time = duration(statistics.mean(times))
    if len(times) > 1:
        time += f" ± {duration(statistics.stdev(times))}"
    runs = f"{len(times)} run" + ("s" if len(times) > 1 else "")
    memory = f"RAM {gib(statistics.mean(r['peak_rss_train_bytes'] for r in rows))}"
    gpu = [r["gpu_peak_bytes"] for r in rows if r.get("gpu_peak_bytes") is not None]
    if gpu:
        memory += f" · VRAM {gib(statistics.mean(gpu))}"
    return f"{time} ({runs})   {memory}" + ("" if same else f"   {text}")


def problems(rows, note):
    """The bar's warnings: slow outliers (more than twice the mean of the other runs) and the
    problems.json note."""
    times = [r["train_seconds"] for r in rows]
    total = sum(times)
    slow = sum(t > 2 * (total - t) / (len(times) - 1) for t in times) if len(times) >= 3 else 0
    found = [f"{slow} of {len(times)} runs > 2× the mean of the others"] if slow else []
    return found + ([note] if note else [])


def chart(dataset, runs, out, notes={}):
    names, ys, headings, span = layout(runs)
    meta = next(run["meta"] for run in runs.values() if run["meta"])
    reps = max(len(rows) for run in runs.values() for rows in run["cases"].values())
    panels = [(p, PANELS[p]) for p in PROTOCOLS if any(p in run["cases"] for run in runs.values())]
    test_set = any(r.get("test_accuracy") is not None for run in runs.values()
                   for rows in run["cases"].values() for r in rows)
    side = len(panels) > 1 and BAR * span > TALL  # tall panels: side by side, sharing the bars' labels
    fig, axes = plt.subplots(1 if side else len(panels), len(panels) if side else 1, sharex=True,
                             sharey=side, squeeze=False,
                             figsize=(10.5 * len(panels) if side else 11,
                                      1.6 + BAR * span + 1.2 if side else 1.4 + len(panels) * (BAR * span + 1.2)))
    axes = axes.ravel()
    times_all = [r["train_seconds"] for run in runs.values() for protocol, rows in run["cases"].items()
                 if protocol in PROTOCOLS for r in rows if r["status"] == "ok"]
    largest, low = max(times_all), min(times_all) / 2
    wide = False
    for ax, (protocol, title) in zip(axes, panels):
        ok = {n: [r for r in runs[n]["cases"].get(protocol, []) if r["status"] == "ok"] for n in names}
        texts = {n: tree_text(rows) for n, rows in ok.items() if rows}
        same = len(set(texts.values())) == 1
        wide |= not same
        for i, name in enumerate(names):
            backend, _ = label(name, runs[name]["machine"])
            rows = ok[name]
            y = ys[name]
            if not rows:
                failed = runs[name]["cases"].get(protocol, [])
                state = runs[name].get("state")
                note = notes.get(runs[name].get("from", ""), {}).get(name)
                note = note.get(protocol) if isinstance(note, dict) else note
                ax.text(low * 1.05, y, (B.failure(failed[0]) if failed else note if state == "unrun" and note else
                                        {"running": "still running", "queued": "queued"}.get(state, "not run")),
                        va="center", fontsize=8, color=MUTED, style="italic")
                continue
            times = [r["train_seconds"] for r in rows]
            mean = statistics.mean(times)
            note = notes.get(runs[name].get("from", ""), {}).get(name)
            note = note.get(protocol) if isinstance(note, dict) else note
            rerun = runs[name].get("rerun")
            warn = problems(rows, f"old build; rerun {'queued' if rerun == 'queued' else 'running'}" if rerun
                            else note)
            ax.barh(y, mean - low, left=low, height=0.62, color=COLORS[BACKENDS[backend][0]],
                    hatch="////" if warn else None, edgecolor=WARN if warn else None, lw=0)
            if len(times) > 1:
                ax.plot([min(times), max(times)], [y, y], color=INK, lw=1)
            ax.text(max(times) * 1.06, y + (0.19 if warn else 0), bar_note(rows, same, texts[name]),
                    va="center", fontsize=8, color=INK)
            if warn:
                ax.text(max(times) * 1.06, y - 0.21, "⚠ " + "; ".join(warn), va="center", fontsize=7, color=WARN)
        ax.set_title(title + (f":  {next(iter(texts.values()))}" if same and texts else ""),
                     loc="left", fontsize=9.5, color=INK)
        ax.title.set_fontweight("bold")
        for y, backend in headings if ax is axes[0] or not side else ():
            ax.text(-0.01, y, BACKENDS[backend][1], transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=8.5, fontweight="bold", color=COLORS[BACKENDS[backend][0]])
        ax.set_yticks([ys[n] for n in names])
        ax.set_yticklabels([label(n, runs[n]["machine"])[1] for n in names], fontsize=8)
        ax.tick_params(axis="y", labelleft=ax is axes[0] or not side)
        ax.set_ylim(-span - 0.1, 0.1)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[-1].set_xscale("log")
    axes[-1].set_xlim(low, largest * (largest / low) ** (0.9 if wide else 0.45))  # room for the labels
    ticks = [m * 10.0 ** e for e in range(-3, 6) for m in (1, 2, 5) if low <= m * 10.0 ** e <= largest * 3]
    axes[-1].set_xticks(ticks if len(ticks) < 9 else [t for t in ticks if str(t).lstrip("0.")[0] == "1"])
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: seconds_tick(v)))
    axes[-1].xaxis.set_minor_formatter(NullFormatter())
    axes[0 if side else -1].set_xlabel(f"training time (log scale): mean ± standard deviation of each bar's runs (1 to {reps}), "
                        "whisker: fastest to slowest\nRAM, VRAM: peak while training, mean of the runs; "
                        "hatched, ⚠: a potential problem (see BENCHMARK.md)", color=MUTED, fontsize=9)
    split = ("the official test set" if meta.get("spec", {}).get("test_source") or
             meta.get("spec", {}).get("split") == "tail" else "held-out rows")
    if "test_from" in meta.get("spec", {}):
        column, value = meta["spec"]["test_from"]
        split = f"every row with {column} ≥ {value}"
    height = fig.get_figheight()
    rows_text = (f"{meta['n_train']:,} training rows, {meta['n_test']:,} test rows ({split})" if test_set
                 else f"{meta['n_train']:,} rows")
    fig.suptitle(f"{NAMES.get(dataset, dataset)}: {rows_text}, {len(meta['features'])} features, "
                 f"{len(meta['classes'])} classes", x=0.01, y=1 - 0.12 / height, ha="left", va="top",
                 fontsize=11, color=INK,
                 fontweight="bold")
    fig.text(0.01, 1 - 0.45 / height,
             ("./tree, depth ≤ 10, no pruning; trained on the training rows, accuracy on the test rows" if test_set
              else "./tree, no depth limit, no pruning; accuracy on the training rows"),
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / height))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def table(dataset, runs):
    """The runs as a Markdown table (BENCHMARK.md), in the chart's order: training time mean ±
    standard deviation per protocol, the number of runs, peak RAM and VRAM (means)."""
    names, _, _, _ = layout(runs)
    panels = [p for p in PROTOCOLS if any(p in run["cases"] for run in runs.values())]
    titles = [PANELS[p].split(",")[0] for p in panels]
    lines = [f"| Run | runs | {' | '.join(titles)} | RAM GB (peak) | VRAM GB (peak) |",
             "|---|---|" + "---|" * len(panels) + "---|---|"]
    for name in names:
        backend, node = name.split("-")[:2]
        ok = {p: [r for r in runs[name]["cases"].get(p, []) if r["status"] == "ok"] for p in panels}
        rows = [r for p in panels for r in ok[p]]
        if not rows:
            continue
        cells = []
        for p in panels:
            times = [r["train_seconds"] for r in ok[p]]
            cells.append("–" if not times else duration(statistics.mean(times)) +
                         (f" ± {duration(statistics.stdev(times))}" if len(times) > 1 else ""))
        counts = sorted({len(ok[p]) for p in panels if ok[p]})
        ram = statistics.mean(r["peak_rss_train_bytes"] for r in rows) / 2**30
        gpu = [r["gpu_peak_bytes"] for r in rows if r.get("gpu_peak_bytes") is not None]
        what = BACKENDS[backend][1].replace("Parallel", "parallel")
        lines.append(f"| {node}, {what} | {'/'.join(map(str, counts))} | {' | '.join(cells)} | "
                     f"{ram:.1f} | {f'{statistics.mean(gpu) / 2**30:.1f}' if gpu else '–'} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--table", action="store_true", help="print the runs as Markdown tables, no chart")
    args = parser.parse_args()
    runs = load(args.folder)
    if not runs:
        raise SystemExit(f"no runs of {', '.join(PROTOCOLS)} in {args.folder}")
    notes = {}  # {"" or "-fix" or "-par2": that folder's problems.json}
    for suffix in ("", "-fix", "-par2"):
        d = args.folder.with_name(args.folder.name + suffix)
        if suffix and d.is_dir():
            overlay(runs, d)
        notes[suffix] = {} if not (d / "problems.json").exists() else json.loads((d / "problems.json").read_text())
    add_unrun(args.folder, runs, notes[""])
    for dataset, by_name in runs.items():
        if args.table:
            print(f"{dataset}:\n\n{table(dataset, by_name)}\n")
            continue
        name = "chart" if len(runs) == 1 else f"chart-{dataset}"
        chart(dataset, by_name, args.folder / f"{name}.png", notes)


if __name__ == "__main__":
    main()
