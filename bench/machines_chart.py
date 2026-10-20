#!/usr/bin/env python3
"""One chart of a ./tree benchmark run with another machine's run as reference:
training time and peak memory of every backend the runs hold (serial, parallel,
CUDA), one panel per protocol (CART above C4.5). Colour follows the backend; the reference run's bars
are hatched. The header names every machine's CPU, RAM and GPU; memory is split
into RAM (host, peak RSS) and VRAM (GPU) bars. When the runs recorded ./tree's
phases, a time bar is split into growing the tree (setup, presort, build) and,
lighter, pruning it.

Refuses to draw when the runs did not read the same data (SHA-256 of the
prepared files in plan.json) or did not build the same trees.

  bench/.venv/bin/python bench/machines_chart.py bench/results/cuda-legion-susy \\
      --reference bench/results/cluster-susy-vs-others-cpu/pc  # -> cuda-legion-susy/chart.png

--spread draws on every time bar a whisker from the fastest to the slowest
run and a dot per run, and writes their min and max.

--log draws the time and memory axes on a log scale (sizes that differ by
orders of magnitude, e.g. serial against CUDA) and writes every case's recorded
pruning time next to its bar, also for the unpruned protocols (where it is only
C4.5's collapse of useless splits) and below 1%.

--more adds the cases of later runs kept in another directory under the same
run names (e.g. cluster-ecbdl/unpruned/cpu-epyc-ecbdl next to
cluster-ecbdl/cpu-epyc-ecbdl), so that one chart shows both.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES  # noqa: E402

# Categorical slots 1-3 of the reference palette (validated all-pairs, CVD dE
# >= 9.2); slot 3 is below 3:1 on white, so every bar carries a value label.
# Slot 4 for ./tree --cuda --gpu-only (every node on the GPU, the CPU threads feed it and prune).
COLORS = {"serial": "#2a78d6", "parallel": "#eb6834", "cuda": "#1baf7a", "gpu_only": "#eda100"}
BACKENDS = {"serial": "Serial", "parallel": "Parallel", "cuda": "CUDA", "gpu_only": "GPU only"}
GPU = {"cuda", "gpu_only"}
TITLES = {"cart_alpha": "CART, fixed α = 1e-5, depth ≤ 30", "c45": "C4.5, CF 0.25",
          "cart_full": "CART, no pruning, depth ≤ 30", "c45_unpruned": "C4.5, no pruning",
          "cart_alpha_nodepth": "CART, fixed α = 1e-5", "cart_full_nodepth": "CART, no pruning"}
PROTOCOLS = {p: TITLES[p] for p in ("cart_alpha", "c45")}  # the panels; --protocols
# No pruning: their "prune" timer holds only C4.5's collapse of useless splits
# and its thresholds, part of growing; the bars are not split.
UNPRUNED = {"cart_full", "c45_unpruned", "cart_full_nodepth"}
# The files ./tree and the evaluation read: equal hashes = the same rows.
DATA_FILES = ("train.csv", "train.f32", "train.y.i32", "test.f32", "test.y.i32")
LEFT = 0.215  # figure fraction left of the panels, for the row labels
# Result directories are <backend>-<machine>-<dataset>; what to call each machine.
MACHINES = {"pc": "desktop", "legion": "laptop"}
MORE = []  # --more: directories with later runs of the same names
NO_VRAM = set()  # --no-vram: run names whose recorded VRAM is not to be trusted
SPREAD = False  # --spread: a min-max whisker and a dot per run on every time bar
LOG = False  # --log: log-scale axes, and every case's pruning time in its notes


def after(value, largest):
    """Where a label goes just right of a bar ending at `value`."""
    return value * 1.06 if LOG else value + largest * 0.012


def duration(seconds):
    if seconds < 0.00005:
        return "0 ms"
    return f"{seconds * 1000:.1f} ms" if seconds < 0.1 else f"{seconds:.2f} s"


def sources(run):
    """The run's directory, then those of its later runs in the --more directories."""
    return [run] + [more / run.name for more in MORE if (more / run.name / "results.jsonl").exists()]


def lighter(color, amount=0.55):
    """`color` mixed with white: the pruning part of a time bar."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)


def where(run):
    name = run.name.split("-")[1]
    return MACHINES.get(name, name)


def short_cpu(info):
    return info["cpu"].split(",")[0].replace("Intel Core ", "")


def short_gpu(info):
    return info["gpu"].split(",")[0].replace("NVIDIA ", "") if info["gpu"] else "GPU"


def bars(run, reference):
    """-> one dict per ./tree case of the run that went well."""
    rows = [json.loads(line) for source in sources(run) for line in open(source / "results.jsonl")]
    info = B.hardware(json.load(open(run / "machine.json")))
    out = []
    for protocol in PROTOCOLS:
        for impl, single in (("tree", True), ("tree", False), ("tree_cuda", False),
                             ("tree_gpu_only", True)):
            ok = [r for r in rows if r["kind"] == "run" and r["status"] == "ok"
                  and r["protocol"] == protocol and r["impl"] == impl
                  and (r["threads"] == 1) == single]
            if not ok:
                continue
            threads = ok[0]["threads"]
            backend = {"tree_cuda": "cuda", "tree_gpu_only": "gpu_only"}.get(
                impl, "serial" if single else "parallel")
            hardware = (f"{short_gpu(info)} + {threads} CPU thread{'s' if threads > 1 else ''}"
                        if backend in GPU
                        else f"{short_cpu(info)}, {threads} thread{'s' if threads > 1 else ''}")
            gpu = [r["gpu_peak_bytes"] for r in ok if r.get("gpu_peak_bytes") is not None]
            prune = [r["prune_seconds"] for r in ok if r.get("prune_seconds") is not None]
            out.append({"protocol": protocol, "backend": backend, "hardware": hardware,
                        "run": run, "reference": reference,
                        "time": statistics.median(r["train_seconds"] for r in ok),
                        "times": [r["train_seconds"] for r in ok],
                        "ram": statistics.median(r["peak_rss_train_bytes"] for r in ok) / 2**30,
                        "vram": (statistics.median(gpu) / 2**30
                                 if gpu and run.name not in NO_VRAM else None),
                        # Median pruning time, if every run recorded it.
                        "prune": (statistics.median(prune)
                                  if len(prune) == len(ok) and protocol not in UNPRUNED else None),
                        "pruned": statistics.median(prune) if len(prune) == len(ok) else None,
                        # Test accuracy, or training accuracy where only that was
                        # recorded (run.py --train-accuracy-only).
                        "tree": (ok[0]["nodes"], ok[0]["depth"],
                                 *((ok[0]["test_accuracy"], "test")
                                   if ok[0].get("test_accuracy") is not None
                                   else (ok[0]["train_accuracy"], "training")))})
    return out


def check_same(runs):
    plans = [json.load(open(source / "plan.json")) for run in runs for source in sources(run)]
    datasets = {d for plan in plans for d in plan["data_files"] if d != B.BASELINE}
    if len(datasets) != 1:
        raise SystemExit(f"the runs used different datasets: {sorted(datasets)}")
    dataset = datasets.pop()
    for name in DATA_FILES:
        if len({plan["data_files"][dataset][name] for plan in plans}) != 1:
            raise SystemExit(f"{dataset}/{name} differs between the runs: not the same data")
    return dataset, plans[0]["datasets"][dataset]


def style(entry):
    color = COLORS[entry["backend"]]
    if entry["reference"]:
        return dict(facecolor="white", edgecolor=color, hatch="////", linewidth=1.2)
    return dict(color=color)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", nargs="*", type=Path, default=[],
                        help="runs drawn hatched as reference (another machine's run)")
    parser.add_argument("--out", type=Path, help="default: <first run>/chart.png")
    parser.add_argument("--protocols", default="cart_alpha,c45",
                        help=f"the protocols to draw, one panel each, of: {', '.join(TITLES)}")
    parser.add_argument("--spread", action="store_true",
                        help="draw each time bar's min-max whisker and every run as a dot")
    parser.add_argument("--no-vram", action="append", default=[], metavar="RUN",
                        help="a run name whose VRAM sample is wrong: drawn without VRAM (repeatable)")
    parser.add_argument("--log", action="store_true",
                        help="log-scale axes, and every case's recorded pruning time")
    parser.add_argument("--more", action="append", type=Path, default=[],
                        help="a directory with later runs of the same names, e.g. other "
                        "protocols (repeatable)")
    args = parser.parse_args()
    MORE.extend(args.more)
    NO_VRAM.update(args.no_vram)
    global SPREAD, LOG
    SPREAD = args.spread
    LOG = args.log
    PROTOCOLS.clear()
    PROTOCOLS.update((p, TITLES[p]) for p in args.protocols.split(","))
    runs = list(args.reference) + list(args.runs)
    out = args.out or args.runs[0] / "chart.png"
    dataset, meta = check_same(runs)
    entries = [b for run in runs for b in bars(run, run in args.reference)]
    for protocol in PROTOCOLS:
        trees = {e["tree"][:2] for e in entries if e["protocol"] == protocol}
        if len(trees) > 1:
            raise SystemExit(f"{protocol}: different trees {sorted(trees)}")

    order = list(BACKENDS)
    per_panel = max(sum(e["protocol"] == p for e in entries) for p in PROTOCOLS)
    gaps = max(len({e["backend"] for e in entries if e["protocol"] == p}) for p in PROTOCOLS) - 1
    # One hardware line per machine, the run being reported first; a machine with
    # several runs (e.g. serial and parallel) lists them all.
    machines = {}
    for run in reversed(runs):
        machines.setdefault(where(run), []).append(run)
    header = 0.42 + 0.2 * len(machines)  # inches of the hardware lines
    footer = 0.18 if any(e["prune"] is not None and not e["reference"] for e in entries) else 0
    footer += 0.18 if SPREAD else 0  # the footer's line on the whiskers
    # At least three bars high: the row labels take two lines each.
    count = len(PROTOCOLS)
    inches = 1.9 + header + footer + count * 0.58 * max(per_panel + 0.45 * gaps, 3)
    split = any(e["prune"] is not None and not e["reference"] for e in entries)
    # Between the panels: the lower one's title and subtitle and the upper one's
    # tick labels, the same in inches however few bars the panels hold.
    gap = 0.92
    panels = inches - (header + 1.05) - (1.08 if split else 0.9) - (0.18 if SPREAD else 0)
    fig, axes = plt.subplots(count, 2, figsize=(13, inches), squeeze=False,
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.12,
                                          "hspace": gap / ((panels - (count - 1) * gap) / count)})
    max_time = max(max(e["times"]) if SPREAD else e["time"] for e in entries)
    max_gib = max(max(e["ram"], e["vram"] or 0) for e in entries)
    min_time = min(min(e["times"]) if SPREAD else e["time"] for e in entries)
    min_gib = min(min(e["ram"], e["vram"] or e["ram"]) for e in entries)
    for row, protocol in enumerate(PROTOCOLS):
        ax_time, ax_mem = axes[row]
        mine = sorted((e for e in entries if e["protocol"] == protocol),
                      key=lambda e: (order.index(e["backend"]), runs.index(e["run"])))
        nodes, depth, accuracy, accuracy_part = mine[0]["tree"]
        y, ys, previous = 0.0, [], None
        for e in mine:
            if previous is not None and e["backend"] != previous:
                y += 0.45
            ys.append(y)
            y += 1
            previous = e["backend"]
        for ax in (ax_time, ax_mem):
            ax.set_ylim(y - 0.35, -0.65)
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
        ax_time.set_title(f"{PROTOCOLS[protocol]}: training time", loc="left", fontsize=10.5,
                          color=INK, fontweight="bold", pad=20)
        ax_time.text(0, 1.02, f"the same tree on every machine and backend: {nodes:,} nodes, "
                     f"depth {depth}, {accuracy_part} accuracy {100 * accuracy:.2f}%",
                     transform=ax_time.transAxes, fontsize=8, color=MUTED, va="bottom")
        ax_mem.set_title("peak memory: RAM (host) and VRAM (GPU)", loc="left", fontsize=10.5,
                         color=INK, fontweight="bold", pad=20)
        times = {(e["run"], e["backend"]): e["time"] for e in mine}
        for y, e in zip(ys, mine):
            run, backend, t = e["run"], e["backend"], e["time"]
            prune = e["prune"]
            if prune is not None and not e["reference"]:
                # Growing (everything but pruning), then pruning in a lighter shade.
                ax_time.barh(y, t - prune, height=0.62, zorder=2, **style(e))
                ax_time.barh(y, prune, left=t - prune, height=0.62, zorder=2,
                             color=lighter(COLORS[backend]))
            else:
                ax_time.barh(y, t, height=0.62, zorder=2, **style(e))
            # Pruning's share; speedup over the same machine's serial run; CUDA
            # also over the same machine's parallel run and the fastest other
            # machine's.
            if LOG and e["pruned"] is not None:
                notes = [f"prune {duration(e['pruned'])}"]
            else:
                notes = ([f"prune {prune:.2f} s ({100 * prune / t:.0f}%)"]
                         if prune is not None and prune >= 0.01 * t else [])
            if backend != "serial" and (run, "serial") in times:
                notes.append(f"{times[(run, 'serial')] / t:.1f}× {where(run)} serial")
            if backend in GPU:
                others = [o for o in runs if o != run and (o, "parallel") in times]
                fastest = min(others, key=lambda o: times[(o, "parallel")], default=None)
                notes += [f"{times[(other, 'parallel')] / t:.1f}× {where(other)} parallel"
                          for other in (run, fastest) if (other, "parallel") in times]
            end = t
            if SPREAD and len(e["times"]) > 1:
                low, high = min(e["times"]), max(e["times"])
                ax_time.plot([low, high], [y, y], color=INK, linewidth=1.1, zorder=3)
                for x in (low, high):
                    ax_time.plot([x, x], [y - 0.17, y + 0.17], color=INK, linewidth=1.1, zorder=3)
                # The runs in run order, spread a little vertically so equal times stay visible.
                n = len(e["times"])
                ax_time.scatter(e["times"], [y - 0.18 + 0.36 * k / (n - 1) for k in range(n)],
                                s=11, color="white", edgecolor=INK, linewidth=0.7, zorder=4)
                end = max(t, high)
                notes.insert(0, f"min {duration(low)} · max {duration(high)}" if LOG
                             else f"min {low:.2f} · max {high:.2f} s")
            ax_time.text(after(end, max_time), y, duration(t) if LOG else f"{t:.2f} s", va="center", fontsize=9,
                         color=INK, fontweight="bold")
            ax_time.annotate(" · ".join(notes), (end, y), xytext=(52, 0),
                             textcoords="offset points", va="center", fontsize=8, color=MUTED)
            name = f"{BACKENDS[backend]} · {where(run)}" + (" (reference)" if e["reference"]
                                                            else "")
            ax_time.text(-0.015, y - 0.14, name, transform=ax_time.get_yaxis_transform(),
                         ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold")
            ax_time.text(-0.015, y + 0.24, e["hardware"], transform=ax_time.get_yaxis_transform(),
                         ha="right", va="center", fontsize=7.8, color=MUTED)
            if e["vram"] is not None:
                ax_mem.barh(y - 0.16, e["ram"], height=0.3, zorder=2, **style(e))
                ax_mem.barh(y + 0.16, e["vram"], height=0.3, zorder=2, facecolor="white",
                            edgecolor=COLORS[backend], linewidth=1.4)
                ax_mem.text(after(e["ram"], max_gib), y - 0.16, f"{e['ram']:.2f} GiB RAM",
                            va="center", fontsize=8.5, color=INK)
                ax_mem.text(after(e["vram"], max_gib), y + 0.16, f"{e['vram']:.2f} GiB VRAM",
                            va="center", fontsize=8.5, color=INK)
            else:
                ax_mem.barh(y, e["ram"], height=0.62, zorder=2, **style(e))
                ax_mem.text(after(e["ram"], max_gib), y, f"{e['ram']:.2f} GiB RAM",
                            va="center", fontsize=8.5, color=INK)
        if LOG:
            # Decades from below the shortest bar; room on the right for the notes.
            ax_time.set_xscale("log")
            ax_time.set_xlim(10 ** math.floor(math.log10(min_time)), max_time * 60)
            ax_time.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
            ax_mem.set_xscale("log")
            ax_mem.set_xlim(10 ** math.floor(math.log10(min_gib)), max_gib * 6)
            ax_mem.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} GiB"))
        else:
            ax_time.set_xlim(0, max_time * 2.0)
            ax_time.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
            ax_mem.set_xlim(0, max_gib * 1.45)
            ax_mem.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax_mem.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} GiB"))

    top = 1 - 0.09 / inches
    fig.suptitle(f"./tree on {NAMES.get(dataset, dataset)} ({meta['n_train']:,} training rows, "
                 f"{len(meta['features'])} features): "
                 + " vs ".join(dict.fromkeys(where(r) for r in reversed(runs))),
                 x=0.012, y=top, ha="left", fontsize=12, color=INK, fontweight="bold")
    # Hardware: one line per machine (see `machines` above).
    for index, (name, mine) in enumerate(machines.items()):
        machine = json.load(open(mine[0] / "machine.json"))
        used_gpu = any(e["run"] in mine and e["backend"] in GPU for e in entries)
        roles, dates = [], []
        for run in mine:
            # Only ./tree is drawn: when its cases were rerun later, that is the date.
            info = json.load(open(run / "machine.json"))
            dates.append(info.get("tree_rerun", info)["started"][:10])
            later = sources(run)[1:]
            dates += [json.load(open(more / "machine.json"))["started"][:10] for more in later]
            roles.append(("reference: " if run in args.reference else "") + run.name
                         + "".join(f" + {more.parent.name}/" for more in later))
        role = " and ".join(roles) + f", run {' and '.join(dict.fromkeys(dates))}"
        line_y = top - (0.36 + 0.2 * index) / inches
        fig.text(0.012, line_y, f"{name.capitalize()}", fontsize=9.2, color=INK,
                 fontweight="bold", va="top")
        line = f"{B.hardware_line(machine, used_gpu)}   ({role})"
        # About 165 characters fit at 9.2 pt; a longer line is set smaller.
        fig.text(0.075, line_y, line, fontsize=min(9.2, 9.2 * 165 / len(line)), color=INK,
                 va="top")
    handles = [Patch(color=COLORS[b], label=BACKENDS[b]) for b in BACKENDS
               if any(e["backend"] == b for e in entries)]
    if split:
        handles.append(Patch(color=lighter(MUTED), label="lighter: pruning"))
    if args.reference:
        handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch="////",
                             label="reference machine (hatched)"))
    handles.append(Patch(color=MUTED, label="RAM: filled / hatched bar"))
    if any(e["vram"] is not None for e in entries):
        handles.append(Patch(facecolor="white", edgecolor=MUTED, linewidth=1.4,
                             label="VRAM: outlined bar"))
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(LEFT - 0.005, top - (header + 0.06) / inches),
               ncol=len(handles), frameon=False, fontsize=8.8, handletextpad=0.4,
               columnspacing=1.3)
    reps = {json.load(open(source / "machine.json"))["settings"]["reps"]
            for run in runs for source in sources(run)}
    runs_text = "one run per case" if reps == {1} else "median over the runs of each case"
    if SPREAD and reps != {1}:
        runs_text += " (whisker: fastest to slowest run; dots: every run).\n"
    fig.text(LEFT, 0.1 / inches,
             f"Bars: {runs_text}{'' if runs_text.endswith(chr(10)) else '. '}Time: ./tree's 'train total' (presort, build, prune; CUDA also "
             "upload and device sort), not reading the CSV."
             + (" Log-scale axes; 'prune': ./tree's prune timer." if LOG else "")
             + ("\nDarker: growing the tree (setup, presort, build); lighter: pruning it "
                "(shown when it takes 1% or more)." if split else "")
             + "\nSame data files (SHA-256) on every "
             "machine. RAM: peak RSS of the process up to the end of training.\nVRAM: peak GPU "
             "memory during the process above idle, CUDA context included (nvidia-smi, every 10 ms)."
             + (f" Not recorded correctly: {', '.join(sorted(NO_VRAM))}." if NO_VRAM else ""),
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=LEFT, right=0.97, top=1 - (header + 1.05) / inches,
                        bottom=((1.08 if split else 0.9) + (0.18 if SPREAD else 0)) / inches)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
