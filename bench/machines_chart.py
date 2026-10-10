#!/usr/bin/env python3
"""One chart of a ./tree benchmark run with another machine's run as reference:
training time and peak memory of every backend the runs hold (serial, parallel,
CUDA), CART above C4.5. Colour follows the backend; the reference run's bars
are hatched. The header names every machine's CPU, RAM and GPU; memory is split
into RAM (host, peak RSS) and VRAM (GPU) bars.

Refuses to draw when the runs did not read the same data (SHA-256 of the
prepared files in plan.json) or did not build the same trees.

  bench/.venv/bin/python bench/machines_chart.py bench/results/cuda-legion-susy \\
      --reference bench/results/cpu-pc-susy          # -> cuda-legion-susy/chart.png
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import benchlib as B  # noqa: E402
from chart import GRID, INK, MUTED, NAMES  # noqa: E402

# Categorical slots 1-3 of the reference palette (validated all-pairs, CVD dE
# >= 9.2); slot 3 is below 3:1 on white, so every bar carries a value label.
COLORS = {"serial": "#2a78d6", "parallel": "#eb6834", "cuda": "#1baf7a"}
BACKENDS = {"serial": "Serial", "parallel": "Parallel", "cuda": "CUDA"}
PROTOCOLS = {"cart_alpha": "CART, fixed α = 1e-5, depth ≤ 30", "c45": "C4.5, CF 0.25"}
# The files ./tree and the evaluation read: equal hashes = the same rows.
DATA_FILES = ("train.csv", "train.f32", "train.y.i32", "test.f32", "test.y.i32")
LEFT = 0.215  # figure fraction left of the panels, for the row labels
# Result directories are <backend>-<machine>-<dataset>; what to call each machine.
MACHINES = {"pc": "desktop", "legion": "laptop"}


def where(run):
    name = run.name.split("-")[1]
    return MACHINES.get(name, name)


def short_cpu(info):
    return info["cpu"].split(",")[0].replace("Intel Core ", "")


def short_gpu(info):
    return info["gpu"].split(",")[0].replace("NVIDIA ", "") if info["gpu"] else "GPU"


def bars(run, reference):
    """-> one dict per ./tree case of the run that went well."""
    rows = [json.loads(line) for line in open(run / "results.jsonl")]
    info = B.hardware(json.load(open(run / "machine.json")))
    out = []
    for protocol in PROTOCOLS:
        for impl, single in (("tree", True), ("tree", False), ("tree_cuda", False)):
            ok = [r for r in rows if r["kind"] == "run" and r["status"] == "ok"
                  and r["protocol"] == protocol and r["impl"] == impl
                  and (r["threads"] == 1) == single]
            if not ok:
                continue
            threads = ok[0]["threads"]
            backend = "cuda" if impl == "tree_cuda" else "serial" if single else "parallel"
            hardware = (f"{short_gpu(info)} + {threads} CPU threads" if backend == "cuda"
                        else f"{short_cpu(info)}, {threads} thread{'s' if threads > 1 else ''}")
            gpu = [r["gpu_peak_bytes"] for r in ok if r.get("gpu_peak_bytes") is not None]
            out.append({"protocol": protocol, "backend": backend, "hardware": hardware,
                        "run": run, "reference": reference,
                        "time": statistics.median(r["train_seconds"] for r in ok),
                        "ram": statistics.median(r["peak_rss_train_bytes"] for r in ok) / 2**30,
                        "vram": statistics.median(gpu) / 2**30 if gpu else None,
                        "tree": (ok[0]["nodes"], ok[0]["depth"], ok[0]["test_accuracy"])})
    return out


def check_same(runs):
    plans = [json.load(open(run / "plan.json")) for run in runs]
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
    args = parser.parse_args()
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
    header = 0.42 + 0.2 * len(runs)  # inches of the hardware lines
    inches = 1.9 + header + 2 * 0.58 * (per_panel + 0.45 * gaps)
    fig, axes = plt.subplots(2, 2, figsize=(13, inches), squeeze=False,
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.12,
                                          "hspace": 0.42 * 8.8 / inches})
    max_time = max(e["time"] for e in entries)
    max_gib = max(max(e["ram"], e["vram"] or 0) for e in entries)
    for row, protocol in enumerate(PROTOCOLS):
        ax_time, ax_mem = axes[row]
        mine = sorted((e for e in entries if e["protocol"] == protocol),
                      key=lambda e: (order.index(e["backend"]), runs.index(e["run"])))
        nodes, depth, accuracy = mine[0]["tree"]
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
                     f"depth {depth}, test accuracy {100 * accuracy:.2f}%",
                     transform=ax_time.transAxes, fontsize=8, color=MUTED, va="bottom")
        ax_mem.set_title("peak memory: RAM (host) and VRAM (GPU)", loc="left", fontsize=10.5,
                         color=INK, fontweight="bold", pad=20)
        times = {(e["run"], e["backend"]): e["time"] for e in mine}
        for y, e in zip(ys, mine):
            run, backend, t = e["run"], e["backend"], e["time"]
            ax_time.barh(y, t, height=0.62, zorder=2, **style(e))
            # Speedup over the same machine's serial run; CUDA also over every
            # machine's parallel run (the fastest CPU result).
            notes = []
            if backend != "serial" and (run, "serial") in times:
                notes.append(f"{times[(run, 'serial')] / t:.1f}× {where(run)} serial")
            if backend == "cuda":
                notes += [f"{times[(other, 'parallel')] / t:.1f}× {where(other)} parallel"
                          for other in reversed(runs) if (other, "parallel") in times]
            ax_time.text(t + max_time * 0.012, y, f"{t:.2f} s", va="center", fontsize=9,
                         color=INK, fontweight="bold")
            ax_time.annotate(" · ".join(notes), (t, y), xytext=(52, 0),
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
                ax_mem.text(e["ram"] + max_gib * 0.015, y - 0.16, f"{e['ram']:.2f} GiB RAM",
                            va="center", fontsize=8.5, color=INK)
                ax_mem.text(e["vram"] + max_gib * 0.015, y + 0.16, f"{e['vram']:.2f} GiB VRAM",
                            va="center", fontsize=8.5, color=INK)
            else:
                ax_mem.barh(y, e["ram"], height=0.62, zorder=2, **style(e))
                ax_mem.text(e["ram"] + max_gib * 0.015, y, f"{e['ram']:.2f} GiB RAM",
                            va="center", fontsize=8.5, color=INK)
        ax_time.set_xlim(0, max_time * 1.6)
        ax_time.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
        ax_mem.set_xlim(0, max_gib * 1.45)
        ax_mem.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} GiB"))

    top = 1 - 0.09 / inches
    fig.suptitle(f"./tree on {NAMES.get(dataset, dataset)} ({meta['n_train']:,} training rows, "
                 f"{len(meta['features'])} features): "
                 + " vs ".join(dict.fromkeys(where(r) for r in reversed(runs))),
                 x=0.012, y=top, ha="left", fontsize=12, color=INK, fontweight="bold")
    # Hardware: one line per machine, the run being reported first.
    for index, run in enumerate(reversed(runs)):
        machine = json.load(open(run / "machine.json"))
        used_gpu = any(e["run"] == run and e["backend"] == "cuda" for e in entries)
        # Only ./tree is drawn: when its cases were rerun later, that is the date.
        started = machine.get("tree_rerun", machine)["started"][:10]
        role = (f"reference: {run.name}, run {started}" if run in args.reference
                else f"{run.name}, run {started}")
        line_y = top - (0.36 + 0.2 * index) / inches
        fig.text(0.012, line_y, f"{where(run).capitalize()}", fontsize=9.2, color=INK,
                 fontweight="bold", va="top")
        fig.text(0.075, line_y, f"{B.hardware_line(machine, used_gpu)}   ({role})",
                 fontsize=9.2, color=INK, va="top")
    handles = [Patch(color=COLORS[b], label=BACKENDS[b]) for b in BACKENDS
               if any(e["backend"] == b for e in entries)]
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
    reps = {json.load(open(r / "machine.json"))["settings"]["reps"] for r in runs}
    runs_text = "one run per case" if reps == {1} else "median over the runs of each case"
    fig.text(LEFT, 0.1 / inches,
             f"Bars: {runs_text}. Time: ./tree's 'train total' (presort, build, prune; CUDA also "
             "upload and device sort), not reading the CSV. Same data files (SHA-256) on every "
             "machine.\nRAM: peak RSS of the process up to the end of training. VRAM: peak GPU "
             "memory during the process above idle, CUDA context included (nvidia-smi, every 10 ms).",
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=LEFT, right=0.97, top=1 - (header + 1.05) / inches,
                        bottom=0.75 / inches)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
