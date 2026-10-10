#!/usr/bin/env python3
"""./tree on several machines: serial, parallel and (where there is a GPU) CUDA
training time and peak memory, CART and C4.5 one above the other, whatever
backends the runs hold. Colour follows the backend; the bars of runs given with
--copied (numbers taken over from an earlier benchmark, not rerun) are hatched
and labelled with the run they come from.

Refuses to draw when the runs did not read the same data (SHA-256 of the
prepared files in plan.json) or did not build the same trees.

  bench/.venv/bin/python bench/machines_chart.py bench/results/cuda-legion-susy \\
      --copied bench/results/cpu-pc-susy --out bench/results/cuda-legion-susy/machines.png
  bench/.venv/bin/python bench/machines_chart.py bench/results/cpu-pc-higgs \
      bench/results/cuda-legion-higgs --out bench/results/cuda-legion-higgs/machines.png
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

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


def machine(run):
    info = json.load(open(run / "machine.json"))
    cpu = re.search(r"Model name:\s+(.*)", info["lscpu"]).group(1)
    cpu = re.sub(r"\(R\)|\(TM\)|Intel|Core|\d+th Gen|\s+CPU.*", "", cpu).split()[-1]
    gpu = info["versions"].get("gpu") or ""
    gpu = gpu.split(",")[0].replace("NVIDIA GeForce ", "") if "unavailable" not in gpu else ""
    return cpu, gpu


def bars(run, copied):
    """-> [(protocol, backend, row label, copied, times, rss, gpu, tree, run)] of ./tree."""
    rows = [json.loads(line) for line in open(run / "results.jsonl")]
    cpu, gpu = machine(run)
    out = []
    for protocol in PROTOCOLS:
        for impl, threads_wanted in (("tree", "1"), ("tree", "n"), ("tree_cuda", "n")):
            ok = [r for r in rows if r["kind"] == "run" and r["status"] == "ok"
                  and r["protocol"] == protocol and r["impl"] == impl
                  and (r["threads"] == 1) == (threads_wanted == "1")]
            if not ok:
                continue
            threads = ok[0]["threads"]
            backend = "cuda" if impl == "tree_cuda" else "serial" if threads == 1 else "parallel"
            label = (f"{gpu} + {threads} CPU threads" if backend == "cuda"
                     else f"{cpu}, {threads} thread{'s' if threads > 1 else ''}")
            out.append((protocol, backend, label, copied,
                        [r["train_seconds"] for r in ok], [r["peak_rss_train_bytes"] for r in ok],
                        [r["gpu_peak_bytes"] for r in ok if r.get("gpu_peak_bytes") is not None],
                        (ok[0]["nodes"], ok[0]["depth"], ok[0]["test_accuracy"]), run))
    return out


def check_same(runs):
    plans = [json.load(open(run / "plan.json")) for run in runs]
    datasets = {d for plan in plans for d in plan["data_files"] if d != "_baseline"}
    if len(datasets) != 1:
        raise SystemExit(f"the runs used different datasets: {sorted(datasets)}")
    dataset = datasets.pop()
    for name in DATA_FILES:
        hashes = {plan["data_files"][dataset][name] for plan in plans}
        if len(hashes) != 1:
            raise SystemExit(f"{dataset}/{name} differs between the runs: not the same data")
    return dataset, plans[0]["datasets"][dataset]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--copied", nargs="*", type=Path, default=[],
                        help="runs whose numbers are copied from an earlier benchmark")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = list(args.copied) + list(args.runs)
    dataset, meta = check_same(runs)
    entries = [b for run in runs for b in bars(run, run in args.copied)]
    for protocol in PROTOCOLS:
        trees = {e[7][:2] for e in entries if e[0] == protocol}
        if len(trees) > 1:
            raise SystemExit(f"{protocol}: different trees {sorted(trees)}")

    order = list(BACKENDS)
    per_panel = max(sum(e[0] == p for e in entries) for p in PROTOCOLS)
    gaps = max(len({e[1] for e in entries if e[0] == p}) for p in PROTOCOLS) - 1
    inches = 1.9 + 2 * 0.58 * (per_panel + 0.45 * gaps)
    fig, axes = plt.subplots(2, 2, figsize=(13, inches), squeeze=False,
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.12,
                                          "hspace": 0.42 * 8.8 / inches})
    max_time = max(statistics.median(e[4]) for e in entries)
    max_gib = max(max(statistics.median(e[5]), statistics.median(e[6]) if e[6] else 0)
                  for e in entries) / 2**30
    for row, protocol in enumerate(PROTOCOLS):
        ax_time, ax_mem = axes[row]
        mine = sorted((e for e in entries if e[0] == protocol),
                      key=lambda e: (order.index(e[1]), runs.index(e[8])))
        nodes, depth, accuracy = mine[0][7]
        y, ys, previous = 0.0, [], None
        for e in mine:
            if previous is not None and e[1] != previous:
                y += 0.45
            ys.append(y)
            y += 1
            previous = e[1]
        height = y
        for ax in (ax_time, ax_mem):
            ax.set_ylim(height - 0.35, -0.65)
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
        ax_mem.set_title("peak memory", loc="left", fontsize=10.5, color=INK, fontweight="bold",
                         pad=20)
        # Speedup over the same machine's serial run, or for CUDA without one,
        # over its parallel run.
        reference = {(e[8], e[1]): statistics.median(e[4]) for e in mine}
        for y, e in zip(ys, mine):
            _, backend, label, copied, times, rss, gpu, _, run = e
            color = COLORS[backend]
            style = dict(facecolor="white", edgecolor=color, hatch="////", linewidth=1.2) \
                if copied else dict(color=color)
            t = statistics.median(times)
            ax_time.barh(y, t, height=0.62, zorder=2, **style)
            base = "serial" if (run, "serial") in reference else "parallel"
            note = (f"   {reference[(run, base)] / t:.1f}× {base}"
                    if backend not in ("serial", base) and (run, base) in reference else "")
            ax_time.text(t + max_time * 0.012, y, f"{t:.2f} s", va="center", fontsize=9,
                         color=INK, fontweight="bold")
            ax_time.text(t + max_time * 0.012, y, f"{' ' * 11}{note}", va="center",
                         fontsize=8, color=MUTED)
            ax_time.text(-0.015, y - 0.14, f"{BACKENDS[backend]} · {where(run)}",
                         transform=ax_time.get_yaxis_transform(), ha="right", va="center",
                         fontsize=9.5, color=INK, fontweight="bold")
            sub = label + (f" · copied from {run.name}" if copied else "")
            ax_time.text(-0.015, y + 0.24, sub, transform=ax_time.get_yaxis_transform(),
                         ha="right", va="center", fontsize=7.8, color=MUTED,
                         style="italic" if copied else "normal")
            gib = statistics.median(rss) / 2**30
            if gpu:
                ax_mem.barh(y - 0.16, gib, height=0.3, zorder=2, **style)
                vram = statistics.median(gpu) / 2**30
                ax_mem.barh(y + 0.16, vram, height=0.3, zorder=2, facecolor="white",
                            edgecolor=color, linewidth=1.2)
                ax_mem.text(gib + max_gib * 0.015, y - 0.16, f"{gib:.2f} GiB RAM", va="center",
                            fontsize=8.5, color=INK)
                ax_mem.text(vram + max_gib * 0.015, y + 0.16, f"{vram:.2f} GiB GPU", va="center",
                            fontsize=8.5, color=INK)
            else:
                ax_mem.barh(y, gib, height=0.62, zorder=2, **style)
                ax_mem.text(gib + max_gib * 0.015, y, f"{gib:.2f} GiB RAM", va="center",
                            fontsize=8.5, color=INK)
        ax_time.set_xlim(0, max_time * 1.32)
        ax_time.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
        ax_mem.set_xlim(0, max_gib * 1.45)
        ax_mem.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} GiB"))

    shown = [b for b in BACKENDS if any(e[1] == b for e in entries)]
    handles = [Patch(color=COLORS[b], label=BACKENDS[b]) for b in shown]
    if args.copied:
        handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch="////", label=", ".join(
            f"{where(r)}: copied from {r.name}" for r in args.copied) + ", not rerun"))
    if any(e[6] for e in entries):
        handles.append(Patch(facecolor="white", edgecolor=MUTED, label="GPU memory (CUDA)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(LEFT - 0.005, 1 - 0.33 / inches),
               ncol=len(handles), frameon=False, fontsize=8.8, handletextpad=0.4,
               columnspacing=1.3)
    machines = " vs ".join(dict.fromkeys(where(r) for r in reversed(runs)))
    fig.suptitle(f"./tree on {NAMES.get(dataset, dataset)} ({meta['n_train']:,} training rows, "
                 f"{len(meta['features'])} features): {machines}",
                 x=LEFT, y=1 - 0.09 / inches, ha="left", fontsize=11.5, color=INK,
                 fontweight="bold")
    reps = {json.load(open(r / "machine.json"))["settings"]["reps"] for r in runs}
    runs_text = "one run per case" if reps == {1} else "median over the runs of each case"
    fig.text(LEFT, 0.1 / inches,
             f"Bars: {runs_text}. Time: ./tree's 'train total' (presort, build, prune; CUDA also "
             "upload and device sort), not reading the CSV. Same data files (SHA-256) on both "
             "machines.\nRAM: peak RSS up to the end of training. GPU: peak device memory "
             "during the process above idle, CUDA context included (nvidia-smi, every 10 ms).",
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=LEFT, right=0.985, top=1 - 1.23 / inches, bottom=0.75 / inches)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
