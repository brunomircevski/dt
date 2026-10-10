#!/usr/bin/env python3
"""Summarise benchmark runs: Markdown tables and CSV files.

  bench/.venv/bin/python bench/report.py                      # the latest run
  bench/.venv/bin/python bench/report.py bench/results/*      # several runs (machines)
  bench/.venv/bin/python bench/report.py RUN --md out.md --csv summary.csv --runs-csv runs.csv

Per case (dataset, protocol, implementation, threads):
- training time: median and min-max over the runs, and the ratio to ./tree
  with the same number of threads;
- peak RSS up to the end of training (median over the same runs), and that
  peak above the tool's runtime footprint (its median peak RSS on the
  _baseline dataset);
- nodes, leaves, depth, training and test accuracy, reported by every run.
  They must be identical in every repetition; a case where they are not is
  marked "varies".
Single-thread and multi-thread cases go to separate tables.

--csv writes one row per case, --runs-csv one row per process (the raw data).
"""

import argparse
import csv
import io
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import benchlib as B

TREE_FIELDS = ("nodes", "leaves", "depth", "train_accuracy", "test_accuracy")


def median(values):
    return statistics.median(values) if values else None


def summarise(run_dir):
    rows = [json.loads(line) for line in open(run_dir / "results.jsonl")]
    metas = json.load(open(run_dir / "plan.json"))["datasets"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["protocol"], row["impl"], row["threads"], row["kind"])].append(row)

    baseline = {}
    for (dataset, protocol, impl, threads, kind), group in groups.items():
        if kind == "baseline":
            ok = [r for r in group if r["status"] == "ok"]
            baseline[impl] = median([r["peak_rss_bytes"] for r in ok])

    cases = {}
    for (dataset, protocol, impl, threads, kind), group in groups.items():
        if kind == "baseline":
            continue
        case = cases.setdefault((dataset, protocol, impl, threads), {
            "run": run_dir.name, "dataset": dataset, "protocol": protocol, "impl": impl,
            "threads": threads, "status": "ok", "consistent": True})
        bad = [r["status"] for r in group if r["status"] != "ok"]
        if bad:
            case["status"] = bad[0]
        ok = [r for r in group if r["status"] == "ok"]
        if kind == "run" and ok:
            times = [r["train_seconds"] for r in ok]
            case.update({k: ok[0].get(k) for k in TREE_FIELDS})
            if len({tuple(r.get(k) for k in TREE_FIELDS) for r in ok}) > 1:
                case["consistent"] = False
            case.update(time_median=median(times), time_min=min(times), time_max=max(times),
                        time_mean=statistics.fmean(times),
                        time_stdev=statistics.stdev(times) if len(times) > 1 else None,
                        time_reps=len(times), wall_median=median([r["wall_seconds"] for r in ok]))
            accuracy, n_test = case.get("test_accuracy"), metas[dataset]["n_test"]
            if accuracy is not None:
                case["test_accuracy_ci95"] = 1.96 * math.sqrt(accuracy * (1 - accuracy) / n_test)
            rss = [r["peak_rss_train_bytes"] for r in ok]
            base_rss = baseline.get(impl)
            case.update(peak_rss=median(rss), peak_rss_min=min(rss), peak_rss_max=max(rss),
                        peak_rss_above_baseline=median(rss) - base_rss if base_rss else None)
            gpu = [r["gpu_peak_bytes"] for r in ok if r.get("gpu_peak_bytes") is not None]
            if gpu:
                case["gpu_peak"] = median(gpu)
    for case in cases.values():
        reference = cases.get((case["dataset"], case["protocol"], "tree", case["threads"]), {})
        if case.get("time_median") and reference.get("time_median"):
            case["ratio_to_tree"] = case["time_median"] / reference["time_median"]
    return cases, baseline


def fmt_seconds(seconds):
    if seconds is None:
        return ""
    if seconds < 1e-2:
        return f"{seconds * 1e3:.2g} ms"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.3g} s"


def fmt_bytes(count):
    if count is None:
        return ""
    if abs(count) < 2**19:
        return "0 MiB"
    if abs(count) < 2**30:
        return f"{count / 2**20:.0f} MiB"
    return f"{count / 2**30:.2f} GiB"


def fmt_percent(value):
    return f"{100 * value:.2f}%" if value is not None else ""


def machine_line(run_dir):
    info = json.load(open(run_dir / "machine.json"))
    model = re.search(r"Model name:\s+(.*)", info.get("lscpu", ""))
    memory = f"{info['mem_total_kb'] / 2**20:.0f} GiB" if info.get("mem_total_kb") else "?"
    settings = info["settings"]
    return (f"**{run_dir.name}**: {model.group(1) if model else '?'}, {info['logical_cpus']} "
            f"logical CPUs (P-cores {info.get('p_cores')}, E-cores {info.get('e_cores')}), "
            f"{memory}, {info.get('kernel')}, governor {', '.join(info.get('governors', []))}, "
            f"turbo {'off' if info.get('intel_no_turbo') == '1' else 'on'}, "
            f"./tree {info['versions'].get('tree_git')}, CPUs in order of use "
            f"{settings.get('cpus') or 'none'}, {settings.get('reps')} run(s) per case, warm-up "
            f"{settings.get('warmup_rows')} rows")


def table(cases):
    out = io.StringIO()
    out.write("| Implementation | Train time (median) | min–max | × ./tree | Peak RSS "
              "| RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |\n"
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    order = list(B.IMPLS)
    for c in sorted(cases, key=lambda c: order.index(c["impl"])):
        label = B.IMPLS[c["impl"]]["label"]
        if not c.get("time_median"):
            out.write(f"| {label} | {c['status']} |" + " |" * 9 + "\n")
            continue
        flags = [] if c["status"] == "ok" else [c["status"]]
        if not c["consistent"]:
            flags.append("varies")
        span = f"{fmt_seconds(c['time_min'])}–{fmt_seconds(c['time_max'])}"
        ratio = f"{c['ratio_to_tree']:.2f}×" if c.get("ratio_to_tree") else ""
        out.write(" | ".join([
            f"| {label}" + (f" ({', '.join(flags)})" if flags else ""),
            fmt_seconds(c["time_median"]), span, ratio, fmt_bytes(c.get("peak_rss"))
            + (f" (+ GPU {fmt_bytes(c['gpu_peak'])})" if c.get("gpu_peak") is not None else ""),
            fmt_bytes(c.get("peak_rss_above_baseline")),
            f"{c['nodes']:,}" if c.get("nodes") is not None else "",
            f"{c['leaves']:,}" if c.get("leaves") is not None else "",
            str(c.get("depth", "")), fmt_percent(c.get("train_accuracy")),
            fmt_percent(c.get("test_accuracy"))]) + " |\n")
    return out.getvalue()


def markdown(run_dirs):
    out = io.StringIO()
    for run_dir in run_dirs:
        cases, baseline = summarise(run_dir)
        out.write(f"## {run_dir.name}\n\n{machine_line(run_dir)}\n\n")
        out.write("Runtime footprint (peak RSS on the 200-row _baseline dataset): " + ", ".join(
            f"{B.IMPLS[impl]['label']} {fmt_bytes(rss)}"
            for impl, rss in sorted(baseline.items())) + "\n")
        metas = json.load(open(run_dir / "plan.json"))["datasets"]
        for dataset in dict.fromkeys(case[0] for case in cases):
            meta = metas[dataset]
            out.write(f"\n### {dataset} ({meta['n_train']:,} train / {meta['n_test']:,} test rows, "
                      f"{len(meta['features'])} features, {len(meta['classes'])} classes)\n")
            for protocol in B.PROTOCOLS:
                mine = [c for key, c in cases.items() if key[0] == dataset and key[1] == protocol]
                for threads in sorted({c["threads"] for c in mine}):
                    rows = [c for c in mine if c["threads"] == threads]
                    heading = "1 thread" if threads == 1 else f"{threads} threads"
                    out.write(f"\n**{B.PROTOCOLS[protocol]['title']}** (`{protocol}`), "
                              f"{heading}\n\n{table(rows)}")
            half_widths = [c["test_accuracy_ci95"] for key, c in cases.items()
                           if key[0] == dataset and c.get("test_accuracy_ci95")]
            if half_widths:
                out.write(f"\nTest accuracy is measured on {meta['n_test']:,} rows: its 95% "
                          f"interval is ±{100 * max(half_widths):.2f} percentage points at most.\n")
    return out.getvalue()


FIELDS = ["run", "dataset", "protocol", "impl", "threads", "status", "consistent",
          "time_median", "time_min", "time_max", "time_mean", "time_stdev", "time_reps",
          "ratio_to_tree", "wall_median", "peak_rss", "peak_rss_min", "peak_rss_max",
          "peak_rss_above_baseline", "nodes", "leaves", "depth", "train_accuracy",
          "test_accuracy", "test_accuracy_ci95", "gpu_peak"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path, help="run directories (default: the latest)")
    parser.add_argument("--md", type=Path, help="write the Markdown here instead of stdout")
    parser.add_argument("--csv", type=Path, help="also write one row per case here")
    parser.add_argument("--runs-csv", type=Path, help="also write one row per process here")
    args = parser.parse_args()
    runs = args.runs or sorted((p for p in B.RESULTS.glob("*") if (p / "results.jsonl").exists()),
                               key=lambda p: p.stat().st_mtime)[-1:]
    if not runs:
        sys.exit("no results in bench/results/")
    text = markdown(runs)
    if args.md:
        args.md.write_text(text)
    else:
        print(text)
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, FIELDS, extrasaction="ignore")
            writer.writeheader()
            for run in runs:
                for case in summarise(run)[0].values():
                    writer.writerow(case)
    if args.runs_csv:
        rows = [json.loads(line) for run in runs for line in open(run / "results.jsonl")]
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with open(args.runs_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
