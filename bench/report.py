#!/usr/bin/env python3
"""Summarise benchmark runs: Markdown tables and one CSV row per case.

  bench/.venv/bin/python bench/report.py                      # the latest run
  bench/.venv/bin/python bench/report.py bench/results/*      # several runs (machines)
  bench/.venv/bin/python bench/report.py RUN --md out.md --csv out.csv

Per case: training time as median [min-max] over the timed repetitions, the
ratio to ./tree on 1 thread (same machine, dataset, protocol), peak RSS and
peak anonymous RSS (medians of the memory repetitions), peak RSS above the
tool's runtime footprint (its median peak RSS on the _baseline dataset), and
the tree size and test accuracy of the check run.
"""

import argparse
import csv
import io
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import benchlib as B


def median(values):
    return statistics.median(values) if values else None


def summarise(run_dir):
    rows = [json.loads(line) for line in open(run_dir / "results.jsonl")]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["protocol"], row["impl"], row["threads"], row["kind"])].append(row)

    baseline = {}
    for (dataset, protocol, impl, threads, kind), group in groups.items():
        if kind == "baseline":
            ok = [r for r in group if r["status"] == "ok"]
            baseline[impl] = (median([r["peak_rss_bytes"] for r in ok]),
                              median([r["peak_anon_bytes"] for r in ok if r["peak_anon_bytes"]]))

    cases = {}
    for (dataset, protocol, impl, threads, kind), group in groups.items():
        if kind == "baseline":
            continue
        case = cases.setdefault((dataset, protocol, impl, threads), {
            "run": run_dir.name, "dataset": dataset, "protocol": protocol, "impl": impl,
            "threads": threads, "status": "ok"})
        bad = [r["status"] for r in group if r["status"] != "ok"]
        if bad:
            case["status"] = bad[0]
        ok = [r for r in group if r["status"] == "ok"]
        if kind == "check" and ok:
            case.update({k: ok[0].get(k) for k in ("nodes", "leaves", "depth", "test_accuracy")})
        elif kind == "time" and ok:
            times = [r["train_seconds"] for r in ok]
            case.update(time_median=median(times), time_min=min(times), time_max=max(times),
                        time_reps=len(times), wall_median=median([r["wall_seconds"] for r in ok]))
        elif kind == "memory" and ok:
            rss = median([r["peak_rss_bytes"] for r in ok])
            anon = median([r["peak_anon_bytes"] for r in ok if r["peak_anon_bytes"]])
            base_rss, base_anon = baseline.get(impl, (None, None))
            case.update(peak_rss=rss, peak_anon=anon,
                        peak_rss_above_baseline=rss - base_rss if base_rss else None,
                        peak_anon_above_baseline=anon - base_anon if anon and base_anon else None)
    for case in cases.values():
        reference = cases.get((case["dataset"], case["protocol"], "tree", 1), {})
        if case.get("time_median") and reference.get("time_median"):
            case["ratio_to_tree_1"] = case["time_median"] / reference["time_median"]
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
    if abs(count) < 2**30:
        return f"{count / 2**20:.0f} MiB"
    return f"{count / 2**30:.2f} GiB"


def machine_line(run_dir):
    info = json.load(open(run_dir / "machine.json"))
    model = re.search(r"Model name:\s+(.*)", info.get("lscpu", ""))
    memory = f"{info['mem_total_kb'] / 2**20:.0f} GiB" if info.get("mem_total_kb") else "?"
    return (f"**{run_dir.name}**: {model.group(1) if model else '?'}, {info['logical_cpus']} "
            f"logical CPUs (P-cores {info.get('p_cores')}, E-cores {info.get('e_cores')}), "
            f"{memory}, {info.get('kernel')}, governor {', '.join(info.get('governors', []))}, "
            f"./tree {info['versions'].get('tree_git')}, pinned to CPUs "
            f"{info['settings'].get('cpus') or 'none'}")


def markdown(run_dirs):
    out = io.StringIO()
    for run_dir in run_dirs:
        cases, baseline = summarise(run_dir)
        out.write(f"## {run_dir.name}\n\n{machine_line(run_dir)}\n\n")
        out.write("Runtime footprint (_baseline dataset, peak RSS / anon): " + ", ".join(
            f"{B.IMPLS[impl]['label']} {fmt_bytes(rss)} / {fmt_bytes(anon)}"
            for impl, (rss, anon) in sorted(baseline.items())) + "\n")
        metas = json.load(open(run_dir / "plan.json"))["datasets"]
        for dataset in dict.fromkeys(case[0] for case in cases):
            meta = metas[dataset]
            out.write(f"\n### {dataset} ({meta['n_train']:,} train / {meta['n_test']:,} test rows, "
                      f"{len(meta['features'])} features, {len(meta['classes'])} classes)\n")
            for protocol in B.PROTOCOLS:
                rows = [c for key, c in cases.items() if key[0] == dataset and key[1] == protocol]
                if not rows:
                    continue
                out.write(f"\n**{B.PROTOCOLS[protocol]['title']}** (`{protocol}`)\n\n")
                out.write("| Implementation | Threads | Train time (median) | min–max | vs ./tree 1 thr "
                          "| Peak RSS | Peak anon | RSS − footprint | Nodes | Leaves | Depth "
                          "| Test acc. |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                order = list(B.IMPLS)
                for c in sorted(rows, key=lambda c: (order.index(c["impl"]), c["threads"])):
                    if c["status"] != "ok" and not c.get("time_median"):
                        out.write(f"| {B.IMPLS[c['impl']]['label']} | {c['threads']} | "
                                  f"{c['status']} |" + " |" * 9 + "\n")
                        continue
                    span = (f"{fmt_seconds(c.get('time_min'))}–{fmt_seconds(c.get('time_max'))}"
                            if c.get("time_min") is not None else "")
                    ratio = f"{c['ratio_to_tree_1']:.2f}×" if c.get("ratio_to_tree_1") else ""
                    accuracy = (f"{100 * c['test_accuracy']:.2f}%"
                                if c.get("test_accuracy") is not None else "")
                    out.write(" | ".join([
                        f"| {B.IMPLS[c['impl']]['label']}", str(c["threads"]),
                        fmt_seconds(c.get("time_median")), span, ratio,
                        fmt_bytes(c.get("peak_rss")), fmt_bytes(c.get("peak_anon")),
                        fmt_bytes(c.get("peak_rss_above_baseline")),
                        f"{c['nodes']:,}" if c.get("nodes") is not None else "",
                        f"{c['leaves']:,}" if c.get("leaves") is not None else "",
                        str(c.get("depth", "")), accuracy]) + " |\n")
    return out.getvalue()


FIELDS = ["run", "dataset", "protocol", "impl", "threads", "status", "time_median", "time_min",
          "time_max", "time_reps", "wall_median", "ratio_to_tree_1", "peak_rss", "peak_anon",
          "peak_rss_above_baseline", "peak_anon_above_baseline", "nodes", "leaves", "depth",
          "test_accuracy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path, help="run directories (default: the latest)")
    parser.add_argument("--md", type=Path, help="write the Markdown here instead of stdout")
    parser.add_argument("--csv", type=Path, help="also write one row per case here")
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


if __name__ == "__main__":
    main()
