#!/usr/bin/env python3
"""Merge extra runs of the same case into one run folder (2026-10-28 top-ups).

  bench/.venv/bin/python bench/merge_runs.py bench/results/cluster-higgs \
      bench/results/cluster-higgs-new bench/results/cluster-higgs-top

For each run folder of the extra benchmark folders (in order), its results.jsonl
rows are appended to <benchmark>/<run>/results.jsonl (the runs keep their own
"time" and "run_id"; "rep" is renumbered per case so it stays unique), its
logs go to <run>/logs/<batch>-<log name> (the batch: the folder's machine.json
start date and host), its machine.json becomes <run>/machine.json when it is
the newest (the older ones stay in logs/), and report.py redraws report.md,
summary.csv and runs.csv. A folder with no results.jsonl (a job that never
started) is skipped. The extra folders are left in place; delete them after.
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


def batch(run):
    m = json.loads((run / "machine.json").read_text())
    return f"{m['started'][:16].replace(':', '').replace('T', '-')}-{m['host']}", m["started"]


def stash(run):
    """Move a folder's own logs and machine.json into logs/ under its batch name (once)."""
    if not (run / "machine.json").exists() or (run / "logs").exists():
        return
    name, _ = batch(run)
    logs = run / "logs"
    logs.mkdir(exist_ok=True)
    for log in sorted(run.glob("run*.log")):
        shutil.move(log, logs / f"{name}-{log.name}")
    shutil.copy(run / "machine.json", logs / f"{name}-machine.json")


def merge(dest, extra):
    if not (extra / "results.jsonl").exists():
        print(f"  skip {extra} (no results)")
        return
    dest.mkdir(exist_ok=True)
    stash(dest)
    rows = [json.loads(l) for l in (dest / "results.jsonl").read_text().splitlines()] \
        if (dest / "results.jsonl").exists() else []
    added = [json.loads(l) for l in (extra / "results.jsonl").read_text().splitlines()]
    reps = Counter()
    out = []
    for row in rows + added:
        if row.get("kind") == "run":
            key = (row["dataset"], row["protocol"], row["impl"], row.get("threads"))
            row["rep"] = reps[key]
            reps[key] += 1
        out.append(json.dumps(row))
    (dest / "results.jsonl").write_text("\n".join(out) + "\n")
    name, started = batch(extra)
    logs = dest / "logs"
    logs.mkdir(exist_ok=True)
    for log in sorted(extra.glob("run*.log")):
        shutil.copy(log, logs / f"{name}-{log.name}")
    shutil.copy(extra / "machine.json", logs / f"{name}-machine.json")
    newest = not (dest / "machine.json").exists() or \
        json.loads((dest / "machine.json").read_text())["started"] < started
    if newest:
        shutil.copy(extra / "machine.json", dest / "machine.json")
        shutil.copy(extra / "plan.json", dest / "plan.json")
    print(f"  {dest.name}: +{len(added)} rows from {extra.parent.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("extra", nargs="+", type=Path)
    args = parser.parse_args()
    touched = []
    for extra in args.extra:
        if not extra.is_dir():
            print(f"  skip {extra} (no such folder)")
            continue
        for run in sorted(p for p in extra.iterdir() if p.is_dir()):
            merge(args.benchmark / run.name, run)
            touched.append(args.benchmark / run.name)
    report = Path(__file__).with_name("report.py")
    for run in dict.fromkeys(touched):
        if (run / "results.jsonl").exists():
            subprocess.run([sys.executable, report, run, "--md", run / "report.md", "--csv", run / "summary.csv",
                            "--runs-csv", run / "runs.csv"], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
