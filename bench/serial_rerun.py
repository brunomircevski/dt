#!/usr/bin/env python3
"""Rerun only ./tree's one-thread (serial) cases of a finished benchmark and put
them in place of the old ones, for when only the serial backend changed.

  bench/.venv/bin/python bench/serial_rerun.py command bench/results/<dir>  # print the run.py command
  bench/.venv/bin/python bench/serial_rerun.py run bench/results/<dir>      # run it here, then merge
  bench/.venv/bin/python bench/serial_rerun.py merge bench/results/<dir>    # merge <dir>/serial-rerun

The rerun takes the run.py command that made ./tree's rows (machine.json:
tree_rerun's command, else the first run's) and changes only --impls tree,
--threads 1, --run-id <dir>/serial-rerun and --no-baseline: the same data,
protocols, CPUs (one thread takes the first one listed), runs per case and
flags. merge checks that the rerun read the same data files (SHA-256) and
built the same trees, replaces the rows of ./tree runs on one thread with its
rows, records it in machine.json (serial_rerun), removes serial-rerun and
rewrites report.md, summary.csv and runs.csv.
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import benchlib as B

NOTE = ("./tree's one-thread (serial) rows rerun after --serial got the parallel backend's "
        "out-of-place partition; the other rows are from the earlier run")


def rerun_args(out):
    """The run.py arguments of the serial rerun of `out`."""
    machine = json.load(open(out / "machine.json"))
    argv = shlex.split(machine.get("tree_rerun", {}).get("command") or machine["command"])
    args = argv[argv.index("bench/run.py") + 1:]

    def put(flag, value):
        if flag in args:
            args[args.index(flag) + 1] = value
        else:
            args.extend([flag, value])
    put("--impls", "tree")
    put("--threads", "1")
    put("--run-id", f"{out.resolve().relative_to(B.RESULTS)}/serial-rerun")
    if "--no-baseline" not in args:
        args.append("--no-baseline")
    return args


def serial(row):
    return row["kind"] == "run" and row["impl"] == "tree" and row["threads"] == 1


def merge(out):
    rerun = out / "serial-rerun"
    old = [json.loads(line) for line in open(out / "results.jsonl")]
    new = [json.loads(line) for line in open(rerun / "results.jsonl")]
    if not new or any(not serial(row) for row in new):
        sys.exit(f"{rerun}: expected only one-thread ./tree runs")
    if any(row["status"] != "ok" for row in new):
        sys.exit(f"{rerun}: some runs failed, nothing merged")
    replaced = [row for row in old if serial(row)]
    key = lambda row: (row["dataset"], row["protocol"])  # noqa: E731
    if {key(row) for row in replaced} != {key(row) for row in new}:
        sys.exit(f"{rerun}: other cases than the old serial ones: "
                 f"{sorted({key(r) for r in new})} vs {sorted({key(r) for r in replaced})}")
    for row in new:  # the same tree as before (and as every other thread count)
        before = {(r["nodes"], r["depth"]) for r in replaced if key(r) == key(row)}
        if (row["nodes"], row["depth"]) not in before:
            sys.exit(f"{rerun}: {key(row)} built another tree: {row['nodes']} nodes, depth "
                     f"{row['depth']}, before {sorted(before)}")
    plan, again = (json.load(open(path / "plan.json")) for path in (out, rerun))
    for dataset, files in again["data_files"].items():
        if dataset != B.BASELINE and plan["data_files"][dataset] != files:
            sys.exit(f"{rerun}: {dataset} is not the same data (SHA-256)")
    with open(out / "results.jsonl", "w") as f:
        for row in [r for r in old if not serial(r)] + new:
            f.write(json.dumps(row) + "\n")
    machine = json.load(open(out / "machine.json"))
    info = json.load(open(rerun / "machine.json"))
    machine["serial_rerun"] = {
        "note": NOTE, "started": info["started"], "command": info["command"],
        "tree_git": info["versions"]["tree_git"], "loadavg_at_start": info["loadavg_at_start"],
        "governors": info["governors"], "reps": info["settings"]["reps"],
        "replaced_rows": len(replaced)}
    json.dump(machine, open(out / "machine.json", "w"), indent=1)
    shutil.rmtree(rerun)
    subprocess.run([sys.executable, str(B.BENCH / "report.py"), str(out), "--md",
                    str(out / "report.md"), "--csv", str(out / "summary.csv"), "--runs-csv",
                    str(out / "runs.csv")], check=True, stdout=subprocess.DEVNULL)
    print(f"{out}: replaced {len(replaced)} serial ./tree rows with {len(new)} new ones")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("command", "run", "merge"))
    parser.add_argument("dir", type=Path, help="a result directory with results.jsonl")
    args = parser.parse_args()
    out = args.dir
    if args.action == "command":
        print(shlex.join(["bench/.venv/bin/python", "bench/run.py", *rerun_args(out)]))
    elif args.action == "run":
        if (out / "serial-rerun").exists():
            sys.exit(f"{out}/serial-rerun exists: merge it or move it away")
        subprocess.run([sys.executable, str(B.BENCH / "run.py"), *rerun_args(out)],
                       cwd=B.ROOT, check=True)
        merge(out)
    else:
        merge(out)


if __name__ == "__main__":
    main()
