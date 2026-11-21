#!/usr/bin/env python3
"""Merge a run measured in parts (several SLURM jobs, each under a few hours, so a
long benchmark does not hold a node for a day) into one run folder.

  merge_parts.py <run folder> [--impls a,b,...] [--protocols a,b,...] [--keep-parts]

The parts are <run folder>/parts/p1, p2, ... (run.py --run-id <dir>/<name>/parts/pN),
each a run.py of its own with some of the repetitions (and maybe some of the
tools: J48's 4.7 h runs on MNIST8M are parts of their own). Rows already in the
run folder (an older run of the same command, topped up) stay first, and so do
the parts its machine.json lists (name a top-up's parts after them, e.g. p7 on). The
merged folder is what one run.py would have written: results.jsonl (rows in the
order measured, run_id the folder, rep renumbered per case), plan.json (every
case), machine.json (the first part's, its command with every tool, the total
repetitions and the folder as run id; "parts" lists each part's host, start,
job and command), run.log (the parts' logs one after the other), then
report.md, summary.csv and runs.csv from report.py. The parts are removed
unless --keep-parts.
"""
import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "bench"


def rows_of(folder):
    f = folder / "results.jsonl"
    return [json.loads(line) for line in open(f)] if f.exists() else []


def set_arg(words, key, value):
    if key in words:
        words[words.index(key) + 1] = value
    else:
        words += [key, value]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--impls", help="the merged command's --impls (default: every part's, in order)")
    ap.add_argument("--protocols", help="the merged command's --protocols (default: every part's, in order)")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()
    target = Path(args.folder).resolve()
    run_id = str(target.relative_to(RESULTS))
    parts = sorted((p for p in (target / "parts").glob("p*") if p.is_dir()),
                   key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    if not parts:
        sys.exit(f"no parts in {target}/parts")
    for p in parts:
        if not (p / "machine.json").exists() or not rows_of(p):
            sys.exit(f"{p}: no results (still running or failed?)")

    old_rows = rows_of(target)
    rows, impls, protocols, cases, reps = list(old_rows), [], [], [], defaultdict(int)
    old_machine = json.loads((target / "machine.json").read_text()) if (target / "machine.json").exists() else None
    old_plan = json.loads((target / "plan.json").read_text()) if (target / "plan.json").exists() else None
    for c in (old_plan or {}).get("cases", []):
        cases.append(c)
    info = []
    for p in parts:
        machine = json.loads((p / "machine.json").read_text())
        plan = json.loads((p / "plan.json").read_text())
        words = shlex.split(machine.get("command", ""))
        for i in (words[words.index("--impls") + 1].split(",") if "--impls" in words else []):
            if i not in impls:
                impls.append(i)
        for i in (words[words.index("--protocols") + 1].split(",") if "--protocols" in words else []):
            if i not in protocols:
                protocols.append(i)
        for c in plan["cases"]:
            if c not in cases:
                cases.append(c)
        prows = rows_of(p)
        rows += prows
        jobs = sorted({r.get("slurm_job") for r in prows if r.get("slurm_job")})
        info.append({"part": p.name, "host": machine.get("host"), "started": machine.get("started"),
                     "command": machine.get("command"), "runs": sum(r.get("kind") == "run" for r in prows),
                     **({"jobs": jobs} if jobs else {})})
    # one rep sequence per case, in the order measured
    for r in rows:
        r["run_id"] = run_id
        key = (r.get("kind"), r.get("protocol"), r.get("impl"), r.get("threads"))
        r["rep"] = reps[key]
        reps[key] += 1
    total = max((n for (kind, *_), n in reps.items() if kind == "run"), default=0)

    first = json.loads((parts[0] / "machine.json").read_text())
    words = shlex.split(first.get("command", ""))
    set_arg(words, "--impls", args.impls or ",".join(impls))
    if args.protocols or protocols:
        set_arg(words, "--protocols", args.protocols or ",".join(protocols))
    set_arg(words, "--reps", str(total))
    set_arg(words, "--run-id", run_id)
    machine = {**first, "command": shlex.join(words), "parts": info}
    if old_machine:  # a top-up: the parts merged before stay listed, else the old run is one "before" part
        machine["parts"] = (old_machine.get("parts")
                            or [{"part": "before", "host": old_machine.get("host"),
                                 "started": old_machine.get("started"), "command": old_machine.get("command"),
                                 "runs": sum(r.get("kind") == "run" for r in old_rows)}]) + info
    plan = {**json.loads((parts[0] / "plan.json").read_text()), "cases": cases}

    (target / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (target / "machine.json").write_text(json.dumps(machine, indent=1))
    (target / "plan.json").write_text(json.dumps(plan, indent=1))
    log = (target / "run.log").read_text() if (target / "run.log").exists() and old_rows else ""
    for p in parts:
        if (p / "run.log").exists():
            log += f"\n##### part {p.name} ({json.loads((p / 'machine.json').read_text()).get('host')})\n"
            log += (p / "run.log").read_text()
        if (p / "profiles").is_dir():
            (target / "profiles").mkdir(exist_ok=True)
            for f in (p / "profiles").iterdir():
                shutil.copy2(f, target / "profiles" / f"{p.name}-{f.name}")
    (target / "run.log").write_text(log)
    subprocess.run([sys.executable, str(ROOT / "bench" / "tools" / "report.py"), str(target), "--md",
                    str(target / "report.md"), "--csv", str(target / "summary.csv"),
                    "--runs-csv", str(target / "runs.csv")], check=True, stdout=subprocess.DEVNULL)
    if not args.keep_parts:
        shutil.rmtree(target / "parts")
    print(f"{run_id}: {len(parts)} parts, {sum(i['runs'] for i in info)} runs merged"
          f"{f' after {len(old_rows)} rows already there' if old_rows else ''}; {total} runs per case at most")


if __name__ == "__main__":
    main()
