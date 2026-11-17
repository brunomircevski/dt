#!/usr/bin/env python3
"""final.sh's helper: keep results that are already there instead of running them again,
and copy runs that one job measured for two benchmarks.

  have_results.py run <dir>/<name> <impls> <protocols> <threads> <runs> <long> [<env>]
      exit 0 if bench/results/<dir>/<name> already holds this plan line's runs, complete
      and current: the same run.py command (impls, protocols, threads, flags), every case
      with <runs> ok runs (<long> if its first run trained for over an hour, run.py
      --long-runs), no failed run, and nothing out of date (benchlib.out_of_date).
  have_results.py gpulibs <results.jsonl> <impls> <runs>
      exit 0 if gpubench's results.jsonl has <runs> current forest-of-10 runs of each impl.
  have_results.py split <vs folder> [<results root>]
      copy ./tree's runs of cluster-cpu-vs-others/vs-epyc-<data> (./tree interleaved with
      the libraries) into cluster-<data>/{simple,serial,cpu}-epyc-<data>, the folders the
      test-set charts and the paper read, so ./tree is measured once. A target is replaced
      only if it is missing, an earlier copy (FROM.txt) or out of date.
"""
import json
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import benchlib as B  # noqa: E402

RESULTS = ROOT / "bench" / "results"
FLAGS = ("--test-accuracy", "--train-accuracy-only", "--hash-trees", "--warm-up", "--profile",
         "--no-eval", "--prefetch")


def rows_of(folder):
    f = folder / "results.jsonl"
    return [json.loads(line) for line in open(f)] if f.exists() else []


def command_args(command):
    """run.py's --impls, --protocols, --threads and flags from machine.json's command."""
    words = shlex.split(command)
    get = lambda key: words[words.index(key) + 1] if key in words else None  # noqa: E731
    return {"impls": get("--impls"), "protocols": get("--protocols"), "threads": get("--threads"),
            "flags": sorted(w for w in words if w in FLAGS)}


def expected_flags(env):
    """The run.py flags job.sh adds for a plan line's environment."""
    values = dict(kv.split("=", 1) for kv in env.split(",") if "=" in kv) if env else {}
    flags = set(values.get("RUN_EXTRA", "").split()) | {values.get("ACCURACY", "--train-accuracy-only")}
    if values.get("WARM_UP"):
        flags.add("--warm-up")
    if values.get("E2E"):
        flags |= {"--prefetch", "--no-eval"}
    return sorted(f for f in flags if f in FLAGS)


def complete(cases, rows, runs, long_runs):
    """Every planned case has its runs, all ok."""
    by_case = defaultdict(list)
    for r in rows:
        if r.get("kind") == "run":
            by_case[(r["protocol"], r["impl"], r["threads"])].append(r)
    for _, protocol, impl, threads in cases:
        got = by_case.get((protocol, impl, threads), [])
        if not got or any(r["status"] != "ok" for r in got):
            return False
        first = got[0].get("train_seconds") or got[0].get("wall_seconds") or 0
        if len(got) < (long_runs if first > 3600 else runs):
            return False
    return True


def have_run(target, impls, protocols, threads, runs, long_runs, env=""):
    folder = RESULTS / target
    machine_f, plan_f = folder / "machine.json", folder / "plan.json"
    if not (machine_f.exists() and plan_f.exists() and (folder / "report.md").exists()):
        return False
    machine, plan = json.loads(machine_f.read_text()), json.loads(plan_f.read_text())
    args = command_args(machine.get("command", ""))
    if (args["impls"] != impls or args["protocols"] != protocols or args["threads"] != threads
            or args["flags"] != expected_flags(env)):
        return False
    rows = rows_of(folder)
    return complete(plan["cases"], rows, runs, long_runs) and not B.rows_out_of_date(rows, machine)


def have_gpulibs(results, impls, runs):
    f = Path(results)
    if not f.exists():
        return False
    rows = [json.loads(line) for line in open(f)]
    forest = [r for r in rows if r.get("trees") == 10 and not r.get("warm") and "startup_seconds" in r]
    return all(len([r for r in forest if r["impl"] == impl]) >= runs
               and not B.rows_out_of_date([r for r in forest if r["impl"] == impl])
               for impl in impls.split(","))


def split(vs, root=RESULTS):
    """./tree's runs of a vs folder -> the per-dataset folders (see the module text)."""
    vs = Path(vs)
    machine = json.loads((vs / "machine.json").read_text())
    plan = json.loads((vs / "plan.json").read_text())
    rows = rows_of(vs)
    if not rows:
        return
    dataset = plan["cases"][0][0]
    top = max(c[3] for c in plan["cases"])
    targets = {("tree_simple", 1): "simple", ("tree", 1): "serial"}
    if top > 1:
        targets[("tree", top)] = "cpu"
    folder = root / B_dir(dataset)
    for (impl, threads), backend in targets.items():
        mine = [r for r in rows if r["impl"] == impl and (r["threads"] == threads or r.get("kind") != "run")]
        runs_ = [r for r in mine if r.get("kind") == "run"]
        if not runs_:
            continue
        target = folder / f"{backend}-epyc-{dataset}"
        if target.exists() and not (target / "FROM.txt").exists():
            old_rows = rows_of(target)
            old_machine = json.loads((target / "machine.json").read_text()) if (target / "machine.json").exists() else {}
            if old_rows and not B.rows_out_of_date(old_rows, old_machine):
                print(f"kept {target.relative_to(root)}: current results of its own (not replaced)")
                continue
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        cases = [c for c in plan["cases"] if c[2] == impl and c[3] == threads]
        (target / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mine))
        (target / "plan.json").write_text(json.dumps({**plan, "cases": cases, "impls": [impl]}, indent=1))
        (target / "machine.json").write_text(json.dumps({**machine, "copied_from": str(vs.relative_to(root))},
                                                        indent=1))
        (target / "FROM.txt").write_text(
            f"Copied from {vs.relative_to(root)} by bench/cluster/have_results.py split: ./tree {impl}, "
            f"{threads} thread(s), measured there interleaved with the libraries (one job, a new shuffled "
            "order each repetition). Not a run of its own.\n")
        subprocess.run([sys.executable, str(ROOT / "bench" / "report.py"), str(target), "--md",
                        str(target / "report.md"), "--csv", str(target / "summary.csv"),
                        "--runs-csv", str(target / "runs.csv")], check=True, stdout=subprocess.DEVNULL)
        print(f"copied {vs.relative_to(root)} ({impl}, {threads} thread(s)) -> {target.relative_to(root)}")


def B_dir(dataset):
    for prefix in ("ecbdl", "sorel"):
        if dataset.startswith(prefix):
            return f"cluster-{prefix}"
    return "cluster-chess" if dataset.startswith("chess_") else f"cluster-{dataset}"


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "run":
        target, impls, protocols, threads, runs, long_runs, *env = rest
        sys.exit(0 if have_run(target, impls, protocols, threads, int(runs), int(long_runs),
                               env[0] if env else "") else 1)
    if mode == "gpulibs":
        results, impls, runs = rest
        sys.exit(0 if have_gpulibs(results, impls, int(runs)) else 1)
    if mode == "split":
        for vs in rest[:1]:
            split(vs, Path(rest[1]) if len(rest) > 1 else RESULTS)
        sys.exit(0)
    sys.exit(__doc__)
