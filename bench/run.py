#!/usr/bin/env python3
"""Run the CPU benchmark: every (dataset, protocol, implementation, threads) case,
timed and memory-measured, one fresh process per measurement.

  bench/.venv/bin/python bench/run.py --dry-run                 # show the plan only
  bench/.venv/bin/python bench/run.py covertype_10k susy_50k    # run these datasets
  bench/.venv/bin/python bench/run.py --protocols c45 --impls tree,yadt --threads 1,2,4,8,all

Every case grows one tree (no cross-validation). cart_alpha prunes at one fixed
alpha (benchlib.CART_ALPHA, or --alpha) given unchanged to every CART tool.

Before anything runs, every prepared file of the datasets is hashed (SHA-256)
and every CSV row is checked against the binary files (plan.json records the
hashes), so every tool provably reads the same rows.

Phases (see bench/README.md for why):
  time       --reps processes per case: training time measured inside the tool
             (after the managed runtimes' untimed warm-up fit); after the timer
             stops, the same process reports tree size and training and test
             accuracy
  memory     --mem-reps processes per case, no warm-up, no evaluation: peak memory
             (their training time is the cold-start time, for an appendix)
  baseline   --mem-reps processes per implementation on the tiny _baseline
             dataset: the tool's runtime footprint
Time and memory repetitions run in a shuffled order (a new order every
repetition), so slow drift (heat, background load) spreads over all cases.
Every process reports the rows and features it loaded; a mismatch with the
dataset is an error.

Results go to bench/results/<run-id>/results.jsonl, with machine.json (hardware,
OS, versions) and plan.json next to it; bench/report.py turns them into tables.
"""

import argparse
import datetime
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import benchlib as B

PYTHON = sys.executable
ADAPTERS = B.BENCH / "adapters"


# ------------------------------------------------------------------ measuring

def process_tree(pid):
    """pid and all its descendants (Linux /proc)."""
    pids, todo = [], [pid]
    while todo:
        current = todo.pop()
        pids.append(current)
        try:
            for task in os.listdir(f"/proc/{current}/task"):
                with open(f"/proc/{current}/task/{task}/children") as handle:
                    todo.extend(int(child) for child in handle.read().split())
        except OSError:
            pass
    return pids


def anon_rss_bytes(pids):
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as handle:
                for line in handle:
                    if line.startswith("RssAnon:"):
                        total += int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    return total


def measure(command, env, timeout, sample_seconds, pin=()):
    """Run `command` to completion and measure it from the outside.

    The command runs under the tiny bench/.tools/rusage launcher (pinned with
    `pin`, e.g. taskset): Linux starts a child's peak RSS at the peak RSS of
    the process it was forked from, so started straight from this (large)
    Python process every tool would report at least this process's peak.

    peak_rss_bytes   the kernel's high-water mark of the process's resident set
                     (ru_maxrss from the launcher's wait4, the number GNU time's
                     %M prints): exact, no sampling, no overhead. For a process
                     that waited for children it is the largest of them, not the
                     sum.
    peak_anon_bytes  the largest sum of RssAnon over the process tree, sampled
                     every sample_seconds: memory the tools allocated, without
                     file-backed pages (memory-mapped input files, program and
                     library code). Sampling can miss a peak shorter than the
                     interval.
    wall_seconds     from fork to exit, including start-up, loading, everything.
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, \
            tempfile.TemporaryDirectory() as folder:
        usage_file = Path(folder) / "rusage"
        start = time.perf_counter()
        process = subprocess.Popen(list(pin) + [str(B.RUSAGE), str(usage_file)] + list(command),
                                   stdout=out, stderr=err, env=env, start_new_session=True)
        done = threading.Event()
        peak_anon = [0]

        def sample():
            while not done.wait(sample_seconds):
                peak_anon[0] = max(peak_anon[0], anon_rss_bytes(process_tree(process.pid)))

        sampler = threading.Thread(target=sample, daemon=True) if sample_seconds > 0 else None
        if sampler:
            sampler.start()
        timed_out = threading.Event()

        def kill():
            timed_out.set()
            os.killpg(process.pid, signal.SIGKILL)

        timer = threading.Timer(timeout, kill)
        timer.start()
        _, status, _ = os.wait4(process.pid, 0)
        wall = time.perf_counter() - start
        timer.cancel()
        done.set()
        if sampler:
            sampler.join()
        process.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        usage = usage_file.read_text().split() if usage_file.exists() else None  # None: killed
        return {"returncode": process.returncode, "timed_out": timed_out.is_set(),
                "stdout": out.read().decode(errors="replace"),
                "stderr": err.read().decode(errors="replace"),
                "wall_seconds": wall,
                "peak_rss_bytes": int(usage[0]) * 1024 if usage else None,
                "peak_anon_bytes": peak_anon[0] or None,
                "user_seconds": float(usage[1]) if usage else None,
                "system_seconds": float(usage[2]) if usage else None}


# ------------------------------------------------------------------ commands

def adapter_args(dataset, threads, warmup_rows, evaluate):
    m = B.meta(dataset)
    return [f"data={B.DATA / dataset}", f"n_train={m['n_train']}", f"n_test={m['n_test']}",
            f"n_features={len(m['features'])}", f"n_classes={len(m['classes'])}",
            f"threads={threads}", f"warmup={warmup_rows}", f"eval={int(evaluate)}"]


def fill(value, alpha):
    return value.format(alpha=repr(alpha)) if isinstance(value, str) and "{" in value else value


def yadt_accuracy(report, title):
    """Accuracy from the confusion matrix YaDT prints after `title` (its error
    percentage is rounded): rows are actual classes, columns predicted ones,
    both led by "?". -> (accuracy, number of rows)."""
    lines = report.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    columns = lines[start + 1].split("\t")[:-1]
    correct = total = 0.0
    for line in lines[start + 2:]:
        cells = line.split("\t")
        if len(cells) != len(columns) + 1:
            break
        counts = [float(cell) for cell in cells[:-1]]
        total += sum(counts)
        if cells[-1] in columns:
            correct += counts[columns.index(cells[-1])]
    return correct / total, int(total)


def build_command(case, kind, args, scratch):
    """-> (argv, extra env, parser of stdout -> result dict)."""
    dataset, protocol, impl, threads = case
    spec = B.PROTOCOLS[protocol]["impls"][impl]
    evaluate = kind == "time"
    m = B.meta(dataset)
    warmup_rows = 0
    if kind == "time" and B.IMPLS[impl]["warmup"]:
        warmup_rows = m["n_train"] if args.warmup_rows == "all" else int(args.warmup_rows)
    data = B.DATA / dataset

    if impl == "tree":
        flags = [fill(flag, args.alpha) for flag in spec]
        backend = ["--serial"] if threads == 1 else ["--parallel", "--threads", str(threads)]
        dump = scratch / "tree.dump"
        argv = [str(B.TREE)] + backend + flags + (["--dump", str(dump)] if evaluate else []) \
            + [str(data / "train.csv")]

        def parse(stdout):
            nodes, leaves, depth = re.search(r"nodes (\d+), leaves (\d+), depth (\d+)",
                                             stdout).groups()
            rows, features = re.search(r"rows (\d+) train, (\d+) features", stdout).groups()
            result = {"train_seconds": float(re.search(r"train total\s+([\d.]+) ms",
                                                       stdout).group(1)) / 1000,
                      "nodes": int(nodes), "leaves": int(leaves), "depth": int(depth),
                      "n_train_loaded": int(rows), "n_features_loaded": int(features)}
            if evaluate:
                # Both accuracies from the dumped tree, so ./tree is measured like the
                # others: by predicting every row. ./tree's own training accuracy
                # (printed to 4 decimals) must agree.
                tree = B.parse_tree_dump(dump, len(m["features"]))
                for part in ("train", "test"):
                    X, y = B.load(dataset, part)
                    result[f"{part}_accuracy"] = float(np.mean(B.predict_tree(tree, X) == y))
                printed = float(re.search(r"train accuracy\s+([\d.]+)%", stdout).group(1))
                if abs(printed - 100 * result["train_accuracy"]) > 1e-4:
                    raise ValueError(f"dump predicts {100 * result['train_accuracy']:.6f}% on "
                                     f"the training rows, ./tree printed {printed}%")
            return result
        return argv, {}, parse

    if impl == "yadt":
        report = scratch / "yadt.txt"
        argv = [str(B.YADT), "-fm", str(data / "yadt.names"), "-fd", str(data / "train.yadt.csv"),
                "-tt", str(threads)] + list(spec)
        if evaluate:
            report.unlink(missing_ok=True)  # never read a previous run's report
            argv += ["-ft", str(data / "test.yadt.csv"), "-t", str(report)]

        def parse(stdout):
            # "load time" is parsing the CSV; the rest of "total time" is YaDT's
            # indexing of the data, its counterpart of ./tree's presort: timed.
            load = float(re.search(r"load time: ([\d.e+-]+) secs", stdout).group(1))
            total = float(re.search(r"total time: ([\d.e+-]+) secs", stdout).group(1))
            steps = re.findall(r"size: (\d+) depth: (\d+) nf: \d+ time: ([\d.e+-]+) secs", stdout)
            features, rows = re.search(r"pred\. atts: (\d+) classes: \d+ .*rows: (\d+)",
                                       stdout).groups()
            nodes = int(steps[-1][0])
            result = {"train_seconds": total - load + sum(float(s[2]) for s in steps),
                      "nodes": nodes, "leaves": (nodes + 1) // 2,  # binary splits only
                      "depth": int(steps[-1][1]),
                      "n_train_loaded": int(rows), "n_features_loaded": int(features)}
            if evaluate:
                text = report.read_text()
                for part, title, expected in (
                        ("train", "MISCLASSIFICATION on training", m["n_train"]),
                        ("test", "MISCLASSIFICATION on test file", m["n_test"])):
                    accuracy, counted = yadt_accuracy(text, title)
                    if counted != expected:
                        raise ValueError(f"YaDT classified {counted} {part} rows, "
                                         f"expected {expected}")
                    result[f"{part}_accuracy"] = accuracy
            return result
        return argv, {"LD_LIBRARY_PATH": str(B.YADT.parent)}, parse

    params = [f"{key}={fill(value, args.alpha)}" for key, value in spec.items()]
    common = adapter_args(dataset, threads, warmup_rows, evaluate)
    if impl == "sklearn":
        argv = [PYTHON, str(ADAPTERS / "sklearn_fit.py")] + common + params
    elif impl == "rpart":
        argv = ["Rscript", "--vanilla", str(ADAPTERS / "rpart_fit.R")] + common + params
    elif impl == "j48":
        argv = ["java", f"-Xmx{args.java_heap}", "-cp", B.JAVA_CLASSPATH, "J48Fit"] + common + params
    else:
        raise ValueError(impl)
    return argv, {}, lambda stdout: json.loads(stdout.strip().splitlines()[-1])


def base_env():
    env = dict(os.environ)
    # No library may add hidden threads; each tool gets its thread count explicitly.
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               LC_ALL="C", PYTHONHASHSEED="0")
    return env


# ------------------------------------------------------------------ machine

def command_output(argv, **kwargs):
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60, **kwargs)
        return (out.stdout + out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def machine_info(args):
    cpus = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"),
                  key=lambda p: int(p.name[3:]))
    governors = {read_text(c / "cpufreq/scaling_governor") for c in cpus} - {None}
    meminfo = read_text("/proc/meminfo") or ""
    import sklearn
    return {
        "host": socket.gethostname(),
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": shlex.join([sys.executable] + sys.argv),
        "lscpu": command_output(["lscpu"]),
        "logical_cpus": os.cpu_count(),
        # Intel hybrid CPUs: performance (cpu_core) and efficiency (cpu_atom) cores.
        "p_cores": read_text("/sys/devices/cpu_core/cpus"),
        "e_cores": read_text("/sys/devices/cpu_atom/cpus"),
        "smt_active": read_text("/sys/devices/system/cpu/smt/active"),
        "governors": sorted(governors),
        "energy_performance_preference": sorted(
            {read_text(c / "cpufreq/energy_performance_preference") for c in cpus} - {None}),
        "intel_no_turbo": read_text("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "cpufreq_boost": read_text("/sys/devices/system/cpu/cpufreq/boost"),
        "loadavg_at_start": read_text("/proc/loadavg"),
        "mem_total_kb": int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1)) if meminfo else None,
        "kernel": command_output(["uname", "-srvm"]),
        "os": command_output(["sh", "-c", ". /etc/os-release && echo $PRETTY_NAME"]),
        "versions": {
            "tree_git": command_output(["git", "-C", str(B.ROOT), "describe", "--always",
                                        "--dirty", "--abbrev=12"]),
            "tree_binary": str(B.TREE),
            "cxx": command_output(["g++", "--version"]).splitlines()[0],
            "cxxflags": next((line for line in (B.ROOT / "Makefile").read_text().splitlines()
                              if line.startswith("CXXFLAGS")), None),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "R": command_output(["Rscript", "-e", "cat(R.version.string)"]),
            "rpart": command_output(["Rscript", "-e", "cat(as.character(packageVersion('rpart')))"]),
            "java": command_output(["java", "-version"]).splitlines()[0],
            "weka": command_output(["unzip", "-p", str(B.TOOLS / "weka.jar"),
                                    "META-INF/maven/nz.ac.waikato.cms.weka/weka-stable/pom.properties"]),
            "yadt": command_output([str(B.YADT)], env={"LD_LIBRARY_PATH": str(B.YADT.parent)})
            .splitlines()[0] if B.YADT.exists() else None,
        },
        "settings": {"alpha": args.alpha, "warmup_rows": args.warmup_rows,
                     "java_heap": args.java_heap, "cpus": args.cpus, "reps": args.reps,
                     "mem_reps": args.mem_reps, "timeout": args.timeout,
                     "sample_ms": args.sample_ms, "cooldown": args.cooldown},
    }


# ------------------------------------------------------------------ plan and run

def parse_cpus(text):
    cpus = []
    for part in text.split(","):
        if "-" in part:
            first, last = map(int, part.split("-"))
            cpus.extend(range(first, last + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def plan_cases(args):
    datasets = args.datasets or [d.name for d in sorted(B.DATA.iterdir())
                                 if (d / "meta.json").exists() and d.name != B.BASELINE]
    protocols = args.protocols.split(",") if args.protocols else list(B.PROTOCOLS)
    impls = args.impls.split(",") if args.impls else list(B.IMPLS)
    all_threads = len(args.cpu_list) if args.cpu_list else os.cpu_count()
    threads = sorted({all_threads if t == "all" else int(t) for t in args.threads.split(",")})
    cases = []
    for dataset in datasets:
        if not (B.DATA / dataset / "meta.json").exists():
            sys.exit(f"{dataset}: not prepared (bench/prepare.py {dataset})")
        for protocol in protocols:
            for impl in B.PROTOCOLS[protocol]["impls"]:
                if impl not in impls:
                    continue
                for t in threads:
                    if t == 1 or B.IMPLS[impl]["threads"] == "multi":
                        cases.append((dataset, protocol, impl, t))
    return cases


# Which protocol measures each tool's runtime footprint on the _baseline dataset.
BASELINE_PROTOCOL = {"tree": "cart_full", "sklearn": "cart_full", "rpart": "cart_full",
                     "j48": "c45", "yadt": "c45"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*", help="default: every prepared dataset")
    parser.add_argument("--protocols", help=f"comma list (default all: {','.join(B.PROTOCOLS)})")
    parser.add_argument("--impls", help=f"comma list (default all: {','.join(B.IMPLS)})")
    parser.add_argument("--threads", default="1,all",
                        help="thread counts for multi-threaded tools, e.g. 1,2,4,8,all")
    parser.add_argument("--alpha", type=float, default=B.CART_ALPHA,
                        help=f"fixed alpha of cart_alpha, the same for every CART tool "
                        f"(default {B.CART_ALPHA})")
    parser.add_argument("--cpus", help="logical CPUs to pin to, in order of use, e.g. 0,2,4,6 "
                        "or 0-11; N threads use the first N (default: no pinning)")
    parser.add_argument("--reps", type=int, default=5, help="timed processes per case")
    parser.add_argument("--mem-reps", type=int, default=5, help="memory processes per case")
    parser.add_argument("--warmup-rows", default=str(B.WARMUP_ROWS),
                        help="rows of the untimed warm-up fit of the managed runtimes "
                        f"(Python, R, JVM); 'all' = the whole training set, 0 = none "
                        f"(default {B.WARMUP_ROWS})")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per process")
    parser.add_argument("--sample-ms", type=float, default=5,
                        help="RssAnon sampling interval (0 = off)")
    parser.add_argument("--cooldown", type=float, default=0, help="seconds of rest between processes")
    parser.add_argument("--java-heap", default=None, help="JVM -Xmx (default: half of RAM)")
    parser.add_argument("--run-id", help="results go to bench/results/<run-id> (default host-date)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the shuffled order")
    parser.add_argument("--skip", default="", help="phases to skip: comma list of "
                        "time,memory,baseline")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()
    args.cpu_list = parse_cpus(args.cpus) if args.cpus else None
    if not args.java_heap:
        total_kb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1))
        args.java_heap = f"{max(1, total_kb // 2 // 1024 // 1024)}g"
    skip = set(filter(None, args.skip.split(",")))
    if skip - {"time", "memory", "baseline"}:
        sys.exit(f"--skip: unknown phase {', '.join(sorted(skip - {'time', 'memory', 'baseline'}))}")
    if args.warmup_rows != "all" and not args.warmup_rows.isdigit():
        sys.exit("--warmup-rows: a number of rows or 'all'")

    cases = plan_cases(args)
    impls_used = sorted({case[2] for case in cases})
    processes = len(cases) * (args.reps * ("time" not in skip)
                              + args.mem_reps * ("memory" not in skip)) \
        + len(impls_used) * args.mem_reps * ("baseline" not in skip)
    print(f"{len(cases)} cases, {processes} processes:")
    for case in cases:
        print("  " + " | ".join(map(str, case)))
    if args.dry_run:
        return
    if args.cpu_list and max(case[3] for case in cases) > len(args.cpu_list):
        sys.exit("--cpus lists fewer CPUs than the largest thread count")
    if not args.cpu_list:
        print("warning: no --cpus: single-thread runs may land on any core (see README)")
    if not B.RUSAGE.exists():
        sys.exit(f"{B.RUSAGE} missing: run bench/setup.sh")

    run_id = args.run_id or f"{socket.gethostname()}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    out_dir = B.RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(machine_info(args), open(out_dir / "machine.json", "w"), indent=1)
    datasets = sorted({case[0] for case in cases} | {B.BASELINE})
    print("checking the data (SHA-256 of every file, every CSV row against the binary files):")
    data_files = {}
    for dataset in datasets:
        start = time.perf_counter()
        data_files[dataset] = B.verify_dataset(dataset)
        print(f"  {dataset}: identical in every format ({time.perf_counter() - start:.0f} s)")
    json.dump({"cases": cases, "protocols": B.PROTOCOLS, "impls": B.IMPLS,
               "warmup_rows": args.warmup_rows,
               "datasets": {d: B.meta(d) for d in datasets}, "data_files": data_files},
              open(out_dir / "plan.json", "w"), indent=1)
    results = open(out_dir / "results.jsonl", "a")

    failed = set()  # cases that timed out or failed: their other runs are skipped
    env = base_env()
    scratch = Path(tempfile.mkdtemp(prefix="bench-"))

    def execute(case, kind, rep):
        if case in failed:
            return
        dataset, protocol, impl, threads = case
        argv, extra_env, parse = build_command(case, kind, args, scratch)
        pin = ["taskset", "-c", ",".join(map(str, args.cpu_list[:threads]))] if args.cpu_list else []
        # The RssAnon sampler polls /proc from this (unpinned) process, so it is
        # left out of the timed runs, where it could steal cycles.
        sampling = args.sample_ms / 1000 if kind != "time" else 0
        run = measure(argv, {**env, **extra_env}, args.timeout, sampling, pin)
        row = {"run_id": run_id, "dataset": dataset, "protocol": protocol, "impl": impl,
               "threads": threads, "kind": kind, "rep": rep,
               "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               **{k: run[k] for k in ("wall_seconds", "peak_rss_bytes", "peak_anon_bytes",
                                      "user_seconds", "system_seconds")}}
        if run["timed_out"]:
            row["status"] = "timeout"
        elif run["returncode"] != 0:
            row["status"] = "error"
            row["error"] = (run["stderr"] or run["stdout"]).strip()[-2000:]
        else:
            try:
                row.update(parse(run["stdout"]))
                row["status"] = "ok"
            except Exception as error:  # noqa: BLE001 - keep the run, record why
                row["status"] = "error"
                row["error"] = f"parse: {error!r}: {run['stdout'][-1000:]}"
            m = B.meta(dataset)
            loaded = (row.get("n_train_loaded"), row.get("n_features_loaded"))
            if row["status"] == "ok" and loaded != (m["n_train"], len(m["features"])):
                row["status"] = "error"
                row["error"] = (f"loaded {loaded[0]} rows x {loaded[1]} features, the dataset "
                                f"has {m['n_train']} x {len(m['features'])}")
        if row["status"] != "ok" and kind != "baseline":
            failed.add(case)
        row["command"] = shlex.join(pin + argv)
        results.write(json.dumps(row) + "\n")
        results.flush()
        shown = f"{row.get('train_seconds', float('nan')):.4g} s" if row["status"] == "ok" \
            else row["status"]
        peak = f"{row['peak_rss_bytes'] / 2**20:.0f} MiB" if row["peak_rss_bytes"] else "?"
        print(f"  {kind:8} {rep} | {' | '.join(map(str, case))}: {shown}, peak {peak}",
              flush=True)
        if args.cooldown:
            time.sleep(args.cooldown)

    rng = random.Random(args.seed)
    for kind, reps in (("time", args.reps), ("memory", args.mem_reps)):
        if kind in skip:
            continue
        print(f"{kind}:")
        for rep in range(reps):
            order = list(cases)
            rng.shuffle(order)
            for case in order:
                execute(case, kind, rep)
    if "baseline" not in skip:
        print("baseline:")
        for rep in range(args.mem_reps):
            for impl in impls_used:
                execute((B.BASELINE, BASELINE_PROTOCOL[impl], impl, 1), "baseline", rep)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"results: {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
