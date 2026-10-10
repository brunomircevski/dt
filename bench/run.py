#!/usr/bin/env python3
"""Run the CPU benchmark: every (dataset, protocol, implementation, threads) case,
--reps fresh processes each; every process gives training time, peak memory,
tree size and accuracy.

  bench/.venv/bin/python bench/run.py --dry-run                 # show the plan only
  bench/.venv/bin/python bench/run.py covertype_10k susy_50k    # run these datasets
  bench/.venv/bin/python bench/run.py --protocols c45 --impls tree,yadt --threads 1,2,4,8,all

Every case grows one tree (no cross-validation). cart_alpha prunes at one fixed
alpha (benchlib.CART_ALPHA, or --alpha) given unchanged to every CART tool.

Before anything runs, every prepared file of the datasets is hashed (SHA-256)
and every CSV row is checked against the binary files (plan.json records the
hashes), so every tool provably reads the same rows.

Each run (see bench/README.md for why):
  1. the managed runtimes (Python, R, JVM) fit one untimed warm-up tree;
  2. the tool times the training itself (data already in memory);
  3. right after it, the peak resident memory so far is read (VmHWM): load,
     warm-up and training, before evaluation allocates anything. YaDT, a closed
     binary, cannot do that, so its measured process only trains (and saves
     the tree); its peak is the process's ru_maxrss, and a second, unmeasured
     YaDT process classifies the test rows with the saved tree;
  4. tree size and training and test accuracy are recorded.
The repetitions run in a shuffled order (a new order every repetition), so slow
drift (heat, background load) spreads over all cases. Every process reports the
rows and features it loaded; a mismatch with the dataset is an error. After the
runs, each tool runs --reps times on the tiny _baseline dataset (no warm-up, no
evaluation): its runtime footprint.

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
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import benchlib as B

PYTHON = sys.executable
ADAPTERS = B.BENCH / "adapters"


# ------------------------------------------------------------------ measuring

def gpu_memory_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                          "--id=0"], capture_output=True, text=True, timeout=30, check=True)
    return int(out.stdout.split()[0])


def measure(command, env, timeout, pin=(), gpu=False):
    """Run `command` to completion and measure it from the outside.

    The command runs under the tiny bench/.tools/rusage launcher (pinned with
    `pin`, e.g. taskset): Linux starts a child's peak RSS at the peak RSS of
    the process it was forked from, so started straight from this (large)
    Python process every tool would report at least this process's peak.

    peak_rss_bytes   the kernel's high-water mark of the process's resident set
                     (ru_maxrss from the launcher's wait4, the number GNU time's
                     %M prints), over the whole process.
    wall_seconds     from fork to exit, including start-up, loading, everything.
    gpu_peak_bytes   (gpu=True) the most GPU memory in use while the process ran,
                     minus what was in use just before it started: nvidia-smi
                     samples the whole device every 10 ms, so nothing else may
                     use the GPU during the run.
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, \
            tempfile.TemporaryDirectory() as folder:
        usage_file = Path(folder) / "rusage"
        sampler = None
        if gpu:
            idle_mib = gpu_memory_used_mib()
            sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits", "--id=0", "-lms", "10"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        start = time.perf_counter()
        # Its own session, so a timeout (or Ctrl+C here) kills the whole tool.
        process = subprocess.Popen(list(pin) + [str(B.RUSAGE), str(usage_file)] + list(command),
                                   stdout=out, stderr=err, env=env, start_new_session=True)
        timed_out = threading.Event()

        def kill():
            timed_out.set()
            os.killpg(process.pid, signal.SIGKILL)

        timer = threading.Timer(timeout, kill)
        timer.start()
        try:
            _, status, _ = os.wait4(process.pid, 0)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            raise
        finally:
            timer.cancel()
            if sampler:
                sampler.terminate()
        wall = time.perf_counter() - start
        gpu_peak = None
        if sampler:
            samples = [int(line) for line in sampler.communicate()[0].split() if line.isdigit()]
            gpu_peak = (max(samples) - idle_mib) * 2**20 if samples else None
        process.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        usage = usage_file.read_text().split() if usage_file.exists() else None  # None: killed
        return {"returncode": process.returncode, "timed_out": timed_out.is_set(),
                "stdout": out.read().decode(errors="replace"),
                "stderr": err.read().decode(errors="replace"),
                "wall_seconds": wall,
                "peak_rss_bytes": int(usage[0]) * 1024 if usage else None,
                "user_seconds": float(usage[1]) if usage else None,
                "system_seconds": float(usage[2]) if usage else None,
                **({"gpu_peak_bytes": gpu_peak} if gpu else {})}


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
    evaluate = kind == "run"
    m = B.meta(dataset)
    warmup_rows = 0
    if kind == "run" and B.IMPLS[impl]["warmup"]:
        warmup_rows = m["n_train"] if args.warmup_rows == "all" else int(args.warmup_rows)
    data = B.DATA / dataset

    if impl in ("tree", "tree_cuda"):
        flags = [fill(flag, args.alpha) for flag in spec]
        if impl == "tree_cuda":
            binary, backend = B.TREE_CUDA, ["--cuda", "--threads", str(threads)]
        else:
            binary = B.TREE
            backend = ["--serial"] if threads == 1 else ["--parallel", "--threads", str(threads)]
        dump = scratch / "tree.dump"
        argv = [str(binary)] + backend + flags + (["--dump", str(dump)] if evaluate else []) \
            + [str(data / "train.csv")]

        def parse(stdout):
            nodes, leaves, depth = re.search(r"nodes (\d+), leaves (\d+), depth (\d+)",
                                             stdout).groups()
            rows, features = re.search(r"rows (\d+) train, (\d+) features", stdout).groups()
            peak = int(re.search(r"peak after training (\d+) KiB", stdout).group(1))
            result = {"train_seconds": float(re.search(r"train total\s+([\d.]+) ms",
                                                       stdout).group(1)) / 1000,
                      "nodes": int(nodes), "leaves": int(leaves), "depth": int(depth),
                      "n_train_loaded": int(rows), "n_features_loaded": int(features),
                      "peak_rss_train_bytes": peak * 1024}
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
        # The measured process only trains: it saves the tree and reports the
        # training accuracy; the test rows are classified afterwards by a second,
        # unmeasured process that loads the saved tree, so they never add to the
        # training process's peak memory.
        report, test_report, saved = (scratch / "yadt.txt", scratch / "yadt_test.txt",
                                      scratch / "yadt.tree")
        argv = [str(B.YADT), "-fm", str(data / "yadt.names"), "-fd", str(data / "train.yadt.csv"),
                "-tt", str(threads)] + list(spec)
        if evaluate:
            for path in (report, test_report, saved):
                path.unlink(missing_ok=True)  # never read a previous run's output
            argv += ["-tb", str(saved), "-t", str(report)]
        environment = {"LD_LIBRARY_PATH": str(B.YADT.parent)}

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
                subprocess.run([str(B.YADT), "-bt", str(saved), "-ft", str(data / "test.yadt.csv"),
                                "-t", str(test_report)], env={**os.environ, **environment},
                               capture_output=True, check=True, timeout=args.timeout)
                for part, text, title, expected in (
                        ("train", report.read_text(), "MISCLASSIFICATION on training",
                         m["n_train"]),
                        ("test", test_report.read_text(), "MISCLASSIFICATION on test file",
                         m["n_test"])):
                    accuracy, counted = yadt_accuracy(text, title)
                    # YaDT prints counts with 6 significant digits (2.14332e+06),
                    # so above a million rows the matrix sum is only that exact.
                    if abs(counted - expected) > max(1, 1e-5 * expected):
                        raise ValueError(f"YaDT classified {counted} {part} rows, "
                                         f"expected {expected}")
                    result[f"{part}_accuracy"] = accuracy
            return result
        return argv, environment, parse

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
            "tree_cuda_binary": str(B.TREE_CUDA),
            "nvcc": command_output([os.environ.get("NVCC", "/opt/cuda/bin/nvcc"), "--version"])
            .splitlines()[-1],
            "gpu": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,"
                                   "power.limit,clocks.max.sm,pstate",
                                   "--format=csv,noheader"]),
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
                     "timeout": args.timeout, "cooldown": args.cooldown},
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
                    mode = B.IMPLS[impl]["threads"]
                    if (mode == "multi" or (mode == "single" and t == 1)
                            or (mode == "all" and t == max(threads))):
                        cases.append((dataset, protocol, impl, t))
    return cases


# Which protocol measures each tool's runtime footprint on the _baseline dataset.
BASELINE_PROTOCOL = {"tree": "cart_full", "tree_cuda": "cart_full", "sklearn": "cart_full",
                     "rpart": "cart_full",
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
    parser.add_argument("--reps", type=int, default=1, help="runs (fresh processes) per case")
    parser.add_argument("--warmup-rows", default=str(B.WARMUP_ROWS),
                        help="rows of the untimed warm-up fit of the managed runtimes "
                        f"(Python, R, JVM); 'all' = the whole training set, 0 = none "
                        f"(default {B.WARMUP_ROWS})")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per process")
    parser.add_argument("--cooldown", type=float, default=0, help="seconds of rest between processes")
    parser.add_argument("--java-heap", default=None, help="JVM -Xmx (default: half of RAM)")
    parser.add_argument("--run-id", help="results go to bench/results/<run-id> (default host-date)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the shuffled order")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the runtime-footprint runs on the _baseline dataset")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()
    args.cpu_list = parse_cpus(args.cpus) if args.cpus else None
    if not args.java_heap:
        total_kb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1))
        args.java_heap = f"{max(1, total_kb // 2 // 1024 // 1024)}g"
    if args.warmup_rows != "all" and not args.warmup_rows.isdigit():
        sys.exit("--warmup-rows: a number of rows or 'all'")

    cases = plan_cases(args)
    impls_used = sorted({case[2] for case in cases})
    processes = (len(cases) + len(impls_used) * (not args.no_baseline)) * args.reps
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
    total = len(cases) * args.reps + (0 if args.no_baseline else len(impls_used) * args.reps)
    progress = {"done": 0, "started": time.perf_counter()}
    walls = defaultdict(list)  # case -> whole-process seconds, for the time left

    def clock(seconds):
        seconds = int(seconds)
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 \
            else f"{seconds // 60}m{seconds % 60:02d}s"

    def describe(case):
        dataset, protocol, impl, threads = case
        algorithm = "CART" if protocol.startswith("cart") else "C4.5"
        return (f"{algorithm} {B.IMPLS[impl]['label']}, {threads} thread"
                f"{'s' if threads > 1 else ''}, {dataset}")

    def time_left(pending):
        """Seconds still to go, from the measured runs of the same cases (None
        while some pending case has not run yet)."""
        if any(not walls[case] for case in pending):
            return None
        return sum(statistics.median(walls[case]) for case in pending)

    def execute(case, kind, rep, pending):
        if case in failed:
            progress["done"] += 1
            print(f"[{progress['done']}/{total}] skipped (failed before): {describe(case)}")
            return
        dataset, protocol, impl, threads = case
        progress["done"] += 1
        print(f"[{progress['done']}/{total}] {datetime.datetime.now():%H:%M:%S} "
              f"{'run ' + str(rep + 1) + '/' + str(args.reps) if kind == 'run' else 'footprint'}"
              f": {describe(case)} ...", flush=True)
        argv, extra_env, parse = build_command(case, kind, args, scratch)
        pin = ["taskset", "-c", ",".join(map(str, args.cpu_list[:threads]))] if args.cpu_list else []
        run = measure(argv, {**env, **extra_env}, args.timeout, pin, gpu=impl == "tree_cuda")
        row = {"run_id": run_id, "dataset": dataset, "protocol": protocol, "impl": impl,
               "threads": threads, "kind": kind, "rep": rep,
               "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               **{k: run[k] for k in ("wall_seconds", "peak_rss_bytes", "user_seconds",
                                      "system_seconds", "gpu_peak_bytes") if k in run}}
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
        if row["status"] == "ok" and "peak_rss_train_bytes" not in row:
            row["peak_rss_train_bytes"] = row["peak_rss_bytes"]  # YaDT: the process only trains
        if row["status"] != "ok" and kind != "baseline":
            failed.add(case)
        row["command"] = shlex.join(pin + argv)
        results.write(json.dumps(row) + "\n")
        results.flush()
        walls[case].append(run["wall_seconds"])
        elapsed = time.perf_counter() - progress["started"]
        left = time_left(pending)
        eta = f", about {clock(left)} left" if left is not None else ""
        if row["status"] == "ok":
            accuracy = (f", test acc {100 * row['test_accuracy']:.2f}%"
                        if row.get("test_accuracy") is not None else "")
            gpu = (f", GPU {row['gpu_peak_bytes'] / 2**20:,.0f} MiB"
                   if row.get("gpu_peak_bytes") is not None else "")
            print(f"    ok: train {row['train_seconds']:.4g} s, peak "
                  f"{row['peak_rss_train_bytes'] / 2**20:,.0f} MiB{gpu}, {row['nodes']:,} nodes, depth "
                  f"{row['depth']}{accuracy} | process {clock(run['wall_seconds'])}, "
                  f"elapsed {clock(elapsed)}{eta}", flush=True)
        else:
            reason = row.get("error", "").strip().splitlines()
            print(f"    {row['status'].upper()}: {reason[0][:300] if reason else ''}\n"
                  f"    (the other runs of this case are skipped; details in results.jsonl) "
                  f"| elapsed {clock(elapsed)}", flush=True)
        if args.cooldown:
            time.sleep(args.cooldown)

    def stop(signum, frame):  # `kill` stops the run like Ctrl+C: measure() kills the tool
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)

    rng = random.Random(args.seed)
    print(f"{len(cases)} cases x {args.reps} run(s), then {len(impls_used)} footprint "
          f"run(s) x {args.reps}: {total} processes. Time left is shown once every case "
          "has run once.")
    try:
        schedule = []
        for rep in range(args.reps):
            order = list(cases)
            rng.shuffle(order)
            schedule += [(case, rep) for case in order]
        for index, (case, rep) in enumerate(schedule):
            execute(case, "run", rep, [c for c, _ in schedule[index + 1:]])
        if not args.no_baseline:
            print("runtime footprint of each tool (200-row _baseline dataset):")
            for rep in range(args.reps):
                for impl in impls_used:
                    execute((B.BASELINE, BASELINE_PROTOCOL[impl], impl, 1), "baseline", rep, [])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"results: {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
