#!/usr/bin/env python3
"""Run the CPU benchmark: every (dataset, protocol, implementation, threads) case,
--reps fresh processes each; every process gives training time, peak memory,
tree size and accuracy.

  bench/tools/.venv/bin/python bench/tools/run.py --dry-run                 # show the plan only
  bench/tools/.venv/bin/python bench/tools/run.py covertype_10k susy_50k    # run these datasets
  bench/tools/.venv/bin/python bench/tools/run.py --protocols c45 --impls tree,yadt --threads 1,2,4,8,all

Every case grows one tree (no cross-validation). cart_alpha prunes at one fixed
alpha (benchlib.CART_ALPHA, or --alpha) given unchanged to every CART tool.

Before anything runs, every prepared file of the datasets is hashed (SHA-256)
and every CSV row is checked against the binary files (plan.json records the
hashes), so every tool provably reads the same rows.

Each run is one fresh process that loads the data, does the tool's start-up
and trains exactly one tree; no tool fits anything before the timed fit (no
warm-up), so ten runs are ten executions of the same command (see
bench/tools/README.md, "How time is measured"):
  1. the tool times the training itself (data already in memory);
  2. right after it, the peak resident memory so far is read (VmHWM): load
     and training, before evaluation allocates anything. YaDT, a closed
     binary, cannot do that, so its measured process only trains (and saves
     the tree); its peak is the process's ru_maxrss, and a second, unmeasured
     YaDT process classifies the test rows with the saved tree;
  3. tree size and training and test accuracy are recorded.
The repetitions run in a shuffled order (a new order every repetition), so slow
drift (heat, background load) spreads over all cases. A case whose first run
trains longer than --long-seconds (an hour) runs --long-runs times in all (default 0:
every case --reps times). Every process reports the
rows and features it loaded; a mismatch with the dataset is an error. After the
runs, each tool runs --reps times on the tiny _baseline dataset (no warm-up, no
evaluation): its runtime footprint.

Results go to bench/<run-id>/results.jsonl, with machine.json (hardware,
OS, versions) and plan.json next to it; bench/tools/report.py turns them into tables.
"""

import argparse
import datetime
import hashlib
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
# The code measured, in every row. A node without git (hopper on mordor3) gets
# it from the submitting shell (final.sh, submit.sh: TREE_GIT_DESCRIBE).
def _commit():
    if os.environ.get("TREE_GIT_DESCRIBE"):
        return os.environ["TREE_GIT_DESCRIBE"]
    try:
        return subprocess.run(["git", "-C", str(B.ROOT), "describe", "--always", "--dirty",
                               "--abbrev=12"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


COMMIT = _commit()


# ------------------------------------------------------------------ measuring

def gpu_index():
    """The GPU ./tree uses, in nvidia-smi's numbering: the first of CUDA_VISIBLE_DEVICES
    (bench/tools/cluster/job.sh sets it with CUDA_DEVICE_ORDER=PCI_BUS_ID, nvidia-smi's order)."""
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return first if first.isdigit() else "0"


def gpu_memory_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                          f"--id={gpu_index()}"], capture_output=True, text=True, timeout=30, check=True)
    return int(out.stdout.split()[0])


_GPU_NAME = {}


def gpu_name():
    """The name of the GPU ./tree and the libraries use (gpu_index())."""
    if gpu_index() not in _GPU_NAME:
        _GPU_NAME[gpu_index()] = command_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", f"--id={gpu_index()}"])
    return _GPU_NAME[gpu_index()]


def measure(command, env, timeout, pin=(), gpu=False):
    """Run `command` to completion and measure it from the outside.

    The command runs under the tiny bench/tools/.tools/rusage launcher (pinned with
    `pin`, e.g. taskset): Linux starts a child's peak RSS at the peak RSS of
    the process it was forked from, so started straight from this (large)
    Python process every tool would report at least this process's peak.

    peak_rss_bytes   the kernel's high-water mark of the process's resident set
                     (ru_maxrss from the launcher's wait4, the number GNU time's
                     %M prints), over the whole process.
    wall_seconds     from fork to exit, including start-up, loading, everything.
    gpu_peak_bytes   (gpu=True) the most GPU memory in use while the process ran,
                     minus what was in use just before it started: nvidia-smi
                     samples the whole device (gpu_index()) every 10 ms, so nothing else may
                     use the GPU during the run. The samples go to a file: in a
                     pipe read only at the end, nvidia-smi blocked once 64 KiB
                     were written (about 170 s) and missed the training.
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, \
            tempfile.TemporaryFile() as gpu_samples, tempfile.TemporaryDirectory() as folder:
        usage_file = Path(folder) / "rusage"
        sampler = None
        if gpu:
            idle_mib = gpu_memory_used_mib()
            sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits", f"--id={gpu_index()}", "-lms", "10"],
                                       stdout=gpu_samples, stderr=subprocess.DEVNULL)
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
            sampler.wait()
            gpu_samples.seek(0)
            samples = [int(line) for line in gpu_samples.read().decode().split() if line.isdigit()]
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

# ./tree's backends: one binary's command line and output.
TREE_IMPLS = ("tree", "tree_cuda", "tree_gpu_only", "tree_simple")
# The GPU libraries, run by cluster-gpu-vs-others' gpubench.py in its venv
# (GPU_PYTHON); only when --impls names them (cluster-gpu-vs-others/one-tree).
GPU_LIBRARIES = ("cuml", "xgboost")
GPU_IMPLS = ("tree_cuda", "tree_gpu_only") + GPU_LIBRARIES
GPUBENCH = B.BENCH / "cluster" / "gpubench.py"
GPU_PYTHON = Path(os.environ.get("GPU_PYTHON", Path.home() / "gpu-dt-bench" / "venv" / "bin" / "python"))


def adapter_args(dataset, threads, evaluate):
    m = B.meta(dataset)
    return [f"train={B.stem(dataset, 'train')}",
            f"test={B.stem(dataset, 'test') if m['n_test'] else ''}",
            f"n_train={m['n_train']}", f"n_test={m['n_test']}",
            f"n_features={len(m['features'])}", f"n_classes={len(m['classes'])}",
            f"threads={threads}", f"eval={int(evaluate)}"]


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


def profile_name(case, rep):
    """./tree --profile: the trace of a run, in <run dir>/profiles/."""
    dataset, protocol, impl, threads = case
    return f"profiles/{protocol}-{impl}-t{threads}-r{rep}.json"


# {(dataset, part, SHA-256 of the dumped tree): accuracy}: one prediction per distinct tree.
ACCURACY = {}


HANDOFF_FILE = B.BENCH / "handoff" / "handoff_rows.json"


def handoff_rows(requested, dataset, flags):
    """--hybrid's --handoff-rows for a run, None for ./tree's own default.
    `requested` (run.py --handoff-rows): a number, "default" (./tree's own), or
    None: the value measured best for this node, dataset and depth limit in
    bench/tools/handoff/handoff_rows.json (written by bench/tools/handoff/handoff_sweep.sh): the
    entry of the shallowest measured depth limit at least as deep as this
    run's ("depth<N>" for -d N, "nodepth" for no limit; -d 5 takes depth10:
    the hand-off acts on the small nodes of the deep levels, so a deeper limit
    is the closer measurement), else ./tree's default."""
    if requested == "default":
        return None
    if requested:
        return int(requested)
    try:
        stored = json.loads(HANDOFF_FILE.read_text())
    except FileNotFoundError:
        return None
    entries = stored.get(socket.gethostname(), {}).get(dataset, {})
    def limit(key):  # "depth10" -> 10, "nodepth" -> infinity
        return float("inf") if key == "nodepth" else int(key.removeprefix("depth"))
    depth = float(flags[flags.index("-d") + 1]) if "-d" in flags else float("inf")
    deeper = [key for key in entries if limit(key) >= depth]
    return int(entries[min(deeper, key=limit)]) if deeper else None


def build_command(case, kind, args, scratch, rep=0):
    """-> (argv, extra env, parser of stdout -> result dict)."""
    dataset, protocol, impl, threads = case
    spec = B.PROTOCOLS[protocol]["impls"][impl]
    evaluate = kind == "run"
    m = B.meta(dataset)
    data = B.DATA / dataset

    if impl in TREE_IMPLS:
        flags = [fill(flag, args.alpha) for flag in spec]
        if impl == "tree_cuda":
            binary, backend = B.TREE_CUDA, ["--hybrid", "--threads", str(threads)]
            rows = handoff_rows(args.handoff_rows, dataset, flags)
            if rows:
                backend += ["--handoff-rows", str(rows)]
        elif impl == "tree_gpu_only":
            binary, backend = B.TREE_CUDA, ["--gpu", "--threads", str(threads)]
        elif impl == "tree_simple":
            binary, backend = B.TREE, ["--simple"]
        else:
            binary = B.TREE
            backend = ["--serial"] if threads == 1 else ["--multicore", "--threads", str(threads)]
        dump = scratch / "tree.dump"
        dump.unlink(missing_ok=True)  # never read a previous run's tree
        predict = evaluate and (args.test_accuracy or not args.train_accuracy_only)
        profile = (["--profile", str(args.out_dir / profile_name(case, rep))]
                   if args.profile and kind == "run" else [])
        argv = [str(binary)] + backend + flags \
            + (["--dump", str(dump)] if predict or args.hash_trees else []) \
            + profile + B.tree_input(dataset)

        def parse(stdout):
            nodes, leaves, depth = re.search(r"nodes (\d+), leaves (\d+), depth (\d+)",
                                             stdout).groups()
            rows, features = re.search(r"rows (\d+) train, (\d+) features", stdout).groups()
            peak = int(re.search(r"peak after training (\d+) KiB", stdout).group(1))
            result = {"train_seconds": float(re.search(r"train total\s+([\d.]+) ms",
                                                       stdout).group(1)) / 1000,
                      # The phases of the training time (gpu setup: CUDA only),
                      # and gpu startup: the CUDA context, memory pool and
                      # staging buffers, made during the load, not in train total;
                      # gpu startup wait: the part of it still running when the
                      # load ended. load: reading the input file (read_seconds).
                      **{f"{'read' if name == 'load' else name.replace(' ', '_')}_seconds":
                         float(ms) / 1000
                         for name, ms in re.findall(r"^  (load|gpu startup wait|gpu startup|gpu setup|"
                                                    r"setup buffers|setup allocate|setup upload|setup finish|"
                                                    r"presort|build|prune)\s+([\d.]+) ms$",
                                                    stdout, re.M)},
                      # When each section ran (CUDA: "span" lines, seconds since
                      # train total started; busy: the CPU subtree tasks' summed work).
                      **({"timeline": [{"name": name, "start": float(a) / 1000, "end": float(b) / 1000,
                                        **({"busy": float(busy) / 1000, "tasks": int(tasks)} if busy else {})}
                                       for name, a, b, busy, tasks in spans]}
                         if (spans := re.findall(r"^  span (.+?): ([\d.]+) ([\d.]+) ms"
                                                 r"(?:, busy ([\d.]+) ms in (\d+) tasks)?$", stdout, re.M))
                         else {}),
                      "nodes": int(nodes), "leaves": int(leaves), "depth": int(depth),
                      "n_train_loaded": int(rows), "n_features_loaded": int(features),
                      "peak_rss_train_bytes": peak * 1024}
            if args.hash_trees:
                # The same tree from every backend: the same dump, byte for byte.
                result["tree_sha256"] = hashlib.sha256(dump.read_bytes()).hexdigest()
            if evaluate and not predict:
                # --train-accuracy-only: ./tree's own training accuracy, no test rows.
                result["train_accuracy"] = float(
                    re.search(r"train accuracy\s+([\d.]+)%", stdout).group(1)) / 100
            elif evaluate:
                # Both accuracies from the dumped tree, so ./tree is measured like the
                # others: by predicting every row. ./tree's own training accuracy
                # (printed to 4 decimals) must agree.
                printed = float(re.search(r"train accuracy\s+([\d.]+)%", stdout).group(1))
                digest = hashlib.sha256(dump.read_bytes()).hexdigest()
                for part in ("test",) if args.test_accuracy else ("train", "test"):
                    if not m[f"n_{part}"]:  # no test set (chess)
                        continue
                    # The runs of a case grow the same tree: its rows are predicted once
                    # (minutes on SOREL's test set), the repeats reuse the accuracy.
                    key = (dataset, part, digest)
                    if key not in ACCURACY:
                        X, y = B.load(dataset, part)
                        ACCURACY[key] = float(np.mean(B.predict_tree(
                            B.parse_tree_dump(dump, *B.csv_names(m)), X) == y))
                        del X, y  # never held while the next process runs
                    result[f"{part}_accuracy"] = ACCURACY[key]
                if args.test_accuracy:
                    # --test-accuracy: the training accuracy as ./tree printed it.
                    result["train_accuracy"] = printed / 100
                elif abs(printed - 100 * result["train_accuracy"]) > 1e-4:
                    raise ValueError(f"dump predicts {100 * result['train_accuracy']:.6f}% on "
                                     f"the training rows, ./tree printed {printed}%")
            return result
        return argv, {}, parse

    if impl == "yadt":
        if "prefix_of" in m:
            raise ValueError(f"{dataset}: YaDT reads whole files, a prefix_of set has none")
        # The measured process only trains: it saves the tree and reports the
        # training accuracy; the test rows are classified afterwards by a second,
        # unmeasured process that loads the saved tree, so they never add to the
        # training process's peak memory.
        report, test_report, saved = (scratch / "yadt.txt", scratch / "yadt_test.txt",
                                      scratch / "yadt.tree")
        argv = [str(B.YADT), "-fm", str(data / "yadt.names"), "-fd", str(data / "train.yadt.csv"),
                "-tt", str(threads)] + list(spec)
        for path in (report, test_report, saved):
            path.unlink(missing_ok=True)  # never read a previous run's output
        if evaluate:
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
            build = sum(float(s[2]) for s in steps)
            result = {"train_seconds": total - load + build,
                      "read_seconds": load, "index_seconds": total - load, "build_seconds": build,
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

    if impl in GPU_LIBRARIES:
        # gpubench.py (cluster-gpu-vs-others): one process, the library's
        # start-up, one timed tree (upload through ./tree's own staged upload,
        # every step timed); its row is its last line of output.
        argv = [str(GPU_PYTHON), str(GPUBENCH), impl, dataset, "cart", str(scratch / "gpubench.jsonl"),
                str(scratch), "1", str(spec["depth"])]
        # OMP_NUM_THREADS: the run's CPUs (base_env's 1 would leave the
        # libraries' host threads at one).
        # CUDA_MODULE_LOADING: LAZY (CUDA's default) unless GPULIB_MODULE_LOADING
        # says otherwise. cuML and XGBoost link their own CUDA runtime, so EAGER
        # loads *all* their kernels at their first call, inside fit (HIGGS on the
        # A100, 2026-11-17: cuML fit 2.19 s EAGER vs 1.71 s LAZY, XGBoost 3.11 vs
        # 2.88 s); ./tree too loads its kernels lazily, in its timed setup.
        environment = {"EVAL": str(int(evaluate)), "GPU_MEMORY": "0", "DATA_ROOT": str(B.DATA),
                       "CUDA_MODULE_LOADING": os.environ.get("GPULIB_MODULE_LOADING", "LAZY"),
                       "OMP_NUM_THREADS": str(threads)}
        return argv, environment, lambda stdout: json.loads(stdout.strip().splitlines()[-1])

    params = [f"{key}={fill(value, args.alpha)}" for key, value in spec.items()]
    common = adapter_args(dataset, threads, evaluate)
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


def ram_modules():
    """Installed memory modules from the DMI data udev exports (no root needed):
    [{"type": "DDR5", "mts": 4800, "bytes": ...}], empty if unknown."""
    text = command_output(["udevadm", "info", "/sys/devices/virtual/dmi/id"])
    found = defaultdict(dict)
    for key, value in re.findall(r"MEMORY_DEVICE_(\d+_[A-Z_]+)=(.*)", text):
        index, field = key.split("_", 1)
        found[int(index)][field] = value
    return [{"type": d["TYPE"], "mts": int(d["CONFIGURED_SPEED_MTS"]), "bytes": int(d["SIZE"])}
            for _, d in sorted(found.items())
            if d.get("SIZE") and d.get("TYPE") not in (None, "Unknown")
            and d.get("CONFIGURED_SPEED_MTS")]


def file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
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
        "ram_modules": ram_modules(),
        "mem_total_kb": int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1)) if meminfo else None,
        "kernel": command_output(["uname", "-srvm"]),
        "os": command_output(["sh", "-c", ". /etc/os-release && echo $PRETTY_NAME"]),
        "versions": {
            # A node without git (hopper on mordor3) gets it from the submitting shell.
            "tree_git": COMMIT,
            "tree_binary": str(B.TREE),
            "tree_cuda_binary": str(B.TREE_CUDA),
            # what ran, whatever the commit stamp says: the binaries' SHA-256 and the
            # commit build.sh built them from (build/COMMIT)
            "tree_binary_sha256": file_sha256(B.TREE),
            "tree_cuda_binary_sha256": file_sha256(B.TREE_CUDA),
            "tree_build_commit": read_text(B.ROOT / "build" / "COMMIT"),
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
        # Free memory per NUMA node at the start: a node whose memory is taken
        # (e.g. by /dev/shm or page cache) sends a pinned run's pages to the other.
        "numa_mem_free_kb": numa_mem_free_kb(),
        "gpu_index": gpu_index(),
        "settings": {"alpha": args.alpha,
                     "long_runs": args.long_runs, "long_seconds": args.long_seconds,
                     "java_heap": args.java_heap, "cpus": args.cpus, "membind": args.membind,
                     "drop_cache": args.drop_cache, "reps": args.reps,
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


def numa_mem_free_kb():
    free = {}
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        match = re.search(r"MemFree:\s+(\d+)", read_text(node / "meminfo") or "")
        if match:
            free[node.name] = int(match.group(1))
    return free


VMSTAT = ("pgscan_kswapd", "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct", "allocstall_normal",
          "allocstall_movable", "compact_stall", "numa_miss", "numa_foreign", "pgmigrate_success",
          # NUMA placement: pages the run got on its own node vs another one
          "numa_local", "numa_other",
          # transparent huge pages: got one, or fell back to 4 KiB pages (fragmented memory)
          "thp_fault_alloc", "thp_fault_fallback", "thp_collapse_alloc")


def throttle_count(cpus):
    """Thermal throttling events of the CPUs so far (Intel: core and package
    counters; none on AMD)."""
    total, found = 0, False
    for cpu in cpus:
        for name in ("core_throttle_count", "package_throttle_count"):
            value = read_text(f"/sys/devices/system/cpu/cpu{cpu}/thermal_throttle/{name}")
            if value and value.strip().isdigit():
                total, found = total + int(value), True
    return total if found else None


class ClockSampler:
    """The run's CPUs' clock (scaling_cur_freq, MHz) every 0.25 s while it runs, from
    this process (a few sysfs reads a second); mean and minimum of the per-sample
    means. Untimed: the tool's own clock is unaffected."""
    def __init__(self, cpus):
        self.files = [f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq" for c in cpus]
        self.means, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.wait(0.25):
            khz = [int(v) for v in (read_text(f) for f in self.files) if v and v.strip().isdigit()]
            if khz:
                self.means.append(sum(khz) / len(khz) / 1000)

    def result(self):
        self.stop.set()
        self.thread.join()
        return ({"cpu_mhz_mean": round(sum(self.means) / len(self.means)), "cpu_mhz_min": round(min(self.means)),
                 "cpu_mhz_samples": len(self.means)} if self.means else {})


def vmstat():
    """The node's page-reclaim and NUMA counters (whole node: the job owns it)."""
    counters = dict(line.split() for line in (read_text("/proc/vmstat") or "").splitlines() if " " in line)
    return {k: int(counters[k]) for k in VMSTAT if k in counters}


def drop_page_cache(root):
    """Drop root's files (the datasets) from the page cache: posix_fadvise
    DONTNEED, which needs no privileges. Cached CSVs of earlier runs pile up
    on one NUMA node (236 GiB of epyc1's node 1 on 2026-10-20) and send a
    whole-node run's pages to the other socket."""
    for path in Path(root).resolve().rglob("*"):
        try:
            if path.is_file():
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
        except OSError:
            pass


def plan_cases(args):
    datasets = args.datasets or [d for d in B.prepared() if d != B.BASELINE]
    protocols = args.protocols.split(",") if args.protocols else list(B.PROTOCOLS)
    impls = args.impls.split(",") if args.impls else [i for i in B.IMPLS if i not in GPU_LIBRARIES]
    all_threads = len(args.cpu_list) if args.cpu_list else os.cpu_count()
    threads = sorted({all_threads if t == "all" else int(t) for t in args.threads.split(",")})
    cases = []
    for dataset in datasets:
        if B.meta_path(dataset) is None:
            sys.exit(f"{dataset}: not prepared (bench/tools/prepare.py {dataset})")
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
BASELINE_PROTOCOL = {"tree": "cart_full", "tree_cuda": "cart_full", "tree_gpu_only": "cart_full",
                     "tree_simple": "cart_full",
                     "sklearn": "cart_full",
                     "rpart": "cart_full",
                     "j48": "c45", "yadt": "c45"}  # the GPU libraries: none


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
    parser.add_argument("--membind", metavar="NODES",
                        help="run every process under numactl --membind=NODES: its memory only "
                        "from those NUMA nodes, e.g. 0 or 0,1 (pair it with --cpus of those nodes)")
    parser.add_argument("--drop-cache", action="store_true",
                        help="drop the datasets from the page cache before every process "
                             "(every run records its free memory per NUMA node, mem_free_node*_kb)")
    parser.add_argument("--reps", type=int, default=1, help="runs (fresh processes) per case")
    parser.add_argument("--long-seconds", type=float, default=3600,
                        help="a case whose first run trains longer (train_seconds; the whole "
                             "process if it failed) "
                             "runs only --long-runs times in all (default 3600)")
    parser.add_argument("--long-runs", type=int, default=0,
                        help="runs of a case whose first run took longer than --long-seconds "
                             "(default 0 = no limit: every case --reps times)")
    parser.add_argument("--spin", type=float, default=0,
                        help="seconds of every CPU busy (untimed) right before the first run, "
                             "after the dataset check: a node idle for a while runs its first "
                             "seconds slow (powersave governor)")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per process")
    parser.add_argument("--cooldown", type=float, default=0, help="seconds of rest between processes")
    parser.add_argument("--java-heap", default=None, help="JVM -Xmx (default: half of RAM)")
    parser.add_argument("--run-id", help="results go to bench/<run-id> (default host-date)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the shuffled order")
    parser.add_argument("--train-accuracy-only", action="store_true",
                        help="./tree: record only its own training accuracy, skip predicting "
                             "the training and test rows from its dumped tree (minutes on big trees)")
    parser.add_argument("--test-accuracy", action="store_true",
                        help="./tree: its own training accuracy, and the test accuracy by "
                             "predicting only the test rows from its dumped tree (the full "
                             "ECBDL's training rows are 80 GB); overrides --train-accuracy-only")
    parser.add_argument("--hash-trees", action="store_true",
                        help="./tree: record the SHA-256 of every tree's --dump (tree_sha256), "
                             "to check that the backends grow the same tree")
    parser.add_argument("--profile", action="store_true",
                        help="./tree: also write where its training time goes (--profile) to "
                             "<run dir>/profiles/<protocol>-<impl>-t<threads>-r<rep>.json")
    parser.add_argument("--handoff-rows",
                        help="./tree --hybrid (tree_cuda): its --handoff-rows, a number or "
                             "'default' (./tree's own); without it, the value measured best for "
                             "the node, dataset and depth limit in bench/tools/handoff/handoff_rows.json, else "
                             "./tree's own; --gpu has none")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()
    args.cpu_list = parse_cpus(args.cpus) if args.cpus else None
    if not args.java_heap:
        total_kb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1))
        args.java_heap = f"{max(1, total_kb // 2 // 1024 // 1024)}g"

    cases = plan_cases(args)
    impls_used = sorted({case[2] for case in cases})
    processes = (len(cases) + len(impls_used)) * args.reps
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
        sys.exit(f"{B.RUSAGE} missing: run bench/tools/setup.sh")

    run_id = args.run_id or f"{socket.gethostname()}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    out_dir = B.RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = out_dir
    if args.profile:
        (out_dir / "profiles").mkdir(exist_ok=True)
    json.dump(machine_info(args), open(out_dir / "machine.json", "w"), indent=1)
    datasets = sorted({case[0] for case in cases} | {B.BASELINE})
    print("checking the data (SHA-256 of every file, every CSV row against the binary files):")
    data_files = {}
    # In a child process: pyarrow keeps the CSV's memory after the check (SOREL:
    # about 135 GiB on epyc), which got C4.5's 370 GiB killed on a 512 GB node.
    import concurrent.futures
    import multiprocessing
    for dataset in datasets:
        start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(
                1, mp_context=multiprocessing.get_context("fork")) as pool:
            data_files[dataset] = pool.submit(B.verify_dataset, dataset).result()
        print(f"  {dataset}: identical in every format ({time.perf_counter() - start:.0f} s)")
    json.dump({"cases": cases, "protocols": B.PROTOCOLS, "impls": B.IMPLS,
               "datasets": {d: B.meta(d) for d in datasets}, "data_files": data_files},
              open(out_dir / "plan.json", "w"), indent=1)
    results = open(out_dir / "results.jsonl", "a")

    failed = set()  # cases that timed out or failed: their other runs are skipped
    env = base_env()
    scratch = Path(tempfile.mkdtemp(prefix="bench-"))
    total = (len(cases) + len(impls_used)) * args.reps
    progress = {"done": 0, "started": time.perf_counter()}
    walls = defaultdict(list)  # case -> whole-process seconds, for the time left
    # case -> its first run's training time (whole process if it failed): the
    # --long-seconds rule (3 runs where one run trains for over an hour)
    first_train = {}

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
        argv, extra_env, parse = build_command(case, kind, args, scratch, rep)
        if args.drop_cache:
            drop_page_cache(B.DATA)
        pin = ["taskset", "-c", ",".join(map(str, args.cpu_list[:threads]))] if args.cpu_list else []
        if args.membind is not None:
            pin = ["numactl", f"--membind={args.membind}"] + pin
        mem_free = numa_mem_free_kb()
        before = vmstat()
        run_cpus = args.cpu_list[:threads] if args.cpu_list else range(os.cpu_count() or 1)
        throttle_before = throttle_count(run_cpus)
        load = (read_text("/proc/loadavg") or "").split()[:3]
        sampler = ClockSampler(run_cpus)
        run = measure(argv, {**env, **extra_env}, args.timeout, pin, gpu=impl in GPU_IMPLS)
        cpu_clock = sampler.result()
        after = vmstat()
        throttle_after = throttle_count(run_cpus)
        row = {"run_id": run_id, "dataset": dataset, "protocol": protocol, "impl": impl,
               "threads": threads, "kind": kind, "rep": rep,
               "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               # where and what ran: the commit, the node, the CPUs, the GPU
               "commit": COMMIT, "host": socket.gethostname(),
               "cpus": ",".join(map(str, args.cpu_list[:threads])) if args.cpu_list else None,
               "membind": args.membind,
               **({"gpu": gpu_name(), "gpu_index": gpu_index()} if impl in GPU_IMPLS else {}),
               **{k: run[k] for k in ("wall_seconds", "peak_rss_bytes", "user_seconds",
                                      "system_seconds", "gpu_peak_bytes") if k in run}}
        if run["timed_out"]:
            row["status"] = "timeout"
        elif run["returncode"] != 0:
            row["status"] = "error"
            row["error"] = (run["stderr"] or run["stdout"]).strip()[-2000:]
        else:
            try:
                # the harness's own fields win (gpubench.py also reports cpus, time, gpu_index)
                row = {**parse(run["stdout"]), **row}
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
            # YaDT: the process only trains; the GPU libraries: the whole process
            row["peak_rss_train_bytes"] = row["peak_rss_bytes"]
        if row["status"] != "ok" and kind != "baseline":
            failed.add(case)
        if args.profile and kind == "run" and impl in TREE_IMPLS:
            row["profile"] = profile_name(case, rep)
        row["command"] = shlex.join(pin + argv)
        row.update({f"mem_free_{node}_kb": kb for node, kb in mem_free.items()})
        # page reclaim and remote allocations during the run (SOREL C4.5 on epyc, 2026-10-28:
        # presort 11 s in one run, 34 s in the next two)
        row.update({f"vmstat_{k}": after[k] - before[k] for k in before if k in after})
        # why a run was slow: the CPU clock over the run, thermal throttling events,
        # the load just before it (with mem_free_* and vmstat_* above)
        row.update(cpu_clock)
        if throttle_before is not None and throttle_after is not None:
            row["throttle_events"] = throttle_after - throttle_before
        if load:
            row["loadavg_before"] = " ".join(load)
        results.write(json.dumps(row) + "\n")
        results.flush()
        walls[case].append(run["wall_seconds"])
        first_train.setdefault(case, row.get("train_seconds") or run["wall_seconds"])
        elapsed = time.perf_counter() - progress["started"]
        left = time_left(pending)
        eta = f", about {clock(left)} left" if left is not None else ""
        if row["status"] == "ok":
            accuracy = (f", test acc {100 * row['test_accuracy']:.2f}%"
                        if row.get("test_accuracy") is not None else
                        f", train acc {100 * row['train_accuracy']:.2f}%"
                        if row.get("train_accuracy") is not None else "")
            gpu = (f", GPU {row['gpu_peak_bytes'] / 2**20:,.0f} MiB"
                   if row.get("gpu_peak_bytes") is not None else "")
            peak = (f", peak {row['peak_rss_train_bytes'] / 2**20:,.0f} MiB"
                    if row.get("peak_rss_train_bytes") else "")
            shape = (f", {row['nodes']:,} nodes, depth {row['depth']}"
                     if row.get("nodes") is not None else "")
            print(f"    ok: train {row['train_seconds']:.4g} s{peak}{gpu}{shape}{accuracy} | process "
                  f"{clock(run['wall_seconds'])}, elapsed {clock(elapsed)}{eta}", flush=True)
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

    if args.spin > 0:  # untimed: wake every CPU after the single-threaded dataset check
        spinners = [subprocess.Popen(["timeout", str(args.spin), "sh", "-c", "while :; do :; done"])
                    for _ in range(os.cpu_count() or 1)]
        for p in spinners:
            p.wait()
        print(f"every CPU busy for {args.spin:g} s before the first run (untimed)")
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
            # One run of this case took over --long-seconds: --long-runs runs in all.
            if (args.long_runs and rep >= args.long_runs and case in first_train
                    and first_train[case] > args.long_seconds):
                progress["done"] += 1
                print(f"[{progress['done']}/{total}] skipped: {describe(case)} has had "
                      f"{args.long_runs} runs, its first trained for {clock(first_train[case])} "
                      f"(over {clock(args.long_seconds)})", flush=True)
                continue
            execute(case, "run", rep, [c for c, _ in schedule[index + 1:]
                                       if not (args.long_runs and c in first_train
                                               and first_train[c] > args.long_seconds)])
        print("runtime footprint of each tool (200-row _baseline dataset):")
        for rep in range(args.reps):
            for impl in [i for i in impls_used if i in BASELINE_PROTOCOL]:
                execute((B.BASELINE, BASELINE_PROTOCOL[impl], impl, 1), "baseline", rep, [])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if args.drop_cache:  # leave the node's memory as we found it
            drop_page_cache(B.DATA)
    print(f"results: {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
