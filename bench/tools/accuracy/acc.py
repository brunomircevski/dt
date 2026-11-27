#!/usr/bin/env python3
"""Pruned-accuracy benchmark: one tree per (dataset, algorithm, tool, depth
limit), pruned and unpruned, scored on the test set. Time is not measured:
the cases run side by side (--jobs), each one process.

  acc.py DATASET --out DIR --side cpu|gpu [--depths 5,10,15,20] [--jobs N]
         [--tools tree,sklearn,...] [--threads T] [--yadt-threads T] [--dry-run]

Pruning, the same for every tool of an algorithm (one fixed setting, nothing
tuned; benchlib.CART_ALPHA and the C4.5 programs' default confidence):
  CART  minimal cost-complexity at alpha = 1e-5 (a misclassification rate):
        ./tree --prune 1e-5; rpart prune(cp = alpha n / root risk); scikit-learn,
        cuML and XGBoost grow their tree, prunelib.ccp_prune prunes it by the
        same rule (none of them has it: scikit-learn's ccp_alpha weighs Gini
        impurity, cuML has only pre-pruning, XGBoost's gamma its own gain).
  C4.5  error-based pruning, CF 0.25 with subtree raising: ./tree --prune 0.25,
        J48 -C 0.25 -M 2, YaDT -ebpg -c 0.25 -m 2.
Unpruned trees (the same depth limit) are the reference for what pruning did.

side cpu: ./tree --multicore (both), --serial (pruned: the tree's hash must
match), scikit-learn, rpart, J48, YaDT. side gpu: cuML, XGBoost, ./tree --gpu
and --hybrid (pruned; hashes checked by report.py).

Every case writes DIR/cases/<case>.json (a list of records) and its test
predictions in DIR/pred/; a case whose json exists is skipped (rerun = resume).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import benchlib as B  # noqa: E402

ALPHA = B.CART_ALPHA  # 1e-5
CF = 0.25
GPU_PYTHON = os.environ.get("GPU_PYTHON", str(Path.home() / "gpu-dt-bench" / "venv" / "bin" / "python"))
TOOLS_DIR = Path(os.environ.get("ACC_TOOLS", B.TOOLS))  # weka.jar, bounce.jar, yadt/
# rpart 4.1.27 indexes its n x p arrays with 32-bit ints; over 2^31 - 1 values
# (MNIST8M, ECBDL'14 15M, SOREL 4M) it runs from this library, built with
# rpart-64bit-index.patch (64-bit offsets only, the same tree)
RPART64 = TOOLS_DIR / "rpart64"
CLASSES = HERE / ".classes"  # J48Fit + J48Acc, compiled by the job
TREE_FLAGS = {("cart", True): ["--cart", "--prune", repr(ALPHA)], ("cart", False): ["--cart"],
              ("c45", True): ["--c45", "--prune", str(CF)], ("c45", False): ["--c45"]}


def cases(dataset, side, depths, tools, yadt_threads=4):
    """-> [(case id, kind, params, threads)]"""
    m = B.meta(dataset)
    rows_x_features = m["n_train"] * len(m["features"])
    yadt = (B.yadt_folder(dataset) / "yadt.names").exists()  # a prefix set's: prepare.py --yadt
    out = []
    for d in depths:
        if side == "cpu":
            for algo in ("cart", "c45"):
                for pruned in (True, False):
                    tag = "pruned" if pruned else "unpruned"
                    if "tree" in tools:
                        out.append((f"{algo}-tree-d{d}-{tag}", "tree", dict(algo=algo, d=d, pruned=pruned,
                                                                              backend="multicore"), None))
                        if pruned:
                            out.append((f"{algo}-tree_serial-d{d}-{tag}", "tree",
                                        dict(algo=algo, d=d, pruned=pruned, backend="serial"), 1))
                    if algo == "c45" and "yadt" in tools and yadt:
                        out.append((f"c45-yadt-d{d}-{tag}", "yadt", dict(d=d, pruned=pruned), yadt_threads))
            if "j48" in tools:  # one fit: the unpruned tree, then the pruned one
                out.append((f"c45-j48-d{d}", "j48", dict(d=d), 1))
            if "sklearn" in tools:
                out.append((f"cart-sklearn-d{d}", "lib", dict(impl="sklearn", d=d), 1))
            if "rpart" in tools and (rows_x_features <= 2**31 - 1 or RPART64.exists()):
                out.append((f"cart-rpart-d{d}", "rpart", dict(d=d), 1))
        else:
            for impl in ("cuml", "xgboost"):
                if impl in tools:
                    out.append((f"cart-{impl}-d{d}", "lib", dict(impl=impl, d=d), None))
            if "tree" in tools:
                for algo in ("cart", "c45"):
                    for backend in ("gpu", "hybrid"):
                        out.append((f"{algo}-tree_{backend}-d{d}-pruned", "tree",
                                    dict(algo=algo, d=d, pruned=True, backend=backend), None))
    return out


def run(argv, log, env=None, timeout=None):
    with open(log, "w") as f:
        f.write(" ".join(map(str, argv)) + "\n")
        f.flush()
        p = subprocess.run(list(map(str, argv)), stdout=subprocess.PIPE, stderr=f, text=True,
                           env={**os.environ, **(env or {})}, timeout=timeout)
        f.write(p.stdout)
    if p.returncode:
        raise RuntimeError(f"exit {p.returncode}: {' '.join(map(str, argv))} (log {log})")
    return p.stdout


_TEST, _TEST_LOCK = {}, __import__("threading").Lock()


def test_set(dataset):
    """The test rows, loaded once per driver (SOREL's are 40 GB)."""
    with _TEST_LOCK:
        if dataset not in _TEST:
            _TEST[dataset] = B.load(dataset, "test")
        return _TEST[dataset]


def do_tree(dataset, cid, p, threads, out):
    m = B.meta(dataset)
    dump = out / "trees" / f"{cid}.dump"
    binary = B.TREE_CUDA if p["backend"] in ("gpu", "hybrid") else B.TREE
    backend = {"multicore": ["--multicore", "--threads", str(threads)], "serial": ["--serial"],
               "gpu": ["--gpu", "--threads", str(threads)], "hybrid": ["--hybrid", "--threads", str(threads)]}
    argv = [binary] + backend[p["backend"]] + TREE_FLAGS[(p["algo"], p["pruned"])] + ["-d", str(p["d"]),
            "--dump", dump] + B.tree_input(dataset)
    start = time.time()
    stdout = run(argv, out / "logs" / f"{cid}.log")
    nodes, leaves, depth = map(int, re.search(r"nodes (\d+), leaves (\d+), depth (\d+)", stdout).groups())
    printed = float(re.search(r"train accuracy\s+([\d.]+)%", stdout).group(1)) / 100
    tree = B.parse_tree_dump(dump, *B.csv_names(m))
    Xt, yt = test_set(dataset)
    pred = B.predict_tree(tree, Xt).astype(np.int16)
    tag = "pruned" if p["pruned"] else "unpruned"
    pred_file = out / "pred" / f"{cid}.npy"
    np.save(pred_file, pred)
    return [{"dataset": dataset, "algo": p["algo"], "tool": "tree", "backend": p["backend"],
             "depth_limit": p["d"], "pruned": p["pruned"],
             "prune": (f"--prune {ALPHA!r}" if p["algo"] == "cart" else f"--prune {CF}") if p["pruned"] else "none",
             "nodes": nodes, "leaves": leaves, "depth": depth, "train_accuracy": printed,
             "test_accuracy": float(np.mean(pred == yt)), "n_train": m["n_train"], "n_test": m["n_test"],
             "tree_sha256": hashlib.sha256(dump.read_bytes()).hexdigest(), "fit_seconds": time.time() - start,
             "pred": str(pred_file), "version": f"./tree {B.ROOT.name} {os.environ.get('TREE_GIT_DESCRIBE', '')}".strip()}]


def do_lib(dataset, cid, p, threads, out):
    py = sys.executable if p["impl"] == "sklearn" else GPU_PYTHON
    out_json = out / "tmp" / f"{cid}.json"
    run([py, HERE / "fit_lib.py", p["impl"], dataset, p["d"], ALPHA, out_json, out / "pred" / cid],
        out / "logs" / f"{cid}.log", env={"DATA_ROOT": str(B.DATA)})
    return json.loads(out_json.read_text())


def do_rpart(dataset, cid, p, threads, out):
    m = B.meta(dataset)
    out_json = out / "tmp" / f"{cid}.json"
    big = m["n_train"] * len(m["features"]) > 2**31 - 1
    run(["Rscript", "--vanilla", HERE / "fit_rpart.R", B.stem(dataset, "train"), B.stem(dataset, "test"),
         m["n_train"], m["n_test"], len(m["features"]), p["d"], ALPHA, out_json, out / "pred" / cid],
        out / "logs" / f"{cid}.log", env={"RPART_LIB": str(RPART64)} if big else None)
    return [{**r, "dataset": dataset} for r in json.loads(out_json.read_text())]


def do_j48(dataset, cid, p, threads, out):
    """J48Acc: one fit, the unpruned tree (after C4.5's collapse) and then the
    pruned one (-C CF -M 2), both evaluated."""
    m = B.meta(dataset)
    options = f"-C {CF} -M 2"
    heap = os.environ.get("JAVA_HEAP", "200g")
    cp = f"{TOOLS_DIR / 'weka.jar'}:{TOOLS_DIR / 'bounce.jar'}:{CLASSES}"
    stdout = run(["java", f"-Xmx{heap}", "-XX:ParallelGCThreads=4", "-XX:ConcGCThreads=1",
                  # G1 keeps all the heap it ever grew to (ECBDL'14: 200 GB for 77 GB
                  # of live data, 2026-11-24): give back what is free after a GC
                  "-XX:MinHeapFreeRatio=5", "-XX:MaxHeapFreeRatio=15", "-cp", cp,
                  "J48Acc", f"train={B.stem(dataset, 'train')}", f"test={B.stem(dataset, 'test')}",
                  f"n_train={m['n_train']}", f"n_test={m['n_test']}", f"n_features={len(m['features'])}",
                  f"n_classes={len(m['classes'])}", f"options={options}", f"max_depth={p['d']}",
                  f"pred={out / 'pred' / cid}"], out / "logs" / f"{cid}.log")
    return [{"dataset": dataset, "algo": "c45", "tool": "j48", "version": "Weka J48", "depth_limit": p["d"],
             "prune": options if r["pruned"] else "none (-U: the same tree before prune())", **r,
             "n_train": m["n_train"], "n_test": m["n_test"]}
            for r in json.loads(stdout.strip().splitlines()[-1])]


def yadt_scores(path, classes):
    """dTcmd -s output: actual,predicted,probability per row -> predicted class indices."""
    index = {name: k for k, name in enumerate(classes)}
    return np.array([index[line.split(",")[1]] for line in open(path)], dtype=np.int16)


def do_yadt(dataset, cid, p, threads, out):
    m = B.meta(dataset)
    yadt = TOOLS_DIR / "yadt" / "dTcmd"
    env = {"LD_LIBRARY_PATH": str(yadt.parent)}
    data = B.yadt_folder(dataset)
    saved = out / "tmp" / f"{cid}.yadt"
    flags = ["-ebpg", "-c", str(CF), "-m", "2"] if p["pruned"] else ["-np", "-m", "2"]
    stdout = run([yadt, "-fm", data / "yadt.names", "-fd", data / "train.yadt.csv", "-tt", threads]
                 + flags + ["-depth", p["d"], "-tb", saved], out / "logs" / f"{cid}.log", env=env)
    nodes, depth = map(int, re.findall(r"size: (\d+) depth: (\d+)", stdout)[-1])
    names = m["classes"]
    accuracy = {}
    for part in ("train", "test"):
        score = out / "tmp" / f"{cid}.{part}.score"
        run([yadt, "-bt", saved, "-fs", data / f"{part}.yadt.csv", "-s", score],
            out / "logs" / f"{cid}.{part}.log", env=env)
        pred = yadt_scores(score, names)
        _, y = B.load(dataset, part)
        if len(pred) != len(y):
            raise RuntimeError(f"YaDT scored {len(pred)} {part} rows, expected {len(y)}")
        accuracy[part] = float(np.mean(pred == y))
        score.unlink()
    pred_file = out / "pred" / f"{cid}.npy"
    np.save(pred_file, pred)
    return [{"dataset": dataset, "algo": "c45", "tool": "yadt", "version": "YaDT 2.3.0", "depth_limit": p["d"],
             "pruned": p["pruned"], "prune": " ".join(flags), "nodes": nodes, "leaves": (nodes + 1) // 2,
             "depth": depth, "train_accuracy": accuracy["train"], "test_accuracy": accuracy["test"],
             "n_train": m["n_train"], "n_test": m["n_test"], "threads": threads, "pred": str(pred_file)}]


def mem_gb(dataset, kind, params):
    """A case's estimated peak memory (GB), from the data's size: V = training
    values as float32, Vt = test values. J48 holds doubles plus Weka's
    per-row objects (MNIST8M at depth 10: 82 GB = 3.2 V); rpart's data.frame of
    doubles and copies (HIGGS: 8.2 GB = 7 V); the library fits hold the
    training and test rows; ./tree its presorted columns."""
    m = B.meta(dataset)
    F = len(m["features"])
    V, Vt = m["n_train"] * F * 4 / 1e9, m["n_test"] * F * 4 / 1e9
    return {"j48": 3.5 * V + 4, "rpart": 8 * V + 2, "lib": 1.3 * V + Vt + 2, "yadt": 4 * V + 2,
            "tree": 3 * V + 2}[kind]


KINDS = {"tree": do_tree, "lib": do_lib, "rpart": do_rpart, "j48": do_j48, "yadt": do_yadt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--side", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--depths", default="5,10,15,20")
    ap.add_argument("--tools", default="tree,sklearn,rpart,j48,yadt,cuml,xgboost")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8, help="./tree's threads")
    ap.add_argument("--yadt-threads", type=int, default=4, help="YaDT's threads (-tt)")
    ap.add_argument("--only", help="regex on case ids")
    ap.add_argument("--mem-gb", type=float, default=400, help="memory budget of the cases running at once")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = a.out / a.dataset
    for sub in ("cases", "pred", "logs", "tmp", "trees"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases(a.dataset, a.side, [int(d) for d in a.depths.split(",")], a.tools.split(","),
                       a.yadt_threads)
            if not (out / "cases" / f"{c[0]}.json").exists() and (not a.only or re.search(a.only, c[0]))]
    print(f"{socket.gethostname()}: {len(todo)} cases for {a.dataset} ({a.side}), {a.jobs} at a time", flush=True)
    if a.dry_run:
        for c in todo:
            print(" ", c[0])
        return

    def one(case):
        cid, kind, params, threads = case
        start = time.time()
        done = out / "cases" / f"{cid}.json"
        if done.exists():  # another job finished it since this one started
            return cid, 0.0, json.loads(done.read_text())
        records = KINDS[kind](a.dataset, cid, params, threads or a.threads, out)
        for r in records:
            r.setdefault("host", socket.gethostname())
            r["case"] = cid
        (out / "cases" / f"{cid}.json").write_text(json.dumps(records, indent=1))
        return cid, time.time() - start, records

    failed = 0
    # The longest kinds first (J48, then the libraries), so they start at once;
    # a case starts only when its estimated memory fits in --mem-gb (or nothing runs).
    order = {"j48": 0, "lib": 1, "rpart": 1, "yadt": 2, "tree": 3}
    todo.sort(key=lambda c: (order[c[1]], -c[2].get("d", 0)))
    lock = threading.Condition()
    used = [0.0]

    def guarded(case):
        need = mem_gb(a.dataset, case[1], case[2])
        with lock:
            lock.wait_for(lambda: used[0] == 0 or used[0] + need <= a.mem_gb)
            used[0] += need
        try:
            return one(case)
        finally:
            with lock:
                used[0] -= need
                lock.notify_all()

    # Cases start in list order: each worker takes the next case and waits for
    # its memory, so a big case is not overtaken forever by small ones.
    with cf.ThreadPoolExecutor(a.jobs) as pool:
        futures = {pool.submit(guarded, c): c[0] for c in todo}
        for f in cf.as_completed(futures):
            try:
                cid, seconds, records = f.result()
                summary = ", ".join(f"{'pruned' if r['pruned'] else 'unpruned'} {100 * r['test_accuracy']:.2f}% "
                                    f"{r['nodes']} nodes" for r in records)
                print(f"{time.strftime('%T')} done {cid} in {seconds:.0f} s: {summary}", flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"{time.strftime('%T')} FAILED {futures[f]}: {e}", flush=True)
    print(f"finished, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
