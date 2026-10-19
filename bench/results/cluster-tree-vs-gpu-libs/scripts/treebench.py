#!/usr/bin/env python3
"""./tree --cuda, REPS fresh processes per case; appends one JSON line per run
to OUT. The last run dumps the tree; its test predictions are saved for the
agreement table.

  treebench.py TREE DATASET CASE REPS OUT PRED_DIR SCRATCH
    DATASET  higgs | airline | ecbdl_15m | chess_1b   (prepared in $DATA_ROOT/<dataset>)
    CASE     cart | c45        depth <= 19, no pruning
"""
import json, os, re, socket, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpubench import DATA_ROOT, DEPTH, GpuMemory, load  # noqa: E402

FLAGS = {"cart": ["--cart", "--no-prune", "-d", str(DEPTH)],
         "c45": ["--c45", "--no-prune", "-d", str(DEPTH)]}


def parse_dump(path):
    """./tree --dump -> preorder arrays (feature, threshold, value, left, right)
    (bench/benchlib.py's parse_tree_dump)."""
    node_re = re.compile(r"^\s*\w+ \[n=\d+\]: (.*)$")
    feature, threshold, value = [], [], []
    for line in open(path):
        match = node_re.match(line)
        if not match:
            continue
        body = match.group(1)
        if body.startswith("Leaf -> "):
            feature.append(-1); threshold.append(0.0); value.append(int(body[len("Leaf -> c"):]))
        else:
            name, cut = re.match(r"if \"?f(\d+)\"? <= (\S+)$", body).groups()
            feature.append(int(name)); threshold.append(float(cut)); value.append(-1)
    feature = np.array(feature)
    count = len(feature)
    size = np.ones(count, dtype=np.int64)
    left = np.full(count, -1)
    right = np.full(count, -1)
    for node in range(count - 1, -1, -1):
        if feature[node] >= 0:
            left[node] = node + 1
            right[node] = node + 1 + size[node + 1]
            size[node] += size[left[node]] + size[right[node]]
    return feature, np.array(threshold), np.array(value), left, right


def predict(tree, X):
    feature, threshold, value, left, right = tree
    node = np.zeros(len(X), dtype=np.int64)
    active = np.arange(len(X))
    while len(active):
        f = feature[node[active]]
        inner = f >= 0
        active = active[inner]
        if not len(active):
            break
        current = node[active]
        goes_left = X[active, f[inner]].astype(np.float64) <= threshold[current]
        node[active] = np.where(goes_left, left[current], right[current])
    return value[node]


def main():
    binary, dataset, case, reps, out, pred_dir, scratch = sys.argv[1:8]
    threads = str(os.cpu_count())
    dump = f"{scratch}/tree-{case}.dump"
    for rep in range(int(reps)):
        last = rep == int(reps) - 1
        argv = [binary, "--cuda", "--threads", threads] + FLAGS[case] \
            + (["--dump", dump] if last else []) + [f"{DATA_ROOT}/{dataset}/train.csv"]
        gpu = GpuMemory()
        stdout = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
        peak = gpu.stop()
        nodes, leaves, depth = map(int, re.search(r"nodes (\d+), leaves (\d+), depth (\d+)",
                                                  stdout).groups())
        row = {"impl": "tree", "dataset": dataset, "case": case, "bins": "exact", "rep": rep,
               "train_seconds": float(re.search(r"train total\s+([\d.]+) ms", stdout).group(1)) / 1000,
               **{f"{k.replace(' ', '_')}_seconds": float(ms) / 1000 for k, ms in
                  re.findall(r"^  (gpu setup|presort|build|prune)\s+([\d.]+) ms$", stdout, re.M)},
               "train_accuracy": float(re.search(r"train accuracy\s+([\d.]+)%", stdout).group(1)) / 100,
               "nodes": nodes, "depth": depth, "gpu_peak_mib": peak,
               "host": socket.gethostname(), "time": time.strftime("%F %T"),
               "command": " ".join(argv)}
        if last:
            tree = parse_dump(dump)
            Xt, yt = load(dataset, "test")
            pred = predict(tree, Xt).astype(np.int8)
            np.save(f"{pred_dir}/tree-{case}.npy", pred)
            row["test_accuracy"] = float(np.mean(pred == yt))
            f, t = tree[0][0], tree[1][0]
            row["root"] = f"f{f} <= {t:.7g}"
            os.remove(dump)
        print(json.dumps(row), flush=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
