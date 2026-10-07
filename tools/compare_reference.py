#!/usr/bin/env python3
"""Compare this project's trees node by node with reference implementations.

  CART vs scikit-learn:
    python3 tools/compare_reference.py cart datasets/iris.csv
  C4.5 vs Quinlan's original c4.5 (Release 8) binary:
    python3 tools/compare_reference.py c45 datasets/iris.csv --c45 /path/to/c4.5

Any other arguments are passed on to ./tree (e.g. -d 5, --min-leaf 3,
--cf 0.1); --min-leaf becomes c4.5's -m. Set TREE_BIN to use another binary (e.g. ./tree_cpu).

Both trees are walked together. Where they pick different splits, the impurity
decrease of both splits is recomputed on the rows of that node: if it is equal,
the two implementations just broke an exact tie differently (scikit-learn
visits features in random order) and the subtrees below are not compared;
otherwise it is reported as a real difference.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np


# Tree model: ("split", feature, threshold, left, right, n) | ("leaf", cls, n)

def parse_own_dump(path):
    """Parse the indented text written by ./tree --dump."""
    node_re = re.compile(r"^(\s*)(\w+) \[n=(\d+)\]: (.*)$")
    lines = [line.rstrip("\n") for line in open(path) if line.strip()]
    position = 0

    def parse():
        nonlocal position
        match = node_re.match(lines[position])
        position += 1
        count = int(match.group(3))
        body = match.group(4)
        if body.startswith("Leaf -> "):
            return ("leaf", body[len("Leaf -> "):], count)
        feature, threshold = re.match(r"if (.+) <= (\S+)$", body).groups()
        left = parse()
        right = parse()
        return ("split", feature, float(threshold), left, right, count)

    return parse()


def node_count(tree):
    return 1 if tree[0] == "leaf" else 1 + node_count(tree[3]) + node_count(tree[4])


class Comparison:
    def __init__(self, X, y, features, criterion, min_objs=2):
        self.X, self.y, self.criterion, self.min_objs = X, y, criterion, min_objs
        self.feature_index = {name: index for index, name in enumerate(features)}
        self.ties = []
        self.differences = []

    def impurity(self, labels):
        if len(labels) == 0:
            return 0.0
        p = np.unique(labels, return_counts=True)[1] / len(labels)
        if self.criterion == "gini":
            return 1.0 - float(np.sum(p * p))
        return float(-np.sum(p * np.log2(p)))

    def c45_ratio(self, rows, feature, threshold):
        """C4.5's gain ratio of `feature <= threshold` at a node (contin.c, build.c)."""
        values = self.X[rows, self.feature_index[feature]].astype(np.float64)
        order = np.argsort(values, kind="stable")
        values, labels = values[order], self.y[rows][order]
        n, classes = len(rows), np.unique(self.y)
        min_split = np.float32(0.10 * n / len(classes))
        min_split = self.min_objs if min_split <= self.min_objs else min(float(min_split), 25.0)
        min_split = max(min_split, self.min_objs)
        low = np.arange(1, n)
        allowed = (low >= min_split) & (low <= n - min_split) & (values[:-1] < values[1:] - 1e-5)
        tries = int(np.sum(allowed))
        info = lambda counts: (counts.sum() * np.log2(counts.sum()) - np.sum(
            counts[counts > 0] * np.log2(counts[counts > 0]))) if counts.sum() else 0.0
        count = lambda part: np.array([np.sum(part == c) for c in classes], dtype=float)
        left = values <= threshold
        gain = (info(count(labels)) - info(count(labels[left])) - info(count(labels[~left]))) / n
        gain -= np.log2(tries) / n
        split_info = info(np.array([left.sum(), (~left).sum()], dtype=float)) / n
        return gain / split_info

    def gain(self, rows, feature, threshold):
        if self.criterion == "c45":
            return self.c45_ratio(rows, feature, threshold)
        goes_left = self.X[rows, self.feature_index[feature]] <= threshold
        left, right = rows[goes_left], rows[~goes_left]
        n = len(rows)
        return (self.impurity(self.y[rows]) - len(left) / n * self.impurity(self.y[left])
                - len(right) / n * self.impurity(self.y[right]))

    def walk(self, ours, ref, rows, path="ROOT"):
        same_split = (ours[0] == ref[0] == "split" and ours[1] == ref[1]
                      and abs(ours[2] - ref[2]) <= 1e-4 * max(1.0, abs(ref[2]))
                      and ours[5] == ref[5])
        if ours[0] == ref[0] == "leaf":
            if ours[1] != ref[1] or ours[2] != ref[2]:
                self.differences.append(f"{path}: leaf ours={ours[1]}/{ours[2]} "
                                        f"ref={ref[1]}/{ref[2]}")
            return
        if not same_split:
            if ours[0] == ref[0] == "split" and self.criterion:
                ours_gain = self.gain(rows, ours[1], ours[2])
                ref_gain = self.gain(rows, ref[1], ref[2])
                # c4.5 computes in single precision: equal up to float noise.
                tolerance = 1e-5 * abs(ref_gain) if self.criterion == "c45" else 1e-9
                if abs(ours_gain - ref_gain) <= tolerance:
                    self.ties.append(path)
                    return
                detail = f" (gain ours {ours_gain:.6g}, ref {ref_gain:.6g})"
            else:
                detail = ""
            describe = lambda t: (f"{t[1]}<={t[2]} [n={t[5]}]" if t[0] == "split"
                                  else f"leaf {t[1]}/{t[2]}")
            self.differences.append(f"{path}: ours {describe(ours)}, ref {describe(ref)}{detail}")
            return
        goes_left = self.X[rows, self.feature_index[ours[1]]] <= ours[2]
        self.walk(ours[3], ref[3], rows[goes_left], path + "/L")
        self.walk(ours[4], ref[4], rows[~goes_left], path + "/R")


def load_csv(path):
    with open(path) as handle:
        header = handle.readline().strip().split(",")
        rows = [line.strip().split(",") for line in handle if line.strip()]
    first = 1 if header[0].lower() == "id" else 0
    features = header[first:-1]
    X = np.array([[float(value) for value in row[first:-1]] for row in rows], dtype=np.float32)
    y = np.array([row[-1] for row in rows])
    return features, X, y


def run_own(dataset, algorithm, extra, dump_path):
    command = [os.environ.get("TREE_BIN", "./tree"), f"--{algorithm}", "--serial", dataset,
               "--dump", dump_path] + extra
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(result.stdout + result.stderr)
    return parse_own_dump(dump_path)


def unpruned(extra):
    """`extra` with --no-prune instead of its pruning options (./tree refuses both)."""
    kept = []
    flags = iter(extra)
    for flag in flags:
        if flag in ("--cf", "--alpha", "--test-sample", "--cv"):
            next(flags, None)  # its value
        elif flag != "--no-prune":
            kept.append(flag)
    return kept + ["--no-prune"]


def sklearn_tree(features, X, y, extra):
    from sklearn.tree import DecisionTreeClassifier

    kwargs = {"criterion": "gini", "random_state": 0}
    for flag, name in (("-d", "max_depth"), ("--min-leaf", "min_samples_leaf")):
        if flag in extra:
            kwargs[name] = int(extra[extra.index(flag) + 1])
    model = DecisionTreeClassifier(**kwargs).fit(X, y)
    tree = model.tree_
    classes = list(model.classes_)

    def build(node):
        count = int(tree.n_node_samples[node])
        if tree.children_left[node] == tree.children_right[node]:
            return ("leaf", classes[int(np.argmax(tree.value[node][0]))], count)
        return ("split", features[tree.feature[node]], float(tree.threshold[node]),
                build(tree.children_left[node]), build(tree.children_right[node]), count)

    return build(0)


def c45_trees(features, X, y, c45_binary, workdir, extra):
    stem = os.path.join(workdir, "DF")
    with open(stem + ".names", "w") as names:
        names.write(", ".join(sorted(set(y))) + ".\n")
        for feature in features:
            names.write(f"{feature}: continuous.\n")
    with open(stem + ".data", "w") as data:
        for row, label in zip(X, y):
            data.write(",".join(repr(float(value)) for value in row) + f",{label}\n")
    command = [c45_binary, "-f", stem]
    if "--min-leaf" in extra:
        command += ["-m", extra[extra.index("--min-leaf") + 1]]
    if "--cf" in extra:
        command += ["-c", f"{float(extra[extra.index('--cf') + 1]) * 100:g}"]
    subprocess.run(command, capture_output=True, text=True, check=True)
    classes = sorted(set(y))
    return (read_c45_tree(stem + ".unpruned", features, classes),
            read_c45_tree(stem + ".tree", features, classes))


def read_c45_tree(path, features, classes):
    """Read the binary tree c4.5 saves (trees.c OutTree; continuous tests only)."""
    import struct
    data = open(path, "rb").read()
    position = 1  # SaveTree writes a '\n' first

    def take(fmt):
        nonlocal position
        values = struct.unpack_from(fmt, data, position)
        position += struct.calcsize(fmt)
        return values

    def node():
        node_type, leaf = take("<hh")
        items, _errors = take("<ff")
        take(f"<{len(classes)}f")  # class distribution
        if node_type == 0:
            return ("leaf", classes[leaf], int(round(items)))
        assert node_type == 2, "only continuous threshold tests are supported"
        tested, forks = take("<hh")
        cut, _lower, _upper = take("<fff")
        assert forks == 2
        left = node()
        right = node()
        return ("split", features[tested], float(cut), left, right, int(round(items)))

    return node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=["cart", "c45"])
    parser.add_argument("dataset")
    parser.add_argument("--c45", help="path to the original c4.5 binary")
    args, extra = parser.parse_known_args()

    features, X, y = load_csv(args.dataset)
    with tempfile.TemporaryDirectory() as workdir:
        dump = os.path.join(workdir, "ours.txt")
        if args.algorithm == "cart":
            pairs = [("CART vs scikit-learn",
                      run_own(args.dataset, "cart", unpruned(extra), dump),
                      sklearn_tree(features, X, y, extra), "gini")]
        else:
            if not args.c45:
                sys.exit("--c45 /path/to/c4.5 is required")
            unpruned_ref, pruned_ref = c45_trees(features, X, y, args.c45, workdir, extra)
            pairs = [("C4.5 unpruned vs c4.5", run_own(args.dataset, "c45", unpruned(extra), dump),
                      unpruned_ref, "c45"),
                     ("C4.5 pruned   vs c4.5", run_own(args.dataset, "c45", extra, dump),
                      pruned_ref, "c45")]

    failed = False
    for title, ours, ref, criterion in pairs:
        min_objs = int(extra[extra.index("--min-leaf") + 1]) if "--min-leaf" in extra else 2
        comparison = Comparison(X, y, features, criterion, min_objs)
        comparison.walk(ours, ref, np.arange(len(y)))
        status = "DIFFERENT" if comparison.differences else (
            "IDENTICAL" if not comparison.ties else "EQUIVALENT")
        print(f"{title}: {status} (ours {node_count(ours)} nodes, reference "
              f"{node_count(ref)} nodes, exact ties broken differently: {len(comparison.ties)})")
        for difference in comparison.differences[:5]:
            print("   ", difference)
        failed |= bool(comparison.differences)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
