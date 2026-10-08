"""Shared paths, dataset files and protocol definitions for the CPU benchmark.

Data layout (bench/data/<dataset>/, made by prepare.py):

  meta.json                  features, classes, row counts, how the split was made
  train.f32, test.f32        features, row-major little-endian float32 (n x F)
  train.y.i32, test.y.i32    class index per row, little-endian int32
  train.csv                  header f0..f{F-1},class; labels c0..c{K-1}  (./tree)
  yadt.names, train.yadt.csv, test.yadt.csv   headerless CSV + metadata  (YaDT)

Every tool sees the same float32 values: the CSVs print them exactly (9
significant digits), the binary files hold them as they are.
"""

import json
import os
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
DATA = BENCH / "data"
TOOLS = BENCH / ".tools"
RESULTS = BENCH / "results"
TREE = Path(os.environ.get("TREE_BIN", ROOT / "tree_cpu"))
YADT = TOOLS / "yadt" / "dTcmd"
JAVA_CLASSPATH = f"{TOOLS / 'weka.jar'}:{TOOLS / 'bounce.jar'}:{TOOLS / 'classes'}"
BASELINE = "_baseline"  # tiny synthetic dataset: each tool's runtime footprint

# CART trees are capped at depth 30 for every tool: rpart cannot grow deeper,
# and a tree that only one tool may grow deeper is not the same work.
CART_DEPTH = 30

# ------------------------------------------------------------------ protocols
#
# A protocol is one well-defined piece of work that every listed implementation
# performs with equivalent settings. Values in braces are filled in per dataset
# by run.py's calibration step (cart_alpha).
#
# ./tree and YaDT entries are command-line flags; the others are key=value
# arguments of the adapters in bench/adapters/.

PROTOCOLS = {
    "cart_full": {
        "title": "CART, grown to pure leaves (depth <= 30), no pruning",
        "impls": {
            "tree": ["--cart", "--no-prune", "-d", str(CART_DEPTH)],
            "sklearn": {"mode": "fit", "max_depth": CART_DEPTH},
            "rpart": {"mode": "fit", "maxdepth": CART_DEPTH},
        },
    },
    "cart_depth12": {
        "title": "CART, grown to depth 12, no pruning",
        "impls": {
            "tree": ["--cart", "--no-prune", "-d", "12"],
            "sklearn": {"mode": "fit", "max_depth": 12},
            "rpart": {"mode": "fit", "maxdepth": 12},
        },
    },
    "cart_alpha": {
        "title": "CART, one tree pruned at a fixed alpha (cost-complexity)",
        "impls": {
            "tree": ["--cart", "-d", str(CART_DEPTH), "--alpha", "{alpha}"],
            "sklearn": {"mode": "fit", "max_depth": CART_DEPTH, "ccp_alpha": "{sk_ccp_alpha}"},
            "rpart": {"mode": "alpha", "maxdepth": CART_DEPTH, "alpha": "{alpha}"},
        },
    },
    "cart_cv10": {
        "title": "CART, alpha chosen by 10-fold CV with the 1-SE rule (whole procedure)",
        "impls": {
            "tree": ["--cart", "-d", str(CART_DEPTH), "--cv", "10"],
            "sklearn": {"mode": "cv", "folds": 10, "candidates": 16, "max_depth": CART_DEPTH},
            "rpart": {"mode": "cv", "folds": 10, "maxdepth": CART_DEPTH},
        },
    },
    "c45": {
        "title": "C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows",
        "impls": {
            "tree": ["--c45"],
            "j48": {"options": "-C 0.25 -M 2"},
            "yadt": ["-ebpg", "-c", "0.25", "-m", "2"],
        },
    },
    "c45_unpruned": {
        "title": "C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows",
        "impls": {
            "tree": ["--c45", "--no-prune"],
            "j48": {"options": "-U -M 2"},
            "yadt": ["-np", "-m", "2"],
        },
    },
}

# threads: "multi" = runs with every thread count asked for, "single" = only 1.
# warmup: what the adapter does before the timed fit, in the same process
#   "full"   one untimed fit on all rows (JIT compilation in the JVM)
#   "subset" one untimed fit on 2,000 rows (imports, first-call costs)
#   "none"   a native executable: run.py runs one untimed process instead
IMPLS = {
    "tree": {"label": "./tree", "threads": "multi", "warmup": "none"},
    "sklearn": {"label": "scikit-learn", "threads": "single", "warmup": "subset"},
    "rpart": {"label": "rpart", "threads": "single", "warmup": "subset"},
    "j48": {"label": "Weka J48", "threads": "single", "warmup": "full"},
    "yadt": {"label": "YaDT", "threads": "multi", "warmup": "none"},
}


def supports_threads(protocol, impl):
    """scikit-learn fits one tree on one thread; only its CV search is parallel."""
    if impl == "sklearn":
        return protocol == "cart_cv10"
    return IMPLS[impl]["threads"] == "multi"


# ------------------------------------------------------------------ datasets

def meta(dataset):
    return json.load(open(DATA / dataset / "meta.json"))


def load(dataset, part):
    m = meta(dataset)
    rows = m[f"n_{part}"]
    X = np.fromfile(DATA / dataset / f"{part}.f32", dtype="<f4").reshape(rows, len(m["features"]))
    y = np.fromfile(DATA / dataset / f"{part}.y.i32", dtype="<i4")
    return X, y


# ------------------------------------------------------------------ ./tree dumps

def parse_tree_dump(path, n_features):
    """./tree --dump -> flat preorder arrays (feature, threshold, value, left, right);
    leaves have feature -1. Features are named f<i>, classes c<i>."""
    import re
    node_re = re.compile(r"^\s*\w+ \[n=\d+\]: (.*)$")
    feature, threshold, value = [], [], []
    for line in open(path):
        match = node_re.match(line)
        if not match:
            continue
        body = match.group(1)
        if body.startswith("Leaf -> "):
            feature.append(-1)
            threshold.append(0.0)
            value.append(int(body[len("Leaf -> c"):]))
        else:
            name, cut = re.match(r"if f(\d+) <= (\S+)$", body).groups()
            assert int(name) < n_features
            feature.append(int(name))
            threshold.append(float(cut))
            value.append(-1)
    feature = np.array(feature)
    count = len(feature)
    size = np.ones(count, dtype=np.int64)
    left = np.full(count, -1)
    right = np.full(count, -1)
    # Preorder: the left child follows its parent, the right child follows the
    # left subtree; one backwards pass gives subtree sizes.
    for node in range(count - 1, -1, -1):
        if feature[node] >= 0:
            left[node] = node + 1
            right[node] = node + 1 + size[node + 1]
            size[node] = 1 + size[left[node]] + size[right[node]]
    return feature, np.array(threshold), np.array(value), left, right


def predict_tree(tree, X):
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
