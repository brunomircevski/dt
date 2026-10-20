"""The tree format every adapter writes, and the code that reads it.

A tree is nested dicts, root first:
  split: {"feature": "x", "op": "<=" | "<", "threshold": 1.0, "n": rows,
          "class": majority class or None, "left": {...}, "right": {...}}
         ("left" holds the rows for which `feature op threshold` is true)
  leaf:  {"class": "A", "n": rows or None, "dist": {"A": rows, "B": rows} or None}

A result file (<impl>-<case>.json) is
  {"meta": {...the run: impl, case, host, times, settings...}, "tree": {...}}
and predict()/stats() here give every library's depth, node count and
accuracies by the same walk.
"""
import json
import numpy as np

DATA_DIR_DEFAULT = "data"
FEATURES = ["x", "y"]
CLASSES = ["A", "B"]


def load(data, part, dtype="f32"):
    """X (rows x 2) and the class index (0 = A, 1 = B) of train or test.
    dtype f32: the float32 values every binary-fed library sees; f64: the
    CSV text parsed as double (YaDT)."""
    meta = json.load(open(f"{data}/meta.json"))
    n = meta[f"n_{part}"]
    y = np.fromfile(f"{data}/{part}.y.i32", dtype="<i4")
    if dtype == "f64":
        X = np.load(f"{data}/{part}.f64.npy")
    else:
        X = np.fromfile(f"{data}/{part}.f32", dtype="<f4").reshape(n, len(FEATURES))
    return X, y


def is_leaf(node):
    return "feature" not in node


def predict(tree, X):
    """Class index per row of X, walking the tree."""
    out = np.empty(len(X), dtype=np.int64)
    stack = [(tree, np.arange(len(X)))]
    while stack:
        node, rows = stack.pop()
        if not len(rows):
            continue
        if is_leaf(node):
            out[rows] = CLASSES.index(node["class"])
            continue
        values = X[rows, FEATURES.index(node["feature"])].astype(np.float64)
        go = values <= node["threshold"] if node["op"] == "<=" else values < node["threshold"]
        stack.append((node["left"], rows[go]))
        stack.append((node["right"], rows[~go]))
    return out


def stats(tree):
    """nodes, leaves, depth (root = depth 0, as every tool here counts it)."""
    nodes = leaves = depth = 0
    stack = [(tree, 0)]
    while stack:
        node, d = stack.pop()
        nodes += 1
        depth = max(depth, d)
        if is_leaf(node):
            leaves += 1
        else:
            stack += [(node["left"], d + 1), (node["right"], d + 1)]
    return nodes, leaves, depth


def fill_counts(tree, X, y):
    """Set n, dist and the majority class of every node from the training
    rows, for tools that do not export them (./tree's dump has only n)."""
    stack = [(tree, np.arange(len(X)))]
    while stack:
        node, rows = stack.pop()
        counts = np.bincount(y[rows], minlength=len(CLASSES))
        node["n"] = int(len(rows))
        node["dist"] = {c: int(k) for c, k in zip(CLASSES, counts)}
        if is_leaf(node):
            continue
        node.setdefault("class", CLASSES[int(np.argmax(counts))] if len(rows) else None)
        values = X[rows, FEATURES.index(node["feature"])].astype(np.float64)
        go = values <= node["threshold"] if node["op"] == "<=" else values < node["threshold"]
        stack.append((node["left"], rows[go]))
        stack.append((node["right"], rows[~go]))


def finish(meta, tree, data, own_test_pred=None, own_train_pred=None):
    """Adds the walk's statistics and accuracies to meta. If the library's own
    predictions are given, the walk must agree with them on every row: that
    checks the export."""
    dtype = meta.get("dtype", "f32")
    X, y = load(data, "train", dtype)
    Xt, yt = load(data, "test", dtype)
    if any("dist" not in n or n.get("dist") is None for n in iter_nodes(tree)):
        fill_counts(tree, X, y)
    pred, pred_t = predict(tree, X), predict(tree, Xt)
    for name, own, walk in (("train", own_train_pred, pred), ("test", own_test_pred, pred_t)):
        if own is not None:
            differ = int(np.sum(np.asarray(own) != walk))
            meta[f"{name}_export_mismatches"] = differ
            if differ:
                print(f"warning: the exported tree predicts {differ} {name} rows differently "
                      f"from {meta['impl']}'s own predict()")
    nodes, leaves, depth = stats(tree)
    meta.update(nodes=nodes, leaves=leaves, depth=depth,
                train_accuracy=float(np.mean(pred == y)), test_accuracy=float(np.mean(pred_t == yt)),
                n_train=int(len(y)), n_test=int(len(yt)))
    return meta


def iter_nodes(tree):
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        if not is_leaf(node):
            stack += [node["left"], node["right"]]


def save(path, meta, tree):
    with open(path, "w") as f:
        json.dump({"meta": meta, "tree": tree}, f)
    print(f"{path}: {meta['nodes']} nodes, depth {meta['depth']}, "
          f"test {100 * meta['test_accuracy']:.2f}%, median {meta['median_seconds']:.4f} s")
