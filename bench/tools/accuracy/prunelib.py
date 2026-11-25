"""Shared code of the pruned-accuracy benchmark (bench/tools/accuracy/).

A grown tree as flat arrays (any tool's: scikit-learn, cuML, XGBoost), routed
on the CPU with NumPy, and pruned by the one CART rule every CART tool gets
here: Breiman's minimal cost-complexity pruning at a fixed alpha, cost =
training misclassification rate + alpha x leaves (docs/ALGORITHMS.md, ./tree's
--prune for --cart; rpart's prune(cp = alpha n / root risk) is the same rule).

  Tree(left, right, feature, threshold, strict)
    left/right: child indices, -1 at a leaf; feature: split feature;
    threshold: the split value; strict: True = "x < t goes left" (XGBoost),
    False = "x <= t goes left" (scikit-learn, cuML).
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Tree:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    strict: bool = False


def route(tree, X, stop=None, chunk=2_000_000):
    """The node every row ends in: a leaf, or the first node with stop[node]
    True (a pruned subtree's root)."""
    out = np.empty(len(X), dtype=np.int64)
    leaf = tree.left < 0 if stop is None else (tree.left < 0) | stop
    for a in range(0, len(X), chunk):
        Xc = X[a:a + chunk]
        node = np.zeros(len(Xc), dtype=np.int64)
        active = np.arange(len(Xc))
        while len(active):
            n = node[active]
            inner = ~leaf[n]
            active, n = active[inner], n[inner]
            if not len(active):
                break
            value = Xc[active, tree.feature[n]].astype(np.float64)
            t = tree.threshold[n].astype(np.float64)
            go_left = value < t if tree.strict else value <= t
            node[active] = np.where(go_left, tree.left[n], tree.right[n])
        out[a:a + chunk] = node
    return out


def bfs(tree):
    """Reachable nodes in breadth-first order, their parent and depth."""
    order, parent, depth = [0], {0: -1}, {0: 0}
    for node in order:
        if tree.left[node] >= 0:
            for child in (tree.left[node], tree.right[node]):
                parent[int(child)] = node
                depth[int(child)] = depth[node] + 1
                order.append(int(child))
    return np.array(order), parent, depth


def node_counts(tree, X, y, classes):
    """Training rows per class at every node: (nodes x classes)."""
    leaves = route(tree, X)
    counts = np.zeros((len(tree.left), classes), dtype=np.int64)
    np.add.at(counts, (leaves, y), 1)
    order, parent, _ = bfs(tree)
    for node in order[::-1]:
        if parent[node] >= 0:
            counts[parent[node]] += counts[node]
    return counts


def ccp_prune(tree, counts, alpha):
    """Minimal cost-complexity pruning at a fixed alpha (misclassification
    rate, n = all training rows): the smallest subtree minimising
    errors + alpha n leaves; a node becomes a leaf when that costs no more
    than its best subtree (ties prune). -> boolean array: node is a pruned leaf."""
    n = counts[0].sum()
    penalty = alpha * n
    errors = counts.sum(axis=1) - counts.max(axis=1)
    order, _, _ = bfs(tree)
    sub_err = np.zeros(len(tree.left), dtype=np.int64)
    sub_leaves = np.zeros(len(tree.left), dtype=np.int64)
    stop = np.zeros(len(tree.left), dtype=bool)
    for node in order[::-1]:
        if tree.left[node] < 0:
            sub_err[node], sub_leaves[node] = errors[node], 1
            continue
        l, r = tree.left[node], tree.right[node]
        e, k = sub_err[l] + sub_err[r], sub_leaves[l] + sub_leaves[r]
        # leaf: errors[node] + penalty; subtree: e + penalty k
        if errors[node] - e <= penalty * (k - 1):
            stop[node], sub_err[node], sub_leaves[node] = True, errors[node], 1
        else:
            sub_err[node], sub_leaves[node] = e, k
    return stop


def size(tree, stop=None):
    """(nodes, leaves, depth) of the tree, cut at the stop nodes."""
    order, depth = [0], {0: 0}
    for node in order:
        if tree.left[node] >= 0 and not (stop is not None and stop[node]):
            for child in (tree.left[node], tree.right[node]):
                depth[int(child)] = depth[node] + 1
                order.append(int(child))
    leaves = sum(1 for node in order if tree.left[node] < 0 or (stop is not None and stop[node]))
    return len(order), leaves, max(depth.values())


def evaluate(tree, counts, X, y, Xt, stop=None):
    """Each row gets the majority training class of the node it ends in.
    -> (train accuracy, test predictions int16)."""
    label = counts.argmax(axis=1)
    train = float(np.mean(label[route(tree, X, stop)] == y))
    return train, label[route(tree, Xt, stop)].astype(np.int16)


# ------------------------------------------------------------------ statistics

def wilson(correct, n, z=1.959964):
    """95% Wilson interval of a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = correct / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(a_right, b_right):
    """Exact-continuity McNemar test of two classifiers on the same rows:
    (rows only a got right, rows only b got right, two-sided p value)."""
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))  # chi-square, 1 degree of freedom
    return b, c, p
