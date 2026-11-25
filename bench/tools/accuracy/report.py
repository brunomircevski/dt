#!/usr/bin/env python3
"""Tables of the pruned-accuracy benchmark from acc.py's case records.

  report.py RUNS_DIR [--md FILE] [--csv FILE]

Per dataset, algorithm and depth limit, every tool's pruned and unpruned tree:
test accuracy with its 95% Wilson interval, balanced accuracy (the mean
recall of the classes: ECBDL'14 is 98% one class), nodes, leaves, depth, training
accuracy; against ./tree's tree of the same pruning: the share of test rows
both predict alike (agreement) and McNemar's test (rows only ./tree got right,
rows only the tool got right, p). ./tree's backends must give one tree
(tree_sha256) for every pruned case.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prunelib as P  # noqa: E402

LABEL = {"tree": "./tree", "sklearn": "scikit-learn", "rpart": "rpart", "cuml": "cuML",
         "xgboost": "XGBoost", "j48": "Weka J48", "yadt": "YaDT"}
ORDER = list(LABEL)
TESTS = {}


def load_pred(path):
    path = Path(path)
    return np.load(path) if path.suffix == ".npy" else np.fromfile(path, dtype="<i2")


def test_labels(dataset, n):
    if dataset not in TESTS:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import benchlib as B
        TESTS[dataset] = B.load(dataset, "test")[1]
    assert len(TESTS[dataset]) == n
    return TESTS[dataset]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    records = [r for f in sorted(a.runs.glob("*/cases/*.json")) for r in json.loads(f.read_text())]
    # ./tree's backends: one tree per (dataset, algo, depth, pruned)
    hashes = defaultdict(dict)
    for r in records:
        if r["tool"] == "tree":
            hashes[(r["dataset"], r["algo"], r["depth_limit"], r["pruned"])][r["backend"]] = r["tree_sha256"]
    lines, rows = [], []
    mismatch = [(k, v) for k, v in hashes.items() if len(set(v.values())) > 1]
    lines.append("./tree backends: " + ("every backend grew the same tree in every case "
                 f"({', '.join(sorted({b for v in hashes.values() for b in v}))})" if not mismatch
                 else f"DIFFERENT TREES: {mismatch}"))
    main_records = [r for r in records if r["tool"] != "tree" or r["backend"] == "multicore"]
    groups = defaultdict(list)
    for r in main_records:
        groups[(r["dataset"], r["algo"], r["depth_limit"])].append(r)
    paper = ["susy", "higgs", "mnist8m", "ecbdl_15m", "sorel_4m"]  # the paper's order, CART first
    for (dataset, algo, depth) in sorted(groups, key=lambda k: (
            paper.index(k[0]) if k[0] in paper else len(paper), k[0], k[1] != "cart", k[2])):
        group = groups[(dataset, algo, depth)]
        y = test_labels(dataset, group[0]["n_test"])
        ref = {r["pruned"]: r for r in group if r["tool"] == "tree"}
        lines.append(f"\n### {dataset}, {'CART' if algo == 'cart' else 'C4.5'}, depth ≤ {depth}\n")
        lines.append("| Tool | Pruned | Test acc (95% CI) | Balanced acc | Nodes | Leaves | Depth | Train acc | "
                     "Agree w/ ./tree | Only ./tree / only tool right | McNemar p |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        group.sort(key=lambda r: (not r["pruned"], ORDER.index(r["tool"])))
        for r in group:
            pred = load_pred(r["pred"])
            right = pred == y
            lo, hi = P.wilson(int(right.sum()), len(y))
            balanced = float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))
            agree = b = c = p = None
            if r["pruned"] in ref and r["tool"] != "tree":
                tp = load_pred(ref[r["pruned"]]["pred"])
                agree = float(np.mean(tp == pred))
                b, c, p = P.mcnemar(tp == y, right)
            row = {"dataset": dataset, "algo": algo, "depth_limit": depth, "tool": r["tool"],
                   "pruned": r["pruned"], "prune": r["prune"], "test_accuracy": r["test_accuracy"],
                   "balanced_accuracy": balanced, "ci_low": lo, "ci_high": hi, "nodes": r["nodes"], "leaves": r["leaves"],
                   "depth": r["depth"], "train_accuracy": r["train_accuracy"], "agreement_with_tree": agree,
                   "only_tree_right": b, "only_tool_right": c, "mcnemar_p": p,
                   "own_predict_agreement": r.get("own_predict_agreement"), "version": r.get("version"),
                   "host": r.get("host")}
            rows.append(row)
            lines.append(f"| {LABEL[r['tool']]} | {'yes' if r['pruned'] else 'no'} | "
                         f"{100 * r['test_accuracy']:.2f}% ({100 * lo:.2f}–{100 * hi:.2f}) | {100 * balanced:.2f}% | {r['nodes']:,} | "
                         f"{r['leaves']:,} | {r['depth']} | {100 * r['train_accuracy']:.2f}% | "
                         + (f"{100 * agree:.2f}% | {b:,} / {c:,} | {p:.3g} |" if agree is not None
                            else "– | – | – |"))
    text = "\n".join(lines) + "\n"
    print(text)
    if a.md:
        a.md.write_text(text)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
