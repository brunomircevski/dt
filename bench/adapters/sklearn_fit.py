"""scikit-learn adapter: fit one tree on a prepared dataset, print one JSON line.

  python sklearn_fit.py data=DIR n_train=N n_test=M n_features=F threads=T \
      warmup=ROWS eval=0|1 mode=fit [max_depth=D] [ccp_alpha=A] [criterion=gini|entropy] \
      [min_samples_leaf=M]

warmup=ROWS: one untimed fit on the first ROWS training rows first (0 = none).
Timed: DecisionTreeClassifier.fit (one tree, on one thread: scikit-learn does
not parallelise a single tree). With ccp_alpha the two parts of fit are timed
apart: growing (fit with ccp_alpha=0) and pruning (_prune_tree() at ccp_alpha,
what fit does last), build_seconds and prune_seconds; train_seconds is their
sum. Loading and prediction are not timed; with
eval=1 the training and test accuracy are measured after the timed fit.
The peak resident memory (VmHWM) is read right after the fit, before evaluation.
"""

import json
import sys
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

args = dict(arg.split("=", 1) for arg in sys.argv[1:])
F = int(args["n_features"])


def read(part, rows):
    X = np.fromfile(f"{args['data']}/{part}.f32", dtype="<f4").reshape(rows, F)
    y = np.fromfile(f"{args['data']}/{part}.y.i32", dtype="<i4")
    return X, y


def tree(**extra):
    depth = int(args["max_depth"]) if "max_depth" in args else None
    return DecisionTreeClassifier(criterion=args.get("criterion", "gini"), max_depth=depth,
                                  min_samples_leaf=int(args.get("min_samples_leaf", 1)),
                                  random_state=0, **extra)


def fit(X, y):
    """-> (model, seconds growing, seconds pruning or None)."""
    start = time.perf_counter()
    model = tree().fit(X, y)
    grown = time.perf_counter()
    if "ccp_alpha" not in args:
        return model, grown - start, None
    model.set_params(ccp_alpha=float(args["ccp_alpha"]))
    model._prune_tree()
    return model, grown - start, time.perf_counter() - grown


X, y = read("train", int(args["n_train"]))
warmup = int(args["warmup"])
if warmup:
    fit(X[:warmup], y[:warmup])
model, build, prune = fit(X, y)
seconds = build + (prune or 0.0)
peak_kib = next(int(line.split()[1]) for line in open("/proc/self/status")
                if line.startswith("VmHWM:"))

result = {"train_seconds": seconds, "build_seconds": build,
          **({"prune_seconds": prune} if prune is not None else {}),
          "nodes": int(model.tree_.node_count),
          "leaves": int(model.get_n_leaves()), "depth": int(model.get_depth()),
          "n_train_loaded": int(X.shape[0]), "n_features_loaded": int(X.shape[1]),
          "peak_rss_train_bytes": peak_kib * 1024}
if args["eval"] == "1":
    result["train_accuracy"] = float(np.mean(model.predict(X) == y))
    X_test, y_test = read("test", int(args["n_test"]))
    result["test_accuracy"] = float(np.mean(model.predict(X_test) == y_test))
print(json.dumps(result))
