"""scikit-learn adapter: fit one tree on a prepared dataset, print one JSON line.

  python sklearn_fit.py data=DIR n_train=N n_test=M n_features=F threads=T \
      warmup=ROWS eval=0|1 mode=fit [max_depth=D] [ccp_alpha=A]

warmup=ROWS: one untimed fit on the first ROWS training rows first (0 = none).
Timed: DecisionTreeClassifier.fit (one tree, on one thread: scikit-learn does
not parallelise a single tree). Loading and prediction are not timed; with
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
    return DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=0, **extra)


def fit(X, y):
    extra = {"ccp_alpha": float(args["ccp_alpha"])} if "ccp_alpha" in args else {}
    return tree(**extra).fit(X, y)


X, y = read("train", int(args["n_train"]))
warmup = int(args["warmup"])
if warmup:
    fit(X[:warmup], y[:warmup])
start = time.perf_counter()
model = fit(X, y)
seconds = time.perf_counter() - start
peak_kib = next(int(line.split()[1]) for line in open("/proc/self/status")
                if line.startswith("VmHWM:"))

result = {"train_seconds": seconds, "nodes": int(model.tree_.node_count),
          "leaves": int(model.get_n_leaves()), "depth": int(model.get_depth()),
          "n_train_loaded": int(X.shape[0]), "n_features_loaded": int(X.shape[1]),
          "peak_rss_train_bytes": peak_kib * 1024}
if args["eval"] == "1":
    result["train_accuracy"] = float(np.mean(model.predict(X) == y))
    X_test, y_test = read("test", int(args["n_test"]))
    result["test_accuracy"] = float(np.mean(model.predict(X_test) == y_test))
print(json.dumps(result))
