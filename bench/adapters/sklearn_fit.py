"""scikit-learn adapter: fit one tree on a prepared dataset, print one JSON line.

  python sklearn_fit.py data=DIR n_train=N n_test=M n_features=F threads=T \
      warmup=subset|none eval=0|1 mode=fit [max_depth=D] [ccp_alpha=A]

Timed: DecisionTreeClassifier.fit (one tree, on one thread: scikit-learn does
not parallelise a single tree). Loading and prediction are not timed.
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
if args["warmup"] == "subset":
    fit(X[:2000], y[:2000])
start = time.perf_counter()
model = fit(X, y)
seconds = time.perf_counter() - start

result = {"train_seconds": seconds, "nodes": int(model.tree_.node_count),
          "leaves": int(model.get_n_leaves()), "depth": int(model.get_depth())}
if args["eval"] == "1":
    X_test, y_test = read("test", int(args["n_test"]))
    result["test_accuracy"] = float(np.mean(model.predict(X_test) == y_test))
print(json.dumps(result))
