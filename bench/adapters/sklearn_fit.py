"""scikit-learn adapter: fit one protocol on a prepared dataset, print one JSON line.

  python sklearn_fit.py data=DIR n_train=N n_test=M n_features=F threads=T \
      warmup=subset|none eval=0|1 mode=fit|cv [max_depth=D] [ccp_alpha=A] \
      [folds=K] [candidates=C]

Timed: DecisionTreeClassifier.fit (mode=fit), or the whole alpha search
(mode=cv): the pruning path of a full tree, a GridSearchCV over `candidates`
ccp_alpha values from it with K folds, the 1-SE choice and the refit. The
search runs its fits on `threads` threads in this one process (joblib's
threading backend; the tree builder releases the GIL), so the process's peak
memory covers all of them. Loading and prediction are not timed.
"""

import json
import sys
import time

import numpy as np
from joblib import parallel_config
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


def candidate_alphas(X, y, count):
    """0 plus count-1 alphas whose pruned trees have geometrically spaced sizes,
    each the geometric midpoint of a pruning-path interval."""
    path = np.unique(tree().cost_complexity_pruning_path(X, y).ccp_alphas)
    path = path[path > 0]
    if len(path) < 2:
        return [0.0]
    steps = np.unique(np.geomspace(1, len(path) - 1, count - 1).astype(int))
    return [0.0] + sorted(np.sqrt(path[len(path) - steps] * path[len(path) - steps - 1]).tolist())


def one_standard_error(folds):
    """GridSearchCV refit callable: the largest alpha (smallest tree) whose CV
    error is within one standard error of the lowest."""
    def choose(results):
        error = 1 - results["mean_test_score"]
        se = results["std_test_score"] / np.sqrt(folds)
        best = int(np.argmin(error))
        alphas = np.array(results["param_ccp_alpha"], dtype=float)
        ok = np.flatnonzero(error <= error[best] + se[best])
        return int(ok[np.argmax(alphas[ok])])
    return choose


def fit(X, y):
    if args["mode"] == "fit":
        extra = {"ccp_alpha": float(args["ccp_alpha"])} if "ccp_alpha" in args else {}
        return tree(**extra).fit(X, y)
    folds = int(args["folds"])
    grid = {"ccp_alpha": candidate_alphas(X, y, int(args["candidates"]))}
    search = GridSearchCV(tree(), grid, cv=StratifiedKFold(folds, shuffle=True, random_state=1),
                          refit=one_standard_error(folds), n_jobs=int(args["threads"]))
    with parallel_config(backend="threading"):
        return search.fit(X, y).best_estimator_


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
