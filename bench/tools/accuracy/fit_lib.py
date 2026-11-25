#!/usr/bin/env python3
"""Grow one CART tree with a library, then report it twice: as grown
(unpruned) and pruned by the benchmark's one CART rule (prunelib.ccp_prune:
minimal cost-complexity, misclassification rate, fixed alpha), the rule ./tree
--prune and rpart apply natively.

  fit_lib.py IMPL DATASET DEPTH ALPHA OUT_JSON PRED_PREFIX
    IMPL   sklearn | cuml | xgboost

The library's tree is exported to flat arrays and every row is routed by
prunelib (the library's own prediction of the grown tree is compared:
own_predict_agreement); a leaf predicts its training rows' majority class.
Library settings as in the timing benchmarks (bench/tools/adapters/sklearn_fit.py,
bench/tools/cluster/gpubench.py): Gini, all rows and features, depth <= DEPTH,
min 1 row per leaf; cuML 128 and XGBoost 256 histogram bins.
Writes OUT_JSON (a list of two records) and PRED_PREFIX-{unpruned,pruned}.npy
(int16 test predictions). Time is recorded (fit_seconds) but not benchmarked.
"""
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cluster"))
import prunelib as P  # noqa: E402


def sklearn_tree(X, y, depth):
    from sklearn.tree import DecisionTreeClassifier
    import sklearn
    model = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=0).fit(X, y)
    t = model.tree_
    tree = P.Tree(t.children_left.astype(np.int64), t.children_right.astype(np.int64),
                  np.maximum(t.feature, 0).astype(np.int64), t.threshold.astype(np.float64), strict=False)
    return tree, model.predict, f"scikit-learn {sklearn.__version__}"


def cuml_tree(X, y, depth):
    import cupy as cp
    import cuml
    from cuml.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=1, bootstrap=False, max_samples=1.0, max_features=1.0,
                                   split_criterion="gini", max_depth=depth, max_leaves=-1, n_bins=128,
                                   min_samples_leaf=1, min_samples_split=2, min_impurity_decrease=0.0,
                                   n_streams=1, random_state=0, output_type="numpy")
    model.fit(cp.asarray(np.asfortranarray(X)), cp.asarray(y))
    t = model.as_treelite().get_tree_accessor(0)
    field = lambda name: np.asarray(t.get_field(name))
    cmp = field("cmp")
    assert set(np.unique(cmp[field("cleft") >= 0])) <= {3}, "cuML: a comparison other than <="
    tree = P.Tree(field("cleft").astype(np.int64), field("cright").astype(np.int64),
                  np.maximum(field("split_index"), 0).astype(np.int64),
                  field("threshold").astype(np.float64), strict=False)

    def predict(A):
        return np.concatenate([model.predict(A[i:i + 1_000_000]) for i in range(0, len(A), 1_000_000)])
    return tree, predict, f"cuML {cuml.__version__}"


def xgboost_tree(X, y, depth):
    import cupy as cp
    import xgboost as xgb
    classes = int(y.max()) + 1
    labels = y.astype(np.float32) if classes == 2 else np.eye(classes, dtype=np.float32)[y]
    dev = cp.asarray(np.asfortranarray(X))
    dtrain = xgb.QuantileDMatrix(dev, label=cp.asarray(labels), max_bin=256)
    del dev  # XGBoost keeps its bins; the raw copy goes (ECBDL'14 15M: 38 GB)
    cp.get_default_memory_pool().free_all_blocks()
    # max_cached_hist_node: how many nodes' histograms XGBoost keeps (a cache:
    # speed only, the tree is the same); the default 65536 runs out of the
    # 80 GB at depth 15 (SOREL 4M: 64 GB for one level's histograms)
    multi = ({"multi_strategy": "multi_output_tree", "max_cached_hist_node": 16} if classes > 2
             else {"max_cached_hist_node": 128})
    params = {"device": "cuda", "tree_method": "hist", "objective": "reg:squarederror", **multi,
              "eta": 1.0, "lambda": 0.0, "alpha": 0.0, "gamma": 0.0, "min_child_weight": 1.0,
              **({"base_score": float(y.mean())} if classes == 2 else {}), "max_bin": 256,
              "subsample": 1.0, "colsample_bytree": 1.0, "max_depth": depth, "grow_policy": "depthwise"}
    booster = xgb.train(params, dtrain, num_boost_round=1)
    j = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"][0]
    left = np.array(j["left_children"], dtype=np.int64)
    tree = P.Tree(left, np.array(j["right_children"], dtype=np.int64),
                  np.array(j["split_indices"], dtype=np.int64),
                  # XGBoost compares float32 values with float32 thresholds; the JSON
                  # prints them as short decimals, which as float64 sit above or below
                  # the float32 (a row equal to the threshold would go left, not right)
                  np.array(j["split_conditions"], dtype=np.float32).astype(np.float64), strict=True)

    def predict(A):
        p = np.concatenate([cp.asnumpy(booster.inplace_predict(cp.asarray(A[i:i + 1_000_000])))
                            for i in range(0, len(A), 1_000_000)])
        return p.argmax(axis=1) if p.ndim == 2 else (p > 0.5).astype(np.int64)
    return tree, predict, f"XGBoost {xgb.__version__}"


def main():
    impl, dataset, depth, alpha, out_json, pred_prefix = sys.argv[1:7]
    depth, alpha = int(depth), float(alpha)
    os.environ.setdefault("DATA_ROOT", str(HERE.parent / "data"))
    import gpubench  # load(): the prepared files, as every GPU run reads them
    X, y = gpubench.load(dataset, "train")
    Xt, yt = gpubench.load(dataset, "test")
    classes = int(max(y.max(), yt.max())) + 1
    start = time.perf_counter()
    tree, library_predict, version = {"sklearn": sklearn_tree, "cuml": cuml_tree,
                                      "xgboost": xgboost_tree}[impl](X, y, depth)
    fit = time.perf_counter() - start
    counts = P.node_counts(tree, X, y, classes)
    stop = P.ccp_prune(tree, counts, alpha)
    base = {"dataset": dataset, "algo": "cart", "tool": impl, "version": version, "depth_limit": depth,
            "n_train": int(len(y)), "n_test": int(len(yt)), "fit_seconds": fit,
            "host": socket.gethostname(), "time": time.strftime("%F %T")}
    records = []
    for pruned in (False, True):
        train_acc, pred = P.evaluate(tree, counts, X, y, Xt, stop if pruned else None)
        nodes, leaves, d = P.size(tree, stop if pruned else None)
        record = {**base, "pruned": pruned,
                  "prune": f"ccp alpha {alpha:g} (misclassification, applied to the grown tree)"
                  if pruned else "none",
                  "nodes": nodes, "leaves": leaves, "depth": d, "train_accuracy": train_acc,
                  "test_accuracy": float(np.mean(pred == yt)),
                  "pred": f"{pred_prefix}-{'pruned' if pruned else 'unpruned'}.npy"}
        if not pruned:
            # the routing above against the library's own prediction of its tree
            record["own_predict_agreement"] = float(np.mean(library_predict(Xt) == pred))
        np.save(record["pred"], pred)
        records.append(record)
    Path(out_json).write_text(json.dumps(records, indent=1))
    print(json.dumps(records))


if __name__ == "__main__":
    main()
