#!/usr/bin/env python3
"""One benchmark case of a GPU decision-tree library: one process, an untimed
warm-up fit, then REPS timed fits of one single tree; appends one JSON line
per timed fit to OUT and saves the last tree's test predictions.

  gpubench.py IMPL DATASET CASE REPS OUT PRED_DIR
    IMPL     cuml | xgboost
    DATASET  higgs | airline   (prepared in $DATA_ROOT/<dataset>)
    CASE     cart   Gini (cuML) or the Gini-equivalent squared error (XGBoost)
             c45    entropy (cuML only; XGBoost has no entropy criterion)

Every library grows one tree of depth <= 19 with binary splits on <= thresholds,
all features, all rows (no bootstrap, no feature sampling), no pruning, on its
default number of histogram bins. The timed part is everything from the host
float32 arrays to the fitted tree: copy to the GPU, quantile binning, growing.
"""
import json, os, socket, subprocess, sys, threading, time
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/dt/bench/data"))
DEPTH = 19
WARMUP_ROWS = 50_000
BINS = {"cuml": 128, "xgboost": 256}  # each library's default


def load(dataset, part):
    m = json.load(open(f"{DATA_ROOT}/{dataset}/meta.json"))
    n = m["n_train"] if part == "train" else m["n_test"]
    X = np.fromfile(f"{DATA_ROOT}/{dataset}/{part}.f32", dtype="<f4").reshape(n, len(m["features"]))
    y = np.fromfile(f"{DATA_ROOT}/{dataset}/{part}.y.i32", dtype="<i4")
    return X, y


class GpuMemory:
    """Peak memory.used of GPU 0 (MiB) above its idle value, sampled every
    20 ms by nvidia-smi. The idle value is read before anything starts."""
    def __init__(self):
        gpu = "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", gpu]
        self.idle = int(subprocess.run(query, capture_output=True, text=True, check=True).stdout)
        self.peak = self.idle
        self.proc = subprocess.Popen(query + ["-lms", "20"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.peak = max(self.peak, int(line.strip()))
            except ValueError:
                pass

    def stop(self):
        self.proc.terminate()
        return self.peak - self.idle


def depth_of(left, right):
    """Depth and node count of a tree given child arrays (-1 = leaf), root 0."""
    depth = np.zeros(len(left), dtype=np.int64)
    order = [0]
    for node in order:  # breadth first
        for child in (left[node], right[node]):
            if child >= 0:
                depth[child] = depth[node] + 1
                order.append(child)
    return int(depth[order].max()), len(order)


# ------------------------------------------------------------------ libraries

def cuml_fit(X, y, case):
    # c45: entropy (information gain; cuML has no gain ratio), and C4.5's
    # smallest split: at least 2 rows on each side, so 4 rows to split.
    from cuml.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=1, bootstrap=False, max_samples=1.0, max_features=1.0,
        split_criterion="gini" if case == "cart" else "entropy",
        max_depth=DEPTH, max_leaves=-1, n_bins=BINS["cuml"],
        min_samples_leaf=1 if case == "cart" else 2,
        min_samples_split=2 if case == "cart" else 4, min_impurity_decrease=0.0,
        n_streams=1, random_state=0, output_type="numpy")
    model.fit(X, y)
    return model


def cuml_info(model, X, y, Xt):
    acc = float(np.mean(model.predict(X) == y))
    pred = model.predict(Xt).astype(np.int8)
    tree = model.as_treelite().get_tree_accessor(0)
    left = np.asarray(tree.get_field("cleft")).astype(np.int64)
    right = np.asarray(tree.get_field("cright")).astype(np.int64)
    d, nodes = depth_of(left, right)
    feat = int(np.asarray(tree.get_field("split_index"))[0])
    thr = float(np.asarray(tree.get_field("threshold"))[0])
    return acc, pred, nodes, d, f"f{feat} <= {thr:.7g}"


def xgb_fit(X, y, case):
    # reg:squarederror with lambda = 0 on 0/1 labels: every row has hessian 1,
    # so the split gain is the drop in the children's squared error, which is
    # n/2 times the drop in Gini impurity: the CART split, on histogram bins.
    assert case == "cart", "XGBoost has no entropy criterion"
    import xgboost as xgb
    dtrain = xgb.QuantileDMatrix(X, label=y.astype(np.float32), max_bin=BINS["xgboost"])
    params = {"device": "cuda", "tree_method": "hist", "objective": "reg:squarederror",
              "eta": 1.0, "lambda": 0.0, "alpha": 0.0, "gamma": 0.0,
              "min_child_weight": 1.0, "base_score": float(y.mean()),
              "max_bin": BINS["xgboost"],
              "subsample": 1.0, "colsample_bytree": 1.0, "nthread": os.cpu_count(),
              "max_depth": DEPTH, "grow_policy": "depthwise"}
    booster = xgb.train(params, dtrain, num_boost_round=1)
    return booster


def xgb_info(booster, X, y, Xt):
    acc = float(np.mean((booster.inplace_predict(X) > 0.5) == y))
    pred = (booster.inplace_predict(Xt) > 0.5).astype(np.int8)
    tree = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"][0]
    left = np.array(tree["left_children"], dtype=np.int64)
    right = np.array(tree["right_children"], dtype=np.int64)
    d, nodes = depth_of(left, right)
    root = f"f{tree['split_indices'][0]} < {tree['split_conditions'][0]:.7g}"
    return acc, pred, nodes, d, root


IMPLS = {"cuml": (cuml_fit, cuml_info), "xgboost": (xgb_fit, xgb_info)}


def version(impl):
    return __import__(impl).__version__


def main():
    impl, dataset, case, reps, out, pred_dir = sys.argv[1:7]
    fit, info = IMPLS[impl]
    X, y = load(dataset, "train")
    Xt, yt = load(dataset, "test")
    gpu = GpuMemory()
    t = time.perf_counter()
    fit(X[:WARMUP_ROWS], y[:WARMUP_ROWS], case)  # imports, CUDA context, JIT
    warmup = time.perf_counter() - t
    for rep in range(int(reps)):
        t = time.perf_counter()
        model = fit(X, y, case)
        seconds = time.perf_counter() - t
        row = {"impl": impl, "dataset": dataset, "case": case, "bins": BINS[impl], "rep": rep,
               "train_seconds": seconds, "warmup_seconds": warmup, "version": version(impl),
               "host": socket.gethostname(), "time": time.strftime("%F %T")}
        if rep == int(reps) - 1:
            train_acc, pred, nodes, d, root = info(model, X, y, Xt)
            np.save(f"{pred_dir}/{impl}-{case}.npy", pred)
            row.update(train_accuracy=train_acc, test_accuracy=float(np.mean(pred == yt)),
                       nodes=nodes, depth=d, root=root, gpu_peak_mib=gpu.stop())
        print(json.dumps(row), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        del model


if __name__ == "__main__":
    main()
