#!/usr/bin/env python3
"""One run of a GPU decision-tree library: one fresh process that loads the
data, does the library's start-up, fits exactly one model (one tree, or one
forest of TREES trees) under the timer, then (EVAL=1) evaluates it; appends one
JSON line to OUT and saves the test predictions. No warm-up fit: ten runs are
ten executions of this command (BENCHMARK-CHANGES.MD, section 1). The warm
variant (WARM=1) first makes one untimed identical fit in the same process.

  gpubench.py IMPL DATASET CASE OUT PRED_DIR [TREES [DEPTH]]
    IMPL     cuml | xgboost
    DATASET  higgs | mnist8m | susy | ecbdl_15m | chess_1b   (prepared in $DATA_ROOT: <dataset>/ or <folder>/<dataset>.*)
    CASE     cart   Gini (cuML) or the Gini-equivalent squared error (XGBoost;
                    with more than 2 classes one multi-output tree on the
                    one-hot labels, whose summed squared error is Gini too).
                    The only case: no GPU library has C4.5's gain ratio, and
                    an entropy tree is not C4.5, so C4.5 has no GPU competitor.
    TREES    trees per fit (default 1): cuML n_estimators, XGBoost
             num_parallel_tree in one round
    DEPTH    depth limit (default 19)

The phases, in order:
  1. read     train.f32 / train.y.i32 into host float32 / int32 NumPy arrays
              (read_seconds; not timed);
  2. start-up import the library (cuML imports RMM; import_seconds), then
              create the CUDA context on the chosen GPU with cudaFree(0)
              (gpu_startup_seconds). CUDA_MODULE_LOADING=LAZY (CUDA's default,
              job.sh): the libraries link their own CUDA runtime, so EAGER
              would load all their kernels inside fit; the kernels a fit uses
              load inside it (timed). RMM keeps cuML's default memory
              resource: nothing is reconfigured or pre-grown, so every
              allocation fit makes stays timed (as ./tree's pool growth does).
              startup_seconds = import_seconds + gpu_startup_seconds; not timed;
              XGBoost's float32 (or one-hot) labels are made here too, untimed
              (the adapter's data preparation: ./tree and cuML get the int32
              labels as read). The same rule as ./tree --gpu-startup context
              (bench/run.py): the CUDA context alone before the timer;
  3. timed    from the host arrays to the fitted model (train_seconds):
              cuML fit (copy to the GPU, binning, growing); XGBoost
              QuantileDMatrix (dmatrix_seconds) + train (build_seconds);
  4. EVAL=1   (default) training and test accuracy, test predictions, the tree
              walk (nodes, depth, root split); EVAL=0 skips all of it, so the
              process ends with the fitted model (the end-to-end benchmark).

Every library grows trees of depth <= DEPTH with binary splits on <= thresholds,
all features, all rows (no bootstrap, no feature sampling), no pruning, on its
default number of histogram bins. With TREES > 1 every tree of the forest is
the same tree as with TREES = 1, so the time per tree is comparable with one
tree's.

Environment:
  EVAL         1 (default) or 0, see above
  GPU_MEMORY   1 (default): sample the GPU's memory.used every 20 ms with
               nvidia-smi (gpu_peak_mib: the peak above the idle value read
               before the start-up); 0: no sampler (run.py samples from outside)
  FOREST       identical (default): the settings above, every tree the same;
               default: each library's own random-forest defaults, only the
               criterion and the depth limit set, so the trees differ:
                 cuML RandomForestClassifier: bootstrap, max_features "sqrt"
                 XGBoost (as XGBRFClassifier): binary:logistic, subsample 0.8,
                 colsample_bynode 0.8, lambda 1e-5, eta 1
               nodes is then the mean over the forest's trees, depth the max.
  N_STREAMS    cuML's n_streams (default: 1 for one tree, cuML's default 4
               for a forest, whose trees it grows concurrently, one per stream)
  TREE_GIT_DESCRIBE  the commit of the benchmark code, recorded in every row
"""
import glob, json, os, socket, subprocess, sys, threading, time
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets"))
DEPTH = 19  # set by main()
TREES = 1   # set by main()
CLASSES = 2  # set by main()
LABEL = None  # XGBoost's labels, made by main() before the timer
EVAL = os.environ.get("EVAL", "1") == "1"
BINS = {"cuml": 128, "xgboost": 256}  # each library's default
FOREST = os.environ.get("FOREST", "identical")
# WARM=1, the warm variant: one untimed fit, the same as the timed one on the
# same data, first in the same process (kernels loaded, allocator caches and
# the driver's mappings made); then the timed fit. ./tree: --warm-up.
WARM = os.environ.get("WARM", "0") == "1"
assert FOREST in ("identical", "default"), FOREST


def meta_path(dataset):
    """<dataset>/meta.json, else <folder>/<dataset>.meta.json (a prefix_of set,
    or a set in a shared folder), as bench/benchlib.py."""
    own = f"{DATA_ROOT}/{dataset}/meta.json"
    return own if os.path.exists(own) else sorted(glob.glob(f"{DATA_ROOT}/*/{dataset}.meta.json"))[0]


def stem(dataset, part):
    """The files' path without .f32 / .y.i32 (bench/benchlib.py stem): a
    prefix_of set (sorel_4m, ecbdl_15m, chess_100m) is the first rows of its
    parent's files."""
    path = meta_path(dataset)
    m = json.load(open(path))
    if "prefix_of" in m:
        return stem(m["prefix_of"], part)
    folder = os.path.dirname(path)
    return f"{folder}/{part}" if path.endswith("/meta.json") else f"{folder}/{dataset}"


def load(dataset, part):
    m = json.load(open(meta_path(dataset)))
    n, F = (m["n_train"] if part == "train" else m["n_test"]), len(m["features"])
    if n == 0:  # no test set
        return np.empty((0, F), np.float32), np.empty(0, np.int32)
    X = np.fromfile(f"{stem(dataset, part)}.f32", dtype="<f4", count=n * F).reshape(n, F)
    y = np.fromfile(f"{stem(dataset, part)}.y.i32", dtype="<i4", count=n)
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
    """Depth and node count of a tree given child arrays (left -1 = leaf), root 0.
    Only the left child marks a leaf: in XGBoost's multi-output trees a leaf's
    right_children entry is its index in leaf_weights."""
    depth = np.zeros(len(left), dtype=np.int64)
    order = [0]
    for node in order:  # breadth first
        if left[node] < 0:
            continue
        for child in (left[node], right[node]):
            if child >= 0:
                depth[child] = depth[node] + 1
                order.append(child)
    return int(depth[order].max()), len(order)


# ------------------------------------------------------------------ libraries

def n_streams():
    return int(os.environ.get("N_STREAMS", 1 if TREES == 1 else 4))


def cuml_fit(X, y, case, trees):
    from cuml.ensemble import RandomForestClassifier
    if FOREST == "default":
        model = RandomForestClassifier(
            n_estimators=trees, split_criterion="gini",
            max_depth=DEPTH, n_streams=n_streams(), random_state=0, output_type="numpy")
        model.fit(X, y)
        return model, {}
    model = RandomForestClassifier(
        n_estimators=trees, bootstrap=False, max_samples=1.0, max_features=1.0,
        split_criterion="gini",
        max_depth=DEPTH, max_leaves=-1, n_bins=BINS["cuml"],
        min_samples_leaf=1, min_samples_split=2, min_impurity_decrease=0.0,
        n_streams=n_streams(), random_state=0, output_type="numpy")
    model.fit(X, y)
    return model, {}


def predict_chunks(predict, X, chunk):
    return np.concatenate([predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def cuml_predict(model, X):
    # In chunks: predict() on ECBDL's 15M x 631 values at once goes wrong past
    # 2^32 values (row 6,806,6xx); 1M-row chunks give the model's real accuracy.
    return predict_chunks(model.predict, X, 1_000_000).astype(np.int8)


def forest_shape(shapes):
    """(depth, nodes) per tree -> max depth, mean node count."""
    depths, nodes = zip(*shapes)
    return max(depths), float(np.mean(nodes))


def cuml_info(model, X, y, Xt):
    acc = float(np.mean(cuml_predict(model, X) == y))
    pred = cuml_predict(model, Xt)
    forest = model.as_treelite()
    shapes = []
    for i in range(forest.num_tree):
        tree = forest.get_tree_accessor(i)
        shapes.append(depth_of(np.asarray(tree.get_field("cleft")).astype(np.int64),
                               np.asarray(tree.get_field("cright")).astype(np.int64)))
    d, nodes = forest_shape(shapes)
    tree = forest.get_tree_accessor(0)
    feat = int(np.asarray(tree.get_field("split_index"))[0])
    thr = float(np.asarray(tree.get_field("threshold"))[0])
    return acc, pred, nodes, d, f"f{feat} <= {thr:.7g}"


def xgb_labels(y):
    """XGBoost's labels, made before the timer (the adapter's data preparation,
    as ./tree's int32 labels and cuML's are read ready): float32 class indices;
    the CART emulation with more than 2 classes: the one-hot labels (the summed
    squared error of a split is n/2 times the drop in Gini impurity for any
    number of classes, whatever base_score is, in one round)."""
    if FOREST == "default" or CLASSES == 2:
        return y.astype(np.float32)
    return np.eye(CLASSES, dtype=np.float32)[y]


def xgb_fit(X, y, case, trees):
    # reg:squarederror with lambda = 0 on 0/1 labels: every row has hessian 1,
    # so the split gain is the drop in the children's squared error, which is
    # n/2 times the drop in Gini impurity: the CART split, on histogram bins.
    # More than 2 classes: one multi-output tree (multi_strategy
    # multi_output_tree) instead of XGBoost's default of one tree per class.
    # Its histograms hold every class's gradients (MNIST8M: 10 x 784 features x
    # 256 bins per node), so at depth 10 the default cache of up to 65,536
    # nodes' histograms runs out of 80 GB: keep fewer (built again when needed;
    # the same tree). With 256 it asked for 28.7 GB with 27.3 GB free on an
    # H100; on MNIST8M one tree takes 5.5 s with 128, 6.4 s with 64, 7.1 s
    # with 32 (H100, 2026-11-10).
    multi = ({"multi_strategy": "multi_output_tree",
              "max_cached_hist_node": int(os.environ.get("XGB_MAX_CACHED_HIST_NODE", 128))}
             if CLASSES > 2 else {})
    if FOREST == "default":  # XGBRFClassifier's defaults, one round
        import xgboost as xgb
        t = time.perf_counter()
        dtrain = xgb.QuantileDMatrix(X, label=LABEL)
        dmatrix = time.perf_counter() - t
        objective = ({"objective": "multi:softprob", "num_class": CLASSES} if CLASSES > 2
                     else {"objective": "binary:logistic"})
        params = {"device": "cuda", "tree_method": "hist", **objective, **multi,
                  "eta": 1.0, "subsample": 0.8, "colsample_bynode": 0.8, "lambda": 1e-5,
                  "max_depth": DEPTH, "num_parallel_tree": trees, "seed": 0}
        t = time.perf_counter()
        booster = xgb.train(params, dtrain, num_boost_round=1)
        return booster, {"dmatrix_seconds": dmatrix, "build_seconds": time.perf_counter() - t}
    # trees > 1: XGBoost's random-forest mode, num_parallel_tree trees in one
    # round on the same gradients; with all rows and features they are the same
    # tree, and XGBoost averages the forest (each leaf scaled by 1/trees).
    import xgboost as xgb
    t = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(X, label=LABEL, max_bin=BINS["xgboost"])
    dmatrix = time.perf_counter() - t
    params = {"device": "cuda", "tree_method": "hist", "objective": "reg:squarederror", **multi,
              "eta": 1.0, "lambda": 0.0, "alpha": 0.0, "gamma": 0.0,
              "min_child_weight": 1.0, **({"base_score": float(y.mean())} if CLASSES == 2 else {}),
              "max_bin": BINS["xgboost"],
              "subsample": 1.0, "colsample_bytree": 1.0,  # nthread: its default, the CPUs it may use
              "max_depth": DEPTH, "grow_policy": "depthwise", "num_parallel_tree": trees}
    t = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=1)
    return booster, {"dmatrix_seconds": dmatrix, "build_seconds": time.perf_counter() - t}


def xgb_predict(booster, X):
    # In chunks, like cuML's (chess_1b: 10^9 rows), each copied to the GPU:
    # given host memory, XGBoost falls back to predicting from a DMatrix
    # (MNIST8M's multi-output tree: 1M rows 7.5 s on the host, 1.5 s on the
    # GPU, the same outputs). More than 2 classes: one output per class, the
    # largest wins.
    import cupy as cp
    p = predict_chunks(lambda c: cp.asnumpy(booster.inplace_predict(cp.asarray(c))), X, 1_000_000)
    return (p.argmax(axis=1) if p.ndim == 2 else p > 0.5).astype(np.int8)


def xgb_info(booster, X, y, Xt):
    acc = float(np.mean(xgb_predict(booster, X) == y))
    pred = xgb_predict(booster, Xt)
    trees = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]
    d, nodes = forest_shape([depth_of(np.array(t["left_children"], dtype=np.int64),
                                      np.array(t["right_children"], dtype=np.int64)) for t in trees])
    tree = trees[0]
    root = f"f{tree['split_indices'][0]} < {tree['split_conditions'][0]:.7g}"
    return acc, pred, nodes, d, root


IMPLS = {"cuml": (cuml_fit, cuml_info, cuml_predict), "xgboost": (xgb_fit, xgb_info, xgb_predict)}


def version(impl):
    return __import__(impl).__version__


def cpu_list():
    """The CPUs this process may run on, as ranges: "0-15,32-47"."""
    cpus, ranges = sorted(os.sched_getaffinity(0)), []
    for cpu in cpus:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def start_up(impl):
    """The library's start-up, before the timer: import it, then create the
    CUDA context (nothing else: no allocator pool, no fit).
    -> (import seconds, context seconds, GPU name)."""
    t = time.perf_counter()
    if impl == "cuml":
        import cuml.ensemble  # noqa: F401  (imports RMM)
    else:
        import xgboost  # noqa: F401
    imported = time.perf_counter() - t
    from cuda.bindings import runtime
    t = time.perf_counter()
    error, = runtime.cudaFree(0)
    context = time.perf_counter() - t
    assert error == runtime.cudaError_t.cudaSuccess, error
    error, props = runtime.cudaGetDeviceProperties(0)
    name = props.name.decode().rstrip("\x00") if error == runtime.cudaError_t.cudaSuccess else None
    return imported, context, name


def main():
    global DEPTH, TREES, CLASSES
    impl, dataset, case, out, pred_dir = sys.argv[1:6]
    assert case == "cart", "CART only: no GPU library grows C4.5 trees"
    TREES = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    DEPTH = int(sys.argv[7]) if len(sys.argv) > 7 else 19
    fit, info, predict = IMPLS[impl]
    gpu = GpuMemory() if os.environ.get("GPU_MEMORY", "1") == "1" else None  # idle value first

    t = time.perf_counter()
    X, y = load(dataset, "train")
    read = time.perf_counter() - t
    CLASSES = int(y.max()) + 1
    imported, context, gpu_name = start_up(impl)
    global LABEL
    LABEL = xgb_labels(y) if impl == "xgboost" else None  # untimed, see xgb_labels

    warmup = None
    if WARM:
        t = time.perf_counter()
        model, _ = fit(X, y, case, TREES)
        del model
        import gc
        gc.collect()
        warmup = time.perf_counter() - t

    t = time.perf_counter()
    model, timing = fit(X, y, case, TREES)
    seconds = time.perf_counter() - t

    row = {"impl": impl, "dataset": dataset, "case": case, "bins": BINS[impl],
           "trees": TREES, "forest": FOREST, "depth_limit": DEPTH, "train_seconds": seconds, **timing,
           "read_seconds": read, "import_seconds": imported, "gpu_startup_seconds": context,
           "startup_seconds": imported + context,
           "n_train_loaded": int(X.shape[0]), "n_features_loaded": int(X.shape[1]),
           "version": version(impl), "commit": os.environ.get("TREE_GIT_DESCRIBE"),
           "host": socket.gethostname(), "cpus": cpu_list(), "gpu": gpu_name,
           "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
           "eval": EVAL, "warm": WARM, "time": time.strftime("%F %T")}
    if WARM:
        row["warmup_seconds"] = warmup
    if impl == "cuml":
        row["n_streams"] = n_streams()
    if EVAL:
        # No test set (chess): predict 50,000 training rows instead, no test accuracy.
        Xt, yt = load(dataset, "test")
        if not len(yt):
            Xt, yt = X[:50_000], None
        train_acc, pred, nodes, d, root = info(model, X, y, Xt)
        if yt is not None:
            np.save(f"{pred_dir}/{impl}-{case}-{TREES}-{FOREST}.npy", pred)
        row.update(train_accuracy=train_acc,
                   test_accuracy=float(np.mean(pred == yt)) if yt is not None else None,
                   nodes=nodes, depth=d, root=root)
    if gpu:
        row["gpu_peak_mib"] = gpu.stop()
    print(json.dumps(row), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
