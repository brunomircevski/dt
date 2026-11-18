#!/usr/bin/env python3
"""One run of a GPU tree library against ./tree: one fresh process that loads
the data, does the library's start-up and grows exactly one CART tree under the
timer, with every step timed; then (EVAL=1) evaluates it. No warm-up, no forest.
Appends one JSON line to OUT and saves the test predictions. bench/run.py runs
it (impls cuml, xgboost), interleaved with ./tree's runs.

  gpubench.py IMPL DATASET CASE OUT PRED_DIR [TREES [DEPTH]]
    IMPL     cuml | xgboost
    DATASET  a prepared dataset in $DATA_ROOT (<dataset>/ or <folder>/<dataset>.*)
    CASE     cart: Gini (cuML) or the Gini-equivalent squared error (XGBoost;
             with more than 2 classes one multi-output tree on the one-hot
             labels, whose summed squared error is Gini too). The only case: no
             GPU library grows C4.5 trees.
    TREES    1 (the only value; kept for run.py's command line)
    DEPTH    depth limit (default 10)

The same for every tool, ./tree included:
  untimed  reading the files into host RAM, as float32 values column-major
           (./tree's loader holds its data column-major too) and the labels
           (XGBoost's float32 or one-hot labels: the adapter's preparation);
           importing the library; the CUDA context alone (cudaFree(0)),
           CUDA_MODULE_LOADING=LAZY (CUDA's default), so the kernels a fit uses
           load inside it, as ./tree's do;
  timed    from the host arrays to the finished tree, in steps:
    buffers  ./tree's page-locked staging buffers and copy threads
             (bench/adapters/uploader.cpp, the same code as ./tree's upload);
    upload   labels and values to the GPU through them, into a column-major
             device array (./tree: its own upload, with the presort overlapped);
    prepare  cuML: its input checks (dtypes, classes); XGBoost: QuantileDMatrix
             (quantile sketch and binning on the GPU), then the raw device copy
             is freed (on MNIST8M it does not fit next to the tree's histograms);
    build    cuML: fit_forest (its binning, then the tree; one C++ call);
             XGBoost: train, one round.
  EVAL=1   (default) training and test accuracy, test predictions, nodes, depth,
           root split. Not timed.

Every library: depth <= DEPTH, binary splits on <= thresholds, all features, all
rows (no bootstrap, no feature sampling), no pruning, its default number of
histogram bins (cuML 128, XGBoost 256).

Environment: EVAL (1), GPU_MEMORY (1: sample nvidia-smi every 20 ms; run.py
passes 0 and samples from outside), UPLOADER (bench/.tools/libuploader.so),
TREE_GIT_DESCRIBE (the commit, recorded in every row).
"""
import ctypes, glob, json, os, socket, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets"))
EVAL = os.environ.get("EVAL", "1") == "1"
BINS = {"cuml": 128, "xgboost": 256}  # each library's default
UPLOADER = os.environ.get("UPLOADER", str(Path(__file__).resolve().parents[3] / ".tools" / "libuploader.so"))
if os.environ.get("WARM", "0") != "0" or os.environ.get("FOREST", "identical") != "identical":
    sys.exit("gpubench.py: one cold tree only (no WARM, no FOREST)")


def meta_path(dataset):
    own = f"{DATA_ROOT}/{dataset}/meta.json"
    return own if os.path.exists(own) else sorted(glob.glob(f"{DATA_ROOT}/*/{dataset}.meta.json"))[0]


def stem(dataset, part):
    """The files' path without .f32 / .y.i32 (bench/benchlib.py stem)."""
    path = meta_path(dataset)
    m = json.load(open(path))
    if "prefix_of" in m:
        return stem(m["prefix_of"], part)
    folder = os.path.dirname(path)
    return f"{folder}/{part}" if path.endswith("/meta.json") else f"{folder}/{dataset}"


def load(dataset, part):
    m = json.load(open(meta_path(dataset)))
    n, F = (m["n_train"] if part == "train" else m["n_test"]), len(m["features"])
    if n == 0:
        return np.empty((0, F), np.float32), np.empty(0, np.int32)
    X = np.fromfile(f"{stem(dataset, part)}.f32", dtype="<f4", count=n * F).reshape(n, F)
    y = np.fromfile(f"{stem(dataset, part)}.y.i32", dtype="<i4", count=n)
    return X, y


class GpuMemory:
    """Peak memory.used of the GPU (MiB) above its idle value, every 20 ms."""
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
    Only the left child marks a leaf (XGBoost's multi-output leaves reuse right)."""
    depth = np.zeros(len(left), dtype=np.int64)
    order = [0]
    for node in order:
        if left[node] < 0:
            continue
        for child in (left[node], right[node]):
            if child >= 0:
                depth[child] = depth[node] + 1
                order.append(child)
    return int(depth[order].max()), len(order)


def predict_chunks(predict, X, chunk=1_000_000):
    # cuML's predict on more than 2^32 values at once goes wrong (ECBDL): chunks.
    return np.concatenate([predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


# ------------------------------------------------------------------ the upload

def numa_nodes():
    return len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit()])


class Uploader:
    """bench/adapters/uploader.cpp: ./tree's staged upload."""
    def __init__(self):
        self.lib = ctypes.CDLL(UPLOADER)
        self.lib.staged_create.restype = ctypes.c_void_p
        self.lib.staged_create.argtypes = []
        self.lib.staged_cpus.argtypes = [ctypes.c_void_p]
        self.lib.staged_copy.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
        self.cpus = 0

    def start(self):  # timed: the buffers and copy threads, pinned to the GPU's node as ./tree's
        self.handle = self.lib.staged_create()
        self.cpus = self.lib.staged_cpus(self.handle)

    def upload(self, host):
        """host array (C- or F-contiguous) -> a CuPy array of the same layout."""
        import cupy as cp
        dev = cp.empty(host.shape, host.dtype, order="F" if not host.flags.c_contiguous else "C")
        error = self.lib.staged_copy(self.handle, dev.data.ptr, host.ctypes.data, host.nbytes)
        assert error == 0, f"upload failed: CUDA error {error}"
        return dev


# ------------------------------------------------------------------ libraries

def cuml_fit(dev, depth, classes, timer):
    from cuml.ensemble import RandomForestClassifier

    class Timed(RandomForestClassifier):
        """fit() is _prepare_fit_inputs() then _fit_forest(): timed apart."""
        def _prepare_fit_inputs(self, *args, **kwargs):
            result = super()._prepare_fit_inputs(*args, **kwargs)
            timer("prepare", "input checks")
            return result

        def _fit_forest(self, *args, **kwargs):
            result = super()._fit_forest(*args, **kwargs)
            timer("build", "build tree")
            return result

    model = Timed(n_estimators=1, bootstrap=False, max_samples=1.0, max_features=1.0,
                  split_criterion="gini", max_depth=depth, max_leaves=-1, n_bins=BINS["cuml"],
                  min_samples_leaf=1, min_samples_split=2, min_impurity_decrease=0.0,
                  n_streams=1, random_state=0, output_type="numpy")
    X, y = dev.pop("X"), dev.pop("y")
    model.fit(X, y)
    timer("finish", "fit's end")  # whatever fit() does after _fit_forest()
    return model


def cuml_info(model, X, y, Xt):
    predict = lambda A: predict_chunks(model.predict, A).astype(np.int8)
    acc = float(np.mean(predict(X) == y))
    pred = predict(Xt)
    tree = model.as_treelite().get_tree_accessor(0)
    d, nodes = depth_of(np.asarray(tree.get_field("cleft")).astype(np.int64),
                        np.asarray(tree.get_field("cright")).astype(np.int64))
    root = f"f{int(np.asarray(tree.get_field('split_index'))[0])} <= {float(np.asarray(tree.get_field('threshold'))[0]):.7g}"
    return acc, pred, nodes, d, root


def xgb_labels(y, classes):
    """float32 class indices; more than 2 classes: one-hot (the multi-output
    tree's summed squared error is n/2 x the drop in Gini impurity)."""
    return y.astype(np.float32) if classes == 2 else np.eye(classes, dtype=np.float32)[y]


def xgb_fit(dev, depth, classes, timer, base_score):
    # reg:squarederror, lambda 0, eta 1, one round: on 0/1 labels the split gain
    # is n/2 x the drop in Gini impurity, the CART split on 256 bins. More than 2
    # classes: one multi-output tree; at most 128 nodes' histograms cached (the
    # default does not fit in 80 GB on MNIST8M; the same tree).
    import cupy as cp
    import xgboost as xgb
    dtrain = xgb.QuantileDMatrix(dev["X"], label=dev["y"], max_bin=BINS["xgboost"])
    timer("prepare", "quantile binning")
    dev.clear()  # the only references to the raw device copy
    cp.get_default_memory_pool().free_all_blocks()  # the raw copy: XGBoost keeps its bins
    timer("prepare", "free raw copy")
    multi = ({"multi_strategy": "multi_output_tree", "max_cached_hist_node": 128} if classes > 2 else {})
    params = {"device": "cuda", "tree_method": "hist", "objective": "reg:squarederror", **multi,
              "eta": 1.0, "lambda": 0.0, "alpha": 0.0, "gamma": 0.0, "min_child_weight": 1.0,
              **({"base_score": base_score} if classes == 2 else {}), "max_bin": BINS["xgboost"],
              "subsample": 1.0, "colsample_bytree": 1.0, "max_depth": depth, "grow_policy": "depthwise"}
    booster = xgb.train(params, dtrain, num_boost_round=1)
    timer("build", "build tree")
    return booster


def xgb_info(booster, X, y, Xt):
    import cupy as cp
    def predict(A):  # on the GPU, in chunks; more than 2 classes: the largest output
        p = predict_chunks(lambda c: cp.asnumpy(booster.inplace_predict(cp.asarray(c))), A)
        return (p.argmax(axis=1) if p.ndim == 2 else p > 0.5).astype(np.int8)
    acc = float(np.mean(predict(X) == y))
    pred = predict(Xt)
    tree = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"][0]
    d, nodes = depth_of(np.array(tree["left_children"], dtype=np.int64),
                        np.array(tree["right_children"], dtype=np.int64))
    return acc, pred, nodes, d, f"f{tree['split_indices'][0]} < {tree['split_conditions'][0]:.7g}"


def cpu_list():
    cpus, ranges = sorted(os.sched_getaffinity(0)), []
    for cpu in cpus:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def main():
    impl, dataset, case, out, pred_dir = sys.argv[1:6]
    assert impl in ("cuml", "xgboost"), impl
    assert case == "cart", "CART only: no GPU library grows C4.5 trees"
    assert (int(sys.argv[6]) if len(sys.argv) > 6 else 1) == 1, "one tree only"
    depth = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    gpu = GpuMemory() if os.environ.get("GPU_MEMORY", "1") == "1" else None

    # untimed: the data in host RAM, column-major as ./tree's; the labels
    t = time.perf_counter()
    X, y = load(dataset, "train")
    X = np.asfortranarray(X)
    classes = int(y.max()) + 1
    labels = xgb_labels(y, classes) if impl == "xgboost" else y
    read = time.perf_counter() - t
    # untimed: the start-up (import, CUDA context)
    t = time.perf_counter()
    import cupy  # noqa: F401
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
    _, props = runtime.cudaGetDeviceProperties(0)
    gpu_name = props.name.decode().rstrip("\x00")
    uploader = Uploader()

    # timed: host arrays -> finished tree, step by step
    steps, timeline = {}, []
    start = last = time.perf_counter()

    def timer(step, section):  # the end of a section of a step: the GPU's queued work included
        nonlocal last
        runtime.cudaDeviceSynchronize()
        now = time.perf_counter()
        steps[f"{step}_seconds"] = steps.get(f"{step}_seconds", 0.0) + now - last
        timeline.append({"name": section, "start": last - start, "end": now - start})
        last = now

    uploader.start()
    timer("buffers", "staging buffers")
    dev = {"y": uploader.upload(np.ascontiguousarray(labels)), "X": uploader.upload(X)}
    timer("upload", "upload")
    if impl == "cuml":
        model = cuml_fit(dev, depth, classes, timer)
    else:
        model = xgb_fit(dev, depth, classes, timer, float(y.mean()))
    seconds = time.perf_counter() - start
    if uploader.cpus == 0 and numa_nodes() > 1:  # ./tree pins its copies there: so must the libraries'
        sys.exit("gpubench.py: the upload's copy threads are not pinned to the GPU's NUMA node")

    row = {"impl": impl, "dataset": dataset, "case": case, "bins": BINS[impl], "trees": 1,
           "depth_limit": depth, "train_seconds": seconds, **steps, "timeline": timeline,
           "upload_bytes": int(X.nbytes + labels.nbytes), "layout": "column-major",
           "read_seconds": read, "import_seconds": imported, "gpu_startup_seconds": context,
           "startup_seconds": imported + context, "uploader_cpus": uploader.cpus,
           "n_train_loaded": int(X.shape[0]), "n_features_loaded": int(X.shape[1]),
           "version": __import__(impl).__version__, "commit": os.environ.get("TREE_GIT_DESCRIBE"),
           "host": socket.gethostname(), "cpus": cpu_list(), "gpu": gpu_name,
           "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
           "eval": EVAL, "time": time.strftime("%F %T")}
    if EVAL:
        Xt, yt = load(dataset, "test")
        if not len(yt):  # no test set (chess): 50,000 training rows, no test accuracy
            Xt, yt = X[:50_000], None
        info = cuml_info if impl == "cuml" else xgb_info
        train_acc, pred, nodes, d, root = info(model, X, y, Xt)
        if yt is not None:
            np.save(f"{pred_dir}/{impl}-{case}.npy", pred)
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
