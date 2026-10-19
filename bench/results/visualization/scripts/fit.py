#!/usr/bin/env python3
"""Fit one tree with one implementation on the 80/20 chess split, time it,
export the tree to the common format (treeio.py) and write <out>/<name>.json.

  fit.py IMPL CASE DATA OUT [--reps 3] [--threads N] [--backend B] [--depth 10]

  IMPL     tree | sklearn | rpart | j48 | yadt | cuml | xgboost
  CASE     cart | c45
  --backend ./tree only: serial | parallel | cuda

Every tool grows one tree with the depth limit, without pruning, on all 800,000
training rows. Timed: each tool's own training (./tree's `train total`; YaDT's
total time without parsing the CSV; the fit/build call of the libraries, which
for cuML and XGBoost includes the copy to the GPU and the binning). ./tree and
YaDT run REPS fresh processes; the libraries run one process with an untimed
warm-up fit on 50,000 rows, then REPS timed fits. The last tree is exported.
Accuracies, depth and node count come from walking the exported tree
(treeio.py); where a library predicts by itself, its test predictions must
agree with the walk on every row (`test_export_mismatches`).
"""
import argparse, json, os, platform, re, socket, statistics, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import treeio  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("DT_REPO", HERE.parents[3]))
TOOLS = Path(os.environ.get("DT_TOOLS", REPO / "bench" / ".tools"))
WARMUP = 50_000


def cpu_name():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                               "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------- ./tree

def parse_tree_dump(path):
    """./tree --dump: preorder, 'ROOT [n=..]: if x <= 1.5' / 'Leaf -> A'."""
    line_re = re.compile(r"^\s*\w+ \[n=(\d+)\]: (.*)$")
    nodes = []
    for line in open(path):
        m = line_re.match(line)
        if not m:
            continue
        n, body = int(m.group(1)), m.group(2)
        if body.startswith("Leaf -> "):
            nodes.append({"class": body[len("Leaf -> "):].strip('"'), "n": n})
        else:
            f, t = re.match(r'if "?(\w+)"? <= (\S+)$', body).groups()
            nodes.append({"feature": f, "op": "<=", "threshold": float(t), "n": n})
    it = iter(nodes)

    def build():
        node = next(it)
        if "feature" in node:
            node["left"] = build()
            node["right"] = build()
        return node
    return build()


def run_tree(a, meta):
    binary = os.environ.get("TREE_BIN", str(REPO / "tree"))
    flags = ["--cart"] if a.case == "cart" else ["--c45"]
    flags += ["--no-prune", "-d", str(a.depth)]
    backend = ["--" + a.backend] + (["--threads", str(a.threads)] if a.backend != "serial" else [])
    times, tree = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(a.reps):
            argv = [binary] + backend + flags + ["--dump", f"{tmp}/t.dump", f"{a.data}/train.csv"]
            out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
            times.append(float(re.search(r"train total\s+([\d.]+) ms", out).group(1)) / 1000)
            printed_acc = float(re.search(r"train accuracy\s+([\d.]+)%", out).group(1)) / 100
        tree = parse_tree_dump(f"{tmp}/t.dump")
    meta.update(command=" ".join(argv[:-1] + ["train.csv"]),
                version=os.environ.get("TREE_GIT_DESCRIBE", "unknown"),
                tree_printed_train_accuracy=printed_acc,
                timer="./tree's `train total` (presort and build; on CUDA also GPU setup and upload), "
                      "one fresh process per run")
    return times, tree, None


# ---------------------------------------------------------------- scikit-learn

def run_sklearn(a, meta):
    import sklearn
    from sklearn.tree import DecisionTreeClassifier
    X, y = treeio.load(a.data, "train")
    Xt, _ = treeio.load(a.data, "test")

    def fit(X, y):
        return DecisionTreeClassifier(criterion="gini", max_depth=a.depth, random_state=0).fit(X, y)
    fit(X[:WARMUP], y[:WARMUP])
    times = []
    for _ in range(a.reps):
        t = time.perf_counter()
        model = fit(X, y)
        times.append(time.perf_counter() - t)
    s = model.tree_

    def build(i):
        n = int(s.n_node_samples[i])
        counts = np.rint(s.value[i][0] * n if s.value[i][0].sum() <= 1.0 + 1e-9 else s.value[i][0])
        dist = {c: int(k) for c, k in zip(treeio.CLASSES, counts)}
        cls = treeio.CLASSES[int(np.argmax(s.value[i][0]))]
        if s.children_left[i] < 0:
            return {"class": cls, "n": n, "dist": dist}
        return {"feature": treeio.FEATURES[s.feature[i]], "op": "<=", "threshold": float(s.threshold[i]),
                "n": n, "dist": dist, "class": cls,
                "left": build(s.children_left[i]), "right": build(s.children_right[i])}
    meta.update(version=f"scikit-learn {sklearn.__version__}",
                command=f"DecisionTreeClassifier(criterion='gini', max_depth={a.depth})",
                timer="DecisionTreeClassifier.fit on the float32 arrays, after a warm-up fit")
    return times, build(0), model.predict(Xt)


# ---------------------------------------------------------------- rpart

def run_rpart(a, meta):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["Rscript", "--vanilla", str(HERE / "rpart_export.R"), a.data, str(a.reps),
                        str(a.depth), f"{tmp}/r"], check=True)
        times = [float(t) for t in open(f"{tmp}/r.times")]
        rows = [line.strip().split(",") for line in open(f"{tmp}/r.frame")][1:]
        pred = np.array([int(p) for p in open(f"{tmp}/r.pred")])
        version = open(f"{tmp}/r.version").read().strip()
    by_id = {int(r[0]): r for r in rows}

    def build(k):
        _, var, n, ca, cb, cls, thr, ncat = by_id[k]
        node = {"n": int(n), "dist": {"A": int(float(ca)), "B": int(float(cb))}, "class": cls}
        if var == "<leaf>":
            return node
        # rpart: ncat -1 sends x < t to the left son, +1 sends x >= t there.
        left, right = build(2 * k), build(2 * k + 1)
        if int(ncat) > 0:
            left, right = right, left
        return {"feature": var, "op": "<", "threshold": float(thr), **node, "left": left, "right": right}
    meta.update(version=version,
                command=f"rpart(method='class', split='gini', minsplit=2, minbucket=1, cp=0, "
                        f"maxdepth={a.depth}, xval=0, maxcompete=0, maxsurrogate=0)",
                timer="the rpart() call, after a warm-up fit; data read from the float32 file")
    return times, build(1), pred


# ---------------------------------------------------------------- Weka J48

def truncate(node, depth, limit):
    """Cut a tree grown without a limit at `limit`: what C4.5's top-down growth
    would build with the limit, since no split above it depends on the ones below.
    Returns the number of splits removed."""
    if treeio.is_leaf(node):
        return 0
    if depth == limit:
        removed = sum(1 for n in treeio.iter_nodes(node) if not treeio.is_leaf(n))
        for k in ("feature", "op", "threshold", "left", "right"):
            node.pop(k)
        return removed
    return truncate(node["left"], depth + 1, limit) + truncate(node["right"], depth + 1, limit)


def run_j48(a, meta):
    cp = f"{TOOLS / 'weka.jar'}:{TOOLS / 'bounce.jar'}:{a.classes}"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["java", "-Xmx8g", "-cp", cp, "J48Export", a.data, str(a.reps), f"{tmp}/j"],
                       check=True)
        times = [float(t) for t in open(f"{tmp}/j.times")]
        lines = [line.split() for line in open(f"{tmp}/j.tree")]
        pred = np.array([int(p) for p in open(f"{tmp}/j.pred")])
        version = open(f"{tmp}/j.version").read().strip()
    it = iter(lines)

    def build():
        parts = next(it)
        ca, cb = float(parts[-2]), float(parts[-1])
        node = {"n": int(round(ca + cb)), "dist": {"A": int(round(ca)), "B": int(round(cb))},
                "class": "A" if ca >= cb else "B"}
        if parts[0] == "leaf":
            node["class"] = parts[1]
            return node
        node.update(feature=parts[1], op="<=", threshold=float(parts[2]))
        node["left"], node["right"] = build(), build()
        return node
    tree = build()
    grown_depth = treeio.stats(tree)[2]
    removed = truncate(tree, 0, a.depth)
    meta.update(version=version, command="J48 -U -M 2 (unpruned, at least 2 rows per side)",
                grown_depth=grown_depth, splits_cut_at_depth_limit=removed,
                depth_limit_note="J48 has no depth limit: the grown tree is cut at the limit afterwards",
                timer="J48.buildClassifier, after a warm-up build; data read from the float32 file")
    return times, tree, pred if removed == 0 else None


# ---------------------------------------------------------------- YaDT

def run_yadt(a, meta):
    binary = TOOLS / "yadt" / "dTcmd"
    env = {**os.environ, "LD_LIBRARY_PATH": str(binary.parent)}
    times = []
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(a.reps):
            argv = [str(binary), "-fm", f"{a.data}/yadt.names", "-fd", f"{a.data}/train.yadt.csv",
                    "-np", "-m", "2", "-depth", str(a.depth), "-tt", str(a.threads),
                    "-x", f"{tmp}/t.xml", "-tb", f"{tmp}/t.bin"]
            out = subprocess.run(argv, capture_output=True, text=True, check=True, env=env).stdout
            load = float(re.search(r"load time: ([\d.e+-]+) secs", out).group(1))
            total = float(re.search(r"total time: ([\d.e+-]+) secs", out).group(1))
            steps = re.findall(r"size: \d+ depth: \d+ nf: \d+ time: ([\d.e+-]+) secs", out)
            times.append(total - load + sum(float(s) for s in steps))
        tree = parse_yadt_xml(f"{tmp}/t.xml")
        # YaDT's own test predictions: the score file of the saved tree.
        subprocess.run([str(binary), "-bt", f"{tmp}/t.bin", "-ft", f"{a.data}/test.yadt.csv",
                        "-s", f"{tmp}/score.txt"], capture_output=True, check=True, env=env)
        pred = parse_yadt_scores(f"{tmp}/score.txt")
        version = subprocess.run([str(binary)], capture_output=True, text=True, env=env).stdout.split("\n")[0]
    meta.update(version=version.split("(c)")[0].strip(), dtype="f64",
                command=" ".join(["dTcmd"] + argv[5:-4]),
                timer="YaDT's total time without loading the CSV (indexing and growing), "
                      "one fresh process per run")
    return times, tree, pred


def parse_yadt_xml(path):
    """YaDT's -x output: PMML-like nested <Node> elements with a predicate."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    top = next(root.iter("Node"))

    def counts(el):
        dist = {c: 0 for c in treeio.CLASSES}
        for sd in el.findall("ScoreDistribution"):
            dist[sd.get("value")] = int(round(float(sd.get("recordCount"))))
        return dist

    def build(el):
        dist = counts(el)
        node = {"n": sum(dist.values()), "dist": dist, "class": el.get("score")}
        children = el.findall("Node")
        if not children:
            return node
        if len(children) != 2:
            raise ValueError("YaDT: not a binary split")
        preds = [c.find("SimplePredicate") for c in children]
        ops = [p.get("operator") for p in preds]
        if ops == ["lessOrEqual", "greaterThan"]:
            left, right = children
        elif ops == ["greaterThan", "lessOrEqual"]:
            right, left = children
        else:
            raise ValueError(f"YaDT: predicates {ops}")
        p = left.find("SimplePredicate")
        node.update(feature=p.get("field"), op="<=", threshold=float(p.get("value")),
                    left=build(left), right=build(right))
        return node
    return build(top)


def parse_yadt_scores(path):
    pred = []
    for line in open(path):
        parts = line.strip().split(",")
        # "<true class>,<predicted class>,<confidence>"
        if len(parts) == 3 and parts[1] in treeio.CLASSES:
            pred.append(treeio.CLASSES.index(parts[1]))
    return np.array(pred) if pred else None


# ---------------------------------------------------------------- cuML

def run_cuml(a, meta):
    import cuml
    from cuml.ensemble import RandomForestClassifier
    X, y = treeio.load(a.data, "train")
    Xt, _ = treeio.load(a.data, "test")

    def fit(X, y):
        return RandomForestClassifier(
            n_estimators=1, bootstrap=False, max_samples=1.0, max_features=1.0,
            split_criterion="gini" if a.case == "cart" else "entropy", max_depth=a.depth,
            max_leaves=-1, n_bins=128, min_samples_leaf=1 if a.case == "cart" else 2,
            min_samples_split=2 if a.case == "cart" else 4, min_impurity_decrease=0.0,
            n_streams=1, random_state=0, output_type="numpy").fit(X, y)
    fit(X[:WARMUP], y[:WARMUP])
    times = []
    for _ in range(a.reps):
        t = time.perf_counter()
        model = fit(X, y)
        times.append(time.perf_counter() - t)
    acc = model.as_treelite().get_tree_accessor(0)

    def field(name):
        return np.asarray(acc.get_field(name))
    left, right = field("cleft").astype(int), field("cright").astype(int)
    feat, thr, cmp = field("split_index"), field("threshold"), field("cmp")
    lv, lvb, lve = field("leaf_vector"), field("leaf_vector_begin"), field("leaf_vector_end")
    ops = {2: "<", 3: "<="}  # treelite's Operator: kLT = 2, kLE = 3

    def build(i):
        if left[i] < 0:
            probs = lv[lvb[i]:lve[i]]
            return {"class": treeio.CLASSES[int(np.argmax(probs))]}
        return {"feature": treeio.FEATURES[int(feat[i])], "op": ops[int(cmp[i])],
                "threshold": float(thr[i]), "left": build(left[i]), "right": build(right[i])}
    meta.update(version=f"cuML {cuml.__version__}",
                command=f"RandomForestClassifier(n_estimators=1, bootstrap=False, max_features=1.0, "
                        f"split_criterion='{'gini' if a.case == 'cart' else 'entropy'}', "
                        f"max_depth={a.depth}, n_bins=128)",
                timer="RandomForestClassifier.fit from the host float32 arrays (copy, binning, growing), "
                      "after a warm-up fit")
    return times, build(0), np.asarray(model.predict(Xt)).astype(int)


# ---------------------------------------------------------------- XGBoost

def run_xgboost(a, meta):
    import xgboost as xgb
    X, y = treeio.load(a.data, "train")
    Xt, _ = treeio.load(a.data, "test")
    base = float(y.mean())

    def fit(X, y):
        d = xgb.QuantileDMatrix(X, label=y.astype(np.float32), max_bin=256)
        return xgb.train({"device": "cuda", "tree_method": "hist", "objective": "reg:squarederror",
                          "eta": 1.0, "lambda": 0.0, "alpha": 0.0, "gamma": 0.0,
                          "min_child_weight": 1.0, "base_score": base, "max_bin": 256,
                          "max_depth": a.depth, "grow_policy": "depthwise"}, d, num_boost_round=1)
    fit(X[:WARMUP], y[:WARMUP])
    times = []
    for _ in range(a.reps):
        t = time.perf_counter()
        booster = fit(X, y)
        times.append(time.perf_counter() - t)
    # The raw model keeps thresholds, leaf values and base_score as the float32s
    # that predict() adds up; the text dump rounds them.
    raw = json.loads(booster.save_raw("json"))
    learner = raw["learner"]
    tree = learner["gradient_booster"]["model"]["trees"][0]
    margin = np.float32(float(learner["learner_model_param"]["base_score"].strip("[]")))
    left, right = tree["left_children"], tree["right_children"]
    feat, cond = tree["split_indices"], tree["split_conditions"]
    hess = tree["sum_hessian"]

    def build(i):
        n = int(round(hess[i]))  # every row has hessian 1
        if left[i] == -1:
            return {"class": "B" if margin + np.float32(cond[i]) > np.float32(0.5) else "A", "n": n}
        return {"feature": treeio.FEATURES[feat[i]], "op": "<", "threshold": float(np.float32(cond[i])),
                "n": n, "left": build(left[i]), "right": build(right[i])}
    meta.update(version=f"XGBoost {xgb.__version__}",
                command=f"xgb.train(objective='reg:squarederror', lambda=0, eta=1, tree_method='hist', "
                        f"max_bin=256, max_depth={a.depth}, device='cuda'), 1 round",
                timer="QuantileDMatrix + xgb.train from the host float32 arrays, after a warm-up fit")
    booster.set_param({"device": "cpu"})
    pred = (booster.inplace_predict(Xt) > 0.5).astype(int)
    return times, build(0), pred


RUNNERS = {"tree": run_tree, "sklearn": run_sklearn, "rpart": run_rpart, "j48": run_j48,
           "yadt": run_yadt, "cuml": run_cuml, "xgboost": run_xgboost}
LABELS = {"tree": "./tree", "sklearn": "scikit-learn", "rpart": "rpart", "j48": "Weka J48",
          "yadt": "YaDT", "cuml": "cuML", "xgboost": "XGBoost"}
ALGORITHMS = {("tree", "cart"): "CART: exact Gini splits",
              ("tree", "c45"): "C4.5: exact gain-ratio splits, unpruned",
              ("sklearn", "cart"): "CART: exact Gini splits",
              ("rpart", "cart"): "CART: exact Gini splits",
              ("j48", "c45"): "C4.5: gain ratio, unpruned",
              ("yadt", "c45"): "C4.5: gain ratio, unpruned",
              ("cuml", "cart"): "Gini on 128 histogram bins (CART-like)",
              ("cuml", "c45"): "entropy on 128 histogram bins (closest to C4.5)",
              ("xgboost", "cart"): "squared error = Gini on 256 histogram bins (CART-like)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("impl", choices=RUNNERS)
    p.add_argument("case", choices=["cart", "c45"])
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--backend", default="parallel", choices=["serial", "parallel", "cuda"])
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--classes", default=str(HERE / "classes"), help="J48Export's compiled class dir")
    a = p.parse_args()
    name = a.impl + ("-" + a.backend if a.impl == "tree" else "") + f"-{a.case}"
    if a.impl == "yadt":
        name = f"yadt-{a.threads}t-{a.case}"
    threads = 1 if a.impl in ("sklearn", "rpart", "j48") or a.backend == "serial" else a.threads
    gpu = a.impl in ("cuml", "xgboost") or (a.impl == "tree" and a.backend == "cuda")
    meta = {"name": name, "impl": a.impl, "label": LABELS[a.impl], "case": a.case,
            "algorithm": ALGORITHMS[(a.impl, a.case)], "backend": a.backend if a.impl == "tree" else
            ("cuda" if gpu else "cpu"), "threads": threads, "depth_limit": a.depth,
            "host": socket.gethostname().split(".")[0], "cpu": cpu_name(),
            "gpu": gpu_name() if gpu else None, "dataset": "chess 3x3, 1M rows (80/20)",
            "date": time.strftime("%F %T"), "reps": a.reps}
    times, tree, own_pred = RUNNERS[a.impl](a, meta)
    meta.update(times=times, median_seconds=statistics.median(times), min_seconds=min(times),
                max_seconds=max(times))
    treeio.finish(meta, tree, a.data, own_test_pred=own_pred)
    if "tree_printed_train_accuracy" in meta and \
            abs(meta["tree_printed_train_accuracy"] - meta["train_accuracy"]) > 1e-6:
        print(f"warning: ./tree printed train accuracy {meta['tree_printed_train_accuracy']}, "
              f"its dump gives {meta['train_accuracy']}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    treeio.save(f"{a.out}/{name}.json", meta, tree)


if __name__ == "__main__":
    main()
