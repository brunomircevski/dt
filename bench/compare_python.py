#!/usr/bin/env python3
"""Compare ./tree's CART and C4.5 with Python CART / C4.5 libraries: time and accuracy.

  python3 bench/compare_python.py prep                   # build train/test splits in WORK
  python3 bench/compare_python.py run covertype_20k      # run the cases, append to WORK/results.jsonl
  python3 bench/compare_python.py table                  # print WORK/results.jsonl as Markdown

WORK defaults to bench/work (BENCH_WORK=... to change; it holds copies of the
training CSVs, ~0.5 GB). Needs numpy pandas pyarrow scikit-learn, and for the
pure-Python C4.5 cases chefboost and c4dot5-decision-tree
(bench/requirements-python.txt).

Every implementation gets the same split: ./tree reads WORK/<name>.train.csv
and writes its tree with --dump, which is then evaluated here on the test rows;
the Python libraries read the same rows from .npy files. Times are training
only (./tree's `train total`; Python: the fit call, including the library's own
preprocessing such as sorting or binning). Loading and prediction are excluded.
Each case runs in a fresh process, one at a time, so runs do not compete for
cores or memory.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("BENCH_WORK", ROOT / "bench" / "work"))
TREE = os.environ.get("TREE_BIN", str(ROOT / "tree_cpu"))
THREADS = os.cpu_count()
SEED = 1

# name: (source CSV, how the 20% test rows are chosen, training-row limit)
DATASETS = {
    "diabetes": ("datasets/diabetes.csv", "shuffle", None),
    "covertype_20k": ("datasets/covertype.csv", "shuffle", 20_000),
    "covertype": ("datasets/covertype.csv", "shuffle", None),
    # SUSY is a Monte Carlo sample in random order and UCI uses its last rows as
    # the test set, so the training CSV is simply the head of the file.
    "susy_1m": ("datasets/supersymmetry.csv", "tail", 1_000_000),
}


# ---------------------------------------------------------------- data

def prep(names):
    import pyarrow.csv as pacsv

    WORK.mkdir(parents=True, exist_ok=True)
    for name in names:
        source, mode, limit = DATASETS[name]
        if (WORK / f"{name}.meta.json").exists():
            print(f"{name}: already prepared")
            continue
        start = time.perf_counter()
        table = pacsv.read_csv(ROOT / source)
        columns = table.column_names
        first = 1 if columns[0].lower() == "id" else 0
        features = columns[first:-1]
        X = np.column_stack([table.column(c).to_numpy().astype(np.float32) for c in features])
        labels = np.array([str(v) for v in table.column(columns[-1]).to_pylist()])
        classes, y = np.unique(labels, return_inverse=True)
        y = y.astype(np.int32)
        n = len(y)
        n_test = int(round(0.2 * n))
        if mode == "shuffle":
            order = np.random.default_rng(SEED).permutation(n)
            train, test = order[n_test:], order[:n_test]
            if limit:
                train, test = train[:limit], test[:limit // 4]
        else:
            n_train = limit or n - n_test
            n_test = n_train // 4 if limit else n_test
            train, test = np.arange(n_train), np.arange(n - n_test, n)
        train_csv = WORK / f"{name}.train.csv"
        if mode == "tail":
            # Byte-identical rows: the head of the original file.
            with open(train_csv, "wb") as out:
                subprocess.run(["head", "-n", str(len(train) + 1), str(ROOT / source)],
                               stdout=out, check=True)
        else:
            import pandas as pd
            frame = pd.DataFrame(X[train], columns=features)
            frame["Label"] = labels[train]
            frame.to_csv(train_csv, index=False, float_format="%.9g")
        for part, rows in (("train", train), ("test", test)):
            np.save(WORK / f"{name}.X_{part}.npy", np.ascontiguousarray(X[rows]))
            np.save(WORK / f"{name}.y_{part}.npy", y[rows])
        json.dump({"features": features, "classes": classes.tolist(),
                   "train": len(train), "test": len(test)},
                  open(WORK / f"{name}.meta.json", "w"))
        print(f"{name}: {len(train)} train / {len(test)} test, {len(features)} features, "
              f"{len(classes)} classes ({time.perf_counter() - start:.1f} s)")


def load(name):
    meta = json.load(open(WORK / f"{name}.meta.json"))
    arrays = {key: np.load(WORK / f"{name}.{key}.npy")
              for key in ("X_train", "y_train", "X_test", "y_test")}
    return meta, arrays


# ---------------------------------------------------------------- ./tree

def parse_dump(path, features, classes):
    """./tree --dump -> flat arrays (preorder); leaves have feature -1."""
    node_re = re.compile(r"^\s*\w+ \[n=\d+\]: (.*)$")
    index = {name: i for i, name in enumerate(features)}
    class_index = {name: i for i, name in enumerate(classes)}
    feature, threshold, value = [], [], []
    for line in open(path):
        match = node_re.match(line)
        if not match:
            continue
        body = match.group(1)
        if body.startswith("Leaf -> "):
            feature.append(-1)
            threshold.append(0.0)
            value.append(class_index[body[len("Leaf -> "):]])
        else:
            name, cut = re.match(r"if (.+) <= (\S+)$", body).groups()
            feature.append(index[name])
            threshold.append(float(cut))
            value.append(-1)
    feature = np.array(feature)
    # Preorder: the left child follows its parent; the right child follows the
    # left subtree. One backwards pass gives subtree sizes.
    count = len(feature)
    size = np.ones(count, dtype=np.int64)
    left = np.full(count, -1)
    right = np.full(count, -1)
    for node in range(count - 1, -1, -1):
        if feature[node] >= 0:
            left[node] = node + 1
            right[node] = node + 1 + size[node + 1]
            size[node] = 1 + size[left[node]] + size[right[node]]
    return feature, np.array(threshold), np.array(value), left, right


def predict_flat(tree, X):
    feature, threshold, value, left, right = tree
    node = np.zeros(len(X), dtype=np.int64)
    active = np.arange(len(X))
    while len(active):
        f = feature[node[active]]
        inner = f >= 0
        active = active[inner]
        if not len(active):
            break
        f = f[inner]
        current = node[active]
        goes_left = X[active, f].astype(np.float64) <= threshold[current]
        node[active] = np.where(goes_left, left[current], right[current])
    return value[node]


def run_tree(name, flags, meta, arrays):
    dump = WORK / f"{name}.dump.txt"
    command = [TREE] + flags + ["--dump", str(dump), str(WORK / f"{name}.train.csv")]
    out = subprocess.run(command, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stdout + out.stderr)
    seconds = float(re.search(r"train total\s+([\d.]+) ms", out.stdout).group(1)) / 1000
    nodes = int(re.search(r"nodes ([\d,]+)", out.stdout).group(1).replace(",", ""))
    depth = int(re.search(r"depth (\d+)", out.stdout).group(1))
    tree = parse_dump(dump, meta["features"], meta["classes"])
    dump.unlink()
    accuracy = float(np.mean(predict_flat(tree, arrays["X_test"]) == arrays["y_test"]))
    return {"seconds": seconds, "nodes": nodes, "depth": depth, "accuracy": accuracy}


# ---------------------------------------------------------------- Python libraries

def timed(fit):
    start = time.perf_counter()
    model = fit()
    return model, time.perf_counter() - start


def sklearn_result(model, seconds, arrays):
    accuracy = float(np.mean(model.predict(arrays["X_test"]) == arrays["y_test"]))
    return {"seconds": seconds, "nodes": int(model.tree_.node_count),
            "depth": int(model.get_depth()), "accuracy": accuracy}


def sk_tree(arrays, threads, **kwargs):
    from sklearn.tree import DecisionTreeClassifier
    X, y = arrays["X_train"], arrays["y_train"]
    model, seconds = timed(lambda: DecisionTreeClassifier(random_state=0, **kwargs).fit(X, y))
    return sklearn_result(model, seconds, arrays)


CANDIDATES = 16


def candidate_alphas(X, y):
    """ccp_alpha candidates from scikit-learn's pruning path (one more full fit):
    0 plus 15 alphas whose pruned trees have geometrically spaced sizes, each
    taken between two path values, as Breiman does."""
    from sklearn.tree import DecisionTreeClassifier
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X, y).ccp_alphas
    path = np.unique(path[path > 0])
    steps = np.unique(np.geomspace(1, len(path) - 1, CANDIDATES - 1).astype(int))
    upper = path[len(path) - steps]
    lower = path[len(path) - steps - 1]
    return [0.0] + sorted(np.sqrt(lower * upper).tolist())


def sk_ccp_holdout(arrays, threads):
    """The scikit-learn way to choose ccp_alpha on a validation set: take
    candidates from the pruning path, fit one tree per candidate on 2/3 of the
    rows (scikit-learn cannot prune a fitted tree), score each on the other 1/3,
    take the simplest within 1 SE of the best (Breiman's rule, as ./tree does).
    The candidate trees are fitted in parallel (threads)."""
    from joblib import Parallel, delayed
    from sklearn.tree import DecisionTreeClassifier
    X, y = arrays["X_train"], arrays["y_train"]

    def fit_score(alpha, Xg, yg, Xv, yv):
        model = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha).fit(Xg, yg)
        return model, float(np.mean(model.predict(Xv) != yv))

    def search():
        order = np.random.default_rng(SEED).permutation(len(y))
        n_val = len(y) // 3
        val, grow = order[:n_val], order[n_val:]
        Xg, yg, Xv, yv = X[grow], y[grow], X[val], y[val]
        alphas = candidate_alphas(Xg, yg)
        fitted = Parallel(n_jobs=threads, prefer="threads")(
            delayed(fit_score)(alpha, Xg, yg, Xv, yv) for alpha in alphas)
        errors = np.array([error for _, error in fitted])
        best = errors.min()
        se = np.sqrt(best * (1 - best) / n_val)
        choice = max(i for i in range(len(alphas)) if errors[i] <= best + se)
        return fitted[choice][0]

    model, seconds = timed(search)
    result = sklearn_result(model, seconds, arrays)
    result["alpha"] = float(model.ccp_alpha)
    return result


def sk_ccp_cv(arrays, threads, folds=10):
    """GridSearchCV over ccp_alpha candidates from the pruning path with K-fold
    CV: 1 + 16 x K + 1 tree fits, in parallel over processes."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    X, y = arrays["X_train"], arrays["y_train"]

    def fit():
        grid = {"ccp_alpha": candidate_alphas(X, y)}
        return GridSearchCV(DecisionTreeClassifier(random_state=0), grid,
                            cv=folds, n_jobs=threads).fit(X, y)

    search, seconds = timed(fit)
    result = sklearn_result(search.best_estimator_, seconds, arrays)
    result["alpha"] = float(search.best_params_["ccp_alpha"])
    return result


def frame_for(arrays, meta, part, target):
    import pandas as pd
    frame = pd.DataFrame(arrays[f"X_{part}"].astype(np.float64), columns=meta["features"])
    # Both libraries expect object columns (pandas 3 would infer its str dtype).
    frame[target] = pd.Series(np.array(meta["classes"], dtype=object)[arrays[f"y_{part}"]],
                              dtype=object)
    return frame


def chefboost_c45(arrays, threads, meta):
    """ChefBoost C4.5 (pure Python/pandas; gain ratio, no pruning). A numeric
    feature with more than 20 values is only cut at its min, max, mean and
    mean +- 1..3 std, not at every value as C4.5 does."""
    import contextlib
    import io
    import logging
    from chefboost import Chefboost
    logging.disable(logging.INFO)
    frame = frame_for(arrays, meta, "train", "Decision")
    config = {"algorithm": "C4.5", "enableParallelism": threads > 1, "num_cores": threads,
              "max_depth": 1000}  # default is 5
    os.chdir(WORK)  # ChefBoost writes its rules as Python files under ./outputs and imports them
    sys.path.insert(0, str(WORK))
    with contextlib.redirect_stdout(io.StringIO()):
        model, seconds = timed(lambda: Chefboost.fit(frame, config=config, silent=True))
    test = frame_for(arrays, meta, "test", "Decision")
    features = test.drop(columns=["Decision"]).values.tolist()
    predictions = [Chefboost.predict(model, row) for row in features]
    accuracy = float(np.mean(np.array(predictions, dtype=object) == test["Decision"].values))
    # Numeric features become binary tests, so the rules form a binary tree.
    rules = open(WORK / "outputs" / "rules" / "rules.py").read().splitlines()
    leaves = sum(line.strip().startswith("return ") for line in rules)
    return {"seconds": seconds, "nodes": 2 * leaves - 1, "accuracy": accuracy}


def c4dot5_c45(arrays, threads, meta):
    """c4dot5-decision-tree (pure Python/pandas C4.5, unpruned)."""
    from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier
    frame = frame_for(arrays, meta, "train", "target")
    attributes = {name: "continuous" for name in meta["features"]}
    model = DecisionTreeClassifier(attributes, max_depth=10_000, node_purity=1.0, min_instances=2)
    _, seconds = timed(lambda: model.fit(frame))
    test = frame_for(arrays, meta, "test", "target").drop(columns=["target"])
    predictions = np.array(model.predict(test), dtype=object)
    expected = np.array(meta["classes"], dtype=object)[arrays["y_test"]]
    return {"seconds": seconds, "nodes": len(model.get_nodes()),
            "accuracy": float(np.mean(predictions == expected))}


# ---------------------------------------------------------------- cases

# (dataset group, algorithm, implementation, threads label, how to run it)
# A runner is ("tree", flags) or ("py", function name, kwargs). "all" = every core.
def cases():
    tree_cart = [
        ("CART, full tree", ["--cart", "--no-prune"]),
        ("CART, test-sample pruning", ["--cart"]),
        ("CART, 10-fold CV pruning", ["--cart", "--cv", "10"]),
        ("CART, depth 12", ["--cart", "--no-prune", "-d", "12"]),
        ("C4.5", ["--c45"]),
        ("C4.5, unpruned", ["--c45", "--no-prune"]),
    ]
    out = []
    for algorithm, flags in tree_cart:
        out.append((algorithm, "./tree", 1, ("tree", ["--serial"] + flags)))
        out.append((algorithm, "./tree", THREADS, ("tree", ["--parallel"] + flags)))
    py = [
        ("CART, full tree", "scikit-learn", [1], "sk_tree", {"criterion": "gini"}),
        ("CART, test-sample pruning", "scikit-learn", [1, THREADS], "sk_ccp_holdout", {}),
        ("CART, 10-fold CV pruning", "scikit-learn", [1, THREADS], "sk_ccp_cv", {}),
        ("CART, depth 12", "scikit-learn", [1], "sk_tree", {"criterion": "gini", "max_depth": 12}),
        ("C4.5, unpruned", "scikit-learn entropy", [1], "sk_tree",
         {"criterion": "entropy", "min_samples_leaf": 2}),
        ("C4.5, unpruned", "ChefBoost", [1, THREADS], "chefboost_c45", {}),
        ("C4.5, unpruned", "c4dot5", [1], "c4dot5_c45", {}),
    ]
    for algorithm, implementation, threads_list, function, kwargs in py:
        for threads in threads_list:
            out.append((algorithm, implementation, threads, ("py", function, kwargs)))
    return out


# What runs on which dataset (slow cases are limited to small data).
SKIP = {
    "chefboost_c45": {"covertype", "susy_1m"},
    "c4dot5_c45": {"covertype", "susy_1m"},
    "sk_ccp_cv": {"susy_1m"},
}


def run_case(name, case, repeat):
    algorithm, implementation, threads, runner = case
    meta, arrays = load(name)
    if runner[0] == "py" and runner[1] in ("sk_tree",):
        # Untimed warm-up (lazy imports, first-call costs).
        small = {key: value[:1000] for key, value in arrays.items()}
        globals()[runner[1]](small, threads, **runner[2])
    best = None
    for _ in range(repeat):
        if runner[0] == "tree":
            flags = runner[1] + (["--threads", str(threads)] if threads > 1 else [])
            result = run_tree(name, flags, meta, arrays)
        else:
            function = globals()[runner[1]]
            kwargs = dict(runner[2])
            if runner[1] in ("chefboost_c45", "c4dot5_c45"):
                kwargs["meta"] = meta
            result = function(arrays, threads, **kwargs)
        if best is None or result["seconds"] < best["seconds"]:
            best = result
    best.update(dataset=name, algorithm=algorithm, implementation=implementation, threads=threads)
    return best


def run(names, only, skip, repeat, timeout):
    results = WORK / "results.jsonl"
    for name in names:
        for case in cases():
            algorithm, implementation, threads, runner = case
            if runner[0] == "py" and name in SKIP.get(runner[1], ()):
                continue
            label = f"{name} | {algorithm} | {implementation} | {threads} thr"
            if (only and not re.search(only, label)) or (skip and re.search(skip, label)):
                continue
            payload = json.dumps([name, cases().index(case), repeat])
            start = time.perf_counter()
            try:
                out = subprocess.run([sys.executable, __file__, "_one", payload],
                                     capture_output=True, text=True, timeout=timeout)
                if out.returncode != 0:
                    raise RuntimeError(out.stderr.strip().splitlines()[-1])
                result = json.loads(out.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                result = {"dataset": name, "algorithm": algorithm, "implementation": implementation,
                          "threads": threads, "timeout": timeout}
            except Exception as error:  # noqa: BLE001 - record and keep going
                result = {"dataset": name, "algorithm": algorithm, "implementation": implementation,
                          "threads": threads, "error": str(error)[:300]}
            with open(results, "a") as handle:
                handle.write(json.dumps(result) + "\n")
            shown = {k: v for k, v in result.items()
                     if k not in ("dataset", "algorithm", "implementation", "threads")}
            print(f"{label}: {shown} [{time.perf_counter() - start:.0f} s wall]", flush=True)


def table():
    rows = [json.loads(line) for line in open(WORK / "results.jsonl")]
    latest = {}
    for row in rows:  # the last run of a case wins
        latest[(row["dataset"], row["algorithm"], row["implementation"], row["threads"])] = row
    print("| Dataset | Algorithm | Implementation | Threads | Train time | Nodes | Test acc. |")
    print("|---|---|---|---:|---:|---:|---:|")
    for (dataset, algorithm, implementation, threads), row in latest.items():
        if "timeout" in row:
            time_text, nodes, accuracy = f"> {row['timeout']} s", "", ""
        elif "error" in row:
            time_text, nodes, accuracy = "error", "", ""
        else:
            time_text = f"{row['seconds']:.3g} s"
            nodes = f"{row['nodes']:,}" if "nodes" in row else ""
            accuracy = f"{100 * row['accuracy']:.2f}%"
        print(f"| {dataset} | {algorithm} | {implementation} | {threads} | {time_text} | "
              f"{nodes} | {accuracy} |")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "_one":
        name, index, repeat = json.loads(sys.argv[2])
        print(json.dumps(run_case(name, cases()[index], repeat)))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prep", "run", "table"])
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--only", help="regex on 'dataset | algorithm | implementation | N thr'")
    parser.add_argument("--skip", help="regex of cases to leave out")
    parser.add_argument("--repeat", type=int, default=1, help="best of N runs")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per case")
    args = parser.parse_args()
    if args.command == "prep":
        prep(args.datasets)
    elif args.command == "run":
        run(args.datasets, args.only, args.skip, args.repeat, args.timeout)
    else:
        table()


if __name__ == "__main__":
    main()
