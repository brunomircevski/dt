#!/usr/bin/env python3
"""Make the train/test split of every dataset in datasets.toml, in each tool's
input format (see benchlib.py for the layout), plus the tiny synthetic
`_baseline` dataset used to measure each tool's runtime footprint.

  bench/.venv/bin/python bench/prepare.py              # every dataset whose source exists
  bench/.venv/bin/python bench/prepare.py covertype_10k susy_50k
  bench/.venv/bin/python bench/prepare.py --force susy # rebuild
"""

import argparse
import json
import shutil
import sys
import time
import tomllib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from benchlib import BASELINE, BENCH, DATA, ROOT, check_csv


def split_rows(n, spec, n_source):
    """Row indices of the training and the test set. With test_source the test
    rows are those read from it, after the source's n_source rows."""
    if "test_source" in spec:
        train, test = np.arange(n_source), np.arange(n_source, n)
        if spec.get("train_rows"):
            train = train[:spec["train_rows"]]
        if spec.get("max_test_rows"):
            test = test[:spec["max_test_rows"]]
        return train, test
    if "test_rows" in spec:
        n_test = int(spec["test_rows"])
    else:
        n_test = int(round(spec.get("test_fraction", 0.2) * n))
    if not 0 < n_test < n:
        raise ValueError(f"test set of {n_test} rows out of {n}")
    if spec.get("split", "random") == "random":
        order = np.random.default_rng(spec.get("seed", 1)).permutation(n)
        test, train = order[:n_test], order[n_test:]
    elif spec["split"] == "tail":
        train, test = np.arange(n - n_test), np.arange(n - n_test, n)
    else:
        raise ValueError(f"unknown split {spec['split']!r}")
    if spec.get("train_rows"):
        train = train[:spec["train_rows"]]
    if spec.get("max_test_rows"):
        test = test[:spec["max_test_rows"]]
    return train, test


def read_source(spec):
    """Features, class names, float32 values and class indices of the source's
    rows, then of test_source's if the spec has one; and the source's row count."""
    table = pacsv.read_csv(ROOT / spec["source"])
    n_source = table.num_rows
    if "test_source" in spec:  # a test set of its own, as for the chess sets
        test = pacsv.read_csv(ROOT / spec["test_source"])
        if test.column_names != table.column_names:
            raise ValueError(f"{spec['test_source']} has other columns than {spec['source']}")
        table = pa.concat_tables([table, test.cast(table.schema)])
    names = table.column_names
    label = names[-1] if spec.get("label", "last") == "last" else spec["label"]
    drop = set(spec.get("drop", [])) | {label}
    features = [name for name in names if name not in drop]
    for name in features:
        if not pa.types.is_integer(table[name].type) and not pa.types.is_floating(table[name].type):
            raise ValueError(f"feature {name!r} is {table[name].type}; only numeric features "
                             "are supported for now")
        if table[name].null_count:
            raise ValueError(f"feature {name!r} has missing values")
    X = np.column_stack([table[name].to_numpy().astype(np.float32) for name in features])
    # The class index is the rank of the label in sorted order, as np.unique
    # would give, without a Python string per row (1G rows for chess_1b).
    encoded = pc.dictionary_encode(pc.cast(table[label], pa.string()))
    encoded = encoded.unify_dictionaries().combine_chunks()
    names = np.array(encoded.dictionary.to_pylist())
    order = np.argsort(names)
    rank = np.empty(len(names), np.int32)
    rank[order] = np.arange(len(names), dtype=np.int32)
    y = rank[encoded.indices.to_numpy()]
    return features, [str(c) for c in names[order]], X, y, n_source


def write_dataset(name, features, classes, X, y, train, test, info, yadt=True):
    out = DATA / name
    tmp = DATA / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    F = X.shape[1]
    parts = {"train": train, "test": test}
    labels = pa.array([f"c{k}" for k in range(len(classes))])
    for part, rows in parts.items():
        Xp = np.ascontiguousarray(X[rows], dtype="<f4")
        Xp.tofile(tmp / f"{part}.f32")
        y[rows].astype("<i4").tofile(tmp / f"{part}.y.i32")
        # Arrow prints float32 in the shortest form that reads back as the same
        # float32 (check_csv checks every row).
        columns = [pa.array(Xp[:, j]) for j in range(F)]
        columns.append(pc.take(labels, pa.array(y[rows])))
        table = pa.table(columns, names=[f"f{j}" for j in range(F)] + ["class"])
        if part == "train":
            pacsv.write_csv(table, tmp / "train.csv", pacsv.WriteOptions(quoting_style="none"))
            check_csv(tmp / "train.csv", Xp, y[rows], header=True)
        if not yadt:
            continue
        pacsv.write_csv(table, tmp / f"{part}.yadt.csv",
                        pacsv.WriteOptions(include_header=False, quoting_style="none"))
        check_csv(tmp / f"{part}.yadt.csv", Xp, y[rows], header=False)
    if yadt:
        with open(tmp / "yadt.names", "w") as names:
            for j in range(F):
                names.write(f"f{j},float,continuous\n")
            names.write("class,string,class\n")
    meta = {"name": name, "features": [f"f{j}" for j in range(F)], "feature_names": features,
            "classes": [f"c{k}" for k in range(len(classes))], "class_names": classes,
            "n_train": int(len(train)), "n_test": int(len(test)), **info}
    json.dump(meta, open(tmp / "meta.json", "w"), indent=1)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


def make_baseline():
    """200 + 50 rows, 4 features, 2 classes: training takes microseconds, so peak
    memory is what the tool needs to start, load and fit at all."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((250, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(250) > 0).astype(np.int32)
    write_dataset(BASELINE, ["x0", "x1", "x2", "x3"], ["0", "1"], X, y,
                  np.arange(200), np.arange(200, 250), {"source": "synthetic"})


def main():
    specs = tomllib.load(open(BENCH / "datasets.toml", "rb"))
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help=f"default: all in datasets.toml ({len(specs)})")
    parser.add_argument("--force", action="store_true", help="rebuild existing datasets")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(specs)
    if unknown:
        sys.exit(f"not in datasets.toml: {', '.join(sorted(unknown))}")
    DATA.mkdir(exist_ok=True)
    if args.force or not (DATA / BASELINE / "meta.json").exists():
        make_baseline()
    sources = {}
    for name in args.datasets or list(specs):
        spec = specs[name]
        if (DATA / name / "meta.json").exists() and not args.force:
            print(f"{name}: already prepared")
            continue
        if not (ROOT / spec["source"]).exists():
            print(f"{name}: skipped, {spec['source']} not found")
            continue
        start = time.perf_counter()
        key = (spec["source"], spec.get("test_source"), spec.get("label", "last"),
               tuple(spec.get("drop", [])))
        if key not in sources:  # variants of one source share one read (kept one at a time)
            sources = {}  # the previous source's arrays go first (chess_1b: tens of GB)
            sources = {key: read_source(spec)}
        features, classes, X, y, n_source = sources[key]
        train, test = split_rows(len(y), spec, n_source)
        write_dataset(name, features, classes, X, y, train, test, {"spec": spec},
                      yadt=spec.get("yadt", True))
        print(f"{name}: {len(train):,} train / {len(test):,} test rows, {len(features)} features, "
              f"{len(classes)} classes ({time.perf_counter() - start:.1f} s)")


if __name__ == "__main__":
    main()
