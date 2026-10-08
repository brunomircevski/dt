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

from benchlib import BASELINE, BENCH, DATA, ROOT


def split_rows(n, spec):
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
    table = pacsv.read_csv(ROOT / spec["source"])
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
    labels = pc.cast(table[label], pa.string()).to_numpy(zero_copy_only=False)
    classes, y = np.unique(labels, return_inverse=True)
    return features, [str(c) for c in classes], X, y.astype(np.int32)


def write_dataset(name, features, classes, X, y, train, test, info):
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
        # float32 (verify_csv checks this on a sample of rows).
        columns = [pa.array(Xp[:, j]) for j in range(F)]
        columns.append(pc.take(labels, pa.array(y[rows])))
        table = pa.table(columns, names=[f"f{j}" for j in range(F)] + ["class"])
        if part == "train":
            pacsv.write_csv(table, tmp / "train.csv", pacsv.WriteOptions(quoting_style="none"))
            verify_csv(tmp / "train.csv", Xp, header=True)
        pacsv.write_csv(table, tmp / f"{part}.yadt.csv",
                        pacsv.WriteOptions(include_header=False, quoting_style="none"))
        verify_csv(tmp / f"{part}.yadt.csv", Xp, header=False)
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


def verify_csv(path, X, header, sample=2000):
    """Sampled rows of the CSV must parse back to exactly the same float32 values."""
    rng = np.random.default_rng(0)
    check = set(rng.choice(len(X), size=min(sample, len(X)), replace=False).tolist())
    with open(path) as handle:
        if header:
            next(handle)
        for i, line in enumerate(handle):
            if i in check:
                values = np.array(line.rstrip("\n").split(",")[:-1], dtype=np.float32)
                if not np.array_equal(values, X[i]):
                    raise AssertionError(f"{path}: row {i} does not round-trip as float32")


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
        key = (spec["source"], spec.get("label", "last"), tuple(spec.get("drop", [])))
        if key not in sources:  # variants of one source share one read (kept one at a time)
            sources = {key: read_source(spec)}
        features, classes, X, y = sources[key]
        train, test = split_rows(len(y), spec)
        write_dataset(name, features, classes, X, y, train, test, {"spec": spec})
        print(f"{name}: {len(train):,} train / {len(test):,} test rows, {len(features)} features, "
              f"{len(classes)} classes ({time.perf_counter() - start:.1f} s)")


if __name__ == "__main__":
    main()
