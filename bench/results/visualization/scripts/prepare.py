#!/usr/bin/env python3
"""Split chess_1m.csv 80/20 into the files every tool reads.

  prepare.py SOURCE.csv OUT_DIR

The rows are i.i.d. uniform points, so the split is the first 800,000 rows
for training and the last 200,000 for testing. Written:
  train.csv, test.csv          header x,y,class, the source's text (./tree)
  train.yadt.csv, test.yadt.csv, yadt.names   the same rows without header (YaDT)
  train.f32, test.f32, *.y.i32 float32 rows and class index A=0, B=1
                               (scikit-learn, rpart, J48, cuML, XGBoost)
  train.f64.npy, test.f64.npy  the text parsed as double (YaDT's view)
  meta.json
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

source, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
lines = source.read_text().splitlines()
assert lines[0] == "x,y,class", lines[0]
rows = lines[1:]
n_train = len(rows) * 8 // 10
parts = {"train": rows[:n_train], "test": rows[n_train:]}
for part, text in parts.items():
    (out / f"{part}.csv").write_text("x,y,class\n" + "\n".join(text) + "\n")
    (out / f"{part}.yadt.csv").write_text("\n".join(text) + "\n")
    cols = [line.split(",") for line in text]
    X = np.array([[float(a), float(b)] for a, b, _ in cols], dtype=np.float64)
    y = np.array([{"A": 0, "B": 1}[c] for _, _, c in cols], dtype="<i4")
    np.save(out / f"{part}.f64.npy", X)
    X.astype("<f4").tofile(out / f"{part}.f32")
    y.tofile(out / f"{part}.y.i32")
    print(f"{part}: {len(y):,} rows, {100 * y.mean():.2f}% B")
(out / "yadt.names").write_text("x,float,continuous\ny,float,continuous\nclass,string,class\n")
meta = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split": "first 80% train, last 20% test", "features": ["x", "y"], "classes": ["A", "B"],
        "n_train": len(parts["train"]), "n_test": len(parts["test"])}
json.dump(meta, open(out / "meta.json", "w"), indent=1)
