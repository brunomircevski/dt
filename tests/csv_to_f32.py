#!/usr/bin/env python3
"""Write CSVs as the binary dataset ./tree also reads (as bench/tools/prepare.py
writes them; standard library only, for tests/check.sh):

  tests/csv_to_f32.py OUT train.csv [test.csv]

OUT/train.f32 (row-major little-endian float32), OUT/train.y.i32 (int32 class
index per row), likewise test.* for test.csv, and OUT/meta.json. As in
prepare.py, meta.json's features / classes are f<i> / c<k>, and the CSV's own
names, if it has others, are csv_features / csv_classes. A CSV as Arrow
writes it (prepare.py: header "f0",...,"class", labels c<k>) has no names of
its own: class c<k> is index k. Other classes are numbered in order of first
appearance, not sorted, so that ./tree must sort them as its CSV loader does.
"""

import json
import re
import struct
import sys
from array import array
from fractions import Fraction
from pathlib import Path


def float32(text):
    """The float32 nearest to the decimal `text` (ties to even), as
    std::from_chars gives it. Python parses to a double first; rounding that
    to float32 is exact unless the double lies on a midpoint of two float32s."""
    value = struct.unpack("<f", struct.pack("<f", float(text)))[0]
    if value != value or value in (float("inf"), float("-inf")):
        return value
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    for step in (-1, 1):
        neighbour = struct.unpack("<f", struct.pack("<I", (bits + step) & 0xFFFFFFFF))[0]
        if float(text) == (value + neighbour) / 2:  # a midpoint: decide on the exact decimal
            exact, middle = Fraction(text), Fraction(value + neighbour) / 2
            if exact != middle:
                return neighbour if (exact > middle) == (neighbour > value) else value
    return value


def read(path):
    """Header names and rows as ./tree's CSV loader splits them: at every ',',
    quotes kept, empty lines skipped."""
    with open(path, newline="") as handle:
        rows = [line.rstrip("\n").removesuffix("\r").split(",") for line in handle]
    rows = [row for row in rows if row != [""]]
    header, body = rows[0], rows[1:]
    skip = 1 if header[0].lower() == "id" else 0
    return header[skip:-1], [([float32(v) for v in row[skip:-1]], row[-1]) for row in body]


def main():
    out, paths = Path(sys.argv[1]), sys.argv[2:]
    out.mkdir(parents=True, exist_ok=True)
    parts = [read(path) for path in paths]
    features = parts[0][0]
    if any(names != features for names, _ in parts):
        sys.exit(f"{paths}: the headers differ")
    arrow = features == [f'"f{j}"' for j in range(len(features))]
    labels_seen = {label for _, rows in parts for _, label in rows}
    if arrow and all(re.fullmatch(r"c\d+", label) for label in labels_seen):
        classes = [f"c{k}" for k in range(1 + max(int(label[1:]) for label in labels_seen))]
    else:
        arrow, classes = False, []
    counts = {}
    for part, (_, rows) in zip(("train", "test"), parts):
        values, labels = array("f"), array("i")
        for row, label in rows:
            if label not in classes:
                classes.append(label)
            values.extend(row)
            labels.append(classes.index(label))
        if sys.byteorder != "little":
            values.byteswap()
            labels.byteswap()
        with open(out / f"{part}.f32", "wb") as handle:
            values.tofile(handle)
        with open(out / f"{part}.y.i32", "wb") as handle:
            labels.tofile(handle)
        counts[f"n_{part}"] = len(rows)
    meta = {"features": [f"f{j}" for j in range(len(features))],
            "classes": [f"c{k}" for k in range(len(classes))],
            "n_train": counts["n_train"], "n_test": counts.get("n_test", 0)}
    if not arrow:
        meta.update(csv_features=features, csv_classes=classes)
    with open(out / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=1)


if __name__ == "__main__":
    main()
