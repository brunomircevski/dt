#!/usr/bin/env python3
"""ECBDL'14 official test set (test_set.arff from cruncher.ico2s.org/bdcomp,
labels released 2014-07-07) -> ecbdl_test.csv, in the same layout as the
training CSV (dt's bench/cluster/make_datasets.py:ecbdl): 539 real attributes
as float32, the 92 nominal ones as integer codes of their declared values in
sorted order, label 0/1 last. The nominal value sets are taken from the
TRAINING ARFF's header so the codes match the training CSV; the test header
must declare the same attributes.

  python -I bench/cluster/ecbdl_test.py <training_set.arff> <test_set.arff> <out.csv>

mordor3: ~/datasets/ecbdl/ecbdl_test.csv (see bench/cluster/README.md, Data).
"""
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


def header(path):
    names, kinds, lines = [], [], []
    with open(path, "rb") as arff:
        while line := arff.readline():
            text = line.decode().strip()
            if text.lower().startswith("@attribute"):
                _, name, kind = text.split(None, 2)
                names.append(name)
                kinds.append(kind)
            elif text.lower() == "@data":
                return names, kinds, arff.tell()
    raise SystemExit(f"{path}: no @data")


def main(train_path, test_path, out_path):
    names, kinds, _ = header(train_path)
    tnames, tkinds, offset = header(test_path)
    assert tnames == names, "attribute names differ"
    for n, a, b in zip(names, kinds, tkinds):
        if a != b:
            print(f"note: {n}: train {a} / test {b}", file=sys.stderr)
    nominal = {}
    for name, kind in zip(names, kinds):
        if kind.startswith("{"):
            nominal[name] = sorted(v.strip() for v in kind.strip("{}").split(","))
        else:
            assert kind.lower() in ("real", "numeric", "integer"), kind
    label = names[-1]
    assert nominal.pop(label) == ["0", "1"], label
    assert (len(names), len(nominal)) == (632, 92), (len(names), len(nominal))
    sets = {name: pa.array(values) for name, values in nominal.items()}
    counts = [0, 0]
    with open(test_path, "rb") as arff, open(out_path, "w") as out:
        arff.seek(offset)
        reader = pacsv.open_csv(
            arff, read_options=pacsv.ReadOptions(column_names=names, block_size=1 << 26),
            convert_options=pacsv.ConvertOptions(column_types={
                name: pa.string() if name in nominal or name == label else pa.float32()
                for name in names}))
        out.write(",".join(names) + "\n")
        rows = 0
        for batch in reader:
            columns = []
            for name, column in zip(names, batch.columns):
                assert not column.null_count, f"{name}: missing values"
                if name in sets:
                    column = pc.index_in(column, value_set=sets[name])
                    assert not column.null_count, f"{name}: a value outside {nominal[name]}"
                elif name == label:
                    assert pc.all(pc.is_in(column, value_set=pa.array(["0", "1"]))).as_py(), \
                        "label not 0/1 (unlabelled test set?)"
                    ones = pc.sum(pc.equal(column, "1")).as_py() or 0
                    counts[1] += ones
                    counts[0] += batch.num_rows - ones
                columns.append(column)
            text = pa.BufferOutputStream()
            pacsv.write_csv(pa.record_batch(columns, names=names), text,
                            pacsv.WriteOptions(include_header=False, quoting_style="none"))
            out.write(text.getvalue().to_pybytes().decode())
            rows += batch.num_rows
    print(f"rows {rows}, columns {len(names)}, label 0: {counts[0]}, label 1: {counts[1]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
