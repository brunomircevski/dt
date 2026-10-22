#!/usr/bin/env python3
"""One-off (2026-10-22): datasets/chess_*.csv from the text of their first build
(x,y,class; A/B) to ./tree's format (make_datasets.chess_text), so that
bench/prepare.py links each set's train.csv to its CSV instead of keeping a
second copy (20.8 GB for chess_1b).

  bench/.venv/bin/python -I bench/cluster/chess_tree_format.py

Each file must have its first build's SHA-256. A training file's new text must
be bench/data/<name>/train.csv byte for byte; that file then replaces the CSV
(nothing is written) and train.csv links to it. The test files are converted.
bench/data/<name> is not prepared again, so its meta.json keeps the names
x, y, A and B.
"""

import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_datasets import CHESS, CHESS_HEADER, OUT, chess_text  # noqa: E402

ROOT = OUT.parent
# make_datasets.EXPECTED of the first build (2026-10-19), header x,y,class.
FIRST_BUILD = {
    "chess_1m": "eeff397d59634960841140a732139a2a0b351db6d538a8bccbda7b385cffdcb1",
    "chess_1m_test": "764f54b147078c4aad8e5d5279aac81a32ef5d0997b55d1e38f80ad3767b653a",
    "chess_10m": "fff4315173bcef7f9672d689b058b3996b57bb98ca6ab8ec7212a853501b0add",
    "chess_10m_test": "758a2c10f767f77de92616d43b89719c75cacba91640cf75e710fe9560b87742",
    "chess_100m": "e25758a82f61198ac562443084baac161661eefe571ca6c2fac6a0a097742555",
    "chess_100m_test": "758a2c10f767f77de92616d43b89719c75cacba91640cf75e710fe9560b87742",
    "chess_1b": "8ca94cd83e586c94ae270643bf52576132faff8504f34e3b4d9ac8aedec6393d",
    "chess_1b_test": "764f54b147078c4aad8e5d5279aac81a32ef5d0997b55d1e38f80ad3767b653a",
}
FIRST_HEADER = b"x,y,class\n"


def blocks(path):
    """The file's rows after the header, in blocks of whole lines."""
    with open(path, "rb") as f:
        assert f.readline() == FIRST_HEADER, f"{path}: not the first build's header"
        tail = b""
        while block := f.read(1 << 26):
            text = tail + block
            cut = text.rfind(b"\n") + 1
            yield text[:cut]
            tail = text[cut:]
        assert not tail, f"{path}: no newline at the end"


def bounded_map(pool, function, items, window=32):
    """pool.map, with at most window items in flight (Executor.map takes them all at once)."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == window:
            yield from pool.map(function, batch)
            batch = []
    yield from pool.map(function, batch)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def convert(name, pool):
    path = OUT / f"{name}.csv"
    with open(path, "rb") as f:
        if f.readline() == CHESS_HEADER.encode():
            print(f"{name}: already in ./tree's format")
            return
    old = sha256(path)
    assert old == FIRST_BUILD[name], f"{path}: sha256 {old}, not the first build's"
    prepared = ROOT / "bench" / "data" / name / "train.csv"
    train = not name.endswith("_test")
    digest = hashlib.sha256(CHESS_HEADER.encode())
    out = None if train else open(path.with_suffix(".csv.partial"), "w", newline="")
    if out:
        out.write(CHESS_HEADER)
    rows = 0
    for text in bounded_map(pool, chess_text, blocks(path)):
        digest.update(text.encode("ascii"))
        rows += text.count("\n")
        if out:
            out.write(text)
    new = digest.hexdigest()
    print(f"{name}: {rows:,} rows, sha256 {old} -> {new}")
    if out:
        out.close()
        path.with_suffix(".csv.partial").rename(path)
        return
    kept = sha256(prepared)
    assert kept == new, f"{prepared}: sha256 {kept}, not the new text's"
    # Each step atomic, so that runs reading train.csv meanwhile never miss it:
    # the CSV becomes a hard link to train.csv, then train.csv a symbolic link
    # to the CSV (what prepare.py makes).
    for temporary in (path.with_suffix(".csv.partial"), prepared.with_suffix(".csv.partial")):
        temporary.unlink(missing_ok=True)
    os.link(prepared, path.with_suffix(".csv.partial"))
    os.replace(path.with_suffix(".csv.partial"), path)
    prepared.with_suffix(".csv.partial").symlink_to(os.path.relpath(path, prepared.parent))
    os.replace(prepared.with_suffix(".csv.partial"), prepared)
    print(f"  {path.relative_to(ROOT)} is {prepared.relative_to(ROOT)}'s bytes; "
          f"{prepared.relative_to(ROOT)} links to it")


def main():
    with ProcessPoolExecutor(min(16, len(os.sched_getaffinity(0)))) as pool:
        for name in CHESS:
            for part in (name, f"{name}_test"):
                convert(part, pool)


if __name__ == "__main__":
    main()
