#!/usr/bin/env python3
"""Create downsampled covtype train/test CSVs (no external deps).

Reads:
  data/covtype-train.csv
  data/covtype-test.csv

Writes:
  data/covtype-train-small.csv  (10x smaller than full)
  data/covtype-test-small.csv
  data/covtype-train-tiny.csv   (100x smaller than full)
  data/covtype-test-tiny.csv

Sampling is uniform without replacement with a fixed seed.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Iterable, Tuple


def count_data_rows(path: str) -> int:
    # Number of non-header lines.
    n = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        next(f)  # header
        for _ in f:
            n += 1
    return n


def sample_indices(n: int, factor: int, seed: int) -> set[int]:
    if factor <= 1:
        raise ValueError("factor must be >= 2")
    k = n // factor
    rng = random.Random(seed)
    return set(rng.sample(range(n), k))


def downsample_csv(in_path: str, out_path: str, factor: int, seed: int) -> Tuple[int, int]:
    n = count_data_rows(in_path)
    selected = sample_indices(n, factor=factor, seed=seed)

    kept = 0
    with open(in_path, "r", encoding="utf-8", newline="") as fin, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as fout:
        header = next(fin)
        fout.write(header)

        # 0-based index over data rows (excluding header)
        i = 0
        for line in fin:
            if i in selected:
                fout.write(line)
                kept += 1
            i += 1

    return n, kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["small", "tiny", "both"],
        default="both",
        help="Which outputs to generate (default: both).",
    )
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed (default: 1337)")
    args = parser.parse_args()

    in_pairs = [("data/covtype-train.csv", "train"), ("data/covtype-test.csv", "test")]
    for in_path, _ in in_pairs:
        if not os.path.exists(in_path):
            raise SystemExit(f"Missing input file: {in_path}")

    jobs: list[tuple[str, str, int]] = []
    if args.mode in ("small", "both"):
        for in_path, kind in in_pairs:
            jobs.append((in_path, f"data/covtype-{kind}-small.csv", 10))
    if args.mode in ("tiny", "both"):
        for in_path, kind in in_pairs:
            jobs.append((in_path, f"data/covtype-{kind}-tiny.csv", 100))

    for in_path, out_path, factor in jobs:
        n, kept = downsample_csv(in_path, out_path, factor=factor, seed=args.seed)
        ratio = (kept / n) if n else 0.0
        print(f"{in_path} -> {out_path}: kept {kept} / {n} rows ({ratio:.2%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
