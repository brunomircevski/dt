#!/usr/bin/env python3
"""Download UCI Covertype and write CSV files for the C++ decision tree loader."""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/5976039"

FEATURE_NAMES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area_1",
    "Wilderness_Area_2",
    "Wilderness_Area_3",
    "Wilderness_Area_4",
    *[f"Soil_Type_{index}" for index in range(1, 41)],
]

CLASS_NAMES = {
    1: "Spruce_Fir",
    2: "Lodgepole_Pine",
    3: "Ponderosa_Pine",
    4: "Cottonwood_Willow",
    5: "Aspen",
    6: "Douglas_Fir",
    7: "Krummholz",
}


def download_covertype() -> np.ndarray:
    print(f"Downloading {DOWNLOAD_URL} ...")
    with urllib.request.urlopen(DOWNLOAD_URL, timeout=120) as response:
        compressed = response.read()

    print("Parsing covertype.data.gz ...")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz_file:
        data = np.genfromtxt(gz_file, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 55:
        raise RuntimeError(f"Unexpected covertype shape: {data.shape}")
    return data


def stratified_sample(data: np.ndarray, target_rows: int, seed: int) -> np.ndarray:
    if target_rows >= data.shape[0]:
        return data

    labels = data[:, -1].astype(int)
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        label_fraction = label_indices.size / labels.size
        label_target = max(1, int(round(target_rows * label_fraction)))
        label_target = min(label_target, label_indices.size)
        chosen = rng.choice(label_indices, size=label_target, replace=False)
        selected_indices.extend(chosen.tolist())

    selected_indices = np.array(selected_indices, dtype=int)
    rng.shuffle(selected_indices)

    if selected_indices.size > target_rows:
        selected_indices = selected_indices[:target_rows]
    elif selected_indices.size < target_rows:
        remaining = np.setdiff1d(np.arange(data.shape[0]), selected_indices, assume_unique=False)
        extra = rng.choice(
            remaining,
            size=target_rows - selected_indices.size,
            replace=False,
        )
        selected_indices = np.concatenate([selected_indices, extra])
        rng.shuffle(selected_indices)

    return data[selected_indices]


def write_csv(path: Path, data: np.ndarray) -> None:
    header = ["Id", *FEATURE_NAMES, "CoverType"]
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(",".join(header) + "\n")
        for row_index, row in enumerate(data, start=1):
            features = ",".join(str(int(value)) for value in row[:-1])
            class_id = int(row[-1])
            label = CLASS_NAMES.get(class_id, str(class_id))
            output.write(f"{row_index},{features},{label}\n")


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    data = download_covertype()
    total_rows = data.shape[0]
    rows_10x = total_rows // 10
    rows_100x = total_rows // 100

    datasets = [
        ("covertype.csv", data),
        ("covertype_10x_smaller.csv", stratified_sample(data, rows_10x, seed=10)),
        ("covertype_100x_smaller.csv", stratified_sample(data, rows_100x, seed=100)),
    ]

    for filename, subset in datasets:
        output_path = DATASETS_DIR / filename
        print(f"Writing {output_path.name} ({subset.shape[0]} rows, {subset.shape[1] - 1} features) ...")
        write_csv(output_path, subset)

    print("\nDone.")
    print(f"  full:  {total_rows:,} rows")
    print(f"  10x:   {rows_10x:,} rows")
    print(f"  100x:  {rows_100x:,} rows")


if __name__ == "__main__":
    main()
