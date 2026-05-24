#!/usr/bin/env python3
"""Benchmark decision-tree implementations and write porównanie.csv + porównanie.svg."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "datasets" / "covertype_10x_smaller.csv"
CSV_OUTPUT = ROOT / "porównanie.csv"
SVG_OUTPUT = ROOT / "porównanie.svg"
BENCHMARK_BINARY = ROOT / ".benchmark_tree"
MAX_DEPTH = 5
REQUIRED_DEPTH = 5
CPP_THREAD_COUNT = 28

COMPILE_COMMAND = [
    "g++",
    "-std=c++17",
    "-O2",
    "-pthread",
    "-I",
    str(ROOT),
    str(PYTHON_DIR / "benchmark_tree_main.cpp"),
    str(ROOT / "c45_tree.cpp"),
    str(ROOT / "dataset.cpp"),
    str(ROOT / "node.cpp"),
    str(ROOT / "pruning" / "cart_prune_cost_complexity.cpp"),
    str(ROOT / "pruning" / "c45_prune_c45_pessimistic.cpp"),
    "-o",
    str(BENCHMARK_BINARY),
]


@dataclass
class BenchmarkRow:
    algorithm: str
    time_ms: float
    accuracy_pct: float
    node_count: int
    tree_depth: int
    sort_key: int


def load_covertype_dataset(csv_path: Path):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    df = df.replace(["NA", "na", "?", "N/A"], np.nan).dropna()

    target_col = df.columns[-1]
    feature_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        try:
            pd.to_numeric(df[col])
            feature_cols.append(col)
        except ValueError:
            continue

    df = df[feature_cols + [target_col]].copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col])
    df[target_col] = df[target_col].astype(str)

    features = df[feature_cols].values.tolist()
    labels = df[target_col].tolist()
    frame = df.rename(columns={target_col: "target"})
    attributes_map = {col: "continuous" for col in feature_cols}
    return feature_cols, features, labels, frame, attributes_map


def compute_python_c45_stats(node):
    if node is None:
        return 0, -1
    if hasattr(node, "get_class_name"):
        return 1, 0
    node_count = 1
    max_depth = -1
    for child in node.get_children():
        child_count, child_depth = compute_python_c45_stats(child)
        node_count += child_count
        max_depth = max(max_depth, child_depth)
    return node_count, 1 + max_depth


def _worker_sklearn_cart(dataset_path: str) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    _, features, labels, _, _ = load_covertype_dataset(Path(dataset_path))
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        ccp_alpha=0.0,
        random_state=0,
    )
    build_start = time.perf_counter()
    model.fit(features, labels)
    build_time_ms = (time.perf_counter() - build_start) * 1000.0

    predictions = model.predict(features)
    correct = sum(prediction == actual for prediction, actual in zip(predictions, labels))

    return {
        "algorithm": "Python CART (scikit-learn)",
        "time_ms": build_time_ms,
        "accuracy_pct": 100.0 * correct / len(labels),
        "node_count": int(model.tree_.node_count),
        "tree_depth": int(model.get_depth()),
        "sort_key": 1,
    }


def _worker_python_c45(dataset_path: str) -> dict:
    from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier

    _, _, _, frame, attributes_map = load_covertype_dataset(Path(dataset_path))
    model = DecisionTreeClassifier(
        attributes_map=attributes_map,
        max_depth=MAX_DEPTH,
        node_purity=1.0,
        min_instances=2,
    )
    build_start = time.perf_counter()
    model.fit(frame)
    build_time_ms = (time.perf_counter() - build_start) * 1000.0

    feature_frame = frame.drop(columns=["target"])
    predictions = model.predict(feature_frame)
    actual_labels = frame["target"].tolist()
    correct = sum(
        prediction == actual
        for prediction, actual in zip(predictions, actual_labels)
    )
    node_count, tree_depth = compute_python_c45_stats(model.get_root_node())

    return {
        "algorithm": "Python C4.5 (c4dot5)",
        "time_ms": build_time_ms,
        "accuracy_pct": 100.0 * correct / len(actual_labels),
        "node_count": node_count,
        "tree_depth": tree_depth,
        "sort_key": 0,
    }


def _worker_cpp(mode: str, binary_path: str, label: str) -> dict:
    completed = subprocess.run(
        [binary_path, mode],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    depth_match = re.search(r"tree depth = (\d+)", output)
    nodes_match = re.search(r"node count = (\d+)", output)
    accuracy_match = re.search(r"accuracy = ([0-9.]+)%", output)
    build_match = re.search(r"build time = ([0-9.]+) ms", output)

    if not all([depth_match, nodes_match, accuracy_match, build_match]):
        raise RuntimeError(f"Could not parse C++ summary for {label}:\n{output[-2000:]}")

    return {
        "algorithm": label,
        "time_ms": float(build_match.group(1)),
        "accuracy_pct": float(accuracy_match.group(1)),
        "node_count": int(nodes_match.group(1)),
        "tree_depth": int(depth_match.group(1)),
        "sort_key": 2 if mode == "cart" else 3,
    }


def compile_benchmark_binary() -> Path:
    print(f"Compiling C++ benchmark binary ({CPP_THREAD_COUNT} threads)...")
    subprocess.run(COMPILE_COMMAND, cwd=ROOT, check=True)
    return BENCHMARK_BINARY


def run_parallel_benchmarks(dataset_path: Path, binary_path: Path) -> list[BenchmarkRow]:
    dataset_arg = str(dataset_path)
    binary_arg = str(binary_path)
    jobs = {
        "python_c45": ( _worker_python_c45, (dataset_arg,) ),
        "sklearn": (_worker_sklearn_cart, (dataset_arg,)),
        "cpp_cart": (_worker_cpp, ("cart", binary_arg, "C++ CART")),
        "cpp_c45": (_worker_cpp, ("c45", binary_arg, "C++ C4.5")),
    }

    rows: list[BenchmarkRow] = []
    print("Running 4 benchmarks in parallel...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_name = {
            executor.submit(worker, *args): name
            for name, (worker, args) in jobs.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            print(f"  finished: {name}")
            result = future.result()
            rows.append(BenchmarkRow(**result))

    rows.sort(key=lambda row: row.sort_key)
    return rows


def verify_depth(rows: list[BenchmarkRow]) -> None:
    for row in rows:
        if row.tree_depth != REQUIRED_DEPTH:
            raise RuntimeError(
                f"{row.algorithm}: expected tree depth {REQUIRED_DEPTH}, got {row.tree_depth}"
            )


def write_csv(rows: list[BenchmarkRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["algorithm", "time_ms", "accuracy_pct", "node_count", "tree_depth"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "algorithm": row.algorithm,
                    "time_ms": f"{row.time_ms:.4f}",
                    "accuracy_pct": f"{row.accuracy_pct:.4f}",
                    "node_count": row.node_count,
                    "tree_depth": row.tree_depth,
                }
            )


def render_comparison_svg(rows: list[BenchmarkRow], path: Path, sample_count: int) -> None:
    width = 1180
    height = 720
    margin_left = 220
    margin_top = 120
    chart_width = width - margin_left - 80
    chart_height = 420
    group_width = chart_width / len(rows)
    bar_width = group_width * 0.22

    max_time = max(row.time_ms for row in rows) * 1.08
    max_accuracy = 100.0
    max_nodes = max(row.node_count for row in rows) * 1.12

    def bar_height(value: float, maximum: float) -> float:
        return chart_height * (value / maximum)

    colors = {
        "time": "#2563eb",
        "accuracy": "#16a34a",
        "nodes": "#dc2626",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: 'Segoe UI', Arial, sans-serif; fill: #111827; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #4b5563; }",
        ".axis { stroke: #9ca3af; stroke-width: 1; }",
        ".legend-box { fill: #f9fafb; stroke: #d1d5db; }",
        ".note-box { fill: rgba(255,255,255,0.92); stroke: #9ca3af; }",
        ".metric-label { font-size: 11px; fill: #374151; }",
        ".value-label { font-size: 10px; font-weight: 600; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="40" y="42">Porównanie algorytmów drzew decyzyjnych</text>',
        '<text class="subtitle" x="40" y="68">'
        f"Zbiór: covertype_10x_smaller.csv | max_depth={MAX_DEPTH} | pruning=wyłączone"
        "</text>",
        f'<rect class="note-box" x="{width - 360}" y="18" width="340" height="88" rx="8"/>',
        f'<text class="metric-label" x="{width - 345}" y="42">covertype_10x_smaller.csv</text>',
        f'<text class="metric-label" x="{width - 345}" y="60">max_depth = {MAX_DEPTH}, pruning = None</text>',
        f'<text class="metric-label" x="{width - 345}" y="78">metryki: czas [ms], accuracy [%], node count</text>',
        f'<text class="metric-label" x="{width - 345}" y="96">próbki: {sample_count}</text>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top + chart_height}" '
        f'x2="{margin_left + chart_width}" y2="{margin_top + chart_height}"/>',
    ]

    metric_specs = [
        ("time", "czas [ms]", max_time, lambda row: row.time_ms),
        ("accuracy", "accuracy [%]", max_accuracy, lambda row: row.accuracy_pct),
        ("nodes", "node count", max_nodes, lambda row: float(row.node_count)),
    ]

    for metric_index, (metric_key, metric_label, maximum, getter) in enumerate(metric_specs):
        for row_index, row in enumerate(rows):
            value = getter(row)
            x_center = margin_left + row_index * group_width + group_width / 2
            x = x_center + (metric_index - 1) * bar_width
            bar_h = bar_height(value, maximum)
            y = margin_top + chart_height - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
                f'fill="{colors[metric_key]}" rx="3"/>'
            )
            parts.append(
                f'<text class="value-label" x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" '
                f'text-anchor="middle">{value:.1f}</text>'
            )

        parts.append(
            f'<text class="metric-label" x="{margin_left + metric_index * 180 + 20}" '
            f'y="{margin_top + chart_height + 52}" fill="{colors[metric_key]}">'
            f"■ {metric_label}</text>"
        )

    for row_index, row in enumerate(rows):
        x_center = margin_left + row_index * group_width + group_width / 2
        label_lines = row.algorithm.replace(" (", "\n(").split("\n")
        for line_index, line in enumerate(label_lines):
            parts.append(
                f'<text class="metric-label" x="{x_center:.1f}" '
                f'y="{margin_top + chart_height + 24 + line_index * 14}" text-anchor="middle">'
                f"{line}</text>"
            )

    legend_y = margin_top + chart_height + 110
    parts.extend(
        [
            f'<rect class="legend-box" x="40" y="{legend_y}" width="560" height="88" rx="8"/>',
            f'<text class="metric-label" x="58" y="{legend_y + 24}">'
            "Python C4.5: c4dot5 | Python CART: sklearn DecisionTreeClassifier (Gini)</text>",
            f'<text class="metric-label" x="58" y="{legend_y + 42}">'
            f"C++ C4.5: Entropy + MeanGainFiltered | C++ CART: Gini + MaxGain | wątki C++: {CPP_THREAD_COUNT}</text>",
            f'<text class="metric-label" x="58" y="{legend_y + 60}">'
            f"Źródło danych: {CSV_OUTPUT.name} | benchmark równoległy (4 procesy)</text>",
            "</svg>",
        ]
    )

    path.write_text("\n".join(parts), encoding="utf-8")


def cleanup_temp_files() -> None:
    if BENCHMARK_BINARY.exists():
        BENCHMARK_BINARY.unlink()
        print(f"Removed temp binary: {BENCHMARK_BINARY.name}")

    temp_log = Path("/tmp/benchmark_comparison.log")
    if temp_log.exists():
        temp_log.unlink()


def main() -> int:
    binary_path: Path | None = None
    try:
        print(f"Dataset: {DATASET_PATH.name}")
        _, _, labels, _, _ = load_covertype_dataset(DATASET_PATH)
        print(f"Loaded samples: {len(labels)}")

        binary_path = compile_benchmark_binary()
        rows = run_parallel_benchmarks(DATASET_PATH, binary_path)

        verify_depth(rows)
        write_csv(rows, CSV_OUTPUT)
        render_comparison_svg(rows, SVG_OUTPUT, len(labels))

        print(f"\nWrote {CSV_OUTPUT}")
        print(f"Wrote {SVG_OUTPUT}\n")
        for row in rows:
            print(
                f"{row.algorithm}: depth={row.tree_depth}, nodes={row.node_count}, "
                f"acc={row.accuracy_pct:.4f}%, time={row.time_ms:.2f} ms"
            )
        return 0
    finally:
        if binary_path is not None and binary_path.exists():
            binary_path.unlink()
        cleanup_temp_files()


if __name__ == "__main__":
    raise SystemExit(main())
