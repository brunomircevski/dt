#!/usr/bin/env python3

import contextlib
import io
import sys
import time
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier
except ModuleNotFoundError:
    print("Missing dependency: pandas, numpy, or c4dot5")
    print("Install them in your active virtual environment.")
    raise SystemExit(1)


def load_any_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)

    # 1. Determine target column
    target_candidates = ["Species", "Outcome", "species"]
    target_col = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        target_col = df.columns[-1]

    # 2. Drop Id column if exists
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # 3. Drop NA/missing rows
    df = df.replace(["NA", "na", "?", "N/A"], np.nan)
    df = df.dropna()

    # 4. Filter only numeric features and the target
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

    feature_names = feature_cols
    labels = df[target_col].tolist()
    features = df[feature_names].values.tolist()
    rows = list(zip(features, labels))

    frame = df.rename(columns={target_col: "target"})
    attributes_map = {col: "continuous" for col in feature_names}

    return feature_names, rows, features, labels, frame, attributes_map


def print_tree(root_node, frame):
    def visit(node, current_frame, depth: int, edge_text: str):
        indent = "  " * depth
        sample_count = len(current_frame)

        if hasattr(node, "get_class_name"):
            print(f"{indent}{edge_text} [n={sample_count}]: Leaf -> {node.get_class_name()}")
            return

        feature_name = node.get_attribute()
        threshold = node._attributes.threshold
        print(f"{indent}{edge_text} [n={sample_count}]: if {feature_name} <= {threshold:.3f}")

        yes_child = None
        no_child = None
        for child in node.get_children():
            label = child.get_label()
            if "<=" in label:
                yes_child = child
            elif ">" in label:
                no_child = child

        yes_frame = current_frame[current_frame[feature_name] <= threshold]
        no_frame = current_frame[current_frame[feature_name] > threshold]

        if yes_child is not None:
            visit(yes_child, yes_frame, depth + 1, "yes")
        if no_child is not None:
            visit(no_child, no_frame, depth + 1, "no")

    visit(root_node, frame, 0, "ROOT")


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
        if child_depth > max_depth:
            max_depth = child_depth
    return node_count, 1 + max_depth


def main():
    # Set the dataset path directly here:
    csv_path = Path(__file__).resolve().parent.parent / "datasets" / "diabetes.csv"

    print(f"Loading dataset: {csv_path.name}")
    feature_names, rows, features, labels, frame, attributes_map = load_any_dataset(csv_path)

    print(f"Loaded samples: {len(rows)}")
    print("Features:", *feature_names)
    print()

    first_features, first_label = rows[0]
    first_parts = [
        f"{feature_names[index]}={first_features[index]}"
        for index in range(len(feature_names))
    ]
    print("First sample:", *first_parts, f"label={first_label}")
    print()

    # Configuration for maximum tree growth
    model = DecisionTreeClassifier(
        attributes_map=attributes_map,
        max_depth=100,
        node_purity=0.95
    )
    build_start = time.perf_counter()
    model.fit(frame)
    build_time_seconds = time.perf_counter() - build_start

    feature_frame = frame.drop(columns=["target"])
    predictions = model.predict(feature_frame)
    actual_labels = frame["target"].tolist()
    correct_predictions = sum(
        1 for prediction, actual in zip(predictions, actual_labels) if prediction == actual
    )
    accuracy = correct_predictions / len(actual_labels)

    print("Learned tree:")
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        print_tree(model.get_root_node(), frame)
    tree_text = f.getvalue()
    print(tree_text, end="")
    print()

    node_count, tree_depth = compute_python_c45_stats(model.get_root_node())

    # Generate tree in SVG format
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        from render_tree_svg import parse_tree, assign_positions, render_svg, MARGIN_X
        root_layout = parse_tree(tree_text.splitlines())
        
        meta_lines_data = [
            f"checked samples = {len(actual_labels)}",
            f"correct predictions = {correct_predictions}",
            f"tree depth = {tree_depth}",
            f"node count = {node_count}",
            f"accuracy = {accuracy * 100.0:.4f}%"
        ]
        
        assign_positions(root_layout, depth=0, next_leaf_x=[MARGIN_X + 260])
        svg_content = render_svg(root_layout, meta_lines=meta_lines_data)
        svg_path = Path(__file__).resolve().parent.parent / "python_c45.svg"
        with open(svg_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(svg_content)
    except Exception as e:
        print(f"Error generating SVG: {e}")
    print("Summary:")
    print(f"  checked samples = {len(actual_labels)}")
    print(f"  correct predictions = {correct_predictions}")
    print(f"  tree depth = {tree_depth}")
    print(f"  node count = {node_count}")
    print(f"  accuracy = {accuracy * 100.0:.4f}%")
    print(f"  build time = {build_time_seconds * 1000.0:.4f} ms")
    print("  prune time = 0.0000 ms")


if __name__ == "__main__":
    main()
