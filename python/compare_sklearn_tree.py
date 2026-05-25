#!/usr/bin/env python3

import contextlib
import io
import sys
import time
from pathlib import Path

try:
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    import numpy as np
except ModuleNotFoundError:
    print("Missing dependency: pandas, numpy, or scikit-learn")
    print("Install it with: pip install pandas numpy scikit-learn")
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


def print_tree(model: DecisionTreeClassifier, feature_names):
    tree = model.tree_
    class_names = list(model.classes_)

    def visit(node_id: int, depth: int, edge_text: str):
        indent = "  " * depth
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        sample_count = int(tree.n_node_samples[node_id])
        if left_child == right_child:
            class_index = np.argmax(tree.value[node_id][0])
            print(f"{indent}{edge_text} [n={sample_count}]: Leaf -> {class_names[class_index]}")
            return

        feature_name = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        print(f"{indent}{edge_text} [n={sample_count}]: if {feature_name} <= {threshold:.3f}")
        visit(left_child, depth + 1, "yes")
        visit(right_child, depth + 1, "no")

    visit(0, 0, "ROOT")


def main():
    # Set the dataset path directly here:
    csv_path = Path(__file__).resolve().parent.parent / "datasets" / "covertype.csv"

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

    # Match training complexity as needed
    model = DecisionTreeClassifier(
        max_depth=5,
        ccp_alpha=0,
        random_state=0
    )
    build_start = time.perf_counter()
    model.fit(features, labels)
    build_time_seconds = time.perf_counter() - build_start

    predictions = model.predict(features)
    correct_predictions = sum(
        1 for prediction, actual in zip(predictions, labels) if prediction == actual
    )
    accuracy = correct_predictions / len(labels)

    print("Learned tree:")
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        print_tree(model, feature_names)
    tree_text = f.getvalue()
    print(tree_text, end="")
    print()

    # Generate tree in SVG format
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        from render_tree_svg import parse_tree, assign_positions, render_svg, MARGIN_X
        root_layout = parse_tree(tree_text.splitlines())
        
        meta_lines_data = [
            f"checked samples = {len(labels)}",
            f"correct predictions = {correct_predictions}",
            f"tree depth = {model.get_depth()}",
            f"node count = {model.tree_.node_count}",
            f"accuracy = {accuracy * 100.0:.4f}%"
        ]
        
        assign_positions(root_layout, depth=0, next_leaf_x=[MARGIN_X + 260])
        svg_content = render_svg(root_layout, meta_lines=meta_lines_data)
        svg_path = Path(__file__).resolve().parent.parent / "python_sckitlearning.svg"
        with open(svg_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(svg_content)
    except Exception as e:
        print(f"Error generating SVG: {e}")

    print("Summary:")
    print(f"  checked samples = {len(labels)}")
    print(f"  correct predictions = {correct_predictions}")
    print(f"  tree depth = {model.get_depth()}")
    print(f"  node count = {model.tree_.node_count}")
    print(f"  accuracy = {accuracy * 100.0:.4f}%")
    print(f"  build time = {build_time_seconds * 1000.0:.4f} ms")
    print("  prune time = 0.0000 ms")


if __name__ == "__main__":
    main()
