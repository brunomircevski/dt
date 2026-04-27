#!/usr/bin/env python3

import csv
from pathlib import Path

import pandas as pd
from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier


def load_iris_csv(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        feature_names = header[1:-1]

        rows = []
        for raw_row in reader:
            if not raw_row:
                continue

            row_features = [float(value) for value in raw_row[1:-1]]
            row_label = raw_row[-1]
            rows.append((row_features, row_label))

    return feature_names, rows


def build_training_frame(csv_path: Path):
    frame = pd.read_csv(csv_path)
    frame = frame.drop(columns=["Id"]).rename(columns={"Species": "target"})
    attributes_map = {column: "continuous" for column in frame.columns if column != "target"}
    return frame, attributes_map


def print_tree(root_node):
    def visit(node, depth: int, edge_text: str):
        indent = "  " * depth

        if hasattr(node, "get_class_name"):
            print(f"{indent}{edge_text}: Leaf -> {node.get_class_name()}")
            return

        threshold = node._attributes.threshold
        print(f"{indent}{edge_text}: if {node.get_attribute()} <= {threshold:.3f}")

        yes_child = None
        no_child = None
        for child in node.get_children():
            label = child.get_label()
            if "<=" in label:
                yes_child = child
            elif ">" in label:
                no_child = child

        if yes_child is not None:
            visit(yes_child, depth + 1, "yes")
        if no_child is not None:
            visit(no_child, depth + 1, "no")

    visit(root_node, 0, "ROOT")


def main():
    csv_path = Path(__file__).resolve().parent.parent / "datasets" / "iris.csv"
    feature_names, rows = load_iris_csv(csv_path)
    frame, attributes_map = build_training_frame(csv_path)

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

    model = DecisionTreeClassifier(
        attributes_map,
        max_depth=20,
        node_purity=1.0,
        min_instances=2,
    )
    model.fit(frame)

    print("Learned tree:")
    print_tree(model.get_root_node())
    print()

    feature_frame = frame.drop(columns=["target"])
    predictions = model.predict(feature_frame)
    actual_labels = frame["target"].tolist()
    correct_predictions = sum(
        1 for prediction, actual in zip(predictions, actual_labels) if prediction == actual
    )
    accuracy = correct_predictions / len(actual_labels)

    print("Prediction summary:")
    print(f"  checked samples = {len(actual_labels)}")
    print(f"  correct predictions = {correct_predictions}")
    print(f"  accuracy = {accuracy * 100.0:.4f}%")
    print("  example predictions:")
    for sample_index in (0, 50, 100):
        prediction = predictions[sample_index]
        actual = actual_labels[sample_index]
        print(f"    sample {sample_index} -> predicted: {prediction}, actual: {actual}")


if __name__ == "__main__":
    main()
