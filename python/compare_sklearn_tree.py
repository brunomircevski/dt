#!/usr/bin/env python3

import csv
from pathlib import Path

try:
    from sklearn.tree import DecisionTreeClassifier
except ModuleNotFoundError:
    print("Missing dependency: scikit-learn")
    print("Install it with: python3 -m pip install scikit-learn")
    raise SystemExit(1)


def load_iris_csv(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        feature_names = header[1:-1]

        rows = []
        features = []
        labels = []

        for raw_row in reader:
            if not raw_row:
                continue

            row_features = [float(value) for value in raw_row[1:-1]]
            row_label = raw_row[-1]

            rows.append((row_features, row_label))
            features.append(row_features)
            labels.append(row_label)

    return feature_names, rows, features, labels


def print_tree(model: DecisionTreeClassifier, feature_names):
    tree = model.tree_
    class_names = list(model.classes_)

    def visit(node_id: int, depth: int, edge_text: str):
        indent = "  " * depth
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        sample_count = int(tree.n_node_samples[node_id])

        if left_child == right_child:
            class_index = max(
                range(len(class_names)),
                key=lambda index: tree.value[node_id][0][index],
            )
            print(f"{indent}{edge_text} [n={sample_count}]: Leaf -> {class_names[class_index]}")
            return

        feature_name = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        print(f"{indent}{edge_text} [n={sample_count}]: if {feature_name} <= {threshold:.3f}")
        visit(left_child, depth + 1, "yes")
        visit(right_child, depth + 1, "no")

    visit(0, 0, "ROOT")


def main():
    csv_path = Path(__file__).resolve().parent.parent / "datasets" / "iris.csv"
    feature_names, rows, features, labels = load_iris_csv(csv_path)

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
        criterion="entropy",
        max_depth=10,
        ccp_alpha=0.04, #post pruning
        random_state=0,
    )
    model.fit(features, labels)

    print("Learned tree:")
    print_tree(model, feature_names)
    print()

    predictions = model.predict(features)
    correct_predictions = sum(
        1 for prediction, actual in zip(predictions, labels) if prediction == actual
    )
    accuracy = correct_predictions / len(labels)

    print("Prediction summary:")
    print(f"  checked samples = {len(labels)}")
    print(f"  correct predictions = {correct_predictions}")
    print(f"  accuracy = {accuracy * 100.0:.4f}%")
    print("  example predictions:")
    for sample_index in (0, 50, 100):
        prediction = predictions[sample_index]
        actual = labels[sample_index]
        print(f"    sample {sample_index} -> predicted: {prediction}, actual: {actual}")


if __name__ == "__main__":
    main()
