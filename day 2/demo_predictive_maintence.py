"""Train a random-forest model for predictive maintenance."""

import csv
import random
from collections import Counter
from pathlib import Path


DATA_FILE = Path(__file__).resolve().parent / "data" / "day2_sample.csv"
FEATURE_NAMES = ("temp", "pressure", "vib", "current", "rpm")


def load_data(file_path=DATA_FILE):
    """Load sensor features and labeled states from day2_sample.csv."""
    with open(file_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("The training CSV contains no records.")

    aliases = {
        "temp": ("temp", "temperature"),
        "pressure": ("pressure",),
        "vib": ("vib", "vibration"),
        "current": ("current",),
        "rpm": ("rpm",),
    }
    columns = set(rows[0])
    feature_columns = []
    for feature in FEATURE_NAMES:
        column = next((name for name in aliases[feature] if name in columns), None)
        if column is None:
            raise ValueError(f"Missing feature column for {feature!r}.")
        feature_columns.append(column)
    if "state" not in columns:
        raise ValueError("The training CSV must contain a 'state' column.")

    features = [[float(row[column]) for column in feature_columns] for row in rows]
    return features, [row["state"] for row in rows]


def _gini(labels):
    counts = Counter(labels)
    total = len(labels)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def _build_tree(features, labels, rng, depth, max_depth, min_samples, max_features):
    majority = Counter(labels).most_common(1)[0][0]
    if (
        depth >= max_depth
        or len(labels) < min_samples
        or len(set(labels)) == 1
    ):
        return {"label": majority}

    feature_count = len(features[0])
    feature_indexes = rng.sample(
        range(feature_count),
        min(max_features, feature_count),
    )
    best = None
    parent_impurity = _gini(labels)

    for feature_index in feature_indexes:
        values = sorted({row[feature_index] for row in features})
        thresholds = [
            (left + right) / 2 for left, right in zip(values, values[1:])
        ]
        for threshold in thresholds:
            left_indexes = [
                index for index, row in enumerate(features)
                if row[feature_index] <= threshold
            ]
            right_indexes = [
                index for index, row in enumerate(features)
                if row[feature_index] > threshold
            ]
            if not left_indexes or not right_indexes:
                continue
            left_labels = [labels[index] for index in left_indexes]
            right_labels = [labels[index] for index in right_indexes]
            impurity = (
                len(left_labels) * _gini(left_labels)
                + len(right_labels) * _gini(right_labels)
            ) / len(labels)
            if best is None or impurity < best[0]:
                best = (impurity, feature_index, threshold, left_indexes, right_indexes)

    if best is None or best[0] >= parent_impurity:
        return {"label": majority}

    _, feature_index, threshold, left_indexes, right_indexes = best
    return {
        "feature": feature_index,
        "threshold": threshold,
        "left": _build_tree(
            [features[index] for index in left_indexes],
            [labels[index] for index in left_indexes],
            rng, depth + 1, max_depth, min_samples, max_features,
        ),
        "right": _build_tree(
            [features[index] for index in right_indexes],
            [labels[index] for index in right_indexes],
            rng, depth + 1, max_depth, min_samples, max_features,
        ),
    }


def train(features, labels, trees=25, max_depth=8, seed=42):
    """Train a random forest using bootstrap samples and random feature splits."""
    if len(features) != len(labels) or not features:
        raise ValueError("Features and labels must be non-empty and equal in length.")
    if trees < 1:
        raise ValueError("trees must be at least 1.")

    rng = random.Random(seed)
    max_features = max(1, int(len(features[0]) ** 0.5))
    forest = []
    for _ in range(trees):
        indexes = [rng.randrange(len(features)) for _ in features]
        sample_features = [features[index] for index in indexes]
        sample_labels = [labels[index] for index in indexes]
        forest.append(
            _build_tree(
                sample_features, sample_labels, rng, 0, max_depth, 2, max_features
            )
        )
    return {"trees": forest}


def _predict_tree(tree, sensor_values):
    while "label" not in tree:
        branch = "left" if sensor_values[tree["feature"]] <= tree["threshold"] else "right"
        tree = tree[branch]
    return tree["label"]


def predict(model, sensor_values):
    """Predict the maintenance state for one sensor reading."""
    if len(sensor_values) != len(FEATURE_NAMES):
        raise ValueError("The prediction must contain all sensor features.")
    votes = [_predict_tree(tree, sensor_values) for tree in model["trees"]]
    return Counter(votes).most_common(1)[0][0]


def accuracy(model, features, labels):
    """Return the proportion of correct predictions."""
    predictions = [predict(model, row) for row in features]
    return sum(actual == predicted for actual, predicted in zip(labels, predictions)) / len(labels)


def main():
    features, labels = load_data()
    combined = list(zip(features, labels))
    random.Random(42).shuffle(combined)
    split_at = max(1, int(len(combined) * 0.8))
    training, testing = combined[:split_at], combined[split_at:]
    model = train([row for row, _ in training], [label for _, label in training])

    test_features = [row for row, _ in testing]
    test_labels = [label for _, label in testing]
    print(f"Training records: {len(training)}")
    print(f"Test records: {len(testing)}")
    print(f"Random forest trees: {len(model['trees'])}")
    print(f"Test accuracy: {accuracy(model, test_features, test_labels):.1%}")
    print(f"Latest reading prediction: {predict(model, features[-1])}")


if __name__ == "__main__":
    main()
