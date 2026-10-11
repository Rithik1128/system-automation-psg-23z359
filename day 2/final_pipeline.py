import csv
import random
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TRAINING_FILE = BASE_DIR / "data" / "day2_sample.csv"
LOG_FILE = BASE_DIR / "final_pipeline.csv"


def generate_sensor_data():
    state = random.choice([
        "NORMAL",
        "OVERHEATING",
        "HIGH_VIBRATION",
        "MOTOR_OVERLOAD",
        "LOW_PRESSURE",
    ])
    reading = {
        "temperature": round(random.uniform(55, 72), 2),
        "pressure": round(random.uniform(4.2, 6.4), 2),
        "vibration": round(random.uniform(1.0, 3.0), 2),
        "current": round(random.uniform(8, 15), 2),
        "rpm": random.randint(1400, 2200),
    }
    if state == "OVERHEATING":
        reading["temperature"] = round(random.uniform(81, 97), 2)
    elif state == "HIGH_VIBRATION":
        reading["vibration"] = round(random.uniform(5.5, 9.0), 2)
    elif state == "MOTOR_OVERLOAD":
        reading["current"] = round(random.uniform(18, 28), 2)
    elif state == "LOW_PRESSURE":
        reading["pressure"] = round(random.uniform(1.2, 3.5), 2)
    return reading


def apply_rules(reading):
    if reading["pressure"] < 4.0:
        return "LOW_PRESSURE", "REDUCE_LOAD"
    if reading["current"] > 16:
        return "MOTOR_OVERLOAD", "STOP_MACHINE"
    if reading["vibration"] > 5.0:
        return "HIGH_VIBRATION", "STOP_MACHINE"
    if reading["temperature"] > 75:
        return "OVERHEATING", "REDUCE_LOAD"
    return "NORMAL", "CONTINUE_NORMAL"


def load_training_data():
    features = []
    labels = []
    with open(TRAINING_FILE, newline="") as file:
        for row in csv.DictReader(file):
            features.append([
                float(row["temperature"]),
                float(row["pressure"]),
                float(row["vibration"]),
                float(row["current"]),
                float(row["rpm"]),
            ])
            labels.append(row["state"])
    return features, labels


def gini(labels):
    counts = Counter(labels)
    return 1 - sum((count / len(labels)) ** 2 for count in counts.values())


def make_tree(features, labels, rng, depth=0):
    majority = Counter(labels).most_common(1)[0][0]
    if depth == 8 or len(labels) < 2 or len(set(labels)) == 1:
        return majority

    best = None
    columns = rng.sample(range(5), 2)
    for column in columns:
        values = sorted(set(row[column] for row in features))
        for split in values[:-1]:
            left = [i for i, row in enumerate(features) if row[column] <= split]
            right = [i for i, row in enumerate(features) if row[column] > split]
            if not left or not right:
                continue
            score = (
                len(left) * gini([labels[i] for i in left])
                + len(right) * gini([labels[i] for i in right])
            ) / len(labels)
            if best is None or score < best[0]:
                best = (score, column, split, left, right)

    if best is None:
        return majority
    _, column, split, left, right = best
    return {
        "column": column,
        "split": split,
        "left": make_tree([features[i] for i in left], [labels[i] for i in left], rng, depth + 1),
        "right": make_tree([features[i] for i in right], [labels[i] for i in right], rng, depth + 1),
    }


def train(features, labels, tree_count=25):
    rng = random.Random(42)
    forest = []
    for _ in range(tree_count):
        indexes = [rng.randrange(len(features)) for _ in features]
        forest.append(make_tree(
            [features[i] for i in indexes],
            [labels[i] for i in indexes],
            rng,
        ))
    return forest


def predict(forest, reading):
    votes = []
    for tree in forest:
        while isinstance(tree, dict):
            side = "left" if reading[tree["column"]] <= tree["split"] else "right"
            tree = tree[side]
        votes.append(tree)
    return Counter(votes).most_common(1)[0][0]


def main(record_count=100):
    features, labels = load_training_data()
    model = train(features, labels)
    fields = [
        "temperature", "pressure", "vibration", "rpm", "current",
        "rule_state", "action", "ai_state",
    ]

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for _ in range(record_count):
            reading = generate_sensor_data()
            rule_state, action = apply_rules(reading)
            ai_reading = [
                reading["temperature"],
                reading["pressure"],
                reading["vibration"],
                reading["current"],
                reading["rpm"],
            ]
            writer.writerow({
                **reading,
                "rule_state": rule_state,
                "action": action,
                "ai_state": predict(model, ai_reading),
            })

    print(f"Logged {record_count} readings to {LOG_FILE}")


if __name__ == "__main__":
    main()