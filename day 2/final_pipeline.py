import csv
import random
from collections import Counter
from pathlib import Path


FOLDER = Path(__file__).parent
TRAIN_FILE = FOLDER / "data" / "day2_sample.csv"
OUTPUT_FILE = FOLDER / "final_pipeline.csv"
SENSORS = ["temperature", "pressure", "vibration", "current", "rpm"]
STATES = ["NORMAL", "OVERHEATING", "HIGH_VIBRATION",
          "MOTOR_OVERLOAD", "LOW_PRESSURE"]


def random_reading():
    condition = random.choice(STATES)
    reading = {
        "temperature": round(random.uniform(55, 72), 2),
        "pressure": round(random.uniform(4.2, 6.4), 2),
        "vibration": round(random.uniform(1, 3), 2),
        "current": round(random.uniform(8, 15), 2),
        "rpm": random.randint(1400, 2200),
    }

    ranges = {
        "OVERHEATING": ("temperature", 81, 97),
        "HIGH_VIBRATION": ("vibration", 5.5, 9),
        "MOTOR_OVERLOAD": ("current", 18, 28),
        "LOW_PRESSURE": ("pressure", 1.2, 3.5),
    }
    if condition != "NORMAL":
        field, low, high = ranges[condition]
        reading[field] = round(random.uniform(low, high), 2)
    return reading


def rules_for(reading):
    checks = (
        ("LOW_PRESSURE", "REDUCE_LOAD", reading["pressure"] < 4),
        ("MOTOR_OVERLOAD", "STOP_MACHINE", reading["current"] > 16),
        ("HIGH_VIBRATION", "STOP_MACHINE", reading["vibration"] > 5),
        ("OVERHEATING", "REDUCE_LOAD", reading["temperature"] > 75),
    )
    for state, action, matched in checks:
        if matched:
            return state, action
    return "NORMAL", "CONTINUE_NORMAL"


def read_training_file():
    inputs = []
    outputs = []
    with TRAIN_FILE.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            inputs.append([float(row[name]) for name in SENSORS])
            outputs.append(row["state"])
    if not inputs:
        raise ValueError("The training file is empty.")
    return inputs, outputs


def impurity(labels):
    total = len(labels)
    return 1 - sum((amount / total) ** 2 for amount in Counter(labels).values())


def grow_tree(inputs, outputs, randomizer, level=0):
    fallback = Counter(outputs).most_common(1)[0][0]
    if level >= 8 or len(outputs) < 2 or len(set(outputs)) == 1:
        return fallback

    best_split = None
    for column in randomizer.sample(range(len(SENSORS)), 2):
        possible_splits = sorted({row[column] for row in inputs})
        for cut in possible_splits[:-1]:
            left_ids = [i for i, row in enumerate(inputs) if row[column] <= cut]
            right_ids = [i for i, row in enumerate(inputs) if row[column] > cut]
            if not left_ids or not right_ids:
                continue

            left_labels = [outputs[i] for i in left_ids]
            right_labels = [outputs[i] for i in right_ids]
            score = (
                len(left_labels) * impurity(left_labels)
                + len(right_labels) * impurity(right_labels)
            ) / len(outputs)
            if best_split is None or score < best_split[0]:
                best_split = (score, column, cut, left_ids, right_ids)

    if best_split is None:
        return fallback

    _, column, cut, left_ids, right_ids = best_split
    return {
        "column": column,
        "cut": cut,
        "low": grow_tree(
            [inputs[i] for i in left_ids],
            [outputs[i] for i in left_ids],
            randomizer,
            level + 1,
        ),
        "high": grow_tree(
            [inputs[i] for i in right_ids],
            [outputs[i] for i in right_ids],
            randomizer,
            level + 1,
        ),
    }


def train(inputs, outputs, tree_count=25):
    randomizer = random.Random(42)
    forest = []
    for _ in range(tree_count):
        sample_ids = [
            randomizer.randrange(len(inputs))
            for _ in range(len(inputs))
        ]
        forest.append(grow_tree(
            [inputs[i] for i in sample_ids],
            [outputs[i] for i in sample_ids],
            randomizer,
        ))
    return forest


def tree_vote(tree, values):
    while isinstance(tree, dict):
        branch = "low" if values[tree["column"]] <= tree["cut"] else "high"
        tree = tree[branch]
    return tree


def ai_state(forest, values):
    predictions = [tree_vote(tree, values) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]


def run(records=100):
    training_inputs, training_outputs = read_training_file()
    forest = train(training_inputs, training_outputs)
    columns = SENSORS + ["rule_state", "action", "ai_state"]

    with OUTPUT_FILE.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for _ in range(records):
            reading = random_reading()
            rule_state, action = rules_for(reading)
            values = [reading[name] for name in SENSORS]
            writer.writerow({
                **reading,
                "rule_state": rule_state,
                "action": action,
                "ai_state": ai_state(forest, values),
            })

    print(f"Logged {records} readings to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
