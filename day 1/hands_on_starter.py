"""
DAY 1 · HANDS-ON — Build Your Rule Engine
23IF02 System Automation

Each function matches a step on the "Hands-on · Build Your Rule Engine"
slide. Run this file to see the simulation:

    python hands_on_starter.py

Need the answer? See hands_on_solution.py — but try it yourself first.
"""

import random
import time

LOG_FILE = "day1_log.txt"


def safe_numeric_value(reading, key, default=0.0):
    """Return a float for a sensor reading or a safe fallback when data is missing."""
    value = reading.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# STEP 1 — Generate simulated sensor data
# ---------------------------------------------------------------------------
def generate_sensor_data():
    """Return a dict with at least 'temperature', 'pressure', 'vibration'."""
    reading = {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(1.0, 4.5), 2),
        "vibration": round(random.uniform(0.1, 2.5), 2),
        "current": round(random.uniform(8, 18), 2),
    }

    # Stretch goal: occasional sensor dropout or spike.
    if random.random() < 0.10:
        reading["temperature"] = None
    if random.random() < 0.08:
        reading["pressure"] = None
    if random.random() < 0.09:
        reading["vibration"] = None
    if random.random() < 0.07:
        reading["current"] = None

    return reading


# ---------------------------------------------------------------------------
# STEP 2 — Write your first automation rules
# STEP 7 — Add a rule of your own (add another dict to this list)
# STEP 8 — Add one priority rule (lower "priority" number checked first)
# ---------------------------------------------------------------------------
RULES = [
    {
        "name": "SENSOR_FAULT",
        "priority": 1,
        "condition": lambda r: any(value is None for value in r.values()),
        "action": "ALERT_OPERATOR",
    },
    {
        "name": "THERMAL_ELECTRICAL_STRESS",
        "priority": 2,
        # High temperature and high current together are more dangerous
        # than either reading alone, so stop this machine immediately.
        "condition": lambda r: (
            safe_numeric_value(r, "temperature") > 75
            and safe_numeric_value(r, "current") > 15
        ),
        "action": "STOP_MACHINE",
    },
    {
        "name": "CRITICAL_VIBRATION",
        "priority": 3,
        "condition": lambda r: safe_numeric_value(r, "vibration") > 2.2,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_TEMP",
        "priority": 4,
        "condition": lambda r: safe_numeric_value(r, "temperature") > 80,
        "action": "REDUCE_LOAD",
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 5,
        "condition": lambda r: safe_numeric_value(r, "current") > 16,
        "action": "ALERT_OPERATOR",
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 6,
        "condition": lambda r: safe_numeric_value(r, "pressure") > 4.0,
        "action": "REDUCE_LOAD",
    },
]


# ---------------------------------------------------------------------------
# STEP 3 — Detect machine state from the rules
# ---------------------------------------------------------------------------
def evaluate_rules(reading, rules):
    """Return (rule_name, action) for the first matching rule, checked in
    priority order (lowest number first). If nothing matches, return
    ("NO_RULE_FIRED", "CONTINUE_NORMAL")."""
    for rule in sorted(rules, key=lambda item: item["priority"]):
        try:
            if rule["condition"](reading):
                return rule["name"], rule["action"]
        except (TypeError, ValueError):
            continue
    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# ---------------------------------------------------------------------------
# STEP 4 — Trigger an action when a rule fires
# ---------------------------------------------------------------------------
def take_action(action):
    """Return a human-readable string describing what happens for this action."""
    messages = {
        "STOP_MACHINE": "Emergency stop activated; machine halted.",
        "REDUCE_LOAD": "Machine load reduced to bring readings back to normal.",
        "ALERT_OPERATOR": "Operator alerted to inspect the machine.",
        "CONTINUE_NORMAL": "Machine continues operating normally.",
    }
    return messages.get(action, f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# STEP 5 — Log every decision made
# ---------------------------------------------------------------------------
def log_decision(reading, rule_name, action, machine_id="MACHINE-1"):
    """Append one line to LOG_FILE with the reading, rule fired, and action."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(
            f"{timestamp} | machine={machine_id} | reading={reading} | "
            f"rule={rule_name} | action={action}\n"
        )


# ---------------------------------------------------------------------------
# STEP 6 — Add a second sensor to the system
# The current sensor is included in generate_sensor_data() and HIGH_CURRENT.
# ---------------------------------------------------------------------------

MACHINE_IDS = [f"MACHINE-{number}" for number in range(1, 6)]
ACTION_URGENCY = {
    "STOP_MACHINE": 1,
    "ALERT_OPERATOR": 2,
    "REDUCE_LOAD": 3,
    "CONTINUE_NORMAL": 4,
}


def find_machine_needing_attention(machine_results, problem_counts=None):
    """Return the most urgent machine result from the current factory tick.

    If urgency and rule priority are tied, the machine that has encountered
    its current problem most often is selected.
    """
    urgent_results = [
        result
        for result in machine_results
        if result["action"] != "CONTINUE_NORMAL"
    ]
    if not urgent_results:
        return None
    if problem_counts is None:
        problem_counts = {}
    return min(
        urgent_results,
        key=lambda result: (
            ACTION_URGENCY.get(result["action"], 5),
            result["rule_priority"],
            -problem_counts.get(
                (result["machine_id"], result["rule_name"]),
                0,
            ),
            result["machine_id"],
        ),
    )


def main():
    problem_counts = {}

    for tick in range(1, 11):
        machine_results = []
        for machine_id in MACHINE_IDS:
            reading = generate_sensor_data()
            rule_name, action = evaluate_rules(reading, RULES)
            problem_key = (machine_id, rule_name)
            if action != "CONTINUE_NORMAL":
                problem_counts[problem_key] = (
                    problem_counts.get(problem_key, 0) + 1
                )
            rule_priority = next(
                (
                    rule["priority"]
                    for rule in RULES
                    if rule["name"] == rule_name
                ),
                len(RULES) + 1,
            )
            result = {
                "machine_id": machine_id,
                "reading": reading,
                "rule_name": rule_name,
                "action": action,
                "rule_priority": rule_priority,
                "problem_frequency": problem_counts.get(problem_key, 0),
            }
            machine_results.append(result)
            print(
                f"[Tick {tick:02d}] {machine_id} -> "
                f"{reading} | {rule_name} "
                f"(seen {result['problem_frequency']} time(s)) | "
                f"{take_action(action)}"
            )
            log_decision(reading, rule_name, action, machine_id)

        urgent_machine = find_machine_needing_attention(
            machine_results,
            problem_counts,
        )
        if urgent_machine is None:
            print("           IMMEDIATE ATTENTION -> None; all machines normal\n")
        else:
            print(
                f"           IMMEDIATE ATTENTION -> "
                f"{urgent_machine['machine_id']} "
                f"({urgent_machine['rule_name']}, "
                f"seen {urgent_machine['problem_frequency']} time(s))\n"
            )
        time.sleep(0.2)


if __name__ == "__main__":
    main()
