import csv
import random
from pathlib import Path


OUTPUT_FILE = Path(__file__).resolve().parent / "sensor_history.csv"
NUM_RECORDS = 250


def determine_state(temperature, pressure, vibration, current):
    """Return the Day 1 severity state for a complete sensor reading."""
    if temperature > 75 and current > 15 or vibration > 2.2:
        return "CRITICAL"
    if temperature > 80 or current > 16 or pressure > 4.0:
        return "WARNING"
    return "NORMAL"


def generate_sensor_history(output_file=OUTPUT_FILE, num_records=NUM_RECORDS):
    """Generate sensor readings and save them as a CSV file."""
    fieldnames = ["temp", "pressure", "vib", "rpm", "current", "state"]

    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(num_records):
            temperature = round(random.uniform(50.0, 95.0), 2)
            pressure = round(random.uniform(1.0, 4.5), 2)
            vibration = round(random.uniform(0.1, 2.5), 2)
            rpm = random.randint(1200, 3600)
            current = round(random.uniform(8.0, 18.0), 2)
 
            state = determine_state(temperature, pressure, vibration, current)

            writer.writerow(
                {
                    "temp": temperature,
                    "pressure": pressure,
                    "vib": vibration,
                    "rpm": rpm,
                    "current": current,
                    "state": state,
                }
            )


if __name__ == "__main__":
    generate_sensor_history()