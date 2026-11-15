

import csv

from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "execution_results.csv"

NODE_IDS = (1, 2, 3, 4)
MIN_SAMPLES = 3


def select_best_node(task_type):
    """Choose a node using historical successful response times."""

    samples = defaultdict(list)

    if not CSV_PATH.exists():
        return 1

    with CSV_PATH.open("r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["Task_Type"] != task_type:
                continue

            if row["Status"].lower() != "success":
                continue

            try:
                node_id = int(row["Node_ID"])
                latency = float(row["Total_Response_Time_s"])
            except (ValueError, KeyError):
                continue

            if node_id in NODE_IDS and latency > 0:
                samples[node_id].append(latency)

    averages = {}

    for node_id in NODE_IDS:
        values = samples[node_id]

        if len(values) >= MIN_SAMPLES:
            averages[node_id] = sum(values) / len(values)

    if not averages:
        return 1

    best_node = min(averages, key=averages.get)

    print("\n===== AUTOMATIC SCHEDULING =====")

    for node_id, average in sorted(averages.items()):
        print(
            f"Node {node_id}: "
            f"{average * 1000:.2f} ms average response"
        )

    print(f"Selected Node: {best_node}")

    return best_node


if __name__ == "__main__":
    selected = select_best_node("matrix_multiplication")
    print("Scheduler decision:", selected)
