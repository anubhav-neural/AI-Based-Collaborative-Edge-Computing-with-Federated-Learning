

"""
Scheduler V2: Live resource-aware edge scheduling.

Uses:
- Historical task response times
- Current running and waiting task counts
- Live TCP node-health checks

This is a heuristic baseline, not an ML model.
"""

import csv
import json
import socket
import time

from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "execution_results.csv"

HOST = "127.0.0.1"

NODES = {
    1: 5001,
    2: 5002,
    3: 5003,
    4: 5004,
}

STATUS_TIMEOUT = 2.0
MIN_SAMPLES = 3

# Experimental heuristic parameters.
RUNNING_PENALTY = 1.0
WAITING_PENALTY = 2.0

DEFAULT_RESPONSE_MS = 100.0
DEFAULT_EXECUTION_MS = 50.0


def get_node_status(node_id, port):
    """Request live status from an edge node."""

    request = {
        "operation": "get_status"
    }

    try:
        start = time.perf_counter()

        with socket.create_connection(
            (HOST, port),
            timeout=STATUS_TIMEOUT
        ) as client:

            client.settimeout(STATUS_TIMEOUT)

            message = json.dumps(request) + "\n"

            client.sendall(message.encode("utf-8"))

            with client.makefile("rb") as stream:
                response = stream.readline(65536)

            if not response or not response.endswith(b"\n"):
                raise ConnectionError("Incomplete status response")

            data = json.loads(response.decode("utf-8"))

        probe_ms = (
            time.perf_counter() - start
        ) * 1000

        if data.get("status") != "success":
            raise RuntimeError(
                data.get("message", "Status request failed")
            )

        if data.get("node_id") != node_id:
            raise ValueError("Unexpected node identity")

        required = (
            "cpu_percent",
            "ram_percent",
            "waiting_tasks",
            "running_tasks",
        )

        for key in required:
            if key not in data:
                raise ValueError(f"Missing status field: {key}")

        waiting = int(data["waiting_tasks"])
        running = int(data["running_tasks"])

        if waiting < 0 or running < 0:
            raise ValueError("Invalid task count")

        return {
            "node_id": node_id,
            "cpu_percent": float(data["cpu_percent"]),
            "ram_percent": float(data["ram_percent"]),
            "waiting_tasks": waiting,
            "running_tasks": running,
            "probe_ms": probe_ms,
        }

    except (OSError, ValueError, RuntimeError, TypeError) as error:
        print(f"Node {node_id} unavailable: {error}")
        return None


def load_history(task_type):
    """Read historical measurements for one task type."""

    samples = defaultdict(list)

    if not CSV_PATH.exists():
        return {}

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row.get("Task_Type") != task_type:
                continue

            if row.get("Status", "").lower() != "success":
                continue

            try:
                node_id = int(row["Node_ID"])

                response_ms = (
                    float(row["Total_Response_Time_s"]) * 1000
                )

                execution_ms = (
                    float(row["Execution_Time_s"]) * 1000
                )

            except (KeyError, ValueError, TypeError):
                continue

            if (
                node_id in NODES
                and response_ms > 0
                and execution_ms >= 0
            ):
                samples[node_id].append(
                    (response_ms, execution_ms)
                )

    history = {}

    for node_id, values in samples.items():

        if len(values) < MIN_SAMPLES:
            continue

        response_avg = sum(
            value[0] for value in values
        ) / len(values)

        execution_avg = sum(
            value[1] for value in values
        ) / len(values)

        history[node_id] = {
            "response_ms": response_avg,
            "execution_ms": execution_avg,
            "samples": len(values),
        }

    return history


def calculate_score(status, profile):
    """
    Heuristic expected response-time score.

    Threaded workers can execute requests concurrently,
    so this penalty is not an exact queueing prediction.
    """

    base_response = profile["response_ms"]
    execution = profile["execution_ms"]

    running = status["running_tasks"]
    waiting = status["waiting_tasks"]

    load_penalty = execution * (
        RUNNING_PENALTY * running
        + WAITING_PENALTY * waiting
    )

    return base_response + load_penalty


def select_best_node(task_type):
    """Select the best reachable node using live metrics."""

    history = load_history(task_type)

    candidates = []

    print("\n===== SCHEDULER V2 =====")
    print("Collecting live node status...")

    for node_id, port in NODES.items():

        status = get_node_status(node_id, port)

        if status is None:
            continue

        profile = history.get(
            node_id,
            {
                "response_ms": DEFAULT_RESPONSE_MS,
                "execution_ms": DEFAULT_EXECUTION_MS,
                "samples": 0,
            }
        )

        score = calculate_score(status, profile)

        candidates.append({
            "node_id": node_id,
            "score_ms": score,
            "status": status,
            "samples": profile["samples"],
        })

        print(
            f"Node {node_id} | "
            f"CPU: {status['cpu_percent']:.1f}% | "
            f"Running: {status['running_tasks']} | "
            f"Waiting: {status['waiting_tasks']} | "
            f"Score: {score:.2f} ms | "
            f"Samples: {profile['samples']}"
        )

    if not candidates:
        raise RuntimeError(
            "No edge nodes are currently reachable"
        )

    best = min(
        candidates,
        key=lambda item: (
            item["score_ms"],
            item["node_id"]
        )
    )

    print("\n===== SCHEDULING DECISION =====")
    print(f"Task: {task_type}")
    print(f"Selected Node: {best['node_id']}")
    print(
        f"Estimated score: "
        f"{best['score_ms']:.2f} ms"
    )

    return best["node_id"]


if __name__ == "__main__":

    selected = select_best_node(
        "matrix_multiplication"
    )

    print("\nFinal selection:", selected)
