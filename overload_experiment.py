"""Controlled overload experiment for Scheduler V2.

Requires Node 1 launched with EDGE_ENABLE_TEST_DELAY=1 and the updated
node_server.py. The artificial service delay is research test instrumentation,
not real CPU saturation or network congestion.
"""

import csv
import json
import socket
import threading
import time
import uuid
from pathlib import Path

from scheduler_v3 import get_node_status, select_best_node

HOST = "127.0.0.1"
PORT = 5001
TASK_TYPE = "matrix_multiplication"
BACKGROUND_TASKS = 4
SIMULATED_DELAY_MS = 2000
TIMEOUT_SECONDS = 20
OUTPUT = Path(__file__).resolve().parent / "overload_experiment_results.csv"


def send_task(index, results, errors):
    task_uid = str(uuid.uuid4())
    request = {
        "task_id": 9000 + index,
        "task_uid": task_uid,
        "node_id": 1,
        "operation": TASK_TYPE,
        "size": 100,
        "test_delay_ms": SIMULATED_DELAY_MS,
        "sent_timestamp": time.time(),
    }
    began = time.perf_counter()
    try:
        with socket.create_connection((HOST, PORT), timeout=5) as sock:
            sock.settimeout(TIMEOUT_SECONDS)
            sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
            with sock.makefile("rb") as stream:
                raw = stream.readline(65536)
            if not raw or not raw.endswith(b"\n"):
                raise ConnectionError("Missing complete task response")
            response = json.loads(raw.decode("utf-8"))
        if response.get("status") != "success" or response.get("task_uid") != task_uid:
            raise RuntimeError(f"Unexpected task result: {response}")
        results[index] = round((time.perf_counter() - began) * 1000, 2)
    except Exception as exc:
        errors[index] = str(exc)


def snapshot(label):
    status = get_node_status(1, PORT)
    if status is None:
        raise RuntimeError("Node 1 is unavailable. Is it running?")
    print(f"{label}: running={status['running_tasks']}, "
          f"waiting={status['waiting_tasks']}, "
          f"cpu={status['cpu_percent']:.1f}%")
    return status


def main():
    print("=== CONTROLLED OVERLOAD EXPERIMENT ===")
    print("This uses artificial service delay, not real CPU or network overload.")
    baseline = snapshot("Before load")
    if baseline["running_tasks"] or baseline["waiting_tasks"]:
        raise RuntimeError("Node 1 is already busy; run during an idle period")

    print("\nBaseline scheduler decision:")
    baseline_selected = select_best_node(TASK_TYPE)

    results, errors = {}, {}
    threads = [threading.Thread(target=send_task, args=(i, results, errors))
               for i in range(BACKGROUND_TASKS)]
    for thread in threads:
        thread.start()

    observed = None
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        status = snapshot("During load")
        if status["running_tasks"] == 1 and status["waiting_tasks"] >= 2:
            observed = status
            break
        time.sleep(0.15)

    overloaded_selected = None
    if observed:
        print("\nScheduler decision during overload:")
        overloaded_selected = select_best_node(TASK_TYPE)
    else:
        print("\nWARNING: Queue buildup not observed. Verify test delay is enabled "
              "and Nodes/node_server.py was restarted.")

    for thread in threads:
        thread.join(timeout=TIMEOUT_SECONDS + 15)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("Background task threads did not finish")

    recovered = snapshot("After load")
    print("\nScheduler decision after recovery:")
    recovery_selected = select_best_node(TASK_TYPE)

    print("\n=== SUMMARY ===")
    print(f"Baseline selected: Node {baseline_selected}")
    print(f"Overload selected: {f'Node {overloaded_selected}' if overloaded_selected else 'not observed'}")
    print(f"Recovery selected: Node {recovery_selected}")
    print(f"Completed background tasks: {len(results)}/{BACKGROUND_TASKS}")
    print(f"Errors: {errors}")
    print(f"Individual response times (ms): {results}")

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["baseline_selected", "peak_running", "peak_waiting",
                         "overload_selected", "recovery_selected", "completed", "errors"])
        writer.writerow([baseline_selected,
                         observed["running_tasks"] if observed else "",
                         observed["waiting_tasks"] if observed else "",
                         overloaded_selected or "", recovery_selected,
                         len(results), len(errors)])
    print(f"Saved: {OUTPUT}")

    if errors:
        raise RuntimeError("Some background tasks failed; inspect the errors above")
    if not observed:
        raise RuntimeError("No observable queue buildup; experiment inconclusive")
    if recovered["running_tasks"] or recovered["waiting_tasks"]:
        raise RuntimeError("Node 1 did not return to idle")
    if overloaded_selected == 1:
        print("WARNING: Scheduler still chose overloaded Node 1. Check historical "+
              "execution-time estimates and penalty weights; this is a valid finding.")
    else:
        print("PASS: Scheduler avoided Node 1 during controlled queue buildup")


if __name__ == "__main__":
    main()
