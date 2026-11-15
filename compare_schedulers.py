"""Controlled V1 vs V2 scheduler benchmark (synthetic queue delay).

Requirements:
- Four updated node servers running at 127.0.0.1:5001..5004
- Node 1 started with EDGE_ENABLE_TEST_DELAY=1
- scheduler.py, scheduler_v2.py, and execution_results.csv in project root

Does not modify the existing controller or historical CSV.
"""

import argparse
import csv
import json
import random
import socket
import statistics
import threading
import time
import uuid
from pathlib import Path

from scheduler import select_best_node as select_v1
from scheduler_v2 import get_node_status, select_best_node as select_v2

HOST = "127.0.0.1"
PORTS = {1: 5001, 2: 5002, 3: 5003, 4: 5004}
TASK_TYPE = "matrix_multiplication"
OUTPUT = Path(__file__).resolve().parent / "scheduler_comparison_results.csv"


def request_task(node_id, task_id, *, delay_ms=0, timeout=45):
    """Measure one task's TCP round trip, excluding scheduling time."""
    uid = str(uuid.uuid4())
    payload = {
        "operation": TASK_TYPE,
        "size": 100,
        "task_id": task_id,
        "task_uid": uid,
        "node_id": node_id,
        "sent_timestamp": time.time(),
    }
    if delay_ms:
        payload["test_delay_ms"] = delay_ms
    started = time.perf_counter()
    with socket.create_connection((HOST, PORTS[node_id]), timeout=5) as sock:
        sock.settimeout(timeout)
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        with sock.makefile("rb") as reader:
            raw = reader.readline(65536)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not raw or not raw.endswith(b"\n"):
        raise RuntimeError(f"Node {node_id}: missing complete response")
    answer = json.loads(raw.decode("utf-8"))
    if (answer.get("status") != "success" or answer.get("task_uid") != uid
            or answer.get("node_id") != node_id):
        raise RuntimeError(f"Node {node_id}: unexpected response {answer}")
    return elapsed_ms


def check_nodes_idle():
    for node_id, port in PORTS.items():
        status = get_node_status(node_id, port)
        if status is None:
            raise RuntimeError(f"Node {node_id} is unavailable; start all four nodes")
        if status["running_tasks"] or status["waiting_tasks"]:
            raise RuntimeError(f"Node {node_id} is busy; stop other experiments first")


def wait_for_queue(expected_waiting, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_node_status(1, PORTS[1])
        if status and status["running_tasks"] == 1 and status["waiting_tasks"] >= expected_waiting:
            return status
        time.sleep(0.10)
    raise RuntimeError("Queue buildup not observed: check Node 1 test delay and restart")


def background_worker(index, delay_ms, outcomes):
    try:
        outcomes[index] = (True, request_task(1, 10000 + index, delay_ms=delay_ms))
    except Exception as exc:
        outcomes[index] = (False, str(exc))


def run_trial(policy, repetition, num_background, delay_ms):
    check_nodes_idle()
    outcomes = {}
    threads = [threading.Thread(target=background_worker,
                                args=(repetition * 100 + i, delay_ms, outcomes))
               for i in range(num_background)]
    for thread in threads:
        thread.start()

    queue_state = None
    selected = None
    scheduling_ms = None
    task_ms = None
    error = ""
    try:
        queue_state = wait_for_queue(num_background - 1)
        sched_started = time.perf_counter()
        selected = select_v1(TASK_TYPE) if policy == "V1" else select_v2(TASK_TYPE)
        scheduling_ms = (time.perf_counter() - sched_started) * 1000
        if selected not in PORTS:
            raise RuntimeError(f"Scheduler selected invalid node {selected}")
        task_ms = request_task(selected, repetition * 1000 + (1 if policy == "V1" else 2))
        print(f"{policy} trial {repetition}: node={selected}, "
              f"scheduling={scheduling_ms:.1f} ms, TCP task={task_ms:.1f} ms, "
              f"total={scheduling_ms + task_ms:.1f} ms")
    except Exception as exc:
        error = str(exc)
        print(f"{policy} trial {repetition} FAILED: {error}")
    finally:
        for thread in threads:
            thread.join(timeout=delay_ms * num_background / 1000 + 20)
        if any(t.is_alive() for t in threads):
            raise RuntimeError("Background workers still alive; experiment aborted")

    bad_background = {i: value for i, (ok, value) in outcomes.items() if not ok}
    if len(outcomes) != num_background or bad_background:
        error = (error + "; " if error else "") + f"background failures: {bad_background}"
    return {
        "policy": policy,
        "trial": repetition,
        "selected_node": selected if selected is not None else "",
        "queue_running": queue_state["running_tasks"] if queue_state else "",
        "queue_waiting": queue_state["waiting_tasks"] if queue_state else "",
        "scheduler_ms": round(scheduling_ms, 3) if scheduling_ms is not None else "",
        "task_tcp_ms": round(task_ms, 3) if task_ms is not None else "",
        "schedule_plus_task_ms": round(scheduling_ms + task_ms, 3)
            if scheduling_ms is not None and task_ms is not None else "",
        "background_completed": len(outcomes) - len(bad_background),
        "error": error,
    }


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--background", type=int, default=4)
    parser.add_argument("--delay-ms", type=int, default=2000)
    args = parser.parse_args()
    if not 1 <= args.trials <= 100:
        parser.error("--trials must be 1..100")
    if not 2 <= args.background <= 8:
        parser.error("--background must be 2..8")
    if not 1 <= args.delay_ms <= 10000:
        parser.error("--delay-ms must be 1..10000")

    print("=== V1 vs V2: CONTROLLED QUEUE COMPARISON ===")
    print("Artificial delay only: not real CPU/network congestion.")
    print("Each policy gets a fresh queue load. Scheduling time is measured separately.")
    check_nodes_idle()

    # Interleave policies, alternating order per pair to limit temporal bias.
    order = [("V1", "V2") if i % 2 == 0 else ("V2", "V1")
             for i in range(args.trials)]
    random.Random(42).shuffle(order)
    rows = []
    for trial_number, pair in enumerate(order, 1):
        for policy in pair:
            print(f"\n--- Trial {trial_number}, policy {policy} ---")
            rows.append(run_trial(policy, trial_number, args.background, args.delay_ms))

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {OUTPUT}")

    for policy in ("V1", "V2"):
        usable = [r for r in rows if r["policy"] == policy and not r["error"]]
        values = [float(r["schedule_plus_task_ms"]) for r in usable]
        if values:
            print(f"{policy}: n={len(values)}, mean={statistics.mean(values):.2f} ms, "
                  f"median={statistics.median(values):.2f} ms, "
                  f"p95={percentile(values, 95):.2f} ms, "
                  f"max={max(values):.2f} ms")
        else:
            print(f"{policy}: no successful trials")

    errors = [row for row in rows if row["error"]]
    if errors:
        raise SystemExit(f"Benchmark completed with {len(errors)} failed trial(s); inspect CSV")
    print("\nCompleted successfully. Compare policies using the CSV results.")


if __name__ == "__main__":
    main()
