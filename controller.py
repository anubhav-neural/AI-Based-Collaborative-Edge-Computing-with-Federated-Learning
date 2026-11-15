"""Interactive controller for the four-node Scheduler V2 demonstration.

Run from the repository root: python controller.py
"""

import csv
import json
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path

from scheduler_v2 import select_best_node
from tasks.task_generator import create_task

HOST = "127.0.0.1"
NODES = {1: 5001, 2: 5002, 3: 5003, 4: 5004}
ROOT = Path(__file__).resolve().parent
EXECUTION_FILE = ROOT / "execution_results.csv"
NETWORK_FILE = ROOT / "network_results.csv"

EXECUTION_HEADERS = [
    "Task_UID", "Task_ID", "Node_ID", "Task_Type",
    "Execution_Time_s", "Communication_Time_s", "Total_Response_Time_s",
    "Status", "Error_Message",
]
NETWORK_HEADERS = [
    "Task_UID", "Task_ID", "Node_ID", "Data_Size_Bytes",
    "Transfer_Time_ms", "Bandwidth_Mbps", "End_to_End_Response_Time_ms",
    "Estimated_Network_Overhead_ms", "Estimated_One_Way_Delay_ms",
]
TASK_TYPES = {
    "1": "matrix_multiplication",
    "2": "sorting",
    "3": "numerical_computation",
    "4": "data_preprocessing",
}
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


def ensure_csv(path, headers):
    """Create a CSV only when missing; never silently overwrite experiment data."""
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != headers:
            raise ValueError(
                f"CSV schema mismatch in {path.name}. "
                "Back up and reconcile this file before running the controller."
            )
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(headers)


def append_csv(path, row):
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def receive_json_line(client):
    """Read one complete newline-delimited JSON response, with a size cap."""
    data = bytearray()
    while True:
        chunk = client.recv(4096)
        if not chunk:
            raise ConnectionError("Node closed connection before sending a complete response")
        data.extend(chunk)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("Node response exceeds 10 MB")
        if b"\n" in chunk:
            break
    return json.loads(bytes(data).split(b"\n", 1)[0].decode("utf-8"))


def execute_selected_task(operation):
    print("\nChecking live resources and historical performance...")
    try:
        node_id = select_best_node(operation)
    except Exception as exc:
        print(f"Scheduler failed: {exc}")
        return

    if node_id not in NODES:
        print(f"Scheduler returned an unknown node: {node_id}")
        return

    try:
        task = create_task(operation)
        if not isinstance(task, dict):
            raise TypeError("Task generator must return a dictionary")
    except Exception as exc:
        print(f"Task generation failed: {exc}")
        return

    task_uid = str(uuid.uuid4())
    task["task_uid"] = task_uid
    task["node_id"] = node_id
    task_id = task.get("task_id", task_uid)

    print("\n===== AUTOMATIC SCHEDULING DECISION =====")
    print(f"Selected Task : {operation}")
    print(f"Selected Node : Node {node_id}")
    print(f"Destination   : {HOST}:{NODES[node_id]}")
    print("\nGenerated Task:")
    print(json.dumps(task, indent=4))

    execution_time_s = 0.0
    response_time_s = 0.0
    nonexecution_time_s = 0.0
    send_time_ms = 0.0
    send_call_mbps = 0.0
    payload_bytes = 0
    status = "Failed"
    error_message = ""
    started = None

    try:
        print(f"\nConnecting to Node {node_id}...")
        started = time.perf_counter()
        with socket.create_connection((HOST, NODES[node_id]), timeout=10) as client:
            client.settimeout(30)
            print("TCP connection established.")

            task["sent_timestamp"] = time.time()
            payload = (json.dumps(task) + "\n").encode("utf-8")
            payload_bytes = len(payload)

            print("Sending task...")
            send_started = time.perf_counter()
            client.sendall(payload)
            send_time_ms = (time.perf_counter() - send_started) * 1000
            if send_time_ms > 0:
                send_call_mbps = payload_bytes * 8 / (send_time_ms * 1000)
            print("Task sent successfully.")

            result = receive_json_line(client)
            response_time_s = time.perf_counter() - started

        if not isinstance(result, dict):
            raise TypeError("Node returned a non-object JSON response")
        if result.get("node_id") != node_id:
            raise ValueError("Response node ID does not match selected node")
        if result.get("task_uid") != task_uid:
            raise ValueError("Response task UID does not match submitted task")

        if "execution_time_ms" in result:
            execution_time_s = float(result["execution_time_ms"]) / 1000
        else:
            execution_time_s = float(result.get("execution_time", 0))
        if execution_time_s < 0:
            raise ValueError("Negative execution duration reported")

        nonexecution_time_s = max(0.0, response_time_s - execution_time_s)
        if str(result.get("status", "")).lower() == "success":
            status = "success"
        else:
            status = "Failed"
            error_message = str(result.get("message", result.get("error", "Node task failed")))

        print("\n===== TASK EXECUTION RESULT =====")
        print(json.dumps(result, indent=4))
        print("\nExecution Summary:")
        print(f"Task ID       : {task_id}")
        print(f"Selected Node : {node_id}")
        print(f"Status        : {status}")
        print(f"Execution Time: {execution_time_s * 1000:.4f} ms")
        print(f"Other Overhead: {nonexecution_time_s * 1000:.4f} ms")
        print(f"Total Response: {response_time_s * 1000:.4f} ms")
        print("\n===== NETWORK INFORMATION =====")
        print(f"Data Size: {payload_bytes} bytes")
        print(f"Local sendall() duration: {send_time_ms:.4f} ms")
        print(f"Send-call throughput estimate: {send_call_mbps:.4f} Mbps")
        print("Note: This is not measured network-link capacity.")
        print("Non-execution overhead also includes server and scheduling effects.")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        status = "Failed"
        error_message = str(exc)
        if started is not None and response_time_s == 0:
            response_time_s = time.perf_counter() - started
        nonexecution_time_s = max(0.0, response_time_s - execution_time_s)
        print("\n===== TASK EXECUTION FAILED =====")
        print(f"Node {node_id}: {error_message}")
    finally:
        overhead_ms = nonexecution_time_s * 1000
        append_csv(EXECUTION_FILE, [
            task_uid, task_id, node_id, operation,
            round(execution_time_s, 6), round(nonexecution_time_s, 6),
            round(response_time_s, 6), status, error_message,
        ])
        append_csv(NETWORK_FILE, [
            task_uid, task_id, node_id, payload_bytes,
            round(send_time_ms, 4), round(send_call_mbps, 4),
            round(response_time_s * 1000, 4), round(overhead_ms, 4),
            round(overhead_ms / 2, 4),
        ])
        print("\nResults saved successfully.")
        print(f"Execution CSV: {EXECUTION_FILE.name}")
        print(f"Network CSV: {NETWORK_FILE.name}")


def main():
    ensure_csv(EXECUTION_FILE, EXECUTION_HEADERS)
    ensure_csv(NETWORK_FILE, NETWORK_HEADERS)
    print("=" * 50)
    print("       INTELLIGENT EDGE COMPUTING CONTROLLER")
    print("=" * 50)
    print("\nAvailable Edge Nodes:")
    for node_id, port in NODES.items():
        print(f"Node {node_id} - Port {port}")
    print("\nScheduling Policy:")
    print("Real-time resource-aware node selection (V2)")

    while True:
        print("\n" + "=" * 50)
        print("TASK MENU")
        print("=" * 50)
        print("1. Matrix Multiplication")
        print("2. Sorting")
        print("3. Numerical Computation")
        print("4. Data Preprocessing")
        print("5. Exit")
        try:
            choice = input("\nSelect task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nController stopped.")
            break
        if choice == "5":
            print("\nController stopped successfully.")
            break
        if choice not in TASK_TYPES:
            print("Invalid task choice. Try again.")
            continue
        execute_selected_task(TASK_TYPES[choice])


if __name__ == "__main__":
    main()
