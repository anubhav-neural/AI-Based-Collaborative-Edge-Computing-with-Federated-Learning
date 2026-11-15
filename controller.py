"""Scheduler V2 TCP controller for four local edge-node servers.

Requires scheduler_v2.py, tasks/task_generator.py and Nodes/node_server.py.
Current network metrics are approximations for local TCP calls, not measured
link bandwidth or one-way propagation delay.
"""

import csv
import json
import shutil
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
CSV_FILE = ROOT / "execution_results.csv"
NETWORK_FILE = ROOT / "network_results.csv"
MAX_RESPONSE_BYTES = 10 * 1024 * 1024

EXECUTION_HEADERS = [
    "Task_UID", "Task_ID", "Node_ID", "Task_Type", "Execution_Time_s",
    "Communication_Time_s", "Total_Response_Time_s", "Status", "Error_Message",
]
NETWORK_HEADERS = [
    "Task_UID", "Task_ID", "Node_ID", "Data_Size_Bytes", "Transfer_Time_ms",
    "Bandwidth_Mbps", "End_to_End_Response_Time_ms",
    "Estimated_Network_Overhead_ms", "Estimated_One_Way_Delay_ms",
]
TASK_TYPES = {
    "1": "matrix_multiplication",
    "2": "sorting",
    "3": "numerical_computation",
    "4": "data_preprocessing",
}


def prepare_csv(path, headers):
    """Keep legacy CSV formats; archive existing files if headers differ."""
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            existing_headers = next(csv.reader(file), [])
        if existing_headers != headers:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = path.with_name(f"{path.stem}_{suffix}_backup.csv")
            shutil.move(str(path), str(backup))
            print(f"Previous CSV backed up to: {backup}")
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(headers)


def read_json_response(client):
    """Read exactly one newline-delimited JSON response with a size cap."""
    response = bytearray()
    while True:
        part = client.recv(4096)
        if not part:
            raise ConnectionError("Connection closed before complete response")
        response.extend(part)
        if len(response) > MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeded 10 MB")
        if b"\n" in part:
            break
    return json.loads(bytes(response).split(b"\n", 1)[0].decode("utf-8"))


def execute_on_node(operation, node_id):
    """Dispatch one task and save its outcome even if execution fails."""
    task = create_task(operation)
    task_uid = str(uuid.uuid4())
    task["task_uid"] = task_uid
    task["node_id"] = node_id
    task_id = task.get("task_id", task_uid)

    execution_time = 0.0
    response_time = 0.0
    nonexecution_time = 0.0
    send_time_ms = 0.0
    send_call_mbps = 0.0
    payload_bytes = 0
    status = "Failed"
    error_message = ""
    started = None

    print("\n===== AUTOMATIC SCHEDULING DECISION =====")
    print(f"Selected Task : {operation}")
    print(f"Selected Node : Node {node_id}")
    print(f"Destination   : {HOST}:{NODES[node_id]}")
    print("\nGenerated Task:")