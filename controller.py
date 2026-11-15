

import socket
import json
import time
import csv
import os
import uuid
import shutil
from datetime import datetime

from scheduler import select_best_node
from tasks.task_generator import create_task


# ============================================================
# 1. NETWORK CONFIGURATION
# ============================================================

HOST = "127.0.0.1"

nodes = {
    1: 5001,
    2: 5002,
    3: 5003,
    4: 5004
}

CSV_FILE = "execution_results.csv"
NETWORK_FILE = "network_results.csv"


# ============================================================
# 2. CSV HEADERS
# ============================================================

EXECUTION_HEADERS = [
    "Task_UID",
    "Task_ID",
    "Node_ID",
    "Task_Type",
    "Execution_Time_s",
    "Communication_Time_s",
    "Total_Response_Time_s",
    "Status",
    "Error_Message"
]

NETWORK_HEADERS = [
    "Task_UID",
    "Task_ID",
    "Node_ID",
    "Data_Size_Bytes",
    "Transfer_Time_ms",
    "Bandwidth_Mbps",
    "End_to_End_Response_Time_ms",
    "Estimated_Network_Overhead_ms",
    "Estimated_One_Way_Delay_ms"
]


# ============================================================
# 3. PREPARE CSV FILES
# ============================================================

def prepare_csv(filename, headers):

    if os.path.exists(filename):

        with open(filename, "r", newline="") as file:
            reader = csv.reader(file)
            old_headers = next(reader, [])

        if old_headers != headers:

            backup = filename.replace(
                ".csv",
                "_" + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                ) + "_backup.csv"
            )

            shutil.move(filename, backup)

            print("Old CSV backed up:", backup)

    if not os.path.exists(filename):

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)


prepare_csv(CSV_FILE, EXECUTION_HEADERS)
prepare_csv(NETWORK_FILE, NETWORK_HEADERS)


# ============================================================
# 4. DISPLAY CONTROLLER INFORMATION
# ============================================================

print("=" * 50)
print("       INTELLIGENT EDGE COMPUTING CONTROLLER")
print("=" * 50)

print("\nAvailable Edge Nodes:")

for node_id, port in nodes.items():
    print(f"Node {node_id} - Port {port}")

print("\nScheduling Policy:")
print("Historical latency-based automatic node selection")


# ============================================================
# 5. MAIN CONTROLLER LOOP
# ============================================================

while True:

    print("\n" + "=" * 50)
    print("TASK MENU")
    print("=" * 50)

    print("1. Matrix Multiplication")
    print("2. Sorting")
    print("3. Numerical Computation")
    print("4. Data Preprocessing")
    print("5. Exit")

    choice = input("\nSelect task: ").strip()

    if choice == "5":
        print("\nController stopped successfully.")
        break

    task_types = {
        "1": "matrix_multiplication",
        "2": "sorting",
        "3": "numerical_computation",
        "4": "data_preprocessing"
    }

    if choice not in task_types:
        print("Invalid task choice. Try again.")
        continue

    operation = task_types[choice]

    # ========================================================
    # 6. AUTOMATIC INTELLIGENT NODE SELECTION
    # ========================================================

    print("\nAnalyzing historical node performance...")

    try:
        node_id = select_best_node(operation)

    except Exception as e:
        print("Scheduler failed:", str(e))
        continue

    if node_id not in nodes:
        print("Scheduler returned an invalid node.")
        continue

    print("\n" + "=" * 50)
    print("AUTOMATIC SCHEDULING DECISION")
    print("=" * 50)

    print(f"Selected Task : {operation}")
    print(f"Selected Node : Node {node_id}")
    print(f"Destination   : {HOST}:{nodes[node_id]}")

    # ========================================================
    # 7. CREATE TASK
    # ========================================================

    task = create_task(operation)

    task_uid = str(uuid.uuid4())

    task["task_uid"] = task_uid
    task["node_id"] = node_id

    task_id = task.get("task_id", task_uid)

    print("\nGenerated Task:")
    print(json.dumps(task, indent=4))

    # ========================================================
    # 8. INITIALIZE MEASUREMENT VARIABLES
    # ========================================================

    execution_time = 0.0
    total_response_time = 0.0
    communication_time = 0.0

    transfer_time_ms = 0.0
    bandwidth = 0.0
    data_size = 0

    network_overhead_ms = 0.0
    estimated_delay = 0.0

    status = "Failed"
    error_message = ""

    client = None
    start = None

    # ========================================================
    # 9. TCP TASK COMMUNICATION
    # ========================================================

    try:

        print(f"\nConnecting to Node {node_id}...")

        start = time.perf_counter()

        client = socket.create_connection(
            (HOST, nodes[node_id]),
            timeout=10
        )

        client.settimeout(30)

        print("TCP connection established.")

        # Add timestamp
        task["sent_timestamp"] = time.time()

        # Encode task as newline-delimited JSON
        message = json.dumps(task) + "\n"
        encoded_message = message.encode("utf-8")

        data_size = len(encoded_message)

        # ====================================================
        # 10. SEND TASK
        # ====================================================

        print("\nSending task...")

        send_start = time.perf_counter()

        client.sendall(encoded_message)

        send_end = time.perf_counter()

        # Duration of the local sendall() operation
        transfer_time_ms = (
            send_end - send_start
        ) * 1000

        if transfer_time_ms > 0:
            bandwidth = (
                (data_size * 8) /
                (transfer_time_ms * 1000)
            )

        print("Task sent successfully.")

        # ====================================================
        # 11. RECEIVE RESPONSE
        # ====================================================

        response = b""

        while b"\n" not in response:

            data = client.recv(4096)

            if not data:
                raise ConnectionError(
                    "Connection closed before full response"
                )

            response += data

            if len(response) > 10 * 1024 * 1024:
                raise ValueError(
                    "Response exceeded 10 MB"
                )

        response_line = response.split(b"\n", 1)[0]

        result = json.loads(
            response_line.decode("utf-8")
        )

        end = time.perf_counter()

        total_response_time = end - start

        # ====================================================
        # 12. EXECUTION TIME
        # ====================================================

        if "execution_time_ms" in result:

            execution_time = (
                float(result["execution_time_ms"]) / 1000
            )

        else:
            execution_time = float(
                result.get("execution_time", 0)
            )

        # This includes non-inference overhead, not only
        # network transmission time.
        communication_time = max(
            0.0,
            total_response_time - execution_time
        )

        network_overhead_ms = (
            communication_time * 1000
        )

        # Rough estimate only; not a measured one-way delay.
        estimated_delay = network_overhead_ms / 2

        # ====================================================
        # 13. CHECK TASK STATUS
        # ====================================================

        result_status = str(
            result.get("status", "Failed")
        ).lower()

        if result_status == "success":
            status = "success"

        else:
            status = "Failed"

            error_message = result.get(
                "error",
                result.get(
                    "message",
                    "Node reported failure"
                )
            )

        # ====================================================
        # 14. DISPLAY EXECUTION RESULTS
        # ====================================================

        print("\n" + "=" * 50)
        print("TASK EXECUTION RESULT")
        print("=" * 50)

        print(json.dumps(result, indent=4))

        print("\nExecution Summary:")

        print(f"Task ID       : {task_id}")
        print(f"Selected Node : {node_id}")
        print(f"Status        : {status}")

        print(
            f"Execution Time: "
            f"{execution_time * 1000:.4f} ms"
        )

        print(
            f"Other Overhead: "
            f"{communication_time * 1000:.4f} ms"
        )

        print(
            f"Total Response: "
            f"{total_response_time * 1000:.4f} ms"
        )

        # ====================================================
        # 15. DISPLAY NETWORK METRICS
        # ====================================================

        print("\n" + "=" * 50)
        print("NETWORK INFORMATION")
        print("=" * 50)

        print(f"Data Size: {data_size} bytes")

        print(
            f"Local sendall() duration: "
            f"{transfer_time_ms:.4f} ms"
        )

        print(
            f"Send-call throughput estimate: "
            f"{bandwidth:.4f} Mbps"
        )

        print(
            f"Estimated Non-execution Overhead: "
            f"{network_overhead_ms:.4f} ms"
        )

        print(
            f"Rough Half-overhead Estimate: "
            f"{estimated_delay:.4f} ms"
        )

        print(
            "\nNote: Send-call throughput is not "
            "measured network link capacity."
        )

    # ========================================================
    # 16. ERROR HANDLING
    # ========================================================

    except Exception as e:

        error_message = str(e)
        status = "Failed"

        if start is not None:
            total_response_time = (
                time.perf_counter() - start
            )

        communication_time = total_response_time

        network_overhead_ms = (
            communication_time * 1000
        )

        estimated_delay = network_overhead_ms / 2

        print("\n" + "=" * 50)
        print("TASK EXECUTION FAILED")
        print("=" * 50)

        print("Task ID:", task_id)
        print("Node ID:", node_id)
        print("Error:", error_message)

    # ========================================================
    # 17. CLOSE CONNECTION AND SAVE RESULTS
    # ========================================================

    finally:

        if client is not None:
            client.close()

        # Save execution measurements
        with open(CSV_FILE, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                task_uid,
                task_id,
                node_id,
                operation,
                round(execution_time, 6),
                round(communication_time, 6),
                round(total_response_time, 6),
                status,
                error_message
            ])

        # Save networking measurements
        with open(NETWORK_FILE, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                task_uid,
                task_id,
                node_id,
                data_size,
                round(transfer_time_ms, 4),
                round(bandwidth, 4),
                round(total_response_time * 1000, 4),
                round(network_overhead_ms, 4),
                round(estimated_delay, 4)
            ])

        print("\nResults saved successfully.")
        print("Execution CSV:", CSV_FILE)
        print("Network CSV:", NETWORK_FILE)
