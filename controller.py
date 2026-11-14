
import socket
import json
import time
import csv
import os
import uuid
import shutil
from datetime import datetime

from tasks.task_generator import create_task

HOST = "127.0.0.1"

nodes = {
    1: 5001,
    2: 5002,
    3: 5003,
    4: 5004
}

CSV_FILE = "execution_results.csv"
NETWORK_FILE = "network_results.csv"

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


# Create CSV or backup old CSV with different headers
def prepare_csv(filename, headers):

    if os.path.exists(filename):

        with open(filename, "r", newline="") as file:
            reader = csv.reader(file)
            old_headers = next(reader, [])

        if old_headers != headers:

            backup = filename.replace(
                ".csv",
                "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_backup.csv"
            )

            shutil.move(filename, backup)

            print("Old CSV backed up:", backup)

    if not os.path.exists(filename):

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)


prepare_csv(CSV_FILE, EXECUTION_HEADERS)
prepare_csv(NETWORK_FILE, NETWORK_HEADERS)


print("===== EDGE COMPUTING CONTROLLER =====")

print("\nAvailable Nodes:")

for node, port in nodes.items():
    print("Node", node, "- Port", port)


while True:

    print("\n===== TASK MENU =====")
    print("1. Matrix Multiplication")
    print("2. Sorting")
    print("3. Numerical Computation")
    print("4. Data Preprocessing")
    print("5. Exit")

    choice = input("\nSelect task: ")

    if choice == "5":
        print("Controller stopped")
        break

    task_types = {
        "1": "matrix_multiplication",
        "2": "sorting",
        "3": "numerical_computation",
        "4": "data_preprocessing"
    }

    if choice not in task_types:
        print("Invalid task choice")
        continue

    try:
        node_id = int(input("Select destination node (1-4): "))
    except ValueError:
        print("Enter a valid node number")
        continue

    if node_id not in nodes:
        print("Invalid node selection")
        continue

    operation = task_types[choice]

    task = create_task(operation)

    # Unique identifier for every task submission
    task_uid = str(uuid.uuid4())

    task["task_uid"] = task_uid
    task["node_id"] = node_id

    task_id = task.get("task_id", task_uid)

    execution_time = 0
    total_response_time = 0
    communication_time = 0

    transfer_time_ms = 0
    bandwidth = 0
    data_size = 0

    network_overhead_ms = 0
    estimated_delay = 0

    status = "Failed"
    error_message = ""

    client = None

    try:

        # Start total response timer
        start = time.perf_counter()

        # Connect to selected node
        client = socket.create_connection(
            (HOST, nodes[node_id]),
            timeout=10
        )

        client.settimeout(30)

        # Add timestamp before sending
        task["sent_timestamp"] = time.time()

        # Convert task into bytes
        message = json.dumps(task) + "\n"
        encoded_message = message.encode("utf-8")

        # Calculate data size
        data_size = len(encoded_message)

        # Measure sending time
        send_start = time.perf_counter()

        client.sendall(encoded_message)

        send_end = time.perf_counter()

        transfer_time_ms = (send_end - send_start) * 1000

        # Calculate estimated bandwidth
        if transfer_time_ms > 0:

            bandwidth = (
                (data_size * 8) / (transfer_time_ms * 1000)
            )

        # Receive response from node
        response = b""

        while b"\n" not in response:

            data = client.recv(4096)

            if not data:
                raise ConnectionError(
                    "Connection closed before response"
                )

            response += data

            if len(response) > 10 * 1024 * 1024:
                raise ValueError("Response exceeded 10 MB")

        # Extract one JSON response
        response_line = response.split(b"\n", 1)[0]

        result = json.loads(response_line.decode("utf-8"))

        # Stop total response timer
        end = time.perf_counter()

        total_response_time = end - start

        # Read execution time from node
        if "execution_time_ms" in result:

            execution_time = (
                result["execution_time_ms"] / 1000
            )

        else:

            execution_time = result.get("execution_time", 0)

        execution_time = float(execution_time)

        # Calculate communication overhead
        communication_time = max(
            0,
            total_response_time - execution_time
        )

        # Estimate network overhead
        network_overhead_ms = communication_time * 1000

        # Estimate one-way delay
        estimated_delay = network_overhead_ms / 2

        # Read task status
        status = result.get("status", "success")

        if status.lower() != "success":
            status = "Failed"
            error_message = result.get(
                "error",
                result.get("message", "Node reported failure")
            )

        print("\n===== TASK EXECUTION RESULT =====")

        print(json.dumps(result, indent=4))

        print("\nExecution Time:", round(execution_time, 6), "seconds")

        print(
            "Communication Time:",
            round(communication_time, 6),
            "seconds"
        )

        print(
            "Total Response Time:",
            round(total_response_time, 6),
            "seconds"
        )

        print("\n===== NETWORK INFORMATION =====")

        print("Data Size:", data_size, "bytes")

        print(
            "Transfer Time:",
            round(transfer_time_ms, 4),
            "ms"
        )

        print(
            "Estimated Bandwidth:",
            round(bandwidth, 4),
            "Mbps"
        )

        print(
            "Estimated Network Overhead:",
            round(network_overhead_ms, 4),
            "ms"
        )

        print(
            "Estimated One-Way Delay:",
            round(estimated_delay, 4),
            "ms"
        )

    except Exception as e:

        error_message = str(e)

        status = "Failed"

        # Record elapsed time even for failed requests
        if "start" in locals():
            total_response_time = time.perf_counter() - start

        communication_time = total_response_time

        print("\n===== TASK FAILED =====")
        print("Task ID:", task_id)
        print("Node ID:", node_id)
        print("Error:", error_message)

    finally:

        if client is not None:
            client.close()

        # Save execution results
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

        # Save network results
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

        print("\nTask record saved to", CSV_FILE)
        print("Network record saved to", NETWORK_FILE)
