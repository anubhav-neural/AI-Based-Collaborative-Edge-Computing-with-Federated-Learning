
import socketserver
import json
import time
import numpy as np
import threading
import sys
from pathlib import Path

# Add main project directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.resource_monitor import ResourceMonitor

HOST = "127.0.0.1"


def execute_task(task):

    operation = task.get("operation")

    if operation == "matrix_multiplication":

        size = int(task.get("size", 10))

        if size < 1 or size > 100:
            raise ValueError("Matrix size must be between 1 and 100")

        a = np.random.rand(size, size)
        b = np.random.rand(size, size)

        result = np.matmul(a, b)

        output = {
            "result_shape": list(result.shape),
            "sample_result": result[:2, :2].tolist()
        }

    elif operation == "sorting":

        size = int(task.get("size", 1000))

        if size < 1 or size > 1000000:
            raise ValueError("Invalid array size")

        arr = np.random.randint(1, 100000, size)

        result = np.sort(arr)

        output = {
            "sorted_count": len(result),
            "first_5": result[:5].tolist(),
            "last_5": result[-5:].tolist()
        }

    elif operation == "numerical_computation":

        size = int(task.get("size", 1000))

        if size < 1 or size > 1000000:
            raise ValueError("Invalid computation size")

        arr = np.arange(1, size + 1)

        result = np.sum(arr ** 2)

        output = {
            "sum_of_squares": int(result)
        }

    elif operation == "data_preprocessing":

        size = int(task.get("size", 1000))

        if size < 1 or size > 1000000:
            raise ValueError("Invalid data size")

        arr = np.random.rand(size)

        minimum = np.min(arr)
        maximum = np.max(arr)

        if maximum == minimum:
            result = np.zeros(size)
        else:
            result = (arr - minimum) / (maximum - minimum)

        output = {
            "processed_count": len(result),
            "first_5": result[:5].tolist()
        }

    else:
        raise ValueError("Unknown operation")

    return {
        "status": "success",
        "node_id": task.get("node_id"),
        "task_id": task.get("task_id"),
        "task_uid": task.get("task_uid"),
        "operation": operation,
        "output": output
    }


class NodeHandler(socketserver.StreamRequestHandler):

    def setup(self):

        super().setup()

        self.connection_start = time.perf_counter()

    def handle(self):

        server = self.server

        task_started = False
        waiting_accounted = True

        response = None

        task = {}

        try:

            line = self.rfile.readline()

            receive_end = time.perf_counter()

            if not line:
                return

            # Decode incoming task
            task = json.loads(line.decode("utf-8"))

            if not isinstance(task, dict):
                raise ValueError("Task must be a JSON object")

            # Update queue and running task count
            server.start_task()

            task_started = True
            waiting_accounted = False

            data_size = len(line)

            # Calculate arrival elapsed time
            sent_timestamp = task.get("sent_timestamp")

            latency_ms = 0

            if sent_timestamp is not None:

                latency_ms = max(
                    0,
                    (time.time() - float(sent_timestamp)) * 1000
                )

            # Estimate receiving duration
            transfer_time_ms = (
                receive_end - self.connection_start
            ) * 1000

            # Estimate bandwidth
            if transfer_time_ms > 0:

                bandwidth = (
                    data_size * 8
                ) / (transfer_time_ms * 1000)

            else:
                bandwidth = 0

            # Record communication metrics
            server.record_metrics(
                latency_ms,
                bandwidth,
                transfer_time_ms
            )

            print("\nTask received:")
            print(task)

            # Execute task exactly once
            start_time = time.perf_counter()

            response = execute_task(task)

            end_time = time.perf_counter()

            # Calculate execution time
            execution_time = end_time - start_time

            execution_time_ms = execution_time * 1000

            # Add timing information
            response["execution_time"] = execution_time

            response["execution_time_ms"] = execution_time_ms

            response["latency_ms"] = latency_ms

        except Exception as e:

            response = {
                "status": "error",
                "node_id": server.node_id,
                "task_id": task.get("task_id"),
                "task_uid": task.get("task_uid"),
                "message": str(e),
                "execution_time": 0,
                "execution_time_ms": 0,
                "latency_ms": 0
            }

            print("\nTask execution failed:")
            print("Error:", str(e))

        finally:

            # Remove request from queue if it never started
            if waiting_accounted:
                server.cancel_waiting()

            try:

                if response is not None:

                    self.wfile.write(
                        (json.dumps(response) + "\n").encode("utf-8")
                    )

                    self.wfile.flush()

                    print("\nTask response:")
                    print(response)

            except Exception as e:

                print("Response sending failed:", str(e))

            finally:

                if task_started:
                    server.finish_task()


class NodeServer(socketserver.ThreadingTCPServer):

    allow_reuse_address = True
    daemon_threads = True

    def setup_monitoring(self, node_id):

        self.node_id = node_id

        self.waiting_tasks = 0
        self.running_tasks = 0

        self.task_lock = threading.Lock()

        self.monitor = ResourceMonitor(f"Node{node_id}")

    def process_request(self, request, client_address):

        with self.task_lock:

            self.waiting_tasks += 1

            self.record_metrics()

        try:

            super().process_request(request, client_address)

        except Exception:

            self.cancel_waiting()

            raise

    def start_task(self):

        with self.task_lock:

            self.waiting_tasks = max(
                0, self.waiting_tasks - 1
            )

            self.running_tasks += 1

            self.record_metrics()

    def cancel_waiting(self):

        with self.task_lock:

            self.waiting_tasks = max(
                0, self.waiting_tasks - 1
            )

            self.record_metrics()

    def finish_task(self):

        with self.task_lock:

            self.running_tasks = max(
                0, self.running_tasks - 1
            )

            self.record_metrics()

    def record_metrics(self, latency=0, bandwidth=0, transfer=0):

        self.monitor.collect(
            queue_length=self.waiting_tasks,
            running_tasks=self.running_tasks,
            latency_ms=latency,
            bandwidth_mbps=bandwidth,
            transfer_time_ms=transfer
        )


def start_node(node_id, port):

    server = NodeServer((HOST, port), NodeHandler)

    server.setup_monitoring(node_id)

    print("=" * 40)
    print("Node", node_id, "started successfully")
    print("IP address:", HOST)
    print("Port:", port)
    print("Waiting for tasks...")
    print("=" * 40)

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print("\nNode", node_id, "stopping...")

    finally:

        server.server_close()
