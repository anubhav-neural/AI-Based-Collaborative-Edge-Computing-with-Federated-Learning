import socketserver
import json
import time
import numpy as np
import threading
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.resource_monitor import ResourceMonitor


HOST = "127.0.0.1"

# Maximum size of an incoming JSON request
MAX_REQUEST_BYTES = 1024 * 1024


# ============================================================
# 1. TASK EXECUTION
# ============================================================

def execute_task(task):

    operation = task.get("operation")

    if operation == "matrix_multiplication":

        size = int(task.get("size", 10))

        if not 1 <= size <= 100:
            raise ValueError(
                "Matrix size must be between 1 and 100"
            )

        a = np.random.rand(size, size)
        b = np.random.rand(size, size)

        result = np.matmul(a, b)

        output = {
            "result_shape": list(result.shape),
            "sample_result": result[:2, :2].tolist()
        }

    elif operation == "sorting":

        size = int(task.get("size", 1000))

        if not 1 <= size <= 1000000:
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

        if not 1 <= size <= 1000000:
            raise ValueError("Invalid computation size")

        arr = np.arange(1, size + 1)

        # Preserve the original numerical operation.
        result = np.sum(arr ** 2)

        output = {
            "sum_of_squares": int(result)
        }

    elif operation == "data_preprocessing":

        size = int(task.get("size", 1000))

        if not 1 <= size <= 1000000:
            raise ValueError("Invalid data size")

        arr = np.random.rand(size)

        minimum = np.min(arr)
        maximum = np.max(arr)

        if maximum == minimum:
            result = np.zeros(size)
        else:
            result = (
                (arr - minimum) /
                (maximum - minimum)
            )

        output = {
            "processed_count": len(result),
            "first_5": result[:5].tolist()
        }

    elif operation == "process_video_stream":
        # Load YOLO lazily so normal status polling remains lightweight.
        from stream_worker import run_stream_task
        output = run_stream_task(task)

    else:
        raise ValueError(
            f"Unknown operation: {operation}"
        )

    return {
        "status": "success",
        "node_id": task.get("node_id"),
        "task_id": task.get("task_id"),
        "task_uid": task.get("task_uid"),
        "operation": operation,
        "output": output
    }


# ============================================================
# 2. TCP REQUEST HANDLER
# ============================================================

class NodeHandler(socketserver.StreamRequestHandler):

    def setup(self):
        super().setup()
        self.connection_start = time.perf_counter()

    def handle(self):

        server = self.server

        task_started = False
        waiting_accounted = True
        slot_acquired = False

        response = None
        task = {}

        try:

            # Limit input size.
            line = self.rfile.readline(
                MAX_REQUEST_BYTES + 1
            )

            receive_end = time.perf_counter()

            if not line:
                return

            if len(line) > MAX_REQUEST_BYTES:
                raise ValueError(
                    "Request exceeds maximum allowed size"
                )

            if not line.endswith(b"\n"):
                raise ValueError(
                    "Incomplete JSON request"
                )

            task = json.loads(
                line.decode("utf-8")
            )

            if not isinstance(task, dict):
                raise ValueError(
                    "Task must be a JSON object"
                )

            operation = task.get("operation")

            # =================================================
            # 3. LIVE RESOURCE STATUS REQUEST
            # =================================================

            if operation == "get_status":

                # Remove the status request from the
                # waiting-task counter.
                server.cancel_waiting()
                waiting_accounted = False

                response = server.get_live_status()

                # The finally block sends the response.
                return

            # =================================================
            # 4. NORMAL COMPUTATION TASK
            # =================================================

            # One execution slot; queued requests stay in waiting_tasks.
            queue_start = time.perf_counter()
            server.worker_semaphore.acquire()
            queue_wait_ms = (time.perf_counter() - queue_start) * 1000
            slot_acquired = True
            server.start_task()

            task_started = True
            waiting_accounted = False

            data_size = len(line)

            sent_timestamp = task.get(
                "sent_timestamp"
            )

            latency_ms = 0.0

            if sent_timestamp is not None:

                latency_ms = max(
                    0.0,
                    (
                        time.time() -
                        float(sent_timestamp)
                    ) * 1000
                )

            # Time from accepted connection to
            # receipt of the request.
            transfer_time_ms = (
                receive_end -
                self.connection_start
            ) * 1000

            if transfer_time_ms > 0:

                bandwidth = (
                    data_size * 8
                ) / (
                    transfer_time_ms * 1000
                )

            else:
                bandwidth = 0.0

            server.record_metrics(
                latency=latency_ms,
                bandwidth=bandwidth,
                transfer=transfer_time_ms
            )

            print("\nTask received:")
            print(task)

            # Measure the execution operation.
            start_time = time.perf_counter()

            response = execute_task(task)

            execution_time = (
                time.perf_counter() -
                start_time
            )

            response["execution_time"] = (
                execution_time
            )

            response["execution_time_ms"] = (
                execution_time * 1000
            )

            response["latency_ms"] = latency_ms
            response["queue_wait_ms"] = round(queue_wait_ms, 3)

        except Exception as error:

            response = {
                "status": "error",
                "node_id": server.node_id,
                "task_id": task.get("task_id"),
                "task_uid": task.get("task_uid"),
                "message": str(error),
                "execution_time": 0,
                "execution_time_ms": 0,
                "latency_ms": 0
            }

            print("\nTask execution failed:")
            print("Error:", error)

        finally:

            # Requests not converted into computation
            # tasks must be removed from waiting count.
            if waiting_accounted:
                server.cancel_waiting()

            try:

                if response is not None:

                    encoded_response = (
                        json.dumps(response) + "\n"
                    ).encode("utf-8")

                    self.wfile.write(
                        encoded_response
                    )

                    self.wfile.flush()

                    # Avoid excessive logging for status polls.
                    if task.get("operation") != "get_status":
                        print("\nTask response:")
                        print(response)

            except Exception as error:

                print(
                    "Response sending failed:",
                    error
                )

            finally:
                try:
                    if task_started:
                        server.finish_task()
                finally:
                    if slot_acquired:
                        server.worker_semaphore.release()


# ============================================================
# 5. NODE SERVER AND RESOURCE TRACKING
# ============================================================

class NodeServer(socketserver.ThreadingTCPServer):

    allow_reuse_address = True
    daemon_threads = True

    def setup_monitoring(self, node_id):

        self.node_id = node_id

        self.waiting_tasks = 0
        self.running_tasks = 0

        self.task_lock = threading.Lock()

        # Experimental configuration: one computation at a time.
        # get_status requests do not consume this semaphore.
        self.max_concurrent_tasks = 1
        self.worker_semaphore = threading.BoundedSemaphore(
            self.max_concurrent_tasks
        )

        self.monitor = ResourceMonitor(
            f"Node{node_id}"
        )

    def process_request(self, request, client_address):

        with self.task_lock:
            self.waiting_tasks += 1

        self.record_metrics()

        try:

            super().process_request(
                request,
                client_address
            )

        except Exception:

            self.cancel_waiting()
            raise

    def start_task(self):

        with self.task_lock:

            self.waiting_tasks = max(
                0,
                self.waiting_tasks - 1
            )

            self.running_tasks += 1

        self.record_metrics()

    def cancel_waiting(self):

        with self.task_lock:

            self.waiting_tasks = max(
                0,
                self.waiting_tasks - 1
            )

        self.record_metrics()

    def finish_task(self):

        with self.task_lock:

            self.running_tasks = max(
                0,
                self.running_tasks - 1
            )

        self.record_metrics()

    def record_metrics(
        self,
        latency=0,
        bandwidth=0,
        transfer=0
    ):

        with self.task_lock:

            waiting = self.waiting_tasks
            running = self.running_tasks

        self.monitor.collect(
            queue_length=waiting,
            running_tasks=running,
            latency_ms=latency,
            bandwidth_mbps=bandwidth,
            transfer_time_ms=transfer
        )

    # ========================================================
    # 6. LIVE STATUS RESPONSE
    # ========================================================

    def get_live_status(self):

        with self.task_lock:

            waiting = self.waiting_tasks
            running = self.running_tasks

        cpu = self.monitor.process.cpu_percent(
            interval=None
        )

        ram = self.monitor.process.memory_percent()

        response = {
            "status": "success",
            "node_id": self.node_id,
            "cpu_percent": round(cpu, 2),
            "ram_percent": round(ram, 2),
            "waiting_tasks": waiting,
            "running_tasks": running,
            "timestamp": time.time()
        }

        return response


# ============================================================
# 7. START EDGE NODE
# ============================================================

def start_node(node_id, port):

    server = NodeServer(
        (HOST, port),
        NodeHandler
    )

    server.setup_monitoring(node_id)

    # V5.2: prepare the model before the node begins accepting requests.
    # Small and nano remain separate; nano is cached on first use.
    # If initialization fails, fail startup instead of advertising a ready node.
    try:
        from stream_worker import preload_model
        preload_model("yolo11s.pt")
    except Exception:
        server.server_close()
        raise

    print("=" * 40)
    print(f"Node {node_id} started successfully")
    print("IP address:", HOST)
    print("Port:", port)
    print("Live monitoring: ENABLED")
    print("Waiting for tasks...")
    print("=" * 40)

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            f"\nNode {node_id} stopping..."
        )

    finally:

        server.server_close()


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print(
        "Run Nodes/Node1.py through Nodes/Node4.py "
        "to start individual edge nodes."
    )