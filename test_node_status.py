

import socket
import json
import time

HOST = "127.0.0.1"

NODES = {
    1: 5001,
    2: 5002,
    3: 5003,
    4: 5004
}


def get_node_status(node_id, port):
    request = {"operation": "get_status"}

    try:
        start = time.perf_counter()

        with socket.create_connection(
            (HOST, port), timeout=3
        ) as client:

            client.settimeout(3)

            message = json.dumps(request) + "\n"
            client.sendall(message.encode("utf-8"))

            with client.makefile("rb") as stream:
                response = stream.readline(65536)

            if not response:
                raise ConnectionError("Empty response")

            result = json.loads(response.decode("utf-8"))

        response_ms = (
            time.perf_counter() - start
        ) * 1000

        if result.get("status") != "success":
            raise RuntimeError(
                result.get("message", "Status request failed")
            )

        if result.get("node_id") != node_id:
            raise ValueError("Unexpected node ID")

        return result, response_ms

    except (OSError, ValueError, RuntimeError) as error:
        print(f"Node {node_id} unavailable: {error}")
        return None, None


def main():
    print("\n===== LIVE EDGE NODE MONITORING =====")

    for node_id, port in NODES.items():
        status, response_ms = get_node_status(node_id, port)

        if status is None:
            continue

        print(f"\nNode {node_id}")
        print("-" * 35)
        print(f"CPU Usage: {status['cpu_percent']}%")
        print(f"RAM Usage: {status['ram_percent']}%")
        print(f"Waiting Tasks: {status['waiting_tasks']}")
        print(f"Running Tasks: {status['running_tasks']}")
        print(f"Status Response Time: {response_ms:.3f} ms")


if __name__ == "__main__":
    main()
