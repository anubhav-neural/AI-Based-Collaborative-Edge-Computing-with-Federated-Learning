
import socket
import json
import time

HOST = "127.0.0.1"
PORT = 5002

task = {
    "task_id": 101,
    "node_id": 2,
    "operation": "matrix_multiplication",
    "size": 100
}

start = time.perf_counter()

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client.connect((HOST, PORT))

message = json.dumps(task) + "\n"

client.sendall(message.encode("utf-8"))

response = b""

while not response.endswith(b"\n"):
    data = client.recv(4096)

    if not data:
        break

    response += data

result = json.loads(response.decode("utf-8"))

end = time.perf_counter()

print("\nResponse received from Node 2:")
print(result)

print("\nTotal response time:", end-start, "seconds")

client.close()
