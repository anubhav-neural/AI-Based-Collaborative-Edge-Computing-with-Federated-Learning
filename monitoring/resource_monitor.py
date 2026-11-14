
import psutil
import csv
import os
import time
import threading
from datetime import datetime


class ResourceMonitor:

    def __init__(self,node_id):

        self.node_id=node_id
        self.process=psutil.Process(os.getpid())

        self.file=f"monitoring/{node_id}_metrics.csv"

        self.lock=threading.Lock()

        os.makedirs("monitoring",exist_ok=True)

        if not os.path.exists(self.file):
            with open(self.file,"w",newline="") as f:
                writer=csv.writer(f)

                writer.writerow([
                    "Timestamp",
                    "Node",
                    "CPU_Usage",
                    "RAM_Usage",
                    "Queue_Length",
                    "Running_Tasks",
                    "Latency_ms",
                    "Bandwidth_Mbps",
                    "Transfer_Time_ms"
                ])

        self.process.cpu_percent(interval=None)

    def collect(self,queue_length,running_tasks,
                latency_ms=0,bandwidth_mbps=0,
                transfer_time_ms=0):

        cpu=self.process.cpu_percent(interval=None)
        ram=self.process.memory_percent()

        with self.lock:

            with open(self.file,"a",newline="") as f:

                writer=csv.writer(f)

                writer.writerow([
                    datetime.now(),
                    self.node_id,
                    round(cpu,2),
                    round(ram,2),
                    queue_length,
                    running_tasks,
                    round(latency_ms,2),
                    round(bandwidth_mbps,2),
                    round(transfer_time_ms,2)
                ])

        print("\n--- RESOURCE MONITOR ---")
        print("Node:",self.node_id)
        print("CPU:",round(cpu,2),"%")
        print("RAM:",round(ram,2),"%")
        print("Waiting Tasks:",queue_length)
        print("Running Tasks:",running_tasks)
