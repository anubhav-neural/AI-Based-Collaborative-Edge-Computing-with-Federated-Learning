"""Scheduler V5: one coordinator, exclusive whole-stream admissions.

Experimental local prototype. All four nodes share one laptop and a video file.
Only stream ASSIGNMENTS (not video bytes) travel over TCP.
Reservations are in memory and only protect dispatches made by this process.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from scheduler_v3 import collect_status_parallel
from scheduler_v4 import load_profiles, propose_placement

ROOT = Path(__file__).resolve().parent
DATA_ROOT = (ROOT / "data").resolve()
PORTS = {1: 5001, 2: 5002, 3: 5003, 4: 5004}


def plan_admissions(profiles, statuses, count, fps, deadline_ms, budget_ms):
    """Batch-admit jobs with one exclusive reservation per idle node.

    The coordinator plans all admissions BEFORE starting the requests; no two
    simultaneous admissions can claim the same node in this process.
    """
    available = {
        node_id: dict(status)
        for node_id, status in statuses.items()
        if node_id in PORTS
        and int(status.get("running_tasks", -1)) == 0
        and int(status.get("waiting_tasks", -1)) == 0
    }
    assignments = []
    refusals = []
    for index in range(1, count + 1):
        choice, _ = propose_placement(profiles, available, fps, deadline_ms, budget_ms) if available else (None, [])
        if choice is None:
            refusals.append({"stream_index": index, "reason": "no available idle node/model pair"})
            continue
        node_id = choice["node_id"]
        assignments.append({"stream_index": index, "node_id": node_id,
                            "model": choice["model"],
                            "heuristic_latency_ms": choice["heuristic_latency_ms"]})
        del available[node_id]  # Reservation: cannot be assigned again in this batch
    return assignments, refusals


def execute_assignment(item, video_rel, frames, deadline_ms, timeout):
    node_id = item["node_id"]
    stream_id = item["stream_id"]
    task_uid = str(uuid.uuid4())
    task = {
        "operation": "process_video_stream", "task_id": 500 + item["stream_index"],
        "task_uid": task_uid, "node_id": node_id, "stream_id": stream_id,
        "video": video_rel, "model": item["model"], "max_frames": frames,
        "deadline_ms": deadline_ms, "sent_timestamp": time.time(),
    }
    started = time.perf_counter()
    try:
        with socket.create_connection(("127.0.0.1", PORTS[node_id]), timeout=8) as client:
            client.settimeout(timeout)
            client.sendall((json.dumps(task) + "\n").encode("utf-8"))
            with client.makefile("rb") as stream:
                line = stream.readline(2 * 1024 * 1024)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not line or not line.endswith(b"\n"):
            raise ConnectionError("Missing/partial response")
        response = json.loads(line.decode("utf-8"))
        if (response.get("task_uid") != task_uid or response.get("node_id") != node_id
                or response.get("operation") != "process_video_stream"):
            raise ValueError("Unexpected response identity/operation")
        if response.get("status") != "success":
            raise RuntimeError(response.get("message", "Worker failed"))
        metrics = response.get("output", {})
        return {**item, "status": "success", "request_wall_ms": round(elapsed_ms, 3),
                "server_queue_wait_ms": response.get("queue_wait_ms"),
                "worker_execution_ms": response.get("execution_time_ms"),
                "frames_processed": metrics.get("frames_processed"),
                "processing_mean_ms": metrics.get("processing_mean_ms"),
                "processing_p95_ms": metrics.get("processing_p95_ms"),
                "processing_deadline_misses": metrics.get("processing_deadline_misses"),
                "metrics": metrics, "error": None}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return {**item, "status": "error", "request_wall_ms": round((time.perf_counter()-started)*1000, 3),
                "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="V5: one-stream-per-idle-node batch admission")
    parser.add_argument("--video", default="data/cars.mp4")
    parser.add_argument("--streams", type=int, default=4, help="number of simultaneous requests (1-16)")
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--overhead-budget-ms", type=float, default=50.0)
    parser.add_argument("--max-frames", type=int, default=30, help="frames per accepted stream (1-10000)")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--execute", action="store_true", help="actually send all admitted streams in parallel")
    args = parser.parse_args()
    video = (ROOT / args.video).resolve()
    if not video.is_relative_to(DATA_ROOT) or not video.is_file():
        parser.error("video must exist inside the project data/ directory")
    if not 1 <= args.streams <= 16 or not 1 <= args.max_frames <= 10000:
        parser.error("streams must be 1-16 and max-frames must be 1-10000")
    values = (args.fps, args.deadline_ms, args.timeout)
    if not all(math.isfinite(n) and n > 0 for n in values):
        parser.error("fps, deadline-ms, and timeout must be positive finite values")
    if not math.isfinite(args.overhead_budget_ms) or args.overhead_budget_ms < 0:
        parser.error("overhead-budget-ms must be finite and non-negative")

    profiles = load_profiles()
    start = time.perf_counter()
    statuses = collect_status_parallel()
    assignments, refusals = plan_admissions(profiles, statuses, args.streams,
                                            args.fps, args.deadline_ms, args.overhead_budget_ms)
    planning_ms = round((time.perf_counter()-start)*1000, 3)
    for item in assignments:
        item["stream_id"] = f"v5-stream-{item['stream_index']:02d}"
    print("\n=== V5 WHOLE-STREAM BATCH ADMISSION ===")
    print(f"Reachable nodes: {sorted(statuses)} | requested={args.streams} | admitted={len(assignments)} | refused={len(refusals)}")
    for item in assignments:
        print(f"ADMIT {item['stream_id']} -> Node {item['node_id']} / {item['model']}")
    for item in refusals:
        print(f"REFUSE stream {item['stream_index']}: {item['reason']}")
    print(f"Polling + admission planning: {planning_ms:.3f} ms")

    results = []
    if args.execute and assignments:
        # The coordinator has reserved all selected nodes before any requests begin.
        print("Sending admitted whole-stream jobs concurrently...")
        with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
            future_map = {pool.submit(execute_assignment, item, str(video.relative_to(ROOT)),
                                      args.max_frames, args.deadline_ms, args.timeout): item
                          for item in assignments}
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                print(f"{result['stream_id']} | Node {result['node_id']} | {result['status']} | "
                      f"wall={result['request_wall_ms']:.1f} ms | frames={result.get('frames_processed', '-')}")
                if result.get("error"):
                    print("  ERROR:", result["error"])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "plan_only",
        "video": str(video.relative_to(ROOT)), "requested_streams": args.streams,
        "max_frames_per_stream": args.max_frames, "fps": args.fps,
        "deadline_ms": args.deadline_ms, "assumed_overhead_budget_ms": args.overhead_budget_ms,
        "planning_ms": planning_ms, "admitted": assignments, "refused": refusals,
        "results": sorted(results, key=lambda x: x["stream_index"]),
        "limitations": [
            "One coordinator; reservations are memory-only, not shared between controller processes",
            "Only currently idle nodes admitted, maximum one simultaneous stream per node",
            "All nodes share a laptop and the same video path; TCP sends metadata, not video frames",
            "250 ms processing-only threshold is not an end-to-end per-frame deadline",
            "All runs use the same host-derived model profiles; simultaneous CPU contention may alter performance",
        ],
    }
    target = ROOT / "results" / "v5_multi_stream_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {target}")
    if args.execute:
        successes = sum(r["status"] == "success" for r in results)
        print(f"Successful dispatched streams: {successes}/{len(assignments)}")
        if successes != len(assignments):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
