"""Select a node/model with Scheduler V4 and dispatch ONE whole-stream job.

Metadata-only local prototype: video is read by the receiving node from the
shared project data/ directory. No video bytes move across TCP.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scheduler_v3 import collect_status_parallel
from scheduler_v4 import load_profiles, propose_placement

ROOT = Path(__file__).resolve().parent
PORTS = {1: 5001, 2: 5002, 3: 5003, 4: 5004}


def dispatch(args):
    video = (ROOT / args.video).resolve()
    data_root = (ROOT / "data").resolve()
    if not video.is_relative_to(data_root) or not video.is_file():
        raise FileNotFoundError("Place the video inside project data/ and use --video data/<file>.mp4")
    if not args.stream_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.stream_id) or len(args.stream_id) > 64:
        raise ValueError("Invalid stream-id; use letters, numbers, '-' or '_'")
    if not (math.isfinite(args.fps) and args.fps > 0 and math.isfinite(args.deadline_ms) and args.deadline_ms > 0):
        raise ValueError("fps and deadline must be positive finite numbers")

    started = time.perf_counter()
    profiles = load_profiles()
    statuses = collect_status_parallel()
    selected, candidates = propose_placement(
        profiles, statuses, args.fps, args.deadline_ms, args.overhead_budget_ms
    )
    placement_ms = (time.perf_counter() - started) * 1000
    if selected is None:
        raise RuntimeError("V4 found no feasible placement under the chosen assumptions")
    node_id = selected["node_id"]
    model = selected["model"]
    print(f"Selected stream placement: Node {node_id} / {model}")
    print(f"Placement decision: {placement_ms:.2f} ms")
    print(f"Heuristic per-frame cost: {selected['heuristic_latency_ms']:.2f} ms")
    task = {
        "operation": "process_video_stream",
        "task_id": 401,
        "task_uid": str(uuid.uuid4()),
        "node_id": node_id,
        "stream_id": args.stream_id,
        "video": str(video.relative_to(ROOT)),
        "model": model,
        "max_frames": args.max_frames,
        "deadline_ms": args.deadline_ms,
        "sent_timestamp": time.time(),
    }
    encoded = (json.dumps(task) + "\n").encode("utf-8")
    task_started = time.perf_counter()
    with socket.create_connection(("127.0.0.1", PORTS[node_id]), timeout=10) as client:
        client.settimeout(args.timeout)
        client.sendall(encoded)
        with client.makefile("rb") as file:
            line = file.readline(2 * 1024 * 1024)
    task_wall_ms = (time.perf_counter() - task_started) * 1000
    if not line or not line.endswith(b"\n"):
        raise ConnectionError("No complete JSON response from selected node")
    result = json.loads(line.decode("utf-8"))
    if result.get("task_uid") != task["task_uid"] or result.get("node_id") != node_id:
        raise ValueError("Response identity mismatch")
    if result.get("status") != "success":
        raise RuntimeError(result.get("message", "Worker failed"))
    if result.get("operation") != "process_video_stream":
        raise ValueError("Node server did not run video-stream operation")
    info = result["output"]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_node": node_id,
        "selected_model": model,
        "stream_id": args.stream_id,
        "placement_ms": round(placement_ms, 3),
        "request_wall_ms": round(task_wall_ms, 3),
        "placement_plus_request_ms": round(placement_ms + task_wall_ms, 3),
        "server_queue_wait_ms": result.get("queue_wait_ms"),
        "worker_execution_ms": result.get("execution_time_ms"),
        "stream_metrics": info,
        "candidates": candidates,
        "limitation": "All nodes share one laptop and video path; TCP carries metadata only. Not true video transport or per-frame end-to-end latency.",
    }
    out = ROOT / "results" / f"stream_dispatch_{args.stream_id}.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== ACTUAL WHOLE-STREAM EXECUTION ===")
    print(f"Node {node_id}, model {model}, frames {info['frames_processed']}")
    print(f"Mean {info['processing_mean_ms']} ms, p95 {info['processing_p95_ms']} ms, p99 {info['processing_p99_ms']} ms")
    print(f"Processing FPS {info['throughput_fps']}, per-frame processing misses {info['processing_deadline_misses']}")
    print(f"Server queue wait: {report['server_queue_wait_ms']} ms")
    print(f"Placement: {placement_ms:.2f} ms; TCP request-to-result: {task_wall_ms:.2f} ms")
    print(f"Saved: {out}")
    print("NOTE: request-to-result measures the FULL batch/stream completion, not a per-frame deadline.")
    print("NOTE: VIDEO BYTES WERE NOT SENT OVER TCP; shared filesystem prototype only.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/cars.mp4")
    parser.add_argument("--stream-id", default="cars-stream-01")
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--overhead-budget-ms", type=float, default=50.0)
    parser.add_argument("--max-frames", type=int, default=10000, help="Process up to N frames; by default, process the whole video up to 10k frames")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if not (0 < args.max_frames <= 10000 and args.timeout > 0 and math.isfinite(args.overhead_budget_ms) and args.overhead_budget_ms >= 0):
        parser.error("Invalid frames, timeout or overhead-budget")
    try:
        dispatch(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
