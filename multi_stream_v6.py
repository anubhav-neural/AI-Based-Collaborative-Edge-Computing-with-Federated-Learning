"""V6 experiment: throughput-screened batch admission of whole video files.

Reuses the working V5.2 node protocol and V5 execution function. No existing
V5 files are changed. All nodes are processes on the same physical laptop.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from multi_stream_v5 import DATA_ROOT, PORTS, ROOT, execute_assignment
from scheduler_v3 import collect_status_parallel
from scheduler_v4 import load_profiles
from scheduler_v6 import MODELS, choose_placement


def plan_admissions(profiles, statuses, count, fps, deadline_ms,
                    overhead_budget_ms, model_mode, safety_factor):
    available = {
        node_id: dict(status)
        for node_id, status in statuses.items()
        if node_id in PORTS
        and int(status.get("running_tasks", -1)) == 0
        and int(status.get("waiting_tasks", -1)) == 0
    }
    admitted, refused = [], []
    for index in range(1, count + 1):
        choice, candidates = choose_placement(
            profiles, available, fps, deadline_ms, overhead_budget_ms,
            model_mode, safety_factor,
        )
        if choice is None:
            reason = "no idle nodes" if not available else "no model satisfies estimated FPS and deadline constraints"
            refused.append({"stream_index": index, "reason": reason,
                            "candidate_count": len(candidates)})
            continue
        admitted.append({
            "stream_index": index,
            "stream_id": f"v6-stream-{index:02d}",
            "node_id": choice["node_id"],
            "model": choice["model"],
            "heuristic_latency_ms": choice["heuristic_latency_ms"],
            "predicted_mean_ms": choice["predicted_mean_ms"],
            "frame_interval_ms": choice["frame_interval_ms"],
        })
        del available[choice["node_id"]]  # batch-local exclusivity only
    return admitted, refused


def main():
    p = argparse.ArgumentParser(description="V6: whole-stream FPS + deadline admission")
    p.add_argument("--video", default="data/cars.mp4")
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--fps", type=float, default=12.5)
    p.add_argument("--deadline-ms", type=float, default=250.0)
    p.add_argument("--overhead-budget-ms", type=float, default=50.0)
    p.add_argument("--safety-factor", type=float, default=1.0,
                   help="Multiply offline mean and p95 latency estimates; 1.0 means no margin")
    p.add_argument("--model", choices=("auto", *MODELS), default="auto",
                   help="auto prefers YOLO11s when heuristically feasible; otherwise screen specified model")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "v6_multi_stream_results.json")
    args = p.parse_args()

    video = (ROOT / args.video).resolve()
    if not video.is_relative_to(DATA_ROOT) or not video.is_file():
        p.error("video must be an existing file inside project data/")
    if not 1 <= args.streams <= 16 or not 1 <= args.max_frames <= 10000:
        p.error("streams must be 1-16 and max-frames must be 1-10000")
    if not all(math.isfinite(v) and v > 0 for v in
               (args.fps, args.deadline_ms, args.safety_factor, args.timeout)):
        p.error("fps, deadline-ms, safety-factor and timeout must be finite and >0")
    if not math.isfinite(args.overhead_budget_ms) or args.overhead_budget_ms < 0:
        p.error("overhead-budget-ms must be finite and >= 0")

    profiles = load_profiles()
    start = time.perf_counter()
    statuses = collect_status_parallel()
    admitted, refused = plan_admissions(
        profiles, statuses, args.streams, args.fps, args.deadline_ms,
        args.overhead_budget_ms, args.model, args.safety_factor,
    )
    planning_ms = round((time.perf_counter() - start) * 1000, 3)
    print("\n=== V6 THROUGHPUT-AWARE WHOLE-STREAM ADMISSION ===")
    print(f"model={args.model} | fps={args.fps:g} | frame interval={1000 / args.fps:.2f} ms")
    print(f"nodes={sorted(statuses)} | requested={args.streams} | admitted={len(admitted)} | refused={len(refused)}")
    for item in admitted:
        print(f"ADMIT {item['stream_id']} -> Node {item['node_id']} / {item['model']}"
              f" | predicted mean={item['predicted_mean_ms']:.2f} ms")
    for item in refused:
        print(f"REFUSE {item['stream_index']}: {item['reason']}")
    print(f"Planning: {planning_ms:.3f} ms")
    print("WARNING: feasibility uses OFFLINE same-host model profiles, not concurrent calibrated predictions")

    results = []
    if args.execute and admitted:
        print("Sending complete-video jobs concurrently...")
        with ThreadPoolExecutor(max_workers=len(admitted)) as pool:
            futures = {
                pool.submit(execute_assignment, item, str(video.relative_to(ROOT)),
                            args.max_frames, args.deadline_ms, args.timeout): item
                for item in admitted
            }
            for future in as_completed(futures):
                result = future.result()
                metrics = result.get("metrics") or {}
                interval = 1000.0 / args.fps
                measured_mean = result.get("processing_mean_ms")
                result["observed_processing_fps_sustainable"] = (
                    float(measured_mean) <= interval
                    if result["status"] == "success" and measured_mean is not None else None
                )
                result["model_reused"] = metrics.get("model_reused")
                results.append(result)
                print(f"{result['stream_id']} | Node {result['node_id']} | {result['model']} | "
                      f"{result['status']} | wall={result.get('request_wall_ms', 0):.1f} ms | "
                      f"mean/frame={result.get('processing_mean_ms')} ms | "
                      f"FPS OK={result['observed_processing_fps_sustainable']}")
                if result.get("error"):
                    print("  ERROR:", result["error"])

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v6_prototype",
        "mode": "execute" if args.execute else "plan_only",
        "video": str(video.relative_to(ROOT)), "requested_streams": args.streams,
        "max_frames_per_stream": args.max_frames, "fps": args.fps,
        "frame_interval_ms": round(1000 / args.fps, 3),
        "deadline_ms": args.deadline_ms, "model_mode": args.model,
        "safety_factor": args.safety_factor,
        "assumed_overhead_budget_ms": args.overhead_budget_ms,
        "planning_ms": planning_ms, "admitted": admitted, "refused": refused,
        "results": sorted(results, key=lambda x: x["stream_index"]),
        "limitations": [
            "4 node processes on one laptop share CPU, RAM, and filesystem",
            "Controller sends video paths and metadata, not video bytes",
            "Offline playback, not a paced live input stream",
            "FPS sustainability is an observed mean processing-time screening check, not live queue stability",
            "250 ms threshold applies only to measured frame processing, not end-to-end delay",
            "Offline benchmark profiles do not account for concurrent CPU contention",
            "One coordinator's memory-only reservations do not protect against other coordinators",
            "YOLO11s preference represents an unvalidated model tier, not an established accuracy advantage",
        ],
    }
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {target}")
    if args.execute:
        successes = sum(r["status"] == "success" for r in results)
        print(f"Successful dispatched streams: {successes}/{len(admitted)}")
        if successes != len(admitted):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
