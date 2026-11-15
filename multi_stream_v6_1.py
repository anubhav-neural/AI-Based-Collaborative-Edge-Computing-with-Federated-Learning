"""V6.1: empirically calibrated, homogeneous four-stream admission.

The calibration is valid ONLY for four simultaneous, same-model jobs on the
single shared host used to collect results. It does not establish live-video
network or end-to-end deadline guarantees.

Uses the existing V5 request protocol/dispatch implementation without changing
node servers, model caches, or V5/V6 scheduler code.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from scheduler_v3 import collect_status_parallel
from multi_stream_v5 import PORTS, ROOT, DATA_ROOT, execute_assignment

CALIBRATION_DEFAULT = ROOT / "results" / "v6_2_calibrated_profiles.json"
OUTPUT_DEFAULT = ROOT / "results" / "v6_1_multi_stream_results.json"
MODELS = ("yolo11n.pt", "yolo11s.pt")


def load_calibration(path: Path) -> dict[str, dict]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError("Calibration must be a JSON array")
    profiles = {}
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("Each calibration entry must be an object")
        model = item.get("model")
        if model not in MODELS or model in profiles:
            raise ValueError(f"Unknown or repeated calibration model: {model}")
        if type(item.get("concurrent_streams")) is not int or item["concurrent_streams"] != 4:
            raise ValueError(f"{model}: this version requires a four-stream calibration")
        for key in ("measured_mean_ms", "conservative_mean_ms", "conservative_p95_ms"):
            value = item.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{model}: invalid {key}")
        if item["conservative_mean_ms"] < item["measured_mean_ms"]:
            raise ValueError(f"{model}: conservative mean is below measured mean")
        profiles[model] = item
    if set(profiles) != set(MODELS):
        raise ValueError("Calibration must contain one entry for each supported model")
    return profiles


def choose_model(profiles: dict, mode: str, fps: float, deadline_ms: float,
                 overhead_ms: float, utilization_limit: float) -> tuple[dict | None, list[dict]]:
    interval = 1000.0 / fps
    candidates = []
    for model in MODELS:
        p = profiles[model]
        mean = float(p["conservative_mean_ms"])
        p95 = float(p["conservative_p95_ms"])
        # Mean/interval measures estimated utilization of a single stream.
        # utilization_limit=1.0 is a theoretical capacity check; less than 1
        # offers headroom for variability. No queueing model is asserted.
        throughput_ok = mean <= interval * utilization_limit
        deadline_ok = p95 + overhead_ms <= deadline_ms
        candidates.append({
            "model": model,
            "calibrated_concurrency": 4,
            "conservative_mean_ms": round(mean, 3),
            "conservative_p95_ms": round(p95, 3),
            "predicted_utilization": round(mean / interval, 4),
            "fps_capacity_estimate": round(1000.0 / mean, 3),
            "fps_feasible": throughput_ok,
            "deadline_feasible": deadline_ok,
            "feasible": throughput_ok and deadline_ok,
        })
    eligible = [c for c in candidates if c["feasible"] and (mode == "auto" or c["model"] == mode)]
    # Preference for the larger tier is NOT an accuracy-quality claim.
    eligible.sort(key=lambda c: (0 if c["model"] == "yolo11s.pt" else 1))
    return (eligible[0] if eligible else None), candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="V6.1 calibrated homogeneous 4-stream scheduler")
    parser.add_argument("--video", default="data/cars.mp4")
    parser.add_argument("--model", choices=("auto", *MODELS), default="auto")
    parser.add_argument("--streams", type=int, default=4, help="must be exactly 4 for this calibration")
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--overhead-budget-ms", type=float, default=50.0,
                        help="assumed (not measured) additional p95 allowance")
    parser.add_argument("--utilization-limit", type=float, default=1.0,
                        help="capacity fraction allowed, (0, 1]; use <1 for headroom")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.streams != 4:
        parser.error("V6.1 only supports exactly 4 homogeneous concurrent streams; do not extrapolate calibration")
    if not 1 <= args.max_frames <= 10000:
        parser.error("max-frames must be between 1 and 10000")
    if not all(math.isfinite(v) and v > 0 for v in (args.fps, args.deadline_ms, args.timeout)):
        parser.error("fps, deadline-ms and timeout must be finite positive numbers")
    if not math.isfinite(args.overhead_budget_ms) or args.overhead_budget_ms < 0:
        parser.error("overhead-budget-ms must be finite and nonnegative")
    if not math.isfinite(args.utilization_limit) or not 0 < args.utilization_limit <= 1:
        parser.error("utilization-limit must be >0 and <=1")
    video_path = (ROOT / args.video).resolve()
    if not video_path.is_relative_to(DATA_ROOT) or not video_path.is_file():
        parser.error("video must be an existing file inside project data/")

    profiles = load_calibration(args.calibration)
    choice, candidates = choose_model(profiles, args.model, args.fps, args.deadline_ms,
                                       args.overhead_budget_ms, args.utilization_limit)
    started = time.perf_counter()
    statuses = collect_status_parallel()
    poll_ms = (time.perf_counter() - started) * 1000
    idle = [node for node in sorted(PORTS) if node in statuses
            and int(statuses[node].get("running_tasks", -1)) == 0
            and int(statuses[node].get("waiting_tasks", -1)) == 0]

    reason = None
    if choice is None:
        reason = "No model satisfies conservative calibrated throughput and deadline checks"
    elif len(idle) != 4:
        reason = f"Homogeneous four-stream calibration requires four idle reachable nodes (found {len(idle)})"

    assignments = []
    refusals = []
    if reason:
        refusals = [{"stream_index": i, "reason": reason} for i in range(1, 5)]
    else:
        assignments = [{"stream_index": i, "node_id": node,
                        "stream_id": f"v6-1-stream-{i:02d}",
                        "model": choice["model"],
                        "heuristic_latency_ms": round(choice["conservative_p95_ms"] + args.overhead_budget_ms, 3)}
                       for i, node in enumerate(idle, 1)]

    print("\n=== V6.1 CONCURRENCY-CALIBRATED ADMISSION ===")
    print(f"mode={args.model} | fps={args.fps:g} | interval={1000/args.fps:.2f} ms | "
          f"utilization limit={args.utilization_limit:g}")
    for candidate in candidates:
        print(f"  {candidate['model']}: mean={candidate['conservative_mean_ms']:.3f}ms, "
              f"p95={candidate['conservative_p95_ms']:.3f}ms, "
              f"FPS_OK={candidate['fps_feasible']}, deadline_OK={candidate['deadline_feasible']}")
    print(f"Reachable nodes={sorted(statuses)} | idle={idle} | "
          f"admitted={len(assignments)} | refused={len(refusals)}")
    for item in assignments:
        print(f"ADMIT {item['stream_id']} -> Node {item['node_id']} / {item['model']}")
    if reason:
        print(f"REFUSE all: {reason}")

    results = []
    if args.execute and assignments:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(execute_assignment, item, str(video_path.relative_to(ROOT)),
                                   args.max_frames, args.deadline_ms, args.timeout): item
                       for item in assignments}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                print(f"{r['stream_id']} | {r['model']} | {r['status']} | "
                      f"job={r['request_wall_ms']:.1f}ms | mean/frame={r.get('processing_mean_ms')}")
                if r.get("error"):
                    print("  ERROR:", r["error"])

    interval = 1000 / args.fps
    for r in results:
        if r.get("status") == "success" and r.get("processing_mean_ms") is not None:
            r["observed_processing_fps_sustainable"] = float(r["processing_mean_ms"]) <= interval
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "plan_only",
        "model_mode": args.model, "fps": args.fps, "deadline_ms": args.deadline_ms,
        "overhead_budget_ms": args.overhead_budget_ms,
        "utilization_limit": args.utilization_limit,
        "calibration_file": str(args.calibration),
        "calibrated_concurrent_streams": 4,
        "polling_ms": round(poll_ms, 3),
        "candidates": candidates, "selected_model": choice["model"] if choice and not reason else None,
        "admitted": assignments, "refused": refusals,
        "results": sorted(results, key=lambda x: x["stream_index"]),
        "limitations": [
            "Calibrated from a single trial per model of four identical-model concurrent jobs on one laptop",
            "Do not generalize to other concurrency levels, mixed models, or independent edge devices",
            "Only task metadata crosses TCP; frames are read from shared disk, not transported",
            "Frame throughput and deadline checks use processing-only measurements, not paced live arrivals",
            "Assumed overhead allowance is not a measured network or end-to-end latency estimate",
            "In-memory coordinator selection is not a distributed reservation protocol",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")
    if args.execute:
        successes = sum(r.get("status") == "success" for r in results)
        print(f"Successful dispatched streams: {successes}/{len(assignments)}")
        if successes != len(assignments) or len(assignments) != 4:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
