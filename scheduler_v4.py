"""Scheduler V4 prototype: choose a whole-stream node/model pair.

This module ONLY recommends placement. It does not start or transfer a video.
All four processes currently share one laptop; the YOLO measurements are
host-level estimates, not independent per-node inference benchmarks.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from scheduler_v3 import collect_status_parallel

ROOT = Path(__file__).resolve().parent
PROFILES_DIR = ROOT / "results" / "video_baseline"
PROFILE_FILES = {
    "yolo11n.pt": "yolo11n_summary.json",
    "yolo11s.pt": "yolo11s_summary.json",
}


def load_profiles(profile_dir=PROFILES_DIR):
    profiles = {}
    for model, filename in PROFILE_FILES.items():
        path = profile_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Save each baseline summary first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model") != model or data.get("device") != "cpu":
            raise ValueError(f"Unexpected model/device in {path}")
        for key in ("processing_mean_ms", "processing_p95_ms", "source_fps", "imgsz", "frames_processed"):
            if key not in data:
                raise ValueError(f"Missing {key} in {path}")
        if float(data["processing_p95_ms"]) <= 0 or float(data["processing_mean_ms"]) <= 0:
            raise ValueError(f"Invalid latency in {path}")
        profiles[model] = data
    a, b = profiles.values()
    for field in ("video", "imgsz", "frames_processed", "source_fps"):
        if a.get(field) != b.get(field):
            raise ValueError(f"YOLO profiles have different '{field}'; benchmark the same workload")
    return profiles


def propose_placement(profiles, statuses, fps, deadline_ms, overhead_budget_ms):
    """Return a ranked node-model choice, or a reason for no feasible pair.

    Conservative simple heuristic: estimated per-frame p95 processing cost
    times (running+waiting+1), plus an explicit unmeasured overhead allowance.
    This is NOT a calibrated queueing model or end-to-end latency prediction.
    """
    if not statuses:
        raise RuntimeError("No reachable nodes; placement refused")
    interval_ms = 1000.0 / fps
    choices = []
    for node_id, status in sorted(statuses.items()):
        running = int(status["running_tasks"])
        waiting = int(status["waiting_tasks"])
        if running < 0 or waiting < 0:
            continue
        contenders = running + waiting
        for model, profile in profiles.items():
            mean_ms = float(profile["processing_mean_ms"])
            p95_ms = float(profile["processing_p95_ms"])
            projected_ms = (contenders + 1) * p95_ms + overhead_budget_ms
            # Approximate compute sustainability with no additional node load.
            fps_feasible = mean_ms <= interval_ms
            deadline_feasible = projected_ms <= deadline_ms
            choices.append({
                "node_id": node_id, "model": model,
                "running_tasks": running, "waiting_tasks": waiting,
                "profile_mean_ms": mean_ms, "profile_p95_ms": p95_ms,
                "heuristic_latency_ms": round(projected_ms, 3),
                "fps_feasible": fps_feasible,
                "deadline_feasible": deadline_feasible,
                "feasible": fps_feasible and deadline_feasible,
            })
    feasible = [c for c in choices if c["feasible"]]
    # Prefer the larger model only as a configurable model-tier preference;
    # no accuracy advantage has been established on this video.
    preferred = sorted(feasible, key=lambda c: (0 if c["model"] == "yolo11s.pt" else 1,
                                                     c["heuristic_latency_ms"], c["node_id"]))
    return (preferred[0] if preferred else None), choices


def main():
    parser = argparse.ArgumentParser(description="Whole-stream node/model placement (decision-only prototype)")
    parser.add_argument("--stream-id", default="cars-stream-01")
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--overhead-budget-ms", type=float, default=50.0,
                        help="Assumed network+queue+controller allowance, NOT measured")
    parser.add_argument("--profiles-dir", type=Path, default=PROFILES_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "scheduler_v4_placement.json")
    args = parser.parse_args()
    if not args.stream_id.strip() or not all(math.isfinite(v) and v > 0 for v in (args.fps, args.deadline_ms)):
        parser.error("stream-id must be nonempty; fps and deadline must be positive finite values")
    if not math.isfinite(args.overhead_budget_ms) or args.overhead_budget_ms < 0:
        parser.error("overhead budget must be finite and nonnegative")

    profiles = load_profiles(args.profiles_dir)
    start = perf_counter()
    statuses = collect_status_parallel()
    selected, candidates = propose_placement(
        profiles, statuses, args.fps, args.deadline_ms, args.overhead_budget_ms
    )
    decision_ms = (perf_counter() - start) * 1000
    print("\n===== SCHEDULER V4: WHOLE-STREAM PLACEMENT (PROTOTYPE) =====")
    print(f"Stream: {args.stream_id} | fps={args.fps:g} | deadline={args.deadline_ms:g} ms")
    print(f"Assumed unmeasured overhead budget: {args.overhead_budget_ms:g} ms")
    print("Node  Model       Load  Score(ms)   FPS OK   Deadline OK")
    for c in candidates:
        print(f"{c['node_id']:>4}  {c['model']:<11} "
              f"{c['running_tasks']}+{c['waiting_tasks']:<3} {c['heuristic_latency_ms']:>9.2f}  "
              f"{str(c['fps_feasible']):>7} {str(c['deadline_feasible']):>11}")
    if selected:
        print(f"\nSelected: Node {selected['node_id']} / {selected['model']}")
        print(f"Heuristic per-frame cost: {selected['heuristic_latency_ms']:.2f} ms")
    else:
        print("\nNo feasible node/model pair under the supplied assumptions")
    print(f"Polling + decision time: {decision_ms:.2f} ms")
    print("WARNING: model benchmark is offline on one host; network cost, accuracy, and actual stream placement remain unvalidated")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stream_id": args.stream_id, "requested_fps": args.fps,
        "deadline_ms": args.deadline_ms, "assumed_overhead_budget_ms": args.overhead_budget_ms,
        "polling_and_decision_ms": round(decision_ms, 3),
        "selected": selected, "candidates": candidates,
        "note": "Decision-only prototype; simulated queue heuristic, local CPU profiles, no stream dispatch or actual end-to-end validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
