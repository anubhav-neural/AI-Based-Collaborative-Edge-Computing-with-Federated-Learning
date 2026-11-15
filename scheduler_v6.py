"""V6 whole-stream placement: processing deadline + input-rate feasibility.

Predictions use offline per-model CPU profiles, NOT calibrated four-node
concurrent profiles. The four nodes currently share one laptop/CPU.
"""
from __future__ import annotations

import math
from scheduler_v4 import load_profiles

MODELS = ("yolo11n.pt", "yolo11s.pt")


def choose_placement(profiles: dict, statuses: dict, fps: float,
                     deadline_ms: float, overhead_budget_ms: float,
                     model_mode: str = "auto", safety_factor: float = 1.0):
    """Return (choice or None, candidates), without modifying statuses.

    This is a *screening heuristic*, not an end-to-end or live-stream guarantee.
    To avoid queueing on already busy nodes, only idle nodes are considered.
    """
    if model_mode not in (*MODELS, "auto"):
        raise ValueError(f"Invalid model selection: {model_mode}")
    if not all(math.isfinite(v) and v > 0 for v in (fps, deadline_ms, safety_factor)):
        raise ValueError("fps, deadline and safety_factor must be finite and > 0")
    if not math.isfinite(overhead_budget_ms) or overhead_budget_ms < 0:
        raise ValueError("overhead_budget_ms must be finite and >= 0")

    frame_interval_ms = 1000.0 / fps
    candidates = []
    for node_id, status in sorted(statuses.items()):
        running = int(status.get("running_tasks", -1))
        waiting = int(status.get("waiting_tasks", -1))
        if running != 0 or waiting != 0:
            continue
        for model in MODELS:
            if model_mode != "auto" and model != model_mode:
                continue
            profile = profiles.get(model)
            if profile is None:
                continue
            mean = float(profile["processing_mean_ms"])
            p95 = float(profile["processing_p95_ms"])
            predicted_mean = mean * safety_factor
            predicted_p95 = p95 * safety_factor
            # Overhead is a user-supplied assumption, NOT measured network latency.
            heuristic_deadline_ms = predicted_p95 + overhead_budget_ms
            fps_ok = predicted_mean <= frame_interval_ms
            deadline_ok = heuristic_deadline_ms <= deadline_ms
            candidates.append({
                "node_id": node_id,
                "model": model,
                "profile_mean_ms": round(mean, 3),
                "profile_p95_ms": round(p95, 3),
                "predicted_mean_ms": round(predicted_mean, 3),
                "predicted_p95_ms": round(predicted_p95, 3),
                "heuristic_latency_ms": round(heuristic_deadline_ms, 3),
                "frame_interval_ms": round(frame_interval_ms, 3),
                "fps_feasible": fps_ok,
                "deadline_feasible": deadline_ok,
                "feasible": fps_ok and deadline_ok,
            })
    viable = [c for c in candidates if c["feasible"]]
    # Prefer the larger model as an *unvalidated quality tier*, not measured accuracy.
    viable.sort(key=lambda c: (c["model"] != "yolo11s.pt", c["predicted_p95_ms"], c["node_id"]))
    return (viable[0] if viable else None), candidates
