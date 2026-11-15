"""V5.2 persistent whole-stream model cache (local 4-node prototype).

Node startup preloads yolo11s.pt. Other allowed models load on first use and
stay cached. The existing server's single execution semaphore prevents two
stream requests from simultaneously using the same model inside a node.
"""
from __future__ import annotations

import json
import re
import threading
import time
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = (ROOT / "data").resolve()
RESULTS_DIR = ROOT / "results" / "stream_jobs"
MODELS = {"yolo11n.pt", "yolo11s.pt"}
_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.RLock()


def preload_model(model_name: str):
    """Return an initialized/warmed YOLO model, caching one per node process."""
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        import numpy as np
        import torch
        from ultralytics import YOLO

        print(f"[V5.2] Preloading {model_name}...", flush=True)
        start = time.perf_counter()
        instance = YOLO(model_name)
        # Same spatial resolution as the existing cars.mp4 test source.
        sample = np.zeros((432, 768, 3), dtype=np.uint8)
        for _ in range(5):
            instance.predict(source=sample, imgsz=320, conf=0.25,
                             device="cpu", verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        _MODEL_CACHE[model_name] = instance
        print(f"[V5.2] {model_name} ready; initialization={elapsed_ms:.2f}ms "
              f"intra_op_threads={torch.get_num_threads()} "
              f"inter_op_threads={torch.get_num_interop_threads()}", flush=True)
        return instance


def run_stream_task(task: dict) -> dict:
    stream_id = str(task.get("stream_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", stream_id):
        raise ValueError("stream_id must be 1-64 letters, digits, '_' or '-'")
    model = task.get("model")
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    node_id = int(task.get("node_id", 0))
    if node_id not in (1, 2, 3, 4):
        raise ValueError("Invalid node_id")
    video = (ROOT / str(task.get("video", ""))).resolve()
    if not video.is_relative_to(DATA_DIR) or not video.is_file():
        raise ValueError("Video must be an existing file under project data/")
    max_frames = int(task.get("max_frames", 10000))
    if not 1 <= max_frames <= 10000:
        raise ValueError("max_frames must be between 1 and 10000")
    deadline_ms = float(task.get("deadline_ms", 250.0))
    if not 0 < deadline_ms <= 60000:
        raise ValueError("deadline_ms must be in (0, 60000]")
    job_uid = str(task.get("task_uid", ""))
    if not re.fullmatch(r"[a-f0-9-]{36}", job_uid):
        raise ValueError("task_uid must be a UUID")

    # Models persist in this node's process; model startup is not per-request.
    persistent_model = preload_model(model)
    from video_inference_baseline import run

    output = RESULTS_DIR / f"node{node_id}" / f"{stream_id}_{job_uid}"
    args = Namespace(
        video=str(video), model=model, device="cpu", imgsz=320,
        frames=max_frames, warmup=5, conf=0.25,
        deadline_ms=deadline_ms, output=str(output), annotate=False,
    )
    run(args, preloaded_model=persistent_model)
    summary_path = output / "video_baseline_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "stream_id": stream_id,
        "model": model,
        "model_reused": summary["model_reused"],
        "frames_processed": summary["frames_processed"],
        "throughput_fps": summary["throughput_fps"],
        "processing_mean_ms": summary["processing_mean_ms"],
        "processing_p95_ms": summary["processing_p95_ms"],
        "processing_p99_ms": summary["processing_p99_ms"],
        "processing_deadline_misses": summary["processing_deadline_misses"],
        "stream_processing_wall_s": summary["wall_time_s"],
        "summary_path": str(summary_path),
        "frame_metrics_path": str(output / "video_frame_metrics.csv"),
    }
