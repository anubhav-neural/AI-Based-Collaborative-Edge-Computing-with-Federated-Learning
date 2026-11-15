"""Whole-video-stream inference baseline (single local node, no offloading).

Usage:
  python video_inference_baseline.py --generate-demo
  python video_inference_baseline.py --video data/demo_stream.mp4 --model yolo11n.pt --frames 120
  python video_inference_baseline.py --video path/to/real_video.mp4 --model yolo11n.pt --frames 300

First execution downloads model weights if not available locally.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def generate_demo_video(path: Path, frames: int = 150, fps: float = 15.0) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Unable to create MP4 demo video; try --video with an existing MP4")
    try:
        for i in range(frames):
            frame = np.full((360, 640, 3), 35, dtype=np.uint8)
            x = 30 + (i * 3) % 520
            cv2.rectangle(frame, (x, 140), (x + 80, 240), (80, 180, 230), -1)
            cv2.circle(frame, (500, 70 + (i * 2) % 180), 22, (200, 130, 90), -1)
            cv2.putText(frame, f"Synthetic demonstration frame {i:03d}", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Created demo video: {path} ({frames} synthetic frames, {fps:.0f} FPS)")
    print("Note: synthetic shapes are for pipeline testing, not detection-accuracy evaluation.")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def run(args: argparse.Namespace, preloaded_model=None) -> None:
    # First-call import costs are measured inside each worker process.
    # Repeated imports in the same process may be cached by Python.
    import_start = time.perf_counter()
    import cv2
    cv2_import_ms = (time.perf_counter() - import_start) * 1000

    import_start = time.perf_counter()
    from ultralytics import YOLO
    ultralytics_import_ms = (time.perf_counter() - import_start) * 1000

    print(f"[PROFILE] OpenCV import: {cv2_import_ms:.3f} ms", flush=True)
    print(f"[PROFILE] Ultralytics import: {ultralytics_import_ms:.3f} ms", flush=True)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video missing: {video_path}. Run --generate-demo or supply --video FILE")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reused_model = preloaded_model is not None
    if reused_model:
        model = preloaded_model
        model_load_ms = 0.0  # Startup cost was paid before this request.
        print(f"[V5.2] Reusing preloaded model: {args.model}", flush=True)
    else:
        print(f"Loading model {args.model} (first run may download weights)...")
        model_load_start = time.perf_counter()
        model = YOLO(args.model)
        model_load_ms = (time.perf_counter() - model_load_start) * 1000
        print(f"Model object loading time: {model_load_ms:.3f} ms")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {source_w}x{source_h}, declared FPS={source_fps:.2f}, frames={source_frames}")

    # Warm-up the SAME model on a sample frame before measuring; avoid treating
    # one-time model initialization as steady-state inference latency.
    ok, sample = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Video contains no decodable frames")
    warmup_start = time.perf_counter()
    warmup_prediction_ms: list[float] = []
    # A persistent model was already warmed up at node startup.
    warmup_count = 0 if reused_model else args.warmup
    for i in range(warmup_count):
        prediction_start = time.perf_counter()
        model.predict(source=sample, imgsz=args.imgsz, conf=args.conf,
                      device=args.device, verbose=False)
        prediction_ms = (time.perf_counter() - prediction_start) * 1000
        warmup_prediction_ms.append(round(prediction_ms, 3))
        print(f"Warm-up prediction {i + 1}: {prediction_ms:.3f} ms")
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    print(f"Total model warm-up time: {warmup_ms:.3f} ms")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Some codecs don't implement seeks reliably. Reopen from the start.
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to reopen video after warm-up")

    writer = None
    if args.annotate:
        out_video = output_dir / "annotated_stream.mp4"
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 source_fps if source_fps > 0 else 15.0,
                                 (source_w, source_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not open annotated video writer")

    rows: list[dict] = []
    stream_start = time.perf_counter()
    try:
        while len(rows) < args.frames:
            read_start = time.perf_counter()
            ok, frame = cap.read()
            decode_ms = (time.perf_counter() - read_start) * 1000
            if not ok:
                break

            model_start = time.perf_counter()
            result = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                                   device=args.device, verbose=False)[0]
            model_wall_ms = (time.perf_counter() - model_start) * 1000
            # Ultralytics' internal preprocess/inference/postprocess times are
            # supplementary measurements, not replacements for wall time.
            speed = getattr(result, "speed", {}) or {}
            total_ms = decode_ms + model_wall_ms
            row = {
                "frame_index": len(rows),
                "decode_ms": round(decode_ms, 4),
                "model_wall_ms": round(model_wall_ms, 4),
                "processing_ms": round(total_ms, 4),
                "model_internal_inference_ms": round(float(speed.get("inference", 0)), 4),
                "detections": len(result.boxes) if result.boxes is not None else 0,
                "deadline_miss": int(total_ms > args.deadline_ms),
            }
            rows.append(row)
            if writer is not None:
                writer.write(result.plot())
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    stream_wall_s = time.perf_counter() - stream_start
    if not rows:
        raise RuntimeError("No frames processed")

    csv_path = output_dir / "video_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    processing = [row["processing_ms"] for row in rows]
    model_walls = [row["model_wall_ms"] for row in rows]
    misses = sum(row["deadline_miss"] for row in rows)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path), "model": args.model, "device": args.device,
        "imgsz": args.imgsz, "confidence_threshold": args.conf,
        "source_fps": source_fps, "source_resolution": [source_w, source_h],
        "frames_processed": len(rows), "warmup_frames": warmup_count,
        "model_reused": reused_model,
        "cv2_import_ms": round(cv2_import_ms, 3),
        "ultralytics_import_ms": round(ultralytics_import_ms, 3),
        "model_load_ms": round(model_load_ms, 3),
        "warmup_total_ms": round(warmup_ms, 3),
        "warmup_prediction_ms": warmup_prediction_ms,
        "wall_time_s": round(stream_wall_s, 4),
        "throughput_fps": round(len(rows) / stream_wall_s, 3),
        "processing_mean_ms": round(statistics.mean(processing), 3),
        "processing_median_ms": round(statistics.median(processing), 3),
        "processing_p95_ms": round(percentile(processing, 95), 3),
        "processing_p99_ms": round(percentile(processing, 99), 3),
        "model_wall_mean_ms": round(statistics.mean(model_walls), 3),
        "deadline_ms": args.deadline_ms,
        "processing_deadline_misses": misses,
        "processing_deadline_miss_rate_pct": round(100 * misses / len(rows), 2),
        "note": "Offline sequential processing of every decoded frame. Processing deadline excludes video arrival, network, scheduling, and queueing; it is NOT full end-to-end stream latency.",
    }
    summary_path = output_dir / "video_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SINGLE-NODE WHOLE-STREAM BASELINE ===")
    print(f"cv2_import_ms: {summary['cv2_import_ms']}")
    print(f"ultralytics_import_ms: {summary['ultralytics_import_ms']}")
    print(f"model_load_ms: {summary['model_load_ms']}")
    print(f"warmup_total_ms: {summary['warmup_total_ms']}")
    for key in ("frames_processed", "throughput_fps", "processing_mean_ms",
                "processing_median_ms", "processing_p95_ms", "processing_p99_ms",
                "model_wall_mean_ms", "processing_deadline_miss_rate_pct"):
        print(f"{key}: {summary[key]}")
    print(f"Per-frame CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    if args.annotate:
        print(f"Annotated video: {output_dir / 'annotated_stream.mp4'}")
    print("Note: offline processing; NOT an end-to-end 250 ms deadline test.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full video stream on one local YOLO model")
    parser.add_argument("--video", default="data/demo_stream.mp4", help="Path to existing video")
    parser.add_argument("--generate-demo", action="store_true", help="Generate a synthetic MP4 then exit")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics detection checkpoint")
    parser.add_argument("--device", default="cpu", help="'cpu' or '0' for GPU")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--output", default="results/video_baseline")
    parser.add_argument("--annotate", action="store_true", help="Save annotated output video; increases overhead")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.imgsz < 32 or args.deadline_ms <= 0:
        parser.error("--frames must be >=1, --warmup >=0, --imgsz >=32 and --deadline-ms >0")
    try:
        if args.generate_demo:
            generate_demo_video(Path(args.video))
        else:
            run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
