"""V7.1: compare drop-oldest vs deadline-aware local video replay.

This is a local-process experiment, NOT a TCP camera stream. Four separate
processes simulate four edge workers on the same physical machine. Each process
preloads its model, then replays frames on a monotonic clock, with a producer
thread and a single inference consumer. Results include the frame release-to-
completion latency, queue wait, drop counts, and deadline misses.

Smoke test without cv2/YOLO:
    python v7_1_deadline_aware.py --backend mock --synthetic --streams 2 --frames 15 --fps 20 --mock-infer-ms 10

Real experiment (requires opencv-python, ultralytics, model weights):
    python v7_1_deadline_aware.py --video data/cars.mp4 --model yolo11n.pt --streams 4 --frames 100 --fps 20 --queue-size 5 --deadline-ms 250 --policy deadline-aware --estimated-infer-ms 80
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * percent / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "p99_ms": round(percentile(values, 99), 3),
        "max_ms": round(max(values), 3),
    }


def run_worker(index: int, config: dict, ready_queue, go_event, results_queue) -> None:
    """Each worker is one Python process, not an existing TCP node server."""
    stream_id = f"v7-stream-{index:02d}"
    try:
        # Constrain heavy math backend BEFORE importing PyTorch/Ultralytics.
        os.environ["OMP_NUM_THREADS"] = str(config["threads"])
        os.environ["MKL_NUM_THREADS"] = str(config["threads"])
        os.environ["OPENBLAS_NUM_THREADS"] = str(config["threads"])
        output_dir = Path(config["output"]) / stream_id
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = Path(config["video"])
        if config["synthetic"]:
            cv2 = None
            model = None
        else:
            import cv2
            from ultralytics import YOLO
            model = YOLO(config["model"]) if config["backend"] == "yolo" else None

        init_start = time.perf_counter()
        if model is not None:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video {video_path}")
            ok, sample = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"No decodable frames in {video_path}")
            for _ in range(config["warmup"]):
                model.predict(source=sample, imgsz=config["imgsz"], conf=0.25,
                              device="cpu", verbose=False)
        init_ms = (time.perf_counter() - init_start) * 1000
        # Readiness is emitted after model preparation and before paced replay.
        ready_queue.put({"stream_id": stream_id, "status": "ready", "initialization_ms": round(init_ms, 3)})
        if not go_event.wait(timeout=config["startup_timeout"]):
            raise TimeoutError("Coordinator did not release start gate")
        # Each process has its own monotonic clock; gate start happens locally.
        start = time.perf_counter()
        frame_interval_s = 1.0 / config["fps"]
        capacity = config["queue_size"]
        frames_queue: queue.Queue = queue.Queue(maxsize=capacity)
        producer_done = threading.Event()
        counts = {"generated": 0, "overflow_dropped": 0, "deadline_dropped": 0, "decode_failures": 0}
        producer_failure: list[str] = []

        def produce() -> None:
            cap = None
            try:
                if not config["synthetic"]:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        raise RuntimeError("Could not open video in producer")
                for frame_index in range(config["frames"]):
                    target = start + frame_index * frame_interval_s
                    sleep = target - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    scheduled = target
                    if config["synthetic"]:
                        frame = frame_index
                    else:
                        ok, frame = cap.read()
                        if not ok:
                            counts["decode_failures"] += 1
                            break
                    arrival = time.perf_counter()
                    counts["generated"] += 1
                    item = (frame_index, scheduled, arrival, frame)
                    try:
                        frames_queue.put_nowait(item)
                    except queue.Full:
                        # Drop oldest queued frame, NEVER interrupt the in-flight one.
                        try:
                            frames_queue.get_nowait()
                            counts["overflow_dropped"] += 1
                        except queue.Empty:
                            pass
                        try:
                            frames_queue.put_nowait(item)
                        except queue.Full:
                            # Defensive fallback (there is only one producer).
                            counts["overflow_dropped"] += 1
            except Exception as exc:
                producer_failure.append(f"{type(exc).__name__}: {exc}")
            finally:
                if cap is not None:
                    cap.release()
                producer_done.set()

        producer = threading.Thread(target=produce, name=f"producer-{index}", daemon=True)
        producer.start()
        rows: list[dict] = []
        while not (producer_done.is_set() and frames_queue.empty()):
            try:
                frame_index, scheduled, arrival, frame = frames_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            inference_start = time.perf_counter()
            elapsed_ms = (inference_start - scheduled) * 1000.0
            # Deadline-aware admission: skip frames that are already unlikely
            # to finish in time. The estimate is deliberately explicit and
            # DOES NOT provide guaranteed completion or interrupt an active job.
            if (config["policy"] == "deadline-aware" and
                    elapsed_ms + config["estimated_infer_ms"] > config["deadline_ms"]):
                counts["deadline_dropped"] += 1
                continue
            if config["backend"] == "mock":
                time.sleep(config["mock_infer_ms"] / 1000.0)
                detections = 0
            else:
                prediction = model.predict(source=frame, imgsz=config["imgsz"],
                                           conf=0.25, device="cpu", verbose=False)[0]
                detections = len(prediction.boxes) if prediction.boxes is not None else 0
            completed = time.perf_counter()
            queue_ms = (inference_start - arrival) * 1000
            inference_ms = (completed - inference_start) * 1000
            # Scheduled replay tick -> local completion. Includes pacing lateness,
            # decoding, queue wait and inference, but NO TCP/video transmission.
            local_latency_ms = (completed - scheduled) * 1000
            rows.append({
                "frame_index": frame_index,
                "producer_lateness_ms": round((arrival - scheduled) * 1000, 4),
                "queue_wait_ms": round(queue_ms, 4),
                "inference_ms": round(inference_ms, 4),
                "local_release_to_completion_ms": round(local_latency_ms, 4),
                "deadline_miss": int(local_latency_ms > config["deadline_ms"]),
                "detections": detections,
            })
        producer.join(timeout=2)
        if producer_failure:
            raise RuntimeError("Producer failed: " + producer_failure[0])
        if producer.is_alive():
            raise RuntimeError("Producer did not terminate")

        csv_path = output_dir / "frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "frame_index", "producer_lateness_ms", "queue_wait_ms", "inference_ms",
                "local_release_to_completion_ms", "deadline_miss", "detections",
            ])
            writer.writeheader()
            writer.writerows(rows)
        generated = counts["generated"]
        processed = len(rows)
        overflow_dropped = counts["overflow_dropped"]
        deadline_dropped = counts["deadline_dropped"]
        dropped = overflow_dropped + deadline_dropped
        if generated != processed + dropped:
            raise RuntimeError(f"Frame accounting mismatch: {generated} != {processed}+{dropped}")
        summary = {
            "stream_id": stream_id, "status": "success", "model": config["model"],
            "backend": config["backend"], "fps_requested": config["fps"],
            "frames_requested": config["frames"], "frames_generated": generated,
            "frames_processed": processed, "frames_dropped": dropped,
            "frames_dropped_overflow": overflow_dropped,
            "frames_dropped_deadline": deadline_dropped,
            "policy": config["policy"],
            "estimated_infer_ms": config["estimated_infer_ms"],
            "frames_not_generated": config["frames"] - generated,
            "drop_rate_pct": round(100 * dropped / generated, 3) if generated else None,
            "deadline_ms": config["deadline_ms"],
            "deadline_misses_on_processed": sum(row["deadline_miss"] for row in rows),
            "frames_on_time": sum(1 for row in rows if not row["deadline_miss"]),
            "on_time_generated_pct": round(
                100 * sum(1 for row in rows if not row["deadline_miss"]) / generated, 3
            ) if generated else None,
            "queue_wait": stats([row["queue_wait_ms"] for row in rows]),
            "inference": stats([row["inference_ms"] for row in rows]),
            "local_release_to_completion": stats([row["local_release_to_completion_ms"] for row in rows]),
            "producer_lateness": stats([row["producer_lateness_ms"] for row in rows]),
            "model_initialization_ms_before_start": round(init_ms, 3),
            "frame_metrics_csv": str(csv_path),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results_queue.put(summary)
    except Exception as exc:
        details = {"stream_id": stream_id, "status": "error", "error": str(exc), "traceback": traceback.format_exc()}
        results_queue.put(details)
        ready_queue.put({"stream_id": stream_id, "status": "error", "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="V7.1 policy comparison: local paced replay with bounded queues")
    parser.add_argument("--video", default="data/cars.mp4")
    parser.add_argument("--model", choices=("yolo11n.pt", "yolo11s.pt"), default="yolo11n.pt")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--deadline-ms", type=float, default=250)
    parser.add_argument("--queue-size", type=int, default=5)
    parser.add_argument("--policy", choices=("drop-oldest", "deadline-aware"), default="drop-oldest")
    parser.add_argument("--estimated-infer-ms", type=float, default=80.0,
                        help="Expected inference duration for deadline precheck; not a guarantee")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--output", default="results/v7_1_policy_replay")
    parser.add_argument("--backend", choices=("yolo", "mock"), default="yolo")
    parser.add_argument("--synthetic", action="store_true", help="Test with numbered frames, no OpenCV or video file")
    parser.add_argument("--mock-infer-ms", type=float, default=20)
    args = parser.parse_args()
    if not (1 <= args.streams <= 4 and 1 <= args.frames <= 10000 and 1 <= args.queue_size <= 100):
        parser.error("streams must be 1..4, frames 1..10000, queue size 1..100")
    if not (1 <= args.threads <= 16 and 0 <= args.warmup <= 100 and args.imgsz >= 32):
        parser.error("invalid threads, warmup, or image size")
    if not all(math.isfinite(value) and value > 0 for value in
               (args.fps, args.deadline_ms, args.startup_timeout, args.mock_infer_ms, args.estimated_infer_ms)):
        parser.error("fps, deadline, startup timeout, mock infer time and estimated infer time must be positive finite")
    if args.synthetic and args.backend != "mock":
        parser.error("--synthetic requires --backend mock")
    root = Path(__file__).resolve().parent
    video = (root / args.video).resolve()
    if not args.synthetic and not video.is_file():
        parser.error(f"video file missing: {video}")
    output_dir = (root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args))
    config.update({"video": str(video), "output": str(output_dir)})
    ctx = mp.get_context("spawn")  # Required on Windows.
    ready_queue = ctx.Queue()
    results_queue = ctx.Queue()
    go_event = ctx.Event()
    processes = [ctx.Process(target=run_worker,
                             args=(index, config, ready_queue, go_event, results_queue))
                 for index in range(1, args.streams + 1)]
    for process in processes:
        process.start()
    print(f"V7.1 ({args.policy}): waiting for {args.streams} worker(s) to initialize...", flush=True)
    ready = []
    deadline = time.monotonic() + args.startup_timeout
    while len(ready) < args.streams:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            status = ready_queue.get(timeout=min(2.0, remaining))
            ready.append(status)
            print(" READY:", status, flush=True)
            if status["status"] != "ready":
                break
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break
    if len(ready) != args.streams or any(s["status"] != "ready" for s in ready):
        print("ERROR: not all workers initialized; stopping", flush=True)
        go_event.set()
        for process in processes:
            process.terminate()
            process.join(timeout=5)
        raise SystemExit(1)
    print("V7.1: releasing simultaneous paced producers", flush=True)
    go_event.set()
    results = []
    max_run_seconds = args.frames / args.fps + max(30, args.frames * args.mock_infer_ms / 1000.0 * 2)
    finish_by = time.monotonic() + max_run_seconds
    while len(results) < args.streams and time.monotonic() < finish_by:
        try:
            result = results_queue.get(timeout=1.0)
            results.append(result)
            if result["status"] == "success":
                latency = result["local_release_to_completion"]
                print(f"{result['stream_id']}: generated={result['frames_generated']} "
                      f"processed={result['frames_processed']} dropped={result['frames_dropped']} "
                      f"(overflow={result['frames_dropped_overflow']}, deadline={result['frames_dropped_deadline']}) "
                      f"on-time={result['frames_on_time']} "
                      f"p95 local={latency['p95_ms']} ms", flush=True)
            else:
                print(f"{result['stream_id']}: ERROR {result['error']}", flush=True)
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "V7.1 local paced replay with deadline-aware policy comparison",
        "configuration": {key: config[key] for key in (
            "video", "model", "streams", "frames", "fps", "deadline_ms", "queue_size",
            "threads", "warmup", "backend", "synthetic", "mock_infer_ms", "policy", "estimated_infer_ms")},
        "results": sorted(results, key=lambda item: item["stream_id"]),
        "limitations": [
            "All workers share the same computer; no independent physical edge nodes",
            "Recorded video decoded locally, not transmitted over TCP or a physical network",
            "Local latency measured from scheduled replay tick to completion, not camera-to-result E2E",
            "Model initialization is performed before the start gate and excluded from frame latency",
            "Each process has its own frame queue; dropped frames are excluded from processed-frame deadline miss counts",
            "Deadline-aware prediction is a fixed user-supplied estimate, not a calibrated queueing or inference model",
            "The local on-time metric uses scheduled replay release, not network or camera timestamps",
        ],
    }
    target = output_dir / "v7_1_results.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {target}")
    if len(results) == args.streams and all(item["status"] == "success" for item in results):
        generated = sum(item["frames_generated"] for item in results)
        processed = sum(item["frames_processed"] for item in results)
        dropped = sum(item["frames_dropped"] for item in results)
        on_time = sum(item["frames_on_time"] for item in results)
        late = sum(item["deadline_misses_on_processed"] for item in results)
        print(f"TOTAL generated={generated}, processed={processed}, dropped={dropped}, "
              f"on_time={on_time}, late={late}, "
              f"on_time_generated_pct={100*on_time/generated if generated else 0:.2f}%", flush=True)
    if len(results) != args.streams or any(item["status"] != "success" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    mp.freeze_support()
    main()
