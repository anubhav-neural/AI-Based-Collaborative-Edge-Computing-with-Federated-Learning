
import json
from pathlib import Path

RESULTS = Path("results/video_baseline")

models = {
    "YOLO11n": RESULTS / "yolo11n_summary.json",
    "YOLO11s": RESULTS / "yolo11s_summary.json"
}

data = {}

for name, path in models.items():
    if not path.exists():
        print(f"Missing result file: {path}")
        raise SystemExit(1)

    with open(path, "r", encoding="utf-8") as f:
        data[name] = json.load(f)

print("\n===== YOLO MODEL COMPARISON =====")

# Print all available metrics first, so that we use
# the actual JSON field names rather than assuming them.
for model, metrics in data.items():
    print(f"\n{model}")
    print("-" * 35)
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            print(f"{key}: {value}")
