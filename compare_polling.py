"""Measure V2 sequential vs V3 parallel live-status polling.

Runs status-only scheduling (no tasks dispatched); prints summaries and saves CSV.
Run with all four servers idle: python compare_polling.py --rounds 10
"""
import argparse
import csv
import random
import statistics
import time
from pathlib import Path

from scheduler_v2 import select_best_node as select_v2
from scheduler_v3 import select_best_node as select_v3

OUTPUT = Path(__file__).resolve().parent / "polling_comparison_results.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--task", default="matrix_multiplication")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")

    policies = [("V2_sequential", select_v2), ("V3_parallel", select_v3)]
    rows = []
    rng = random.Random(42)
    for round_number in range(1, args.rounds + 1):
        # Alternate ordering to reduce simple first-run/order bias.
        order = policies[:]
        rng.shuffle(order)
        for name, fn in order:
            start = time.perf_counter()
            try:
                selected = fn(args.task)
                error = ""
            except Exception as exc:
                selected = ""
                error = repr(exc)
            elapsed = (time.perf_counter() - start) * 1000
            rows.append({
                "round": round_number,
                "policy": name,
                "selected_node": selected,
                "scheduling_ms": round(elapsed, 3),
                "error": error,
            })
            print(f"ROUND {round_number} | {name} | {elapsed:.2f} ms | "
                  f"selected={selected} | error={error or 'none'}")
        time.sleep(0.15)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== SUMMARY (scheduling only, no task dispatch) ===")
    for name, _ in policies:
        successful = [row["scheduling_ms"] for row in rows
                      if row["policy"] == name and not row["error"]]
        if successful:
            print(f"{name}: n={len(successful)}, mean={statistics.mean(successful):.2f} ms, "
                  f"median={statistics.median(successful):.2f} ms, "
                  f"max={max(successful):.2f} ms")
        else:
            print(f"{name}: no successful rounds")
    print(f"Saved: {OUTPUT}")
    print("Important: results are local loopback latency, not independent network links.")


if __name__ == "__main__":
    main()
