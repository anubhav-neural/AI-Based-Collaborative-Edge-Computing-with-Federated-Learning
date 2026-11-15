"""Scheduler V3: parallel live-status polling with V2's unchanged scoring rule.

Place alongside scheduler_v2.py. Compatible with select_best_node(task_type).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from scheduler_v2 import (
    NODES,
    DEFAULT_RESPONSE_MS,
    DEFAULT_EXECUTION_MS,
    calculate_score,
    get_node_status,
    load_history,
)


def collect_status_parallel():
    """Query all nodes concurrently; return {node_id: status} for reachable nodes."""
    result = {}
    with ThreadPoolExecutor(max_workers=len(NODES)) as executor:
        futures = {
            executor.submit(get_node_status, node_id, port): node_id
            for node_id, port in NODES.items()
        }
        for future in as_completed(futures):
            node_id = futures[future]
            try:
                status = future.result()
            except Exception as error:
                print(f"Node {node_id} status probe failed: {error}")
                continue
            if status is not None:
                result[node_id] = status
    return result


def select_best_node(task_type):
    """Choose lowest V2 heuristic score among currently reachable nodes."""
    started = perf_counter()
    history = load_history(task_type)
    statuses = collect_status_parallel()

    print("\n===== SCHEDULER V3 (PARALLEL STATUS POLLING) =====")
    if not statuses:
        raise RuntimeError("No edge nodes are currently reachable")

    candidates = []
    for node_id in sorted(statuses):
        status = statuses[node_id]
        profile = history.get(node_id, {
            "response_ms": DEFAULT_RESPONSE_MS,
            "execution_ms": DEFAULT_EXECUTION_MS,
            "samples": 0,
        })
        score = calculate_score(status, profile)
        candidates.append((score, node_id))
        print(
            f"Node {node_id} | CPU: {status['cpu_percent']:.1f}% | "
            f"Running: {status['running_tasks']} | "
            f"Waiting: {status['waiting_tasks']} | "
            f"Score: {score:.2f} ms | Samples: {profile['samples']}"
        )

    score, selected = min(candidates)
    elapsed_ms = (perf_counter() - started) * 1000
    print(f"Selected Node: {selected} | Score: {score:.2f} ms")
    print(f"Scheduling time (including polls): {elapsed_ms:.2f} ms")
    return selected


if __name__ == "__main__":
    select_best_node("matrix_multiplication")
