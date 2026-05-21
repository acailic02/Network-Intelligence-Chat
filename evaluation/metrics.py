import json
import os
from pathlib import Path

logs_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs/llm_calls.jsonl")

def query_type_accuracy(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.get("details", {}).get("query_type"))
    
    by_type = {}
    for qtype in ["LOOKUP", "DISCOVERY", "HYBRID"]:
        subset = [r for r in results if r.get("type") == qtype]
        if subset:
            subset_correct = sum(1 for r in subset if r.get("details", {}).get("query_type"))
            by_type[qtype] = {
                "correct": subset_correct,
                "total": len(subset),
                "accuracy": round(subset_correct / len(subset) * 100, 1)
            }
    
    return {
        "overall": round(correct / total * 100, 1) if total else 0,
        "by_type": by_type
    }


def field_accuracy(results: list[dict]) -> dict:
    fields = [
        "query_type",
        "contextual_need_populated",
        "contextual_trigger_populated",
        "lookup_filters"
    ]
    
    field_stats = {}
    for field in fields:
        total = len(results)
        correct = sum(1 for r in results if r.get("details", {}).get(field))
        field_stats[field] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total * 100, 1) if total else 0
        }
    
    return field_stats


def false_positive_rate(results: list[dict]) -> dict:
    contextual_need_fp = 0
    contextual_trigger_fp = 0
    total = len(results)

    for r in results:
        details = r.get("details", {})
        # populated when should be None
        if not details.get("contextual_need_populated") and not details.get("query_type"):
            contextual_need_fp += 1
        if not details.get("contextual_trigger_populated"):
            contextual_trigger_fp += 1

    return {
        "contextual_need_false_positive": round(contextual_need_fp / total * 100, 1) if total else 0,
        "contextual_trigger_false_positive": round(contextual_trigger_fp / total * 100, 1) if total else 0,
    }


def avg_latency(logs_path) -> dict:
    path = Path(logs_path)
    if not path.exists():
        return {"avg_latency_s": None, "p95_latency_s": None}

    latencies = []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "latency_s" in entry:
                    latencies.append(entry["latency_s"])
            except json.JSONDecodeError:
                continue

    if not latencies:
        return {"avg_latency_s": None, "p95_latency_s": None}

    latencies.sort()
    avg = round(sum(latencies) / len(latencies), 3)
    p95_index = int(len(latencies) * 0.95)
    p95 = latencies[min(p95_index, len(latencies) - 1)]

    return {
        "avg_latency_s": avg,
        "p95_latency_s": round(p95, 3),
        "total_calls": len(latencies)
    }


def difficulty_breakdown(results: list[dict]) -> dict:
    breakdown = {}
    for difficulty in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.get("difficulty") == difficulty]
        if subset:
            passed = sum(1 for r in subset if r.get("status") == "PASS")
            breakdown[difficulty] = {
                "passed": passed,
                "total": len(subset),
                "accuracy": round(passed / len(subset) * 100, 1)
            }
    return breakdown


def compute_all(results: list[dict], logs_path: str = "logs/llm_calls.jsonl"):
    print("\n" + "=" * 40)
    print("METRICS SUMMARY")
    print("=" * 40)

    qt = query_type_accuracy(results)
    print(f"\nQuery Type Accuracy: {qt['overall']}%")
    for qtype, stats in qt["by_type"].items():
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")

    print("\nField Accuracy:")
    for field, stats in field_accuracy(results).items():
        print(f"  {field}: {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")

    print("\nFalse Positive Rate:")
    for field, rate in false_positive_rate(results).items():
        print(f"  {field}: {rate}%")

    print("\nDifficulty Breakdown:")
    for difficulty, stats in difficulty_breakdown(results).items():
        print(f"  {difficulty}: {stats['passed']}/{stats['total']} ({stats['accuracy']}%)")

    latency = avg_latency(logs_path)
    print(f"\nLatency:")
    print(f"  avg: {latency['avg_latency_s']}s")
    print(f"  p95: {latency['p95_latency_s']}s")
    print(f"  total calls: {latency['total_calls']}")
    print("=" * 40)