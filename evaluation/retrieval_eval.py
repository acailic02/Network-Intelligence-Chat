import json
import os

from src.agents.workflow import build_network_intelligence_workflow

VALIDATION_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_validation_set.json")
workflow = build_network_intelligence_workflow()


def precision_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def evaluate_query(id: str, query_type: str, user_input: str, relevant_ids: list[str], k: int = 10) -> dict:
    result = workflow.invoke({"user_input": user_input, "conversation_history": []})
    profiles = result["profiles_data"]
    retrieved_ids = [p.get("linkedin_url") for p in profiles]
    relevant_set = set(relevant_ids)

    pk = precision_k(retrieved_ids, relevant_set, k)
    rk = recall_k(retrieved_ids, relevant_set, k)
    p3 = precision_k(retrieved_ids, relevant_set, k=3)

    return {
        "id": id,
        "query_type": query_type,
        "user_input": user_input,
        "retrieved_ids": retrieved_ids,
        "relevant_ids": relevant_ids,
        "precision_10": round(pk, 4),
        "precision_3": round(p3, 4),
        "recall_10": round(rk, 4),
        "num_retrieved": len(retrieved_ids),
        "num_relevant": len(relevant_ids),
    }


def print_stats(results: list[dict]):
    structured_results = [result for result in results if result.get("query_type") == "LOOKUP"]
    semantic_results = [result for result in results if result.get("query_type") == "DISCOVERY"]
    hybrid_results = [result for result in results if result.get("query_type") == "HYBRID"]

    if structured_results:
        struct_stats = \
            {
                "query_type": "LOOKUP",
                "avg_precision_10": sum([r.get("precision_10") for r in structured_results]) / len(structured_results),
                "avg_precision_3": sum([r.get("precision_3") for r in structured_results]) / len(structured_results),
                "avg_recall_10": sum([r.get("recall_10") for r in structured_results]) / len(structured_results),
            }
    if semantic_results:
        semantic_stats = \
            {
                "query_type": "DISCOVERY",
                "avg_precision_10": sum([r.get("precision_10") for r in semantic_results]) / len(semantic_results),
                "avg_recall_10": sum([r.get("recall_10") for r in semantic_results]) / len(semantic_results),
            }
    if hybrid_results:
        hybrid_stats = \
            {
                "query_type": "HYBRID",
                "avg_precision_10": sum([r.get("precision_10") for r in hybrid_results]) / len(hybrid_results),
                "avg_recall_10": sum([r.get("recall_10") for r in hybrid_results]) / len(hybrid_results),
            }

    print("STRUCTURED STATISTICS: \n")
    for r in structured_results:
        print(f"QUERY: {r.get('user_input')}\n")
        print(f"PRECISION@10: {r.get('precision_10')}\n")
        print(f"PRECISION@3: {r.get('precision_3')}\n")
        print(f"RECALL@10: {r.get('recall_10')}\n")
    print("AVG PRECISION@10: ", struct_stats["avg_precision_10"])
    print("AVG PRECISION@3: ", struct_stats["avg_precision_3"])
    print("AVG RECALL@10: ", struct_stats["avg_recall_10"])
    print("\nDISCOVERY STATISTICS: \n")
    for r in semantic_results:
        print(f"QUERY: {r.get('user_input')}\n")
        print(f"PRECISION@10: {r.get('precision_10')}\n")
        print(f"RECALL@10: {r.get('recall_10')}\n")
    print("AVG PRECISION@10: ", semantic_stats["avg_precision_10"])
    print("AVG RECALL@10: ", semantic_stats["avg_recall_10"])
    print("\nHYBRID STATISTICS: \n")
    for r in hybrid_results:
        print(f"QUERY: {r.get('user_input')}\n")
        print(f"PRECISION@10: {r.get('precision_10')}\n")
        print(f"RECALL@10: {r.get('recall_10')}\n")
    print("AVG PRECISION@10: ", hybrid_stats["avg_precision_10"])
    print("AVG RECALL@10: ", hybrid_stats["avg_recall_10"])


def run_eval():
    with open(VALIDATION_SET_PATH, 'r', encoding='utf-8') as f:
        validation_set = json.load(f)
        results = []
        for query in validation_set:
            results.append(evaluate_query(**query))

        print_stats(results)


if __name__ == "__main__":
    run_eval()