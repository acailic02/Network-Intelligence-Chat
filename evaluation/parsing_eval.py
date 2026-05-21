import json
import os

from evaluation.metrics import compute_all
from src.agents.query_understanding import understand

VALIDATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parsing_validation_set.json")

def load_validation_set():
    with open(VALIDATION_PATH) as f:
        return json.load(f)

def compare(expected: dict, actual: dict) -> dict:
    results = {}
    
    # query_type
    results["query_type"] = actual.get("query_type") == expected.get("query_type")
    
    # contextual_need — samo da li je None ili populated, ne exact match
    expected_cn = expected.get("contextual_need")
    actual_cn = actual.get("contextual_need")
    results["contextual_need_populated"] = (expected_cn is None) == (actual_cn is None)
    
    # contextual_trigger — isto
    expected_ct = expected.get("contextual_trigger")
    actual_ct = actual.get("contextual_trigger")
    results["contextual_trigger_populated"] = (expected_ct is None) == (actual_ct is None)
    
    # lookup_filters — proveri svaki filter
    expected_filters = expected.get("lookup_filters", {})
    actual_filters = actual.get("lookup_filters", {})
    results["lookup_filters"] = compare_filters(expected_filters, actual_filters)
    
    results["passed"] = all([
        results["query_type"],
        results["contextual_need_populated"],
        results["contextual_trigger_populated"],
        results["lookup_filters"]
    ])
    
    return results

def compare_filters(expected: dict, actual: dict) -> bool:
    for key, expected_val in expected.items():
        actual_val = actual.get(key)
        
        # ako je expected None, actual mora biti None
        if expected_val is None:
            if actual_val is not None:
                return False
        # ako je expected dict, rekurzivno poredi
        elif isinstance(expected_val, dict):
            if not isinstance(actual_val, dict):
                return False
            if not compare_filters(expected_val, actual_val):
                return False
        # ako je lista, poredi bez reda
        elif isinstance(expected_val, list):
            if not isinstance(actual_val, list):
                return False
            if set(map(str.lower, expected_val)) != set(map(str.lower, actual_val)):
                return False
        # scalar vrednosti
        else:
            if str(expected_val).lower() != str(actual_val).lower():
                return False
    return True

def run():
    tests = load_validation_set()
    results = []
    passed = 0

    for test in tests:
        print(f"Running test {test['id']}: {test['description']}...")
        
        try:
            actual = understand(test["input"])
            actual_dict = actual.model_dump()
            comparison = compare(test["expected"], actual_dict)
            
            if comparison["passed"]:
                passed += 1
                status = "PASS"
            else:
                status = "FAIL"
            
            results.append({
                "id": test["id"],
                "difficulty": test["difficulty"],
                "type": test["type"],
                "description": test["description"],
                "status": status,
                "details": comparison
            })
            
            print(f"  {status}")
            if status == "FAIL":
                print(f"  Expected: {json.dumps(test['expected'], indent=2)}")
                print(f"  Actual:   {json.dumps(actual_dict, indent=2)}")
        
        except Exception as e:
            results.append({
                "id": test["id"],
                "status": "ERROR",
                "error": str(e)
            })
            print(f"  ERROR: {e}")

    total = len(tests)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed ({round(passed/total*100)}%)")
    
    # breakdown po difficulty
    for difficulty in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.get("difficulty") == difficulty]
        subset_passed = sum(1 for r in subset if r["status"] == "PASS")
        print(f"  {difficulty}: {subset_passed}/{len(subset)}")
    
    # breakdown po type
    for qtype in ["LOOKUP", "DISCOVERY", "HYBRID"]:
        subset = [r for r in results if r.get("type") == qtype]
        subset_passed = sum(1 for r in subset if r["status"] == "PASS")
        print(f"  {qtype}: {subset_passed}/{len(subset)}")

    compute_all(results)

if __name__ == "__main__":
    run()