"""
Unit tests for src/agents/retrieval_strategy.py
Tests only deterministic functions — no LLM calls, no DB, no vector store.
"""
from unittest.mock import patch, MagicMock


# ---- Tests for get_filters() ----

def test_get_filters_removes_none_values():
    from src.agents.retrieval_strategy import get_filters
    state = {"filters": {"city": ["Belgrade"], "country": None, "skills": None, "owners": ["Petar"]}}
    result = get_filters(state)
    assert "city" in result
    assert "owners" in result
    assert "country" not in result
    assert "skills" not in result


def test_get_filters_empty():
    from src.agents.retrieval_strategy import get_filters
    state = {"filters": {"city": None, "country": None}}
    result = get_filters(state)
    assert result == {}


def test_get_filters_all_present():
    from src.agents.retrieval_strategy import get_filters
    state = {"filters": {"city": ["Belgrade"], "country": ["Serbia"]}}
    result = get_filters(state)
    assert len(result) == 2


# ---- Tests for irrelevant_profiles_drop_off() ----

def test_irrelevant_profiles_drop_off_removes_low_scores():
    from src.agents.retrieval_strategy import irrelevant_profiles_drop_off
    state = {
        "results": [
            {"name": "Marko", "relevance_score": 0.9},
            {"name": "Ana", "relevance_score": 0.1},
            {"name": "Stefan", "relevance_score": 0.5},
        ]
    }
    result = irrelevant_profiles_drop_off(state)
    names = [p["name"] for p in result["results"]]
    assert "Marko" in names
    assert "Stefan" in names
    assert "Ana" not in names


def test_irrelevant_profiles_drop_off_threshold():
    """Profiles with score exactly 0.15 should be removed (not > 0.15)."""
    from src.agents.retrieval_strategy import irrelevant_profiles_drop_off
    state = {
        "results": [
            {"name": "A", "relevance_score": 0.15},
            {"name": "B", "relevance_score": 0.16},
        ]
    }
    result = irrelevant_profiles_drop_off(state)
    names = [p["name"] for p in result["results"]]
    assert "A" not in names
    assert "B" in names


def test_irrelevant_profiles_drop_off_no_score():
    """Profiles without relevance_score should be removed (default 0.0)."""
    from src.agents.retrieval_strategy import irrelevant_profiles_drop_off
    state = {
        "results": [
            {"name": "A"},
            {"name": "B", "relevance_score": 0.8},
        ]
    }
    result = irrelevant_profiles_drop_off(state)
    names = [p["name"] for p in result["results"]]
    assert "A" not in names
    assert "B" in names


def test_irrelevant_profiles_drop_off_empty():
    from src.agents.retrieval_strategy import irrelevant_profiles_drop_off
    state = {"results": []}
    result = irrelevant_profiles_drop_off(state)
    assert result["results"] == []


# ---- Tests for final_profile_sorting() ----

def test_final_profile_sorting_orders_by_score():
    from src.agents.retrieval_strategy import final_profile_sorting
    state = {
        "results": [
            {"name": "A", "relevance_score": 0.3},
            {"name": "B", "relevance_score": 0.9},
            {"name": "C", "relevance_score": 0.6},
        ]
    }
    result = final_profile_sorting(state)
    names = [p["name"] for p in result["results"]]
    assert names == ["B", "C", "A"]


def test_final_profile_sorting_empty():
    from src.agents.retrieval_strategy import final_profile_sorting
    state = {"results": []}
    result = final_profile_sorting(state)
    assert result["results"] == []


# ---- Tests for relaxation_node() ----

def test_relaxation_node_removes_relaxed_filters():
    from src.agents.retrieval_strategy import relaxation_node
    mock_decision = MagicMock()
    mock_decision.switch_strategy_to = "no_change"

    state = {
        "filters": {"city": ["Belgrade"], "country": ["Serbia"], "skills": ["Python"]},
        "relaxed_filters": ["city"],
        "retrieval_type": "structured_filter",
        "attempts": 1,
        "decision": mock_decision,
    }
    result = relaxation_node(state)
    assert "city" not in result["filters"]
    assert "country" in result["filters"]


def test_relaxation_node_switches_strategy():
    from src.agents.retrieval_strategy import relaxation_node
    mock_decision = MagicMock()
    mock_decision.switch_strategy_to = "semantic_search"

    state = {
        "filters": {"city": ["Belgrade"]},
        "relaxed_filters": [],
        "retrieval_type": "structured_filter",
        "attempts": 0,
        "decision": mock_decision,
    }
    result = relaxation_node(state)
    assert result["retrieval_type"] == "semantic_search"


def test_relaxation_node_increments_attempts():
    from src.agents.retrieval_strategy import relaxation_node
    mock_decision = MagicMock()
    mock_decision.switch_strategy_to = "no_change"

    state = {
        "filters": {},
        "relaxed_filters": [],
        "retrieval_type": "structured_filter",
        "attempts": 2,
        "decision": mock_decision,
    }
    result = relaxation_node(state)
    assert result["attempts"] == 3


def test_relaxation_node_keeps_strategy_on_no_change():
    from src.agents.retrieval_strategy import relaxation_node
    mock_decision = MagicMock()
    mock_decision.switch_strategy_to = "no_change"

    state = {
        "filters": {},
        "relaxed_filters": [],
        "retrieval_type": "hybrid_search",
        "attempts": 0,
        "decision": mock_decision,
    }
    result = relaxation_node(state)
    assert result["retrieval_type"] == "hybrid_search"


# ---- Tests for route_retrieval_type() ----

def test_route_retrieval_type_structured():
    from src.agents.retrieval_strategy import route_retrieval_type
    state = {"retrieval_type": "structured_filter"}
    assert route_retrieval_type(state) == "structured_filter"


def test_route_retrieval_type_semantic():
    from src.agents.retrieval_strategy import route_retrieval_type
    state = {"retrieval_type": "semantic_search"}
    assert route_retrieval_type(state) == "semantic_search"


def test_route_retrieval_type_hybrid():
    from src.agents.retrieval_strategy import route_retrieval_type
    state = {"retrieval_type": "hybrid_search"}
    assert route_retrieval_type(state) == "hybrid_search"


# ---- Tests for route_next_action() ----

def test_route_next_action_finish():
    from src.agents.retrieval_strategy import route_next_action
    mock_decision = MagicMock()
    mock_decision.action = "finish"
    state = {"decision": mock_decision}
    assert route_next_action(state) == "finish"


def test_route_next_action_relax():
    from src.agents.retrieval_strategy import route_next_action
    mock_decision = MagicMock()
    mock_decision.action = "relax"
    state = {"decision": mock_decision}
    assert route_next_action(state) == "relax"


def test_route_next_action_rerank():
    from src.agents.retrieval_strategy import route_next_action
    mock_decision = MagicMock()
    mock_decision.action = "rerank"
    state = {"decision": mock_decision}
    assert route_next_action(state) == "rerank"