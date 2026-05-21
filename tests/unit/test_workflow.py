"""
Unit tests for src/agents/workflow.py
Tests graph structure only — no LLM calls.
"""
from unittest.mock import patch, MagicMock


def test_workflow_builds_without_error():
    """build_network_intelligence_workflow() should return a compiled graph."""
    from src.agents.workflow import build_network_intelligence_workflow
    workflow = build_network_intelligence_workflow()
    assert workflow is not None


def test_workflow_has_correct_nodes():
    """Graph should contain all three required nodes."""
    from src.agents.workflow import build_network_intelligence_workflow
    workflow = build_network_intelligence_workflow()
    nodes = set(workflow.get_graph().nodes.keys())
    assert "query_understanding_node" in nodes
    assert "retrieval_strategy_node" in nodes
    assert "synthesis_node" in nodes


def test_workflow_state_has_required_fields():
    """WorkflowState TypedDict should have all required fields."""
    from src.agents.workflow import build_network_intelligence_workflow
    import inspect

    workflow = build_network_intelligence_workflow()

    # Get WorkflowState from the closure
    source = inspect.getsource(build_network_intelligence_workflow)
    required_fields = ["user_input", "query_understanding_state", "results", "answer", "profiles_data"]
    for field in required_fields:
        assert field in source, f"Missing field in WorkflowState: {field}"


@patch("src.agents.workflow.understand")
@patch("src.agents.workflow.retrieve")
@patch("src.agents.workflow.synthesize")
def test_workflow_invoke_calls_all_agents(mock_synthesize, mock_retrieve, mock_understand):
    """Invoking the workflow should call all three agents in order."""
    from src.agents.query_understanding import UserQuery, LookupAttributes, QueryType
    from src.agents.workflow import build_network_intelligence_workflow

    mock_understand.return_value = UserQuery(
        lookup_filters=LookupAttributes(),
        contextual_need=None,
        contextual_trigger=None,
        query_type=QueryType.LOOKUP,
    )
    mock_retrieve.return_value = []
    mock_synthesize.return_value = ("No results found.", [])

    workflow = build_network_intelligence_workflow()
    result = workflow.invoke({
        "user_input": "Who works at Stripe?",
        "conversation_history": [],
    })

    mock_understand.assert_called_once()
    mock_retrieve.assert_called_once()
    mock_synthesize.assert_called_once()


@patch("src.agents.workflow.understand")
@patch("src.agents.workflow.retrieve")
@patch("src.agents.workflow.synthesize")
def test_workflow_returns_answer(mock_synthesize, mock_retrieve, mock_understand):
    """Workflow result should contain answer and profiles_data."""
    from src.agents.query_understanding import UserQuery, LookupAttributes, QueryType
    from src.agents.workflow import build_network_intelligence_workflow

    mock_understand.return_value = UserQuery(
        lookup_filters=LookupAttributes(),
        contextual_need=None,
        contextual_trigger=None,
        query_type=QueryType.LOOKUP,
    )
    mock_retrieve.return_value = [{"name": "Marko", "linkedin_url": "https://linkedin.com/in/marko"}]
    mock_synthesize.return_value = ("Marko is a strong connection.", [{"name": "Marko"}])

    workflow = build_network_intelligence_workflow()
    result = workflow.invoke({
        "user_input": "Who works at Stripe?",
        "conversation_history": [],
    })

    assert result["answer"] == "Marko is a strong connection."
    assert len(result["profiles_data"]) == 1