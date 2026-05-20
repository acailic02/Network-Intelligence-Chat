# src/agents/workflow.py
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.query_understanding import understand, UserQuery
from src.agents.retrieval_strategy import retrieve, RetrievalState
from src.agents.synthesis import synthesize


def build_network_intelligence_workflow():
    class WorkflowState(TypedDict):
        user_input: str
        query_understanding_state: UserQuery
        conversation_history: list[dict]
        results: list[dict]
        answer: str
        profiles_data: list[dict]

    def query_understanding_node(state: WorkflowState) -> dict:
        output = understand(state["user_input"], state["conversation_history"])
        return {"query_understanding_state": output}

    def retrieval_strategy_node(state: WorkflowState) -> dict:
        output = retrieve(state["query_understanding_state"], state["user_input"])
        return {"results": output}

    def synthesis_node(state: WorkflowState) -> dict:
        profiles = state["results"]
        answer, profiles_data = synthesize(state["user_input"], profiles, state["conversation_history"])
        return {"answer": answer, "profiles_data": profiles_data}


    graph = StateGraph(WorkflowState)

    graph.add_node("query_understanding_node", query_understanding_node)
    graph.add_node("retrieval_strategy_node", retrieval_strategy_node)
    graph.add_node("synthesis_node", synthesis_node)

    graph.set_entry_point("query_understanding_node")
    graph.add_edge("query_understanding_node", "retrieval_strategy_node")
    graph.add_edge("retrieval_strategy_node", "synthesis_node")
    graph.add_edge("synthesis_node", END)

    return graph.compile()
