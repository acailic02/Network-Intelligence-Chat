from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.query_understanding import UserQuery, LookupAttributes, QueryType
from src.tools.tools import structured_filter, semantic_search, count_matches

class RetrievalState(TypedDict):
    retrieval_type: Literal["structured_filter", "semantic_search"]
    filters: dict
    query_text: str
    results: list[dict]


def router(state: RetrievalState) -> dict:
    #num_matches = count_matches(**state["filters"])
    #TODO: if too many matches do reranking else or if too little (or none) relax filters
    return {}


def decide_search_type(state: RetrievalState) -> str:
    return state["retrieval_type"]

def structured_retrieval(state: RetrievalState) -> dict:
    filters = {k: v for k, v in state["filters"].items() if v is not None}
    results = structured_filter.invoke(filters)
    return {"results": results}

def semantic_retrieval(state: RetrievalState) -> dict:
    results = semantic_search.invoke(state["query_text"])
    return {"results": results}

graph = StateGraph(RetrievalState)
graph.add_node("router", router)
graph.add_node("structured_retrieval", structured_retrieval)
graph.add_node("semantic_retrieval", semantic_retrieval)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", decide_search_type, {"structured_filter": "structured_retrieval", "semantic_search": "semantic_retrieval"})
graph.add_edge("semantic_retrieval", END)
graph.add_edge("structured_retrieval", END)

retrieval_strategy = graph.compile()

def retrieve(specification: UserQuery) -> list[dict]:
    filters = {
        "company": specification.lookup_filters.company,
        "country": specification.lookup_filters.location,
        "owners_all": specification.lookup_filters.network_connection,
    }
    if specification.query_type.value == "LOOKUP":
        rt = "structured_filter"
    else:
        rt = "semantic_search"
    spec = {
        "retrieval_type": rt,
        "filters": filters,
        "query_text": specification.contextual_need,
    }
    result = retrieval_strategy.invoke(spec)
    return result["results"]