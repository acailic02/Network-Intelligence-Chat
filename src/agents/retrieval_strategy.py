from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.query_understanding import UserQuery, LookupAttributes, QueryType
from src.tools.tools import structured_filter, semantic_search, count_matches, hybrid_search


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

def hybrid_retrieval(state: RetrievalState) -> dict:
    kwargs = {
        "filters": state["filters"],
        "query_text": state["query_text"]
    }
    results = hybrid_search.invoke(kwargs)
    return {"results": results}

graph = StateGraph(RetrievalState)
graph.add_node("router", router)
graph.add_node("structured_retrieval", structured_retrieval)
graph.add_node("semantic_retrieval", semantic_retrieval)
graph.add_node("hybrid_retrieval", hybrid_retrieval)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", decide_search_type, {"structured_filter": "structured_retrieval", "semantic_search": "semantic_retrieval", "hybrid_search": "hybrid_retrieval"})
graph.add_edge("semantic_retrieval", END)
graph.add_edge("structured_retrieval", END)
graph.add_edge("hybrid_retrieval", END)

retrieval_strategy = graph.compile()

def retrieve(specification: UserQuery) -> list[dict]:
    position = specification.lookup_filters.position
    connection = specification.lookup_filters.connection
    education = specification.lookup_filters.education
    #TODO:insert filters for all possible parameters for get_connections() (leave untill query_understanding.py adopts changes)
    filters = {
        "current_company_name": position.company_name[0] if position is not None and position.company_name is not None else None,
        "company_location": position.company_location if position is not None else None,
        "current_job_title": position.title[0] if position is not None and position.title is not None else None,
        "country": connection.country[0] if connection is not None and connection.country is not None else None,
        "city": connection.city[0] if connection is not None and connection.city is not None else None,
        "skills": connection.skills if connection is not None else None,
        "degree": education.degree if education is not None else None,
        "school_name": education.school_name if education is not None else None,
        "owners_all": specification.lookup_filters.owner,
    }
    print(filters)
    query_type = specification.query_type.value
    if query_type == "LOOKUP":
        rt = "structured_filter"
    elif query_type == "DISCOVERY":
        rt = "semantic_search"
    else:
        rt = "hybrid_search"

    if rt in ["semantic_search, hybrid_search"] and not specification.contextual_need: #guard
        rt = "structured_filter"

    spec = {
        "retrieval_type": rt,
        "filters": filters,
        "query_text": specification.contextual_need,
    }
    result = retrieval_strategy.invoke(spec)
    return result["results"]