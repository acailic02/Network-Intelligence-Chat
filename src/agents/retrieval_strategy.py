from typing import TypedDict, Annotated, Literal, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from src.agents.query_understanding import UserQuery, LookupAttributes, QueryType
from src.tools.tools import structured_filter, semantic_search, count_matches, hybrid_search
from src.llm.client import get_llm

MAX_ATTEMPTS = 5

EVALUATOR_PROMPT = """
You are the Retrieval Evaluator in a network intelligence assistant. The system 
searches a combined LinkedIn network of a small team to answer the user's question. 
Your job is to look at the current retrieval results and decide what to do next.

# Your task
Decide ONE of three actions:
- "finish"  — the current results are sufficient to answer the user's query well. 
The synthesis step will use them as-is.
- "relax"   — the results are too few, empty, or too narrow to answer the query. 
You must decide how to broaden the search: drop one or more filters, and/or 
switch retrieval strategy.
- "rerank"  — the results contain enough relevant candidates but there are too 
many or they are poorly ordered for this query. A re-ranking pass would help 
surface the best ones. Only choose this when the strategy was "structured_filter" 
or "hybrid_search"; semantic results are already similarity-ranked, so reranking 
them rarely helps.

# How to judge "sufficient"
There is NO fixed minimum or maximum number of results. Judge based on the query:
- A specific lookup ("Who works at Stripe?") may need only 1–3 hits to be answerable.
- A broad strategic query ("Who can introduce me to Series A investors?") usually
needs a wider, diverse pool (5–15+) to allow good synthesis.
- Empty or near-empty results almost always mean "relax" unless every filter
has already been dropped — in that case, finish and let synthesis explain the gap.
- Very large result sets (dozens+) for a focused query usually mean "rerank".

# How to relax
If you choose "relax", you MUST propose at least one concrete change:
- `drop_filters`: list of filter names to remove. Pick filters that are most
likely overly restrictive given the user's intent. Do NOT drop filters that are
the core of the query (e.g. don't drop "current_company_name" if the user
specifically asked about a company).
- `switch_strategy_to`: optionally switch the retrieval strategy. Use
"semantic_search" when filters can't capture the user's intent (fuzzy or
conceptual queries). Use "hybrid_search" when you want both filtering and
similarity ranking. Leave null to keep the current strategy.

You may do both (drop filters AND switch strategy) in the same step.

Do NOT propose dropping a filter that is already in `relaxed_fields` — it has
already been removed.

Do NOT propose "relax" with an empty `drop_filters` and no `switch_strategy_to`
— that would be a no-op. In that case, choose "finish" instead.

# Inputs you receive
- original_query:   the user's natural-language question.
- current_strategy: the retrieval strategy that produced the current results.
- active_filters:   filters currently in effect (non-null only).
- relaxed_fields:   filters already dropped in earlier iterations.
- attempts:         how many evaluation rounds have already happened.
- results_summary:  compact view of retrieved profiles (name, headline, owners).
This is what synthesis will work with.

# Output
Return a structured decision matching the EvaluatorDecision schema:
- action:               "finish" | "relax" | "rerank"
- drop_filters:         list of filter names (only if action == "relax")
- switch_strategy_to:   one of the three strategies, or null
- reasoning:            one or two sentences explaining your choice. Be specific:
reference the query, the result count, and why the chosen action helps.

# Important rules
1. Prefer "finish" when results plausibly answer the query, even if imperfect —
synthesis can still produce a useful response, and looping is expensive.
2. Never propose changes that would have no effect (no-op relax, reranking
already similarity-ranked results, dropping already-dropped filters).
3. The "owners_all" filter represents which team members know the connection.
Almost never drop this — it is a hard constraint of the user's question, not a
preference.
4. Be honest in `reasoning`. If you are unsure, say so and lean toward "finish".
"""

class Decision(BaseModel):
    action: Literal["finish", "relax", "rerank"]
    reasoning: str
    drop_filters: list[str] = Field(description="Filters that were dropped due to the reasoning both in this iteration and earlier ones. Use only when action is 'relax'.")
    switch_strategy_to: Literal["structured_filter", "semantic_search", "hybrid_search", ""]


class RetrievalState(TypedDict):
    retrieval_type: Literal["structured_filter", "semantic_search", "hybrid_search"]
    filters: dict
    query_text: str
    user_input: str
    results: list[dict]
    attempts: int
    relaxed_filters: list[str]
    decision: Optional[Decision]
    decision_log: list[dict]


llm = get_llm()
evaluator = llm.with_structured_output(Decision)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def route_retrieval_type(state: RetrievalState) -> str:
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

def llm_evaluation(state: RetrievalState) -> dict:
    if state["attempts"] >= MAX_ATTEMPTS:
        return {"decision": Decision(action="finish", reasoning="No more attempts left.")}

    additional_info = {
        "user_question": state["user_input"],
        "number_of_results": len(state["results"]),
        "active_filters": {k: v for k, v in state["filters"].items() if v is not None},
        "relaxed_fields": state["relaxed_filters"],
        "attempts": state["attempts"],
    }
    decision = evaluator.invoke([
        SystemMessage(content=EVALUATOR_PROMPT),
        HumanMessage(content=f"""
        User question: {additional_info["user_question"]}
        Number of results: {additional_info["number_of_results"]}
        Active filters: {additional_info["active_filters"]}
        Relaxed fields: {additional_info["relaxed_fields"]}
        Attempt number: {additional_info["attempts"]}
        """),
    ])

    return {
        "decision": decision,
        "relaxed_filters": state["relaxed_filters"] + decision.drop_filters,
        "decision_log": state.get("decision_log", []) + [decision.model_dump()],
    }

def route_next_action(state: RetrievalState) -> Literal["finish", "relax", "rerank"]:
    return state["decision"].action

def relaxation_node(state: RetrievalState) -> dict:
    active_filters = dict(state["filters"])
    for f in state["relaxed_filters"]:
        if f in active_filters:
            active_filters.pop(f)

    return {"retrieval_type": state["decision"].switch_strategy_to or state["retrieval_type"], "filters": active_filters, "attempts": state["attempts"] + 1}

def reranking_node(state: RetrievalState) -> dict:
    results = {r.get("linkedin_url"): r for r in state["results"]}
    result_ids = list(results.keys())
    query_text = state["query_text"]
    scores = cross_encoder.predict([(query_text, results[rid].get("headline", "") + "\\n" + results[rid].get("summary", "")) for rid in result_ids])

    ranked = sorted(zip(result_ids, scores), key=lambda x: x[1], reverse=True)

    top_k = state["filters"].get("top_k", 10)

    results = [results[rid] for rid, _ in ranked[:top_k]]

    return {"results": results}

graph = StateGraph(RetrievalState)
graph.add_node("structured_retrieval", structured_retrieval)
graph.add_node("semantic_retrieval", semantic_retrieval)
graph.add_node("hybrid_retrieval", hybrid_retrieval)
graph.add_node("llm_evaluation", llm_evaluation)
graph.add_node("relaxation_node", relaxation_node)
graph.add_node("reranking_node", reranking_node)


graph.add_conditional_edges(START, route_retrieval_type, {"structured_filter": "structured_retrieval", "semantic_search": "semantic_retrieval", "hybrid_search": "hybrid_retrieval"})
graph.add_edge("semantic_retrieval", "llm_evaluation")
graph.add_edge("structured_retrieval", "llm_evaluation")
graph.add_edge("hybrid_retrieval", "llm_evaluation")
graph.add_conditional_edges("llm_evaluation", route_next_action, {"finish": END, "relax": "relaxation_node", "rerank": "reranking_node"})
graph.add_conditional_edges("relaxation_node", route_retrieval_type, {"structured_filter": "structured_retrieval", "semantic_search": "semantic_retrieval", "hybrid_search": "hybrid_retrieval"})
graph.add_edge("reranking_node", END)

retrieval_strategy = graph.compile()

def retrieve(specification: UserQuery, user_input: str) -> list[dict]:
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

    if rt in ["semantic_search", "hybrid_search"] and not specification.contextual_need: #guard
        rt = "structured_filter"

    spec = {
        "retrieval_type": rt,
        "filters": filters,
        "user_input": user_input,
        "query_text": specification.contextual_need if specification.contextual_need else user_input,
        "attempts": 0,
        "relaxed_filters": [],
        "decision_log": []
    }
    result = retrieval_strategy.invoke(spec)
    return result["results"]