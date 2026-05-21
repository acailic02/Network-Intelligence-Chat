import copy
import json
import os
from typing import TypedDict, Annotated, Literal, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from src.agents.query_understanding import UserQuery, LookupAttributes, QueryType
from src.tools.tools import structured_filter, semantic_search, count_matches, hybrid_search, Prof
from src.llm.client import get_llm
from src.config import LOGS_DIR

MAX_ATTEMPTS = 5

EVALUATOR_PROMPT = """
You are the Retrieval Evaluator in a network intelligence assistant. The system 
searches a combined LinkedIn network of a small team to answer the user's question. 
Your job is to look at the current retrieval results and decide what to do next.

# Your task
Decide ONE of three actions:
- "finish"  — the current results are sufficient to answer the user's query well. 
The synthesis step will use them as-is.
- "relax" means: structured filters are preventing retrieval of relevant results,
  not that the result count is low. 
You must decide how to broaden the search: drop one or more filters, and/or 
switch retrieval strategy.
- "rerank"  — the results contain enough relevant candidates but there are too 
many or they are poorly ordered for this query. A re-ranking pass would help 
surface the best ones. Only choose this when the strategy was "structured_filter" 
or "hybrid_search"; semantic results are already similarity-ranked


# How to relax
If you choose "relax", you MUST propose at least one concrete change:
- `drop_filters`: list of filter names to remove. Pick filters that are most
likely overly restrictive given the user's intent. Do NOT drop filters that are
the core of the query (e.g. don't drop "current_company_name" if the user
specifically asked about a company). Be very careful to preserve users intent, if you cant dont drop any filters.
- `switch_strategy_to`: optionally switch the retrieval strategy. Use
"semantic_search" when filters can't capture the user's intent (fuzzy or
conceptual queries). Use "hybrid_search" when you want both filtering and
similarity ranking. Use "no_change" to keep the current strategy. 
(e.g. if user asks for people working in an area/field that is not exact position (can't be captured by structured filters, e.g. "game developement", "machine learning" etc.), and current strategy is 
structured_filter, you should switch to either hybrid_search or semantic_search)

You may do both (drop filters AND switch strategy) in the same step.

If you cant preserve users intent just switch to another strategy without dropping any filters.
DONT switch strategies if users question is easily captured with structured filters.
DONT drop filters if by doing so the user's intent would be lost.

Do NOT propose dropping a filter that is already in `relaxed_fields` — it has
already been removed.

Do NOT propose "relax" with an empty `drop_filters` 

# Inputs you receive
- original_query:   the user's natural-language question.
- current_strategy: the retrieval strategy that produced the current results.
- active_filters:   filters currently in effect (non-null only).
- relaxed_filters:   filters already dropped in earlier iterations.
- attempts:         how many evaluation rounds have already happened.
- results:          retrieval results (dicts with information about each profile).

# Output
Return a structured decision matching the EvaluatorDecision schema:
- action:               "finish" | "relax" | "rerank"
- drop_filters:         list of filter names (only if action == "relax", empty list if no drop_filters)
- switch_strategy_to:   one of the three strategies, or "no_change" if no change
- reasoning:            one or two sentences explaining your choice. Be specific:
reference the query, the results, and why the chosen action helps.

# Important rules
1. Prefer "finish" when results plausibly answer the query, even if imperfect —
synthesis can still produce a useful response, and looping is expensive.
2. Never propose changes that would have no effect (no-op relax, reranking
already similarity-ranked results, dropping already-dropped filters).
3. The "owners_all" filter represents which team members know the connection.
Almost never drop this — it is a hard constraint of the user's question, not a
preference.
4. If you are unsure, say so and lean toward "finish".
4b. A small number of highly relevant structured results is sufficient.
Do NOT switch away from structured retrieval solely because the result count is low.

If results match the active filters and satisfy the user's explicit constraints,
prefer "finish" even if the result set is small (e.g. 1–10 results).

5. Never recommend "semantic_search" if structured_filter already returned
results that satisfy all active_filters.
Semantic search is ONLY for cases where structured filters fail to capture intent.

6.You should only switch strategie to "semantic_search" if you dropped already dropped all filters.
Otherwise you should switch to "hybrid_search" to cover more cases.
"""

RELEVANCE_EVAL_PROMPT = """
You are a Relevance Evaluator for a network intelligence assistant that searches 
LinkedIn profiles across a team's combined network to answer user questions.

# Task
Evaluate how relevant each retrieved profile is for answering the user's question.
Score each profile from 0.0 to 1.0:
- 1.0 = Highly relevant, directly helps answer the question
- 0.5-0.9 = Partially relevant, provides useful context
- 0.0-0.4 = Low relevance, minimal help

# What makes a profile relevant?
Consider ONLY the following factors:
    - Job title/role matches the query domain
    - Skills/expertise align with what the user is asking
    - Experience/industry is directly related
    - Location matters (if query is geography-specific)
    - Company/education background is pertinent

Connection constraints should have much lower impact on relevance score than above mentioned factors

# Guidelines
- Be strict: only high scores (>0.7) for profiles that DIRECTLY help answer the question
- Profiles tangentially related (adjacent domain but no explicit mention of relevant keywords/experience) should score 0.3-0.7
- Empty/incomplete profiles should score low (<0.3)
- Provide clear reasoning in relevance_summary (1-2 sentences explaining WHY)

# Inputs
- user_question: the user's natural-language question
- retrieved_profiles: list of profile dictionaries, each containing:
  * name, headline, summary
  * current company, job title
  * skills, location, etc.

# Output
For each profile, provide:
- id: THE EXACT id from input (DONT CHANGE).
- relevance_score: float between 0.0 and 1.0
- relevance_summary: brief explanation (1-2 sentences)

# Scoring formula
For each profile:
1. Identify all user constraints.
2. Mark each constraint as critical, high, medium, or low based on the wording.
3. Mark each constraint as satisfied, partially_satisfied, or missing using only profile data.
4. Score by weighted coverage:
   - critical = 3
   - high = 2
   - medium = 1
   - low = 0.5
   satisfied = full weight
   partially_satisfied = half weight
   missing = 0
5. For AND-style queries, never score above 0.70 if any critical constraint is missing.
6. For AND-style queries, never score above 0.85 unless all critical/high constraints are satisfied.
7. For OR-style queries, a profile can score highly if it satisfies at least one major branch well.

# Examples by number of constraints

## 2 constraints

user_question: "Find connections working at Microsoft"
profile: { current_company: "Microsoft", location: "Belgrade" }
score: 0.95 | "Works at Microsoft, single constraint fully satisfied."

profile: { current_company: "Google", positions: [{ company: "Microsoft" }] }
score: 0.35 | "Previously at Microsoft but no longer there. Main constraint partially satisfied."

profile: { current_company: "Amazon", location: "Belgrade" }
score: 0.05 | "Never worked at Microsoft. Constraint not satisfied."

---

user_question: "Find QA engineers in our network"
profile: { current_job_title: "QA Engineer", current_company: "Nordeus" }
score: 0.95 | "QA Engineer title directly satisfies the role constraint."

profile: { current_job_title: "QA Automation Engineer", skills: ["Selenium", "TestNG"] }
score: 0.75 | "Automation variant of QA with relevant skills — intent matched."

profile: { current_job_title: "Software Engineer", skills: ["testing", "Jest"] }
score: 0.3 | "Testing skills present but no QA title — tangentially related."

---

## 3 constraints

user_question: "Find data engineers from Croatia not Jelenas connections"
profile: { current_job_title: "Data Engineer", location: "Zagreb, Croatia", owners: ["Petar"] }
score: 0.9 | "Data Engineer from Croatia, not Jelena's connection. All three constraints satisfied."

profile: { current_job_title: "Data Engineer", location: "Zagreb, Croatia", owners: ["Jelena"] }
score: 0.8 | "Data Engineer from Croatia — two main constraints satisfied. Jelena ownership is minor negative."

profile: { current_job_title: "Data Engineer", location: "Belgrade, Serbia", owners: ["Petar"] }
score: 0.6 | "Data Engineer role satisfied but not from Croatia. Owner constraint satisfied. One significant miss."

profile: { current_job_title: "Data Analyst", location: "Zagreb, Croatia", owners: ["Petar"] }
score: 0.55 | "From Croatia, not Jelena's connection, but Data Analyst not Data Engineer — role is adjacent not exact."

profile: { current_job_title: "Backend Engineer", location: "Belgrade, Serbia", owners: ["Jelena"] }
score: 0.1 | "None of the three constraints satisfied."

---

user_question: "Find people working at Google and are not from Serbia"
profile: { current_company: "Google", location: "Dublin, Ireland", education: [{ school: "Trinity College Dublin" }] }
score: 0.95 | "Works at Google, Irish location and education confirm non-Serbian origin. Both constraints satisfied."

profile: { current_company: "Google", location: "Belgrade, Serbia", education: [{ school: "University of Belgrade" }] }
score: 0.45 | "Works at Google which is the primary constraint, but location and education both confirm Serbian origin. One of two constraints violated."

profile: { current_company: "Google", location: "Zurich, Switzerland", education: [{ school: "University of Belgrade" }] }
score: 0.7 | "Works at Google and lives in Switzerland, but Belgrade education is a soft signal of Serbian origin — nationality uncertain, main constraint satisfied."

profile: { current_company: "Nordeus", location: "Dublin, Ireland" }
score: 0.25 | "Not from Serbia satisfies nationality constraint but does not work at Google — primary constraint not satisfied."

---

## 4 constraints

user_question: "Find backend engineers in London not at Amazon, not Jelenas connections"
profile: { current_job_title: "Backend Engineer", location: "London, UK", current_company: "Spotify", owners: ["Petar"] }
score: 0.95 | "Backend Engineer in London, not at Amazon, not Jelena's connection. All four constraints satisfied."

profile: { current_job_title: "Backend Engineer", location: "London, UK", current_company: "Amazon", owners: ["Petar"] }
score: 0.65 | "Backend Engineer in London, not Jelena's — three of four satisfied. Works at Amazon which violates company exclusion."

profile: { current_job_title: "Backend Engineer", location: "London, UK", current_company: "Spotify", owners: ["Jelena"] }
score: 0.85 | "Backend Engineer in London, not at Amazon — three of four satisfied. Jelena ownership is minor negative."

profile: { current_job_title: "Backend Engineer", location: "Berlin, Germany", current_company: "Amazon", owners: ["Jelena"] }
score: 0.25 | "Role constraint satisfied but wrong city, wrong company, and Jelena connection — three of four violated."

profile: { current_job_title: "Frontend Engineer", location: "Berlin, Germany", current_company: "Amazon", owners: ["Jelena"] }
score: 0.05 | "None of the four constraints satisfied."

---

user_question: "Find people from Novi Sad who worked at Levi9 and know Python, connected to Aleksandar"
profile: { location: "Novi Sad", positions: [{ company: "Levi9" }], skills: ["Python", "Django"], owners: ["Aleksandar"] }
score: 0.95 | "From Novi Sad, Levi9 experience, Python skills, Aleksandar's connection. All four satisfied."

profile: { location: "Novi Sad", positions: [{ company: "Levi9" }], skills: ["Java", "Spring"], owners: ["Aleksandar"] }
score: 0.7 | "Novi Sad, Levi9, Aleksandar's connection satisfied — but no Python skills. Three of four constraints met."

profile: { location: "Belgrade", positions: [{ company: "Levi9" }], skills: ["Python"], owners: ["Aleksandar"] }
score: 0.65 | "Levi9 experience, Python, Aleksandar's connection — but from Belgrade not Novi Sad. Three of four satisfied."

profile: { location: "Novi Sad", positions: [{ company: "Nordeus" }], skills: ["Python"], owners: ["Mihajlo"] }
score: 0.35 | "From Novi Sad and knows Python, but no Levi9 experience and not Aleksandar's connection. Two of four satisfied."

profile: { location: "Belgrade", positions: [{ company: "Nordeus" }], skills: ["Java"], owners: ["Jelena"] }
score: 0.05 | "None of the four constraints satisfied."

---

## Past experience vs current (special case)

user_question: "Find people who worked at Nutanix and FIS"
profile: { positions: [{ company: "Nutanix" }, { company: "FIS" }], current_company: "Google" }
score: 0.95 | "Has experience at both Nutanix and FIS — AND constraint fully satisfied regardless of current employer."

profile: { positions: [{ company: "Nutanix" }], current_company: "FIS" }
score: 0.95 | "Nutanix in past experience, currently at FIS — both companies present, AND satisfied."

profile: { positions: [{ company: "Nutanix" }], current_company: "Google" }
score: 0.3 | "Only Nutanix found, FIS missing entirely. AND constraint requires both companies."

profile: { current_company: "Nutanix", positions: [{ company: "IBM" }] }
score: 0.3 | "Currently at Nutanix but no FIS experience. AND constraint not satisfied."
"""

class Decision(BaseModel):
    action: Literal["finish", "relax", "rerank"]
    reasoning: str
    drop_filters: list[str] = Field(description="Filters that were dropped due to the reasoning both in this iteration and earlier ones. Use only when action is 'relax'.")
    switch_strategy_to: Literal["structured_filter", "semantic_search", "hybrid_search", "no_change"]

class ConstraintScore(BaseModel):
    constraint: str
    importance: Literal["critical", "high", "medium", "low"]
    match: Literal["satisfied", "partial", "missing"]
    evidence: str | None

class EvaluatedProf(BaseModel):
    id: int = Field(description="Profile id (from the input).")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score of the profile.")
    relevance_summary: str = Field(description="Short (1-2 sentence) summary of why the profile is relevant to the query.")
    constraint_scores: list[ConstraintScore]

class RelevanceEvaluation(BaseModel):
    evaluated_results: list[EvaluatedProf]

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
retrieval_evaluator = llm.with_structured_output(Decision)
relevance_evaluator = llm.with_structured_output(RelevanceEvaluation)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_filters(state: RetrievalState):
    return {k: v for k, v in state["filters"].items() if v is not None}

def route_retrieval_type(state: RetrievalState) -> str:
    return state["retrieval_type"]

def structured_retrieval(state: RetrievalState) -> dict:
    print("======================================================================================STRUCTURED RETRIEVAL=======================================================================================")
    print(state["filters"])
    results = structured_filter.invoke(get_filters(state))
    return {"results": results}

def semantic_retrieval(state: RetrievalState) -> dict:
    print("======================================================================================SEMANTIC RETRIEVAL=======================================================================================")
    results = semantic_search.invoke({"query": state["query_text"], "filters": state["filters"]})
    return {"results": results}

def hybrid_retrieval(state: RetrievalState) -> dict:
    print("======================================================================================HYBRID RETRIEVAL=======================================================================================")
    print(state["filters"])
    kwargs = {
        "filters": get_filters(state),
        "query_text": state["query_text"]
    }
    results = hybrid_search.invoke(kwargs)
    return {"results": results}

def retrieval_evaluation(state: RetrievalState) -> dict:
    if state["attempts"] >= MAX_ATTEMPTS:
        return {"decision": Decision(action="finish", reasoning="No more attempts left.")}

    additional_info = {
        "user_question": state["user_input"],
        "retrieval_results": state["results"],
        "active_filters": get_filters(state),
        "relaxed_filters": state["relaxed_filters"],
        "attempts": state["attempts"],
    }
    decision = retrieval_evaluator.invoke([
        SystemMessage(content=EVALUATOR_PROMPT),
        HumanMessage(content=f"""
        User question: {additional_info["user_question"]}
        Retrieval results: {additional_info["retrieval_results"]}
        Active filters: {additional_info["active_filters"]}
        Relaxed filters: {additional_info["relaxed_filters"]}
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

    if state["decision"].switch_strategy_to != "no_change":
        new_strategy = state["decision"].switch_strategy_to
    else:
        new_strategy = state["retrieval_type"]

    return {"retrieval_type": new_strategy , "filters": active_filters, "attempts": state["attempts"] + 1}

def reranking_node(state: RetrievalState) -> dict:
    results = {r.get("linkedin_url"): r for r in state["results"]}
    result_ids = list(results.keys())
    query_text = state["query_text"]
    scores = cross_encoder.predict([(query_text, (results[rid].get("headline", "") or "") + "\\n" + (results[rid].get("summary", "") or "")) for rid in result_ids])

    ranked = sorted(zip(result_ids, scores), key=lambda x: x[1], reverse=True)

    top_k = state["filters"].get("top_k", 10)

    results = [results[rid] for rid, _ in ranked[:top_k]]

    return {"results": results}

def relevance_evaluation(state: RetrievalState) -> dict:
    results = copy.deepcopy(state["results"])

    for i, r in enumerate(results):
        r["id"] = i
        r["name"] = f"placeholder_name_{i}"
        r["linkedin_url"] = f"placeholder_linkedin_url_{i}"

    query_text = state["query_text"]
    evaluated = relevance_evaluator.invoke([
        SystemMessage(content=RELEVANCE_EVAL_PROMPT),
        HumanMessage(content=f"""
        User question: {query_text}
        Retrieval results: {results}
        """)
    ])

    results_by_id = {r["id"]: r for r in results}
    enriched = [
        {
            **state["results"][ev.id],
            "relevance_score": ev.relevance_score,
            "relevance_summary": ev.relevance_summary
        }
        for ev in evaluated.evaluated_results
        if ev.id in results_by_id
    ]

    return {"results": enriched}

def irrelevant_profiles_drop_off(state: RetrievalState) -> dict:
    results = [result for result in state["results"] if result.get("relevance_score", 0.0) > 0.15]
    return {"results": results}

def final_profile_sorting(state: RetrievalState) -> dict:
    results = state["results"]
    return {"results": sorted(results, key=lambda x: x.get("relevance_score", 0.0), reverse=True)}

def log_retrieval(state: RetrievalState) -> dict:
    final_retrieval = {
        "retrieval_type": state["retrieval_type"],
        "filters": state["filters"],
        "query_text": state["query_text"],
        "user_input": state["user_input"],
    }

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "final_retrieval_per_query.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(final_retrieval) + "\n")

    return {}


graph = StateGraph(RetrievalState)
graph.add_node("structured_retrieval", structured_retrieval)
graph.add_node("semantic_retrieval", semantic_retrieval)
graph.add_node("hybrid_retrieval", hybrid_retrieval)
graph.add_node("retrieval_evaluation", retrieval_evaluation)
graph.add_node("relaxation_node", relaxation_node)
graph.add_node("reranking_node", reranking_node)
graph.add_node("relevance_evaluation", relevance_evaluation)
graph.add_node("irrelevant_profiles_drop_off", irrelevant_profiles_drop_off)
graph.add_node("final_profile_sorting", final_profile_sorting)
graph.add_node("log_retrieval", log_retrieval)


graph.add_conditional_edges(START, route_retrieval_type, {"structured_filter": "structured_retrieval", "semantic_search": "semantic_retrieval", "hybrid_search": "hybrid_retrieval"})
graph.add_edge("semantic_retrieval", "retrieval_evaluation")
graph.add_edge("structured_retrieval", "retrieval_evaluation")
graph.add_edge("hybrid_retrieval", "retrieval_evaluation")
graph.add_conditional_edges("retrieval_evaluation", route_next_action, {"finish": "relevance_evaluation", "relax": "relaxation_node", "rerank": "reranking_node"})
graph.add_conditional_edges("relaxation_node", route_retrieval_type, {"structured_filter": "structured_retrieval", "semantic_search": "semantic_retrieval", "hybrid_search": "hybrid_retrieval"})
graph.add_edge("reranking_node", "relevance_evaluation")
graph.add_edge("relevance_evaluation", "irrelevant_profiles_drop_off")
graph.add_edge("irrelevant_profiles_drop_off", "final_profile_sorting")
graph.add_edge("final_profile_sorting", "log_retrieval")
graph.add_edge("log_retrieval", END)

retrieval_strategy = graph.compile()

def retrieve(specification: UserQuery, user_input: str) -> list[dict]:
    position = specification.lookup_filters.position
    connection = specification.lookup_filters.connection
    education = specification.lookup_filters.education
    filters = {
        "country": connection.country if connection is not None else None,
        "exclude_country": connection.exc_country if connection is not None else None,
        "city": connection.city if connection is not None else None,
        "exclude_city": connection.exc_city if connection is not None else None,
        "skills": connection.skills if connection is not None else None,
        "skills_operator": connection.skills_operator if connection is not None else None,
        "exclude_skills": connection.exc_skills if connection is not None else None,
        "owners": specification.lookup_filters.owner if specification.lookup_filters.owner is not None else None,
        "owners_operator": specification.lookup_filters.owner_operator if specification.lookup_filters.owner_operator is not None else None,
        "exclude_owners": specification.lookup_filters.exc_owner if specification.lookup_filters.exc_owner is not None else None,
        "current_company_name": connection.current_company_name if connection is not None else None,
        "current_job_title": connection.current_job_title if connection is not None else None,
        "company_location": position.company_location if position is not None else None,
        "company_name": position.company_name if position is not None else None,
        "exclude_company_name": position.exc_company_name if position is not None else None,
        "company_name_operator": position.company_name_operator if position is not None else None,
        "school_name": education.school_name if education is not None else None,
        "school_name_operator": education.school_name_operator if education is not None else None,
        "exclude_school_name": education.exc_school_name if education is not None else None,
        "degree": education.degree if education is not None else None,
        "degree_operator": education.degree_operator if education is not None else None,
        "exclude_degree": education.exc_degree if education is not None else None,
        "job_title": position.title if position is not None else None,
        "job_title_operator": position.title_operator if position is not None else None,
        "exclude_job_title": position.exc_title if position is not None else None,
        "limit": specification.limit,
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