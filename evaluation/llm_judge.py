import json
import os
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import START
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.config import LLM_MODEL
from src.llm.client import get_llm
from src.agents.workflow import build_network_intelligence_workflow

VALIDATION_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_validation_set.json")

OVERALL_PASS_THRESHOLD = 0.8

class CriterionJudgement(BaseModel):
    criterion: Literal["groundedness", "completeness", "citation_owner_accuracy", "recommendation_quality"]
    score: float = Field(ge=0.0, le=1.0, description="0.0 means severe failure, 1.0 means the criterion is fully satisfied.",)
    passed: bool = Field(description="True if this criterion is acceptable for production use.")
    reasoning: str = Field(description="Brief explanation of the score.")


class JudgeState(TypedDict, total=False):
    query: str
    profiles: list[dict]
    answer: str
    groundedness: CriterionJudgement
    completeness: CriterionJudgement
    citation_owner_accuracy: CriterionJudgement
    recommendation_quality: CriterionJudgement
    overall_score: float
    passed: bool

COMMON_JUDGE_CONTEXT = """
You are judging the output of the synthesis agent in a LinkedIn network search
assistant.

The synthesis agent receives:
- the user's original query
- retrieved LinkedIn profile data
- connection owner metadata

Judge only the final answer. Use the retrieved profiles as the only source of
truth. Do not reward facts that are plausible but absent from the profile data.
Return a structured judgement matching the requested schema.
"""

GROUNDEDNESS_PROMPT = f"""
{COMMON_JUDGE_CONTEXT}

Criterion: groundedness.

Evaluate whether every factual claim in the answer is supported by the supplied
profile data. Check claims about names, companies, roles, skills, locations,
industries, relevance, connection strength, and missing information.

Scoring guide:
- 1.0: all factual claims are directly supported.
- 0.7: mostly grounded, with minor vague claims or weakly supported phrasing.
- 0.4: several unsupported claims, but the answer is still partially based on data.
- 0.0: answer substantially invents facts or contradicts the profiles.

Set criterion to "groundedness".
"""

COMPLETENESS_PROMPT = f"""
{COMMON_JUDGE_CONTEXT}

Criterion: completeness.

Evaluate whether the answer satisfies the user's query given the retrieved
profiles and the synthesis instructions:
- starts by stating the total number of relevant profiles found
- gives a strategic recommendation, not a raw profile dump
- highlights the strongest 2-4 relevant connections when available
- explains why those connections are strong enough for the user's need
- explicitly includes a Missing section only when something important was not
  found or could not be answered from the profiles

Scoring guide:
- 1.0: fully answers the query and handles missing information correctly.
- 0.7: useful answer with a small omission.
- 0.4: partially answers but misses important requested details.
- 0.0: does not answer the user's query.

Set criterion to "completeness".
"""

CITATION_OWNER_PROMPT = f"""
{COMMON_JUDGE_CONTEXT}

Criterion: citation_owner_accuracy.

Evaluate whether cited people and connection owners are correct:
- every mentioned profile should use this exact format:
  <a href="FULL_LINKEDIN_URL">Full Name</a> (connection: Owner)
- LinkedIn URLs must exactly match the supplied profile data
- owner names must exactly match the supplied profile owners
- if a profile has multiple owners, the answer should explicitly mention them
- the answer must not invent owners, shorten URLs, or cite a person not present
  in the retrieved profiles

Do not penalize generic text that does not mention a person.

Scoring guide:
- 1.0: all citations and owners are exact.
- 0.7: minor formatting issue, but URLs and owners are recoverable.
- 0.4: multiple citation/owner mistakes.
- 0.0: citations or owners are mostly fabricated or missing.

Set criterion to "citation_owner_accuracy".
"""

RECOMMENDATION_QUALITY_PROMPT = f"""
{COMMON_JUDGE_CONTEXT}

Criterion: recommendation_quality.

Evaluate whether the answer is a concise, useful strategic recommendation:
- prioritizes the strongest people instead of listing everyone
- gives clear reasons for the recommendation
- treats multiple owners as a stronger connection when relevant
- avoids filler, unsupported confidence, and excessive length
- stays within the requested 3-5 sentence style unless a short Missing section
  is necessary

Scoring guide:
- 1.0: concise, actionable, and well-prioritized.
- 0.7: useful but slightly verbose, generic, or under-prioritized.
- 0.4: mostly a list or too vague to guide the user.
- 0.0: unusable recommendation.

Set criterion to "recommendation_quality".
"""

llm = get_llm(LLM_MODEL)
judge = llm.with_structured_output(CriterionJudgement)


def context_builder(state: JudgeState) -> str:
    return f"""
User query:
{state["query"]}

Retrieved profiles:
{state["profiles"]}

Final synthesis answer:
{state["answer"]}
"""


def judge_groundedness(state: JudgeState) -> dict:
    judgement = judge.invoke([
        SystemMessage(content=GROUNDEDNESS_PROMPT),
        HumanMessage(content=context_builder(state))
    ])

    return {"groundedness": judgement}

def judge_completeness(state: JudgeState) -> dict:
    judgement = judge.invoke([
        SystemMessage(content=COMPLETENESS_PROMPT),
        HumanMessage(content=context_builder(state))
    ])

    return {"completeness": judgement}

def judge_citation_owner_accuracy(state: JudgeState) -> dict:
    judgement = judge.invoke([
        SystemMessage(content=CITATION_OWNER_PROMPT),
        HumanMessage(content=context_builder(state))
    ])

    return {"citation_owner_accuracy": judgement}

def judge_recommendation_quality(state: JudgeState) -> dict:
    judgement = judge.invoke([
        SystemMessage(content=RECOMMENDATION_QUALITY_PROMPT),
        HumanMessage(content=context_builder(state))
    ])

    return {"recommendation_quality": judgement}


def final_judgement(state: JudgeState) -> dict:
    criterions = ["groundedness", "completeness", "citation_owner_accuracy", "recommendation_quality"]

    overall_score = sum([state[f"{criterion}"].score for criterion in criterions]) / len(criterions)
    passed = overall_score >= OVERALL_PASS_THRESHOLD

    return {"overall_score": overall_score, "passed": passed}


graph = StateGraph(JudgeState)
graph.add_node("judge_groundedness", judge_groundedness)
graph.add_node("judge_completeness", judge_completeness)
graph.add_node("judge_citation_owner_accuracy", judge_citation_owner_accuracy)
graph.add_node("judge_recommendation_quality", judge_recommendation_quality)
graph.add_node("final_judgement", final_judgement)

graph.add_edge(START, "judge_groundedness")
graph.add_edge(START, "judge_completeness")
graph.add_edge(START, "judge_citation_owner_accuracy")
graph.add_edge(START, "judge_recommendation_quality")
graph.add_edge("judge_groundedness", "final_judgement")
graph.add_edge("judge_completeness", "final_judgement")
graph.add_edge("judge_citation_owner_accuracy", "final_judgement")
graph.add_edge("judge_recommendation_quality", "final_judgement")
graph.add_edge("final_judgement", END)

answer_judge = graph.compile()

def judge_answer(query: str, answer: str, profiles: list[dict]):
    return answer_judge.invoke({"query": query, "answer": answer, "profiles": profiles})


def run_eval():
    workflow = build_network_intelligence_workflow()
    with open(VALIDATION_SET_PATH, 'r', encoding='utf-8') as f:
        validation_set = json.load(f)
        results = []
        for example in validation_set:
            query = example["user_input"]
            output = workflow.invoke({"user_input": query, "conversation_history": []})

            profiles = output["profiles_data"]
            answer = output["answer"]
            results.append(judge_answer(query, answer, profiles))

        for r in results:
            print(f"Query: {r['query']}")
            print(f"Answer: {r['answer']}")
            print(f"Groundedness score: {r['groundedness'].score:.2f} ({r['groundedness'].passed})")
            print(f"Groundedness reasoning: {r['groundedness'].reasoning}")
            print(f"Completeness score: {r['completeness'].score:.2f} ({r['completeness'].passed})")
            print(f"Completeness reasoning: {r['completeness'].reasoning}")
            print(f"Citation/owner accuracy score: {r['citation_owner_accuracy'].score:.2f} ({r['citation_owner_accuracy'].passed})")
            print(f"Citation/owner accuracy reasoning: {r['citation_owner_accuracy'].reasoning}")
            print(f"Recommendation quality score: {r['recommendation_quality'].score:.2f} ({r['recommendation_quality'].passed})")
            print(f"Recommendation quality reasoning: {r['recommendation_quality'].reasoning}")
            print(f"Overall score: {r['overall_score']}")
            print(f"Passed: {r['passed']}")
            print()


if __name__ == "__main__":
    run_eval()