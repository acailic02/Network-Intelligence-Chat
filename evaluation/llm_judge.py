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

Important product context:
- The UI always shows the full retrieved profile list separately in an expander.
- The final answer is only a concise summary/recommendation above that list.
- Do not penalize the answer for not enumerating every retrieved profile.
- For queries like "give me all" or "list all", it is acceptable for the answer
  to state the count and highlight the strongest matches, because the full list
  is visible in the UI.

The synthesis agent receives:
- the user's original query
- retrieved LinkedIn profile data
- connection owner metadata

Important owner semantics:
- The owners field is authoritative LinkedIn connection metadata.
- "Petar's connection" means a profile whose owners list contains "Petar".
- "not Petar's connection" means the owners list does not contain "Petar".
- Do not confuse a person's first name with a connection owner.
  Example: a person named Petar with owners ["Mihajlo"] is NOT Petar's connection.
- If owners are ["Petar"], it is correct to cite the person as
  (connection: Petar), and it is correct to say they are not Aleksandar's connection.

Scoring guide:
- 0.90-1.00: Excellent. Fully satisfies the criterion, with no issue or only
  trivial wording/style issues.
- 0.75-0.89: Good. Satisfies the criterion for production use, with minor
  omissions, mild ambiguity, or small unsupported phrasing that does not change
  the answer materially.
- 0.50-0.74: Mixed. Partially satisfies the criterion, but has noticeable issues
  that reduce reliability or usefulness.
- 0.25-0.49: Poor. Major issues are present, though some parts are still based
  on the data.
- 0.00-0.24: Failing. Mostly incorrect, fabricated, contradictory, or unusable
  for this criterion.

Use the full range, not just the boundary values.
Set passed=true when score >= 0.75.
Set passed=false when score < 0.75.

Judge only the final answer, using the retrieved profiles as the source of truth.
Do not reward facts that are plausible but absent from the profile data.
"""

GROUNDEDNESS_PROMPT = f"""
{COMMON_JUDGE_CONTEXT}

Criterion: groundedness.

Evaluate whether every factual claim in the answer is supported by the supplied
profile data. Check claims about names, companies, roles, skills, locations,
industries, relevance, connection strength, and missing information.

Connection owner metadata is authoritative. If a user asks for people who are
not connected to Petar, and a profile's owners list does not include Petar,
then the answer may state that the person is not Petar's connection. This is
grounded in the supplied owner metadata.

Set criterion to "groundedness".
"""

COMPLETENESS_PROMPT = f"""
{COMMON_JUDGE_CONTEXT}

Criterion: completeness.

Evaluate whether the answer satisfies the user's query given the retrieved
profiles, remembering that the full profile list is shown separately in the UI.

A complete answer should:
- state the total number of retrieved/relevant profiles found, if profiles exist
- give a strategic recommendation, not a raw profile dump
- highlight the strongest 2-4 relevant connections when available
- explain briefly why those highlighted connections fit the user's need
- not be penalized for omitting other retrieved profiles from the prose answer
- include a Missing section only when an important part of the query cannot be
  answered from the retrieved profiles and that limitation is not already stated
  clearly in the answer

For "all", "list all", or "give me all" queries:
- Do not require the answer to enumerate all profiles.
- It is enough to state the count and summarize/highlight the best matches,
  because the full result list is visible in the UI expander.
  
Do not require a literal "Missing:" section when the answer clearly states the
limitation in normal prose.

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

For owner exclusions, absence from the owners list is valid evidence.
Example: if owners are ["Aleksandar"] and the user asks for people who are not
Mihajlo's connections, it is correct to say the person is not Mihajlo's connection.

A citation is correct if:
- it uses <a href="FULL_LINKEDIN_URL">Displayed Name</a> (connection: Owner)
- the URL exactly matches a retrieved profile
- the displayed name refers to that retrieved profile
- the owner string contains exactly the supplied owners

Do not penalize:
- accented vs unaccented display if the URL identifies the exact retrieved profile
- count statements
- repeated citations of the same person
- not listing every retrieved profile
- generic owner-exclusion statements that are supported by owners metadata


Set criterion to "citation_owner_accuracy".
"""

RECOMMENDATION_QUALITY_PROMPT = f"""
{COMMON_JUDGE_CONTEXT}

Criterion: recommendation_quality.

Evaluate whether the answer is a concise, useful strategic recommendation.

If retrieved profiles are empty:
- a concise no-results answer should receive 1.0 if it does not invent candidates
- do not penalize it for not recommending people

For non-empty results, the answer should:
- prioritize the strongest people instead of listing everyone
- give clear reasons for the recommendation
- treat multiple owners as a stronger introduction path when relevant
- avoid filler, unsupported confidence, and excessive length
- stay within the requested concise style unless a short Missing section is necessary

Set criterion to "recommendation_quality".
"""

llm = get_llm(LLM_MODEL)
judge = llm.with_structured_output(CriterionJudgement)


def context_builder(state: JudgeState) -> str:
    return f"""
User query:
{state["query"]}

Number of retrieved profiles:
{len(state["profiles"])}

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

        passed = 0
        overall_score = 0
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
            overall_score += r['overall_score']
            print(f"Passed: {r['passed']}")
            if r["passed"]:
                passed += 1
            print()

        print(f"PASSED: = {passed} / {len(results)}")
        print(f"Overall score: {overall_score} / {len(results)}")

if __name__ == "__main__":
    run_eval()