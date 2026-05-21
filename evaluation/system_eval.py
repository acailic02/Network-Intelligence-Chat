import json
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import START
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from evaluation.llm_judge import JudgeState
from src.agents.synthesis import format_profiles_for_prompt, synthesize


class EvalState(TypedDict):
    llm_judge_state: JudgeState
