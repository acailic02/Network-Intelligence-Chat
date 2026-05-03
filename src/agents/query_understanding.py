from pydantic import BaseModel, Field
from src.llm.client import parse
from enum import Enum
from typing import Optional, List

SYSTEM_PROMPT = """
    You are a query parser for a LinkedIn network search tool.
    Your job is to extract structured search parameters from the user's message.

    QUERY TYPE RULES:
    1. LOOKUP: at least one LookupAttributes field is populated AND contextual_need is None
    2. DISCOVERY: ALL LookupAttributes fields ARE None AND contextual_need IS populated
    3. HYBRID: at LEAST one LookupAttributes field IS populated AND contextual_need IS populated.
        e.g.: - "I need a technical co-founder with ML experience based in London" 
              -- location: London (SQL) + "technical co-founder with ML experience" (vector)
              - "Who in the fintech space could open doors for us in the US?" 
              -- industry: fintech (SQL) + "open doors" (vector)

    CONTEXTUAL NEED RULES:
    1. Set to null if LookupAttributes fully captures the user's intent
    2. Populate only if there is a context that cannot be expressed as a attribute in LookupAttributes class
        e. g.: Example of null: "I need people from Serbia" -> location: Serbia, contextual_need: null
               Example of populated: "I need someone who can open doors in the markets" -> contextual_need: "person who can open doors in markets"

    GENERAL RULES:
    1. Never infer or assume values that are not stated or clearly implied
"""

class QueryType(str, Enum):
    LOOKUP = "LOOKUP"
    DISCOVERY = "DISCOVERY"
    HYBRID = "HYBRID"

class LookupAttributes(BaseModel):
    company: Optional[str] = Field(default=None, description="What company is user looking for, if not stated set as None.")
    location: Optional[str] = Field(default=None, description="Is it stated in what country user is looking for connections, if not stated set as None.")
    industry: Optional[List[str]] = Field(default=None, description="What industry users message is asking for, if not stated set as None.")
    employment_status: Optional[str] = Field(default=None, description="Does user ask for someone that is employed or NOT, None if there is no information about this atribute.")
    network_connection: Optional[List[str]] = Field(default=None, description="Explicit person names mentioned by user whose connections to search through, e.g. ['John Smith', 'AnaKović']. Only real person names, never verbs or roles.")

class UserQuery(BaseModel):
    query_type: QueryType = Field(description="What type of query did user give to LLM, is it LOOKUP/DISCOVERY/HYBRID. This is contextual not factual.")
    lookup_filters: LookupAttributes = Field(description="Looking at users message make LookupAttributes class with attributes parsed from user message.")
    contextual_need: Optional[str] = Field(default=None, description="What kind of people is user searching for contextualy given by users query?")
    contextual_trigger: Optional[str] = Field(default=None, description="What is the reason behind users search for a person given by messages context?")

def understand(user_input: str) -> UserQuery:
    return parse(
        messages=[{"content": user_input}],
        response_format=UserQuery,
        system=SYSTEM_PROMPT,
    )
