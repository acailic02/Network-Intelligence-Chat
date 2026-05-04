from pydantic import BaseModel, Field
from src.llm.client import parse
from enum import Enum
from typing import Optional, List

SYSTEM_PROMPT = """
    You are a query parser for a LinkedIn network search tool.
    Your job is to extract structured search parameters from the user's message.

    QUERY TYPE RULES:
    1. LOOKUP: at least ONE LookupAttributes field is populated AND contextual_need is None
    2. DISCOVERY: ALL LookupAttributes fields are None AND contextual_need IS populated
    3. HYBRID: at least ONE LookupAttributes field IS populated AND contextual_need IS populated
        e.g.: - "I need a technical co-founder with ML experience based in London"
                -- position.title: ["co-founder"], connection.city: ["London"], contextual_need: "technical co-founder with ML experience"
            - "Who in the fintech space could open doors for us in the US?"
                -- position.company_location: ["US"], contextual_need: "person who can open doors in fintech"

    LOOKUP ATTRIBUTES RULES:
    1. connection: populate country, city, skills if mentioned
    2. position: populate title, company_name, company_location if mentioned. Set recently_changed: true if user asks for people who recently changed jobs
    3. education: populate degree, school_name if mentioned
    4. owner: ONLY real person names explicitly mentioned by the user, e.g. ["John Smith"]. Never verbs or roles.
    5. If there are no information to fill in fields inside these classes, set those class fields as None in LookupAttributes
    

    CONTEXTUAL NEED RULES:
    1. Set to None if LookupAttributes fully captures the user's intent
    2. Populate only if there is a semantic dimension that cannot be expressed as a structured filter
        e.g.: - "I need people from Serbia" -> connection.country: ["Serbia"], contextual_need: None
              - "I need someone who can open doors in the market" -> contextual_need: "person who can open doors in the market"

    GENERAL RULES:
    1. Never infer or assume values that are not stated or clearly implied
    2. Never populate owner with verbs, roles or descriptions — only explicit person names
"""

class QueryType(str, Enum):
    LOOKUP = "LOOKUP"
    DISCOVERY = "DISCOVERY"
    HYBRID = "HYBRID"

class ConnectionFilters(BaseModel):
    country: Optional[List[str]] = Field(default=None, description="All countries explicitly mentioned in the user's query.")
    city: Optional[List[str]] = Field(default=None, description="All cities explicitly mentioned in the user's query. e.g. 'I need people from Zagreb, Belgrade too' -> ['Zagreb', 'Belgrade']")
    skills: Optional[List[str]] = Field(default=None, description="Skills explicitly mentioned in the user's query. Do not infer skills from context.")

class PositionFilters(BaseModel):
    title: Optional[List[str]] = Field(default=None, description="Job titles explicitly mentioned in the user's query, e.g. 'developer', 'HR', 'ML engineer'.")
    company_name: Optional[List[str]] = Field(default=None, description="Specific company names mentioned in the user's query, e.g. 'Microsoft', 'Stripe'.")
    company_location: Optional[List[str]] = Field(default=None, description="Locations of companies mentioned in the user's query, e.g. 'companies based in London'.")
    recently_changed: Optional[bool] = Field(default=None, description="Set to true if user is looking for people who recently changed jobs.")

class EducationFilters(BaseModel):
    degree: Optional[List[str]] = Field(default=None, description="Specific degrees mentioned in the user's query, e.g. 'master', 'phd'.")
    school_name: Optional[List[str]] = Field(default=None, description="Specific schools or universities mentioned in the user's query.")

class LookupAttributes(BaseModel):
    connection: Optional[ConnectionFilters] = Field(default=None, description="Filters related to the person's personal info, location and skills. If none of fields in class have value, set as None.")
    position: Optional[PositionFilters] = Field(default=None, description="Filters related to the person's work experience and companies. If none of fields in class have value, set as None.")
    education: Optional[EducationFilters] = Field(default=None, description="Filters related to the person's educational background. If none of fields in class have value, set as None.")
    owner: Optional[List[str]] = Field(default=None,description="Explicit person names mentioned by the user whose connections to search through. Only real person names, never verbs or roles.")

class UserQuery(BaseModel):
    lookup_filters: LookupAttributes = Field(description="Structured filters extracted from the user's query.")
    contextual_need: Optional[str] = Field(default=None, description="Free-form semantic description of the person the user is looking for, used for vector search. Populate only if LookupAttributes does not fully capture the user's intent. Do not populate for pure LOOKUP queries.")
    contextual_trigger: Optional[str] = Field(default=None, description="The reason or situation behind the user's search, explicitly stated or clearly implied. e.g. 'raising a Series A', 'speaking at Web Summit'.")
    query_type: QueryType = Field(
        description="""Derived from what you populated above:
        - LOOKUP: at least ONE LookupAttributes field is populated AND contextual_need is None
        - DISCOVERY: ALL LookupAttributes fields are None AND contextual_need IS populated
        - HYBRID: at least ONE LookupAttributes field IS populated AND contextual_need IS populated"""
    )

def understand(user_input: str) -> UserQuery:
    return parse(
        messages=[{"content": user_input}],
        response_format=UserQuery,
        system=SYSTEM_PROMPT,
    )
