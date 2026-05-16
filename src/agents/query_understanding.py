from pydantic import BaseModel, Field
from src.llm.client import parse
from enum import Enum
from typing import Optional, List, Literal

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
    3. All fields must be populated in English regardless of the language the user writes in.
        e.g. user writes in Serbian: "Nađi mi ljude iz Srbije" -> connection.country: ["Serbia"]
    4. Expand abbreviations and acronyms to their full English name before populating fields.
        e.g. "MATF" -> "Faculty of Mathematics"
             "ETF" -> "School of Electrical Engineering"
             "RAF" -> "School of Computing"
             "FON" -> "Faculty of Organizational Sciences"
             "MIT" -> "Massachusetts Institute of Technology"
             "NYU" -> "New York University"
             "CTO" -> "Chief Technology Officer"
             "CEO" -> "Chief Executive Officer"
             "CFO" -> "Chief Financial Officer"
             "CMO" -> "Chief Marketing Officer"
             "COO" -> "Chief Operating Officer"
             "VP" -> "Vice President"
             "PM" -> "Product Manager"
             "EM" -> "Engineering Manager"
             "ML" -> "Machine Learning Engineer" (when used as a job title)
             "HR" -> "Human Resources"
             "BD" -> "Business Development"
             "QA" -> "Quality Assurance"
    
    OPERATOR RULES:
    1. Set operator to None if only one value is in the list
    2. Set operator to ANY if user uses "or" logic — e.g. "works at Stripe or Revolut" -> company_name_operator: ANY
    3. Set operator to ALL if user uses "and" logic — e.g. "knows both Python and JavaScript" -> skills_operator: ALL
    4. If user does not specify, set operator to None

    current_company_name AND current_job_title RULES:
    - Populate ONLY when user explicitly uses "currently", "right now", "at the moment" or similar present tense indicators
    - If current_company_name is populated, company_name MUST also be populated with the same value
    - If current_job_title is populated, title MUST also be populated with the same value
    - e.g. "Who currently works at Microsoft" -> current_company_name: "Microsoft", position.company_name: ["Microsoft"]
    - e.g. "Who is currently a CTO" -> current_job_title: "CTO", position.title: ["CTO"]
    - e.g. "Who works at Microsoft" -> current_company_name: None, position.company_name: ["Microsoft"]
"""

class QueryType(str, Enum):
    LOOKUP = "LOOKUP"
    DISCOVERY = "DISCOVERY"
    HYBRID = "HYBRID"

class ConnectionFilters(BaseModel):
    country: Optional[List[str]] = Field(default=None, description="All countries explicitly mentioned in the user's query.")
    city: Optional[List[str]] = Field(default=None, description="All cities explicitly mentioned in the user's query. e.g. 'I need people from Zagreb, Belgrade too' -> ['Zagreb', 'Belgrade']")
    skills: Optional[List[str]] = Field(default=None, description="Skills explicitly mentioned in the user's query. Do not infer skills from context.")
    skills_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many skills does SQL have to look for. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for someone that has skill with C++ and JavaScript, value is set as ALL.")
    current_company_name: Optional[str] = Field(default=None, description="The current company the user is looking for, explicitly mentioned as current or present. e.g. 'Who currently works at Google' -> 'Google'. Do not populate if user is asking for any past or general company experience.")
    current_job_title: Optional[str] = Field(default=None, description="The current job title the user is looking for, explicitly mentioned as current or present. e.g. 'Who is currently a CTO' -> 'CTO'. Do not populate if user is asking for any past or general title experience.")

class PositionFilters(BaseModel):
    title: Optional[List[str]] = Field(default=None, description="Job titles explicitly mentioned in the user's query, e.g. 'developer', 'HR', 'ML engineer'.")
    title_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many titles does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connections that have titles HR and Engineer, value is set as ALL.")
    company_name: Optional[List[str]] = Field(default=None, description="Specific company names mentioned in the user's query, e.g. 'Microsoft', 'Stripe'.")
    company_name_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many company names does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connections that work in Rivian or Microsoft, value is set as ANY.")
    company_location: Optional[List[str]] = Field(default=None, description="Locations of companies mentioned in the user's query, e.g. 'companies based in London'.")
    recently_changed: Optional[bool] = Field(default=None, description="Set to true if user is looking for people who recently changed jobs.")

class EducationFilters(BaseModel):
    degree: Optional[List[str]] = Field(default=None, description="Specific degrees mentioned in the user's query, e.g. 'master', 'phd'.")
    degree_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many degrees does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connections that has to have any degree user asked for, value is set as ANY.")
    school_name: Optional[List[str]] = Field(default=None, description="Specific schools or universities mentioned in the user's query.")
    school_name_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many schools does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connections that went to all schools mentioned, value is set as ALL.")

class LookupAttributes(BaseModel):
    connection: Optional[ConnectionFilters] = Field(default=None, description="Filters related to the person's personal info, location and skills. If none of fields in class have value, set as None.")
    position: Optional[PositionFilters] = Field(default=None, description="Filters related to the person's work experience and companies. If none of fields in class have value, set as None.")
    education: Optional[EducationFilters] = Field(default=None, description="Filters related to the person's educational background. If none of fields in class have value, set as None.")
    owner: Optional[List[str]] = Field(default=None,description="Explicit person names mentioned by the user whose connections to search through. Only real person names, never verbs or roles.")
    owner_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many owners does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connection that has connections with Jelena and Aleksandar, value is set as ALL.")

class UserQuery(BaseModel):
    lookup_filters: LookupAttributes = Field(description="Structured filters extracted from the user's query.")
    contextual_need: Optional[str] = Field(default=None, description="Free-form semantic description of the person the user is looking for, used for vector search. Populate only if LookupAttributes does not fully capture the user's intent. Do not populate for pure LOOKUP queries.")
    contextual_trigger: Optional[str] = Field(default=None, description="The reason or situation behind the user's search, explicitly stated or clearly implied. e.g. 'raising a Series A', 'speaking at Web Summit'.")
    limit: int = Field(default=10,
    description="""Number of results to return. 
            If user explicitly states a number, use that. 
            If not specified, estimate based on the nature of the query: 
                use a lower number (5-10) for specific targeted searches, 
                higher (20-50) for broad exploratory searches. 
                e.g. 'Find me 3 people from Serbia' -> 3; 
                     'Who in our network works in fintech' -> 20; 
                     'I need a technical co-founder' -> 5.""")
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
