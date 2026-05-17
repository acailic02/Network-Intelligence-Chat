from pydantic import BaseModel, Field
from src.llm.client import parse
from enum import Enum
from typing import Optional, List, Literal

SYSTEM_PROMPT = """
    You are a query parser for a LinkedIn network search tool.
    Your job is to extract structured search parameters from the user's message.

    GENERAL RULES:
    1. Never infer or assume values that are not stated or clearly implied
    2. Never populate owner with verbs, roles or descriptions — only explicit person names
    3. All fields MUST BE populated IN ENGLISH regardless of the language the user writes in.
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
             "ML" -> "Machine Learning" (when used as a skill)
             "ML Engineer" -> "Machine Learning Engineer" (when used as a job title)
             "HR" -> "Human Resources"
             "BD" -> "Business Development"
             "QA" -> "Quality Assurance"
    5. company_name is ONLY for real, specific company names like "Microsoft", "Stripe".
       Never populate with industry descriptions, categories, or startup types like "B2B SaaS startups".
    6. Do not populate title with roles that are descriptions of a person type rather than a standard job title.
        e.g. "co-founder", "advisor", "mentor" -> contextual_need, not position.title
    7. contextual_trigger is the REASON behind the search, not a description of the person being searched for.
        e.g. "between jobs" describes the person -> contextual_trigger: None ; but recently_changed: true
             "raising a Series A" is the reason -> contextual_trigger: "raising a Series A"
    8. If user mentions "US market", "Berlin", "London" — connection.country or connection.city must be populated
    9. If industry is mentioned intagrate it into contextual_need.
    
    LOOKUP ATTRIBUTES RULES:
    1. If there are NO information about fields for some class, SET value if that class fields as None IN LookupAttributes
    2. connection: populate country, city, skills if mentioned
    3. position: populate title, company_name, company_location if mentioned. Set recently_changed: true if user asks for people who recently changed jobs or are between jobs
    4. education: populate degree, school_name if mentioned
    5. owner: ONLY real person names explicitly mentioned by the user, e.g. ["John Smith"]. Never verbs or roles.    
    
    QUERY TYPE RULES:
    1. LOOKUP: at least ONE LookupAttributes field is populated AND contextual_need is None
    2. DISCOVERY: ALL LookupAttributes fields are None AND contextual_need IS populated
    3. HYBRID: at least ONE LookupAttributes field IS populated AND contextual_need IS populated

    WHEN TO USE HYBRID vs DISCOVERY:
    - If user mentions ANY of the following, it is ALWAYS a structured filter and query is HYBRID:
        * A location (country, city, region) -> connection.country or connection.city
        * A specific company name -> position.company_name
        * A specific job title -> position.title
        * A specific skill -> connection.skills
        * A specific school or degree -> education.school_name or education.degree
        * Recently changed jobs or between jobs -> position.recently_changed
    - DISCOVERY is ONLY when the user's intent CANNOT be expressed with ANY structured filter at all
    - When in doubt between HYBRID and DISCOVERY, choose HYBRID

    HYBRID EXAMPLES:
    - "We are opening an office in Amsterdam, who in our network works in HR or recruiting there?"
        -> connection.city: ["Amsterdam"] (structured) + contextual_need: "person who works in HR or recruiting" -> HYBRID
    - "I have a board meeting next week, who in our network has experience with corporate governance in Germany?"
        -> connection.country: ["Germany"] (structured) + contextual_need: "person with experience in corporate governance" -> HYBRID
    - "Looking for someone with sales experience who could help us break into healthcare"
        -> connection.skills: ["sales"] (structured) + contextual_need: "person who can help break into healthcare industry" -> HYBRID

    DISCOVERY EXAMPLES:
    - "Who in our network would be the best person to cold email about a partnership?" -> DISCOVERY
    - "Who could introduce us to the right people in the gaming industry?" -> DISCOVERY
    - "Who in our network has the most interesting career story?" -> DISCOVERY

    CONTEXTUAL NEED RULES:
    1. Set to None if LookupAttributes fully captures the user's intent
    2. Populate only if there is a semantic dimension that cannot be expressed as a structured filter
        e.g.: - "I need people from Serbia" -> connection.country: ["Serbia"], contextual_need: None
              - "I need someone who can open doors in the market" -> contextual_need: "person who can open doors in the market"
    3. Skills that describe a person's expertise in context (not explicitly listed skills) go into contextual_need, not connection.skills
        e.g.: - "technical co-founder with ML experience" -> contextual_need: "technical co-founder with ML experience", NOT skills: ["Machine Learning"]
              - "someone who knows Python and JavaScript" -> skills: ["Python", "JavaScript"], NOT contextual_need

    OPERATOR RULES:
    1. Set operator to None if only one value is in the list
    2. Set operator to ANY if user uses "or" logic — e.g. "works at Stripe or Revolut" -> company_name_operator: ANY
    3. Set operator to ALL if user uses "and" logic — e.g. "knows both Python and JavaScript" -> skills_operator: ALL
    4. If user does not specify, set operator to None

    EXCLUSION RULES:
    1. Exclusion fields (exc_*) are populated when user explicitly does NOT want certain values
    2. Never infer exclusions — only populate if user explicitly states they do not want something
        e.g. "Find developers but not from Microsoft" 
                -> exc_company_name: ["Microsoft"]
             "I need people from Serbia, not from Belgrade" 
                -> country: ["Serbia"], exc_city: ["Belgrade"]
             "Find ML engineers but not junior ones" 
                -> title: ["Machine Learning Engineer"], exc_title: ["Junior Machine Learning Engineer"]
             "Everyone from our network except Jelena's connections" 
                -> exc_title: ["Jelena"] (owner exclusion)
    3. Exclusion and inclusion can be populated at the same time
        e.g. "Find Python developers in Berlin, but not from Google"
                -> city: ["Berlin"], skills: ["Python"], exc_company_name: ["Google"]
    4. If user says "only" or "exclusively" that is an inclusion, not an exclusion
        e.g. "Only people from Serbia" 
                -> country: ["Serbia"], NOT exc_country of everything else

    current_company_name and current_job_title RULES:
    1. These fields represent current snapshot information directly associated with the person's profile
    2. They live in ConnectionFilters, NOT PositionFilters
    3. Populate ONLY when user explicitly uses "currently", "right now", "at the moment" or similar present tense indicators
    4. If current_company_name is populated, position.company_name MUST also be populated with the same value
    5. If current_job_title is populated, position.title MUST also be populated with the same value
        e.g. "Who is currently a CTO in Belgrade" -> connection.current_job_title: "Chief Technology Officer", position.title: ["Chief Technology Officer"]
"""

class QueryType(str, Enum):
    LOOKUP = "LOOKUP"
    DISCOVERY = "DISCOVERY"
    HYBRID = "HYBRID"

class ConnectionFilters(BaseModel):
    country: Optional[List[str]] = Field(default=None, description="All countries explicitly mentioned in the user's query.")
    exc_country: Optional[List[str]] = Field(default=None, description="User explicitly DOES NOT want connections from these countries.")
    city: Optional[List[str]] = Field(default=None, description="All cities explicitly mentioned in the user's query. e.g. 'I need people from Zagreb, Belgrade too' -> ['Zagreb', 'Belgrade']")
    exc_city: Optional[List[str]] = Field(default=None, description="User explicitly DOES NOT want connections from these cities.")
    skills: Optional[List[str]] = Field(default=None, 
                                        description="""Skills explicitly mentioned in the user's query. 
                                        Look for keywords as: 'knows', 'has/is experience', 'works with', 'proficient', 'expert', 'background in'...  
                                        Do not infer skills from context.""")
    skills_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many skills does SQL have to look for. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for someone that has skill with C++ and JavaScript, value is set as ALL.")
    exc_skills: Optional[List[str]] = Field(default=None, description="User explicitly DOES NOT want connections with these skills.")
    current_company_name: Optional[str] = Field(default=None, description="The current company the user is looking for, explicitly mentioned as current or present. e.g. 'Who currently works at Google' -> 'Google'. Do not populate if user is asking for any past or general company experience.")
    current_job_title: Optional[str] = Field(default=None, description="The current job title the user is looking for, explicitly mentioned as current or present. e.g. 'Who is currently a CTO' -> 'CTO'. Do not populate if user is asking for any past or general title experience.")

class PositionFilters(BaseModel):
    title: Optional[List[str]] = Field(default=None, description="Job titles explicitly mentioned in the user's query, e.g. 'developer', 'HR', 'ML engineer'.")
    title_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many titles does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connections that have titles HR and Engineer, value is set as ALL.")
    exc_title: Optional[List[str]] = Field(default=None, description="User explicitly DOES NOT want connections with these titles.")
    company_name: Optional[List[str]] = Field(default=None, description="Specific company names mentioned in the user's query, e.g. 'Microsoft', 'Stripe'.")
    company_name_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many company names does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connections that work in Rivian or Microsoft, value is set as ANY.")
    exc_company_name: Optional[List[str]] = Field(default=None, description="User explicitly DOES NOT want connections working in these companies.")
    company_location: Optional[List[str]] = Field(default=None, description="Locations of companies mentioned in the user's query, e.g. 'companies based in London'; 'working in game developing in Serbia'")
    recently_changed: Optional[bool] = Field(default=None, description="Set to true if user is looking for people who recently changed jobs.")

class EducationFilters(BaseModel):
    degree: Optional[List[str]] = Field(default=None, description="Specific degrees mentioned in the user's query, e.g. 'master', 'phd'.")
    degree_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many degrees does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connections that has to have any degree user asked for, value is set as ANY.")
    exc_degree: Optional[List[str]] = Field(default=None, description="User explicitly DOES NOT want connections with these degrees.")
    school_name: Optional[List[str]] = Field(default=None, description="Specific schools or universities mentioned in the user's query.")
    school_name_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many schools does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connections that went to all schools mentioned, value is set as ALL.")
    exc_school_name: Optional[List[str]] = Field(default=None, description="User explicitly DOES NOT want connections that went to these schools.")

class LookupAttributes(BaseModel):
    connection: Optional[ConnectionFilters] = Field(default=None, description="Filters related to the person's personal info, location and skills. If none of fields in class have value, set as None.")
    position: Optional[PositionFilters] = Field(default=None, description="Filters related to the person's work experience and companies. If none of fields in class have value, set as None.")
    education: Optional[EducationFilters] = Field(default=None, description="Filters related to the person's educational background. If none of fields in class have value, set as None.")
    owner: Optional[List[Literal["Jelena", "Aleksandar", "Mihajlo", "Petar"]]] = Field(default=None,description="Explicit person names mentioned by the user whose connections to search through. Only real person names, never verbs or roles.")
    owner_operator: Optional[Literal["ANY", "ALL"]] = Field(default=None, description="How many owners does found connections have to satisfy. Possible values are ANY/ALL, if user didnt specify set as None. If user asked for connection that has connections with Jelena and Aleksandar, value is set as ALL.")
    exc_owner: Optional[Literal["Jelena", "Aleksandar", "Mihajlo", "Petar"]] = Field(default=None, description="User explicitly DOES NOT want connections from these owners.")

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
