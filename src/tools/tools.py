from typing import TypedDict

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from src.storage.db import engine
from src.retrieval.sql_query import get_connections
from src.retrieval.vector_query import semantic_query


class Position(TypedDict):
    title: str | None
    company_name: str | None
    company_location: str | None

class Education(TypedDict):
    degree: str | None
    school_name: str | None

class Profile(TypedDict):
    name: str
    headline: str | None
    summary: str | None
    linkedin_url: str
    location: str | None
    current_company: str | None
    positions: list[Position] | None
    education: list[Education] | None
    owners: list[str]

@tool
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search LinkedIn profiles by semantic similarity over their headlines and summaries.
    Use this tool for fuzzy or conceptual queries that can't be looked up with structured SQL queries.
    Returns a ranked list of relevant profiles with their owners.
    """
    result = semantic_query(query, top_k=top_k)

    return [
        {
            "name": f"{metadata['first_name']} {metadata['last_name']}",
            "headline": metadata['headline'],
            "summary": metadata['summary'],
            "linkedin_url": metadata['linkedin_url'],
            "owners": metadata["owners"],
        }
        for metadata in result["metadatas"][0]
    ]


@tool
def structured_filter(
        country: str = None,
        city: str = None,
        skills_all: list[str] = None,
        skills_any: list[str] = None,
        owners_any: list[str] = None,
        owners_all: list[str] = None,
        current_company_name: str = None,
        company_location: str = None,
        multiple_comapny_names_all: list[str] = None,
        any_company_name: list[str] = None,
        school_name: str = None,
        degree: list[str] = None,
        current_job_title: str = None,
        multiple_job_titles_all: list[str] = None,
        any_job_title: list[str] = None,
        limit: int = None,
        offset: int = None
) -> list[Profile]:
    """
    Filter LinkedIn connections using structured SQL queries.
    Use this tool for exact lookups: specific company, country, city, job title,
    school, degree, skills, or owners. All parameters are optional and combined with AND.
    Returns a list of matching profiles with their owners.
    """
    with Session(engine) as session:
        results = get_connections(
            session,
            country=country,
            city=city,
            current_company_name=current_company_name,
            company_location=company_location,
            multiple_comapny_names_all=multiple_comapny_names_all,
            any_company_name=any_company_name,
            current_job_title=current_job_title,
            any_job_title=any_job_title,
            multiple_job_titles_all=multiple_job_titles_all,
            school_name=school_name,
            degree=degree,
            skills_all=skills_all,
            skills_any=skills_any,
            owners_any=owners_any,
            owners_all=owners_all,
            limit=limit,
        )
        print(f"COUNTRY: {country}")
        print(f"CITY: {city}")
        print(f"COMPANY: {any_company_name}")
        print(f"JOB_TITLE: {any_job_title}")
        print(f"OWNERS: {owners_all}")
        profiles = [
            {
                "name": f"{r.first_name} {r.last_name}",
                "headline": None,
                "summary": None,
                "linkedin_url": r.linkedin_url,
                "location": f"{r.city}, {r.country}",
                # TODO: move current_company to positions (leave it for now until synthesis.py adopts change)
                "current_company": current_company_name if current_company_name else None, # if any profile is fetched that means we found profile with this current position
                "positions":[
                    {
                        "title": p.title,
                        "company_name": p.company_name,
                        "company_location": p.company_location,
                    }
                for p in r.positions],
                "education":[
                    {
                        "degree": e.degree,
                        "school_name": e.school_name
                    }
                for e in r.education],
                "owners": r.owners,
            }
            for r in results
        ]
        return profiles


@tool
def count_matches(results: list) -> int:
    """Count how many profiles were retrieved."""
    return len(results)


tools = [structured_filter, semantic_search, count_matches]