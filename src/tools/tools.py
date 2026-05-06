from langchain_core.tools import tool
from sqlalchemy.orm import Session

from src.storage.db import engine
from src.retrieval.sql_query import get_connections
from src.retrieval.vector_query import semantic_query


# ─── TOOLS ───────────────────────────────────────────────────────────────────

@tool
def structured_filter(
        country: str = None,
        city: str = None,
        skills: list[str] = None,
        owners_any: list[str] = None,
        owners_all: list[str] = None,
        current_company_name: str = None,
        multiple_comapny_names_all: list[str] = None,
        multiple_comapny_names_any: list[str] = None,
        any_company_name: str = None,
        school_name: str = None,
        degree: str = None,
        current_job_title: str = None,
        any_job_title: str = None,
        multiple_job_titles_all: list[str] = None,
        multiple_job_titles_any: list[str] = None,
        limit: int = None,
        offset: int = None
) -> list[dict]:
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
            multiple_comapny_names_all=multiple_comapny_names_all,
            multiple_comapny_names_any=multiple_comapny_names_any,
            any_company_name=any_company_name,
            current_job_title=current_job_title,
            any_job_title=any_job_title,
            multiple_job_titles_all=multiple_job_titles_all,
            multiple_job_titles_any=multiple_job_titles_any,
            school_name=school_name,
            degree=degree,
            skills=skills,
            owners_any=owners_any,
            owners_all=owners_all,
            limit=limit,
        )
        return [
            {
                "name": f"{r.first_name} {r.last_name}",
                "linkedin_url": {r.linkedin_url},
                "location": f"{r.city}, {r.country}",
                "current_title": current_job_title if current_job_title else None,
                "current_company": current_company_name if current_company_name else None,
                "owners": r.owners,
            }
            for r in results
        ]


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
def count_matches(results: list) -> int:
    """Count how many profiles were retrieved."""
    return len(results)


tools = [structured_filter, semantic_search, count_matches]