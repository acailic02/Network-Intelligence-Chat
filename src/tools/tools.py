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
    company_name: str = None,
    job_title: str = None,
    school_name: str = None,
    degree: str = None,
    skills: list[str] = None,
    owners_any: list[str] = None,
    owners_all: list[str] = None,
    limit: int = 10,
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
            company_name=company_name,
            job_title=job_title,
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
            "owners": metadata["owners"],
        }
        for metadata in result["metadatas"]
    ]


@tool
def count_matches(
    country: str = None,
    city: str = None,
    company_name: str = None,
    job_title: str = None,
    skills: list[str] = None,
    owners_any: list[str] = None,
) -> int:
    """
    Count how many profiles match the given filters without fetching full results.
    Use this before structured_filter to check if filters are too narrow or too broad.
    """
    with Session(engine) as session:
        results = get_connections(
            session,
            country=country,
            city=city,
            company_name=company_name,
            job_title=job_title,
            skills=skills,
            owners_any=owners_any,
        )
        return len(results)

tools = [structured_filter, semantic_search, count_matches]