from typing import TypedDict, Optional, Literal

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from src.storage.db import engine
from src.retrieval.sql_query import get_connections
from src.retrieval.vector_query import semantic_query


class Pos(TypedDict):
    title: str | None
    company_name: str | None
    company_location: str | None

class Edu(TypedDict):
    degree: str | None
    school_name: str | None

class Prof(TypedDict):
    name: str
    headline: str | None
    summary: str | None
    skills: list[str] | None
    linkedin_url: str
    location: str | None
    current_company: str | None
    current_job_title: str | None
    positions: list[Pos] | None
    education: list[Edu] | None
    owners: list[str]

@tool
def semantic_search(query: str, filters: dict = None, top_k: int = 10) -> list[Prof]:
    """
    Search LinkedIn profiles by semantic similarity over their headlines and summaries.
    Use this tool for fuzzy or conceptual queries that can't be looked up with structured SQL queries.
    Returns a ranked list of relevant profiles with their owners.
    """
    owners = filters.get("owners", None)
    owners_operator = filters.get("owners_operator", None)
    if owners:
        owners = [owner.capitalize() for owner in owners]
        if owners_operator == "ANY":
            semantic_filters = {"$or": [{"owners": {"$contains": owner}} for owner in owners]}
        elif owners_operator == "ALL":
            semantic_filters = {"$and": [{"owners": {"$contains": owner}} for owner in owners]}
    else:
        semantic_filters = None
    result = semantic_query(query, semantic_filters, top_k=top_k)

    profiles = [
        {
            "name": f"{metadata['first_name']} {metadata['last_name']}",
            "headline": metadata['headline'],
            "summary": metadata['summary'],
            "skills": None,
            "linkedin_url": metadata['linkedin_url'],
            "location": None,
            "current_company": None,
            "current_job_title": None,
            "positions": None,
            "education": None,
            "owners": metadata["owners"],
        }
        for metadata in result["metadatas"][0]
    ]

    return profiles


@tool
def structured_filter(
        country: list[str] = None,
        exclude_country: list[str] = None,
        city: list[str] = None,
        exclude_city: list[str] = None,
        skills: list[str] = None,
        skills_operator: Optional[Literal["ANY", "ALL"]] = None,
        exclude_skills: list[str] = None,
        owners: list[str] = None,
        owners_operator: Optional[Literal["ANY", "ALL"]] = None,
        exclude_owners: list[str] = None,
        current_company_name: str = None,
        current_job_title: str = None,
        company_location: list[str] = None,
        company_name: list[str] = None,
        company_name_operator: Optional[Literal["ANY", "ALL"]] = None,
        exclude_company_name: list[str] = None,
        school_name: list[str] = None,
        school_name_operator: Optional[Literal["ANY", "ALL"]] = None,
        exclude_school_name: list[str] = None,
        degree: list[str] = None,
        degree_operator: Optional[Literal["ANY", "ALL"]] = None,
        exclude_degree: list[str] = None,
        job_title: list[str] = None,
        job_title_operator: Optional[Literal["ANY", "ALL"]] = None,
        exclude_job_title: list[str] = None,
        limit: int = 50,
        offset: int = None
) -> list[Prof]:
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
            exclude_country=exclude_country,
            city=city,
            exclude_city=exclude_city,
            skills=skills,
            skills_operator=skills_operator,
            exclude_skills=exclude_skills,
            owners=owners,
            owners_operator=owners_operator,
            exclude_owners=exclude_owners,
            current_company_name=current_company_name,
            current_job_title=current_job_title,
            company_location=company_location,
            company_name=company_name,
            company_name_operator=company_name_operator,
            exclude_company_name=exclude_company_name,
            school_name=school_name,
            school_name_operator=school_name_operator,
            exclude_school_name=exclude_school_name,
            degree=degree,
            degree_operator=degree_operator,
            exclude_degree=exclude_degree,
            job_title=job_title,
            job_title_operator=job_title_operator,
            exclude_job_title=exclude_job_title,
            offset=offset,
            limit=50,
        )

        profiles = [
            {
                "name": f"{r.first_name} {r.last_name}",
                "headline": None,
                "summary": None,
                "skills": r.skills,
                "linkedin_url": r.linkedin_url,
                "location": f"{r.city}, {r.country}",
                "current_company": current_company_name if current_company_name else None, # if any profile is fetched that means we found profile with this current position
                "current_job_title": current_job_title if current_job_title else None, # same as for the current_company_name
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
def hybrid_search(filters: dict, query_text: str, top_k: int = 50) -> list[Prof]:
    """
    Combine structured SQL filters with semantic search and return a deduplicated,
    enriched list of profiles. Profiles found by both sources keep their structured
    fields and get enriched with headline/summary from the semantic side.
    """
    # Reuse the existing tools so the dict shape stays consistent and we don't
    # duplicate the ORM-to-dict conversion logic.
    structured_results = structured_filter.invoke(filters)
    semantic_results = semantic_search.invoke({"query": query_text, "filters": filters, "top_k": top_k})

    semantic_by_url = {p["linkedin_url"]: p for p in semantic_results}

    results = []
    seen_urls = set()

    for r in structured_results:
        url = r["linkedin_url"]
        if url in seen_urls:
            continue
        sem = semantic_by_url.get(url, {})
        r["headline"] = sem.get("headline")
        r["summary"] = sem.get("summary")
        results.append(r)
        seen_urls.add(url)

    for r in semantic_results:
        url = r["linkedin_url"]
        if url not in seen_urls:
            results.append(r)
            seen_urls.add(url)

    return results

@tool
def count_matches(results: list[dict]) -> int:
    """Count how many profiles were retrieved."""
    return len(results)


tools = [structured_filter, semantic_search, count_matches]