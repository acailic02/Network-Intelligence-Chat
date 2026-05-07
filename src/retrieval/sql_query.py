from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import Session
from src.storage.models import Profile, Positions, Education


def get_connections(session: Session,
                    country: str = None,
                    city: str = None,
                    skills_all: list[str] = None,
                    skills_any: list[str] = None,
                    owners_any: list[str] = None,
                    owners_all: list[str] = None,
                    current_company_name: str = None,
                    company_location: list[str] = None,
                    multiple_comapny_names_all: list[str] = None,
                    any_company_name: list[str] = None,
                    school_name: str = None,
                    degree: list[str] = None,
                    current_job_title: str = None,
                    multiple_job_titles_all: list[str] = None,
                    any_job_title: list[str] = None,
                    limit: int = None,
                    offset: int = None,):
    query = select(Profile).limit(limit).offset(offset)

    # "Current position" = first position row inserted for the profile (MIN(id)).
    # Correlate to the OUTER Profile table; the inner Positions reference must
    # NOT be correlated away (otherwise it gets confused with the Positions
    # alias used inside Profile.positions.any(...)).
    min_position_id = (
        select(func.min(Positions.id))
        .where(Positions.linkedin_url == Profile.linkedin_url)
        .correlate(Profile)
        .scalar_subquery()
    )

    if country:
        query = query.where(Profile.country.ilike(f"%{country}%"))
    if city:
        query = query.where(Profile.city.ilike(f"%{city}%"))
    if skills_all:
        query = query.where(and_(*[func.array_to_string(Profile.skills, "|").ilike(f"%{skill}%") for skill in skills_all]))
    if skills_any:
        query = query.where(or_(*[func.array_to_string(Profile.skills, "|").ilike(f"%{skill}%") for skill in skills_any]))
    if owners_any:
        query = query.where(or_(*[func.array_to_string(Profile.owners, "|").ilike(f"%{owner}%") for owner in owners_any]))
    if owners_all:
        query = query.where(and_(*[func.array_to_string(Profile.owners, "|").ilike(f"%{owner}%") for owner in owners_all]))
    if current_company_name:
        query = query.where(Profile.positions.any((Positions.id == min_position_id) & (Positions.company_name.ilike(f"%{current_company_name}%"))))
    if company_location:
        query = query.where(Profile.positions.any(or_(*[Positions.company_location.ilike(f"%{location}%") for location in company_location])))
    if multiple_comapny_names_all:
        for company_name in multiple_comapny_names_all:
            query = query.where(Profile.positions.any(Positions.company_name.ilike(f"%{company_name}%")))
    if any_company_name:
        query = query.where(Profile.positions.any(or_(*[Positions.company_name.ilike(f"%{company_name}%") for company_name in any_company_name])))
    if school_name:
        query = query.where(Profile.education.any(Education.school_name.ilike(f"%{school_name}%")))
    if degree:
        query = query.where(Profile.education.any(or_(*[Education.degree.ilike(f"%{d}%") for d in degree])))
    if current_job_title:
        query = query.where(Profile.positions.any((Positions.id == min_position_id) & (Positions.title.ilike(f"%{current_job_title}%"))))
    if any_job_title:
        query = query.where(Profile.positions.any(or_(*[Positions.title.ilike(f"%{title}%") for title in any_job_title])))
    if multiple_job_titles_all:
        for job_title in multiple_job_titles_all:
            query = query.where(Profile.positions.any(Positions.title.ilike(f"%{job_title}%")))

    return session.scalars(query).all()
