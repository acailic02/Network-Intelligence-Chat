from typing import Literal, Optional

from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import Session
from src.storage.models import Profile, Positions, Education


def get_connections(session: Session,
                    country: list[str] = None,
                    city: list[str] = None,
                    skills: list[str] = None,
                    skills_operator: Optional[Literal["ANY", "ALL"]] = None,
                    owners: list[str] = None,
                    owners_operator: Optional[Literal["ANY", "ALL"]] = None,
                    current_company_name: str = None,
                    current_job_title: str = None,
                    company_location: list[str] = None,
                    company_name: list[str] = None,
                    company_name_operator: Optional[Literal["ANY", "ALL"]] = None,
                    school_name: list[str] = None,
                    school_name_operator: Optional[Literal["ANY", "ALL"]] = None,
                    degree: list[str] = None,
                    degree_operator: Optional[Literal["ANY", "ALL"]] = None,
                    job_title: list[str] = None,
                    job_title_operator: Optional[Literal["ANY", "ALL"]] = None,
                    limit: int = 50,
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
        conditions = []
        for c in country:
            conditions.append(Profile.country.ilike(f"%{c}%"))

        query = query.where(or_(*conditions))
    if city:
        conditions = []
        for c in city:
            conditions.append(Profile.city.ilike(f"%{c}%"))

        query = query.where(or_(*conditions))
    if skills:
        conditions = []
        for skill in skills:
            conditions.append(Profile.skills.any(f"%{skill}%", operator=func.ilike))

        match skills_operator:
            case "ANY":
                query = query.where(or_(*conditions))
            case "ALL":
                query = query.where(and_(*conditions))
    if owners:
        conditions = []
        for owner in owners:
            conditions.append(Profile.owners.any(f"%{owner}%", operator=func.ilike))

        match owners_operator:
            case "ANY":
                query = query.where(or_(*conditions))
            case "ALL":
                query = query.where(and_(*conditions))

    if current_company_name:
        query = query.where(Profile.positions.any((Positions.id == min_position_id) & (Positions.company_name.ilike(f"%{current_company_name}%"))))
    if company_location:
        conditions = []
        for location in company_location:
            conditions.append(Profile.positions.any(Positions.company_location.ilike(f"{location}")))

        match company_name_operator:
            case "ANY":
                query = query.where(or_(*conditions))
            case "ALL":
                query = query.where(and_(*conditions))
    if company_name:
        conditions = []
        for name in company_name:
            conditions.append(Profile.positions.any(Positions.company_name.ilike(f"{name}")))

        match company_name_operator:
            case "ANY":
                query = query.where(or_(*conditions))
            case "ALL":
                query = query.where(and_(*conditions))
    if school_name:
        conditions = []
        for name in school_name:
            conditions.append(Profile.education.any(Education.school_name.ilike(f"{name}")))

        match school_name_operator:
            case "ANY":
                query = query.where(or_(*conditions))
            case "ALL":
                query = query.where(and_(*conditions))
    if degree:
        conditions = []
        for deg in degree:
            conditions.append(Profile.education.any(Education.degree.ilike(f"{deg}")))

        match degree_operator:
            case "ANY":
                query = query.where(or_(*conditions))
            case "ALL":
                query = query.where(and_(*conditions))
    if current_job_title:
        query = query.where(Profile.positions.any((Positions.id == min_position_id) & (Positions.title.ilike(f"%{current_job_title}%"))))
    if job_title:
        conditions = []
        for title in job_title:
            conditions.append(Profile.positions.any(Positions.title.ilike(f"{title}")))

        match job_title_operator:
            case "ANY":
                query = query.where(or_(*conditions))
            case "ALL":
                query = query.where(and_(*conditions))

    return session.scalars(query).all()
