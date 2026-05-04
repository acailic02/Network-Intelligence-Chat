from sqlalchemy import select
from sqlalchemy.orm import Session
from src.storage.models import Profile, Positions, Education


def get_connections(session: Session,
                    country: str = None,
                    city: str = None,
                    skills: list[str] = None,
                    owners_any: list[str] = None,
                    owners_all: list[str] = None,
                    company_name: str = None,
                    school_name: str = None,
                    degree: str = None,
                    job_title: str = None,
                    limit: int = None,
                    offset: int = None,):
    query = select(Profile).limit(limit).offset(offset)

    if country:
        query = query.where(Profile.country == country)
    if city:
        query = query.where(Profile.city == city)
    if skills:
        query = query.where(Profile.skills.contains(skills))
    if owners_any:
        query = query.where(Profile.owners.overlap(owners_any))
    if owners_all:
        query = query.where(Profile.owners.contains(owners_all))
    if company_name:
        query = query.where(Profile.positions.any(Positions.company_name == company_name))
    if school_name:
        query = query.where(Profile.education.any(Education.school_name == school_name))
    if degree:
        query = query.where(Profile.education.any(Education.degree == degree))
    if job_title:
        query = query.where(Profile.positions.any(Positions.title == job_title))

    return session.scalars(query).all()
