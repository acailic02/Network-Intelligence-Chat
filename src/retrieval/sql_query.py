from sqlalchemy import select, func
from sqlalchemy.orm import Session
from src.storage.models import Profile, Positions, Education


def get_connections(session: Session,
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
                    offset: int = None,):
    query = select(Profile).limit(limit).offset(offset)

    #za pronalazenje trenutne pozicije (pretpostavljamo da je to poslednja pozicija koja je dodata na profil)
    min_position_id = (
        select(func.min(Positions.id))
        .where(Positions.linkedin_url == Profile.linkedin_url)
        .scalar_subquery()
    )


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
    if current_company_name:
        query = query.where(Profile.positions.any((Positions.id == min_position_id) & (Positions.company_name == current_company_name)))
    if any_company_name:
        query = query.where(Profile.positions.any(Positions.company_name == any_company_name))
    if multiple_comapny_names_all:
        query = query.where(Profile.positions.contains(multiple_comapny_names_all))
    if multiple_comapny_names_any:
        query = query.where(Profile.positions.overlaps(multiple_comapny_names_any))
    if school_name:
        query = query.where(Profile.education.any(Education.school_name == school_name))
    if degree:
        query = query.where(Profile.education.any(Education.degree == degree))
    if current_job_title:
        query = query.where(Profile.positions.any((Positions.id == min_position_id) & (Positions.title == current_job_title)))
    if any_job_title:
        query = query.where(Profile.positions.any(Positions.title == any_job_title))
    if multiple_job_titles_all:
        query = query.where(Profile.positions.contains(multiple_job_titles_all))
    if multiple_job_titles_any:
        query = query.where(Profile.positions.overlaps(multiple_job_titles_any))

    return session.scalars(query).all()
