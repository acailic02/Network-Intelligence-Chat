from sqlalchemy.orm import Session
from src.config import DATABASE_URL
from sqlalchemy import create_engine
from src.storage.models import Base, Profile, Positions, Education, Owner
import json

engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    Base.metadata.create_all(engine)


def load_data():
    with Session(engine) as session:
        try:
            with open("../../data/snapshot.json", 'r') as f:
                data = json.load(f)
                for entry in data:
                    lin_url = entry["source_row"]["url"]
                    connection = Profile(
                        linkedin_url=lin_url,
                        first_name=entry["source_row"]["first_name"],
                        last_name=entry["source_row"]["last_name"],
                        email=entry["source_row"]["email"],
                        country=entry["enrichment"].get("person", {}).get("location", {}).get("country", None),
                        city=entry["enrichment"].get("person", {}).get("location", {}).get("city", None),
                        skills=entry["enrichment"].get("person", {}).get("skills", []),
                        owners=entry["owners"],
                    )

                    positions = []
                    for pos in entry["enrichment"].get("person", {}).get("positions", {}).get("positionHistory", []):
                        position = Positions(
                            linkedin_url=lin_url,
                            title=pos["title"],
                            company_name=pos["companyName"],
                            company_location=pos["companyLocation"],
                            start_month=pos["startEndDate"]["start"]["month"] if pos["startEndDate"]["start"] else None,
                            start_year=pos["startEndDate"]["start"]["year"] if pos["startEndDate"]["start"] else None,
                            end_month=pos["startEndDate"]["end"]["month"] if pos["startEndDate"]["end"] else None,
                            end_year=pos["startEndDate"]["end"]["year"] if pos["startEndDate"]["end"] else None,
                        )
                        positions.append(position)

                    educations = []
                    for edu in entry["enrichment"].get("person", {}).get("schools", {}).get("educationHistory", []):
                        education = Education(
                            linkedin_url=lin_url,
                            degree=edu["degreeName"],
                            school_name=edu["schoolName"],
                            start_month=edu["startEndDate"]["start"]["month"] if edu["startEndDate"]["start"] else None,
                            start_year=edu["startEndDate"]["start"]["year"] if edu["startEndDate"]["start"] else None,
                            end_month=edu["startEndDate"]["end"]["month"] if edu["startEndDate"]["end"] else None,
                            end_year=edu["startEndDate"]["end"]["year"] if edu["startEndDate"]["end"] else None,
                        )
                        educations.append(education)

                    session.add(connection)
                    session.add_all(positions)
                    session.add_all(educations)

            owners = [
                Owner(name="Jelena", linkedin_url="https://www.linkedin.com/in/jgraovac"),
                Owner(name="Petar", linkedin_url="https://www.linkedin.com/in/petar-pavlovic-048a4022b"),
                Owner(name="Mihajlo", linkedin_url="https://www.linkedin.com/in/mihajlo-trifunovic-49325321a"),
                Owner(name="Aleksandar", linkedin_url="https://www.linkedin.com/in/aleksandar-ilic-a9bb59357"),
            ]
            connection = Profile(
                linkedin_url="https://www.linkedin.com/in/jgraovac",
                first_name="Jelena",
                last_name="Graovac",
                email="jgraovac12@gmail.com",
                skills=["Artificial Intelligence (AI)", "Large Language Models (LLM)", "Machine Learning", "Computer Science", "Algorithms", "Software Development", "Programming", "SQL", "C++", "C", "LaTeX", "C#", "Matlab", "Java", "Perl", "Python", "Eclipse", "Software Engineering", "Linux", "Bash"],
                owners=["Petar", "Aleksandar"]
            )
            session.add(connection)
            session.add_all(owners)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e


if __name__ == "__main__":
    init_db()
    load_data()