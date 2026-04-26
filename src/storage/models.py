from typing import List, Optional, Literal
from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY


class Base(DeclarativeBase):
    pass

class Owner(Base):
    __tablename__ = "owners"
    name: Mapped[Literal["Mihajlo", "Jelena", "Petar", "Aleksandar"]] = mapped_column("name", primary_key=True)
    linkedin_url: Mapped[str] = mapped_column("linkedin_url", ForeignKey("connections.linkedin_url"))

    profile: Mapped[List["Profile"]] = relationship()

class Profile(Base):
    __tablename__ = "connections"
    linkedin_url: Mapped[str] = mapped_column("linkedin_url", primary_key=True)
    first_name: Mapped[str] = mapped_column("first_name")
    last_name: Mapped[str] = mapped_column("last_name")
    email: Mapped[Optional[str]] = mapped_column("email")
    country: Mapped[Optional[str]] = mapped_column("country")
    city: Mapped[Optional[str]] = mapped_column("city")
    skills: Mapped[Optional[List[str]]] = mapped_column("skills", ARRAY(String))
    owners: Mapped[List[str]] = mapped_column("owners", ARRAY(String))

    positions: Mapped[Optional[List["Positions"]]] = relationship(back_populates="connection")
    education: Mapped[Optional[List["Education"]]] = relationship(back_populates="connection")

class Positions(Base):
    __tablename__ = "positions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    linkedin_url: Mapped[str] = mapped_column("linkedin_url", ForeignKey("connections.linkedin_url"))
    title: Mapped[str] = mapped_column("title")
    company_name: Mapped[Optional[str]] = mapped_column("company_name")
    company_location: Mapped[Optional[str]] = mapped_column("company_location")
    start_month: Mapped[Optional[int]] = mapped_column("start_month")
    start_year: Mapped[Optional[int]] = mapped_column("start_year")
    end_month: Mapped[Optional[int]] = mapped_column("end_month")
    end_year: Mapped[Optional[int]] = mapped_column("end_year")

    connection: Mapped["Profile"] = relationship(back_populates="positions")


class Education(Base):
    __tablename__ = "education"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    linkedin_url: Mapped[str] = mapped_column("linkedin_url", ForeignKey("connections.linkedin_url"))
    degree: Mapped[Optional[str]] = mapped_column("degree")
    school_name: Mapped[Optional[str]] = mapped_column("school_name")
    start_month: Mapped[Optional[int]] = mapped_column("start_month")
    start_year: Mapped[Optional[int]] = mapped_column("start_year")
    end_month: Mapped[Optional[int]] = mapped_column("end_month")
    end_year: Mapped[Optional[int]] = mapped_column("end_year")

    connection: Mapped["Profile"] = relationship(back_populates="education")