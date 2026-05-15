import pytest
from src.agents.synthesis import format_profiles_for_prompt, synthesize

SINGLE_PROFILE = [
    {
        "name": "Marko Petrovic",
        "linkedin_url": "https://linkedin.com/in/marko-petrovic",
        "owners": ["Aleksandar"],
        "current_title": "Senior PM",
        "current_company": "N26",
        "location": "Berlin, Germany",
        "industry": "Financial Services",
        "headline": "Senior PM at N26 | Fintech",
        "summary": "PM in fintech with 7 years of experience.",
        "skills": ["Product Management", "Fintech"],
        "experience": [
            {"title": "Senior PM", "company": "N26", "start": "2020", "end": "2024"}
        ],
    }
]

SHARED_CONNECTION = [
    {
        "name": "Ana Anic",
        "linkedin_url": "https://linkedin.com/in/ana-anic",
        "owners": ["Mihajlo", "Petar"],
        "current_title": "Engineering Lead",
        "current_company": "Solarisbank",
        "location": "Berlin, Germany",
        "industry": "Financial Services",
        "headline": "Engineering Lead at Solarisbank",
        "summary": "Software engineer in banking-as-a-service.",
        "skills": ["Python", "Banking APIs"],
        "experience": [
            {"title": "Engineering Lead", "company": "Solarisbank", "start": "2022", "end": None}
        ],
    }
]

MULTIPLE_PROFILES = SINGLE_PROFILE + SHARED_CONNECTION


def test_format_contains_name():
    result = format_profiles_for_prompt(SINGLE_PROFILE)
    assert "Marko Petrovic" in result


def test_format_contains_owner():
    result = format_profiles_for_prompt(SINGLE_PROFILE)
    assert "Aleksandar" in result


def test_format_contains_company():
    result = format_profiles_for_prompt(SINGLE_PROFILE)
    assert "N26" in result


def test_format_contains_linkedin_url():
    result = format_profiles_for_prompt(SINGLE_PROFILE)
    assert "https://linkedin.com/in/marko-petrovic" in result


def test_format_empty_list():
    result = format_profiles_for_prompt([])
    assert "No relevant profiles found" in result


def test_format_multiple_owners():
    result = format_profiles_for_prompt(SHARED_CONNECTION)
    assert "Mihajlo" in result
    assert "Petar" in result


def test_format_multiple_profiles():
    result = format_profiles_for_prompt(MULTIPLE_PROFILES)
    assert "Marko Petrovic" in result
    assert "Ana Anic" in result


def test_format_experience():
    result = format_profiles_for_prompt(SINGLE_PROFILE)
    assert "Senior PM" in result


def test_format_skills():
    result = format_profiles_for_prompt(SINGLE_PROFILE)
    assert "Fintech" in result


def test_synthesize_empty_profiles():
    """Should return a message when no profiles are found, without calling LLM."""
    result = synthesize(query="anyone in Stripe?", profiles=[])
    assert isinstance(result, str)
    assert len(result) > 0