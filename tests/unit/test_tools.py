"""
Unit tests for src/tools/tools.py
Mocks get_connections() and semantic_query() so no real DB or vector store is needed.
"""
from unittest.mock import patch, MagicMock


# ---- Mock helpers ----

def make_profile(first_name, last_name, linkedin_url, city="Belgrade", country="Serbia",
                 owners=None, skills=None, positions=None, education=None):
    """Creates a mock Profile ORM object."""
    p = MagicMock()
    p.first_name = first_name
    p.last_name = last_name
    p.linkedin_url = linkedin_url
    p.city = city
    p.country = country
    p.owners = owners or ["Aleksandar"]
    p.skills = skills or []

    pos_list = []
    for pos in (positions or []):
        m = MagicMock()
        m.title = pos.get("title")
        m.company_name = pos.get("company_name")
        m.company_location = pos.get("company_location")
        pos_list.append(m)
    p.positions = pos_list

    edu_list = []
    for edu in (education or []):
        m = MagicMock()
        m.degree = edu.get("degree")
        m.school_name = edu.get("school_name")
        edu_list.append(m)
    p.education = edu_list

    return p


def make_semantic_result(first_name, last_name, linkedin_url, headline="", summary="", owners=None):
    return {
        "first_name": first_name,
        "last_name": last_name,
        "linkedin_url": linkedin_url,
        "headline": headline,
        "summary": summary,
        "owners": owners or ["Aleksandar"],  # lista, ne string
    }


MOCK_DB_PROFILES = [
    make_profile("Marko", "Petrovic", "https://linkedin.com/in/marko-petrovic",
                 owners=["Aleksandar"],
                 positions=[{"title": "Senior PM", "company_name": "N26", "company_location": "Berlin"}]),
    make_profile("Ana", "Anic", "https://linkedin.com/in/ana-anic",
                 owners=["Mihajlo", "Petar"],
                 positions=[{"title": "Engineering Lead", "company_name": "Solarisbank", "company_location": "Berlin"}]),
]

MOCK_SEMANTIC_RESULTS = {
    "metadatas": [[
        make_semantic_result("Stefan", "Jovanovic", "https://linkedin.com/in/stefan-jovanovic",
                             headline="ML Engineer", summary="ML experience at Google", owners=["Petar"]),
        make_semantic_result("Jelena", "Markovic", "https://linkedin.com/in/jelena-markovic",
                             headline="VC Investor", summary="B2B SaaS investor", owners=["Aleksandar"]),
    ]]
}


# ---- Tests for structured_filter ----

@patch("src.tools.tools.Session")
@patch("src.tools.tools.get_connections")
def test_structured_filter_returns_profiles(mock_get_connections, mock_session):
    mock_get_connections.return_value = MOCK_DB_PROFILES
    mock_session.return_value.__enter__ = lambda s: MagicMock()
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.tools.tools import structured_filter
    result = structured_filter.invoke({"city": ["Belgrade"]})

    assert isinstance(result, list)
    assert len(result) == 2


@patch("src.tools.tools.Session")
@patch("src.tools.tools.get_connections")
def test_structured_filter_profile_has_required_fields(mock_get_connections, mock_session):
    mock_get_connections.return_value = [MOCK_DB_PROFILES[0]]
    mock_session.return_value.__enter__ = lambda s: MagicMock()
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.tools.tools import structured_filter
    result = structured_filter.invoke({})

    p = result[0]
    assert "name" in p
    assert "linkedin_url" in p
    assert "owners" in p
    assert "positions" in p
    assert "education" in p


@patch("src.tools.tools.Session")
@patch("src.tools.tools.get_connections")
def test_structured_filter_owners_preserved(mock_get_connections, mock_session):
    mock_get_connections.return_value = [MOCK_DB_PROFILES[1]]
    mock_session.return_value.__enter__ = lambda s: MagicMock()
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.tools.tools import structured_filter
    result = structured_filter.invoke({})

    assert result[0]["owners"] == ["Mihajlo", "Petar"]


@patch("src.tools.tools.Session")
@patch("src.tools.tools.get_connections")
def test_structured_filter_empty_result(mock_get_connections, mock_session):
    mock_get_connections.return_value = []
    mock_session.return_value.__enter__ = lambda s: MagicMock(return_value=False)
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.tools.tools import structured_filter
    result = structured_filter.invoke({"city": ["Tokyo"]})

    assert result == []


@patch("src.tools.tools.Session")
@patch("src.tools.tools.get_connections")
def test_structured_filter_positions_mapped(mock_get_connections, mock_session):
    mock_get_connections.return_value = [MOCK_DB_PROFILES[0]]
    mock_session.return_value.__enter__ = lambda s: MagicMock()
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    from src.tools.tools import structured_filter
    result = structured_filter.invoke({})

    positions = result[0]["positions"]
    assert len(positions) == 1
    assert positions[0]["title"] == "Senior PM"
    assert positions[0]["company_name"] == "N26"


# ---- Tests for semantic_search ----

@patch("src.tools.tools.semantic_query")
def test_semantic_search_returns_profiles(mock_semantic_query):
    mock_semantic_query.return_value = MOCK_SEMANTIC_RESULTS

    from src.tools.tools import semantic_search
    result = semantic_search.invoke({"query": "ML engineer"})

    assert isinstance(result, list)
    assert len(result) == 2


@patch("src.tools.tools.semantic_query")
def test_semantic_search_profile_has_required_fields(mock_semantic_query):
    mock_semantic_query.return_value = MOCK_SEMANTIC_RESULTS

    from src.tools.tools import semantic_search
    result = semantic_search.invoke({"query": "investor"})

    p = result[0]
    assert "name" in p
    assert "linkedin_url" in p
    assert "owners" in p
    assert "headline" in p
    assert "summary" in p


@patch("src.tools.tools.semantic_query")
def test_semantic_search_name_constructed(mock_semantic_query):
    mock_semantic_query.return_value = MOCK_SEMANTIC_RESULTS

    from src.tools.tools import semantic_search
    result = semantic_search.invoke({"query": "ML"})

    assert result[0]["name"] == "Stefan Jovanovic"


@patch("src.tools.tools.semantic_query")
def test_semantic_search_owners_is_list(mock_semantic_query):
    """owners field must be a list, not a string."""
    mock_semantic_query.return_value = MOCK_SEMANTIC_RESULTS

    from src.tools.tools import semantic_search
    result = semantic_search.invoke({"query": "ML"})

    assert isinstance(result[0]["owners"], list)


# ---- Tests for hybrid_search ----

@patch("src.tools.tools.semantic_query")
@patch("src.tools.tools.Session")
@patch("src.tools.tools.get_connections")
def test_hybrid_search_deduplication(mock_get_connections, mock_session, mock_semantic_query):
    """Same profile from both sources should appear only once."""
    mock_get_connections.return_value = [MOCK_DB_PROFILES[0]]
    mock_session.return_value.__enter__ = lambda s: MagicMock()
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    mock_semantic_query.return_value = {"metadatas": [[
        make_semantic_result("Marko", "Petrovic", "https://linkedin.com/in/marko-petrovic",
                             headline="Senior PM", owners=["Aleksandar"])
    ]]}

    from src.tools.tools import hybrid_search
    result = hybrid_search.invoke({"filters": {}, "query_text": "PM fintech"})

    urls = [p["linkedin_url"] for p in result]
    assert len(urls) == len(set(urls)), "Duplicate profiles found in hybrid_search result"


@patch("src.tools.tools.semantic_query")
@patch("src.tools.tools.Session")
@patch("src.tools.tools.get_connections")
def test_hybrid_search_enriches_with_headline(mock_get_connections, mock_session, mock_semantic_query):
    """Structured profile should get headline/summary from semantic match."""
    mock_get_connections.return_value = [MOCK_DB_PROFILES[0]]
    mock_session.return_value.__enter__ = lambda s: MagicMock()
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    mock_semantic_query.return_value = {"metadatas": [[
        make_semantic_result("Marko", "Petrovic", "https://linkedin.com/in/marko-petrovic",
                             headline="Senior PM at N26", summary="Fintech expert", owners=["Aleksandar"])
    ]]}

    from src.tools.tools import hybrid_search
    result = hybrid_search.invoke({"filters": {}, "query_text": "PM"})

    marko = next(p for p in result if "Marko" in p["name"])
    assert marko["headline"] == "Senior PM at N26"
    assert marko["summary"] == "Fintech expert"


@patch("src.tools.tools.semantic_query")
@patch("src.tools.tools.Session")
@patch("src.tools.tools.get_connections")
def test_hybrid_search_combines_both_sources(mock_get_connections, mock_session, mock_semantic_query):
    """Hybrid should return profiles from both structured and semantic."""
    mock_get_connections.return_value = [MOCK_DB_PROFILES[0]]
    mock_session.return_value.__enter__ = lambda s: MagicMock()
    mock_session.return_value.__exit__ = MagicMock(return_value=False)

    mock_semantic_query.return_value = {"metadatas": [[
        make_semantic_result("Stefan", "Jovanovic", "https://linkedin.com/in/stefan-jovanovic",
                             headline="ML Engineer", owners=["Petar"])
    ]]}

    from src.tools.tools import hybrid_search
    result = hybrid_search.invoke({"filters": {}, "query_text": "engineer"})

    names = [p["name"] for p in result]
    assert any("Marko" in n for n in names)
    assert any("Stefan" in n for n in names)


# ---- Tests for count_matches ----

def test_count_matches_correct():
    from src.tools.tools import count_matches
    result = count_matches.invoke({"results": [{"name": "A"}, {"name": "B"}, {"name": "C"}]})
    assert result == 3


def test_count_matches_empty():
    from src.tools.tools import count_matches
    result = count_matches.invoke({"results": []})
    assert result == 0