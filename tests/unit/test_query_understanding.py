from src.agents.query_understanding import (
    understand,
    UserQuery,
    QueryType,
    LookupAttributes,
    ConnectionFilters,
    PositionFilters,
    EducationFilters,
)



def test_query_type_values():
    assert QueryType.LOOKUP == "LOOKUP"
    assert QueryType.DISCOVERY == "DISCOVERY"
    assert QueryType.HYBRID == "HYBRID"


def test_user_query_defaults():
    query = UserQuery(
        lookup_filters=LookupAttributes(),
        contextual_need=None,
        contextual_trigger=None,
        query_type=QueryType.DISCOVERY,
    )
    assert query.contextual_need is None
    assert query.contextual_trigger is None


def test_lookup_attributes_all_none():
    filters = LookupAttributes(
        connection=None,
        position=None,
        education=None,
        owner=None,
    )
    assert filters.connection is None
    assert filters.position is None


def test_connection_filters():
    conn = ConnectionFilters(country=["Serbia"], city=["Belgrade"], skills=["Python"])
    assert "Serbia" in conn.country
    assert "Belgrade" in conn.city
    assert "Python" in conn.skills


def test_position_filters_recently_changed():
    pos = PositionFilters(recently_changed=True)
    assert pos.recently_changed is True


def test_education_filters():
    edu = EducationFilters(degree=["PhD"], school_name=["ETH Zurich"])
    assert "PhD" in edu.degree
    assert "ETH Zurich" in edu.school_name



def test_understand_simple_company_lookup():
    result = understand("Who in our network works at Stripe?")
    assert isinstance(result, UserQuery)
    assert result.query_type == QueryType.LOOKUP
    assert result.lookup_filters.position is not None
    assert result.lookup_filters.position.company_name is not None
    assert any("Stripe" in c for c in result.lookup_filters.position.company_name)


def test_understand_discovery_query():
    result = understand("Who can open doors for us in the US market?")
    assert isinstance(result, UserQuery)
    assert result.query_type in [QueryType.DISCOVERY, QueryType.HYBRID]
    assert result.contextual_need is not None


def test_understand_hybrid_query():
    result = understand("I need a technical co-founder with ML experience based in London")
    assert isinstance(result, UserQuery)
    assert result.query_type == QueryType.HYBRID
    assert result.contextual_need is not None


def test_understand_returns_user_query_instance():
    result = understand("Find ML engineers in Berlin")
    assert isinstance(result, UserQuery)


def test_understand_location_lookup():
    result = understand("Who in our network is based in Berlin?")
    assert isinstance(result, UserQuery)
    assert result.lookup_filters.connection is not None
    assert result.lookup_filters.connection.city is not None
    assert any("Berlin" in c for c in result.lookup_filters.connection.city)


def test_understand_recently_changed_jobs():
    result = understand("Find connections who recently changed jobs")
    assert isinstance(result, UserQuery)
    assert result.lookup_filters.position is not None
    assert result.lookup_filters.position.recently_changed is True