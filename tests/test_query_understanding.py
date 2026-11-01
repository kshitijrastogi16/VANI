import pytest
from app.settings import settings
from rag.query_router import QueryRouter


@pytest.fixture(scope="module")
def router():
    return QueryRouter(settings.database_path)


@pytest.mark.parametrize(
    ("name", "expected_num", "expected_name"),
    [
        ("Bala", 1, "Baala Kanda"),
        ("Baala Kanda", 1, "Baala Kanda"),
        ("Ayodhya", 2, "Ayodhya Kanda"),
        ("Ayodhaya", 2, "Ayodhya Kanda"),
        ("Sundara", 5, "Sundara Kanda"),
        ("Sundar Kand", 5, "Sundara Kanda"),
        ("Kishkindha", 4, "Kishkindha Kanda"),
        ("Yudha", 6, "Yuddha Kanda"),
        ("Lanka", 6, "Yuddha Kanda"),
    ],
)
def test_known_kanda_aliases(router, name, expected_num, expected_name):
    kanda_num, canonical_name = router._normalize_kanda_identifier(name)
    assert kanda_num == expected_num
    assert canonical_name == expected_name


@pytest.mark.parametrize(
    ("question", "expected_target", "expected_parameters"),
    [
        (
            "How many verses are in Sarga 1 of Bala Kanda?",
            "verses_in_sarga",
            {"kanda": "bala", "sarga": 1},
        ),
        (
            "How many shlokas are in Sarga 1 of Bala Kanda?",
            "verses_in_sarga",
            {"kanda": "bala", "sarga": 1},
        ),
        (
            "How many verses does Sundara Kanda contain?",
            "verses_in_kanda",
            {"kanda": "sundara"},
        ),
        (
            "Tell me the verse count of Sarga 2 in Ayodhya Kanda.",
            "verses_in_sarga",
            {"kanda": "ayodhya", "sarga": 2},
        ),
    ],
)
def test_count_query_understanding(
    router, question, expected_target, expected_parameters
):
    analysis = router.analyze_query(question)

    assert analysis.type == "count"
    assert analysis.target == expected_target
    assert analysis.parameters == expected_parameters
    assert analysis.requires_llm is False


def test_unsupported_kanda_is_not_a_valid_database_entity(router):
    kanda_num, canonical_name = router._normalize_kanda_identifier("Uttara Kanda")

    assert kanda_num is None
    assert canonical_name is None


def test_invalid_sarga_returns_no_match(router):
    analysis = router.analyze_query(
        "How many verses are in Sarga 999 of Bala Kanda?"
    )

    assert analysis.type == "count"
    assert analysis.target == "verses_in_sarga"

    result = router.handle_count_query(analysis)

    assert result["count"] == 0
    assert "No match found" in result["answer"]
