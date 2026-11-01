import json

import pytest

from app.settings import settings
from rag.query_planner import (
    QueryPlanner,
    detect_response_language,
    normalize_retrieval_query,
)
from rag.query_router import QueryRouter


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt, system_prompt, query_type="general"):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "query_type": query_type})
        return self.response


@pytest.fixture(scope="module")
def router():
    return QueryRouter(settings.database_path)


def test_planner_validates_entities_and_canonicalizes_kanda(router):
    client = FakeLLMClient(json.dumps({
        "type": "count",
        "target": "verses_in_sarga",
        "parameters": {"kanda": "Sundra Kanda", "sarga": 3},
        "retrieval_query": "Tell me about Sundra Kanda Sarga 3",
        "response_language": "en",
        "needs_clarification": False,
        "clarification_question": None,
    }))

    result = QueryPlanner(client, router).plan(
        "How many verses does Sundra Kanda Sarga 3 have?", "auto"
    )

    assert result is not None
    assert result.analysis is not None
    assert result.analysis.type == "count"
    assert result.analysis.target == "verses_in_sarga"
    assert result.analysis.parameters == {"kanda": "Sundara Kanda", "sarga": 3}
    assert result.retrieval_query == "Sundra Kanda Sarga 3"
    assert result.to_provenance()["llm_used_for_planning"] is True


def test_planner_requests_clarification_for_kanda_outside_dataset(router):
    client = FakeLLMClient(json.dumps({
        "type": "count",
        "target": "verses_in_kanda",
        "parameters": {"kanda": "Uttara Kanda"},
        "retrieval_query": "Uttara Kanda verse count",
        "response_language": "en",
        "needs_clarification": False,
        "clarification_question": None,
    }))

    result = QueryPlanner(client, router).plan(
        "How many verses are in Uttara Kanda?", "auto"
    )

    assert result is not None
    assert result.needs_clarification is True
    assert "Available Kandas" in result.clarification_question
    assert result.analysis is None


def test_malformed_llm_plan_falls_back_without_crashing(router):
    result = QueryPlanner(FakeLLMClient("Not valid JSON"), router).plan(
        "Tell me about Rama's exile", "auto"
    )
    assert result is None


def test_explicit_language_preference_wins_over_model_output(router):
    client = FakeLLMClient(json.dumps({
        "type": "general",
        "target": None,
        "parameters": {},
        "retrieval_query": "Rama exile",
        "response_language": "en",
        "needs_clarification": False,
        "clarification_question": None,
    }))

    result = QueryPlanner(client, router).plan(
        "Tell me about Rama's exile", "hi"
    )

    assert result is not None
    assert result.response_language == "hi"


def test_language_detection_supports_hindi_and_hinglish():
    assert detect_response_language("auto", "राम वन क्यों गए?") == "hi"
    assert detect_response_language("auto", "Rama van mein kyun gaye") == "hinglish"
    assert detect_response_language("en", "राम वन क्यों गए?") == "en"


def test_retrieval_query_removes_conversational_scaffolding_only():
    assert normalize_retrieval_query("Tell me about Rama's exile") == "Rama exile"
    assert normalize_retrieval_query("Can you explain Sita's abduction?") == "Sita abduction"
    assert normalize_retrieval_query("Rama's exile") == "Rama exile"


def test_auto_language_uses_detected_query_language_not_model_mislabelling(router):
    client = FakeLLMClient(json.dumps({
        "type": "general",
        "target": None,
        "parameters": {},
        "retrieval_query": "Rama van mein kyun gaye",
        "response_language": "en",
        "needs_clarification": False,
        "clarification_question": None,
    }))
    result = QueryPlanner(client, router).plan("Rama van mein kyun gaye?", "auto")
    assert result is not None
    assert result.response_language == "hinglish"


def test_planner_rejects_non_boolean_clarification_flag(router):
    client = FakeLLMClient(json.dumps({
        "type": "general",
        "target": None,
        "parameters": {},
        "retrieval_query": "Rama exile",
        "response_language": "en",
        "needs_clarification": "false",
        "clarification_question": None,
    }))
    assert QueryPlanner(client, router).plan("Tell me about Rama's exile") is None
