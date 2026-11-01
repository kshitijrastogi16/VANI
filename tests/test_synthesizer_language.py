from rag.synthesizer import ResponseSynthesizer


def test_hindi_sarga_count_uses_hindi_kanda_name_and_is_database_backed():
    result = ResponseSynthesizer(None).synthesize_count_response(
        {
            "answer": "Sarga 2 of Ayodhya Kanda contains 54 verses.",
            "target": "verses_in_sarga",
            "count": 54,
            "kanda": "Ayodhya Kanda",
            "sarga": 2,
            "entity_resolved": True,
            "sarga_resolved": True,
        },
        response_language="hi",
    )
    assert result.answer == "अयोध्याकाण्ड के सर्ग 2 में 54 श्लोक हैं।"
    assert result.llm_used is False
    assert result.query_type == "count"


def test_hindi_invalid_sarga_is_clarification_not_zero_count():
    result = ResponseSynthesizer(None).synthesize_count_response(
        {
            "answer": "No match found.",
            "target": "verses_in_sarga",
            "count": 0,
            "kanda": "Baala Kanda",
            "sarga": 999,
            "entity_resolved": True,
            "sarga_resolved": False,
        },
        response_language="hi",
    )
    assert "बालकाण्ड" in result.answer
    assert "999" in result.answer
    assert result.query_type == "clarification"
    assert result.confidence == 0.0


def test_template_fallback_preserves_verifiable_source_citations():
    from types import SimpleNamespace

    source = SimpleNamespace(
        verse_id="2-12-101",
        kanda_name="Ayodhya Kanda",
        sarga_num=12,
        verse_index=101,
        translation_en="Whose are these dreadful words, this inconceivable plan of Rama's exile and Bharata's enthronement.",
        translation_hi="राम के वनवास और भरत के राज्याभिषेक की योजना सुनकर राजा चकित रह गए।",
        shloka_sanskrit=None,
        shloka_iast=None,
        sarga_titles=["King Dasaratha lamented."],
        final_score=0.69,
        verse_ids=[],
    )
    result = ResponseSynthesizer(None).synthesize(
        query="Tell me about Rama's exile",
        documents=[source],
        query_type="general",
        max_sources=1,
        response_language="en",
    )
    assert "2.12.101" in result.answer
    assert result.citations == ["2.12.101"]
    assert "most relevant reference is from" not in result.answer


def test_llm_citations_are_filtered_against_retrieved_source_ids():
    from types import SimpleNamespace
    import time

    source = SimpleNamespace(
        verse_id="2-12-101",
        kanda_name="Ayodhya Kanda",
        sarga_num=12,
        verse_index=101,
        translation_en="Rama's exile is discussed here.",
        translation_hi="राम के वनवास का वर्णन यहाँ है।",
        shloka_sanskrit=None,
        shloka_iast=None,
        sarga_titles=[],
        final_score=0.69,
        verse_ids=[],
    )
    result = ResponseSynthesizer(None)._process_llm_response(
        query="Rama exile",
        response_text="Rama's exile is discussed in 2.12.101 and 9.9.9.",
        documents=[source],
        query_type="general",
        start_time=time.time(),
        llm_used=True,
        response_language="en",
    )
    assert "2.12.101" in result.answer
    assert "9.9.9" not in result.answer
