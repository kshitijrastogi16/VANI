from app.settings import settings
from core.search.text_search import search_bm25
from rag.query_planner import normalize_retrieval_query


def test_conversational_query_returns_relevant_exile_passages_in_top_five():
    query = normalize_retrieval_query("Tell me about Rama's exile")
    results = search_bm25(settings.database_path, settings.fts_table, query, limit=5)
    retrieved_ids = [row.get("verse_id") for row in results]

    # These IDs were verified in the supplied corpus and explicitly discuss Rama's exile.
    assert "2-12-101" in retrieved_ids
    assert "2-12-93" in retrieved_ids
    assert "2-38-10" in retrieved_ids


def test_kaikeyi_exile_query_retrieves_the_demands_passage():
    query = normalize_retrieval_query("What demands did Kaikeyi make regarding Rama's exile?")
    results = search_bm25(settings.database_path, settings.fts_table, query, limit=5)
    retrieved_ids = [row.get("verse_id") for row in results]

    assert "2-12-51" in retrieved_ids


def test_hybrid_retrieval_preserves_topical_intent_when_entity_boost_enabled(monkeypatch):
    from core.entities.resolver import EntityResolver
    from core.search.hybrid_engine import HybridSearchEngine
    from rag.retriever import RAGRetriever

    monkeypatch.setattr(settings, "entity_boost_for_general", True)
    resolver = EntityResolver(
        characters_path=settings.characters_path,
        locations_path=settings.locations_path,
        confidence_threshold=settings.entity_confidence_threshold,
    )
    engine = HybridSearchEngine(
        db_path=settings.database_path,
        vector_store=None,
        entity_resolver=resolver,
        bm25_weight=settings.bm25_weight,
        vector_weight=settings.vector_weight,
        fts_table=settings.fts_table,
    )
    retriever = RAGRetriever(
        search_engine=engine,
        entity_resolver=resolver,
    )

    result = retriever.retrieve("Rama exile", top_k=5, entity_boost=True)
    ids = [doc.verse_id for doc in result.results]

    # The topical term "exile" must not be erased by a broad OR expansion to
    # generic Rama aliases, which previously put unrelated Yuddha Kanda passages first.
    assert ids[0] == "2-12-101"
    assert {"2-12-101", "2-38-10", "2-57-15", "2-25-43", "2-12-93"}.issubset(set(ids))
    assert "6-102-43" not in ids


def test_hybrid_retrieval_ranks_kaikeyis_exile_demands_relevantly(monkeypatch):
    from core.entities.resolver import EntityResolver
    from core.search.hybrid_engine import HybridSearchEngine
    from rag.retriever import RAGRetriever

    monkeypatch.setattr(settings, "entity_boost_for_general", True)
    resolver = EntityResolver(
        characters_path=settings.characters_path,
        locations_path=settings.locations_path,
        confidence_threshold=settings.entity_confidence_threshold,
    )
    engine = HybridSearchEngine(
        db_path=settings.database_path,
        vector_store=None,
        entity_resolver=resolver,
        bm25_weight=settings.bm25_weight,
        vector_weight=settings.vector_weight,
        fts_table=settings.fts_table,
    )
    retriever = RAGRetriever(search_engine=engine, entity_resolver=resolver)

    result = retriever.retrieve(
        "What demands did Kaikeyi make regarding Rama exile?",
        top_k=5,
        entity_boost=True,
    )
    ids = [doc.verse_id for doc in result.results]
    assert "2-12-51" in ids
    assert ids[0] == "2-12-51"


def test_fts5_bm25_similarity_preserves_higher_relevance_for_more_negative_scores():
    from core.search.text_search import _bm25_similarity

    assert _bm25_similarity(-22.0) > _bm25_similarity(-12.0)
    assert 0.0 < _bm25_similarity(-22.0) < 1.0
    assert _bm25_similarity(0.0) == 0.0
