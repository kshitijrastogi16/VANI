"""
VANI V3 Search API Routes - FIXED VERSION
Enforcing: Proper async patterns, error handling, and input validation
"""

import asyncio
import time
import re
import logging
from typing import List, Dict, Any, Optional, Literal
from fastapi import APIRouter, HTTPException, Depends, Query, Request, Path
from pydantic import BaseModel, Field, validator
from slowapi import _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from app.dependencies import get_database_status, get_embeddings_status
from app.settings import settings
from core.embeddings.storage import VectorStore
from core.search.hybrid_engine import HybridSearchEngine
from core.entities.resolver import EntityResolver
from core.search.vector_search import VectorSearchEngine
from rag.query_router import QueryRouter
from rag.synthesizer import ResponseSynthesizer
from app.ratelimit import limiter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])

# Rate limiter comes from app.ratelimit (shared instance)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    # Use Literal for strict allowed values (better compatibility with pydantic v2)
    search_type: Literal["hybrid", "bm25", "vector", "exact"] = "hybrid"
    max_results: int = Field(default=10, ge=1, le=50)
    include_metadata: bool = Field(default=True)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @validator('query')
    def sanitize_query(cls, v):
        """Sanitize search query to prevent injection"""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>;&|`$]', '', v.strip())
        
        if len(sanitized) != len(v.strip()):
            logger.warning(f"Query sanitized: '{v}' -> '{sanitized}'")
        
        return sanitized

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int
    search_type: str
    query_time_ms: float
    metadata: Dict[str, Any]

@router.post("/", response_model=SearchResponse)
@limiter.limit("30/minute")  # Rate limiting
async def search_verses(
    request: Request,
    search_request: SearchRequest,
    db_status: bool = Depends(get_database_status),
    embeddings_status: bool = Depends(get_embeddings_status)
):
    """
    Main search endpoint with proper async handling
    """
    
    if not db_status:
        raise HTTPException(status_code=503, detail="Database not available")
    
    start_time = time.time()
    
    try:
        # Execute search without loading large embedding models on free hosting
        vector_search_enabled = (
            embeddings_status and not settings.disable_vector_search
        )

        if search_request.search_type == "bm25":
            results = await asyncio.to_thread(
                _execute_bm25_search, search_request
            )

        elif search_request.search_type == "vector":
            if not vector_search_enabled:
                raise HTTPException(
                    status_code=503,
                    detail="Vector search is disabled on this deployment."
                )
            results = await asyncio.to_thread(
                _execute_vector_search, request, search_request
            )

        elif search_request.search_type == "exact":
            results = await asyncio.to_thread(
                _execute_exact_search, search_request
            )

        else:  # hybrid
            if not vector_search_enabled:
                results = await asyncio.to_thread(
                    _execute_bm25_search, search_request
                )
            else:
                results = await asyncio.to_thread(
                    _execute_hybrid_search, request, search_request
                )
        
        # Apply score filtering
        if search_request.min_score is not None:
            results = [r for r in results if r.get('score', 0) >= search_request.min_score]
        
        # Limit results
        results = results[:search_request.max_results]
        
        query_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=results,
            total_found=len(results),
            search_type=search_request.search_type,
            query_time_ms=round(query_time, 2),
            metadata={
                "database_available": db_status,
                "embeddings_available": embeddings_status,
                "query": search_request.query,
                "client_ip": get_remote_address(request)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal search error")

@router.get("/suggest")
@limiter.limit("60/minute")
async def get_suggestions(
    request: Request,
    q: str = Query(..., min_length=1, max_length=100, description="Partial query"),
    limit: int = Query(default=5, ge=1, le=10, description="Max suggestions")
):
    """Get search suggestions with rate limiting"""
    
    try:
        # Sanitize input
        clean_query = re.sub(r'[^\w\s]', '', q).strip()
        if not clean_query:
            return {"suggestions": [], "query": q}
        
        # Use default thread executor
        suggestions = await asyncio.to_thread(_get_search_suggestions, clean_query, limit)
        
        return {
            "suggestions": suggestions,
            "query": q
        }
        
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        return {"suggestions": [], "query": q}

@router.get("/verse/{verse_id}")
@limiter.limit("60/minute")
async def get_verse_by_id(
    request: Request,
    # Accept hyphen or dot style: 1-2-3 or 1.2.3, with optional -V2 range
    verse_id: str = Path(..., pattern=r"^\d+(?:[.-]\d+){2}(?:-\d+)?$"),
    include_context: bool = Query(default=True)
):
    """Get specific verse by ID with proper validation"""
    
    try:
        # Normalize input (dot -> hyphen) and validate
        normalized_id = verse_id.replace('.', '-')
        if not re.match(r'^\d+-\d+-\d+(?:-\d+)?$', normalized_id):
            raise HTTPException(status_code=400, detail="Invalid verse ID format")
        
        # Use default thread executor
        verse = await asyncio.to_thread(_get_verse_by_id, normalized_id)
        # Optionally fetch context window via QueryRouter's citation handler
        sources = []
        context_summary = ""
        if verse and include_context:
            citation_result = await asyncio.to_thread(_get_citation_context, normalized_id)
            if isinstance(citation_result, dict):
                all_sources = citation_result.get("sources", [])
                # Drop the main verse; show only neighbor verses in context
                sources = all_sources[1:] if len(all_sources) > 1 else []
                try:
                    # Build a context summary using summarizer (LLM if available)
                    syn = ResponseSynthesizer(getattr(getattr(request.app, 'state', object()), 'llm_client', None))
                    # IMPORTANT: Pass the full list (main verse first) for better summarization
                    context_summary = syn.summarize_citation_context({"sources": all_sources})
                except Exception:
                    context_summary = ""
        
        if not verse:
            raise HTTPException(status_code=404, detail=f"Verse {verse_id} not found")
        
        return {
            "verse": verse,
            # Return normalized database-style ID here; clients can render to dot-style if desired
            "verse_id": normalized_id,
            "context_included": include_context,
            "sources": sources if include_context else [],
            "context_summary": context_summary if include_context else ""
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verse lookup error: {e}")
        raise HTTPException(status_code=500, detail="Verse lookup failed")

@router.post("/diagnostics")
@limiter.limit("30/minute")
async def search_diagnostics(
    request: Request,
    search_request: SearchRequest,
    db_status: bool = Depends(get_database_status),
    embeddings_status: bool = Depends(get_embeddings_status)
):
    """Diagnostics endpoint to inspect expanded query and search stats for a given query.
    Useful for validating entity expansion, alias/context usage, and hybrid fusion.
    """
    if not db_status:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        diag = await asyncio.to_thread(_execute_diagnostics, request, search_request, embeddings_status)
        return diag
    except Exception as e:
        logger.error(f"Diagnostics error: {e}")
        raise HTTPException(status_code=500, detail="Diagnostics failed")

# Helper functions (executed in thread pool)
def _execute_bm25_search(request: SearchRequest) -> List[Dict]:
    """Execute BM25 search safely"""
    try:
        from core.search.text_search import search_bm25
        return search_bm25(
            db_path=settings.database_path,
            fts_table=settings.fts_table,
            query=request.query,
            limit=request.max_results,
            min_score=request.min_score
        )
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        return []

def _execute_vector_search(req_request: Request, request: SearchRequest) -> List[Dict]:
    """Execute vector search safely"""
    try:
        # Prefer app-initialized vector store for efficiency
        vector_store = getattr(getattr(req_request, 'app', object()), 'state', None)
        vector_store = getattr(vector_store, 'vector_store', None)
        if not vector_store:
            vector_store = VectorStore(
                embeddings_dir=settings.embeddings_dir,
                primary_model=settings.primary_embedding_model,
                fallback_model=settings.fallback_embedding_model,
                tertiary_model=getattr(settings, 'tertiary_embedding_model', None)
            )
            vector_store.load()
        vec_engine = VectorSearchEngine(vector_store)
        vresults = vec_engine.search(request.query, top_k=request.max_results)
        # Map VectorSearchResult -> dicts compatible with API schema
        results: List[Dict] = []
        for vr in vresults:
            item = {
                'verse_id': vr.verse_id,
                'score': vr.similarity_score,
                'search_type': 'vector_similarity'
            }
            if isinstance(vr.metadata, dict):
                item.update(vr.metadata)
            results.append(item)
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

def _execute_exact_search(request: SearchRequest) -> List[Dict]:
    """Execute exact search safely"""
    try:
        from core.search.text_search import search_exact_verse
        
        # Check if query looks like verse ID
        if re.match(r'^\d+-\d+-\d+(?:-\d+)?$', request.query):
            verse = search_exact_verse(settings.database_path, request.query)
            return [verse] if verse else []
        
        # Otherwise exact phrase search
        from core.search.text_search import search_bm25
        return search_bm25(
            db_path=settings.database_path,
            fts_table=settings.fts_table,
            query=f'"{request.query}"',  # Exact phrase
            limit=request.max_results
        )
    except Exception as e:
        logger.error(f"Exact search failed: {e}")
        return []

def _execute_hybrid_search(req_request: Request, request: SearchRequest) -> List[Dict]:
    """Execute hybrid search safely"""
    try:
        # Prepare vector store (reuse app state when available)
        vector_store = getattr(getattr(req_request, 'app', object()), 'state', None)
        vector_store = getattr(vector_store, 'vector_store', None)
        if not vector_store:
            vector_store = VectorStore(
                embeddings_dir=settings.embeddings_dir,
                primary_model=settings.primary_embedding_model,
                fallback_model=settings.fallback_embedding_model,
                tertiary_model=getattr(settings, 'tertiary_embedding_model', None)
            )
            vector_store.load()

        # Prepare entity resolver (reuse app.state if available)
        state_obj = getattr(req_request, 'app', object())
        state_obj = getattr(state_obj, 'state', None)
        entity_resolver = getattr(state_obj, 'entity_resolver', None) if state_obj else None
        if not entity_resolver:
            entity_resolver = EntityResolver(
                characters_path=settings.characters_path,
                locations_path=settings.locations_path,
                confidence_threshold=settings.entity_confidence_threshold
            )

        # Initialize hybrid engine
        engine = HybridSearchEngine(
            db_path=settings.database_path,
            vector_store=vector_store,
            entity_resolver=entity_resolver,
            bm25_weight=settings.bm25_weight,
            vector_weight=settings.vector_weight,
            fts_table=settings.fts_table
        )

        return engine.search(request.query, top_k=request.max_results)
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        # Fallback to BM25
        return _execute_bm25_search(request)

def _execute_diagnostics(req_request: Request, request: SearchRequest, embeddings_status: bool) -> Dict[str, Any]:
    """Execute a RAG retrieval and return diagnostics including expanded_query and stats."""
    try:
        # Prepare vector store (reuse app state when available)
        vector_store = getattr(getattr(req_request, 'app', object()), 'state', None)
        vector_search_enabled = (
            embeddings_status and not settings.disable_vector_search
        )

        vector_store = (
            getattr(vector_store, 'vector_store', None)
            if vector_search_enabled else None
        )

        if vector_search_enabled and not vector_store:
            vector_store = VectorStore(
                embeddings_dir=settings.embeddings_dir,
                primary_model=settings.primary_embedding_model,
                fallback_model=settings.fallback_embedding_model,
                tertiary_model=getattr(settings, 'tertiary_embedding_model', None)
            )
            vector_store.load()

        # Prepare entity resolver (reuse app.state if available)
        state_obj = getattr(req_request, 'app', object())
        state_obj = getattr(state_obj, 'state', None)
        entity_resolver = getattr(state_obj, 'entity_resolver', None) if state_obj else None
        if not entity_resolver:
            entity_resolver = EntityResolver(
                characters_path=settings.characters_path,
                locations_path=settings.locations_path,
                confidence_threshold=settings.entity_confidence_threshold
            )

        # Initialize hybrid engine
        engine = HybridSearchEngine(
            db_path=settings.database_path,
            vector_store=vector_store,
            entity_resolver=entity_resolver,
            bm25_weight=settings.bm25_weight,
            vector_weight=settings.vector_weight,
            fts_table=settings.fts_table
        )

        # Initialize retriever
        from rag.retriever import RAGRetriever
        from rag.query_router import QueryRouter
        retriever = RAGRetriever(
            search_engine=engine,
            entity_resolver=entity_resolver,
            relationship_graph=getattr(state_obj, 'relationship_graph', None) if state_obj else None,
            confidence_threshold=settings.entity_confidence_threshold,
            router=QueryRouter(settings.database_path)
        )

        # Run retrieval with entity boost preference for general queries as configured
        ret = retriever.retrieve(
            query=request.query,
            top_k=max(10, request.max_results),
            include_context=True,
            entity_boost=getattr(settings, 'entity_boost_for_general', False)
        )

        # Build diagnostics payload
        entities = [
            {"canonical_name": e.canonical_name, "confidence": e.confidence, "type": e.entity_type}
            for e in (ret.entities_found or [])
        ]
        top1 = float(ret.results[0].final_score) if ret.results else 0.0
        stats = engine.get_search_stats()
        return {
            "query": request.query,
            "expanded_query": ret.expanded_query or request.query,
            "entities": entities,
            "candidate_count": len(ret.results or []),
            "fused_top1_score": top1,
            "search_engine": stats,
            "settings": {
                "entity_boost_for_general": getattr(settings, 'entity_boost_for_general', False),
                "entity_expansion_router_confidence": getattr(settings, 'entity_expansion_router_confidence', 0.8),
                "hybrid_alias_limit": getattr(settings, 'hybrid_alias_limit', 1),
                "enable_entity_context_in_hybrid": getattr(settings, 'enable_entity_context_in_hybrid', False),
                "hybrid_context_limit": getattr(settings, 'hybrid_context_limit', 0),
                "retriever_identity_alias_limit": getattr(settings, 'retriever_identity_alias_limit', 1),
                "retriever_alias_limit": getattr(settings, 'retriever_alias_limit', 2),
                "english_only_query_expansion": getattr(settings, 'english_only_query_expansion', True),
                "enable_description_expansion": getattr(settings, 'enable_description_expansion', False),
            }
        }
    except Exception as e:
        logger.error(f"Diagnostics internal failure: {e}")
        return {
            "query": request.query,
            "expanded_query": request.query,
            "entities": [],
            "candidate_count": 0,
            "fused_top1_score": 0.0,
            "search_engine": {"error": str(e)}
        }

def _get_search_suggestions(query: str, limit: int) -> List[str]:
    """Get search suggestions safely"""
    try:
        from core.search.text_search import get_search_suggestions
        return get_search_suggestions(settings.database_path, query, limit)
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        return []

def _get_verse_by_id(verse_id: str) -> Optional[Dict]:
    """Get verse by ID safely"""
    try:
        from core.search.text_search import search_exact_verse
        return search_exact_verse(settings.database_path, verse_id)
    except Exception as e:
        logger.error(f"Verse lookup failed: {e}")
        return None

def _get_citation_context(verse_id: str) -> Dict:
    """Use QueryRouter to build verse + context window as sources."""
    try:
        router = QueryRouter(settings.database_path)
        analysis = router.analyze_query(verse_id)
        if analysis and analysis.type == "citation":
            return router.handle_citation_query(analysis)
        # Fallback: force citation analysis if analyzer didn't catch it
        from rag.query_router import QueryAnalysis  # local import to avoid top-level coupling
        forced = QueryAnalysis(
            type="citation", confidence=1.0, target="verse_lookup",
            parameters={"verse_id": verse_id}, requires_llm=False
        )
        return router.handle_citation_query(forced)
    except Exception as e:
        logger.error(f"Citation context retrieval failed: {e}")
        return {"sources": []}
