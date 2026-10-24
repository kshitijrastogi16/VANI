"""
VANI V3 Main Application - FIXED VERSION
Enforcing: Production middleware, CORS, graceful shutdown, proper error handling
"""

import asyncio
import signal
import time
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, status, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
try:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from app.ratelimit import limiter, endpoint_rate_limit
except Exception:
    # Provide lightweight no-op fallbacks when SlowAPI is not available (e.g., test env)
    class RateLimitExceeded(Exception):
        pass
    def _rate_limit_exceeded_handler(request, exc):
        # Minimal handler; production uses SlowAPI implementation
        return Response(content="Rate limit exceeded", status_code=429)
    class _DummyLimiter:
        def limit(self, *args, **kwargs):
            def _decorator(func):
                return func
            return _decorator
    limiter = _DummyLimiter()
    def endpoint_rate_limit() -> str:
        return "100/minute"
    SlowAPIMiddleware = None
from pydantic import BaseModel, Field

from app.settings import settings
from app.dependencies import get_database_status, get_embeddings_status, get_llm_status
from api.routes import health, search
from utils.logging import setup_logging
from rag.citations import denormalize_verse_id
from app.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    RAG_QUERY_COUNT,
    RAG_DOCUMENTS_RETURNED,
    metrics_payload,
    CONTENT_TYPE_LATEST,
)

# Setup logging
setup_logging(settings.log_level, settings.log_file)
logger = logging.getLogger(__name__)

# Global shutdown handler
class GracefulShutdown:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.shutdown = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")

shutdown_handler = GracefulShutdown()

# Rate limiter comes from app.ratelimit (shared instance)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper startup/shutdown"""
    
    # Startup
    logger.info("🚀 VANI V3 starting up...")
    
    try:
        # Validate environment and data files unless explicitly skipped
        if getattr(settings, 'skip_startup_checks', False):
            logger.warning("⚠️  Skipping startup environment/data checks as configured (SKIP_STARTUP_CHECKS=true)")
        else:
            # Validate environment
            _validate_environment()
            # Validate data files
            _validate_data_files()
        
        # Initialize components
        await _initialize_components()
        
        logger.info("✅ VANI V3 startup complete")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🔄 VANI V3 shutting down...")
        await _cleanup_resources()
        logger.info("✅ VANI V3 shutdown complete")

# FastAPI app with lifespan management
app = FastAPI(
    title="VANI - Voice of Ancient Narrative Intelligence",
    description="Production RAG system for Ramayana question answering",
    version=settings.project_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=(
        ["localhost", "127.0.0.1", "*.local", "*.onrender.com"]
        if settings.debug
        else ["localhost", "127.0.0.1", "*.onrender.com"]
    ),
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=600
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
if SlowAPIMiddleware:
    app.add_middleware(SlowAPIMiddleware)

# Request size validation middleware
@app.middleware("http")
async def validate_request_size(request: Request, call_next):
    """Validate request size to prevent abuse"""
    
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=413, detail="Request too large")
    
    # Check for shutdown
    if shutdown_handler.shutdown:
        raise HTTPException(status_code=503, detail="Server shutting down")
    
    response = await call_next(request)
    return response

# Request ID middleware
@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    setattr(request.state, "request_id", request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Prometheus metrics middleware
@app.middleware("http")
async def prometheus_http_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    try:
        # Use route template (e.g., /search/verse/{verse_id}) to reduce label cardinality
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        method = request.method
        status_code = getattr(response, "status_code", 0)
        try:
            status_label = f"{int(status_code) // 100}xx"
        except Exception:
            status_label = str(status_code)
        REQUEST_COUNT.labels(method=method, path=path, status=status_label).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(time.time() - start)
    except Exception:
        # Metrics must never break the request path
        pass
    return response

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers"""
    
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Content Security Policy:
    # Inline scripts/styles have been moved to external files to enable a strict policy.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com; "
        "worker-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    
    if not settings.debug:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Static files and templates
app.mount("/static", StaticFiles(directory=settings.web_static_dir), name="static")
templates = Jinja2Templates(directory=settings.web_templates_dir)

# API routes
app.include_router(health.router)
app.include_router(search.router)

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    include_context: bool = Field(default=True)
    max_results: int = Field(default=10, ge=1, le=50)

class QueryResponse(BaseModel):
    success: bool
    answer: str
    sources: list
    citations: list
    confidence: float
    response_time: float
    query_type: str

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
@limiter.limit(endpoint_rate_limit)
async def query_endpoint(
    request: Request,
    query_request: QueryRequest,
    db_status: bool = Depends(get_database_status),
    embeddings_status: bool = Depends(get_embeddings_status),
    llm_status: bool = Depends(get_llm_status)
):
    """Main RAG query endpoint with comprehensive error handling"""
    
    start_time = time.time()
    
    try:
        # System availability check
        if not db_status:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Execute RAG pipeline using default thread executor
        result = await asyncio.to_thread(
            _execute_rag_pipeline, query_request, embeddings_status, llm_status
        )
        
        processing_time = time.time() - start_time
        # RAG metrics
        try:
            RAG_QUERY_COUNT.labels(
                query_type=result.get("query_type", "general"), success=True
            ).inc()
            RAG_DOCUMENTS_RETURNED.observe(len(result.get("sources", [])))
        except Exception:
            pass
        
        return QueryResponse(
            success=True,
            answer=result.get("answer", "No answer generated"),
            sources=result.get("sources", []),
            citations=result.get("citations", []),
            confidence=result.get("confidence", 0.0),
            response_time=round(processing_time, 3),
            query_type=result.get("query_type", "general")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        
        processing_time = time.time() - start_time
        try:
            RAG_QUERY_COUNT.labels(query_type="error", success=False).inc()
        except Exception:
            pass
        
        return QueryResponse(
            success=False,
            answer="I apologize, but I encountered an error while processing your question. Please try again.",
            sources=[],
            citations=[],
            confidence=0.0,
            response_time=round(processing_time, 3),
            query_type="error"
        )

# Web interface
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve main web interface"""
    
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        raise HTTPException(status_code=500, detail="Template error")

# Health check for load balancers
@app.get("/ping")
async def ping():
    """Simple ping endpoint for load balancers"""
    return {"status": "pong", "timestamp": time.time()}

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    return Response(content=metrics_payload(), media_type=CONTENT_TYPE_LATEST)

# Helper functions
def _validate_environment():
    """Validate required environment variables"""
    
    # LLM API key is optional; system can run without LLM (graceful degrade)
    required_vars = ["DATABASE_PATH", "EMBEDDINGS_DIR"]
    missing = [var for var in required_vars if not getattr(settings, var.lower(), None)]
    
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")
    
    if not getattr(settings, 'groq_api_key', None):
        logger.warning("GROQ_API_KEY not set; LLM features will be disabled.")

def _validate_data_files():
    """Validate required data files exist"""
    
    import os
    required_files = [
        settings.database_path,
        settings.characters_path,
        settings.locations_path
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"Missing required data files: {missing}")

async def _initialize_components():
    """Initialize system components"""
    
    try:
        # In test/dev mode, allow skipping heavy component initialization
        if getattr(settings, 'skip_startup_checks', False):
            logger.warning("⚠️  SKIP_STARTUP_CHECKS=true: Skipping DB/vector/entity/LLM initialization for fast startup")
            # Ensure attributes exist on app.state for handlers that reference them
            app.state.db_manager = None
            app.state.vector_store = None
            app.state.llm_client = None
            app.state.relationship_graph = None
            app.state.entity_resolver = None
            app.state.query_router = None
            return

        # Apply tuned weights if configured
        try:
            import os, json
            if getattr(settings, 'load_tuned_weights', True):
                path = getattr(settings, 'tuned_weights_path', '')
                if path and os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        tw = json.load(f)
                    bm = float(tw.get('bm25_weight', settings.bm25_weight))
                    vc = float(tw.get('vector_weight', settings.vector_weight))
                    setattr(settings, 'bm25_weight', bm)
                    setattr(settings, 'vector_weight', vc)
                    if isinstance(tw.get('bm25_column_weights'), list):
                        setattr(settings, 'bm25_column_weights', tw['bm25_column_weights'])
                    if 'mmr_lambda' in tw:
                        setattr(settings, 'enable_mmr_diversity', True)
                        setattr(settings, 'mmr_lambda', float(tw['mmr_lambda']))
                    if 'mmr_k' in tw:
                        setattr(settings, 'enable_mmr_diversity', True)
                        setattr(settings, 'mmr_k', int(tw['mmr_k']))
                    logger.info(f"✅ Loaded tuned weights from {path}")
        except Exception as e:
            logger.warning(f"Failed to load tuned weights: {e}")

        # Initialize database connection pool (global + app state)
        from core.database.connection import get_database_manager
        db_manager = get_database_manager(settings.database_path)
        app.state.db_manager = db_manager
        
        # Initialize vector store only when vector search is enabled
        if settings.embeddings_dir and not settings.disable_vector_search:
            try:
                from core.embeddings.storage import VectorStore
                vector_store = VectorStore(
                    embeddings_dir=settings.embeddings_dir,
                    primary_model=settings.primary_embedding_model,
                    fallback_model=settings.fallback_embedding_model,
                    tertiary_model=getattr(settings, 'tertiary_embedding_model', None)
                )
                vector_store.load()
                app.state.vector_store = vector_store
                logger.info("Vector store initialized")
            except Exception as e:
                logger.warning(f"Vector store initialization failed: {e}")
                app.state.vector_store = None
        else:
            app.state.vector_store = None
            logger.info("Vector search disabled; skipping vector store initialization")
        
        # Initialize LLM client
        try:
            from llm.client import GroqClient
            llm_client = GroqClient(
                api_key=settings.groq_api_key,
                models=settings.groq_models
            )
            app.state.llm_client = llm_client
            logger.info("✅ LLM client initialized")
        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}")
            app.state.llm_client = None
        
        # Initialize Relationship Graph
        try:
            from core.entities.relationship import RelationshipGraph
            app.state.relationship_graph = RelationshipGraph(
                characters_path=settings.characters_path,
                locations_path=settings.locations_path
            )
            logger.info("✅ Relationship graph initialized")
        except Exception as e:
            logger.warning(f"Relationship graph initialization failed: {e}")
            app.state.relationship_graph = None
        
        # Initialize Entity Resolver (singleton for app)
        try:
            from core.entities.resolver import EntityResolver
            app.state.entity_resolver = EntityResolver(
                characters_path=settings.characters_path,
                locations_path=settings.locations_path,
                confidence_threshold=settings.entity_confidence_threshold
            )
            logger.info("✅ Entity resolver initialized")
        except Exception as e:
            logger.warning(f"Entity resolver initialization failed: {e}")
            app.state.entity_resolver = None
            
        # Initialize Query Router (singleton)
        try:
            from rag.query_router import QueryRouter
            app.state.query_router = QueryRouter(settings.database_path)
            logger.info("✅ Query router initialized")
            
            # Log router configuration
            if settings.enable_router_embeddings:
                logger.info(f"Router embeddings enabled with model: {settings.router_embedding_model}")
            else:
                logger.info("Router embeddings are disabled")
                
        except Exception as e:
            logger.warning(f"Query router initialization failed: {e}")
            app.state.query_router = None
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        raise

async def _cleanup_resources():
    """Clean up resources on shutdown"""
    
    try:
        # Close database connections
        if hasattr(app.state, 'db_manager'):
            app.state.db_manager.close_all()
        
        # Clean up vector store
        if hasattr(app.state, 'vector_store') and app.state.vector_store:
            app.state.vector_store.cleanup()
        
        # Clean up LLM client
        if hasattr(app.state, 'llm_client'):
            del app.state.llm_client
            
        # Clean up Query Router
        if hasattr(app.state, 'query_router'):
            if hasattr(app.state.query_router, 'embedding_model'):
                del app.state.query_router.embedding_model
            del app.state.query_router
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def _execute_rag_pipeline(query_request: QueryRequest, embeddings_status: bool, llm_status: bool) -> Dict[str, Any]:
    """Execute RAG pipeline safely"""
    
    try:
        from rag.query_router import QueryRouter
        from rag.retriever import RAGRetriever
        from rag.synthesizer import ResponseSynthesizer
        from core.entities.resolver import EntityResolver
        from core.search.hybrid_engine import HybridSearchEngine
        
        # Router
        query_router = QueryRouter(settings.database_path)
        analysis = query_router.analyze_query(query_request.question)

        # Count queries (direct DB)
        if analysis.type == "count":
            count_result = query_router.handle_count_query(analysis)
            synthesizer = ResponseSynthesizer(app.state.llm_client if llm_status else None)
            syn = synthesizer.synthesize_count_response(count_result)
            return {
                "answer": syn.answer,
                "sources": syn.sources,
                "citations": syn.citations,
                "confidence": syn.confidence,
                "query_type": "count"
            }

        # Citation lookup (direct verse)
        if analysis.type == "citation":
            citation_result = query_router.handle_citation_query(analysis)
            verse_id = citation_result.get("verse_id")
            # Convert to dot-style display format for UI
            citations = [denormalize_verse_id(verse_id)] if verse_id else []
            # Respect include_context flag: if false, include only the main verse (first source)
            all_sources = citation_result.get("sources", [])
            # For direct citation, avoid repeating the main verse in the Sources panel.
            # If context is requested, show only neighbor verses; otherwise show none.
            if query_request.include_context:
                sources = all_sources[1:] if len(all_sources) > 1 else []
            else:
                sources = []
            # Generate a concise context summary using surrounding verses when requested
            _syn = ResponseSynthesizer(app.state.llm_client if llm_status else None)
            context_summary = ""
            try:
                # Always use full list (main verse first, then neighbors) for summarization
                context_summary = _syn.summarize_citation_context({
                    "sources": all_sources
                })
            except Exception:
                context_summary = ""
            answer_text = citation_result.get("answer", "")
            if context_summary:
                answer_text = f"{answer_text}\n\nContext Summary: {context_summary}"
            return {
                "answer": answer_text,
                "sources": sources,
                "citations": citations,
                "confidence": 1.0 if verse_id else 0.5,
                "query_type": "citation"
            }

        # Build retrieval stack for relationship/general using singleton resolver
        entity_resolver = getattr(app.state, 'entity_resolver', None)
        if not entity_resolver:
            try:
                from core.entities.resolver import EntityResolver as _ER
                entity_resolver = _ER(
                    characters_path=settings.characters_path,
                    locations_path=settings.locations_path,
                    confidence_threshold=settings.entity_confidence_threshold
                )
                app.state.entity_resolver = entity_resolver
            except Exception:
                # As a last resort, set to None; downstream should handle
                entity_resolver = None

        vector_store = getattr(app.state, 'vector_store', None) if embeddings_status else None
        search_engine = HybridSearchEngine(
            db_path=settings.database_path,
            vector_store=vector_store,
            entity_resolver=entity_resolver,
            bm25_weight=settings.bm25_weight,
            vector_weight=settings.vector_weight,
            fts_table=settings.fts_table
        )
        retriever = RAGRetriever(
            search_engine=search_engine,
            entity_resolver=entity_resolver,
            relationship_graph=getattr(app.state, 'relationship_graph', None),
            confidence_threshold=settings.entity_confidence_threshold
        )

        if analysis.type == "relationship":
            # Try KG-first deterministic answering when possible
            try:
                from core.entities.relationship import RelationshipGraph
                kg = getattr(app.state, 'relationship_graph', None)
            except Exception:
                kg = None

            subject = None
            relation = None
            if getattr(analysis, 'parameters', None):
                subject = (analysis.parameters or {}).get('subject')
                relation = (analysis.parameters or {}).get('relation')

            # If we have a parsed subject+relation and KG answering is enabled
            if subject and relation and kg and getattr(settings, 'kg_relationship_answers', True):
                # Resolve subject to canonical name
                subj_resolved = entity_resolver.resolve(subject)
                if subj_resolved:
                    inferred = kg.infer_family(subj_resolved.canonical_name, relation)
                    if inferred:
                        # Build concise deterministic answer
                        subj_disp = subj_resolved.canonical_name
                        rel_disp = relation

                        # Helper: produce clean display names (strip trailing parentheticals like "(Dasharatha's son)")
                        def _display_name(n: str) -> str:
                            s = str(n or '').strip()
                            if s.endswith(')'):
                                lb = s.rfind('(')
                                if lb != -1:
                                    return s[:lb].strip()
                            return s

                        # Grammar: singular vs plural and label pluralization when multiple
                        is_plural = relation in {"parents", "children", "siblings", "spouses", "grandparents", "grandchildren"} or len(inferred) > 1
                        verb = "are" if is_plural else "is"

                        # Pluralize label for multiple inferred results
                        def _pluralize(label: str) -> str:
                            m = label.lower()
                            return {
                                "father": "fathers",
                                "mother": "mothers",
                                "parent": "parents",
                                "son": "sons",
                                "daughter": "daughters",
                                "child": "children",
                                "brother": "brothers",
                                "sister": "sisters",
                                "sibling": "siblings",
                                "husband": "husbands",
                                "wife": "wives",
                                "spouse": "spouses",
                                "grandfather": "grandfathers",
                                "grandmother": "grandmothers",
                                "grandparent": "grandparents",
                            }.get(m, label)

                        if is_plural:
                            rel_disp = _pluralize(rel_disp)

                        # Join inferred names in a natural list (sanitize names for display)
                        disp_names = [_display_name(n) for n in inferred]
                        if len(disp_names) == 1:
                            names_text = disp_names[0]
                        elif len(disp_names) == 2:
                            names_text = f"{disp_names[0]} and {disp_names[1]}"
                        else:
                            names_text = ", ".join(disp_names[:-1]) + f", and {disp_names[-1]}"

                        sources = []
                        citations = []

                        if query_request.include_context:
                            # Collect supporting sources via targeted hybrid search
                            try:
                                support_queries = set()
                                # Always include natural phrasing around parent/child relations for each inferred name
                                # Use sanitized display names for queries to reduce noise
                                for name in disp_names:
                                    # General neutral phrasing (may help vector search)
                                    support_queries.add(f"{subj_disp} {relation} {name}")
                                    support_queries.add(f"{name} {relation} of {subj_disp}")
                                    # Parent/child specific expansions only when relation is parent/child category
                                    if (relation or '').lower() in {"father", "mother", "parent", "parents", "son", "daughter", "child", "children", "grandfather", "grandmother", "grandparent", "grandparents"}:
                                        # Parent side (name -> subject)
                                        support_queries.add(f"{name} father of {subj_disp}")
                                        support_queries.add(f"{name} mother of {subj_disp}")
                                        support_queries.add(f"{name} parent of {subj_disp}")
                                        # Child side (subject -> name)
                                        support_queries.add(f"{subj_disp} son of {name}")
                                        support_queries.add(f"{subj_disp} daughter of {name}")
                                        support_queries.add(f"{subj_disp} child of {name}")
                                    # Two-sided additions gated by settings
                                    if getattr(settings, 'enable_two_sided_relationship_context', True):
                                        # Spouse side (bidirectional phrasing)
                                        if relation.lower() in {"wife", "husband", "spouse", "spouses"}:
                                            support_queries.add(f"{subj_disp} wife {name}")
                                            support_queries.add(f"{subj_disp} husband {name}")
                                            support_queries.add(f"{subj_disp} spouse {name}")
                                            support_queries.add(f"{subj_disp} married to {name}")
                                            support_queries.add(f"{subj_disp} married {name}")
                                            support_queries.add(f"{name} wife of {subj_disp}")
                                            support_queries.add(f"{name} husband of {subj_disp}")
                                            support_queries.add(f"{name} spouse of {subj_disp}")
                                            support_queries.add(f"{name} married to {subj_disp}")
                                        # Sibling side (bidirectional phrasing) – use neutral term only to avoid gender errors
                                        if relation.lower() in {"brother", "sister", "sibling", "siblings"}:
                                            support_queries.add(f"{subj_disp} sibling {name}")
                                            support_queries.add(f"{name} sibling of {subj_disp}")
                                # If grand- relation, also search via parent bridge
                                if relation in {"grandfather", "grandmother", "grandparent", "grandparents"}:
                                    try:
                                        parents_bridge = []
                                        parents_bridge.extend(kg.infer_family(subj_disp, "father"))
                                        parents_bridge.extend(kg.infer_family(subj_disp, "mother"))
                                    except Exception:
                                        parents_bridge = []
                                    for p in parents_bridge:
                                        for g in inferred:
                                            # Natural grand-bridge phrasing
                                            support_queries.add(f"{g} father of {p}")
                                            support_queries.add(f"{g} mother of {p}")
                                            support_queries.add(f"{g} parent of {p}")
                                            support_queries.add(f"{p} son of {g}")
                                            support_queries.add(f"{p} daughter of {g}")
                                            support_queries.add(f"{p} child of {g}")
                                # Diagnostics: log support queries when enabled
                                if getattr(settings, 'enable_relationship_diagnostics', False):
                                    try:
                                        logger.info(f"[RELATION_CTX] {subj_disp} ~{relation}: {list(support_queries)[:10]}")
                                    except Exception:
                                        pass
                                # Execute searches and aggregate results
                                all_docs = []
                                seen_vids = set()
                                # honor UI cap (up to 10)
                                limit = max(1, min(query_request.max_results, 10))
                                # modest cap per query
                                per_q_k = min(getattr(settings, 'hybrid_top_k', 10), 5)
                                for sq in list(support_queries)[:5]:
                                    try:
                                        docs = search_engine.search(sq, top_k=per_q_k, include_entities=True) or []
                                        for d in docs:
                                            vid = getattr(d, 'verse_id', None)
                                            if vid and vid not in seen_vids:
                                                seen_vids.add(vid)
                                                all_docs.append(d)
                                    except Exception:
                                        continue

                                # Re-rank aggregated results to prefer genealogy mentions
                                def _genealogy_score(doc) -> float:
                                    try:
                                        text = f"{doc.translation_en} {doc.translation_hi} {doc.shloka_iast} {' '.join(doc.sarga_titles)}".lower()
                                    except Exception:
                                        text = ""
                                    score = float(getattr(doc, 'final_score', 0.0) or 0.0)
                                    subj_l = subj_disp.lower()
                                    has_subj = subj_l in text
                                    has_any_name = any((n or '').lower() in text for n in inferred)
                                    has_rel_words = any(w in text for w in ["father", "mother", "parent", "son", "daughter", "child", "grand", "descendant", "offspring"])
                                    if has_subj:
                                        score += 0.08
                                    if has_any_name:
                                        score += 0.08
                                    if has_rel_words:
                                        score += 0.08
                                    if has_subj and has_any_name:
                                        score += 0.05
                                    return score

                                all_docs.sort(key=_genealogy_score, reverse=True)

                                # Build sources and citations honoring UI cap
                                for doc in all_docs[:limit]:
                                    sources.append({
                                        "verse_id": doc.verse_id,
                                        "citation_format": denormalize_verse_id(doc.verse_id),
                                        "kanda_name": doc.kanda_name,
                                        "sarga_num": doc.sarga_num,
                                        "verse_index": doc.verse_index,
                                        "translation_en": doc.translation_en,
                                        "translation_hi": doc.translation_hi,
                                        "shloka_sanskrit": doc.shloka_sanskrit,
                                        "shloka_iast": doc.shloka_iast,
                                        "sarga_titles": doc.sarga_titles,
                                        "search_score": doc.final_score,
                                    })
                                    if len(citations) < min(3, limit):
                                        citations.append(denormalize_verse_id(doc.verse_id))
                            except Exception:
                                sources = []
                                citations = []

                        # Build answer; include inline primary citation only when context included
                        # Inline a primary citation only when exactly one name is present
                        primary_citation = ""
                        if query_request.include_context and citations and len(inferred) == 1:
                            primary_citation = f" ({citations[0]})"
                        answer_text = f"{subj_disp}'s {rel_disp} {verb} {names_text}{primary_citation}."

                        # If context was requested but we could not find supporting citations, be explicit
                        confidence = 0.98
                        if query_request.include_context and not citations:
                            confidence = 0.92
                            answer_text += "\n\nNote: No direct verse citation could be located for this relationship in the retrieved top results. The answer is derived from the canonical relationship graph."

                        return {
                            "answer": answer_text,
                            "sources": sources,
                            "citations": citations,
                            "confidence": confidence,
                            "query_type": "relationship"
                        }

            # Fallback A: On-the-fly relation inference (pattern + BM25) with citations
            if (
                getattr(settings, 'enable_relation_inference', True)
                and subject and relation and entity_resolver
            ):
                try:
                    from core.entities.relation_infer import RelationInferencer
                    inferencer = RelationInferencer(
                        db_path=settings.database_path,
                        entity_resolver=entity_resolver,
                        fts_table=settings.fts_table,
                    )
                    inf = inferencer.infer(
                        subject=subject,
                        relation=relation,
                        top_k=getattr(settings, 'relation_infer_top_k', 40),
                        per_query_k=getattr(settings, 'relation_infer_per_query_k', 6),
                        max_queries=getattr(settings, 'relation_infer_max_queries', 6),
                    )
                    cands = (inf or {}).get('candidates', [])
                    if cands:
                        # Select top candidates (spouse typically singular; others allow small list)
                        rel_lower = (relation or '').lower()
                        take_n = 1 if rel_lower in {"wife", "husband", "spouse", "spouses"} else min(3, len(cands))
                        top = cands[:take_n]

                        subj_disp = (inf or {}).get('subject', subject)
                        rel_disp = relation
                        is_plural = rel_lower in {"parents", "children", "siblings", "spouses", "grandparents", "grandchildren"} or len(top) > 1
                        verb = "are" if is_plural else "is"

                        def _pluralize(label: str) -> str:
                            m = label.lower()
                            return {
                                "father": "fathers",
                                "mother": "mothers",
                                "parent": "parents",
                                "son": "sons",
                                "daughter": "daughters",
                                "child": "children",
                                "brother": "brothers",
                                "sister": "sisters",
                                "sibling": "siblings",
                                "husband": "husbands",
                                "wife": "wives",
                                "spouse": "spouses",
                                "grandfather": "grandfathers",
                                "grandmother": "grandmothers",
                                "grandparent": "grandparents",
                            }.get(m, label)

                        if is_plural:
                            rel_disp = _pluralize(rel_disp)

                        names = [t.get('object_canonical') or t.get('raw_name') for t in top]
                        names = [n for n in names if n]
                        # Sanitize names for display (strip trailing parentheticals like "(Dasharatha's son)")
                        def _disp(n: str) -> str:
                            s = str(n or '').strip()
                            if s.endswith(')'):
                                lb = s.rfind('(')
                                if lb != -1:
                                    return s[:lb].strip()
                            return s
                        disp_names = [_disp(n) for n in names]
                        if len(disp_names) == 1:
                            names_text = disp_names[0]
                        elif len(disp_names) == 2:
                            names_text = f"{disp_names[0]} and {disp_names[1]}"
                        else:
                            names_text = ", ".join(disp_names[:-1]) + f", and {disp_names[-1]}"

                        # Build sources and citations from top candidates
                        sources = []
                        citations = []
                        seen_vids = set()
                        limit = max(1, min(query_request.max_results, 10))
                        for t in top:
                            vid = t.get('verse_id')
                            if vid and vid not in seen_vids:
                                seen_vids.add(vid)
                                sources.append({
                                    "verse_id": vid,
                                    "citation_format": t.get('citation'),
                                    "kanda_name": t.get('kanda_name'),
                                    "sarga_num": t.get('sarga_num'),
                                    "verse_index": t.get('verse_index'),
                                    "translation_en": t.get('translation_en'),
                                    "translation_hi": t.get('translation_hi'),
                                    "shloka_sanskrit": None,  # not used here to reduce payload
                                    "shloka_iast": t.get('shloka_iast'),
                                    "search_score": None,
                                })
                                if len(citations) < min(3, limit) and t.get('citation'):
                                    citations.append(t['citation'])

                        # If context requested, expand with two-sided retrieval to strengthen citations
                        if query_request.include_context:
                            try:
                                support_queries = set()
                                for name in disp_names:
                                    # Neutral phrasing
                                    support_queries.add(f"{subj_disp} {relation} {name}")
                                    support_queries.add(f"{name} {relation} of {subj_disp}")
                                    # Spouse side
                                    if (relation or '').lower() in {"wife", "husband", "spouse", "spouses"}:
                                        support_queries.add(f"{subj_disp} wife {name}")
                                        support_queries.add(f"{subj_disp} husband {name}")
                                        support_queries.add(f"{subj_disp} spouse {name}")
                                        support_queries.add(f"{subj_disp} married to {name}")
                                        support_queries.add(f"{subj_disp} married {name}")
                                        support_queries.add(f"{name} wife of {subj_disp}")
                                        support_queries.add(f"{name} husband of {subj_disp}")
                                        support_queries.add(f"{name} spouse of {subj_disp}")
                                        support_queries.add(f"{name} married to {subj_disp}")
                                    # Sibling side (neutral phrasing only)
                                    if (relation or '').lower() in {"brother", "sister", "sibling", "siblings"}:
                                        support_queries.add(f"{subj_disp} sibling {name}")
                                        support_queries.add(f"{name} sibling of {subj_disp}")

                                # modest cap per query
                                per_q_k = min(getattr(settings, 'hybrid_top_k', 10), 5)
                                for sq in list(support_queries)[:5]:
                                    try:
                                        docs = search_engine.search(sq, top_k=per_q_k, include_entities=True) or []
                                        for d in docs:
                                            vid = getattr(d, 'verse_id', None)
                                            if vid and vid not in seen_vids and len(sources) < limit:
                                                seen_vids.add(vid)
                                                sources.append({
                                                    "verse_id": d.verse_id,
                                                    "citation_format": denormalize_verse_id(d.verse_id),
                                                    "kanda_name": d.kanda_name,
                                                    "sarga_num": d.sarga_num,
                                                    "verse_index": d.verse_index,
                                                    "translation_en": d.translation_en,
                                                    "translation_hi": d.translation_hi,
                                                    "shloka_sanskrit": d.shloka_sanskrit,
                                                    "shloka_iast": d.shloka_iast,
                                                    "search_score": d.final_score,
                                                })
                                                if len(citations) < min(3, limit):
                                                    citations.append(denormalize_verse_id(d.verse_id))
                                    except Exception:
                                        continue
                            except Exception:
                                pass

                        # Inline a primary citation only when exactly one name is present
                        primary_citation = ""
                        if query_request.include_context and citations and len(names) == 1:
                            primary_citation = f" ({citations[0]})"
                        answer_text = f"{subj_disp}'s {rel_disp} {verb} {names_text}{primary_citation}."

                        # Confidence calibration from top candidate score
                        try:
                            top_score = float(cands[0].get('score', 0.0) or 0.0)
                        except Exception:
                            top_score = 0.0
                        confidence = 0.92
                        if top_score >= 1.2:
                            confidence = 0.94
                        elif top_score < 1.0:
                            confidence = 0.88

                        return {
                            "answer": answer_text,
                            "sources": sources[:limit],
                            "citations": citations,
                            "confidence": confidence,
                            "query_type": "relationship"
                        }
                except Exception as _e:
                    logger.warning(f"Relation inference fallback failed: {_e}")

            # Fallback B: relationship-aware retrieval (use parsed params when available)
            entity_for_retrieval = subject if subject else query_request.question
            retrieving_k = (
                getattr(settings, 'adaptive_max_k', query_request.max_results)
                if getattr(settings, 'adaptive_retrieval', False)
                else query_request.max_results
            )
            ret = retriever.retrieve_relationship(
                entity_name=entity_for_retrieval,
                rel_type=relation,
                top_k=retrieving_k
            )
        else:
            # general/default path
            retrieving_k = (
                getattr(settings, 'adaptive_max_k', query_request.max_results)
                if getattr(settings, 'adaptive_retrieval', False)
                else query_request.max_results
            )
            # In production we often want to avoid entity-driven OR expansion for general queries
            # Use settings.entity_boost_for_general to control this behavior (default False)
            ret = retriever.retrieve(
                query=query_request.question,
                top_k=retrieving_k,
                include_context=query_request.include_context,
                entity_boost=getattr(settings, 'entity_boost_for_general', False)
            )

        synthesizer = ResponseSynthesizer(app.state.llm_client if llm_status else None)
        syn = synthesizer.synthesize(
            query=query_request.question,
            documents=ret.results,
            query_type=analysis.type or "general",
            max_sources=min(query_request.max_results, 10)
        )

        return {
            "answer": syn.answer,
            "sources": syn.sources,
            "citations": syn.citations,
            "confidence": syn.confidence,
            "query_type": syn.query_type
        }
        
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        return {
            "answer": "Error in RAG pipeline",
            "sources": [],
            "citations": [],
            "confidence": 0.0,
            "query_type": "error"
        }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.auto_reload and settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )
