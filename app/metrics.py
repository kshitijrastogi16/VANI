"""
Prometheus metrics for VANI.
Provides globally shared counters and histograms and a /metrics endpoint helper.
"""
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# HTTP-level metrics
REQUEST_COUNT = Counter(
    "vani_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
)

REQUEST_LATENCY = Histogram(
    "vani_request_latency_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "path"),
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2, 3, 5, 8, 13)
)

# RAG-specific metrics
RAG_QUERY_COUNT = Counter(
    "vani_rag_query_total",
    "Total RAG queries processed",
    labelnames=("query_type", "success"),
)

RAG_DOCUMENTS_RETURNED = Histogram(
    "vani_rag_documents_returned",
    "Number of documents returned by retriever",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21)
)

# Retrieval observability
RAG_FUSED_TOP1_SCORE = Histogram(
    "vani_rag_fused_top1_score",
    "Top-1 fused score distribution",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)
)

RAG_CANDIDATES_CONSIDERED = Histogram(
    "vani_rag_candidates_considered",
    "Number of candidates considered after fusion and before enrichment/rerank",
    buckets=(1, 2, 3, 5, 8, 13, 21, 34)
)

RAG_ABSTENTIONS = Counter(
    "vani_rag_abstentions_total",
    "Total abstentions due to insufficient evidence",
    labelnames=("reason",)
)

RAG_STRICT_CITATION_FALLBACK = Counter(
    "vani_rag_strict_citation_fallback_total",
    "Total times strict citation fallback template was used"
)

RAG_MMR_APPLIED = Counter(
    "vani_rag_mmr_applied_total",
    "Total times MMR diversity step was applied"
)


def metrics_payload() -> bytes:
    """Return the latest Prometheus metrics payload."""
    return generate_latest()
