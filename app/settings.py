"""
VANI V3 Settings - Centralized Configuration Management
Enforcing: V2 pydantic settings with .env support
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from pydantic import ConfigDict

class Settings(BaseSettings):
    """Centralized application settings with environment variable support"""
    
    # Model configuration
    model_config = ConfigDict(
        extra='ignore',
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Project Info
    project_name: str = "VANI - Voice of Ancient Narrative Intelligence"
    project_description: str = "Voice of Ancient Narrative Intelligence - Ramayana RAG System"
    version: str = "3.0.0"
    project_version: str = Field(default="3.0.0", env="PROJECT_VERSION")
    
    # Application settings
    debug: bool = Field(default=False, env="DEBUG")
    # Testing/dev convenience: skip startup validations (env/data checks)
    skip_startup_checks: bool = Field(default=False, env="SKIP_STARTUP_CHECKS")
    
    # Router settings
    enable_router_embeddings: bool = Field(
        default=False,
        description="Enable/disable embedding model in QueryRouter"
    )
    router_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding model for QueryRouter"
    )
    
    # LLM settings
    llm_timeout: int = Field(
        default=30,
        description="Timeout in seconds for LLM API calls"
    )
    llm_max_retries: int = Field(
        default=2,
        description="Maximum number of retries for LLM API calls"
    )
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    auto_reload: bool = Field(default=False, env="AUTO_RELOAD")
    
    # Database Configuration
    database_path: str = "data/processed/ramayana.db"
    fts_table: str = "verses_fts"
    
    # Web Directories
    web_static_dir: str = Field(default="web/static", env="WEB_STATIC_DIR")
    web_templates_dir: str = Field(default="web/templates", env="WEB_TEMPLATES_DIR")
    
    # Embeddings Configuration (V3 Hybrid Strategy)
    embeddings_dir: str = "data/processed/embeddings"
    # Primary: V2's choice for performance
    primary_embedding_model: str = "BAAI/bge-m3"
    # Fallbacks in order: secondary then tertiary
    fallback_embedding_model: str = "intfloat/multilingual-e5-large"
    tertiary_embedding_model: str = "l3cube-pune/indic-sentence-similarity-sbert"
    batch_size: int = 32
    
    # Search Configuration
    hybrid_top_k: int = 10
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    # Optional per-column BM25 weights (match FTS column count).
    # Example when Hindi is present: [0.0, 0.0, 0.8, 1.2, 0.0, 0.5]
    # Example when Hindi column is removed: [0.0, 0.0, 0.8, 1.2, 0.5]
    bm25_column_weights: List[float] | None = Field(default=None, env="BM25_COLUMN_WEIGHTS")
    # Fielded BM25 quota for English/IAST focused retrieval (tunable)
    fielded_top_k: int = Field(default=40, env="FIELDED_TOP_K")
    # Hybrid score calibration
    hybrid_score_norm: str = Field(default="minmax", env="HYBRID_SCORE_NORM")  # "minmax" or "zscore"
    hybrid_log_scale: bool = Field(default=True, env="HYBRID_LOG_SCALE")
    hybrid_log_alpha: float = Field(default=1.5, env="HYBRID_LOG_ALPHA")
    # Heuristic re-ranking and dynamic weighting (no re-embedding)
    enable_heuristic_rerank: bool = Field(default=True, env="ENABLE_HEURISTIC_RERANK")
    reranker_top_k: int = Field(default=30, env="RERANKER_TOP_K")
    dynamic_hybrid_weights: bool = Field(default=True, env="DYNAMIC_HYBRID_WEIGHTS")
    # Entity expansion inside HybridSearchEngine
    hybrid_alias_limit: int = Field(default=4, env="HYBRID_ALIAS_LIMIT")
    enable_entity_context_in_hybrid: bool = Field(default=True, env="ENABLE_ENTITY_CONTEXT_IN_HYBRID")
    hybrid_context_limit: int = Field(default=2, env="HYBRID_CONTEXT_LIMIT")
    # Adaptive retrieval (auto-select K based on score distribution)
    adaptive_retrieval: bool = Field(default=True, env="ADAPTIVE_RETRIEVAL")
    adaptive_min_k: int = Field(default=5, env="ADAPTIVE_MIN_K")
    adaptive_max_k: int = Field(default=20, env="ADAPTIVE_MAX_K")
    # When the relative drop between successive scores exceeds this ratio, we cut off
    # Example: if s[i] - s[i+1] >= adaptive_drop_ratio * s[0], stop at i+1
    adaptive_drop_ratio: float = Field(default=0.15, env="ADAPTIVE_DROP_RATIO")
    # Query expansion policy
    english_only_query_expansion: bool = Field(default=False, env="ENGLISH_ONLY_QUERY_EXPANSION")
    # Diversity (optional MMR reordering after heuristic rerank)
    enable_mmr_diversity: bool = Field(default=False, env="ENABLE_MMR_DIVERSITY")
    mmr_lambda: float = Field(default=0.7, env="MMR_LAMBDA")
    mmr_k: int = Field(default=20, env="MMR_K")
    # Pseudo-Relevance Feedback (PRF) expansion (English-only terms)
    enable_prf: bool = Field(default=False, env="ENABLE_PRF")
    prf_bm25_k: int = Field(default=8, env="PRF_BM25_K")
    prf_terms_k: int = Field(default=2, env="PRF_TERMS_K")
    prf_min_word_len: int = Field(default=4, env="PRF_MIN_WORD_LEN")
    # Retrieval cache (LRU + TTL)
    retrieval_cache_enabled: bool = Field(default=True, env="RETRIEVAL_CACHE_ENABLED")
    retrieval_cache_ttl_secs: int = Field(default=60, env="RETRIEVAL_CACHE_TTL_SECS")
    retrieval_cache_max_entries: int = Field(default=256, env="RETRIEVAL_CACHE_MAX_ENTRIES")
    # Optional cross-encoder reranker for final candidates
    enable_cross_encoder_rerank: bool = Field(default=False, env="ENABLE_CROSS_ENCODER_RERANK")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", env="CROSS_ENCODER_MODEL")
    cross_encoder_top_k: int = Field(default=30, env="CROSS_ENCODER_TOP_K")
    cross_encoder_weight: float = Field(default=0.6, env="CROSS_ENCODER_WEIGHT")
    cross_encoder_timeout_secs: float = Field(default=1.0, env="CROSS_ENCODER_TIMEOUT_SECS")
    
    # Entity Resolution
    characters_path: str = "data/raw/characters.json"
    locations_path: str = "data/raw/locations.json"
    entity_confidence_threshold: float = 0.7
    # Prefer knowledge-graph answers for direct relationship questions when possible
    kg_relationship_answers: bool = Field(default=True, env="KG_RELATIONSHIP_ANSWERS")
    # Citation window size for verse context (used by QueryRouter)
    citation_context_window: int = Field(default=2, env="CITATION_CONTEXT_WINDOW")
    # Relationship Inference (runtime, no re-embedding)
    enable_relation_inference: bool = Field(default=True, env="ENABLE_RELATION_INFERENCE")
    relation_infer_top_k: int = Field(default=40, env="RELATION_INFER_TOP_K")
    relation_infer_per_query_k: int = Field(default=6, env="RELATION_INFER_PER_QUERY_K")
    relation_infer_max_queries: int = Field(default=6, env="RELATION_INFER_MAX_QUERIES")
    # Description-based query expansion (BM25 only; no re-embedding)
    enable_description_expansion: bool = Field(default=True, env="ENABLE_DESCRIPTION_EXPANSION")
    
    # Canonicality policy
    # If true, prompts and synthesis will be constrained to Valmiki Ramayana only (Baala, Ayodhya, Aranya, Kishkindha, Sundara, Yuddha)
    restrict_to_valmiki: bool = Field(default=True, env="RESTRICT_TO_VALMIKI")
    # If true, explicitly suppress mentions of Uttara Kanda in generated answers
    disallow_uttara_kanda_mentions: bool = Field(default=True, env="DISALLOW_UTTARA_KANDA")
    
    # FTS configuration: optionally unindex or remove Hindi column
    fts_unindex_hi: bool = Field(default=True, env="FTS_UNINDEX_HI")
    fts_remove_hi: bool = Field(default=False, env="FTS_REMOVE_HI")
    
    # Tuned weights persistence
    tuned_weights_path: str = Field(default="data/config/tuned_weights.json", env="TUNED_WEIGHTS_PATH")
    load_tuned_weights: bool = Field(default=True, env="LOAD_TUNED_WEIGHTS")
    # Retrieval/context behavior for relationships
    enable_two_sided_relationship_context: bool = Field(default=True, env="ENABLE_TWO_SIDED_REL_CTX")
    enable_relationship_diagnostics: bool = Field(default=False, env="ENABLE_REL_DIAGNOSTICS")
    
    # Retriever expansion controls
    retriever_expand_context: bool = Field(default=False, env="RETRIEVER_EXPAND_CONTEXT")
    retriever_alias_limit: int = Field(default=3, env="RETRIEVER_ALIAS_LIMIT")
    retriever_identity_alias_limit: int = Field(default=2, env="RETRIEVER_IDENTITY_ALIAS_LIMIT")
    retriever_hard_filter_relationship_terms: bool = Field(default=True, env="RETRIEVER_FILTER_REL_TERMS")
    retriever_double_expansion_guard: bool = Field(default=True, env="RETRIEVER_DOUBLE_EXPANSION_GUARD")
    # In production, avoid entity-driven OR query expansions for general queries by default
    entity_boost_for_general: bool = Field(default=True, env="ENTITY_BOOST_FOR_GENERAL")
    # Router-gated entity expansion confidence threshold for relationship/narrative queries
    entity_expansion_router_confidence: float = Field(default=0.7, env="ENTITY_EXPANSION_ROUTER_CONFIDENCE")
    
    # LLM Configuration
    # Current Groq hosted model IDs. Override with GROQ_MODELS in .env.
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_models: List[str] = Field(
        default=[
            "openai/gpt-oss-120b",  # Primary production
            "qwen/qwen3.8-27b",     # Complex narrative
            "openai/gpt-oss-20b",   # Speed fallback
        ],
        env="GROQ_MODELS",
    )
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    # llm_timeout configured above
    
    # Security & Performance
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/vani.log"

    # Answer Quality & Citations
    # If true, responses must include citations that match provided sources; otherwise fall back to deterministic template
    strict_citations_required: bool = Field(default=True, env="STRICT_CITATIONS_REQUIRED")
    # If the top fused score is below this threshold, abstain with a helpful message instead of low-quality answer
    abstain_below_score: float = Field(default=0.18, env="ABSTAIN_BELOW_SCORE")


# Global settings instance
settings = Settings()
