"""
VANI V3 Hybrid Search Engine - V1 Sophistication + V2 Architecture
Enforcing: Advanced weighted fusion with entity enhancement
"""

import sqlite3
import json
import re
import logging
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from statistics import mean, pstdev

from core.search.text_search import search_bm25, search_bm25_fielded, search_bm25_with_filters
from core.search.vector_search import VectorSearchEngine
from core.database.connection import get_database_manager

from core.entities.resolver import EntityResolver, ResolvedEntity
from app.settings import settings
from app.metrics import (
    RAG_FUSED_TOP1_SCORE,
    RAG_CANDIDATES_CONSIDERED,
    RAG_MMR_APPLIED,
)
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    verse_id: str
    kanda_name: str
    sarga_num: int
    verse_index: int
    translation_en: str
    translation_hi: str
    shloka_sanskrit: str
    shloka_iast: str
    sarga_titles: List[str]
    bm25_score: float
    vector_score: float
    final_score: float
    entities_found: List[str]
    # Windowed embeddings metadata (optional)
    verse_ids: List[str] = None
    start_verse_index: int = 0
    end_verse_index: int = 0
    anchor_verse_id: str = ""
    # Aggregated window content (if vector result came from a window)
    window_translation_en: str = ""
    window_shloka_iast: str = ""


class HybridSearchEngine:
    """
    V1 Sophisticated Hybrid Search with V2 Architecture
    Features: Entity enhancement, weighted fusion, contextual enrichment
    """
    
    def __init__(self, 
                 db_path: str,
                 vector_store,
                 entity_resolver: EntityResolver,
                 bm25_weight: float = 0.4,
                 vector_weight: float = 0.6,
                 fts_table: str = "verses_fts"):
        
        self.db_path = db_path
        self.vector_store = vector_store
        self.entity_resolver = entity_resolver
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.fts_table = fts_table
        
        # Validate weights
        if abs((bm25_weight + vector_weight) - 1.0) > 0.01:
            logger.warning("BM25 and vector weights don't sum to 1.0")
        # Simple in-memory retrieval cache (LRU by timestamp, TTL-based)
        self._cache: Dict[str, Tuple[float, List[SearchResult]]] = {}
        # Lazy cross-encoder cache
        self._ce_model = None
    
    def search(self, query: str, top_k: int = 10, include_entities: bool = True) -> List[SearchResult]:
        """
        Main hybrid search with V1 sophistication
        """
        
        logger.info(f"Hybrid search for: '{query}' (top_k={top_k})")
        # Retrieval cache lookup
        try:
            if getattr(settings, 'retrieval_cache_enabled', True):
                ttl = int(getattr(settings, 'retrieval_cache_ttl_secs', 60))
                key = f"{query}|{top_k}|{int(include_entities)}"
                hit = self._cache.get(key)
                if hit and (time.time() - hit[0]) <= ttl:
                    return hit[1]
        except Exception:
            pass
        
        # 1. Entity-enhanced query expansion (V1 intelligence)
        # Guard: if retriever already expanded into an OR-joined quoted query, don't enhance again
        do_enhance = include_entities
        try:
            if getattr(settings, 'retriever_double_expansion_guard', True):
                if (' OR ' in query) or (query.count('"') >= 4):
                    do_enhance = False
        except Exception:
            pass
        enhanced_query = self._enhance_query_with_entities(query) if do_enhance else query
        
        # 2. Get BM25 results
        bm25_results = self._get_bm25_results(enhanced_query, top_k * 3)
        
        # 3. Get vector similarity results  
        vector_results = self._get_vector_results(enhanced_query, top_k * 3)
        
        # 4. Perform sophisticated fusion (V1 strategy)
        fused_results = self._fuse_results(bm25_results, vector_results, query)
        # Metrics: fused top-1 score
        try:
            if fused_results:
                RAG_FUSED_TOP1_SCORE.observe(float(fused_results[0][0]))
        except Exception:
            pass

        # 4b. Candidate selection for enrichment/rerank (no engine-level adaptive cutoff)
        try:
            candidate_k = top_k
            if getattr(settings, 'enable_heuristic_rerank', False) and fused_results:
                rr_top = int(getattr(settings, 'reranker_top_k', 30))
                candidate_k = min(max(candidate_k, top_k), rr_top)
        except Exception:
            candidate_k = top_k
        # Metrics: candidates considered
        try:
            RAG_CANDIDATES_CONSIDERED.observe(int(candidate_k))
        except Exception:
            pass

        # 5. Enrich with contextual metadata on candidate set
        enriched_results = self._enrich_results(fused_results[:candidate_k])

        # 5a. Heuristic reranking on enriched documents (no re-embedding)
        try:
            if getattr(settings, 'enable_heuristic_rerank', False):
                enriched_results = self._heuristic_rerank(enriched_results, query)
        except Exception as e:
            logger.warning(f"Heuristic rerank failed: {e}")

        # 5a.2 Optional MMR diversity to reduce redundancy
        try:
            if getattr(settings, 'enable_mmr_diversity', False) and enriched_results:
                lam = float(getattr(settings, 'mmr_lambda', 0.7))
                k = int(getattr(settings, 'mmr_k', 20))
                enriched_results = self._apply_mmr(enriched_results, lam=lam, k=k)
                try:
                    RAG_MMR_APPLIED.inc()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"MMR diversity failed: {e}")

        # 5b. Apply Valmiki-only filter to exclude Uttara Kanda if configured
        try:
            if getattr(settings, 'restrict_to_valmiki', False) and getattr(settings, 'disallow_uttara_kanda_mentions', False):
                enriched_results = [r for r in enriched_results if str(r.kanda_name or '').lower() != 'uttara kanda']
        except Exception:
            pass
        
        # 5c. Optional cross-encoder reranking for semantic precision on top candidates
        try:
            enriched_results = self._cross_encoder_rerank(enriched_results, query)
        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed: {e}")

        # 6. Entity annotation
        if include_entities:
            enriched_results = self._annotate_entities(enriched_results, query)
        
        # 7. Final cut to top_k
        final_results = enriched_results[:top_k]
        logger.info(f" Hybrid search complete: {len(final_results)} results")
        # Populate cache
        try:
            if getattr(settings, 'retrieval_cache_enabled', True):
                ttl = int(getattr(settings, 'retrieval_cache_ttl_secs', 60))
                max_entries = int(getattr(settings, 'retrieval_cache_max_entries', 256))
                key = f"{query}|{top_k}|{int(include_entities)}"
                self._cache[key] = (time.time(), final_results)
                # Evict oldest if over capacity
                if len(self._cache) > max_entries:
                    oldest_key = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
                    self._cache.pop(oldest_key, None)
        except Exception:
            pass
        return final_results
    
    def _enhance_query_with_entities(self, query: str) -> str:
        """
        Enhance query with resolved entities and their aliases (V1 intelligence)
        """
        
        # Resolve entities in the query
        resolved_entities = self.entity_resolver.resolve_multiple(query)
        
        if not resolved_entities:
            return query
        
        # Build enhanced query with aliases and context using quoted OR terms
        enhancement_terms: List[str] = []
        
        # Limits and gating from settings
        try:
            alias_limit = max(0, int(getattr(settings, 'hybrid_alias_limit', 1)))
        except Exception:
            alias_limit = 1
        enable_ctx = bool(getattr(settings, 'enable_entity_context_in_hybrid', False))
        try:
            ctx_limit = max(0, int(getattr(settings, 'hybrid_context_limit', 0))) if enable_ctx else 0
        except Exception:
            ctx_limit = 0

        def _sanitize(term: str) -> str:
            try:
                t = str(term or "")
                t = t.replace('"', ' ')
                t = re.sub(r"[()]+", " ", t)
                t = re.sub(r"[-_]+", " ", t)
                t = re.sub(r"\s+", " ", t).strip()
                return t
            except Exception:
                return term

        def _quote(term: str) -> str:
            s = _sanitize(term)
            return f'"{s}"' if s else s

        def _is_english(s: str) -> bool:
            try:
                return bool(re.fullmatch(r"[A-Za-z0-9 ,.;:!?\-]+", s))
            except Exception:
                return True

        # Always include the original query as a quoted phrase first
        if query and query.strip():
            enhancement_terms.append(_quote(query))

        # Use settings-based confidence threshold (align with resolver/retriever)
        try:
            conf_thresh = float(getattr(settings, 'entity_confidence_threshold', 0.7))
        except Exception:
            conf_thresh = 0.7

        # Use only the top-1 high-confidence entity to avoid bleed from secondary fuzzy matches
        top_entities = [e for e in resolved_entities if float(e.confidence or 0.0) >= conf_thresh][:1]

        for entity in top_entities:
            # Canonical name
            canon = _sanitize(entity.canonical_name)
            if not getattr(settings, 'english_only_query_expansion', True) or _is_english(canon):
                enhancement_terms.append(_quote(canon))

            # High-value aliases
            for alias in entity.aliases[:alias_limit]:
                if isinstance(alias, str) and alias.strip() and alias.lower() != entity.canonical_name.lower():
                    a = _sanitize(alias)
                    if not getattr(settings, 'english_only_query_expansion', True) or _is_english(a):
                        enhancement_terms.append(_quote(a))

            # Optional context phrases for disambiguation
            if enable_ctx and ctx_limit > 0:
                added = 0
                for context in entity.context[:ctx_limit]:
                    if not isinstance(context, str) or not context.strip():
                        continue
                    c = _sanitize(context)
                    if not getattr(settings, 'english_only_query_expansion', True) or _is_english(c):
                        enhancement_terms.append(_quote(c))
                        added += 1
                        if added >= ctx_limit:
                            break

            # Optional: add a couple of salient description terms (feature-flag)
            try:
                if getattr(settings, 'enable_description_expansion', False):
                    # Fetch raw entity metadata using resolver's normalization
                    try:
                        norm = self.entity_resolver._normalize_text(entity.canonical_name)
                        meta = self.entity_resolver.canonical_lookup.get(norm)
                    except Exception:
                        meta = None
                    if meta and isinstance(meta.get('description'), str):
                        import re as _re
                        desc = meta['description']
                        cands = _re.findall(r"\b[A-Z][a-z]{3,}\b", desc)
                        stop = {"Rama","Sita","Hanuman","Ravana","King","Prince","Son","Daughter","Mother","Father"}
                        picked: List[str] = []
                        for w in cands:
                            if w in stop:
                                continue
                            if w not in picked:
                                picked.append(w)
                            if len(picked) >= 2:
                                break
                        for w in picked:
                            enhancement_terms.append(_quote(w))
            except Exception:
                pass

        # Deduplicate while preserving order
        deduped: List[str] = []
        seen = set()
        for t in enhancement_terms:
            if t and t not in seen:
                seen.add(t)
                deduped.append(t)

        # Join with OR for FTS to treat as disjunctive expansion
        enhanced_query = ' OR '.join(deduped) if deduped else query

        # Trim overly long queries to avoid search issues
        if len(enhanced_query) > 220:
            enhanced_query = ' OR '.join(deduped[:5])

        logger.debug(f"Enhanced query: '{query}' → '{enhanced_query}'")
        return enhanced_query
    
    def _heuristic_rerank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Apply lightweight heuristic reranking on enriched results.
        Signals: exact phrase, entity overlap, title/kanda alignment, reasonable length.
        Also adds small penalties for extremely short/long content and a small boost
        when window-level English content is reasonably sized."""
        try:
            q = (query or '').strip().lower()
            # Resolve query entities once
            ents = []
            try:
                ents = self.entity_resolver.resolve_multiple(query)
            except Exception:
                ents = []
            ent_names = {e.canonical_name.lower() for e in (ents or [])}

            scored: List[Tuple[float, SearchResult]] = []
            for r in results:
                boost = 0.0
                # Exact phrase (English)
                try:
                    en = (r.translation_en or '').lower()
                    if q and q in en:
                        boost += 0.02
                except Exception:
                    pass
                # Entity overlap
                try:
                    hay = f"{(r.translation_en or '').lower()} {' '.join(r.sarga_titles or [])}"
                    matches = sum(1 for n in ent_names if n and n in hay)
                    boost += min(0.05, 0.01 * matches)
                except Exception:
                    pass
                # Kanda alignment
                try:
                    kn = (r.kanda_name or '').lower()
                    if q and any(tok in kn for tok in q.split() if len(tok) >= 5):
                        boost += 0.03
                except Exception:
                    pass
                # Reasonable length of English translation
                try:
                    ln = len((r.translation_en or '').strip())
                    if 60 <= ln <= 400:
                        boost += 0.01
                    elif ln < 30:
                        boost -= 0.01
                    elif ln > 600:
                        boost -= 0.01
                except Exception:
                    pass
                # Window-level English content size (if available)
                try:
                    wln = len(getattr(r, 'window_translation_en', '') or '')
                    if 120 <= wln <= 1200:
                        boost += 0.01
                except Exception:
                    pass
                final = r.final_score + boost
                r.final_score = final
                scored.append((final, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored]
        except Exception:
            return results

    def _apply_mmr(self, results: List[SearchResult], lam: float = 0.7, k: int = 20) -> List[SearchResult]:
        """Apply Maximal Marginal Relevance using token-set similarity on English snippets.
        lam controls relevance vs diversity (higher favors relevance)."""
        try:
            if not results:
                return results
            k = max(1, min(k, len(results)))
            selected: List[SearchResult] = []
            candidates = results[:]
            # Precompute text
            def text_of(r: SearchResult) -> str:
                try:
                    return f"{r.translation_en or ''} {' '.join(r.sarga_titles or [])}".strip()
                except Exception:
                    return r.translation_en or ''
            texts = {id(r): text_of(r) for r in candidates}
            # Prepare verse_ids sets for overlap-based redundancy if available
            verse_sets = {}
            try:
                for r in candidates:
                    s = set(r.verse_ids or [])
                    if s:
                        verse_sets[id(r)] = s
            except Exception:
                verse_sets = {}
            # Select the first (highest score)
            selected.append(candidates.pop(0))
            while candidates and len(selected) < k:
                best_idx = 0
                best_score = -1e9
                for idx, c in enumerate(candidates):
                    # Base relevance from current ordering (final_score)
                    try:
                        rel = float(getattr(c, 'final_score', 0.0) or 0.0)
                    except Exception:
                        rel = 0.0
                    # Diversity penalty: max similarity to any selected
                    try:
                        sim = 0.0
                        for s in selected:
                            # Textual redundancy
                            t1 = texts.get(id(c), '')
                            t2 = texts.get(id(s), '')
                            if t1 and t2:
                                sim = max(sim, fuzz.token_set_ratio(t1, t2) / 100.0)
                            # Verse-ID overlap redundancy (Jaccard)
                            v1 = verse_sets.get(id(c))
                            v2 = verse_sets.get(id(s))
                            if v1 and v2:
                                inter = len(v1 & v2)
                                union = len(v1 | v2) or 1
                                jacc = inter / union
                                sim = max(sim, jacc)
                    except Exception:
                        sim = 0.0
                    mmr = lam * rel - (1 - lam) * sim
                    if mmr > best_score:
                        best_score = mmr
                        best_idx = idx
                selected.append(candidates.pop(best_idx))
            return selected + candidates
        except Exception:
            return results
    
    def _get_bm25_results(self, query: str, limit: int) -> List[Dict]:
        """Get BM25/FTS5 search results"""
        
        try:
            # Base BM25 across all fields (already sanitized and normalized)
            results = search_bm25(
                db_path=self.db_path,
                fts_table=self.fts_table,
                query=query,
                limit=limit
            )
            # Fielded BM25: prefer English translations when available
            try:
                fielded = search_bm25_fielded(
                    db_path=self.db_path,
                    fts_table=self.fts_table,
                    query=query,
                    field="translation_en",
                    limit=min(limit, int(getattr(settings, 'fielded_top_k', 40)))
                )
            except Exception:
                fielded = []

            # Kanda-filtered BM25 if query hints at a specific kanda
            try:
                kanda_name = self._detect_kanda(query)
                filtered = []
                if kanda_name:
                    filtered = search_bm25_with_filters(
                        db_path=self.db_path,
                        fts_table=self.fts_table,
                        query=query,
                        limit=min(limit, 20),
                        kanda_name=kanda_name
                    )
            except Exception:
                filtered = []

            # Deduplicate by verse_id while keeping the highest BM25 similarity
            merged: Dict[str, Dict] = {}
            def _merge(items: List[Dict]):
                for it in (items or []):
                    vid = it.get('verse_id')
                    if not vid:
                        continue
                    prev = merged.get(vid)
                    if not prev or float(it.get('score', 0.0)) > float(prev.get('score', 0.0)):
                        merged[vid] = it

            _merge(results)
            _merge(fielded)
            _merge(filtered)

            merged_list = list(merged.values())
            logger.debug(f"BM25 returned {len(merged_list)} results (merged)")
            return merged_list
            
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    
    def _get_vector_results(self, query: str, limit: int) -> List[Dict]:
        """Get vector similarity results"""
        
        try:
            if not self.vector_store:
                return []
            engine = VectorSearchEngine(self.vector_store)
            vresults = engine.search(query, top_k=limit)
            # Map VectorSearchResult -> dict expected downstream
            mapped: List[Dict] = []
            for vr in vresults:
                d = {
                    'verse_id': vr.verse_id,
                    'score': vr.similarity_score,
                    'search_type': 'vector_similarity'
                }
                # include metadata (contains mapping info)
                if isinstance(vr.metadata, dict):
                    d.update(vr.metadata)
                mapped.append(d)
            logger.debug(f"Vector search returned {len(mapped)} results")
            return mapped
        
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _fuse_results(self, bm25_results: List[Dict], vector_results: List[Dict], original_query: str) -> List[Tuple[float, Dict]]:
        """
        Sophisticated result fusion with normalized scores and dynamic weighting
        """
        
        combined_results: Dict[str, Dict] = {}

        # Prepare per-batch normalizers
        bm_raw = [float(r.get('score', 0.0) or 0.0) for r in (bm25_results or [])]
        vc_raw = [float(r.get('score', 0.0) or 0.0) for r in (vector_results or [])]

        def make_normalizer(values: List[float]):
            norm_type = str(getattr(settings, 'hybrid_score_norm', 'minmax') or 'minmax').lower()
            log_scale = bool(getattr(settings, 'hybrid_log_scale', True))
            alpha = float(getattr(settings, 'hybrid_log_alpha', 1.5))
            if not values:
                def _zero(_: float) -> float:
                    return 0.0
                return _zero
            if norm_type == 'zscore':
                mu = mean(values)
                try:
                    sd = pstdev(values)
                except Exception:
                    sd = 0.0
                if sd <= 1e-9:
                    # Degenerate distribution: map all to mid value
                    def _const(_: float) -> float:
                        v = 1.0
                        # optional log compression keeps 1.0 invariant
                        return v
                    return _const
                def _z(x: float) -> float:
                    # Map z-score to [0,1] via sigmoid
                    z = (x - mu) / sd
                    s = 1.0 / (1.0 + math.exp(-z))
                    if log_scale:
                        # Compress peaks while preserving order in [0,1]
                        return math.log1p(alpha * s) / math.log1p(alpha)
                    return s
                return _z
            # default: min-max
            mn = min(values)
            mx = max(values)
            if mx - mn <= 1e-12:
                def _ones(_: float) -> float:
                    return 1.0
                return _ones
            def _mm(x: float) -> float:
                s = (x - mn) / (mx - mn)
                if log_scale:
                    return math.log1p(alpha * s) / math.log1p(alpha)
                return s
            return _mm

        bm_norm = make_normalizer(bm_raw)
        vc_norm = make_normalizer(vc_raw)

        # Collect maxima for dynamic weighting after normalization
        try:
            bm25_max = max((bm_norm(r.get('score', 0.0) or 0.0) for r in bm25_results), default=0.0)
        except Exception:
            bm25_max = 0.0
        try:
            vec_max = max((vc_norm(r.get('score', 0.0) or 0.0) for r in vector_results), default=0.0)
        except Exception:
            vec_max = 0.0

        # Effective weights (optionally dynamic)
        bm_w = self.bm25_weight
        vc_w = self.vector_weight
        try:
            if getattr(settings, 'dynamic_hybrid_weights', False):
                if bm25_max > 0.7 and bm25_max > (vec_max + 0.05):
                    bm_w = min(0.8, bm_w + 0.2)
                    vc_w = max(0.2, 1.0 - bm_w)
                elif vec_max > 0.7 and vec_max > (bm25_max + 0.05):
                    vc_w = min(0.8, vc_w + 0.2)
                    bm_w = max(0.2, 1.0 - vc_w)
        except Exception:
            pass
        # Normalize to sum 1.0
        try:
            total_w = bm_w + vc_w
            if total_w > 0:
                bm_w, vc_w = bm_w / total_w, vc_w / total_w
        except Exception:
            pass
        
        # Process BM25 results using batch-normalized similarity
        for rank, result in enumerate(bm25_results):
            verse_id = result.get('verse_id', '')
            if not verse_id:
                continue
            bm25_score = bm_norm(float(result.get('score', 0.0) or 0.0))
            combined_results[verse_id] = {
                'result': result,
                'bm25_score': bm25_score,
                'vector_score': 0.0,
                'bm25_rank': rank,
                'vector_rank': None
            }

        # Process vector results using batch-normalized vector similarity
        for rank, result in enumerate(vector_results):
            verse_id = result.get('verse_id', '')
            if not verse_id:
                continue
            vector_score = vc_norm(float(result.get('score', 0.0) or 0.0))
            # If this vector result is a window, and any BM25 verse_id lies within its verse_ids,
            # merge under that BM25 verse_id to align units for fusion.
            fusion_key = verse_id
            try:
                vset = set(result.get('verse_ids', []) or [])
                if vset:
                    # Find first BM25 key that is contained in this window
                    for bm_key in list(combined_results.keys()):
                        if bm_key in vset:
                            fusion_key = bm_key
                            break
            except Exception:
                pass

            # Ensure the result dict reflects the fusion key for downstream enrichment
            if fusion_key != verse_id:
                result['verse_id'] = fusion_key

            if fusion_key in combined_results:
                combined_results[fusion_key]['vector_score'] = vector_score
                combined_results[fusion_key]['vector_rank'] = rank
                # Prefer richer metadata (keep window metadata fields if present)
                try:
                    combined_results[fusion_key]['result'].update(result)
                except Exception:
                    pass
            else:
                combined_results[fusion_key] = {
                    'result': result,
                    'bm25_score': 0.0,
                    'vector_score': vector_score,
                    'bm25_rank': None,
                    'vector_rank': rank
                }
        
        # Calculate final weighted scores
        scored_results: List[Tuple[float, Dict]] = []
        for verse_id, data in combined_results.items():
            base_score = (bm_w * data['bm25_score']) + (vc_w * data['vector_score'])
            # Apply query-specific boosting (V1 intelligence)
            final_score = self._apply_query_boosting(base_score, data['result'], original_query)
            scored_results.append((final_score, data['result']))
        
        # Sort by final score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"Fused {len(scored_results)} unique results (bm_w={bm_w:.2f}, vc_w={vc_w:.2f})")
        return scored_results

    def _cross_encoder_rerank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Optional lightweight cross-encoder reranker for top-N candidates.
        Combines normalized cross-encoder scores with normalized current final scores.
        """
        try:
            if not results:
                return results
            if not getattr(settings, 'enable_cross_encoder_rerank', False):
                return results
            top_n = max(1, int(getattr(settings, 'cross_encoder_top_k', 30)))
            top_n = min(top_n, len(results))
            # Time budget: if model isn't loaded and budget is tiny, skip
            budget = float(getattr(settings, 'cross_encoder_timeout_secs', 1.0))
            t0 = time.time()

            # Lazy-load CrossEncoder
            if self._ce_model is None:
                # If the budget is very small, avoid loading large weights
                if budget < 0.5:
                    return results
                try:
                    from sentence_transformers import CrossEncoder
                    model_name = str(getattr(settings, 'cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'))
                    self._ce_model = CrossEncoder(model_name)
                except Exception as e:
                    logger.warning(f"Cross-encoder load failed: {e}")
                    return results

            # Build query-document pairs
            pairs = []
            selected = results[:top_n]
            for r in selected:
                text = (r.window_translation_en or '').strip() or (r.translation_en or '').strip() or (r.shloka_iast or '').strip()
                # Include titles lightly
                if r.sarga_titles:
                    text = ("; ".join(r.sarga_titles) + ": " + text).strip()
                # Fallback to kanda name if text empty
                if not text:
                    text = f"{r.kanda_name} {r.verse_index}"
                pairs.append((query, text))

            # Predict scores (may take a few hundred ms)
            try:
                ce_scores = self._ce_model.predict(pairs, show_progress_bar=False)
            except Exception as e:
                logger.warning(f"Cross-encoder inference failed: {e}")
                return results

            # Respect time budget
            if (time.time() - t0) > budget:
                return results

            # Normalize CE scores to [0,1]
            ce_scores = [float(s) for s in ce_scores]
            cmin, cmax = min(ce_scores), max(ce_scores)
            ce_norm = [(s - cmin) / (cmax - cmin) if cmax > cmin else 1.0 for s in ce_scores]

            # Normalize current final scores of the selected set to [0,1]
            cur = [float(getattr(r, 'final_score', 0.0) or 0.0) for r in selected]
            mni, mxi = min(cur), max(cur)
            cur_norm = [(v - mni) / (mxi - mni) if mxi > mni else 1.0 for v in cur]

            w = float(getattr(settings, 'cross_encoder_weight', 0.6))
            combined: List[Tuple[float, SearchResult]] = []
            for idx, r in enumerate(selected):
                f = (1.0 - w) * cur_norm[idx] + w * ce_norm[idx]
                r.final_score = f
                combined.append((f, r))

            # Sort the reranked head and append the tail unchanged
            combined.sort(key=lambda x: x[0], reverse=True)
            head = [r for _, r in combined]
            tail = results[top_n:]
            return head + tail
        except Exception:
            return results
    
    def _apply_query_boosting(self, base_score: float, result: Dict, query: str) -> float:
        """
        Apply query-specific score boosting (V1 intelligence)
        """
        
        boosted_score = base_score
        query_lower = query.lower()
        
        # Boost results where query terms appear in multiple fields
        fields_to_check = ['translation_en', 'shloka_iast']
        matching_fields = 0
        
        for field in fields_to_check:
            field_content = str(result.get(field, '')).lower()
            if any(term in field_content for term in query_lower.split()):
                matching_fields += 1
        
        # Multi-field boost
        if matching_fields >= 2:
            boosted_score *= 1.1
        if matching_fields >= 3:
            boosted_score *= 1.2
        
        # Boost important kandas (narrative significance)
        kanda_name = result.get('kanda_name', '').lower()
        important_kandas = ['sundara kanda', 'yuddha kanda', 'ayodhya kanda']
        
        if any(important in kanda_name for important in important_kandas):
            boosted_score *= 1.05

        # Extra boost when query explicitly mentions a kanda matching this result
        try:
            q_kanda = self._detect_kanda(query)
            if q_kanda and q_kanda.lower() == result.get('kanda_name', '').lower():
                boosted_score *= 1.05
        except Exception:
            pass
        
        return boosted_score
    
    def _enrich_results(self, scored_results: List[Tuple[float, Dict]]) -> List[SearchResult]:
        """
        Enrich results with contextual metadata (V1 approach)
        """
        
        if not scored_results:
            return []
        
        # Collect all verse_ids we may want to enrich: anchors + full windows
        anchor_ids = []
        all_ids_set = set()
        for _, res in scored_results:
            vid = res.get('verse_id')
            if vid:
                anchor_ids.append(vid)
                all_ids_set.add(vid)
            vlist = res.get('verse_ids') or []
            for v in vlist:
                all_ids_set.add(v)
        all_ids = [v for v in all_ids_set if v]
        if not all_ids:
            return []
        
        # Batch fetch enriched data for all involved verses
        placeholders = ','.join('?' * len(all_ids))
        query = f"""
        SELECT 
            verse_id, kanda_name, sarga_num, verse_index,
            translation_en, translation_hi, shloka_sanskrit, shloka_iast,
            sarga_titles_json
        FROM verses 
        WHERE verse_id IN ({placeholders})
        """
        try:
            dbm = get_database_manager(self.db_path)
            with dbm.get_connection() as conn:
                rows = conn.execute(query, all_ids).fetchall()
            enrichment_data = {row['verse_id']: dict(row) for row in rows}
        except Exception as e:
            logger.error(f"Error enriching results: {e}")
            enrichment_data = {}
        
        # Build enriched results
        enriched_results = []
        
        for final_score, result in scored_results:
            verse_id = result.get('verse_id', '')
            enrichment = enrichment_data.get(verse_id, {})

            # If enrichment is missing for a single-verse anchor (common when the DB stores
            # merged multi-verse entries like k-s-v1-v2), resolve the proper combined verse_id
            # that contains this verse index. This ensures compact range citations (e.g., 2.12.27-28)
            # and prevents fallback to large window content.
            if not enrichment and isinstance(verse_id, str):
                try:
                    import re as _re
                    m = _re.match(r"^(\d+)-(\d+)-(\d+)$", verse_id)
                    if m:
                        k_num = int(m.group(1))
                        s_num = int(m.group(2))
                        v_idx = int(m.group(3))
                        # Search for any merged range row k-s-v1-v2 where v1 <= v_idx <= v2
                        dbm = get_database_manager(self.db_path)
                        with dbm.get_connection() as _conn:
                            rows = _conn.execute(
                                """
                                SELECT verse_id, kanda_name, sarga_num, verse_index,
                                       translation_en, translation_hi, shloka_sanskrit, shloka_iast,
                                       sarga_titles_json
                                FROM verses
                                WHERE kanda_num = ? AND sarga_num = ?
                                  AND verse_index <= ?
                                  AND verse_id LIKE ?
                                ORDER BY verse_index DESC
                                LIMIT 50
                                """,
                                (k_num, s_num, v_idx, f"{k_num}-{s_num}-%-%")
                            ).fetchall()
                        for r in rows or []:
                            vid = str(r["verse_id"] or "")
                            m2 = _re.match(r"^(\d+)-(\d+)-(\d+)-(\d+)$", vid)
                            if not m2:
                                continue
                            v_start = int(m2.group(3))
                            v_end = int(m2.group(4))
                            if v_start <= v_idx <= v_end:
                                # Found the combined entry; switch to it
                                enrichment = dict(r)
                                enrichment_data[vid] = enrichment
                                verse_id = vid
                                # Also update the fused result verse_id so downstream uses the range
                                try:
                                    result['verse_id'] = vid
                                except Exception:
                                    pass
                                break
                except Exception:
                    # On any error, keep original verse_id and proceed with best-effort enrichment
                    pass
            
            # Parse sarga titles
            try:
                sarga_titles = json.loads(enrichment.get('sarga_titles_json', '[]'))
            except (json.JSONDecodeError, TypeError):
                sarga_titles = []

            # Build window-level concatenations (English + IAST) if window metadata present
            window_ids = result.get('verse_ids') or []
            window_en = []
            window_iast = []
            window_hi = []
            window_skt = []
            if window_ids:
                for vid in window_ids:
                    r = enrichment_data.get(vid, {})
                    # Compose compact lines: "K.S.V text"
                    try:
                        ksv = str(vid).replace('-', '.')
                    except Exception:
                        ksv = str(vid)
                    en_txt = (r.get('translation_en') or '').strip()
                    iast_txt = (r.get('shloka_iast') or '').strip()
                    hi_txt = (r.get('translation_hi') or '').strip()
                    skt_txt = (r.get('shloka_sanskrit') or '').strip()
                    if en_txt:
                        window_en.append(f"{ksv} {en_txt}")
                    if iast_txt:
                        window_iast.append(f"{ksv} {iast_txt}")
                    if hi_txt:
                        window_hi.append(f"{ksv} {hi_txt}")
                    if skt_txt:
                        window_skt.append(f"{ksv} {skt_txt}")
            # Fallback: if no window, leave empty strings
            window_en_text = ' '.join(window_en) if window_en else ''
            window_iast_text = ' '.join(window_iast) if window_iast else ''
            window_hi_text = ' '.join(window_hi) if window_hi else ''
            window_skt_text = ' '.join(window_skt) if window_skt else ''

            # Robust fallbacks when enrichment data is missing (e.g., vector-only windows
            # whose anchor verse_id doesn't exist as a standalone row because DB stores
            # a combined multi-verse id). Use metadata from fused 'result' when available.
            def _kanda_num_to_name(kn: int) -> str:
                mapping = {
                    1: 'Baala Kanda',
                    2: 'Ayodhya Kanda',
                    3: 'Aranya Kanda',
                    4: 'Kishkindha Kanda',
                    5: 'Sundara Kanda',
                    6: 'Yuddha Kanda'
                }
                return mapping.get(int(kn or 0), 'Unknown')

            kanda_name = enrichment.get('kanda_name')
            if not kanda_name:
                kanda_name = result.get('kanda_name') or _kanda_num_to_name(result.get('kanda_num') or result.get('kanda'))

            sarga_num = enrichment.get('sarga_num')
            if not sarga_num:
                sarga_num = result.get('sarga_num') or result.get('sarga') or 0

            verse_index_val = enrichment.get('verse_index')
            if not verse_index_val:
                verse_index_val = result.get('verse_index') or result.get('verse') or 0

            # Titles fallback from vector metadata ('titles') or fused 'result'
            if not sarga_titles:
                try:
                    meta_titles = result.get('sarga_titles') or result.get('titles') or []
                    if isinstance(meta_titles, list):
                        sarga_titles = [t for t in meta_titles if isinstance(t, str)]
                    elif isinstance(meta_titles, str) and meta_titles.strip():
                        sarga_titles = [meta_titles]
                except Exception:
                    sarga_titles = []

            # Content fallbacks (BM25 results already carry these; vector-only may not)
            translation_en = enrichment.get('translation_en') or result.get('translation_en') or ''
            translation_hi = enrichment.get('translation_hi') or result.get('translation_hi') or ''
            shloka_sanskrit = enrichment.get('shloka_sanskrit') or result.get('shloka_sanskrit') or ''
            shloka_iast = enrichment.get('shloka_iast') or result.get('shloka_iast') or ''

            # If anchor-level verse text is missing (common for window-only vector hits),
            # fallback to aggregated window-level content so UI shows complete details.
            if not translation_en and window_en_text:
                translation_en = window_en_text
            if not shloka_iast and window_iast_text:
                shloka_iast = window_iast_text
            if not translation_hi and window_hi_text:
                translation_hi = window_hi_text
            if not shloka_sanskrit and window_skt_text:
                shloka_sanskrit = window_skt_text

            search_result = SearchResult(
                verse_id=verse_id,
                # Leave empty when unknown so frontend shows 'Unknown Kanda' instead of literal 'Unknown'
                kanda_name=kanda_name or '',
                sarga_num=int(sarga_num or 0),
                verse_index=int(verse_index_val or 0),
                translation_en=translation_en,
                translation_hi=translation_hi,
                shloka_sanskrit=shloka_sanskrit,
                shloka_iast=shloka_iast,
                sarga_titles=sarga_titles,
                bm25_score=result.get('bm25_score', 0.0),
                vector_score=result.get('vector_score', 0.0),
                final_score=final_score,
                entities_found=[],  # Will be populated by _annotate_entities
                # Window metadata passthrough (if present in fused result)
                verse_ids=window_ids,
                start_verse_index=int(result.get('start_verse_index', 0) or 0),
                end_verse_index=int(result.get('end_verse_index', 0) or 0),
                anchor_verse_id=result.get('anchor_verse_id', ''),
                window_translation_en=window_en_text,
                window_shloka_iast=window_iast_text,
            )

            enriched_results.append(search_result)
        
        return enriched_results

    def _detect_kanda(self, query: str) -> Optional[str]:
        """Lightweight kanda name detector from free-text query.
        Returns canonical kanda name (e.g., 'Sundara Kanda') or None.
        """
        try:
            q = (query or '').lower()
            mapping = {
                'bala': 'Baala Kanda', 'baala': 'Baala Kanda', 'bala kanda': 'Baala Kanda', 'baala kanda': 'Baala Kanda',
                'ayodhya': 'Ayodhya Kanda', 'ayodhya kanda': 'Ayodhya Kanda',
                'aranya': 'Aranya Kanda', 'aranya kanda': 'Aranya Kanda',
                'kishkindha': 'Kishkindha Kanda', 'kishkinda': 'Kishkindha Kanda', 'kishkindha kanda': 'Kishkindha Kanda',
                'sundara': 'Sundara Kanda', 'sundar': 'Sundara Kanda', 'sundara kanda': 'Sundara Kanda',
                'yuddha': 'Yuddha Kanda', 'yudha': 'Yuddha Kanda', 'lanka': 'Yuddha Kanda', 'yuddha kanda': 'Yuddha Kanda',
            }
            # Look for longest matching key present in query
            candidates = [mapping[k] for k in mapping.keys() if k in q]
            if not candidates:
                return None
            # Prefer specific full names where present
            priority = [
                'Baala Kanda', 'Ayodhya Kanda', 'Aranya Kanda', 'Kishkindha Kanda', 'Sundara Kanda', 'Yuddha Kanda'
            ]
            for name in priority:
                if name in candidates:
                    return name
            return candidates[0]
        except Exception:
            return None
    
    def _annotate_entities(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Annotate results with found entities (V1 intelligence)
        """
        
        for result in results:
            # Combine text fields for entity detection
            combined_text = f"{result.translation_en} {' '.join(result.sarga_titles)}"
            try:
                found_entities = self.entity_resolver.resolve_multiple(combined_text)
                # Fallback to query-based detection if none found in text
                if not found_entities and query:
                    found_entities = self.entity_resolver.resolve_multiple(query)

                unique_names = []
                seen = set()
                for ent in (found_entities or []):
                    name = ent.canonical_name
                    if name and name not in seen:
                        seen.add(name)
                        unique_names.append(name)
                result.entities_found = unique_names
            except Exception as e:
                logger.warning(f"Entity annotation failed: {e}")
                result.entities_found = []
        
        return results

    def get_search_stats(self) -> Dict:
        """Return search engine and vector store statistics"""
        stats: Dict[str, any] = {
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "db_path": self.db_path,
        }
        try:
            if hasattr(self.vector_store, 'get_stats'):
                stats["vector_store"] = self.vector_store.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get vector store stats: {e}")
            stats["vector_store"] = {"error": str(e)}
        return stats
