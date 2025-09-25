"""
VANI V3 RAG Retriever - Orchestrated Retrieval with Entity Intelligence
Enforcing: Relationship-first processing and entity-enhanced search
"""

import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from core.entities.resolver import EntityResolver, ResolvedEntity
from core.search.text_search import search_bm25
from core.entities.relationship import RelationshipGraph, RelationshipType
from core.search.hybrid_engine import HybridSearchEngine, SearchResult
from app.settings import settings
from rag.query_router import QueryRouter

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    type: str  # "direct_relationship", "entity_enhanced", "standard_search"
    query: str
    expanded_query: Optional[str]
    results: List[SearchResult]
    entities_found: List[ResolvedEntity]
    confidence: float
    processing_time: float


class RAGRetriever:
    """
    Production RAG Retrieval Orchestrator
    Combines entity resolution, relationship intelligence, and hybrid search
    """
    
    def __init__(self, 
                 search_engine: HybridSearchEngine,
                 entity_resolver: EntityResolver,
                 relationship_graph: RelationshipGraph | None = None,
                 confidence_threshold: float = 0.7,
                 router: QueryRouter | None = None):
        
        self.search_engine = search_engine
        self.entity_resolver = entity_resolver
        self.relationship_graph = relationship_graph
        # Router for intent/intent-confidence gating
        self.router = router or QueryRouter(settings.database_path)
        self.confidence_threshold = confidence_threshold
        
        logger.info("✅ RAG Retriever initialized")
    
    def retrieve(self, 
                query: str, 
                top_k: int = 10,
                include_context: bool = True,
                entity_boost: bool = True) -> RetrievalResult:
        """
        Main retrieval method with intelligent processing pipeline
        """
        
        import time
        start_time = time.time()
        
        logger.info(f"Retrieving for query: '{query}' (top_k={top_k})")
        
        # 0. Analyze query intent to gate entity expansion
        try:
            qa = self.router.analyze_query(query)
        except Exception:
            qa = None
        allow_by_router = False
        try:
            if qa and qa.type in {"relationship", "narrative"}:
                allow_by_router = float(qa.confidence or 0.0) >= float(getattr(settings, 'entity_expansion_router_confidence', 0.8))
        except Exception:
            allow_by_router = False
        # Optional support for general queries if explicitly enabled
        if (qa and qa.type == "general" and getattr(settings, 'entity_boost_for_general', False)):
            allow_by_router = True

        # Final allowance respects caller flag too
        allow_entity_boost = bool(entity_boost) and bool(allow_by_router)

        # 1. Entity resolution and enhancement (only when allowed)
        entities_found = []
        expanded_query = query
        if allow_entity_boost:
            entities_found = self.entity_resolver.resolve_multiple(query)
            high_confidence_entities = [
                entity for entity in entities_found 
                if entity.confidence >= self.confidence_threshold
            ]
            if high_confidence_entities:
                expanded_query = self._enhance_query_with_entities(query, high_confidence_entities)
                logger.debug(f"Entity-enhanced query: '{expanded_query}'")
        
        # 2. Optional PRF expansion (English-only) after entity enhancement
        # Guard: if the query is already an OR-joined quoted expansion, skip PRF to avoid noise
        # Additionally: only allow PRF when entity_boost is enabled for this query
        try:
            if allow_entity_boost and getattr(settings, 'enable_prf', False):
                if (' OR ' not in expanded_query) and (expanded_query.count('"') < 4):
                    expanded_query = self._apply_prf(expanded_query)
        except Exception as e:
            logger.warning(f"PRF expansion failed: {e}")

        # 3. Execute hybrid search (over-fetch for adaptive retrieval if enabled)
        fetch_k = top_k
        if getattr(settings, 'adaptive_retrieval', False):
            try:
                fetch_k = max(top_k, getattr(settings, 'adaptive_max_k', top_k))
            except Exception:
                fetch_k = top_k

        search_results = self.search_engine.search(
            query=expanded_query,
            top_k=fetch_k,
            include_entities=allow_entity_boost
        )
        # 3b. Adaptive cut-off based on score drop ratio
        if getattr(settings, 'adaptive_retrieval', False) and search_results:
            try:
                base = max(0.0, float(search_results[0].final_score))
                drop_ratio = float(getattr(settings, 'adaptive_drop_ratio', 0.15))
                min_k = int(getattr(settings, 'adaptive_min_k', 5))
                # Determine cut index
                cut = len(search_results)
                if base > 0.0:
                    for i in range(min(len(search_results) - 1, fetch_k - 1)):
                        cur = float(search_results[i].final_score)
                        nxt = float(search_results[i + 1].final_score)
                        if (cur - nxt) >= (drop_ratio * base) and (i + 1) >= min_k:
                            cut = i + 1
                            break
                # Respect user-provided top_k as upper bound
                cut = min(cut, top_k)
                search_results = search_results[:max(1, cut)]
            except Exception:
                # On any error, fall back to first top_k
                search_results = search_results[:top_k]
        
        # 4. Determine result type and confidence
        result_type = "standard_search"
        confidence = 0.8  # Base confidence
        
        if entities_found:
            result_type = "entity_enhanced"
            # Boost confidence based on entity quality
            max_entity_confidence = max(entity.confidence for entity in entities_found)
            confidence = min(0.9, 0.7 + (max_entity_confidence * 0.2))
        
        if expanded_query != query:
            result_type = "entity_enhanced"
        
        processing_time = time.time() - start_time
        
        retrieval_result = RetrievalResult(
            type=result_type,
            query=query,
            expanded_query=expanded_query if expanded_query != query else None,
            results=search_results,
            entities_found=entities_found,
            confidence=confidence,
            processing_time=processing_time
        )
        
        logger.info(f"✅ Retrieved {len(search_results)} results in {processing_time:.3f}s")
        return retrieval_result

    def _apply_prf(self, query_text: str) -> str:
        """Apply lightweight pseudo-relevance feedback using top BM25 EN snippets.
        - English-only terms respected via settings.english_only_query_expansion
        - Adds up to prf_terms_k tokens not already present in the query.
        """
        try:
            # Run a small BM25 over-fetch on the current query
            k = int(getattr(settings, 'prf_bm25_k', 8))
            results = search_bm25(
                db_path=settings.database_path,
                fts_table=settings.fts_table,
                query=query_text,
                limit=max(3, min(20, k))
            )
            if not results:
                return query_text
            # Collect English tokens
            import re as _re
            import math as _math
            import sqlite3 as _sqlite3
            min_len = int(getattr(settings, 'prf_min_word_len', 4))
            # Build base token set from the current query to avoid duplicates
            base_tokens = set(t.lower() for t in _re.findall(r"[A-Za-z]+", query_text))
            freq: dict[str, int] = {}
            for r in results:
                en = (r.get('translation_en') or '')
                for tok in _re.findall(r"[A-Za-z]+", en):
                    tl = tok.lower()
                    if len(tl) < min_len:
                        continue
                    # Enforce English-only expansion if configured
                    if getattr(settings, 'english_only_query_expansion', True) and not _re.fullmatch(r"[A-Za-z]+", tl):
                        continue
                    if tl in base_tokens:
                        continue
                    # Basic stopword filter
                    if tl in {"and","or","the","a","an","in","of","to","for","on","with","by","as","is","are","was","were","be","been","this","that","these","those","from","at","it","its","into"}:
                        continue
                    freq[tl] = freq.get(tl, 0) + 1
            if not freq:
                return query_text
            # Try to re-score with IDF from FTS5 vocab if available
            additions: List[str] = []
            try:
                conn = _sqlite3.connect(settings.database_path)
                conn.row_factory = _sqlite3.Row
                # Attempt to create a vocab view/table; if unsupported, this will fail
                try:
                    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.vocab USING fts5vocab(verses_fts, 'row')")
                except Exception:
                    # Some SQLite builds require the DB name without quotes
                    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(verses_fts, row)")
                # Total document count for IDF
                total_docs = 1
                try:
                    cur = conn.execute("SELECT COUNT(*) as c FROM verses")
                    total_docs = int(cur.fetchone()[0]) or 1
                except Exception:
                    total_docs = max(1, sum(1 for _ in results))
                # Query doc frequency for candidate tokens
                cand_list = list(freq.keys())[:50]
                if cand_list:
                    placeholders = ",".join(["?"] * len(cand_list))
                    # Prefer temp.vocab if created, else fallback to main.vocab name
                    tf_rows = []
                    try:
                        cur = conn.execute(f"SELECT term, doc FROM temp.vocab WHERE term IN ({placeholders})", cand_list)
                        tf_rows = cur.fetchall()
                    except Exception:
                        try:
                            cur = conn.execute(f"SELECT term, doc FROM vocab WHERE term IN ({placeholders})", cand_list)
                            tf_rows = cur.fetchall()
                        except Exception:
                            tf_rows = []
                    df_map = {str(r[0]): int(r[1]) for r in (tf_rows or [])}
                    scored: List[tuple[str, float]] = []
                    for t, tf in freq.items():
                        df = max(1, int(df_map.get(t, 1)))
                        idf = _math.log(total_docs / df)
                        scored.append((t, tf * idf))
                    scored.sort(key=lambda kv: kv[1], reverse=True)
                    n = int(getattr(settings, 'prf_terms_k', 2))
                    additions = [w for w, _ in scored[:n]]
                conn.close()
            except Exception:
                additions = []
            # Fallback to pure frequency if vocab path failed/empty
            if not additions:
                n = int(getattr(settings, 'prf_terms_k', 2))
                additions = [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:n]]
            if not additions:
                return query_text
            # Append tokens to the query (let downstream _prepare_fts_query handle quoting)
            expanded = f"{query_text} {' '.join(additions)}".strip()
            # Trim overly long queries
            if len(expanded) > 220:
                expanded = expanded[:220].rsplit(' ', 1)[0]
            logger.debug(f"PRF expanded query: '{query_text}' -> '{expanded}' with terms {additions}")
            return expanded
        except Exception:
            return query_text
    
    def _enhance_query_with_entities(self, 
                                   original_query: str, 
                                   entities: List[ResolvedEntity]) -> str:
        """
        Enhance query with resolved entities (conservative, identity-focused by default).
        - Include sanitized original query as a quoted phrase
        - Add canonical name and a small number of aliases
        - Context expansion is disabled by default and filtered when enabled
        """
        
        # Start with sanitized original query as a quoted phrase to prevent raw syntax issues
        enhancement_terms: List[str] = []
        def _sanitize_phrase_for_fts(term: str) -> str:
            try:
                t = str(term or "")
                t = t.replace('"', ' ')
                t = re.sub(r"[-_]+", " ", t)
                t = re.sub(r"[()]+", " ", t)
                t = re.sub(r"\s+", " ", t).strip()
                return t
            except Exception:
                return term
        if original_query and original_query.strip():
            enhancement_terms.append(f'"{_sanitize_phrase_for_fts(original_query)}"')
        
        def _sanitize_phrase_for_fts(term: str) -> str:
            try:
                t = str(term or "")
                t = t.replace('"', ' ')
                t = re.sub(r"[-_]+", " ", t)
                t = re.sub(r"[()]+", " ", t)
                t = re.sub(r"\s+", " ", t).strip()
                return t
            except Exception:
                return term

        # Controls
        alias_lim = int(getattr(settings, 'retriever_identity_alias_limit', 1))
        expand_ctx = bool(getattr(settings, 'retriever_expand_context', False))
        ban_rel = bool(getattr(settings, 'retriever_hard_filter_relationship_terms', True))
        rel_pat = re.compile(r"\b(son|daughter|child|father|mother|parent|grand\w+|wife|husband|spouse|brother|sister|sibling|ancestor|descendant|lineage)\b", re.IGNORECASE)
        of_pat = re.compile(r"\bof\b", re.IGNORECASE)

        # Use only the top-1 entity to avoid bleed from secondary matches
        for entity in entities[:1]:
            if entity.confidence >= self.confidence_threshold:
                
                # Add canonical name
                enhancement_terms.append(f'"{_sanitize_phrase_for_fts(entity.canonical_name)}"')
                
                # Add best aliases
                for alias in entity.aliases[:max(0, alias_lim)]:
                    if alias.lower() != entity.canonical_name.lower():
                        enhancement_terms.append(f'"{_sanitize_phrase_for_fts(alias)}"')
                
                # Optional contextual phrases (disabled by default)
                if expand_ctx:
                    ctx_added = 0
                    for context in entity.context:
                        if not context:
                            continue
                        c = _sanitize_phrase_for_fts(context)
                        # Filters: keep short, avoid genealogical/"of" patterns, avoid capitalized proper-name phrases
                        if len(c.split()) > 4:
                            continue
                        if ban_rel and (rel_pat.search(c) or of_pat.search(c)):
                            continue
                        if re.search(r"\b[A-Z][a-z]+\b", context):
                            continue
                        enhancement_terms.append(f'"{c}"')
                        ctx_added += 1
                        if ctx_added >= int(getattr(settings, 'retriever_alias_limit', 2)):
                            break

                # Optional: add a couple of salient description terms (feature-flag)
                try:
                    if getattr(settings, 'enable_description_expansion', False):
                        meta = self._get_entity_meta(entity.canonical_name)
                        if meta and isinstance(meta.get('description'), str):
                            desc_terms = self._extract_description_terms(meta['description'], limit=2)
                            for t in desc_terms:
                                enhancement_terms.append(f'"{t}"')
                except Exception:
                    pass
        
        # Deduplicate while preserving order
        seen = set()
        dedup_terms: List[str] = []
        for t in enhancement_terms:
            if t not in seen:
                seen.add(t)
                dedup_terms.append(t)
        enhanced_query = ' OR '.join(dedup_terms)
        
        # Limit query length to prevent search issues
        if len(enhanced_query) > 200:
            enhanced_query = ' OR '.join(enhancement_terms[:5])  # Truncate
        
        return enhanced_query
    
    def retrieve_by_entity(self, 
                          entity_name: str, 
                          top_k: int = 10,
                          include_relationships: bool = True) -> RetrievalResult:
        """
        Retrieve content specifically related to a named entity
        """
        
        import time
        start_time = time.time()
        
        # Resolve the entity
        resolved_entity = self.entity_resolver.resolve(entity_name)
        
        if not resolved_entity:
            logger.warning(f"Could not resolve entity: {entity_name}")
            return RetrievalResult(
                type="entity_not_found",
                query=entity_name,
                expanded_query=None,
                results=[],
                entities_found=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Build comprehensive search query for the entity
        search_terms = [resolved_entity.canonical_name]
        
        # Add aliases
        search_terms.extend(resolved_entity.aliases[:5])
        
        # Add relationship context if requested
        if include_relationships and resolved_entity.context:
            search_terms.extend(resolved_entity.context[:3])
        
        # Create search query
        entity_query = ' OR '.join(f'"{term}"' for term in search_terms)
        
        # Execute search
        search_results = self.search_engine.search(
            query=entity_query,
            top_k=top_k,
            include_entities=True
        )
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            type="entity_focused",
            query=entity_name,
            expanded_query=entity_query,
            results=search_results,
            entities_found=[resolved_entity],
            confidence=resolved_entity.confidence,
            processing_time=processing_time
        )
    
    def retrieve_similar_passages(self, 
                                reference_verse_id: str,
                                top_k: int = 10) -> RetrievalResult:
        """
        Find passages similar to a reference verse
        """
        
        import time
        start_time = time.time()
        
        # Get reference verse content
        try:
            import sqlite3
            conn = sqlite3.connect(self.search_engine.db_path)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT translation_en, translation_hi, shloka_iast
                FROM verses
                WHERE verse_id = ?
            """, (reference_verse_id,))
            
            reference = cursor.fetchone()
            conn.close()
            
            if not reference:
                raise ValueError(f"Reference verse {reference_verse_id} not found")
            
            # Create search query from reference content
            reference_text = f"{reference['translation_en']} {reference['shloka_iast']}"
            
        except Exception as e:
            logger.error(f"Error retrieving reference verse: {e}")
            return RetrievalResult(
                type="error",
                query=reference_verse_id,
                expanded_query=None,
                results=[],
                entities_found=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Search for similar content
        search_results = self.search_engine.search(
            query=reference_text,
            top_k=top_k + 5,  # Get extra to filter out reference verse
            include_entities=False
        )
        
        # Remove the reference verse from results
        filtered_results = [
            result for result in search_results 
            if result.verse_id != reference_verse_id
        ][:top_k]
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            type="similarity_search",
            query=reference_verse_id,
            expanded_query=reference_text[:100] + "...",
            results=filtered_results,
            entities_found=[],
            confidence=0.85,
            processing_time=processing_time
        )

    def retrieve_relationship(self,
                              entity_name: str,
                              rel_type: str | None = None,
                              top_k: int = 10) -> RetrievalResult:
        """
        Retrieve content focused on an entity's relationships using RelationshipGraph.
        """
        import time
        start_time = time.time()

        # Resolve entity via resolver first
        resolved = self.entity_resolver.resolve(entity_name)
        if not resolved:
            logger.warning(f"Could not resolve entity for relationship query: {entity_name}")
            # Fallback: perform standard retrieval
            std = self.retrieve(entity_name, top_k=top_k, include_context=True, entity_boost=True)
            return RetrievalResult(
                type="relationship_fallback",
                query=entity_name,
                expanded_query=std.expanded_query,
                results=std.results,
                entities_found=std.entities_found,
                confidence=std.confidence,
                processing_time=time.time() - start_time
            )

        # Build relationship-aware search terms
        enhancement_terms: list[str] = [resolved.canonical_name]

        # Use relationship graph if available
        if self.relationship_graph:
            try:
                # Family tree labels -> names
                fam = self.relationship_graph.find_family_members(resolved.canonical_name)
                for label, names in fam.items():
                    enhancement_terms.extend(names[:3])  # keep it short

                # General relationships (no type filter)
                rels = self.relationship_graph.get_relationships(resolved.canonical_name)
                enhancement_terms.extend([r.target_entity for r in rels[:5]])
            except Exception as e:
                logger.warning(f"Relationship graph access failed: {e}")

        # Add top aliases/context
        for alias in resolved.aliases[:3]:
            enhancement_terms.append(alias)
        for ctx in resolved.context[:2]:
            enhancement_terms.append(ctx)

        # Add relation synonyms to improve recall (wife/husband/spouse etc.)
        if rel_type:
            rel = (rel_type or '').lower().strip()
            syn_map = {
                'wife': ['wife', 'spouse', 'married', 'wedded'],
                'husband': ['husband', 'spouse', 'married', 'wedded'],
                'spouse': ['husband', 'wife', 'spouse', 'married'],
                'spouses': ['husband', 'wife', 'spouse', 'married'],
                'father': ['father', 'parent'],
                'mother': ['mother', 'parent'],
                'parent': ['parent', 'father', 'mother'],
                'parents': ['parents', 'father', 'mother'],
                'son': ['son', 'child'],
                'daughter': ['daughter', 'child'],
                'child': ['child', 'son', 'daughter'],
                'children': ['children', 'son', 'daughter'],
                'brother': ['brother', 'sibling'],
                'sister': ['sister', 'sibling'],
                'sibling': ['sibling'],
                'siblings': ['siblings', 'sibling'],
                'grandfather': ['grandfather', 'grandparent'],
                'grandmother': ['grandmother', 'grandparent'],
                'grandparent': ['grandparent', 'grandfather', 'grandmother'],
                'grandparents': ['grandparents', 'grandfather', 'grandmother'],
                'grandson': ['grandson', 'grandchild'],
                'granddaughter': ['granddaughter', 'grandchild'],
                'grandchild': ['grandchild', 'grandson', 'granddaughter'],
                'grandchildren': ['grandchildren', 'grandson', 'granddaughter'],
            }
            for token in syn_map.get(rel, [])[:3]:  # keep short
                enhancement_terms.append(token)

        # Optional: add a couple of salient description terms (feature-flag)
        try:
            if getattr(settings, 'enable_description_expansion', False):
                meta = self._get_entity_meta(resolved.canonical_name)
                if meta and isinstance(meta.get('description'), str):
                    desc_terms = self._extract_description_terms(meta['description'], limit=2)
                    enhancement_terms.extend(desc_terms)
        except Exception:
            pass

        # Create expanded OR query with quoting and sanitization
        def _sanitize_phrase_for_fts(term: str) -> str:
            try:
                t = str(term or "")
                t = t.replace('"', ' ')
                t = re.sub(r"[-_]+", " ", t)
                t = re.sub(r"\s+", " ", t).strip()
                return t
            except Exception:
                return term
        quoted_terms = [f'"{_sanitize_phrase_for_fts(t)}"' for t in enhancement_terms if isinstance(t, str) and t]
        expanded_query = ' OR '.join(quoted_terms) if quoted_terms else resolved.canonical_name

        # Execute hybrid search with entity enhancement
        results = self.search_engine.search(
            query=expanded_query,
            top_k=top_k,
            include_entities=True
        )

        return RetrievalResult(
            type="relationship",
            query=entity_name,
            expanded_query=expanded_query,
            results=results,
            entities_found=[resolved],
            confidence=max(self.confidence_threshold, resolved.confidence),
            processing_time=time.time() - start_time
        )
    
    # --------------------
    # Helpers
    # --------------------
    def _get_entity_meta(self, canonical_name: str) -> Optional[Dict[str, Any]]:
        """Fetch raw entity metadata dict from resolver given a canonical name."""
        try:
            norm = self.entity_resolver._normalize_text(canonical_name)  # best available
            return self.entity_resolver.canonical_lookup.get(norm)
        except Exception:
            return None

    def _extract_description_terms(self, text: str, limit: int = 2) -> List[str]:
        """Extract a few salient capitalized tokens from a description string."""
        if not text:
            return []
        # Simple heuristic: pick capitalized words of length >=4 that are not common stopwords
        candidates = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
        stop = {"Rama", "Sita", "Hanuman", "Ravana", "King", "Prince", "Son", "Daughter", "Mother", "Father"}
        out: List[str] = []
        for w in candidates:
            if w in stop:
                continue
            if w not in out:
                out.append(w)
            if len(out) >= limit:
                break
        return out
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval system statistics
        """
        
        entity_stats = {
            "total_characters": len(self.entity_resolver.canonical_lookup),
            "confidence_threshold": self.confidence_threshold
        }
        
        search_stats = self.search_engine.get_search_stats()
        
        return {
            "entity_resolver": entity_stats,
            "search_engine": search_stats,
            "supported_retrieval_types": [
                "standard_search",
                "entity_enhanced", 
                "entity_focused",
                "similarity_search"
            ]
        }
