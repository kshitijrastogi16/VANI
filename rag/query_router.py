"""
VANI V3 Query Router - Intelligent Query Classification and Direct Routing
Enforcing: Count queries, citation lookup, and type-aware processing
"""

import re
import sqlite3
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from app.settings import settings
from rag.citations import denormalize_verse_id

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    type: str  # "count", "citation", "relationship", "narrative", "analysis", "general"
    confidence: float
    target: Optional[str] = None
    parameters: Optional[Dict] = None
    requires_llm: bool = True


class QueryRouter:
    """
    Intelligent Query Classification and Direct Database Routing
    Handles count queries, citations, and determines processing strategy
    
    Uses singleton pattern and lazy loading for the embedding model.
    """
    _instance = None
    _model_initialized = False
    
    def __new__(cls, db_path: str = None):
        if cls._instance is None:
            if db_path is None:
                raise ValueError("db_path must be provided for the first initialization")
            cls._instance = super(QueryRouter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str):
        if self._initialized:
            return
            
        self.db_path = db_path
        self.embedding_model = None
        self.settings = settings
        self._validate_database()
        self._initialized = True
    
    def _ensure_embedding_model(self):
        """Router embeddings disabled; stubbed to no-op."""
        return None

    def _analyze_narrative_or_analysis_query(self, q: str) -> Optional[QueryAnalysis]:
        """Detect narrative/analysis style queries.
        - Narrative: who/why/where/when/how did/what happened
        - Analysis: compare/difference/versus X and Y
        """
        try:
            qs = (q or "").strip()
            ql = qs.lower()
            # Compare pattern → analysis
            cmp_patterns = [
                r"\bcompare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)",
                r"\bhow\s+is\s+(.+?)\s+different\s+from\s+(.+)",
                r"\bdifference\s+between\s+(.+?)\s+and\s+(.+)",
            ]
            for p in cmp_patterns:
                m = re.search(p, ql)
                if m:
                    a = (m.group(1) or '').strip()
                    b = (m.group(2) or '').strip()
                    return QueryAnalysis(
                        type="analysis",
                        confidence=0.9,
                        target="compare",
                        parameters={"a": a, "b": b},
                        requires_llm=True,
                    )
            # Who is/was X? → narrative
            m = re.match(r"^who\s+(?:is|was)\s+(.+?)\s*[\?\.!]*$", ql)
            if m:
                subj = m.group(1).strip()
                return QueryAnalysis(
                    type="narrative",
                    confidence=0.9,
                    target="who_is",
                    parameters={"subject": subj},
                    requires_llm=True,
                )
            # Why / Where / When / How did … happen → narrative
            if re.match(r"^(why\b|where\b|when\b)", ql) or re.match(r"^how\s+did\b", ql) or re.search(r"\bwhat\s+happened\b", ql):
                return QueryAnalysis(
                    type="narrative",
                    confidence=0.85,
                    target="narrative",
                    requires_llm=True,
                )
        except Exception:
            pass
        return None
    
    def get_query_embedding(self, query: str) -> Optional[None]:
        """Router embeddings disabled; returns None."""
        return None
    
    def _validate_database(self):
        """Ensure database is accessible"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT COUNT(*) FROM verses LIMIT 1")
            conn.close()
            logger.info("✅ Query router database validation successful")
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            raise
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query to determine optimal processing strategy
        """
        
        if not query or not query.strip():
            return QueryAnalysis(type="error", confidence=0.0)
        
        query_lower = query.lower().strip()
        
        # 1. Count queries (direct database, no LLM needed)
        count_analysis = self._analyze_count_query(query_lower)
        if count_analysis:
            return count_analysis
        
        # 2. Citation queries (direct lookup, no LLM needed)  
        citation_analysis = self._analyze_citation_query(query.strip())
        if citation_analysis:
            return citation_analysis
        
        # 3. Relationship queries (may need entity resolution)
        relationship_analysis = self._analyze_relationship_query(query_lower)
        if relationship_analysis:
            return relationship_analysis
        # 4. Narrative/analysis patterns (who/why/where/when/compare)
        na = self._analyze_narrative_or_analysis_query(query)
        if na:
            return na
        
        # 5. Default to general RAG processing
        return QueryAnalysis(
            type="general", 
            confidence=1.0,
            requires_llm=True
        )
    
    def _analyze_count_query(self, query: str) -> Optional[QueryAnalysis]:
        """
        Detect and classify count queries for direct database processing
        """
        
        # 1) Specific counts FIRST to avoid being shadowed by generic patterns
        # Verses in a specific sarga within a specific kanda
        sarga_verse_match = re.search(
            r"\bhow many\s+(?:shlokas|verses?)\b(?:\s+(?:are\s+there\s+in|are\s+in|in))?\s+(?:sarga|chapter|canto)\s+(\d+)\s+(?:of|in)\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)*)\s+kanda\b",
            query,
        )
        if sarga_verse_match:
            sarga_num = int(sarga_verse_match.group(1))
            kanda_name = sarga_verse_match.group(2).strip()
            return QueryAnalysis(
                type="count",
                confidence=0.98,
                target="verses_in_sarga",
                parameters={"kanda": kanda_name, "sarga": sarga_num},
                requires_llm=False,
            )

        # Sarga count in specific kanda (handle "are there in" / "are in" / "in")
        sarga_match = re.search(
            r"\bhow many\s+(?:sargas?|chapters?|cantos?)\b(?:\s+(?:are\s+there\s+in|are\s+in|in))?\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)*)\s+kanda\b",
            query,
        )
        if sarga_match:
            kanda_name = sarga_match.group(1).strip()
            return QueryAnalysis(
                type="count",
                confidence=0.95,
                target="sargas",
                parameters={"kanda": kanda_name},
                requires_llm=False,
            )

        # Verse count in specific kanda (support verse/shloka synonyms)
        kanda_verse_match = re.search(
            r"\bhow many\s+(?:shlokas|verses?)\b(?:\s+(?:are\s+there\s+in|are\s+in|in))?\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)*)\s+kanda\b",
            query,
        )
        if kanda_verse_match:
            kanda_name = kanda_verse_match.group(1).strip()
            return QueryAnalysis(
                type="count",
                confidence=0.95,
                target="verses_in_kanda",
                parameters={"kanda": kanda_name},
                requires_llm=False,
            )

        # Occurrences of a name (count verses mentioning a name)
        occ_patterns = [
            r"\bhow many times\s+([a-zA-Z'\-\.\s]+?)\s+(?:appears?|is\s+mentioned)\b",
            r"\bhow many verses\s+(?:mention|contain)\s+([a-zA-Z'\-\.\s]+?)\b",
            r"\boccurrences\s+of\s+([a-zA-Z'\-\.\s]+?)\b",
        ]
        for pattern in occ_patterns:
            m = re.search(pattern, query)
            if m:
                name = m.group(1).strip().strip("'\" ")
                if name:
                    return QueryAnalysis(
                        type="count",
                        confidence=0.95,
                        target="occurrences",
                        parameters={"name": name},
                        requires_llm=False,
                    )

        # Characters total count
        char_patterns = [
            r"\bhow many\s+(?:characters|people|persons)\b(?:.*\bramayana\b)?",
            r"\btotal\s+(?:characters|people|persons)\b",
            r"\bnumber\s+of\s+(?:characters|people|persons)\b",
        ]
        for pattern in char_patterns:
            if re.search(pattern, query):
                return QueryAnalysis(
                    type="count",
                    confidence=0.95,
                    target="characters_total",
                    requires_llm=False,
                )

        # Locations total count
        loc_patterns = [
            r"\bhow many\s+(?:locations|places)\b(?:.*\bramayana\b)?",
            r"\btotal\s+(?:locations|places)\b",
            r"\bnumber\s+of\s+(?:locations|places)\b",
        ]
        for pattern in loc_patterns:
            if re.search(pattern, query):
                return QueryAnalysis(
                    type="count",
                    confidence=0.95,
                    target="locations_total",
                    requires_llm=False,
                )

        # Per-kanda breakdown requests (tabular-ish)
        breakdown_patterns = [
            r"\blist\s+(?:sarga|chapter|canto)s?\s+counts?\s+for\s+all\s+kandas\b",
            r"\b(sarga|chapter|canto)s?\s+per\s+kanda\b",
            r"\bverses\s+per\s+kanda\b",
            r"\bper\-?kanda\s+(?:breakdown|summary)\b",
        ]
        for pattern in breakdown_patterns:
            if re.search(pattern, query):
                return QueryAnalysis(
                    type="count",
                    confidence=0.95,
                    target="kanda_breakdown",
                    requires_llm=False,
                )

        # 2) Kanda count patterns
        kanda_patterns = [
            r"\bhow many kandas?\b",
            r"\btotal kandas?\b",
            r"\bnumber of kandas?\b",
            r"\bkandas in ramayana\b",
        ]
        for pattern in kanda_patterns:
            if re.search(pattern, query):
                return QueryAnalysis(
                    type="count",
                    confidence=0.95,
                    target="kandas",
                    requires_llm=False,
                )

        # 3) Total sarga count across Ramayana
        sarga_total_patterns = [
            r"\bhow many\s+(?:sargas?|chapters?|cantos?)\b(?:\s+(?:are\s+there\s+in|are\s+in|in))?\s+(?:the\s+)?ramayana\b",
            r"\btotal\s+(?:sargas?|chapters?|cantos?)\b",
            r"\bnumber\s+of\s+(?:sargas?|chapters?|cantos?)\b",
        ]
        for pattern in sarga_total_patterns:
            if re.search(pattern, query) and ("kanda" not in query):
                return QueryAnalysis(
                    type="count",
                    confidence=0.95,
                    target="sargas_total",
                    requires_llm=False,
                )

        # 4) Total verse count patterns (skip if query mentions a kanda/sarga)
        if ("kanda" not in query) and ("sarga" not in query):
            verse_patterns = [
                r"\bhow many verses?\b",
                r"\btotal verses?\b",
                r"\bnumber of verses?\b",
                r"\bverses in ramayana\b",
                r"\bhow many shlokas?\b",
                r"\btotal shlokas?\b",
            ]
            for pattern in verse_patterns:
                if re.search(pattern, query):
                    return QueryAnalysis(
                        type="count",
                        confidence=0.95,
                        target="verses",
                        requires_llm=False,
                    )
        
        return None
    
    def _analyze_citation_query(self, query: str) -> Optional[QueryAnalysis]:
        """
        Detect direct verse citation requests
        """
        
        # Direct verse ID formats
        # 1) Hyphen format: 1-1-1 or 2-15-23-30
        verse_id_match = re.match(r"^(\d+)-(\d+)-(\d+)(?:-(\d+))?$", query.strip())
        if verse_id_match:
            return QueryAnalysis(
                type="citation",
                confidence=1.0,
                target="verse_lookup",
                parameters={"verse_id": query.strip()},
                requires_llm=False
            )
        # 2) Dot format: 1.1.1 or 6.129.5-9
        # Allow whitespace around dots, e.g., "1 . 1 . 1"
        verse_id_match_dot = re.match(r"^(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)(?:\s*-\s*(\d+))?$", query.strip())
        if verse_id_match_dot:
            k, s, v = verse_id_match_dot.group(1), verse_id_match_dot.group(2), verse_id_match_dot.group(3)
            v2 = verse_id_match_dot.group(4)
            verse_id_hyphen = f"{k}-{s}-{v}" + (f"-{v2}" if v2 else "")
            return QueryAnalysis(
                type="citation",
                confidence=1.0,
                target="verse_lookup",
                parameters={"verse_id": verse_id_hyphen},
                requires_llm=False
            )
        
        # Citation request patterns  
        citation_patterns = [
            # hyphen variants with optional range
            r"verse\s+(\d+)-(\d+)-(\d+)(?:-(\d+))?",
            r"shloka\s+(\d+)-(\d+)-(\d+)(?:-(\d+))?",
            r"show me verse\s+(\d+)-(\d+)-(\d+)(?:-(\d+))?",
            r"find verse\s+(\d+)-(\d+)-(\d+)(?:-(\d+))?",
            # dot variants with optional range
            r"verse\s+(\d+)\.(\d+)\.(\d+)(?:-(\d+))?",
            r"shloka\s+(\d+)\.(\d+)\.(\d+)(?:-(\d+))?",
            r"show me verse\s+(\d+)\.(\d+)\.(\d+)(?:-(\d+))?",
            r"find verse\s+(\d+)\.(\d+)\.(\d+)(?:-(\d+))?"
        ]
        
        for pattern in citation_patterns:
            match = re.search(pattern, query.lower())
            if match:
                # Normalize to hyphen and include optional range if present
                base = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                rng = f"-{match.group(4)}" if match.lastindex and match.group(4) else ""
                verse_id = base + rng
                return QueryAnalysis(
                    type="citation",
                    confidence=0.90,
                    target="verse_lookup",
                    parameters={"verse_id": verse_id},
                    requires_llm=False
                )
        
        return None
    
    def _analyze_relationship_query(self, query: str) -> Optional[QueryAnalysis]:
        """
        Detect relationship and family tree queries
        """
        
        q = query.strip()
        # Canonical relation labels (singular + plural when applicable)
        rel_labels = {
            'father', 'mother', 'parent', 'parents',
            'son', 'daughter', 'child', 'children',
            'brother', 'sister', 'sibling', 'siblings',
            'husband', 'wife', 'spouse', 'spouses',
            'grandfather', 'grandmother', 'grandparent', 'grandparents',
            'grandson', 'granddaughter', 'grandchild', 'grandchildren'
        }

        # Spouse/marriage direct questions
        # S1: "Who married <subject>?"
        m = re.search(r"\bwho\s+married\s+(.+?)[\?\.!]*$", q)
        if m:
            subject = m.group(1).strip().strip("?.!, ")
            if subject:
                return QueryAnalysis(
                    type="relationship",
                    confidence=0.92,
                    target="entity_relationship",
                    parameters={"subject": subject, "relation": "spouse"},
                    requires_llm=False,
                )

        # S2: "Whom did <subject> marry?"
        m = re.search(r"\bwhom\s+did\s+(.+?)\s+(?:marry|wed|wedded)\b", q)
        if m:
            subject = m.group(1).strip().strip("?.!, ")
            if subject:
                return QueryAnalysis(
                    type="relationship",
                    confidence=0.92,
                    target="entity_relationship",
                    parameters={"subject": subject, "relation": "spouse"},
                    requires_llm=False,
                )

        # S3: "Who did <subject> marry?"
        m = re.search(r"\bwho\s+did\s+(.+?)\s+(?:marry|wed|wedded)\b", q)
        if m:
            subject = m.group(1).strip().strip("?.!, ")
            if subject:
                return QueryAnalysis(
                    type="relationship",
                    confidence=0.92,
                    target="entity_relationship",
                    parameters={"subject": subject, "relation": "spouse"},
                    requires_llm=False,
                )

        # S4: "Who is/was <subject> married to?"
        m = re.search(r"\bwho\s+(?:is|was)\s+(.+?)\s+married\s+to\b", q)
        if m:
            subject = m.group(1).strip().strip("?.!, ")
            if subject:
                return QueryAnalysis(
                    type="relationship",
                    confidence=0.92,
                    target="entity_relationship",
                    parameters={"subject": subject, "relation": "spouse"},
                    requires_llm=False,
                )

        # S5: "Who is/was the <relation> of <subject>?" (e.g., "Who is the wife of Shatrughna?")
        m = re.search(r"\bwho\s+(?:is|was|are|were)\s+(?:the\s+)?([a-zA-Z\-]+)\s+of\s+(.+?)[\?\.!]*$", q)
        if m:
            rel = (m.group(1) or '').strip().lower()
            rel = re.sub(r'^(?:the\s+)?', '', rel)
            subject = (m.group(2) or '').strip().strip("?.!, ")
            if rel in rel_labels and subject:
                return QueryAnalysis(
                    type="relationship",
                    confidence=0.93,
                    target="entity_relationship",
                    parameters={"subject": subject, "relation": rel},
                    requires_llm=False,
                )

        def _norm_rel(rel: str) -> str:
            r = re.sub(r"\s+", " ", (rel or '').strip().lower())
            # collapse common spaced forms
            r = r.replace('grand father', 'grandfather').replace('grand mother', 'grandmother')
            r = r.replace('grand son', 'grandson').replace('grand daughter', 'granddaughter')
            # normalize plurals to defined set
            return r

        # Pattern A: "who is <subject>'s <relation>" or "who are <subject>'s <relation>"
        m = re.search(r"\bwho\s+(?:is|are)\s+(.+?)(?:'s|’s)\s+([a-zA-Z\-\s]+)\b", q)
        if m:
            subject = m.group(1).strip().strip("?.!, ")
            rel = _norm_rel(m.group(2))
            # Drop leading articles like 'the', 'a', 'an' before relation label
            rel = re.sub(r'^(?:the|a|an)\s+', '', rel)
            if rel in rel_labels:
                return QueryAnalysis(
                    type="relationship",
                    confidence=0.9,
                    target="entity_relationship",
                    parameters={"subject": subject, "relation": rel},
                    requires_llm=False
                )

        # Pattern B: "<relation> of <subject>"
        m = re.search(r"\b([a-zA-Z\-\s]+)\b\s+of\s+(.+)$", q)
        if m:
            rel = _norm_rel(m.group(1))
            # Drop leading articles like 'the', 'a', 'an' before relation label
            rel = re.sub(r'^(?:the|a|an)\s+', '', rel)
            subject = m.group(2).strip().strip("?.!, ")
            if rel in rel_labels:
                return QueryAnalysis(
                    type="relationship",
                    confidence=0.9,
                    target="entity_relationship",
                    parameters={"subject": subject, "relation": rel},
                    requires_llm=False
                )

        # Generic family query fallbacks
        relationship_patterns = [
            r"family of .+",
            r"children of .+",
            r"parents of .+",
            r"siblings of .+"
        ]
        for pattern in relationship_patterns:
            if re.search(pattern, q):
                return QueryAnalysis(
                    type="relationship",
                    confidence=0.85,
                    target="entity_relationship",
                    requires_llm=True
                )
        return None
    
    def handle_count_query(self, analysis: QueryAnalysis) -> Dict:
        """
        Execute count queries directly against database
        """
        
        if analysis.type != "count":
            raise ValueError("Invalid query type for count handler")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            if analysis.target == "kandas":
                # Prefer canonical count from kandas table
                cursor = conn.execute("SELECT COUNT(*) as count FROM kandas")
                count = cursor.fetchone()["count"]
                
                return {
                    "type": "count",
                    "answer": f"The Ramayana is divided into {count} kandas (books).",
                    "count": count,
                    "target": "kandas",
                    "sources": []  # No sources needed for count queries
                }
            
            elif analysis.target == "verses":
                # True verse (shloka) count: sum canonical per-kanda counts
                cursor = conn.execute("SELECT COALESCE(SUM(verse_count_in_kanda), 0) as count FROM kandas")
                count = cursor.fetchone()["count"] or 0
                
                return {
                    "type": "count", 
                    "answer": f"The complete Ramayana contains {count:,} verses across all kandas.",
                    "count": count,
                    "target": "verses",
                    "sources": []
                }

            elif analysis.target == "sargas_total":
                # Total sargas across all kandas (canonical)
                cursor = conn.execute("SELECT COALESCE(SUM(sarga_count), 0) as count FROM kandas")
                count = cursor.fetchone()["count"] or 0

                return {
                    "type": "count",
                    "answer": f"The complete Ramayana contains {count} sargas (chapters) across all kandas.",
                    "count": count,
                    "target": "sargas_total",
                    "sources": [],
                }
            
            elif analysis.target == "sargas" and analysis.parameters:
                kanda_name = analysis.parameters["kanda"]
                knum, _canon = self._normalize_kanda_identifier(kanda_name)
                
                # Use canonical sarga_count from kandas table (handles merged multi-verse entries)
                if knum:
                    cursor = conn.execute(
                        "SELECT sarga_count as count, kanda_name FROM kandas WHERE kanda_num = ?",
                        (knum,),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT sarga_count as count, kanda_name
                        FROM kandas
                        WHERE LOWER(kanda_name) LIKE LOWER(?)
                        """,
                        (f"%{kanda_name}%",),
                    )
                
                result = cursor.fetchone()
                if result:
                    count = result["count"]
                    actual_kanda_name = result["kanda_name"]
                    
                    return {
                        "type": "count",
                        "answer": f"The {actual_kanda_name} contains {count} sargas (chapters).",
                        "count": count,
                        "target": "sargas",
                        "kanda": actual_kanda_name,
                        "sources": []
                    }

            elif analysis.target == "characters_total":
                # Count characters from JSON file
                try:
                    path = getattr(settings, "characters_path", None)
                    cnt = 0
                    if path and os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                cnt = len(data)
                            elif isinstance(data, dict):
                                cnt = len(data.get("characters", data))
                    return {
                        "type": "count",
                        "answer": f"There are {cnt} characters in the dataset.",
                        "count": cnt,
                        "target": "characters_total",
                        "sources": []
                    }
                except Exception:
                    return {
                        "type": "count",
                        "answer": "Unable to compute character count at this time.",
                        "count": 0,
                        "target": "characters_total",
                        "sources": []
                    }

            elif analysis.target == "locations_total":
                # Count locations from JSON file
                try:
                    path = getattr(settings, "locations_path", None)
                    cnt = 0
                    if path and os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                cnt = len(data)
                            elif isinstance(data, dict):
                                cnt = len(data.get("locations", data))
                    return {
                        "type": "count",
                        "answer": f"There are {cnt} locations in the dataset.",
                        "count": cnt,
                        "target": "locations_total",
                        "sources": []
                    }
                except Exception:
                    return {
                        "type": "count",
                        "answer": "Unable to compute location count at this time.",
                        "count": 0,
                        "target": "locations_total",
                        "sources": []
                    }

            elif analysis.target == "occurrences" and analysis.parameters:
                # Count number of verses that mention the given name (approximate occurrences)
                name = analysis.parameters.get("name", "").strip()
                if not name:
                    return {
                        "type": "count",
                        "answer": "Please specify a valid name to search for.",
                        "count": 0,
                        "target": "occurrences",
                        "sources": []
                    }

                term = f"%{name.lower()}%"
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) as cnt
                    FROM verses
                    WHERE LOWER(translation_en) LIKE ?
                       OR LOWER(translation_hi) LIKE ?
                       OR LOWER(shloka_iast) LIKE ?
                       OR LOWER(shloka_sanskrit) LIKE ?
                    """,
                    (term, term, term, term),
                )
                cnt = cursor.fetchone()["cnt"]
                return {
                    "type": "count",
                    "answer": f"{cnt} verses mention '{name}'.",
                    "count": cnt,
                    "target": "occurrences",
                    "parameters": {"name": name},
                    "sources": []
                }

            elif analysis.target == "kanda_breakdown":
                # Provide per-kanda sarga and verse counts
                cursor = conn.execute(
                    "SELECT kanda_num, kanda_name, sarga_count, verse_count_in_kanda FROM kandas ORDER BY kanda_num"
                )
                rows = cursor.fetchall()
                breakdown = [
                    {
                        "kanda_num": r["kanda_num"],
                        "kanda_name": r["kanda_name"],
                        "sarga_count": r["sarga_count"],
                        "verse_count_in_kanda": r["verse_count_in_kanda"],
                    }
                    for r in rows
                ]
                # Build a concise textual answer
                parts = [
                    f"{b['kanda_name']}: {b['sarga_count']} sargas, {b['verse_count_in_kanda']:,} verses"
                    for b in breakdown
                ]
                answer = "Per-kanda breakdown — " + "; ".join(parts)
                return {
                    "type": "count",
                    "answer": answer,
                    "count": len(breakdown),
                    "target": "kanda_breakdown",
                    "breakdown": breakdown,
                    "sources": []
                }
            
            elif analysis.target == "verses_in_kanda" and analysis.parameters:
                kanda_name = analysis.parameters["kanda"]
                knum, _canon = self._normalize_kanda_identifier(kanda_name)
                
                # True verse (shloka) count for a kanda from canonical table
                if knum:
                    cursor = conn.execute(
                        "SELECT verse_count_in_kanda as count, kanda_name FROM kandas WHERE kanda_num = ?",
                        (knum,),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT verse_count_in_kanda as count, kanda_name
                        FROM kandas
                        WHERE LOWER(kanda_name) LIKE LOWER(?)
                        """,
                        (f"%{kanda_name}%",),
                    )
                
                result = cursor.fetchone()
                if result:
                    count = result["count"]
                    actual_kanda_name = result["kanda_name"]
                    
                    return {
                        "type": "count",
                        "answer": f"The {actual_kanda_name} contains {count:,} verses.",
                        "count": count,
                        "target": "verses_in_kanda", 
                        "kanda": actual_kanda_name,
                        "sources": []
                    }
                else:
                    return {
                        "type": "count",
                        "answer": f"No kanda found matching '{kanda_name}'. Please check the spelling.",
                        "count": 0,
                        "target": "verses_in_kanda",
                        "sources": []
                    }

            elif analysis.target == "verses_in_sarga" and analysis.parameters:
                kanda_name = analysis.parameters.get("kanda")
                sarga_num = int(analysis.parameters.get("sarga", 0))

                if not sarga_num:
                    return {
                        "type": "count",
                        "answer": "Please specify a valid sarga number.",
                        "count": 0,
                        "target": "verses_in_sarga",
                        "sources": []
                    }

                # Resolve via join to get canonical kanda name and sarga verse count
                knum, _canon = self._normalize_kanda_identifier(kanda_name)
                if knum:
                    cursor = conn.execute(
                        """
                        SELECT s.verse_count_in_sarga as count, k.kanda_name, s.sarga_num
                        FROM sargas s
                        JOIN kandas k ON s.kanda_num = k.kanda_num
                        WHERE s.sarga_num = ? AND s.kanda_num = ?
                        LIMIT 1
                        """,
                        (sarga_num, knum),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT s.verse_count_in_sarga as count, k.kanda_name, s.sarga_num
                        FROM sargas s
                        JOIN kandas k ON s.kanda_num = k.kanda_num
                        WHERE s.sarga_num = ? AND LOWER(k.kanda_name) LIKE LOWER(?)
                        LIMIT 1
                        """,
                        (sarga_num, f"%{kanda_name}%"),
                    )

                result = cursor.fetchone()
                if result:
                    count = result["count"]
                    actual_kanda_name = result["kanda_name"]
                    actual_sarga_num = result["sarga_num"]

                    return {
                        "type": "count",
                        "answer": f"Sarga {actual_sarga_num} of {actual_kanda_name} contains {count} verses.",
                        "count": count,
                        "target": "verses_in_sarga",
                        "kanda": actual_kanda_name,
                        "sarga": actual_sarga_num,
                        "sources": []
                    }
                else:
                    return {
                        "type": "count",
                        "answer": f"No match found for Sarga {sarga_num} in '{kanda_name}'. Please check the spelling.",
                        "count": 0,
                        "target": "verses_in_sarga",
                        "sources": []
                    }
            
            else:
                raise ValueError(f"Unknown count target: {analysis.target}")
        
        finally:
            conn.close()

    # --------------------
    # Helpers
    # --------------------
    def _normalize_kanda_identifier(self, name_or_phrase: str) -> Tuple[Optional[int], Optional[str]]:
        """Normalize kanda names to kanda_num and canonical name.
        Accepts inputs like 'Baala', 'Bala Kanda', 'kanda 1', etc.
        Returns (kanda_num, canonical_name) or (None, None) if not recognized.
        """
        try:
            text = (name_or_phrase or "").lower().strip()
            # Extract numeric form like 'kanda 2'
            m = re.search(r"\bkanda\s*(\d)\b", text)
            if m:
                num = int(m.group(1))
                return num, None
            # Strip trailing 'kanda'
            text = re.sub(r"\bkanda\b", "", text).strip()
            mapping = {
                "baala": (1, "Baala Kanda"), "bala": (1, "Baala Kanda"),
                "ayodhya": (2, "Ayodhya Kanda"), "ayodhaya": (2, "Ayodhya Kanda"),
                "aranya": (3, "Aranya Kanda"),
                "kishkindha": (4, "Kishkindha Kanda"), "kiskindha": (4, "Kishkindha Kanda"),
                "sundara": (5, "Sundara Kanda"), "sundar": (5, "Sundara Kanda"),
                "yuddha": (6, "Yuddha Kanda"), "yudha": (6, "Yuddha Kanda"), "lanka": (6, "Yuddha Kanda"),
                "uttara": (7, "Uttara Kanda"), "uttar": (7, "Uttara Kanda"),
            }
            # Try exact key
            if text in mapping:
                return mapping[text]
            # Try to match first token
            first = text.split()[0] if text else ""
            if first in mapping:
                return mapping[first]
        except Exception:
            pass
        return None, None
    
    def handle_citation_query(self, analysis: QueryAnalysis) -> Dict:
        """
        Execute direct verse lookup queries
        """
        
        if analysis.type != "citation":
            raise ValueError("Invalid query type for citation handler")
        
        verse_id = analysis.parameters["verse_id"]
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.execute("""
                SELECT verse_id, kanda_num, kanda_name, sarga_num, verse_index,
                       translation_en, translation_hi, shloka_sanskrit, shloka_iast,
                       sarga_titles_json
                FROM verses
                WHERE verse_id = ?
            """, (verse_id,))
            
            result = cursor.fetchone()

            # Fallback: if exact verse_id not found, resolve possible multi-verse range like K-S-start-end
            # Only attempt when input is a single-verse id K-S-V
            if not result:
                import re as _re
                m = _re.match(r"^(\d+)-(\d+)-(\d+)$", verse_id)
                if m:
                    kanda_num = int(m.group(1))
                    sarga_num = int(m.group(2))
                    target_verse = int(m.group(3))
                    # Search rows in same sarga to find a range that includes target_verse
                    cursor = conn.execute("""
                        SELECT verse_id, kanda_num, kanda_name, sarga_num, verse_index,
                               translation_en, translation_hi, shloka_sanskrit, shloka_iast,
                               sarga_titles_json
                        FROM verses
                        WHERE kanda_num = ? AND sarga_num = ?
                    """, (kanda_num, sarga_num))
                    rows = cursor.fetchall()
                    for r in rows:
                        try:
                            parts = str(r["verse_id"]).split('-')
                            if len(parts) == 4:
                                start_v = int(parts[2])
                                end_v = int(parts[3])
                                if start_v <= target_verse <= end_v:
                                    result = r
                                    break
                        except Exception:
                            continue
            
            if result:
                # Parse sarga titles
                try:
                    sarga_titles = json.loads(result["sarga_titles_json"])
                except (json.JSONDecodeError, TypeError):
                    sarga_titles = []
                
                # Format comprehensive citation
                _res = dict(result)
                source_info = {
                    "verse_id": _res.get("verse_id"),
                    "kanda_num": _res.get("kanda_num"),
                    "kanda_name": _res.get("kanda_name"),
                    "sarga_num": _res.get("sarga_num"),
                    "verse_index": _res.get("verse_index"),
                    "translation_en": _res.get("translation_en"),
                    "translation_hi": _res.get("translation_hi"),
                    "shloka_sanskrit": _res.get("shloka_sanskrit"),
                    "shloka_iast": _res.get("shloka_iast"),
                    "sarga_titles": sarga_titles
                }
                
                # Create concise header answer; detailed verse content is available in sources
                display_citation = denormalize_verse_id(_res['verse_id'])
                answer = f"{display_citation} ({_res['kanda_name']}, Sarga {_res['sarga_num']})\n\n"
                # Include full verse content in the answer for direct lookup
                if _res.get('shloka_sanskrit'):
                    answer += f"Sanskrit: {_res['shloka_sanskrit']}\n\n"
                if _res.get('shloka_iast'):
                    answer += f"IAST: {_res['shloka_iast']}\n\n"
                if _res.get('translation_en'):
                    answer += f"English: {_res['translation_en']}\n\n"
                if _res.get('translation_hi'):
                    answer += f"Hindi: {_res['translation_hi']}"
                
                if sarga_titles:
                    answer += f"\n\nContext: {sarga_titles[0]}"
                
                # Build surrounding context window (±2 verses by index) when a single verse was requested
                context_sources = []
                try:
                    import re as _re
                    window = max(0, getattr(settings, "citation_context_window", 2))
                    anchor_index = None
                    m = _re.match(r"^(\d+)-(\d+)-(\d+)$", verse_id)
                    if m:
                        anchor_index = int(m.group(3))
                    else:
                        # For range requests, anchor around the stored verse_index
                        anchor_index = int(_res["verse_index"]) if _res.get("verse_index") is not None else None
                    if anchor_index is not None and window > 0:
                        # Build dynamic ±window neighbor indices, excluding the anchor itself
                        neighbor_set = set()
                        for d in range(1, window + 1):
                            neighbor_set.add(anchor_index - d)
                            neighbor_set.add(anchor_index + d)
                        neighbor_indices = sorted(neighbor_set)
                    else:
                        neighbor_indices = []

                    # Scope neighbors by kanda_num + sarga_num
                    neighbor_rows = []
                    if neighbor_indices:
                        placeholders = ','.join(['?'] * len(neighbor_indices))
                        q = f"""
                            SELECT verse_id, kanda_num, kanda_name, sarga_num, verse_index,
                                   translation_en, translation_hi, shloka_sanskrit, shloka_iast,
                                   sarga_titles_json
                            FROM verses
                            WHERE kanda_num = ? AND sarga_num = ? AND verse_index IN ({placeholders})
                            ORDER BY verse_index
                        """
                        _res2 = dict(result)
                        k_num_val = _res2.get("kanda_num", locals().get("kanda_num"))
                        s_num_val = _res2.get("sarga_num", locals().get("sarga_num"))
                        params = [k_num_val, s_num_val, *neighbor_indices]
                        cur2 = conn.execute(q, params)
                        neighbor_rows = cur2.fetchall()
                        for nr in neighbor_rows:
                            if nr["verse_id"] == result["verse_id"]:
                                continue  # skip duplicate main verse
                            try:
                                n_titles = json.loads(nr["sarga_titles_json"]) if nr["sarga_titles_json"] else []
                            except (json.JSONDecodeError, TypeError):
                                n_titles = []
                            _nr = dict(nr)
                            context_sources.append({
                                "verse_id": _nr.get("verse_id"),
                                "kanda_num": _nr.get("kanda_num"),
                                "kanda_name": _nr.get("kanda_name"),
                                "sarga_num": _nr.get("sarga_num"),
                                "verse_index": _nr.get("verse_index"),
                                "translation_en": _nr.get("translation_en"),
                                "translation_hi": _nr.get("translation_hi"),
                                "shloka_sanskrit": _nr.get("shloka_sanskrit"),
                                "shloka_iast": _nr.get("shloka_iast"),
                                "sarga_titles": n_titles
                            })
                except Exception:
                    context_sources = []
                
                sources_combined = [source_info] + context_sources
                
                return {
                    "type": "citation",
                    "answer": answer,
                    "verse_id": verse_id,
                    "sources": sources_combined
                }
            else:
                return {
                    "type": "citation",
                    "answer": f"Verse {verse_id} not found. Please verify the verse ID format (Kanda-Sarga-Verse).",
                    "verse_id": verse_id,
                    "sources": []
                }
        
        finally:
            conn.close()
    
    def get_router_stats(self) -> Dict:
        """
        Get query routing statistics
        """
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Get database statistics
            stats = {}
            
            # Canonical counts from metadata tables
            cursor = conn.execute("SELECT COUNT(*) as kandas FROM kandas")
            stats["total_kandas"] = cursor.fetchone()["kandas"]

            cursor = conn.execute("SELECT COALESCE(SUM(sarga_count), 0) as sargas FROM kandas")
            stats["total_sargas"] = cursor.fetchone()["sargas"]

            cursor = conn.execute("SELECT COALESCE(SUM(verse_count_in_kanda), 0) as verses FROM kandas")
            stats["total_verses"] = cursor.fetchone()["verses"]

            # Also report total entries (rows) in verses table for diagnostics
            cursor = conn.execute("SELECT COUNT(*) as entries FROM verses")
            stats["total_entries"] = cursor.fetchone()["entries"]
            
            return {
                "database_stats": stats,
                "supported_query_types": ["count", "citation", "relationship", "general"],
                "direct_processing": [
                    "count", "citation",
                    # Count subtypes
                    "kandas", "sargas_total", "verses", "sargas", "verses_in_kanda", "verses_in_sarga",
                    "characters_total", "locations_total", "occurrences", "kanda_breakdown",
                ]
            }
        
        finally:
            conn.close()
