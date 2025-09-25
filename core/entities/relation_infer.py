"""
Relation inference from verses (pattern-based) with citations.
Non-invasive: uses BM25 over existing FTS; no re-embedding required.
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional, Tuple

from app.settings import settings
from core.search.text_search import search_bm25
from rag.citations import denormalize_verse_id
from core.entities.resolver import EntityResolver

logger = logging.getLogger(__name__)


class RelationInferencer:
    """Infer missing relations on-the-fly using regex patterns over retrieved verses.
    - Searches the corpus with small set of crafted queries for a (subject, relation)
    - Extracts candidate objects via regex
    - Resolves candidates to canonical entities; returns citations and confidences
    """

    def __init__(self, db_path: str, entity_resolver: EntityResolver, fts_table: str = None):
        self.db_path = db_path
        self.resolver = entity_resolver
        self.fts_table = fts_table or settings.fts_table
        # Precompile generic patterns keyed by relation
        self._compiled_cache: Dict[str, List[re.Pattern]] = {}

    # -----------------------------
    # Public API
    # -----------------------------
    def infer(
        self,
        subject: str,
        relation: str,
        top_k: int = None,
        max_queries: int = None,
        per_query_k: int = None,
    ) -> Dict:
        """Infer relation objects for the given subject and relation.
        Returns a dict with candidate objects and supporting citations.
        """
        if not subject or not relation or not self.resolver:
            return {"candidates": []}

        # Defaults from settings
        top_k = top_k or getattr(settings, "relation_infer_top_k", 40)
        max_queries = max_queries or getattr(settings, "relation_infer_max_queries", 6)
        per_query_k = per_query_k or getattr(settings, "relation_infer_per_query_k", 6)

        # Resolve subject to canonical
        subj_resolved = self.resolver.resolve(subject)
        if not subj_resolved:
            return {"candidates": []}
        subj = subj_resolved.canonical_name

        # Generate small set of search queries
        queries = self._generate_queries(subj, relation)[:max_queries]
        if not queries:
            return {"candidates": []}

        # Search and extract
        candidates: Dict[str, Dict] = {}
        for q in queries:
            try:
                docs = search_bm25(self.db_path, self.fts_table, q, limit=per_query_k) or []
            except Exception as e:
                logger.warning(f"RelationInferencer BM25 error for '{q}': {e}")
                continue

            for d in docs:
                # Use only English translation to avoid cross-language noise
                text = str(d.get("translation_en", ""))
                matches = self._extract_candidates(text, subj, relation)
                for raw_name, pat_weight in matches:
                    # Resolve raw candidate name
                    resolved = self.resolver.resolve(raw_name)
                    if not resolved:
                        continue
                    conf = float(resolved.confidence or 0.0)
                    if conf < max(0.7, getattr(settings, "entity_confidence_threshold", 0.7)):
                        continue

                    # Aggregate by canonical
                    canon = resolved.canonical_name
                    cur = candidates.get(canon)
                    # Combine evidence score from pattern weight + bm25 similarity + resolver conf
                    bm25_sim = float(d.get("score", 0.0) or 0.0)  # already normalized to [0,1]
                    score = pat_weight + 0.25 * bm25_sim + 0.25 * conf

                    if not cur or score > cur["score"]:
                        candidates[canon] = {
                            "object_canonical": canon,
                            "raw_name": raw_name,
                            "score": score,
                            "verse_id": d.get("verse_id"),
                            "citation": denormalize_verse_id(str(d.get("verse_id", ""))) if d.get("verse_id") else None,
                            "kanda_name": d.get("kanda_name"),
                            "sarga_num": d.get("sarga_num"),
                            "verse_index": d.get("verse_index"),
                            "translation_en": d.get("translation_en"),
                            "shloka_iast": d.get("shloka_iast"),
                        }

        # Sort candidates by score
        ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return {
            "subject": subj,
            "relation": relation,
            "candidates": ranked,
        }

    # -----------------------------
    # Internals
    # -----------------------------
    def _generate_queries(self, subject: str, relation: str) -> List[str]:
        r = relation.lower().strip()
        subj_q = f'"{subject}"'
        queries: List[str] = []
        # Spouse
        if r in {"wife", "husband", "spouse", "spouses"}:
            for token in ["wife", "husband", "spouse", "married", "married to", "wedded"]:
                queries.append(f"{subj_q} {token}")
                if token not in {"married", "wedded"}:
                    queries.append(f"{token} of {subj_q}")
        # Parents/children
        if r in {"father", "mother", "parent", "parents"}:
            for token in ["father", "mother", "parent"]:
                queries.append(f"{subj_q} {token}")
                queries.append(f"{token} of {subj_q}")
        if r in {"son", "daughter", "child", "children"}:
            for token in ["son", "daughter", "child"]:
                queries.append(f"{subj_q} {token}")
                queries.append(f"{token} of {subj_q}")
        # Siblings: prefer neutral phrasing by default to avoid gender errors
        if r in {"sibling", "siblings"}:
            for token in ["sibling"]:
                queries.append(f"{subj_q} {token}")
                queries.append(f"{token} of {subj_q}")
        elif r == "brother":
            for token in ["brother", "sibling"]:
                queries.append(f"{subj_q} {token}")
                queries.append(f"{token} of {subj_q}")
        elif r == "sister":
            for token in ["sister", "sibling"]:
                queries.append(f"{subj_q} {token}")
                queries.append(f"{token} of {subj_q}")
        # Grand relations
        if r in {"grandfather", "grandmother", "grandparent", "grandparents", "grandson", "granddaughter", "grandchild", "grandchildren"}:
            for token in ["grandfather", "grandmother", "grandson", "granddaughter", "grandchild", "grandparent"]:
                queries.append(f"{subj_q} {token}")
                queries.append(f"{token} of {subj_q}")
        # Dedup while preserving order
        seen = set()
        uniq: List[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                uniq.append(q)
        return uniq

    def _extract_candidates(self, text: str, subject: str, relation: str) -> List[Tuple[str, float]]:
        """Return list of (raw_name, pattern_weight) found in the text.
        pattern_weight encodes the confidence of the pattern (0-1).
        """
        if not text:
            return []
        r = relation.lower().strip()
        subj = subject
        patterns = self._get_patterns(subj, r)
        # Filter tokens that are not likely to be proper names
        stop_tokens = {
            "King", "Queen", "Lord", "Prince", "Princess", "Rakshasa", "Rakshasas",
            "Raksasa", "Demon", "Demons", "Army", "Minister", "Mother", "Father",
            "Brother", "Sister", "Son", "Daughter", "City", "Lanka", "Ayodhya"
        }
        out: List[Tuple[str, float]] = []
        for weight, pat in patterns:
            for m in pat.finditer(text):
                obj = (m.group("obj") or "").strip(" ,.;:!?\n\t\r")
                # Proximity check: ensure match span is reasonably short (subject near relation)
                span_len = m.end() - m.start()
                if span_len > 180:
                    continue
                if len(obj) >= 2 and obj.lower() != subj.lower() and obj not in stop_tokens:
                    out.append((obj, weight))
        return out

    def _get_patterns(self, subject: str, rel: str) -> List[Tuple[float, re.Pattern]]:
        key = f"{rel}::{subject.lower()}"
        cached = self._compiled_cache.get(key)
        if cached is not None:
            return cached  # type: ignore

        # Escape subject for regex; allow possessive forms
        subj_re = re.escape(subject)
        subj_possessive = rf"{subj_re}(?:'s|’s)?"
        patterns: List[Tuple[float, str]] = []

        # Spouse patterns
        if rel in {"wife", "husband", "spouse", "spouses"}:
            patterns.extend([
                # subject's wife/husband/spouse OBJ
                (0.82, rf"(?i)\b{subj_possessive}\s+(?:the\s+)?(?:wife|husband|spouse)s?\s+(?P<obj>[A-Z][A-Za-z\-']+)"),
                # OBJ, the wife/husband/spouse of subject
                (0.80, rf"(?i)\b(?P<obj>[A-Z][A-Za-z\-']+)[,]?\s*(?:the\s+)?(?:wife|husband|spouse)s?\s+of\s+{subj_re}\b"),
                # subject is the wife/husband/spouse of OBJ
                (0.78, rf"(?i)\b{subj_re}\s+(?:is|was)\s+(?:the\s+)?(?:wife|husband|spouse)s?\s+of\s+(?P<obj>[A-Z][A-Za-z\-']+)"),
                # subject married OBJ / married to OBJ / wedded OBJ
                (0.75, rf"(?i)\b{subj_re}\s+(?:married\s+to|married|wedded)\s+(?P<obj>[A-Z][A-Za-z\-']+)")
            ])
        # Parents
        if rel in {"father", "mother", "parent", "parents"}:
            patterns.extend([
                # subject's father/mother/parent OBJ
                (0.77, rf"(?i)\b{subj_possessive}\s+(?:the\s+)?(?:father|mother|parent)s?\s+(?P<obj>[A-Z][A-Za-z\-']+)"),
                # OBJ, the father/mother/parent of subject
                (0.75, rf"(?i)\b(?P<obj>[A-Z][A-Za-z\-']+)[,]?\s*(?:the\s+)?(?:father|mother|parent)s?\s+of\s+{subj_re}\b"),
                # subject is son/daughter/child of OBJ (implies parents)
                (0.74, rf"(?i)\b{subj_re}\s+(?:is|was)\s+(?:a?n?\s+)?(?:son|daughter|child)\s+of\s+(?P<obj>[A-Z][A-Za-z\-']+)")
            ])
        # Children
        if rel in {"son", "daughter", "child", "children"}:
            patterns.extend([
                # subject's son/daughter/child OBJ
                (0.74, rf"(?i)\b{subj_possessive}\s+(?:the\s+)?(?:son|daughter|child)\s+(?P<obj>[A-Z][A-Za-z\-']+)"),
                # OBJ, the son/daughter/child of subject
                (0.72, rf"(?i)\b(?P<obj>[A-Z][A-Za-z\-']+)[,]?\s*(?:the\s+)?(?:son|daughter|child)\s+of\s+{subj_re}\b"),
                # OBJ is son/daughter/child of subject (reverse phrasing)
                (0.70, rf"(?i)\b(?P<obj>[A-Z][A-Za-z\-']+)\s+(?:is|was)\s+(?:a?n?\s+)?(?:son|daughter|child)\s+of\s+{subj_re}\b"),
            ])
        # Siblings
        if rel in {"brother", "sister", "sibling", "siblings"}:
            patterns.extend([
                # subject's (elder/younger) brother/sister OBJ
                (0.74, rf"(?i)\b{subj_possessive}\s+(?:the\s+)?(?:elder\s+|younger\s+)?(?:brother|sister|sibling)s?\s+(?P<obj>[A-Z][A-Za-z\-']+)"),
                # OBJ, the (elder/younger) brother/sister of subject
                (0.72, rf"(?i)\b(?P<obj>[A-Z][A-Za-z\-']+)[,]?\s*(?:the\s+)?(?:elder\s+|younger\s+)?(?:brother|sister|sibling)s?\s+of\s+{subj_re}\b"),
                # subject is (the) (elder/younger) brother/sister of OBJ
                (0.70, rf"(?i)\b{subj_re}\s+(?:is|was)\s+(?:the\s+)?(?:elder\s+|younger\s+)?(?:brother|sister|sibling)s?\s+of\s+(?P<obj>[A-Z][A-Za-z\-']+)"),
            ])
        # Grands
        if rel in {"grandfather", "grandmother", "grandparent", "grandparents"}:
            patterns.extend([
                (0.70, rf"(?i)\b{subj_possessive}\s+(?:the\s+)?(?:grandfather|grandmother|grandparent)\s+(?P<obj>[A-Z][A-Za-z\-']+)"),
                (0.68, rf"(?i)\b(?P<obj>[A-Z][A-Za-z\-']+)[,]?\s*(?:the\s+)?(?:grandfather|grandmother|grandparent)\s+of\s+{subj_re}\b"),
            ])
        if rel in {"grandson", "granddaughter", "grandchild", "grandchildren"}:
            patterns.extend([
                (0.68, rf"(?i)\b{subj_possessive}\s+(?:the\s+)?(?:grandson|granddaughter|grandchild)\s+(?P<obj>[A-Z][A-Za-z\-']+)"),
                (0.66, rf"(?i)\b(?P<obj>[A-Z][A-Za-z\-']+)[,]?\s*(?:the\s+)?(?:grandson|granddaughter|grandchild)\s+of\s+{subj_re}\b"),
            ])

        compiled: List[Tuple[float, re.Pattern]] = []
        for w, p in patterns:
            try:
                compiled.append((w, re.compile(p)))
            except re.error:
                continue
        self._compiled_cache[key] = compiled  # cache
        return compiled
