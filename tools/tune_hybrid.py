"""
Hybrid retrieval tuning harness
- Grid search BM25 column weights, hybrid bm25/vector weights, and MMR params
- Evaluate on a small golden set with token-based scoring
- Persist best config to settings.tuned_weights_path (JSON)

Run:
    python tools/tune_hybrid.py --golden data/config/golden_queries.json \
        --save data/config/tuned_weights.json --max-k 10

Notes:
- Uses existing RAG pipeline (QueryRouter -> RAGRetriever -> HybridSearchEngine)
- No LLM calls are made
- Safe to run offline
"""
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from app.settings import settings
from rag.query_router import QueryRouter
from rag.retriever import RAGRetriever
from core.entities.resolver import EntityResolver
from core.search.hybrid_engine import HybridSearchEngine

@dataclass
class GoldenCase:
    query: str
    type: str = "general"
    must_have: List[str] = None
    should_have: List[str] = None
    avoid: List[str] = None
    max_results: int = 10

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GoldenCase":
        return GoldenCase(
            query=d.get("query", ""),
            type=d.get("type", "general"),
            must_have=d.get("must_have", []) or [],
            should_have=d.get("should_have", []) or [],
            avoid=d.get("avoid", []) or [],
            max_results=int(d.get("max_results", 10) or 10),
        )


def load_golden(path: str) -> List[GoldenCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[GoldenCase] = []
    for row in (data or []):
        out.append(GoldenCase.from_dict(row))
    return out


def build_engine() -> Tuple[HybridSearchEngine, EntityResolver, QueryRouter]:
    # Build (non-LLM) components
    entity_resolver = EntityResolver(
        characters_path=settings.characters_path,
        locations_path=settings.locations_path,
        confidence_threshold=settings.entity_confidence_threshold,
    )
    engine = HybridSearchEngine(
        db_path=settings.database_path,
        vector_store=getattr(getattr(__import__("app.main").main.app, "state", object()), "vector_store", None),
        entity_resolver=entity_resolver,
        bm25_weight=settings.bm25_weight,
        vector_weight=settings.vector_weight,
        fts_table=settings.fts_table,
    )
    router = QueryRouter(settings.database_path)
    return engine, entity_resolver, router


def run_query(engine: HybridSearchEngine, router: QueryRouter, q: GoldenCase) -> List[Any]:
    # Let router classify but ultimately we call engine directly (no LLM)
    _ = router.analyze_query(q.query)
    docs = engine.search(q.query, top_k=q.max_results, include_entities=True)
    return docs


def score_case(docs: List[Any], case: GoldenCase) -> float:
    """Token-based scoring on English text and titles.
    Score = +2 per must-have present, +1 per should-have, -2 per avoid term.
    Terms are matched case-insensitively as substrings.
    """
    en_blob = " ".join([(d.translation_en or "") for d in docs])
    titles_blob = " ".join([" ".join(d.sarga_titles or []) for d in docs])
    blob = f"{en_blob} {titles_blob}".lower()
    score = 0.0
    for t in (case.must_have or []):
        if t and t.lower() in blob:
            score += 2.0
    for t in (case.should_have or []):
        if t and t.lower() in blob:
            score += 1.0
    for t in (case.avoid or []):
        if t and t.lower() in blob:
            score -= 2.0
    # Light bonus based on top-1 fused score quality
    try:
        if docs and float(getattr(docs[0], "final_score", 0.0)) > 0.25:
            score += 0.5
    except Exception:
        pass
    return score


def grid(values: List) -> List:
    return list(values)


def search_hyperparams(goldens: List[GoldenCase], max_k: int, save_path: str) -> Dict[str, Any]:
    # Candidate grids
    bm25_vec_weights = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5)]
    mmr_lambdas = [0.6, 0.7, 0.8]
    mmr_ks = [10, 15, 20]
    # If Hindi is removed from FTS, 5 columns [0.0,0.0,0.8,1.2,0.5]; else 6 columns with Hindi weight 0.0
    default_cols = [0.0, 0.0, 0.8, 1.2, 0.5] if getattr(settings, "fts_remove_hi", False) else [0.0, 0.0, 0.8, 1.2, 0.0, 0.5]
    bm25_column_sets = [
        default_cols,
        (default_cols[:-1] + [0.6]) if len(default_cols) == 5 else (default_cols[:-1] + [0.6]),
        (default_cols[:-1] + [0.4]) if len(default_cols) == 5 else (default_cols[:-1] + [0.4]),
    ]

    best_score = -1e9
    best_cfg: Dict[str, Any] = {}

    # Save original settings to restore
    orig = {
        "bm25_weight": settings.bm25_weight,
        "vector_weight": settings.vector_weight,
        "bm25_column_weights": settings.bm25_column_weights,
        "enable_mmr_diversity": settings.enable_mmr_diversity,
        "mmr_lambda": settings.mmr_lambda,
        "mmr_k": settings.mmr_k,
    }

    try:
        total = 0
        for bm_w, vc_w in bm25_vec_weights:
            for lam in mmr_lambdas:
                for mk in mmr_ks:
                    for cols in bm25_column_sets:
                        total += 1
        print(f"Evaluating {total} combinations on {len(goldens)} golden cases...")

        tried = 0
        for bm_w, vc_w in bm25_vec_weights:
            # Apply weights
            settings.bm25_weight = bm_w
            settings.vector_weight = vc_w
            for lam in mmr_lambdas:
                settings.enable_mmr_diversity = True
                settings.mmr_lambda = lam
                for mk in mmr_ks:
                    settings.mmr_k = mk
                    for cols in bm25_column_sets:
                        settings.bm25_column_weights = cols
                        tried += 1
                        # Build engine with current settings
                        engine, er, router = build_engine()
                        total_score = 0.0
                        for case in goldens:
                            docs = run_query(engine, router, case)
                            sc = score_case(docs, case)
                            total_score += sc
                        if total_score > best_score:
                            best_score = total_score
                            best_cfg = {
                                "bm25_weight": bm_w,
                                "vector_weight": vc_w,
                                "mmr_lambda": lam,
                                "mmr_k": mk,
                                "bm25_column_weights": cols,
                                "score": total_score,
                            }
                        if tried % 10 == 0:
                            print(f"Tried {tried} / {total}, current best: {best_cfg}")
        print(f"Best config: {best_cfg}")
        # Persist to JSON
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(best_cfg, f, ensure_ascii=False, indent=2)
            print(f"Saved tuned weights to {save_path}")
        return best_cfg
    finally:
        # Restore original settings
        settings.bm25_weight = orig["bm25_weight"]
        settings.vector_weight = orig["vector_weight"]
        settings.bm25_column_weights = orig["bm25_column_weights"]
        settings.enable_mmr_diversity = orig["enable_mmr_diversity"]
        settings.mmr_lambda = orig["mmr_lambda"]
        settings.mmr_k = orig["mmr_k"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True, help="Path to golden queries JSON")
    parser.add_argument("--save", default=None, help="Where to save tuned weights JSON")
    parser.add_argument("--max-k", type=int, default=10, help="Top K to evaluate")
    args = parser.parse_args()

    goldens = load_golden(args.golden)
    # Cap max_results per case
    for g in goldens:
        g.max_results = min(g.max_results or 10, args.max_k)

    t0 = time.time()
    cfg = search_hyperparams(goldens, args.max_k, args.save or settings.tuned_weights_path)
    dt = time.time() - t0
    print(f"Tuning completed in {dt:.2f}s. Best: {cfg}")


if __name__ == "__main__":
    main()
