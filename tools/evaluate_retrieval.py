"""Small deterministic BM25 evaluation against curated, corpus-verified verse IDs.

Run from the repository root with: python tools/evaluate_retrieval.py --k 5
This starter benchmark is intentionally small; expand it with manually verified cases.
"""

import argparse
import json
from pathlib import Path
import sys

# Running a script by path places tools/, not the repository root, on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.settings import settings
from core.search.text_search import search_bm25
from rag.query_planner import normalize_retrieval_query


def reciprocal_rank(expected_ids, retrieved_ids):
    expected = set(expected_ids)
    for rank, verse_id in enumerate(retrieved_ids, start=1):
        if verse_id in expected:
            return 1.0 / rank
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval cutoff")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    case_path = Path(__file__).resolve().parents[1] / "tests" / "data" / "rag_eval_cases.json"
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    recall_values = []
    reciprocal_ranks = []

    for case in cases:
        query = normalize_retrieval_query(case["question"])
        results = search_bm25(settings.database_path, settings.fts_table, query, limit=args.k)
        retrieved = [str(row.get("verse_id")) for row in results if row.get("verse_id")]
        relevant = set(case["expected_relevant_verse_ids"])
        recall = len(relevant & set(retrieved)) / len(relevant) if relevant else 0.0
        mrr = reciprocal_rank(relevant, retrieved)
        recall_values.append(recall)
        reciprocal_ranks.append(mrr)
        print(f"[{case['id']}] query={query!r}")
        print(f"  relevant_in_top_{args.k}={sorted(relevant & set(retrieved))}")
        print(f"  Recall@{args.k}={recall:.3f}  ReciprocalRank={mrr:.3f}")

    count = len(cases)
    print(f"\nCases: {count}")
    print(f"Mean Recall@{args.k}: {sum(recall_values) / count:.3f}")
    print(f"MRR@{args.k}: {sum(reciprocal_ranks) / count:.3f}")
    print("Note: this is a small lexical-retrieval smoke benchmark, not an industry-wide quality claim.")


if __name__ == "__main__":
    main()
