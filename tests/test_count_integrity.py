import sqlite3
from pathlib import Path
import pytest

DB_PATH = Path("data/processed/ramayana.db")

@pytest.fixture
def db():
    if not DB_PATH.exists():
        pytest.fail(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def scalar(conn, query):
    return conn.execute(query).fetchone()[0]

def test_searchable_record_count(db):
    assert scalar(db, "SELECT COUNT(*) FROM verses") == 18136

def test_kanda_verse_totals(db):
    assert scalar(
        db, "SELECT SUM(verse_count_in_kanda) FROM kandas"
    ) == 19951

def test_sarga_verse_totals(db):
    assert scalar(
        db, "SELECT SUM(verse_count_in_sarga) FROM sargas"
    ) == 19951

def test_kanda_and_sarga_summaries_agree(db):
    kanda_total = scalar(
        db, "SELECT SUM(verse_count_in_kanda) FROM kandas"
    )
    sarga_total = scalar(
        db, "SELECT SUM(verse_count_in_sarga) FROM sargas"
    )
    assert kanda_total == sarga_total

def test_kanda_and_sarga_counts(db):
    assert scalar(db, "SELECT COUNT(*) FROM kandas") == 6
    assert scalar(db, "SELECT COUNT(*) FROM sargas") == 537

def test_each_sarga_has_a_positive_verse_count(db):
    invalid = scalar(
        db,
        """
        SELECT COUNT(*)
        FROM sargas
        WHERE verse_count_in_sarga IS NULL
           OR verse_count_in_sarga <= 0
        """,
    )
    assert invalid == 0

def test_expanded_verse_ids_match_each_sarga_metadata(db):
    """Grouped verse IDs must expand to the exact per-Sarga counts in SQLite."""
    from collections import Counter

    expanded_ids = set()
    actual_by_sarga = Counter()

    for row in db.execute("SELECT verse_id FROM verses"):
        verse_id = str(row[0])
        parts = verse_id.split("-")
        assert len(parts) in (3, 4), f"Unrecognized verse ID: {verse_id}"

        try:
            kanda_num, sarga_num = int(parts[0]), int(parts[1])
            if len(parts) == 3:
                verse_indices = [int(parts[2])]
            else:
                start, end = int(parts[2]), int(parts[3])
                assert end >= start, f"Invalid grouped verse range: {verse_id}"
                verse_indices = range(start, end + 1)
        except ValueError as exc:
            pytest.fail(f"Non-numeric verse ID component in {verse_id}: {exc}")

        actual_by_sarga[(kanda_num, sarga_num)] += len(verse_indices)
        for verse_index in verse_indices:
            individual_id = f"{kanda_num}-{sarga_num}-{verse_index}"
            assert individual_id not in expanded_ids, (
                f"Duplicate individual verse after expanding grouped IDs: {individual_id}"
            )
            expanded_ids.add(individual_id)

    expected_by_sarga = {
        (int(row[0]), int(row[1])): int(row[2])
        for row in db.execute(
            "SELECT kanda_num, sarga_num, verse_count_in_sarga FROM sargas"
        )
    }
    assert actual_by_sarga == expected_by_sarga
    assert len(expanded_ids) == 19951
