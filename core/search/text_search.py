"""
VANI V3 Text Search - FIXED VERSION
Enforcing: SQL injection prevention, proper parameterization, input validation
"""

import sqlite3
import logging
import re
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
from app.settings import settings

logger = logging.getLogger(__name__)

def _bm25_weights_clause() -> str:
    """Return a comma-separated weight clause for bm25() based on settings.
    If settings.bm25_column_weights is provided, use it as-is.
    Otherwise fall back to defaults, adapting when Hindi column is removed.
    """
    try:
        remove_hi = bool(getattr(settings, 'fts_remove_hi', False))
        expected = 5 if remove_hi else 6
        default_5 = [0.0, 0.0, 0.8, 1.2, 0.5]
        default_6 = [0.0, 0.0, 0.8, 1.2, 0.0, 0.5]
        weights = getattr(settings, 'bm25_column_weights', None)
        seq = None
        if isinstance(weights, (list, tuple)):
            # Trim/pad to expected length
            seq = list(weights)[:expected]
            if len(seq) < expected:
                pad = (default_5 if expected == 5 else default_6)[len(seq):expected]
                seq.extend(pad)
        else:
            seq = default_5 if expected == 5 else default_6
        return ", ".join(str(float(w)) for w in seq)
    except Exception:
        return "0.0, 0.0, 0.8, 1.2, 0.0, 0.5"

def search_bm25(db_path: str, 
                fts_table: str, 
                query: str, 
                limit: int = 10,
                min_score: Optional[float] = None) -> List[Dict]:
    """
    Execute BM25 search using SQLite FTS5 with SQL injection prevention
    
    Args:
        db_path: Path to SQLite database
        fts_table: Name of FTS5 table (validated)
        query: Search query string (sanitized)
        limit: Maximum number of results
        min_score: Optional minimum BM25 score threshold
    
    Returns:
        List of search result dictionaries
    """
    
    if not query or not query.strip():
        logger.warning("Empty search query provided")
        return []
    
    # Validate table name (prevent injection)
    if not _validate_table_name(fts_table):
        logger.error(f"Invalid FTS table name: {fts_table}")
        return []
    
    # Clean and prepare query for FTS5
    cleaned_query = _prepare_fts_query(query.strip())
    if not cleaned_query:
        logger.warning(f"Query could not be cleaned: {query}")
        return []
    
    # Validate limit parameter
    limit = max(1, min(limit, 1000))  # Reasonable bounds
    
    logger.debug(f"Searching '{cleaned_query}' in {fts_table} (limit={limit})")
    
    try:
        with _get_db_connection(db_path) as conn:
            weights = _bm25_weights_clause()
            # Build parameterized search SQL (prevents injection)
            search_sql = f"""
            SELECT 
                v.id,
                v.verse_id,
                v.kanda_num,
                v.sarga_num,
                v.verse_index,
                v.kanda_name,
                v.shloka_sanskrit,
                v.shloka_iast,
                v.translation_en,
                v.translation_hi,
                v.sarga_titles_json,
                bm25({fts_table}, {weights}) as bm25_score
            FROM {fts_table}
            JOIN verses v ON v.id = {fts_table}.rowid
            WHERE {fts_table} MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """
            
            # Execute with parameters (safe from injection)
            cursor = conn.execute(search_sql, (cleaned_query, limit))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                # Use raw BM25 score from FTS5 (lower = better). Normalize to similarity [0,1].
                try:
                    bm25_raw = float(row['bm25_score'])
                except Exception:
                    bm25_raw = 0.0
                # Simple normalization: higher is better, bounded in (0,1]
                similarity = 1.0 / (1.0 + max(0.0, bm25_raw))
                
                # Apply minimum score filter if specified
                if min_score is not None and similarity < float(min_score):
                    continue
                
                # Convert to dictionary and add metadata
                result = dict(row)
                result['search_type'] = 'bm25'
                # Keep raw bm25_score from SELECT; expose normalized similarity in 'score'
                result['score'] = similarity
                
                # Parse sarga titles safely
                try:
                    import json
                    result['sarga_titles'] = json.loads(row['sarga_titles_json'] or "[]")
                except (json.JSONDecodeError, TypeError):
                    result['sarga_titles'] = []
                
                results.append(result)
            
            logger.debug(f"BM25 search returned {len(results)} results")
            return results

    except sqlite3.Error as e:
        logger.error(f"SQLite error in BM25 search: {e}")
        # Fallback: attempt simplified query once
        try:
            simple = _simple_fts_query(query)
            if not simple or simple == cleaned_query:
                return []
            with _get_db_connection(db_path) as conn:
                weights = _bm25_weights_clause()
                search_sql = f"""
                SELECT 
                    v.id,
                    v.verse_id,
                    v.kanda_num,
                    v.sarga_num,
                    v.verse_index,
                    v.kanda_name,
                    v.shloka_sanskrit,
                    v.shloka_iast,
                    v.translation_en,
                    v.translation_hi,
                    v.sarga_titles_json,
                    bm25({fts_table}, {weights}) as bm25_score
                FROM {fts_table}
                JOIN verses v ON v.id = {fts_table}.rowid
                WHERE {fts_table} MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """
                cursor = conn.execute(search_sql, (simple, limit))
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    try:
                        bm25_raw = float(row['bm25_score'])
                    except Exception:
                        bm25_raw = 0.0
                    similarity = 1.0 / (1.0 + max(0.0, bm25_raw))
                    result = dict(row)
                    result['search_type'] = 'bm25'
                    result['score'] = similarity
                    try:
                        import json
                        result['sarga_titles'] = json.loads(row['sarga_titles_json'] or "[]")
                    except (json.JSONDecodeError, TypeError):
                        result['sarga_titles'] = []
                    results.append(result)
                return results
        except Exception as _:
            return []
    except Exception as e:
        logger.error(f"Unexpected error in BM25 search: {e}")
        return []

# -----------------------------
# Fielded and filtered BM25 helpers (non-breaking additions)
# -----------------------------

def _build_fielded_query(raw: str, field: str) -> str:
    """Build a conservative field-scoped FTS query from a raw string.
    Uses _simple_fts_query to avoid complex transformations and prefixes each token/phrase with field:.
    Example output: field:"rama sita" OR field:rama OR field:sita
    """
    base = _simple_fts_query(raw)
    if not base:
        return ""
    parts = [p.strip() for p in base.split(" OR ") if p and p.strip()]
    scoped: List[str] = []
    for p in parts:
        if p.startswith('"') and p.endswith('"') and len(p) > 2:
            scoped.append(f"{field}:{p}")
        else:
            # single token; ensure it is alnum
            tok = re.sub(r"[^A-Za-z0-9]", "", p)
            if len(tok) >= 2:
                scoped.append(f"{field}:{tok}")
    return " OR ".join(scoped)

def search_bm25_fielded(db_path: str,
                        fts_table: str,
                        query: str,
                        field: str,
                        limit: int = 10,
                        min_score: Optional[float] = None) -> List[Dict]:
    """Execute BM25 search restricted to a specific FTS field (e.g., translation_en).
    Safe and parameterized; field name must be a known FTS column.
    """
    if not query or not field:
        return []
    # Allow only known safe fields (EN + IAST focus)
    allowed_fields = {"translation_en", "shloka_iast"}
    if field not in allowed_fields:
        return []
    scoped = _build_fielded_query(query, field)
    if not scoped:
        return []
    # Reuse main search, but pass the fielded MATCH query
    try:
        with _get_db_connection(db_path) as conn:
            weights = _bm25_weights_clause()
            search_sql = f"""
            SELECT 
                v.id,
                v.verse_id,
                v.kanda_num,
                v.sarga_num,
                v.verse_index,
                v.kanda_name,
                v.shloka_sanskrit,
                v.shloka_iast,
                v.translation_en,
                v.translation_hi,
                v.sarga_titles_json,
                bm25({fts_table}, {weights}) as bm25_score
            FROM {fts_table}
            JOIN verses v ON v.id = {fts_table}.rowid
            WHERE {fts_table} MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """
            cursor = conn.execute(search_sql, (scoped, max(1, min(limit, 1000))))
            rows = cursor.fetchall()
            results: List[Dict] = []
            for row in rows:
                try:
                    bm25_raw = float(row['bm25_score'])
                except Exception:
                    bm25_raw = 0.0
                similarity = 1.0 / (1.0 + max(0.0, bm25_raw))
                if min_score is not None and similarity < float(min_score):
                    continue
                result = dict(row)
                result['search_type'] = 'bm25_fielded'
                result['score'] = similarity
                try:
                    import json
                    result['sarga_titles'] = json.loads(row['sarga_titles_json'] or "[]")
                except (json.JSONDecodeError, TypeError):
                    result['sarga_titles'] = []
                results.append(result)
            return results
    except Exception:
        return []

def search_bm25_with_filters(db_path: str,
                             fts_table: str,
                             query: str,
                             limit: int = 10,
                             min_score: Optional[float] = None,
                             kanda_name: Optional[str] = None) -> List[Dict]:
    """Execute BM25 with optional metadata filters (currently supports kanda_name).
    Safe and parameterized; uses same scoring normalization as search_bm25.
    """
    if not query:
        return []
    if not _validate_table_name(fts_table):
        logger.error(f"Invalid FTS table name: {fts_table}")
        return []
    cleaned_query = _prepare_fts_query(query.strip())
    if not cleaned_query:
        return []
    limit = max(1, min(limit, 1000))
    try:
        with _get_db_connection(db_path) as conn:
            weights = _bm25_weights_clause()
            sql = f"""
            SELECT 
                v.id,
                v.verse_id,
                v.kanda_num,
                v.sarga_num,
                v.verse_index,
                v.kanda_name,
                v.shloka_sanskrit,
                v.shloka_iast,
                v.translation_en,
                v.translation_hi,
                v.sarga_titles_json,
                bm25({fts_table}, {weights}) as bm25_score
            FROM {fts_table}
            JOIN verses v ON v.id = {fts_table}.rowid
            WHERE {fts_table} MATCH ?
            """
            params: List = [cleaned_query]
            if kanda_name:
                sql += " AND v.kanda_name = ?"
                params.append(kanda_name)
            sql += " ORDER BY bm25_score ASC LIMIT ?"
            params.append(limit)
            cursor = conn.execute(sql, tuple(params))
            rows = cursor.fetchall()
            out: List[Dict] = []
            for row in rows:
                try:
                    bm25_raw = float(row['bm25_score'])
                except Exception:
                    bm25_raw = 0.0
                similarity = 1.0 / (1.0 + max(0.0, bm25_raw))
                if min_score is not None and similarity < float(min_score):
                    continue
                d = dict(row)
                d['search_type'] = 'bm25_filtered'
                d['score'] = similarity
                try:
                    import json
                    d['sarga_titles'] = json.loads(row['sarga_titles_json'] or "[]")
                except (json.JSONDecodeError, TypeError):
                    d['sarga_titles'] = []
                out.append(d)
            return out
    except sqlite3.Error as e:
        logger.error(f"SQLite error in BM25 filtered search: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in BM25 filtered search: {e}")
        return []

def _prepare_fts_query(query: str) -> str:
    """
    Prepare query string for FTS5 search with injection prevention
    
    Handles:
    - Special character sanitization
    - FTS5 operator validation
    - Query structure validation
    - Length limits
    """
    if not query:
        return ""
    # Remove potentially dangerous characters for FTS5
    # Keep only alphanumeric, spaces, quotes, hyphens, apostrophes
    cleaned = re.sub(r"[^\w\s\-'\"()]", " ", query)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return ""
    # Remove apostrophes to avoid FTS5 parse errors in MATCH phrases
    # Example problematic phrase: "Hanuman's Father"
    cleaned = cleaned.replace("'", "")
    # Length limit for safety
    if len(cleaned) > 200:
        cleaned = cleaned[:200].rsplit(' ', 1)[0]  # Break at word boundary
    parts: List[str] = []
    # Extract and normalize quoted phrases (if quotes balanced)
    if cleaned.count('"') % 2 == 0 and '"' in cleaned:
        raw_phrases = re.findall(r'"([^"]+)"', cleaned)
        norm_phrases: List[str] = []
        for ph in raw_phrases:
            ph2 = re.sub(r"[-_]+", " ", ph)
            ph2 = re.sub(r"[()]+", " ", ph2)
            ph2 = re.sub(r"\s+", " ", ph2).strip()
            if ph2:
                norm_phrases.append(ph2)
        parts.extend([f'"{p}"' for p in norm_phrases])
        # Remove quoted segments from remainder
        cleaned = re.sub(r'"[^\"]+"', ' ', cleaned)
    # Sanitize remainder (unquoted segment)
    rem = cleaned.replace('"', '')
    rem = re.sub(r'[()]+', ' ', rem)
    rem = re.sub(r"\s+", " ", rem).strip()
    if rem:
        words = rem.split()
        # Remove FTS reserved keywords
        reserved = {"or", "and", "not", "near"}
        valid_words = [w for w in words if len(w) >= 2 and w.isalnum() and w.lower() not in reserved]
        if valid_words:
            phrase_part = f'"{" ".join(valid_words)}"'
            words_part = ' OR '.join(valid_words)
            parts.append(f'({phrase_part}) OR ({words_part})')
    # If we built any parts, join with OR
    if parts:
        return ' OR '.join(parts)
    # Fallback to minimal cleaned token
    return rem

def _validate_table_name(table_name: str) -> bool:
    """
    Validate FTS table name to prevent SQL injection
    Only allow alphanumeric characters, underscores, and known table names
    """
    
    if not table_name:
        return False
    
    # Allow only safe characters
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', table_name):
        return False
    
    # Whitelist known FTS tables
    allowed_tables = ['verses_fts', 'characters_fts', 'locations_fts']
    return table_name in allowed_tables

def _simple_fts_query(raw: str) -> str:
    """
    Build a conservative FTS query: quoted phrase of alnum tokens + OR-joined tokens.
    Avoids punctuation and reserved keywords.
    """
    if not raw:
        return ""
    # Strip quotes and parentheses, keep only words
    s = re.sub(r'"', ' ', raw)
    s = re.sub(r'[()]+', ' ', s)
    s = re.sub(r"\s+", " ", s).strip()
    words = [w for w in re.split(r"\s+|\bOR\b|\bAND\b|\bNOT\b", s, flags=re.IGNORECASE) if w]
    words = [re.sub(r"[^A-Za-z0-9]", "", w) for w in words]
    words = [w for w in words if len(w) >= 2]
    if not words:
        return ""
    # Remove reserved
    reserved = {"or", "and", "not", "near"}
    words = [w for w in words if w.lower() not in reserved]
    if not words:
        return ""
    phrase = '"' + ' '.join(words) + '"'
    return f"{phrase} OR {' OR '.join(words)}"

@contextmanager
def _get_db_connection(db_path: str):
    """Get database connection with proper error handling and cleanup"""
    
    conn = None
    try:
        conn = sqlite3.connect(
            db_path,
            timeout=30,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        
        # Set safe PRAGMA settings
        conn.execute("PRAGMA query_only = ON")  # Read-only mode
        
        yield conn
        
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def search_exact_verse(db_path: str, verse_id: str) -> Optional[Dict]:
    """
    Search for exact verse by verse_id with input validation
    
    Args:
        db_path: Path to SQLite database
        verse_id: Exact verse identifier (validated format)
    
    Returns:
        Verse dictionary or None if not found
    """
    
    if not verse_id:
        return None
    
    # Validate verse ID format strictly
    if not re.match(r'^\d{1,2}-\d{1,3}-\d{1,4}(?:-\d{1,4})?$', verse_id):
        logger.warning(f"Invalid verse ID format: {verse_id}")
        return None
    
    try:
        with _get_db_connection(db_path) as conn:
            # Parameterized query (safe from injection)
            cursor = conn.execute("""
            SELECT 
                id, verse_id, kanda_num, sarga_num, verse_index, kanda_name,
                shloka_sanskrit, shloka_iast, translation_en, translation_hi, sarga_titles_json
            FROM verses
            WHERE verse_id = ?
            """, (verse_id,))
            
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                result['search_type'] = 'exact_verse'
                result['score'] = 1.0
                
                # Parse sarga titles safely
                try:
                    import json
                    result['sarga_titles'] = json.loads(row['sarga_titles_json'] or "[]")
                except (json.JSONDecodeError, TypeError):
                    result['sarga_titles'] = []
                
                return result
            
            return None
            
    except sqlite3.Error as e:
        logger.error(f"SQLite error in exact verse search: {e}")
        return None

def search_by_coordinates(db_path: str, 
                         kanda: int, 
                         sarga: int, 
                         verse_start: int,
                         verse_end: Optional[int] = None) -> List[Dict]:
    """
    Search verses by coordinate range with input validation
    """
    
    # Validate coordinate ranges
    if not (1 <= kanda <= 7):
        logger.warning(f"Invalid kanda number: {kanda}")
        return []
    
    if not (1 <= sarga <= 200):  # Reasonable upper bound
        logger.warning(f"Invalid sarga number: {sarga}")
        return []
    
    if not (1 <= verse_start <= 500):  # Reasonable upper bound
        logger.warning(f"Invalid verse number: {verse_start}")
        return []
    
    if verse_end and (verse_end < verse_start or verse_end > 500):
        logger.warning(f"Invalid verse range: {verse_start}-{verse_end}")
        return []
    
    try:
        with _get_db_connection(db_path) as conn:
            if verse_end and verse_end > verse_start:
                # Range query with parameters
                cursor = conn.execute("""
                SELECT 
                    id, verse_id, kanda_num, sarga_num, verse_index, kanda_name,
                    shloka_sanskrit, shloka_iast, translation_en, translation_hi, sarga_titles_json
                FROM verses
                WHERE kanda_num = ? AND sarga_num = ? 
                AND verse_index >= ? AND verse_index <= ?
                ORDER BY verse_index
                """, (kanda, sarga, verse_start, verse_end))
            else:
                # Single verse query with parameters
                cursor = conn.execute("""
                SELECT 
                    id, verse_id, kanda_num, sarga_num, verse_index, kanda_name,
                    shloka_sanskrit, shloka_iast, translation_en, translation_hi, sarga_titles_json
                FROM verses
                WHERE kanda_num = ? AND sarga_num = ? AND verse_index = ?
                """, (kanda, sarga, verse_start))
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                result['search_type'] = 'coordinate'
                result['score'] = 1.0
                
                # Parse sarga titles safely
                try:
                    import json
                    result['sarga_titles'] = json.loads(row['sarga_titles_json'] or "[]")
                except (json.JSONDecodeError, TypeError):
                    result['sarga_titles'] = []
                
                results.append(result)
            
            return results
            
    except sqlite3.Error as e:
        logger.error(f"SQLite error in coordinate search: {e}")
        return []

def get_search_suggestions(db_path: str, partial_query: str, limit: int = 5) -> List[str]:
    """
    Get search suggestions based on partial query with input validation
    """
    
    if not partial_query or len(partial_query) < 2:
        return []
    
    # Sanitize input
    clean_query = re.sub(r'[^\w\s]', '', partial_query).strip()
    if not clean_query:
        return []
    
    # Length limit
    if len(clean_query) > 50:
        clean_query = clean_query[:50]
    
    # Limit bounds
    limit = max(1, min(limit, 20))
    
    try:
        with _get_db_connection(db_path) as conn:
            # Use parameterized LIKE query (safe)
            like_pattern = f"{clean_query}%"

            # Wrap the CASE selection as a subquery to filter by alias safely
            cursor = conn.execute(
                """
                SELECT suggestion FROM (
                    SELECT DISTINCT 
                        CASE 
                            WHEN translation_en LIKE ? THEN translation_en
                            WHEN shloka_iast LIKE ? THEN shloka_iast
                            ELSE kanda_name
                        END AS suggestion
                    FROM verses
                    WHERE (translation_en LIKE ? OR shloka_iast LIKE ? OR kanda_name LIKE ?)
                ) 
                WHERE suggestion IS NOT NULL AND length(suggestion) > 0
                ORDER BY length(suggestion)
                LIMIT ?
                """,
                (like_pattern, like_pattern, like_pattern, like_pattern, like_pattern, limit)
            )

            suggestions = [row['suggestion'] for row in cursor.fetchall() if row['suggestion']]
            return suggestions
            
    except sqlite3.Error as e:
        logger.error(f"SQLite error in search suggestions: {e}")
        return []

def validate_fts_setup(db_path: str, fts_table: str = "verses_fts") -> Dict[str, bool]:
    """
    Validate FTS5 setup and functionality with proper error handling
    """
    
    results = {
        "fts_table_exists": False,
        "fts_populated": False,
        "fts_triggers_exist": False,
        "search_functional": False
    }
    
    # Validate table name
    if not _validate_table_name(fts_table):
        logger.error(f"Invalid FTS table name: {fts_table}")
        return results
    
    try:
        with _get_db_connection(db_path) as conn:
            # Check if FTS table exists
            cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
            """, (fts_table,))
            
            if cursor.fetchone():
                results["fts_table_exists"] = True
                
                # Check if FTS table is populated
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {fts_table}")
                count = cursor.fetchone()["count"]
                results["fts_populated"] = count > 0
                
                # Check for FTS triggers
                cursor = conn.execute("""
                SELECT COUNT(*) as count FROM sqlite_master 
                WHERE type='trigger' AND name LIKE '%fts%'
                """)
                trigger_count = cursor.fetchone()["count"]
                results["fts_triggers_exist"] = trigger_count > 0
                
                # Test basic search functionality safely
                if results["fts_populated"]:
                    try:
                        cursor = conn.execute(f"""
                        SELECT COUNT(*) as count FROM {fts_table} 
                        WHERE {fts_table} MATCH ?
                        LIMIT 1
                        """, ("rama",))  # Safe test query
                        cursor.fetchone()
                        results["search_functional"] = True
                    except sqlite3.Error as e:
                        logger.warning(f"FTS search test failed: {e}")
                        results["search_functional"] = False
    
    except sqlite3.Error as e:
        logger.error(f"FTS validation error: {e}")
    
    return results
