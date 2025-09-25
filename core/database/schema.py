"""
VANI V3 Database Schema - Comprehensive Database Setup and Validation
Enforcing: V2 robust database management with WAL mode and FTS5 optimization
"""

import sqlite3
import os
import logging
from typing import Dict, List, Optional, Tuple
import json
from app.settings import settings

from core.database.models import get_create_table_sql, get_create_indexes_sql, get_create_triggers_sql

logger = logging.getLogger(__name__)


def _get_unicode61_tokenizer_clause(conn: sqlite3.Connection) -> str:
    """
    Detect whether the current SQLite build supports 'unicode61 remove_diacritics=2'.
    Falls back to plain 'unicode61' if the extended option is not supported.
    This avoids parse-error warnings during normal schema creation.
    """
    try:
        # Ensure probe table is absent
        conn.execute("DROP TABLE IF EXISTS fts_tokenizer_probe")
        # Try advanced diacritics folding option (SQLite 3.27+)
        conn.execute("""
            CREATE VIRTUAL TABLE fts_tokenizer_probe USING fts5(
                x, tokenize='unicode61 remove_diacritics=2'
            )
        """)
        conn.execute("DROP TABLE IF EXISTS fts_tokenizer_probe")
        return "unicode61 remove_diacritics=2"
    except sqlite3.Error:
        # Fall back silently to plain unicode61
        try:
            conn.execute("DROP TABLE IF EXISTS fts_tokenizer_probe")
        except sqlite3.Error:
            pass
        return "unicode61"

def ensure_database_schema(db_path: str) -> bool:
    """
    Ensure database exists with proper schema, indexes, and FTS setup
    
    Args:
        db_path: Path to SQLite database file
    
    Returns:
        True if schema setup successful, False otherwise
    """
    
    logger.info(f"Setting up database schema: {db_path}")
    
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect and configure database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Enable WAL mode for better concurrency (V2 approach)
        _configure_database_settings(conn)
        
        # Create all tables
        _create_tables(conn)
        
        # Create indexes for performance
        _create_indexes(conn)
        
        # Set up FTS5 triggers
        _create_fts_triggers(conn)
        
        # Validate schema
        validation_results = _validate_schema(conn)
        
        # Check if critical validations passed (ignore FTS-related checks)
        critical_validations = {
            k: v for k, v in validation_results.items() 
            if k not in ('fts_table_functional', 'triggers_created')
        }
        
        all_critical_passed = all(critical_validations.values())
        
        if not validation_results['fts_table_functional']:
            logger.warning("⚠️  FTS5 functionality validation failed. Full-text search may not work properly.")
            logger.warning("    The application will continue to run with limited search functionality.")
            logger.warning("    To enable full-text search, please ensure your SQLite installation has FTS5 support.")
        
        if all_critical_passed:
            logger.info("✅ Database schema setup complete")
            if not validation_results['fts_table_functional']:
                logger.info("   Note: Full-text search is disabled")
            return True
        else:
            logger.error(f"❌ Critical schema validation failed: {validation_results}")
            return False
            
    except Exception as e:
        logger.error(f"Database schema setup failed: {e}")
        return False


def _configure_database_settings(conn: sqlite3.Connection):
    """
    Configure optimal database settings for production use
    """
    
    settings = [
        # Enable WAL mode for better concurrency
        "PRAGMA journal_mode=WAL",
        
        # Enable foreign key constraints
        "PRAGMA foreign_keys=ON",
        
        # Optimize performance settings
        "PRAGMA synchronous=NORMAL",
        "PRAGMA cache_size=10000",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA mmap_size=268435456",  # 256MB
        
        # Optimize for text search
        "PRAGMA optimize"
    ]
    
    for setting in settings:
        try:
            conn.execute(setting)
            logger.debug(f"Applied setting: {setting}")
        except sqlite3.Error as e:
            logger.warning(f"Failed to apply setting '{setting}': {e}")
    
    conn.commit()


def _create_tables(conn: sqlite3.Connection):
    """
    Create all required tables with proper constraints
    """
    
    # First check if FTS5 is available
    try:
        cursor = conn.execute("PRAGMA compile_options")
        compile_options = [row[0] for row in cursor.fetchall()]
        fts5_available = 'ENABLE_FTS5' in compile_options
    except sqlite3.Error:
        fts5_available = False
    
    table_sql = get_create_table_sql()
    
    # Create tables in dependency order
    table_order = ['kandas', 'sargas', 'verses']
    
    # Only include FTS table if FTS5 is available
    if fts5_available:
        table_order.append('verses_fts')
    else:
        logger.warning("FTS5 is not available. Full-text search will be disabled.")
    
    for table_name in table_order:
        if table_name in table_sql:
            try:
                if table_name == 'verses_fts':
                    # Proactively select the right tokenizer to avoid parse errors
                    clause = _get_unicode61_tokenizer_clause(conn)
                    # Build FTS column list based on settings
                    try:
                        hi_col = ''
                        if getattr(settings, 'fts_remove_hi', False):
                            hi_col = ''  # omit column entirely
                        else:
                            if getattr(settings, 'fts_unindex_hi', True):
                                hi_col = '\n                            translation_hi UNINDEXED,'
                            else:
                                hi_col = '\n                            translation_hi,'
                        ddl = f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS verses_fts USING fts5(
                            verse_id UNINDEXED,
                            kanda_name UNINDEXED,
                            shloka_iast,
                            translation_en,{hi_col}
                            sarga_titles_json UNINDEXED,
                            content='verses',
                            content_rowid='id',
                            tokenize='{clause}'
                        )
                        """
                        conn.execute(ddl)
                        logger.info(f"✅ FTS table created with tokenizer '{clause}' (remove_hi={getattr(settings, 'fts_remove_hi', False)}, unindex_hi={getattr(settings, 'fts_unindex_hi', True)})")
                    except sqlite3.Error as e:
                        raise
                else:
                    conn.execute(table_sql[table_name])
                    logger.debug(f"Created table: {table_name}")
            except sqlite3.Error as e:
                if table_name == 'verses_fts':
                    logger.warning(f"Failed to create FTS table (continuing without FTS): {e}")
                else:
                    logger.error(f"Failed to create table {table_name}: {e}")
                    raise
    
    conn.commit()


def _create_indexes(conn: sqlite3.Connection):
    """
    Create performance indexes
    """
    
    indexes = get_create_indexes_sql()
    
    for index_sql in indexes:
        try:
            conn.execute(index_sql)
            logger.debug(f"Created index: {index_sql}")
        except sqlite3.Error as e:
            logger.warning(f"Failed to create index: {e}")
    
    conn.commit()


def _create_fts_triggers(conn: sqlite3.Connection):
    """
    Create FTS5 synchronization triggers
    """
    
    # First check if FTS5 is available
    try:
        cursor = conn.execute("PRAGMA compile_options")
        compile_options = [row[0] for row in cursor.fetchall()]
        
        if 'ENABLE_FTS5' not in compile_options:
            logger.warning("FTS5 is not enabled in this SQLite build. FTS triggers will not be created.")
            return False
            
        # Check if the FTS table exists
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='verses_fts'")
        if cursor.fetchone()[0] == 0:
            logger.warning("FTS table 'verses_fts' does not exist. FTS triggers will not be created.")
            return False
            
        # If we get here, FTS5 is available and the table exists
        triggers = get_create_triggers_sql()
        
        for trigger_sql in triggers:
            try:
                conn.execute(trigger_sql)
                logger.debug("Created FTS trigger")
            except sqlite3.Error as e:
                logger.warning(f"Failed to create FTS trigger (continuing without FTS): {e}")
                return False
        
        conn.commit()
        return True
        
    except sqlite3.Error as e:
        logger.warning(f"Error setting up FTS triggers (continuing without FTS): {e}")
        return False


def _validate_schema(conn: sqlite3.Connection) -> Dict[str, bool]:
    """
    Validate database schema completeness and functionality
    """
    
    validation = {
        "required_tables_exist": False,
        "fts_table_functional": False,
        "indexes_created": False,
        "triggers_created": False,
        "foreign_keys_enabled": False,
        "wal_mode_enabled": False
    }
    
    try:
        # Check required base tables (FTS table is optional)
        cursor = conn.execute("""
        SELECT COUNT(*) as count FROM sqlite_master 
        WHERE type='table' AND name IN ('kandas', 'sargas', 'verses')
        """)
        table_count = cursor.fetchone()["count"]
        validation["required_tables_exist"] = (table_count == 3)
        
        # Check FTS5 functionality
        try:
            # First check if FTS5 is available
            cursor = conn.execute("PRAGMA compile_options")
            compile_options = [row[0] for row in cursor.fetchall()]
            
            if 'ENABLE_FTS5' not in compile_options:
                logger.warning("FTS5 is not enabled in this SQLite build. Full-text search will be disabled.")
                validation["fts_table_functional"] = False
            else:
                # Check if the FTS table exists and is accessible
                cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='verses_fts'")
                if cursor.fetchone()[0] == 1:
                    # Try a simple query on the FTS table
                    cursor = conn.execute("SELECT COUNT(*) FROM verses_fts LIMIT 1")
                    cursor.fetchone()
                    validation["fts_table_functional"] = True
                else:
                    logger.warning("FTS table 'verses_fts' does not exist")
                    validation["fts_table_functional"] = False
                    
        except sqlite3.Error as e:
            logger.warning(f"FTS5 validation warning: {e}")
            validation["fts_table_functional"] = False
        
        # Check indexes
        cursor = conn.execute("""
        SELECT COUNT(*) as count FROM sqlite_master 
        WHERE type='index' AND name LIKE 'idx_%'
        """)
        index_count = cursor.fetchone()["count"]
        validation["indexes_created"] = (index_count >= 3)
        
        # Check triggers
        cursor = conn.execute("""
        SELECT COUNT(*) as count FROM sqlite_master 
        WHERE type='trigger' AND name LIKE '%fts%'
        """)
        trigger_count = cursor.fetchone()["count"]
        validation["triggers_created"] = (trigger_count >= 3)
        
        # Check foreign keys
        cursor = conn.execute("PRAGMA foreign_keys")
        fk_enabled = cursor.fetchone()[0]
        validation["foreign_keys_enabled"] = bool(fk_enabled)
        
        # Check WAL mode
        cursor = conn.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        validation["wal_mode_enabled"] = (journal_mode.upper() == 'WAL')
        
    except sqlite3.Error as e:
        logger.error(f"Schema validation error: {e}")
    
    return validation


def get_database_stats(db_path: str) -> Dict[str, any]:
    """
    Get comprehensive database statistics
    """
    
    if not os.path.exists(db_path):
        return {"error": "Database file not found"}
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        stats = {}
        
        # Basic counts
        cursor = conn.execute("SELECT COUNT(*) as count FROM verses")
        stats["total_verses"] = cursor.fetchone()["count"]
        
        cursor = conn.execute("SELECT COUNT(DISTINCT kanda_num) as count FROM verses")
        stats["total_kandas"] = cursor.fetchone()["count"]
        
        cursor = conn.execute("SELECT COUNT(DISTINCT sarga_num) as count FROM verses")
        stats["total_sargas"] = cursor.fetchone()["count"]
        
        # FTS statistics
        try:
            cursor = conn.execute("SELECT COUNT(*) as count FROM verses_fts")
            stats["fts_entries"] = cursor.fetchone()["count"]
        except sqlite3.Error:
            stats["fts_entries"] = 0
        
        # Database file size
        stats["file_size_mb"] = round(os.path.getsize(db_path) / (1024 * 1024), 2)
        
        # Kanda breakdown
        cursor = conn.execute("""
        SELECT kanda_name, COUNT(*) as verse_count
        FROM verses
        GROUP BY kanda_name
        ORDER BY kanda_num
        """)
        stats["kanda_breakdown"] = {row["kanda_name"]: row["verse_count"] for row in cursor.fetchall()}
        
        conn.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e)}


def rebuild_fts_index(db_path: str) -> bool:
    """
    Rebuild FTS5 index from scratch
    """
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Check if FTS5 is available
        cursor = conn.execute("PRAGMA compile_options")
        compile_options = [row[0] for row in cursor.fetchall()]
        if 'ENABLE_FTS5' not in compile_options:
            logger.warning("FTS5 is not enabled in this SQLite build. Cannot rebuild FTS index.")
            return False
            
        logger.info("🔄 Rebuilding FTS5 search index...")
        
        # Drop existing FTS table and triggers
        conn.execute("DROP TABLE IF EXISTS verses_fts")
        conn.execute("DROP TRIGGER IF EXISTS verses_fts_insert")
        conn.execute("DROP TRIGGER IF EXISTS verses_fts_delete")
        conn.execute("DROP TRIGGER IF EXISTS verses_fts_update")
        
        # Recreate FTS table using proactively detected tokenizer clause
        try:
            clause = _get_unicode61_tokenizer_clause(conn)
            # Build DDL based on settings
            hi_col = ''
            if getattr(settings, 'fts_remove_hi', False):
                hi_col = ''
            else:
                if getattr(settings, 'fts_unindex_hi', True):
                    hi_col = '\n                    translation_hi UNINDEXED,'
                else:
                    hi_col = '\n                    translation_hi,'
            ddl = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS verses_fts USING fts5(
                verse_id UNINDEXED,
                kanda_name UNINDEXED,
                shloka_iast,
                translation_en,{hi_col}
                sarga_titles_json UNINDEXED,
                content='verses',
                content_rowid='id',
                tokenize='{clause}'
            )
            """
            conn.execute(ddl)
            logger.info(f"✅ FTS5 table created with tokenizer '{clause}' (remove_hi={getattr(settings, 'fts_remove_hi', False)}, unindex_hi={getattr(settings, 'fts_unindex_hi', True)})")
        except sqlite3.Error as e:
            logger.error(f"FTS rebuild failed creating table: {e}")
            conn.rollback()
            conn.close()
            return False
        
        # Recreate triggers
        triggers = get_create_triggers_sql()
        for trigger_sql in triggers:
            conn.execute(trigger_sql)
        
        # Rebuild FTS content from verses table
        try:
            if getattr(settings, 'fts_remove_hi', False):
                conn.execute("""
                INSERT INTO verses_fts(
                    rowid, verse_id, kanda_name,
                    shloka_iast, translation_en, sarga_titles_json
                )
                SELECT 
                    id, verse_id, kanda_name,
                    shloka_iast, translation_en, sarga_titles_json
                FROM verses
                """)
            else:
                conn.execute("""
                INSERT INTO verses_fts(
                    rowid, verse_id, kanda_name, 
                    shloka_iast, translation_en, translation_hi, sarga_titles_json
                )
                SELECT 
                    id, verse_id, kanda_name, 
                    shloka_iast, translation_en, translation_hi, sarga_titles_json
                FROM verses
                """)
        except sqlite3.Error as e:
            logger.error(f"FTS content rebuild failed: {e}")
            conn.rollback()
            conn.close()
            return False
        
        # Optimize FTS index
        try:
            conn.execute("INSERT INTO verses_fts(verses_fts) VALUES('optimize')")
        except sqlite3.Error as e:
            logger.warning(f"FTS optimize failed (continuing): {e}")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ FTS5 index rebuilt successfully")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"FTS rebuild failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False


def check_database_integrity(db_path: str) -> Dict[str, any]:
    """
    Comprehensive database integrity check
    """
    
    if not os.path.exists(db_path):
        return {"status": "error", "message": "Database file not found"}
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        integrity_report = {
            "status": "healthy",
            "checks": {},
            "issues": []
        }
        
        # SQLite integrity check
        cursor = conn.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        integrity_report["checks"]["sqlite_integrity"] = (integrity_result == "ok")
        
        if integrity_result != "ok":
            integrity_report["issues"].append(f"SQLite integrity: {integrity_result}")
            integrity_report["status"] = "degraded"
        
        # Foreign key check
        cursor = conn.execute("PRAGMA foreign_key_check")
        fk_violations = cursor.fetchall()
        integrity_report["checks"]["foreign_key_integrity"] = (len(fk_violations) == 0)
        
        if fk_violations:
            integrity_report["issues"].append(f"Foreign key violations: {len(fk_violations)}")
            integrity_report["status"] = "degraded"
        
        # Check for orphaned FTS entries
        cursor = conn.execute("""
        SELECT COUNT(*) as count FROM verses_fts 
        WHERE rowid NOT IN (SELECT id FROM verses)
        """)
        orphaned_fts = cursor.fetchone()["count"]
        integrity_report["checks"]["fts_sync"] = (orphaned_fts == 0)
        
        if orphaned_fts > 0:
            integrity_report["issues"].append(f"Orphaned FTS entries: {orphaned_fts}")
            integrity_report["status"] = "degraded"
        
        # Check verse ID consistency (accept single and range formats: k-s-v or k-s-v1-v2)
        try:
            import re
            cursor = conn.execute("""
            SELECT verse_id, kanda_num, sarga_num, verse_index FROM verses
            """)
            rows = cursor.fetchall()
            bad = 0
            pattern = re.compile(r"^(\d+)-(\d+)-(\d+)(?:-(\d+))?$")
            for row in rows:
                vid = row["verse_id"] or ""
                km = int(row["kanda_num"])
                sm = int(row["sarga_num"])
                vi = int(row["verse_index"])
                m = pattern.match(vid)
                if not m:
                    bad += 1
                    continue
                k, s, v_start, v_end = int(m.group(1)), int(m.group(2)), int(m.group(3)), (int(m.group(4)) if m.group(4) else None)
                if k != km or s != sm:
                    bad += 1
                    continue
                # For ranges, ensure verse_index equals range start; for single, equals the verse
                if v_start != vi:
                    bad += 1
            integrity_report["checks"]["verse_id_consistency"] = (bad == 0)
            if bad > 0:
                integrity_report["issues"].append(f"Inconsistent verse IDs: {bad}")
                integrity_report["status"] = "degraded"
        except Exception as e:
            integrity_report["checks"]["verse_id_consistency"] = False
            integrity_report["issues"].append(f"Verse ID consistency check failed: {e}")
            integrity_report["status"] = "degraded"
        
        conn.close()
        
        return integrity_report
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "checks": {},
            "issues": [f"Integrity check failed: {e}"]
        }
