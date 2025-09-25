"""
VANI V3 Dependencies - FastAPI Dependency Injection
Enforcing: V2 proper dependency management patterns
"""

import sqlite3
import os
from typing import Dict, Any
from fastapi import Depends, HTTPException

from app.settings import settings


def get_database_status() -> bool:
    """Check database connectivity and schema validity"""
    try:
        if not os.path.exists(settings.database_path):
            return False
            
        conn = sqlite3.connect(settings.database_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check required base tables exist (FTS is optional)
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('verses', 'kandas', 'sargas')
            """
        )
        base_tables = {row[0] for row in cursor.fetchall()}

        has_base = all(t in base_tables for t in ('verses', 'kandas', 'sargas'))

        # Optionally check FTS table presence and basic accessibility, but do not fail overall
        try:
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='verses_fts'")
            fts_exists = cursor.fetchone()[0] == 1
            if fts_exists:
                try:
                    cursor.execute("SELECT COUNT(*) FROM verses_fts LIMIT 1")
                    _ = cursor.fetchone()
                except Exception:
                    # FTS present but not accessible; treat as degraded, not failure
                    pass
        except Exception:
            # Ignore FTS check errors entirely
            pass

        conn.close()
        return has_base
        
    except Exception:
        return False


def get_embeddings_status() -> bool:
    """Check embeddings availability"""
    try:
        required_files = [
            "embeddings.npy",
            "mappings.json", 
            "metadata.json"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(settings.embeddings_dir, file)):
                return False
                
        return True
    except Exception:
        return False


def get_llm_status() -> bool:
    """Check LLM client availability"""
    try:
        return bool(settings.groq_api_key)
    except Exception:
        return False


def get_app_state() -> Dict[str, Any]:
    """Get application state for dependency injection"""
    # This will be populated by the lifespan manager
    return {}
