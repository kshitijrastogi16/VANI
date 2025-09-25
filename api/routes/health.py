"""
VANI V3 Health Check Endpoints
Enforcing: Production-grade health monitoring and diagnostics
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import time
import sqlite3
import os

from app.dependencies import get_database_status, get_embeddings_status, get_llm_status
from app.settings import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "VANI",
        "version": "3.0.0",
        "timestamp": time.time()
    }


@router.get("/detailed")
async def detailed_health_check(
    db_status: bool = Depends(get_database_status),
    embeddings_status: bool = Depends(get_embeddings_status),
    llm_status: bool = Depends(get_llm_status)
) -> Dict[str, Any]:
    """Comprehensive health check with component status"""
    
    components = {
        "database": db_status,
        "embeddings": embeddings_status,
        "llm": llm_status
    }
    
    # Overall system status
    overall_status = "healthy" if all(components.values()) else "degraded"
    
    # Get database statistics if available
    db_stats = {}
    if db_status:
        try:
            conn = sqlite3.connect(settings.database_path)
            cursor = conn.execute("SELECT COUNT(*) as count FROM verses")
            db_stats["total_verses"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM verses_fts")  
            db_stats["fts_entries"] = cursor.fetchone()[0]
            conn.close()
        except:
            db_stats = {"error": "Could not fetch database stats"}
    
    return {
        "status": overall_status,
        "service": "VANI", 
        "version": "3.0.0",
        "timestamp": time.time(),
        "components": components,
        "database_stats": db_stats,
        "configuration": {
            "database_path": settings.database_path,
            "embeddings_dir": settings.embeddings_dir,
            "embedding_model": settings.primary_embedding_model,
            "llm_models_available": len(settings.groq_models)
        }
    }


@router.get("/ready")
async def readiness_check(
    db_status: bool = Depends(get_database_status),
    embeddings_status: bool = Depends(get_embeddings_status)
) -> Dict[str, Any]:
    """Kubernetes/Docker readiness probe"""
    
    is_ready = db_status and embeddings_status
    
    return {
        "ready": is_ready,
        "components": {
            "database": db_status,
            "embeddings": embeddings_status
        }
    }


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Kubernetes/Docker liveness probe"""
    return {"status": "alive"}
