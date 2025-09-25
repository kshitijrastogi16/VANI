"""
VANI V3 Database Connection Management - FIXED VERSION
Enforcing: Connection pooling, proper resource management, thread safety
"""

import sqlite3
import threading
import time
import os
import logging
from queue import Queue, Empty
from contextlib import contextmanager
from typing import Optional, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConnectionStats:
    total_connections: int
    active_connections: int
    created_connections: int
    closed_connections: int
    total_queries: int
    avg_query_time: float

class DatabaseManager:
    """
    Production-grade database connection manager with pooling
    Features: Connection pooling, automatic cleanup, thread safety
    """
    
    def __init__(self, 
                 db_path: str, 
                 pool_size: int = 10,
                 max_overflow: int = 5,
                 timeout: int = 30,
                 recycle_time: int = 3600):
        
        self.db_path = db_path
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.recycle_time = recycle_time
        
        # Connection pool
        self.pool = Queue(maxsize=pool_size + max_overflow)
        self.overflow_connections = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = ConnectionStats(0, 0, 0, 0, 0, 0.0)
        self._total_query_time = 0.0
        
        # Initialize pool
        self._initialize_pool()
        
        logger.info(f"✅ Database manager initialized (pool_size={pool_size})")
    
    def _initialize_pool(self):
        """Initialize connection pool with base connections"""
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        with self.lock:
            for _ in range(self.pool_size):
                conn = self._create_connection()
                if conn:
                    self.pool.put({
                        'connection': conn,
                        'created_at': time.time(),
                        'query_count': 0
                    })
                    self.stats.created_connections += 1
            
            self.stats.total_connections = self.pool.qsize()
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection with optimal settings"""
        
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            
            # Configure connection for optimal performance
            conn.row_factory = sqlite3.Row
            
            # Apply PRAGMA settings
            pragmas = [
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=NORMAL", 
                "PRAGMA cache_size=10000",
                "PRAGMA temp_store=MEMORY",
                "PRAGMA mmap_size=268435456",  # 256MB
                "PRAGMA foreign_keys=ON"
            ]
            
            for pragma in pragmas:
                try:
                    conn.execute(pragma)
                except sqlite3.Error as e:
                    logger.warning(f"Failed to apply {pragma}: {e}")
            
            return conn
            
        except sqlite3.Error as e:
            logger.error(f"Failed to create database connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a connection from pool with automatic cleanup
        
        Usage:
            with db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM verses")
                results = cursor.fetchall()
        """
        
        conn_info = None
        start_time = time.time()
        
        try:
            # Get connection from pool
            conn_info = self._get_connection_from_pool()
            
            if not conn_info:
                raise RuntimeError("Unable to get database connection")
            
            connection = conn_info['connection']
            
            # Validate connection
            if not self._validate_connection(connection):
                # Connection is stale, create new one
                connection.close()
                connection = self._create_connection()
                if not connection:
                    raise RuntimeError("Failed to create new database connection")
                conn_info['connection'] = connection
                conn_info['created_at'] = time.time()
            
            with self.lock:
                self.stats.active_connections += 1
            
            yield connection
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            # Mark connection as potentially bad
            if conn_info and 'connection' in conn_info:
                try:
                    conn_info['connection'].rollback()
                except:
                    pass
            raise
            
        finally:
            # Return connection to pool
            query_time = time.time() - start_time
            
            with self.lock:
                self.stats.active_connections = max(0, self.stats.active_connections - 1)
                self.stats.total_queries += 1
                self._total_query_time += query_time
                self.stats.avg_query_time = self._total_query_time / self.stats.total_queries
            
            if conn_info:
                self._return_connection_to_pool(conn_info)
    
    def _get_connection_from_pool(self) -> Optional[dict]:
        """Get connection from pool with overflow handling"""
        
        try:
            # Try to get from pool
            return self.pool.get(timeout=5)
            
        except Empty:
            # Pool is empty, try overflow
            with self.lock:
                if self.overflow_connections < self.max_overflow:
                    conn = self._create_connection()
                    if conn:
                        self.overflow_connections += 1
                        self.stats.created_connections += 1
                        return {
                            'connection': conn,
                            'created_at': time.time(),
                            'query_count': 0,
                            'is_overflow': True
                        }
            
            logger.error("Connection pool exhausted")
            return None
    
    def _return_connection_to_pool(self, conn_info: dict):
        """Return connection to pool or close if overflow/stale"""
        
        try:
            connection = conn_info['connection']
            
            # Check if connection should be recycled
            if (time.time() - conn_info['created_at']) > self.recycle_time:
                connection.close()
                with self.lock:
                    self.stats.closed_connections += 1
                    if conn_info.get('is_overflow'):
                        self.overflow_connections = max(0, self.overflow_connections - 1)
                return
            
            # If overflow connection, close it
            if conn_info.get('is_overflow'):
                connection.close()
                with self.lock:
                    self.overflow_connections = max(0, self.overflow_connections - 1)
                    self.stats.closed_connections += 1
                return
            
            # Return to pool
            conn_info['query_count'] += 1
            self.pool.put(conn_info)
            
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
            # Ensure overflow counter is decremented
            if conn_info.get('is_overflow'):
                with self.lock:
                    self.overflow_connections = max(0, self.overflow_connections - 1)
    
    def _validate_connection(self, connection: sqlite3.Connection) -> bool:
        """Validate that connection is still usable"""
        
        try:
            connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute query with connection management"""
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: list) -> int:
        """Execute multiple queries in a transaction"""
        
        with self.get_connection() as conn:
            cursor = conn.executemany(query, params_list)
            return cursor.rowcount
    
    def get_stats(self) -> ConnectionStats:
        """Get connection pool statistics"""
        
        with self.lock:
            self.stats.total_connections = self.pool.qsize() + self.overflow_connections
            return self.stats
    
    def health_check(self) -> dict:
        """Perform health check on database"""
        
        try:
            with self.get_connection() as conn:
                # Test basic operations
                conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
                
                # Test FTS if available
                try:
                    conn.execute("SELECT COUNT(*) FROM verses_fts LIMIT 1").fetchone()
                    fts_healthy = True
                except sqlite3.Error:
                    fts_healthy = False
                
                stats = self.get_stats()
                
                return {
                    "status": "healthy",
                    "database_accessible": True,
                    "fts_available": fts_healthy,
                    "pool_stats": {
                        "total_connections": stats.total_connections,
                        "active_connections": stats.active_connections,
                        "created_connections": stats.created_connections,
                        "closed_connections": stats.closed_connections,
                        "total_queries": stats.total_queries,
                        "avg_query_time": round(stats.avg_query_time, 4)
                    }
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_accessible": False
            }
    
    def close_all(self):
        """Close all connections in pool"""
        
        logger.info("Closing all database connections...")
        
        with self.lock:
            # Close pooled connections
            while not self.pool.empty():
                try:
                    conn_info = self.pool.get_nowait()
                    conn_info['connection'].close()
                    self.stats.closed_connections += 1
                except Empty:
                    break
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            # Reset counters
            self.overflow_connections = 0
            self.stats.total_connections = 0
            self.stats.active_connections = 0
        
        logger.info("✅ All database connections closed")

# Global database manager instance
_db_manager = None

def get_database_manager(db_path: str = None) -> DatabaseManager:
    """Get global database manager instance"""
    
    global _db_manager
    
    if _db_manager is None and db_path:
        _db_manager = DatabaseManager(db_path)
    
    return _db_manager

def close_database_manager():
    """Close global database manager"""
    
    global _db_manager
    
    if _db_manager:
        _db_manager.close_all()
        _db_manager = None
