"""
VANI V3 Logging Configuration - Centralized Logging Setup
Enforcing: Production-grade logging with structured output
"""

import logging
import logging.handlers
import os
from typing import Optional
import sys
from datetime import datetime


# Idempotence guard to prevent duplicate initialization
_LOGGING_INITIALIZED = False
_LAST_CONFIG = (None, None)  # (level, log_file)


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Setup centralized logging configuration for VANI
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    global _LOGGING_INITIALIZED, _LAST_CONFIG
    # If already initialized, just update levels and return existing logger without re-adding handlers
    if _LOGGING_INITIALIZED:
        try:
            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)
            for handler in root_logger.handlers:
                try:
                    handler.setLevel(numeric_level)
                except Exception:
                    pass
        except Exception:
            pass
        return logging.getLogger("vani")
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers (ensure single console/file handlers)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log startup message only once
    logger = logging.getLogger("vani")
    logger.info(f"🚀 VANI logging initialized (level: {log_level})")
    _LOGGING_INITIALIZED = True
    _LAST_CONFIG = (log_level.upper(), (log_file or ""))
    
    return logger


class VANILogger:
    """
    VANI-specific logger wrapper with context management
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"vani.{name}")
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context"""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context"""  
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context"""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context"""
        self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional context"""
        self.logger.critical(self._format_message(message, kwargs))
    
    def _format_message(self, message: str, context: dict) -> str:
        """Format message with context information"""
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            return f"{message} | {context_str}"
        return message


# Component-specific loggers
def get_component_logger(component: str) -> VANILogger:
    """Get logger for specific VANI component"""
    return VANILogger(component)


# Performance logging decorator
def log_performance(logger: logging.Logger):
    """Decorator to log function performance"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator
