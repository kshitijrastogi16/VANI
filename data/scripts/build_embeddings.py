"""
VANI V3 Build Embeddings Script - Production Embedding Generation
Enforcing: V1 fallback strategy + V2 performance + V3 bug fixes
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.embeddings.generator import EmbeddingsGenerator
from app.settings import settings
from utils.logging import setup_logging

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    """Command-line interface for embedding generation"""
    
    parser = argparse.ArgumentParser(description="Generate embeddings for VANI (DB or JSON windowed)")
    parser.add_argument(
        "--db-path", 
        default=settings.database_path,
        help="Path to SQLite database (for DB mode)"
    )
    parser.add_argument(
        "--output-dir",
        default=settings.embeddings_dir, 
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--primary-model",
        default=settings.primary_embedding_model,
        help="Primary embedding model"
    )
    parser.add_argument(
        "--fallback-model", 
        default=settings.fallback_embedding_model,
        help="Fallback embedding model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.batch_size,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings"
    )
    # JSON windowing options
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="Build windowed embeddings from raw JSON instead of DB"
    )
    parser.add_argument(
        "--json-path",
        default="data/raw/ramayana.json",
        help="Path to raw Ramayana JSON (used with --from-json)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Window size in atomic verse units (default 5)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride in atomic verse units (default 4 for 1-unit overlap with W=5)"
    )
    parser.add_argument(
        "--min-window-size",
        type=int,
        default=1,
        help="Minimum tail window size (default 1)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs for selected mode
    if args.from_json:
        if not os.path.exists(args.json_path):
            logger.error(f"JSON not found: {args.json_path}")
            return 1
    else:
        if not os.path.exists(args.db_path):
            logger.error(f"Database not found: {args.db_path}")
            return 1
    
    if os.path.exists(args.output_dir) and not args.force:
        logger.error(f"Output directory exists: {args.output_dir}. Use --force to overwrite.")
        return 1
    
    # If rebuilding in an existing directory, remove stale FAISS index to force regeneration
    try:
        index_path = os.path.join(args.output_dir, "index.faiss")
        if os.path.exists(index_path):
            os.remove(index_path)
            logger.info(f"🧹 Removed stale FAISS index: {index_path}")
    except Exception as e:
        logger.warning(f"Could not remove existing index.faiss: {e}")
    
    try:
        # Initialize generator
        generator = EmbeddingsGenerator(
            primary_model=args.primary_model,
            fallback_model=args.fallback_model,
            batch_size=args.batch_size
        )
        
        # Generate embeddings
        logger.info(f"🚀 Starting embedding generation...")
        logger.info(f"   Output: {args.output_dir}")
        logger.info(f"   Primary model: {args.primary_model}")
        logger.info(f"   Fallback model: {args.fallback_model}")

        if args.from_json:
            logger.info(f"   Mode: JSON windowed")
            logger.info(f"   JSON: {args.json_path}")
            logger.info(f"   Window: size={args.window_size}, stride={args.stride}, min={args.min_window_size}")
            result = generator.generate_embeddings_from_json(
                json_path=args.json_path,
                output_dir=args.output_dir,
                window_size_units=args.window_size,
                stride_units=args.stride,
                min_window_size_units=args.min_window_size,
            )
        else:
            logger.info(f"   Mode: DB per-verse")
            logger.info(f"   Database: {args.db_path}")
            result = generator.generate_embeddings(args.db_path, args.output_dir)
        
        logger.info("✅ Embedding generation complete!")
        logger.info(f"   Shape: {result['embeddings_shape']}")
        logger.info(f"   Model used: {result['model_id']}")
        logger.info(f"   Output directory: {result['output_dir']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
