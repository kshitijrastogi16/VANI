"""
VANI V3 Vector Storage - FIXED V2 Bugs + Enhanced Features
Enforcing: Fix array indexing bugs and add E5 asymmetric support
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Production-grade vector storage with FIXED bugs"""
    
    def __init__(self, embeddings_dir: str, primary_model: str, fallback_model: str, tertiary_model: Optional[str] = None):
        self.embeddings_dir = embeddings_dir
        self.primary_model_id = primary_model
        self.fallback_model_id = fallback_model
        self.tertiary_model_id = tertiary_model
        
        # Initialize components
        self.model: Optional[SentenceTransformer] = None
        self.current_model_id: Optional[str] = None
        self.index: Optional[faiss.Index] = None
        self.mappings: List[Dict] = []
        self.metadata: Dict = {}
        # Note: Encoder model is loaded lazily in load() after reading metadata
    
    def _load_model(self):
        """Load embedding model with fallback strategy"""
        
        try:
            logger.info(f"Loading primary embedding model: {self.primary_model_id}")
            self.model = SentenceTransformer(self.primary_model_id)
            logger.info("✅ Primary model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            
            try:
                logger.info(f"Loading fallback model: {self.fallback_model_id}")
                self.model = SentenceTransformer(self.fallback_model_id)
                logger.info("✅ Fallback model loaded successfully")
                
            except Exception as e2:
                logger.warning(f"Fallback model failed: {e2}")
                # Try tertiary if provided
                if self.tertiary_model_id:
                    try:
                        logger.info(f"Loading tertiary model: {self.tertiary_model_id}")
                        self.model = SentenceTransformer(self.tertiary_model_id)
                        logger.info("✅ Tertiary model loaded successfully")
                        return
                    except Exception as e3:
                        logger.error(f"All models failed (primary, fallback, tertiary): {e3}")
                        raise RuntimeError("Failed to load any embedding model")
                else:
                    logger.error("Both primary and fallback models failed, and no tertiary model provided.")
                    raise RuntimeError("Failed to load any embedding model")
    
    def load(self):
        """Load vector index and metadata with FIXED BUGS"""
        
        # Load metadata
        metadata_path = os.path.join(self.embeddings_dir, "metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load mappings
        mappings_path = os.path.join(self.embeddings_dir, "mappings.json")
        with open(mappings_path, 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
        
        # Decide which encoder to load (prefer embeddings' model for compatibility)
        preferred_model = (self.metadata.get("model_id") or "").strip()
        source = "metadata"
        if not preferred_model:
            preferred_model = (self.primary_model_id or "").strip()
            source = "settings.primary"
        if not preferred_model:
            preferred_model = (self.fallback_model_id or "").strip()
            source = "settings.fallback"
        if not preferred_model and self.tertiary_model_id:
            preferred_model = (self.tertiary_model_id or "").strip()
            source = "settings.tertiary"
        
        # Load encoder only once, aligned to the embeddings space
        if (self.model is None) or (self.current_model_id or "") != preferred_model:
            try:
                logger.info(f"Loading embedding encoder ({source}): {preferred_model}")
                self.model = SentenceTransformer(preferred_model)
                self.current_model_id = preferred_model
                logger.info("✅ Encoder model loaded")
            except Exception as e:
                logger.warning(f"Preferred encoder load failed ({preferred_model}): {e}")
                # Fallback sequence
                for alt in [self.primary_model_id, self.fallback_model_id, self.tertiary_model_id]:
                    alt = (alt or "").strip()
                    if not alt or alt == preferred_model:
                        continue
                    try:
                        logger.info(f"Loading fallback encoder: {alt}")
                        self.model = SentenceTransformer(alt)
                        self.current_model_id = alt
                        logger.info("✅ Fallback encoder loaded")
                        break
                    except Exception as e2:
                        logger.warning(f"Fallback encoder failed ({alt}): {e2}")
                if self.model is None:
                    raise RuntimeError("Failed to load any embedding encoder")
        
        # Load or create FAISS index
        index_path = os.path.join(self.embeddings_dir, "index.faiss")
        embeddings_path = os.path.join(self.embeddings_dir, "embeddings.npy")
        
        if os.path.exists(index_path):
            # Load existing index
            self.index = faiss.read_index(index_path)
            logger.info(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
            
        else:
            # Create index from embeddings
            logger.info("Creating FAISS index from embeddings...")
            
            embeddings = np.load(embeddings_path).astype(np.float32)
            
            if embeddings.ndim != 2 or embeddings.size == 0:
                raise RuntimeError(f"Invalid embeddings shape: {embeddings.shape}")
            
            # CRITICAL BUG FIX: Correct dimension access
            vector_dim = embeddings.shape[1]  # FIX: was embeddings.shape[4]
            
            # Normalize for cosine similarity via inner product
            faiss.normalize_L2(embeddings)
            
            # Create index
            self.index = faiss.IndexFlatIP(vector_dim)  # FIX: was embeddings.shape[4]
            self.index.add(embeddings)
            
            # Save index
            faiss.write_index(self.index, index_path)
            logger.info(f"✅ Created and saved FAISS index: {self.index.ntotal} vectors")
        
        # Validate consistency
        if len(self.mappings) != self.index.ntotal:
            logger.warning(f"Mappings count ({len(self.mappings)}) != index count ({self.index.ntotal})")
        # Encoder is already aligned to metadata (if provided). No further reconciliation needed.
    
    def similarity_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform semantic similarity search"""
        
        if not self.model or not self.index:
            raise RuntimeError("Vector store not properly loaded")
        
        # Encode query with E5 prefix if needed
        if "e5" in self.metadata.get("model_id", "").lower():
            query_text = f"query: {query}"
        else:
            query_text = query
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)
        query_vector = query_embedding.astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        # CRITICAL BUG FIX: Correct score/index access
        results = []
        for i in range(scores.shape[1]):  # FIX: was scores.shape[4]
            idx = indices[0][i]
            score = float(scores[0][i])
            
            if idx < len(self.mappings):
                mapping = self.mappings[idx]
                result = {**mapping, 'score': score, 'search_type': 'vector_similarity'}
                # Windowed embeddings compatibility: ensure a singular verse_id for fusion
                if not result.get('verse_id'):
                    anchor = result.get('anchor_verse_id')
                    if anchor:
                        result['verse_id'] = anchor
                results.append(result)
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Use appropriate prefix for E5 models
        if "e5" in self.metadata.get("model_id", "").lower():
            text = f"passage: {text}"
        
        return self.model.encode([text], normalize_embeddings=True)[0]
    
    def add_documents(self, texts: List[str], metadata: List[Dict]) -> bool:
        """Add new documents to the vector store (for future expansion)"""
        
        if not self.model or not self.index:
            raise RuntimeError("Vector store not properly loaded")
        
        if len(texts) != len(metadata):
            raise ValueError("Texts and metadata must have same length")
        
        # Generate embeddings
        embeddings = []
        for text in texts:
            if "e5" in self.metadata.get("model_id", "").lower():
                text = f"passage: {text}"
            
            embedding = self.model.encode([text], normalize_embeddings=True)[0]
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Update mappings
        # Ensure each metadata entry has a compatible verse_id (fallback to anchor_verse_id)
        fixed_meta = []
        for m in metadata:
            if 'verse_id' not in m or not m.get('verse_id'):
                anchor = m.get('anchor_verse_id')
                if anchor:
                    m = {**m, 'verse_id': anchor}
            fixed_meta.append(m)
        self.mappings.extend(fixed_meta)
        
        # Save updated index and mappings
        index_path = os.path.join(self.embeddings_dir, "index.faiss")
        mappings_path = os.path.join(self.embeddings_dir, "mappings.json")
        
        faiss.write_index(self.index, index_path)
        
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(self.mappings, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Added {len(texts)} new documents")
        return True
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        
        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "vector_dimension": self.metadata.get("dim", 0),
            "model_id": self.metadata.get("model_id", "unknown"),
            "creation_time": self.metadata.get("creation_time", "unknown"),
            "normalization": self.metadata.get("normalization", "unknown")
        }
        
        return stats

    def cleanup(self) -> None:
        """Release resources and reset in-memory state (safe no-op).
        Matches app shutdown expectations (app.main -> _cleanup_resources)."""
        try:
            # Explicitly drop references so Python can GC the objects
            self.index = None
        except Exception as e:
            logger.debug(f"VectorStore cleanup (index): {e}")
        try:
            self.model = None
        except Exception as e:
            logger.debug(f"VectorStore cleanup (model): {e}")
        # Clear lightweight structures
        self.mappings = []
        self.metadata = {}
        logger.info("✅ VectorStore resources released")
