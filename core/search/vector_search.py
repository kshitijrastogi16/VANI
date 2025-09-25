"""
VANI V3 Vector Search - Semantic Similarity Search Utilities
Enforcing: Clean vector search operations with FAISS integration
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    verse_id: str
    similarity_score: float
    vector_index: int
    metadata: Dict


class VectorSearchEngine:
    """
    Semantic similarity search using vector embeddings
    Integrates with core.embeddings.storage.VectorStore
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def search(self, query: str, top_k: int = 10, min_similarity: float = 0.0) -> List[VectorSearchResult]:
        """
        Perform semantic similarity search
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of VectorSearchResult objects
        """
        
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Use the vector store's similarity search
            raw_results = self.vector_store.similarity_search(query, top_k=top_k)
            
            # Convert to VectorSearchResult format
            results = []
            for idx, result in enumerate(raw_results):
                if result.get('score', 0) >= min_similarity:
                    vector_result = VectorSearchResult(
                        verse_id=result.get('verse_id', ''),
                        similarity_score=result.get('score', 0.0),
                        vector_index=idx,
                        metadata=result
                    )
                    results.append(vector_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def get_similar_verses(self, reference_verse_id: str, top_k: int = 5) -> List[VectorSearchResult]:
        """
        Find verses similar to a reference verse
        """
        
        # This would require getting the embedding for the reference verse
        # and searching for similar embeddings
        # Implementation depends on vector store capabilities
        
        logger.info(f"Finding verses similar to {reference_verse_id}")
        # Placeholder implementation
        return []


def search_by_embedding(vector_store, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
    """
    Direct search using pre-computed embedding
    """
    
    if vector_store and hasattr(vector_store, 'index'):
        try:
            # Direct FAISS search
            scores, indices = vector_store.index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 
                top_k
            )
            
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])
                
                if idx < len(vector_store.mappings):
                    mapping = vector_store.mappings[idx]
                    result = {**mapping, 'score': score, 'search_type': 'direct_embedding'}
                    # Windowed compatibility: ensure a singular verse_id for fusion/citation
                    if not result.get('verse_id'):
                        anchor = result.get('anchor_verse_id')
                        if anchor:
                            result['verse_id'] = anchor
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Direct embedding search error: {e}")
    
    return []
