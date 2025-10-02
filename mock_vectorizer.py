"""
Mock vectorizer to bypass OpenAI quota issues.
Creates simple hash-based vectors for testing purposes.
"""

import hashlib
import random
import numpy as np
from typing import List, Any
import logging

logger = logging.getLogger(__name__)

class MockVectorizer:
    """Mock vectorizer that creates deterministic vectors from text."""
    
    def __init__(self, vector_dim: int = 1536):
        """Initialize mock vectorizer with specified dimension."""
        self.vector_dim = vector_dim
        logger.info(f"Initialized mock vectorizer with dimension {vector_dim}")
    
    def vectorize(self, text: str) -> List[float]:
        """
        Create a mock vector from text using hash-based approach.
        
        Args:
            text: Input text to vectorize
            
        Returns:
            List of floats representing the vector
        """
        # Create a deterministic seed from text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        
        # Set random seed for reproducible results
        random.seed(seed)
        
        # Generate vector with some structure
        vector = []
        for i in range(self.vector_dim):
            # Create some patterns based on text characteristics
            if i < len(text):
                char_val = ord(text[i % len(text)])
                val = (char_val / 255.0) * 2 - 1  # Normalize to [-1, 1]
            else:
                val = random.uniform(-1, 1)
            
            vector.append(val)
        
        # Normalize the vector
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def batch_vectorize(self, texts: List[str]) -> List[List[float]]:
        """
        Vectorize multiple texts.
        
        Args:
            texts: List of texts to vectorize
            
        Returns:
            List of vectors
        """
        return [self.vectorize(text) for text in texts]

# Global instance
mock_vectorizer = MockVectorizer()

def get_mock_vector(text: str) -> List[float]:
    """Get a mock vector for text."""
    return mock_vectorizer.vectorize(text)
