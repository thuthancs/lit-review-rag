"""
Simple file-based storage system to completely bypass Weaviate and OpenAI.
Stores papers as JSON files with basic text search capabilities.
"""

import json
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

class SimplePaperStorage:
    """Simple file-based storage for research papers."""
    
    def __init__(self, storage_dir: str = "paper_storage"):
        """Initialize simple storage."""
        self.storage_dir = storage_dir
        self.papers_file = os.path.join(storage_dir, "papers.json")
        self.chunks_file = os.path.join(storage_dir, "chunks.json")
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage files."""
        if not os.path.exists(self.papers_file):
            with open(self.papers_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.chunks_file):
            with open(self.chunks_file, 'w') as f:
                json.dump({}, f)
    
    def _load_papers(self) -> Dict[str, Any]:
        """Load papers from storage."""
        try:
            with open(self.papers_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading papers: {e}")
            return {}
    
    def _save_papers(self, papers: Dict[str, Any]):
        """Save papers to storage."""
        try:
            with open(self.papers_file, 'w') as f:
                json.dump(papers, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving papers: {e}")
    
    def _load_chunks(self) -> Dict[str, Any]:
        """Load chunks from storage."""
        try:
            with open(self.chunks_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return {}
    
    def _save_chunks(self, chunks: Dict[str, Any]):
        """Save chunks to storage."""
        try:
            with open(self.chunks_file, 'w') as f:
                json.dump(chunks, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
    
    def insert_papers(self, papers_data: List[Dict[str, Any]]) -> bool:
        """
        Insert processed papers into storage.
        
        Args:
            papers_data: List of paper dictionaries with chunks and metadata
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            papers = self._load_papers()
            chunks = self._load_chunks()
            
            total_chunks = 0
            
            for paper in papers_data:
                paper_id = str(uuid.uuid4())
                
                # Store paper metadata
                papers[paper_id] = {
                    "title": paper.get('title', 'Unknown Title'),
                    "authors": paper.get('authors', []),
                    "abstract": paper.get('abstract', ''),
                    "publication_year": paper.get('publication_year', 0),
                    "doi": paper.get('doi', ''),
                    "chunk_count": len(paper.get('chunks', [])),
                    "uploaded_at": datetime.now().isoformat()
                }
                
                # Store each chunk
                for chunk_idx, chunk_text in enumerate(paper.get('chunks', [])):
                    chunk_id = f"{paper_id}_chunk_{chunk_idx}"
                    chunks[chunk_id] = {
                        "paper_id": paper_id,
                        "chunk_index": chunk_idx,
                        "content": chunk_text,
                        "title": paper.get('title', 'Unknown Title'),
                        "authors": paper.get('authors', []),
                        "doi": paper.get('doi', ''),
                        "uploaded_at": datetime.now().isoformat()
                    }
                    total_chunks += 1
            
            # Save to files
            self._save_papers(papers)
            self._save_chunks(chunks)
            
            logger.info(f"Successfully inserted {len(papers_data)} papers with {total_chunks} total chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting papers: {str(e)}")
            return False
    
    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using simple text matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            chunks = self._load_chunks()
            results = []
            
            # Simple text matching
            query_words = set(re.findall(r'\w+', query.lower()))
            
            for chunk_id, chunk_data in chunks.items():
                content = chunk_data.get('content', '').lower()
                content_words = set(re.findall(r'\w+', content))
                
                # Calculate simple similarity (word overlap)
                overlap = len(query_words.intersection(content_words))
                if overlap > 0:
                    similarity = overlap / len(query_words)
                    
                    result = {
                        "content": chunk_data.get('content', ''),
                        "title": chunk_data.get('title', 'Unknown'),
                        "authors": chunk_data.get('authors', []),
                        "doi": chunk_data.get('doi', ''),
                        "chunk_index": chunk_data.get('chunk_index', 0),
                        "paper_id": chunk_data.get('paper_id', ''),
                        "similarity": similarity
                    }
                    results.append(result)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching chunks: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        try:
            papers = self._load_papers()
            chunks = self._load_chunks()
            
            return {
                "total_papers": len(papers),
                "total_chunks": len(chunks)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"total_papers": 0, "total_chunks": 0}
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Get all papers with metadata."""
        try:
            papers = self._load_papers()
            return [{"paper_id": pid, **data} for pid, data in papers.items()]
        except Exception as e:
            logger.error(f"Error getting papers: {str(e)}")
            return []

# Global instance
simple_storage = SimplePaperStorage()

def get_simple_storage():
    """Get the simple storage instance."""
    return simple_storage
