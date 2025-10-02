"""
Mock RAG query functions that bypass OpenAI quota issues.
Uses mock vectorization and simple text matching for testing.
"""

from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime

from config import config
from weaviate_setup_mock import mock_weaviate_manager, get_mock_weaviate_collection
from mock_vectorizer import mock_vectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockRAGQueryManager:
    """Mock RAG manager that works without OpenAI."""
    
    def __init__(self):
        """Initialize mock RAG query manager."""
        self.collection = None
    
    def _get_collection(self):
        """Get the Weaviate collection."""
        if not self.collection:
            if not mock_weaviate_manager.connect():
                raise ValueError("Failed to setup mock Weaviate connection")
            self.collection = mock_weaviate_manager.get_collection()
            if self.collection is None:
                raise ValueError("Mock Weaviate collection not available. Please run setup first.")
            logger.info(f"Mock collection retrieved successfully: {type(self.collection)}")
        return self.collection
    
    def insert_papers(self, papers_data: List[Dict[str, Any]]) -> bool:
        """
        Insert processed papers into Weaviate collection using mock vectors.
        
        Args:
            papers_data: List of paper dictionaries with chunks and metadata
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            total_chunks = 0
            
            for paper in papers_data:
                paper_id = str(uuid.uuid4())
                
                # Insert each chunk as a separate object with mock vectors
                for chunk_idx, chunk_text in enumerate(paper.get('chunks', [])):
                    chunk_data = {
                        "title": paper.get('title', 'Unknown Title'),
                        "authors": paper.get('authors', []),
                        "abstract": paper.get('abstract', ''),
                        "content_chunk": chunk_text,
                        "publication_year": paper.get('publication_year', 0),
                        "doi": paper.get('doi', ''),
                        "chunk_index": chunk_idx,
                        "paper_id": paper_id
                    }
                    
                    # Insert with mock vector
                    success = mock_weaviate_manager.insert_with_mock_vectors(chunk_data)
                    if success:
                        total_chunks += 1
                        logger.info(f"Inserted chunk {chunk_idx + 1} for paper: {paper.get('title', 'Unknown')}")
                    else:
                        logger.warning(f"Failed to insert chunk {chunk_idx + 1}")
            
            logger.info(f"Successfully inserted {len(papers_data)} papers with {total_chunks} total chunks (mock mode)")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting papers: {str(e)}")
            return False
    
    def search_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant text chunks using mock vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
            
        try:
            collection = self._get_collection()
            
            # Generate mock vector for query
            query_vector = mock_vectorizer.vectorize(query)
            
            # Perform vector search
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=["distance"]
            )
            
            results = []
            for obj in response.objects:
                chunk_data = {
                    "content": obj.properties["content_chunk"],
                    "title": obj.properties["title"],
                    "authors": obj.properties["authors"],
                    "doi": obj.properties["doi"],
                    "chunk_index": obj.properties["chunk_index"],
                    "paper_id": obj.properties["paper_id"],
                    "similarity": 1 - obj.metadata.distance if obj.metadata.distance else 0
                }
                results.append(chunk_data)
            
            logger.info(f"Found {len(results)} relevant chunks for query: {query} (mock mode)")
            return results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {str(e)}")
            return []
    
    def generate_gap_analysis(self, topic: str) -> Dict[str, Any]:
        """
        Generate research gap analysis using simple text analysis (no OpenAI).
        
        Args:
            topic: Research topic to analyze
            
        Returns:
            Dictionary with gap analysis results
        """
        try:
            # Search for relevant papers
            relevant_chunks = self.search_relevant_chunks(topic, top_k=10)
            
            if not relevant_chunks:
                return {
                    "success": False,
                    "error": "No relevant papers found for the given topic",
                    "gaps": [],
                    "opportunities": []
                }
            
            # Simple analysis without OpenAI
            analysis_text = f"""
# Research Gap Analysis for "{topic}"

Based on the {len(relevant_chunks)} relevant text chunks found in your research papers, here's a basic analysis:

## Key Research Areas Identified:
{self._extract_key_terms(relevant_chunks)}

## Potential Research Gaps:
1. **Methodological Limitations**: The papers may have limitations in their experimental design or data collection methods.
2. **Limited Scope**: Research might be focused on specific populations or contexts that could be expanded.
3. **Technology Gaps**: There may be opportunities to apply newer technologies or approaches.
4. **Cross-disciplinary Integration**: Limited integration with other fields of study.
5. **Long-term Studies**: Most research appears to be short-term; longitudinal studies could be beneficial.

## Future Research Opportunities:
1. **Expanded Methodology**: Apply different research methods to validate findings.
2. **Broader Applications**: Extend research to different domains or populations.
3. **Technology Integration**: Incorporate emerging technologies and tools.
4. **Collaborative Research**: Multi-institutional and cross-disciplinary collaborations.
5. **Real-world Implementation**: Focus on practical applications and implementation studies.

## Note:
This analysis is based on simple text matching and keyword extraction. For more sophisticated analysis, OpenAI API access is required.
"""

            return {
                "success": True,
                "topic": topic,
                "analysis": analysis_text,
                "gaps": ["Methodological limitations", "Limited scope", "Technology gaps"],
                "opportunities": ["Expanded methodology", "Broader applications", "Technology integration"],
                "sources": self._format_sources(relevant_chunks),
                "num_papers": len(set(chunk["paper_id"] for chunk in relevant_chunks))
            }
            
        except Exception as e:
            logger.error(f"Error generating gap analysis: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating analysis: {str(e)}",
                "gaps": [],
                "opportunities": []
            }
    
    def chat_with_papers(self, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Chat with papers using simple text matching (no OpenAI).
        
        Args:
            question: User's question
            conversation_history: Previous conversation context
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            # Search for relevant chunks
            relevant_chunks = self.search_relevant_chunks(question, top_k=8)
            
            if not relevant_chunks:
                return {
                    "success": False,
                    "error": "No relevant information found in the papers",
                    "answer": "",
                    "sources": []
                }
            
            # Simple answer generation without OpenAI
            answer = f"""
Based on the {len(relevant_chunks)} relevant text chunks found in your research papers, here's what I found regarding your question: "{question}"

## Relevant Information:

{self._format_chunks_for_answer(relevant_chunks)}

## Summary:
The research papers contain relevant information about your question. The above excerpts provide context and details related to your query.

**Note**: This is a basic text-matching response. For more sophisticated analysis and natural language responses, OpenAI API access is required.
"""

            return {
                "success": True,
                "question": question,
                "answer": answer,
                "sources": self._format_sources(relevant_chunks),
                "num_sources": len(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in chat with papers: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating answer: {str(e)}",
                "answer": "",
                "sources": []
            }
    
    def _extract_key_terms(self, chunks: List[Dict[str, Any]]) -> str:
        """Extract key terms from chunks."""
        terms = []
        for chunk in chunks[:5]:  # Use first 5 chunks
            content = chunk.get("content", "")
            # Simple keyword extraction
            words = content.split()
            key_words = [word for word in words if len(word) > 4 and word.isalpha()]
            terms.extend(key_words[:3])  # Take first 3 words from each chunk
        
        return ", ".join(list(set(terms))[:10])  # Return unique terms
    
    def _format_chunks_for_answer(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for display in answer."""
        formatted = []
        for i, chunk in enumerate(chunks[:3], 1):  # Use first 3 chunks
            title = chunk.get("title", "Unknown")
            content = chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", "")
            formatted.append(f"{i}. **{title}**: {content}")
        
        return "\n\n".join(formatted)
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format sources for display."""
        sources = []
        seen_papers = set()
        
        for chunk in chunks:
            paper_id = chunk.get("paper_id")
            if paper_id not in seen_papers:
                source = {
                    "title": chunk.get("title", "Unknown"),
                    "authors": ", ".join(chunk.get("authors", [])),
                    "doi": chunk.get("doi", ""),
                    "chunk_index": chunk.get("chunk_index", 0)
                }
                sources.append(source)
                seen_papers.add(paper_id)
        
        return sources

# Global instance
mock_rag_manager = None

def get_mock_rag_manager():
    """Get the global mock RAG manager instance."""
    global mock_rag_manager
    if mock_rag_manager is None:
        mock_rag_manager = MockRAGQueryManager()
    return mock_rag_manager

def insert_papers_mock(papers_data: List[Dict[str, Any]]) -> bool:
    """Convenience function to insert papers with mock vectors."""
    return get_mock_rag_manager().insert_papers(papers_data)

def generate_gap_analysis_mock(topic: str) -> Dict[str, Any]:
    """Convenience function to generate gap analysis without OpenAI."""
    return get_mock_rag_manager().generate_gap_analysis(topic)

def chat_with_papers_mock(question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """Convenience function to chat with papers without OpenAI."""
    return get_mock_rag_manager().chat_with_papers(question, conversation_history)

if __name__ == "__main__":
    # Test the mock RAG functions
    print("Testing mock RAG functions...")
    
    manager = get_mock_rag_manager()
    
    # Test search
    test_query = "machine learning applications"
    results = manager.search_relevant_chunks(test_query, top_k=3)
    print(f"✅ Search test: Found {len(results)} results")
    
    # Test gap analysis
    gap_result = manager.generate_gap_analysis("artificial intelligence")
    if gap_result["success"]:
        print("✅ Gap analysis test: Success")
    else:
        print(f"❌ Gap analysis test: {gap_result.get('error', 'Unknown error')}")
    
    # Test chat
    chat_result = manager.chat_with_papers("What are the main challenges in AI?")
    if chat_result["success"]:
        print("✅ Chat test: Success")
    else:
        print(f"❌ Chat test: {chat_result.get('error', 'Unknown error')}")
    
    print("Mock RAG testing completed!")
