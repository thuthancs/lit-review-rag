"""
Simple RAG query functions using file-based storage.
Completely bypasses Weaviate and OpenAI for testing.
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from simple_storage import get_simple_storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAGQueryManager:
    """Simple RAG manager using file storage."""
    
    def __init__(self):
        """Initialize simple RAG query manager."""
        self.storage = get_simple_storage()
    
    def insert_papers(self, papers_data: List[Dict[str, Any]]) -> bool:
        """
        Insert processed papers into simple storage.
        
        Args:
            papers_data: List of paper dictionaries with chunks and metadata
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            return self.storage.insert_papers(papers_data)
        except Exception as e:
            logger.error(f"Error inserting papers: {str(e)}")
            return False
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant text chunks using simple text matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            return self.storage.search_chunks(query, top_k)
        except Exception as e:
            logger.error(f"Error searching chunks: {str(e)}")
            return []
    
    def generate_gap_analysis(self, topic: str) -> Dict[str, Any]:
        """
        Generate research gap analysis using simple text analysis.
        
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
            
            # Simple analysis
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
This analysis is based on simple text matching and keyword extraction. For more sophisticated analysis, AI-powered tools would be required.
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
        Chat with papers using simple text matching.
        
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
            
            # Simple answer generation
            answer = f"""
Based on the {len(relevant_chunks)} relevant text chunks found in your research papers, here's what I found regarding your question: "{question}"

## Relevant Information:

{self._format_chunks_for_answer(relevant_chunks)}

## Summary:
The research papers contain relevant information about your question. The above excerpts provide context and details related to your query.

**Note**: This is a basic text-matching response. For more sophisticated analysis and natural language responses, AI-powered tools would be required.
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
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        return self.storage.get_collection_stats()
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Get all papers with metadata."""
        return self.storage.get_all_papers()
    
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
simple_rag_manager = None

def get_simple_rag_manager():
    """Get the global simple RAG manager instance."""
    global simple_rag_manager
    if simple_rag_manager is None:
        simple_rag_manager = SimpleRAGQueryManager()
    return simple_rag_manager

def insert_papers_simple(papers_data: List[Dict[str, Any]]) -> bool:
    """Convenience function to insert papers into simple storage."""
    return get_simple_rag_manager().insert_papers(papers_data)

def generate_gap_analysis_simple(topic: str) -> Dict[str, Any]:
    """Convenience function to generate gap analysis."""
    return get_simple_rag_manager().generate_gap_analysis(topic)

def chat_with_papers_simple(question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """Convenience function to chat with papers."""
    return get_simple_rag_manager().chat_with_papers(question, conversation_history)

if __name__ == "__main__":
    # Test the simple RAG functions
    print("Testing simple RAG functions...")
    
    manager = get_simple_rag_manager()
    
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
    
    print("Simple RAG testing completed!")
