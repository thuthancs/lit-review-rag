"""
RAG query functions for Literature Review system.
Handles paper insertion, gap analysis, and chat functionality.
"""

from openai import OpenAI
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime

from config import config
from weaviate_setup import get_weaviate_collection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with new API format
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

class RAGQueryManager:
    """Manages RAG operations for literature review."""
    
    def __init__(self):
        """Initialize RAG query manager."""
        self.collection = None
    
    def _get_collection(self):
        """Get the Weaviate collection, connecting if necessary."""
        if not self.collection:
            from weaviate_setup import weaviate_manager
            if not weaviate_manager.connect():
                raise ValueError("Failed to setup Weaviate connection")
            self.collection = weaviate_manager.get_collection()
            if self.collection is None:
                raise ValueError("Weaviate collection not available. Please run setup first.")
            logger.info(f"Collection retrieved successfully: {type(self.collection)}")
        return self.collection
    
    def insert_papers(self, papers_data: List[Dict[str, Any]]) -> bool:
        """
        Insert processed papers into Weaviate collection.
        
        Args:
            papers_data: List of paper dictionaries with chunks and metadata
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            total_chunks = 0
            
            for paper in papers_data:
                paper_id = str(uuid.uuid4())
                
                # Insert each chunk as a separate object
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
                    
                    # Insert into Weaviate with error handling
                    try:
                        self._get_collection().data.insert(chunk_data)
                        total_chunks += 1
                        logger.info(f"Inserted chunk {chunk_idx + 1} for paper: {paper.get('title', 'Unknown')}")
                    except Exception as insert_error:
                        error_msg = str(insert_error)
                        logger.error(f"Failed to insert chunk {chunk_idx + 1}: {error_msg}")
                        
                        # Handle OpenAI quota issues specifically
                        if "429" in error_msg or "quota" in error_msg.lower():
                            logger.error("OpenAI API quota exceeded during paper insertion. Cannot vectorize content.")
                            raise ValueError("OpenAI API quota exceeded. Cannot upload papers without vectorization.")
                        else:
                            # For other errors, continue with next chunk
                            logger.warning(f"Skipping chunk {chunk_idx + 1} due to error")
            
            logger.info(f"Successfully inserted {len(papers_data)} papers with {total_chunks} total chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting papers: {str(e)}")
            return False
    
    def search_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant text chunks using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
            
        try:
            # Perform semantic search with error handling for OpenAI quota issues
            response = self._get_collection().query.near_text(
                query=query,
                limit=top_k,
                return_metadata=["distance", "certainty"]
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
            
            logger.info(f"Found {len(results)} relevant chunks for query: {query}")
            return results
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error searching chunks: {error_msg}")
            
            # Handle specific OpenAI quota exceeded error
            if "429" in error_msg or "quota" in error_msg.lower():
                logger.error("OpenAI API quota exceeded. Please check your OpenAI billing and usage limits.")
                return []
            
            return []
    
    def generate_gap_analysis(self, topic: str) -> Dict[str, Any]:
        """
        Generate research gap analysis for a given topic.
        
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
            
            # Prepare context from relevant chunks
            context = self._prepare_context(relevant_chunks)
            
            # Generate gap analysis using OpenAI
            prompt = f"""
Based on these research papers about "{topic}", identify key research gaps and future opportunities. 
Focus on methodological limitations, unexplored areas, and conflicting findings.

Research Papers Context:
{context}

Please provide:
1. Key research gaps (3-5 specific gaps)
2. Future research opportunities (3-5 opportunities)
3. Methodological limitations identified
4. Areas that need more investigation

Format your response as a structured analysis.
"""
            
            response = openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research analyst expert in identifying research gaps and opportunities."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse the response (simple parsing - could be enhanced)
            gaps = self._extract_gaps_from_analysis(analysis_text)
            opportunities = self._extract_opportunities_from_analysis(analysis_text)
            
            return {
                "success": True,
                "topic": topic,
                "analysis": analysis_text,
                "gaps": gaps,
                "opportunities": opportunities,
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
        Chat with papers using RAG to answer questions.
        
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
            
            # Prepare context
            context = self._prepare_context(relevant_chunks)
            
            # Build conversation context
            messages = [
                {"role": "system", "content": "You are a research assistant. Answer questions based on the provided research papers. Include specific citations and references when possible."}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-4:]:  # Keep last 4 messages for context
                    messages.append(msg)
            
            # Add current question with context
            prompt = f"""
Based on the following research papers, answer the question: {question}

Research Papers Context:
{context}

Please provide a comprehensive answer with specific citations when possible.
"""
            
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            answer = response.choices[0].message.content
            
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
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context string from relevant chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Unknown")
            authors = ", ".join(chunk.get("authors", []))
            content = chunk.get("content", "")
            doi = chunk.get("doi", "")
            
            context_part = f"""
Paper {i}: {title}
Authors: {authors}
DOI: {doi}
Content: {content}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
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
    
    def _extract_gaps_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract research gaps from analysis text (simple parsing)."""
        # Simple extraction - could be enhanced with more sophisticated parsing
        lines = analysis_text.split('\n')
        gaps = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['gap', 'limitation', 'missing', 'unexplored']):
                if line and not line.startswith('#') and len(line) > 20:
                    gaps.append(line)
        
        return gaps[:5]  # Return top 5 gaps
    
    def _extract_opportunities_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract research opportunities from analysis text (simple parsing)."""
        # Simple extraction - could be enhanced with more sophisticated parsing
        lines = analysis_text.split('\n')
        opportunities = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['opportunity', 'future', 'recommend', 'suggest']):
                if line and not line.startswith('#') and len(line) > 20:
                    opportunities.append(line)
        
        return opportunities[:5]  # Return top 5 opportunities

# Global instance - will be created when first accessed
rag_manager = None

def get_rag_manager():
    """Get the global RAG manager instance, creating it if necessary."""
    global rag_manager
    if rag_manager is None:
        rag_manager = RAGQueryManager()
    return rag_manager

def insert_papers(papers_data: List[Dict[str, Any]]) -> bool:
    """Convenience function to insert papers."""
    return get_rag_manager().insert_papers(papers_data)

def generate_gap_analysis(topic: str) -> Dict[str, Any]:
    """Convenience function to generate gap analysis."""
    return get_rag_manager().generate_gap_analysis(topic)

def chat_with_papers(question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """Convenience function to chat with papers."""
    return get_rag_manager().chat_with_papers(question, conversation_history)

if __name__ == "__main__":
    # Test the RAG functions
    print("Testing RAG functions...")
    
    manager = get_rag_manager()
    
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
    
    print("RAG testing completed!")
