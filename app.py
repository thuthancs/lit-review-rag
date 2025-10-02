"""
Streamlit UI for Literature Review RAG system.
This is a basic version for testing integration.
"""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Literature Review RAG",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Literature Review RAG System")
    st.markdown("---")
    
    # Check if configuration is working
    try:
        from config import config
        st.success("✅ Configuration loaded successfully")
        
        # Display config status
        with st.expander("Configuration Status"):
            st.write(f"Collection Name: {config.COLLECTION_NAME}")
            st.write(f"Chunk Size: {config.CHUNK_SIZE}")
            st.write(f"Chunk Overlap: {config.CHUNK_OVERLAP}")
            st.write(f"OpenAI Model: {config.OPENAI_MODEL}")
            
    except Exception as e:
        st.error(f"❌ Configuration error: {str(e)}")
        st.stop()
    
    # Check if Weaviate setup is working
    try:
        from weaviate_setup import setup_weaviate, get_weaviate_collection
        
        if st.button("Test Weaviate Connection"):
            with st.spinner("Connecting to Weaviate..."):
                if setup_weaviate():
                    st.success("✅ Weaviate connection successful!")
                    
                    # Test collection access
                    collection = get_weaviate_collection()
                    if collection:
                        st.success("✅ Collection access successful!")
                    else:
                        st.error("❌ Collection access failed!")
                else:
                    st.error("❌ Weaviate connection failed!")
                    
    except Exception as e:
        st.error(f"❌ Weaviate setup error: {str(e)}")
    
    # Check if RAG queries are working
    try:
        from rag_queries import rag_manager
        
        st.subheader("RAG Functions Test")
        
        # Test search function
        test_query = st.text_input("Test Search Query", value="machine learning")
        
        if st.button("Test Search"):
            with st.spinner("Searching..."):
                results = rag_manager.search_relevant_chunks(test_query, top_k=3)
                if results:
                    st.success(f"✅ Found {len(results)} results")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}: {result.get('title', 'Unknown')}"):
                            st.write(f"**Content:** {result.get('content', '')[:200]}...")
                            st.write(f"**Authors:** {', '.join(result.get('authors', []))}")
                            st.write(f"**DOI:** {result.get('doi', 'N/A')}")
                else:
                    st.warning("⚠️ No results found (this is expected if no papers are uploaded yet)")
        
        # Test gap analysis
        topic = st.text_input("Test Gap Analysis Topic", value="artificial intelligence")
        
        if st.button("Test Gap Analysis"):
            with st.spinner("Generating gap analysis..."):
                result = rag_manager.generate_gap_analysis(topic)
                if result["success"]:
                    st.success("✅ Gap analysis generated successfully!")
                    st.write("**Analysis:**")
                    st.write(result["analysis"])
                else:
                    st.warning(f"⚠️ Gap analysis failed: {result.get('error', 'Unknown error')}")
        
        # Test chat function
        question = st.text_input("Test Chat Question", value="What are the main challenges?")
        
        if st.button("Test Chat"):
            with st.spinner("Generating answer..."):
                result = rag_manager.chat_with_papers(question)
                if result["success"]:
                    st.success("✅ Chat response generated successfully!")
                    st.write("**Answer:**")
                    st.write(result["answer"])
                else:
                    st.warning(f"⚠️ Chat failed: {result.get('error', 'Unknown error')}")
                    
    except Exception as e:
        st.error(f"❌ RAG queries error: {str(e)}")
    
    # Instructions
    st.markdown("---")
    st.subheader("📋 Next Steps")
    st.markdown("""
    This is a basic integration test. To complete the system:
    
    1. **PDF Processor**: Developer 2 needs to implement `pdf_processor.py`
    2. **Full UI**: Developer 2 needs to implement the complete Streamlit interface
    3. **Environment Setup**: Create a `.env` file with your API keys
    4. **Upload Papers**: Use the PDF processor to upload research papers
    
    **Required Environment Variables:**
    ```
    WEAVIATE_URL=your_weaviate_url
    WEAVIATE_API_KEY=your_weaviate_api_key
    OPENAI_API_KEY=your_openai_api_key
    ```
    """)

if __name__ == "__main__":
    main()
