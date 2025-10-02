"""
Streamlit app for Literature Review RAG Assistant
Run with: streamlit run app.py
"""

import streamlit as st
from weaviate_setup import (
    get_client,
    import_chunks,
    get_collection_stats,
    create_collection,
)
from pdf_processor import PDFProcessor
from rag_queries import (
    find_research_gaps,
    chat_with_papers,
    get_papers_summary,
    search_specific_topic,
)
import tempfile
import os
from datetime import datetime
from pathlib import Path


# Page config
st.set_page_config(
    page_title="Literature Review Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .paper-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "client" not in st.session_state:
        st.session_state.client = None
        st.session_state.connected = False
        st.session_state.connection_error = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processed_papers" not in st.session_state:
        st.session_state.processed_papers = []

    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 250

    if "overlap_size" not in st.session_state:
        st.session_state.overlap_size = 50

    if "search_limit" not in st.session_state:
        st.session_state.search_limit = 8

    if "last_gap_analysis" not in st.session_state:
        st.session_state.last_gap_analysis = None

    if "collection_initialized" not in st.session_state:
        st.session_state.collection_initialized = False


def connect_to_weaviate():
    """Establish connection to Weaviate"""
    try:
        if not st.session_state.connected:
            with st.spinner("Connecting to Weaviate..."):
                client = get_client()
                st.session_state.client = client
                st.session_state.connected = True
                st.session_state.connection_error = None

                # Initialize collection if needed
                if not st.session_state.collection_initialized:
                    try:
                        create_collection(client)
                        st.session_state.collection_initialized = True
                    except Exception as e:
                        # Collection might already exist
                        st.session_state.collection_initialized = True

                return True
    except Exception as e:
        st.session_state.connection_error = str(e)
        st.session_state.connected = False
        return False


def display_connection_status():
    """Display connection status in sidebar"""
    st.sidebar.markdown("### 🔌 Connection Status")
    if st.session_state.connected:
        st.sidebar.success("✓ Connected to Weaviate")
    else:
        st.sidebar.error("✗ Not connected")
        if st.session_state.connection_error:
            with st.sidebar.expander("Error Details"):
                st.code(st.session_state.connection_error)
        if st.sidebar.button("Retry Connection"):
            connect_to_weaviate()
            st.rerun()


def upload_papers_section():
    """Handle PDF upload and processing"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Upload Papers")

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 3-10 papers for best results",
        key="pdf_uploader",
    )

    # Manual metadata input (optional)
    with st.sidebar.expander("📝 Add Metadata (Optional)"):
        st.markdown("*Leave blank for auto-detection*")
        manual_title = st.text_input("Paper Title", key="manual_title")
        manual_authors = st.text_input("Authors", key="manual_authors")
        manual_year = st.number_input("Year", 1990, 2025, 2024, key="manual_year")

    if uploaded_files:
        st.sidebar.info(f"📎 {len(uploaded_files)} file(s) selected")

        if st.sidebar.button(
            "🚀 Process & Upload Papers", type="primary", use_container_width=True
        ):
            if not st.session_state.connected:
                st.sidebar.error("Please connect to Weaviate first!")
                return

            process_uploaded_papers(
                uploaded_files,
                manual_title if manual_title else None,
                manual_authors if manual_authors else None,
                manual_year,
            )


def process_uploaded_papers(uploaded_files, manual_title, manual_authors, manual_year):
    """Process and upload papers to Weaviate"""
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    processor = PDFProcessor(
        chunk_size=st.session_state.chunk_size,
        overlap_size=st.session_state.overlap_size,
    )

    total_chunks = 0
    successful = 0
    failed = []

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            progress = (idx) / len(uploaded_files)
            progress_bar.progress(progress)

            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Prepare metadata
            metadata = None
            if manual_title or manual_authors:
                metadata = {
                    "title": manual_title or uploaded_file.name.replace(".pdf", ""),
                    "authors": manual_authors or "Unknown",
                    "year": str(manual_year),
                }

            # Process paper
            chunks = processor.process_paper(tmp_path, metadata)

            # Import to Weaviate
            import_chunks(st.session_state.client, chunks)

            # Track success
            total_chunks += len(chunks)
            successful += 1
            st.session_state.processed_papers.append(
                {
                    "name": uploaded_file.name,
                    "chunks": len(chunks),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            # Clean up
            os.unlink(tmp_path)

        except Exception as e:
            failed.append((uploaded_file.name, str(e)))
            st.sidebar.error(f"Failed: {uploaded_file.name}")

    # Complete progress
    progress_bar.progress(1.0)
    status_text.empty()
    progress_bar.empty()

    # Show results
    if successful > 0:
        st.sidebar.success(
            f"✅ Successfully uploaded {successful}/{len(uploaded_files)} papers!"
        )
        st.sidebar.info(f"📊 Total chunks created: {total_chunks}")

    if failed:
        with st.sidebar.expander(f"⚠️ {len(failed)} file(s) failed"):
            for filename, error in failed:
                st.write(f"**{filename}**")
                st.code(error, language=None)

    # Rerun to update stats
    st.rerun()


def display_collection_stats():
    """Display collection statistics"""
    if not st.session_state.connected:
        return

    try:
        stats = get_collection_stats(st.session_state.client)
        papers = get_papers_summary(st.session_state.client)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Collection Stats")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Papers", len(papers))
        with col2:
            st.metric("Chunks", stats["total_chunks"])

        # Show paper list
        if papers:
            with st.sidebar.expander("📚 View All Papers"):
                for paper in papers:
                    st.markdown(
                        f"""
                    <div class="paper-card">
                        <strong>{paper['title'][:60]}...</strong><br/>
                        <small>{paper['authors']} ({paper['year']})</small><br/>
                        <small>Chunks: {paper['chunk_count']}</small>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.sidebar.error(f"Error loading stats: {e}")


def settings_section():
    """Display settings in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")

    with st.sidebar.expander("Chunking Settings"):
        st.session_state.chunk_size = st.slider(
            "Chunk Size (words)",
            100,
            400,
            st.session_state.chunk_size,
            help="Number of words per chunk",
        )
        st.session_state.overlap_size = st.slider(
            "Overlap Size (words)",
            0,
            100,
            st.session_state.overlap_size,
            help="Word overlap between chunks",
        )

    with st.sidebar.expander("Search Settings"):
        st.session_state.search_limit = st.slider(
            "Search Results",
            3,
            20,
            st.session_state.search_limit,
            help="Number of chunks to retrieve",
        )


def gap_analysis_tab():
    """Gap Analysis tab content"""
    st.markdown("### 🔍 Research Gap Analysis")
    st.markdown("*Identify unexplored areas and opportunities in your literature*")

    if not st.session_state.connected:
        st.warning("⚠️ Please connect to Weaviate first!")
        return

    # Check if papers exist
    try:
        stats = get_collection_stats(st.session_state.client)
        if stats["total_chunks"] == 0:
            st.info(
                "📚 Please upload some research papers first to perform gap analysis."
            )
            return
    except:
        st.error("Error checking collection status")
        return

    # Input section
    col1, col2 = st.columns([3, 1])

    with col1:
        focus_area = st.text_input(
            "🎯 Focus Area (optional)",
            placeholder="e.g., methodology, population, theoretical framework",
            help="Narrow the analysis to a specific aspect",
            key="gap_focus",
        )

    with col2:
        num_chunks = st.number_input(
            "Chunks to analyze",
            min_value=5,
            max_value=20,
            value=st.session_state.search_limit,
            help="More chunks = broader analysis",
        )

    # Analysis button
    if st.button("🔍 Analyze Research Gaps", type="primary", use_container_width=True):
        with st.spinner("🤔 Analyzing papers for research gaps..."):
            try:
                gaps = find_research_gaps(
                    st.session_state.client,
                    focus_area if focus_area else None,
                    limit=num_chunks,
                )

                st.session_state.last_gap_analysis = {
                    "gaps": gaps,
                    "focus": focus_area,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                return

    # Display results
    if st.session_state.last_gap_analysis:
        st.markdown("---")
        st.success("✅ Analysis Complete!")

        # Metadata
        analysis = st.session_state.last_gap_analysis
        col1, col2 = st.columns([2, 1])
        with col1:
            if analysis["focus"]:
                st.markdown(f"**Focus Area:** {analysis['focus']}")
        with col2:
            st.markdown(f"**Generated:** {analysis['timestamp']}")

        # Results
        st.markdown("### 📋 Identified Research Gaps")
        st.markdown(analysis["gaps"])

        # Download option
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="📥 Download as TXT",
                data=analysis["gaps"],
                file_name=f"research_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.last_gap_analysis = None
                st.rerun()


def chat_tab():
    """Chat with Papers tab content"""
    st.markdown("### 💬 Chat with Your Papers")
    st.markdown("*Ask questions and get answers grounded in your research papers*")

    if not st.session_state.connected:
        st.warning("⚠️ Please connect to Weaviate first!")
        return

    # Check if papers exist
    try:
        stats = get_collection_stats(st.session_state.client)
        if stats["total_chunks"] == 0:
            st.info("📚 Please upload some research papers first to start chatting.")
            return
    except:
        st.error("Error checking collection status")
        return

    # Chat history display
    st.markdown("### 📜 Conversation History")
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(
                    f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br/>
                    {msg['content']}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong><br/>
                    {msg['content']}
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("💡 No conversation yet. Ask a question below to get started!")

    st.markdown("---")

    # Input section
    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.text_input(
            "🤔 Ask a question:",
            placeholder="e.g., What methodologies are most commonly used? What are the main findings?",
            key="chat_question",
        )

    with col2:
        num_sources = st.number_input(
            "Sources",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of chunks to use as context",
        )

    # Send button
    if st.button("📤 Send Question", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("⚠️ Please enter a question")
            return

        # Add user message to history
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )

        # Get answer
        with st.spinner("🔍 Searching papers and generating answer..."):
            try:
                answer = chat_with_papers(
                    st.session_state.client, question, limit=num_sources
                )

                # Add assistant message to history
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                )

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error getting answer: {e}")
                # Remove user message if failed
                st.session_state.chat_history.pop()


def main():
    """Main application"""
    # Initialize
    init_session_state()

    # Header
    st.markdown(
        '<p class="main-header">📚 Literature Review Assistant</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">AI-powered research gap analysis and paper Q&A using Weaviate RAG</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        # Connection
        if not st.session_state.connected:
            if st.button(
                "🔌 Connect to Weaviate", type="primary", use_container_width=True
            ):
                if connect_to_weaviate():
                    st.success("✅ Connected!")
                    st.rerun()
                else:
                    st.error("❌ Connection failed!")

        display_connection_status()

        # Only show other sections if connected
        if st.session_state.connected:
            upload_papers_section()
            display_collection_stats()
            settings_section()

        # Footer
        st.markdown("---")
        st.markdown(
            """
        <small>
        Built with Weaviate, OpenAI, and Streamlit<br/>
        <a href="https://docs.weaviate.io" target="_blank">📖 Weaviate Docs</a>
        </small>
        """,
            unsafe_allow_html=True,
        )

    # Main content
    if not st.session_state.connected:
        st.info("👆 Please connect to Weaviate using the sidebar to get started!")

        # Show quick start guide
        st.markdown("### 🚀 Quick Start Guide")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            **1. Connect** 🔌
            - Click "Connect to Weaviate"
            - Ensure API keys are set
            """
            )

        with col2:
            st.markdown(
                """
            **2. Upload** 📄
            - Upload 3-10 PDF papers
            - Add metadata (optional)
            - Click "Process & Upload"
            """
            )

        with col3:
            st.markdown(
                """
            **3. Analyze** 🔍
            - Find research gaps
            - Chat with papers
            - Export results
            """
            )

        return

    # Main tabs
    tab1, tab2 = st.tabs(["🔍 Gap Analysis", "💬 Chat with Papers"])

    with tab1:
        gap_analysis_tab()

    with tab2:
        chat_tab()


if __name__ == "__main__":
    main()
