# Literature Review RAG MVP - Task Breakdown

## Overview
90-minute MVP for literature review assistant using Weaviate RAG system.

## Project Structure
```
literature_rag/
├── config.py          # API keys, constants
├── pdf_processor.py   # Extract & chunk PDFs
├── weaviate_setup.py  # Collection creation
├── rag_queries.py     # Gap analysis & chat functions
└── app.py            # Streamlit UI
```

## Task Breakdown

### Phase 1: Foundation Setup (15 minutes)

#### Task 1A: Configuration Setup 
**Assigned to: Developer 1**
**Time: 15 minutes**

**Prompt for AI:**
```
Create a config.py file for a literature review RAG system with Weaviate. Include:
- Environment variable loading for API keys (Weaviate, OpenAI)
- Configuration constants for chunk sizes, overlap, collection names
- Error handling for missing environment variables
- Use python-dotenv for environment management
```

**Deliverable:** `config.py` with environment management

---

### Phase 2: Core Infrastructure (30 minutes)

#### Task 2A: Weaviate Setup & Collection Creation
**Assigned to: Developer 1** 
**Time: 30 minutes**

**Prompt for AI:**
```
Create weaviate_setup.py that:
1. Connects to Weaviate cloud instance using config
2. Creates a collection for research papers with schema:
   - title (text)
   - authors (text[])
   - abstract (text)
   - content_chunks (text[])
   - publication_year (int)
   - doi (text)
   - chunk_index (int)
3. Configures vectorizer (text2vec-openai)
4. Includes functions to check if collection exists and create if needed
5. Add error handling for connection issues
```

**Deliverable:** `weaviate_setup.py` with collection management

#### Task 2B: PDF Processing Pipeline
**Assigned to: Developer 2**
**Time: 30 minutes**

**Prompt for AI:**
```
Create pdf_processor.py that:
1. Extracts text from PDF files using PyPDF2 or pdfplumber
2. Splits text into semantic chunks (200-300 words with 50-word overlap)
3. Extracts metadata (title, authors, abstract if available)
4. Returns structured data for Weaviate insertion
5. Handles multiple PDFs in a directory
6. Includes error handling for corrupted PDFs
7. Uses sliding window approach for chunking
```

**Deliverable:** `pdf_processor.py` with text extraction and chunking

---

### Phase 3: RAG Implementation (30 minutes)

#### Task 3A: RAG Query Functions
**Assigned to: Developer 1**
**Time: 30 minutes**

**Prompt for AI:**
```
Create rag_queries.py with:
1. Function to insert processed papers into Weaviate
2. Function for gap analysis using grouped_task RAG:
   - Search for papers on specific topic
   - Generate study gaps and research opportunities
   - Use prompt: "Based on these research papers, identify key research gaps and future opportunities. Focus on methodological limitations, unexplored areas, and conflicting findings."
3. Function for chat with papers:
   - Semantic search for relevant chunks
   - Generate answers with citations
   - Use prompt: "Answer the question based on the provided research papers. Include specific citations and page references when possible."
4. Include error handling and result formatting
```

**Deliverable:** `rag_queries.py` with gap analysis and chat functions

#### Task 3B: Streamlit UI Development
**Assigned to: Developer 2**
**Time: 30 minutes**

**Prompt for AI:**
```
Create app.py with Streamlit interface:
1. File upload section for PDF papers
2. Two main tabs:
   - "Gap Analysis": Input topic, generate research gaps
   - "Chat with Papers": Q&A interface with papers
3. Display results with proper formatting
4. Show loading states during processing
5. Include sidebar for settings (chunk size, etc.)
6. Add error handling and user feedback
7. Use st.session_state for conversation history
```

**Deliverable:** `app.py` with complete UI

---

### Phase 4: Integration & Testing (15 minutes)

#### Task 4A: Integration Testing
**Assigned to: Both Developers**
**Time: 15 minutes**

**Tasks:**
- Test end-to-end workflow
- Fix any integration issues
- Add requirements.txt
- Test with sample PDFs
- Document usage instructions

---

## Parallel Work Strategy

### Developer 1 Timeline:
1. **0-15 min:** Config setup
2. **15-45 min:** Weaviate setup
3. **45-75 min:** RAG queries
4. **75-90 min:** Integration testing

### Developer 2 Timeline:
1. **0-30 min:** PDF processing
2. **30-60 min:** Streamlit UI
3. **60-75 min:** UI integration with RAG functions
4. **75-90 min:** Integration testing

## Key Dependencies & Handoffs

### Critical Integration Points:
1. **Config → All modules:** Environment variables and constants
2. **PDF Processor → RAG Queries:** Data format for Weaviate insertion
3. **Weaviate Setup → RAG Queries:** Collection schema and connection
4. **RAG Queries → Streamlit UI:** Function interfaces and return formats

### Communication Checkpoints:
- **15 min:** Share config structure
- **45 min:** Align on data formats between PDF processor and RAG
- **75 min:** Test integration between UI and RAG functions

## Success Criteria for MVP
- [ ] Upload and process PDF papers
- [ ] Generate research gaps for a given topic
- [ ] Chat interface with paper content
- [ ] Proper citations and references
- [ ] Error handling and user feedback
- [ ] Clean, functional UI

## Required Dependencies
```python
# requirements.txt
streamlit
weaviate-client
openai
python-dotenv
PyPDF2
pdfplumber
numpy
```

## Environment Variables Needed
Create a `.env` file with:
```
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Quick Start Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
