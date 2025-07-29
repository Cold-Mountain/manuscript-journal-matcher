# Manuscript Journal Matcher - Step-by-Step Implementation Plan

**Goal**: Create a working MVP that can match manuscripts to journals through semantic similarity search.

---

## Step 1: Project Foundation & Environment Setup
**Estimated Time**: 30 minutes  
**Prerequisites**: Python 3.8+, basic understanding of virtual environments

### What to do:
1. **Create project structure**
   ```bash
   cd /Users/aryanpathak/manuscript-journal-matcher
   mkdir -p src data tests scripts docs
   mkdir -p data/sample_manuscripts data/api_cache
   mkdir -p tests/fixtures
   ```

2. **Create requirements.txt**
   ```bash
   # Copy the dependencies from README.md into requirements.txt
   # Start with essential packages only
   ```

3. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Create basic configuration files**
   - `.gitignore` (Python-specific)
   - `.env.example` (environment variables template)
   - `src/__init__.py` and `tests/__init__.py` (empty files)

### Expected Outcome:
- Clean project structure
- Working Python environment with dependencies installed
- Ready for development

### Next person reads: Project structure and requirements.txt to understand dependencies

---

## Step 2: Document Extraction Engine
**Estimated Time**: 2-3 hours  
**Prerequisites**: Completed Step 1, sample PDF/DOCX files for testing

### What to do:
1. **Create `src/extractor.py`**
   - Implement PDF text extraction using `pdfplumber`
   - Implement DOCX text extraction using `python-docx`
   - Create functions to detect and extract title and abstract
   - Handle common document formats and edge cases

2. **Key functions to implement**:
   ```python
   def extract_text_from_pdf(file_path: str) -> str:
       """Extract all text from PDF file"""
   
   def extract_text_from_docx(file_path: str) -> str:
       """Extract all text from DOCX file"""
   
   def extract_title_and_abstract(text: str) -> tuple[str, str]:
       """Extract title and abstract from document text"""
   
   def extract_manuscript_data(file_path: str) -> dict:
       """Main function to extract metadata from any supported file"""
   ```

3. **Create test files**
   - Add sample PDF and DOCX files to `data/sample_manuscripts/`
   - Create `tests/test_extractor.py` with basic tests

4. **Test extraction quality**
   - Verify title extraction accuracy
   - Verify abstract extraction accuracy
   - Handle documents without clear abstracts

### Expected Outcome:
- Working document processing pipeline
- Reliable title and abstract extraction
- Error handling for malformed files

### Next person reads: `src/extractor.py` to understand text extraction logic and available functions

---

## Step 3: Embedding System
**Estimated Time**: 1-2 hours  
**Prerequisites**: Completed Step 2, understanding of text embeddings

### What to do:
1. **Create `src/embedder.py`**
   - Initialize sentence-transformers model (`all-MiniLM-L6-v2`)
   - Implement text embedding functions
   - Add cosine similarity calculations
   - Handle batch processing for efficiency

2. **Key functions to implement**:
   ```python
   def initialize_embedding_model() -> SentenceTransformer:
       """Initialize and return the embedding model"""
   
   def embed_text(text: str, model: SentenceTransformer) -> np.ndarray:
       """Generate embedding for single text"""
   
   def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
       """Generate embeddings for multiple texts"""
   
   def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
       """Calculate cosine similarity between two vectors"""
   ```

3. **Create `src/config.py`**
   - Define configuration constants (model name, embedding dimensions, etc.)
   - Environment variable handling
   - Default settings

4. **Test embedding functionality**
   - Create `tests/test_embedder.py`
   - Test embedding generation consistency
   - Test similarity calculations

### Expected Outcome:
- Working embedding pipeline
- Consistent vector representations
- Efficient batch processing capability

### Next person reads: `src/embedder.py` and `src/config.py` to understand embedding model and configuration

---

## Step 4: Journal Database Builder (OpenAlex Integration)
**Estimated Time**: 3-4 hours  
**Prerequisites**: Completed Step 3, basic API knowledge

### What to do:
1. **Create `src/journal_db_builder.py`**
   - Implement OpenAlex API client
   - Create journal data fetching functions
   - Build journal semantic fingerprints
   - Implement data persistence

2. **Key functions to implement**:
   ```python
   def fetch_openalex_journals(limit: int = 1000) -> list[dict]:
       """Fetch journal data from OpenAlex API"""
   
   def create_semantic_fingerprint(journal_data: dict) -> str:
       """Create semantic fingerprint from journal scope and sample articles"""
   
   def build_journal_embeddings(journals: list[dict]) -> tuple[list[dict], np.ndarray]:
       """Generate embeddings for all journals"""
   
   def save_journal_database(journals: list[dict], embeddings: np.ndarray):
       """Save journal data and embeddings to disk"""
   ```

3. **Implement data schema**
   - Define journal data structure (as per README)
   - Handle missing fields gracefully
   - Validate data quality

4. **Create `scripts/build_database.py`**
   - Script to build initial journal database
   - Progress tracking and error handling
   - Resume capability for interrupted builds

5. **Test database building**
   - Start with small dataset (100 journals)
   - Verify data quality and completeness
   - Test embedding generation

### Expected Outcome:
- Working OpenAlex API integration
- Journal database with embeddings
- Persistent storage system

### Next person reads: `src/journal_db_builder.py` and `scripts/build_database.py` to understand data collection and storage

---

## Step 5: Vector Search Implementation
**Estimated Time**: 2-3 hours  
**Prerequisites**: Completed Step 4, FAISS library understanding

### What to do:
1. **Create `src/match_journals.py`**
   - Implement FAISS index creation and loading
   - Create similarity search functions
   - Add result ranking and filtering
   - Handle search result formatting

2. **Key functions to implement**:
   ```python
   def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
       """Create and return FAISS index from embeddings"""
   
   def load_journal_database() -> tuple[list[dict], faiss.Index]:
       """Load journal metadata and FAISS index from disk"""
   
   def search_similar_journals(query_embedding: np.ndarray, index: faiss.Index, 
                              journals: list[dict], top_k: int = 10) -> list[dict]:
       """Find most similar journals to query"""
   
   def rank_and_filter_results(results: list[dict], filters: dict = None) -> list[dict]:
       """Apply additional ranking and filtering to results"""
   ```

3. **Implement search optimization**
   - Efficient index loading
   - Batch search capabilities
   - Memory management for large databases

4. **Create `src/utils.py`**
   - Common utility functions
   - File handling helpers
   - Data validation functions

5. **Test search functionality**
   - Create `tests/test_matching.py`
   - Test search accuracy with known queries
   - Verify ranking quality

### Expected Outcome:
- Fast vector similarity search
- Ranked journal recommendations
- Filtered results based on criteria

### Next person reads: `src/match_journals.py` and `src/utils.py` to understand search logic and utilities

---

## Step 6: Basic Streamlit Interface
**Estimated Time**: 2-3 hours  
**Prerequisites**: Completed Step 5, basic Streamlit knowledge

### What to do:
1. **Create `src/main.py`**
   - Build file upload interface
   - Implement processing workflow
   - Create results display
   - Add basic error handling

2. **Key interface components**:
   ```python
   def main():
       """Main Streamlit application"""
       st.title("Manuscript Journal Matcher")
       
       # File upload section
       uploaded_file = st.file_uploader(...)
       
       # Processing section
       if st.button("Find Matching Journals"):
           process_manuscript(uploaded_file)
       
       # Results display
       display_results(search_results)
   
   def process_manuscript(file) -> list[dict]:
       """Process uploaded file and return journal matches"""
   
   def display_results(results: list[dict]):
       """Display search results in formatted table"""
   ```

3. **Implement core workflow**:
   - File validation and processing
   - Text extraction and embedding
   - Journal search and ranking
   - Results formatting and display

4. **Add user experience features**:
   - Progress indicators
   - Error messages
   - Clear instructions
   - Download functionality

5. **Test end-to-end workflow**
   - Upload sample manuscripts
   - Verify complete pipeline works
   - Check result quality and formatting

### Expected Outcome:
- Working web interface
- Complete manuscript-to-journal matching pipeline
- User-friendly experience

### Next person reads: `src/main.py` to understand the complete application workflow and UI components

---

## Step 7: Testing & Validation
**Estimated Time**: 2-3 hours  
**Prerequisites**: Completed Step 6, pytest knowledge

### What to do:
1. **Expand test suite**
   - Complete unit tests for all modules
   - Add integration tests for end-to-end workflow
   - Create performance benchmarks

2. **Test categories to implement**:
   ```python
   # tests/test_extractor.py - Document processing tests
   # tests/test_embedder.py - Embedding functionality tests  
   # tests/test_matching.py - Search and ranking tests
   # tests/test_integration.py - End-to-end workflow tests
   ```

3. **Create test fixtures**
   - Sample manuscripts with known characteristics
   - Expected extraction results
   - Reference journal matches

4. **Validate system accuracy**
   - Test with real academic papers
   - Verify journal recommendations make sense
   - Check edge cases and error handling

5. **Performance testing**
   - Measure processing times
   - Test with large journal databases
   - Memory usage optimization

### Expected Outcome:
- Comprehensive test coverage
- Validated system accuracy
- Performance benchmarks

### Next person reads: Test files in `tests/` directory to understand system validation and expected behavior

---

## Step 8: DOAJ Integration (Open Access Data)
**Estimated Time**: 2-3 hours  
**Prerequisites**: Completed Step 7, API integration experience

### What to do:
1. **Extend `src/journal_db_builder.py`**
   - Add DOAJ API client functions
   - Implement open access data enrichment
   - Handle API rate limits and errors

2. **New functions to add**:
   ```python
   def fetch_doaj_data(issn: str) -> dict:
       """Fetch open access info from DOAJ for given ISSN"""
   
   def enrich_journals_with_doaj(journals: list[dict]) -> list[dict]:
       """Add DOAJ data to existing journal records"""
   
   def update_journal_database():
       """Update existing database with new DOAJ data"""
   ```

3. **Update database schema**
   - Add open access status fields
   - Include APC (Article Processing Charge) information
   - Handle missing DOAJ data gracefully

4. **Update search and display**
   - Add OA filtering options in search
   - Display APC information in results
   - Update `src/main.py` interface

### Expected Outcome:
- Enhanced journal database with open access information
- OA-based filtering capabilities
- APC cost information for researchers

### Next person reads: Updated `src/journal_db_builder.py` to understand DOAJ integration and enhanced data schema

---

## Step 9: CrossRef Integration & Publisher Data
**Estimated Time**: 2 hours  
**Prerequisites**: Completed Step 8

### What to do:
1. **Add CrossRef integration to `src/journal_db_builder.py`**
   - Implement CrossRef API client
   - Fetch publisher and additional metadata
   - Handle rate limiting (50 requests/second)

2. **New functions to implement**:
   ```python
   def fetch_crossref_data(issn: str) -> dict:
       """Fetch publisher info from CrossRef"""
   
   def enrich_journals_with_crossref(journals: list[dict]) -> list[dict]:
       """Add CrossRef data to journal records"""
   ```

3. **Final database schema**
   - Complete journal records with all metadata
   - Publisher information
   - Subject classifications

4. **Update interface**
   - Display publisher information
   - Add publisher-based filtering
   - Improve results table formatting

### Expected Outcome:
- Complete journal database with comprehensive metadata
- Publisher information for all journals
- Enhanced filtering and display options

### Next person reads: Final version of `src/journal_db_builder.py` to understand complete data integration pipeline

---

## Step 10: Production Readiness & Documentation
**Estimated Time**: 2-3 hours  
**Prerequisites**: Completed Step 9

### What to do:
1. **Create production configuration**
   - Environment variable handling
   - Production-ready settings
   - Error logging and monitoring

2. **Add deployment files**:
   ```bash
   # Create Dockerfile for containerization
   # Create docker-compose.yml for local development
   # Create requirements-prod.txt with pinned versions
   ```

3. **Performance optimization**
   - Database loading optimization
   - Memory usage improvements
   - Caching strategies

4. **Create comprehensive documentation**:
   ```bash
   # docs/user_guide.md - End-user instructions
   # docs/api_documentation.md - Technical API docs
   # docs/deployment_guide.md - Production deployment
   ```

5. **Final testing**
   - Load testing with multiple users
   - Large file processing tests
   - Error recovery testing

### Expected Outcome:
- Production-ready application
- Complete documentation
- Deployment instructions

### Next person reads: Documentation files in `docs/` directory for deployment and usage instructions

---

## Step 11: Optional Enhancements
**Estimated Time**: Variable (1-4 hours each)  
**Prerequisites**: Completed Step 10

Choose one or more enhancements based on priorities:

### A. Advanced Text Processing
- Better title/abstract detection using ML
- Multi-language support
- Reference extraction and analysis

### B. Enhanced UI/UX
- React/Next.js frontend (replace Streamlit)
- Advanced filtering controls
- Batch processing multiple manuscripts

### C. Analytics & Insights
- Usage tracking and analytics
- Journal recommendation accuracy metrics
- User feedback collection

### D. API Development
- REST API for programmatic access
- Authentication and rate limiting
- API documentation with OpenAPI/Swagger

### Expected Outcome:
- Enhanced functionality based on user needs
- Improved user experience
- Extended capabilities

---

## Quick Start Guide for Each Step

### For the Next Developer:
1. **Read the README.md** - Understand overall project goals
2. **Check current step completion** - Look at existing code in `src/` directory
3. **Read the relevant step above** - Follow detailed instructions
4. **Check tests** - Run existing tests to verify current functionality
5. **Implement step** - Follow the "What to do" section
6. **Verify outcome** - Ensure "Expected Outcome" is achieved
7. **Update documentation** - Add comments and update relevant docs

### Code Quality Guidelines:
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for public functions
- Create tests for new functionality
- Handle errors gracefully
- Use meaningful variable and function names

### Common Commands:
```bash
# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run application
streamlit run src/main.py

# Build database
python scripts/build_database.py

# Run with debugging
streamlit run src/main.py --logger.level=debug
```

---

## Troubleshooting Common Issues

### Step 2 (Document Extraction):
- **PDF extraction fails**: Try alternative libraries (PyMuPDF, pdfminer.six)
- **No abstract found**: Check document structure, adjust regex patterns
- **Encoding issues**: Handle different text encodings explicitly

### Step 4 (Database Building):
- **API rate limits**: Implement exponential backoff and retry logic
- **Memory issues**: Process journals in batches, clear unused data
- **Network timeouts**: Add robust error handling and resume capability

### Step 6 (Streamlit Interface):
- **Large file uploads**: Adjust Streamlit file size limits
- **Slow processing**: Add progress bars and async processing
- **Memory leaks**: Ensure proper cleanup of temporary files

### Next Developer Notes:
- Each step builds on previous steps - don't skip unless absolutely necessary
- Test thoroughly at each step before moving to the next
- Keep the README.md updated with any significant changes
- Document any deviations from the original plan

---

---

## 🎉 IMPLEMENTATION STATUS UPDATE

### ✅ CSV Journal Database Integration - COMPLETED (July 28, 2025)

**Major Enhancement**: The system has been successfully upgraded with comprehensive CSV integration capabilities, transforming it from a 10-journal prototype to a production-ready system with 7,000+ medical journals.

#### Completed Components:
1. **CSV Journal Importer** (`src/csv_journal_importer.py`) ✅
   - Processes 7,678 medical journals from Medicine Journal Rankings 2024.csv
   - Handles European decimal format and data cleaning
   - Chunked processing for memory efficiency

2. **Schema Mapper** (`src/csv_schema_mapper.py`) ✅
   - Maps all 24 CSV columns to database schema
   - Enhanced semantic fingerprints with ranking context
   - DOAJ integration compatibility maintained

3. **Data Processor** (`src/csv_data_processor.py`) ✅
   - Quality validation and filtering
   - Duplicate detection and quality reports
   - Configurable filtering criteria

4. **Enhanced Build Script** (`scripts/build_database.py`) ✅
   - CSV processing options added
   - Quality filtering capabilities
   - DOAJ enrichment integration

5. **Comprehensive Test Suite** (`tests/test_csv_integration.py`) ✅
   - Unit and integration tests
   - Performance testing capabilities
   - End-to-end workflow validation

#### Integration Test Results:
- **📊 7,664 journals successfully processed** (from 7,678 raw entries)
- **🏆 2,041 Q1 journals** (highest quality tier)
- **🔝 100 top-ranked journals** available
- **🌍 104 countries** represented
- **✅ All validation tests passed**

#### New Capabilities:
```bash
# Build production database from CSV
python scripts/build_database.py \
    --csv-file "Medicine Journal Rankings 2024.csv" \
    --csv-chunk-size 500 \
    --doaj-rate-limit 1.0

# Build with quality filtering
python scripts/build_database.py \
    --csv-file "Medicine Journal Rankings 2024.csv" \
    --quality-filter \
    --min-h-index 20 \
    --allowed-quartiles Q1 Q2 \
    --max-rank 2000
```

### Next Steps:
- **Step 11 Enhancements**: Production database build with 7,000+ journals
- **Advanced Features**: Consider implementing optional enhancements
- **Performance Optimization**: Scale testing with full dataset

---

**Final Note**: This implementation plan is designed to create a working MVP in approximately 15-20 hours of development time, spread across multiple sessions. Each step produces tangible results that can be tested and verified independently.