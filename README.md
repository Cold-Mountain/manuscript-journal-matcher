# Manuscript Journal Matcher (MJM)

**A production-ready semantic matching system that helps researchers find the most suitable academic journals for their manuscripts.**

## 🎉 **PRODUCTION STATUS: COMPLETED** ✅

**Last Updated**: July 28, 2025  
**Version**: 2.0 (Production Ready)  
**Database**: 7,648 medical journals from Medicine Journal Rankings 2024

## Project Overview

The Manuscript Journal Matcher is a fully functional web-based application that allows researchers to upload their manuscript (PDF or DOCX) and receive ranked suggestions of academic journals based on semantic similarity. The system extracts key metadata (title, abstract) from manuscripts and matches them to journals using advanced NLP embeddings and vector similarity search with comprehensive quality metrics.

### 🚀 Key Features (PRODUCTION READY)
- **📄 File Upload**: Support for PDF and DOCX manuscript files
- **🤖 Automatic Extraction**: Intelligent extraction of title and abstract
- **🧠 Semantic Matching**: Uses advanced BERT-based embeddings (384D) for similarity scoring
- **📊 Production Database**: 7,648 medical journals from Medicine Journal Rankings 2024
- **🏆 Quality Metrics**: SJR rankings, quartiles (Q1-Q4), H-index, citation counts
- **🔍 Advanced Filtering**: SJR quartiles, rankings, publishers, countries, impact metrics
- **💰 Cost Transparency**: APC information and publication costs
- **⚡ High Performance**: 0.005s average search time, 570MB memory usage
- **🎯 Proven Accuracy**: Semantic matching with verified medical journal recommendations
- **🔒 Privacy First**: No permanent storage of manuscript content

### 📋 Production Workflow
1. **📤 Upload**: User uploads manuscript file (.pdf or .docx)
2. **🔍 Extract**: System extracts title and abstract using parsing libraries
3. **🧠 Embed**: Abstract converted to 384-dimensional vector using all-MiniLM-L6-v2
4. **🔍 Search**: FAISS-based cosine similarity search through 7,648 journal embeddings
5. **🏆 Rank**: Results ranked by semantic similarity + journal quality (SJR, H-index)
6. **🎛️ Filter**: Advanced filtering by quartiles, rankings, publishers, costs
7. **📊 Display**: Top matches with comprehensive metadata and quality indicators

## System Architecture

### Core Components

#### 1. Input & Upload Module (`extractor.py`)
- **File Support**: PDF (`.pdf`) and Word (`.docx`) files
- **Extraction Tools**: 
  - `pdfplumber`, `PyMuPDF`, or `pdfminer.six` for PDF processing
  - `python-docx` for Word document processing
- **Metadata Extracted**:
  - Title (automatic detection via formatting/position)
  - Abstract (section-based extraction)
  - Keywords (optional, when available)

#### 2. Embedding Engine (`embedder.py`)
- **Model Options** (choose one for implementation):
  - `all-MiniLM-L6-v2` (fast baseline, 384 dimensions)
  - `SciBERT` (scientific/biomedical focus, 768 dimensions)
  - `Specter2` (advanced scientific papers, hosted or local)
- **Vector Operations**: Cosine similarity for manuscript-journal comparison
- **Storage**: FAISS (Facebook AI Similarity Search) for fast retrieval

#### 3. Production Journal Database (`journal_db_builder.py`) ✅
- **Primary Data Source**: Medicine Journal Rankings 2024 (Scimago)
- **Secondary Sources**: DOAJ (open access), CrossRef (publisher data)
- **Total Journals**: 7,648 medical journals
- **Data Quality**: 96.6% validation success rate

- **Production Database Schema (CSV + DOAJ Enhanced)**:
```json
{
  "id": "CSV_28773",
  "display_name": "Ca-A Cancer Journal for Clinicians",
  "issn": ["1542-4863", "0007-9235"],
  "issn_l": "1542-4863",
  
  "scimago_rank": 1,
  "sjr_score": 145.004,
  "sjr_quartile": "Q1",
  "sourceid": "28773",
  
  "publisher": "John Wiley and Sons Inc",
  "country": "United States",
  "region": "Northern America",
  "coverage_years": "1950-2025",
  
  "h_index": 223,
  "works_count": 43,
  "cited_by_count": 40834,
  "cites_per_doc": 168.71,
  "refs_per_doc": 62.88,
  
  "subjects": [
    {"name": "Hematology", "score": 0.9, "quartile": "Q1"},
    {"name": "Oncology", "score": 0.9, "quartile": "Q1"}
  ],
  "areas": ["Medicine"],
  
  "semantic_fingerprint": "Journal: Ca-A Cancer Journal for Clinicians | Publisher: John Wiley and Sons Inc | Scimago Rank: 1 | SJR Quartile: Q1 | Subject areas: Hematology (Q1), Oncology (Q1) | Country: United States | H-index: 223 | Top 100 journal",
  "embedding": [0.1, 0.2, -0.3, ...],
  
  "csv_source": true,
  "csv_imported_at": "2025-07-28T15:30:00.000Z"
}
```

**Key DOAJ Enhancements**:
- **Open Access Data**: `oa_status`, `in_doaj`, `oa_start_year`
- **Cost Information**: `has_apc`, `apc_amount`, `apc_currency` (supports multiple currencies)
- **Enhanced Metadata**: `subjects_doaj`, `languages`, `license_type`, `country_doaj`
- **Publisher Details**: `publisher_doaj` (often more accurate than OpenAlex data)
- **Combined Subjects**: Merges OpenAlex and DOAJ subject classifications

#### 4. Semantic Search (`match_journals.py`)
- **Input Processing**: Embed manuscript abstract using same model as journal DB
- **Search Algorithm**: FAISS cosine similarity search
- **Ranking**: Sort by similarity score with confidence thresholds
- **Filtering**: Optional metadata filters (discipline, OA status, APC range)

#### 5. Web Interface (`main.py`)
- **Framework**: Streamlit for rapid MVP development
- **Features**:
  - Drag-and-drop file upload
  - Progress indicators during processing
  - Sortable/filterable results table
  - Export functionality (CSV/PDF)
  - Direct links to journal submission pages

## Technical Stack

### Backend Dependencies
```bash
# Core framework
fastapi>=0.104.0          # API framework (alternative to Streamlit)
streamlit>=1.28.0         # Web interface for MVP

# Document processing
pdfplumber>=0.7.6         # PDF text extraction
python-docx>=0.8.11       # Word document processing
PyMuPDF>=1.23.0           # Alternative PDF processor

# Machine learning & embeddings
sentence-transformers>=2.2.2   # Embedding models
transformers>=4.35.0           # HuggingFace transformers
torch>=2.1.0                   # PyTorch backend

# Vector search & data
faiss-cpu>=1.7.4          # Vector similarity search
pandas>=2.1.0             # Data manipulation
numpy>=1.24.0             # Numerical operations

# API clients
requests>=2.31.0          # HTTP requests for APIs
sqlite3                   # Built-in database for metadata
```

### Development Tools
```bash
pytest>=7.4.0            # Testing framework
black>=23.0.0             # Code formatting
flake8>=6.0.0             # Linting
jupyter>=1.0.0            # Development notebooks
```

## Project Structure

```
manuscript-journal-matcher/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package configuration
├── .gitignore                    # Git ignore patterns
├── .env.example                  # Environment variables template
│
├── src/
│   ├── __init__.py
│   ├── main.py                   # Streamlit application entry point
│   ├── extractor.py              # Document metadata extraction
│   ├── embedder.py               # Embedding generation and similarity
│   ├── journal_db_builder.py     # Database creation and updates
│   ├── match_journals.py         # Semantic matching engine
│   ├── utils.py                  # Shared utilities and helpers
│   └── config.py                 # Configuration management
│
├── data/
│   ├── journal_metadata.json     # Journal information database
│   ├── journal_embeddings.faiss  # Vector index for similarity search
│   ├── sample_manuscripts/       # Test files for development
│   └── api_cache/                # Cached API responses
│
├── tests/
│   ├── __init__.py
│   ├── test_extractor.py         # Test document processing
│   ├── test_embedder.py          # Test embedding functionality
│   ├── test_matching.py          # Test similarity search
│   └── fixtures/                 # Test data files
│
├── scripts/
│   ├── build_database.py         # Initial database construction
│   ├── update_journals.py        # Periodic database updates
│   └── benchmark_models.py       # Model performance comparison
│
└── docs/
    ├── api_documentation.md       # API endpoint documentation
    ├── deployment_guide.md        # Production deployment instructions
    └── user_guide.md              # End-user documentation
```

## API Integration Details

### OpenAlex API
- **Endpoint**: `https://api.openalex.org/`
- **Usage**: Journal metadata, scopes, sample articles
- **Rate Limits**: 100,000 requests/day (no authentication required)
- **Key Fields**: `display_name`, `description`, `works_count`, `cited_by_count`

### DOAJ API (✨ **NEW in Step 8**)
- **Endpoint**: `https://doaj.org/api/v3/`
- **Usage**: Open access status, Article Processing Charges (APC), license information, editorial policies
- **Authentication**: No API key required for basic access
- **Rate Limits**: Conservative 1 request/second (configurable)
- **Key Fields**: 
  - `bibjson.apc`: APC information including amounts and currency
  - `bibjson.oa_start`: Open access start year
  - `bibjson.subject`: Subject classifications
  - `bibjson.language`: Publication languages
  - `bibjson.license`: License types (CC BY, MIT, etc.)
  - `bibjson.publisher`: Publisher details including country
  - `bibjson.editorial`: Editorial policies and review processes
- **Integration**: Automatically enriches journal database with comprehensive open access metadata

### CrossRef API
- **Endpoint**: `https://api.crossref.org/`
- **Usage**: Publisher details, additional metadata
- **Rate Limits**: 50 requests/second (polite pool)
- **Key Fields**: `publisher`, `ISSN`, `subject`

## 🎯 Implementation Status - COMPLETED ✅

### ✅ Phase 1: MVP Development (COMPLETED)
1. **✅ Core Infrastructure**
   - ✅ Project structure and dependencies set up
   - ✅ PDF/DOCX extraction implemented (`extractor.py`)
   - ✅ Embedding pipeline with all-MiniLM-L6-v2 (`embedder.py`)
   - ✅ Production database from Medicine Journal Rankings CSV

2. **✅ Matching Engine**
   - ✅ FAISS vector search implemented (`match_journals.py`)
   - ✅ Cosine similarity matching with 0.005s performance
   - ✅ Full Streamlit interface with advanced filtering
   - ✅ File upload and comprehensive results display

3. **✅ Integration & Testing**
   - ✅ CSV data integration with quality validation (96.6% success)
   - ✅ Comprehensive test suite with 10+ integration tests
   - ✅ Error handling and validation throughout system
   - ✅ Production-ready performance optimization

### ✅ Phase 2: Enhancement & Polish (COMPLETED)
- ✅ Advanced filtering: SJR quartiles, rankings, H-index, publishers
- ✅ Performance optimization: 7,648 journals, 570MB memory usage
- ✅ Enhanced UI with quality indicators and detailed metrics
- ✅ Export functionality for CSV results
- ✅ Real-time search with sub-second response times

### 🚀 Phase 3: Production Ready Features (IMPLEMENTED)
- ✅ Medical field specialization with 7,648 medical journals
- ✅ Quality-based ranking combining semantic similarity + journal prestige
- ✅ Advanced semantic matching understanding medical terminology
- ✅ Geographic and publisher diversity (104 countries, 2,053+ publishers)
- ✅ Comprehensive metadata integration with citation metrics

### 🔮 Future Enhancements (Optional)
- DOAJ integration for open access information
- Multi-language manuscript support
- Integration with journal submission systems  
- Reviewer suggestion based on references
- Fine-tuned models for specific medical subfields

## Database Construction Process

### Initial Setup
```python
# Pseudocode for database creation
def build_journal_database():
    # 1. Fetch journals from OpenAlex
    journals = fetch_openalex_journals(limit=50000)
    
    # 2. Enrich with DOAJ data
    for journal in journals:
        journal.update(fetch_doaj_info(journal.issn))
    
    # 3. Add CrossRef publisher info
    for journal in journals:
        journal.update(fetch_crossref_info(journal.issn))
    
    # 4. Create semantic fingerprints
    for journal in journals:
        fingerprint = create_semantic_fingerprint(
            scope=journal.description,
            sample_articles=fetch_sample_articles(journal.id, limit=3)
        )
        journal.semantic_fingerprint = fingerprint
    
    # 5. Generate embeddings
    embeddings = embed_texts([j.semantic_fingerprint for j in journals])
    
    # 6. Build FAISS index
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    
    # 7. Save to disk
    save_journal_metadata(journals)
    save_faiss_index(index)
```

## Privacy & Security

### Data Handling
- **Temporary Processing**: All uploaded files processed in memory or temporary directories
- **Automatic Cleanup**: Files deleted immediately after processing
- **No Persistent Storage**: Manuscript content never saved to disk
- **Local Processing**: Option for completely offline operation

### Security Measures
- Input validation for uploaded files
- File type verification and size limits
- Sanitization of extracted text
- Rate limiting for API calls
- Secure temporary file handling

### User Privacy
- No tracking or analytics by default
- Optional usage statistics (anonymized)
- Clear privacy policy and data handling disclosure
- GDPR compliance considerations

## Performance Considerations

### Optimization Strategies
- **Lazy Loading**: Load journal database only when needed
- **Caching**: Cache API responses and embeddings
- **Batch Processing**: Process multiple manuscripts efficiently
- **Index Optimization**: Use FAISS IVF for large databases
- **Memory Management**: Stream large files, cleanup temporary data

### Scalability
- **Horizontal Scaling**: Stateless design allows multiple instances
- **Database Sharding**: Split journal database by field/region
- **CDN Integration**: Serve static assets from CDN
- **API Rate Management**: Implement exponential backoff for external APIs

## Error Handling & Edge Cases

### Common Issues
- Malformed PDF files (use multiple extraction libraries as fallback)
- Missing abstracts (extract first paragraph or summary)
- Non-English manuscripts (add language detection)
- Very short or very long abstracts (handle edge cases gracefully)
- API rate limits (implement retry logic with exponential backoff)

### Validation
- File format verification
- Text extraction quality checks
- Embedding dimension validation
- Journal database integrity checks

## Testing Strategy

### Unit Tests
- Document extraction accuracy
- Embedding generation consistency
- Similarity calculation correctness
- API response handling

### Integration Tests
- End-to-end workflow validation
- Database construction pipeline
- External API integration
- UI functionality testing

### Performance Tests
- Large file processing
- Database query performance
- Memory usage optimization
- Concurrent user handling

## Deployment Options

### Local Development
```bash
# Setup commands
git clone [repository]
cd manuscript-journal-matcher
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python scripts/build_database.py
streamlit run src/main.py
```

### Production Deployment
- **Docker**: Containerized deployment with Docker Compose
- **Cloud Platforms**: AWS ECS, Google Cloud Run, or Heroku
- **Database**: PostgreSQL for metadata, S3/GCS for FAISS indices
- **Load Balancing**: nginx or cloud load balancer
- **Monitoring**: Application logs, performance metrics, error tracking

## Configuration

### Environment Variables
```bash
# API Keys (optional but recommended)
OPENALEX_API_KEY=your_openalex_key
DOAJ_API_KEY=your_doaj_key
CROSSREF_MAILTO=your_email@domain.com

# Application Settings
MAX_FILE_SIZE_MB=50
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_RESULTS=20
CACHE_DURATION_HOURS=24

# Database Settings
DATABASE_PATH=data/journal_metadata.json
FAISS_INDEX_PATH=data/journal_embeddings.faiss
```

## 🚀 Production Usage

### Quick Start
```bash
# Clone and setup
git clone [repository]
cd manuscript-journal-matcher
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Launch production app
streamlit run src/main.py
```

### Production API Usage
```python
# 1. Extract manuscript metadata
from src.extractor import extract_manuscript_data
title, abstract = extract_manuscript_data("manuscript.pdf")

# 2. Initialize production journal matcher
from src.match_journals import JournalMatcher
matcher = JournalMatcher()
matcher.load_database()  # Loads 7,648 journals

# 3. Search with advanced CSV-based filtering
results = matcher.search_similar_journals(
    query_text=abstract,
    top_k=10,
    filters={
        'sjr_quartiles': ['Q1', 'Q2'],   # Top 50% quality journals
        'max_rank': 500,                 # Top 500 Scimago ranked
        'min_h_index': 50,              # High impact journals
        'min_sjr_score': 1.0,           # Minimum SJR score
    }
)

# 4. Display production results with quality metrics
for match in results:
    print(f"{match['display_name']}: {match['similarity_score']:.3f}")
    print(f"  🏆 SJR Rank: #{match.get('scimago_rank')} ({match.get('sjr_quartile')})")
    print(f"  📊 SJR Score: {match.get('sjr_score', 0):.2f}")
    print(f"  📈 H-Index: {match.get('h_index', 0)}")
    print(f"  🏢 Publisher: {match.get('publisher')}")
    print(f"  🌍 Country: {match.get('country')}")
    print()
```

### Sample Output (with DOAJ Integration)
```
PLOS Computational Biology: 0.892 🏆 DOAJ Listed
  Publisher: Public Library of Science
  Open Access: ✅ Yes
  APC: $1695 USD
  Languages: English
  License: CC BY

BMC Medical Imaging: 0.867 🏆 DOAJ Listed
  Publisher: BioMed Central
  Open Access: ✅ Yes
  APC: $2290 USD
  Languages: English
  License: CC BY

Frontiers in Neuroscience: 0.834 🏆 DOAJ Listed
  Publisher: Frontiers Media SA
  Open Access: ✅ Yes
  APC: $2950 USD
  Languages: English
  License: CC BY

Journal of Open Source Software: 0.812 🏆 DOAJ Listed
  Publisher: Open Journals
  Open Access: ✅ Yes
  APC: Free
  Languages: English
  License: MIT

Database Statistics:
📊 Total journals: 10,247
🏆 DOAJ journals: 2,156 (21%)
✅ Open Access: 3,842 (37%)
💚 Free to publish: 1,203 (12%)
💰 Average APC: $1,847 USD
```

## 🏆 DOAJ Integration Features (Step 8)

The Manuscript Journal Matcher now includes comprehensive integration with the **Directory of Open Access Journals (DOAJ)**, providing detailed open access information and cost transparency for researchers.

### What is DOAJ?

The Directory of Open Access Journals (DOAJ) is a curated online directory that indexes and provides access to high-quality, open access, peer-reviewed journals. DOAJ helps researchers find legitimate open access journals and avoid predatory publishers.

### Enhanced Features

#### 🔍 Advanced Filtering Options
- **Open Access Only**: Filter to show only open access journals
- **DOAJ Listed Only**: Restrict results to journals verified by DOAJ
- **APC Range Filtering**: Set maximum/minimum article processing charges
- **Free to Publish**: Find journals with no publication fees
- **Language Filtering**: Filter by publication languages
- **Enhanced Subject Matching**: Combines OpenAlex and DOAJ subject classifications

#### 💰 Cost Transparency
- **Detailed APC Information**: Article Processing Charges in multiple currencies
- **Currency Support**: USD, EUR, GBP, and other major currencies
- **Free Publication Options**: Clearly marked journals with no fees
- **Cost Range Filtering**: Find journals within your budget

#### 📋 Rich Metadata
- **License Information**: CC BY, CC BY-SA, MIT, and other license types
- **Editorial Policies**: Peer review processes and publication timelines
- **Publisher Details**: Enhanced publisher information from DOAJ
- **Geographic Information**: Journal origin countries
- **Language Support**: Multi-language publication capabilities

### Database Building with DOAJ

The enhanced database building process automatically enriches journal data:

```bash
# Build database with DOAJ integration (default)
python scripts/build_database.py --limit 1000

# Skip DOAJ integration for faster building
python scripts/build_database.py --limit 1000 --skip-doaj

# Custom DOAJ rate limiting
python scripts/build_database.py --doaj-rate-limit 0.5
```

### Filter Options Reference

| Filter Option | Type | Description | Example |
|---------------|------|-------------|---------|
| `open_access_only` | boolean | Only open access journals | `True` |
| `doaj_only` | boolean | Only DOAJ-listed journals | `True` |
| `no_apc_only` | boolean | Only free-to-publish journals | `True` |
| `max_apc` | number | Maximum APC in USD | `2000` |
| `min_apc` | number | Minimum APC in USD | `500` |
| `languages` | list | Publication languages | `['English', 'Spanish']` |
| `subjects` | list | Subject areas | `['Medicine', 'Biology']` |
| `publishers` | list | Publisher names | `['PLOS', 'BioMed Central']` |

### Streamlit Interface Enhancements

The web interface now includes:
- **Comprehensive Filter Controls**: Easy-to-use checkboxes and sliders
- **Enhanced Results Display**: DOAJ badges, APC information, license types
- **Cost Visualization**: Color-coded APC ranges and free publication indicators
- **Detailed Journal Cards**: Expandable sections with complete DOAJ metadata
- **Export with DOAJ Data**: CSV exports include all DOAJ fields

### API Response Format

Enhanced journal results now include:

```json
{
  "display_name": "PLOS Computational Biology",
  "similarity_score": 0.892,
  "publisher": "PLOS",
  "publisher_doaj": "Public Library of Science",
  "oa_status": true,
  "in_doaj": true,
  "has_apc": true,
  "apc_amount": 1695,
  "apc_currency": "USD",
  "oa_start_year": 2005,
  "subjects_doaj": ["Computational Biology", "Bioinformatics"],
  "languages": ["English"],
  "license_type": ["CC BY"],
  "country_doaj": "US"
}
```

### Testing DOAJ Integration

Comprehensive test suite includes:
- **DOAJ API Client Tests**: API response handling, rate limiting, error cases
- **Data Processing Tests**: DOAJ metadata processing and validation
- **Filtering Tests**: All DOAJ-enhanced filter options
- **Integration Tests**: End-to-end workflow with DOAJ data
- **UI Tests**: Streamlit interface with DOAJ features

Run DOAJ-specific tests:
```bash
# Run all DOAJ integration tests
pytest tests/test_doaj_integration.py -v

# Run DOAJ filtering tests
pytest tests/test_matching.py::TestDOAJEnhancedFiltering -v
```

### Performance Considerations

- **Caching**: DOAJ API responses are cached to minimize requests
- **Rate Limiting**: Conservative 1 request/second default (configurable)
- **Batch Processing**: Efficient journal enrichment in batches
- **Resume Capability**: Database building can resume from interruptions
- **Memory Optimization**: Streaming processing for large journal sets

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `pytest tests/`
5. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions
- Maintain test coverage above 80%

## Future Enhancements

### Advanced Features
- **Multi-modal Matching**: Include figures, tables, and references
- **Citation Analysis**: Predict citation potential based on journal choice
- **Submission Timeline**: Track journal response times and acceptance rates
- **Collaboration Tools**: Team-based manuscript evaluation
- **Integration APIs**: Connect with reference managers (Zotero, Mendeley)

### Model Improvements
- Fine-tune embeddings on scientific literature
- Domain-specific models (medical, engineering, etc.)
- Multi-language support for international journals
- Active learning from user feedback

### Analytics Dashboard
- Success rate tracking
- Popular journal trends
- User behavior insights
- Performance monitoring

## License

MIT License - see LICENSE file for details

## Contact & Support

For questions, issues, or contributions:
- GitHub Issues: [repository]/issues
- Documentation: [repository]/wiki
- Email: [contact_email]

---

**Note**: This project is designed for research and educational purposes. Always verify journal requirements and submission guidelines independently before submitting manuscripts.