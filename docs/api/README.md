# API Reference Documentation

Welcome to the Manuscript Journal Matcher API documentation. This comprehensive reference provides detailed information about all public functions, classes, and modules in the system.

## 📚 Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **[extractor](extractor.md)** | Document text extraction | `extract_manuscript_data()`, `extract_text_from_pdf()` |
| **[embedder](embedder.md)** | Text embedding generation | `embed_text()`, `embed_texts()`, `cosine_similarity_single()` |
| **[match_journals](match_journals.md)** | Journal matching & search | `JournalMatcher.search_similar_journals()` |
| **[journal_db_builder](journal_db_builder.md)** | Database construction | `OpenAlexAPI`, `DOAJIntegration` |
| **[config](config.md)** | Configuration management | `get_embedding_dimension()`, `validate_file_size()` |
| **[utils](utils.md)** | Utility functions | `validate_file()`, `clean_text()`, `save_cache()` |

## 🚀 Quick Start Examples

### Basic Usage Pattern

```python
from src.extractor import extract_manuscript_data
from src.match_journals import JournalMatcher

# 1. Extract manuscript content
manuscript_data = extract_manuscript_data("paper.pdf")
abstract = manuscript_data['abstract']

# 2. Initialize matcher and search
matcher = JournalMatcher()
matcher.load_database()

# 3. Find similar journals
results = matcher.search_similar_journals(
    query_text=abstract,
    top_k=10,
    filters={'open_access_only': True}
)

# 4. Process results
for journal in results:
    print(f"{journal['display_name']}: {journal['similarity_score']:.3f}")
```

### Advanced Filtering Example

```python
# Advanced search with multiple filters
results = matcher.search_similar_journals(
    query_text=abstract,
    top_k=15,
    min_similarity=0.7,
    filters={
        'doaj_only': True,           # DOAJ-listed journals only
        'max_apc': 2000,            # Maximum $2000 APC
        'subjects': ['Medicine'],    # Medical journals
        'languages': ['English'],   # English language
        'min_h_index': 50           # High-impact journals
    }
)
```

## 📖 Core Concepts

### Document Processing Pipeline

```mermaid
graph LR
    A[Upload File] --> B[Validate File]
    B --> C[Extract Text]
    C --> D[Parse Title/Abstract]
    D --> E[Generate Embeddings]
    E --> F[Search Similar Journals]
```

### Embedding & Similarity

The system uses **sentence-transformers** with the `all-MiniLM-L6-v2` model:
- **Dimension**: 384-dimensional vectors
- **Similarity**: Cosine similarity for semantic matching
- **Performance**: <0.1s search time for 7,600+ journals

### FAISS Vector Search

```python
# Index types used based on dataset size
if n_journals < 1000:
    index = faiss.IndexFlatIP(dimension)  # Exact search
else:
    index = faiss.IndexIVFFlat(quantizer, dimension, n_centroids)  # Approximate
```

## 🔧 Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |
| `MAX_RESULTS` | `20` | Default number of results |
| `MIN_SIMILARITY` | `0.1` | Minimum similarity threshold |
| `CACHE_DURATION_HOURS` | `24` | API response cache duration |

### File Paths

```python
# Default paths (configurable)
DATA_DIR = "data/"
JOURNAL_METADATA_PATH = "data/journal_metadata.json"
FAISS_INDEX_PATH = "data/journal_embeddings.faiss"
API_CACHE_DIR = "data/api_cache/"
```

## 🚦 Error Handling

All modules use custom exceptions for clear error handling:

```python
# Exception hierarchy
Exception
├── ExtractionError          # Document processing errors
├── EmbeddingError          # Embedding generation errors
├── MatchingError           # Journal matching errors
├── JournalDatabaseError    # Database building errors
├── ValidationError         # Data validation errors
└── FileError              # File handling errors
```

## 📊 Performance Characteristics

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| PDF extraction | 0.5-2s | 50MB | Depends on file size |
| Embedding generation | 0.1s | 100MB | Per abstract |
| Journal search | <0.1s | 570MB | 7,600+ journals loaded |
| Database loading | 2-5s | 600MB | One-time initialization |

## 🔗 API Integration

### OpenAlex API

```python
from src.journal_db_builder import OpenAlexAPI

api = OpenAlexAPI(api_key="optional_key")
journals = api.fetch_journals(limit=1000, filters={'type': 'journal'})
```

### DOAJ Integration

```python
from src.journal_db_builder import DOAJIntegration

doaj = DOAJIntegration(rate_limit=1.0)  # 1 second between requests
oa_info = doaj.get_journal_info(issn="1234-5678")
```

## 📝 Data Formats

### Journal Metadata Schema

```python
{
    "id": "openalex_id",
    "display_name": "Journal of Example Research",
    "publisher": "Example Publisher",
    "issn": ["1234-5678", "8765-4321"],
    "issn_l": "1234-5678",
    "is_oa": True,
    "oa_status": True,           # DOAJ data
    "in_doaj": True,             # DOAJ listed
    "has_apc": True,             # Has APC
    "apc_amount": 1500,          # APC in USD
    "apc_currency": "USD",       # Currency
    "subjects": [...],           # OpenAlex subjects
    "subjects_doaj": [...],      # DOAJ subjects
    "languages": ["English"],    # Publication languages
    "license_type": ["CC BY"],   # License types
    "homepage_url": "https://...",
    "works_count": 5000,         # Total publications
    "cited_by_count": 150000,    # Total citations
    "h_index": 85,               # H-index
    "country_doaj": "US",        # Publisher country
    "oa_start_year": 2010        # Open access start year
}
```

### Search Results Format

```python
{
    "rank": 1,
    "journal_name": "Nature Medicine",
    "similarity_score": 0.892,
    "publisher": "Nature Publishing Group",
    "issn": "1078-8956",
    "is_open_access": False,
    "in_doaj": False,
    "apc_amount": 9750,
    "apc_currency": "USD",
    "has_apc": True,
    "homepage_url": "https://www.nature.com/nm/",
    "works_count": 15000,
    "cited_by_count": 2500000,
    "h_index": 350,
    "subjects": ["Medicine", "Biomedical Research"],
    "languages": ["English"],
    "license_type": ["All rights reserved"],
    "country": "US"
}
```

## 🎯 Best Practices

### Memory Management

```python
# Efficient batch processing
embeddings = embed_texts(
    texts,
    batch_size=32,        # Process in batches
    show_progress=True    # Monitor progress
)

# Clean up temporary files
from src.utils import safe_delete_file
safe_delete_file(temp_file_path)
```

### Error Recovery

```python
# Robust error handling
try:
    results = matcher.search_similar_journals(query)
except MatchingError as e:
    logger.error(f"Search failed: {e}")
    # Fallback to cached results or simplified search
    results = fallback_search(query)
```

### Caching Strategy

```python
from src.utils import save_cache, load_cache

# Cache expensive operations
cache_key = f"search_{hash(query_text)}"
results = load_cache(cache_key)

if results is None:
    results = expensive_search_operation(query_text)
    save_cache(cache_key, results, expiry_hours=1)
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues)
- **Documentation**: [User Guide](../user/README.md)
- **Contributing**: [Developer Guide](../developer/README.md)

---

*This API reference is automatically generated from source code docstrings. Last updated: July 29, 2025*