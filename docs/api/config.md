# Configuration API Reference

The `config` module handles application settings, environment variables, and default configurations for the journal matching system.

## 📋 Module Overview

```python
from src.config import (
    # Core configuration
    PROJECT_ROOT, DATA_DIR, JOURNAL_METADATA_PATH, FAISS_INDEX_PATH,
    
    # Embedding configuration
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSIONS, EMBEDDING_BATCH_SIZE,
    get_embedding_dimension,
    
    # File processing
    MAX_FILE_SIZE_MB, MAX_FILE_SIZE_BYTES, SUPPORTED_FILE_TYPES,
    validate_file_size, get_supported_extensions,
    
    # API configuration
    OPENALEX_BASE_URL, DOAJ_BASE_URL, get_api_headers,
    
    # Search parameters
    DEFAULT_TOP_K_RESULTS, MIN_SIMILARITY_THRESHOLD,
    
    # Text patterns
    TITLE_PATTERNS, ABSTRACT_PATTERNS, ABSTRACT_KEYWORDS,
    
    # Utilities
    ensure_directories_exist
)
```

## 🚀 Quick Start

```python
from src.config import *

# Check current configuration
print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"Max file size: {MAX_FILE_SIZE_MB}MB")
print(f"Data directory: {DATA_DIR}")
print(f"Default results: {DEFAULT_TOP_K_RESULTS}")

# Validate file before processing
if validate_file_size(file_size_bytes):
    print("✅ File size OK")
else:
    print("❌ File too large")
```

## 📁 Path Configuration

### Core Directories

```python
# Project structure
PROJECT_ROOT = Path(__file__).parent.parent  # Root directory
DATA_DIR = PROJECT_ROOT / "data"              # Data storage
API_CACHE_DIR = DATA_DIR / "api_cache"        # API response cache
SAMPLE_MANUSCRIPTS_DIR = DATA_DIR / "sample_manuscripts"  # Test files

# Test directories
TESTS_DIR = PROJECT_ROOT / "tests"
TEST_FIXTURES_DIR = TESTS_DIR / "fixtures"
```

**Example Usage:**
```python
from src.config import DATA_DIR, API_CACHE_DIR

# Ensure directories exist
from src.config import ensure_directories_exist
ensure_directories_exist()

# Check if directories exist
print(f"📁 Data directory exists: {DATA_DIR.exists()}")
print(f"📁 Cache directory exists: {API_CACHE_DIR.exists()}")

# Build paths for data files
journal_file = DATA_DIR / "custom_journals.json"
print(f"📄 Journal file path: {journal_file}")
```

### Default File Paths

```python
# Core data files
JOURNAL_METADATA_PATH = DATA_DIR / "journal_metadata.json"
FAISS_INDEX_PATH = DATA_DIR / "journal_embeddings.faiss"

# Usage example
from pathlib import Path

# Check if database files exist
if JOURNAL_METADATA_PATH.exists():
    size_mb = JOURNAL_METADATA_PATH.stat().st_size / (1024 * 1024)
    print(f"📊 Journal metadata: {size_mb:.1f}MB")
else:
    print("⚠️ Journal metadata not found - run database builder")

if FAISS_INDEX_PATH.exists():
    print("✅ FAISS index found")
else:
    print("⚠️ FAISS index not found - will be created automatically")
```

## 🤖 Embedding Configuration

### Model Settings

```python
# Default embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Supported models and their dimensions
EMBEDDING_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,        # Fast, good quality (default)
    "all-MiniLM-L12-v2": 384,       # Better quality, slower
    "SciBERT": 768,                 # Scientific texts
    "specter2": 768,                # Scientific papers
    "all-mpnet-base-v2": 768        # Best quality, slowest
}
```

**Example Usage:**
```python
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSIONS, get_embedding_dimension

# Get current model info
current_model = EMBEDDING_MODEL_NAME
current_dim = get_embedding_dimension()

print(f"🤖 Current model: {current_model}")
print(f"📐 Embedding dimension: {current_dim}")

# Check available models
print("\n📚 Available models:")
for model, dim in EMBEDDING_DIMENSIONS.items():
    status = "✅ (current)" if model == current_model else "  "
    print(f"{status} {model}: {dim} dimensions")

# Validate dimension for current model
expected_dim = EMBEDDING_DIMENSIONS.get(current_model, 384)
print(f"\n🔍 Expected dimension: {expected_dim}")
```

### Processing Parameters

```python
# Embedding processing configuration
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2048"))
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # "auto", "cpu", "cuda"
```

**Example Usage:**
```python
from src.config import *

# Display processing configuration
print("⚙️ Embedding Processing Configuration:")
print(f"  Batch size: {EMBEDDING_BATCH_SIZE}")
print(f"  Max text length: {MAX_TEXT_LENGTH} characters")
print(f"  Cache size: {EMBEDDING_CACHE_SIZE} embeddings")
print(f"  Device preference: {EMBEDDING_DEVICE}")

# Adjust for your hardware
import os
if os.getenv("CUDA_VISIBLE_DEVICES"):
    print("🚀 GPU detected - using CUDA acceleration")
else:
    print("💻 Using CPU for embeddings")
```

## 📄 File Processing Configuration

### File Size Limits

```python
# File size limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def validate_file_size(file_size: int) -> bool:
    """Check if file size is within allowed limits."""
    return file_size <= MAX_FILE_SIZE_BYTES
```

**Example Usage:**
```python
from src.config import MAX_FILE_SIZE_MB, validate_file_size
from pathlib import Path

# Check file size before processing
file_path = Path("large_manuscript.pdf")
if file_path.exists():
    file_size = file_path.stat().st_size
    size_mb = file_size / (1024 * 1024)
    
    print(f"📄 File: {file_path.name}")
    print(f"📊 Size: {size_mb:.1f}MB (limit: {MAX_FILE_SIZE_MB}MB)")
    
    if validate_file_size(file_size):
        print("✅ File size OK for processing")
    else:
        print(f"❌ File too large! Maximum allowed: {MAX_FILE_SIZE_MB}MB")
        print("💡 Try compressing the PDF or splitting into smaller files")
```

### Supported File Types

```python
# Supported file extensions
SUPPORTED_FILE_TYPES = [".pdf", ".docx"]

def get_supported_extensions() -> list[str]:
    """Get list of supported file extensions."""
    return SUPPORTED_FILE_TYPES.copy()
```

**Example Usage:**
```python
from src.config import SUPPORTED_FILE_TYPES, get_supported_extensions
from pathlib import Path

# Check file type support
def check_file_support(file_path):
    """Check if file type is supported."""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    supported = get_supported_extensions()
    
    print(f"📄 File: {path.name}")
    print(f"🔍 Extension: {extension}")
    print(f"📚 Supported types: {', '.join(supported)}")
    
    if extension in supported:
        print("✅ File type supported")
        return True
    else:
        print("❌ File type not supported")
        print("💡 Supported formats: PDF (.pdf) and Word documents (.docx)")
        return False

# Usage
check_file_support("research_paper.pdf")    # ✅ Supported
check_file_support("manuscript.docx")       # ✅ Supported  
check_file_support("document.txt")          # ❌ Not supported
```

## 🌐 API Configuration

### API Endpoints

```python
# API base URLs
OPENALEX_BASE_URL = "https://api.openalex.org"
DOAJ_BASE_URL = "https://doaj.org/api/v2"
CROSSREF_BASE_URL = "https://api.crossref.org"
```

### API Keys and Headers

```python
# API authentication (optional but recommended)
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", None)
DOAJ_API_KEY = os.getenv("DOAJ_API_KEY", None)
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "researcher@example.com")

def get_api_headers() -> dict[str, str]:
    """Get API headers with optional authentication."""
    headers = {
        "User-Agent": "ManuscriptJournalMatcher/1.0 (mailto:researcher@example.com)"
    }
    
    if CROSSREF_MAILTO:
        headers["mailto"] = CROSSREF_MAILTO
    
    return headers
```

**Example Usage:**
```python
from src.config import *
import os

# Set up API configuration
os.environ["CROSSREF_MAILTO"] = "your.email@university.edu"
# os.environ["OPENALEX_API_KEY"] = "your_openalex_key"  # Optional

# Get configured headers
headers = get_api_headers()
print("🔧 API Headers:")
for key, value in headers.items():
    print(f"  {key}: {value}")

# Check API key status
print("\n🔑 API Authentication Status:")
print(f"  OpenAlex API Key: {'✅ Set' if OPENALEX_API_KEY else '❌ Not set (using public access)'}")
print(f"  DOAJ API Key: {'✅ Set' if DOAJ_API_KEY else '❌ Not set (using public access)'}")
print(f"  CrossRef mailto: {'✅ Set' if CROSSREF_MAILTO else '❌ Not set'}")
```

## 🔍 Search Configuration

### Default Parameters

```python
# Search and matching parameters
DEFAULT_TOP_K_RESULTS = int(os.getenv("MAX_RESULTS", "20"))
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY", "0.1"))
CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "24"))
```

**Example Usage:**
```python
from src.config import *

# Display search configuration
print("🔍 Search Configuration:")
print(f"  Default max results: {DEFAULT_TOP_K_RESULTS}")
print(f"  Minimum similarity: {MIN_SIMILARITY_THRESHOLD}")
print(f"  Cache duration: {CACHE_DURATION_HOURS} hours")

# Use in search
from src.match_journals import JournalMatcher

matcher = JournalMatcher()
results = matcher.search_similar_journals(
    query_text="machine learning in healthcare",
    top_k=DEFAULT_TOP_K_RESULTS,
    min_similarity=MIN_SIMILARITY_THRESHOLD
)

print(f"📊 Found {len(results)} results using default configuration")
```

## 📝 Text Extraction Patterns

### Title Detection Patterns

```python
# Regex patterns for title extraction
TITLE_PATTERNS = [
    r"^(.{1,200}?)\n",                          # First line (up to 200 chars)
    r"title:\s*(.+?)(?:\n|$)",                  # "Title:" followed by text
    r"^([A-Z].*?[.!?])\s*\n",                  # Sentence-case first line ending with punctuation
]
```

### Abstract Detection Patterns

```python
# Regex patterns for abstract extraction
ABSTRACT_PATTERNS = [
    r"abstract:?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.|references)|$)",
    r"summary:?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.|references)|$)",
    r"(.*?)(?=\n\s*(?:keywords?|introduction|1\.|references))",
]

# Keywords for section detection
ABSTRACT_KEYWORDS = [
    "abstract", "summary", "synopsis", "overview", "background"
]

INTRODUCTION_KEYWORDS = [
    "introduction", "1.", "1 introduction", "background"
]
```

**Example Usage:**
```python
from src.config import TITLE_PATTERNS, ABSTRACT_PATTERNS, ABSTRACT_KEYWORDS
import re

# Test title extraction
sample_text = """
Machine Learning Applications in Medical Diagnosis: A Comprehensive Review

Abstract: This study reviews recent advances in machine learning
for medical diagnosis applications...

1. Introduction
Machine learning has transformed healthcare...
"""

# Try title patterns
print("🔍 Testing title extraction patterns:")
for i, pattern in enumerate(TITLE_PATTERNS):
    match = re.search(pattern, sample_text, re.IGNORECASE | re.MULTILINE)
    if match:
        title = match.group(1).strip()
        print(f"✅ Pattern {i+1} found title: '{title}'")
        break
else:
    print("❌ No title pattern matched")

# Try abstract detection
print("\n📝 Testing abstract detection:")
for keyword in ABSTRACT_KEYWORDS:
    if keyword in sample_text.lower():
        print(f"✅ Found '{keyword}' keyword")
        
        # Find text after keyword
        start_idx = sample_text.lower().find(keyword)
        if start_idx != -1:
            remaining_text = sample_text[start_idx:]
            for pattern in ABSTRACT_PATTERNS:
                match = re.search(pattern, remaining_text, re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1).strip()
                    print(f"📄 Extracted abstract: '{abstract[:100]}...'")
                    break
        break
```

## 🔧 Environment Variables

### Configuration via Environment

All configuration values can be overridden using environment variables:

```bash
# Embedding configuration
export EMBEDDING_MODEL="all-mpnet-base-v2"
export EMBEDDING_BATCH_SIZE="64"
export EMBEDDING_DEVICE="cuda"

# File processing
export MAX_FILE_SIZE_MB="100"
export MAX_TEXT_LENGTH="4096"

# Search parameters
export MAX_RESULTS="50"
export MIN_SIMILARITY="0.2"
export CACHE_DURATION_HOURS="48"

# API configuration
export OPENALEX_API_KEY="your_key_here"
export CROSSREF_MAILTO="your.email@domain.com"

# Logging
export LOG_LEVEL="DEBUG"
```

**Python Usage:**
```python
import os

# Set configuration programmatically
os.environ["EMBEDDING_MODEL"] = "SciBERT"
os.environ["MAX_RESULTS"] = "30"
os.environ["MIN_SIMILARITY"] = "0.5"

# Import config after setting environment variables
from src.config import *

print(f"🤖 Model: {EMBEDDING_MODEL_NAME}")          # SciBERT
print(f"📊 Max results: {DEFAULT_TOP_K_RESULTS}")   # 30
print(f"🎯 Min similarity: {MIN_SIMILARITY_THRESHOLD}")  # 0.5
```

## 🛠️ Utility Functions

### Directory Management

```python
def ensure_directories_exist() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        API_CACHE_DIR,
        SAMPLE_MANUSCRIPTS_DIR,
        TEST_FIXTURES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
```

**Example Usage:**
```python
from src.config import ensure_directories_exist, DATA_DIR, API_CACHE_DIR

# Ensure all directories exist
ensure_directories_exist()

# Verify directories were created
directories_to_check = [
    ("Data directory", DATA_DIR),
    ("API cache", API_CACHE_DIR),
    ("Sample manuscripts", SAMPLE_MANUSCRIPTS_DIR),
    ("Test fixtures", TEST_FIXTURES_DIR)
]

print("📁 Directory Status:")
for name, path in directories_to_check:
    status = "✅ Exists" if path.exists() else "❌ Missing"
    print(f"  {name}: {status} ({path})")
```

### Configuration Validation

```python
def validate_configuration():
    """Validate current configuration settings."""
    issues = []
    
    # Check embedding model
    if EMBEDDING_MODEL_NAME not in EMBEDDING_DIMENSIONS:
        issues.append(f"Unknown embedding model: {EMBEDDING_MODEL_NAME}")
    
    # Check file size limits
    if MAX_FILE_SIZE_MB <= 0:
        issues.append(f"Invalid max file size: {MAX_FILE_SIZE_MB}MB")
    
    # Check search parameters
    if DEFAULT_TOP_K_RESULTS <= 0:
        issues.append(f"Invalid default results count: {DEFAULT_TOP_K_RESULTS}")
    
    if not (0 <= MIN_SIMILARITY_THRESHOLD <= 1):
        issues.append(f"Invalid similarity threshold: {MIN_SIMILARITY_THRESHOLD}")
    
    # Check directories
    ensure_directories_exist()
    
    return issues

# Usage
issues = validate_configuration()
if issues:
    print("⚠️ Configuration Issues:")
    for issue in issues:
        print(f"  • {issue}")
else:
    print("✅ Configuration is valid")
```

## 📊 Configuration Summary

```python
def print_configuration_summary():
    """Print a comprehensive configuration summary."""
    
    print("=" * 60)
    print("📋 MANUSCRIPT JOURNAL MATCHER CONFIGURATION")
    print("=" * 60)
    
    print(f"\n🤖 EMBEDDING CONFIGURATION")
    print(f"  Model: {EMBEDDING_MODEL_NAME}")
    print(f"  Dimension: {get_embedding_dimension()}")
    print(f"  Batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"  Device: {EMBEDDING_DEVICE}")
    print(f"  Max text length: {MAX_TEXT_LENGTH:,} chars")
    
    print(f"\n📄 FILE PROCESSING")
    print(f"  Max file size: {MAX_FILE_SIZE_MB}MB")
    print(f"  Supported types: {', '.join(SUPPORTED_FILE_TYPES)}")
    
    print(f"\n🔍 SEARCH PARAMETERS")
    print(f"  Default max results: {DEFAULT_TOP_K_RESULTS}")
    print(f"  Min similarity: {MIN_SIMILARITY_THRESHOLD}")
    print(f"  Cache duration: {CACHE_DURATION_HOURS} hours")
    
    print(f"\n📁 FILE PATHS")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Journal metadata: {JOURNAL_METADATA_PATH}")
    print(f"  FAISS index: {FAISS_INDEX_PATH}")
    
    print(f"\n🌐 API CONFIGURATION")
    print(f"  OpenAlex: {OPENALEX_BASE_URL}")
    print(f"  DOAJ: {DOAJ_BASE_URL}")
    print(f"  API key status: {'✅' if OPENALEX_API_KEY else '❌'}")
    
    print("=" * 60)

# Usage
print_configuration_summary()
```

---

*For advanced configuration and deployment settings, see the [Deployment Guide](../deployment/) and [Environment Setup](../user/installation-guide.md).*