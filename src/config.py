"""
Configuration management for Manuscript Journal Matcher.

This module handles application settings, environment variables,
and default configurations for the journal matching system.
"""

import os
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
JOURNAL_METADATA_PATH = DATA_DIR / "journal_metadata.json"
FAISS_INDEX_PATH = DATA_DIR / "journal_embeddings.faiss"
API_CACHE_DIR = DATA_DIR / "api_cache"
SAMPLE_MANUSCRIPTS_DIR = DATA_DIR / "sample_manuscripts"

# Test directories
TESTS_DIR = PROJECT_ROOT / "tests"
TEST_FIXTURES_DIR = TESTS_DIR / "fixtures"

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "SciBERT": 768,
    "specter2": 768,
    "all-mpnet-base-v2": 768
}

# Embedding processing configuration
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2048"))  # Maximum characters for embedding
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))  # Number of embeddings to cache

# Device configuration for embeddings
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # "auto", "cpu", "cuda"

# File processing limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Supported file types
SUPPORTED_FILE_TYPES = [".pdf", ".docx"]

# API configuration
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", None)
DOAJ_API_KEY = os.getenv("DOAJ_API_KEY", None)
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "researcher@example.com")

# API endpoints
OPENALEX_BASE_URL = "https://api.openalex.org"
DOAJ_BASE_URL = "https://doaj.org/api/v2"
CROSSREF_BASE_URL = "https://api.crossref.org"

# Search and matching parameters
DEFAULT_TOP_K_RESULTS = int(os.getenv("MAX_RESULTS", "20"))
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY", "0.1"))
CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "24"))

# Text extraction patterns
TITLE_PATTERNS = [
    r"^(.{1,200}?)\n",  # First line (up to 200 chars)
    r"title:\s*(.+?)(?:\n|$)",  # "Title:" followed by text
    r"^([A-Z].*?[.!?])\s*\n",  # Sentence-case first line ending with punctuation
]

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

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_embedding_dimension() -> int:
    """Get the embedding dimension for the configured model."""
    return EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL_NAME, 384)


def validate_file_size(file_size: int) -> bool:
    """Check if file size is within allowed limits."""
    return file_size <= MAX_FILE_SIZE_BYTES


def get_supported_extensions() -> list[str]:
    """Get list of supported file extensions."""
    return SUPPORTED_FILE_TYPES.copy()


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


def get_api_headers() -> dict[str, str]:
    """Get API headers with optional authentication."""
    headers = {
        "User-Agent": "ManuscriptJournalMatcher/1.0 (mailto:researcher@example.com)"
    }
    
    if CROSSREF_MAILTO:
        headers["mailto"] = CROSSREF_MAILTO
    
    return headers


# Initialize directories on import
ensure_directories_exist()