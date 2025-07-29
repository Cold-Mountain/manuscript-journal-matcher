"""
Utility functions and helpers for Manuscript Journal Matcher.

This module provides common utilities for file handling, data validation,
text processing, and other shared functionality across the application.
"""

import logging
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import re
from datetime import datetime, timedelta
import tempfile
import shutil

try:
    from .config import (
        MAX_FILE_SIZE_BYTES,
        SUPPORTED_FILE_TYPES,
        API_CACHE_DIR,
        CACHE_DURATION_HOURS
    )
except ImportError:
    from config import (
        MAX_FILE_SIZE_BYTES,
        SUPPORTED_FILE_TYPES,
        API_CACHE_DIR,
        CACHE_DURATION_HOURS
    )

# Set up logging
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class FileError(Exception):
    """Custom exception for file handling errors."""
    pass


def validate_file(file_path: Union[str, Path], 
                 check_size: bool = True, 
                 check_extension: bool = True) -> Dict[str, Any]:
    """
    Validate a file for processing.
    
    Args:
        file_path: Path to the file to validate
        check_size: Whether to check file size limits
        check_extension: Whether to check file extension
        
    Returns:
        Dictionary with validation results and file info
        
    Raises:
        ValidationError: If validation fails
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")
    
    # Get file info
    file_info = {
        'path': str(file_path),
        'name': file_path.name,
        'extension': file_path.suffix.lower(),
        'size_bytes': file_path.stat().st_size,
        'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
        'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime),
        'is_valid': True,
        'validation_errors': []
    }
    
    # Check file size
    if check_size and file_info['size_bytes'] > MAX_FILE_SIZE_BYTES:
        error_msg = (f"File too large: {file_info['size_mb']}MB "
                    f"(max: {MAX_FILE_SIZE_BYTES / (1024 * 1024)}MB)")
        file_info['validation_errors'].append(error_msg)
        file_info['is_valid'] = False
    
    # Check file extension
    if check_extension and file_info['extension'] not in SUPPORTED_FILE_TYPES:
        error_msg = (f"Unsupported file type: {file_info['extension']} "
                    f"(supported: {', '.join(SUPPORTED_FILE_TYPES)})")
        file_info['validation_errors'].append(error_msg)
        file_info['is_valid'] = False
    
    # Check if file is readable
    try:
        with open(file_path, 'rb') as f:
            f.read(1)  # Try to read first byte
    except PermissionError:
        error_msg = f"Permission denied reading file: {file_path}"
        file_info['validation_errors'].append(error_msg)
        file_info['is_valid'] = False
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        file_info['validation_errors'].append(error_msg)
        file_info['is_valid'] = False
    
    if not file_info['is_valid']:
        logger.error(f"File validation failed: {file_info['validation_errors']}")
        raise ValidationError(f"File validation failed: {'; '.join(file_info['validation_errors'])}")
    
    logger.debug(f"File validation passed: {file_path.name} ({file_info['size_mb']}MB)")
    return file_info


def clean_text(text: str, 
               remove_extra_whitespace: bool = True,
               remove_special_chars: bool = False,
               max_length: Optional[int] = None) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text to clean
        remove_extra_whitespace: Whether to normalize whitespace
        remove_special_chars: Whether to remove special characters
        max_length: Maximum length to truncate to
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    cleaned = text
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
    
    # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    if remove_special_chars:
        cleaned = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-\(\)]', '', cleaned)
    
    # Truncate if needed
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + "..."
        logger.debug(f"Text truncated to {max_length} characters")
    
    return cleaned


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        top_k: Number of top keywords to return
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Clean text and convert to lowercase
    cleaned = clean_text(text, remove_special_chars=True).lower()
    
    # Split into words
    words = cleaned.split()
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'we', 'our', 'us', 'they', 'their',
        'them', 'i', 'me', 'my', 'you', 'your', 'he', 'him', 'his', 'she',
        'her', 'it', 'its', 'can', 'may', 'might', 'must', 'shall', 'not',
        'no', 'yes', 'all', 'any', 'some', 'each', 'every', 'both', 'either',
        'neither', 'one', 'two', 'three', 'first', 'second', 'last', 'next',
        'more', 'most', 'many', 'much', 'few', 'less', 'least', 'also',
        'however', 'therefore', 'thus', 'hence', 'although', 'though', 'while',
        'since', 'because', 'if', 'unless', 'when', 'where', 'how', 'why',
        'what', 'which', 'who', 'whom', 'whose', 'very', 'quite', 'rather',
        'too', 'so', 'such', 'just', 'only', 'even', 'still', 'yet', 'already',
        'now', 'then', 'here', 'there', 'up', 'down', 'over', 'under',
        'through', 'between', 'among', 'during', 'before', 'after', 'above',
        'below', 'from', 'into', 'onto', 'off', 'out', 'about', 'around'
    }
    
    # Filter words (length >= 3, not stop words, not purely numeric)
    filtered_words = []
    for word in words:
        if (len(word) >= 3 and 
            word not in stop_words and 
            not word.isdigit() and
            word.isalpha()):
            filtered_words.append(word)
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top k
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, count in sorted_words[:top_k]]
    
    logger.debug(f"Extracted {len(keywords)} keywords from {len(words)} words")
    return keywords


def compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of text for caching and comparison.
    
    Args:
        text: Input text
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def save_cache(cache_key: str, data: Any, expiry_hours: int = None) -> None:
    """
    Save data to cache with expiration.
    
    Args:
        cache_key: Unique key for the cached data
        data: Data to cache (must be JSON serializable)
        expiry_hours: Cache expiration in hours (defaults to config value)
    """
    if expiry_hours is None:
        expiry_hours = CACHE_DURATION_HOURS
    
    cache_data = {
        'data': data,
        'cached_at': datetime.now().isoformat(),
        'expires_at': (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
    }
    
    try:
        API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = API_CACHE_DIR / f"{cache_key}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Cached data with key: {cache_key}")
        
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_key}: {e}")


def load_cache(cache_key: str) -> Optional[Any]:
    """
    Load data from cache if not expired.
    
    Args:
        cache_key: Cache key to load
        
    Returns:
        Cached data if valid, None if expired or not found
    """
    try:
        cache_file = API_CACHE_DIR / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check expiration
        expires_at = datetime.fromisoformat(cache_data['expires_at'])
        if datetime.now() > expires_at:
            logger.debug(f"Cache expired: {cache_key}")
            cache_file.unlink()  # Delete expired cache
            return None
        
        logger.debug(f"Cache hit: {cache_key}")
        return cache_data['data']
        
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None


def clear_expired_cache() -> int:
    """
    Clear all expired cache files.
    
    Returns:
        Number of files cleared
    """
    if not API_CACHE_DIR.exists():
        return 0
    
    cleared_count = 0
    current_time = datetime.now()
    
    try:
        for cache_file in API_CACHE_DIR.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                expires_at = datetime.fromisoformat(cache_data['expires_at'])
                if current_time > expires_at:
                    cache_file.unlink()
                    cleared_count += 1
                    
            except Exception as e:
                logger.debug(f"Error processing cache file {cache_file}: {e}")
                # Delete corrupted cache files
                cache_file.unlink()
                cleared_count += 1
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
    
    if cleared_count > 0:
        logger.info(f"Cleared {cleared_count} expired cache files")
    
    return cleared_count


def create_temp_file(content: Union[str, bytes], suffix: str = "") -> Path:
    """
    Create a temporary file with given content.
    
    Args:
        content: Content to write to file
        suffix: File suffix/extension
        
    Returns:
        Path to temporary file
        
    Raises:
        FileError: If file creation fails
    """
    try:
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        temp_path = Path(temp_path)
        
        # Write content
        mode = 'wb' if isinstance(content, bytes) else 'w'
        encoding = None if isinstance(content, bytes) else 'utf-8'
        
        with open(temp_path, mode, encoding=encoding) as f:
            f.write(content)
        
        # Close file descriptor
        import os
        os.close(fd)
        
        logger.debug(f"Created temporary file: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to create temporary file: {e}")
        raise FileError(f"Temporary file creation failed: {e}")


def safe_delete_file(file_path: Union[str, Path]) -> bool:
    """
    Safely delete a file with error handling.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted file: {file_path}")
            return True
        return True  # File doesn't exist, consider it "deleted"
        
    except Exception as e:
        logger.warning(f"Failed to delete file {file_path}: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def timing_decorator(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.3f}s")
        return result
    
    return wrapper


def retry_with_backoff(max_retries: int = 3, 
                      base_delay: float = 1.0,
                      max_delay: float = 60.0,
                      backoff_multiplier: float = 2.0):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            raise last_exception
        
        return wrapper
    return decorator


def validate_json_data(data: Any, required_fields: List[str]) -> Dict[str, Any]:
    """
    Validate that JSON data contains required fields.
    
    Args:
        data: Data to validate
        required_fields: List of required field names
        
    Returns:
        Validation result dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    result = {
        'is_valid': True,
        'missing_fields': [],
        'invalid_fields': [],
        'warnings': []
    }
    
    if not isinstance(data, dict):
        raise ValidationError("Data must be a dictionary")
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            result['missing_fields'].append(field)
            result['is_valid'] = False
        elif data[field] is None:
            result['invalid_fields'].append(f"{field} is None")
            result['is_valid'] = False
        elif isinstance(data[field], str) and not data[field].strip():
            result['invalid_fields'].append(f"{field} is empty string")
            result['is_valid'] = False
    
    if not result['is_valid']:
        error_msg = "Validation failed: "
        if result['missing_fields']:
            error_msg += f"Missing fields: {', '.join(result['missing_fields'])}. "
        if result['invalid_fields']:
            error_msg += f"Invalid fields: {', '.join(result['invalid_fields'])}"
        
        raise ValidationError(error_msg)
    
    return result


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and logging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'current_directory': str(Path.cwd()),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add memory info if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['memory_total_gb'] = round(memory.total / (1024**3), 2)
        info['memory_available_gb'] = round(memory.available / (1024**3), 2)
        info['memory_percent_used'] = memory.percent
    except ImportError:
        info['memory_info'] = 'psutil not available'
    
    return info


def setup_logging(level: str = "INFO", 
                 log_file: Optional[Path] = None,
                 include_timestamp: bool = True) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        include_timestamp: Whether to include timestamps in log messages
    """
    # Create formatter
    if include_timestamp:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging configured at {level} level")