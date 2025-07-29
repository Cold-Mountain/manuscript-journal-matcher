"""
Document text extraction module for Manuscript Journal Matcher.

This module handles extraction of text, titles, and abstracts from
PDF and DOCX manuscript files using multiple parsing strategies.
"""

import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import pdfplumber
import fitz  # PyMuPDF
from docx import Document
try:
    from .config import (
        TITLE_PATTERNS, 
        ABSTRACT_PATTERNS, 
        ABSTRACT_KEYWORDS,
        INTRODUCTION_KEYWORDS,
        validate_file_size,
        get_supported_extensions
    )
except ImportError:
    from config import (
        TITLE_PATTERNS, 
        ABSTRACT_PATTERNS, 
        ABSTRACT_KEYWORDS,
        INTRODUCTION_KEYWORDS,
        validate_file_size,
        get_supported_extensions
    )

# Set up logging
logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Custom exception for document extraction errors."""
    pass


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Extract all text from a PDF file using pdfplumber with PyMuPDF fallback.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        ExtractionError: If text extraction fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ExtractionError(f"File not found: {file_path}")
    
    if not validate_file_size(file_path.stat().st_size):
        raise ExtractionError(f"File too large: {file_path}")
    
    # Primary extraction method using pdfplumber
    try:
        with pdfplumber.open(file_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            text = "\n".join(text_parts)
            if text.strip():
                logger.info(f"Successfully extracted text from PDF using pdfplumber: {file_path}")
                return text.strip()
    
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed for {file_path}: {e}")
    
    # Fallback extraction method using PyMuPDF
    try:
        pdf_document = fitz.open(file_path)
        text_parts = []
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
        
        pdf_document.close()
        text = "\n".join(text_parts)
        
        if text.strip():
            logger.info(f"Successfully extracted text from PDF using PyMuPDF: {file_path}")
            return text.strip()
    
    except Exception as e:
        logger.error(f"PyMuPDF extraction also failed for {file_path}: {e}")
    
    raise ExtractionError(f"Could not extract text from PDF: {file_path}")


def extract_text_from_docx(file_path: Union[str, Path]) -> str:
    """
    Extract all text from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text as a string
        
    Raises:
        ExtractionError: If text extraction fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ExtractionError(f"File not found: {file_path}")
    
    if not validate_file_size(file_path.stat().st_size):
        raise ExtractionError(f"File too large: {file_path}")
    
    try:
        doc = Document(file_path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())
        
        text = "\n".join(text_parts)
        
        if text.strip():
            logger.info(f"Successfully extracted text from DOCX: {file_path}")
            return text.strip()
        else:
            raise ExtractionError(f"No text content found in DOCX: {file_path}")
    
    except Exception as e:
        logger.error(f"DOCX extraction failed for {file_path}: {e}")
        raise ExtractionError(f"Could not extract text from DOCX: {file_path}")


def extract_title_and_abstract(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract title and abstract from document text using pattern matching.
    
    Args:
        text: Full document text
        
    Returns:
        Tuple of (title, abstract) - either may be None if not found
    """
    if not text or not text.strip():
        return None, None
    
    lines = text.split('\n')
    
    # Extract title using various patterns
    title = _extract_title(text, lines)
    
    # Extract abstract using various patterns  
    abstract = _extract_abstract(text, lines)
    
    logger.info(f"Extraction results - Title: {'Found' if title else 'Not found'}, "
                f"Abstract: {'Found' if abstract else 'Not found'}")
    
    return title, abstract


def _extract_title(text: str, lines: list[str]) -> Optional[str]:
    """
    Extract title from document text using multiple strategies.
    
    Args:
        text: Full document text
        lines: Text split into lines
        
    Returns:
        Extracted title or None
    """
    # Strategy 1: Try predefined title patterns
    for pattern in TITLE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            title = match.group(1).strip()
            if 10 <= len(title) <= 200:  # Reasonable title length
                return title
    
    # Strategy 2: Use first non-empty line as title
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and len(line) < 200:
            # Skip lines that look like headers, page numbers, etc.
            if not re.match(r'^(page|p\.|fig|figure|table|abstract)', line, re.IGNORECASE):
                return line
    
    # Strategy 3: Look for lines in title case or all caps
    for line in lines[:10]:  # Check first 10 lines only
        line = line.strip()
        if line and 10 <= len(line) <= 200:
            # Check if line is in title case or all caps
            if line.istitle() or line.isupper():
                return line
    
    return None


def _extract_abstract(text: str, lines: list[str]) -> Optional[str]:
    """
    Extract abstract from document text using multiple strategies.
    
    Args:
        text: Full document text
        lines: Text split into lines
        
    Returns:
        Extracted abstract or None
    """
    text_lower = text.lower()
    
    # Strategy 1: Look for explicit abstract section
    for keyword in ABSTRACT_KEYWORDS:
        # Find abstract section header
        abstract_start = text_lower.find(keyword)
        if abstract_start != -1:
            # Extract text after the abstract keyword
            abstract_text = text[abstract_start:]
            
            # Try to find the end of abstract (before introduction, keywords, etc.)
            for pattern in ABSTRACT_PATTERNS:
                match = re.search(pattern, abstract_text, re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1).strip()
                    # Remove the abstract keyword from the beginning if present
                    abstract = re.sub(r'^(abstract|summary|synopsis|overview|background):?\s*', 
                                    '', abstract, flags=re.IGNORECASE)
                    if 50 <= len(abstract) <= 3000:  # Reasonable abstract length
                        return abstract
    
    # Strategy 2: Look for text between title and introduction
    intro_keywords_pattern = '|'.join(INTRODUCTION_KEYWORDS)
    intro_match = re.search(f'({intro_keywords_pattern})', text_lower)
    
    if intro_match:
        potential_abstract = text[:intro_match.start()].strip()
        
        # Remove potential title (first few lines)
        abstract_lines = potential_abstract.split('\n')[2:]  # Skip first 2 lines
        abstract = '\n'.join(abstract_lines).strip()
        
        if 50 <= len(abstract) <= 3000:
            return abstract
    
    # Strategy 3: Extract first substantial paragraph (fallback)
    paragraphs = text.split('\n\n')
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if 100 <= len(paragraph) <= 3000:  # Substantial paragraph
            # Skip if it looks like a title or header
            if not paragraph.isupper() and not paragraph.istitle():
                return paragraph
    
    return None


def extract_manuscript_data(file_path: Union[str, Path]) -> Dict[str, Optional[str]]:
    """
    Main function to extract metadata from manuscript files.
    
    Args:
        file_path: Path to the manuscript file (.pdf or .docx)
        
    Returns:
        Dictionary containing extracted metadata:
        {
            'title': str or None,
            'abstract': str or None,
            'full_text': str,
            'file_type': str,
            'file_name': str
        }
        
    Raises:
        ExtractionError: If file processing fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ExtractionError(f"File does not exist: {file_path}")
    
    file_ext = file_path.suffix.lower()
    if file_ext not in get_supported_extensions():
        raise ExtractionError(f"Unsupported file type: {file_ext}. "
                            f"Supported types: {get_supported_extensions()}")
    
    logger.info(f"Processing manuscript file: {file_path}")
    
    try:
        # Extract text based on file type
        if file_ext == '.pdf':
            full_text = extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            full_text = extract_text_from_docx(file_path)
        else:
            raise ExtractionError(f"Unsupported file extension: {file_ext}")
        
        # Extract title and abstract
        title, abstract = extract_title_and_abstract(full_text)
        
        # Prepare result
        result = {
            'title': title,
            'abstract': abstract,
            'full_text': full_text,
            'file_type': file_ext,
            'file_name': file_path.name
        }
        
        logger.info(f"Successfully processed {file_path.name}: "
                   f"Title={'✓' if title else '✗'}, Abstract={'✓' if abstract else '✗'}")
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to process manuscript {file_path}: {e}")
        raise ExtractionError(f"Failed to process manuscript: {e}")


def validate_extracted_data(data: Dict[str, Optional[str]]) -> Dict[str, str]:
    """
    Validate and clean extracted manuscript data.
    
    Args:
        data: Dictionary containing extracted data
        
    Returns:
        Dictionary with validation results and warnings
    """
    warnings = []
    
    if not data.get('title'):
        warnings.append("No title found - using filename as fallback")
    
    if not data.get('abstract'):
        warnings.append("No abstract found - matching quality may be reduced")
    elif len(data['abstract']) < 50:
        warnings.append("Abstract is very short - matching quality may be reduced")
    elif len(data['abstract']) > 3000:
        warnings.append("Abstract is very long - may contain additional content")
    
    if not data.get('full_text'):
        warnings.append("No text content extracted from file")
    
    return {
        'status': 'valid' if data.get('full_text') else 'invalid',
        'warnings': warnings
    }