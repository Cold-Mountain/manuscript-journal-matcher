# Document Extraction API Reference

The `extractor` module handles document text extraction from PDF and DOCX manuscript files using multiple parsing strategies for robust content extraction.

## 📋 Module Overview

```python
from src.extractor import (
    extract_manuscript_data,    # Main extraction function
    extract_text_from_pdf,      # PDF text extraction
    extract_text_from_docx,     # DOCX text extraction
    extract_title_and_abstract, # Parse title/abstract
    validate_extracted_data,    # Validate extraction results
    ExtractionError            # Custom exception
)
```

## 🚀 Quick Start

```python
from src.extractor import extract_manuscript_data

# Extract all data from manuscript
data = extract_manuscript_data("paper.pdf")

print(f"Title: {data['title']}")
print(f"Abstract: {data['abstract'][:100]}...")
print(f"File type: {data['file_type']}")
print(f"Text length: {len(data['full_text'])} characters")
```

## 📖 Core Functions

### extract_manuscript_data()

**Main function to extract metadata from manuscript files.**

```python
def extract_manuscript_data(file_path: Union[str, Path]) -> Dict[str, Optional[str]]
```

**Parameters:**
- `file_path` (str | Path): Path to the manuscript file (.pdf or .docx)

**Returns:**
- `Dict[str, Optional[str]]`: Dictionary containing extracted metadata:
  ```python
  {
      'title': str or None,        # Extracted title
      'abstract': str or None,     # Extracted abstract  
      'full_text': str,           # Complete document text
      'file_type': str,           # File extension (.pdf/.docx)
      'file_name': str            # Original filename
  }
  ```

**Raises:**
- `ExtractionError`: If file processing fails

**Example:**
```python
# Basic extraction
data = extract_manuscript_data("research_paper.pdf")

if data['title']:
    print(f"📄 Title: {data['title']}")
else:
    print("⚠️ No title found")

if data['abstract']:
    print(f"📝 Abstract: {len(data['abstract'])} characters")
    print(f"Preview: {data['abstract'][:200]}...")
else:
    print("⚠️ No abstract found - using full text for matching")

# Validation
from src.extractor import validate_extracted_data
validation = validate_extracted_data(data)
if validation['warnings']:
    for warning in validation['warnings']:
        print(f"⚠️ {warning}")
```

### extract_text_from_pdf()

**Extract all text from a PDF file using multiple extraction methods.**

```python
def extract_text_from_pdf(file_path: Union[str, Path]) -> str
```

**Parameters:**
- `file_path` (str | Path): Path to the PDF file

**Returns:**
- `str`: Extracted text as a string

**Raises:**
- `ExtractionError`: If text extraction fails

**Features:**
- **Primary Method**: pdfplumber for accurate text extraction
- **Fallback Method**: PyMuPDF for problematic PDFs
- **File Validation**: Size and existence checks
- **Error Recovery**: Automatic fallback between methods

**Example:**
```python
try:
    # Extract text from PDF
    text = extract_text_from_pdf("paper.pdf")
    print(f"Extracted {len(text)} characters")
    
    # First 500 characters
    print("Preview:", text[:500])
    
except ExtractionError as e:
    print(f"❌ PDF extraction failed: {e}")
    # Handle corrupted or unsupported PDF
```

### extract_text_from_docx()

**Extract all text from a DOCX file.**

```python
def extract_text_from_docx(file_path: Union[str, Path]) -> str
```

**Parameters:**
- `file_path` (str | Path): Path to the DOCX file

**Returns:**
- `str`: Extracted text as a string

**Raises:**
- `ExtractionError`: If text extraction fails

**Features:**
- **Paragraph Extraction**: Processes document paragraphs sequentially
- **Text Cleaning**: Removes empty paragraphs and extra whitespace
- **File Validation**: Size and accessibility checks

**Example:**
```python
try:
    # Extract from Word document
    text = extract_text_from_docx("manuscript.docx")
    
    # Count paragraphs (rough estimate)
    paragraphs = text.count('\n\n')
    print(f"Document has ~{paragraphs} paragraphs")
    
except ExtractionError as e:
    print(f"❌ DOCX extraction failed: {e}")
```

### extract_title_and_abstract()

**Extract title and abstract from document text using pattern matching.**

```python
def extract_title_and_abstract(text: str) -> Tuple[Optional[str], Optional[str]]
```

**Parameters:**
- `text` (str): Full document text

**Returns:**
- `Tuple[Optional[str], Optional[str]]`: (title, abstract) - either may be None

**Title Extraction Strategies:**
1. **Pattern Matching**: Uses predefined regex patterns
2. **First Line**: Uses first substantial line as title
3. **Title Case**: Looks for lines in title case or all caps

**Abstract Extraction Strategies:**
1. **Explicit Sections**: Searches for "Abstract:", "Summary:", etc.
2. **Section Detection**: Text between title and introduction
3. **First Paragraph**: Falls back to first substantial paragraph

**Example:**
```python
# Extract title and abstract from raw text
text = """
Machine Learning in Medical Diagnosis: A Comprehensive Review

Abstract: This study reviews the application of machine learning 
techniques in medical diagnosis, examining over 200 research papers
published between 2020-2024. We analyze the effectiveness of various
algorithms including neural networks, support vector machines, and
ensemble methods across different medical specialties.

1. Introduction
Machine learning has revolutionized many fields...
"""

title, abstract = extract_title_and_abstract(text)

print(f"📄 Title: {title}")
print(f"📝 Abstract: {abstract}")

# Output:
# 📄 Title: Machine Learning in Medical Diagnosis: A Comprehensive Review
# 📝 Abstract: This study reviews the application of machine learning...
```

### validate_extracted_data()

**Validate and clean extracted manuscript data.**

```python
def validate_extracted_data(data: Dict[str, Optional[str]]) -> Dict[str, str]
```

**Parameters:**
- `data` (Dict): Dictionary containing extracted data

**Returns:**
- `Dict[str, str]`: Dictionary with validation results and warnings:
  ```python
  {
      'status': 'valid' | 'invalid',
      'warnings': List[str]  # List of warning messages
  }
  ```

**Validation Checks:**
- **Title Presence**: Warns if no title found
- **Abstract Quality**: Checks length and content quality
- **Content Availability**: Ensures some text was extracted

**Example:**
```python
# Validate extraction results
data = extract_manuscript_data("paper.pdf")
validation = validate_extracted_data(data)

print(f"Status: {validation['status']}")

if validation['warnings']:
    print("⚠️ Warnings:")
    for warning in validation['warnings']:
        print(f"  • {warning}")
        
# Example output:
# Status: valid
# ⚠️ Warnings:
#   • Abstract is very short - matching quality may be reduced
```

## 🔧 Configuration

### Supported File Types

```python
from src.config import get_supported_extensions

supported = get_supported_extensions()
print(f"Supported formats: {supported}")
# Output: ['.pdf', '.docx']
```

### File Size Limits

```python
from src.config import MAX_FILE_SIZE_MB, validate_file_size

print(f"Maximum file size: {MAX_FILE_SIZE_MB}MB")

# Check specific file
file_size = 25 * 1024 * 1024  # 25MB in bytes
is_valid = validate_file_size(file_size)
print(f"25MB file valid: {is_valid}")
```

### Text Extraction Patterns

The module uses configurable regex patterns for content detection:

```python
from src.config import TITLE_PATTERNS, ABSTRACT_PATTERNS, ABSTRACT_KEYWORDS

# Title detection patterns
print("Title patterns:", len(TITLE_PATTERNS))

# Abstract detection keywords
print("Abstract keywords:", ABSTRACT_KEYWORDS)
# Output: ['abstract', 'summary', 'synopsis', 'overview', 'background']
```

## 🚨 Error Handling

### ExtractionError

Custom exception for document extraction errors:

```python
from src.extractor import ExtractionError

try:
    data = extract_manuscript_data("corrupted.pdf")
except ExtractionError as e:
    print(f"❌ Extraction failed: {e}")
    
    # Common error types:
    # - File not found: /path/to/file.pdf
    # - File too large: /path/to/file.pdf  
    # - Unsupported file type: .txt
    # - Could not extract text from PDF: /path/to/file.pdf
```

### Robust Extraction Pattern

```python
def robust_extract(file_path):
    """Robust extraction with fallbacks."""
    try:
        # Primary extraction
        data = extract_manuscript_data(file_path)
        
        # Validate results
        validation = validate_extracted_data(data)
        
        if validation['status'] == 'valid':
            return data
        else:
            print("⚠️ Extraction completed with warnings")
            return data
            
    except ExtractionError as e:
        print(f"❌ Extraction failed: {e}")
        
        # Fallback: return minimal data structure
        return {
            'title': None,
            'abstract': None,
            'full_text': '',
            'file_type': Path(file_path).suffix,
            'file_name': Path(file_path).name
        }
```

## 📊 Performance Considerations

### File Processing Times

| File Type | Size | Typical Time | Notes |
|-----------|------|--------------|-------|
| PDF (Text) | 1-5MB | 0.5-2s | Clean text extraction |
| PDF (Scanned) | 1-5MB | 2-10s | OCR required (not supported) |
| DOCX | 1-5MB | 0.2-1s | Fast paragraph extraction |

### Memory Usage

```python
# Efficient processing for large files
def process_large_file(file_path):
    """Process large files efficiently."""
    
    # Check file size first
    file_info = Path(file_path).stat()
    size_mb = file_info.st_size / (1024 * 1024)
    
    if size_mb > 25:
        print(f"⚠️ Large file ({size_mb:.1f}MB) - processing may be slow")
    
    # Extract with progress tracking
    start_time = time.time()
    data = extract_manuscript_data(file_path)
    processing_time = time.time() - start_time
    
    print(f"✅ Processed in {processing_time:.2f}s")
    return data
```

### Batch Processing

```python
def extract_batch(file_paths):
    """Extract data from multiple files."""
    results = []
    failed_files = []
    
    for file_path in file_paths:
        try:
            data = extract_manuscript_data(file_path)
            results.append({
                'file_path': file_path,
                'data': data,
                'status': 'success'
            })
        except ExtractionError as e:
            failed_files.append({
                'file_path': file_path,
                'error': str(e),
                'status': 'failed'
            })
    
    print(f"✅ Processed {len(results)} files successfully")
    if failed_files:
        print(f"❌ Failed to process {len(failed_files)} files")
    
    return results, failed_files
```

## 🎯 Best Practices

### 1. File Validation First

```python
from src.utils import validate_file

# Always validate before extraction
try:
    file_info = validate_file(file_path)
    print(f"✅ File valid: {file_info['name']} ({file_info['size_mb']}MB)")
    
    data = extract_manuscript_data(file_path)
    
except ValidationError as e:
    print(f"❌ File validation failed: {e}")
```

### 2. Handle Missing Content Gracefully

```python
def safe_extract_for_matching(file_path):
    """Extract content suitable for journal matching."""
    data = extract_manuscript_data(file_path)
    
    # Use abstract if available, otherwise use first part of full text
    matching_text = data['abstract']
    
    if not matching_text:
        # Use first 1000 characters of full text
        matching_text = data['full_text'][:1000]
        print("ℹ️ Using full text excerpt for matching (no abstract found)")
    
    return {
        'text_for_matching': matching_text,
        'title': data['title'] or data['file_name'],
        'original_data': data
    }
```

### 3. Progress Monitoring

```python
def extract_with_progress(file_paths):
    """Extract with progress monitoring."""
    from tqdm import tqdm
    
    results = []
    
    for file_path in tqdm(file_paths, desc="Extracting manuscripts"):
        try:
            data = extract_manuscript_data(file_path)
            results.append(data)
            
            # Log progress details
            tqdm.write(f"✅ {Path(file_path).name}: "
                      f"Title={'✓' if data['title'] else '✗'}, "
                      f"Abstract={'✓' if data['abstract'] else '✗'}")
                      
        except ExtractionError as e:
            tqdm.write(f"❌ {Path(file_path).name}: {e}")
    
    return results
```

---

*For more examples and advanced usage, see the [User Guide](../user/) and [Integration Examples](../examples/).*