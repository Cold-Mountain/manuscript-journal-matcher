"""
Tests for document extraction functionality.

This module contains unit tests for the extractor module,
testing PDF/DOCX processing and text extraction capabilities.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.extractor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_title_and_abstract,
    extract_manuscript_data,
    validate_extracted_data,
    ExtractionError,
    _extract_title,
    _extract_abstract
)


class TestTextExtraction:
    """Test text extraction from PDF and DOCX files."""
    
    def test_extract_text_from_pdf_nonexistent_file(self):
        """Test PDF extraction with non-existent file."""
        with pytest.raises(ExtractionError, match="File not found"):
            extract_text_from_pdf("nonexistent.pdf")
    
    def test_extract_text_from_docx_nonexistent_file(self):
        """Test DOCX extraction with non-existent file."""
        with pytest.raises(ExtractionError, match="File not found"):
            extract_text_from_docx("nonexistent.docx")
    
    @patch('src.extractor.validate_file_size')
    def test_file_size_validation(self, mock_validate):
        """Test file size validation."""
        mock_validate.return_value = False
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_file:
            with pytest.raises(ExtractionError, match="File too large"):
                extract_text_from_pdf(tmp_file.name)
    
    @patch('pdfplumber.open')
    def test_pdf_extraction_success(self, mock_pdfplumber):
        """Test successful PDF text extraction."""
        # Mock pdfplumber
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF text content"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_file:
            with patch('src.extractor.validate_file_size', return_value=True):
                result = extract_text_from_pdf(tmp_file.name)
                assert result == "Sample PDF text content"
    
    @patch('pdfplumber.open')
    @patch('fitz.open')
    def test_pdf_extraction_fallback_to_pymupdf(self, mock_fitz, mock_pdfplumber):
        """Test PDF extraction falling back to PyMuPDF when pdfplumber fails."""
        # Mock pdfplumber to fail
        mock_pdfplumber.side_effect = Exception("pdfplumber failed")
        
        # Mock PyMuPDF to succeed
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PyMuPDF extracted text"
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_file:
            with patch('src.extractor.validate_file_size', return_value=True):
                result = extract_text_from_pdf(tmp_file.name)
                assert result == "PyMuPDF extracted text"
    
    @patch('src.extractor.Document')
    def test_docx_extraction_success(self, mock_document):
        """Test successful DOCX text extraction."""
        # Mock docx Document
        mock_paragraph = MagicMock()
        mock_paragraph.text = "Sample DOCX paragraph"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.docx') as tmp_file:
            with patch('src.extractor.validate_file_size', return_value=True):
                result = extract_text_from_docx(tmp_file.name)
                assert result == "Sample DOCX paragraph"


class TestTitleExtraction:
    """Test title extraction from document text."""
    
    def test_extract_title_from_first_line(self):
        """Test title extraction from first line."""
        text = "This is a Sample Research Title\n\nAbstract: This is the abstract content..."
        lines = text.split('\n')
        title = _extract_title(text, lines)
        assert title == "This is a Sample Research Title"
    
    def test_extract_title_with_title_keyword(self):
        """Test title extraction with explicit title keyword."""
        text = "Title: Advanced Machine Learning Methods\n\nAbstract content follows..."
        lines = text.split('\n')
        title = _extract_title(text, lines)
        assert title == "Advanced Machine Learning Methods"
    
    def test_extract_title_too_short(self):
        """Test that very short titles are rejected."""
        text = "Short\n\nLonger content follows with more substantial text..."
        lines = text.split('\n')
        title = _extract_title(text, lines)
        # Should skip the short title and find a longer line
        assert title != "Short"
    
    def test_extract_title_too_long(self):
        """Test that very long titles are rejected."""
        long_title = "This is an extremely long title that goes on and on " * 10
        text = f"{long_title}\n\nNormal content follows..."
        lines = text.split('\n')
        title = _extract_title(text, lines)
        # Should not return the overly long title
        assert title != long_title
    
    def test_extract_title_none_found(self):
        """Test when no suitable title is found."""
        text = "page 1\nfig 1\nabstract\nshort content"
        lines = text.split('\n')
        title = _extract_title(text, lines)
        assert title is None


class TestAbstractExtraction:
    """Test abstract extraction from document text."""
    
    def test_extract_abstract_with_keyword(self):
        """Test abstract extraction with explicit abstract keyword."""
        text = """Title: Sample Paper
        
        Abstract: This is a comprehensive study examining the effects of 
        various methodological approaches on research outcomes. The study 
        demonstrates significant improvements in accuracy.
        
        Keywords: research, methodology, accuracy
        
        1. Introduction
        This paper presents..."""
        
        lines = text.split('\n')
        abstract = _extract_abstract(text, lines)
        assert abstract is not None
        assert "comprehensive study" in abstract
        assert "significant improvements" in abstract
    
    def test_extract_abstract_with_summary_keyword(self):
        """Test abstract extraction with summary keyword."""
        text = """Title: Research Paper
        
        Summary: This research investigates novel approaches to data processing
        and analysis. Results show marked improvements in processing efficiency
        and output quality compared to traditional methods.
        
        Introduction
        The field of data processing..."""
        
        lines = text.split('\n')
        abstract = _extract_abstract(text, lines)
        assert abstract is not None
        assert "novel approaches" in abstract
        assert "marked improvements" in abstract
    
    def test_extract_abstract_before_introduction(self):
        """Test abstract extraction from text before introduction."""
        text = """Sample Research Title
        
        This study presents a comprehensive analysis of machine learning
        algorithms applied to biomedical data. The research demonstrates
        significant improvements in diagnostic accuracy through novel
        feature selection techniques.
        
        1. Introduction
        Machine learning has become increasingly important..."""
        
        lines = text.split('\n')
        abstract = _extract_abstract(text, lines)
        assert abstract is not None
        assert "comprehensive analysis" in abstract
        assert "diagnostic accuracy" in abstract
    
    def test_extract_abstract_too_short(self):
        """Test that very short abstracts are rejected."""
        text = """Title
        
        Abstract: Short.
        
        Introduction
        Longer content..."""
        
        lines = text.split('\n')
        abstract = _extract_abstract(text, lines)
        # Should not return the very short abstract
        assert abstract != "Short."
    
    def test_extract_abstract_none_found(self):
        """Test when no suitable abstract is found."""
        text = "Title\nVery short content\nMore short text\nBrief note"
        lines = text.split('\n')
        abstract = _extract_abstract(text, lines)
        assert abstract is None


class TestTitleAndAbstractExtraction:
    """Test combined title and abstract extraction."""
    
    def test_extract_title_and_abstract_success(self):
        """Test successful extraction of both title and abstract."""
        text = """Advanced Machine Learning in Healthcare Applications
        
        Abstract: This research presents a novel approach to applying machine
        learning algorithms in healthcare diagnostics. The study evaluates
        multiple algorithmic approaches and demonstrates significant improvements
        in diagnostic accuracy and processing speed.
        
        Keywords: machine learning, healthcare, diagnostics
        
        1. Introduction
        The healthcare industry has seen rapid adoption..."""
        
        title, abstract = extract_title_and_abstract(text)
        
        assert title == "Advanced Machine Learning in Healthcare Applications"
        assert abstract is not None
        assert "novel approach" in abstract
        assert "diagnostic accuracy" in abstract
    
    def test_extract_title_and_abstract_partial(self):
        """Test extraction when only title or abstract is found."""
        text = """Research Paper Title That Is Sufficiently Long
        
        Short intro without clear abstract.
        
        Introduction
        The research begins..."""
        
        title, abstract = extract_title_and_abstract(text)
        
        assert title == "Research Paper Title That Is Sufficiently Long"
        assert abstract is None
    
    def test_extract_empty_text(self):
        """Test extraction from empty or whitespace text."""
        title, abstract = extract_title_and_abstract("")
        assert title is None
        assert abstract is None
        
        title, abstract = extract_title_and_abstract("   \n   \n   ")
        assert title is None
        assert abstract is None


class TestManuscriptDataExtraction:
    """Test the main manuscript data extraction function."""
    
    def test_unsupported_file_type(self):
        """Test extraction with unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp_file:
            with pytest.raises(ExtractionError, match="Unsupported file type"):
                extract_manuscript_data(tmp_file.name)
    
    def test_nonexistent_file(self):
        """Test extraction with non-existent file."""
        with pytest.raises(ExtractionError, match="File does not exist"):
            extract_manuscript_data("nonexistent.pdf")
    
    @patch('src.extractor.extract_text_from_pdf')
    @patch('src.extractor.extract_title_and_abstract')
    def test_successful_pdf_extraction(self, mock_extract_title, mock_extract_pdf):
        """Test successful PDF manuscript data extraction."""
        # Mock the extraction functions
        mock_extract_pdf.return_value = "Full document text content"
        mock_extract_title.return_value = ("Sample Title", "Sample Abstract")
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_file:
            result = extract_manuscript_data(tmp_file.name)
            
            assert result['title'] == "Sample Title"
            assert result['abstract'] == "Sample Abstract"
            assert result['full_text'] == "Full document text content"
            assert result['file_type'] == '.pdf'
            assert result['file_name'] == Path(tmp_file.name).name
    
    @patch('src.extractor.extract_text_from_docx')
    @patch('src.extractor.extract_title_and_abstract')
    def test_successful_docx_extraction(self, mock_extract_title, mock_extract_docx):
        """Test successful DOCX manuscript data extraction."""
        # Mock the extraction functions
        mock_extract_docx.return_value = "DOCX document content"
        mock_extract_title.return_value = ("DOCX Title", "DOCX Abstract")
        
        with tempfile.NamedTemporaryFile(suffix='.docx') as tmp_file:
            result = extract_manuscript_data(tmp_file.name)
            
            assert result['title'] == "DOCX Title"
            assert result['abstract'] == "DOCX Abstract"
            assert result['full_text'] == "DOCX document content"
            assert result['file_type'] == '.docx'


class TestValidation:
    """Test validation of extracted data."""
    
    def test_validate_complete_data(self):
        """Test validation of complete extracted data."""
        data = {
            'title': 'Sample Research Title',
            'abstract': 'This is a comprehensive abstract with sufficient length ' * 3,
            'full_text': 'Complete document text',
            'file_type': '.pdf',
            'file_name': 'sample.pdf'
        }
        
        validation = validate_extracted_data(data)
        assert validation['status'] == 'valid'
        assert len(validation['warnings']) == 0
    
    def test_validate_missing_title(self):
        """Test validation with missing title."""
        data = {
            'title': None,
            'abstract': 'Valid abstract with sufficient length for testing purposes.',
            'full_text': 'Complete document text',
            'file_type': '.pdf',
            'file_name': 'sample.pdf'
        }
        
        validation = validate_extracted_data(data)
        assert validation['status'] == 'valid'
        assert any('title' in warning.lower() for warning in validation['warnings'])
    
    def test_validate_missing_abstract(self):
        """Test validation with missing abstract."""
        data = {
            'title': 'Valid Title',
            'abstract': None,
            'full_text': 'Complete document text',
            'file_type': '.pdf',
            'file_name': 'sample.pdf'
        }
        
        validation = validate_extracted_data(data)
        assert validation['status'] == 'valid'
        assert any('abstract' in warning.lower() for warning in validation['warnings'])
    
    def test_validate_short_abstract(self):
        """Test validation with short abstract."""
        data = {
            'title': 'Valid Title',
            'abstract': 'Short abstract.',
            'full_text': 'Complete document text',
            'file_type': '.pdf',
            'file_name': 'sample.pdf'
        }
        
        validation = validate_extracted_data(data)
        assert validation['status'] == 'valid'
        assert any('very short' in warning.lower() for warning in validation['warnings'])
    
    def test_validate_no_text(self):
        """Test validation with no extracted text."""
        data = {
            'title': 'Valid Title',
            'abstract': 'Valid abstract',
            'full_text': None,
            'file_type': '.pdf',
            'file_name': 'sample.pdf'
        }
        
        validation = validate_extracted_data(data)
        assert validation['status'] == 'invalid'
        assert any('no text content' in warning.lower() for warning in validation['warnings'])