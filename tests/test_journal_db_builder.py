"""
Tests for journal database builder functionality.

This module contains unit tests for the journal_db_builder module,
testing OpenAlex API integration, semantic fingerprinting, and database creation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import numpy as np

from src.journal_db_builder import (
    OpenAlexAPI,
    create_semantic_fingerprint,
    build_journal_embeddings,
    save_journal_database,
    load_journal_database,
    JournalDatabaseError
)


class TestOpenAlexAPI:
    """Test OpenAlex API client functionality."""
    
    def test_api_initialization(self):
        """Test API client initialization."""
        api = OpenAlexAPI(rate_limit=0.1)
        assert api.base_url == "https://api.openalex.org"
        assert api.rate_limit == 0.1
        
    @patch('src.journal_db_builder.requests.Session.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response
        
        api = OpenAlexAPI()
        result = api._make_request("test_endpoint")
        
        assert result == {"test": "data"}
        mock_get.assert_called_once()
    
    @patch('src.journal_db_builder.requests.Session.get')
    def test_make_request_failure(self, mock_get):
        """Test API request failure handling."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        api = OpenAlexAPI()
        with pytest.raises(JournalDatabaseError, match="OpenAlex API request failed"):
            api._make_request("test_endpoint")
    
    def test_process_journal_data_valid(self):
        """Test processing valid journal data."""
        raw_data = {
            'id': 'https://openalex.org/S123456',
            'display_name': 'Test Journal',
            'issn_l': '1234-5678',
            'issn': ['1234-5678', '8765-4321'],
            'host_organization_name': 'Test Publisher',
            'is_oa': True,
            'works_count': 1000,
            'cited_by_count': 5000,
            'x_concepts': [
                {'display_name': 'Biology', 'score': 0.8},
                {'display_name': 'Medicine', 'score': 0.6}
            ]
        }
        
        api = OpenAlexAPI()
        result = api._process_journal_data(raw_data)
        
        assert result is not None
        assert result['id'] == 'S123456'
        assert result['display_name'] == 'Test Journal'
        assert result['issn'] == ['1234-5678', '8765-4321']
        assert result['publisher'] == 'Test Publisher'
        assert result['is_oa'] == True
        assert result['works_count'] == 1000
        assert len(result['subjects']) == 2
    
    def test_process_journal_data_invalid(self):
        """Test processing invalid journal data."""
        api = OpenAlexAPI()
        
        # Missing essential data
        result = api._process_journal_data({})
        assert result is None
        
        # Missing display name
        result = api._process_journal_data({'id': 'test'})
        assert result is None
    
    def test_clean_abstract(self):
        """Test abstract cleaning functionality."""
        api = OpenAlexAPI()
        
        # Valid abstract (needs to be >50 chars after cleaning)
        raw_abstract = "<p>This is a comprehensive test abstract with <b>HTML</b> tags that contains enough content to pass the length requirement.</p>"
        cleaned = api._clean_abstract(raw_abstract)
        expected = "This is a comprehensive test abstract with HTML tags that contains enough content to pass the length requirement."
        assert cleaned == expected
        
        # Too short abstract
        short_abstract = "Short"
        cleaned = api._clean_abstract(short_abstract)
        assert cleaned is None
        
        # None input
        cleaned = api._clean_abstract(None)
        assert cleaned is None


class TestSemanticFingerprinting:
    """Test semantic fingerprint creation."""
    
    def test_create_semantic_fingerprint_complete(self):
        """Test creating fingerprint with complete data."""
        journal_data = {
            'display_name': 'Nature',
            'publisher': 'Springer Nature',
            'subjects': [
                {'name': 'Biology', 'score': 0.8},
                {'name': 'Chemistry', 'score': 0.6}
            ]
        }
        
        sample_articles = [
            {
                'title': 'Sample Article 1',
                'abstract': 'This is a sample abstract about biology research.',
                'concepts': ['Biology', 'Research', 'Science']
            },
            {
                'title': 'Sample Article 2',
                'abstract': 'Another abstract about chemistry.',
                'concepts': ['Chemistry', 'Analysis']
            }
        ]
        
        fingerprint = create_semantic_fingerprint(journal_data, sample_articles)
        
        assert 'Journal: Nature' in fingerprint
        assert 'Publisher: Springer Nature' in fingerprint
        assert 'Biology, Chemistry' in fingerprint
        assert 'Sample Article 1' in fingerprint
        assert 'biology research' in fingerprint
    
    def test_create_semantic_fingerprint_minimal(self):
        """Test creating fingerprint with minimal data."""
        journal_data = {'display_name': 'Minimal Journal'}
        sample_articles = []
        
        fingerprint = create_semantic_fingerprint(journal_data, sample_articles)
        
        assert 'Journal: Minimal Journal' in fingerprint
        assert len(fingerprint) > 0


class TestEmbeddingGeneration:
    """Test embedding generation for journals."""
    
    @patch('src.journal_db_builder.embed_texts')
    def test_build_journal_embeddings_success(self, mock_embed_texts):
        """Test successful embedding generation."""
        # Mock embedding function
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_embed_texts.return_value = mock_embeddings
        
        journals = [
            {
                'id': 'J1',
                'display_name': 'Journal 1',
                'semantic_fingerprint': 'Test fingerprint 1'
            },
            {
                'id': 'J2',
                'display_name': 'Journal 2',
                'semantic_fingerprint': 'Test fingerprint 2'
            }
        ]
        
        updated_journals, embeddings = build_journal_embeddings(journals)
        
        assert len(updated_journals) == 2
        assert embeddings.shape == (2, 3)
        assert updated_journals[0]['embedding'] == [0.1, 0.2, 0.3]
        assert updated_journals[1]['embedding'] == [0.4, 0.5, 0.6]
        
        mock_embed_texts.assert_called_once()
    
    def test_build_journal_embeddings_no_fingerprints(self):
        """Test embedding generation with no valid fingerprints."""
        journals = [
            {'id': 'J1', 'display_name': 'Journal 1'},  # No fingerprint
            {'id': 'J2', 'display_name': 'Journal 2', 'semantic_fingerprint': ''}  # Empty fingerprint
        ]
        
        with pytest.raises(JournalDatabaseError, match="No valid semantic fingerprints found"):
            build_journal_embeddings(journals)


class TestDatabasePersistence:
    """Test database saving and loading functionality."""
    
    def test_save_and_load_database(self):
        """Test saving and loading journal database."""
        journals = [
            {
                'id': 'J1',
                'display_name': 'Test Journal 1',
                'embedding': [0.1, 0.2, 0.3],
                'issn': ['1234-5678']
            },
            {
                'id': 'J2',
                'display_name': 'Test Journal 2',
                'embedding': [0.4, 0.5, 0.6],
                'issn': ['8765-4321']
            }
        ]
        
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Mock the global path
            with patch('src.journal_db_builder.JOURNAL_METADATA_PATH', tmp_path):
                # Save database
                save_journal_database(journals, embeddings)
                
                # Verify file was created
                assert tmp_path.exists()
                
                # Load database
                loaded_journals, loaded_embeddings = load_journal_database()
                
                # Verify loaded data
                assert len(loaded_journals) == 2
                assert loaded_journals[0]['display_name'] == 'Test Journal 1'
                assert loaded_journals[1]['display_name'] == 'Test Journal 2'
                
                assert loaded_embeddings is not None
                assert loaded_embeddings.shape == (2, 3)
                np.testing.assert_array_equal(loaded_embeddings, embeddings)
        
        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_load_database_not_found(self):
        """Test loading database when file doesn't exist."""
        nonexistent_path = Path('/tmp/nonexistent_database.json')
        
        with patch('src.journal_db_builder.JOURNAL_METADATA_PATH', nonexistent_path):
            with pytest.raises(JournalDatabaseError, match="Journal database not found"):
                load_journal_database()
    
    def test_save_database_without_embeddings_array(self):
        """Test saving database without separate embeddings array."""
        journals = [
            {
                'id': 'J1',
                'display_name': 'Test Journal',
                'embedding': [0.1, 0.2, 0.3]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            with patch('src.journal_db_builder.JOURNAL_METADATA_PATH', tmp_path):
                save_journal_database(journals)  # No embeddings array
                
                # Verify file was created and contains correct metadata
                with open(tmp_path, 'r') as f:
                    data = json.load(f)
                
                assert data['total_journals'] == 1
                assert data['embedding_dimension'] == 3
                assert len(data['journals']) == 1
        
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @patch('src.journal_db_builder.OpenAlexAPI.fetch_journals')
    @patch('src.journal_db_builder.OpenAlexAPI.fetch_sample_articles')
    @patch('src.journal_db_builder.embed_texts')
    def test_end_to_end_workflow(self, mock_embed, mock_articles, mock_journals):
        """Test complete database building workflow."""
        # Mock journal data
        mock_journals.return_value = [
            {
                'id': 'S123',
                'display_name': 'Test Journal',
                'issn': ['1234-5678'],
                'publisher': 'Test Publisher',
                'subjects': [{'name': 'Biology', 'score': 0.8}]
            }
        ]
        
        # Mock sample articles
        mock_articles.return_value = [
            {
                'title': 'Sample Article',
                'abstract': 'This is a test abstract about biology.',
                'concepts': ['Biology', 'Research']
            }
        ]
        
        # Mock embeddings
        mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Create API instance and test workflow
        api = OpenAlexAPI()
        
        # Fetch journals
        journals = api.fetch_journals(limit=1)
        assert len(journals) == 1
        
        # Create fingerprints
        sample_articles = api.fetch_sample_articles('S123')
        fingerprint = create_semantic_fingerprint(journals[0], sample_articles)
        journals[0]['semantic_fingerprint'] = fingerprint
        
        # Generate embeddings
        journals_with_embeddings, embeddings = build_journal_embeddings(journals)
        
        # Verify results
        assert len(journals_with_embeddings) == 1
        assert journals_with_embeddings[0]['embedding'] == [0.1, 0.2, 0.3]
        assert embeddings.shape == (1, 3)
        
        # Verify fingerprint content
        assert 'Test Journal' in fingerprint
        assert 'biology' in fingerprint.lower()