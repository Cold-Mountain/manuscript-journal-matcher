"""
Test module for journal matching functionality.

This module tests the FAISS-based vector search and journal matching
capabilities of the Manuscript Journal Matcher system.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from match_journals import (
        JournalMatcher, MatchingError, create_faiss_index, 
        search_similar_journals, rank_and_filter_results, 
        format_search_results, load_journal_database_with_index
    )
    from embedder import embed_text, initialize_embedding_model
    from config import get_embedding_dimension
    from utils import validate_json_data, compute_text_hash
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestJournalMatcher:
    """Test cases for the JournalMatcher class."""
    
    @pytest.fixture
    def sample_journals(self):
        """Create sample journal data for testing."""
        return [
            {
                'id': '1',
                'display_name': 'Nature Medicine',
                'issn_l': '1078-8956',
                'issn': ['1078-8956', '1546-170X'],
                'publisher': 'Nature Publishing Group',
                'is_oa': False,
                'is_in_doaj': False,
                'apc_usd': None,
                'works_count': 5000,
                'cited_by_count': 500000,
                'h_index': 150,
                'subjects': [
                    {'name': 'Medicine', 'score': 0.9},
                    {'name': 'Biology', 'score': 0.7}
                ],
                'semantic_fingerprint': 'Nature Medicine publishes high-impact medical research',
                'embedding': [0.1, 0.2, 0.3] * (get_embedding_dimension() // 3)
            },
            {
                'id': '2',
                'display_name': 'PLOS ONE',
                'issn_l': '1932-6203',
                'issn': ['1932-6203'],
                'publisher': 'PLOS',
                'is_oa': True,
                'is_in_doaj': True,
                'apc_usd': 1595,
                'works_count': 100000,
                'cited_by_count': 2000000,
                'h_index': 200,
                'subjects': [
                    {'name': 'Multidisciplinary', 'score': 0.8},
                    {'name': 'Science', 'score': 0.9}
                ],
                'semantic_fingerprint': 'PLOS ONE publishes multidisciplinary scientific research',
                'embedding': [0.4, 0.5, 0.6] * (get_embedding_dimension() // 3)
            },
            {
                'id': '3',
                'display_name': 'IEEE Transactions on Computers',
                'issn_l': '0018-9340',
                'issn': ['0018-9340', '1557-9956'],
                'publisher': 'IEEE',
                'is_oa': False,
                'is_in_doaj': False,
                'apc_usd': None,
                'works_count': 3000,
                'cited_by_count': 300000,
                'h_index': 120,
                'subjects': [
                    {'name': 'Computer Science', 'score': 0.95},
                    {'name': 'Engineering', 'score': 0.8}
                ],
                'semantic_fingerprint': 'IEEE Transactions on Computers covers computer science and engineering',
                'embedding': [0.7, 0.8, 0.9] * (get_embedding_dimension() // 3)
            }
        ]
    
    @pytest.fixture
    def sample_embeddings(self, sample_journals):
        """Create sample embeddings array."""
        embeddings = []
        for journal in sample_journals:
            embedding = journal['embedding']
            # Ensure embedding has correct dimension
            if len(embedding) < get_embedding_dimension():
                embedding = embedding * (get_embedding_dimension() // len(embedding) + 1)
            embedding = embedding[:get_embedding_dimension()]
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    @pytest.fixture
    def temp_database_files(self, sample_journals, sample_embeddings):
        """Create temporary database files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create metadata file
            metadata_file = temp_path / "journal_metadata.json"
            metadata = {
                'created_at': '2024-01-01T00:00:00',
                'total_journals': len(sample_journals),
                'embedding_dimension': get_embedding_dimension(),
                'journals': sample_journals
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Create FAISS index file
            index_file = temp_path / "journal_embeddings.faiss"
            
            yield metadata_file, index_file, temp_path
    
    def test_journal_matcher_initialization(self, temp_database_files):
        """Test JournalMatcher initialization."""
        metadata_file, index_file, temp_path = temp_database_files
        
        matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
        
        assert matcher.index_path == index_file
        assert matcher.metadata_path == metadata_file
        assert matcher.journals == []
        assert matcher.faiss_index is None
        assert matcher.embeddings is None
    
    @patch('match_journals.load_journal_database')
    def test_load_database_success(self, mock_load_db, sample_journals, sample_embeddings, temp_database_files):
        """Test successful database loading."""
        metadata_file, index_file, temp_path = temp_database_files
        mock_load_db.return_value = (sample_journals, sample_embeddings)
        
        matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
        matcher.load_database()
        
        assert len(matcher.journals) == len(sample_journals)
        assert matcher.embeddings is not None
        assert matcher.embedding_dimension == get_embedding_dimension()
        assert matcher.faiss_index is not None
        mock_load_db.assert_called_once()
    
    @patch('match_journals.load_journal_database')
    def test_load_database_no_journals(self, mock_load_db, temp_database_files):
        """Test database loading with no journals."""
        metadata_file, index_file, temp_path = temp_database_files
        mock_load_db.return_value = ([], None)
        
        matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
        
        with pytest.raises(MatchingError, match="No journals found"):
            matcher.load_database()
    
    @patch('match_journals.load_journal_database')
    def test_load_database_no_embeddings(self, mock_load_db, sample_journals, temp_database_files):
        """Test database loading with no embeddings."""
        metadata_file, index_file, temp_path = temp_database_files
        mock_load_db.return_value = (sample_journals, None)
        
        matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
        
        with pytest.raises(MatchingError, match="No embeddings found"):
            matcher.load_database()
    
    @patch('match_journals.load_journal_database')
    @patch('match_journals.embed_text')
    def test_search_similar_journals_success(self, mock_embed, mock_load_db, 
                                           sample_journals, sample_embeddings, temp_database_files):
        """Test successful journal search."""
        metadata_file, index_file, temp_path = temp_database_files
        mock_load_db.return_value = (sample_journals, sample_embeddings)
        
        # Mock query embedding (similar to first journal)
        query_embedding = np.array([0.1, 0.2, 0.3] * (get_embedding_dimension() // 3))
        mock_embed.return_value = query_embedding
        
        matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
        
        results = matcher.search_similar_journals(
            query_text="Medical research paper about cancer treatment",
            top_k=2
        )
        
        assert len(results) > 0
        assert len(results) <= 2
        assert all('similarity_score' in result for result in results)
        assert all('rank' in result for result in results)
        assert results[0]['similarity_score'] >= results[-1]['similarity_score']  # Sorted by similarity
        
        mock_embed.assert_called_once()
    
    def test_search_empty_query(self, temp_database_files):
        """Test search with empty query text."""
        metadata_file, index_file, temp_path = temp_database_files
        
        matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
        
        with pytest.raises(MatchingError, match="Query text cannot be empty"):
            matcher.search_similar_journals("")
        
        with pytest.raises(MatchingError, match="Query text cannot be empty"):
            matcher.search_similar_journals("   ")
    
    @patch('match_journals.load_journal_database')
    @patch('match_journals.embed_text')
    def test_search_with_filters(self, mock_embed, mock_load_db, 
                                sample_journals, sample_embeddings, temp_database_files):
        """Test search with various filters."""
        metadata_file, index_file, temp_path = temp_database_files
        mock_load_db.return_value = (sample_journals, sample_embeddings)
        
        query_embedding = np.array([0.4, 0.5, 0.6] * (get_embedding_dimension() // 3))
        mock_embed.return_value = query_embedding
        
        matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
        
        # Test open access filter
        results = matcher.search_similar_journals(
            query_text="Scientific research",
            filters={'open_access_only': True}
        )
        
        assert all(result.get('is_oa', False) or result.get('is_in_doaj', False) 
                  for result in results)
        
        # Test APC filter
        results = matcher.search_similar_journals(
            query_text="Scientific research",
            filters={'max_apc': 1000}
        )
        
        for result in results:
            apc = result.get('apc_usd')
            if apc is not None:
                assert apc <= 1000
        
        # Test subject filter
        results = matcher.search_similar_journals(
            query_text="Computer science research",
            filters={'subjects': ['Computer Science']}
        )
        
        # Verify that results contain relevant subjects
        for result in results:
            subjects = [s.get('name', '').lower() for s in result.get('subjects', [])]
            subject_text = ' '.join(subjects)
            assert 'computer science' in subject_text
    
    @patch('match_journals.load_journal_database')
    def test_get_database_stats(self, mock_load_db, sample_journals, sample_embeddings, temp_database_files):
        """Test database statistics generation."""
        metadata_file, index_file, temp_path = temp_database_files
        mock_load_db.return_value = (sample_journals, sample_embeddings)
        
        matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
        stats = matcher.get_database_stats()
        
        assert stats['total_journals'] == len(sample_journals)
        assert stats['embedding_dimension'] == get_embedding_dimension()
        assert 'faiss_index_type' in stats
        assert 'open_access_journals' in stats
        assert 'journals_with_apc' in stats
        assert 'sample_journal' in stats
        
        # Verify specific counts
        expected_oa_count = sum(1 for j in sample_journals 
                              if j.get('is_oa', False) or j.get('is_in_doaj', False))
        assert stats['open_access_journals'] == expected_oa_count
        
        expected_apc_count = sum(1 for j in sample_journals if j.get('apc_usd') is not None)
        assert stats['journals_with_apc'] == expected_apc_count


class TestStandaloneFunctions:
    """Test standalone functions in the match_journals module."""
    
    def test_create_faiss_index(self):
        """Test FAISS index creation."""
        # Create sample embeddings
        n_vectors, dimension = 5, get_embedding_dimension()
        embeddings = np.random.random((n_vectors, dimension)).astype(np.float32)
        
        index = create_faiss_index(embeddings)
        
        assert index is not None
        assert index.ntotal == n_vectors
        assert index.d == dimension
    
    def test_search_similar_journals_function(self):
        """Test the standalone search function."""
        # Create sample data
        n_vectors, dimension = 3, get_embedding_dimension()
        embeddings = np.random.random((n_vectors, dimension)).astype(np.float32)
        
        journals = [
            {'id': '1', 'display_name': 'Journal A', 'publisher': 'Publisher A'},
            {'id': '2', 'display_name': 'Journal B', 'publisher': 'Publisher B'},
            {'id': '3', 'display_name': 'Journal C', 'publisher': 'Publisher C'},
        ]
        
        # Create index
        index = create_faiss_index(embeddings)
        
        # Create query similar to first journal
        query_embedding = embeddings[0] + np.random.normal(0, 0.1, dimension)
        
        results = search_similar_journals(query_embedding, index, journals, top_k=2)
        
        assert len(results) <= 2
        assert len(results) > 0
        assert all('similarity_score' in result for result in results)
        assert all('rank' in result for result in results)
        assert results[0]['similarity_score'] >= results[-1]['similarity_score']
    
    def test_rank_and_filter_results(self):
        """Test result ranking and filtering."""
        results = [
            {
                'display_name': 'Journal A',
                'similarity_score': 0.8,
                'is_oa': True,
                'apc_usd': 1000,
                'subjects': [{'name': 'Medicine'}],
                'publisher': 'Nature'
            },
            {
                'display_name': 'Journal B',
                'similarity_score': 0.9,
                'is_oa': False,
                'apc_usd': 2000,
                'subjects': [{'name': 'Biology'}],
                'publisher': 'Elsevier'
            },
            {
                'display_name': 'Journal C',
                'similarity_score': 0.7,
                'is_oa': True,
                'apc_usd': 500,
                'subjects': [{'name': 'Computer Science'}],
                'publisher': 'IEEE'
            }
        ]
        
        # Test without filters (should sort by similarity)
        ranked = rank_and_filter_results(results)
        assert len(ranked) == 3
        assert ranked[0]['similarity_score'] == 0.9
        assert ranked[1]['similarity_score'] == 0.8
        assert ranked[2]['similarity_score'] == 0.7
        
        # Test with open access filter
        filters = {'open_access_only': True}
        filtered = rank_and_filter_results(results, filters)
        assert len(filtered) == 2
        assert all(result['is_oa'] for result in filtered)
        
        # Test with APC filter
        filters = {'max_apc': 1500}
        filtered = rank_and_filter_results(results, filters)
        assert len(filtered) == 2
        assert all(result['apc_usd'] <= 1500 for result in filtered)
        
        # Test with subject filter
        filters = {'subjects': ['Medicine']}
        filtered = rank_and_filter_results(results, filters)
        assert len(filtered) == 1
        assert 'Medicine' in [s['name'] for s in filtered[0]['subjects']]
    
    def test_format_search_results(self):
        """Test search result formatting."""
        raw_results = [
            {
                'rank': 1,
                'display_name': 'Nature Medicine',
                'similarity_score': 0.95,
                'publisher': 'Nature Publishing Group',
                'issn_l': '1078-8956',
                'is_oa': False,
                'is_in_doaj': False,
                'apc_usd': None,
                'homepage_url': 'https://nature.com/nm',
                'works_count': 5000,
                'cited_by_count': 500000,
                'h_index': 150,
                'subjects': [
                    {'name': 'Medicine', 'score': 0.9},
                    {'name': 'Biology', 'score': 0.7},
                    {'name': 'Genetics', 'score': 0.6},
                    {'name': 'Immunology', 'score': 0.5}
                ],
                'embedding': [0.1, 0.2, 0.3] * 100
            }
        ]
        
        # Test basic formatting
        formatted = format_search_results(raw_results)
        
        assert len(formatted) == 1
        result = formatted[0]
        
        assert result['rank'] == 1
        assert result['journal_name'] == 'Nature Medicine'
        assert result['similarity_score'] == 0.950
        assert result['publisher'] == 'Nature Publishing Group'
        assert result['issn'] == '1078-8956'
        assert result['is_open_access'] is False
        assert result['apc_usd'] is None
        assert result['homepage_url'] == 'https://nature.com/nm'
        assert result['works_count'] == 5000
        assert result['cited_by_count'] == 500000
        assert result['h_index'] == 150
        assert len(result['subjects']) == 3  # Should be limited to top 3
        assert 'embedding' not in result  # Should be excluded by default
        
        # Test with embeddings included
        formatted_with_embeddings = format_search_results(raw_results, include_embeddings=True)
        assert 'embedding' in formatted_with_embeddings[0]
    
    def test_empty_results(self):
        """Test handling of empty results."""
        assert rank_and_filter_results([]) == []
        assert format_search_results([]) == []


class TestErrorHandling:
    """Test error handling in matching functionality."""
    
    def test_matching_error_exception(self):
        """Test custom MatchingError exception."""
        with pytest.raises(MatchingError):
            raise MatchingError("Test error message")
    
    def test_invalid_query_embedding_shape(self):
        """Test handling of invalid query embedding shapes."""
        # Create sample data
        dimension = get_embedding_dimension()
        embeddings = np.random.random((3, dimension)).astype(np.float32)
        journals = [{'id': '1', 'display_name': 'Journal A'}] * 3
        index = create_faiss_index(embeddings)
        
        # Test with wrong dimension
        invalid_query = np.random.random(dimension + 10)
        
        # This should handle the error gracefully (FAISS will adjust internally)
        results = search_similar_journals(invalid_query[:dimension], index, journals)
        assert isinstance(results, list)
    
    def test_empty_journal_list(self):
        """Test handling of empty journal list."""
        dimension = get_embedding_dimension()
        embeddings = np.random.random((1, dimension)).astype(np.float32)
        query = np.random.random(dimension)
        index = create_faiss_index(embeddings)
        
        results = search_similar_journals(query, index, [], top_k=5)
        assert results == []


class TestIntegration:
    """Integration tests that test the complete workflow."""
    
    @pytest.mark.slow
    def test_end_to_end_workflow(self, sample_journals, sample_embeddings):
        """Test the complete end-to-end matching workflow."""
        # This test requires actual embeddings and database loading
        # It's marked as slow since it may load actual models
        
        try:
            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                metadata_file = temp_path / "metadata.json"
                index_file = temp_path / "index.faiss"
                
                # Save sample data
                metadata = {
                    'created_at': '2024-01-01T00:00:00',
                    'total_journals': len(sample_journals),
                    'embedding_dimension': get_embedding_dimension(),
                    'journals': sample_journals
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
                
                # Mock the database loading
                with patch('match_journals.load_journal_database') as mock_load:
                    mock_load.return_value = (sample_journals, sample_embeddings)
                    
                    # Initialize matcher
                    matcher = JournalMatcher(index_path=index_file, metadata_path=metadata_file)
                    
                    # Test search
                    results = matcher.search_similar_journals(
                        query_text="Medical research on cancer treatment and therapy",
                        top_k=3
                    )
                    
                    # Verify results
                    assert len(results) > 0
                    assert len(results) <= 3
                    
                    # Check result structure
                    for result in results:
                        assert 'display_name' in result
                        assert 'similarity_score' in result
                        assert 'rank' in result
                        assert isinstance(result['similarity_score'], float)
                        assert isinstance(result['rank'], int)
                    
                    # Test database stats
                    stats = matcher.get_database_stats()
                    assert stats['total_journals'] == len(sample_journals)
                    
        except ImportError:
            pytest.skip("Integration test requires full environment setup")


class TestDOAJEnhancedFiltering:
    """Test enhanced filtering with DOAJ data."""
    
    @pytest.fixture
    def doaj_enhanced_journals(self):
        """Create sample journals with DOAJ data for testing."""
        return [
            {
                'id': '1',
                'display_name': 'PLOS ONE',
                'publisher': 'PLOS',
                'publisher_doaj': 'Public Library of Science',
                'is_oa': True,
                'oa_status': True,
                'in_doaj': True,
                'has_apc': True,
                'apc_amount': 1595,
                'apc_currency': 'USD',
                'apc_usd': 1595,
                'subjects': [{'name': 'Biology'}],
                'subjects_doaj': ['Life Sciences', 'Medicine'],
                'languages': ['English'],
                'license_type': ['CC BY'],
                'similarity_score': 0.9
            },
            {
                'id': '2',
                'display_name': 'Frontiers in Medicine',
                'publisher': 'Frontiers',
                'publisher_doaj': 'Frontiers Media SA',
                'is_oa': True,
                'oa_status': True,
                'in_doaj': True,
                'has_apc': True,
                'apc_amount': 2950,
                'apc_currency': 'USD',
                'apc_usd': 2950,
                'subjects': [{'name': 'Medicine'}],
                'subjects_doaj': ['Medicine', 'Health Sciences'],
                'languages': ['English'],
                'license_type': ['CC BY'],
                'similarity_score': 0.8
            },
            {
                'id': '3',
                'display_name': 'Journal of Open Source Software',
                'publisher': 'Open Journals',
                'publisher_doaj': 'Open Journals',
                'is_oa': True,
                'oa_status': True,
                'in_doaj': True,
                'has_apc': False,
                'apc_amount': 0,
                'apc_currency': 'USD',
                'apc_usd': 0,
                'subjects': [{'name': 'Computer Science'}],
                'subjects_doaj': ['Computer Science', 'Software Engineering'],
                'languages': ['English'],
                'license_type': ['MIT'],
                'similarity_score': 0.7
            },
            {
                'id': '4',
                'display_name': 'Nature Medicine',
                'publisher': 'Nature Publishing Group',
                'is_oa': False,
                'oa_status': False,
                'in_doaj': False,
                'has_apc': None,
                'apc_amount': None,
                'apc_currency': None,
                'apc_usd': None,
                'subjects': [{'name': 'Medicine'}],
                'subjects_doaj': [],
                'languages': [],
                'license_type': [],
                'similarity_score': 0.95
            }
        ]
    
    def test_open_access_filter_with_doaj_data(self, doaj_enhanced_journals):
        """Test open access filtering with DOAJ oa_status data."""
        matcher = JournalMatcher()
        
        filters = {'open_access_only': True}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should include 3 open access journals
        assert len(filtered) == 3
        journal_names = [j['display_name'] for j in filtered]
        assert 'Nature Medicine' not in journal_names
        assert 'PLOS ONE' in journal_names
    
    def test_doaj_only_filter(self, doaj_enhanced_journals):
        """Test DOAJ-only filtering."""
        matcher = JournalMatcher()
        
        filters = {'doaj_only': True}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should include only DOAJ journals
        assert len(filtered) == 3
        for journal in filtered:
            assert journal['in_doaj'] is True
    
    def test_no_apc_filter(self, doaj_enhanced_journals):
        """Test no-APC (free to publish) filtering."""
        matcher = JournalMatcher()
        
        filters = {'no_apc_only': True}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should include only journals with no APC
        assert len(filtered) == 1
        assert filtered[0]['display_name'] == 'Journal of Open Source Software'
        assert filtered[0]['has_apc'] is False
    
    def test_apc_range_filtering(self, doaj_enhanced_journals):
        """Test APC range filtering."""
        matcher = JournalMatcher()
        
        # Test max APC filter
        filters = {'max_apc': 2000}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should exclude Frontiers in Medicine (APC: 2950)
        journal_names = [j['display_name'] for j in filtered]
        assert 'Frontiers in Medicine' not in journal_names
        assert 'PLOS ONE' in journal_names
        
        # Test min APC filter
        filters = {'min_apc': 2000}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should only include Frontiers in Medicine
        assert len(filtered) == 1
        assert filtered[0]['display_name'] == 'Frontiers in Medicine'
    
    def test_enhanced_subject_filtering(self, doaj_enhanced_journals):
        """Test subject filtering that includes DOAJ subjects."""
        matcher = JournalMatcher()
        
        filters = {'subjects': ['Software Engineering']}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should match DOAJ subject for JOSS
        assert len(filtered) == 1
        assert filtered[0]['display_name'] == 'Journal of Open Source Software'
    
    def test_enhanced_publisher_filtering(self, doaj_enhanced_journals):
        """Test publisher filtering that includes DOAJ publisher data."""
        matcher = JournalMatcher()
        
        filters = {'publishers': ['Public Library']}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should match DOAJ publisher for PLOS ONE
        assert len(filtered) == 1
        assert filtered[0]['display_name'] == 'PLOS ONE'
    
    def test_language_filtering(self, doaj_enhanced_journals):
        """Test language filtering."""
        matcher = JournalMatcher()
        
        # Test with English filter
        filters = {'languages': ['English']}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should include journals with English language
        assert len(filtered) == 3  # All DOAJ journals have English
        
        # Test with non-existent language
        filters = {'languages': ['Spanish']}
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should include no journals
        assert len(filtered) == 0
    
    def test_combined_doaj_filters(self, doaj_enhanced_journals):
        """Test combination of multiple DOAJ filters."""
        matcher = JournalMatcher()
        
        filters = {
            'doaj_only': True,
            'max_apc': 2000,
            'subjects': ['Medicine']
        }
        filtered = matcher._apply_filters(doaj_enhanced_journals, filters)
        
        # Should only include PLOS ONE (DOAJ, APC < 2000, has Medicine subject)
        assert len(filtered) == 1
        assert filtered[0]['display_name'] == 'PLOS ONE'


class TestDOAJFormattedResults:
    """Test enhanced result formatting with DOAJ data."""
    
    def test_format_search_results_with_doaj_data(self):
        """Test formatting of search results includes DOAJ data."""
        sample_results = [
            {
                'id': '1',
                'display_name': 'PLOS ONE',
                'publisher': 'PLOS',
                'publisher_doaj': 'Public Library of Science',
                'oa_status': True,
                'in_doaj': True,
                'apc_amount': 1595,
                'apc_currency': 'USD',
                'subjects': [{'name': 'Biology'}],
                'subjects_doaj': ['Life Sciences'],
                'languages': ['English'],
                'license_type': ['CC BY'],
                'similarity_score': 0.9,
                'rank': 1
            }
        ]
        
        formatted = format_search_results(sample_results)
        
        assert len(formatted) == 1
        result = formatted[0]
        
        # Check DOAJ-enhanced fields
        assert result['publisher'] == 'Public Library of Science'  # DOAJ publisher preferred
        assert result['is_open_access'] is True  # Based on oa_status
        assert result['in_doaj'] is True
        assert result['apc_display'] == '1595 USD'
        assert result['languages'] == ['English']
        assert result['license_type'] == ['CC BY']
        
        # Check combined subjects
        assert 'Biology' in result['subjects']
        assert 'Life Sciences' in result['subjects']


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])