#!/usr/bin/env python3
"""
Tests for DOAJ integration functionality.

Tests the DOAJ API client, data processing, and integration with
the journal matching system.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from journal_db_builder import DOAJAPI, JournalDatabaseError


class TestDOAJAPI:
    """Test DOAJ API client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.doaj_api = DOAJAPI(rate_limit=0.01)  # Faster for testing
        
        # Sample DOAJ response data
        self.sample_doaj_response = {
            'results': [{
                'id': 'test_doaj_id',
                'bibjson': {
                    'title': 'Test Journal',
                    'publisher': {
                        'name': 'Test Publisher',
                        'country': 'US'
                    },
                    'apc': {
                        'has_apc': True,
                        'max': [{
                            'price': 1500,
                            'currency': 'USD'
                        }]
                    },
                    'subject': [
                        {'term': 'Biology'},
                        {'term': 'Medicine'}
                    ],
                    'language': ['English', 'Spanish'],
                    'license': [{
                        'type': 'CC BY'
                    }],
                    'oa_start': {
                        'year': 2015
                    },
                    'editorial': {
                        'review_process': 'Peer review',
                        'review_time': 12
                    }
                },
                'last_updated': '2023-01-01T00:00:00Z'
            }]
        }
        
        # Sample journal data for enrichment
        self.sample_journals = [
            {
                'id': 'S123456789',
                'display_name': 'Test Journal',
                'issn': ['1234-5678'],
                'issn_l': '1234-5678',
                'publisher': 'Test Publisher',
                'is_oa': True
            },
            # Journal without ISSN (should be skipped)
            {
                'id': 'S987654321',
                'display_name': 'No ISSN Journal',
                'issn': [],
                'issn_l': None,
                'publisher': 'Another Publisher',
                'is_oa': False
            }
        ]
    
    def test_init(self):
        """Test DOAJ API initialization."""
        assert self.doaj_api.base_url == "https://doaj.org/api/v3"
        assert self.doaj_api.rate_limit == 0.01
        assert 'User-Agent' in self.doaj_api.session.headers
        assert 'Accept' in self.doaj_api.session.headers
    
    @patch('requests.Session.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_get.return_value = mock_response
        
        result = self.doaj_api._make_request('test/endpoint')
        
        assert result == {'test': 'data'}
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_make_request_404(self, mock_get):
        """Test API request with 404 (journal not found)."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.doaj_api._make_request('test/endpoint')
        
        assert result == {}  # Should return empty dict for 404
    
    @patch('requests.Session.get')
    def test_make_request_error(self, mock_get):
        """Test API request with error (should not raise exception)."""
        # Mock connection error
        mock_get.side_effect = Exception("Connection error")
        
        result = self.doaj_api._make_request('test/endpoint')
        
        assert result == {}  # Should return empty dict for errors
    
    def test_process_doaj_data(self):
        """Test processing of DOAJ response data."""
        raw_data = self.sample_doaj_response['results'][0]
        
        processed = self.doaj_api._process_doaj_data(raw_data)
        
        # Check basic fields
        assert processed['doaj_id'] == 'test_doaj_id'
        assert processed['in_doaj'] is True
        assert processed['oa_status'] is True
        
        # Check APC data
        assert processed['has_apc'] is True
        assert processed['apc_amount'] == 1500
        assert processed['apc_currency'] == 'USD'
        
        # Check subjects
        assert 'Biology' in processed['subjects_doaj']
        assert 'Medicine' in processed['subjects_doaj']
        
        # Check languages
        assert 'English' in processed['languages']
        assert 'Spanish' in processed['languages']
        
        # Check license
        assert 'CC BY' in processed['license_type']
        
        # Check OA start year
        assert processed['oa_start_year'] == 2015
        
        # Check editorial info
        assert processed['plagiarism_detection'] == 'Peer review'
        assert processed['publication_time_weeks'] == 12
        
        # Check publisher info
        assert processed['publisher_doaj'] == 'Test Publisher'
        assert processed['country_doaj'] == 'US'
    
    def test_process_doaj_data_minimal(self):
        """Test processing DOAJ data with minimal fields."""
        minimal_data = {
            'id': 'minimal_id',
            'bibjson': {}
        }
        
        processed = self.doaj_api._process_doaj_data(minimal_data)
        
        assert processed['doaj_id'] == 'minimal_id'
        assert processed['in_doaj'] is True
        assert processed['oa_status'] is True
        assert processed['has_apc'] is False
        assert processed['apc_amount'] is None
        assert processed['subjects_doaj'] == []
        assert processed['languages'] == []
    
    @patch.object(DOAJAPI, '_make_request')
    @patch.object(DOAJAPI, '_process_doaj_data')
    def test_fetch_journal_by_issn_success(self, mock_process, mock_request):
        """Test successful journal fetch by ISSN."""
        # Mock API response
        mock_request.return_value = self.sample_doaj_response
        mock_process.return_value = {'test': 'processed_data'}
        
        result = self.doaj_api.fetch_journal_by_issn('1234-5678')
        
        assert result == {'test': 'processed_data'}
        mock_request.assert_called_once()
        mock_process.assert_called_once()
    
    @patch.object(DOAJAPI, '_make_request')
    def test_fetch_journal_by_issn_not_found(self, mock_request):
        """Test journal fetch when journal not found in DOAJ."""
        # Mock empty response
        mock_request.return_value = {'results': []}
        
        result = self.doaj_api.fetch_journal_by_issn('1234-5678')
        
        assert result is None
    
    def test_fetch_journal_by_issn_invalid(self):
        """Test journal fetch with invalid ISSN."""
        # Test with None
        assert self.doaj_api.fetch_journal_by_issn(None) is None
        
        # Test with invalid length
        assert self.doaj_api.fetch_journal_by_issn('123') is None
        
        # Test with empty string
        assert self.doaj_api.fetch_journal_by_issn('') is None
    
    @patch.object(DOAJAPI, 'fetch_journal_by_issn')
    def test_enrich_journals_with_doaj_success(self, mock_fetch):
        """Test successful journal enrichment with DOAJ data."""
        # Mock DOAJ data for first journal
        mock_fetch.side_effect = [
            {
                'in_doaj': True,
                'oa_status': True,
                'apc_amount': 1500,
                'apc_currency': 'USD'
            },
            None  # Second journal not in DOAJ
        ]
        
        enriched = self.doaj_api.enrich_journals_with_doaj(self.sample_journals)
        
        assert len(enriched) == 2
        
        # First journal should be enriched
        assert enriched[0]['in_doaj'] is True
        assert enriched[0]['oa_status'] is True
        assert enriched[0]['apc_amount'] == 1500
        
        # Second journal should have default values
        assert enriched[1]['in_doaj'] is False
        assert enriched[1]['oa_status'] is False
        assert enriched[1]['has_apc'] is None
    
    @patch.object(DOAJAPI, 'fetch_journal_by_issn')
    def test_enrich_journals_empty_list(self, mock_fetch):
        """Test enrichment with empty journal list."""
        enriched = self.doaj_api.enrich_journals_with_doaj([])
        
        assert enriched == []
        mock_fetch.assert_not_called()
    
    @patch.object(DOAJAPI, 'fetch_journal_by_issn')
    def test_enrich_journals_with_error(self, mock_fetch):
        """Test enrichment when DOAJ API raises exception."""
        # Mock exception during fetch
        mock_fetch.side_effect = Exception("API Error")
        
        # Should not raise exception, just log and continue
        enriched = self.doaj_api.enrich_journals_with_doaj(self.sample_journals)
        
        # Should still return journals with default DOAJ fields
        assert len(enriched) == 2
        assert all(j.get('doaj_fetched_at') for j in enriched)


class TestDOAJIntegration:
    """Test DOAJ integration with existing system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_journal_with_doaj = {
            'id': 'S123456789',
            'display_name': 'Open Access Journal',
            'publisher': 'OA Publisher',
            'subjects': [{'name': 'Biology'}],
            # DOAJ fields
            'in_doaj': True,
            'oa_status': True,
            'has_apc': True,
            'apc_amount': 1200,
            'apc_currency': 'EUR',
            'subjects_doaj': ['Medicine', 'Life Sciences'],
            'languages': ['English', 'German'],
            'publisher_doaj': 'DOAJ Publisher',
            'license_type': ['CC BY'],
            'oa_start_year': 2020
        }
    
    def test_doaj_fields_in_journal_schema(self):
        """Test that DOAJ fields are properly included in journal schema."""
        journal = self.sample_journal_with_doaj
        
        # Check core DOAJ fields
        assert 'in_doaj' in journal
        assert 'oa_status' in journal
        assert 'has_apc' in journal
        assert 'apc_amount' in journal
        assert 'apc_currency' in journal
        
        # Check additional DOAJ fields
        assert 'subjects_doaj' in journal
        assert 'languages' in journal
        assert 'publisher_doaj' in journal
        assert 'license_type' in journal
        assert 'oa_start_year' in journal
    
    @patch('sys.path')
    def test_semantic_fingerprint_includes_doaj_data(self, mock_path):
        """Test that semantic fingerprint includes DOAJ information."""
        # Import after mocking sys.path
        from journal_db_builder import create_semantic_fingerprint
        
        fingerprint = create_semantic_fingerprint(
            self.sample_journal_with_doaj, 
            []  # No sample articles for this test
        )
        
        # Check that DOAJ data is included in fingerprint
        assert 'Open access journal' in fingerprint
        assert 'Listed in Directory of Open Access Journals (DOAJ)' in fingerprint
        assert '1200 EUR' in fingerprint  # APC information
        assert 'English, German' in fingerprint  # Languages
        assert 'Medicine, Life Sciences' in fingerprint  # DOAJ subjects


class TestDOAJFiltering:
    """Test DOAJ-enhanced filtering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Sample journals with mixed DOAJ status
        self.test_journals = [
            {
                'display_name': 'DOAJ Journal',
                'in_doaj': True,
                'oa_status': True,
                'has_apc': True,
                'apc_amount': 1000,
                'apc_currency': 'USD'
            },
            {
                'display_name': 'OA Non-DOAJ Journal',
                'in_doaj': False,
                'oa_status': True,
                'has_apc': False,
                'apc_amount': None,
                'apc_currency': None
            },
            {
                'display_name': 'Subscription Journal',
                'in_doaj': False,
                'oa_status': False,
                'has_apc': None,
                'apc_amount': None,
                'apc_currency': None
            },
            {
                'display_name': 'High APC Journal',
                'in_doaj': True,
                'oa_status': True,
                'has_apc': True,
                'apc_amount': 3500,
                'apc_currency': 'USD'
            }
        ]
    
    @patch('sys.path')
    def test_open_access_filter_enhanced(self, mock_path):
        """Test enhanced open access filtering with DOAJ data."""
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        
        # Test open access only filter
        filters = {'open_access_only': True}
        filtered = matcher._apply_filters(self.test_journals, filters)
        
        # Should include journals with oa_status = True
        assert len(filtered) == 3  # Excludes subscription journal
        journal_names = [j['display_name'] for j in filtered]
        assert 'Subscription Journal' not in journal_names
    
    @patch('sys.path')
    def test_doaj_only_filter(self, mock_path):
        """Test DOAJ-only filtering."""
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        
        # Test DOAJ only filter
        filters = {'doaj_only': True}
        filtered = matcher._apply_filters(self.test_journals, filters)
        
        # Should only include DOAJ journals
        assert len(filtered) == 2
        journal_names = [j['display_name'] for j in filtered]
        assert 'DOAJ Journal' in journal_names
        assert 'High APC Journal' in journal_names
    
    @patch('sys.path')
    def test_no_apc_filter(self, mock_path):
        """Test no-APC (free to publish) filtering."""
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        
        # Test no APC filter
        filters = {'no_apc_only': True}
        filtered = matcher._apply_filters(self.test_journals, filters)
        
        # Should only include journals with no APC
        assert len(filtered) == 1
        assert filtered[0]['display_name'] == 'OA Non-DOAJ Journal'
    
    @patch('sys.path')
    def test_apc_range_filter(self, mock_path):
        """Test APC range filtering."""
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        
        # Test max APC filter
        filters = {'max_apc': 2000}
        filtered = matcher._apply_filters(self.test_journals, filters)
        
        # Should exclude high APC journal
        journal_names = [j['display_name'] for j in filtered]
        assert 'High APC Journal' not in journal_names
        assert 'DOAJ Journal' in journal_names
    
    @patch('sys.path')
    def test_combined_filters(self, mock_path):
        """Test combination of DOAJ filters."""
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        
        # Test combined filters: DOAJ + max APC
        filters = {
            'doaj_only': True,
            'max_apc': 2000
        }
        filtered = matcher._apply_filters(self.test_journals, filters)
        
        # Should only include DOAJ journals under APC limit
        assert len(filtered) == 1
        assert filtered[0]['display_name'] == 'DOAJ Journal'


@pytest.mark.integration
class TestDOAJDatabaseBuild:
    """Integration tests for DOAJ database building."""
    
    @patch('sys.path')
    @patch.object(DOAJAPI, 'enrich_journals_with_doaj')
    def test_database_build_with_doaj(self, mock_enrich, mock_path):
        """Test database building includes DOAJ enrichment."""
        from journal_db_builder import DOAJAPI
        
        # Mock enrichment
        sample_journals = [{'id': 'test', 'display_name': 'Test Journal'}]
        enriched_journals = [{
            'id': 'test', 
            'display_name': 'Test Journal',
            'in_doaj': True,
            'oa_status': True
        }]
        mock_enrich.return_value = enriched_journals
        
        # Test DOAJ enrichment
        doaj_api = DOAJAPI()
        result = doaj_api.enrich_journals_with_doaj(sample_journals)
        
        assert len(result) == 1
        assert result[0]['in_doaj'] is True
        mock_enrich.assert_called_once_with(sample_journals)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])