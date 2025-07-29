"""
Integration tests for the complete Manuscript Journal Matcher workflow.

This module tests the entire pipeline from document upload to journal matching,
ensuring all components work together correctly.
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from extractor import extract_manuscript_data, validate_extracted_data
    from embedder import embed_text, get_model
    from journal_db_builder import load_journal_database, save_journal_database
    from match_journals import JournalMatcher, format_search_results
    from utils import validate_file, clean_text
    from config import get_embedding_dimension
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestEndToEndWorkflow:
    """Test the complete manuscript-to-journal matching workflow."""
    
    @pytest.fixture
    def sample_manuscript_content(self):
        """Sample manuscript content for testing."""
        return """
        Advanced Machine Learning Approaches for Medical Diagnosis
        
        Abstract: This research investigates the application of deep learning algorithms
        for automated medical diagnosis using electronic health records. We developed
        a novel neural network architecture that achieves 94.2% accuracy in predicting
        cardiovascular diseases from patient data. Our approach combines convolutional
        neural networks with attention mechanisms to identify relevant patterns in
        complex medical datasets. The results demonstrate significant improvements
        over traditional machine learning methods, with potential applications in
        clinical decision support systems.
        
        Keywords: machine learning, medical diagnosis, deep learning, cardiovascular disease
        
        1. Introduction
        The healthcare industry has witnessed rapid technological advancement in recent years...
        """
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Create a simple PDF-like content for testing."""
        # This would be binary PDF content in a real scenario
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    
    @pytest.fixture
    def temp_manuscript_file(self, sample_manuscript_content):
        """Create a temporary manuscript file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_manuscript_content)
            f.flush()
            yield Path(f.name)
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_journal_database(self):
        """Create a mock journal database for testing."""
        journals = [
            {
                'id': 'test1',
                'display_name': 'Journal of Medical AI',
                'publisher': 'Medical Publishing',
                'is_oa': True,
                'is_in_doaj': True,
                'apc_usd': 2000,
                'subjects': [{'name': 'Medicine', 'score': 0.9}],
                'semantic_fingerprint': 'Medical AI journal focusing on machine learning applications in healthcare',
                'embedding': np.random.random(get_embedding_dimension()).tolist()
            },
            {
                'id': 'test2',
                'display_name': 'Computer Science Review',
                'publisher': 'Tech Publishers',
                'is_oa': False,
                'is_in_doaj': False,
                'apc_usd': None,
                'subjects': [{'name': 'Computer Science', 'score': 0.95}],
                'semantic_fingerprint': 'Comprehensive reviews in computer science and artificial intelligence',
                'embedding': np.random.random(get_embedding_dimension()).tolist()
            }
        ]
        
        embeddings = np.array([journal['embedding'] for journal in journals])
        return journals, embeddings
    
    def test_complete_workflow_with_text_file(self, temp_manuscript_file, mock_journal_database):
        """Test the complete workflow with a text manuscript file."""
        journals, embeddings = mock_journal_database
        
        with patch('journal_db_builder.load_journal_database') as mock_load_db:
            mock_load_db.return_value = (journals, embeddings)
            
            # Step 1: File validation
            file_info = validate_file(temp_manuscript_file, check_extension=False)
            assert file_info['is_valid']
            assert file_info['size_bytes'] > 0
            
            # Step 2: Text extraction
            manuscript_data = extract_manuscript_data(str(temp_manuscript_file))
            assert manuscript_data['title'] is not None
            assert manuscript_data['abstract'] is not None
            assert len(manuscript_data['abstract']) > 100
            
            # Step 3: Data validation
            validation = validate_extracted_data(manuscript_data)
            assert validation['status'] in ['valid', 'warning']
            
            # Step 4: Text cleaning
            cleaned_abstract = clean_text(manuscript_data['abstract'])
            assert len(cleaned_abstract) > 0
            assert cleaned_abstract != manuscript_data['abstract']  # Should be cleaned
            
            # Step 5: Embedding generation
            abstract_embedding = embed_text(manuscript_data['abstract'])
            assert abstract_embedding.shape[0] == get_embedding_dimension()
            assert not np.all(abstract_embedding == 0)
            
            # Step 6: Journal matching
            matcher = JournalMatcher()
            matcher.load_database()
            
            results = matcher.search_similar_journals(
                query_text=manuscript_data['abstract'],
                top_k=2
            )
            
            assert len(results) > 0
            assert len(results) <= 2
            
            # Step 7: Results validation
            for result in results:
                assert 'display_name' in result
                assert 'similarity_score' in result
                assert 'publisher' in result
                assert 0 <= result['similarity_score'] <= 1
            
            # Step 8: Results formatting
            formatted_results = format_search_results(results)
            
            assert len(formatted_results) == len(results)
            for formatted in formatted_results:
                assert 'journal_name' in formatted
                assert 'similarity_score' in formatted
                assert 'rank' in formatted
                assert 'is_open_access' in formatted
    
    def test_workflow_error_handling(self):
        """Test error handling throughout the workflow."""
        # Test with non-existent file
        with pytest.raises(Exception):
            validate_file("non_existent_file.pdf")
        
        # Test with empty abstract
        with pytest.raises(Exception):
            embed_text("")
        
        # Test matcher without database
        matcher = JournalMatcher()
        with pytest.raises(Exception):
            matcher.search_similar_journals("test query")
    
    def test_workflow_with_different_file_types(self, sample_manuscript_content):
        """Test workflow with different file formats."""
        # Test with .txt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_manuscript_content)
            f.flush()
            
            try:
                file_info = validate_file(f.name, check_extension=False)
                assert file_info['is_valid']
                
                manuscript_data = extract_manuscript_data(f.name)
                assert manuscript_data['title'] is not None
            finally:
                Path(f.name).unlink(missing_ok=True)
    
    @patch('journal_db_builder.load_journal_database')
    def test_matcher_performance(self, mock_load_db, mock_journal_database):
        """Test matcher performance with multiple queries."""
        import time
        
        journals, embeddings = mock_journal_database
        mock_load_db.return_value = (journals, embeddings)
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        test_queries = [
            "machine learning medical diagnosis",
            "computer science artificial intelligence",
            "data analysis healthcare applications",
            "neural networks pattern recognition",
            "automated clinical decision support"
        ]
        
        start_time = time.time()
        
        for query in test_queries:
            results = matcher.search_similar_journals(query, top_k=1)
            assert len(results) > 0
        
        end_time = time.time()
        avg_time_per_query = (end_time - start_time) / len(test_queries)
        
        # Should process queries relatively quickly
        assert avg_time_per_query < 2.0  # Less than 2 seconds per query
    
    def test_results_quality_validation(self, mock_journal_database):
        """Test that search results are meaningful and well-ranked."""
        journals, embeddings = mock_journal_database
        
        with patch('journal_db_builder.load_journal_database') as mock_load_db:
            mock_load_db.return_value = (journals, embeddings)
            
            matcher = JournalMatcher()
            matcher.load_database()
            
            # Test medical query should rank medical journal higher
            medical_query = "machine learning applications in medical diagnosis and healthcare"
            results = matcher.search_similar_journals(medical_query, top_k=2)
            
            assert len(results) > 0
            
            # Results should be sorted by similarity score (descending)
            for i in range(len(results) - 1):
                assert results[i]['similarity_score'] >= results[i+1]['similarity_score']
            
            # Test that results have reasonable similarity scores
            for result in results:
                assert result['similarity_score'] > 0  # Should have some similarity
            
            # Test filtering functionality
            oa_results = matcher.search_similar_journals(
                medical_query, 
                top_k=2,
                filters={'open_access_only': True}
            )
            
            for result in oa_results:
                assert result.get('is_oa', False) or result.get('is_in_doaj', False)


class TestSystemIntegration:
    """Test integration between different system components."""
    
    @pytest.fixture
    def system_components(self):
        """Initialize all system components for testing."""
        return {
            'extractor_available': True,
            'embedder_available': True,
            'matcher_available': True,
            'database_available': True
        }
    
    def test_component_availability(self, system_components):
        """Test that all required components are available."""
        # Test extractor
        try:
            from extractor import extract_manuscript_data
            assert system_components['extractor_available']
        except ImportError:
            system_components['extractor_available'] = False
        
        # Test embedder
        try:
            from embedder import embed_text
            model = get_model()
            assert model is not None
            assert system_components['embedder_available']
        except ImportError:
            system_components['embedder_available'] = False
        
        # Test matcher
        try:
            from match_journals import JournalMatcher
            assert system_components['matcher_available']
        except ImportError:
            system_components['matcher_available'] = False
        
        # All components should be available
        assert all(system_components.values()), f"Missing components: {system_components}"
    
    def test_data_flow_consistency(self):
        """Test that data flows consistently between components."""
        sample_text = "This is a test abstract about machine learning in medical applications."
        
        # Test embedding consistency
        embedding1 = embed_text(sample_text)
        embedding2 = embed_text(sample_text)
        
        # Same text should produce same embeddings
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
        
        # Test that different texts produce different embeddings
        different_text = "This is about quantum computing and physics research."
        different_embedding = embed_text(different_text)
        
        # Should not be identical
        assert not np.allclose(embedding1, different_embedding)
    
    def test_configuration_consistency(self):
        """Test that all components use consistent configuration."""
        from config import get_embedding_dimension, EMBEDDING_MODEL_NAME
        
        # Test embedding dimension consistency
        expected_dim = get_embedding_dimension()
        
        # Test actual embedding dimension
        test_embedding = embed_text("test")
        assert test_embedding.shape[0] == expected_dim
        
        # Test model name consistency
        model = get_model()
        assert hasattr(model, '_model_name')
        assert model._model_name == EMBEDDING_MODEL_NAME


class TestPerformanceBenchmarks:
    """Performance benchmark tests for the system."""
    
    def test_embedding_performance(self):
        """Benchmark embedding generation performance."""
        import time
        
        test_texts = [
            "Short abstract.",
            "Medium length abstract with some technical terms and methodology description.",
            "Long abstract with detailed methodology, results, and conclusions. " * 10
        ]
        
        times = []
        for text in test_texts:
            start_time = time.time()
            embedding = embed_text(text)
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert embedding.shape[0] == get_embedding_dimension()
        
        # Performance should be reasonable
        assert max(times) < 5.0  # No single embedding should take more than 5 seconds
        assert sum(times) / len(times) < 2.0  # Average should be under 2 seconds
    
    def test_search_performance(self, mock_journal_database):
        """Benchmark search performance."""
        import time
        
        journals, embeddings = mock_journal_database
        
        with patch('journal_db_builder.load_journal_database') as mock_load_db:
            mock_load_db.return_value = (journals, embeddings)
            
            matcher = JournalMatcher()
            
            # Test database loading time
            start_time = time.time()
            matcher.load_database()
            load_time = time.time() - start_time
            
            assert load_time < 5.0  # Database loading should be fast
            
            # Test search time
            query = "machine learning medical diagnosis"
            
            start_time = time.time()
            results = matcher.search_similar_journals(query, top_k=5)
            search_time = time.time() - start_time
            
            assert search_time < 1.0  # Search should be very fast
            assert len(results) > 0
    
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load model and generate embeddings
            model = get_model()
            test_texts = ["test text"] * 100
            
            for text in test_texts:
                embed_text(text)
            
            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 1GB)
            assert memory_increase < 1000, f"Memory increased by {memory_increase:.1f}MB"
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])