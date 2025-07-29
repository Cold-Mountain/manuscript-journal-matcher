"""
Tests for text embedding functionality.

This module contains unit tests for the embedder module,
testing embedding generation, similarity calculations, and model management.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from sentence_transformers import SentenceTransformer

from src.embedder import (
    initialize_embedding_model,
    get_model,
    embed_text,
    embed_texts,
    cosine_similarity_single,
    cosine_similarity_matrix,
    find_most_similar,
    validate_embedding_dimension,
    get_embedding_info,
    _preprocess_text,
    EmbeddingError
)


class TestModelInitialization:
    """Test embedding model initialization and management."""
    
    @patch('src.embedder.SentenceTransformer')
    @patch('src.embedder.torch.cuda.is_available')
    def test_initialize_embedding_model_cpu(self, mock_cuda, mock_st):
        """Test model initialization with CPU device."""
        mock_cuda.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        model = initialize_embedding_model("all-MiniLM-L6-v2")
        
        mock_st.assert_called_once_with("all-MiniLM-L6-v2", device='cpu')
        assert model._model_name == "all-MiniLM-L6-v2"
        assert model == mock_model
    
    @patch('src.embedder.SentenceTransformer')
    @patch('src.embedder.torch.cuda.is_available')
    def test_initialize_embedding_model_gpu(self, mock_cuda, mock_st):
        """Test model initialization with GPU device."""
        mock_cuda.return_value = True
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        model = initialize_embedding_model("all-MiniLM-L6-v2")
        
        mock_st.assert_called_once_with("all-MiniLM-L6-v2", device='cuda')
        assert model._model_name == "all-MiniLM-L6-v2"
    
    @patch('src.embedder.SentenceTransformer')
    def test_initialize_embedding_model_failure(self, mock_st):
        """Test model initialization failure handling."""
        mock_st.side_effect = Exception("Model loading failed")
        
        with pytest.raises(EmbeddingError, match="Could not load embedding model"):
            initialize_embedding_model("invalid-model")
    
    @patch('src.embedder._model_instance', None)
    @patch('src.embedder.initialize_embedding_model')
    def test_get_model_initializes_if_none(self, mock_init):
        """Test get_model initializes model if none exists."""
        mock_model = MagicMock()
        mock_init.return_value = mock_model
        
        result = get_model()
        
        mock_init.assert_called_once()
        assert result == mock_model


class TestTextEmbedding:
    """Test text embedding generation."""
    
    def test_embed_text_empty_string(self):
        """Test embedding empty or whitespace text raises error."""
        with pytest.raises(EmbeddingError, match="Cannot embed empty"):
            embed_text("")
        
        with pytest.raises(EmbeddingError, match="Cannot embed empty"):
            embed_text("   \n   ")
    
    @patch('src.embedder.get_model')
    @patch('src.embedder._preprocess_text')
    def test_embed_text_success(self, mock_preprocess, mock_get_model):
        """Test successful text embedding."""
        mock_preprocess.return_value = "cleaned text"
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_model.return_value = mock_model
        
        result = embed_text("sample text")
        
        mock_preprocess.assert_called_once_with("sample text")
        mock_model.encode.assert_called_once_with("cleaned text", convert_to_numpy=True)
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3]))
    
    @patch('src.embedder.get_model')
    def test_embed_text_with_2d_output(self, mock_get_model):
        """Test embedding with 2D output gets flattened."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])  # 2D array
        mock_get_model.return_value = mock_model
        
        result = embed_text("sample text")
        
        assert result.ndim == 1
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3]))
    
    @patch('src.embedder.get_model')
    def test_embed_text_model_failure(self, mock_get_model):
        """Test handling of model encoding failure."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model encoding failed")
        mock_get_model.return_value = mock_model
        
        with pytest.raises(EmbeddingError, match="Embedding generation failed"):
            embed_text("sample text")


class TestBatchEmbedding:
    """Test batch text embedding generation."""
    
    def test_embed_texts_empty_list(self):
        """Test embedding empty text list raises error."""
        with pytest.raises(EmbeddingError, match="Cannot embed empty text list"):
            embed_texts([])
    
    @patch('src.embedder.get_model')
    @patch('src.embedder._preprocess_text')
    def test_embed_texts_success(self, mock_preprocess, mock_get_model):
        """Test successful batch embedding."""
        mock_preprocess.side_effect = lambda x: f"cleaned {x}"
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = mock_model
        
        texts = ["text1", "text2"]
        result = embed_texts(texts, batch_size=1)
        
        # Check preprocessing was called for each text
        assert mock_preprocess.call_count == 2
        
        # Check model was called with cleaned texts
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert call_args == ["cleaned text1", "cleaned text2"]
        
        # Check result shape
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))
    
    @patch('src.embedder.get_model')
    @patch('src.embedder._preprocess_text')
    def test_embed_texts_with_empty_texts(self, mock_preprocess, mock_get_model):
        """Test batch embedding with some empty texts."""
        # Mock preprocessing to return empty for some texts
        mock_preprocess.side_effect = ["cleaned text1", "", "cleaned text2"]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = mock_model
        
        texts = ["text1", "", "text2"]
        result = embed_texts(texts)
        
        # Should have embeddings for all 3 texts (with zeros for empty)
        assert result.shape == (3, 2)
        # First and third should have actual embeddings
        np.testing.assert_array_equal(result[0], np.array([0.1, 0.2]))
        np.testing.assert_array_equal(result[2], np.array([0.3, 0.4]))
        # Second should be zeros
        np.testing.assert_array_equal(result[1], np.array([0.0, 0.0]))
    
    @patch('src.embedder.get_model')
    @patch('src.embedder._preprocess_text')
    def test_embed_texts_all_empty(self, mock_preprocess, mock_get_model):
        """Test batch embedding with all empty texts."""
        mock_preprocess.return_value = ""
        mock_get_model.return_value = MagicMock()
        
        with pytest.raises(EmbeddingError, match="No valid texts to embed"):
            embed_texts(["", "   ", "\n"])


class TestSimilarityCalculations:
    """Test similarity calculation functions."""
    
    def test_cosine_similarity_single_success(self):
        """Test successful cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        similarity = cosine_similarity_single(vec1, vec2)
        
        assert abs(similarity - 1.0) < 1e-6  # Should be very close to 1.0
    
    def test_cosine_similarity_single_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        
        similarity = cosine_similarity_single(vec1, vec2)
        
        assert abs(similarity - 0.0) < 1e-6  # Should be very close to 0.0
    
    def test_cosine_similarity_single_shape_mismatch(self):
        """Test cosine similarity with mismatched vector shapes."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        with pytest.raises(EmbeddingError, match="Vector shape mismatch"):
            cosine_similarity_single(vec1, vec2)
    
    def test_cosine_similarity_matrix_success(self):
        """Test similarity matrix calculation."""
        embeddings1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        embeddings2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        
        similarity_matrix = cosine_similarity_matrix(embeddings1, embeddings2)
        
        assert similarity_matrix.shape == (2, 3)
        # Check specific values
        assert abs(similarity_matrix[0, 0] - 1.0) < 1e-6  # Same vectors
        assert abs(similarity_matrix[0, 1] - 0.0) < 1e-6  # Orthogonal vectors
    
    def test_cosine_similarity_matrix_dimension_mismatch(self):
        """Test similarity matrix with dimension mismatch."""
        embeddings1 = np.array([[1.0, 0.0]])  # 2D
        embeddings2 = np.array([[1.0, 0.0, 0.0]])  # 3D
        
        with pytest.raises(EmbeddingError, match="Embedding dimension mismatch"):
            cosine_similarity_matrix(embeddings1, embeddings2)


class TestSimilaritySearch:
    """Test similarity search functionality."""
    
    def test_find_most_similar_success(self):
        """Test successful similarity search."""
        query = np.array([1.0, 0.0])
        candidates = np.array([
            [1.0, 0.0],    # Perfect match
            [0.0, 1.0],    # Orthogonal
            [0.5, 0.5],    # Partial match
        ])
        
        results = find_most_similar(query, candidates, top_k=2)
        
        assert len(results) == 2
        # First result should be perfect match (index 0)
        assert results[0][0] == 0
        assert abs(results[0][1] - 1.0) < 1e-6
        # Second result should be partial match (index 2)
        assert results[1][0] == 2
    
    def test_find_most_similar_query_wrong_dimension(self):
        """Test similarity search with wrong query dimension."""
        query = np.array([[1.0, 0.0]])  # 2D instead of 1D
        candidates = np.array([[1.0, 0.0]])
        
        with pytest.raises(EmbeddingError, match="Query embedding must be 1D"):
            find_most_similar(query, candidates)
    
    def test_find_most_similar_candidates_wrong_dimension(self):
        """Test similarity search with wrong candidate dimension."""
        query = np.array([1.0, 0.0])
        candidates = np.array([1.0, 0.0])  # 1D instead of 2D
        
        with pytest.raises(EmbeddingError, match="Candidate embeddings must be 2D"):
            find_most_similar(query, candidates)
    
    def test_find_most_similar_dimension_mismatch(self):
        """Test similarity search with dimension mismatch."""
        query = np.array([1.0, 0.0])  # 2D
        candidates = np.array([[1.0, 0.0, 0.0]])  # 3D
        
        with pytest.raises(EmbeddingError, match="Dimension mismatch"):
            find_most_similar(query, candidates)
    
    def test_find_most_similar_top_k_larger_than_candidates(self):
        """Test similarity search with top_k larger than candidate count."""
        query = np.array([1.0, 0.0])
        candidates = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        results = find_most_similar(query, candidates, top_k=10)
        
        # Should return only available candidates
        assert len(results) == 2


class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_preprocess_text_basic_cleaning(self):
        """Test basic text preprocessing."""
        text = "  This is   sample    text  \n\n  "
        result = _preprocess_text(text)
        assert result == "This is sample text"
    
    def test_preprocess_text_empty(self):
        """Test preprocessing empty text."""
        result = _preprocess_text("")
        assert result == ""
        
        result = _preprocess_text("   \n   ")
        assert result == ""
    
    def test_preprocess_text_truncation(self):
        """Test text truncation for long texts."""
        long_text = "word " * 1000  # Very long text
        result = _preprocess_text(long_text)
        
        # Should be truncated and end with "..."
        assert len(result) <= 2048 + 3  # Max length + "..."
        assert result.endswith("...")
    
    def test_validate_embedding_dimension_1d(self):
        """Test embedding dimension validation for 1D array."""
        embedding = np.array([0.1, 0.2, 0.3])  # 3D embedding
        
        assert validate_embedding_dimension(embedding, expected_dim=3) == True
        assert validate_embedding_dimension(embedding, expected_dim=4) == False
    
    def test_validate_embedding_dimension_2d(self):
        """Test embedding dimension validation for 2D array."""
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2x3 array
        
        assert validate_embedding_dimension(embeddings, expected_dim=3) == True
        assert validate_embedding_dimension(embeddings, expected_dim=2) == False
    
    def test_validate_embedding_dimension_wrong_shape(self):
        """Test embedding dimension validation for wrong shape."""
        embedding = np.array([[[0.1, 0.2]]])  # 3D array
        
        assert validate_embedding_dimension(embedding, expected_dim=2) == False
    
    @patch('src.embedder.get_model')
    def test_get_embedding_info_success(self, mock_get_model):
        """Test getting embedding model information."""
        mock_model = MagicMock()
        mock_model._model_name = "test-model"
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 512
        mock_model.device = "cpu"
        mock_get_model.return_value = mock_model
        
        info = get_embedding_info()
        
        assert info['model_name'] == "test-model"
        assert info['dimension'] == 384
        assert info['max_sequence_length'] == 512
        assert info['device'] == "cpu"
        assert info['loaded'] == True
    
    @patch('src.embedder.get_model')
    def test_get_embedding_info_failure(self, mock_get_model):
        """Test getting embedding model information when model fails to load."""
        mock_get_model.side_effect = Exception("Model loading failed")
        
        info = get_embedding_info()
        
        assert info['loaded'] == False
        assert 'error' in info
        assert "Model loading failed" in info['error']