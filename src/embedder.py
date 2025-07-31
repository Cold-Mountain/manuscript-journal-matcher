"""
Text embedding module for Manuscript Journal Matcher.

This module handles text embedding generation using sentence-transformers
and provides utilities for similarity calculations and batch processing.
"""

import os
# Disable tokenizer parallelism to prevent multiprocessing issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
import numpy as np
from typing import List, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .config import (
        EMBEDDING_MODEL_NAME,
        get_embedding_dimension
    )
except ImportError:
    from config import (
        EMBEDDING_MODEL_NAME,
        get_embedding_dimension
    )

# Set up logging
logger = logging.getLogger(__name__)

# Global model instance (loaded lazily)
_model_instance: Optional[SentenceTransformer] = None


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


def initialize_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Initialize and return the sentence transformer model.
    
    Args:
        model_name: Optional model name override. If None, uses config default.
        
    Returns:
        Initialized SentenceTransformer model
        
    Raises:
        EmbeddingError: If model initialization fails
    """
    global _model_instance
    
    if model_name is None:
        model_name = EMBEDDING_MODEL_NAME
    
    # Return cached instance if it matches the requested model
    if _model_instance is not None and hasattr(_model_instance, '_model_name'):
        if _model_instance._model_name == model_name:
            return _model_instance
    
    try:
        logger.info(f"Loading embedding model: {model_name}")
        
        # Set device (prefer GPU if available, but use CPU as fallback)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Initialize the model with settings to avoid multiprocessing issues
        model = SentenceTransformer(
            model_name, 
            device=device,
            tokenizer_kwargs={
                'clean_up_tokenization_spaces': False,
                'use_fast': False  # Use slower but more stable tokenizer
            }
        )
        
        # Store model name for future reference
        model._model_name = model_name
        
        # Cache the model instance
        _model_instance = model
        
        logger.info(f"Successfully loaded model {model_name} on {device}")
        logger.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")
        
        return model
    
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {model_name}: {e}")
        raise EmbeddingError(f"Could not load embedding model: {e}")


def get_model() -> SentenceTransformer:
    """
    Get the current embedding model instance, initializing if necessary.
    
    Returns:
        SentenceTransformer model instance
    """
    if _model_instance is None:
        return initialize_embedding_model()
    return _model_instance


def embed_text(text: str, model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Input text to embed
        model: Optional model instance. If None, uses default model.
        
    Returns:
        Numpy array containing the text embedding
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty or whitespace-only text")
    
    if model is None:
        model = get_model()
    
    try:
        # Clean and normalize text
        cleaned_text = _preprocess_text(text)
        
        # Generate embedding
        embedding = model.encode(cleaned_text, convert_to_numpy=True)
        
        # Ensure it's a 1D array
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        logger.debug(f"Generated embedding for text (length: {len(text)}, "
                    f"embedding shape: {embedding.shape})")
        
        return embedding
    
    except Exception as e:
        logger.error(f"Failed to generate embedding for text: {e}")
        raise EmbeddingError(f"Embedding generation failed: {e}")


def embed_texts(texts: List[str], model: Optional[SentenceTransformer] = None, 
                batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """
    Generate embeddings for multiple texts efficiently.
    
    Args:
        texts: List of input texts to embed
        model: Optional model instance. If None, uses default model.
        batch_size: Number of texts to process in each batch
        show_progress: Whether to show progress bar
        
    Returns:
        2D numpy array where each row is an embedding
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    if not texts:
        raise EmbeddingError("Cannot embed empty text list")
    
    if model is None:
        model = get_model()
    
    try:
        # Clean and normalize all texts
        cleaned_texts = [_preprocess_text(text) for text in texts]
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(cleaned_texts):
            if text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            raise EmbeddingError("No valid texts to embed after preprocessing")
        
        logger.info(f"Generating embeddings for {len(valid_texts)} texts "
                   f"(batch_size: {batch_size})")
        
        # Generate embeddings in batches
        embeddings = model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Handle case where we filtered out some texts
        if len(valid_indices) < len(texts):
            # Create full array with zeros for invalid texts
            full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
            full_embeddings[valid_indices] = embeddings
            embeddings = full_embeddings
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise EmbeddingError(f"Batch embedding generation failed: {e}")


def cosine_similarity_single(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
        
    Raises:
        EmbeddingError: If similarity calculation fails
    """
    if vec1.shape != vec2.shape:
        raise EmbeddingError(f"Vector shape mismatch: {vec1.shape} vs {vec2.shape}")
    
    try:
        # Reshape to 2D arrays for sklearn
        vec1_2d = vec1.reshape(1, -1)
        vec2_2d = vec2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vec1_2d, vec2_2d)[0, 0]
        
        return float(similarity)
    
    except Exception as e:
        logger.error(f"Failed to calculate cosine similarity: {e}")
        raise EmbeddingError(f"Similarity calculation failed: {e}")


def cosine_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity matrix between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings (N x D)
        embeddings2: Second set of embeddings (M x D)
        
    Returns:
        Similarity matrix (N x M) where entry (i,j) is similarity between 
        embeddings1[i] and embeddings2[j]
        
    Raises:
        EmbeddingError: If similarity calculation fails
    """
    if embeddings1.shape[1] != embeddings2.shape[1]:
        raise EmbeddingError(f"Embedding dimension mismatch: "
                           f"{embeddings1.shape[1]} vs {embeddings2.shape[1]}")
    
    try:
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
        logger.debug(f"Calculated similarity matrix shape: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    except Exception as e:
        logger.error(f"Failed to calculate similarity matrix: {e}")
        raise EmbeddingError(f"Similarity matrix calculation failed: {e}")


def find_most_similar(query_embedding: np.ndarray, 
                     candidate_embeddings: np.ndarray,
                     top_k: int = 10) -> List[Tuple[int, float]]:
    """
    Find the most similar embeddings to a query embedding.
    
    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: Array of candidate embeddings
        top_k: Number of top results to return
        
    Returns:
        List of (index, similarity_score) tuples, sorted by similarity descending
        
    Raises:
        EmbeddingError: If similarity search fails
    """
    if query_embedding.ndim != 1:
        raise EmbeddingError("Query embedding must be 1D array")
    
    if candidate_embeddings.ndim != 2:
        raise EmbeddingError("Candidate embeddings must be 2D array")
    
    if query_embedding.shape[0] != candidate_embeddings.shape[1]:
        raise EmbeddingError(f"Dimension mismatch: query {query_embedding.shape[0]} "
                           f"vs candidates {candidate_embeddings.shape[1]}")
    
    try:
        # Calculate similarities
        query_2d = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_2d, candidate_embeddings)[0]
        
        # Get top k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort descending
        
        # Create result list
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        logger.debug(f"Found top {len(results)} similar embeddings, "
                    f"best similarity: {results[0][1]:.4f}")
        
        return results
    
    except Exception as e:
        logger.error(f"Failed to find similar embeddings: {e}")
        raise EmbeddingError(f"Similarity search failed: {e}")


def _preprocess_text(text: str) -> str:
    """
    Clean and normalize text for embedding.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text ready for embedding
    """
    if not text:
        return ""
    
    # Basic cleaning
    cleaned = text.strip()
    
    # Remove excessive whitespace
    cleaned = ' '.join(cleaned.split())
    
    # Truncate if too long (most models have token limits)
    max_length = 512 * 4  # Rough estimate for token limit
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
        logger.warning(f"Text truncated to {max_length} characters")
    
    return cleaned


def validate_embedding_dimension(embedding: np.ndarray, expected_dim: Optional[int] = None) -> bool:
    """
    Validate that an embedding has the expected dimension.
    
    Args:
        embedding: Embedding array to validate
        expected_dim: Expected dimension. If None, uses config default.
        
    Returns:
        True if dimension is correct, False otherwise
    """
    if expected_dim is None:
        expected_dim = get_embedding_dimension()
    
    if embedding.ndim == 1:
        actual_dim = embedding.shape[0]
    elif embedding.ndim == 2:
        actual_dim = embedding.shape[1]
    else:
        return False
    
    return actual_dim == expected_dim


def get_embedding_info() -> dict:
    """
    Get information about the current embedding model.
    
    Returns:
        Dictionary with model information
    """
    try:
        model = get_model()
        return {
            'model_name': getattr(model, '_model_name', EMBEDDING_MODEL_NAME),
            'dimension': model.get_sentence_embedding_dimension(),
            'max_sequence_length': getattr(model, 'max_seq_length', 'Unknown'),
            'device': str(model.device),
            'loaded': True
        }
    except Exception as e:
        return {
            'model_name': EMBEDDING_MODEL_NAME,
            'dimension': get_embedding_dimension(),
            'error': str(e),
            'loaded': False
        }


def cleanup_model():
    """
    Clean up the global model instance to prevent resource leaks.
    """
    global _model_instance
    if _model_instance is not None:
        try:
            # Move model to CPU to free GPU memory
            if hasattr(_model_instance, 'to'):
                _model_instance.to('cpu')
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.debug("Cleaned up embedding model resources")
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
        finally:
            _model_instance = None


import atexit
# Register cleanup function to run when script exits
atexit.register(cleanup_model)