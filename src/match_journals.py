"""
Vector search and journal matching module for Manuscript Journal Matcher.

This module implements FAISS-based similarity search to find the most
relevant journals for a given manuscript based on semantic embeddings.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from pathlib import Path
import time

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is required for vector search. Install with: pip install faiss-cpu"
    )

try:
    from .config import (
        JOURNAL_METADATA_PATH,
        FAISS_INDEX_PATH,
        DEFAULT_TOP_K_RESULTS,
        MIN_SIMILARITY_THRESHOLD,
        get_embedding_dimension
    )
    from .journal_db_builder import load_journal_database
    from .embedder import embed_text, get_model
except ImportError:
    from config import (
        JOURNAL_METADATA_PATH,
        FAISS_INDEX_PATH,
        DEFAULT_TOP_K_RESULTS,
        MIN_SIMILARITY_THRESHOLD,
        get_embedding_dimension
    )
    from journal_db_builder import load_journal_database
    from embedder import embed_text, get_model

# Set up logging
logger = logging.getLogger(__name__)


class MatchingError(Exception):
    """Custom exception for journal matching errors."""
    pass


class JournalMatcher:
    """
    Main class for journal matching using FAISS-based vector search.
    """
    
    def __init__(self, index_path: Optional[Path] = None, 
                 metadata_path: Optional[Path] = None):
        """
        Initialize the journal matcher.
        
        Args:
            index_path: Path to FAISS index file (optional)
            metadata_path: Path to journal metadata file (optional)
        """
        self.index_path = index_path or FAISS_INDEX_PATH
        self.metadata_path = metadata_path or JOURNAL_METADATA_PATH
        
        self.journals: List[Dict[str, Any]] = []
        self.faiss_index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dimension: Optional[int] = None
        
        logger.info(f"Initialized JournalMatcher with index: {self.index_path}")
    
    def load_database(self, force_reload: bool = False) -> None:
        """
        Load journal database and create/load FAISS index.
        
        Args:
            force_reload: Whether to force reload even if already loaded
            
        Raises:
            MatchingError: If database loading fails
        """
        if not force_reload and self.journals and self.faiss_index is not None:
            logger.debug("Database already loaded, skipping")
            return
        
        try:
            logger.info("Loading journal database...")
            
            # Load journal metadata and embeddings
            self.journals, self.embeddings = load_journal_database()
            
            if not self.journals:
                raise MatchingError("No journals found in database")
            
            if self.embeddings is None:
                raise MatchingError("No embeddings found in database")
            
            logger.info(f"Loaded {len(self.journals)} journals")
            
            # Set embedding dimension
            self.embedding_dimension = self.embeddings.shape[1]
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
            # Create or load FAISS index
            self.faiss_index = self._create_or_load_faiss_index()
            
            logger.info("✅ Journal database loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load journal database: {e}")
            raise MatchingError(f"Database loading failed: {e}")
    
    def _create_or_load_faiss_index(self) -> faiss.Index:
        """
        Create FAISS index from embeddings or load existing index.
        
        Returns:
            FAISS index ready for similarity search
        """
        # Try to load existing index first
        if self.index_path.exists():
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                index = faiss.read_index(str(self.index_path))
                
                # Verify index matches our data
                if index.ntotal == len(self.journals) and index.d == self.embedding_dimension:
                    logger.info(f"✅ Loaded FAISS index with {index.ntotal} vectors")
                    return index
                else:
                    logger.warning(f"Index size mismatch. Expected {len(self.journals)} vectors, "
                            f"got {index.ntotal}. Rebuilding index.")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
        
        # Create new index
        return self._create_faiss_index()
    
    def _create_faiss_index(self) -> faiss.Index:
        """
        Create a new FAISS index from the embeddings.
        
        Returns:
            New FAISS index
        """
        logger.info("Creating new FAISS index...")
        
        if self.embeddings is None:
            raise MatchingError("No embeddings available for index creation")
        
        # Choose index type based on dataset size and dimension
        n_vectors, dimension = self.embeddings.shape
        
        if n_vectors < 1000:
            # For small datasets, use flat index (exact search)
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            logger.info(f"Using flat index for {n_vectors} vectors")
        else:
            # For larger datasets, use IVF index for faster search
            n_centroids = min(int(np.sqrt(n_vectors)), 100)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_centroids)
            
            # Train the index
            logger.info(f"Training IVF index with {n_centroids} centroids...")
            index.train(self.embeddings.astype(np.float32))
            logger.info(f"Using IVF index for {n_vectors} vectors")
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self._normalize_embeddings(self.embeddings)
        
        # Add vectors to index
        index.add(normalized_embeddings.astype(np.float32))
        
        # Save index to disk
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self.index_path))
            logger.info(f"💾 Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.warning(f"Failed to save FAISS index: {e}")
        
        logger.info(f"✅ Created FAISS index with {index.ntotal} vectors")
        return index
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity search.
        
        Args:
            embeddings: Input embeddings array
            
        Returns:
            L2-normalized embeddings
        """
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def search_similar_journals(self, query_text: str, top_k: int = None,
                               min_similarity: float = None,
                               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find journals most similar to the given query text.
        
        Args:
            query_text: Text to search for (typically manuscript abstract)
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            filters: Additional filters to apply to results
            
        Returns:
            List of matching journals with similarity scores
            
        Raises:
            MatchingError: If search fails
        """
        if not query_text or not query_text.strip():
            raise MatchingError("Query text cannot be empty")
        
        # Load database if not already loaded
        self.load_database()
        
        # Set defaults
        if top_k is None:
            top_k = DEFAULT_TOP_K_RESULTS
        if min_similarity is None:
            min_similarity = MIN_SIMILARITY_THRESHOLD
        
        try:
            start_time = time.time()
            
            # Generate embedding for query
            logger.debug(f"Generating embedding for query (length: {len(query_text)})")
            query_embedding = embed_text(query_text)
            query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))[0]
            
            # Perform similarity search
            logger.debug(f"Searching for top {top_k} similar journals")
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 
                min(top_k * 2, len(self.journals))  # Get more results for filtering
            )
            
            # Process results
            results = []
            for i, (similarity, journal_idx) in enumerate(zip(similarities[0], indices[0])):
                if journal_idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if similarity < min_similarity:
                    continue
                
                if journal_idx >= len(self.journals):
                    logger.warning(f"Invalid journal index: {journal_idx}")
                    continue
                
                journal = self.journals[journal_idx].copy()
                journal['similarity_score'] = float(similarity)
                journal['rank'] = i + 1
                
                results.append(journal)
            
            # Apply additional filters
            if filters:
                results = self._apply_filters(results, filters)
            
            # Sort by similarity and limit results
            results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
            results = results[:top_k]
            
            search_time = time.time() - start_time
            logger.info(f"Found {len(results)} matching journals in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search journals: {e}")
            raise MatchingError(f"Journal search failed: {e}")
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply additional filters to search results.
        
        Args:
            results: List of journal results
            filters: Dictionary of filter criteria
            
        Returns:
            Filtered results list
        """
        filtered_results = []
        
        for journal in results:
            # Check open access filter (enhanced with DOAJ data)
            if 'open_access_only' in filters and filters['open_access_only']:
                # Use DOAJ oa_status if available, fall back to OpenAlex is_oa
                is_oa = journal.get('oa_status', journal.get('is_oa', False))
                if not is_oa:
                    continue
            
            # Check DOAJ-specific filter
            if 'doaj_only' in filters and filters['doaj_only']:
                if not journal.get('in_doaj', False):
                    continue
            
            # Check APC range filter (enhanced with DOAJ data)
            if 'max_apc' in filters and filters['max_apc'] is not None:
                # Use DOAJ APC if available, fall back to OpenAlex APC
                apc = journal.get('apc_amount') or journal.get('apc_usd')
                if apc is not None and apc > filters['max_apc']:
                    continue
            
            # Check minimum APC filter
            if 'min_apc' in filters and filters['min_apc'] is not None:
                apc = journal.get('apc_amount') or journal.get('apc_usd')
                if apc is None or apc < filters['min_apc']:
                    continue
            
            # Check no-APC filter (free to publish)
            if 'no_apc_only' in filters and filters['no_apc_only']:
                has_apc = journal.get('has_apc', False)
                apc_amount = journal.get('apc_amount') or journal.get('apc_usd')
                # Include journals that explicitly have no APC or have zero APC
                if has_apc or (apc_amount is not None and apc_amount > 0):
                    continue
            
            # Check subject/field filter (enhanced with DOAJ subjects)
            if 'subjects' in filters and filters['subjects']:
                # Combine OpenAlex and DOAJ subjects
                journal_subjects = []
                
                # Add OpenAlex subjects
                for s in journal.get('subjects', []):
                    if isinstance(s, dict) and s.get('name'):
                        journal_subjects.append(s['name'].lower())
                    elif isinstance(s, str):
                        journal_subjects.append(s.lower())
                
                # Add DOAJ subjects
                for s in journal.get('subjects_doaj', []):
                    if isinstance(s, str):
                        journal_subjects.append(s.lower())
                
                filter_subjects = [s.lower() for s in filters['subjects']]
                
                # Check if any filter subject matches journal subjects
                if not any(fs in ' '.join(journal_subjects) for fs in filter_subjects):
                    continue
            
            # Check publisher filter (enhanced with DOAJ data)
            if 'publishers' in filters and filters['publishers']:
                # Use both OpenAlex and DOAJ publisher information
                publishers_to_check = []
                if journal.get('publisher'):
                    publishers_to_check.append(journal['publisher'].lower())
                if journal.get('publisher_doaj'):
                    publishers_to_check.append(journal['publisher_doaj'].lower())
                
                filter_publishers = [p.lower() for p in filters['publishers']]
                
                # Check if any filter publisher matches journal publishers
                match_found = False
                for jp in publishers_to_check:
                    if any(fp in jp for fp in filter_publishers):
                        match_found = True
                        break
                
                if not match_found:
                    continue
            
            # Check language filter
            if 'languages' in filters and filters['languages']:
                journal_languages = journal.get('languages', [])
                filter_languages = [lang.lower() for lang in filters['languages']]
                
                # Check if any filter language matches journal languages
                if not any(lang.lower() in filter_languages for lang in journal_languages):
                    continue
            
            # Check minimum citation count
            if 'min_citations' in filters and filters['min_citations'] is not None:
                citations = journal.get('cited_by_count', 0)
                if citations < filters['min_citations']:
                    continue
            
            # Check minimum h-index
            if 'min_h_index' in filters and filters['min_h_index'] is not None:
                h_index = journal.get('h_index', 0)
                if h_index < filters['min_h_index']:
                    continue
            
            filtered_results.append(journal)
        
        logger.debug(f"Filtered {len(results)} results to {len(filtered_results)}")
        return filtered_results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded journal database.
        
        Returns:
            Dictionary with database statistics
        """
        self.load_database()
        
        stats = {
            'total_journals': len(self.journals),
            'embedding_dimension': self.embedding_dimension,
            'faiss_index_type': type(self.faiss_index).__name__,
            'index_size': self.faiss_index.ntotal if self.faiss_index else 0,
        }
        
        # Analyze journal characteristics
        if self.journals:
            # Count different types of journals
            oa_count = sum(1 for j in self.journals if j.get('oa_status', False))
            doaj_count = sum(1 for j in self.journals if j.get('in_doaj', False))
            with_apc_count = sum(1 for j in self.journals 
                               if j.get('apc_amount') is not None or j.get('apc_usd') is not None)
            no_apc_count = sum(1 for j in self.journals 
                             if j.get('has_apc', False) == False and j.get('oa_status', False))
            
            # Calculate average APC
            apc_values = []
            for j in self.journals:
                apc = j.get('apc_amount') or j.get('apc_usd')
                if apc is not None and apc > 0:
                    apc_values.append(apc)
            
            avg_apc = sum(apc_values) / len(apc_values) if apc_values else 0
            
            stats.update({
                'open_access_journals': oa_count,
                'doaj_journals': doaj_count,
                'journals_with_apc': with_apc_count,
                'free_to_publish_journals': no_apc_count,
                'average_apc': round(avg_apc, 2),
                'sample_journal': self.journals[0].get('display_name', 'Unknown'),
            })
        
        return stats


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Create FAISS index from embeddings array.
    
    Args:
        embeddings: 2D numpy array of embeddings
        
    Returns:
        FAISS index ready for similarity search
    """
    matcher = JournalMatcher()
    matcher.embeddings = embeddings
    matcher.embedding_dimension = embeddings.shape[1]
    return matcher._create_faiss_index()


def load_journal_database_with_index() -> Tuple[List[Dict[str, Any]], faiss.Index]:
    """
    Load journal metadata and FAISS index from disk.
    
    Returns:
        Tuple of (journals, faiss_index)
    """
    matcher = JournalMatcher()
    matcher.load_database()
    return matcher.journals, matcher.faiss_index


def search_similar_journals(query_embedding: np.ndarray, 
                           index: faiss.Index,
                           journals: List[Dict[str, Any]], 
                           top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Find most similar journals to query using pre-built index.
    
    Args:
        query_embedding: Query embedding vector
        index: FAISS index
        journals: List of journal metadata
        top_k: Number of top results to return
        
    Returns:
        List of matching journals with similarity scores
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Normalize query embedding
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm
    
    # Search
    similarities, indices = index.search(query_embedding.astype(np.float32), top_k)
    
    # Process results
    results = []
    for i, (similarity, journal_idx) in enumerate(zip(similarities[0], indices[0])):
        if journal_idx == -1 or journal_idx >= len(journals):
            continue
        
        journal = journals[journal_idx].copy()
        journal['similarity_score'] = float(similarity)
        journal['rank'] = i + 1
        
        results.append(journal)
    
    return results


def rank_and_filter_results(results: List[Dict[str, Any]], 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Apply additional ranking and filtering to results.
    
    Args:
        results: List of journal results with similarity scores
        filters: Optional filter criteria
        
    Returns:
        Filtered and ranked results
    """
    if not results:
        return results
    
    # Apply filters if provided
    if filters:
        matcher = JournalMatcher()
        results = matcher._apply_filters(results, filters)
    
    # Additional ranking logic can be added here
    # For now, just sort by similarity score
    results = sorted(results, key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    return results


def format_search_results(results: List[Dict[str, Any]], 
                         include_embeddings: bool = False) -> List[Dict[str, Any]]:
    """
    Format search results for display or export.
    
    Args:
        results: Raw search results
        include_embeddings: Whether to include embedding vectors in output
        
    Returns:
        Formatted results ready for display
    """
    formatted = []
    
    for journal in results:
        # Create clean result dict
        issn_list = journal.get('issn', [])
        issn = journal.get('issn_l') or (issn_list[0] if issn_list else None)
        
        # Combine subjects from both OpenAlex and DOAJ
        all_subjects = []
        for s in journal.get('subjects', []):
            if isinstance(s, dict) and s.get('name'):
                all_subjects.append(s['name'])
            elif isinstance(s, str):
                all_subjects.append(s)
        
        for s in journal.get('subjects_doaj', []):
            if isinstance(s, str) and s not in all_subjects:
                all_subjects.append(s)
        
        # Get APC information (prioritize DOAJ data)
        apc_amount = journal.get('apc_amount') or journal.get('apc_usd')
        apc_currency = journal.get('apc_currency', 'USD')
        
        formatted_journal = {
            'rank': journal.get('rank', 0),
            'journal_name': journal.get('display_name', 'Unknown'),
            'similarity_score': round(journal.get('similarity_score', 0), 3),
            'publisher': journal.get('publisher_doaj') or journal.get('publisher', 'Unknown'),
            'issn': issn,
            'is_open_access': journal.get('oa_status', journal.get('is_oa', False)),
            'in_doaj': journal.get('in_doaj', False),
            'apc_amount': apc_amount,
            'apc_currency': apc_currency,
            'apc_display': f"{apc_amount} {apc_currency}" if apc_amount else None,
            'has_apc': journal.get('has_apc'),
            'homepage_url': journal.get('homepage_url'),
            'works_count': journal.get('works_count', 0),
            'cited_by_count': journal.get('cited_by_count', 0),
            'h_index': journal.get('h_index', 0),
            'subjects': all_subjects[:5],  # Show top 5 subjects
            'languages': journal.get('languages', []),
            'license_type': journal.get('license_type', []),
            'oa_start_year': journal.get('oa_start_year'),
            'country': journal.get('country_doaj') or journal.get('country_code'),
            
            # Legacy fields for backward compatibility
            'apc_usd': journal.get('apc_usd'),  # Keep for compatibility
        }
        
        # Add embeddings if requested
        if include_embeddings and 'embedding' in journal:
            formatted_journal['embedding'] = journal['embedding']
        
        formatted.append(formatted_journal)
    
    return formatted