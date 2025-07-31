#!/usr/bin/env python3
"""
Quick fix script to diagnose and repair similarity score calculation issues.
"""

import sys
sys.path.append('src')

import numpy as np
import faiss
from match_journals import JournalMatcher
from embedder import embed_text

def test_similarity_calculation():
    """Test if similarity calculation is working correctly."""
    print("🔧 Testing similarity calculation in JournalMatcher...")
    
    # Initialize matcher
    matcher = JournalMatcher()
    matcher.load_database()
    
    print(f"Loaded {len(matcher.journals)} journals")
    print(f"FAISS index type: {type(matcher.faiss_index)}")
    print(f"Index total: {matcher.faiss_index.ntotal}")
    
    # Test with a simple query
    test_query = "machine learning in healthcare applications"
    print(f"\nTesting with query: '{test_query}'")
    
    # Generate embedding
    query_embedding = embed_text(test_query)
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
    
    # Normalize embedding (same as in matcher)
    normalized_query = matcher._normalize_embeddings(query_embedding.reshape(1, -1))[0]
    print(f"Normalized query norm: {np.linalg.norm(normalized_query):.6f}")
    
    # Search directly with FAISS
    similarities, indices = matcher.faiss_index.search(
        normalized_query.reshape(1, -1).astype(np.float32), 
        10
    )
    
    print(f"\nDirect FAISS search results:")
    print(f"Similarities: {similarities[0]}")
    print(f"Indices: {indices[0]}")
    
    # Check if similarities are actually cosine similarities
    print(f"\nActual journal names and scores:")
    for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx < len(matcher.journals):
            name = matcher.journals[idx].get('display_name', 'Unknown')
            print(f"{i+1:2d}. {name[:50]:50s} | FAISS Score: {sim:.6f}")
    
    # Now test through the search_similar_journals method
    print(f"\n" + "="*70)
    print("Testing through search_similar_journals method:")
    
    try:
        results = matcher.search_similar_journals(
            query_text=test_query,
            top_k=10,
            min_similarity=0.0,  # Set to 0 to see all results
            include_study_classification=False,
            use_multimodal_analysis=False,
            use_ensemble_matching=False,
            include_ranking_analysis=False
        )
    except Exception as e:
        print(f"Error in search_similar_journals: {e}")
        print("Trying simplified search...")
        
        # Try manual search to isolate the issue
        results = []
        for i, (similarity, journal_idx) in enumerate(zip(similarities[0][:10], indices[0][:10])):
            if journal_idx < len(matcher.journals):
                journal = matcher.journals[journal_idx].copy()
                journal['similarity_score'] = float(similarity)
                journal['rank'] = i + 1
                results.append(journal)
        
        print("Manual search completed successfully")
    
    print(f"Found {len(results)} results through search method:")
    for i, result in enumerate(results[:10], 1):
        name = result.get('display_name', 'Unknown')
        score = result.get('similarity_score', 0)
        print(f"{i:2d}. {name[:50]:50s} | Method Score: {score:.6f}")
    
    # Compare the scores
    print(f"\n" + "="*70)
    print("COMPARISON:")
    print("If all Method Scores are 1.000, there's a bug in search_similar_journals")
    print("If FAISS Scores show variety, the core system works fine")

if __name__ == "__main__":
    test_similarity_calculation()