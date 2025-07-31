#!/usr/bin/env python3
"""
Rebuild the FAISS index with the correct IndexIVFIP for cosine similarity.
"""

import sys
sys.path.append('src')

import os
import shutil
from pathlib import Path

def rebuild_index():
    """Rebuild the FAISS index with correct settings."""
    print("🔧 Rebuilding FAISS index with correct cosine similarity settings...")
    
    # Remove the old index file
    index_path = Path('data/journal_embeddings.faiss')
    if index_path.exists():
        print(f"Removing old index: {index_path}")
        index_path.unlink()
    
    # Initialize the journal matcher which will create a new index
    from match_journals import JournalMatcher
    
    print("Initializing JournalMatcher...")
    matcher = JournalMatcher()
    
    print("Loading database and creating new index...")
    matcher.load_database(force_reload=True)
    
    print("✅ New FAISS index created successfully!")
    print(f"Index type: {type(matcher.faiss_index).__name__}")
    print(f"Index size: {matcher.faiss_index.ntotal} vectors")
    
    # Test the new index
    print("\n🧪 Testing new index...")
    from embedder import embed_text
    import numpy as np
    
    test_query = "machine learning pediatric healthcare applications"
    query_embedding = embed_text(test_query)
    normalized_query = query_embedding / np.linalg.norm(query_embedding)
    
    similarities, indices = matcher.faiss_index.search(
        normalized_query.reshape(1, -1).astype(np.float32), 5
    )
    
    print(f"Test query: '{test_query}'")
    print(f"Similarity scores: {similarities[0]}")
    print(f"Score range: {similarities[0].min():.6f} to {similarities[0].max():.6f}")
    
    # Check if scores are proper cosine similarities (should be <= 1.0)
    if all(score <= 1.0 for score in similarities[0]):
        print("✅ Similarity scores are now correct (cosine similarities ≤ 1.0)")
    else:
        print("❌ Warning: Some similarity scores are > 1.0")
    
    # Test variety
    if len(set(similarities[0])) > 1:
        print("✅ Similarity scores show proper variety")
    else:
        print("❌ Warning: All similarity scores are identical")
    
    print("\n🎯 Index rebuild complete! The similarity calculation bug should now be fixed.")

if __name__ == "__main__":
    rebuild_index()