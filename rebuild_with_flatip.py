#!/usr/bin/env python3
"""
Rebuild FAISS index using IndexFlatIP for guaranteed accuracy.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import faiss
import json

def rebuild_with_flatip():
    """Rebuild FAISS index using IndexFlatIP which directly returns cosine similarity."""
    print("🔧 REBUILDING WITH IndexFlatIP")
    print("=" * 50)
    
    # Load journal data
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    
    # Extract embeddings
    embeddings = []
    for journal in journals:
        if journal.get('embedding'):
            embeddings.append(journal['embedding'])
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings_array.shape}")
    
    # Verify embeddings are normalized
    norms = np.linalg.norm(embeddings_array, axis=1)
    print(f"Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}")
    
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("Normalizing embeddings...")
        embeddings_array = embeddings_array / norms[:, np.newaxis]
    
    # Create IndexFlatIP (returns cosine similarity directly)
    dimension = embeddings_array.shape[1]
    print(f"Creating IndexFlatIP for dimension {dimension}")
    
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    print(f"Index created with {index.ntotal} vectors")
    
    # Test the index
    print("\nTesting new index...")
    test_emb = embeddings_array[0]
    similarities, indices = index.search(test_emb.reshape(1, -1), 5)
    
    print(f"Test similarities (should be cosine): {similarities[0]}")
    print(f"First similarity (self): {similarities[0][0]:.6f} (should be ~1.0)")
    
    if similarities[0][0] > 0.99:
        print("✅ IndexFlatIP working correctly!")
        
        # Save the new index
        faiss.write_index(index, 'data/journal_embeddings.faiss')
        print("✅ Saved new IndexFlatIP index")
        
        return True
    else:
        print("❌ IndexFlatIP test failed")
        return False

if __name__ == "__main__":
    rebuild_with_flatip()