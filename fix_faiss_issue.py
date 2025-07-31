#!/usr/bin/env python3
"""
Fix the FAISS issue causing all distances to be 1.0
"""

import sys
sys.path.append('src')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import faiss
import json
from pathlib import Path

def diagnose_faiss_issue():
    """Diagnose why FAISS returns distance=1.0 for all queries."""
    print("🔧 DIAGNOSING FAISS ISSUE")
    print("=" * 50)
    
    # Load data
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    
    # Load FAISS index
    index = faiss.read_index('data/journal_embeddings.faiss')
    print(f"FAISS index: {type(index).__name__}")
    print(f"Index vectors: {index.ntotal}")
    print(f"Index dimension: {index.d}")
    
    # Check if it's an IVF index that needs training/setup
    if hasattr(index, 'nprobe'):
        print(f"Current nprobe: {index.nprobe}")
        # IVF indices need nprobe to be set properly
        if index.nprobe < 10:
            print("⚠️  nprobe is very low, increasing it")
            index.nprobe = min(50, index.nlist // 2)
            print(f"Set nprobe to: {index.nprobe}")
    
    # Test with a journal's own embedding (should return distance ≈ 0)
    print("\n1. Testing self-similarity...")
    first_journal_emb = np.array(journals[0]['embedding']).astype(np.float32)
    print(f"Query embedding norm: {np.linalg.norm(first_journal_emb):.6f}")
    
    distances, indices = index.search(first_journal_emb.reshape(1, -1), 5)
    print(f"Self-search distances: {distances[0]}")
    print(f"Self-search indices: {indices[0]}")
    
    if distances[0][0] > 0.1:
        print("❌ PROBLEM: Self-similarity is not near 0!")
        print("This indicates the FAISS index is corrupted or incorrectly built")
        return True  # Needs rebuild
    
    # Test with a different embedding
    print("\n2. Testing with embedder...")
    try:
        from embedder import embed_text
        
        # Generate a query embedding
        query_emb = embed_text("medical research in healthcare")
        print(f"Generated embedding norm: {np.linalg.norm(query_emb):.6f}")
        
        # Search
        distances, indices = index.search(query_emb.reshape(1, -1).astype(np.float32), 5)
        print(f"Query search distances: {distances[0]}")
        
        if all(abs(d - 1.0) < 0.001 for d in distances[0]):
            print("❌ PROBLEM: All distances are exactly 1.0!")
            print("This suggests embedding dimension mismatch or index corruption")
            return True
        else:
            print("✅ Distances show variety")
            
    except Exception as e:
        print(f"Error with embedder: {e}")
    
    return False

def rebuild_faiss_index():
    """Rebuild the FAISS index correctly."""
    print("\n🔧 REBUILDING FAISS INDEX")
    print("=" * 50)
    
    # Load journal data
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    
    # Extract embeddings
    embeddings = []
    valid_indices = []
    
    for i, journal in enumerate(journals):
        if journal.get('embedding'):
            embeddings.append(journal['embedding'])
            valid_indices.append(i)
    
    if not embeddings:
        print("❌ No embeddings found in journal data!")
        return False
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Embedding norms (first 5): {[np.linalg.norm(emb) for emb in embeddings_array[:5]]}")
    
    # Check if embeddings are normalized
    norms = np.linalg.norm(embeddings_array, axis=1)
    print(f"Norm statistics: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
    
    # Normalize embeddings if needed
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("⚠️  Embeddings not normalized, normalizing...")
        embeddings_array = embeddings_array / norms[:, np.newaxis]
        print("✅ Embeddings normalized")
    
    # Create new FAISS index
    dimension = embeddings_array.shape[1]
    n_vectors = embeddings_array.shape[0]
    
    print(f"Creating FAISS index for {n_vectors} vectors of dimension {dimension}")
    
    if n_vectors < 1000:
        # Use flat index for small datasets
        print("Using IndexFlatIP (inner product)")
        index = faiss.IndexFlatIP(dimension)
    else:
        # Use IVF for larger datasets  
        n_centroids = min(int(np.sqrt(n_vectors)), 100)
        print(f"Using IndexIVFFlat with {n_centroids} centroids")
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_centroids)
        
        # Train the index
        print("Training IVF index...")
        index.train(embeddings_array)
        
        # Set reasonable nprobe
        index.nprobe = min(50, n_centroids // 2)
        print(f"Set nprobe to {index.nprobe}")
    
    # Add vectors to index
    print("Adding vectors to index...")
    index.add(embeddings_array)
    
    # Test the new index
    print("\n🧪 Testing new index...")
    test_emb = embeddings_array[0]  # Use first journal's embedding
    distances, indices = index.search(test_emb.reshape(1, -1), 5)
    
    print(f"Test distances: {distances[0]}")
    print(f"Test indices: {indices[0]}")
    
    if distances[0][0] < 0.01 and len(set(distances[0])) > 1:
        print("✅ New index works correctly!")
        
        # Save the new index
        print("Saving new FAISS index...")
        faiss.write_index(index, 'data/journal_embeddings.faiss')
        print("✅ New index saved")
        return True
    else:
        print("❌ New index still has issues")
        return False

def main():
    """Main function to fix FAISS issues."""
    needs_rebuild = diagnose_faiss_issue()
    
    if needs_rebuild:
        print("\n🔧 Index needs rebuilding...")
        success = rebuild_faiss_index()
        if success:
            print("\n✅ FAISS index rebuilt successfully!")
            print("🚀 The similarity calculation should now work correctly.")
        else:
            print("\n❌ Failed to rebuild index")
    else:
        print("\n✅ FAISS index appears to be working correctly")

if __name__ == "__main__":
    main()