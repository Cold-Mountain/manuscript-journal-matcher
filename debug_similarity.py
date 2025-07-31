#!/usr/bin/env python3
"""
Debug script for similarity calculation issues
"""

import sys
sys.path.append('src')
import numpy as np
import json
from pathlib import Path

def test_database_variety():
    """Test if journal embeddings are actually different"""
    print("=== Testing Journal Database Variety ===")
    
    # Load journal metadata
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    
    journals = data['journals']
    print(f"Total journals: {len(journals)}")
    
    # Get first 10 embeddings
    embeddings = []
    names = []
    for i, journal in enumerate(journals[:10]):
        if journal.get('embedding'):
            embeddings.append(journal['embedding'])
            names.append(journal.get('display_name', f'Journal {i}'))
    
    embeddings = np.array(embeddings)
    print(f"Sample embeddings shape: {embeddings.shape}")
    
    # Check if embeddings are identical
    first_embedding = embeddings[0]
    similarities_to_first = []
    
    print("\nSimilarities to first journal:")
    for i, (embedding, name) in enumerate(zip(embeddings, names)):
        similarity = np.dot(first_embedding, embedding)
        similarities_to_first.append(similarity)
        print(f"{i+1:2d}. {name[:50]:50s} | Similarity: {similarity:.6f}")
    
    # Statistics
    similarities = np.array(similarities_to_first)
    print(f"\nSimilarity statistics:")
    print(f"Min: {similarities.min():.6f}")
    print(f"Max: {similarities.max():.6f}")
    print(f"Mean: {similarities.mean():.6f}")
    print(f"Std: {similarities.std():.6f}")
    print(f"Unique values: {len(np.unique(similarities))}")
    
    return similarities


def test_embedding_generation():
    """Test if embedding generation is working correctly"""
    print("\n=== Testing Embedding Generation ===")
    
    try:
        from embedder import embed_text
        
        test_texts = [
            "machine learning in healthcare research",
            "pediatric cardiology clinical studies", 
            "quantum physics theoretical models",
            "environmental science climate change",
            "artificial intelligence neural networks"
        ]
        
        embeddings = []
        for text in test_texts:
            embedding = embed_text(text)
            embeddings.append(embedding)
            print(f"Text: '{text[:30]}...' | Norm: {np.linalg.norm(embedding):.6f}")
        
        # Check similarities between different texts
        print("\nCross-similarity matrix:")
        embeddings = np.array(embeddings)
        for i in range(len(test_texts)):
            similarities = []
            for j in range(len(test_texts)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
            print(f"Text {i+1}: {similarities}")
        
        return embeddings
        
    except Exception as e:
        print(f"Error in embedding generation: {e}")
        return None


def test_faiss_search():
    """Test FAISS search functionality"""
    print("\n=== Testing FAISS Search ===")
    
    try:
        import faiss
        
        # Load FAISS index
        index = faiss.read_index('data/journal_embeddings.faiss')
        print(f"FAISS index: {index.ntotal} vectors, {index.d} dimensions")
        
        # Test with a simple query
        from embedder import embed_text
        query = embed_text("machine learning healthcare")
        
        # Search
        k = 5
        distances, indices = index.search(query.reshape(1, -1).astype(np.float32), k)
        
        print(f"Search results:")
        print(f"Distances: {distances[0]}")
        print(f"Indices: {indices[0]}")
        
        # Load journal names to see what we found
        with open('data/journal_metadata.json', 'r') as f:
            data = json.load(f)
        journals = data['journals']
        
        print("\nTop results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(journals):
                name = journals[idx].get('display_name', f'Journal {idx}')
                print(f"{i+1}. {name} | Distance: {dist:.6f} | Similarity: {1-dist:.6f}")
        
        return distances, indices
        
    except Exception as e:
        print(f"Error in FAISS search: {e}")
        return None, None


def main():
    """Run all diagnostic tests"""
    print("🔍 DEBUGGING JOURNAL MATCHING ISSUES")
    print("=" * 60)
    
    # Test 1: Database variety
    similarities = test_database_variety()
    
    # Test 2: Embedding generation
    embeddings = test_embedding_generation()
    
    # Test 3: FAISS search
    distances, indices = test_faiss_search()
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSIS SUMMARY:")
    
    if similarities is not None:
        if len(np.unique(similarities)) == 1:
            print("❌ PROBLEM: All journal embeddings are identical!")
        elif similarities.std() < 0.01:
            print("⚠️  WARNING: Journal embeddings have very low variance")
        else:
            print("✅ Journal embeddings show good variety")
    
    if embeddings is not None:
        print("✅ Embedding generation is working")
    else:
        print("❌ PROBLEM: Embedding generation failed")
    
    if distances is not None:
        if len(np.unique(distances[0])) == 1:
            print("❌ PROBLEM: FAISS search returns identical distances!")
        else:
            print("✅ FAISS search shows variety in results")
    else:
        print("❌ PROBLEM: FAISS search failed")


if __name__ == "__main__":
    main()