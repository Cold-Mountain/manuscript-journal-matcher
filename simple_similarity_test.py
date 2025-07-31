#!/usr/bin/env python3
"""
Simple test to isolate the similarity score issue.
"""

import sys
sys.path.append('src')

import numpy as np
import faiss
import json
from pathlib import Path
from embedder import embed_text

def simple_test():
    """Simple test to isolate the similarity issue."""
    print("🔧 Simple Similarity Test")
    print("="*50)
    
    # Load journal metadata directly
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    
    journals = data['journals']
    print(f"Loaded {len(journals)} journals")
    
    # Load FAISS index directly  
    index = faiss.read_index('data/journal_embeddings.faiss')
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    
    # Test query
    test_query = "machine learning applications in pediatric healthcare"
    print(f"\nQuery: '{test_query}'")
    
    # Generate and normalize embedding
    query_embedding = embed_text(test_query)
    query_norm = np.linalg.norm(query_embedding)
    normalized_query = query_embedding / query_norm if query_norm > 0 else query_embedding
    
    print(f"Query embedding norm after normalization: {np.linalg.norm(normalized_query):.6f}")
    
    # Search with FAISS
    similarities, indices = index.search(normalized_query.reshape(1, -1).astype(np.float32), 10)
    
    print(f"\nSearch Results:")
    print("Rank | Journal Name                                    | FAISS Score")
    print("-" * 70)
    
    for i, (sim, idx) in enumerate(zip(similarities[0], indices[0]), 1):
        if idx < len(journals):
            name = journals[idx].get('display_name', 'Unknown')[:42]
            print(f"{i:4d} | {name:42s} | {sim:10.6f}")
    
    # Check if we're getting the pediatric journals issue
    pediatric_count = 0
    for sim, idx in zip(similarities[0], indices[0]):
        if idx < len(journals):
            name = journals[idx].get('display_name', '').lower()
            if 'pediatric' in name or 'child' in name:
                pediatric_count += 1
    
    print(f"\nPediatric journals in top 10: {pediatric_count}")
    
    # Check if similarity scores are all 1.0
    unique_scores = len(set(similarities[0]))
    print(f"Unique similarity scores: {unique_scores}")
    if unique_scores == 1 and similarities[0][0] == 1.0:
        print("❌ BUG CONFIRMED: All scores are 1.0!")
    else:
        print("✅ Similarity scores show variety")
    
    return similarities[0], [journals[idx].get('display_name', 'Unknown') for idx in indices[0] if idx < len(journals)]

if __name__ == "__main__":
    scores, names = simple_test()
    
    print(f"\nDETAILED ANALYSIS:")
    print(f"Score range: {scores.min():.6f} to {scores.max():.6f}")
    print(f"Score std dev: {scores.std():.6f}")
    print(f"Are all scores identical? {len(set(scores)) == 1}")