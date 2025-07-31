#!/usr/bin/env python3
"""
Quick test of the similarity fix without rebuilding the whole index.
"""

import sys
sys.path.append('src')

# Suppress multiprocessing warnings
import warnings
import multiprocessing
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

import numpy as np
import faiss
import json
from embedder import embed_text

def test_similarity_fix():
    """Test the similarity calculation fix."""
    print("🧪 Testing similarity calculation fix...")
    
    # Load existing index (even though it's wrong type, we can test the conversion)
    index = faiss.read_index('data/journal_embeddings.faiss')
    
    # Load journal metadata
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    
    # Test query
    test_query = "machine learning applications in pediatric healthcare diagnosis"
    query_embedding = embed_text(test_query)
    
    # Normalize query
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm
    
    # Search
    distances, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 10)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Raw FAISS distances: {distances[0]}")
    
    # Apply the similarity conversion
    converted_similarities = []
    for distance in distances[0]:
        if isinstance(index, faiss.IndexFlatIP):
            similarity = float(distance)  # Already cosine similarity
        else:
            # For IndexIVFFlat: convert distance to similarity
            similarity = max(0.0, 1.0 - (float(distance) / 2.0))
        converted_similarities.append(similarity)
    
    print(f"Converted similarities: {converted_similarities}")
    
    # Show results with journal names
    print(f"\nTop results:")
    print("Rank | Journal Name                                    | Distance  | Similarity")
    print("-" * 80)
    
    for i, (dist, sim, idx) in enumerate(zip(distances[0], converted_similarities, indices[0]), 1):
        if idx < len(journals):
            name = journals[idx].get('display_name', 'Unknown')[:42]
            print(f"{i:4d} | {name:42s} | {dist:8.6f} | {sim:10.6f}")
    
    # Check if similarities are reasonable
    min_sim = min(converted_similarities)
    max_sim = max(converted_similarities)
    unique_sims = len(set(converted_similarities))
    
    print(f"\nSimilarity Analysis:")
    print(f"Range: {min_sim:.6f} to {max_sim:.6f}")
    print(f"Unique values: {unique_sims}")
    
    # Check for pediatric journals dominance
    pediatric_count = 0
    for idx in indices[0]:
        if idx < len(journals):
            name = journals[idx].get('display_name', '').lower()
            if 'pediatric' in name or 'child' in name or 'paediatric' in name:
                pediatric_count += 1
    
    print(f"Pediatric journals in top 10: {pediatric_count}")
    
    if unique_sims > 1 and max_sim <= 1.0:
        print("✅ Similarity calculation appears to be working correctly!")
    else:
        print("❌ There may still be issues with similarity calculation")
    
    return converted_similarities

if __name__ == "__main__":
    test_similarity_fix()