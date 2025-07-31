#!/usr/bin/env python3
"""
Test to reproduce the exact user issue with 0.500 similarities and pediatric bias.
"""

import sys
sys.path.append('src')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import faiss
import json

def main():
    print("🎯 REPRODUCING USER'S EXACT ISSUE")
    print("=" * 50)
    
    # Load data
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    
    index = faiss.read_index('data/journal_embeddings.faiss')
    print(f"Loaded {len(journals)} journals, FAISS index: {index.ntotal} vectors")
    
    # Test with the embedder but avoid hanging
    try:
        from embedder import embed_text
        
        # User's likely query (pediatric + healthcare)  
        query = "machine learning applications in pediatric healthcare diagnosis and treatment optimization"
        print(f"\nQuery: {query}")
        
        # Generate embedding
        embedding = embed_text(query)
        print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
        
        # Search
        distances, indices = index.search(embedding.reshape(1, -1).astype(np.float32), 15)
        
        print(f"\nRaw FAISS distances: {distances[0][:5]}")
        
        # Apply the current conversion formula
        similarities = [max(0.0, 1.0 - (d / 2.0)) for d in distances[0]]
        print(f"Converted similarities: {[f'{s:.3f}' for s in similarities[:5]]}")
        
        # Check for the exact issues user reported
        print(f"\n=== CHECKING USER'S REPORTED ISSUES ===")
        
        # Issue 1: All similarities around 0.500?
        around_500 = sum(1 for s in similarities if 0.4 < s < 0.6)
        print(f"1. Similarities around 0.500: {around_500}/{len(similarities)}")
        if around_500 >= 10:
            print("   ❌ CONFIRMED: Clustering around 0.500")
        else:
            print("   ✅ No 0.500 clustering")
        
        # Issue 2: Pediatric journal bias?
        pediatric_count = 0
        lancet_child_found = False
        
        print(f"\n2. Journal results analysis:")
        for i, (sim, idx) in enumerate(zip(similarities, indices[0]), 1):
            if idx < len(journals):
                name = journals[idx].get('display_name', 'Unknown')
                is_pediatric = any(kw in name.lower() for kw in 
                                 ['pediatric', 'paediatric', 'child', 'infant', 'neonat'])
                
                if is_pediatric:
                    pediatric_count += 1
                
                if 'lancet' in name.lower() and 'child' in name.lower():
                    lancet_child_found = True
                
                if i <= 10:  # Show top 10
                    print(f"   {i:2d}. {name[:45]:45s} | Sim: {sim:.3f} | {'PEDIATRIC' if is_pediatric else ''}")
        
        print(f"\n   Pediatric journals in top 15: {pediatric_count}")
        print(f"   Lancet Child & Adolescent Health found: {lancet_child_found}")
        
        if pediatric_count >= 8:
            print("   ❌ CONFIRMED: High pediatric bias")
        else:
            print("   ✅ No significant pediatric bias")
            
        if lancet_child_found:
            print("   ❌ CONFIRMED: Lancet Child found (matches user report)")
        
    except Exception as e:
        print(f"Error with embedder: {e}")
        # Fallback test with random vector
        print("Falling back to random vector test...")
        test_vector = np.random.randn(384).astype(np.float32)
        test_vector = test_vector / np.linalg.norm(test_vector)
        distances, indices = index.search(test_vector.reshape(1, -1), 10)
        similarities = [max(0.0, 1.0 - (d / 2.0)) for d in distances[0]]
        print(f"Random vector similarities: {[f'{s:.3f}' for s in similarities]}")

if __name__ == "__main__":
    main()