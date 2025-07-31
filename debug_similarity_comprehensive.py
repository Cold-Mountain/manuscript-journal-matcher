#!/usr/bin/env python3
"""
Comprehensive debugging of similarity calculation issues.
"""

import sys
sys.path.append('src')

import numpy as np
import faiss
import json
from embedder import embed_text

def test_current_system():
    """Test the current system to reproduce the 0.500 similarity issue."""
    print("🔍 COMPREHENSIVE SIMILARITY DEBUG")
    print("=" * 60)
    
    # Load current system
    print("1. Loading current system...")
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    
    index = faiss.read_index('data/journal_embeddings.faiss')
    print(f"   ✓ Loaded {len(journals)} journals")
    print(f"   ✓ FAISS index: {type(index).__name__} with {index.ntotal} vectors")
    
    # Test with user's likely query
    print("\n2. Testing with healthcare query...")
    query_text = "machine learning applications in pediatric healthcare diagnosis"
    query_embedding = embed_text(query_text)
    
    # Normalize query (crucial step)
    query_norm = np.linalg.norm(query_embedding)
    normalized_query = query_embedding / query_norm if query_norm > 0 else query_embedding
    
    print(f"   Query: '{query_text}'")
    print(f"   Original embedding norm: {query_norm:.6f}")
    print(f"   Normalized embedding norm: {np.linalg.norm(normalized_query):.6f}")
    
    # FAISS search
    distances, indices = index.search(normalized_query.reshape(1, -1).astype(np.float32), 10)
    
    print(f"\n3. FAISS search results:")
    print(f"   Raw distances: {distances[0]}")
    print(f"   Distance range: {distances[0].min():.4f} to {distances[0].max():.4f}")
    
    # Apply current conversion formula
    print(f"\n4. Current similarity conversion:")
    similarities_current = []
    for d in distances[0]:
        # Current formula: similarity = max(0.0, 1.0 - (distance / 2.0))
        sim = max(0.0, 1.0 - (float(d) / 2.0))
        similarities_current.append(sim)
    
    print(f"   Current formula: max(0.0, 1.0 - (distance / 2.0))")
    print(f"   Converted similarities: {similarities_current}")
    print(f"   Similarity range: {min(similarities_current):.4f} to {max(similarities_current):.4f}")
    
    # Check if all similarities are around 0.5
    around_half = [0.4 < s < 0.6 for s in similarities_current]
    print(f"   Similarities around 0.5: {sum(around_half)}/{len(around_half)}")
    
    # Show actual journal results
    print(f"\n5. Journal results analysis:")
    pediatric_count = 0
    print("   Rank | Journal Name                                    | Distance | Current Sim | Pediatric")
    print("   " + "-" * 95)
    
    for i, (dist, sim, idx) in enumerate(zip(distances[0], similarities_current, indices[0]), 1):
        if idx >= 0 and idx < len(journals):
            name = journals[idx].get('display_name', 'Unknown')[:42]
            is_pediatric = any(keyword in name.lower() for keyword in 
                             ['pediatric', 'paediatric', 'child', 'infant', 'neonat'])
            if is_pediatric:
                pediatric_count += 1
            print(f"   {i:4d} | {name:42s} | {dist:8.4f} | {sim:11.4f} | {'YES' if is_pediatric else 'NO'}")
    
    print(f"\n   Pediatric journals in top 10: {pediatric_count}")
    
    # Test correct formula for comparison
    print(f"\n6. Testing alternative similarity formulas:")
    
    # For normalized vectors with L2 distance: cosine_sim = 1 - (L2_dist^2 / 2)
    # But FAISS might be returning squared distances already
    similarities_alt1 = [1.0 - (d / 2.0) for d in distances[0]]  # Current formula
    similarities_alt2 = [1.0 - (d / 4.0) for d in distances[0]]  # If d is already squared
    similarities_alt3 = [max(0.0, 2.0 - d) / 2.0 for d in distances[0]]  # Alternative approach
    
    print(f"   Formula 1 (current): {[f'{s:.3f}' for s in similarities_alt1[:5]]}")
    print(f"   Formula 2 (d/4):     {[f'{s:.3f}' for s in similarities_alt2[:5]]}")
    print(f"   Formula 3 (2-d)/2:   {[f'{s:.3f}' for s in similarities_alt3[:5]]}")
    
    return distances[0], similarities_current, pediatric_count, indices[0]

def test_manual_similarity():
    """Test manual similarity calculation to verify FAISS behavior."""
    print(f"\n7. Manual similarity verification:")
    
    # Load first few journal embeddings
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    
    # Get embeddings for first 5 journals
    embeddings = []
    names = []
    for i, journal in enumerate(data['journals'][:5]):
        if journal.get('embedding'):
            embeddings.append(np.array(journal['embedding']))
            names.append(journal.get('display_name', f'Journal {i}'))
    
    # Test query
    query_text = "machine learning healthcare"
    query_embedding = embed_text(query_text)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    print(f"   Testing manual cosine similarity vs FAISS:")
    print("   Journal                                    | Manual Cosine | Manual L2^2  | FAISS Distance")
    print("   " + "-" * 85)
    
    # Load FAISS index for comparison
    index = faiss.read_index('data/journal_embeddings.faiss')
    faiss_dists, faiss_indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 5)
    
    for i, (emb, name) in enumerate(zip(embeddings, names)):
        # Manual cosine similarity
        cosine_sim = np.dot(query_embedding, emb)
        
        # Manual L2 squared distance
        l2_squared = np.sum((query_embedding - emb) ** 2)
        
        # Find corresponding FAISS distance
        faiss_dist = faiss_dists[0][i] if i < len(faiss_dists[0]) else "N/A"
        
        print(f"   {name[:40]:40s} | {cosine_sim:11.6f} | {l2_squared:10.6f} | {faiss_dist}")
        
        # Verify relationship: L2^2 = 2(1 - cosine_sim) for normalized vectors
        expected_l2_squared = 2 * (1 - cosine_sim)
        print(f"   Expected L2^2 from cosine: {expected_l2_squared:.6f}")

def main():
    """Run comprehensive similarity debugging."""
    try:
        distances, similarities, pediatric_count, indices = test_current_system()
        test_manual_similarity()
        
        print(f"\n" + "=" * 60)
        print("🎯 DIAGNOSIS SUMMARY:")
        
        # Diagnose similarity issue
        if all(0.4 < s < 0.6 for s in similarities):
            print("❌ CONFIRMED: All similarities clustered around 0.5")
            print("   This indicates incorrect distance-to-similarity conversion")
        else:
            print("✅ Similarity scores show proper variety")
        
        # Diagnose pediatric bias
        if pediatric_count >= 5:
            print(f"❌ CONFIRMED: High pediatric bias ({pediatric_count}/10 results)")
            print("   This indicates semantic clustering or query bias issue")
        else:
            print("✅ No significant pediatric bias detected")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("1. Fix similarity conversion formula based on FAISS distance type")
        print("2. Test IndexFlatIP as alternative to IndexIVFFlat")
        print("3. Implement query diversification if semantic bias confirmed")
        print("4. Add similarity validation and bounds checking")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()