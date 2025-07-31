#!/usr/bin/env python3
import sys
sys.path.append('src')

# Suppress the multiprocessing warning 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

try:
    print("🔍 TESTING HEALTHCARE QUERY (like user would input)")
    print("=" * 60)
    
    import numpy as np
    import faiss
    import json
    from embedder import embed_text
    
    # Load data
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    index = faiss.read_index('data/journal_embeddings.faiss')
    
    # Test with actual healthcare/medical queries that might trigger pediatric results
    test_queries = [
        "machine learning applications in medical diagnosis and healthcare",
        "artificial intelligence in clinical decision making",
        "deep learning for medical image analysis",
        "predictive models for patient outcomes in hospital settings",
        "automated diagnosis systems using electronic health records"
    ]
    
    for query_text in test_queries:
        print(f"\n📝 Query: '{query_text}'")
        
        # Generate embedding (this is what Streamlit does)
        query_embedding = embed_text(query_text)
        
        # Normalize (crucial step)
        query_norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / query_norm if query_norm > 0 else query_embedding
        
        # Search
        distances, indices = index.search(normalized_query.reshape(1, -1).astype(np.float32), 10)
        
        # Convert similarities using current formula
        similarities = [max(0.0, 1.0 - (d / 2.0)) for d in distances[0]]
        
        print(f"   Distance range: {distances[0].min():.4f} to {distances[0].max():.4f}")
        print(f"   Similarity range: {min(similarities):.4f} to {max(similarities):.4f}")
        
        # Check for 0.5 clustering
        around_half = sum(1 for s in similarities if 0.4 < s < 0.6)
        if around_half >= 8:
            print(f"   ❌ ISSUE: {around_half}/10 similarities around 0.5")
        else:
            print(f"   ✅ Good variety: {around_half}/10 around 0.5")
        
        # Check pediatric bias
        pediatric_count = 0
        pediatric_names = []
        
        print(f"   Top 5 results:")
        for i, (dist, sim, idx) in enumerate(zip(distances[0][:5], similarities[:5], indices[0][:5])):
            if idx < len(journals):
                name = journals[idx].get('display_name', 'Unknown')
                is_pediatric = any(kw in name.lower() for kw in ['pediatric', 'paediatric', 'child', 'infant'])
                if is_pediatric:
                    pediatric_count += 1
                    pediatric_names.append(name)
                print(f"   {i+1}. {name[:45]:45s} | Sim: {sim:.4f} | {'PEDIATRIC' if is_pediatric else ''}")
        
        if pediatric_count >= 3:
            print(f"   ❌ HIGH PEDIATRIC BIAS: {pediatric_count}/5 results")
            print(f"   Pediatric journals: {pediatric_names}")
        else:
            print(f"   ✅ Low pediatric bias: {pediatric_count}/5 results")
        
        print("   " + "-" * 50)
    
    print(f"\n🎯 SUMMARY:")
    print("If any query shows clustering around 0.5 similarities, that's the bug.")
    print("If any query shows high pediatric bias, that indicates semantic clustering.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()