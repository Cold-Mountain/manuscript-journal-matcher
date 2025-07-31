#!/usr/bin/env python3
import sys
sys.path.append('src')

try:
    print("1. Testing imports...")
    import numpy as np
    print("   ✓ numpy imported")
    
    import faiss
    print("   ✓ faiss imported")
    
    import json
    print("   ✓ json imported")
    
    print("2. Testing FAISS index loading...")
    index = faiss.read_index('data/journal_embeddings.faiss')
    print(f"   ✓ FAISS index loaded: {index.ntotal} vectors")
    
    print("3. Testing journal metadata loading...")
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    print(f"   ✓ Journal metadata loaded: {len(journals)} journals")
    
    print("4. Testing embedder...")
    from embedder import embed_text
    print("   ✓ Embedder imported")
    
    test_text = "machine learning healthcare"
    print(f"   Testing with: '{test_text}'")
    embedding = embed_text(test_text)
    print(f"   ✓ Embedding generated: shape {embedding.shape}, norm {np.linalg.norm(embedding):.6f}")
    
    print("5. Testing FAISS search...")
    norm_embedding = embedding / np.linalg.norm(embedding)
    distances, indices = index.search(norm_embedding.reshape(1, -1).astype(np.float32), 3)
    print(f"   ✓ FAISS search completed")
    print(f"   Distances: {distances[0]}")
    print(f"   Indices: {indices[0]}")
    
    print("6. Testing similarity conversion...")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(journals):
            name = journals[idx].get('display_name', 'Unknown')
            similarity = max(0.0, 1.0 - (float(dist) / 2.0))
            print(f"   {i+1}. {name[:40]:40s} | Dist: {dist:.6f} | Sim: {similarity:.6f}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()