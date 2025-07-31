#!/usr/bin/env python3
import sys
sys.path.append('src')

try:
    print("Step 1: Testing basic imports...")
    import numpy as np
    import faiss
    import json
    print("✓ Basic imports successful")
    
    print("\nStep 2: Loading journal data...")
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    print(f"✓ Loaded {len(journals)} journals")
    
    print("\nStep 3: Loading FAISS index...")
    index = faiss.read_index('data/journal_embeddings.faiss')
    print(f"✓ Loaded FAISS index: {type(index).__name__} with {index.ntotal} vectors")
    
    print("\nStep 4: Testing simple search...")
    # Create a simple test vector
    test_vector = np.random.randn(384).astype(np.float32)
    test_vector = test_vector / np.linalg.norm(test_vector)
    
    distances, indices = index.search(test_vector.reshape(1, -1), 5)
    print(f"✓ FAISS search completed")
    print(f"  Raw distances: {distances[0]}")
    
    print("\nStep 5: Testing similarity conversion...")
    similarities = [max(0.0, 1.0 - (d / 2.0)) for d in distances[0]]
    print(f"  Converted similarities: {similarities}")
    print(f"  Range: {min(similarities):.3f} to {max(similarities):.3f}")
    
    # Check if all similarities are around 0.5
    if all(0.4 < s < 0.6 for s in similarities):
        print("❌ ISSUE CONFIRMED: All similarities around 0.5")
    else:
        print("✅ Similarities show variety")
    
    print("\nStep 6: Analyzing journal results...")
    pediatric_count = 0
    for i, idx in enumerate(indices[0]):
        if idx < len(journals):
            name = journals[idx].get('display_name', 'Unknown')
            is_pediatric = any(kw in name.lower() for kw in ['pediatric', 'child', 'infant'])
            if is_pediatric:
                pediatric_count += 1
            print(f"  {i+1}. {name[:50]} | Pediatric: {is_pediatric}")
    
    print(f"\nPediatric journals in results: {pediatric_count}/5")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()