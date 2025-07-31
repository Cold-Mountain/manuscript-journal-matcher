#!/usr/bin/env python3
"""
Very simple direct test of journal matching without any complex features.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def direct_search_test():
    """Test direct search functionality step by step."""
    print("🔍 DIRECT SEARCH TEST")
    print("=" * 40)
    
    try:
        from journal_db_builder import load_journal_database
        from embedder import embed_text
        import faiss
        import numpy as np
        
        # Load everything directly
        print("Loading data...")
        journals, embeddings = load_journal_database()
        index = faiss.read_index('data/journal_embeddings.faiss')
        
        print(f"✅ Loaded: {len(journals)} journals")
        print(f"✅ Index: {type(index).__name__} with {index.ntotal} vectors")
        
        # Test query
        query = "Laparoscopic bladder surgery technique"
        print(f"\n🔍 Testing: '{query}'")
        
        # Generate embedding
        embedding = embed_text(query)
        print(f"✅ Generated embedding: norm={np.linalg.norm(embedding):.6f}")
        
        # Search
        similarities, indices = index.search(embedding.reshape(1, -1).astype(np.float32), 5)
        
        print(f"\n📊 Results:")
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(journals):
                name = journals[idx].get('display_name', 'Unknown')
                print(f"  {i+1}. [{idx}] {name[:50]} | {sim:.6f}")
        
        # Check if we got good results
        best_similarity = float(similarities[0][0])
        if best_similarity > 0.1:  # Should be much higher for a relevant query
            print(f"\n✅ Direct search working correctly! Best: {best_similarity:.6f}")
            return True
        else:
            print(f"\n❌ Low similarity scores - possible issue with embeddings")
            return False
            
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = direct_search_test()
    
    if success:
        print("\n✅ DIRECT SEARCH WORKS - Issue is in JournalMatcher wrapper")
    else:
        print("\n❌ DIRECT SEARCH BROKEN - Fundamental problem with data/embeddings")