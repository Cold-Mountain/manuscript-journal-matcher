#!/usr/bin/env python3
"""
Debug the exact FAISS call that JournalMatcher is making.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def debug_faiss_search():
    """Debug the FAISS search call inside JournalMatcher."""
    print("🔍 DEBUGGING FAISS CALL IN JournalMatcher")
    print("=" * 60)
    
    try:
        from match_journals import JournalMatcher
        from embedder import embed_text
        import numpy as np
        
        # Create matcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        print(f"✅ JournalMatcher loaded with {len(matcher.journals)} journals")
        
        # Generate the same embedding as the matcher would
        query = "Laparoscopic and robotic bladder diverticulectomy surgical technique"
        query_embedding = embed_text(query)
        
        print(f"✅ Generated embedding: norm={np.linalg.norm(query_embedding):.6f}")
        
        # Make the EXACT same FAISS call that JournalMatcher makes
        print(f"\n🔍 Making exact FAISS call as JournalMatcher...")
        
        top_k = 5
        similarities, indices = matcher.faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            min(top_k * 2, len(matcher.journals))  # Same logic as JournalMatcher
        )
        
        print(f"FAISS returned {len(similarities[0])} results")
        
        print(f"\n📊 Raw FAISS results (first 10):")
        for i, (similarity, journal_idx) in enumerate(zip(similarities[0][:10], indices[0][:10])):
            if journal_idx == -1:
                print(f"  {i+1}. ❌ Invalid index: -1")
                continue
            
            if journal_idx >= len(matcher.journals):
                print(f"  {i+1}. ❌ Index out of range: {journal_idx} >= {len(matcher.journals)}")
                continue
            
            journal = matcher.journals[journal_idx]
            journal_name = journal.get('display_name', 'Unknown')
            
            # Process similarity exactly like JournalMatcher does
            if isinstance(matcher.faiss_index, type(matcher.faiss_index)):
                similarity_score = float(similarity)
            
            print(f"  {i+1}. [{journal_idx}] {journal_name[:40]} | raw={similarity:.6f} | processed={similarity_score:.6f}")
        
        # Now let's process results exactly like JournalMatcher does
        print(f"\n🔍 Processing results like JournalMatcher...")
        
        results = []
        for i, (similarity, journal_idx) in enumerate(zip(similarities[0], indices[0])):
            if journal_idx == -1:
                continue
            
            if similarity < 0.0:  # min_similarity = 0.0
                continue
            
            if journal_idx >= len(matcher.journals):
                continue
            
            journal = matcher.journals[journal_idx].copy()
            similarity_score = float(similarity)
            journal['similarity_score'] = similarity_score
            journal['rank'] = i + 1
            
            results.append(journal)
        
        # Sort by similarity 
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        results = results[:top_k]
        
        print(f"\n📊 Processed results ({len(results)} found):")
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity_score', 0)
            name = result.get('display_name', 'Unknown')
            print(f"  {i}. {name[:40]} | {similarity:.6f}")
        
        # Check if results make sense
        if results and results[0].get('similarity_score', 0) > 0.1:
            print(f"\n✅ Results look correct!")
            return False
        else:
            print(f"\n❌ Results are wrong!")
            return True
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return True

if __name__ == "__main__":
    has_issues = debug_faiss_search()
    
    if has_issues:
        print("\n❌ CRITICAL: FAISS call in JournalMatcher is broken")
    else:
        print("\n✅ FAISS call looks correct - issue is elsewhere")