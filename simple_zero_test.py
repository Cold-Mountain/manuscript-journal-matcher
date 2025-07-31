#!/usr/bin/env python3
"""
Simple test to replicate 0.000 similarities issue without complex imports.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def simple_search_test():
    """Simple search test without complex features."""
    print("🔍 SIMPLE SEARCH TEST")
    print("=" * 50)
    
    try:
        # Basic imports only
        from journal_db_builder import load_journal_database
        from embedder import embed_text
        import faiss
        import numpy as np
        
        # Load data
        print("Loading data...")
        journals, embeddings = load_journal_database()
        index = faiss.read_index('data/journal_embeddings.faiss')
        
        print(f"Loaded: {len(journals)} journals, index: {type(index).__name__}")
        
        # Test the exact query user mentioned
        test_query = "This study investigates the effectiveness of machine learning algorithms in diagnosing cardiovascular diseases using ECG data from 10,000 patients."
        
        print(f"\nTesting: '{test_query[:50]}...'")
        
        # Generate embedding
        query_embedding = embed_text(test_query)
        print(f"Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
        
        # Search
        similarities, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 10)
        
        print(f"\nRaw FAISS results:")
        processed_results = []
        
        for i, (similarity, journal_idx) in enumerate(zip(similarities[0], indices[0])):
            if journal_idx >= len(journals):
                continue
                
            journal = journals[journal_idx]
            
            # Process similarity exactly like JournalMatcher does
            if isinstance(index, faiss.IndexFlatIP):
                similarity_score = float(similarity)
            else:
                raw_distance = float(similarity)
                similarity_score = max(0.0, 1.0 - (raw_distance / 2.0))
            
            result = {
                'rank': i + 1,
                'display_name': journal.get('display_name', 'Unknown'),
                'similarity_score': similarity_score,
                'publisher': journal.get('publisher', 'Unknown')
            }
            
            processed_results.append(result)
            
            print(f"  {i+1}. {result['display_name'][:40]} | {similarity_score:.6f}")
        
        # Check for zeros
        zero_count = sum(1 for r in processed_results if r['similarity_score'] == 0.0)
        
        if zero_count > 0:
            print(f"\n❌ Found {zero_count} results with 0.000 similarity!")
            return True
        else:
            print(f"\n✅ All similarities > 0 (range: {min(r['similarity_score'] for r in processed_results):.3f}-{max(r['similarity_score'] for r in processed_results):.3f})")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return True

def test_min_similarity_filtering():
    """Test if min_similarity filtering is causing the issue."""
    print(f"\n🔍 MIN_SIMILARITY FILTERING TEST")
    print("=" * 50)
    
    try:
        from journal_db_builder import load_journal_database
        from embedder import embed_text
        import faiss
        import numpy as np
        
        journals, embeddings = load_journal_database()
        index = faiss.read_index('data/journal_embeddings.faiss')
        
        query = "machine learning in healthcare"
        query_embedding = embed_text(query)
        
        # Get more results than we need
        similarities, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 20)
        
        print(f"Testing min_similarity filtering...")
        
        # Process all results
        all_processed = []
        for similarity, journal_idx in zip(similarities[0], indices[0]):
            if journal_idx >= len(journals):
                continue
                
            if isinstance(index, faiss.IndexFlatIP):
                similarity_score = float(similarity)
            else:
                similarity_score = max(0.0, 1.0 - (float(similarity) / 2.0))
            
            all_processed.append(similarity_score)
        
        print(f"All similarities: {[f'{s:.3f}' for s in all_processed[:10]]}")
        
        # Test different thresholds
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4]:
            filtered = [s for s in all_processed if s >= threshold]
            print(f"  Threshold {threshold:.1f}: {len(filtered)}/{len(all_processed)} results")
            
            if threshold > 0.0 and len(filtered) == 0:
                print(f"    ⚠️  All results filtered out at threshold {threshold:.1f}")
                
                # This could be the issue - if user accidentally sets high min_similarity
                # or if there's a bug setting it incorrectly
                if max(all_processed) < threshold:
                    print(f"    ❌ Max similarity {max(all_processed):.3f} < threshold {threshold:.1f}")
                    print(f"    This would cause 0 results, appearing as 0.000 similarities!")
                    return True
        
        return False
        
    except Exception as e:
        print(f"❌ Filtering test failed: {e}")
        return True

def main():
    """Run simple tests."""
    print("🎯 SIMPLE 0.000 SIMILARITY INVESTIGATION")
    print("=" * 60)
    
    issues = []
    
    if simple_search_test():
        issues.append("Core search returning 0.000 similarities")
    
    if test_min_similarity_filtering():
        issues.append("Min_similarity filtering causing no results")
    
    print(f"\n" + "=" * 60)
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Core similarity calculation working correctly")
        print("\n💡 NEXT STEPS:")
        print("   1. Test in actual Streamlit interface")
        print("   2. Check if user is setting high min_similarity")
        print("   3. Look for session state or UI bugs")
        print("   4. Test with exact user input (copy/paste artifacts)")

if __name__ == "__main__":
    main()