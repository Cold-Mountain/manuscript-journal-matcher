#!/usr/bin/env python3
"""
Debug JournalMatcher to find why it's returning wrong results.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def debug_journal_matcher():
    """Debug JournalMatcher step by step."""
    print("🔍 DEBUGGING JournalMatcher")
    print("=" * 60)
    
    user_abstract = "Laparoscopic and robotic bladder diverticulectomy surgical technique"
    
    try:
        from match_journals import JournalMatcher
        from journal_db_builder import load_journal_database
        from embedder import embed_text
        import faiss
        import numpy as np
        
        # Step 1: Check if JournalMatcher uses same data as direct load
        print("STEP 1: Checking data consistency")
        print("-" * 40)
        
        # Load data directly
        direct_journals, direct_embeddings = load_journal_database()
        direct_index = faiss.read_index('data/journal_embeddings.faiss')
        
        # Load via JournalMatcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        print(f"Direct load: {len(direct_journals)} journals, {direct_embeddings.shape} embeddings")
        print(f"JournalMatcher: {len(matcher.journals)} journals, {matcher.embeddings.shape if matcher.embeddings is not None else 'None'} embeddings")
        print(f"Direct index: {type(direct_index).__name__}, {direct_index.ntotal} vectors")
        print(f"Matcher index: {type(matcher.faiss_index).__name__}, {matcher.faiss_index.ntotal} vectors")
        
        # Check if first few journals match
        print(f"\nFirst 3 journals comparison:")
        for i in range(3):
            direct_name = direct_journals[i].get('display_name', 'Unknown')
            matcher_name = matcher.journals[i].get('display_name', 'Unknown')
            match = "✅" if direct_name == matcher_name else "❌"
            print(f"  {i}: Direct='{direct_name}' | Matcher='{matcher_name}' {match}")
        
        print(f"\n" + "=" * 60)
        
        # Step 2: Test minimal JournalMatcher search
        print("STEP 2: Minimal JournalMatcher search")
        print("-" * 40)
        
        # Generate embedding
        query_embedding = embed_text(user_abstract)
        print(f"Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
        
        # Test direct search with matcher's index
        similarities, indices = matcher.faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            5
        )
        
        print(f"Raw FAISS search on matcher's index:")
        for i, (similarity, journal_idx) in enumerate(zip(similarities[0], indices[0])):
            if journal_idx >= len(matcher.journals):
                print(f"  {i+1}. ❌ INDEX ERROR: journal_idx={journal_idx} >= {len(matcher.journals)}")
                continue
                
            journal = matcher.journals[journal_idx]
            journal_name = journal.get('display_name', 'Unknown')
            print(f"  {i+1}. [{journal_idx}] {journal_name[:40]} | {similarity:.6f}")
        
        print(f"\n" + "=" * 60)
        
        # Step 3: Test JournalMatcher with minimal parameters
        print("STEP 3: JournalMatcher with minimal parameters")
        print("-" * 40)
        
        # Test with absolutely minimal parameters
        results = matcher.search_similar_journals(
            query_text=user_abstract,
            top_k=5,
            min_similarity=0.0,
            filters=None,
            include_study_classification=False,
            use_multimodal_analysis=False,
            use_ensemble_matching=False,
            include_ranking_analysis=False
        )
        
        print(f"Minimal JournalMatcher results ({len(results)} found):")
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity_score', 0)
            name = result.get('display_name', 'Unknown')
            rank = result.get('rank', 'N/A')
            print(f"  {i}. [{rank}] {name[:40]} | {similarity:.6f}")
        
        return len(results) == 0 or all(r.get('similarity_score', 0) == 0 for r in results)
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return True

def main():
    """Run debug."""
    has_issues = debug_journal_matcher()
    
    if has_issues:
        print("\n❌ JournalMatcher has serious issues!")
    else:
        print("\n✅ JournalMatcher working correctly")

if __name__ == "__main__":
    main()