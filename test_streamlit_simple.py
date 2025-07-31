#!/usr/bin/env python3
"""
Simplified test to reproduce the Streamlit issues.
"""

import sys
sys.path.append('src')

try:
    print("🧪 SIMPLIFIED STREAMLIT TEST")
    print("=" * 50)
    
    from match_journals import JournalMatcher
    
    # Initialize with minimal components
    matcher = JournalMatcher()
    print("Loading database...")
    
    # Load just the core components, skip problematic ones
    import json
    import faiss
    from journal_db_builder import load_journal_database
    
    matcher.journals, matcher.embeddings = load_journal_database()
    matcher.faiss_index = faiss.read_index('data/journal_embeddings.faiss')
    matcher.embedding_dimension = matcher.embeddings.shape[1]
    
    print(f"✓ Loaded {len(matcher.journals)} journals")
    
    # Test the exact query flow
    query_text = "machine learning applications in pediatric healthcare diagnosis"
    
    # Test WITHOUT multimodal/ranking to isolate the issue
    print(f"\nTesting basic search (no multimodal/ranking)...")
    results_basic = matcher.search_similar_journals(
        query_text=query_text,
        top_k=10,
        min_similarity=0.0,
        filters=None,
        include_study_classification=False,
        use_multimodal_analysis=False,
        use_ensemble_matching=False,
        include_ranking_analysis=False
    )
    
    print(f"Basic search: {len(results_basic)} results")
    
    # Analyze basic results
    similarities_basic = [r.get('similarity_score', 0) for r in results_basic]
    print(f"Basic similarities: {[f'{s:.3f}' for s in similarities_basic[:5]]}")
    
    around_500_basic = sum(1 for s in similarities_basic if abs(s - 0.5) < 0.1)
    print(f"Around 0.500: {around_500_basic}/{len(similarities_basic)}")
    
    # Check pediatric bias in basic results
    pediatric_basic = 0
    print("Basic results:")
    for i, result in enumerate(results_basic[:5], 1):
        name = result.get('display_name', 'Unknown')
        is_pediatric = 'pediatric' in name.lower() or 'child' in name.lower()
        if is_pediatric:
            pediatric_basic += 1
        print(f"  {i}. {name[:40]} {'(PEDIATRIC)' if is_pediatric else ''}")
    
    print(f"Basic pediatric count: {pediatric_basic}/5")
    
    # Now test WITH multimodal analysis to see if that's the issue
    print(f"\nTesting with study classification only...")
    results_study = matcher.search_similar_journals(
        query_text=query_text,
        top_k=10,
        min_similarity=0.0,
        filters=None,
        include_study_classification=True,
        use_multimodal_analysis=False,
        use_ensemble_matching=False,
        include_ranking_analysis=False
    )
    
    similarities_study = [r.get('similarity_score', 0) for r in results_study]
    print(f"Study classification similarities: {[f'{s:.3f}' for s in similarities_study[:5]]}")
    
    # Compare results
    if around_500_basic >= 8:
        print("❌ Issue is in BASIC search (not multimodal)")
    else:
        print("✅ Basic search works fine")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()