#!/usr/bin/env python3
"""
Test that exactly replicates Streamlit app behavior.
"""

import sys
sys.path.append('src')

try:
    print("🧪 EXACT STREAMLIT APP REPLICATION TEST")
    print("=" * 60)
    
    # Initialize matcher exactly like Streamlit does
    from match_journals import JournalMatcher
    
    print("Initializing JournalMatcher...")
    matcher = JournalMatcher()
    matcher.load_database()
    print(f"✓ Matcher loaded with {len(matcher.journals)} journals")
    
    # Test query similar to what user would input
    query_text = "machine learning applications in pediatric healthcare diagnosis and treatment"
    
    print(f"\nTesting with query: '{query_text}'")
    
    # Call search_similar_journals exactly like Streamlit does
    print("Calling search_similar_journals with Streamlit parameters...")
    results = matcher.search_similar_journals(
        query_text=query_text,
        top_k=10,
        min_similarity=0.0,  # Same as Streamlit default
        filters=None,
        include_study_classification=True,  # Same as Streamlit
        use_multimodal_analysis=True,      # Same as Streamlit  
        use_ensemble_matching=False,       # Same as Streamlit default
        include_ranking_analysis=True      # Same as Streamlit
    )
    
    print(f"✓ Search completed, got {len(results)} results")
    
    # Analyze results exactly like the user reported
    print(f"\nAnalyzing results for the reported issues:")
    
    similarities = [r.get('similarity_score', 0) for r in results]
    
    print(f"Similarity scores: {[f'{s:.3f}' for s in similarities[:10]]}")
    print(f"Similarity range: {min(similarities):.3f} to {max(similarities):.3f}")
    
    # Check for 0.500 clustering
    around_500 = sum(1 for s in similarities if abs(s - 0.5) < 0.1)
    print(f"Similarities around 0.500 (±0.1): {around_500}/{len(similarities)}")
    
    if around_500 >= 8:
        print("❌ ISSUE CONFIRMED: Most similarities around 0.500")
    else:
        print("✅ Similarities show proper variety")
    
    # Check pediatric bias
    pediatric_count = 0
    pediatric_names = []
    
    print(f"\nTop 10 journal results:")
    for i, result in enumerate(results[:10], 1):
        name = result.get('display_name', 'Unknown')
        similarity = result.get('similarity_score', 0)
        
        is_pediatric = any(keyword in name.lower() for keyword in 
                          ['pediatric', 'paediatric', 'child', 'infant', 'neonat'])
        
        if is_pediatric:
            pediatric_count += 1
            pediatric_names.append(name)
        
        print(f"{i:2d}. {name[:50]:50s} | Sim: {similarity:.3f} | {'PEDIATRIC' if is_pediatric else ''}")
    
    print(f"\nPediatric journals in top 10: {pediatric_count}")
    if pediatric_count >= 5:
        print("❌ HIGH PEDIATRIC BIAS CONFIRMED")
        print(f"Pediatric journals: {pediatric_names}")
    else:
        print("✅ No significant pediatric bias")
    
    # Check if first result is "The Lancet Child and Adolescent Health"
    if results and 'lancet' in results[0].get('display_name', '').lower() and 'child' in results[0].get('display_name', '').lower():
        print("❌ CONFIRMED: Top result is Lancet Child as user reported")
    
    print(f"\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()