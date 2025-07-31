#!/usr/bin/env python3
"""
Debug Streamlit-specific issues causing 0.000 similarities
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_streamlit_matcher_directly():
    """Test the exact matcher used by Streamlit."""
    print("🔍 TESTING STREAMLIT MATCHER DIRECTLY")
    print("=" * 60)
    
    try:
        from match_journals import JournalMatcher
        
        # Initialize exactly as Streamlit does
        matcher = JournalMatcher()
        
        # Load components manually to avoid issues
        from journal_db_builder import load_journal_database
        import faiss
        
        print("Loading database components...")
        matcher.journals, matcher.embeddings = load_journal_database()
        matcher.faiss_index = faiss.read_index('data/journal_embeddings.faiss')
        matcher.embedding_dimension = matcher.embeddings.shape[1]
        
        print(f"✅ Loaded {len(matcher.journals)} journals")
        print(f"✅ FAISS index: {type(matcher.faiss_index).__name__}")
        
        # Test with very simple text first
        simple_queries = [
            "machine learning",
            "healthcare artificial intelligence", 
            "medical diagnosis using AI",
            "pediatric cardiology research",
            "physical chemistry analysis"
        ]
        
        for query in simple_queries:
            print(f"\nTesting: '{query}'")
            
            try:
                # Call search exactly as Streamlit does but with minimal params
                results = matcher.search_similar_journals(
                    query_text=query,
                    top_k=5,
                    min_similarity=0.0,
                    filters=None,
                    include_study_classification=False,
                    use_multimodal_analysis=False,
                    use_ensemble_matching=False,
                    include_ranking_analysis=False
                )
                
                if results:
                    similarities = [r.get('similarity_score', 0) for r in results]
                    print(f"   ✅ Similarities: {[f'{s:.3f}' for s in similarities[:3]]}")
                    
                    if all(s == 0.0 for s in similarities):
                        print(f"   ❌ All similarities are 0.000!")
                        return True
                    elif any(s < 0.001 for s in similarities):
                        print(f"   ⚠️  Some similarities very low")
                else:
                    print(f"   ❌ No results returned")
                    return True
                    
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
                import traceback
                traceback.print_exc()
                return True
        
        print(f"\n✅ All simple queries working")
        return False
        
    except Exception as e:
        print(f"❌ Matcher initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return True

def test_problematic_inputs():
    """Test inputs that might cause 0.000 similarities."""
    print(f"\n🔍 TESTING PROBLEMATIC INPUTS")
    print("=" * 60)
    
    # These are examples of inputs that might cause issues
    problematic_inputs = [
        "",  # Empty
        "   ",  # Just whitespace
        "a",  # Too short
        "test test test test test",  # Repetitive
        "12345 67890 numbers only",  # Mostly numbers
        "!@#$% special chars only &*()",  # Special characters
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",  # Generic lorem ipsum
    ]
    
    try:
        from match_journals import JournalMatcher
        from journal_db_builder import load_journal_database
        import faiss
        
        matcher = JournalMatcher()
        matcher.journals, matcher.embeddings = load_journal_database()
        matcher.faiss_index = faiss.read_index('data/journal_embeddings.faiss')
        matcher.embedding_dimension = matcher.embeddings.shape[1]
        
        for i, query in enumerate(problematic_inputs, 1):
            print(f"\nTest {i}: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            try:
                results = matcher.search_similar_journals(
                    query_text=query,
                    top_k=3,
                    min_similarity=0.0,
                    filters=None,
                    include_study_classification=False,
                    use_multimodal_analysis=False,
                    use_ensemble_matching=False,
                    include_ranking_analysis=False
                )
                
                if results:
                    similarities = [r.get('similarity_score', 0) for r in results[:3]]
                    print(f"   Similarities: {[f'{s:.3f}' for s in similarities]}")
                    
                    if all(s == 0.0 for s in similarities):
                        print(f"   ❌ Causes 0.000 similarities")
                else:
                    print(f"   No results (likely filtered out)")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")

def check_min_similarity_filtering():
    """Check if min_similarity filtering is causing issues."""
    print(f"\n🔍 CHECKING MIN_SIMILARITY FILTERING")
    print("=" * 60)
    
    try:
        from match_journals import JournalMatcher
        from journal_db_builder import load_journal_database
        import faiss
        
        matcher = JournalMatcher()
        matcher.journals, matcher.embeddings = load_journal_database()
        matcher.faiss_index = faiss.read_index('data/journal_embeddings.faiss')
        matcher.embedding_dimension = matcher.embeddings.shape[1]
        
        test_query = "machine learning in healthcare diagnosis"
        
        # Test with different min_similarity thresholds
        thresholds = [0.0, 0.1, 0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            results = matcher.search_similar_journals(
                query_text=test_query,
                top_k=10,
                min_similarity=threshold,
                filters=None,
                include_study_classification=False,
                use_multimodal_analysis=False,
                use_ensemble_matching=False,
                include_ranking_analysis=False
            )
            
            if results:
                similarities = [r.get('similarity_score', 0) for r in results]
                print(f"   Threshold {threshold:.1f}: {len(results)} results, max_sim={max(similarities):.3f}")
            else:
                print(f"   Threshold {threshold:.1f}: 0 results (filtered out)")
        
    except Exception as e:
        print(f"❌ Filtering test failed: {e}")

def main():
    """Main debugging function."""
    issues = []
    
    if test_streamlit_matcher_directly():
        issues.append("Streamlit matcher returning 0.000 similarities")
    
    test_problematic_inputs()
    check_min_similarity_filtering()
    
    print(f"\n" + "=" * 60)
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Streamlit matcher working correctly")
        print("   The 0.000 similarity issue may be specific to:")
        print("   - The exact text you entered")
        print("   - Streamlit's text preprocessing")
        print("   - Browser copy/paste issues with special characters")

if __name__ == "__main__":
    main()