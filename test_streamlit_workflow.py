#!/usr/bin/env python3
"""
Test the exact same workflow that Streamlit uses, step by step.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_streamlit_workflow():
    """Test the exact workflow that Streamlit uses."""
    print("🔍 TESTING STREAMLIT WORKFLOW")
    print("=" * 50)
    
    try:
        # Import exactly what Streamlit imports
        from match_journals import JournalMatcher
        
        # Create matcher exactly like Streamlit does
        print("Step 1: Creating JournalMatcher...")
        matcher = JournalMatcher()
        
        print("Step 2: Loading database...")
        matcher.load_database()
        
        print(f"✅ Loaded {len(matcher.journals)} journals")
        
        # Test query
        abstract_text = "Laparoscopic bladder surgery technique"
        
        print("Step 3: Performing search with exact Streamlit parameters...")
        
        # Use the EXACT same call that Streamlit makes
        results = matcher.search_similar_journals(
            query_text=abstract_text,
            top_k=10,
            min_similarity=0.0,
            filters=None,
            include_study_classification=False,
            use_multimodal_analysis=False,
            use_ensemble_matching=False,
            include_ranking_analysis=False
        )
        
        print(f"Step 4: Processing results...")
        print(f"Got {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            similarity = result.get('similarity_score', 0)
            name = result.get('display_name', 'Unknown')
            print(f"  {i}. {name[:40]} | {similarity:.6f}")
        
        return len(results) > 0 and results[0].get('similarity_score', 0) > 0.1
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing the exact Streamlit workflow...")
    success = test_streamlit_workflow()
    
    if success:
        print("\n✅ Streamlit workflow should work!")
    else:
        print("\n❌ Streamlit workflow still broken")