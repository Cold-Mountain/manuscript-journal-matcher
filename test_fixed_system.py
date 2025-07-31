#!/usr/bin/env python3
"""
Test the fixed system with simplified search.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_fixed_system():
    """Test the fixed system with all problematic features disabled."""
    print("🔍 TESTING FIXED SYSTEM")
    print("=" * 50)
    
    try:
        from match_journals import JournalMatcher
        
        # Create matcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        print(f"✅ Loaded {len(matcher.journals)} journals")
        
        # Test with the user's problematic abstract
        abstract = """Introduction: Laparoscopic and robotic bladder diverticulectomy is a successful option to correct bladder diverticula (BD). Nevertheless, the identification of BD could be a tricky step, due to the presence of pneumoperitoneum compressing the bladder."""
        
        print(f"\n🔍 Testing with simplified search...")
        
        # Test with all problematic features disabled (same as Streamlit now uses)
        results = matcher.search_similar_journals(
            query_text=abstract,
            top_k=5,
            min_similarity=0.0,
            filters=None,
            include_study_classification=False,
            use_multimodal_analysis=False,
            use_ensemble_matching=False,
            include_ranking_analysis=False
        )
        
        print(f"\n📊 Results ({len(results)} found):")
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity_score', 0)
            name = result.get('display_name', 'Unknown')
            print(f"  {i}. {name[:50]} | {similarity:.6f}")
        
        # Check if we got good results
        if results and results[0].get('similarity_score', 0) > 0.1:
            print(f"\n✅ SYSTEM FIXED! Best similarity: {results[0]['similarity_score']:.6f}")
            print(f"✅ Relevant journal found: {results[0]['display_name']}")
            return True
        else:
            print(f"\n❌ System still broken - getting wrong results")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_system()
    
    if success:
        print("\n🎉 SYSTEM IS NOW WORKING!")
        print("💡 The Streamlit interface should now show correct results")
    else:
        print("\n😞 System still has issues")