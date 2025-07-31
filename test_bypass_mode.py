#!/usr/bin/env python3
"""
Test the new bypass mode in JournalMatcher.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

def test_bypass_mode():
    """Test JournalMatcher with bypass mode enabled."""
    print("🔍 TESTING BYPASS MODE")
    print("=" * 60)
    
    user_abstract = "Laparoscopic and robotic bladder diverticulectomy surgical technique"
    
    try:
        from match_journals import JournalMatcher
        
        # Create matcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        print(f"✅ Loaded {len(matcher.journals)} journals")
        
        # Test with bypass mode enabled
        print(f"\n🚀 Testing BYPASS MODE...")
        
        results = matcher.search_similar_journals(
            query_text=user_abstract,
            top_k=5,
            min_similarity=0.0,
            filters=None,
            bypass_mode=True  # Enable bypass mode
        )
        
        print(f"\n📊 BYPASS MODE Results ({len(results)} found):")
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity_score', 0)
            name = result.get('display_name', 'Unknown')
            print(f"  {i}. {name[:50]} | {similarity:.6f}")
        
        # Check if results are good
        if results and results[0].get('similarity_score', 0) > 0.1:
            print(f"\n✅ BYPASS MODE WORKS! Best similarity: {results[0]['similarity_score']:.6f}")
            return True
        else:
            print(f"\n❌ BYPASS MODE failed - still getting wrong results")
            return False
        
    except Exception as e:
        print(f"❌ Bypass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bypass_mode()
    
    if success:
        print("\n🎉 BYPASS MODE IS THE SOLUTION!")
        print("💡 Enable bypass_mode=True in Streamlit for working results")
    else:
        print("\n😞 Even bypass mode doesn't work - deeper issue")