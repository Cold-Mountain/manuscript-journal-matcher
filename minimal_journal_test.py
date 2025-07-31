#!/usr/bin/env python3
"""
Minimal test to isolate the JournalMatcher issue by bypassing all complex features.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def minimal_matcher_test():
    """Test JournalMatcher with absolutely minimal configuration."""
    print("🔍 MINIMAL JournalMatcher TEST")
    print("=" * 60)
    
    user_abstract = "Laparoscopic and robotic bladder diverticulectomy surgical technique"
    
    try:
        from match_journals import JournalMatcher
        import numpy as np
        
        # Create matcher and load database
        matcher = JournalMatcher()
        matcher.load_database()
        
        print(f"✅ Loaded {len(matcher.journals)} journals")
        print(f"✅ FAISS index: {type(matcher.faiss_index).__name__}")
        print(f"✅ Index vectors: {matcher.faiss_index.ntotal}")
        
        # Test with absolutely minimal parameters - no extra features
        print(f"\n🔍 Testing minimal search...")
        
        results = matcher.search_similar_journals(
            query_text=user_abstract,
            top_k=5,
            min_similarity=0.0,  # Set to 0.0 to avoid filtering
            filters=None,  # No filters
            include_study_classification=False,  # Disabled
            use_multimodal_analysis=False,  # Disabled
            use_ensemble_matching=False,  # Disabled
            include_ranking_analysis=False  # Disabled
        )
        
        print(f"\n📊 Results ({len(results)} found):")
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity_score', 0)
            name = result.get('display_name', 'Unknown')
            print(f"  {i}. {name[:40]} | {similarity:.6f}")
        
        # Check if we got the expected results
        if len(results) > 0 and results[0].get('similarity_score', 0) > 0:
            print(f"\n✅ JournalMatcher working correctly!")
            return False
        else:
            print(f"\n❌ JournalMatcher still returning 0.000 similarities!")
            return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return True

if __name__ == "__main__":
    has_issues = minimal_matcher_test()
    
    if has_issues:
        print("\n❌ CRITICAL ISSUE: JournalMatcher is fundamentally broken")
        print("💡 The problem is NOT in complex features - it's in the core search logic")
    else:
        print("\n✅ JournalMatcher working - issue was in complex features")