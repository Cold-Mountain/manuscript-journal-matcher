#!/usr/bin/env python3
"""
Quick test of bypass mode without excessive logging.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable INFO logging to reduce noise
import logging
logging.getLogger().setLevel(logging.WARNING)

def quick_bypass_test():
    """Quick test of bypass mode."""
    print("🔍 QUICK BYPASS TEST")
    print("=" * 40)
    
    try:
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        # User's problematic abstract
        abstract = """Introduction: Laparoscopic and robotic bladder diverticulectomy is a successful option to correct bladder diverticula (BD). Nevertheless, the identification of BD could be a tricky step, due to the presence of pneumoperitoneum compressing the bladder."""
        
        # Test bypass mode
        results = matcher.search_similar_journals(
            query_text=abstract,
            top_k=3,
            min_similarity=0.0,
            bypass_mode=True
        )
        
        print(f"Results: {len(results)}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.get('display_name', 'Unknown')[:40]} | {r.get('similarity_score', 0):.3f}")
        
        if results and results[0].get('similarity_score', 0) > 0.1:
            print("✅ BYPASS MODE WORKING!")
            return True
        else:
            print("❌ BYPASS MODE FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    quick_bypass_test()