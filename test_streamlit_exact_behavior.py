#!/usr/bin/env python3
"""
Test to replicate the exact Streamlit behavior and find the 0.000 similarity issue.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_streamlit_exact_flow():
    """Test the exact same flow as Streamlit does."""
    print("🔍 TESTING STREAMLIT EXACT FLOW")
    print("=" * 60)
    
    try:
        from match_journals import JournalMatcher
        
        # Initialize exactly as Streamlit does
        print("1. Initializing matcher...")
        matcher = JournalMatcher()
        
        print("2. Loading database...")
        matcher.load_database()
        
        print(f"✅ Loaded {len(matcher.journals)} journals")
        print(f"✅ FAISS index: {type(matcher.faiss_index).__name__}")
        
        # Test the exact sample abstracts from Streamlit
        sample_abstracts = {
            "Medical Research": "This study investigates the effectiveness of machine learning algorithms in diagnosing cardiovascular diseases using ECG data from 10,000 patients.",
            "Computer Science": "We present a novel deep learning architecture for natural language processing tasks, achieving state-of-the-art performance on multiple benchmarks.",
            "Biology": "Our research examines the genetic mechanisms underlying cellular differentiation in stem cells using CRISPR-Cas9 gene editing techniques.",
        }
        
        # Test each sample with exact Streamlit parameters
        for name, abstract in sample_abstracts.items():
            print(f"\n3. Testing '{name}' sample...")
            print(f"   Abstract: '{abstract[:50]}...'")
            
            # Call with exact same parameters as Streamlit default
            results = matcher.search_similar_journals(
                query_text=abstract,
                top_k=10,                    # Default from Streamlit
                min_similarity=0.0,          # Default from Streamlit  
                filters=None,                # No filters by default
                include_study_classification=True,   # Default from Streamlit
                use_multimodal_analysis=True,        # Default from Streamlit
                use_ensemble_matching=False,         # Default from Streamlit
                include_ranking_analysis=True       # Default from Streamlit
            )
            
            if results:
                similarities = [r.get('similarity_score', 0) for r in results[:5]]
                print(f"   ✅ Got {len(results)} results")
                print(f"   Similarities: {[f'{s:.3f}' for s in similarities]}")
                
                # Check for 0.000 issue
                zero_count = sum(1 for s in similarities if s == 0.0)
                if zero_count > 0:
                    print(f"   ❌ Found {zero_count} zero similarities!")
                    
                    # Debug the first zero result
                    for i, result in enumerate(results[:3]):
                        sim = result.get('similarity_score', 0)
                        name = result.get('display_name', 'Unknown')
                        print(f"      Result {i+1}: {name[:30]} -> {sim:.6f}")
                    
                    return True  # Found the issue
                else:
                    print(f"   ✅ All similarities > 0")
            else:
                print(f"   ❌ No results returned!")
                return True
        
        print(f"\n✅ All Streamlit sample abstracts working correctly")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return True

def test_user_problematic_input():
    """Test the type of input that might cause 0.000 similarities."""
    print(f"\n🔍 TESTING USER PROBLEMATIC INPUT")
    print("=" * 60)
    
    # Examples of inputs that might cause issues based on user feedback
    problematic_inputs = [
        # Based on "physical chemistry" issue user mentioned
        "Physical chemistry molecular dynamics simulations and computational methods for analyzing chemical reactions",
        
        # Very short input
        "machine learning",
        
        # Generic medical text
        "This study examines patients with medical conditions",
        
        # Copy-paste artifacts (common issue)
        "Title: Machine Learning\n\nAbstract: This study...\n\nKeywords: ML, AI",
        
        # Text with unusual characters or formatting
        "This study investigates the effectiveness of AI algorithms—using data from 10,000 patients—in medical diagnosis.",
    ]
    
    try:
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        for i, query in enumerate(problematic_inputs, 1):
            print(f"\nTest {i}: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            try:
                results = matcher.search_similar_journals(
                    query_text=query,
                    top_k=5,
                    min_similarity=0.0,
                    filters=None,
                    include_study_classification=True,
                    use_multimodal_analysis=True,
                    use_ensemble_matching=False,
                    include_ranking_analysis=True
                )
                
                if results:
                    similarities = [r.get('similarity_score', 0) for r in results]
                    print(f"   Similarities: {[f'{s:.3f}' for s in similarities]}")
                    
                    if all(s == 0.0 for s in similarities):
                        print(f"   ❌ ALL SIMILARITIES ARE 0.000 - FOUND THE ISSUE!")
                        
                        # Debug this specific case
                        print(f"   🔍 Debugging this input...")
                        print(f"      Query length: {len(query)} chars")
                        print(f"      First result: {results[0].get('display_name', 'Unknown')}")
                        print(f"      Raw similarity: {results[0].get('similarity_score')}")
                        
                        return True
                    elif any(s == 0.0 for s in similarities):
                        zero_count = sum(1 for s in similarities if s == 0.0)
                        print(f"   ⚠️  {zero_count} zero similarities found")
                    else:
                        print(f"   ✅ All similarities > 0")
                else:
                    print(f"   ❌ No results returned")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return True

def main():
    """Run all tests to find the 0.000 similarity issue."""
    print("🎯 FINDING THE 0.000 SIMILARITY ISSUE")
    print("=" * 70)
    
    issues_found = []
    
    # Test 1: Streamlit exact flow
    if test_streamlit_exact_flow():
        issues_found.append("Streamlit default samples showing 0.000 similarities")
    
    # Test 2: User problematic inputs  
    if test_user_problematic_input():
        issues_found.append("User-type inputs causing 0.000 similarities")
    
    print(f"\n" + "=" * 70)
    print("🎯 DIAGNOSIS:")
    
    if issues_found:
        print("❌ Issues found:")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("✅ No 0.000 similarity issues found")
        print("   The problem might be:")
        print("   - Specific to certain text inputs not tested")
        print("   - Related to browser copy/paste formatting")
        print("   - Caused by Streamlit session state issues")
        print("   - Min similarity filtering in UI")

if __name__ == "__main__":
    main()