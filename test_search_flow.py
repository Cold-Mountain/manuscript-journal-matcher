#!/usr/bin/env python3
"""
Test the full search flow to find where 0.000 similarities appear.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_search_flow_step_by_step():
    """Test the search flow step by step to find where zeros appear."""
    print("🔍 STEP-BY-STEP SEARCH FLOW TEST")
    print("=" * 60)
    
    try:
        from match_journals import JournalMatcher
        import numpy as np
        
        # Initialize
        matcher = JournalMatcher()
        matcher.load_database()
        
        print(f"✅ Initialized matcher with {len(matcher.journals)} journals")
        print(f"✅ FAISS index: {type(matcher.faiss_index).__name__}")
        
        # Test query that user mentioned causing issues
        query = "This study investigates the effectiveness of machine learning algorithms in diagnosing cardiovascular diseases using ECG data from 10,000 patients."
        
        print(f"\n🔍 Testing problematic query: '{query[:60]}...'")
        
        # Step 1: Generate embedding
        from embedder import embed_text
        query_embedding = embed_text(query)
        print(f"1. Query embedding generated: norm={np.linalg.norm(query_embedding):.6f}")
        
        # Step 2: FAISS search directly
        similarities, indices = matcher.faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            10  # Top 10 results
        )
        
        print(f"2. Raw FAISS search results:")
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(matcher.journals):
                journal_name = matcher.journals[idx].get('display_name', 'Unknown')
                print(f"   {i+1}. {journal_name[:40]}: {sim:.6f}")
        
        # Step 3: Test with different min_similarity values
        print(f"\n3. Testing with different min_similarity thresholds:")
        
        for min_sim in [0.0, 0.1, 0.3, 0.5]:
            print(f"\n   Testing min_similarity = {min_sim}")
            
            try:
                results = matcher.search_similar_journals(
                    query_text=query,
                    top_k=5,
                    min_similarity=min_sim,
                    filters=None,
                    include_study_classification=False,  # Disable to simplify
                    use_multimodal_analysis=False,       # Disable to simplify
                    use_ensemble_matching=False,
                    include_ranking_analysis=False
                )
                
                if results:
                    sims = [r.get('similarity_score', 0) for r in results]
                    print(f"      Got {len(results)} results: {[f'{s:.3f}' for s in sims]}")
                    
                    # Check for zeros
                    zero_count = sum(1 for s in sims if s == 0.0)
                    if zero_count > 0:
                        print(f"      ❌ Found {zero_count} zero similarities!")
                        
                        # Debug the first zero result
                        for result in results:
                            if result.get('similarity_score', 0) == 0.0:
                                print(f"         Zero result: {result.get('display_name', 'Unknown')}")
                                print(f"         Raw data: {result.get('similarity_score')}")
                                return True
                else:
                    print(f"      No results (filtered out by min_similarity)")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        # Step 4: Test with all complex features enabled (like Streamlit does)
        print(f"\n4. Testing with full Streamlit-like configuration:")
        
        try:
            results = matcher.search_similar_journals(
                query_text=query,
                top_k=10,
                min_similarity=0.0,
                filters=None,
                include_study_classification=True,
                use_multimodal_analysis=True,
                use_ensemble_matching=False,
                include_ranking_analysis=True
            )
            
            if results:
                sims = [r.get('similarity_score', 0) for r in results[:5]]
                print(f"   Full config results: {[f'{s:.3f}' for s in sims]}")
                
                zero_count = sum(1 for s in sims if s == 0.0)
                if zero_count > 0:
                    print(f"   ❌ Found {zero_count} zero similarities with full config!")
                    return True
                else:
                    print(f"   ✅ All similarities > 0 with full config")
            else:
                print(f"   ❌ No results with full config!")
                return True
                
        except Exception as e:
            print(f"   ❌ Full config error: {e}")
            import traceback
            traceback.print_exc()
            return True
        
        print(f"\n✅ All search flow tests passed - no 0.000 similarities found")
        return False
        
    except Exception as e:
        print(f"❌ Search flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return True

def test_specific_user_case():
    """Test the specific case the user mentioned."""
    print(f"\n🔍 TESTING USER-SPECIFIC CASE")
    print("=" * 60)
    
    # The user mentioned "physical chemistry" was showing up incorrectly
    # and giving 0.000 similarity scores
    
    user_queries = [
        # User mentioned this was giving physical chemistry results incorrectly
        "Machine Learning for Predictive Analytics in Pediatric Emergency Medicine. Emergency departments face increasing pressure to provide rapid, accurate diagnoses for pediatric patients.",
        
        # User mentioned getting "bunch of the same pediatric journals"
        "pediatric emergency medicine diagnosis treatment",
        
        # Physical chemistry query that might be confusing the system
        "physical chemistry molecular dynamics computational analysis",
    ]
    
    try:
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        for i, query in enumerate(user_queries, 1):
            print(f"\nUser case {i}: '{query[:50]}...'")
            
            results = matcher.search_similar_journals(
                query_text=query,
                top_k=10,
                min_similarity=0.0,
                filters=None,
                include_study_classification=True,
                use_multimodal_analysis=True,
                use_ensemble_matching=False,
                include_ranking_analysis=True
            )
            
            if results:
                print(f"   Got {len(results)} results:")
                
                # Check for clustering issues (same specialty)
                specialties = []
                zero_sims = []
                
                for j, result in enumerate(results[:5], 1):
                    sim = result.get('similarity_score', 0)
                    name = result.get('display_name', 'Unknown')
                    
                    # Extract specialty from name
                    name_lower = name.lower()
                    if 'pediatric' in name_lower or 'paediatric' in name_lower:
                        specialty = 'pediatric'
                    elif 'chemistry' in name_lower or 'chemical' in name_lower:
                        specialty = 'chemistry'
                    elif 'emergency' in name_lower:
                        specialty = 'emergency'
                    else:
                        specialty = 'other'
                    
                    specialties.append(specialty)
                    
                    print(f"   {j}. {name[:40]} | {sim:.3f} | {specialty}")
                    
                    if sim == 0.0:
                        zero_sims.append(j)
                
                # Analysis
                if zero_sims:
                    print(f"   ❌ Found zero similarities at positions: {zero_sims}")
                    return True
                
                # Check for clustering
                from collections import Counter
                specialty_counts = Counter(specialties)
                if any(count >= 4 for count in specialty_counts.values()):
                    most_common = specialty_counts.most_common(1)[0]
                    print(f"   ⚠️  Specialty clustering detected: {most_common[1]} '{most_common[0]}' journals")
                    # This isn't a 0.000 issue but explains user's "same pediatric journals" complaint
                
            else:
                print(f"   ❌ No results returned")
        
        return False
        
    except Exception as e:
        print(f"❌ User case test failed: {e}")
        return True

def main():
    """Run search flow tests."""
    issues = []
    
    if test_search_flow_step_by_step():
        issues.append("Search flow producing 0.000 similarities")
    
    if test_specific_user_case():
        issues.append("User-specific cases showing 0.000 similarities")
    
    print(f"\n" + "=" * 60)
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("   1. Check Streamlit session state persistence")
        print("   2. Test in actual Streamlit interface")
        print("   3. Look for copy/paste formatting issues")
        print("   4. Check browser console for JavaScript errors")
    else:
        print("✅ No search flow issues found")
        print("   The 0.000 similarity issue might be:")
        print("   - Specific to Streamlit environment")
        print("   - Browser-related formatting issues")
        print("   - Session state or caching problems")
        print("   - User interface interaction bugs")

if __name__ == "__main__":
    main()