#!/usr/bin/env python3
"""
Test the exact user abstract that's showing 0.000 similarities in Streamlit.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_user_abstract():
    """Test the exact abstract that user reported showing 0.000 similarities."""
    print("🔍 TESTING USER'S EXACT ABSTRACT")
    print("=" * 60)
    
    # The exact abstract user provided
    user_abstract = """Introduction: Laparoscopic and robotic bladder diverticulectomy is a successful option to correct bladder diverticula (BD). Nevertheless, the identification of BD could be a tricky step, due to the presence of pneumoperitoneum compressing the bladder. This occurrence could be particularly evident for the posterior or postero-lateral location of BDs. We present a novel technique to overcome this concern based on a rigid guidewire previously endoscopically placed and coiled inside BD, to ensure it expands and remains stable during the dissection. The technique was used in cases of diverticulectomy concomi- tant to other prostatic procedures. Methods: This is a multicentric series of laparoscopic and robotic diverticulectomy performed with this original technique in 34 patients. The procedure was concomitant to other prostatic intervention in most of the cases: TURP or bladder neck incision (16); radical prostatectomy (three); Millin adenomectomy (four cases). Surgical procedure: The first step of the procedure endoscopic, consisting of the retrograde insertion of a stiff guidewire inside the BD via cystoscopy; the guidewire is pushed in until it coils inside the diverticulum, and then enlarged to make it visible transperitoneally. The guidewire stretches the diverticulum and guides the dissection up to identify its neck. The primary endpoint is to address the feasibility of the technique by considering the operative time (OT, min) and the complication rate. Results: The median size of the BDs was 5.1 cm. The location of the BD was postero-lateral or posterior in all except one case. Bladder diverticulectomy was laparoscopically performed in 25 and robotically assisted in nine cases. Median OT was 179 min (DS 42). The post-operative course was uneventful for all except two patients with symptomatic urinary tract infections. Conclusions: The use of a stiff guidewire coiling and expanding the BD is a simple and useful trick to aid BD's identification and dissection; it aids diverticulectomy and is also concomitant to other prostatic procedures."""
    
    try:
        # Test 1: Direct core search (like simple_zero_test.py)
        print("TEST 1: Direct FAISS search")
        print("-" * 30)
        
        from journal_db_builder import load_journal_database
        from embedder import embed_text
        import faiss
        import numpy as np
        
        journals, embeddings = load_journal_database()
        index = faiss.read_index('data/journal_embeddings.faiss')
        
        print(f"Loaded: {len(journals)} journals, index: {type(index).__name__}")
        
        # Generate embedding
        query_embedding = embed_text(user_abstract)
        print(f"Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
        
        # Direct FAISS search
        similarities, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 10)
        
        print(f"Direct FAISS results:")
        for i, (similarity, journal_idx) in enumerate(zip(similarities[0], indices[0])):
            if journal_idx >= len(journals):
                continue
                
            journal = journals[journal_idx]
            
            # Process similarity exactly like JournalMatcher does
            if isinstance(index, faiss.IndexFlatIP):
                similarity_score = float(similarity)
            else:
                similarity_score = max(0.0, 1.0 - (float(similarity) / 2.0))
            
            journal_name = journal.get('display_name', 'Unknown')
            print(f"  {i+1}. {journal_name[:40]} | {similarity_score:.6f}")
        
        print(f"\n" + "=" * 60)
        
        # Test 2: JournalMatcher search (how Streamlit uses it)
        print("TEST 2: JournalMatcher.search_similar_journals()")
        print("-" * 30)
        
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Use the exact same parameters as Streamlit might use
        results = matcher.search_similar_journals(
            query_text=user_abstract,
            top_k=10,
            min_similarity=0.0,  # Start with 0.0 to avoid filtering
            filters=None,
            include_study_classification=True,
            use_multimodal_analysis=True,
            use_ensemble_matching=False,
            include_ranking_analysis=True
        )
        
        print(f"JournalMatcher results ({len(results)} found):")
        for i, result in enumerate(results[:10], 1):
            similarity = result.get('similarity_score', 0)
            name = result.get('display_name', 'Unknown')
            print(f"  {i}. {name[:40]} | {similarity:.6f}")
            
        print(f"\n" + "=" * 60)
        
        # Test 3: Check if there's a difference in similarity values
        if results:
            direct_similarities = [float(s) for s in similarities[0][:10]]
            matcher_similarities = [r.get('similarity_score', 0) for r in results[:10]]
            
            print("COMPARISON:")
            print("-" * 30)
            print("Direct FAISS vs JournalMatcher results:")
            for i, (direct, matcher) in enumerate(zip(direct_similarities, matcher_similarities)):
                diff = abs(direct - matcher)
                status = "✅" if diff < 0.001 else "❌"
                print(f"  {i+1}. Direct: {direct:.6f} | Matcher: {matcher:.6f} | Diff: {diff:.6f} {status}")
            
            if any(abs(d - m) > 0.001 for d, m in zip(direct_similarities, matcher_similarities)):
                print("\n❌ FOUND DISCREPANCY between direct FAISS and JournalMatcher!")
                return True
            else:
                print("\n✅ Direct FAISS and JournalMatcher results match")
                return False
        else:
            print("❌ JournalMatcher returned NO RESULTS!")
            return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return True

def main():
    """Run user abstract test."""
    print("🎯 USER ABSTRACT INVESTIGATION")
    print("=" * 60)
    
    has_issues = test_user_abstract()
    
    print(f"\n" + "=" * 60)
    if has_issues:
        print("❌ Issues found with user's abstract processing")
        print("\n💡 LIKELY CAUSES:")
        print("   1. JournalMatcher parameter handling issue")
        print("   2. Session state problems in Streamlit")
        print("   3. Filtering logic removing all results")
        print("   4. Multi-modal analysis interference")
    else:
        print("✅ User abstract processing working correctly")
        print("\n💡 ISSUE LIKELY IN STREAMLIT UI:")
        print("   1. Session state not properly initialized")
        print("   2. UI parameter passing incorrectly")
        print("   3. Result formatting/display bug")

if __name__ == "__main__":
    main()