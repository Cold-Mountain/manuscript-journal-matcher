#!/usr/bin/env python3
"""
Direct test of similarity calculation to find 0.000 issue.
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np

def test_direct_similarity():
    """Test similarity calculation directly."""
    print("🔍 DIRECT SIMILARITY TEST")
    print("=" * 50)
    
    try:
        # Import the core components
        from journal_db_builder import load_journal_database
        from embedder import embed_text
        import faiss
        
        print("1. Loading database...")
        journals, embeddings = load_journal_database()
        print(f"   Loaded {len(journals)} journals, embeddings shape: {embeddings.shape}")
        
        print("2. Loading FAISS index...")
        index = faiss.read_index('data/journal_embeddings.faiss')
        print(f"   Index type: {type(index).__name__}")
        print(f"   Index vectors: {index.ntotal}")
        
        # Test queries that might cause issues
        test_queries = [
            "machine learning algorithms in diagnosing cardiovascular diseases using ECG data",
            "physical chemistry molecular dynamics simulations",
            "This study investigates the effectiveness of machine learning algorithms in diagnosing cardiovascular diseases using ECG data from 10,000 patients.",
            "deep learning architecture for natural language processing tasks",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n3.{i} Testing query: '{query[:50]}...'")
            
            # Generate embedding
            query_embedding = embed_text(query)
            print(f"     Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
            
            # Search with FAISS
            similarities, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 5)
            
            print(f"     Raw FAISS results:")
            print(f"       Similarities: {similarities[0]}")
            print(f"       Indices: {indices[0]}")
            
            # Convert to similarity scores like the matcher does
            processed_similarities = []
            for similarity, journal_idx in zip(similarities[0], indices[0]):
                if isinstance(index, faiss.IndexFlatIP):
                    # Direct cosine similarity from IndexFlatIP
                    similarity_score = float(similarity)
                else:
                    # Legacy support for distance-based indices
                    raw_distance = float(similarity)
                    similarity_score = max(0.0, 1.0 - (raw_distance / 2.0))
                
                processed_similarities.append(similarity_score)
                
                if journal_idx < len(journals):
                    journal_name = journals[journal_idx].get('display_name', 'Unknown')
                    print(f"       {journal_name[:30]}: raw={similarity:.6f} -> processed={similarity_score:.6f}")
            
            # Check for zeros
            zero_count = sum(1 for s in processed_similarities if s == 0.0)
            if zero_count > 0:
                print(f"     ❌ Found {zero_count} zero similarities!")
                return True
            else:
                print(f"     ✅ All similarities > 0 (range: {min(processed_similarities):.3f}-{max(processed_similarities):.3f})")
        
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return True

def test_edge_cases():
    """Test edge cases that might cause 0.000 similarities."""
    print(f"\n🔍 EDGE CASE TEST")
    print("=" * 50)
    
    try:
        from embedder import embed_text
        import faiss
        from journal_db_builder import load_journal_database
        
        journals, embeddings = load_journal_database()
        index = faiss.read_index('data/journal_embeddings.faiss')
        
        edge_cases = [
            "",  # Empty string
            "a",  # Single character
            "the the the the the",  # Repetitive words
            "1234567890",  # Only numbers
            "!@#$%^&*()",  # Only special characters
            "machine learning " * 100,  # Very long repetitive text
        ]
        
        for case in edge_cases:
            if not case.strip():
                print(f"\nTesting empty/whitespace...")
            else:
                print(f"\nTesting: '{case[:30]}{'...' if len(case) > 30 else ''}'")
            
            try:
                if not case.strip():
                    print("     ⚠️  Skipping empty string (would cause error)")
                    continue
                    
                query_embedding = embed_text(case)
                similarities, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 3)
                
                # Process similarities
                processed = []
                for sim in similarities[0]:
                    if isinstance(index, faiss.IndexFlatIP):
                        processed.append(float(sim))
                    else:
                        processed.append(max(0.0, 1.0 - (float(sim) / 2.0)))
                
                print(f"     Similarities: {[f'{s:.6f}' for s in processed]}")
                
                if any(s == 0.0 for s in processed):
                    print(f"     ❌ Found zero similarities with edge case!")
                    return True
                
            except Exception as e:
                print(f"     ❌ Error with edge case: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return True

def main():
    """Run direct similarity tests."""
    issues = []
    
    if test_direct_similarity():
        issues.append("Direct similarity calculation showing zeros")
    
    if test_edge_cases():
        issues.append("Edge cases causing zero similarities")
    
    print(f"\n" + "=" * 50)
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ No similarity calculation issues found")
        print("   The 0.000 issue might be elsewhere:")
        print("   - In the matcher's filtering logic")
        print("   - In min_similarity threshold application")
        print("   - In the Streamlit session state")

if __name__ == "__main__":
    main()