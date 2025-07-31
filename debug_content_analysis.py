#!/usr/bin/env python3
"""
Debug the content analysis pipeline to identify why similarity scores are 0.000
"""

import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import numpy as np

def test_journal_semantic_fingerprints():
    """Test the quality of journal semantic fingerprints."""
    print("🔍 TESTING JOURNAL SEMANTIC FINGERPRINTS")
    print("=" * 60)
    
    # Load journal data
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    journals = data['journals']
    
    print(f"Total journals: {len(journals)}")
    
    # Check semantic fingerprints
    fingerprint_lengths = []
    empty_fingerprints = 0
    sample_fingerprints = []
    
    for i, journal in enumerate(journals[:10]):  # Check first 10
        fingerprint = journal.get('semantic_fingerprint', '')
        name = journal.get('display_name', 'Unknown')
        
        if not fingerprint or len(fingerprint.strip()) < 50:
            empty_fingerprints += 1
            print(f"❌ {name}: Empty/minimal fingerprint ({len(fingerprint)} chars)")
        else:
            fingerprint_lengths.append(len(fingerprint))
            if len(sample_fingerprints) < 3:
                sample_fingerprints.append((name, fingerprint[:200] + "..."))
            print(f"✅ {name}: {len(fingerprint)} chars")
    
    print(f"\nFingerprint Analysis:")
    print(f"Empty/minimal fingerprints: {empty_fingerprints}/10")
    if fingerprint_lengths:
        print(f"Average fingerprint length: {sum(fingerprint_lengths)/len(fingerprint_lengths):.0f} chars")
    
    print(f"\nSample fingerprints:")
    for name, fp in sample_fingerprints:
        print(f"  {name}: {fp}")
    
    return empty_fingerprints > 5  # More than half are empty

def test_embedding_generation():
    """Test if embedding generation is working properly."""
    print(f"\n🔍 TESTING EMBEDDING GENERATION")
    print("=" * 60)
    
    try:
        from embedder import embed_text
        
        # Test with different types of content
        test_texts = [
            "machine learning applications in medical diagnosis",
            "pediatric cardiology interventional procedures",
            "artificial intelligence healthcare optimization",
            "physical chemistry molecular dynamics simulations"
        ]
        
        print("Testing embedding generation with various texts:")
        for text in test_texts:
            try:
                embedding = embed_text(text)
                norm = np.linalg.norm(embedding)
                print(f"✅ '{text[:40]}...': norm={norm:.6f}, shape={embedding.shape}")
            except Exception as e:
                print(f"❌ '{text[:40]}...': ERROR - {e}")
                return True
        
        # Test embedding similarity between related texts
        emb1 = embed_text("machine learning in healthcare")
        emb2 = embed_text("artificial intelligence in medical diagnosis")
        emb3 = embed_text("quantum physics theoretical calculations")
        
        sim_related = np.dot(emb1, emb2)
        sim_unrelated = np.dot(emb1, emb3)
        
        print(f"\nSimilarity test:")
        print(f"Related texts (ML healthcare vs AI medical): {sim_related:.3f}")
        print(f"Unrelated texts (ML healthcare vs quantum physics): {sim_unrelated:.3f}")
        
        if sim_related <= sim_unrelated:
            print("❌ Embedding model not distinguishing related vs unrelated content!")
            return True
        else:
            print("✅ Embedding model working correctly")
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return True
    
    return False

def test_title_abstract_extraction():
    """Test title and abstract extraction."""
    print(f"\n🔍 TESTING TITLE AND ABSTRACT EXTRACTION")
    print("=" * 60)
    
    try:
        from extractor import extract_title_and_abstract
        
        # Test with sample manuscript text
        sample_manuscript = """
        Machine Learning Applications in Pediatric Healthcare: A Comprehensive Review
        
        Abstract
        
        Background: Machine learning (ML) has emerged as a powerful tool in healthcare, 
        with particular promise in pediatric applications. This review examines the current 
        state of machine learning applications in pediatric healthcare settings.
        
        Methods: We conducted a systematic review of literature published between 2018-2023, 
        focusing on ML applications in pediatric diagnosis, treatment planning, and outcome prediction.
        
        Results: We identified 127 studies demonstrating successful ML implementations in pediatric 
        healthcare. Key applications included diagnostic imaging analysis (45%), clinical decision 
        support systems (32%), and predictive modeling for patient outcomes (23%).
        
        Conclusions: Machine learning shows significant promise in pediatric healthcare applications, 
        with demonstrated improvements in diagnostic accuracy and clinical decision-making.
        
        Keywords: machine learning, pediatric healthcare, artificial intelligence, clinical decision support
        
        Introduction
        
        The integration of artificial intelligence and machine learning technologies in healthcare 
        has accelerated rapidly in recent years...
        """
        
        title, abstract = extract_title_and_abstract(sample_manuscript)
        
        print(f"Extracted title: '{title}'")
        print(f"Title length: {len(title) if title else 0} characters")
        
        print(f"\nExtracted abstract: '{abstract[:200] if abstract else 'None'}{'...' if abstract and len(abstract) > 200 else ''}'")
        print(f"Abstract length: {len(abstract) if abstract else 0} characters")
        
        if not title or len(title) < 10:
            print("❌ Title extraction failed or too short")
            return True
        
        if not abstract or len(abstract) < 100:
            print("❌ Abstract extraction failed or too short")
            return True
        
        print("✅ Title and abstract extraction working")
        
        # Test with the extracted content
        if title and abstract:
            combined_text = f"{title} {abstract}"
            try:
                from embedder import embed_text
                embedding = embed_text(combined_text)
                print(f"✅ Successfully generated embedding from extracted content: norm={np.linalg.norm(embedding):.6f}")
            except Exception as e:
                print(f"❌ Failed to generate embedding from extracted content: {e}")
                return True
        
    except Exception as e:
        print(f"❌ Title/Abstract extraction failed: {e}")
        return True
    
    return False

def test_streamlit_processing_pipeline():
    """Test how Streamlit processes manuscript text."""
    print(f"\n🔍 TESTING STREAMLIT PROCESSING PIPELINE")
    print("=" * 60)
    
    # Simulate what happens when user pastes text into Streamlit
    user_input = """
    Title: Machine Learning for Predictive Analytics in Pediatric Emergency Medicine
    
    Abstract: Emergency departments face increasing pressure to provide rapid, accurate diagnoses 
    for pediatric patients. This study evaluates machine learning algorithms for predicting 
    clinical outcomes in pediatric emergency cases. We analyzed 5,000 pediatric emergency visits...
    """
    
    print(f"Simulating user input ({len(user_input)} characters):")
    print(f"'{user_input[:150]}...'")
    
    # Test the processing steps
    try:
        # Step 1: Title/Abstract extraction
        from extractor import extract_title_and_abstract
        title, abstract = extract_title_and_abstract(user_input)
        print(f"\n1. Extraction Results:")
        print(f"   Title: '{title}'")
        print(f"   Abstract: '{abstract[:100] if abstract else 'None'}...'")
        
        # Step 2: Embedding generation
        if title or abstract:
            content_for_search = f"{title or ''} {abstract or ''}".strip()
            if not content_for_search:
                content_for_search = user_input
        else:
            content_for_search = user_input
        
        print(f"\n2. Content for search ({len(content_for_search)} chars):")
        print(f"   '{content_for_search[:100]}...'")
        
        from embedder import embed_text
        query_embedding = embed_text(content_for_search)
        print(f"\n3. Query embedding: norm={np.linalg.norm(query_embedding):.6f}")
        
        # Step 3: Search simulation
        import faiss
        index = faiss.read_index('data/journal_embeddings.faiss')
        
        similarities, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 5)
        print(f"\n4. Search results:")
        print(f"   Similarities: {similarities[0]}")
        print(f"   Range: {similarities[0].min():.3f} to {similarities[0].max():.3f}")
        
        if all(s < 0.01 for s in similarities[0]):
            print("❌ All similarities near 0 - no semantic match found!")
            return True
        else:
            print("✅ Found semantic matches")
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return True
    
    return False

def main():
    """Run comprehensive content analysis debugging."""
    print("🔧 COMPREHENSIVE CONTENT ANALYSIS DEBUG")
    print("=" * 70)
    
    issues_found = []
    
    # Test 1: Journal fingerprints
    if test_journal_semantic_fingerprints():
        issues_found.append("Poor journal semantic fingerprints")
    
    # Test 2: Embedding generation
    if test_embedding_generation():
        issues_found.append("Embedding generation problems")
    
    # Test 3: Title/Abstract extraction
    if test_title_abstract_extraction():
        issues_found.append("Title/Abstract extraction problems")
    
    # Test 4: Full pipeline
    if test_streamlit_processing_pipeline():
        issues_found.append("Streamlit processing pipeline problems")
    
    print(f"\n" + "=" * 70)
    print("🎯 DIAGNOSIS SUMMARY:")
    
    if issues_found:
        print("❌ Issues identified:")
        for issue in issues_found:
            print(f"   - {issue}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if "Poor journal semantic fingerprints" in issues_found:
            print("   1. Rebuild journal database with richer semantic fingerprints")
        if "Title/Abstract extraction problems" in issues_found:
            print("   2. Improve title/abstract extraction patterns")
        if "Embedding generation problems" in issues_found:
            print("   3. Check embedding model and tokenizer settings")
        if "Streamlit processing pipeline problems" in issues_found:
            print("   4. Debug query processing and normalization")
            
    else:
        print("✅ All content analysis components working correctly")
        print("   The 0.000 similarity issue may be due to:")
        print("   - Mismatch between query content and journal database")
        print("   - Need for more specific journal categorization")
        print("   - Requirement for better semantic matching algorithms")

if __name__ == "__main__":
    main()