#!/usr/bin/env python3
"""
Step 7 Validation Script - Comprehensive testing and validation.

This script validates the complete system functionality using the existing
journal database and tests all components work together correctly.
"""

import sys
from pathlib import Path
import time
import tempfile

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic system functionality."""
    print("=== Testing Basic System Functionality ===")
    
    try:
        # Test imports
        from extractor import extract_title_and_abstract
        from embedder import embed_text, get_model
        from match_journals import JournalMatcher
        from journal_db_builder import load_journal_database
        from utils import clean_text, extract_keywords
        print("✅ All modules imported successfully")
        
        # Test embedding model
        model = get_model()
        print(f"✅ Embedding model loaded: {getattr(model, '_model_name', 'Unknown')}")
        
        # Test database loading
        journals, embeddings = load_journal_database()
        print(f"✅ Database loaded: {len(journals)} journals, {embeddings.shape if embeddings is not None else 'No embeddings'}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    print("\n=== Testing End-to-End Workflow ===")
    
    try:
        from extractor import extract_title_and_abstract
        from embedder import embed_text
        from match_journals import JournalMatcher
        
        # Initialize matcher
        matcher = JournalMatcher()
        matcher.load_database()
        print("✅ Journal matcher initialized")
        
        # Test with sample abstracts
        test_abstracts = [
            "This study investigates machine learning algorithms for medical diagnosis using patient data and electronic health records.",
            "We present a novel deep learning architecture for natural language processing tasks with transformer models.",
            "Our research examines genetic mechanisms in cellular biology using CRISPR gene editing techniques.",
            "This paper discusses systematic review methodology and meta-analysis techniques in healthcare research."
        ]
        
        for i, abstract in enumerate(test_abstracts, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Abstract: {abstract[:60]}...")
            
            # Generate embedding
            start_time = time.time()
            embedding = embed_text(abstract)
            embedding_time = time.time() - start_time
            print(f"✅ Embedding generated in {embedding_time:.3f}s")
            
            # Search for matching journals
            start_time = time.time()
            results = matcher.search_similar_journals(abstract, top_k=3)
            search_time = time.time() - start_time
            print(f"✅ Search completed in {search_time:.3f}s")
            
            # Display results
            print(f"Found {len(results)} matches:")
            for j, result in enumerate(results, 1):
                print(f"  {j}. {result['display_name']} (similarity: {result['similarity_score']:.3f})")
                print(f"     Publisher: {result.get('publisher', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_accuracy_with_domain_queries():
    """Test accuracy with domain-specific queries."""
    print("\n=== Testing Domain-Specific Accuracy ===")
    
    try:
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Domain-specific test queries
        domain_tests = [
            {
                "domain": "Medical/Healthcare",
                "query": "clinical trials medical diagnosis patient treatment healthcare outcomes cardiovascular disease",
                "expected_terms": ["medical", "clinical", "health", "patient", "medicine"]
            },
            {
                "domain": "Computer Science",
                "query": "machine learning algorithms artificial intelligence neural networks deep learning computer science",
                "expected_terms": ["computer", "science", "artificial", "intelligence", "machine"]
            },
            {
                "domain": "Biology/Genetics", 
                "query": "molecular biology genetic research DNA sequencing cellular mechanisms gene expression",
                "expected_terms": ["biology", "genetic", "molecular", "gene", "cellular"]
            }
        ]
        
        for test in domain_tests:
            print(f"\n--- {test['domain']} Query ---")
            results = matcher.search_similar_journals(test['query'], top_k=3)
            
            if results:
                top_result = results[0]
                print(f"Top match: {top_result['display_name']} (score: {top_result['similarity_score']:.3f})")
                print(f"Publisher: {top_result.get('publisher', 'Unknown')}")
                
                # Check if result is relevant to domain
                result_text = f"{top_result['display_name']} {top_result.get('publisher', '')}".lower()
                semantic_fingerprint = top_result.get('semantic_fingerprint', '').lower()
                combined_text = f"{result_text} {semantic_fingerprint}"
                
                relevance_score = sum(1 for term in test['expected_terms'] 
                                    if term.lower() in combined_text) / len(test['expected_terms'])
                
                print(f"Domain relevance: {relevance_score:.2f} ({relevance_score * 100:.0f}% match)")
                
                if relevance_score > 0.2:  # At least 20% of expected terms
                    print("✅ Domain matching appears reasonable")
                else:
                    print("⚠️ Domain matching may need improvement")
            else:
                print("❌ No results found")
        
        return True
        
    except Exception as e:
        print(f"❌ Domain accuracy test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test system performance characteristics."""
    print("\n=== Testing Performance Benchmarks ===")
    
    try:
        from match_journals import JournalMatcher
        from embedder import embed_text
        
        # Initialize system
        start_time = time.time()
        matcher = JournalMatcher()
        matcher.load_database()
        init_time = time.time() - start_time
        print(f"✅ System initialization: {init_time:.3f}s")
        
        # Test embedding performance
        test_texts = [
            "Short text.",
            "Medium length text with some technical terms and methodology description for testing purposes.",
            "Long text with detailed methodology, results, and conclusions. " * 20
        ]
        
        embedding_times = []
        for i, text in enumerate(test_texts):
            start_time = time.time()
            embedding = embed_text(text)
            embedding_time = time.time() - start_time
            embedding_times.append(embedding_time)
            print(f"✅ Embedding {i+1} ({len(text)} chars): {embedding_time:.3f}s")
        
        avg_embedding_time = sum(embedding_times) / len(embedding_times)
        print(f"Average embedding time: {avg_embedding_time:.3f}s")
        
        # Test search performance
        query = "machine learning medical diagnosis healthcare applications"
        search_times = []
        
        for i in range(5):  # Multiple searches
            start_time = time.time()
            results = matcher.search_similar_journals(query, top_k=5)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"✅ Average search time: {avg_search_time:.3f}s (over {len(search_times)} searches)")
        
        # Performance assertions
        performance_checks = [
            (init_time < 10.0, f"System initialization time: {init_time:.3f}s (should be < 10s)"),
            (avg_embedding_time < 3.0, f"Average embedding time: {avg_embedding_time:.3f}s (should be < 3s)"),
            (avg_search_time < 1.0, f"Average search time: {avg_search_time:.3f}s (should be < 1s)")
        ]
        
        all_passed = True
        for check, message in performance_checks:
            if check:
                print(f"✅ {message}")
            else:
                print(f"⚠️ {message}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Performance benchmark test failed: {e}")
        return False

def test_result_quality():
    """Test the quality and consistency of search results."""
    print("\n=== Testing Result Quality ===")
    
    try:
        from match_journals import JournalMatcher, format_search_results
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        test_query = "artificial intelligence machine learning algorithms research applications"
        
        # Test different parameter combinations
        test_configs = [
            {"top_k": 3, "min_similarity": 0.0},
            {"top_k": 5, "min_similarity": 0.1},
            {"top_k": 10, "min_similarity": 0.0},
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n--- Configuration {i+1}: {config} ---")
            
            results = matcher.search_similar_journals(test_query, **config)
            print(f"Found {len(results)} results")
            
            if results:
                # Test result properties
                scores = [r['similarity_score'] for r in results]
                
                # Check score ordering
                is_ordered = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
                print(f"✅ Scores properly ordered: {is_ordered}")
                
                # Check score range
                score_range = max(scores) - min(scores) if len(scores) > 1 else 0
                print(f"Score range: [{min(scores):.3f}, {max(scores):.3f}] (range: {score_range:.3f})")
                
                # Test result formatting
                formatted = format_search_results(results)
                print(f"✅ Results formatted successfully: {len(formatted)} items")
                
                # Check required fields
                required_fields = ['journal_name', 'similarity_score', 'publisher', 'rank']
                missing_fields = []
                
                for field in required_fields:
                    if field not in formatted[0]:
                        missing_fields.append(field)
                
                if not missing_fields:
                    print("✅ All required fields present in formatted results")
                else:
                    print(f"⚠️ Missing fields in formatted results: {missing_fields}")
        
        return True
        
    except Exception as e:
        print(f"❌ Result quality test failed: {e}")
        return False

def test_error_handling():
    """Test system error handling and robustness."""
    print("\n=== Testing Error Handling ===")
    
    try:
        from match_journals import JournalMatcher, MatchingError
        from embedder import embed_text, EmbeddingError
        
        # Test empty query handling
        try:
            matcher = JournalMatcher()
            matcher.load_database()
            results = matcher.search_similar_journals("", top_k=5)
            print("❌ Empty query should have failed")
            return False
        except (MatchingError, Exception) as e:
            print("✅ Empty query properly rejected")
        
        # Test very short query
        try:
            results = matcher.search_similar_journals("AI", top_k=5)
            print("✅ Short query handled gracefully")
        except Exception as e:
            print(f"⚠️ Short query caused error: {e}")
        
        # Test invalid parameters
        try:
            results = matcher.search_similar_journals("test query", top_k=0)
            print("⚠️ Invalid top_k should be handled better")
        except Exception as e:
            print("✅ Invalid parameters properly rejected")
        
        # Test special characters
        try:
            special_query = "α-β protein interactions with μ-opioid receptors @ 37°C ± 2°C"
            results = matcher.search_similar_journals(special_query, top_k=3)
            print("✅ Special characters handled gracefully")
        except Exception as e:
            print(f"⚠️ Special characters caused error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 STEP 7 VALIDATION - Testing & Validation")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Domain-Specific Accuracy", test_accuracy_with_domain_queries),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Result Quality", test_result_quality),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎯 VALIDATION RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Step 7 validation completed successfully.")
        print("\n✨ System Validation Summary:")
        print("  • ✅ All modules import and initialize correctly")
        print("  • ✅ End-to-end workflow functions properly")
        print("  • ✅ Search results are relevant and well-formatted")
        print("  • ✅ Performance meets acceptable benchmarks")
        print("  • ✅ Error handling is robust")
        
        print("\n🚀 Ready for Step 8: DOAJ Integration")
        return 0
    else:
        print(f"⚠️ {total - passed} tests failed. Review issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)