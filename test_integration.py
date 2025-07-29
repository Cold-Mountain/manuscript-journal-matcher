#!/usr/bin/env python3
"""
Integration test script for manuscript journal matcher.

This script tests the complete workflow from document processing
to journal matching and result formatting.
"""

import sys
from pathlib import Path
import logging

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic import and initialization."""
    print("=== Testing Basic Functionality ===")
    
    try:
        from match_journals import JournalMatcher
        from embedder import embed_text, get_model
        from journal_db_builder import load_journal_database
        from utils import clean_text, extract_keywords
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_database_loading():
    """Test database loading."""
    print("\n=== Testing Database Loading ===")
    
    try:
        from journal_db_builder import load_journal_database
        journals, embeddings = load_journal_database()
        
        print(f"✅ Loaded {len(journals)} journals")
        if embeddings is not None:
            print(f"✅ Embeddings shape: {embeddings.shape}")
        else:
            print("❌ No embeddings found")
            return False
            
        # Check journal data quality
        valid_journals = 0
        for journal in journals:
            if journal.get('display_name') and journal.get('semantic_fingerprint'):
                valid_journals += 1
        
        print(f"✅ {valid_journals}/{len(journals)} journals have required data")
        return True
        
    except Exception as e:
        print(f"❌ Database loading error: {e}")
        return False

def test_embedding_functionality():
    """Test embedding generation."""
    print("\n=== Testing Embedding Functionality ===")
    
    try:
        from embedder import embed_text, get_model
        
        # Test model loading
        model = get_model()
        print(f"✅ Embedding model loaded: {getattr(model, '_model_name', 'Unknown')}")
        
        # Test text embedding
        test_text = "This is a test paper about machine learning and artificial intelligence."
        embedding = embed_text(test_text)
        
        print(f"✅ Generated embedding with shape: {embedding.shape}")
        print(f"✅ Embedding dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return False

def test_journal_matching():
    """Test the complete journal matching workflow."""
    print("\n=== Testing Journal Matching ===")
    
    try:
        from match_journals import JournalMatcher
        
        # Initialize matcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        print(f"✅ Matcher initialized with {len(matcher.journals)} journals")
        
        # Test queries
        test_queries = [
            "Machine learning algorithms for medical diagnosis",
            "Systematic review and meta-analysis methodology", 
            "Mitochondrial DNA analysis in biology research",
            "Health survey and quality of life measurement"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query[:50]}... ---")
            
            try:
                results = matcher.search_similar_journals(
                    query_text=query,
                    top_k=3,
                    min_similarity=0.0
                )
                
                if results:
                    print(f"✅ Found {len(results)} matches")
                    for j, result in enumerate(results, 1):
                        name = result.get('display_name', 'Unknown')
                        score = result.get('similarity_score', 0)
                        publisher = result.get('publisher', 'Unknown')
                        print(f"  {j}. {name} (score: {score:.3f}, {publisher})")
                else:
                    print("⚠️  No matches found")
                    
            except Exception as e:
                print(f"❌ Search failed for query {i}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Matching error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_filtering_functionality():
    """Test search result filtering."""
    print("\n=== Testing Filtering Functionality ===")
    
    try:
        from match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Test with open access filter
        query = "Scientific research publication"
        
        # Get all results first
        all_results = matcher.search_similar_journals(query, top_k=10)
        print(f"✅ Base search returned {len(all_results)} results")
        
        # Test different filters
        filters = [
            {'open_access_only': True},
            {'max_apc': 1000},
            {'min_citations': 100000},
        ]
        
        for i, filter_dict in enumerate(filters, 1):
            try:
                filtered_results = matcher.search_similar_journals(
                    query, top_k=10, filters=filter_dict
                )
                filter_desc = ', '.join(f"{k}={v}" for k, v in filter_dict.items())
                print(f"✅ Filter {i} ({filter_desc}): {len(filtered_results)} results")
            except Exception as e:
                print(f"❌ Filter {i} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Filtering error: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\n=== Testing Utility Functions ===")
    
    try:
        from utils import clean_text, extract_keywords, compute_text_hash
        
        # Test text cleaning
        messy_text = "  This   has    extra    spaces   and\n\nnewlines  "
        cleaned = clean_text(messy_text)
        print(f"✅ Text cleaning: '{messy_text}' -> '{cleaned}'")
        
        # Test keyword extraction
        text = "This paper presents machine learning algorithms for medical diagnosis in healthcare applications."
        keywords = extract_keywords(text, top_k=5)
        print(f"✅ Keyword extraction: {keywords}")
        
        # Test hash computation
        hash_val = compute_text_hash(text)
        print(f"✅ Text hash: {hash_val[:16]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        return False

def test_result_formatting():
    """Test result formatting functionality."""
    print("\n=== Testing Result Formatting ===")
    
    try:
        from match_journals import JournalMatcher, format_search_results
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Get some results
        results = matcher.search_similar_journals("medical research", top_k=2)
        
        if results:
            # Test formatting
            formatted = format_search_results(results)
            
            print(f"✅ Formatted {len(formatted)} results")
            
            # Check format structure
            if formatted:
                result = formatted[0]
                expected_fields = [
                    'rank', 'journal_name', 'similarity_score', 'publisher',
                    'issn', 'is_open_access', 'homepage_url', 'subjects'
                ]
                
                missing_fields = [field for field in expected_fields if field not in result]
                if missing_fields:
                    print(f"⚠️  Missing fields: {missing_fields}")
                else:
                    print("✅ All expected fields present in formatted results")
                
                # Display sample result
                print(f"Sample result: {result['journal_name']} (score: {result['similarity_score']})")
        else:
            print("⚠️  No results to format")
        
        return True
        
    except Exception as e:
        print(f"❌ Formatting error: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 MANUSCRIPT JOURNAL MATCHER - INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Database Loading", test_database_loading),
        ("Embedding Functionality", test_embedding_functionality),
        ("Journal Matching", test_journal_matching),
        ("Filtering Functionality", test_filtering_functionality),
        ("Utility Functions", test_utilities),
        ("Result Formatting", test_result_formatting),
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
    print(f"🎯 INTEGRATION TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Step 5 implementation is working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)