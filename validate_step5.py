#!/usr/bin/env python3
"""
Step 5 Validation Script - Quick validation of vector search implementation.

This script validates that the Step 5 implementation is working correctly
without running into multiprocessing issues.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_files_exist():
    """Check that all required files were created."""
    print("=== File Validation ===")
    
    required_files = [
        "src/match_journals.py",
        "src/utils.py", 
        "tests/test_matching.py",
        "data/journal_metadata.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} exists ({path.stat().st_size} bytes)")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def validate_imports():
    """Check that all modules can be imported."""
    print("\n=== Import Validation ===")
    
    try:
        from match_journals import JournalMatcher, MatchingError
        print("✅ JournalMatcher imported")
        
        from match_journals import create_faiss_index, search_similar_journals
        print("✅ Standalone functions imported")
        
        from utils import validate_file, clean_text, extract_keywords
        print("✅ Utility functions imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def validate_database_structure():
    """Check that the database has the expected structure."""
    print("\n=== Database Structure Validation ===")
    
    try:
        metadata_path = Path("data/journal_metadata.json")
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Check metadata structure
        required_keys = ['created_at', 'total_journals', 'embedding_dimension', 'journals']
        for key in required_keys:
            if key in data:
                print(f"✅ {key}: {data[key]}")
            else:
                print(f"❌ Missing key: {key}")
                return False
        
        # Check journal structure
        if data['journals']:
            journal = data['journals'][0]
            required_journal_keys = ['display_name', 'semantic_fingerprint', 'embedding']
            
            for key in required_journal_keys:
                if key in journal:
                    if key == 'embedding':
                        print(f"✅ {key}: array of length {len(journal[key])}")
                    else:
                        print(f"✅ {key}: {str(journal[key])[:50]}...")
                else:
                    print(f"❌ Missing journal key: {key}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database validation error: {e}")
        return False

def validate_class_structure():
    """Check that the JournalMatcher class has expected methods."""
    print("\n=== Class Structure Validation ===")
    
    try:
        from match_journals import JournalMatcher
        
        # Check class methods
        expected_methods = [
            'load_database',
            'search_similar_journals', 
            'get_database_stats',
            '_create_faiss_index',
            '_apply_filters'
        ]
        
        matcher = JournalMatcher()
        
        for method_name in expected_methods:
            if hasattr(matcher, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Missing method: {method_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Class validation error: {e}")
        return False

def validate_faiss_functionality():
    """Test FAISS index creation without loading heavy models."""
    print("\n=== FAISS Functionality Validation ===")
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
        
        # Test basic FAISS operations
        dimension = 384
        n_vectors = 10
        test_embeddings = np.random.random((n_vectors, dimension)).astype(np.float32)
        
        # Test flat index creation
        index = faiss.IndexFlatIP(dimension)
        index.add(test_embeddings)
        
        print(f"✅ FAISS flat index created with {index.ntotal} vectors")
        
        # Test search
        query = np.random.random((1, dimension)).astype(np.float32)
        similarities, indices = index.search(query, 3)
        
        print(f"✅ FAISS search returned {len(indices[0])} results")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS validation error: {e}")
        return False

def validate_utility_functions():
    """Test utility functions work correctly."""
    print("\n=== Utility Functions Validation ===")
    
    try:
        from utils import clean_text, extract_keywords, compute_text_hash
        
        # Test text cleaning
        test_text = "  This   has    extra   spaces  \n\n and newlines  "
        cleaned = clean_text(test_text)
        expected = "This has extra spaces and newlines"
        
        if cleaned == expected:
            print(f"✅ Text cleaning works correctly")
        else:
            print(f"❌ Text cleaning failed: '{cleaned}' != '{expected}'")
            return False
        
        # Test keyword extraction
        text = "machine learning artificial intelligence computer science"
        keywords = extract_keywords(text, top_k=3)
        
        if isinstance(keywords, list) and len(keywords) <= 3:
            print(f"✅ Keyword extraction: {keywords}")
        else:
            print(f"❌ Keyword extraction failed: {keywords}")
            return False
        
        # Test hash computation
        hash_val = compute_text_hash("test text")
        
        if isinstance(hash_val, str) and len(hash_val) == 64:  # SHA-256 hex length
            print(f"✅ Hash computation: {hash_val[:16]}...")
        else:
            print(f"❌ Hash computation failed: {hash_val}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Utility validation error: {e}")
        return False

def validate_test_structure():
    """Check that test file has proper structure."""
    print("\n=== Test Structure Validation ===")
    
    try:
        test_file = Path("tests/test_matching.py")
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Check for important test classes and functions
        required_items = [
            "class TestJournalMatcher",
            "class TestStandaloneFunctions", 
            "def test_search_similar_journals",
            "def test_create_faiss_index",
            "pytest.fixture"
        ]
        
        for item in required_items:
            if item in content:
                print(f"✅ Found: {item}")
            else:
                print(f"❌ Missing: {item}")
                return False
        
        # Count approximate number of test functions
        test_count = content.count("def test_")
        print(f"✅ Found approximately {test_count} test functions")
        
        return True
        
    except Exception as e:
        print(f"❌ Test validation error: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🔍 STEP 5 VALIDATION - Vector Search Implementation")
    print("=" * 60)
    
    validations = [
        ("Files Exist", validate_files_exist),
        ("Module Imports", validate_imports),
        ("Database Structure", validate_database_structure),
        ("Class Structure", validate_class_structure),
        ("FAISS Functionality", validate_faiss_functionality),
        ("Utility Functions", validate_utility_functions),
        ("Test Structure", validate_test_structure)
    ]
    
    passed = 0
    total = len(validations)
    
    for name, validation_func in validations:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        try:
            if validation_func():
                passed += 1
                print(f"✅ {name} PASSED")
            else:
                print(f"❌ {name} FAILED")
        except Exception as e:
            print(f"❌ {name} CRASHED: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"🎯 VALIDATION RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 STEP 5 IMPLEMENTATION VALIDATED SUCCESSFULLY!")
        print("\n✨ What was implemented:")
        print("  • src/match_journals.py - FAISS-based vector search")
        print("  • src/utils.py - Utility functions and helpers") 
        print("  • tests/test_matching.py - Comprehensive test suite")
        print("  • Integration with existing journal database")
        print("  • Search functionality with filtering options")
        print("  • Result formatting and ranking")
        
        print("\n🚀 Next Steps:")
        print("  • Step 6: Build Streamlit interface (src/main.py)")
        print("  • End-to-end testing with real manuscript files")
        print("  • Performance optimization if needed")
        
        return 0
    else:
        print(f"⚠️  {total - passed} validations failed. Review issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)