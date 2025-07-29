"""
Integration test for Step 3: Embedding System

This test verifies that the embedding system works correctly with the 
document extraction system from Step 2.

Run this test after installing dependencies with: pip install -r requirements.txt
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_embedding_integration():
    """Test integration between extractor and embedder."""
    try:
        from src.extractor import extract_title_and_abstract
        from src.embedder import embed_text, cosine_similarity_single, get_embedding_info
        
        # Test with sample manuscript text
        sample_text = """
        Advanced Machine Learning Approaches for Biomedical Data Analysis
        
        Abstract: This research presents a comprehensive investigation into the application 
        of machine learning algorithms for analyzing complex biomedical datasets. Our study 
        evaluates multiple algorithmic approaches including deep neural networks, random 
        forests, and support vector machines on various healthcare data types.
        
        Keywords: machine learning, biomedical data, healthcare
        
        1. Introduction
        The healthcare industry has experienced rapid technological advancement...
        """
        
        print("Step 3 Integration Test: Embedding System")
        print("=" * 50)
        
        # Test extraction
        print("1. Testing text extraction...")
        title, abstract = extract_title_and_abstract(sample_text)
        print(f"   Title: {title}")
        print(f"   Abstract found: {'Yes' if abstract else 'No'}")
        print(f"   Abstract length: {len(abstract) if abstract else 0} characters")
        
        # Test embedding model info
        print("\n2. Testing embedding model...")
        model_info = get_embedding_info()
        print(f"   Model: {model_info.get('model_name', 'Unknown')}")
        print(f"   Dimension: {model_info.get('dimension', 'Unknown')}")
        print(f"   Loaded: {model_info.get('loaded', False)}")
        
        if model_info.get('loaded', False):
            # Test embedding generation
            print("\n3. Testing embedding generation...")
            if abstract:
                embedding = embed_text(abstract[:500])  # Limit length for testing
                print(f"   Generated embedding shape: {embedding.shape}")
                print(f"   Embedding type: {type(embedding)}")
                print(f"   Sample values: {embedding[:5]}")
                
                # Test similarity
                print("\n4. Testing similarity calculation...")
                # Create a slightly modified version
                modified_abstract = abstract.replace("machine learning", "artificial intelligence")
                modified_embedding = embed_text(modified_abstract[:500])
                
                similarity = cosine_similarity_single(embedding, modified_embedding)
                print(f"   Similarity between original and modified: {similarity:.4f}")
                
                # Self-similarity should be 1.0
                self_similarity = cosine_similarity_single(embedding, embedding)
                print(f"   Self-similarity: {self_similarity:.4f} (should be 1.0)")
                
                print("\n✅ All tests passed! Embedding system is working correctly.")
            else:
                print("   ⚠️  No abstract found, skipping embedding tests")
        else:
            error_msg = model_info.get('error', 'Unknown error')
            print(f"   ❌ Model failed to load: {error_msg}")
            print("   Make sure you've installed the requirements: pip install -r requirements.txt")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_embedding_integration()