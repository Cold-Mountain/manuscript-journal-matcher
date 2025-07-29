#!/usr/bin/env python3
"""
Simple script to test embedding functionality.

Usage:
    python scripts/test_embeddings.py

This script helps verify that the embedding system is working correctly
after installing the required dependencies.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """Main testing function."""
    print("Testing Embedding System")
    print("=" * 40)
    
    try:
        # Import modules
        from embedder import (
            initialize_embedding_model, 
            embed_text, 
            cosine_similarity_single,
            get_embedding_info
        )
        
        # Test 1: Check model info
        print("1. Checking embedding model...")
        info = get_embedding_info()
        print(f"   Model: {info.get('model_name', 'Unknown')}")
        print(f"   Dimension: {info.get('dimension', 'Unknown')}")
        print(f"   Device: {info.get('device', 'Unknown')}")
        print(f"   Loaded: {info.get('loaded', False)}")
        
        if not info.get('loaded', False):
            print(f"   Error: {info.get('error', 'Unknown error')}")
            return False
        
        # Test 2: Generate embeddings
        print("\n2. Testing embedding generation...")
        test_texts = [
            "Machine learning algorithms for healthcare applications",
            "Deep neural networks in medical diagnosis",
            "Traditional statistical methods in medicine",
            "Computer vision for medical imaging"
        ]
        
        embeddings = []
        for i, text in enumerate(test_texts):
            embedding = embed_text(text)
            embeddings.append(embedding)
            print(f"   Text {i+1}: Shape {embedding.shape}, "
                  f"Sample: [{embedding[0]:.3f}, {embedding[1]:.3f}, ...]")
        
        # Test 3: Calculate similarities
        print("\n3. Testing similarity calculations...")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = cosine_similarity_single(embeddings[i], embeddings[j])
                print(f"   Text {i+1} vs Text {j+1}: {similarity:.4f}")
        
        # Test 4: Self-similarity (should be 1.0)
        print("\n4. Testing self-similarity...")
        self_sim = cosine_similarity_single(embeddings[0], embeddings[0])
        print(f"   Self-similarity: {self_sim:.6f} (should be 1.000000)")
        
        if abs(self_sim - 1.0) < 1e-6:
            print("   ✅ Self-similarity test passed")
        else:
            print("   ❌ Self-similarity test failed")
            return False
        
        print("\n✅ All embedding tests passed successfully!")
        print("\nThe embedding system is ready for use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease install dependencies:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)