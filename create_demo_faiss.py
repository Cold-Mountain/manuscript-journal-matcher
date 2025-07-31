#!/usr/bin/env python3
"""
Create FAISS index for the demo dataset.
"""

import json
import sys
sys.path.append('src')

from pathlib import Path
import numpy as np
import faiss
import os

# Set environment variables for stability
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def create_demo_faiss_simple():
    """Create FAISS index for demo journals using existing embedding approach."""
    print("🔍 Creating FAISS index for demo dataset...")
    
    # Load demo journals
    with open('data/demo_journal_metadata.json', 'r') as f:
        demo_journals = json.load(f)
    
    print(f"📊 Processing {len(demo_journals)} journals...")
    
    # Import after setting environment variables
    from embedder import embed_text, get_model
    
    # Initialize model once
    print("🤖 Loading embedding model...")
    model = get_model()
    
    # Create embeddings
    embeddings = []
    valid_journals = []
    
    for i, journal in enumerate(demo_journals):
        if i % 25 == 0:
            print(f"  📈 Progress: {i}/{len(demo_journals)} ({i/len(demo_journals)*100:.1f}%)")
            
        # Create text to embed
        title = journal.get('display_name', '')
        
        # Extract subject names
        subjects = []
        for subj in journal.get('subjects', []):
            if isinstance(subj, dict):
                subjects.append(subj.get('name', ''))
            else:
                subjects.append(str(subj))
        subjects_text = ' '.join(subjects)
        
        # Include areas
        areas = ' '.join(journal.get('areas', []))
        
        text_to_embed = f"{title} {subjects_text} {areas}".strip()
        
        if text_to_embed:
            try:
                embedding = embed_text(text_to_embed, model)
                embeddings.append(embedding)
                valid_journals.append(journal)
            except Exception as e:
                print(f"⚠️  Skipping journal {i} ({title[:30]}): {e}")
                continue
    
    print(f"✅ Created embeddings for {len(embeddings)} journals")
    
    if len(embeddings) == 0:
        print("❌ No embeddings created!")
        return
    
    # Create FAISS index
    print("🔍 Building FAISS index...")
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    # Create index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    # Save FAISS index
    demo_faiss_path = Path('data/demo_journal_embeddings.faiss')
    faiss.write_index(index, str(demo_faiss_path))
    
    # Update metadata to only include journals with embeddings
    demo_metadata_path = Path('data/demo_journal_metadata.json')
    with open(demo_metadata_path, 'w') as f:
        json.dump(valid_journals, f, indent=2)
    
    print(f"💾 Saved FAISS index: {demo_faiss_path}")
    print(f"📏 FAISS size: {demo_faiss_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"💾 Updated metadata: {demo_metadata_path}")  
    print(f"📏 Metadata size: {demo_metadata_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Verify index works
    print("🧪 Testing FAISS index...")
    test_query = "cancer treatment oncology"
    test_embedding = embed_text(test_query, model)
    test_embedding = test_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(test_embedding)
    
    scores, indices = index.search(test_embedding, 3)
    print(f"✅ Test search for '{test_query}':")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        journal_name = valid_journals[idx].get('display_name', 'Unknown')
        print(f"  {i+1}. {journal_name[:40]} (similarity: {score:.3f})")

if __name__ == "__main__":
    try:
        create_demo_faiss_simple()
        print(f"\n🎉 Demo FAISS index created successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()