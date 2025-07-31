#!/usr/bin/env python3
"""
Create a very small demo with just 50 journals for Streamlit Cloud.
"""

import json
import sys
sys.path.append('src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def create_mini_demo():
    """Create a mini demo dataset."""
    print("🔍 Loading full dataset...")
    
    # Load full dataset
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    
    full_journals = data.get('journals', data)
    if isinstance(full_journals, dict) and 'journals' in full_journals:
        full_journals = full_journals['journals']
    
    print(f"📊 Full dataset: {len(full_journals)} journals")
    
    # Select top 50 journals by SJR score
    scored_journals = []
    for journal in full_journals:
        sjr = journal.get('sjr_score', 0)
        if sjr > 0:
            scored_journals.append((sjr, journal))
    
    # Sort by SJR score descending
    scored_journals.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 50
    mini_journals = [journal for _, journal in scored_journals[:50]]
    
    print(f"✅ Selected top {len(mini_journals)} journals")
    
    # Save mini dataset
    with open('data/mini_journal_metadata.json', 'w') as f:
        json.dump(mini_journals, f, indent=2)
    
    # Check file size
    size_mb = os.path.getsize('data/mini_journal_metadata.json') / 1024 / 1024
    print(f"📏 Mini metadata size: {size_mb:.1f} MB")
    
    # Now create embeddings
    print("🤖 Creating embeddings...")
    
    from embedder import embed_text, get_model
    import numpy as np
    import faiss
    from pathlib import Path
    
    model = get_model()
    embeddings = []
    valid_journals = []
    
    for i, journal in enumerate(mini_journals):
        print(f"  📈 {i+1}/{len(mini_journals)}: {journal.get('display_name', 'Unknown')[:40]}")
        
        # Create embedding text
        title = journal.get('display_name', '')
        subjects = []
        for subj in journal.get('subjects', []):
            if isinstance(subj, dict):
                subjects.append(subj.get('name', ''))
        subjects_text = ' '.join(subjects)
        areas = ' '.join(journal.get('areas', []))
        
        text_to_embed = f"{title} {subjects_text} {areas}".strip()
        
        if text_to_embed:
            try:
                embedding = embed_text(text_to_embed, model)
                embeddings.append(embedding)
                valid_journals.append(journal)
            except Exception as e:
                print(f"    ⚠️  Failed: {e}")
    
    print(f"✅ Created {len(embeddings)} embeddings")
    
    # Create FAISS index
    if embeddings:
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        
        # Save
        faiss.write_index(index, 'data/mini_journal_embeddings.faiss')
        
        with open('data/mini_journal_metadata.json', 'w') as f:
            json.dump(valid_journals, f, indent=2)
        
        # Check sizes
        faiss_size = os.path.getsize('data/mini_journal_embeddings.faiss') / 1024 / 1024
        meta_size = os.path.getsize('data/mini_journal_metadata.json') / 1024 / 1024
        
        print(f"💾 FAISS size: {faiss_size:.1f} MB")
        print(f"💾 Metadata size: {meta_size:.1f} MB")
        print(f"💾 Total size: {faiss_size + meta_size:.1f} MB")
        
        # Quick test
        test_embedding = embed_text("cancer research oncology", model)
        test_embedding = test_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(test_embedding)
        
        scores, indices = index.search(test_embedding, 3)
        print(f"\n🧪 Test search results:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            name = valid_journals[idx].get('display_name', 'Unknown')
            print(f"  {i+1}. {name} (similarity: {score:.3f})")

if __name__ == "__main__":
    try:
        create_mini_demo()
        print(f"\n🎉 Mini demo created!")
        print(f"📁 Files: mini_journal_metadata.json, mini_journal_embeddings.faiss")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()