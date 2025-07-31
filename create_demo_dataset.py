#!/usr/bin/env python3
"""
Create a small demo dataset for Streamlit Cloud deployment.
Extract ~500 high-quality journals from the full dataset.
"""

import json
import sys
sys.path.append('src')

from pathlib import Path
import numpy as np
import faiss
from embedder import embed_text, get_model

def create_demo_dataset():
    """Create a smaller dataset suitable for Streamlit Cloud."""
    
    print("🔍 Loading full dataset...")
    
    # Load full dataset
    with open('data/journal_metadata.json', 'r') as f:
        data = json.load(f)
    
    full_journals = data.get('journals', data)  # Handle both formats
    if isinstance(full_journals, dict) and 'journals' in full_journals:
        full_journals = full_journals['journals']
    
    print(f"📊 Full dataset: {len(full_journals)} journals")
    
    # Filter criteria for demo dataset
    demo_journals = []
    
    # Priority 1: High impact journals (SJR score > 2)
    high_impact = [j for j in full_journals if j.get('sjr_score', 0) > 2.0]
    demo_journals.extend(high_impact[:100])  # Top 100 high impact
    
    # Priority 2: Open access journals 
    open_access = [j for j in full_journals if j.get('is_oa', False) and j not in demo_journals]
    demo_journals.extend(open_access[:150])  # Top 150 open access
    
    # Priority 3: Popular medical specialties
    medical_keywords = ['medicine', 'medical', 'surgery', 'clinical', 'health', 'disease', 'patient', 'treatment', 'oncology', 'cardiology']
    medical_journals = []
    for journal in full_journals:
        if journal in demo_journals:
            continue
        title = journal.get('display_name', '').lower()
        
        # Extract subject names from subject objects
        subjects = []
        for subj in journal.get('subjects', []):
            if isinstance(subj, dict):
                subjects.append(subj.get('name', ''))
            else:
                subjects.append(str(subj))
        subjects_text = ' '.join(subjects).lower()
        
        # Check areas too
        areas = ' '.join(journal.get('areas', [])).lower()
        
        if any(keyword in title or keyword in subjects_text or keyword in areas for keyword in medical_keywords):
            medical_journals.append(journal)
    
    demo_journals.extend(medical_journals[:200])  # Top 200 medical
    
    # Priority 4: Fill remaining with diverse journals
    remaining = [j for j in full_journals if j not in demo_journals]
    demo_journals.extend(remaining[:50])  # Top 50 others
    
    # Remove duplicates and limit size
    seen_ids = set()
    final_demo = []
    for journal in demo_journals:
        journal_id = journal.get('id') or journal.get('display_name')
        if journal_id not in seen_ids:
            seen_ids.add(journal_id)
            final_demo.append(journal)
            
        if len(final_demo) >= 500:  # Limit to 500 journals
            break
    
    print(f"✅ Created demo dataset: {len(final_demo)} journals")
    
    # Save demo dataset
    demo_path = Path('data/demo_journal_metadata.json')
    with open(demo_path, 'w') as f:
        json.dump(final_demo, f, indent=2)
    
    print(f"💾 Saved to: {demo_path}")
    print(f"📏 File size: {demo_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return final_demo

def create_demo_faiss_index(demo_journals):
    """Create FAISS index for demo journals."""
    print("🔍 Creating FAISS index for demo dataset...")
    
    # Initialize embedding model
    model = get_model()
    
    # Create embeddings for demo journals
    embeddings = []
    valid_journals = []
    
    for i, journal in enumerate(demo_journals):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(demo_journals)} journals...")
            
        # Create text to embed (same as full system)
        title = journal.get('display_name', '')
        
        # Extract subject names from subject objects
        subjects = []
        for subj in journal.get('subjects', []):
            if isinstance(subj, dict):
                subjects.append(subj.get('name', ''))
            else:
                subjects.append(str(subj))
        subjects_text = ' '.join(subjects)
        
        # Include areas
        areas = ' '.join(journal.get('areas', []))
        
        description = journal.get('description', '')
        
        text_to_embed = f"{title} {subjects_text} {areas} {description}".strip()
        
        if text_to_embed:
            try:
                embedding = embed_text(text_to_embed, model)
                embeddings.append(embedding)
                valid_journals.append(journal)
            except Exception as e:
                print(f"⚠️  Skipping journal {i}: {e}")
                continue
    
    print(f"✅ Created {len(embeddings)} embeddings")
    
    # Create FAISS index
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Normalize embeddings for IndexFlatIP (cosine similarity)
    faiss.normalize_L2(embeddings_array)
    
    # Create index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    # Save demo FAISS index
    demo_faiss_path = Path('data/demo_journal_embeddings.faiss')
    faiss.write_index(index, str(demo_faiss_path))
    
    # Update demo journal metadata to match embeddings
    demo_metadata_path = Path('data/demo_journal_metadata.json') 
    with open(demo_metadata_path, 'w') as f:
        json.dump(valid_journals, f, indent=2)
    
    print(f"💾 Saved FAISS index: {demo_faiss_path}")
    print(f"📏 FAISS file size: {demo_faiss_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"💾 Updated metadata: {demo_metadata_path}")
    print(f"📏 Metadata file size: {demo_metadata_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return len(valid_journals)

if __name__ == "__main__":
    print("🚀 Creating demo dataset for Streamlit Cloud...")
    
    try:
        # Create demo journal dataset
        demo_journals = create_demo_dataset()
        
        # Create demo FAISS index
        num_journals = create_demo_faiss_index(demo_journals) 
        
        print(f"\n🎉 Demo dataset created successfully!")
        print(f"📊 {num_journals} journals with embeddings")
        print(f"📁 Files created:")
        print(f"  - data/demo_journal_metadata.json")
        print(f"  - data/demo_journal_embeddings.faiss")
        print(f"\n💡 These files should be small enough for GitHub/Streamlit Cloud")
        
    except Exception as e:
        print(f"❌ Error creating demo dataset: {e}")
        import traceback
        traceback.print_exc()