# Interactive API Examples

This document provides comprehensive, runnable examples for all major functions in the Manuscript Journal Matcher API. Each example includes setup code, expected outputs, and practical use cases.

## 🚀 Getting Started

### Basic Setup

```python
# Import core modules
from src.extractor import extract_manuscript_data
from src.embedder import embed_text, embed_texts, cosine_similarity_single
from src.match_journals import JournalMatcher
from src.config import *
from src.utils import validate_file, clean_text

# Create sample data directory for examples
import tempfile
from pathlib import Path

# Set up temporary working directory
EXAMPLE_DIR = Path(tempfile.mkdtemp()) / "journal_matcher_examples"
EXAMPLE_DIR.mkdir(exist_ok=True)
print(f"📁 Examples directory: {EXAMPLE_DIR}")
```

## 📄 Document Extraction Examples

### Example 1: Basic PDF Extraction

```python
# Create a sample PDF text for demonstration
sample_pdf_text = """
Deep Learning for Medical Image Analysis: A Comprehensive Survey

Abstract: Medical image analysis has been revolutionized by deep learning 
techniques in recent years. This survey reviews the latest advances in 
convolutional neural networks, generative adversarial networks, and 
transformer architectures for medical imaging applications including 
radiology, pathology, and ophthalmology. We analyze over 150 research 
papers published between 2020-2024 and identify key trends and future 
research directions.

Keywords: deep learning, medical imaging, CNN, GAN, transformers

1. Introduction
Medical imaging plays a crucial role in modern healthcare diagnosis...
"""

# Simulate extraction from this text
def demo_pdf_extraction():
    from src.extractor import extract_title_and_abstract
    
    print("📄 Document Extraction Demo")
    print("=" * 50)
    
    # Extract title and abstract
    title, abstract = extract_title_and_abstract(sample_pdf_text)
    
    print(f"📋 Title: {title}")
    print(f"\n📝 Abstract ({len(abstract)} chars):")
    print(f"{abstract}")
    
    # Simulate full extraction result
    mock_data = {
        'title': title,
        'abstract': abstract,
        'full_text': sample_pdf_text,
        'file_type': '.pdf',
        'file_name': 'sample_paper.pdf'
    }
    
    # Validate extraction
    from src.extractor import validate_extracted_data
    validation = validate_extracted_data(mock_data)
    
    print(f"\n✅ Validation Status: {validation['status']}")
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"⚠️ {warning}")
    
    return mock_data

# Run the demo
extracted_data = demo_pdf_extraction()
```

**Expected Output:**
```
📄 Document Extraction Demo
==================================================
📋 Title: Deep Learning for Medical Image Analysis: A Comprehensive Survey

📝 Abstract (334 chars):
Medical image analysis has been revolutionized by deep learning 
techniques in recent years. This survey reviews the latest advances in 
convolutional neural networks, generative adversarial networks, and 
transformer architectures for medical imaging applications including 
radiology, pathology, and ophthalmology...

✅ Validation Status: valid
```

### Example 2: Batch File Processing

```python
def demo_batch_extraction():
    """Demonstrate batch processing of multiple documents."""
    
    # Sample abstracts for different medical domains
    sample_documents = {
        "cardiology_paper.pdf": {
            "text": """
            Artificial Intelligence in Cardiovascular Disease Diagnosis
            
            Abstract: This study presents a novel AI-powered approach for 
            early detection of cardiovascular diseases using ECG analysis 
            and machine learning algorithms. We achieved 94.2% accuracy 
            on a dataset of 10,000 patient records.
            """,
            "domain": "Cardiology"
        },
        "radiology_paper.pdf": {
            "text": """
            Automated Chest X-Ray Analysis Using Deep Learning
            
            Abstract: We develop a convolutional neural network for 
            automated detection of pneumonia and other respiratory 
            conditions in chest X-rays. The model demonstrates superior 
            performance compared to traditional methods.
            """,
            "domain": "Radiology"
        },
        "oncology_paper.pdf": {
            "text": """
            Machine Learning for Cancer Prognosis Prediction
            
            Abstract: This research investigates the use of ensemble 
            machine learning methods for predicting cancer patient 
            outcomes based on genomic and clinical data.
            """,
            "domain": "Oncology"
        }
    }
    
    print("📚 Batch Document Processing Demo")
    print("=" * 50)
    
    batch_results = []
    
    for filename, doc_info in sample_documents.items():
        print(f"\n📄 Processing: {filename}")
        
        # Extract from document text
        from src.extractor import extract_title_and_abstract
        title, abstract = extract_title_and_abstract(doc_info["text"])
        
        result = {
            'filename': filename,
            'domain': doc_info['domain'],
            'title': title,
            'abstract': abstract,
            'abstract_length': len(abstract) if abstract else 0,
            'status': 'success' if abstract else 'warning'
        }
        
        batch_results.append(result)
        
        print(f"  📋 Title: {title}")
        print(f"  📝 Abstract: {abstract[:100]}...")
        print(f"  🏷️ Domain: {doc_info['domain']}")
        print(f"  📊 Length: {result['abstract_length']} chars")
    
    # Summary statistics
    successful = len([r for r in batch_results if r['status'] == 'success'])
    avg_length = sum(r['abstract_length'] for r in batch_results) / len(batch_results)
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"  Total documents: {len(batch_results)}")
    print(f"  Successful extractions: {successful}")
    print(f"  Average abstract length: {avg_length:.0f} characters")
    
    return batch_results

# Run batch processing demo
batch_results = demo_batch_extraction()
```

## 🧠 Embedding Examples

### Example 3: Single Text Embedding

```python
def demo_single_embedding():
    """Demonstrate single text embedding generation."""
    
    print("🧠 Single Text Embedding Demo")
    print("=" * 50)
    
    # Sample medical research text
    medical_text = """
    This study investigates the application of transformer neural networks 
    for automated analysis of electronic health records to predict patient 
    readmission risk within 30 days of discharge.
    """
    
    from src.embedder import embed_text, get_embedding_info
    
    # Get model information
    model_info = get_embedding_info()
    print(f"🤖 Model: {model_info['model_name']}")
    print(f"📐 Dimension: {model_info['dimension']}")
    print(f"🖥️ Device: {model_info['device']}")
    
    # Generate embedding
    print(f"\n📝 Input text ({len(medical_text)} chars):")
    print(f"'{medical_text.strip()}'")
    
    embedding = embed_text(medical_text)
    
    print(f"\n🔢 Generated embedding:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Data type: {embedding.dtype}")
    print(f"  L2 norm: {np.linalg.norm(embedding):.4f}")
    print(f"  Min value: {embedding.min():.4f}")
    print(f"  Max value: {embedding.max():.4f}")
    print(f"  First 10 values: {embedding[:10]}")
    
    return embedding

# Run single embedding demo
sample_embedding = demo_single_embedding()
```

### Example 4: Batch Embedding Generation

```python
def demo_batch_embeddings():
    """Demonstrate batch embedding generation with different medical topics."""
    
    print("📚 Batch Embedding Generation Demo")
    print("=" * 50)
    
    # Medical research abstracts from different domains
    medical_abstracts = [
        "Machine learning algorithms for predicting cardiovascular disease risk using ECG data",
        "Deep learning approaches for automated radiology image analysis and diagnosis",
        "Natural language processing techniques for clinical text mining and information extraction",
        "Computer vision methods for pathology slide analysis and cancer detection",
        "Artificial intelligence applications in drug discovery and pharmaceutical research"
    ]
    
    print(f"📝 Processing {len(medical_abstracts)} abstracts...")
    
    from src.embedder import embed_texts
    import time
    
    # Generate embeddings with timing
    start_time = time.time()
    embeddings = embed_texts(
        medical_abstracts,
        batch_size=16,
        show_progress=True
    )
    processing_time = time.time() - start_time
    
    print(f"\n✅ Batch Processing Complete!")
    print(f"  Processing time: {processing_time:.2f} seconds")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Speed: {len(medical_abstracts)/processing_time:.1f} texts/second")
    
    # Analyze embeddings
    print(f"\n📊 Embedding Analysis:")
    for i, (text, embedding) in enumerate(zip(medical_abstracts, embeddings)):
        norm = np.linalg.norm(embedding)
        print(f"  {i+1}. {text[:60]}...")
        print(f"     L2 norm: {norm:.4f}")
    
    return embeddings, medical_abstracts

# Run batch embedding demo
batch_embeddings, batch_texts = demo_batch_embeddings()
```

### Example 5: Similarity Calculations

```python
def demo_similarity_calculations():
    """Demonstrate various similarity calculation methods."""
    
    print("🔍 Similarity Calculation Demo")
    print("=" * 50)
    
    # Sample texts for comparison
    texts = {
        "cardiology": "ECG analysis for heart disease detection using machine learning",
        "radiology": "X-ray image analysis using deep learning for pneumonia detection", 
        "related_cardiology": "Cardiac imaging and AI-based diagnosis of heart conditions",
        "unrelated": "Stock market prediction using time series analysis"
    }
    
    from src.embedder import embed_text, cosine_similarity_single, cosine_similarity_matrix
    
    # Generate embeddings
    embeddings = {}
    print("🧠 Generating embeddings...")
    for key, text in texts.items():
        embeddings[key] = embed_text(text)
        print(f"  ✅ {key}: {text}")
    
    # Single similarity calculations
    print(f"\n🔢 Pairwise Similarity Calculations:")
    
    comparisons = [
        ("cardiology", "related_cardiology", "Related medical domains"),
        ("cardiology", "radiology", "Different medical domains"),
        ("cardiology", "unrelated", "Completely different topics"),
        ("radiology", "related_cardiology", "Medical imaging topics")
    ]
    
    for text1, text2, description in comparisons:
        similarity = cosine_similarity_single(
            embeddings[text1], 
            embeddings[text2]
        )
        
        # Interpret similarity score
        if similarity >= 0.8:
            interpretation = "🟢 Highly similar"
        elif similarity >= 0.6:
            interpretation = "🔵 Moderately similar"
        elif similarity >= 0.4:
            interpretation = "🟡 Somewhat similar"
        else:
            interpretation = "🔴 Not similar"
            
        print(f"  {text1} ↔ {text2}")
        print(f"    Score: {similarity:.3f} - {interpretation}")
        print(f"    Context: {description}")
        print()
    
    # Matrix similarity calculation
    print("📊 Similarity Matrix:")
    embedding_list = list(embeddings.values())
    text_keys = list(embeddings.keys())
    
    # Convert to numpy array for matrix calculation
    embedding_array = np.array(embedding_list)
    similarity_matrix = cosine_similarity_matrix(embedding_array, embedding_array)
    
    # Display matrix
    print("     ", end="")
    for key in text_keys:
        print(f"{key[:12]:>12}", end="")
    print()
    
    for i, key1 in enumerate(text_keys):
        print(f"{key1[:4]:>4} ", end="")
        for j, key2 in enumerate(text_keys):
            score = similarity_matrix[i, j]
            print(f"{score:>12.3f}", end="")
        print()
    
    return similarity_matrix

# Run similarity demo
similarity_matrix = demo_similarity_calculations()
```

## 📊 Journal Matching Examples

### Example 6: Basic Journal Search

```python
def demo_basic_journal_search():
    """Demonstrate basic journal matching functionality."""
    
    print("🔍 Basic Journal Search Demo")
    print("=" * 50)
    
    # Sample manuscript abstract
    manuscript_abstract = """
    This study presents a novel deep learning framework for automated 
    detection of diabetic retinopathy in fundus photographs. Using a 
    dataset of 50,000 retinal images, we developed a convolutional neural 
    network that achieves 96.3% accuracy in classifying disease severity. 
    The model demonstrates significant improvement over existing methods 
    and shows potential for deployment in clinical settings.
    """
    
    print(f"📝 Manuscript Abstract:")
    print(f"'{manuscript_abstract.strip()}'")
    print(f"Length: {len(manuscript_abstract)} characters")
    
    try:
        # Initialize matcher
        from src.match_journals import JournalMatcher
        matcher = JournalMatcher()
        
        print(f"\n🔄 Loading journal database...")
        matcher.load_database()
        
        # Get database statistics
        stats = matcher.get_database_stats()
        print(f"📊 Database loaded: {stats['total_journals']:,} journals")
        
        # Perform search
        print(f"\n🔍 Searching for similar journals...")
        results = matcher.search_similar_journals(
            query_text=manuscript_abstract,
            top_k=5,
            min_similarity=0.3
        )
        
        # Display results
        print(f"\n🏆 Top {len(results)} Journal Matches:")
        print("=" * 70)
        
        for i, journal in enumerate(results, 1):
            print(f"{i}. {journal['display_name']}")
            print(f"   📊 Similarity: {journal['similarity_score']:.3f}")
            print(f"   🏢 Publisher: {journal.get('publisher', 'Unknown')}")
            
            # Show additional metrics if available
            if journal.get('h_index'):
                print(f"   📈 H-index: {journal['h_index']}")
            if journal.get('is_oa'):
                print(f"   🌐 Open Access: {'Yes' if journal['is_oa'] else 'No'}")
            if journal.get('apc_usd'):
                print(f"   💰 APC: ${journal['apc_usd']:,}")
            
            print()
        
        return results
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        print("💡 This demo requires the journal database to be built first.")
        print("   Run: python scripts/build_database.py")
        return []

# Run basic search demo
search_results = demo_basic_journal_search()
```

### Example 7: Advanced Filtering

```python
def demo_advanced_filtering():
    """Demonstrate advanced journal filtering capabilities."""
    
    print("🎯 Advanced Journal Filtering Demo")
    print("=" * 50)
    
    # Clinical research abstract
    clinical_abstract = """
    A randomized controlled trial investigating the efficacy of a novel 
    telemedicine intervention for managing type 2 diabetes in rural 
    populations. We enrolled 1,200 patients across 15 rural clinics 
    and measured HbA1c levels over 12 months. Results show significant 
    improvement in glycemic control compared to standard care.
    """
    
    print(f"📝 Clinical Research Abstract:")
    print(f"'{clinical_abstract.strip()}'")
    
    try:
        from src.match_journals import JournalMatcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Define different filtering scenarios
        filter_scenarios = [
            {
                "name": "Open Access Only", 
                "filters": {"open_access_only": True},
                "description": "Only open access journals"
            },
            {
                "name": "High Impact",
                "filters": {"min_h_index": 50, "min_citations": 20000},
                "description": "High-impact journals (H-index ≥50, Citations ≥20k)"
            },
            {
                "name": "Budget Friendly",
                "filters": {"open_access_only": True, "max_apc": 2000},
                "description": "Open access with APC ≤$2000"
            },
            {
                "name": "Premium Journals",
                "filters": {"min_h_index": 100, "min_similarity": 0.7},
                "description": "Top-tier journals with high similarity"
            }
        ]
        
        # Test each filtering scenario
        for scenario in filter_scenarios:
            print(f"\n🎯 Scenario: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"🔧 Filters: {scenario['filters']}")
            
            results = matcher.search_similar_journals(
                query_text=clinical_abstract,
                top_k=5,
                filters=scenario['filters']
            )
            
            print(f"📊 Found {len(results)} matching journals")
            
            if results:
                print("🏆 Top matches:")
                for i, journal in enumerate(results[:3], 1):
                    similarity = journal['similarity_score']
                    name = journal['display_name']
                    h_index = journal.get('h_index', 0)
                    is_oa = journal.get('is_oa', False)
                    apc = journal.get('apc_usd', 0)
                    
                    print(f"  {i}. {name}")
                    print(f"     Similarity: {similarity:.3f}, H-index: {h_index}")
                    print(f"     Open Access: {'Yes' if is_oa else 'No'}")
                    if apc:
                        print(f"     APC: ${apc:,}")
            else:
                print("  ❌ No journals match these criteria")
            
            print("-" * 60)
        
    except Exception as e:
        print(f"❌ Advanced filtering demo failed: {e}")

# Run advanced filtering demo
demo_advanced_filtering()
```

### Example 8: Multi-Criteria Journal Ranking

```python
def demo_multi_criteria_ranking():
    """Demonstrate custom journal ranking using multiple criteria."""
    
    print("📈 Multi-Criteria Journal Ranking Demo")
    print("=" * 50)
    
    def calculate_composite_score(journal, weights):
        """Calculate composite score based on multiple criteria."""
        
        # Normalize similarity score (already 0-1)
        similarity_score = journal.get('similarity_score', 0)
        
        # Normalize H-index (cap at 200)
        h_index = journal.get('h_index', 0)
        h_index_score = min(h_index / 200.0, 1.0)
        
        # Open access bonus
        oa_score = 1.0 if journal.get('is_oa', False) else 0.5
        
        # Cost score (lower cost = higher score)
        apc = journal.get('apc_usd', 0)
        if apc == 0:
            cost_score = 1.0  # Free is best
        else:
            cost_score = max(0, 1.0 - (apc / 5000))  # Normalize to $5k max
        
        # Calculate weighted composite score
        composite = (
            weights['similarity'] * similarity_score +
            weights['impact'] * h_index_score + 
            weights['openness'] * oa_score +
            weights['cost'] * cost_score
        )
        
        return composite, {
            'similarity': similarity_score,
            'impact': h_index_score,
            'openness': oa_score,
            'cost': cost_score
        }
    
    # AI research abstract
    ai_abstract = """
    We propose a novel attention mechanism for medical image segmentation 
    that significantly improves accuracy in identifying tumor boundaries. 
    Our method combines transformer architectures with traditional CNNs 
    to achieve state-of-the-art results on three benchmark datasets.
    """
    
    try:
        from src.match_journals import JournalMatcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Get initial results
        initial_results = matcher.search_similar_journals(
            query_text=ai_abstract,
            top_k=20,  # Get more results for ranking
            min_similarity=0.4
        )
        
        if not initial_results:
            print("❌ No initial results found")
            return
        
        print(f"📊 Initial results: {len(initial_results)} journals")
        
        # Define ranking weights
        ranking_scenarios = [
            {
                "name": "Similarity Focused",
                "weights": {"similarity": 0.7, "impact": 0.2, "openness": 0.05, "cost": 0.05},
                "description": "Prioritize semantic similarity"
            },
            {
                "name": "Impact Focused", 
                "weights": {"similarity": 0.4, "impact": 0.4, "openness": 0.1, "cost": 0.1},
                "description": "Balance similarity with journal impact"
            },
            {
                "name": "Open Science",
                "weights": {"similarity": 0.3, "impact": 0.2, "openness": 0.4, "cost": 0.1},
                "description": "Prefer open access journals"
            },
            {
                "name": "Budget Conscious",
                "weights": {"similarity": 0.4, "impact": 0.2, "openness": 0.1, "cost": 0.3},
                "description": "Consider publication costs"
            }
        ]
        
        # Apply different ranking scenarios
        for scenario in ranking_scenarios:
            print(f"\n🎯 Ranking Scenario: {scenario['name']}")
            print(f"📋 {scenario['description']}")
            print(f"⚖️ Weights: {scenario['weights']}")
            
            # Calculate composite scores
            scored_journals = []
            for journal in initial_results:
                composite, breakdown = calculate_composite_score(journal, scenario['weights'])
                journal_copy = journal.copy()
                journal_copy['composite_score'] = composite
                journal_copy['score_breakdown'] = breakdown
                scored_journals.append(journal_copy)
            
            # Sort by composite score
            scored_journals.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Display top 3 results
            print(f"🏆 Top 3 Journals:")
            for i, journal in enumerate(scored_journals[:3], 1):
                print(f"\n  {i}. {journal['display_name']}")
                print(f"     🎯 Composite Score: {journal['composite_score']:.3f}")
                print(f"     📊 Similarity: {journal['score_breakdown']['similarity']:.3f}")
                print(f"     📈 Impact: {journal['score_breakdown']['impact']:.3f}")
                print(f"     🌐 Openness: {journal['score_breakdown']['openness']:.3f}")
                print(f"     💰 Cost: {journal['score_breakdown']['cost']:.3f}")
                
                # Show actual values
                h_index = journal.get('h_index', 0)
                is_oa = journal.get('is_oa', False)
                apc = journal.get('apc_usd', 0)
                print(f"     📋 H-index: {h_index}, OA: {'Yes' if is_oa else 'No'}, APC: ${apc:,}")
            
            print("-" * 70)
        
    except Exception as e:
        print(f"❌ Multi-criteria ranking demo failed: {e}")

# Run multi-criteria ranking demo
demo_multi_criteria_ranking()
```

## 🔧 Configuration Examples

### Example 9: Configuration Management

```python
def demo_configuration():
    """Demonstrate configuration management and customization."""
    
    print("⚙️ Configuration Management Demo")
    print("=" * 50)
    
    from src.config import *
    import os
    
    # Display current configuration
    print("📋 Current Configuration:")
    print(f"  Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Embedding dimension: {get_embedding_dimension()}")
    print(f"  Max file size: {MAX_FILE_SIZE_MB}MB")
    print(f"  Default results: {DEFAULT_TOP_K_RESULTS}")
    print(f"  Min similarity: {MIN_SIMILARITY_THRESHOLD}")
    print(f"  Supported file types: {', '.join(SUPPORTED_FILE_TYPES)}")
    
    # Test file validation
    print(f"\n📄 File Validation Examples:")
    
    test_files = [
        ("small_file.pdf", 5 * 1024 * 1024),      # 5MB
        ("large_file.pdf", 100 * 1024 * 1024),    # 100MB  
        ("document.docx", 10 * 1024 * 1024),      # 10MB
        ("text_file.txt", 1 * 1024 * 1024),       # 1MB
    ]
    
    for filename, size_bytes in test_files:
        file_ext = Path(filename).suffix.lower()
        size_mb = size_bytes / (1024 * 1024)
        
        size_ok = validate_file_size(size_bytes)
        type_ok = file_ext in SUPPORTED_FILE_TYPES
        
        status = "✅" if (size_ok and type_ok) else "❌"
        
        print(f"  {status} {filename} ({size_mb:.1f}MB, {file_ext})")
        if not size_ok:
            print(f"     ⚠️ Exceeds size limit of {MAX_FILE_SIZE_MB}MB")
        if not type_ok:
            print(f"     ⚠️ Unsupported file type")
    
    # Test environment variable override
    print(f"\n🔧 Configuration Override Example:")
    
    # Save original values
    original_max_results = DEFAULT_TOP_K_RESULTS
    original_model = EMBEDDING_MODEL_NAME
    
    print(f"  Original max results: {original_max_results}")
    
    # Set environment variable (this would normally be done before import)
    print(f"  Setting MAX_RESULTS=15 via environment...")
    os.environ["MAX_RESULTS"] = "15"
    
    # Note: In practice, you'd need to reload the config module
    # For demo purposes, we'll just show the concept
    new_max_results = int(os.getenv("MAX_RESULTS", "20"))
    print(f"  New max results would be: {new_max_results}")
    
    # Directory management
    print(f"\n📁 Directory Management:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Data dir exists: {'✅' if DATA_DIR.exists() else '❌'}")
    
    # Ensure directories exist
    ensure_directories_exist()
    print(f"  ✅ All required directories created")
    
    # API configuration
    print(f"\n🌐 API Configuration:")
    headers = get_api_headers()
    for key, value in headers.items():
        print(f"  {key}: {value}")
    
    # Clean up environment
    if "MAX_RESULTS" in os.environ:
        del os.environ["MAX_RESULTS"]

# Run configuration demo
demo_configuration()
```

### Example 10: Utility Functions

```python
def demo_utility_functions():
    """Demonstrate utility functions for text processing and validation."""
    
    print("🛠️ Utility Functions Demo")
    print("=" * 50)
    
    from src.utils import (
        clean_text, extract_keywords, compute_text_hash,
        save_cache, load_cache, format_file_size
    )
    
    # Text cleaning demo
    messy_text = """
        This    is  a   messy    text   with    lots  of      whitespace
        and special characters: @#$%^&*(){}[]|\\:";'<>?,./
        
        It also has multiple
        
        
        line breaks.
    """
    
    print("📝 Text Cleaning Demo:")
    print(f"Original text ({len(messy_text)} chars):")
    print(f"'{messy_text[:100]}...'")
    
    # Clean with different options
    cleaning_options = [
        {"remove_extra_whitespace": True, "remove_special_chars": False},
        {"remove_extra_whitespace": True, "remove_special_chars": True},
        {"remove_extra_whitespace": True, "remove_special_chars": True, "max_length": 100}
    ]
    
    for i, options in enumerate(cleaning_options, 1):
        cleaned = clean_text(messy_text, **options)
        print(f"\nCleaning option {i}: {options}")
        print(f"Result ({len(cleaned)} chars): '{cleaned[:80]}...'")
    
    # Keyword extraction demo
    print(f"\n🔍 Keyword Extraction Demo:")
    
    medical_text = """
    This research investigates machine learning applications in medical 
    diagnosis, specifically focusing on deep learning algorithms for 
    analyzing medical images. The study examines convolutional neural 
    networks, image classification, pattern recognition, and automated 
    diagnosis systems in healthcare settings.
    """
    
    keywords = extract_keywords(medical_text, top_k=10)
    print(f"Text: '{medical_text[:100]}...'")
    print(f"Extracted keywords: {', '.join(keywords)}")
    
    # Text hashing demo
    print(f"\n🔐 Text Hashing Demo:")
    texts = [
        "Machine learning in healthcare",
        "Deep learning for medical diagnosis", 
        "Machine learning in healthcare"  # Duplicate
    ]
    
    hashes = {}
    for text in texts:
        text_hash = compute_text_hash(text)
        if text_hash in hashes:
            print(f"📄 '{text}' - Hash: {text_hash[:16]}... (DUPLICATE)")
        else:
            print(f"📄 '{text}' - Hash: {text_hash[:16]}...")
            hashes[text_hash] = text
    
    # Caching demo
    print(f"\n💾 Caching Demo:")
    
    # Save some data to cache
    cache_data = {
        "search_results": ["Journal A", "Journal B", "Journal C"],
        "timestamp": "2024-07-29",
        "query": "machine learning medical"
    }
    
    cache_key = "demo_search_results"
    print(f"Saving data to cache with key: {cache_key}")
    save_cache(cache_key, cache_data, expiry_hours=1)
    
    # Load from cache
    loaded_data = load_cache(cache_key)
    if loaded_data:
        print(f"✅ Successfully loaded from cache:")
        print(f"  Results: {loaded_data['search_results']}")
        print(f"  Query: {loaded_data['query']}")
    else:
        print("❌ Failed to load from cache")
    
    # File size formatting demo
    print(f"\n📊 File Size Formatting Demo:")
    
    sizes_bytes = [1024, 1024**2, 1024**3, 1.5 * 1024**3, 0, 500]
    
    for size in sizes_bytes:
        formatted = format_file_size(size)
        print(f"  {size:>12,} bytes = {formatted}")

# Run utility functions demo
demo_utility_functions()
```

## 🎯 Complete Workflow Examples

### Example 11: End-to-End Manuscript Processing

```python
def demo_complete_workflow():
    """Demonstrate complete end-to-end manuscript processing workflow."""
    
    print("🔄 Complete Manuscript Processing Workflow")
    print("=" * 60)
    
    # Simulate a complete manuscript processing pipeline
    manuscript_text = """
    Federated Learning for Privacy-Preserving Medical AI: 
    A Multi-Institutional Study
    
    Abstract: This study presents a federated learning framework 
    for training machine learning models on distributed medical 
    datasets while preserving patient privacy. We conducted 
    experiments across 12 hospitals with a combined dataset of 
    100,000 patient records. Our approach achieves comparable 
    accuracy to centralized training while maintaining strict 
    privacy constraints. The framework demonstrates significant 
    potential for collaborative medical AI research without 
    compromising sensitive health information.
    
    Keywords: federated learning, medical AI, privacy, healthcare
    
    1. Introduction
    The healthcare industry generates vast amounts of data...
    """
    
    try:
        # Step 1: Document Extraction
        print("📄 Step 1: Document Extraction")
        print("-" * 30)
        
        from src.extractor import extract_title_and_abstract, validate_extracted_data
        
        title, abstract = extract_title_and_abstract(manuscript_text)
        
        manuscript_data = {
            'title': title,
            'abstract': abstract,
            'full_text': manuscript_text,
            'file_type': '.pdf',
            'file_name': 'federated_learning_paper.pdf'
        }
        
        validation = validate_extracted_data(manuscript_data)
        
        print(f"✅ Title extracted: {title}")
        print(f"✅ Abstract extracted: {len(abstract)} characters")
        print(f"✅ Validation: {validation['status']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"⚠️ {warning}")
        
        # Step 2: Text Preprocessing  
        print(f"\n🔧 Step 2: Text Preprocessing")
        print("-" * 30)
        
        from src.utils import clean_text, extract_keywords
        
        # Clean abstract for better matching
        clean_abstract = clean_text(
            abstract, 
            remove_extra_whitespace=True,
            max_length=1000
        )
        
        # Extract keywords for analysis
        keywords = extract_keywords(clean_abstract, top_k=8)
        
        print(f"✅ Text cleaned: {len(clean_abstract)} characters")
        print(f"✅ Keywords extracted: {', '.join(keywords)}")
        
        # Step 3: Embedding Generation
        print(f"\n🧠 Step 3: Embedding Generation")
        print("-" * 30)
        
        from src.embedder import embed_text, get_embedding_info
        
        model_info = get_embedding_info()
        abstract_embedding = embed_text(clean_abstract)
        
        print(f"✅ Model: {model_info['model_name']}")
        print(f"✅ Embedding generated: {abstract_embedding.shape}")
        print(f"✅ Embedding norm: {np.linalg.norm(abstract_embedding):.4f}")
        
        # Step 4: Journal Matching
        print(f"\n🔍 Step 4: Journal Matching")
        print("-" * 30)
        
        from src.match_journals import JournalMatcher
        
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Search with progressive filtering
        search_scenarios = [
            {
                "name": "High Impact Open Access",
                "params": {
                    "top_k": 5,
                    "min_similarity": 0.6,
                    "filters": {
                        "open_access_only": True,
                        "min_h_index": 50
                    }
                }
            },
            {
                "name": "Budget Friendly", 
                "params": {
                    "top_k": 5,
                    "min_similarity": 0.5,
                    "filters": {
                        "open_access_only": True,
                        "max_apc": 2000
                    }
                }
            },
            {
                "name": "All Options",
                "params": {
                    "top_k": 10,
                    "min_similarity": 0.4
                }
            }
        ]
        
        all_recommendations = {}
        
        for scenario in search_scenarios:
            print(f"\n  🎯 {scenario['name']}:")
            
            results = matcher.search_similar_journals(
                query_text=clean_abstract,
                **scenario['params']
            )
            
            all_recommendations[scenario['name']] = results
            
            if results:
                print(f"  ✅ Found {len(results)} journals")
                for i, journal in enumerate(results[:3], 1):
                    name = journal['display_name']
                    similarity = journal['similarity_score']
                    h_index = journal.get('h_index', 0)
                    is_oa = journal.get('is_oa', False)
                    
                    print(f"    {i}. {name}")
                    print(f"       Similarity: {similarity:.3f}, H-index: {h_index}")
                    print(f"       Open Access: {'Yes' if is_oa else 'No'}")
            else:
                print(f"  ❌ No journals found for this scenario")
        
        # Step 5: Results Analysis and Export
        print(f"\n📊 Step 5: Results Analysis")
        print("-" * 30)
        
        # Analyze recommendations across scenarios
        unique_journals = set()
        total_recommendations = 0
        
        for scenario_name, results in all_recommendations.items():
            total_recommendations += len(results)
            for journal in results:
                unique_journals.add(journal['display_name'])
        
        print(f"✅ Total recommendations: {total_recommendations}")
        print(f"✅ Unique journals: {len(unique_journals)}")
        
        # Find top overall recommendations
        if all_recommendations.get("All Options"):
            top_journals = all_recommendations["All Options"][:5]
            
            print(f"\n🏆 Top 5 Overall Recommendations:")
            for i, journal in enumerate(top_journals, 1):
                print(f"  {i}. {journal['display_name']}")
                print(f"     📊 Similarity: {journal['similarity_score']:.3f}")
                print(f"     🏢 Publisher: {journal.get('publisher', 'Unknown')}")
                
                if journal.get('homepage_url'):
                    print(f"     🔗 URL: {journal['homepage_url']}")
        
        # Step 6: Generate Summary Report
        print(f"\n📋 Step 6: Processing Summary")
        print("-" * 30)
        
        summary = {
            "manuscript": {
                "title": title,
                "abstract_length": len(abstract),
                "keywords": keywords,
                "domain": "AI/Healthcare" if any(kw in keywords for kw in ["learning", "medical", "healthcare"]) else "Unknown"
            },
            "processing": {
                "extraction_success": bool(title and abstract),
                "embedding_dimension": abstract_embedding.shape[0],
                "total_recommendations": total_recommendations,
                "unique_journals": len(unique_journals)
            },
            "top_recommendation": top_journals[0] if top_journals else None
        }
        
        print(f"✅ Processing completed successfully!")
        print(f"  📋 Title extracted: {'Yes' if summary['manuscript']['title'] else 'No'}")
        print(f"  📝 Abstract: {summary['manuscript']['abstract_length']} chars")
        print(f"  🏷️ Domain: {summary['manuscript']['domain']}")
        print(f"  🧠 Embedding: {summary['processing']['embedding_dimension']}D")
        print(f"  📊 Recommendations: {summary['processing']['total_recommendations']}")
        
        if summary['top_recommendation']:
            top = summary['top_recommendation']
            print(f"  🏆 Top match: {top['display_name']} ({top['similarity_score']:.3f})")
        
        return summary
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        print("💡 Make sure the journal database is built and accessible")
        return None

# Run complete workflow demo
workflow_summary = demo_complete_workflow()
```

## 🎉 Running All Examples

```python
def run_all_examples():
    """Run all API examples in sequence."""
    
    print("🚀 MANUSCRIPT JOURNAL MATCHER - API EXAMPLES")
    print("=" * 80)
    print("Running comprehensive examples for all API functions...")
    print()
    
    examples = [
        ("Document Extraction", demo_pdf_extraction),
        ("Batch Processing", demo_batch_extraction), 
        ("Single Embedding", demo_single_embedding),
        ("Batch Embeddings", demo_batch_embeddings),
        ("Similarity Calculations", demo_similarity_calculations),
        ("Basic Journal Search", demo_basic_journal_search),
        ("Advanced Filtering", demo_advanced_filtering),
        ("Multi-Criteria Ranking", demo_multi_criteria_ranking),
        ("Configuration", demo_configuration),
        ("Utility Functions", demo_utility_functions),
        ("Complete Workflow", demo_complete_workflow)
    ]
    
    results = {}
    
    for name, example_func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = example_func()
            results[name] = {"status": "success", "result": result}
            print(f"✅ {name} completed successfully")
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            print(f"❌ {name} failed: {e}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("📊 EXAMPLES SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    total = len(results)
    
    print(f"Total examples: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    print("\n📋 Detailed Results:")
    for name, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        print(f"  {status} {name}")
        if result["status"] == "error":
            print(f"    Error: {result['error']}")
    
    return results

# Uncomment the line below to run all examples
# all_results = run_all_examples()
```

---

*These examples provide comprehensive, hands-on demonstrations of the Manuscript Journal Matcher API. Each example includes detailed explanations, expected outputs, and practical use cases. For additional examples and integration patterns, see the [User Guide](../user/) and [API Reference](README.md).*