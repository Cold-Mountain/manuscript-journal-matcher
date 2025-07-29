# Text Embedding API Reference

The `embedder` module handles text embedding generation using sentence-transformers and provides utilities for similarity calculations and batch processing with the all-MiniLM-L6-v2 model.

## 📋 Module Overview

```python
from src.embedder import (
    embed_text,                    # Single text embedding
    embed_texts,                   # Batch text embeddings
    initialize_embedding_model,    # Model initialization
    get_model,                     # Get current model instance
    cosine_similarity_single,      # Single similarity calculation
    cosine_similarity_matrix,      # Batch similarity calculation
    find_most_similar,             # Top-K similarity search
    validate_embedding_dimension,  # Dimension validation
    get_embedding_info,           # Model information
    EmbeddingError                # Custom exception
)
```

## 🚀 Quick Start

```python
from src.embedder import embed_text, cosine_similarity_single

# Generate embeddings for two texts
text1 = "Machine learning in medical diagnosis"
text2 = "AI applications in healthcare"

embedding1 = embed_text(text1)
embedding2 = embed_text(text2)

# Calculate similarity
similarity = cosine_similarity_single(embedding1, embedding2)
print(f"Similarity: {similarity:.3f}")  # Output: Similarity: 0.756
```

## 🧠 Core Functions

### embed_text()

**Generate embedding for a single text string.**

```python
def embed_text(text: str, model: Optional[SentenceTransformer] = None) -> np.ndarray
```

**Parameters:**
- `text` (str): Input text to embed
- `model` (SentenceTransformer, optional): Model instance. If None, uses default model.

**Returns:**
- `np.ndarray`: 384-dimensional embedding vector

**Raises:**
- `EmbeddingError`: If embedding generation fails

**Example:**
```python
# Basic embedding generation
abstract = """
This study investigates the use of machine learning algorithms 
for early detection of cardiovascular diseases using ECG data.
"""

embedding = embed_text(abstract)
print(f"Embedding shape: {embedding.shape}")  # (384,)
print(f"Embedding type: {type(embedding)}")   # <class 'numpy.ndarray'>

# Verify embedding properties
assert embedding.ndim == 1
assert len(embedding) == 384
print(f"✅ Generated 384-dimensional embedding")
```

### embed_texts()

**Generate embeddings for multiple texts efficiently.**

```python
def embed_texts(
    texts: List[str], 
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32, 
    show_progress: bool = False
) -> np.ndarray
```

**Parameters:**
- `texts` (List[str]): List of input texts to embed
- `model` (SentenceTransformer, optional): Model instance
- `batch_size` (int): Number of texts to process in each batch (default: 32)
- `show_progress` (bool): Whether to show progress bar (default: False)

**Returns:**
- `np.ndarray`: 2D array where each row is an embedding (N × 384)

**Raises:**
- `EmbeddingError`: If embedding generation fails

**Example:**
```python
# Batch embedding generation
abstracts = [
    "Machine learning for medical diagnosis",
    "Deep learning in radiology imaging", 
    "AI-powered drug discovery methods",
    "Natural language processing in healthcare",
    "Computer vision for pathology analysis"
]

# Generate embeddings with progress bar
embeddings = embed_texts(
    abstracts, 
    batch_size=16,
    show_progress=True
)

print(f"Generated embeddings: {embeddings.shape}")  # (5, 384)

# Process each embedding
for i, embedding in enumerate(embeddings):
    print(f"Text {i+1}: {abstracts[i][:30]}...")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.3f}")
```

### initialize_embedding_model()

**Initialize and return the sentence transformer model.**

```python
def initialize_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer
```

**Parameters:**
- `model_name` (str, optional): Model name override. If None, uses config default.

**Returns:**
- `SentenceTransformer`: Initialized model instance

**Raises:**
- `EmbeddingError`: If model initialization fails

**Supported Models:**
- `all-MiniLM-L6-v2` (default): 384 dimensions, fast and accurate
- `all-MiniLM-L12-v2`: 384 dimensions, higher accuracy
- `all-mpnet-base-v2`: 768 dimensions, best accuracy
- `SciBERT`: 768 dimensions, scientific texts

**Example:**
```python
# Initialize default model
model = initialize_embedding_model()
print(f"Model: {model._model_name}")
print(f"Dimension: {model.get_sentence_embedding_dimension()}")
print(f"Device: {model.device}")

# Initialize specific model
scientific_model = initialize_embedding_model("SciBERT")
print(f"Scientific model dimension: {scientific_model.get_sentence_embedding_dimension()}")

# Model comparison
texts = ["Machine learning in medical research"]

default_emb = model.encode(texts)[0]
scientific_emb = scientific_model.encode(texts)[0]

print(f"Default model: {len(default_emb)} dims")      # 384
print(f"SciBERT model: {len(scientific_emb)} dims")   # 768
```

### cosine_similarity_single()

**Calculate cosine similarity between two vectors.**

```python
def cosine_similarity_single(vec1: np.ndarray, vec2: np.ndarray) -> float
```

**Parameters:**
- `vec1` (np.ndarray): First vector
- `vec2` (np.ndarray): Second vector

**Returns:**
- `float`: Cosine similarity score between -1 and 1

**Raises:**
- `EmbeddingError`: If similarity calculation fails

**Example:**
```python
# Compare different medical texts
cardiology_text = "Echocardiogram shows normal cardiac function"
neurology_text = "MRI reveals no brain abnormalities" 
related_cardiology = "Cardiac ultrasound demonstrates healthy heart"

# Generate embeddings
emb1 = embed_text(cardiology_text)
emb2 = embed_text(neurology_text)  
emb3 = embed_text(related_cardiology)

# Calculate similarities
sim_different = cosine_similarity_single(emb1, emb2)
sim_related = cosine_similarity_single(emb1, emb3)

print(f"Cardiology vs Neurology: {sim_different:.3f}")    # ~0.456
print(f"Cardiology vs Related: {sim_related:.3f}")       # ~0.834

# Interpretation
if sim_related > 0.8:
    print("✅ Highly similar topics")
elif sim_related > 0.6:
    print("📊 Moderately similar topics")
else:
    print("📝 Different topics")
```

### cosine_similarity_matrix()

**Calculate cosine similarity matrix between two sets of embeddings.**

```python
def cosine_similarity_matrix(
    embeddings1: np.ndarray, 
    embeddings2: np.ndarray
) -> np.ndarray
```

**Parameters:**
- `embeddings1` (np.ndarray): First set of embeddings (N × D)
- `embeddings2` (np.ndarray): Second set of embeddings (M × D)

**Returns:**
- `np.ndarray`: Similarity matrix (N × M) where entry (i,j) is similarity between embeddings1[i] and embeddings2[j]

**Raises:**
- `EmbeddingError`: If similarity calculation fails

**Example:**
```python
# Compare manuscript abstracts to journal descriptions
manuscript_abstracts = [
    "Machine learning for cardiovascular disease prediction",
    "Deep learning in radiology image analysis",
    "NLP for clinical text mining"
]

journal_descriptions = [
    "Cardiology research and clinical practice",
    "Medical imaging and diagnostic radiology", 
    "Health informatics and medical AI",
    "General medicine and clinical studies"
]

# Generate embeddings
manuscript_embs = embed_texts(manuscript_abstracts)
journal_embs = embed_texts(journal_descriptions)

# Calculate similarity matrix
similarity_matrix = cosine_similarity_matrix(manuscript_embs, journal_embs)

print(f"Similarity matrix shape: {similarity_matrix.shape}")  # (3, 4)

# Find best matches
for i, abstract in enumerate(manuscript_abstracts):
    best_journal_idx = np.argmax(similarity_matrix[i])
    best_similarity = similarity_matrix[i, best_journal_idx]
    
    print(f"\n📄 Manuscript {i+1}: {abstract[:50]}...")
    print(f"🏆 Best match: {journal_descriptions[best_journal_idx]}")
    print(f"📊 Similarity: {best_similarity:.3f}")
```

### find_most_similar()

**Find the most similar embeddings to a query embedding.**

```python
def find_most_similar(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int = 10
) -> List[Tuple[int, float]]
```

**Parameters:**
- `query_embedding` (np.ndarray): Query embedding vector (1D)
- `candidate_embeddings` (np.ndarray): Array of candidate embeddings (2D)
- `top_k` (int): Number of top results to return (default: 10)

**Returns:**
- `List[Tuple[int, float]]`: List of (index, similarity_score) tuples, sorted by similarity descending

**Raises:**
- `EmbeddingError`: If similarity search fails

**Example:**
```python
# Find most similar journals for a manuscript
manuscript_abstract = """
This study presents a novel deep learning approach for automated 
analysis of chest X-rays to detect pneumonia and other respiratory 
conditions using convolutional neural networks.
"""

# Sample journal descriptions
journal_descriptions = [
    "Radiology and medical imaging research",
    "Artificial intelligence in healthcare",
    "Respiratory medicine and pulmonology", 
    "General medical practice and clinical care",
    "Computer science and machine learning",
    "Medical AI and diagnostic systems"
]

# Generate embeddings
query_embedding = embed_text(manuscript_abstract)
journal_embeddings = embed_texts(journal_descriptions)

# Find top matches
top_matches = find_most_similar(
    query_embedding, 
    journal_embeddings, 
    top_k=3
)

print("🔍 Top journal matches:")
for rank, (idx, score) in enumerate(top_matches, 1):
    print(f"{rank}. {journal_descriptions[idx]}")
    print(f"   Similarity: {score:.3f}")
    print()

# Expected output:
# 1. Medical AI and diagnostic systems
#    Similarity: 0.892
# 2. Radiology and medical imaging research  
#    Similarity: 0.847
# 3. Artificial intelligence in healthcare
#    Similarity: 0.823
```

## 🔧 Model Management

### get_model()

**Get the current embedding model instance, initializing if necessary.**

```python
def get_model() -> SentenceTransformer
```

**Returns:**
- `SentenceTransformer`: Current model instance

**Example:**
```python
# Get current model (lazy initialization)
model = get_model()

# Check model properties
info = {
    'name': getattr(model, '_model_name', 'Unknown'),
    'dimension': model.get_sentence_embedding_dimension(),
    'device': str(model.device),
    'max_length': getattr(model, 'max_seq_length', 'Unknown')
}

print("🤖 Current embedding model:")
for key, value in info.items():
    print(f"  {key}: {value}")
```

### get_embedding_info()

**Get information about the current embedding model.**

```python
def get_embedding_info() -> dict
```

**Returns:**
- `dict`: Dictionary with model information

**Example:**
```python
# Get detailed model information
info = get_embedding_info()

if info.get('loaded'):
    print("✅ Model loaded successfully")
    print(f"📊 Model: {info['model_name']}")
    print(f"📐 Dimensions: {info['dimension']}")
    print(f"🖥️ Device: {info['device']}")
    print(f"📏 Max sequence length: {info['max_sequence_length']}")
else:
    print(f"❌ Model not loaded: {info['error']}")
```

### validate_embedding_dimension()

**Validate that an embedding has the expected dimension.**

```python
def validate_embedding_dimension(
    embedding: np.ndarray, 
    expected_dim: Optional[int] = None
) -> bool
```

**Parameters:**
- `embedding` (np.ndarray): Embedding array to validate
- `expected_dim` (int, optional): Expected dimension. If None, uses config default (384).

**Returns:**
- `bool`: True if dimension is correct, False otherwise

**Example:**
```python
# Validate embedding dimensions
text = "Sample text for embedding"
embedding = embed_text(text)

# Validate against default dimension (384)
is_valid = validate_embedding_dimension(embedding)
print(f"Default validation: {is_valid}")  # True

# Validate against specific dimension
is_768_dim = validate_embedding_dimension(embedding, expected_dim=768)
print(f"768-dim validation: {is_768_dim}")  # False

# Validate batch embeddings
batch_embeddings = embed_texts(["text1", "text2", "text3"])
for i, emb in enumerate(batch_embeddings):
    valid = validate_embedding_dimension(emb)
    print(f"Embedding {i+1} valid: {valid}")
```

## 🚨 Error Handling

### EmbeddingError

Custom exception for embedding-related errors:

```python
from src.embedder import EmbeddingError

try:
    # This will fail - empty text
    embedding = embed_text("")
except EmbeddingError as e:
    print(f"❌ Embedding error: {e}")
    # Output: Cannot embed empty or whitespace-only text

try:
    # This will fail - dimension mismatch
    vec1 = np.random.rand(384)
    vec2 = np.random.rand(768)  # Wrong dimension
    similarity = cosine_similarity_single(vec1, vec2)
except EmbeddingError as e:
    print(f"❌ Similarity error: {e}")
    # Output: Vector shape mismatch: (384,) vs (768,)
```

### Robust Error Handling

```python
def safe_embed_text(text, fallback=""):
    """Safely embed text with fallback."""
    try:
        if not text or not text.strip():
            if fallback:
                return embed_text(fallback)
            else:
                raise ValueError("No text to embed")
        
        return embed_text(text)
        
    except EmbeddingError as e:
        print(f"⚠️ Embedding failed: {e}")
        if fallback:
            print(f"🔄 Using fallback text: {fallback[:50]}...")
            return embed_text(fallback)
        else:
            # Return zero vector as last resort
            from src.config import get_embedding_dimension
            return np.zeros(get_embedding_dimension())
```

## 📊 Performance Optimization

### Batch Processing

```python
def efficient_batch_embedding(texts, batch_size=64):
    """Efficiently process large batches of texts."""
    
    print(f"🔄 Processing {len(texts)} texts in batches of {batch_size}")
    
    # Filter out empty texts
    valid_texts = [text for text in texts if text and text.strip()]
    print(f"📊 {len(valid_texts)} valid texts after filtering")
    
    # Process in optimal batch size
    embeddings = embed_texts(
        valid_texts,
        batch_size=batch_size,
        show_progress=True
    )
    
    print(f"✅ Generated {embeddings.shape[0]} embeddings")
    return embeddings

# Usage example
large_text_list = ["text " + str(i) for i in range(1000)]
embeddings = efficient_batch_embedding(large_text_list, batch_size=128)
```

### Memory Management

```python
def memory_efficient_similarity_search(query_text, candidate_texts):
    """Memory-efficient similarity search for large datasets."""
    
    # Generate query embedding once
    query_embedding = embed_text(query_text)
    
    # Process candidates in chunks to avoid memory issues
    chunk_size = 1000
    all_similarities = []
    
    for i in range(0, len(candidate_texts), chunk_size):
        chunk = candidate_texts[i:i + chunk_size]
        
        # Process chunk
        chunk_embeddings = embed_texts(chunk, batch_size=64)
        
        # Calculate similarities for this chunk
        query_2d = query_embedding.reshape(1, -1)
        chunk_similarities = cosine_similarity_matrix(query_2d, chunk_embeddings)[0]
        
        # Store with original indices
        for j, sim in enumerate(chunk_similarities):
            all_similarities.append((i + j, sim))
        
        # Clear chunk from memory
        del chunk_embeddings
    
    # Sort and return top results
    all_similarities.sort(key=lambda x: x[1], reverse=True)
    return all_similarities[:50]  # Top 50 results
```

### Caching Embeddings

```python
import hashlib
from src.utils import save_cache, load_cache

def cached_embed_text(text):
    """Embed text with caching for repeated requests."""
    
    # Create cache key from text hash
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_key = f"embedding_{text_hash}"
    
    # Try to load from cache
    cached_embedding = load_cache(cache_key)
    if cached_embedding is not None:
        print("📄 Loaded embedding from cache")
        return np.array(cached_embedding)
    
    # Generate new embedding
    embedding = embed_text(text)
    
    # Save to cache (embeddings expire after 24 hours)
    save_cache(cache_key, embedding.tolist(), expiry_hours=24)
    print("💾 Saved embedding to cache")
    
    return embedding
```

## 🎯 Best Practices

### 1. Text Preprocessing

```python
def preprocess_for_embedding(text):
    """Preprocess text for optimal embedding quality."""
    
    if not text:
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate very long texts (model has token limits)
    max_chars = 2048  # Rough estimate for token limit
    if len(text) > max_chars:
        # Truncate at word boundary
        text = text[:max_chars].rsplit(' ', 1)[0] + "..."
        print(f"⚠️ Text truncated to {max_chars} characters")
    
    return text

# Usage
raw_text = "   Very long manuscript with lots of whitespace...   "
clean_text = preprocess_for_embedding(raw_text)
embedding = embed_text(clean_text)
```

### 2. Similarity Thresholds

```python
def interpret_similarity(score):
    """Interpret cosine similarity scores."""
    if score >= 0.9:
        return "🟢 Highly similar (>90%)"
    elif score >= 0.8:
        return "🔵 Very similar (80-90%)"
    elif score >= 0.7:
        return "🟡 Moderately similar (70-80%)"
    elif score >= 0.6:
        return "🟠 Somewhat similar (60-70%)"
    elif score >= 0.5:
        return "🔴 Slightly similar (50-60%)"
    else:
        return "⚫ Not similar (<50%)"

# Usage
similarity = cosine_similarity_single(emb1, emb2)
interpretation = interpret_similarity(similarity)
print(f"Similarity: {similarity:.3f} - {interpretation}")
```

### 3. Model Selection

```python
def choose_optimal_model(text_type="general"):
    """Choose the best model for different text types."""
    
    model_recommendations = {
        "general": "all-MiniLM-L6-v2",      # Fast, good quality
        "scientific": "SciBERT",             # Scientific papers
        "long_documents": "all-mpnet-base-v2", # Better for long texts
        "multilingual": "distiluse-base-multilingual-cased"
    }
    
    recommended = model_recommendations.get(text_type, "all-MiniLM-L6-v2")
    print(f"💡 Recommended model for '{text_type}': {recommended}")
    
    return initialize_embedding_model(recommended)

# Usage
scientific_model = choose_optimal_model("scientific")
general_model = choose_optimal_model("general")
```

---

*For more examples and integration patterns, see the [User Guide](../user/) and [Journal Matching API](match_journals.md).*