# Journal Matching API Reference

The `match_journals` module implements FAISS-based similarity search to find the most relevant journals for a given manuscript based on semantic embeddings with support for advanced filtering and quality metrics.

## 📋 Module Overview

```python
from src.match_journals import (
    JournalMatcher,                    # Main matching class
    create_faiss_index,               # Index creation utility
    load_journal_database_with_index, # Database loading utility
    search_similar_journals,          # Standalone search function
    rank_and_filter_results,          # Result processing
    format_search_results,            # Result formatting
    MatchingError                     # Custom exception
)
```

## 🚀 Quick Start

```python
from src.match_journals import JournalMatcher

# Initialize matcher
matcher = JournalMatcher()
matcher.load_database()

# Search for similar journals
results = matcher.search_similar_journals(
    query_text="Machine learning for medical diagnosis",
    top_k=10
)

# Display results
for journal in results:
    print(f"{journal['display_name']}: {journal['similarity_score']:.3f}")
```

## 🏆 JournalMatcher Class

The main class for journal matching using FAISS-based vector search.

### Initialization

```python
def __init__(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None)
```

**Parameters:**
- `index_path` (Path, optional): Path to FAISS index file
- `metadata_path` (Path, optional): Path to journal metadata file

**Example:**
```python
# Use default paths
matcher = JournalMatcher()

# Use custom paths
matcher = JournalMatcher(
    index_path=Path("custom_data/my_index.faiss"),
    metadata_path=Path("custom_data/my_journals.json")
)

print(f"📊 Index path: {matcher.index_path}")
print(f"📄 Metadata path: {matcher.metadata_path}")
```

### load_database()

**Load journal database and create/load FAISS index.**

```python
def load_database(self, force_reload: bool = False) -> None
```

**Parameters:**
- `force_reload` (bool): Whether to force reload even if already loaded

**Raises:**
- `MatchingError`: If database loading fails

**Example:**
```python
# Load database (lazy loading)
matcher.load_database()
print(f"✅ Loaded {len(matcher.journals)} journals")

# Force reload (useful during development)
matcher.load_database(force_reload=True)
print("🔄 Database reloaded")

# Check loading status
if matcher.faiss_index is not None:
    print(f"📊 FAISS index ready: {matcher.faiss_index.ntotal} vectors")
else:
    print("❌ FAISS index not loaded")
```

### search_similar_journals()

**Find journals most similar to the given query text.**

```python
def search_similar_journals(
    self,
    query_text: str,
    top_k: int = None,
    min_similarity: float = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

**Parameters:**
- `query_text` (str): Text to search for (typically manuscript abstract)
- `top_k` (int, optional): Number of top results to return (default: 20)
- `min_similarity` (float, optional): Minimum similarity threshold (default: 0.1)
- `filters` (Dict, optional): Additional filters to apply to results

**Returns:**
- `List[Dict[str, Any]]`: List of matching journals with similarity scores

**Raises:**
- `MatchingError`: If search fails

**Basic Example:**
```python
# Simple search
results = matcher.search_similar_journals(
    query_text="Deep learning applications in radiology",
    top_k=5
)

for i, journal in enumerate(results, 1):
    print(f"#{i} {journal['display_name']}")
    print(f"   Similarity: {journal['similarity_score']:.3f}")
    print(f"   Publisher: {journal.get('publisher', 'Unknown')}")
    print()
```

**Advanced Filtering Example:**
```python
# Advanced search with comprehensive filters
results = matcher.search_similar_journals(
    query_text=manuscript_abstract,
    top_k=15,
    min_similarity=0.6,
    filters={
        # Open access and quality filters
        'open_access_only': True,        # Only open access journals
        'doaj_only': True,              # Only DOAJ-listed journals
        'no_apc_only': False,           # Allow journals with APCs
        
        # Cost filters
        'max_apc': 2500,                # Maximum $2500 APC
        'min_apc': None,                # No minimum APC
        
        # Subject and content filters
        'subjects': ['Medicine', 'Biology'],  # Subject areas
        'languages': ['English'],        # Publication languages
        'publishers': ['PLOS', 'BioMed Central'],  # Preferred publishers
        
        # Quality filters  
        'min_citations': 10000,         # Minimum citations
        'min_h_index': 50              # Minimum H-index
    }
)

print(f"🔍 Found {len(results)} journals matching criteria")
```

## 🎯 Filtering Options

The system supports comprehensive filtering based on journal metadata:

### Open Access Filters

```python
# Open access only
results = matcher.search_similar_journals(
    query_text=abstract,
    filters={'open_access_only': True}
)

# DOAJ-listed journals only (high quality open access)
results = matcher.search_similar_journals(
    query_text=abstract,
    filters={'doaj_only': True}
)

# Free to publish (no APCs)
results = matcher.search_similar_journals(
    query_text=abstract,
    filters={'no_apc_only': True}
)
```

### Cost Filters

```python
# Budget-conscious search
results = matcher.search_similar_journals(
    query_text=abstract,
    filters={
        'max_apc': 1500,      # Maximum $1500 APC
        'open_access_only': True
    }
)

# Premium journals (high APC, high quality)
results = matcher.search_similar_journals(
    query_text=abstract,
    filters={
        'min_apc': 3000,      # Minimum $3000 APC
        'min_h_index': 100    # High H-index
    }
)
```

### Subject and Content Filters

```python
# Medical AI research
results = matcher.search_similar_journals(
    query_text=abstract,
    filters={
        'subjects': ['Medicine', 'Computer Science', 'Artificial Intelligence'],
        'languages': ['English'],
        'min_citations': 50000
    }
)

# Specific publishers
results = matcher.search_similar_journals(
    query_text=abstract,
    filters={
        'publishers': ['Nature Publishing Group', 'Elsevier', 'Springer'],
        'min_h_index': 80
    }
)
```

### Quality Filters

```python
# High-impact journals only
results = matcher.search_similar_journals(
    query_text=abstract,
    filters={
        'min_citations': 100000,    # Highly cited journals
        'min_h_index': 100,        # High H-index
        'min_similarity': 0.8      # High semantic similarity
    }
)
```

## 📊 Database Statistics

### get_database_stats()

**Get statistics about the loaded journal database.**

```python
def get_database_stats(self) -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Dictionary with database statistics

**Example:**
```python
# Get comprehensive database statistics
stats = matcher.get_database_stats()

print("📊 Database Statistics:")
print(f"  Total journals: {stats['total_journals']:,}")
print(f"  Embedding dimension: {stats['embedding_dimension']}")
print(f"  FAISS index type: {stats['faiss_index_type']}")
print(f"  Index size: {stats['index_size']:,} vectors")
print()

print("🌐 Journal Distribution:")
print(f"  Open access journals: {stats['open_access_journals']:,}")
print(f"  DOAJ journals: {stats['doaj_journals']:,}")
print(f"  Journals with APC info: {stats['journals_with_apc']:,}")
print(f"  Free to publish: {stats['free_to_publish_journals']:,}")
print(f"  Average APC: ${stats['average_apc']:,.2f}")
print()

print(f"📄 Sample journal: {stats['sample_journal']}")
```

## 🔍 Result Processing

### format_search_results()

**Format search results for display or export.**

```python
def format_search_results(
    results: List[Dict[str, Any]], 
    include_embeddings: bool = False
) -> List[Dict[str, Any]]
```

**Parameters:**
- `results` (List[Dict]): Raw search results
- `include_embeddings` (bool): Whether to include embedding vectors in output

**Returns:**
- `List[Dict[str, Any]]`: Formatted results ready for display

**Example:**
```python
# Get search results
results = matcher.search_similar_journals(abstract, top_k=5)

# Format for display
formatted = format_search_results(results)

for journal in formatted:
    print(f"🏆 Rank #{journal['rank']}: {journal['journal_name']}")
    print(f"📊 Similarity: {journal['similarity_score']:.3f}")
    print(f"🏢 Publisher: {journal['publisher']}")
    print(f"🌐 Open Access: {'✅' if journal['is_open_access'] else '❌'}")
    print(f"💰 APC: {journal['apc_display'] or 'Not specified'}")
    print(f"📚 Works: {journal['works_count']:,}")
    print(f"📈 Citations: {journal['cited_by_count']:,}")
    print(f"🎯 H-index: {journal['h_index']}")
    print(f"🏷️ Subjects: {', '.join(journal['subjects'][:3])}")
    if journal['homepage_url']:
        print(f"🔗 Homepage: {journal['homepage_url']}")
    print("-" * 50)
```

### rank_and_filter_results()

**Apply additional ranking and filtering to results.**

```python
def rank_and_filter_results(
    results: List[Dict[str, Any]], 
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

**Parameters:**
- `results` (List[Dict]): List of journal results with similarity scores
- `filters` (Dict, optional): Optional filter criteria

**Returns:**
- `List[Dict[str, Any]]`: Filtered and ranked results

**Example:**
```python
# Get initial results without filters
initial_results = matcher.search_similar_journals(
    query_text=abstract,
    top_k=50,  # Get more results initially
    min_similarity=0.3
)

# Apply post-processing filters
filtered_results = rank_and_filter_results(
    initial_results,
    filters={
        'open_access_only': True,
        'max_apc': 2000,
        'min_h_index': 25
    }
)

print(f"📊 Initial results: {len(initial_results)}")
print(f"🎯 After filtering: {len(filtered_results)}")
```

## 🏗️ Utility Functions

### create_faiss_index()

**Create FAISS index from embeddings array.**

```python
def create_faiss_index(embeddings: np.ndarray) -> faiss.Index
```

**Parameters:**
- `embeddings` (np.ndarray): 2D numpy array of embeddings

**Returns:**
- `faiss.Index`: FAISS index ready for similarity search

**Example:**
```python
import numpy as np

# Create sample embeddings
sample_embeddings = np.random.rand(1000, 384).astype(np.float32)

# Create FAISS index
index = create_faiss_index(sample_embeddings)

print(f"📊 Created index with {index.ntotal} vectors")
print(f"📐 Index dimension: {index.d}")
print(f"🔍 Index type: {type(index).__name__}")
```

### load_journal_database_with_index()

**Load journal metadata and FAISS index from disk.**

```python
def load_journal_database_with_index() -> Tuple[List[Dict[str, Any]], faiss.Index]
```

**Returns:**
- `Tuple`: (journals, faiss_index)

**Example:**
```python
# Load database and index directly
journals, index = load_journal_database_with_index()

print(f"📄 Loaded {len(journals)} journal records")
print(f"📊 Index contains {index.ntotal} vectors")

# Use for custom search
from src.embedder import embed_text

query_embedding = embed_text("cardiovascular disease research")
similarities, indices = index.search(
    query_embedding.reshape(1, -1).astype(np.float32), 
    10
)

for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
    if idx < len(journals):
        journal = journals[idx]
        print(f"{i+1}. {journal['display_name']}: {similarity:.3f}")
```

## 🚨 Error Handling

### MatchingError

Custom exception for journal matching errors:

```python
from src.match_journals import MatchingError

try:
    # This will fail - empty query
    results = matcher.search_similar_journals("")
except MatchingError as e:
    print(f"❌ Matching error: {e}")
    # Output: Query text cannot be empty

try:
    # Database not loaded
    unloaded_matcher = JournalMatcher()
    results = unloaded_matcher.search_similar_journals("test query")
except MatchingError as e:
    print(f"❌ Database error: {e}")
    # Output: Database loading failed: ...
```

### Robust Search Pattern

```python
def robust_journal_search(query_text, max_retries=3):
    """Robust journal search with fallbacks."""
    
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty")
    
    matcher = JournalMatcher()
    
    # Attempt with progressively relaxed criteria
    search_configs = [
        # First try: strict criteria
        {
            'top_k': 10,
            'min_similarity': 0.7,
            'filters': {'open_access_only': True, 'min_h_index': 50}
        },
        # Second try: moderate criteria
        {
            'top_k': 15,
            'min_similarity': 0.5,
            'filters': {'open_access_only': True}
        },
        # Final try: minimal criteria
        {
            'top_k': 20,
            'min_similarity': 0.3,
            'filters': None
        }
    ]
    
    last_error = None
    
    for attempt, config in enumerate(search_configs, 1):
        try:
            print(f"🔍 Search attempt {attempt}/{len(search_configs)}")
            
            matcher.load_database()
            results = matcher.search_similar_journals(
                query_text=query_text,
                **config
            )
            
            if results:
                print(f"✅ Found {len(results)} results on attempt {attempt}")
                return results
            else:
                print(f"⚠️ No results found on attempt {attempt}")
                
        except MatchingError as e:
            last_error = e
            print(f"❌ Attempt {attempt} failed: {e}")
            
            if attempt < len(search_configs):
                print("🔄 Trying with relaxed criteria...")
    
    # All attempts failed
    if last_error:
        raise MatchingError(f"All search attempts failed. Last error: {last_error}")
    else:
        return []  # No results found but no errors
```

## 📈 Performance Optimization

### Batch Processing

```python
def batch_journal_matching(abstracts, batch_size=50):
    """Process multiple abstracts efficiently."""
    
    matcher = JournalMatcher()
    matcher.load_database()  # Load once
    
    all_results = []
    
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i + batch_size]
        batch_results = []
        
        print(f"🔄 Processing batch {i//batch_size + 1}/{(len(abstracts)-1)//batch_size + 1}")
        
        for abstract in batch:
            try:
                results = matcher.search_similar_journals(
                    query_text=abstract,
                    top_k=5,
                    min_similarity=0.6
                )
                batch_results.append({
                    'abstract': abstract[:100] + "...",
                    'results': results,
                    'status': 'success'
                })
            except MatchingError as e:
                batch_results.append({
                    'abstract': abstract[:100] + "...",
                    'error': str(e),
                    'status': 'failed'
                })
        
        all_results.extend(batch_results)
    
    successful = sum(1 for r in all_results if r['status'] == 'success')
    print(f"✅ Processed {successful}/{len(abstracts)} abstracts successfully")
    
    return all_results
```

### Memory Management

```python
def memory_efficient_search(query_text, large_journal_db=True):
    """Memory-efficient search for large databases."""
    
    matcher = JournalMatcher()
    
    if large_journal_db:
        # Load database in chunks if memory is limited
        print("🧠 Using memory-efficient loading...")
        matcher.load_database()
        
        # Clear embedding cache periodically
        import gc
        gc.collect()
    
    # Use smaller top_k for initial search, then expand if needed
    results = matcher.search_similar_journals(
        query_text=query_text,
        top_k=50,  # Get more candidates initially
        min_similarity=0.3
    )
    
    # Apply post-processing to get final results
    final_results = rank_and_filter_results(
        results,
        filters={'min_similarity': 0.6}
    )[:10]  # Top 10 after filtering
    
    return final_results
```

## 🎯 Best Practices

### 1. Query Optimization

```python
def optimize_query_text(abstract):
    """Optimize abstract text for better matching."""
    
    if not abstract:
        return ""
    
    # Remove common academic boilerplate
    boilerplate_phrases = [
        "in this study", "in this paper", "this research",
        "we present", "we propose", "we demonstrate",
        "our results show", "our findings indicate"
    ]
    
    optimized = abstract.lower()
    for phrase in boilerplate_phrases:
        optimized = optimized.replace(phrase, "")
    
    # Focus on key content (first 500 characters are often most informative)
    key_content = optimized[:500].strip()
    
    return key_content

# Usage
raw_abstract = "In this study, we present a novel approach..."
optimized_abstract = optimize_query_text(raw_abstract)

results = matcher.search_similar_journals(
    query_text=optimized_abstract,
    top_k=10
)
```

### 2. Progressive Filtering

```python
def progressive_journal_search(query_text):
    """Progressive search with increasingly specific filters."""
    
    # Start with broad search
    broad_results = matcher.search_similar_journals(
        query_text=query_text,
        top_k=100,
        min_similarity=0.3
    )
    
    print(f"📊 Broad search: {len(broad_results)} results")
    
    # Apply quality filters progressively
    quality_levels = [
        {'min_h_index': 100, 'min_citations': 50000, 'label': 'Top-tier'},
        {'min_h_index': 50, 'min_citations': 20000, 'label': 'High-quality'},
        {'min_h_index': 25, 'min_citations': 10000, 'label': 'Good-quality'},
        {'min_h_index': 10, 'min_citations': 5000, 'label': 'Standard'},
    ]
    
    recommendations = {}
    
    for level in quality_levels:
        filtered = rank_and_filter_results(
            broad_results,
            filters={
                'min_h_index': level['min_h_index'],
                'min_citations': level['min_citations'],
                'min_similarity': 0.6
            }
        )
        
        if filtered:
            recommendations[level['label']] = filtered[:5]
            print(f"🏆 {level['label']} journals: {len(filtered)} found")
    
    return recommendations
```

### 3. Multi-Criteria Ranking

```python
def multi_criteria_ranking(results, weights=None):
    """Rank journals using multiple criteria."""
    
    if weights is None:
        weights = {
            'similarity': 0.4,    # 40% semantic similarity
            'h_index': 0.3,       # 30% journal impact
            'openness': 0.2,      # 20% open access preference
            'cost': 0.1           # 10% cost consideration
        }
    
    scored_results = []
    
    for journal in results:
        # Normalize scores to 0-1 range
        similarity_score = journal.get('similarity_score', 0)
        h_index_score = min(journal.get('h_index', 0) / 200, 1.0)  # Cap at 200
        
        # Open access bonus
        openness_score = 1.0 if journal.get('is_open_access') else 0.5
        
        # Cost score (lower cost = higher score)
        apc = journal.get('apc_amount', 0)
        if apc == 0:
            cost_score = 1.0  # Free is best
        else:
            cost_score = max(0, 1.0 - (apc / 5000))  # Normalize to $5000 max
        
        # Calculate composite score
        composite_score = (
            weights['similarity'] * similarity_score +
            weights['h_index'] * h_index_score +
            weights['openness'] * openness_score +
            weights['cost'] * cost_score
        )
        
        journal['composite_score'] = composite_score
        scored_results.append(journal)
    
    # Sort by composite score
    scored_results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return scored_results
```

---

*For complete integration examples and workflows, see the [User Guide](../user/) and [API Examples](../examples/).*