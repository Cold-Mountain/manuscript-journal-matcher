# Chunked Processing Architecture Design

**Document Version**: 1.0  
**Date**: July 28, 2025  
**Purpose**: Technical architecture for processing 7,678 journals efficiently using chunked processing

## Overview

Processing 7,678 medical journals requires careful memory management and error handling. This document outlines the chunked processing architecture designed to handle large-scale journal data processing while maintaining system stability and recovery capabilities.

## Architecture Components

### System Requirements
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space for database and embeddings
- **Processing Time**: 8-12 hours for full pipeline with DOAJ enrichment
- **Network**: Stable internet connection for DOAJ API calls

### Processing Pipeline Overview
```
CSV File (7,678 journals)
    ↓
Chunk Split (500-1000 journals per chunk)
    ↓
For Each Chunk:
    ├─ Parse & Validate
    ├─ Map to Schema
    ├─ DOAJ Enrichment (optional)
    ├─ Create Semantic Fingerprints
    ├─ Generate Embeddings
    └─ Save Progress
    ↓
Combine Results
    ↓
Build FAISS Index
    ↓
Save Final Database
```

## Chunk Processing Strategy

### Chunk Size Determination

#### Optimal Chunk Sizes by Processing Stage

| Processing Stage | Recommended Chunk Size | Memory Usage | Processing Time |
|------------------|------------------------|--------------|-----------------|
| CSV Parsing | 2000-5000 journals | ~100MB | 30 seconds |
| Schema Mapping | 1000-2000 journals | ~200MB | 1-2 minutes |
| DOAJ Enrichment | 50-100 journals | ~50MB | 10-30 minutes |
| Embedding Generation | 32-64 journals | ~500MB | 5-15 minutes |
| FAISS Index Building | All journals | ~1GB | 2-5 minutes |

#### Dynamic Chunk Sizing Logic
```python
def calculate_optimal_chunk_size(total_journals: int, available_memory_gb: int, 
                                include_doaj: bool = True) -> Dict[str, int]:
    """Calculate optimal chunk sizes based on system resources."""
    
    # Base chunk sizes
    base_sizes = {
        'parsing': 2000,
        'mapping': 1000,
        'doaj': 100 if include_doaj else 0,
        'embedding': 64
    }
    
    # Adjust based on available memory
    memory_multiplier = min(available_memory_gb / 8.0, 2.0)  # Cap at 2x for 16GB+
    
    optimized_sizes = {}
    for stage, base_size in base_sizes.items():
        if stage == 'doaj' and not include_doaj:
            optimized_sizes[stage] = 0
        else:
            optimized_sizes[stage] = int(base_size * memory_multiplier)
    
    # Ensure chunks don't exceed total journals
    for stage in optimized_sizes:
        if optimized_sizes[stage] > total_journals:
            optimized_sizes[stage] = total_journals
    
    return optimized_sizes
```

### Memory Management

#### Memory Monitoring
```python
import psutil
import gc
from typing import Dict, Any

class MemoryManager:
    """Monitor and manage memory usage during chunked processing."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        """
        Initialize memory manager.
        
        Args:
            warning_threshold: Memory usage percentage to trigger warning
            critical_threshold: Memory usage percentage to force cleanup
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percentage': memory.percent / 100.0
        }
    
    def check_memory_status(self) -> str:
        """Check current memory status and return status level."""
        usage = self.get_memory_usage()
        
        if usage['percentage'] >= self.critical_threshold:
            return 'CRITICAL'
        elif usage['percentage'] >= self.warning_threshold:
            return 'WARNING'
        else:
            return 'OK'
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        # Clear any large variables that might be lingering
        gc.collect()
        
        # Force garbage collection multiple times for stubborn objects
        for _ in range(3):
            gc.collect()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        current = self.get_memory_usage()
        
        return {
            'current_usage': current,
            'memory_increase_gb': current['used_gb'] - self.initial_memory['used_gb'],
            'status': self.check_memory_status(),
            'recommendation': self._get_recommendation(current)
        }
    
    def _get_recommendation(self, usage: Dict[str, float]) -> str:
        """Get memory management recommendation."""
        if usage['percentage'] >= self.critical_threshold:
            return "URGENT: Reduce chunk size or restart process"
        elif usage['percentage'] >= self.warning_threshold:
            return "WARNING: Consider reducing chunk size"
        elif usage['available_gb'] < 2.0:
            return "CAUTION: Low available memory"
        else:
            return "OK: Memory usage is normal"
```

#### Memory-Efficient Processing
```python
def process_chunk_with_memory_management(chunk: pd.DataFrame, 
                                       memory_manager: MemoryManager) -> List[Dict[str, Any]]:
    """Process a chunk with active memory management."""
    
    # Check memory before processing
    memory_status = memory_manager.check_memory_status()
    if memory_status == 'CRITICAL':
        memory_manager.force_cleanup()
        
        # Recheck after cleanup
        if memory_manager.check_memory_status() == 'CRITICAL':
            raise MemoryError("Insufficient memory to process chunk")
    
    try:
        # Process chunk
        processed_journals = []
        
        for idx, row in chunk.iterrows():
            # Process individual journal
            journal = process_single_journal(row)
            if journal:
                processed_journals.append(journal)
            
            # Periodic memory check within chunk
            if idx % 100 == 0:  # Check every 100 journals
                if memory_manager.check_memory_status() == 'CRITICAL':
                    memory_manager.force_cleanup()
        
        return processed_journals
        
    finally:
        # Always cleanup after chunk processing
        memory_manager.force_cleanup()
```

## Progress Tracking and Resume Capability

### Progress State Management
```python
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

class ProgressTracker:
    """Track and persist processing progress for resume capability."""
    
    def __init__(self, progress_file: str = "csv_processing_progress.json"):
        """Initialize progress tracker."""
        self.progress_file = Path(progress_file)
        self.state = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load existing progress or create new state."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupted progress file, start fresh
                pass
        
        return {
            'session_id': self._generate_session_id(),
            'started_at': datetime.now().isoformat(),
            'total_journals': 0,
            'processed_chunks': [],
            'failed_chunks': [],
            'current_stage': 'initialization',
            'stage_progress': {},
            'errors': [],
            'warnings': []
        }
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def save_progress(self):
        """Persist current progress to disk."""
        self.state['last_updated'] = datetime.now().isoformat()
        
        with open(self.progress_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def start_stage(self, stage_name: str, total_items: int):
        """Start a new processing stage."""
        self.state['current_stage'] = stage_name
        self.state['stage_progress'][stage_name] = {
            'started_at': datetime.now().isoformat(),
            'total_items': total_items,
            'completed_items': 0,
            'failed_items': 0,
            'status': 'in_progress'
        }
        self.save_progress()
    
    def update_stage_progress(self, stage_name: str, completed: int, failed: int = 0):
        """Update progress for current stage."""
        if stage_name in self.state['stage_progress']:
            self.state['stage_progress'][stage_name]['completed_items'] = completed
            self.state['stage_progress'][stage_name]['failed_items'] = failed
            self.save_progress()
    
    def complete_stage(self, stage_name: str, success: bool = True):
        """Mark stage as completed."""
        if stage_name in self.state['stage_progress']:
            self.state['stage_progress'][stage_name]['status'] = 'completed' if success else 'failed'
            self.state['stage_progress'][stage_name]['completed_at'] = datetime.now().isoformat()
            self.save_progress()
    
    def add_processed_chunk(self, chunk_id: str, chunk_info: Dict[str, Any]):
        """Record successful chunk processing."""
        chunk_record = {
            'chunk_id': chunk_id,
            'processed_at': datetime.now().isoformat(),
            **chunk_info
        }
        self.state['processed_chunks'].append(chunk_record)
        self.save_progress()
    
    def add_failed_chunk(self, chunk_id: str, error: str, chunk_info: Dict[str, Any]):
        """Record failed chunk processing."""
        failure_record = {
            'chunk_id': chunk_id,
            'failed_at': datetime.now().isoformat(),
            'error': error,
            **chunk_info
        }
        self.state['failed_chunks'].append(failure_record)
        self.save_progress()
    
    def add_error(self, error: str, context: Dict[str, Any] = None):
        """Add error to error log."""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'context': context or {}
        }
        self.state['errors'].append(error_record)
        self.save_progress()
    
    def add_warning(self, warning: str, context: Dict[str, Any] = None):
        """Add warning to warning log."""
        warning_record = {
            'timestamp': datetime.now().isoformat(),
            'warning': warning,
            'context': context or {}
        }
        self.state['warnings'].append(warning_record)
        self.save_progress()
    
    def get_resume_point(self) -> Dict[str, Any]:
        """Get information needed to resume processing."""
        return {
            'session_id': self.state['session_id'],
            'processed_chunks': len(self.state['processed_chunks']),
            'failed_chunks': len(self.state['failed_chunks']),
            'current_stage': self.state['current_stage'],
            'can_resume': len(self.state['processed_chunks']) > 0,
            'completion_percentage': self._calculate_completion_percentage()
        }
    
    def _calculate_completion_percentage(self) -> float:
        """Calculate overall completion percentage."""
        if self.state['total_journals'] == 0:
            return 0.0
        
        completed_journals = sum(
            chunk.get('journal_count', 0) 
            for chunk in self.state['processed_chunks']
        )
        
        return (completed_journals / self.state['total_journals']) * 100.0
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get comprehensive progress report."""
        return {
            'session_info': {
                'session_id': self.state['session_id'],
                'started_at': self.state['started_at'],
                'duration': self._calculate_duration()
            },
            'overall_progress': {
                'total_journals': self.state['total_journals'],
                'completion_percentage': self._calculate_completion_percentage(),
                'processed_chunks': len(self.state['processed_chunks']),
                'failed_chunks': len(self.state['failed_chunks'])
            },
            'stage_progress': self.state['stage_progress'],
            'current_stage': self.state['current_stage'],
            'errors': len(self.state['errors']),
            'warnings': len(self.state['warnings']),
            'can_resume': self.get_resume_point()['can_resume']
        }
    
    def _calculate_duration(self) -> str:
        """Calculate processing duration."""
        start = datetime.fromisoformat(self.state['started_at'])
        now = datetime.now()
        duration = now - start
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
```

### Resume Functionality
```python
def resume_processing(csv_file: str, progress_tracker: ProgressTracker) -> bool:
    """Resume processing from previous session."""
    
    resume_info = progress_tracker.get_resume_point()
    
    if not resume_info['can_resume']:
        logger.info("No previous session found, starting fresh processing")
        return False
    
    logger.info(f"Resuming session {resume_info['session_id']}")
    logger.info(f"Previous progress: {resume_info['completion_percentage']:.1f}%")
    logger.info(f"Processed chunks: {resume_info['processed_chunks']}")
    logger.info(f"Failed chunks: {resume_info['failed_chunks']}")
    
    # Load CSV and determine remaining chunks
    importer = CSVJournalImporter(csv_file)
    importer.load_csv()
    importer.clean_and_validate()
    
    all_chunks = importer.get_processed_chunks()
    processed_chunk_ids = {chunk['chunk_id'] for chunk in progress_tracker.state['processed_chunks']}
    
    # Find unprocessed chunks
    remaining_chunks = []
    for i, chunk in enumerate(all_chunks):
        chunk_id = f"chunk_{i:04d}"
        if chunk_id not in processed_chunk_ids:
            remaining_chunks.append((chunk_id, chunk))
    
    if not remaining_chunks:
        logger.info("All chunks already processed, moving to final assembly")
        return True
    
    logger.info(f"Resuming with {len(remaining_chunks)} remaining chunks")
    
    # Continue processing remaining chunks
    for chunk_id, chunk in remaining_chunks:
        try:
            process_single_chunk(chunk_id, chunk, progress_tracker)
        except Exception as e:
            progress_tracker.add_failed_chunk(chunk_id, str(e), {'chunk_size': len(chunk)})
            logger.error(f"Failed to process {chunk_id}: {e}")
    
    return True
```

## Error Handling and Recovery

### Comprehensive Error Handling Strategy
```python
class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass

class ChunkProcessingError(ProcessingError):
    """Error during chunk processing."""
    pass

class DOAJEnrichmentError(ProcessingError):
    """Error during DOAJ enrichment."""
    pass

class EmbeddingGenerationError(ProcessingError):
    """Error during embedding generation."""
    pass

def process_chunk_with_error_handling(chunk_id: str, chunk: pd.DataFrame, 
                                     progress_tracker: ProgressTracker) -> Dict[str, Any]:
    """Process chunk with comprehensive error handling."""
    
    logger.info(f"Processing {chunk_id} with {len(chunk)} journals")
    
    try:
        # Stage 1: Schema Mapping
        progress_tracker.start_stage(f"{chunk_id}_mapping", len(chunk))
        mapped_journals = []
        
        for idx, row in chunk.iterrows():
            try:
                journal = map_csv_row_to_schema(row)
                if journal:
                    mapped_journals.append(journal)
            except Exception as e:
                error_msg = f"Failed to map journal {row.get('Title', 'Unknown')}: {e}"
                progress_tracker.add_warning(error_msg, {'row_index': idx})
                logger.warning(error_msg)
            
            progress_tracker.update_stage_progress(f"{chunk_id}_mapping", idx + 1)
        
        progress_tracker.complete_stage(f"{chunk_id}_mapping")
        
        if not mapped_journals:
            raise ChunkProcessingError(f"No journals successfully mapped in {chunk_id}")
        
        # Stage 2: DOAJ Enrichment (if enabled)
        if should_enrich_with_doaj():
            progress_tracker.start_stage(f"{chunk_id}_doaj", len(mapped_journals))
            
            try:
                enriched_journals = enrich_chunk_with_doaj(mapped_journals, progress_tracker)
                mapped_journals = enriched_journals
                progress_tracker.complete_stage(f"{chunk_id}_doaj")
            except Exception as e:
                error_msg = f"DOAJ enrichment failed for {chunk_id}: {e}"
                progress_tracker.add_warning(error_msg)
                logger.warning(f"{error_msg} - continuing without DOAJ data")
                progress_tracker.complete_stage(f"{chunk_id}_doaj", success=False)
        
        # Stage 3: Semantic Fingerprints
        progress_tracker.start_stage(f"{chunk_id}_fingerprints", len(mapped_journals))
        
        for i, journal in enumerate(mapped_journals):
            try:
                journal['semantic_fingerprint'] = create_semantic_fingerprint(journal)
            except Exception as e:
                error_msg = f"Failed to create fingerprint for {journal.get('display_name', 'Unknown')}: {e}"
                progress_tracker.add_warning(error_msg)
                # Use fallback fingerprint
                journal['semantic_fingerprint'] = create_fallback_fingerprint(journal)
            
            progress_tracker.update_stage_progress(f"{chunk_id}_fingerprints", i + 1)
        
        progress_tracker.complete_stage(f"{chunk_id}_fingerprints")
        
        # Stage 4: Embedding Generation
        progress_tracker.start_stage(f"{chunk_id}_embeddings", len(mapped_journals))
        
        try:
            journals_with_embeddings, embeddings = generate_embeddings_for_chunk(mapped_journals)
            progress_tracker.complete_stage(f"{chunk_id}_embeddings")
        except Exception as e:
            raise EmbeddingGenerationError(f"Embedding generation failed for {chunk_id}: {e}")
        
        # Success - record chunk completion
        chunk_info = {
            'journal_count': len(journals_with_embeddings),
            'embedding_shape': embeddings.shape if embeddings is not None else None,
            'doaj_enriched': sum(1 for j in journals_with_embeddings if j.get('in_doaj')),
            'processing_time': calculate_processing_time()
        }
        
        progress_tracker.add_processed_chunk(chunk_id, chunk_info)
        
        return {
            'chunk_id': chunk_id,
            'journals': journals_with_embeddings,
            'embeddings': embeddings,
            'info': chunk_info
        }
        
    except Exception as e:
        # Record failure
        error_context = {
            'chunk_size': len(chunk),
            'error_type': type(e).__name__,
            'processing_stage': progress_tracker.state['current_stage']
        }
        
        progress_tracker.add_failed_chunk(chunk_id, str(e), error_context)
        logger.error(f"Chunk {chunk_id} processing failed: {e}")
        
        # Decide whether to continue or abort
        if isinstance(e, EmbeddingGenerationError):
            # Critical error - abort processing
            raise
        elif len(progress_tracker.state['failed_chunks']) > 10:
            # Too many failures - abort
            raise ProcessingError("Too many chunk failures, aborting processing")
        else:
            # Non-critical error - continue with next chunk
            return None
```

### Retry Logic for External API Calls
```python
import time
import random
from functools import wraps

def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic with exponential backoff."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(0.1, 0.5)
                    total_delay = delay + jitter
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.1f}s")
                    time.sleep(total_delay)
            
            # All attempts failed
            logger.error(f"All {max_retries + 1} attempts failed. Last error: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator

@retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
def fetch_doaj_data_with_retry(issn: str) -> Dict[str, Any]:
    """Fetch DOAJ data with retry logic."""
    return doaj_api.fetch_journal_by_issn(issn)
```

## Performance Optimization

### Parallel Processing Considerations
```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

class ParallelChunkProcessor:
    """Process multiple chunks in parallel while respecting resource limits."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            use_processes: Whether to use processes (True) or threads (False)
        """
        if max_workers is None:
            # Conservative default based on CPU count and memory
            cpu_count = mp.cpu_count()
            max_workers = min(cpu_count // 2, 4)  # Use half CPUs, max 4
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    def process_chunks_parallel(self, chunks: List[Tuple[str, pd.DataFrame]], 
                               progress_tracker: ProgressTracker) -> List[Dict[str, Any]]:
        """Process chunks in parallel."""
        
        logger.info(f"Processing {len(chunks)} chunks with {self.max_workers} workers")
        
        results = []
        failed_chunks = []
        
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(process_chunk_with_error_handling, chunk_id, chunk, progress_tracker): 
                (chunk_id, chunk)
                for chunk_id, chunk in chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_id, chunk = future_to_chunk[future]
                
                try:
                    result = future.result()
                    if result:  # Successfully processed
                        results.append(result)
                        logger.info(f"✅ Completed {chunk_id}")
                    else:  # Failed but handled
                        failed_chunks.append(chunk_id)
                        logger.warning(f"⚠️ Failed {chunk_id}")
                        
                except Exception as e:
                    # Unhandled exception
                    failed_chunks.append(chunk_id)
                    progress_tracker.add_error(f"Unhandled error in {chunk_id}: {e}")
                    logger.error(f"❌ Error in {chunk_id}: {e}")
        
        logger.info(f"Parallel processing complete: {len(results)} successful, {len(failed_chunks)} failed")
        
        return results
```

### I/O Optimization
```python
class OptimizedFileOperations:
    """Optimized file operations for large-scale processing."""
    
    @staticmethod
    def save_chunk_results(chunk_results: List[Dict[str, Any]], 
                          temp_dir: Path = Path("temp_chunks")):
        """Save chunk results to temporary files for memory efficiency."""
        temp_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        for result in chunk_results:
            chunk_id = result['chunk_id']
            
            # Save journals metadata
            journals_file = temp_dir / f"{chunk_id}_journals.json"
            with open(journals_file, 'w') as f:
                json.dump(result['journals'], f, separators=(',', ':'))  # Compact JSON
            
            # Save embeddings separately (more efficient)
            embeddings_file = temp_dir / f"{chunk_id}_embeddings.npy"
            np.save(embeddings_file, result['embeddings'])
            
            saved_files.append({
                'chunk_id': chunk_id,
                'journals_file': str(journals_file),
                'embeddings_file': str(embeddings_file),
                'journal_count': result['info']['journal_count']
            })
        
        return saved_files
    
    @staticmethod
    def load_and_combine_chunks(saved_files: List[Dict[str, Any]]) -> Tuple[List[Dict], np.ndarray]:
        """Load and combine chunk results from temporary files."""
        all_journals = []
        all_embeddings = []
        
        for file_info in saved_files:
            # Load journals
            with open(file_info['journals_file'], 'r') as f:
                chunk_journals = json.load(f)
                all_journals.extend(chunk_journals)
            
            # Load embeddings
            chunk_embeddings = np.load(file_info['embeddings_file'])
            all_embeddings.append(chunk_embeddings)
        
        # Combine embeddings
        combined_embeddings = np.vstack(all_embeddings) if all_embeddings else None
        
        return all_journals, combined_embeddings
    
    @staticmethod
    def cleanup_temp_files(saved_files: List[Dict[str, Any]]):
        """Clean up temporary chunk files."""
        for file_info in saved_files:
            try:
                Path(file_info['journals_file']).unlink(missing_ok=True)
                Path(file_info['embeddings_file']).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_info['chunk_id']}: {e}")
```

## Complete Processing Pipeline

### Main Processing Function
```python
def process_csv_with_chunking(csv_file: str, 
                             chunk_size: int = 500,
                             include_doaj: bool = True,
                             parallel_processing: bool = False,
                             resume: bool = True) -> Dict[str, Any]:
    """
    Complete CSV processing pipeline with chunking.
    
    Args:
        csv_file: Path to Medicine Journal Rankings CSV
        chunk_size: Number of journals per chunk
        include_doaj: Whether to include DOAJ enrichment
        parallel_processing: Whether to process chunks in parallel
        resume: Whether to resume from previous session
        
    Returns:
        Processing results and statistics
    """
    
    # Initialize components
    progress_tracker = ProgressTracker()
    memory_manager = MemoryManager()
    
    # Check for resume
    if resume and progress_tracker.get_resume_point()['can_resume']:
        logger.info("Attempting to resume previous session...")
        if resume_processing(csv_file, progress_tracker):
            logger.info("Successfully resumed processing")
        else:
            logger.info("Resume failed, starting fresh")
    
    try:
        # Stage 1: CSV Loading and Preparation
        logger.info("🔄 Stage 1: Loading and preparing CSV data...")
        progress_tracker.start_stage('csv_loading', 1)
        
        importer = CSVJournalImporter(csv_file)
        importer.load_csv()
        importer.clean_and_validate()
        
        stats = importer.get_statistics()
        progress_tracker.state['total_journals'] = stats['total_journals']
        
        logger.info(f"Loaded {stats['total_journals']} journals for processing")
        progress_tracker.complete_stage('csv_loading')
        
        # Stage 2: Chunk Creation
        logger.info("🔄 Stage 2: Creating processing chunks...")
        chunks = importer.get_processed_chunks(chunk_size)
        
        # Create chunk identifiers
        chunk_list = [(f"chunk_{i:04d}", chunk) for i, chunk in enumerate(chunks)]
        
        logger.info(f"Created {len(chunk_list)} chunks of ~{chunk_size} journals each")
        
        # Stage 3: Chunk Processing
        logger.info("🔄 Stage 3: Processing chunks...")
        progress_tracker.start_stage('chunk_processing', len(chunk_list))
        
        if parallel_processing and len(chunk_list) > 1:
            processor = ParallelChunkProcessor(max_workers=2)  # Conservative for memory
            chunk_results = processor.process_chunks_parallel(chunk_list, progress_tracker)
        else:
            chunk_results = []
            for chunk_id, chunk in chunk_list:
                result = process_chunk_with_error_handling(chunk_id, chunk, progress_tracker)
                if result:
                    chunk_results.append(result)
        
        progress_tracker.complete_stage('chunk_processing')
        
        if not chunk_results:
            raise ProcessingError("No chunks processed successfully")
        
        # Stage 4: Memory-Efficient Combination
        logger.info("🔄 Stage 4: Combining chunk results...")
        progress_tracker.start_stage('combination', 1)
        
        # Save chunk results to temporary files
        temp_files = OptimizedFileOperations.save_chunk_results(chunk_results)
        
        # Clear chunk results from memory  
        del chunk_results
        memory_manager.force_cleanup()
        
        # Load and combine from files
        all_journals, combined_embeddings = OptimizedFileOperations.load_and_combine_chunks(temp_files)
        
        progress_tracker.complete_stage('combination')
        
        # Stage 5: Final Database Creation
        logger.info("🔄 Stage 5: Creating final database...")
        progress_tracker.start_stage('database_creation', 1)
        
        # Build FAISS index
        faiss_index = create_faiss_index(combined_embeddings)
        
        # Save final database
        save_journal_database(all_journals, combined_embeddings)
        
        progress_tracker.complete_stage('database_creation')
        
        # Cleanup temporary files
        OptimizedFileOperations.cleanup_temp_files(temp_files)
        
        # Generate final report
        final_report = {
            'success': True,
            'total_journals': len(all_journals),
            'embedding_shape': combined_embeddings.shape,
            'doaj_enriched': sum(1 for j in all_journals if j.get('in_doaj')),
            'processing_time': progress_tracker._calculate_duration(),
            'progress_report': progress_tracker.get_progress_report(),
            'memory_report': memory_manager.get_memory_report()
        }
        
        logger.info("✅ CSV processing completed successfully!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   - Total journals: {final_report['total_journals']}")
        logger.info(f"   - DOAJ journals: {final_report['doaj_enriched']}")
        logger.info(f"   - Processing time: {final_report['processing_time']}")
        
        return final_report
        
    except Exception as e:
        logger.error(f"❌ CSV processing failed: {e}")
        progress_tracker.add_error(f"Processing failed: {e}")
        
        return {
            'success': False,
            'error': str(e),
            'progress_report': progress_tracker.get_progress_report(),
            'partial_results': progress_tracker.state.get('processed_chunks', [])
        }
    
    finally:
        # Always save final progress
        progress_tracker.save_progress()
```

This chunked processing architecture ensures efficient, reliable processing of the 7,678 medical journals while providing comprehensive error handling, memory management, and resume capabilities.