#!/usr/bin/env python3
"""
Script to build the initial journal database from OpenAlex.

Usage:
    python scripts/build_database.py [--limit 1000] [--resume] [--test]
    
This script fetches journal data from OpenAlex, creates semantic fingerprints,
generates embeddings, and saves the database for journal matching.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from journal_db_builder import (
        OpenAlexAPI,
        DOAJAPI,
        create_semantic_fingerprint,
        build_journal_embeddings,
        save_journal_database,
        load_journal_database,
        JournalDatabaseError
    )
    from config import JOURNAL_METADATA_PATH, API_CACHE_DIR
    from embedder import get_embedding_info
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'database_build.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build journal database from OpenAlex or CSV')
    parser.add_argument(
        '--limit', 
        type=int, 
        default=1000,
        help='Number of journals to fetch (default: 1000)'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume from existing database'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Test mode - fetch only 10 journals'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.1,
        help='Rate limit between API requests in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--skip-doaj',
        action='store_true',
        help='Skip DOAJ data enrichment (faster but less complete)'
    )
    parser.add_argument(
        '--doaj-rate-limit',
        type=float,
        default=1.0,
        help='Rate limit for DOAJ API requests in seconds (default: 1.0)'
    )
    # CSV-specific arguments
    parser.add_argument(
        '--csv-file',
        type=str,
        help='Path to CSV file for journal data (alternative to OpenAlex)'
    )
    parser.add_argument(
        '--csv-chunk-size',
        type=int,
        default=500,
        help='Number of CSV journals to process per chunk (default: 500)'
    )
    parser.add_argument(
        '--csv-only',
        action='store_true',
        help='Use only CSV data, skip OpenAlex API calls'
    )
    parser.add_argument(
        '--quality-filter',
        action='store_true',
        help='Apply high-quality journal filtering for CSV data'
    )
    parser.add_argument(
        '--min-h-index',
        type=int,
        default=10,
        help='Minimum H-index for quality filtering (default: 10)'
    )
    parser.add_argument(
        '--min-works',
        type=int,
        default=20,
        help='Minimum number of works for quality filtering (default: 20)'
    )
    parser.add_argument(
        '--max-rank',
        type=int,
        help='Maximum Scimago rank for filtering (e.g., 2000 for top 2000)'
    )
    parser.add_argument(
        '--allowed-quartiles',
        type=str,
        nargs='+',
        help='Allowed quartiles for filtering (e.g., Q1 Q2)'
    )
    return parser.parse_args()


def check_prerequisites():
    """Check that all prerequisites are met."""
    logger.info("Checking prerequisites...")
    
    # Check embedding model
    model_info = get_embedding_info()
    if not model_info.get('loaded', False):
        logger.error(f"Embedding model failed to load: {model_info.get('error', 'Unknown error')}")
        return False
    
    logger.info(f"✅ Embedding model loaded: {model_info['model_name']} ({model_info['dimension']}D)")
    
    # Check API cache directory
    API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ API cache directory: {API_CACHE_DIR}")
    
    return True


def build_database(limit: int, resume: bool = False, batch_size: int = 32, 
                  rate_limit: float = 0.1, skip_doaj: bool = False, doaj_rate_limit: float = 1.0):
    """
    Build the journal database with DOAJ enrichment.
    
    Args:
        limit: Number of journals to fetch
        resume: Whether to resume from existing database
        batch_size: Batch size for embedding generation
        rate_limit: Rate limit between OpenAlex API requests
        skip_doaj: Whether to skip DOAJ data enrichment
        doaj_rate_limit: Rate limit between DOAJ API requests
    """
    start_time = time.time()
    
    # Initialize API client
    logger.info("Initializing OpenAlex API client...")
    api = OpenAlexAPI(rate_limit=rate_limit)
    
    existing_journals = []
    if resume and JOURNAL_METADATA_PATH.exists():
        logger.info("Resuming from existing database...")
        try:
            existing_journals, _ = load_journal_database()
            logger.info(f"Loaded {len(existing_journals)} existing journals")
        except Exception as e:
            logger.warning(f"Could not load existing database: {e}")
            existing_journals = []
    
    # Calculate how many more journals we need
    remaining_limit = max(0, limit - len(existing_journals))
    if remaining_limit == 0:
        logger.info(f"Database already has {len(existing_journals)} journals (>= limit of {limit})")
        return existing_journals
    
    logger.info(f"Fetching {remaining_limit} additional journals...")
    
    # Fetch journal data
    try:
        new_journals = api.fetch_journals(
            limit=remaining_limit,
            offset=len(existing_journals)
        )
        
        if not new_journals:
            logger.warning("No journals fetched from API")
            return existing_journals
        
        logger.info(f"Successfully fetched {len(new_journals)} journals")
        
    except Exception as e:
        logger.error(f"Failed to fetch journals: {e}")
        if existing_journals:
            logger.info(f"Proceeding with {len(existing_journals)} existing journals")
            return existing_journals
        raise
    
    # Combine with existing journals
    all_journals = existing_journals + new_journals
    
    # Enrich with DOAJ data
    if not skip_doaj:
        logger.info("Enriching journals with DOAJ data...")
        doaj_api = DOAJAPI(rate_limit=doaj_rate_limit)
        
        try:
            # Only enrich journals that don't already have DOAJ data
            journals_to_enrich = [j for j in all_journals if not j.get('doaj_fetched_at')]
            
            if journals_to_enrich:
                logger.info(f"Enriching {len(journals_to_enrich)} journals with DOAJ data...")
                enriched_journals = doaj_api.enrich_journals_with_doaj(journals_to_enrich)
                
                # Replace enriched journals in the main list
                enriched_dict = {j['id']: j for j in enriched_journals}
                for i, journal in enumerate(all_journals):
                    if journal['id'] in enriched_dict:
                        all_journals[i] = enriched_dict[journal['id']]
                
                doaj_count = sum(1 for j in all_journals if j.get('in_doaj', False))
                logger.info(f"✅ DOAJ enrichment completed. Found {doaj_count} journals in DOAJ")
            else:
                logger.info("All journals already have DOAJ data")
                
        except Exception as e:
            logger.warning(f"DOAJ enrichment failed: {e}")
            logger.info("Continuing without DOAJ data...")
    else:
        logger.info("Skipping DOAJ enrichment as requested")
    
    # Create semantic fingerprints
    logger.info("Creating semantic fingerprints...")
    journals_with_fingerprints = []
    
    for i, journal in enumerate(all_journals):
        if journal.get('semantic_fingerprint'):
            # Already has fingerprint (from resume)
            journals_with_fingerprints.append(journal)
            continue
        
        try:
            # Fetch sample articles for new journals
            journal_id = journal['id']
            sample_articles = api.fetch_sample_articles(journal_id, limit=3)
            
            # Create semantic fingerprint
            fingerprint = create_semantic_fingerprint(journal, sample_articles)
            journal['semantic_fingerprint'] = fingerprint
            journal['sample_articles_count'] = len(sample_articles)
            
            journals_with_fingerprints.append(journal)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Created fingerprints for {i + 1}/{len(all_journals)} journals")
                
        except Exception as e:
            logger.warning(f"Failed to create fingerprint for journal {journal.get('display_name', 'Unknown')}: {e}")
            # Add journal without fingerprint (will be skipped in embedding)
            journals_with_fingerprints.append(journal)
    
    logger.info(f"Created fingerprints for {len(journals_with_fingerprints)} journals")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    try:
        journals_with_embeddings, embeddings = build_journal_embeddings(
            journals_with_fingerprints, 
            batch_size=batch_size
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise
    
    # Save database
    logger.info("Saving journal database...")
    try:
        save_journal_database(journals_with_embeddings, embeddings)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✅ Database build completed successfully!")
        logger.info(f"📊 Stats:")
        logger.info(f"   - Total journals: {len(journals_with_embeddings)}")
        logger.info(f"   - With embeddings: {len([j for j in journals_with_embeddings if j.get('embedding')])}")
        logger.info(f"   - Build time: {duration:.2f} seconds")
        logger.info(f"   - Database saved to: {JOURNAL_METADATA_PATH}")
        
        return journals_with_embeddings
        
    except Exception as e:
        logger.error(f"Failed to save database: {e}")
        raise


def validate_database():
    """Validate the built database."""
    logger.info("Validating database...")
    
    try:
        journals, embeddings = load_journal_database()
        
        # Basic validation
        total_journals = len(journals)
        with_embeddings = len([j for j in journals if j.get('embedding')])
        with_fingerprints = len([j for j in journals if j.get('semantic_fingerprint')])
        
        # DOAJ statistics
        in_doaj = len([j for j in journals if j.get('in_doaj', False)])
        open_access = len([j for j in journals if j.get('oa_status', False)])
        with_apc = len([j for j in journals if j.get('apc_amount') is not None])
        free_to_publish = len([j for j in journals if j.get('has_apc', None) == False])
        
        logger.info(f"✅ Database validation:")
        logger.info(f"   - Total journals: {total_journals}")
        logger.info(f"   - With fingerprints: {with_fingerprints}")
        logger.info(f"   - With embeddings: {with_embeddings}")
        logger.info(f"   - In DOAJ: {in_doaj}")
        logger.info(f"   - Open Access: {open_access}")
        logger.info(f"   - With APC data: {with_apc}")
        logger.info(f"   - Free to publish: {free_to_publish}")
        
        if embeddings is not None:
            logger.info(f"   - Embeddings shape: {embeddings.shape}")
            logger.info(f"   - Embedding dimension: {embeddings.shape[1] if embeddings.ndim > 1 else 'N/A'}")
        
        # Sample journal data
        if journals:
            sample = journals[0]
            logger.info(f"   - Sample journal: {sample.get('display_name', 'Unknown')}")
            logger.info(f"   - Sample ISSN: {sample.get('issn', ['None'])[0] if sample.get('issn') else 'None'}")
            logger.info(f"   - Sample publisher: {sample.get('publisher', 'Unknown')}")
        
        # Validation checks
        issues = []
        if with_embeddings < total_journals * 0.8:  # Less than 80% have embeddings
            issues.append(f"Only {with_embeddings}/{total_journals} journals have embeddings")
        
        if with_fingerprints < total_journals * 0.9:  # Less than 90% have fingerprints
            issues.append(f"Only {with_fingerprints}/{total_journals} journals have fingerprints")
        
        if issues:
            logger.warning("⚠️ Database validation issues:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ Database validation passed!")
        
        return len(issues) == 0
        
    except Exception as e:
        logger.error(f"Database validation failed: {e}")
        return False


def build_database_from_csv(csv_file: str, chunk_size: int = 500, 
                           skip_doaj: bool = False, doaj_rate_limit: float = 1.0,
                           quality_filter: bool = False, min_h_index: int = 10,
                           min_works: int = 20, max_rank: int = None,
                           allowed_quartiles: List[str] = None, batch_size: int = 32):
    """
    Build database from CSV file instead of OpenAlex API.
    
    Args:
        csv_file: Path to Medicine Journal Rankings CSV
        chunk_size: Number of journals per processing chunk
        skip_doaj: Whether to skip DOAJ enrichment
        doaj_rate_limit: Rate limit for DOAJ API calls
        quality_filter: Whether to apply quality filtering
        min_h_index: Minimum H-index for quality filtering
        min_works: Minimum works count for quality filtering
        max_rank: Maximum acceptable Scimago rank
        allowed_quartiles: List of allowed quartiles
        batch_size: Batch size for embedding generation
    """
    start_time = time.time()
    
    # Import CSV processing modules
    try:
        from csv_journal_importer import CSVJournalImporter
        from csv_schema_mapper import CSVSchemaMapper
        from csv_data_processor import CSVDataProcessor
    except ImportError as e:
        logger.error(f"Failed to import CSV processing modules: {e}")
        raise
    
    # Initialize components
    importer = CSVJournalImporter(csv_file)
    mapper = CSVSchemaMapper()
    processor = CSVDataProcessor()
    
    # Load and process CSV
    logger.info("Loading CSV file...")
    importer.load_csv()
    importer.clean_and_validate()
    
    # Get statistics
    stats = importer.get_statistics()
    logger.info(f"📊 CSV Statistics:")
    logger.info(f"   - Total journals: {stats['total_journals']}")
    logger.info(f"   - Q1 journals: {stats['q1_journals']}")
    logger.info(f"   - Q2 journals: {stats['q2_journals']}")
    logger.info(f"   - Top 100: {stats['top_100']}")
    logger.info(f"   - Top 500: {stats['top_500']}")
    logger.info(f"   - Countries: {stats['countries']}")
    logger.info(f"   - Publishers: {stats['publishers']}")
    
    # Process in chunks
    chunks = importer.get_processed_chunks(chunk_size)
    all_journals = []
    
    logger.info(f"Processing {len(chunks)} chunks of ~{chunk_size} journals each...")
    
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} journals)...")
        
        # Map CSV data to database schema
        chunk_journals = []
        for _, row in chunk.iterrows():
            journal = mapper.map_journal(row.to_dict())
            if journal:
                # Create semantic fingerprint
                fingerprint = mapper.create_semantic_fingerprint(journal)
                journal['semantic_fingerprint'] = fingerprint
                chunk_journals.append(journal)
        
        all_journals.extend(chunk_journals)
        
        mapping_stats = mapper.get_mapping_stats()
        logger.info(f"   Mapped {len(chunk_journals)} journals (errors: {mapping_stats['errors']})")
    
    logger.info(f"Successfully mapped {len(all_journals)} journals from CSV")
    
    # Generate quality report
    logger.info("Generating quality report...")
    quality_report = processor.generate_quality_report(all_journals)
    logger.info(f"📈 Quality Report:")
    logger.info(f"   - Quality score: {quality_report['quality_metrics']['quality_score']:.1f}%")
    logger.info(f"   - Completeness score: {quality_report['quality_metrics']['completeness_score']:.1f}%")
    logger.info(f"   - Journals with warnings: {quality_report['quality_metrics']['journals_with_warnings']}")
    
    if quality_report['recommendations']:
        logger.info("📋 Recommendations:")
        for rec in quality_report['recommendations'][:3]:
            logger.info(f"   - {rec}")
    
    # Apply quality filtering if requested
    if quality_filter:
        logger.info("Applying quality filters...")
        original_count = len(all_journals)
        all_journals = processor.filter_high_quality_journals(
            all_journals,
            min_works=min_works,
            min_h_index=min_h_index,
            max_rank=max_rank,
            allowed_quartiles=allowed_quartiles
        )
        logger.info(f"Quality filtering: {original_count} → {len(all_journals)} journals")
    
    # DOAJ enrichment (optional)
    if not skip_doaj and all_journals:
        logger.info("Enriching journals with DOAJ data...")
        
        try:
            from journal_db_builder import DOAJAPI
            doaj_api = DOAJAPI(rate_limit=doaj_rate_limit)
            
            # Process in smaller batches for DOAJ
            doaj_batch_size = min(chunk_size // 5, 100)  # Smaller batches for DOAJ
            enriched_journals = doaj_api.enrich_journals_with_doaj(
                all_journals, 
                batch_size=doaj_batch_size
            )
            all_journals = enriched_journals
            
            doaj_count = sum(1 for j in all_journals if j.get('in_doaj', False))
            oa_count = sum(1 for j in all_journals if j.get('oa_status', False))
            logger.info(f"✅ DOAJ enrichment completed:")
            logger.info(f"   - DOAJ journals: {doaj_count}")
            logger.info(f"   - Open access: {oa_count}")
            
        except Exception as e:
            logger.warning(f"DOAJ enrichment failed: {e}")
            logger.info("Continuing without DOAJ data...")
    
    # Optimize for embeddings
    logger.info("Optimizing journals for embedding generation...")
    all_journals = processor.optimize_for_embedding(all_journals)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    try:
        journals_with_embeddings, embeddings = build_journal_embeddings(
            all_journals, 
            batch_size=batch_size
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise
    
    # Save database
    logger.info("Saving journal database...")
    try:
        save_journal_database(journals_with_embeddings, embeddings)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✅ CSV database build completed successfully!")
        logger.info(f"📊 Final Stats:")
        logger.info(f"   - Total journals: {len(journals_with_embeddings)}")
        logger.info(f"   - With embeddings: {len([j for j in journals_with_embeddings if j.get('embedding')])}")
        logger.info(f"   - DOAJ journals: {sum(1 for j in journals_with_embeddings if j.get('in_doaj', False))}")
        logger.info(f"   - Build time: {duration:.2f} seconds")
        logger.info(f"   - Database saved to: {JOURNAL_METADATA_PATH}")
        
        return journals_with_embeddings
        
    except Exception as e:
        logger.error(f"Failed to save database: {e}")
        raise


def main():
    """Main function."""
    args = parse_args()
    
    # Set test mode
    if args.test:
        args.limit = 10
        logger.info("🧪 Running in test mode (10 journals)")
    
    logger.info("=" * 60)
    if args.csv_file:
        logger.info("🏗️  JOURNAL DATABASE BUILDER (CSV Mode)")
    else:
        logger.info("🏗️  JOURNAL DATABASE BUILDER (OpenAlex Mode)")
    logger.info("=" * 60)
    logger.info(f"Settings:")
    
    if args.csv_file:
        logger.info(f"  - CSV file: {args.csv_file}")
        logger.info(f"  - CSV chunk size: {args.csv_chunk_size}")
        logger.info(f"  - Quality filter: {args.quality_filter}")
        if args.quality_filter:
            logger.info(f"  - Min H-index: {args.min_h_index}")
            logger.info(f"  - Min works: {args.min_works}")
            if args.max_rank:
                logger.info(f"  - Max rank: {args.max_rank}")
            if args.allowed_quartiles:
                logger.info(f"  - Allowed quartiles: {', '.join(args.allowed_quartiles)}")
    else:
        logger.info(f"  - Limit: {args.limit} journals")
        logger.info(f"  - Resume: {args.resume}")
        logger.info(f"  - OpenAlex rate limit: {args.rate_limit}s")
    
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Skip DOAJ: {args.skip_doaj}")
    if not args.skip_doaj:
        logger.info(f"  - DOAJ rate limit: {args.doaj_rate_limit}s")
    logger.info("=" * 60)
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            sys.exit(1)
        
        # Build database from CSV or OpenAlex
        if args.csv_file:
            logger.info(f"📁 Building database from CSV: {args.csv_file}")
            
            # Validate CSV file exists
            if not Path(args.csv_file).exists():
                logger.error(f"CSV file not found: {args.csv_file}")
                sys.exit(1)
            
            journals = build_database_from_csv(
                csv_file=args.csv_file,
                chunk_size=args.csv_chunk_size,
                skip_doaj=args.skip_doaj,
                doaj_rate_limit=args.doaj_rate_limit,
                quality_filter=args.quality_filter,
                min_h_index=args.min_h_index,
                min_works=args.min_works,
                max_rank=args.max_rank,
                allowed_quartiles=args.allowed_quartiles,
                batch_size=args.batch_size
            )
        else:
            logger.info("🌐 Building database from OpenAlex API")
            journals = build_database(
                limit=args.limit,
                resume=args.resume,
                batch_size=args.batch_size,
                rate_limit=args.rate_limit,
                skip_doaj=args.skip_doaj,
                doaj_rate_limit=args.doaj_rate_limit
            )
        
        # Validate database
        if validate_database():
            logger.info("🎉 Database build and validation completed successfully!")
            sys.exit(0)
        else:
            logger.warning("⚠️ Database built but validation found issues")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Build interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Database build failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()