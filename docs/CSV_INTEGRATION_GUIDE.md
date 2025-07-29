# Medical Journal Rankings CSV Integration Guide

**Document Version**: 1.0  
**Date**: July 28, 2025  
**Author**: Development Team  
**Status**: Implementation Ready

## Overview

This guide provides complete instructions for integrating the **Medicine Journal Rankings 2024.csv** file (7,678 journals) into the Manuscript Journal Matcher system. The integration will transform the system from a 10-journal test database to a production-ready system with comprehensive medical journal coverage.

## Table of Contents

1. [Data Analysis & Structure](#data-analysis--structure)
2. [Prerequisites](#prerequisites)
3. [Implementation Architecture](#implementation-architecture)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Testing & Validation](#testing--validation)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Future Enhancements](#future-enhancements)

## Data Analysis & Structure

### CSV File Details
- **File**: `Medicine Journal Rankings 2024.csv`
- **Total Records**: 7,678 journals (plus header)
- **Separator**: Semicolon (`;`)
- **Encoding**: UTF-8
- **Data Source**: Scimago Journal Rank (SJR) 2024

### CSV Column Structure

| Column | Field Name | Type | Description | Example |
|--------|------------|------|-------------|---------|
| 1 | Rank | Integer | Scimago ranking position | `1`, `2`, `3` |
| 2 | Sourceid | String | Scopus source identifier | `28773`, `19434` |
| 3 | Title | String | Journal title | `"Ca-A Cancer Journal for Clinicians"` |
| 4 | Type | String | Publication type | `"journal"` |
| 5 | Issn | String | ISSN numbers (comma-separated) | `"15424863, 00079235"` |
| 6 | SJR | Float | SJR score | `145,004`, `41,754` |
| 7 | SJR Best Quartile | String | Best quartile ranking | `"Q1"`, `"Q2"`, `"Q3"`, `"Q4"` |
| 8 | H index | Integer | H-index score | `223`, `155` |
| 9 | Total Docs. (2024) | Integer | Documents published in 2024 | `43`, `6` |
| 10 | Total Docs. (3years) | Integer | Documents in last 3 years | `122`, `15` |
| 11 | Total Refs. | Integer | Total references | `2704`, `1652` |
| 12 | Total Cites (3years) | Integer | Citations in last 3 years | `40834`, `1308` |
| 13 | Citable Docs. (3years) | Integer | Citable documents | `81`, `15` |
| 14 | Cites / Doc. (2years) | Float | Citations per document | `168,71`, `75,11` |
| 15 | Ref. / Doc. | Float | References per document | `62,88`, `275,33` |
| 16 | %Female | Float | Percentage female authors | `48,21`, `75,93` |
| 17 | Overton | Integer | Overton score | `4`, `1` |
| 18 | SDG | Integer | Sustainable Development Goals | `37`, `5` |
| 19 | Country | String | Publisher country | `"United States"`, `"United Kingdom"` |
| 20 | Region | String | Geographic region | `"Northern America"`, `"Western Europe"` |
| 21 | Publisher | String | Publisher name | `"John Wiley and Sons Inc"` |
| 22 | Coverage | String | Publication years | `"1950-2025"`, `"1990-2024"` |
| 23 | Categories | String | Subject categories with quartiles | `"Hematology (Q1); Oncology (Q1)"` |
| 24 | Areas | String | Broad subject areas | `"Medicine"`, `"Environmental Science; Medicine"` |

### Data Quality Notes

1. **Missing Values**: Some fields may be empty (represented as empty strings)
2. **Discontinued Journals**: Some journals are marked as "(discontinued)" in the title
3. **Multiple ISSNs**: ISSNs are comma-separated when multiple exist
4. **Decimal Format**: Uses comma as decimal separator (European format)
5. **Multiple Categories**: Categories and areas are semicolon-separated

## Prerequisites

### System Requirements
- Python 3.8+
- Existing Manuscript Journal Matcher system (Step 8 DOAJ integration complete)
- At least 8GB RAM for processing 7,678 journals
- 2GB free disk space for database and embeddings

### Dependencies
```bash
# Core dependencies (already installed)
pandas>=2.1.0
numpy>=1.24.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4

# Additional for CSV processing
chardet>=5.0.0  # For encoding detection
tqdm>=4.64.0    # For progress bars
```

### File Preparation
1. Ensure `Medicine Journal Rankings 2024.csv` is in project root
2. Verify file integrity:
   ```bash
   wc -l "Medicine Journal Rankings 2024.csv"  # Should show 7679 lines
   head -1 "Medicine Journal Rankings 2024.csv"  # Verify header
   ```

## Implementation Architecture

### Component Overview
```
CSV Integration System
├── CSV Parser (src/csv_journal_importer.py)
├── Data Processor (src/csv_data_processor.py)
├── Chunked Builder (enhanced build_database.py)
├── Schema Mapper (src/csv_schema_mapper.py)
└── Validation Suite (tests/test_csv_integration.py)
```

### Processing Flow
```
CSV File → Parse & Clean → Chunk Processing → DOAJ Enrichment → Embedding Generation → Database Save
```

### Memory Management Strategy
- **Chunk Size**: 500-1000 journals per batch
- **Streaming**: Process one chunk at a time
- **Garbage Collection**: Explicit cleanup between chunks
- **Progress Persistence**: Save progress to enable resume

## Step-by-Step Implementation

### Step 1: Create CSV Journal Importer

**File**: `src/csv_journal_importer.py`

```python
#!/usr/bin/env python3
"""
CSV Journal Importer for Medicine Journal Rankings 2024

Handles parsing, cleaning, and initial processing of the CSV file
containing 7,678 medical journals from Scimago Journal Rank.
"""

import pandas as pd
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class CSVJournalImporter:
    """Import and process medical journal rankings CSV file."""
    
    def __init__(self, csv_file_path: str):
        """
        Initialize CSV importer.
        
        Args:
            csv_file_path: Path to Medicine Journal Rankings 2024.csv
        """
        self.csv_file_path = Path(csv_file_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_csv(self) -> pd.DataFrame:
        """Load and parse the CSV file with proper handling of European format."""
        try:
            # Read CSV with semicolon separator
            df = pd.read_csv(
                self.csv_file_path,
                sep=';',
                encoding='utf-8',
                dtype=str,  # Read all as strings initially
                na_values=['', 'null', 'NULL', '-']
            )
            
            logger.info(f"Loaded CSV with {len(df)} journals")
            self.raw_data = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def clean_and_validate(self) -> pd.DataFrame:
        """Clean and validate the CSV data."""
        if self.raw_data is None:
            raise ValueError("CSV not loaded. Call load_csv() first.")
        
        df = self.raw_data.copy()
        
        # 1. Remove discontinued journals
        active_mask = ~df['Title'].str.contains(r'\(discontinued\)', case=False, na=False)
        df = df[active_mask]
        logger.info(f"Filtered out discontinued journals. Remaining: {len(df)}")
        
        # 2. Filter journals with valid ISSNs
        issn_mask = df['Issn'].notna() & (df['Issn'].str.strip() != '')
        df = df[issn_mask]
        logger.info(f"Filtered journals with valid ISSNs. Remaining: {len(df)}")
        
        # 3. Clean numeric fields (European format with commas)
        numeric_fields = ['SJR', 'H index', 'Total Docs. (2024)', 'Total Docs. (3years)',
                         'Total Refs.', 'Total Cites (3years)', 'Cites / Doc. (2years)',
                         'Ref. / Doc.', '%Female']
        
        for field in numeric_fields:
            if field in df.columns:
                # Replace comma with dot for decimal conversion
                df[field] = df[field].str.replace(',', '.', regex=False)
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        # 4. Convert rank to integer
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        
        # 5. Clean and normalize text fields
        text_fields = ['Title', 'Publisher', 'Country', 'Region', 'Categories', 'Areas']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].str.strip()
        
        self.processed_data = df
        logger.info(f"Data cleaning complete. Final count: {len(df)} journals")
        return df
    
    def split_categories(self, categories_str: str) -> List[Dict[str, Any]]:
        """Parse categories string into structured format."""
        if pd.isna(categories_str) or not categories_str:
            return []
        
        categories = []
        # Split by semicolon and parse each category
        for cat in categories_str.split(';'):
            cat = cat.strip()
            if not cat:
                continue
                
            # Extract quartile if present: "Oncology (Q1)"
            quartile_match = re.search(r'\((Q[1-4])\)', cat)
            if quartile_match:
                quartile = quartile_match.group(1)
                name = cat.replace(f'({quartile})', '').strip()
            else:
                quartile = None
                name = cat
            
            categories.append({
                'name': name,
                'quartile': quartile
            })
        
        return categories
    
    def process_issns(self, issn_str: str) -> Dict[str, Any]:
        """Process ISSN string into structured format."""
        if pd.isna(issn_str) or not issn_str:
            return {'issn_list': [], 'issn_l': None}
        
        # Split comma-separated ISSNs and clean
        issns = [issn.strip() for issn in issn_str.split(',') if issn.strip()]
        
        # Validate ISSN format (8 digits)
        valid_issns = []
        for issn in issns:
            clean_issn = re.sub(r'[^\d]', '', issn)
            if len(clean_issn) == 8:
                # Add hyphen for standard format: XXXXXXXX -> XXXX-XXXX
                formatted_issn = f"{clean_issn[:4]}-{clean_issn[4:]}"
                valid_issns.append(formatted_issn)
        
        return {
            'issn_list': valid_issns,
            'issn_l': valid_issns[0] if valid_issns else None
        }
    
    def get_processed_chunks(self, chunk_size: int = 500) -> List[pd.DataFrame]:
        """Split processed data into manageable chunks."""
        if self.processed_data is None:
            raise ValueError("Data not processed. Call clean_and_validate() first.")
        
        chunks = []
        df = self.processed_data
        
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks of size ~{chunk_size}")
        return chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            return {"error": "No processed data available"}
        
        df = self.processed_data
        
        return {
            "total_journals": len(df),
            "q1_journals": len(df[df['SJR Best Quartile'] == 'Q1']),
            "q2_journals": len(df[df['SJR Best Quartile'] == 'Q2']),
            "q3_journals": len(df[df['SJR Best Quartile'] == 'Q3']),
            "q4_journals": len(df[df['SJR Best Quartile'] == 'Q4']),
            "top_100": len(df[df['Rank'] <= 100]),
            "top_500": len(df[df['Rank'] <= 500]),
            "with_sjr_score": len(df[df['SJR'].notna()]),
            "countries": df['Country'].nunique(),
            "publishers": df['Publisher'].nunique(),
            "avg_h_index": df['H index'].mean(),
            "avg_sjr_score": df['SJR'].mean()
        }

# Usage Example
if __name__ == "__main__":
    importer = CSVJournalImporter("Medicine Journal Rankings 2024.csv")
    
    # Load and process data
    raw_df = importer.load_csv()
    processed_df = importer.clean_and_validate()
    
    # Get chunks for processing
    chunks = importer.get_processed_chunks(chunk_size=500)
    
    # Display statistics
    stats = importer.get_statistics()
    print(f"Processing {stats['total_journals']} journals in {len(chunks)} chunks")
```

### Step 2: Create Schema Mapper

**File**: `src/csv_schema_mapper.py`

```python
#!/usr/bin/env python3
"""
CSV to Database Schema Mapper

Maps Medicine Journal Rankings CSV data to the existing
journal database schema with DOAJ integration.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class CSVSchemaMapper:
    """Maps CSV data to journal database schema."""
    
    def __init__(self):
        """Initialize schema mapper."""
        self.mapping_stats = {
            'mapped': 0,
            'errors': 0,
            'warnings': []
        }
    
    def map_journal(self, csv_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a single CSV row to journal database schema.
        
        Args:
            csv_row: Dictionary representing one CSV row
            
        Returns:
            Journal data in database schema format
        """
        try:
            # Process ISSNs
            issn_data = self._process_issns(csv_row.get('Issn', ''))
            
            # Process categories
            categories = self._process_categories(csv_row.get('Categories', ''))
            
            # Create base journal record
            journal = {
                # Basic identification
                'id': f"CSV_{csv_row.get('Sourceid', '')}",
                'display_name': csv_row.get('Title', '').strip('"'),
                'issn': issn_data['issn_list'],
                'issn_l': issn_data['issn_l'],
                
                # Publisher information
                'publisher': csv_row.get('Publisher', '').strip('"'),
                'homepage_url': None,  # Not available in CSV
                'country_code': self._map_country_code(csv_row.get('Country', '')),
                'type': csv_row.get('Type', 'journal'),
                
                # Scimago-specific fields
                'scimago_rank': self._safe_int(csv_row.get('Rank')),
                'sjr_score': self._safe_float(csv_row.get('SJR')),
                'sjr_quartile': csv_row.get('SJR Best Quartile', '').strip(),
                'sourceid': csv_row.get('Sourceid', ''),
                
                # Citation metrics
                'works_count': self._safe_int(csv_row.get('Total Docs. (2024)', 0)),
                'cited_by_count': self._safe_int(csv_row.get('Total Cites (3years)', 0)),
                'h_index': self._safe_int(csv_row.get('H index', 0)),
                'cites_per_doc': self._safe_float(csv_row.get('Cites / Doc. (2years)')),
                'refs_per_doc': self._safe_float(csv_row.get('Ref. / Doc.')),
                
                # Additional metrics
                'female_percentage': self._safe_float(csv_row.get('%Female')),
                'overton_score': self._safe_int(csv_row.get('Overton')),
                'sdg_score': self._safe_int(csv_row.get('SDG')),
                
                # Geographic information
                'country': csv_row.get('Country', ''),
                'region': csv_row.get('Region', ''),
                
                # Coverage and areas
                'coverage_years': csv_row.get('Coverage', ''),
                'areas': csv_row.get('Areas', '').split(';') if csv_row.get('Areas') else [],
                
                # Subjects (from categories)
                'subjects': categories,
                
                # Placeholders for DOAJ integration (filled later)
                'is_oa': None,  # Will be determined by DOAJ
                'is_in_doaj': None,
                'oa_status': None,
                'in_doaj': None,
                'has_apc': None,
                'apc_amount': None,
                'apc_currency': None,
                'apc_usd': None,  # Legacy field
                'subjects_doaj': [],
                'languages': [],
                'license_type': [],
                'publisher_doaj': None,
                'country_doaj': None,
                
                # System fields
                'description': None,
                'scope_text': None,
                'semantic_fingerprint': None,
                'embedding': None,
                'fetched_at': datetime.now().isoformat(),
                'csv_source': True,
                'csv_imported_at': datetime.now().isoformat()
            }
            
            self.mapping_stats['mapped'] += 1
            return journal
            
        except Exception as e:
            self.mapping_stats['errors'] += 1
            self.mapping_stats['warnings'].append(f"Error mapping journal {csv_row.get('Title', 'Unknown')}: {e}")
            logger.error(f"Error mapping journal: {e}")
            return None
    
    def _process_issns(self, issn_str: str) -> Dict[str, Any]:
        """Process ISSN string into structured format."""
        if not issn_str or str(issn_str).strip() == '':
            return {'issn_list': [], 'issn_l': None}
        
        # Remove quotes and split by comma
        issn_str = str(issn_str).strip('"')
        issns = [issn.strip() for issn in issn_str.split(',') if issn.strip()]
        
        # Format ISSNs properly
        formatted_issns = []
        for issn in issns:
            # Remove any existing hyphens and format as XXXX-XXXX
            clean_issn = re.sub(r'[^\d]', '', issn)
            if len(clean_issn) == 8:
                formatted_issn = f"{clean_issn[:4]}-{clean_issn[4:]}"
                formatted_issns.append(formatted_issn)
        
        return {
            'issn_list': formatted_issns,
            'issn_l': formatted_issns[0] if formatted_issns else None
        }
    
    def _process_categories(self, categories_str: str) -> List[Dict[str, Any]]:
        """Process categories string into subjects format."""
        if not categories_str or str(categories_str).strip() == '':
            return []
        
        categories = []
        categories_str = str(categories_str).strip('"')
        
        for cat in categories_str.split(';'):
            cat = cat.strip()
            if not cat:
                continue
            
            # Extract quartile: "Oncology (Q1)" -> name="Oncology", quartile="Q1"
            quartile_match = re.search(r'\((Q[1-4])\)', cat)
            if quartile_match:
                quartile = quartile_match.group(1)
                name = cat.replace(f'({quartile})', '').strip()
                score = self._quartile_to_score(quartile)
            else:
                name = cat
                quartile = None
                score = 0.5  # Default score
            
            categories.append({
                'name': name,
                'score': score,
                'quartile': quartile
            })
        
        return categories
    
    def _quartile_to_score(self, quartile: str) -> float:
        """Convert quartile to numerical score."""
        quartile_scores = {
            'Q1': 0.9,
            'Q2': 0.7,
            'Q3': 0.5,
            'Q4': 0.3
        }
        return quartile_scores.get(quartile, 0.5)
    
    def _map_country_code(self, country: str) -> Optional[str]:
        """Map country name to country code."""
        # Basic country mapping - extend as needed
        country_mapping = {
            'United States': 'US',
            'United Kingdom': 'GB',
            'Germany': 'DE',
            'France': 'FR',
            'Netherlands': 'NL',
            'Spain': 'ES',
            'Italy': 'IT',
            'Canada': 'CA',
            'Australia': 'AU',
            'Japan': 'JP',
            'China': 'CN',
            'India': 'IN',
            'Brazil': 'BR',
            'Russia': 'RU',
            'South Korea': 'KR',
            'Switzerland': 'CH',
            'Sweden': 'SE',
            'Norway': 'NO',
            'Denmark': 'DK',
            'Finland': 'FI'
        }
        return country_mapping.get(country, None)
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to integer."""
        try:
            if value is None or str(value).strip() == '':
                return None
            return int(float(str(value)))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        try:
            if value is None or str(value).strip() == '':
                return None
            return float(str(value))
        except (ValueError, TypeError):
            return None
    
    def create_semantic_fingerprint(self, journal: Dict[str, Any]) -> str:
        """Create semantic fingerprint for CSV journal."""
        parts = []
        
        # Journal name
        if journal.get('display_name'):
            parts.append(f"Journal: {journal['display_name']}")
        
        # Publisher
        if journal.get('publisher'):
            parts.append(f"Publisher: {journal['publisher']}")
        
        # Ranking information
        if journal.get('scimago_rank'):
            parts.append(f"Scimago Rank: {journal['scimago_rank']}")
        
        if journal.get('sjr_quartile'):
            parts.append(f"SJR Quartile: {journal['sjr_quartile']}")
        
        # Subject areas
        if journal.get('subjects'):
            subject_names = [s['name'] for s in journal['subjects'][:3]]
            parts.append(f"Subject areas: {', '.join(subject_names)}")
        
        # Geographic info
        if journal.get('country'):
            parts.append(f"Country: {journal['country']}")
        
        # Areas
        if journal.get('areas'):
            areas = ', '.join(journal['areas'][:3])
            parts.append(f"Research areas: {areas}")
        
        # Quality indicators
        if journal.get('h_index'):
            parts.append(f"H-index: {journal['h_index']}")
        
        if journal.get('sjr_score'):
            parts.append(f"SJR score: {journal['sjr_score']}")
        
        return ' | '.join(parts)
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about the mapping process."""
        return self.mapping_stats.copy()

# Usage Example
if __name__ == "__main__":
    mapper = CSVSchemaMapper()
    
    # Example CSV row (simplified)
    csv_row = {
        'Rank': '1',
        'Sourceid': '28773',
        'Title': '"Ca-A Cancer Journal for Clinicians"',
        'Issn': '"15424863, 00079235"',
        'SJR': '145,004',
        'SJR Best Quartile': 'Q1',
        'H index': '223',
        'Publisher': '"John Wiley and Sons Inc"',
        'Country': 'United States',
        'Categories': '"Hematology (Q1); Oncology (Q1)"'
    }
    
    # Map to database schema
    journal = mapper.map_journal(csv_row)
    
    # Create semantic fingerprint
    fingerprint = mapper.create_semantic_fingerprint(journal)
    journal['semantic_fingerprint'] = fingerprint
    
    print(f"Mapped journal: {journal['display_name']}")
    print(f"Fingerprint: {fingerprint}")
```

### Step 3: Enhance Database Builder

**File**: Modifications to `scripts/build_database.py`

Add the following command-line arguments and functionality:

```python
# Add to parse_args() function
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

# Add new function for CSV-based database building
def build_database_from_csv(csv_file: str, chunk_size: int = 500, 
                           skip_doaj: bool = False, doaj_rate_limit: float = 1.0):
    """
    Build database from CSV file instead of OpenAlex API.
    
    Args:
        csv_file: Path to Medicine Journal Rankings CSV
        chunk_size: Number of journals per processing chunk
        skip_doaj: Whether to skip DOAJ enrichment
        doaj_rate_limit: Rate limit for DOAJ API calls
    """
    from csv_journal_importer import CSVJournalImporter
    from csv_schema_mapper import CSVSchemaMapper
    from journal_db_builder import DOAJAPI
    
    start_time = time.time()
    
    # Initialize components
    importer = CSVJournalImporter(csv_file)
    mapper = CSVSchemaMapper()
    
    # Load and process CSV
    logger.info("Loading CSV file...")
    importer.load_csv()
    importer.clean_and_validate()
    
    # Get statistics
    stats = importer.get_statistics()
    logger.info(f"Processing {stats['total_journals']} journals from CSV")
    
    # Process in chunks
    chunks = importer.get_processed_chunks(chunk_size)
    all_journals = []
    
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
        logger.info(f"Mapped {len(chunk_journals)} journals in chunk {i}")
    
    logger.info(f"Successfully mapped {len(all_journals)} journals from CSV")
    
    # DOAJ enrichment (optional)
    if not skip_doaj:
        logger.info("Enriching journals with DOAJ data...")
        doaj_api = DOAJAPI(rate_limit=doaj_rate_limit)
        
        try:
            enriched_journals = doaj_api.enrich_journals_with_doaj(
                all_journals, 
                batch_size=min(chunk_size, 50)  # Smaller batches for DOAJ
            )
            all_journals = enriched_journals
            
            doaj_count = sum(1 for j in all_journals if j.get('in_doaj', False))
            logger.info(f"✅ DOAJ enrichment completed. Found {doaj_count} journals in DOAJ")
            
        except Exception as e:
            logger.warning(f"DOAJ enrichment failed: {e}")
            logger.info("Continuing without DOAJ data...")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    try:
        journals_with_embeddings, embeddings = build_journal_embeddings(
            all_journals, 
            batch_size=32
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
        logger.info(f"📊 Stats:")
        logger.info(f"   - Total journals: {len(journals_with_embeddings)}")
        logger.info(f"   - With embeddings: {len([j for j in journals_with_embeddings if j.get('embedding')])}")
        logger.info(f"   - DOAJ journals: {sum(1 for j in journals_with_embeddings if j.get('in_doaj', False))}")
        logger.info(f"   - Build time: {duration:.2f} seconds")
        logger.info(f"   - Database saved to: {JOURNAL_METADATA_PATH}")
        
        return journals_with_embeddings
        
    except Exception as e:
        logger.error(f"Failed to save database: {e}")
        raise

# Update main() function to handle CSV option
def main():
    """Main function with CSV support."""
    args = parse_args()
    
    # ... existing setup code ...
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            sys.exit(1)
        
        # Build database from CSV or OpenAlex
        if args.csv_file:
            logger.info(f"Building database from CSV: {args.csv_file}")
            journals = build_database_from_csv(
                csv_file=args.csv_file,
                chunk_size=args.csv_chunk_size,
                skip_doaj=args.skip_doaj,
                doaj_rate_limit=args.doaj_rate_limit
            )
        else:
            logger.info("Building database from OpenAlex API")
            journals = build_database(
                limit=args.limit,
                resume=args.resume,
                batch_size=args.batch_size,
                rate_limit=args.rate_limit,
                skip_doaj=args.skip_doaj,
                doaj_rate_limit=args.doaj_rate_limit
            )
        
        # ... rest of existing validation code ...
```

### Step 4: Create Data Processing Module

**File**: `src/csv_data_processor.py`

```python
#!/usr/bin/env python3
"""
CSV Data Processor

Handles advanced processing, validation, and quality assurance
for Medicine Journal Rankings CSV data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

class CSVDataProcessor:
    """Advanced processing and validation for CSV journal data."""
    
    def __init__(self):
        """Initialize data processor."""
        self.validation_results = {
            'total_processed': 0,
            'validation_errors': [],
            'quality_warnings': [],
            'statistics': {}
        }
    
    def validate_journal_quality(self, journal: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate journal data quality and flag potential issues.
        
        Args:
            journal: Journal data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Critical validations (would exclude journal)
        if not journal.get('display_name'):
            issues.append("CRITICAL: Missing journal title")
        
        if not journal.get('issn') or len(journal.get('issn', [])) == 0:
            issues.append("CRITICAL: Missing ISSN")
        
        if journal.get('scimago_rank') and journal['scimago_rank'] <= 0:
            issues.append("CRITICAL: Invalid Scimago rank")
        
        # Quality warnings (journal included but flagged)
        if not journal.get('publisher'):
            issues.append("WARNING: Missing publisher information")
        
        if not journal.get('subjects') or len(journal.get('subjects', [])) == 0:
            issues.append("WARNING: No subject categories")
        
        if journal.get('works_count', 0) < 10:
            issues.append("WARNING: Very low publication count")
        
        if not journal.get('sjr_score') or journal.get('sjr_score', 0) <= 0:
            issues.append("WARNING: Missing or invalid SJR score")
        
        if not journal.get('h_index') or journal.get('h_index', 0) <= 0:
            issues.append("WARNING: Missing or invalid H-index")
        
        # Check for critical issues
        critical_issues = [issue for issue in issues if issue.startswith("CRITICAL")]
        is_valid = len(critical_issues) == 0
        
        return is_valid, issues
    
    def detect_duplicates(self, journals: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """
        Detect potential duplicate journals.
        
        Args:
            journals: List of journal dictionaries
            
        Returns:
            List of tuples indicating duplicate pairs (index1, index2)
        """
        duplicates = []
        
        # Group by ISSN
        issn_groups = {}
        for i, journal in enumerate(journals):
            issns = journal.get('issn', [])
            for issn in issns:
                if issn not in issn_groups:
                    issn_groups[issn] = []
                issn_groups[issn].append(i)
        
        # Find ISSN duplicates
        for issn, indices in issn_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        duplicates.append((indices[i], indices[j]))
        
        # Group by title similarity
        title_groups = {}
        for i, journal in enumerate(journals):
            title = journal.get('display_name', '').lower().strip()
            # Simple normalization
            normalized_title = re.sub(r'[^\w\s]', '', title)
            normalized_title = re.sub(r'\s+', ' ', normalized_title)
            
            if normalized_title not in title_groups:
                title_groups[normalized_title] = []
            title_groups[normalized_title].append(i)
        
        # Find title duplicates
        for title, indices in title_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        # Avoid adding same pair twice
                        pair = (indices[i], indices[j])
                        reverse_pair = (indices[j], indices[i])
                        if pair not in duplicates and reverse_pair not in duplicates:
                            duplicates.append(pair)
        
        logger.info(f"Found {len(duplicates)} potential duplicate pairs")
        return duplicates
    
    def generate_quality_report(self, journals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for journal data.
        
        Args:
            journals: List of processed journal dictionaries
            
        Returns:
            Quality report dictionary
        """
        report = {
            'total_journals': len(journals),
            'timestamp': datetime.now().isoformat(),
            'quality_metrics': {},
            'data_completeness': {},
            'distribution_analysis': {},
            'recommendations': []
        }
        
        if not journals:
            return report
        
        # Quality metrics
        valid_count = 0
        warning_count = 0
        
        for journal in journals:
            is_valid, issues = self.validate_journal_quality(journal)
            if is_valid:
                valid_count += 1
            if any('WARNING' in issue for issue in issues):
                warning_count += 1
        
        report['quality_metrics'] = {
            'valid_journals': valid_count,
            'journals_with_warnings': warning_count,
            'quality_score': (valid_count / len(journals)) * 100 if journals else 0
        }
        
        # Data completeness
        fields_to_check = [
            'display_name', 'issn', 'publisher', 'subjects', 'scimago_rank',
            'sjr_score', 'h_index', 'works_count', 'country', 'sjr_quartile'
        ]
        
        completeness = {}
        for field in fields_to_check:
            complete_count = sum(1 for j in journals if j.get(field) is not None and j.get(field) != '')
            completeness[field] = {
                'count': complete_count,
                'percentage': (complete_count / len(journals)) * 100
            }
        
        report['data_completeness'] = completeness
        
        # Distribution analysis
        # Quartile distribution
        quartile_dist = Counter(j.get('sjr_quartile') for j in journals if j.get('sjr_quartile'))
        report['distribution_analysis']['quartile_distribution'] = dict(quartile_dist)
        
        # Country distribution (top 10)
        country_dist = Counter(j.get('country') for j in journals if j.get('country'))
        report['distribution_analysis']['top_countries'] = dict(country_dist.most_common(10))
        
        # Publisher distribution (top 10)
        publisher_dist = Counter(j.get('publisher') for j in journals if j.get('publisher'))
        report['distribution_analysis']['top_publishers'] = dict(publisher_dist.most_common(10))
        
        # Numerical statistics
        numerical_fields = ['scimago_rank', 'sjr_score', 'h_index', 'works_count']
        for field in numerical_fields:
            values = [j.get(field) for j in journals if j.get(field) is not None]
            if values:
                report['distribution_analysis'][f'{field}_stats'] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Generate recommendations
        recommendations = []
        
        if report['quality_metrics']['quality_score'] < 90:
            recommendations.append("Consider additional data cleaning - quality score below 90%")
        
        if completeness['publisher']['percentage'] < 95:
            recommendations.append("Many journals missing publisher information")
        
        if completeness['subjects']['percentage'] < 95:
            recommendations.append("Many journals missing subject classifications")
        
        if len(self.detect_duplicates(journals)) > 0:
            recommendations.append("Potential duplicate journals detected - review before processing")
        
        report['recommendations'] = recommendations
        
        return report
    
    def filter_high_quality_journals(self, journals: List[Dict[str, Any]], 
                                   min_works: int = 50, 
                                   min_h_index: int = 10,
                                   require_sjr: bool = True) -> List[Dict[str, Any]]:
        """
        Filter journals based on quality criteria.
        
        Args:
            journals: List of journal dictionaries
            min_works: Minimum number of works published
            min_h_index: Minimum H-index
            require_sjr: Whether SJR score is required
            
        Returns:
            Filtered list of high-quality journals
        """
        filtered = []
        
        for journal in journals:
            # Basic validation
            is_valid, _ = self.validate_journal_quality(journal)
            if not is_valid:
                continue
            
            # Quality filters
            if journal.get('works_count', 0) < min_works:
                continue
            
            if journal.get('h_index', 0) < min_h_index:
                continue
            
            if require_sjr and (not journal.get('sjr_score') or journal.get('sjr_score', 0) <= 0):
                continue
            
            filtered.append(journal)
        
        logger.info(f"Filtered {len(journals)} journals to {len(filtered)} high-quality journals")
        return filtered
    
    def optimize_for_embedding(self, journals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize journal data for embedding generation.
        
        Args:
            journals: List of journal dictionaries
            
        Returns:
            Optimized journal list
        """
        optimized = []
        
        for journal in journals:
            # Enhance semantic fingerprint
            if not journal.get('semantic_fingerprint'):
                continue
            
            # Add ranking context to fingerprint
            fingerprint = journal['semantic_fingerprint']
            
            # Add quartile information
            if journal.get('sjr_quartile'):
                fingerprint += f" | Journal quality: {journal['sjr_quartile']} quartile"
            
            # Add ranking context
            if journal.get('scimago_rank') and journal['scimago_rank'] <= 100:
                fingerprint += f" | Top 100 journal (rank {journal['scimago_rank']})"
            elif journal.get('scimago_rank') and journal['scimago_rank'] <= 500:
                fingerprint += f" | Top 500 journal (rank {journal['scimago_rank']})"
            
            # Add citation context
            if journal.get('h_index') and journal['h_index'] > 100:
                fingerprint += f" | High impact journal (H-index: {journal['h_index']})"
            
            journal['semantic_fingerprint'] = fingerprint
            optimized.append(journal)
        
        return optimized

# Usage Example
if __name__ == "__main__":
    processor = CSVDataProcessor()
    
    # Example: Generate quality report
    sample_journals = [
        {
            'display_name': 'Test Journal',
            'issn': ['1234-5678'],
            'publisher': 'Test Publisher',
            'scimago_rank': 1,
            'sjr_score': 10.5,
            'h_index': 150,
            'works_count': 1000,
            'subjects': [{'name': 'Medicine'}]
        }
    ]
    
    report = processor.generate_quality_report(sample_journals)
    print(f"Quality report generated for {report['total_journals']} journals")
    print(f"Quality score: {report['quality_metrics']['quality_score']:.1f}%")
```

### Step 5: Create Test Suite

**File**: `tests/test_csv_integration.py`

```python
#!/usr/bin/env python3
"""
Test suite for CSV integration functionality.

Tests CSV parsing, data processing, schema mapping,
and integration with existing system.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import sys
from unittest.mock import Mock, patch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from csv_journal_importer import CSVJournalImporter
from csv_schema_mapper import CSVSchemaMapper
from csv_data_processor import CSVDataProcessor

class TestCSVJournalImporter:
    """Test CSV journal importer functionality."""
    
    def create_test_csv(self, content: str) -> str:
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(content)
            return f.name
    
    def test_csv_loading(self):
        """Test basic CSV loading functionality."""
        csv_content = '''Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index
1;28773;"Test Journal";journal;"1234-5678";10,5;Q1;100
2;28774;"Another Journal";journal;"8765-4321";5,2;Q2;50'''
        
        csv_file = self.create_test_csv(csv_content)
        importer = CSVJournalImporter(csv_file)
        
        df = importer.load_csv()
        
        assert len(df) == 2
        assert 'Title' in df.columns
        assert df.iloc[0]['Title'] == '"Test Journal"'
        
        Path(csv_file).unlink()  # Cleanup
    
    def test_data_cleaning(self):
        """Test data cleaning and validation."""
        csv_content = '''Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index
1;28773;"Test Journal";journal;"1234-5678";10,5;Q1;100
2;28774;"Discontinued Journal (discontinued)";journal;"8765-4321";5,2;Q2;50
3;28775;"No ISSN Journal";journal;"";3,1;Q3;25'''
        
        csv_file = self.create_test_csv(csv_content)
        importer = CSVJournalImporter(csv_file)
        
        importer.load_csv()
        cleaned_df = importer.clean_and_validate()
        
        # Should filter out discontinued and no-ISSN journals
        assert len(cleaned_df) == 1
        assert cleaned_df.iloc[0]['Title'] == '"Test Journal"'
        
        Path(csv_file).unlink()  # Cleanup
    
    def test_chunking(self):
        """Test data chunking functionality."""
        csv_content = '''Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index
1;1;"Journal 1";journal;"1111-1111";10,5;Q1;100
2;2;"Journal 2";journal;"2222-2222";5,2;Q2;50
3;3;"Journal 3";journal;"3333-3333";3,1;Q3;25
4;4;"Journal 4";journal;"4444-4444";2,5;Q4;15
5;5;"Journal 5";journal;"5555-5555";1,8;Q4;10'''
        
        csv_file = self.create_test_csv(csv_content)
        importer = CSVJournalImporter(csv_file)
        
        importer.load_csv()
        importer.clean_and_validate()
        chunks = importer.get_processed_chunks(chunk_size=2)
        
        assert len(chunks) == 3  # 5 journals in chunks of 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1
        
        Path(csv_file).unlink()  # Cleanup
    
    def test_statistics(self):
        """Test statistics generation."""
        csv_content = '''Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index
1;1;"Journal 1";journal;"1111-1111";10,5;Q1;100
2;2;"Journal 2";journal;"2222-2222";5,2;Q1;50
3;3;"Journal 3";journal;"3333-3333";3,1;Q2;25'''
        
        csv_file = self.create_test_csv(csv_content)
        importer = CSVJournalImporter(csv_file)
        
        importer.load_csv()
        importer.clean_and_validate()
        stats = importer.get_statistics()
        
        assert stats['total_journals'] == 3
        assert stats['q1_journals'] == 2
        assert stats['q2_journals'] == 1
        assert stats['top_100'] == 3
        
        Path(csv_file).unlink()  # Cleanup


class TestCSVSchemaMapper:
    """Test CSV to database schema mapping."""
    
    def test_basic_mapping(self):
        """Test basic journal mapping."""
        mapper = CSVSchemaMapper()
        
        csv_row = {
            'Rank': '1',
            'Sourceid': '28773',
            'Title': '"Test Journal"',
            'Issn': '"1234-5678, 8765-4321"',
            'SJR': '10.5',
            'SJR Best Quartile': 'Q1',
            'H index': '100',
            'Publisher': '"Test Publisher"',
            'Country': 'United States',
            'Categories': '"Medicine (Q1); Biology (Q2)"'
        }
        
        journal = mapper.map_journal(csv_row)
        
        assert journal is not None
        assert journal['display_name'] == 'Test Journal'
        assert journal['scimago_rank'] == 1
        assert journal['sjr_score'] == 10.5
        assert journal['sjr_quartile'] == 'Q1'
        assert len(journal['issn']) == 2
        assert journal['issn_l'] == '1234-5678'
        assert len(journal['subjects']) == 2
    
    def test_issn_processing(self):
        """Test ISSN processing and formatting."""
        mapper = CSVSchemaMapper()
        
        # Test multiple ISSNs
        issn_data = mapper._process_issns('"12345678, 87654321"')
        assert len(issn_data['issn_list']) == 2
        assert issn_data['issn_list'][0] == '1234-5678'
        assert issn_data['issn_list'][1] == '8765-4321'
        assert issn_data['issn_l'] == '1234-5678'
        
        # Test empty ISSN
        issn_data = mapper._process_issns('')
        assert issn_data['issn_list'] == []
        assert issn_data['issn_l'] is None
    
    def test_categories_processing(self):
        """Test categories processing with quartiles."""
        mapper = CSVSchemaMapper()
        
        categories = mapper._process_categories('"Medicine (Q1); Biology (Q2); Chemistry"')
        
        assert len(categories) == 3
        assert categories[0]['name'] == 'Medicine'
        assert categories[0]['quartile'] == 'Q1'
        assert categories[0]['score'] == 0.9
        
        assert categories[1]['name'] == 'Biology'
        assert categories[1]['quartile'] == 'Q2'
        assert categories[1]['score'] == 0.7
        
        assert categories[2]['name'] == 'Chemistry'
        assert categories[2]['quartile'] is None
        assert categories[2]['score'] == 0.5
    
    def test_semantic_fingerprint_creation(self):
        """Test semantic fingerprint creation."""
        mapper = CSVSchemaMapper()
        
        journal = {
            'display_name': 'Test Medical Journal',
            'publisher': 'Test Publisher',
            'scimago_rank': 1,
            'sjr_quartile': 'Q1',
            'subjects': [{'name': 'Medicine'}, {'name': 'Biology'}],
            'country': 'United States',
            'areas': ['Medicine', 'Life Sciences'],
            'h_index': 150,
            'sjr_score': 25.5
        }
        
        fingerprint = mapper.create_semantic_fingerprint(journal)
        
        assert 'Test Medical Journal' in fingerprint
        assert 'Test Publisher' in fingerprint
        assert 'Scimago Rank: 1' in fingerprint
        assert 'SJR Quartile: Q1' in fingerprint
        assert 'Medicine, Biology' in fingerprint
        assert 'H-index: 150' in fingerprint


class TestCSVDataProcessor:
    """Test CSV data processing and validation."""
    
    def test_journal_validation(self):
        """Test journal quality validation."""
        processor = CSVDataProcessor()
        
        # Valid journal
        valid_journal = {
            'display_name': 'Test Journal',
            'issn': ['1234-5678'],
            'publisher': 'Test Publisher',
            'scimago_rank': 1,
            'sjr_score': 10.5,
            'subjects': [{'name': 'Medicine'}],
            'works_count': 100,
            'h_index': 50
        }
        
        is_valid, issues = processor.validate_journal_quality(valid_journal)
        assert is_valid
        assert len([i for i in issues if 'CRITICAL' in i]) == 0
        
        # Invalid journal (missing critical fields)
        invalid_journal = {
            'display_name': '',  # Missing title
            'issn': [],  # Missing ISSN
            'scimago_rank': -1  # Invalid rank
        }
        
        is_valid, issues = processor.validate_journal_quality(invalid_journal)
        assert not is_valid
        assert len([i for i in issues if 'CRITICAL' in i]) > 0
    
    def test_duplicate_detection(self):
        """Test duplicate journal detection."""
        processor = CSVDataProcessor()
        
        journals = [
            {
                'display_name': 'Test Journal',
                'issn': ['1234-5678']
            },
            {
                'display_name': 'Different Journal',
                'issn': ['8765-4321']
            },
            {
                'display_name': 'Test Journal',  # Same title
                'issn': ['9999-9999']
            },
            {
                'display_name': 'Another Journal',
                'issn': ['1234-5678']  # Same ISSN
            }
        ]
        
        duplicates = processor.detect_duplicates(journals)
        
        # Should find duplicates based on both title and ISSN
        assert len(duplicates) > 0
    
    def test_quality_report_generation(self):
        """Test quality report generation."""
        processor = CSVDataProcessor()
        
        journals = [
            {
                'display_name': 'Complete Journal',
                'issn': ['1234-5678'],
                'publisher': 'Publisher 1',
                'scimago_rank': 1,
                'sjr_score': 10.5,
                'sjr_quartile': 'Q1',
                'h_index': 100,
                'works_count': 500,
                'country': 'United States',
                'subjects': [{'name': 'Medicine'}]
            },
            {
                'display_name': 'Incomplete Journal',
                'issn': ['8765-4321'],
                # Missing several fields
                'scimago_rank': 2,
                'sjr_quartile': 'Q2'
            }
        ]
        
        report = processor.generate_quality_report(journals)
        
        assert report['total_journals'] == 2
        assert 'quality_metrics' in report
        assert 'data_completeness' in report
        assert 'distribution_analysis' in report
        assert 'recommendations' in report
        
        # Check completeness metrics
        assert report['data_completeness']['display_name']['count'] == 2
        assert report['data_completeness']['publisher']['count'] == 1
    
    def test_high_quality_filtering(self):
        """Test high-quality journal filtering."""
        processor = CSVDataProcessor()
        
        journals = [
            {
                'display_name': 'High Quality Journal',
                'issn': ['1234-5678'],
                'works_count': 1000,
                'h_index': 100,
                'sjr_score': 15.5
            },
            {
                'display_name': 'Low Quality Journal',
                'issn': ['8765-4321'],
                'works_count': 10,  # Too low
                'h_index': 5,       # Too low
                'sjr_score': 0.5
            },
            {
                'display_name': 'No SJR Journal',
                'issn': ['9999-9999'],
                'works_count': 200,
                'h_index': 50
                # Missing SJR score
            }
        ]
        
        filtered = processor.filter_high_quality_journals(
            journals, 
            min_works=50, 
            min_h_index=20, 
            require_sjr=True
        )
        
        # Should only include the high quality journal
        assert len(filtered) == 1
        assert filtered[0]['display_name'] == 'High Quality Journal'


@pytest.mark.integration
class TestCSVIntegrationWorkflow:
    """Integration tests for complete CSV workflow."""
    
    def test_complete_workflow(self):
        """Test complete CSV processing workflow."""
        # Create test CSV
        csv_content = '''Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Publisher;Country;Categories
1;1;"Medical Journal";journal;"1234-5678";15,5;Q1;150;"Test Publisher";"United States";"Medicine (Q1)"
2;2;"Biology Journal";journal;"8765-4321";8,2;Q1;100;"Bio Publisher";"United Kingdom";"Biology (Q1)"'''
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            csv_file = f.name
        
        try:
            # Step 1: Import CSV
            importer = CSVJournalImporter(csv_file)
            importer.load_csv()
            importer.clean_and_validate()
            
            # Step 2: Get chunks
            chunks = importer.get_processed_chunks(chunk_size=10)
            assert len(chunks) == 1
            
            # Step 3: Map to schema
            mapper = CSVSchemaMapper()
            mapped_journals = []
            
            for chunk in chunks:
                for _, row in chunk.iterrows():
                    journal = mapper.map_journal(row.to_dict())
                    if journal:
                        fingerprint = mapper.create_semantic_fingerprint(journal)
                        journal['semantic_fingerprint'] = fingerprint
                        mapped_journals.append(journal)
            
            assert len(mapped_journals) == 2
            
            # Step 4: Process and validate
            processor = CSVDataProcessor()
            report = processor.generate_quality_report(mapped_journals)
            
            assert report['total_journals'] == 2
            assert report['quality_metrics']['quality_score'] > 80
            
            # Step 5: Optimize for embeddings
            optimized = processor.optimize_for_embedding(mapped_journals)
            assert len(optimized) == 2
            
            # Verify semantic fingerprints are enhanced
            for journal in optimized:
                assert 'semantic_fingerprint' in journal
                assert len(journal['semantic_fingerprint']) > 0
                if journal.get('sjr_quartile') == 'Q1':
                    assert 'Q1 quartile' in journal['semantic_fingerprint']
        
        finally:
            Path(csv_file).unlink()  # Cleanup


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Step 6: Usage Instructions

**Commands to run the CSV integration:**

```bash
# Navigate to project directory
cd /Users/aryanpathak/Desktop/manuscript-journal-matcher

# Activate virtual environment
source venv/bin/activate

# Install any additional dependencies
pip install chardet tqdm

# Build database from CSV (with DOAJ enrichment)
python scripts/build_database.py \
    --csv-file "Medicine Journal Rankings 2024.csv" \
    --csv-chunk-size 500 \
    --doaj-rate-limit 1.0

# Build database from CSV (without DOAJ - faster)
python scripts/build_database.py \
    --csv-file "Medicine Journal Rankings 2024.csv" \
    --csv-chunk-size 1000 \
    --skip-doaj

# Build database with high-quality filtering
python scripts/build_database.py \
    --csv-file "Medicine Journal Rankings 2024.csv" \
    --csv-chunk-size 500 \
    --limit 5000  # Process only top 5000 journals

# Test the CSV integration
pytest tests/test_csv_integration.py -v

# Run the application with new database
streamlit run src/main.py
```

### Expected Timeline

- **Implementation**: 6-8 hours of development time
- **Database Building**: 8-12 hours (with DOAJ enrichment for 7,678 journals)
- **Testing & Validation**: 2-3 hours
- **Total**: 16-23 hours

### Expected Results

After successful implementation, you will have:
- **6,000-7,000 high-quality medical journals** in the database
- **Enhanced search capabilities** with SJR ranking and quartile filtering
- **Geographic and publisher-based filtering**
- **Comprehensive metadata** including citation metrics and quality indicators
- **Production-ready system** capable of providing excellent journal recommendations

This implementation will transform the Manuscript Journal Matcher from a prototype into a fully functional, production-ready journal recommendation system specifically optimized for medical research.