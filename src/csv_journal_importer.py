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
                         'Total Refs.', 'Total Cites (3years)', 'Citable Docs. (3years)', 
                         'Cites / Doc. (2years)', 'Ref. / Doc.', '%Female', 'Overton', 'SDG']
        
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
            "top_1000": len(df[df['Rank'] <= 1000]),
            "with_sjr_score": len(df[df['SJR'].notna()]),
            "countries": df['Country'].nunique(),
            "publishers": df['Publisher'].nunique(),
            "avg_h_index": df['H index'].mean(),
            "avg_sjr_score": df['SJR'].mean(),
            "median_works_count": df['Total Docs. (2024)'].median(),
            "high_impact_journals": len(df[df['H index'] > 100]),
            "recent_journals": len(df[df['Total Docs. (2024)'] > 50])
        }

# Usage Example
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    importer = CSVJournalImporter("Medicine Journal Rankings 2024.csv")
    
    # Load and process data
    raw_df = importer.load_csv()
    processed_df = importer.clean_and_validate()
    
    # Get chunks for processing
    chunks = importer.get_processed_chunks(chunk_size=500)
    
    # Display statistics
    stats = importer.get_statistics()
    print(f"Processing {stats['total_journals']} journals in {len(chunks)} chunks")
    print(f"Q1 journals: {stats['q1_journals']}")
    print(f"Top 100: {stats['top_100']}")
    print(f"Countries: {stats['countries']}")
    print(f"Publishers: {stats['publishers']}")