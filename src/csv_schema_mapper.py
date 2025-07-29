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
    
    def map_journal(self, csv_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            
            # Process areas
            areas = self._process_areas(csv_row.get('Areas', ''))
            
            # Create base journal record
            journal = {
                # Basic identification
                'id': f"CSV_{csv_row.get('Sourceid', '')}",
                'display_name': self._clean_text(csv_row.get('Title', '')),
                'issn': issn_data['issn_list'],
                'issn_l': issn_data['issn_l'],
                
                # Publisher information
                'publisher': self._clean_text(csv_row.get('Publisher', '')),
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
                'works_count_3y': self._safe_int(csv_row.get('Total Docs. (3years)', 0)),
                'cited_by_count': self._safe_int(csv_row.get('Total Cites (3years)', 0)),
                'h_index': self._safe_int(csv_row.get('H index', 0)),
                'cites_per_doc': self._safe_float(csv_row.get('Cites / Doc. (2years)')),
                'refs_per_doc': self._safe_float(csv_row.get('Ref. / Doc.')),
                'total_refs': self._safe_int(csv_row.get('Total Refs.')),
                'citable_docs_3y': self._safe_int(csv_row.get('Citable Docs. (3years)')),
                
                # Additional metrics
                'female_percentage': self._safe_float(csv_row.get('%Female')),
                'overton_score': self._safe_int(csv_row.get('Overton')),
                'sdg_score': self._safe_int(csv_row.get('SDG')),
                
                # Geographic information
                'country': csv_row.get('Country', ''),
                'region': csv_row.get('Region', ''),
                
                # Coverage and areas
                'coverage_years': csv_row.get('Coverage', ''),
                'areas': areas,
                
                # Subjects (from categories)
                'subjects': categories,
                
                # Placeholders for DOAJ integration (filled later)
                'is_oa': None,  # Legacy field
                'is_in_doaj': None,  # Legacy field
                'oa_status': None,
                'in_doaj': None,
                'has_apc': None,
                'apc_amount': None,
                'apc_currency': None,
                'apc_usd': None,  # Legacy field
                'oa_start_year': None,
                'subjects_doaj': [],
                'languages': [],
                'license_type': [],
                'publisher_doaj': None,
                'country_doaj': None,
                'doaj_fetched_at': None,
                
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
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text field."""
        if not text:
            return ""
        
        # Remove surrounding quotes
        text = str(text).strip('"').strip("'")
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _process_issns(self, issn_str: str) -> Dict[str, Any]:
        """Process ISSN string into structured format."""
        if not issn_str or str(issn_str).strip() == '':
            return {'issn_list': [], 'issn_l': None}
        
        # Remove quotes and split by comma
        issn_str = self._clean_text(str(issn_str))
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
        categories_str = self._clean_text(str(categories_str))
        
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
    
    def _process_areas(self, areas_str: str) -> List[str]:
        """Process areas string into list."""
        if not areas_str or str(areas_str).strip() == '':
            return []
        
        areas_str = self._clean_text(str(areas_str))
        areas = [area.strip() for area in areas_str.split(';') if area.strip()]
        return areas
    
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
        # Comprehensive country mapping
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
            'Finland': 'FI',
            'Belgium': 'BE',
            'Austria': 'AT',
            'Poland': 'PL',
            'Portugal': 'PT',
            'Israel': 'IL',
            'South Africa': 'ZA',
            'Mexico': 'MX',
            'Argentina': 'AR',
            'Chile': 'CL',
            'Colombia': 'CO',
            'Turkey': 'TR',
            'Iran': 'IR',
            'Egypt': 'EG',
            'Saudi Arabia': 'SA',
            'United Arab Emirates': 'AE',
            'Singapore': 'SG',
            'Malaysia': 'MY',
            'Thailand': 'TH',
            'New Zealand': 'NZ',
            'Ireland': 'IE',
            'Greece': 'GR',
            'Czech Republic': 'CZ',
            'Hungary': 'HU',
            'Romania': 'RO',
            'Croatia': 'HR',
            'Slovenia': 'SI',
            'Slovakia': 'SK',
            'Lithuania': 'LT',
            'Latvia': 'LV',
            'Estonia': 'EE'
        }
        return country_mapping.get(country, None)
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to integer."""
        try:
            if value is None or str(value).strip() == '' or str(value).lower() == 'nan':
                return None
            return int(float(str(value)))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        try:
            if value is None or str(value).strip() == '' or str(value).lower() == 'nan':
                return None
            return float(str(value))
        except (ValueError, TypeError):
            return None
    
    def create_semantic_fingerprint(self, journal: Dict[str, Any]) -> str:
        """Create enhanced semantic fingerprint for CSV journal."""
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
            
            # Add ranking context
            if journal['scimago_rank'] <= 10:
                parts.append("Top 10 journal")
            elif journal['scimago_rank'] <= 50:
                parts.append("Top 50 journal")
            elif journal['scimago_rank'] <= 100:
                parts.append("Top 100 journal")
            elif journal['scimago_rank'] <= 500:
                parts.append("Top 500 journal")
            elif journal['scimago_rank'] <= 1000:
                parts.append("Top 1000 journal")
        
        if journal.get('sjr_quartile'):
            parts.append(f"SJR Quartile: {journal['sjr_quartile']}")
            
            # Add quality context
            if journal['sjr_quartile'] == 'Q1':
                parts.append("First quartile journal")
            elif journal['sjr_quartile'] == 'Q2':
                parts.append("Second quartile journal")
            elif journal['sjr_quartile'] == 'Q3':
                parts.append("Third quartile journal")
            elif journal['sjr_quartile'] == 'Q4':
                parts.append("Fourth quartile journal")
        
        # Subject areas with quartile information
        if journal.get('subjects'):
            subject_parts = []
            for subject in journal['subjects'][:3]:  # Top 3 subjects
                if subject.get('quartile'):
                    subject_parts.append(f"{subject['name']} ({subject['quartile']})")
                else:
                    subject_parts.append(subject['name'])
            if subject_parts:
                parts.append(f"Subject areas: {', '.join(subject_parts)}")
        
        # Geographic information
        if journal.get('country'):
            parts.append(f"Country: {journal['country']}")
        
        # Research areas
        if journal.get('areas'):
            areas = ', '.join(journal['areas'][:3])
            parts.append(f"Research areas: {areas}")
        
        # Quality indicators
        if journal.get('h_index') and journal['h_index'] > 0:
            parts.append(f"H-index: {journal['h_index']}")
            
            # Add impact context
            if journal['h_index'] >= 200:
                parts.append("Very high impact journal")
            elif journal['h_index'] >= 100:
                parts.append("High impact journal")
            elif journal['h_index'] >= 50:
                parts.append("Medium impact journal")
            elif journal['h_index'] >= 20:
                parts.append("Moderate impact journal")
        
        if journal.get('sjr_score') and journal['sjr_score'] > 0:
            parts.append(f"SJR score: {journal['sjr_score']}")
            
            # Add SJR context
            if journal['sjr_score'] >= 10:
                parts.append("Very high SJR score")
            elif journal['sjr_score'] >= 5:
                parts.append("High SJR score")
            elif journal['sjr_score'] >= 2:
                parts.append("Good SJR score")
        
        # Publication activity
        if journal.get('works_count') and journal['works_count'] > 0:
            if journal['works_count'] >= 1000:
                parts.append("High publication volume")
            elif journal['works_count'] >= 500:
                parts.append("Medium publication volume")
        
        return ' | '.join(parts)
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about the mapping process."""
        return self.mapping_stats.copy()

# Usage Example
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    mapper = CSVSchemaMapper()
    
    # Example CSV row (simplified)
    csv_row = {
        'Rank': '1',
        'Sourceid': '28773',
        'Title': '"Ca-A Cancer Journal for Clinicians"',
        'Issn': '"15424863, 00079235"',
        'SJR': '145.004',
        'SJR Best Quartile': 'Q1',
        'H index': '223',
        'Publisher': '"John Wiley and Sons Inc"',
        'Country': 'United States',
        'Categories': '"Hematology (Q1); Oncology (Q1)"',
        'Areas': '"Medicine"'
    }
    
    # Map to database schema
    journal = mapper.map_journal(csv_row)
    
    if journal:
        # Create semantic fingerprint
        fingerprint = mapper.create_semantic_fingerprint(journal)
        journal['semantic_fingerprint'] = fingerprint
        
        print(f"Mapped journal: {journal['display_name']}")
        print(f"ISSNs: {journal['issn']}")
        print(f"Rank: {journal['scimago_rank']}")
        print(f"Subjects: {len(journal['subjects'])}")
        print(f"Fingerprint: {fingerprint[:100]}...")
    
    stats = mapper.get_mapping_stats()
    print(f"Mapping stats: {stats}")