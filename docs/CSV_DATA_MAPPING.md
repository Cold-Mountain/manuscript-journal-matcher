# CSV Data Mapping Documentation

**Document Version**: 1.0  
**Date**: July 28, 2025  
**Purpose**: Define complete mapping between Medicine Journal Rankings 2024.csv and database schema

## Overview

This document provides detailed specifications for mapping the 24 columns from the Medicine Journal Rankings 2024 CSV file to the existing journal database schema with DOAJ integration support.

## CSV Source Data Structure

### File Format
- **File**: Medicine Journal Rankings 2024.csv
- **Separator**: Semicolon (`;`)
- **Total Records**: 7,678 journals
- **Encoding**: UTF-8
- **Decimal Format**: European (comma as decimal separator)

### Complete Column Mapping

| CSV Position | CSV Column | CSV Type | Database Field | Database Type | Transformation Notes |
|--------------|------------|----------|----------------|---------------|---------------------|
| 1 | Rank | Integer | `scimago_rank` | Integer | Direct mapping |
| 2 | Sourceid | String | `sourceid` | String | Scopus source identifier |
| 3 | Title | String | `display_name` | String | Remove surrounding quotes |
| 4 | Type | String | `type` | String | Usually "journal" |
| 5 | Issn | String | `issn`, `issn_l` | List[String], String | Parse multiple ISSNs, format as XXXX-XXXX |
| 6 | SJR | Float | `sjr_score` | Float | Convert comma decimal to dot |
| 7 | SJR Best Quartile | String | `sjr_quartile` | String | Q1, Q2, Q3, Q4 values |
| 8 | H index | Integer | `h_index` | Integer | Citation impact metric |
| 9 | Total Docs. (2024) | Integer | `works_count` | Integer | Current year publications |
| 10 | Total Docs. (3years) | Integer | `works_count_3y` | Integer | 3-year publication count |
| 11 | Total Refs. | Integer | `total_refs` | Integer | Total references |
| 12 | Total Cites (3years) | Integer | `cited_by_count` | Integer | 3-year citation count |
| 13 | Citable Docs. (3years) | Integer | `citable_docs_3y` | Integer | Citable documents count |
| 14 | Cites / Doc. (2years) | Float | `cites_per_doc` | Float | Citation ratio |
| 15 | Ref. / Doc. | Float | `refs_per_doc` | Float | Reference ratio |
| 16 | %Female | Float | `female_percentage` | Float | Female author percentage |
| 17 | Overton | Integer | `overton_score` | Integer | Policy impact score |
| 18 | SDG | Integer | `sdg_score` | Integer | SDG alignment score |
| 19 | Country | String | `country` | String | Publisher country |
| 20 | Region | String | `region` | String | Geographic region |
| 21 | Publisher | String | `publisher` | String | Remove surrounding quotes |
| 22 | Coverage | String | `coverage_years` | String | Publication year range |
| 23 | Categories | String | `subjects` | List[Dict] | Parse categories with quartiles |
| 24 | Areas | String | `areas` | List[String] | Split by semicolon |

## Detailed Field Mappings

### 1. Basic Identification Fields

#### Journal Title (CSV Column 3 → `display_name`)
```python
# CSV: '"Ca-A Cancer Journal for Clinicians"'
# Transform: Remove quotes, normalize whitespace
display_name = csv_row['Title'].strip('"').strip()
# Result: "Ca-A Cancer Journal for Clinicians"
```

#### ISSN Processing (CSV Column 5 → `issn`, `issn_l`)
```python
# CSV: '"15424863, 00079235"'
# Transform: Split, format, validate
issn_string = csv_row['Issn'].strip('"')
issns = [issn.strip() for issn in issn_string.split(',')]

formatted_issns = []
for issn in issns:
    # Remove any existing formatting
    clean_issn = re.sub(r'[^\d]', '', issn)
    if len(clean_issn) == 8:
        # Format as XXXX-XXXX
        formatted_issn = f"{clean_issn[:4]}-{clean_issn[4:]}"
        formatted_issns.append(formatted_issn)

# Result: 
# issn = ["1542-4863", "0007-9235"]
# issn_l = "1542-4863"  # First ISSN as linking ISSN
```

### 2. Ranking and Quality Metrics

#### Scimago Ranking (CSV Column 1 → `scimago_rank`)
```python
# CSV: "1", "2", "3"
# Transform: Convert to integer
scimago_rank = int(csv_row['Rank'])
# Result: 1, 2, 3
```

#### SJR Score (CSV Column 6 → `sjr_score`)
```python
# CSV: "145,004" (European decimal format)
# Transform: Replace comma with dot, convert to float
sjr_score = float(csv_row['SJR'].replace(',', '.'))
# Result: 145.004
```

#### SJR Quartile (CSV Column 7 → `sjr_quartile`)
```python
# CSV: "Q1", "Q2", "Q3", "Q4"
# Transform: Direct mapping
sjr_quartile = csv_row['SJR Best Quartile'].strip()
# Result: "Q1"
```

### 3. Citation Metrics

#### H-Index (CSV Column 8 → `h_index`)
```python
# CSV: "223", "155"
# Transform: Convert to integer
h_index = int(csv_row['H index'])
# Result: 223
```

#### Citation Metrics (Columns 12, 14, 15)
```python
# Total Citations (3 years)
cited_by_count = int(csv_row['Total Cites (3years)'])

# Citations per Document (2 years) - European decimal format
cites_per_doc = float(csv_row['Cites / Doc. (2years)'].replace(',', '.'))

# References per Document - European decimal format  
refs_per_doc = float(csv_row['Ref. / Doc.'].replace(',', '.'))
```

### 4. Subject Classifications

#### Categories Processing (CSV Column 23 → `subjects`)
```python
# CSV: '"Hematology (Q1); Oncology (Q1); Medicine (miscellaneous) (Q2)"'
# Transform: Parse categories with quartile information

def parse_categories(categories_str):
    if not categories_str:
        return []
    
    categories = []
    categories_str = categories_str.strip('"')
    
    for cat in categories_str.split(';'):
        cat = cat.strip()
        if not cat:
            continue
        
        # Extract quartile: "Oncology (Q1)" -> name="Oncology", quartile="Q1"
        quartile_match = re.search(r'\((Q[1-4])\)', cat)
        if quartile_match:
            quartile = quartile_match.group(1)
            name = cat.replace(f'({quartile})', '').strip()
            score = quartile_to_score(quartile)  # Q1=0.9, Q2=0.7, Q3=0.5, Q4=0.3
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

# Result:
# subjects = [
#     {'name': 'Hematology', 'score': 0.9, 'quartile': 'Q1'},
#     {'name': 'Oncology', 'score': 0.9, 'quartile': 'Q1'},
#     {'name': 'Medicine (miscellaneous)', 'score': 0.7, 'quartile': 'Q2'}
# ]
```

#### Areas Processing (CSV Column 24 → `areas`)
```python
# CSV: '"Medicine; Environmental Science; Health Professions"'
# Transform: Split by semicolon, clean whitespace
areas_str = csv_row['Areas'].strip('"')
areas = [area.strip() for area in areas_str.split(';') if area.strip()]
# Result: ["Medicine", "Environmental Science", "Health Professions"]
```

### 5. Geographic and Publisher Information

#### Country Mapping (CSV Column 19 → `country`, `country_code`)
```python
# CSV: "United States", "United Kingdom", "Germany"
# Transform: Direct mapping + country code lookup

country_code_mapping = {
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
    # ... extend as needed
}

country = csv_row['Country']
country_code = country_code_mapping.get(country, None)
```

#### Publisher Processing (CSV Column 21 → `publisher`)
```python
# CSV: '"John Wiley and Sons Inc"'
# Transform: Remove quotes, normalize
publisher = csv_row['Publisher'].strip('"').strip()
# Result: "John Wiley and Sons Inc"
```

### 6. Additional Metrics

#### Female Author Percentage (CSV Column 16 → `female_percentage`)
```python
# CSV: "48,21" (European decimal format)
# Transform: Convert decimal format
female_percentage = float(csv_row['%Female'].replace(',', '.'))
# Result: 48.21
```

#### Impact Scores (Columns 17, 18)
```python
# Overton Score (policy impact)
overton_score = int(csv_row['Overton']) if csv_row['Overton'] else None

# SDG Score (Sustainable Development Goals alignment)
sdg_score = int(csv_row['SDG']) if csv_row['SDG'] else None
```

## Integration with Existing Schema

### Preserving DOAJ Integration Fields

The CSV mapping preserves all existing DOAJ integration fields, which will be populated during the enrichment phase:

```python
# DOAJ fields (populated later during enrichment)
journal_record = {
    # ... CSV mapped fields above ...
    
    # DOAJ fields (initially None, filled during enrichment)
    'oa_status': None,           # Will be determined by DOAJ lookup
    'in_doaj': None,             # DOAJ directory status
    'has_apc': None,             # Article Processing Charge status
    'apc_amount': None,          # APC amount in original currency
    'apc_currency': None,        # APC currency
    'oa_start_year': None,       # Open access start year
    'subjects_doaj': [],         # DOAJ subject classifications
    'languages': [],             # Publication languages
    'license_type': [],          # License types (CC BY, etc.)
    'publisher_doaj': None,      # DOAJ publisher information
    'country_doaj': None,        # DOAJ country information
    'doaj_fetched_at': None,     # DOAJ enrichment timestamp
    
    # System fields
    'csv_source': True,          # Indicates CSV origin
    'csv_imported_at': datetime.now().isoformat()
}
```

### Enhanced Semantic Fingerprint

The semantic fingerprint creation is enhanced to include CSV-specific ranking information:

```python
def create_enhanced_semantic_fingerprint(journal):
    parts = []
    
    # Basic journal information
    if journal.get('display_name'):
        parts.append(f"Journal: {journal['display_name']}")
    
    if journal.get('publisher'):
        parts.append(f"Publisher: {journal['publisher']}")
    
    # CSV-specific enhancements
    if journal.get('scimago_rank'):
        parts.append(f"Scimago Rank: {journal['scimago_rank']}")
    
    if journal.get('sjr_quartile'):
        parts.append(f"SJR Quartile: {journal['sjr_quartile']}")
        
        # Add quality context
        if journal['sjr_quartile'] == 'Q1':
            parts.append("Top quartile journal")
        elif journal['sjr_quartile'] == 'Q2':
            parts.append("Second quartile journal")
    
    # Subject areas with quartile information
    if journal.get('subjects'):
        subject_parts = []
        for subject in journal['subjects'][:3]:  # Top 3 subjects
            if subject.get('quartile'):
                subject_parts.append(f"{subject['name']} ({subject['quartile']})")
            else:
                subject_parts.append(subject['name'])
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
        if journal['h_index'] > 100:
            parts.append("High impact journal")
        elif journal['h_index'] > 50:
            parts.append("Medium impact journal")
    
    if journal.get('sjr_score') and journal['sjr_score'] > 0:
        parts.append(f"SJR score: {journal['sjr_score']}")
    
    # Ranking context
    if journal.get('scimago_rank'):
        if journal['scimago_rank'] <= 100:
            parts.append("Top 100 journal")
        elif journal['scimago_rank'] <= 500:
            parts.append("Top 500 journal")
        elif journal['scimago_rank'] <= 1000:
            parts.append("Top 1000 journal")
    
    return ' | '.join(parts)
```

## Data Quality Considerations

### Validation Rules

#### Critical Fields (Must be Present)
```python
CRITICAL_FIELDS = {
    'display_name': 'Journal title is required',
    'issn': 'At least one ISSN is required',
    'scimago_rank': 'Scimago ranking is required'
}
```

#### Quality Thresholds
```python
QUALITY_THRESHOLDS = {
    'min_works_count': 10,       # Minimum publications
    'min_h_index': 5,            # Minimum H-index
    'max_rank': 10000,           # Maximum acceptable rank
    'min_sjr_score': 0.1         # Minimum SJR score
}
```

### Data Cleaning Rules

#### Text Normalization
```python
def normalize_text_field(text):
    if not text:
        return None
    
    # Remove quotes
    text = text.strip('"').strip("'")
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle empty strings
    return text if text else None
```

#### Numeric Field Conversion
```python
def convert_european_decimal(value_str):
    """Convert European decimal format (comma) to standard (dot)."""
    if not value_str or str(value_str).strip() == '':
        return None
    
    try:
        # Replace comma with dot for decimal conversion
        normalized = str(value_str).replace(',', '.')
        return float(normalized)
    except (ValueError, TypeError):
        return None
```

## Complete Mapping Example

Here's a complete example showing the transformation of one CSV row to database format:

### Input CSV Row
```csv
1;28773;"Ca-A Cancer Journal for Clinicians";journal;"15424863, 00079235";145,004;Q1;223;43;122;2704;40834;81;168,71;62,88;48,21;4;37;United States;Northern America;"John Wiley and Sons Inc";"1950-2025";"Hematology (Q1); Oncology (Q1)";"Medicine"
```

### Output Database Record
```json
{
  "id": "CSV_28773",
  "display_name": "Ca-A Cancer Journal for Clinicians",
  "issn": ["1542-4863", "0007-9235"],
  "issn_l": "1542-4863",
  "type": "journal",
  
  "publisher": "John Wiley and Sons Inc",
  "country": "United States",
  "country_code": "US",
  "region": "Northern America",
  "coverage_years": "1950-2025",
  
  "scimago_rank": 1,
  "sourceid": "28773",
  "sjr_score": 145.004,
  "sjr_quartile": "Q1",
  
  "h_index": 223,
  "works_count": 43,
  "works_count_3y": 122,
  "cited_by_count": 40834,
  "citable_docs_3y": 81,
  "cites_per_doc": 168.71,
  "refs_per_doc": 62.88,
  
  "female_percentage": 48.21,
  "overton_score": 4,
  "sdg_score": 37,
  
  "subjects": [
    {"name": "Hematology", "score": 0.9, "quartile": "Q1"},
    {"name": "Oncology", "score": 0.9, "quartile": "Q1"}
  ],
  "areas": ["Medicine"],
  
  "semantic_fingerprint": "Journal: Ca-A Cancer Journal for Clinicians | Publisher: John Wiley and Sons Inc | Scimago Rank: 1 | SJR Quartile: Q1 | Top quartile journal | Subject areas: Hematology (Q1), Oncology (Q1) | Country: United States | Research areas: Medicine | H-index: 223 | High impact journal | SJR score: 145.004 | Top 100 journal",
  
  "csv_source": true,
  "csv_imported_at": "2025-07-28T15:30:00.000Z",
  "fetched_at": "2025-07-28T15:30:00.000Z",
  
  "oa_status": null,
  "in_doaj": null,
  "has_apc": null,
  "apc_amount": null,
  "apc_currency": null,
  "subjects_doaj": [],
  "languages": [],
  "license_type": [],
  "publisher_doaj": null,
  "country_doaj": null,
  "doaj_fetched_at": null,
  
  "homepage_url": null,
  "description": null,
  "scope_text": null,
  "embedding": null
}
```

## Validation and Testing

### Field Validation Tests
```python
def test_field_mappings():
    """Test that all CSV fields are properly mapped."""
    csv_row = load_sample_csv_row()
    journal = map_csv_to_journal(csv_row)
    
    # Test critical fields
    assert journal['display_name'] is not None
    assert len(journal['issn']) > 0
    assert journal['scimago_rank'] > 0
    
    # Test numeric conversions
    assert isinstance(journal['sjr_score'], float)
    assert isinstance(journal['h_index'], int)
    
    # Test subject parsing
    assert len(journal['subjects']) > 0
    assert all('name' in subject for subject in journal['subjects'])
    
    # Test ISSN formatting
    for issn in journal['issn']:
        assert re.match(r'\d{4}-\d{4}', issn)
```

### Data Quality Tests  
```python
def test_data_quality():
    """Test data quality after mapping."""
    journals = process_full_csv_file()
    
    # Test completeness
    complete_titles = sum(1 for j in journals if j.get('display_name'))
    assert complete_titles == len(journals)
    
    # Test ranking consistency
    ranks = [j['scimago_rank'] for j in journals if j.get('scimago_rank')]
    assert len(set(ranks)) == len(ranks)  # All ranks should be unique
    
    # Test quartile distribution
    quartiles = [j.get('sjr_quartile') for j in journals]
    q1_count = quartiles.count('Q1')
    assert q1_count > 0  # Should have some Q1 journals
```

This mapping specification ensures consistent, high-quality transformation of the CSV data into the existing database schema while preserving compatibility with DOAJ integration and maintaining data integrity throughout the process.