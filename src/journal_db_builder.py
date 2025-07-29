"""
Journal database builder for Manuscript Journal Matcher.

This module handles fetching journal data from OpenAlex API,
creating semantic fingerprints, and building the journal database
with embeddings for similarity search.
"""

import json
import logging
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from datetime import datetime
from urllib.parse import urlencode
import hashlib

try:
    from .config import (
        OPENALEX_BASE_URL,
        OPENALEX_API_KEY,
        get_api_headers,
        JOURNAL_METADATA_PATH,
        API_CACHE_DIR
    )
    from .embedder import embed_texts, get_model
except ImportError:
    from config import (
        OPENALEX_BASE_URL,
        OPENALEX_API_KEY,
        get_api_headers,
        JOURNAL_METADATA_PATH,
        API_CACHE_DIR
    )
    from embedder import embed_texts, get_model

# DOAJ API Configuration
DOAJ_BASE_URL = "https://doaj.org/api/v3"
DOAJ_RATE_LIMIT = 1.0  # 1 second between requests (conservative)

# Set up logging
logger = logging.getLogger(__name__)


class JournalDatabaseError(Exception):
    """Custom exception for journal database building errors."""
    pass


class OpenAlexAPI:
    """OpenAlex API client for fetching journal and publication data."""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 0.1):
        """
        Initialize OpenAlex API client.
        
        Args:
            api_key: Optional API key for higher rate limits
            rate_limit: Minimum seconds between requests
        """
        self.base_url = OPENALEX_BASE_URL
        self.api_key = api_key or OPENALEX_API_KEY
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        
        # Set up headers
        headers = get_api_headers()
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        self.session.headers.update(headers)
        
        logger.info(f"Initialized OpenAlex API client (authenticated: {bool(self.api_key)})")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make rate-limited request to OpenAlex API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data
            
        Raises:
            JournalDatabaseError: If request fails
        """
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        url = f"{self.base_url}/{endpoint}"
        if params:
            url += "?" + urlencode(params)
        
        try:
            logger.debug(f"Making request to: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            self.last_request_time = time.time()
            
            data = response.json()
            logger.debug(f"Received {len(str(data))} characters of data")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise JournalDatabaseError(f"OpenAlex API request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise JournalDatabaseError(f"Invalid JSON response from OpenAlex: {e}")
    
    def fetch_journals(self, limit: int = 1000, offset: int = 0, 
                      filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Fetch journal data from OpenAlex.
        
        Args:
            limit: Maximum number of journals to fetch
            offset: Number of journals to skip
            filters: Additional filters for journal selection
            
        Returns:
            List of journal data dictionaries
        """
        params = {
            'per-page': min(limit, 200),  # OpenAlex max per page
            'cursor': '*',  # Use cursor pagination for large datasets
        }
        
        # Add filters
        if filters:
            for key, value in filters.items():
                params[key] = value
        
        journals = []
        fetched = 0
        
        while fetched < limit:
            try:
                response = self._make_request('sources', params)
                
                results = response.get('results', [])
                if not results:
                    logger.info("No more journals to fetch")
                    break
                
                # Process each journal
                for journal_data in results:
                    if fetched >= limit:
                        break
                    
                    processed_journal = self._process_journal_data(journal_data)
                    if processed_journal:
                        journals.append(processed_journal)
                        fetched += 1
                
                # Update cursor for next page
                next_cursor = response.get('meta', {}).get('next_cursor')
                if not next_cursor:
                    logger.info("Reached end of journal data")
                    break
                
                params['cursor'] = next_cursor
                logger.info(f"Fetched {fetched}/{limit} journals")
                
            except Exception as e:
                logger.error(f"Error fetching journals at offset {fetched}: {e}")
                if fetched == 0:  # If no journals fetched yet, re-raise
                    raise
                break  # Otherwise, return what we have
        
        logger.info(f"Successfully fetched {len(journals)} journals")
        return journals
    
    def _process_journal_data(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process raw OpenAlex journal data into our format.
        
        Args:
            raw_data: Raw journal data from OpenAlex API
            
        Returns:
            Processed journal data or None if invalid
        """
        try:
            # Extract basic information
            journal_id = raw_data.get('id', '').split('/')[-1]  # Extract ID from URL
            display_name = raw_data.get('display_name', '')
            
            # Skip if missing essential data
            if not journal_id or not display_name:
                logger.debug(f"Skipping journal with missing ID or name")
                return None
            
            # Extract ISSN
            issns = []
            if raw_data.get('issn_l'):
                issns.append(raw_data['issn_l'])
            if raw_data.get('issn'):
                issns.extend(raw_data['issn'])
            
            # Remove duplicates
            issns = list(set(filter(None, issns)))
            
            # Extract additional metadata
            processed = {
                'id': journal_id,
                'display_name': display_name,
                'issn': issns,
                'issn_l': raw_data.get('issn_l'),
                'publisher': raw_data.get('host_organization_name'),
                'homepage_url': raw_data.get('homepage_url'),
                'is_oa': raw_data.get('is_oa', False),
                'is_in_doaj': raw_data.get('is_in_doaj', False),
                'country_code': raw_data.get('country_code'),
                'type': raw_data.get('type'),
                'apc_usd': raw_data.get('apc_usd'),
                'works_count': raw_data.get('works_count', 0),
                'cited_by_count': raw_data.get('cited_by_count', 0),
                'h_index': raw_data.get('summary_stats', {}).get('h_index', 0),
                'subjects': [],
                'description': None,
                'scope_text': None,
                'semantic_fingerprint': None,
                'embedding': None,
                'fetched_at': datetime.now().isoformat()
            }
            
            # Extract subject areas
            if raw_data.get('x_concepts'):
                subjects = []
                for concept in raw_data['x_concepts'][:5]:  # Top 5 concepts
                    if concept.get('display_name') and concept.get('score', 0) > 0.1:
                        subjects.append({
                            'name': concept['display_name'],
                            'score': concept['score']
                        })
                processed['subjects'] = subjects
            
            logger.debug(f"Processed journal: {display_name}")
            return processed
            
        except Exception as e:
            logger.error(f"Error processing journal data: {e}")
            return None
    
    def fetch_sample_articles(self, journal_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch sample articles from a journal for semantic fingerprinting.
        
        Args:
            journal_id: OpenAlex journal ID
            limit: Number of sample articles to fetch
            
        Returns:
            List of article data
        """
        params = {
            'filter': f'primary_location.source.id:https://openalex.org/S{journal_id}',
            'per-page': limit,
            'sort': 'cited_by_count:desc',  # Get highly cited articles
        }
        
        try:
            response = self._make_request('works', params)
            articles = []
            
            for work in response.get('results', []):
                article = {
                    'title': work.get('title', ''),
                    'abstract': self._clean_abstract(work.get('abstract')),
                    'concepts': [c.get('display_name', '') for c in work.get('concepts', [])[:5]],
                    'cited_by_count': work.get('cited_by_count', 0)
                }
                
                if article['title'] or article['abstract']:
                    articles.append(article)
            
            logger.debug(f"Fetched {len(articles)} sample articles for journal {journal_id}")
            return articles
            
        except Exception as e:
            logger.warning(f"Could not fetch sample articles for journal {journal_id}: {e}")
            return []
    
    def _clean_abstract(self, raw_abstract: Optional[str]) -> Optional[str]:
        """Clean and normalize abstract text."""
        if not raw_abstract:
            return None
        
        # Remove HTML tags and extra whitespace
        import re
        cleaned = re.sub(r'<[^>]+>', '', raw_abstract)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Return None if too short
        if len(cleaned) < 50:
            return None
            
        return cleaned


class DOAJAPI:
    """DOAJ API client for fetching open access journal information."""
    
    def __init__(self, rate_limit: float = DOAJ_RATE_LIMIT):
        """
        Initialize DOAJ API client.
        
        Args:
            rate_limit: Minimum seconds between requests
        """
        self.base_url = DOAJ_BASE_URL
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        
        # Set up headers
        self.session.headers.update({
            'User-Agent': 'Manuscript-Journal-Matcher/1.0',
            'Accept': 'application/json'
        })
        
        logger.info("Initialized DOAJ API client")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make rate-limited request to DOAJ API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data
            
        Raises:
            JournalDatabaseError: If request fails
        """
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            logger.debug(f"Making DOAJ request to: {url}")
            response = self.session.get(url, params=params, timeout=30)
            
            # DOAJ may return 404 for journals not in directory
            if response.status_code == 404:
                logger.debug(f"Journal not found in DOAJ: {url}")
                return {}
            
            response.raise_for_status()
            self.last_request_time = time.time()
            
            data = response.json()
            logger.debug(f"Received DOAJ data: {len(str(data))} characters")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"DOAJ API request failed: {e}")
            return {}  # Return empty dict instead of raising for non-critical API
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response from DOAJ: {e}")
            return {}
    
    def fetch_journal_by_issn(self, issn: str) -> Optional[Dict[str, Any]]:
        """
        Fetch journal information from DOAJ by ISSN.
        
        Args:
            issn: Journal ISSN (print or electronic)
            
        Returns:
            DOAJ journal data or None if not found
        """
        if not issn:
            return None
        
        # Clean ISSN (remove hyphens)
        clean_issn = issn.replace('-', '').replace(' ', '')
        if len(clean_issn) != 8:
            logger.debug(f"Invalid ISSN format: {issn}")
            return None
        
        # Check cache first
        cache_key = f"doaj_issn_{clean_issn}"
        cached = load_cached_response(cache_key)
        if cached:
            logger.debug(f"Using cached DOAJ data for ISSN: {issn}")
            return cached
        
        # Search by ISSN
        endpoint = "search/journals"
        params = {
            'q': f'bibjson.identifier.id:"{issn}" OR bibjson.identifier.id:"{clean_issn}"',
            'pageSize': 1
        }
        
        response = self._make_request(endpoint, params)
        
        if response and response.get('results'):
            journal_data = response['results'][0]
            processed_data = self._process_doaj_data(journal_data)
            
            # Cache the result
            cache_api_response(cache_key, processed_data)
            
            return processed_data
        
        logger.debug(f"Journal with ISSN {issn} not found in DOAJ")
        # Cache empty result to avoid repeated requests
        cache_api_response(cache_key, {})
        return None
    
    def _process_doaj_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw DOAJ journal data into our format.
        
        Args:
            raw_data: Raw journal data from DOAJ API
            
        Returns:
            Processed DOAJ data
        """
        bibjson = raw_data.get('bibjson', {})
        
        # Extract APC information
        apc_info = bibjson.get('apc', {})
        has_apc = apc_info.get('has_apc', False)
        apc_amount = None
        apc_currency = None
        
        if has_apc and apc_info.get('max'):
            for apc_entry in apc_info.get('max', []):
                if apc_entry.get('price'):
                    apc_amount = apc_entry['price']
                    apc_currency = apc_entry.get('currency', 'USD')
                    break
        
        # Extract subject areas
        subjects = []
        for subject in bibjson.get('subject', []):
            if subject.get('term'):
                subjects.append(subject['term'])
        
        # Extract language information
        languages = []
        for lang in bibjson.get('language', []):
            if isinstance(lang, str):
                languages.append(lang)
            elif isinstance(lang, dict) and lang.get('name'):
                languages.append(lang['name'])
        
        processed = {
            'doaj_id': raw_data.get('id'),
            'in_doaj': True,
            'oa_status': True,  # All DOAJ journals are open access
            'has_apc': has_apc,
            'apc_amount': apc_amount,
            'apc_currency': apc_currency,
            'oa_start_year': bibjson.get('oa_start', {}).get('year'),
            'subjects_doaj': subjects[:5],  # Limit to 5 subjects
            'languages': languages,
            'publisher_doaj': bibjson.get('publisher', {}).get('name'),
            'country_doaj': bibjson.get('publisher', {}).get('country'),
            'license_type': [],
            'plagiarism_detection': bibjson.get('editorial', {}).get('review_process'),
            'publication_time_weeks': bibjson.get('editorial', {}).get('review_time'),
            'doaj_updated': raw_data.get('last_updated'),
            'doaj_fetched_at': datetime.now().isoformat()
        }
        
        # Extract license information
        for license_info in bibjson.get('license', []):
            if license_info.get('type'):
                processed['license_type'].append(license_info['type'])
        
        return processed
    
    def enrich_journals_with_doaj(self, journals: List[Dict[str, Any]], 
                                  batch_size: int = 50) -> List[Dict[str, Any]]:
        """
        Enrich journal database with DOAJ information.
        
        Args:
            journals: List of journal data to enrich
            batch_size: Number of journals to process in each batch
            
        Returns:
            Enriched journal data
        """
        logger.info(f"Enriching {len(journals)} journals with DOAJ data")
        
        enriched_journals = []
        processed = 0
        
        for i, journal in enumerate(journals):
            # Try to get DOAJ data using ISSNs
            doaj_data = None
            
            # Try all available ISSNs
            issns_to_try = []
            if journal.get('issn_l'):
                issns_to_try.append(journal['issn_l'])
            if journal.get('issn'):
                if isinstance(journal['issn'], list):
                    issns_to_try.extend(journal['issn'])
                else:
                    issns_to_try.append(journal['issn'])
            
            for issn in issns_to_try:
                doaj_data = self.fetch_journal_by_issn(issn)
                if doaj_data:
                    break
            
            # Merge DOAJ data with journal data
            enriched_journal = journal.copy()
            if doaj_data:
                enriched_journal.update(doaj_data)
                logger.debug(f"Enriched journal: {journal.get('display_name', 'Unknown')}")
            else:
                # Add default DOAJ fields for journals not in DOAJ
                enriched_journal.update({
                    'in_doaj': False,
                    'oa_status': journal.get('is_oa', False),  # Use OpenAlex OA status as fallback
                    'has_apc': None,
                    'apc_amount': journal.get('apc_usd'),  # Use OpenAlex APC if available
                    'apc_currency': 'USD' if journal.get('apc_usd') else None,
                    'doaj_fetched_at': datetime.now().isoformat()
                })
            
            enriched_journals.append(enriched_journal)
            processed += 1
            
            # Progress logging
            if processed % batch_size == 0 or processed == len(journals):
                logger.info(f"Processed {processed}/{len(journals)} journals with DOAJ data")
        
        doaj_count = sum(1 for j in enriched_journals if j.get('in_doaj', False))
        logger.info(f"Found {doaj_count} journals in DOAJ out of {len(enriched_journals)} total")
        
        return enriched_journals


def create_semantic_fingerprint(journal_data: Dict[str, Any], 
                               sample_articles: List[Dict[str, Any]]) -> str:
    """
    Create semantic fingerprint for a journal from its metadata and sample articles.
    
    Args:
        journal_data: Processed journal metadata
        sample_articles: Sample articles from the journal
        
    Returns:
        Combined text representing the journal's semantic fingerprint
    """
    fingerprint_parts = []
    
    # Add journal name and publisher
    if journal_data.get('display_name'):
        fingerprint_parts.append(f"Journal: {journal_data['display_name']}")
    
    if journal_data.get('publisher'):
        fingerprint_parts.append(f"Publisher: {journal_data['publisher']}")
    
    # Add subject areas (combine OpenAlex and DOAJ subjects)
    all_subjects = []
    if journal_data.get('subjects'):
        all_subjects.extend([s['name'] for s in journal_data['subjects'][:3]])
    if journal_data.get('subjects_doaj'):
        all_subjects.extend(journal_data['subjects_doaj'][:2])  # Add DOAJ subjects
    
    if all_subjects:
        # Remove duplicates while preserving order
        unique_subjects = list(dict.fromkeys(all_subjects))[:5]  # Top 5 unique subjects
        fingerprint_parts.append(f"Subject areas: {', '.join(unique_subjects)}")
    
    # Add open access information
    if journal_data.get('oa_status'):
        fingerprint_parts.append("Open access journal")
        if journal_data.get('in_doaj'):
            fingerprint_parts.append("Listed in Directory of Open Access Journals (DOAJ)")
    
    # Add APC information for context
    if journal_data.get('has_apc') and journal_data.get('apc_amount'):
        currency = journal_data.get('apc_currency', 'USD')
        fingerprint_parts.append(f"Article processing charge: {journal_data['apc_amount']} {currency}")
    
    # Add language information
    if journal_data.get('languages'):
        langs = ', '.join(journal_data['languages'][:3])  # Top 3 languages
        fingerprint_parts.append(f"Languages: {langs}")
    
    # Add sample article titles and abstracts
    for i, article in enumerate(sample_articles[:3], 1):
        if article.get('title'):
            fingerprint_parts.append(f"Sample article {i} title: {article['title']}")
        
        if article.get('abstract'):
            # Use first 200 characters of abstract
            abstract_snippet = article['abstract'][:200]
            fingerprint_parts.append(f"Sample article {i} abstract: {abstract_snippet}")
        
        if article.get('concepts'):
            concepts = ', '.join(article['concepts'][:3])
            fingerprint_parts.append(f"Sample article {i} concepts: {concepts}")
    
    # Join all parts
    fingerprint = ' | '.join(fingerprint_parts)
    
    logger.debug(f"Created fingerprint of {len(fingerprint)} characters")
    return fingerprint


def build_journal_embeddings(journals: List[Dict[str, Any]], 
                            batch_size: int = 32) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Generate embeddings for all journals using their semantic fingerprints.
    
    Args:
        journals: List of journal data with semantic fingerprints
        batch_size: Batch size for embedding generation
        
    Returns:
        Tuple of (updated_journals, embeddings_array)
    """
    logger.info(f"Generating embeddings for {len(journals)} journals")
    
    # Extract semantic fingerprints
    fingerprints = []
    valid_indices = []
    
    for i, journal in enumerate(journals):
        fingerprint = journal.get('semantic_fingerprint')
        if fingerprint and len(fingerprint.strip()) > 0:
            fingerprints.append(fingerprint)
            valid_indices.append(i)
        else:
            logger.warning(f"No fingerprint for journal: {journal.get('display_name', 'Unknown')}")
    
    if not fingerprints:
        raise JournalDatabaseError("No valid semantic fingerprints found")
    
    # Generate embeddings
    embeddings = embed_texts(fingerprints, batch_size=batch_size, show_progress=True)
    
    # Update journal data with embeddings
    updated_journals = journals.copy()
    embedding_index = 0
    
    for i in valid_indices:
        updated_journals[i]['embedding'] = embeddings[embedding_index].tolist()  # Convert to list for JSON
        embedding_index += 1
    
    logger.info(f"Generated embeddings for {len(valid_indices)} journals")
    return updated_journals, embeddings


def save_journal_database(journals: List[Dict[str, Any]], 
                         embeddings: Optional[np.ndarray] = None) -> None:
    """
    Save journal database to disk.
    
    Args:
        journals: List of journal data with embeddings
        embeddings: Optional numpy array of embeddings (will extract from journals if None)
    """
    # Ensure directory exists
    JOURNAL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_journals': len(journals),
        'embedding_dimension': None,
        'journals': journals
    }
    
    # Determine embedding dimension
    if embeddings is not None:
        metadata['embedding_dimension'] = embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings)
    elif journals and journals[0].get('embedding'):
        metadata['embedding_dimension'] = len(journals[0]['embedding'])
    
    # Save JSON metadata
    with open(JOURNAL_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved journal database with {len(journals)} journals to {JOURNAL_METADATA_PATH}")


def load_journal_database() -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Load journal database from disk.
    
    Returns:
        Tuple of (journals, embeddings_array)
    """
    if not JOURNAL_METADATA_PATH.exists():
        raise JournalDatabaseError(f"Journal database not found at {JOURNAL_METADATA_PATH}")
    
    with open(JOURNAL_METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    journals = metadata.get('journals', [])
    
    # Extract embeddings from journal data
    embeddings = []
    for journal in journals:
        if journal.get('embedding'):
            embeddings.append(journal['embedding'])
    
    embeddings_array = np.array(embeddings) if embeddings else None
    
    logger.info(f"Loaded journal database with {len(journals)} journals")
    return journals, embeddings_array


def cache_api_response(cache_key: str, data: Any) -> None:
    """Cache API response to disk."""
    cache_file = API_CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_cached_response(cache_key: str) -> Optional[Any]:
    """Load cached API response from disk."""
    cache_file = API_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None