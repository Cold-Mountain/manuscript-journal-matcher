"""
Citation network analysis module for Manuscript Journal Matcher.

This module implements citation network analysis to enhance journal matching
by analyzing reference patterns, citation networks, co-authorship patterns,
and academic impact metrics to find journals with similar research ecosystems.
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx

try:
    from .config import DATA_DIR
    from .embedder import embed_text, cosine_similarity_single
    from .utils import clean_text, extract_keywords
    from .study_classifier import StudyType
except ImportError:
    from config import DATA_DIR
    from embedder import embed_text, cosine_similarity_single
    from utils import clean_text, extract_keywords
    from study_classifier import StudyType

# Set up logging
logger = logging.getLogger(__name__)


class CitationMetric(Enum):
    """Enumeration of citation analysis metrics."""
    REFERENCE_OVERLAP = "reference_overlap"
    AUTHOR_NETWORK = "author_network"
    JOURNAL_PRESTIGE = "journal_prestige"
    TEMPORAL_PATTERNS = "temporal_patterns"
    SUBJECT_CLUSTERING = "subject_clustering"
    IMPACT_PROPAGATION = "impact_propagation"
    COLLABORATION_NETWORK = "collaboration_network"


@dataclass
class ReferenceData:
    """Parsed reference information."""
    title: Optional[str]
    authors: List[str]
    journal: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    pmid: Optional[str]
    raw_text: str
    confidence: float  # Parsing confidence


@dataclass
class AuthorNetwork:
    """Author collaboration network data."""
    authors: Set[str]
    collaborations: Dict[Tuple[str, str], int]  # (author1, author2) -> collaboration_count
    institutions: Dict[str, Set[str]]  # author -> set of institutions
    research_areas: Dict[str, Set[str]]  # author -> set of research areas


@dataclass
class CitationAnalysis:
    """Complete citation network analysis result."""
    references: List[ReferenceData]
    cited_journals: Dict[str, int]  # journal -> citation_count
    author_network: AuthorNetwork
    temporal_distribution: Dict[int, int]  # year -> reference_count
    subject_areas: Dict[str, float]  # subject -> relevance_score
    impact_metrics: Dict[str, float]
    network_centrality: Dict[str, float]  # journal -> centrality_score
    research_ecosystem_score: float
    metadata: Dict[str, Any]


class CitationNetworkAnalyzer:
    """
    Advanced citation network analyzer for academic manuscript analysis.
    
    Analyzes citation patterns, author networks, and research ecosystems
    to provide enhanced journal matching based on academic connections.
    """
    
    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize the citation network analyzer.
        
        Args:
            cache_path: Optional path for caching analysis results
        """
        self.cache_path = cache_path or (DATA_DIR / "citation_cache.json")
        
        # Initialize citation patterns and known journals
        self._initialize_journal_patterns()
        self._initialize_author_patterns()
        
        # Load cached data if available
        self.citation_cache = self._load_cache()
        
        # Build journal network graph
        self.journal_network = nx.Graph()
        
        logger.info("CitationNetworkAnalyzer initialized")
    
    def _initialize_journal_patterns(self) -> None:
        """Initialize patterns for journal name recognition."""
        
        # Common journal name patterns
        self.journal_patterns = [
            r'([A-Z][a-zA-Z\s&]+(?:Journal|Review|Proceedings|Letters|Science|Medicine|Research|Analysis|Computing|Engineering))',
            r'([A-Z][a-zA-Z\s]+(?:of|in|for)\s+[A-Z][a-zA-Z\s]+)',
            r'(Nature(?:\s+[A-Z][a-zA-Z]+)?)',
            r'(Science(?:\s+[A-Z][a-zA-Z]+)?)',
            r'([A-Z]{2,8})',  # Acronym journals like PNAS, JAMA
            r'([A-Z][a-z]+(?:[A-Z][a-z]+)*)',  # CamelCase journals
        ]
        
        # Known high-impact journals
        self.prestigious_journals = {
            'nature', 'science', 'cell', 'lancet', 'nejm', 'jama', 'pnas',
            'nature medicine', 'nature biotechnology', 'nature methods',
            'science translational medicine', 'cell metabolism', 'immunity',
            'neuron', 'cancer cell', 'molecular cell', 'developmental cell'
        }
        
        # Journal impact tiers
        self.journal_tiers = {
            'tier1': {'nature', 'science', 'cell'},  # Top-tier
            'tier2': {'lancet', 'nejm', 'jama', 'pnas'},  # High-impact clinical/general
            'tier3': {'plos one', 'scientific reports', 'bmc'},  # Open access/broad scope
        }
    
    def _initialize_author_patterns(self) -> None:
        """Initialize patterns for author name recognition."""
        
        # Author name patterns
        self.author_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?)*\s+[A-Z][a-z]+)',  # FirstName M. LastName
            r'([A-Z][a-z]+,\s+[A-Z]\.?(?:\s*[A-Z]\.?)*)',    # LastName, F.M.
            r'([A-Z]\.?\s*[A-Z]\.?\s+[A-Z][a-z]+)',          # F.M. LastName
        ]
        
        # Institution patterns
        self.institution_patterns = [
            r'(University\s+of\s+[A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+\s+University)',
            r'([A-Z][a-zA-Z\s]+\s+Institute(?:\s+of\s+[A-Z][a-zA-Z\s]+)?)',
            r'([A-Z][a-zA-Z\s]+\s+Hospital)',
            r'([A-Z][a-zA-Z\s]+\s+Medical\s+Center)',
        ]
    
    def analyze_citations(self, manuscript_text: str, 
                         include_network_analysis: bool = True) -> CitationAnalysis:
        """
        Perform comprehensive citation network analysis.
        
        Args:
            manuscript_text: Full manuscript text including references
            include_network_analysis: Whether to perform network analysis
            
        Returns:
            CitationAnalysis with detailed citation network information
        """
        if not manuscript_text or not manuscript_text.strip():
            return self._create_empty_analysis()
        
        logger.info("Starting citation network analysis")
        
        # Extract references from manuscript
        references = self._extract_references(manuscript_text)
        logger.info(f"Extracted {len(references)} references")
        
        # Analyze cited journals
        cited_journals = self._analyze_cited_journals(references)
        
        # Build author network
        author_network = self._build_author_network(references)
        
        # Analyze temporal patterns
        temporal_distribution = self._analyze_temporal_patterns(references)
        
        # Extract subject areas from citations
        subject_areas = self._extract_subject_areas(references)
        
        # Calculate impact metrics
        impact_metrics = self._calculate_impact_metrics(references, cited_journals)
        
        # Perform network analysis if requested
        network_centrality = {}
        if include_network_analysis:
            network_centrality = self._analyze_journal_network(cited_journals)
        
        # Calculate research ecosystem score
        ecosystem_score = self._calculate_ecosystem_score(
            cited_journals, author_network, impact_metrics
        )
        
        # Create metadata
        metadata = {
            'total_references': len(references),
            'unique_journals': len(cited_journals),
            'unique_authors': len(author_network.authors),
            'year_span': self._calculate_year_span(temporal_distribution),
            'analysis_timestamp': np.datetime64('now').astype(str),
            'high_impact_citations': sum(1 for j in cited_journals.keys() 
                                       if j.lower() in self.prestigious_journals)
        }
        
        return CitationAnalysis(
            references=references,
            cited_journals=cited_journals,
            author_network=author_network,
            temporal_distribution=temporal_distribution,
            subject_areas=subject_areas,
            impact_metrics=impact_metrics,
            network_centrality=network_centrality,
            research_ecosystem_score=ecosystem_score,
            metadata=metadata
        )
    
    def _extract_references(self, text: str) -> List[ReferenceData]:
        """
        Extract and parse references from manuscript text.
        
        Args:
            text: Full manuscript text
            
        Returns:
            List of parsed reference data
        """
        references = []
        
        # Find references section
        ref_section = self._find_references_section(text)
        if not ref_section:
            logger.warning("No references section found")
            return references
        
        # Split into individual references
        ref_lines = self._split_references(ref_section)
        
        for ref_text in ref_lines:
            if len(ref_text.strip()) < 20:  # Skip too short references
                continue
            
            # Parse individual reference
            ref_data = self._parse_single_reference(ref_text)
            if ref_data:
                references.append(ref_data)
        
        return references
    
    def _find_references_section(self, text: str) -> Optional[str]:
        """Find and extract the references section from text."""
        
        # Common reference section headers
        ref_headers = [
            r'\n\s*references?\s*\n',
            r'\n\s*bibliography\s*\n',
            r'\n\s*works?\s+cited\s*\n',
            r'\n\s*literature\s+cited\s*\n'
        ]
        
        text_lower = text.lower()
        
        for pattern in ref_headers:
            match = re.search(pattern, text_lower)
            if match:
                # Extract from match position to end or next major section
                start_pos = match.end()
                
                # Look for section endings
                end_patterns = [
                    r'\n\s*(?:appendix|supplementary|acknowledgment|author\s+contribution)',
                    r'\n\s*figure\s+\d+',
                    r'\n\s*table\s+\d+'
                ]
                
                end_pos = len(text)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, text_lower[start_pos:])
                    if end_match:
                        end_pos = start_pos + end_match.start()
                        break
                
                return text[start_pos:end_pos]
        
        return None
    
    def _split_references(self, ref_section: str) -> List[str]:
        """Split references section into individual references."""
        
        # Try numbered references first (1., 2., etc.)
        numbered_refs = re.split(r'\n\s*\d+\.?\s+', ref_section)
        if len(numbered_refs) > 3:  # Found numbered references
            return [ref.strip() for ref in numbered_refs[1:] if ref.strip()]
        
        # Try bracketed references [1], [2], etc.
        bracketed_refs = re.split(r'\n\s*\[\d+\]\s*', ref_section)
        if len(bracketed_refs) > 3:
            return [ref.strip() for ref in bracketed_refs[1:] if ref.strip()]
        
        # Fallback: split by double newlines
        paragraph_refs = re.split(r'\n\s*\n', ref_section)
        if len(paragraph_refs) > 1:
            return [ref.strip() for ref in paragraph_refs if ref.strip()]
        
        # Last resort: split by single newlines
        line_refs = ref_section.split('\n')
        return [ref.strip() for ref in line_refs if len(ref.strip()) > 20]
    
    def _parse_single_reference(self, ref_text: str) -> Optional[ReferenceData]:
        """
        Parse a single reference text into structured data.
        
        Args:
            ref_text: Raw reference text
            
        Returns:
            ReferenceData object or None if parsing fails
        """
        ref_text = ref_text.strip()
        if not ref_text:
            return None
        
        # Initialize reference data
        title = None
        authors = []
        journal = None
        year = None
        doi = None
        pmid = None
        confidence = 0.5  # Base confidence
        
        # Extract DOI
        doi_match = re.search(r'doi[:\s]*([0-9]+\.[0-9]+/[^\s]+)', ref_text, re.IGNORECASE)
        if doi_match:
            doi = doi_match.group(1)
            confidence += 0.2
        
        # Extract PMID
        pmid_match = re.search(r'pmid[:\s]*([0-9]+)', ref_text, re.IGNORECASE)
        if pmid_match:
            pmid = pmid_match.group(1)
            confidence += 0.1
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
        if year_match:
            year = int(year_match.group())
            confidence += 0.1
        
        # Extract authors (simplified - first few words before year/title)
        author_text = ref_text.split('.')[0] if '.' in ref_text else ref_text[:100]
        authors = self._extract_authors_from_text(author_text)
        if authors:
            confidence += 0.1
        
        # Extract journal name
        journal = self._extract_journal_from_reference(ref_text)
        if journal:
            confidence += 0.2
        
        # Extract title (often in quotes or between specific patterns)
        title = self._extract_title_from_reference(ref_text)
        if title:
            confidence += 0.1
        
        return ReferenceData(
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            doi=doi,
            pmid=pmid,
            raw_text=ref_text,
            confidence=min(confidence, 1.0)
        )
    
    def _extract_journal_from_reference(self, ref_text: str) -> Optional[str]:
        """Extract journal name from reference text."""
        
        # Look for italic text (common journal formatting)
        italic_match = re.search(r'<i>([^<]+)</i>', ref_text)
        if italic_match:
            return italic_match.group(1).strip()
        
        # Try journal patterns
        for pattern in self.journal_patterns:
            match = re.search(pattern, ref_text)
            if match:
                journal_candidate = match.group(1).strip()
                
                # Validate journal name
                if self._is_valid_journal_name(journal_candidate):
                    return journal_candidate
        
        # Look for common journal abbreviations
        abbrev_match = re.search(r'\b([A-Z][a-z]*\.?\s*){1,4}[A-Z][a-z]*\.?\b', ref_text)
        if abbrev_match:
            return abbrev_match.group().strip()
        
        return None
    
    def _extract_title_from_reference(self, ref_text: str) -> Optional[str]:
        """Extract article title from reference text."""
        
        # Look for quoted titles
        quote_match = re.search(r'"([^"]+)"', ref_text)
        if quote_match:
            return quote_match.group(1).strip()
        
        # Look for titles after authors and before journal
        # This is a simplified heuristic
        parts = ref_text.split('.')
        if len(parts) >= 3:
            # Often title is the second part after authors
            title_candidate = parts[1].strip()
            if 10 < len(title_candidate) < 200:  # Reasonable title length
                return title_candidate
        
        return None
    
    def _extract_authors_from_text(self, author_text: str) -> List[str]:
        """Extract author names from text."""
        
        authors = []
        
        # Try different author patterns
        for pattern in self.author_patterns:
            matches = re.findall(pattern, author_text)
            authors.extend(matches)
        
        # Clean and deduplicate authors
        clean_authors = []
        for author in authors:
            author = author.strip().rstrip(',')
            if author and len(author) > 2:
                clean_authors.append(author)
        
        return list(set(clean_authors))[:10]  # Limit to first 10 authors
    
    def _is_valid_journal_name(self, name: str) -> bool:
        """Validate if a string is likely a journal name."""
        
        # Check length
        if len(name) < 3 or len(name) > 100:
            return False
        
        # Check for journal indicators
        journal_indicators = ['journal', 'review', 'proceedings', 'letters', 
                            'science', 'medicine', 'research', 'analysis']
        
        name_lower = name.lower()
        
        # Direct match with known journals
        if name_lower in self.prestigious_journals:
            return True
        
        # Contains journal indicators
        if any(indicator in name_lower for indicator in journal_indicators):
            return True
        
        # Title case pattern (common for journals)
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', name):
            return True
        
        return False
    
    def _analyze_cited_journals(self, references: List[ReferenceData]) -> Dict[str, int]:
        """Analyze journals cited in references."""
        
        cited_journals = Counter()
        
        for ref in references:
            if ref.journal:
                # Normalize journal name
                journal_name = self._normalize_journal_name(ref.journal)
                cited_journals[journal_name] += 1
        
        return dict(cited_journals)
    
    def _normalize_journal_name(self, journal_name: str) -> str:
        """Normalize journal name for consistent analysis."""
        
        # Convert to lowercase and strip
        normalized = journal_name.lower().strip()
        
        # Remove common suffixes
        normalized = re.sub(r'\s*\(online\)|\s*\(print\)', '', normalized)
        
        # Handle common abbreviations
        abbreviations = {
            'proc natl acad sci': 'pnas',
            'proceedings of the national academy of sciences': 'pnas',
            'new england journal of medicine': 'nejm',
            'journal of the american medical association': 'jama'
        }
        
        if normalized in abbreviations:
            normalized = abbreviations[normalized]
        
        return normalized
    
    def _build_author_network(self, references: List[ReferenceData]) -> AuthorNetwork:
        """Build author collaboration network from references."""
        
        all_authors = set()
        collaborations = defaultdict(int)
        institutions = defaultdict(set)
        research_areas = defaultdict(set)
        
        for ref in references:
            if not ref.authors or len(ref.authors) < 2:
                continue
            
            # Add authors
            ref_authors = [self._normalize_author_name(author) for author in ref.authors]
            all_authors.update(ref_authors)
            
            # Record collaborations (pairwise)
            for i, author1 in enumerate(ref_authors):
                for author2 in ref_authors[i+1:]:
                    collab_key = tuple(sorted([author1, author2]))
                    collaborations[collab_key] += 1
            
            # Extract institutions (simplified)
            institutions_found = self._extract_institutions(ref.raw_text)
            for author in ref_authors:
                institutions[author].update(institutions_found)
            
            # Extract research areas from journal/title
            areas = self._extract_research_areas(ref.journal, ref.title)
            for author in ref_authors:
                research_areas[author].update(areas)
        
        return AuthorNetwork(
            authors=all_authors,
            collaborations=dict(collaborations),
            institutions=dict(institutions),
            research_areas=dict(research_areas)
        )
    
    def _normalize_author_name(self, author_name: str) -> str:
        """Normalize author name for consistent analysis."""
        
        # Basic normalization
        normalized = author_name.strip().lower()
        
        # Handle "LastName, FirstName" format
        if ',' in normalized:
            parts = normalized.split(',', 1)
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_part = parts[1].strip()
                
                # Extract initials from first part
                initials = re.findall(r'\b[a-z]', first_part)
                if initials:
                    normalized = f"{last_name} {' '.join(initials)}"
        
        return normalized
    
    def _extract_institutions(self, ref_text: str) -> Set[str]:
        """Extract institution names from reference text."""
        
        institutions = set()
        
        for pattern in self.institution_patterns:
            matches = re.findall(pattern, ref_text)
            institutions.update(match.strip() for match in matches)
        
        return institutions
    
    def _extract_research_areas(self, journal: Optional[str], 
                               title: Optional[str]) -> Set[str]:
        """Extract research areas from journal and title."""
        
        areas = set()
        
        # Extract from journal name
        if journal:
            journal_lower = journal.lower()
            
            # Map journal patterns to research areas
            area_mappings = {
                'medicine|medical|clinical': 'medicine',
                'computer|computing|informatics': 'computer_science',
                'biology|biological|life': 'biology',
                'physics|physical': 'physics',
                'chemistry|chemical': 'chemistry',
                'engineering': 'engineering',
                'neuroscience|neurology|brain': 'neuroscience',
                'psychology|psychological': 'psychology',
                'economics|economic': 'economics'
            }
            
            for pattern, area in area_mappings.items():
                if re.search(pattern, journal_lower):
                    areas.add(area)
        
        # Extract from title keywords
        if title:
            title_lower = title.lower()
            for pattern, area in area_mappings.items():
                if re.search(pattern, title_lower):
                    areas.add(area)
        
        return areas
    
    def _analyze_temporal_patterns(self, references: List[ReferenceData]) -> Dict[int, int]:
        """Analyze temporal distribution of references."""
        
        year_counts = Counter()
        
        for ref in references:
            if ref.year and 1950 <= ref.year <= 2024:  # Reasonable year range
                year_counts[ref.year] += 1
        
        return dict(year_counts)
    
    def _extract_subject_areas(self, references: List[ReferenceData]) -> Dict[str, float]:
        """Extract subject areas from citations with relevance scores."""
        
        subject_scores = defaultdict(float)
        
        # Combine all citation text
        all_citation_text = ' '.join([
            (ref.title or '') + ' ' + (ref.journal or '')
            for ref in references
        ])
        
        # Extract keywords and map to subjects
        keywords = extract_keywords(all_citation_text, top_k=50)
        
        # Subject area keywords mapping
        subject_keywords = {
            'medicine': ['medical', 'clinical', 'patient', 'treatment', 'therapy', 'diagnosis'],
            'biology': ['biological', 'cell', 'molecular', 'gene', 'protein', 'organism'],
            'computer_science': ['algorithm', 'computing', 'software', 'data', 'artificial'],
            'physics': ['physics', 'quantum', 'particle', 'energy', 'matter'],
            'chemistry': ['chemical', 'reaction', 'molecule', 'compound', 'synthesis'],
            'engineering': ['engineering', 'design', 'system', 'technology', 'mechanical'],
            'neuroscience': ['brain', 'neural', 'cognitive', 'neuron', 'mind'],
            'psychology': ['psychological', 'behavior', 'mental', 'social', 'cognition']
        }
        
        # Calculate subject relevance scores
        for subject, subject_kws in subject_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword.lower() in subject_kws:
                    score += 1.0
                elif any(sk in keyword.lower() for sk in subject_kws):
                    score += 0.5
            
            if score > 0:
                subject_scores[subject] = score / len(keywords)
        
        return dict(subject_scores)
    
    def _calculate_impact_metrics(self, references: List[ReferenceData], 
                                 cited_journals: Dict[str, int]) -> Dict[str, float]:
        """Calculate various impact metrics."""
        
        metrics = {}
        
        # Reference quality score
        high_quality_refs = sum(1 for ref in references if ref.confidence > 0.7)
        metrics['reference_quality'] = high_quality_refs / len(references) if references else 0
        
        # Prestigious journal citations
        prestigious_citations = sum(count for journal, count in cited_journals.items()
                                  if journal.lower() in self.prestigious_journals)
        metrics['prestigious_citations'] = prestigious_citations / len(references) if references else 0
        
        # Diversity metrics
        metrics['journal_diversity'] = len(cited_journals) / len(references) if references else 0
        
        # Recent citations (last 5 years)
        current_year = 2024
        recent_refs = sum(1 for ref in references 
                         if ref.year and ref.year >= current_year - 5)
        metrics['recency_score'] = recent_refs / len(references) if references else 0
        
        # DOI availability (indicator of formal publication)
        doi_refs = sum(1 for ref in references if ref.doi)
        metrics['doi_coverage'] = doi_refs / len(references) if references else 0
        
        return metrics
    
    def _analyze_journal_network(self, cited_journals: Dict[str, int]) -> Dict[str, float]:
        """Analyze journal citation network and calculate centrality measures."""
        
        # Build network graph
        G = nx.Graph()
        
        # Add journal nodes
        for journal in cited_journals.keys():
            G.add_node(journal, weight=cited_journals[journal])
        
        # Add edges based on co-citation (simplified)
        # In a full implementation, this would use actual co-citation data
        journal_list = list(cited_journals.keys())
        for i, journal1 in enumerate(journal_list):
            for journal2 in journal_list[i+1:]:
                # Simple heuristic: journals in similar fields are connected
                if self._journals_related(journal1, journal2):
                    weight = min(cited_journals[journal1], cited_journals[journal2])
                    G.add_edge(journal1, journal2, weight=weight)
        
        # Calculate centrality measures
        centrality_scores = {}
        
        if G.number_of_edges() > 0:
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(G, weight='weight')
            
            # Eigenvector centrality (if graph is connected)
            try:
                eigenvector = nx.eigenvector_centrality(G, weight='weight')
            except:
                eigenvector = {node: 0.0 for node in G.nodes()}
            
            # Combine centrality measures
            for journal in cited_journals.keys():
                centrality_scores[journal] = (
                    betweenness.get(journal, 0.0) * 0.5 +
                    eigenvector.get(journal, 0.0) * 0.5
                )
        
        return centrality_scores
    
    def _journals_related(self, journal1: str, journal2: str) -> bool:
        """Determine if two journals are related (simplified heuristic)."""
        
        # Check if journals share common keywords
        j1_words = set(journal1.lower().split())
        j2_words = set(journal2.lower().split())
        
        # Remove common words
        common_stop_words = {'of', 'the', 'and', 'for', 'in', 'on', 'journal', 'review'}
        j1_words -= common_stop_words
        j2_words -= common_stop_words
        
        # Check overlap
        if j1_words & j2_words:  # Non-empty intersection
            return True
        
        # Check if both are in same tier
        for tier_journals in self.journal_tiers.values():
            if journal1.lower() in tier_journals and journal2.lower() in tier_journals:
                return True
        
        return False
    
    def _calculate_ecosystem_score(self, cited_journals: Dict[str, int],
                                  author_network: AuthorNetwork,
                                  impact_metrics: Dict[str, float]) -> float:
        """Calculate overall research ecosystem score."""
        
        # Combine different aspects of the research ecosystem
        ecosystem_components = {
            'journal_diversity': min(len(cited_journals) / 20, 1.0),  # Normalize to max 20 journals
            'author_network_size': min(len(author_network.authors) / 50, 1.0),  # Max 50 authors
            'prestigious_citations': impact_metrics.get('prestigious_citations', 0.0),
            'reference_quality': impact_metrics.get('reference_quality', 0.0),
            'collaboration_density': min(len(author_network.collaborations) / 100, 1.0)
        }
        
        # Weighted combination
        weights = {
            'journal_diversity': 0.25,
            'author_network_size': 0.20,
            'prestigious_citations': 0.25,
            'reference_quality': 0.20,
            'collaboration_density': 0.10
        }
        
        ecosystem_score = sum(
            score * weights[component]
            for component, score in ecosystem_components.items()
        )
        
        return min(ecosystem_score, 1.0)
    
    def _calculate_year_span(self, temporal_distribution: Dict[int, int]) -> int:
        """Calculate the span of years in references."""
        if not temporal_distribution:
            return 0
        
        years = list(temporal_distribution.keys())
        return max(years) - min(years) if years else 0
    
    def _create_empty_analysis(self) -> CitationAnalysis:
        """Create empty analysis for invalid input."""
        return CitationAnalysis(
            references=[],
            cited_journals={},
            author_network=AuthorNetwork(set(), {}, {}, {}),
            temporal_distribution={},
            subject_areas={},
            impact_metrics={},
            network_centrality={},
            research_ecosystem_score=0.0,
            metadata={'error': 'No valid content provided'}
        )
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cached analysis results."""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load citation cache: {e}")
        
        return {}
    
    def _save_cache(self) -> None:
        """Save analysis results to cache."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(self.citation_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save citation cache: {e}")
    
    def get_journal_compatibility_score(self, analysis: CitationAnalysis, 
                                      target_journal: str) -> float:
        """
        Calculate compatibility score between manuscript and target journal.
        
        Args:
            analysis: Citation analysis result
            target_journal: Target journal name
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        if not analysis.cited_journals:
            return 0.0
        
        target_normalized = self._normalize_journal_name(target_journal)
        compatibility_score = 0.0
        
        # Direct citation score
        if target_normalized in analysis.cited_journals:
            citation_count = analysis.cited_journals[target_normalized]
            max_citations = max(analysis.cited_journals.values())
            compatibility_score += (citation_count / max_citations) * 0.4
        
        # Network centrality score
        if target_normalized in analysis.network_centrality:
            centrality = analysis.network_centrality[target_normalized]
            compatibility_score += centrality * 0.3
        
        # Subject area alignment
        target_areas = self._extract_research_areas(target_journal, None)
        if target_areas and analysis.subject_areas:
            area_overlap = len(target_areas & set(analysis.subject_areas.keys()))
            area_alignment = area_overlap / len(target_areas) if target_areas else 0
            compatibility_score += area_alignment * 0.3
        
        return min(compatibility_score, 1.0)


def analyze_manuscript_citations(manuscript_text: str) -> CitationAnalysis:
    """
    Convenience function to analyze citations in a manuscript.
    
    Args:
        manuscript_text: Full manuscript text
        
    Returns:
        CitationAnalysis result
    """
    analyzer = CitationNetworkAnalyzer()
    return analyzer.analyze_citations(manuscript_text)