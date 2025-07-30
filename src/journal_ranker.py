"""
Journal ranking integration module for Manuscript Journal Matcher.

This module integrates various journal ranking systems including impact factors,
SJR scores, h-index metrics, prestige indicators, and quality assessments to
enhance journal matching with ranking-aware recommendations.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict
import re

try:
    from .config import DATA_DIR
    from .utils import clean_text
except ImportError:
    from config import DATA_DIR
    from utils import clean_text

# Set up logging
logger = logging.getLogger(__name__)


class RankingSystem(Enum):
    """Enumeration of journal ranking systems."""
    SJR_SCIMAGO = "sjr_scimago"
    JCR_IMPACT_FACTOR = "jcr_impact_factor"
    H_INDEX = "h_index"
    CITATIONS_PERCENTILE = "citations_percentile"
    EIGENFACTOR = "eigenfactor"
    ARTICLE_INFLUENCE = "article_influence"
    PRESTIGE_SCORE = "prestige_score"
    QUALITY_INDICATOR = "quality_indicator"


class PrestigeLevel(Enum):
    """Journal prestige levels."""
    ELITE = "elite"          # Top 1% (Nature, Science, Cell)
    PREMIER = "premier"      # Top 5% (High-impact specialized)
    EXCELLENT = "excellent"  # Top 10% (Well-regarded in field)
    GOOD = "good"           # Top 25% (Solid reputation)
    AVERAGE = "average"     # Top 50% (Standard journals)
    EMERGING = "emerging"   # Newer or regional journals


@dataclass
class RankingMetrics:
    """Comprehensive ranking metrics for a journal."""
    journal_id: str
    journal_name: str
    
    # Core ranking metrics
    sjr_score: Optional[float]
    sjr_rank: Optional[int]
    sjr_quartile: Optional[str]
    impact_factor: Optional[float]
    if_rank: Optional[int]
    h_index: Optional[int]
    
    # Citation metrics
    total_citations: Optional[int]
    citations_per_doc: Optional[float]
    self_citations_ratio: Optional[float]
    
    # Quality indicators
    prestige_level: PrestigeLevel
    quality_score: float
    reputation_score: float
    
    # Specialized metrics
    eigenfactor_score: Optional[float]
    article_influence_score: Optional[float]
    immediacy_index: Optional[float]
    
    # Metadata
    subject_areas: List[str]
    publisher: Optional[str]
    country: Optional[str]
    open_access_status: bool
    
    # Computed scores
    composite_ranking_score: float
    field_normalized_score: float


@dataclass
class RankingAnalysis:
    """Analysis of journal rankings for matching."""
    manuscript_quality_indicators: Dict[str, float]
    target_prestige_level: PrestigeLevel
    recommended_ranking_range: Tuple[int, int]
    ranking_explanation: str
    quality_alignment_score: float


class JournalRankingIntegrator:
    """
    Advanced journal ranking system integrator.
    
    Combines multiple ranking systems and quality metrics to provide
    comprehensive journal quality assessment and ranking-aware matching.
    """
    
    def __init__(self, ranking_data_path: Optional[Path] = None):
        """
        Initialize the journal ranking integrator.
        
        Args:
            ranking_data_path: Path to journal ranking data
        """
        self.ranking_data_path = ranking_data_path or (DATA_DIR / "journal_rankings.json")
        
        # Load ranking data
        self.ranking_data = self._load_ranking_data()
        
        # Initialize prestige hierarchies
        self._initialize_prestige_hierarchies()
        
        # Load subject area rankings
        self._initialize_subject_rankings()
        
        # Initialize quality indicators
        self._initialize_quality_indicators()
        
        logger.info(f"JournalRankingIntegrator initialized with {len(self.ranking_data)} journals")
    
    def _load_ranking_data(self) -> Dict[str, RankingMetrics]:
        """Load journal ranking data from various sources."""
        ranking_data = {}
        
        # Load from file if exists
        if self.ranking_data_path.exists():
            try:
                with open(self.ranking_data_path, 'r') as f:
                    data = json.load(f)
                    # Convert to RankingMetrics objects
                    for journal_id, metrics in data.items():
                        ranking_data[journal_id] = self._dict_to_ranking_metrics(metrics)
            except Exception as e:
                logger.warning(f"Failed to load ranking data: {e}")
        
        # If no data loaded, create from available journal database
        if not ranking_data:
            ranking_data = self._create_ranking_data_from_database()
        
        return ranking_data
    
    def _dict_to_ranking_metrics(self, data: Dict[str, Any]) -> RankingMetrics:
        """Convert dictionary to RankingMetrics object."""
        return RankingMetrics(
            journal_id=data.get('journal_id', ''),
            journal_name=data.get('journal_name', ''),
            sjr_score=data.get('sjr_score'),
            sjr_rank=data.get('sjr_rank'),
            sjr_quartile=data.get('sjr_quartile'),
            impact_factor=data.get('impact_factor'),
            if_rank=data.get('if_rank'),
            h_index=data.get('h_index'),
            total_citations=data.get('total_citations'),
            citations_per_doc=data.get('citations_per_doc'),
            self_citations_ratio=data.get('self_citations_ratio'),
            prestige_level=PrestigeLevel(data.get('prestige_level', 'average')),
            quality_score=data.get('quality_score', 0.5),
            reputation_score=data.get('reputation_score', 0.5),
            eigenfactor_score=data.get('eigenfactor_score'),
            article_influence_score=data.get('article_influence_score'),
            immediacy_index=data.get('immediacy_index'),
            subject_areas=data.get('subject_areas', []),
            publisher=data.get('publisher'),
            country=data.get('country'),
            open_access_status=data.get('open_access_status', False),
            composite_ranking_score=data.get('composite_ranking_score', 0.5),
            field_normalized_score=data.get('field_normalized_score', 0.5)
        )
    
    def _create_ranking_data_from_database(self) -> Dict[str, RankingMetrics]:
        """Create ranking data from existing journal database."""
        ranking_data = {}
        
        try:
            # Import journal database
            from .journal_db_builder import load_journal_database
            journals, _ = load_journal_database()
            
            for i, journal in enumerate(journals):
                journal_id = journal.get('id', str(i))
                
                # Extract available metrics
                sjr_score = journal.get('sjr_score')
                sjr_rank = journal.get('scimago_rank')
                sjr_quartile = journal.get('sjr_quartile')
                h_index = journal.get('h_index')
                total_citations = journal.get('cited_by_count')
                
                # Calculate derived metrics
                prestige_level = self._determine_prestige_level(
                    sjr_score, sjr_rank, h_index, total_citations
                )
                
                quality_score = self._calculate_quality_score(journal)
                reputation_score = self._calculate_reputation_score(journal)
                
                # Create ranking metrics
                ranking_data[journal_id] = RankingMetrics(
                    journal_id=journal_id,
                    journal_name=journal.get('display_name', ''),
                    sjr_score=sjr_score,
                    sjr_rank=sjr_rank,
                    sjr_quartile=sjr_quartile,
                    impact_factor=None,  # Not available in current data
                    if_rank=None,
                    h_index=h_index,
                    total_citations=total_citations,
                    citations_per_doc=None,
                    self_citations_ratio=None,
                    prestige_level=prestige_level,
                    quality_score=quality_score,
                    reputation_score=reputation_score,
                    eigenfactor_score=None,
                    article_influence_score=None,
                    immediacy_index=None,
                    subject_areas=[s.get('name', '') if isinstance(s, dict) else str(s) 
                                 for s in journal.get('subjects', [])],
                    publisher=journal.get('publisher'),
                    country=journal.get('country'),
                    open_access_status=journal.get('is_oa', False),
                    composite_ranking_score=self._calculate_composite_score(
                        sjr_score, sjr_rank, h_index, total_citations, quality_score
                    ),
                    field_normalized_score=0.5  # Will be computed later
                )
            
        except Exception as e:
            logger.error(f"Failed to create ranking data from database: {e}")
        
        return ranking_data
    
    def _determine_prestige_level(self, sjr_score: Optional[float], 
                                sjr_rank: Optional[int],
                                h_index: Optional[int], 
                                citations: Optional[int]) -> PrestigeLevel:
        """Determine journal prestige level based on metrics."""
        
        # Elite journals (top 1%)
        if sjr_rank and sjr_rank <= 50:  # Top 50 globally
            return PrestigeLevel.ELITE
        
        if sjr_score and sjr_score >= 3.0:  # Very high SJR
            return PrestigeLevel.ELITE
        
        # Premier journals (top 5%)
        if sjr_rank and sjr_rank <= 250:
            return PrestigeLevel.PREMIER
        
        if sjr_score and sjr_score >= 1.5:
            return PrestigeLevel.PREMIER
        
        if h_index and h_index >= 150:
            return PrestigeLevel.PREMIER
        
        # Excellent journals (top 10%)
        if sjr_rank and sjr_rank <= 500:
            return PrestigeLevel.EXCELLENT
        
        if sjr_score and sjr_score >= 1.0:
            return PrestigeLevel.EXCELLENT
        
        if h_index and h_index >= 100:
            return PrestigeLevel.EXCELLENT
        
        # Good journals (top 25%)
        if sjr_rank and sjr_rank <= 1500:
            return PrestigeLevel.GOOD
        
        if sjr_score and sjr_score >= 0.5:
            return PrestigeLevel.GOOD
        
        if h_index and h_index >= 50:
            return PrestigeLevel.GOOD
        
        # Average journals (top 50%)
        if sjr_rank and sjr_rank <= 3000:
            return PrestigeLevel.AVERAGE
        
        # Emerging journals
        return PrestigeLevel.EMERGING
    
    def _calculate_quality_score(self, journal: Dict[str, Any]) -> float:
        """Calculate journal quality score from available metrics."""
        quality_factors = []
        
        # SJR contribution
        sjr_score = journal.get('sjr_score')
        if sjr_score:
            # Normalize SJR score (max around 20 for top journals)
            sjr_normalized = min(sjr_score / 5.0, 1.0)
            quality_factors.append(('sjr', sjr_normalized, 0.3))
        
        # H-index contribution
        h_index = journal.get('h_index')
        if h_index:
            # Normalize h-index (max around 500 for top journals)
            h_normalized = min(h_index / 200.0, 1.0)
            quality_factors.append(('h_index', h_normalized, 0.25))
        
        # Citation count contribution
        citations = journal.get('cited_by_count')
        if citations:
            # Normalize citations (log scale)
            citation_normalized = min(np.log10(citations + 1) / 7.0, 1.0)  # 10M citations = 1.0
            quality_factors.append(('citations', citation_normalized, 0.2))
        
        # Publisher prestige
        publisher = journal.get('publisher', '').lower()
        publisher_score = self._get_publisher_prestige_score(publisher)
        quality_factors.append(('publisher', publisher_score, 0.15))
        
        # Open access bonus
        if journal.get('is_oa', False):
            quality_factors.append(('open_access', 0.1, 0.1))
        
        # Calculate weighted average
        if quality_factors:
            total_score = sum(score * weight for _, score, weight in quality_factors)
            total_weight = sum(weight for _, _, weight in quality_factors)
            return total_score / total_weight if total_weight > 0 else 0.5
        
        return 0.5  # Default quality score
    
    def _get_publisher_prestige_score(self, publisher: str) -> float:
        """Get publisher prestige score."""
        publisher_scores = {
            'nature': 1.0,
            'science': 1.0,
            'cell': 1.0,
            'elsevier': 0.8,
            'springer': 0.8,
            'wiley': 0.75,
            'oxford': 0.85,
            'cambridge': 0.85,
            'taylor': 0.7,
            'sage': 0.7,
            'ieee': 0.8,
            'acm': 0.75,
            'plos': 0.7,
            'frontiers': 0.6,
            'hindawi': 0.5,
            'mdpi': 0.5
        }
        
        for pub_name, score in publisher_scores.items():
            if pub_name in publisher:
                return score
        
        return 0.5  # Default publisher score
    
    def _calculate_reputation_score(self, journal: Dict[str, Any]) -> float:
        """Calculate journal reputation score."""
        reputation_factors = []
        
        # Age/establishment factor
        works_count = journal.get('works_count', 0)
        if works_count > 0:
            # More publications indicate established journal
            establishment_score = min(np.log10(works_count + 1) / 5.0, 1.0)
            reputation_factors.append(establishment_score * 0.3)
        
        # Quartile ranking
        quartile = journal.get('sjr_quartile')
        if quartile:
            quartile_scores = {'Q1': 1.0, 'Q2': 0.75, 'Q3': 0.5, 'Q4': 0.25}
            reputation_factors.append(quartile_scores.get(quartile, 0.25) * 0.4)
        
        # Subject area diversity
        subjects = journal.get('subjects', [])
        if subjects:
            # More focused journals often have higher reputation in their field
            diversity_penalty = min(len(subjects) / 10.0, 0.3)
            reputation_factors.append((1.0 - diversity_penalty) * 0.3)
        
        return sum(reputation_factors) if reputation_factors else 0.5
    
    def _calculate_composite_score(self, sjr_score: Optional[float],
                                 sjr_rank: Optional[int],
                                 h_index: Optional[int],
                                 citations: Optional[int],
                                 quality_score: float) -> float:
        """Calculate composite ranking score."""
        
        components = []
        
        # SJR component
        if sjr_score:
            sjr_component = min(sjr_score / 5.0, 1.0)
            components.append(('sjr', sjr_component, 0.35))
        
        # Rank component (inverted - lower rank is better)
        if sjr_rank:
            rank_component = max(0, 1.0 - (sjr_rank / 10000.0))  # Normalize by 10k journals
            components.append(('rank', rank_component, 0.25))
        
        # H-index component
        if h_index:
            h_component = min(h_index / 200.0, 1.0)
            components.append(('h_index', h_component, 0.2))
        
        # Quality component
        components.append(('quality', quality_score, 0.2))
        
        # Calculate weighted average
        if components:
            total_score = sum(score * weight for _, score, weight in components)
            total_weight = sum(weight for _, _, weight in components)
            return total_score / total_weight if total_weight > 0 else 0.5
        
        return 0.5
    
    def _initialize_prestige_hierarchies(self) -> None:
        """Initialize journal prestige hierarchies."""
        
        # Elite journals (widely recognized top-tier)
        self.elite_journals = {
            'nature', 'science', 'cell', 'lancet', 'nejm', 'jama',
            'nature medicine', 'nature biotechnology', 'nature methods',
            'nature genetics', 'nature neuroscience', 'nature chemistry',
            'science translational medicine', 'cell metabolism',
            'immunity', 'neuron', 'cancer cell', 'molecular cell'
        }
        
        # Premier journals by field
        self.premier_journals = {
            'medicine': ['bmj', 'pnas', 'circulation', 'blood', 'gastroenterology'],
            'computer_science': ['cacm', 'ieee computer', 'acm computing surveys'],
            'biology': ['plos biology', 'elife', 'current biology', 'developmental cell'],
            'physics': ['physical review letters', 'nature physics', 'physics today'],
            'chemistry': ['jacs', 'angewandte chemie', 'chemical reviews']
        }
        
        # Impact factor tiers
        self.if_tiers = {
            'tier1': (20, float('inf')),  # IF > 20
            'tier2': (10, 20),            # IF 10-20
            'tier3': (5, 10),             # IF 5-10
            'tier4': (2, 5),              # IF 2-5
            'tier5': (0, 2)               # IF < 2
        }
    
    def _initialize_subject_rankings(self) -> None:
        """Initialize subject-specific ranking data."""
        
        self.subject_rankings = {
            'medicine': {
                'top_journals': ['nejm', 'lancet', 'jama', 'bmj', 'nature medicine'],
                'ranking_weight': 0.4  # High importance in medicine
            },
            'computer_science': {
                'top_journals': ['cacm', 'ieee computer', 'nature machine intelligence'],
                'ranking_weight': 0.3  # Moderate importance (conferences also matter)
            },
            'biology': {
                'top_journals': ['nature', 'science', 'cell', 'plos biology'],
                'ranking_weight': 0.4  # High importance in biology
            },
            'physics': {
                'top_journals': ['nature', 'science', 'physical review letters'],
                'ranking_weight': 0.35
            }
        }
    
    def _initialize_quality_indicators(self) -> None:
        """Initialize quality assessment indicators."""
        
        self.quality_indicators = {
            'editorial_quality': {
                'peer_review_rigor': 0.3,
                'editorial_board_prestige': 0.2,
                'review_turnaround_time': 0.15,
                'acceptance_rate': 0.1,
                'publication_ethics': 0.25
            },
            'technical_quality': {
                'publication_standards': 0.4,
                'data_sharing_requirements': 0.2,
                'reproducibility_standards': 0.2,
                'statistical_review': 0.2
            },
            'reach_impact': {
                'global_readership': 0.3,
                'interdisciplinary_scope': 0.2,
                'media_coverage': 0.2,
                'policy_influence': 0.3
            }
        }
    
    def analyze_manuscript_for_ranking(self, manuscript_text: str,
                                     subject_areas: Optional[List[str]] = None) -> RankingAnalysis:
        """
        Analyze manuscript to determine appropriate journal ranking tier.
        
        Args:
            manuscript_text: Full manuscript text
            subject_areas: Optional subject areas
            
        Returns:
            RankingAnalysis with recommendations
        """
        
        # Analyze manuscript quality indicators
        quality_indicators = self._assess_manuscript_quality(manuscript_text)
        
        # Determine target prestige level
        target_prestige = self._determine_target_prestige_level(
            quality_indicators, subject_areas
        )
        
        # Calculate recommended ranking range
        ranking_range = self._calculate_ranking_range(target_prestige, quality_indicators)
        
        # Generate explanation
        explanation = self._generate_ranking_explanation(
            target_prestige, quality_indicators, ranking_range
        )
        
        # Calculate quality alignment score
        alignment_score = self._calculate_quality_alignment(quality_indicators)
        
        return RankingAnalysis(
            manuscript_quality_indicators=quality_indicators,
            target_prestige_level=target_prestige,
            recommended_ranking_range=ranking_range,
            ranking_explanation=explanation,
            quality_alignment_score=alignment_score
        )
    
    def _assess_manuscript_quality(self, text: str) -> Dict[str, float]:
        """Assess manuscript quality indicators."""
        
        indicators = {}
        text_lower = text.lower()
        
        # Methodological rigor
        method_indicators = [
            'randomized', 'controlled', 'blinded', 'placebo', 'meta-analysis',
            'systematic review', 'statistical analysis', 'power analysis',
            'confidence interval', 'p-value', 'hypothesis testing'
        ]
        method_score = sum(1 for indicator in method_indicators if indicator in text_lower)
        indicators['methodological_rigor'] = min(method_score / 5.0, 1.0)
        
        # Sample size and statistical power
        sample_patterns = [
            r'n\s*=\s*(\d+)', r'sample\s+size.*?(\d+)', r'participants.*?(\d+)',
            r'patients.*?(\d+)', r'subjects.*?(\d+)'
        ]
        sample_sizes = []
        for pattern in sample_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    sample_sizes.extend([int(match) for match in matches])
                except ValueError:
                    continue
        
        if sample_sizes:
            max_sample = max(sample_sizes)
            sample_score = min(np.log10(max_sample + 1) / 4.0, 1.0)  # Log scale
            indicators['sample_size_adequacy'] = sample_score
        else:
            indicators['sample_size_adequacy'] = 0.3
        
        # Innovation indicators
        innovation_terms = [
            'novel', 'new', 'first', 'innovative', 'breakthrough', 'pioneering',
            'cutting-edge', 'state-of-the-art', 'unprecedented', 'original'
        ]
        innovation_score = sum(1 for term in innovation_terms if term in text_lower)
        indicators['innovation_level'] = min(innovation_score / 3.0, 1.0)
        
        # Clinical significance
        clinical_terms = [
            'clinical trial', 'patient', 'treatment', 'therapy', 'diagnosis',
            'medical', 'healthcare', 'clinical significance', 'therapeutic'
        ]
        clinical_score = sum(1 for term in clinical_terms if term in text_lower)
        indicators['clinical_significance'] = min(clinical_score / 4.0, 1.0)
        
        # International collaboration
        collab_indicators = [
            'university', 'institute', 'hospital', 'center', 'consortium',
            'collaboration', 'multi-center', 'international'
        ]
        collab_score = sum(1 for indicator in collab_indicators if indicator in text_lower)
        indicators['collaboration_scope'] = min(collab_score / 3.0, 1.0)
        
        # Technical sophistication
        tech_terms = [
            'algorithm', 'machine learning', 'artificial intelligence', 'big data',
            'genomics', 'proteomics', 'bioinformatics', 'computational',
            'molecular', 'genetic', 'imaging', 'spectroscopy'
        ]
        tech_score = sum(1 for term in tech_terms if term in text_lower)
        indicators['technical_sophistication'] = min(tech_score / 3.0, 1.0)
        
        return indicators
    
    def _determine_target_prestige_level(self, quality_indicators: Dict[str, float],
                                       subject_areas: Optional[List[str]]) -> PrestigeLevel:
        """Determine target journal prestige level based on manuscript quality."""
        
        # Calculate overall quality score
        quality_score = np.mean(list(quality_indicators.values()))
        
        # Adjust based on specific indicators
        high_impact_indicators = [
            'methodological_rigor', 'innovation_level', 'clinical_significance'
        ]
        high_impact_score = np.mean([
            quality_indicators.get(indicator, 0.5) 
            for indicator in high_impact_indicators
        ])
        
        # Determine prestige level
        if quality_score >= 0.8 and high_impact_score >= 0.7:
            return PrestigeLevel.ELITE
        elif quality_score >= 0.7 and high_impact_score >= 0.6:
            return PrestigeLevel.PREMIER
        elif quality_score >= 0.6:
            return PrestigeLevel.EXCELLENT
        elif quality_score >= 0.5:
            return PrestigeLevel.GOOD
        elif quality_score >= 0.4:
            return PrestigeLevel.AVERAGE
        else:
            return PrestigeLevel.EMERGING
    
    def _calculate_ranking_range(self, prestige_level: PrestigeLevel,
                               quality_indicators: Dict[str, float]) -> Tuple[int, int]:
        """Calculate recommended journal ranking range."""
        
        # Base ranges for each prestige level
        base_ranges = {
            PrestigeLevel.ELITE: (1, 50),
            PrestigeLevel.PREMIER: (50, 200),
            PrestigeLevel.EXCELLENT: (200, 500),
            PrestigeLevel.GOOD: (500, 1500),
            PrestigeLevel.AVERAGE: (1500, 3000),
            PrestigeLevel.EMERGING: (3000, 7000)
        }
        
        base_min, base_max = base_ranges[prestige_level]
        
        # Adjust based on quality indicators
        quality_score = np.mean(list(quality_indicators.values()))
        
        # Narrow range for higher quality manuscripts
        if quality_score >= 0.8:
            adjustment_factor = 0.5
        elif quality_score >= 0.6:
            adjustment_factor = 0.7
        else:
            adjustment_factor = 1.0
        
        range_size = (base_max - base_min) * adjustment_factor
        mid_point = (base_min + base_max) / 2
        
        adjusted_min = max(1, int(mid_point - range_size / 2))
        adjusted_max = min(7000, int(mid_point + range_size / 2))
        
        return (adjusted_min, adjusted_max)
    
    def _generate_ranking_explanation(self, prestige_level: PrestigeLevel,
                                    quality_indicators: Dict[str, float],
                                    ranking_range: Tuple[int, int]) -> str:
        """Generate explanation for ranking recommendation."""
        
        explanations = []
        
        # Prestige level explanation
        prestige_explanations = {
            PrestigeLevel.ELITE: "manuscript shows exceptional quality suitable for top-tier journals",
            PrestigeLevel.PREMIER: "manuscript demonstrates high quality for premier journals",
            PrestigeLevel.EXCELLENT: "manuscript shows excellent quality for well-regarded journals",
            PrestigeLevel.GOOD: "manuscript demonstrates good quality for established journals",
            PrestigeLevel.AVERAGE: "manuscript suitable for standard academic journals",
            PrestigeLevel.EMERGING: "manuscript appropriate for emerging or regional journals"
        }
        
        explanations.append(prestige_explanations[prestige_level])
        
        # Highlight strong quality indicators
        strong_indicators = [
            (indicator, score) for indicator, score in quality_indicators.items()
            if score >= 0.7
        ]
        
        if strong_indicators:
            strong_names = [indicator.replace('_', ' ') for indicator, _ in strong_indicators]
            explanations.append(f"Strong indicators: {', '.join(strong_names[:3])}")
        
        # Ranking range explanation
        min_rank, max_rank = ranking_range
        explanations.append(f"Target journals ranked #{min_rank}-{max_rank} globally")
        
        return "; ".join(explanations)
    
    def _calculate_quality_alignment(self, quality_indicators: Dict[str, float]) -> float:
        """Calculate manuscript-journal quality alignment score."""
        
        # Weight different quality aspects
        weights = {
            'methodological_rigor': 0.25,
            'innovation_level': 0.20,
            'clinical_significance': 0.15,
            'technical_sophistication': 0.15,
            'sample_size_adequacy': 0.15,
            'collaboration_scope': 0.10
        }
        
        weighted_score = sum(
            quality_indicators.get(indicator, 0.5) * weight
            for indicator, weight in weights.items()
        )
        
        return min(weighted_score, 1.0)
    
    def get_journal_ranking_score(self, journal_id: str) -> Optional[RankingMetrics]:
        """Get ranking metrics for a specific journal."""
        return self.ranking_data.get(journal_id)
    
    def rank_journals_for_manuscript(self, manuscript_text: str,
                                   journal_candidates: List[Dict[str, Any]],
                                   subject_areas: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank journal candidates based on manuscript quality and journal rankings.
        
        Args:
            manuscript_text: Manuscript text
            journal_candidates: List of journal data
            subject_areas: Optional subject areas
            
        Returns:
            List of (journal, ranking_score) tuples sorted by ranking score
        """
        
        # Analyze manuscript
        manuscript_analysis = self.analyze_manuscript_for_ranking(
            manuscript_text, subject_areas
        )
        
        ranked_journals = []
        
        for journal in journal_candidates:
            journal_id = journal.get('id', str(journal_candidates.index(journal)))
            ranking_metrics = self.get_journal_ranking_score(journal_id)
            
            if ranking_metrics:
                # Calculate ranking compatibility score
                ranking_score = self._calculate_ranking_compatibility(
                    manuscript_analysis, ranking_metrics
                )
            else:
                # Fallback scoring for journals without ranking data
                ranking_score = self._calculate_fallback_ranking_score(journal)
            
            ranked_journals.append((journal, ranking_score))
        
        # Sort by ranking score (descending)
        ranked_journals.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_journals
    
    def _calculate_ranking_compatibility(self, manuscript_analysis: RankingAnalysis,
                                       ranking_metrics: RankingMetrics) -> float:
        """Calculate compatibility between manuscript and journal ranking."""
        
        compatibility_score = 0.0
        
        # Prestige level alignment
        manuscript_prestige = manuscript_analysis.target_prestige_level
        journal_prestige = ranking_metrics.prestige_level
        
        prestige_compatibility = self._calculate_prestige_compatibility(
            manuscript_prestige, journal_prestige
        )
        compatibility_score += prestige_compatibility * 0.4
        
        # Ranking range alignment
        min_rank, max_rank = manuscript_analysis.recommended_ranking_range
        journal_rank = ranking_metrics.sjr_rank
        
        if journal_rank:
            if min_rank <= journal_rank <= max_rank:
                rank_compatibility = 1.0
            else:
                # Penalize based on distance from range
                if journal_rank < min_rank:
                    distance = min_rank - journal_rank
                    rank_compatibility = max(0, 1.0 - distance / min_rank)
                else:
                    distance = journal_rank - max_rank
                    rank_compatibility = max(0, 1.0 - distance / max_rank)
        else:
            rank_compatibility = 0.5  # Neutral if no rank data
        
        compatibility_score += rank_compatibility * 0.3
        
        # Quality alignment
        quality_alignment = manuscript_analysis.quality_alignment_score
        journal_quality = ranking_metrics.quality_score
        
        quality_compatibility = 1.0 - abs(quality_alignment - journal_quality)
        compatibility_score += quality_compatibility * 0.3
        
        return min(compatibility_score, 1.0)
    
    def _calculate_prestige_compatibility(self, manuscript_prestige: PrestigeLevel,
                                        journal_prestige: PrestigeLevel) -> float:
        """Calculate compatibility between prestige levels."""
        
        # Create prestige ordering
        prestige_order = [
            PrestigeLevel.EMERGING, PrestigeLevel.AVERAGE, PrestigeLevel.GOOD,
            PrestigeLevel.EXCELLENT, PrestigeLevel.PREMIER, PrestigeLevel.ELITE
        ]
        
        manuscript_level = prestige_order.index(manuscript_prestige)
        journal_level = prestige_order.index(journal_prestige)
        
        # Perfect match
        if manuscript_level == journal_level:
            return 1.0
        
        # Calculate compatibility based on distance
        distance = abs(manuscript_level - journal_level)
        max_distance = len(prestige_order) - 1
        
        # Allow slight upgrades (submitting to higher prestige)
        if journal_level > manuscript_level and distance <= 1:
            return 0.9
        
        # Penalize significant mismatches
        compatibility = max(0, 1.0 - (distance / max_distance) * 0.7)
        
        return compatibility
    
    def _calculate_fallback_ranking_score(self, journal: Dict[str, Any]) -> float:
        """Calculate ranking score for journals without ranking data."""
        
        fallback_score = 0.5  # Base score
        
        # Publisher-based adjustment
        publisher = journal.get('publisher', '').lower()
        publisher_bonus = self._get_publisher_prestige_score(publisher) - 0.5
        fallback_score += publisher_bonus * 0.3
        
        # Citation-based adjustment
        citations = journal.get('cited_by_count', 0)
        if citations > 0:
            citation_score = min(np.log10(citations + 1) / 7.0, 0.3)
            fallback_score += citation_score
        
        # Open access bonus
        if journal.get('is_oa', False):
            fallback_score += 0.1
        
        return min(fallback_score, 1.0)


def integrate_journal_rankings(journals: List[Dict[str, Any]], 
                             manuscript_text: str) -> List[Dict[str, Any]]:
    """
    Convenience function to integrate ranking data into journal results.
    
    Args:
        journals: List of journal data
        manuscript_text: Manuscript text for analysis
        
    Returns:
        Enhanced journal list with ranking information
    """
    
    ranker = JournalRankingIntegrator()
    
    # Analyze manuscript for ranking
    manuscript_analysis = ranker.analyze_manuscript_for_ranking(manuscript_text)
    
    # Enhance journals with ranking data
    enhanced_journals = []
    
    for journal in journals:
        journal_id = journal.get('id', str(journals.index(journal)))
        ranking_metrics = ranker.get_journal_ranking_score(journal_id)
        
        # Add ranking information to journal data
        enhanced_journal = journal.copy()
        
        if ranking_metrics:
            enhanced_journal.update({
                'ranking_metrics': {
                    'prestige_level': ranking_metrics.prestige_level.value,
                    'quality_score': ranking_metrics.quality_score,
                    'reputation_score': ranking_metrics.reputation_score,
                    'composite_ranking_score': ranking_metrics.composite_ranking_score
                },
                'manuscript_compatibility': ranker._calculate_ranking_compatibility(
                    manuscript_analysis, ranking_metrics
                )
            })
        else:
            # Add fallback ranking data
            fallback_score = ranker._calculate_fallback_ranking_score(journal)
            enhanced_journal.update({
                'ranking_metrics': {
                    'prestige_level': 'average',
                    'quality_score': fallback_score,
                    'reputation_score': fallback_score,
                    'composite_ranking_score': fallback_score
                },
                'manuscript_compatibility': fallback_score
            })
        
        enhanced_journals.append(enhanced_journal)
    
    return enhanced_journals