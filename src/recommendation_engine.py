"""
Advanced filtering and recommendation engine for Manuscript Journal Matcher.

This module provides sophisticated filtering capabilities and intelligent
recommendations based on manuscript analysis, journal rankings, citation patterns,
and user preferences to deliver highly targeted journal suggestions.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import json
from pathlib import Path
import re

try:
    from .config import DATA_DIR
    from .journal_ranker import JournalRankingIntegrator, PrestigeLevel, RankingAnalysis
    from .citation_analyzer import CitationNetworkAnalyzer, CitationAnalysis
    from .study_classifier import StudyTypeClassifier, StudyType
    from .match_journals import JournalMatcher
    from .utils import clean_text
except ImportError:
    from config import DATA_DIR
    from journal_ranker import JournalRankingIntegrator, PrestigeLevel, RankingAnalysis
    from citation_analyzer import CitationNetworkAnalyzer, CitationAnalysis
    from study_classifier import StudyTypeClassifier, StudyType
    from match_journals import JournalMatcher
    from utils import clean_text

# Set up logging
logger = logging.getLogger(__name__)


class FilterType(Enum):
    """Types of filters available."""
    PRESTIGE_LEVEL = "prestige_level"
    OPEN_ACCESS = "open_access"
    APC_RANGE = "apc_range"
    SUBJECT_AREA = "subject_area"
    PUBLISHER = "publisher"
    IMPACT_FACTOR = "impact_factor"
    CITATION_COUNT = "citation_count"
    H_INDEX = "h_index"
    STUDY_TYPE_MATCH = "study_type_match"
    TEMPORAL_RELEVANCE = "temporal_relevance"
    GEOGRAPHIC_SCOPE = "geographic_scope"
    LANGUAGE = "language"
    MANUSCRIPT_QUALITY = "manuscript_quality"


class RecommendationStrategy(Enum):
    """Recommendation strategies."""
    CONSERVATIVE = "conservative"    # Safe, established journals
    AMBITIOUS = "ambitious"         # Higher prestige targets
    BALANCED = "balanced"           # Mix of safe and ambitious
    EXPLORATORY = "exploratory"     # Include newer/emerging journals
    COST_CONSCIOUS = "cost_conscious"  # Prioritize low/no APC
    OPEN_ACCESS = "open_access"     # Prioritize OA journals


@dataclass
class FilterCriteria:
    """Comprehensive filter criteria."""
    # Prestige and quality filters
    min_prestige_level: Optional[PrestigeLevel] = None
    max_prestige_level: Optional[PrestigeLevel] = None
    min_quality_score: Optional[float] = None
    min_manuscript_compatibility: Optional[float] = None
    
    # Access and cost filters
    open_access_only: bool = False
    no_apc_only: bool = False
    max_apc: Optional[float] = None
    min_apc: Optional[float] = None
    doaj_only: bool = False
    
    # Impact and citation filters
    min_impact_factor: Optional[float] = None
    max_impact_factor: Optional[float] = None
    min_citation_count: Optional[int] = None
    min_h_index: Optional[int] = None
    min_works_count: Optional[int] = None
    
    # Subject and scope filters
    required_subjects: List[str] = field(default_factory=list)
    excluded_subjects: List[str] = field(default_factory=list)
    required_publishers: List[str] = field(default_factory=list)
    excluded_publishers: List[str] = field(default_factory=list)
    
    # Geographic and language filters
    preferred_countries: List[str] = field(default_factory=list)
    excluded_countries: List[str] = field(default_factory=list)
    required_languages: List[str] = field(default_factory=list)
    
    # Study type and temporal filters
    matching_study_types: List[StudyType] = field(default_factory=list)
    min_citation_recency: Optional[int] = None  # Years
    prefer_recent_journals: bool = False
    
    # Advanced filters
    exclude_predatory: bool = True
    min_editorial_board_size: Optional[int] = None
    require_peer_review: bool = True
    exclude_mega_journals: bool = False


@dataclass
class RecommendationWeights:
    """Weights for different recommendation factors."""
    similarity_score: float = 0.25
    prestige_match: float = 0.20
    quality_alignment: float = 0.15
    citation_compatibility: float = 0.15
    study_type_match: float = 0.10
    cost_effectiveness: float = 0.05
    temporal_relevance: float = 0.05
    publication_speed: float = 0.05


@dataclass
class RecommendationResult:
    """Individual recommendation result."""
    journal_data: Dict[str, Any]
    recommendation_score: float
    confidence: float
    recommendation_reasons: List[str]
    risk_factors: List[str]
    estimated_acceptance_probability: float
    estimated_time_to_publication: Optional[int]  # Days
    cost_analysis: Dict[str, Any]
    match_explanation: str


@dataclass
class RecommendationSuite:
    """Complete recommendation suite."""
    primary_recommendations: List[RecommendationResult]
    alternative_recommendations: List[RecommendationResult]
    aspirational_recommendations: List[RecommendationResult]
    cost_effective_recommendations: List[RecommendationResult]
    open_access_recommendations: List[RecommendationResult]
    
    manuscript_analysis_summary: Dict[str, Any]
    filter_summary: Dict[str, Any]
    recommendation_strategy: RecommendationStrategy
    total_journals_considered: int
    recommendation_timestamp: str


class AdvancedRecommendationEngine:
    """
    Advanced recommendation engine with sophisticated filtering and analysis.
    
    Combines multiple analysis methods to provide targeted journal recommendations
    with detailed explanations and risk assessments.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            data_dir: Directory containing recommendation data
        """
        self.data_dir = data_dir or DATA_DIR
        
        # Initialize analysis components
        self.journal_matcher = JournalMatcher()
        self.ranking_integrator = JournalRankingIntegrator()
        self.citation_analyzer = CitationNetworkAnalyzer()
        self.study_classifier = StudyTypeClassifier()
        
        # Load recommendation data
        self._load_recommendation_data()
        
        # Initialize recommendation strategies
        self._initialize_strategies()
        
        logger.info("AdvancedRecommendationEngine initialized")
    
    def _load_recommendation_data(self) -> None:
        """Load recommendation-specific data."""
        
        # Load predatory journal blacklist
        self.predatory_journals = set()
        predatory_file = self.data_dir / "predatory_journals.json"
        if predatory_file.exists():
            try:
                with open(predatory_file, 'r') as f:
                    data = json.load(f)
                    self.predatory_journals = set(data.get('journals', []))
            except Exception as e:
                logger.warning(f"Failed to load predatory journal list: {e}")
        
        # Load publication speed data
        self.publication_speeds = {}
        speed_file = self.data_dir / "publication_speeds.json"
        if speed_file.exists():
            try:
                with open(speed_file, 'r') as f:
                    self.publication_speeds = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load publication speed data: {e}")
        
        # Load acceptance rate data
        self.acceptance_rates = {}
        acceptance_file = self.data_dir / "acceptance_rates.json"
        if acceptance_file.exists():
            try:
                with open(acceptance_file, 'r') as f:
                    self.acceptance_rates = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load acceptance rate data: {e}")
    
    def _initialize_strategies(self) -> None:
        """Initialize recommendation strategies."""
        
        self.strategy_weights = {
            RecommendationStrategy.CONSERVATIVE: RecommendationWeights(
                similarity_score=0.30,
                prestige_match=0.15,
                quality_alignment=0.20,
                citation_compatibility=0.15,
                study_type_match=0.10,
                cost_effectiveness=0.05,
                temporal_relevance=0.03,
                publication_speed=0.02
            ),
            RecommendationStrategy.AMBITIOUS: RecommendationWeights(
                similarity_score=0.20,
                prestige_match=0.35,
                quality_alignment=0.25,
                citation_compatibility=0.10,
                study_type_match=0.05,
                cost_effectiveness=0.02,
                temporal_relevance=0.02,
                publication_speed=0.01
            ),
            RecommendationStrategy.BALANCED: RecommendationWeights(
                similarity_score=0.25,
                prestige_match=0.20,
                quality_alignment=0.15,
                citation_compatibility=0.15,
                study_type_match=0.10,
                cost_effectiveness=0.05,
                temporal_relevance=0.05,
                publication_speed=0.05
            ),
            RecommendationStrategy.COST_CONSCIOUS: RecommendationWeights(
                similarity_score=0.25,
                prestige_match=0.10,
                quality_alignment=0.15,
                citation_compatibility=0.10,
                study_type_match=0.10,
                cost_effectiveness=0.20,
                temporal_relevance=0.05,
                publication_speed=0.05
            ),
            RecommendationStrategy.OPEN_ACCESS: RecommendationWeights(
                similarity_score=0.25,
                prestige_match=0.15,
                quality_alignment=0.15,
                citation_compatibility=0.10,
                study_type_match=0.10,
                cost_effectiveness=0.15,
                temporal_relevance=0.05,
                publication_speed=0.05
            )
        }
    
    def generate_recommendations(self,
                               manuscript_text: str,
                               filter_criteria: Optional[FilterCriteria] = None,
                               strategy: RecommendationStrategy = RecommendationStrategy.BALANCED,
                               max_recommendations: int = 20) -> RecommendationSuite:
        """
        Generate comprehensive journal recommendations.
        
        Args:
            manuscript_text: Full manuscript text
            filter_criteria: Filtering criteria
            strategy: Recommendation strategy
            max_recommendations: Maximum recommendations per category
            
        Returns:
            Complete recommendation suite
        """
        
        logger.info(f"Generating recommendations with {strategy.value} strategy")
        
        # Initialize filter criteria if not provided
        if filter_criteria is None:
            filter_criteria = FilterCriteria()
        
        # Analyze manuscript comprehensively
        manuscript_analysis = self._analyze_manuscript_comprehensively(manuscript_text)
        
        # Get initial journal candidates
        journal_candidates = self._get_journal_candidates(
            manuscript_text, manuscript_analysis, filter_criteria
        )
        
        logger.info(f"Found {len(journal_candidates)} journal candidates")
        
        # Apply advanced filtering
        filtered_journals = self._apply_advanced_filters(
            journal_candidates, manuscript_analysis, filter_criteria
        )
        
        logger.info(f"After filtering: {len(filtered_journals)} journals")
        
        # Generate recommendations for each category
        recommendations = self._generate_categorized_recommendations(
            filtered_journals, manuscript_analysis, strategy, max_recommendations
        )
        
        # Create recommendation suite
        suite = RecommendationSuite(
            primary_recommendations=recommendations['primary'],
            alternative_recommendations=recommendations['alternative'],
            aspirational_recommendations=recommendations['aspirational'],
            cost_effective_recommendations=recommendations['cost_effective'],
            open_access_recommendations=recommendations['open_access'],
            manuscript_analysis_summary=self._create_analysis_summary(manuscript_analysis),
            filter_summary=self._create_filter_summary(filter_criteria),
            recommendation_strategy=strategy,
            total_journals_considered=len(journal_candidates),
            recommendation_timestamp=self._get_timestamp()
        )
        
        logger.info("Recommendation suite generated successfully")
        return suite
    
    def _analyze_manuscript_comprehensively(self, manuscript_text: str) -> Dict[str, Any]:
        """Perform comprehensive manuscript analysis."""
        
        analysis = {
            'text_length': len(manuscript_text),
            'word_count': len(manuscript_text.split())
        }
        
        # Study type classification
        try:
            study_classification = self.study_classifier.classify_study_type(manuscript_text)
            analysis['study_classification'] = study_classification
            logger.info(f"Study type: {study_classification.primary_type.value}")
        except Exception as e:
            logger.warning(f"Study classification failed: {e}")
            analysis['study_classification'] = None
        
        # Ranking analysis
        try:
            ranking_analysis = self.ranking_integrator.analyze_manuscript_for_ranking(manuscript_text)
            analysis['ranking_analysis'] = ranking_analysis
            logger.info(f"Target prestige: {ranking_analysis.target_prestige_level.value}")
        except Exception as e:
            logger.warning(f"Ranking analysis failed: {e}")
            analysis['ranking_analysis'] = None
        
        # Citation analysis
        try:
            citation_analysis = self.citation_analyzer.analyze_citations(manuscript_text)
            analysis['citation_analysis'] = citation_analysis
            logger.info(f"Citations found: {len(citation_analysis.references)}")
        except Exception as e:
            logger.warning(f"Citation analysis failed: {e}")
            analysis['citation_analysis'] = None
        
        # Content quality assessment
        analysis['content_quality'] = self._assess_content_quality(manuscript_text)
        
        # Subject area detection
        analysis['detected_subjects'] = self._detect_subject_areas(manuscript_text)
        
        return analysis
    
    def _get_journal_candidates(self,
                              manuscript_text: str,
                              manuscript_analysis: Dict[str, Any],
                              filter_criteria: FilterCriteria) -> List[Dict[str, Any]]:
        """Get initial journal candidates."""
        
        # Load journal database
        self.journal_matcher.load_database()
        
        # Get base similarity matches
        candidates = self.journal_matcher.search_similar_journals(
            query_text=manuscript_text,
            top_k=500,  # Get large initial set
            min_similarity=0.1,
            include_ranking_analysis=True,
            use_multimodal_analysis=True,
            include_study_classification=True
        )
        
        return candidates
    
    def _apply_advanced_filters(self,
                              journals: List[Dict[str, Any]],
                              manuscript_analysis: Dict[str, Any],
                              filter_criteria: FilterCriteria) -> List[Dict[str, Any]]:
        """Apply advanced filtering criteria."""
        
        filtered = []
        
        for journal in journals:
            if self._passes_all_filters(journal, manuscript_analysis, filter_criteria):
                filtered.append(journal)
        
        return filtered
    
    def _passes_all_filters(self,
                          journal: Dict[str, Any],
                          manuscript_analysis: Dict[str, Any],
                          criteria: FilterCriteria) -> bool:
        """Check if journal passes all filter criteria."""
        
        # Predatory journal filter
        if criteria.exclude_predatory:
            journal_name = journal.get('display_name', '').lower()
            if journal_name in self.predatory_journals:
                return False
        
        # Prestige level filters
        ranking_data = journal.get('ranking_metrics', {})
        journal_prestige = ranking_data.get('prestige_level', 'average')
        
        if criteria.min_prestige_level:
            if not self._meets_min_prestige(journal_prestige, criteria.min_prestige_level):
                return False
        
        if criteria.max_prestige_level:
            if not self._meets_max_prestige(journal_prestige, criteria.max_prestige_level):
                return False
        
        # Quality score filter
        if criteria.min_quality_score:
            quality_score = ranking_data.get('quality_score', 0)
            if quality_score < criteria.min_quality_score:
                return False
        
        # Manuscript compatibility filter  
        if criteria.min_manuscript_compatibility:
            compatibility = journal.get('manuscript_compatibility', 0)
            if compatibility < criteria.min_manuscript_compatibility:
                return False
        
        # Open access filters
        if criteria.open_access_only:
            if not journal.get('oa_status', journal.get('is_oa', False)):
                return False
        
        if criteria.doaj_only:
            if not journal.get('in_doaj', False):
                return False
        
        # APC filters
        apc_amount = journal.get('apc_amount') or journal.get('apc_usd')
        if apc_amount is None:
            apc_amount = 0
        
        if criteria.no_apc_only:
            has_apc = journal.get('has_apc', False)
            if has_apc or (apc_amount and apc_amount > 0):
                return False
        
        if criteria.max_apc and apc_amount:
            if apc_amount > criteria.max_apc:
                return False
        
        if criteria.min_apc and apc_amount:
            if apc_amount < criteria.min_apc:
                return False
        
        # Citation and impact filters
        if criteria.min_citation_count:
            citations = journal.get('cited_by_count', 0)
            if citations < criteria.min_citation_count:
                return False
        
        if criteria.min_h_index:
            h_index = journal.get('h_index', 0)
            if h_index < criteria.min_h_index:
                return False
        
        if criteria.min_works_count:
            works = journal.get('works_count', 0)
            if works < criteria.min_works_count:
                return False
        
        # Subject area filters
        journal_subjects = self._get_journal_subjects(journal)
        
        if criteria.required_subjects:
            if not any(req_subj.lower() in subject.lower() 
                      for req_subj in criteria.required_subjects 
                      for subject in journal_subjects):
                return False
        
        if criteria.excluded_subjects:
            if any(excl_subj.lower() in subject.lower() 
                  for excl_subj in criteria.excluded_subjects 
                  for subject in journal_subjects):
                return False
        
        # Publisher filters
        publisher = journal.get('publisher', '').lower()
        
        if criteria.required_publishers:
            if not any(req_pub.lower() in publisher 
                      for req_pub in criteria.required_publishers):
                return False
        
        if criteria.excluded_publishers:
            if any(excl_pub.lower() in publisher 
                  for excl_pub in criteria.excluded_publishers):
                return False
        
        # Language filters
        if criteria.required_languages:
            journal_languages = journal.get('languages', [])
            if not any(lang.lower() in [jl.lower() for jl in journal_languages] 
                      for lang in criteria.required_languages):
                return False
        
        # Study type matching
        if criteria.matching_study_types and manuscript_analysis.get('study_classification'):
            manuscript_study_type = manuscript_analysis['study_classification'].primary_type
            if manuscript_study_type not in criteria.matching_study_types:
                # This is a soft filter - don't exclude but will affect scoring
                pass
        
        return True
    
    def _generate_categorized_recommendations(self,
                                           journals: List[Dict[str, Any]],
                                           manuscript_analysis: Dict[str, Any],
                                           strategy: RecommendationStrategy,
                                           max_per_category: int) -> Dict[str, List[RecommendationResult]]:
        """Generate recommendations for each category."""
        
        # Score all journals
        scored_journals = []
        for journal in journals:
            recommendation_result = self._create_recommendation_result(
                journal, manuscript_analysis, strategy
            )
            scored_journals.append(recommendation_result)
        
        # Sort by recommendation score
        scored_journals.sort(key=lambda x: x.recommendation_score, reverse=True)
        
        # Categorize recommendations
        categories = {
            'primary': [],
            'alternative': [],
            'aspirational': [],
            'cost_effective': [],
            'open_access': []
        }
        
        # Primary recommendations (top overall scores)
        categories['primary'] = scored_journals[:max_per_category]
        
        # Alternative recommendations (good scores, different from primary)
        used_journals = {r.journal_data.get('id') for r in categories['primary']}
        alternatives = [r for r in scored_journals[max_per_category:] 
                       if r.journal_data.get('id') not in used_journals]
        categories['alternative'] = alternatives[:max_per_category]
        
        # Aspirational recommendations (high prestige)
        aspirational = [r for r in scored_journals 
                       if self._is_aspirational(r) and 
                       r.journal_data.get('id') not in used_journals]
        categories['aspirational'] = aspirational[:max_per_category]
        
        # Cost-effective recommendations (low/no APC)
        cost_effective = [r for r in scored_journals 
                         if self._is_cost_effective(r) and 
                         r.journal_data.get('id') not in used_journals]
        categories['cost_effective'] = cost_effective[:max_per_category]
        
        # Open access recommendations
        open_access = [r for r in scored_journals 
                      if self._is_open_access(r) and 
                      r.journal_data.get('id') not in used_journals]
        categories['open_access'] = open_access[:max_per_category]
        
        return categories
    
    def _create_recommendation_result(self,
                                    journal: Dict[str, Any],
                                    manuscript_analysis: Dict[str, Any],
                                    strategy: RecommendationStrategy) -> RecommendationResult:
        """Create a comprehensive recommendation result."""
        
        # Calculate recommendation score
        rec_score = self._calculate_recommendation_score(
            journal, manuscript_analysis, strategy
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(journal, manuscript_analysis)
        
        # Generate recommendation reasons
        reasons = self._generate_recommendation_reasons(journal, manuscript_analysis)
        
        # Identify risk factors
        risks = self._identify_risk_factors(journal, manuscript_analysis)
        
        # Estimate acceptance probability
        acceptance_prob = self._estimate_acceptance_probability(journal, manuscript_analysis)
        
        # Estimate time to publication
        pub_time = self._estimate_publication_time(journal)
        
        # Analyze costs
        cost_analysis = self._analyze_costs(journal)
        
        # Generate match explanation
        explanation = self._generate_match_explanation(journal, manuscript_analysis)
        
        return RecommendationResult(
            journal_data=journal,
            recommendation_score=rec_score,
            confidence=confidence,
            recommendation_reasons=reasons,
            risk_factors=risks,
            estimated_acceptance_probability=acceptance_prob,
            estimated_time_to_publication=pub_time,
            cost_analysis=cost_analysis,
            match_explanation=explanation
        )
    
    def _calculate_recommendation_score(self,
                                      journal: Dict[str, Any],
                                      manuscript_analysis: Dict[str, Any],
                                      strategy: RecommendationStrategy) -> float:
        """Calculate comprehensive recommendation score."""
        
        weights = self.strategy_weights[strategy]
        score = 0.0
        
        # Similarity score component
        similarity = journal.get('similarity_score', 0)
        score += similarity * weights.similarity_score
        
        # Prestige match component
        prestige_match = self._calculate_prestige_match(journal, manuscript_analysis)
        score += prestige_match * weights.prestige_match
        
        # Quality alignment component
        quality_alignment = journal.get('manuscript_compatibility', 0.5)
        score += quality_alignment * weights.quality_alignment
        
        # Citation compatibility component
        citation_compat = self._calculate_citation_compatibility(journal, manuscript_analysis)
        score += citation_compat * weights.citation_compatibility
        
        # Study type match component
        study_match = self._calculate_study_type_match(journal, manuscript_analysis)
        score += study_match * weights.study_type_match
        
        # Cost effectiveness component
        cost_effectiveness = self._calculate_cost_effectiveness(journal)
        score += cost_effectiveness * weights.cost_effectiveness
        
        # Temporal relevance component
        temporal_relevance = self._calculate_temporal_relevance(journal, manuscript_analysis)
        score += temporal_relevance * weights.temporal_relevance
        
        # Publication speed component
        pub_speed = self._calculate_publication_speed_score(journal)
        score += pub_speed * weights.publication_speed
        
        return min(score, 1.0)
    
    def _meets_min_prestige(self, journal_prestige: str, min_prestige: PrestigeLevel) -> bool:
        """Check if journal meets minimum prestige level."""
        prestige_order = [
            PrestigeLevel.EMERGING, PrestigeLevel.AVERAGE, PrestigeLevel.GOOD,
            PrestigeLevel.EXCELLENT, PrestigeLevel.PREMIER, PrestigeLevel.ELITE
        ]
        
        try:
            journal_level = PrestigeLevel(journal_prestige)
            journal_index = prestige_order.index(journal_level)
            min_index = prestige_order.index(min_prestige)
            return journal_index >= min_index
        except (ValueError, KeyError):
            return False
    
    def _meets_max_prestige(self, journal_prestige: str, max_prestige: PrestigeLevel) -> bool:
        """Check if journal meets maximum prestige level."""
        prestige_order = [
            PrestigeLevel.EMERGING, PrestigeLevel.AVERAGE, PrestigeLevel.GOOD,
            PrestigeLevel.EXCELLENT, PrestigeLevel.PREMIER, PrestigeLevel.ELITE
        ]
        
        try:
            journal_level = PrestigeLevel(journal_prestige)
            journal_index = prestige_order.index(journal_level)
            max_index = prestige_order.index(max_prestige)
            return journal_index <= max_index
        except (ValueError, KeyError):
            return True  # Default to allowing if unclear
    
    def _get_journal_subjects(self, journal: Dict[str, Any]) -> List[str]:
        """Extract journal subject areas."""
        subjects = []
        
        # OpenAlex subjects
        for s in journal.get('subjects', []):
            if isinstance(s, dict) and s.get('name'):
                subjects.append(s['name'])
            elif isinstance(s, str):
                subjects.append(s)
        
        # DOAJ subjects
        for s in journal.get('subjects_doaj', []):
            if isinstance(s, str):
                subjects.append(s)
        
        return subjects
    
    def _calculate_prestige_match(self, journal: Dict[str, Any], 
                                manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate prestige level match."""
        ranking_analysis = manuscript_analysis.get('ranking_analysis')
        if not ranking_analysis:
            return 0.5
        
        manuscript_prestige = ranking_analysis.target_prestige_level
        journal_ranking = journal.get('ranking_metrics', {})
        journal_prestige = journal_ranking.get('prestige_level', 'average')
        
        try:
            journal_prestige_enum = PrestigeLevel(journal_prestige)
        except ValueError:
            return 0.5
        
        # Calculate compatibility
        prestige_order = [
            PrestigeLevel.EMERGING, PrestigeLevel.AVERAGE, PrestigeLevel.GOOD,
            PrestigeLevel.EXCELLENT, PrestigeLevel.PREMIER, PrestigeLevel.ELITE
        ]
        
        manuscript_level = prestige_order.index(manuscript_prestige)
        journal_level = prestige_order.index(journal_prestige_enum)
        
        # Perfect match
        if manuscript_level == journal_level:
            return 1.0
        
        # Allow slight upgrades
        if journal_level > manuscript_level and (journal_level - manuscript_level) <= 1:
            return 0.9
        
        # Penalize significant mismatches
        distance = abs(manuscript_level - journal_level)
        max_distance = len(prestige_order) - 1
        return max(0, 1.0 - (distance / max_distance) * 0.7)
    
    def _calculate_citation_compatibility(self, journal: Dict[str, Any],
                                        manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate citation pattern compatibility."""
        citation_analysis = manuscript_analysis.get('citation_analysis')
        if not citation_analysis:
            return 0.5
        
        # Get journal name
        journal_name = journal.get('display_name', '')
        
        # Check if journal appears in citations
        compatibility = self.citation_analyzer.get_journal_compatibility_score(
            citation_analysis, journal_name
        )
        
        return compatibility
    
    def _calculate_study_type_match(self, journal: Dict[str, Any],
                                  manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate study type matching score."""
        study_classification = manuscript_analysis.get('study_classification')
        if not study_classification:
            return 0.5
        
        # This would require subject-specific matching logic
        # For now, return neutral score
        return 0.5
    
    def _calculate_cost_effectiveness(self, journal: Dict[str, Any]) -> float:
        """Calculate cost effectiveness score."""
        apc_amount = journal.get('apc_amount') or journal.get('apc_usd')
        
        if apc_amount is None or apc_amount == 0:
            return 1.0  # Free publication
        
        # Normalize APC (assuming max reasonable APC is $5000)
        normalized_apc = min(apc_amount / 5000.0, 1.0)
        cost_effectiveness = 1.0 - normalized_apc
        
        # Bonus for open access
        if journal.get('oa_status', journal.get('is_oa', False)):
            cost_effectiveness += 0.1
        
        return min(cost_effectiveness, 1.0)
    
    def _calculate_temporal_relevance(self, journal: Dict[str, Any],
                                    manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate temporal relevance score."""
        citation_analysis = manuscript_analysis.get('citation_analysis')
        if not citation_analysis or not citation_analysis.temporal_distribution:
            return 0.5
        
        # Check recency of citations
        recent_years = [year for year in citation_analysis.temporal_distribution.keys() 
                       if year >= 2020]
        
        if not recent_years:
            return 0.3  # Old citations
        
        recent_ratio = len(recent_years) / len(citation_analysis.temporal_distribution)
        return min(recent_ratio * 1.5, 1.0)  # Boost recent citations
    
    def _calculate_publication_speed_score(self, journal: Dict[str, Any]) -> float:
        """Calculate publication speed score."""
        journal_name = journal.get('display_name', '')
        
        # Look up publication speed data
        speed_data = self.publication_speeds.get(journal_name)
        if not speed_data:
            return 0.5  # Neutral if unknown
        
        # Convert days to score (faster = better)
        days = speed_data.get('average_days', 180)
        # Normalize assuming 180 days is average, 60 days is excellent
        speed_score = max(0, 1.0 - (days - 60) / 120.0)
        
        return min(speed_score, 1.0)
    
    def _calculate_confidence(self, journal: Dict[str, Any],
                            manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate recommendation confidence."""
        
        confidence_factors = []
        
        # Similarity confidence
        similarity = journal.get('similarity_score', 0)
        confidence_factors.append(similarity)
        
        # Data completeness
        ranking_data = journal.get('ranking_metrics', {})
        if ranking_data.get('prestige_level'):
            confidence_factors.append(0.8)
        
        if journal.get('manuscript_compatibility'):
            confidence_factors.append(0.7)
        
        # Citation analysis confidence
        citation_analysis = manuscript_analysis.get('citation_analysis')
        if citation_analysis and len(citation_analysis.references) > 5:
            confidence_factors.append(0.8)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _generate_recommendation_reasons(self, journal: Dict[str, Any],
                                       manuscript_analysis: Dict[str, Any]) -> List[str]:
        """Generate reasons for recommendation."""
        
        reasons = []
        
        # Similarity reason
        similarity = journal.get('similarity_score', 0)
        if similarity >= 0.7:
            reasons.append(f"High content similarity ({similarity:.2f})")
        elif similarity >= 0.5:
            reasons.append(f"Good content similarity ({similarity:.2f})")
        
        # Prestige reason
        ranking_data = journal.get('ranking_metrics', {})
        prestige = ranking_data.get('prestige_level')
        if prestige in ['elite', 'premier']:
            reasons.append(f"High prestige journal ({prestige})")
        
        # Quality reason
        quality = ranking_data.get('quality_score', 0)
        if quality >= 0.7:
            reasons.append(f"High quality score ({quality:.2f})")
        
        # Open access reason
        if journal.get('oa_status', journal.get('is_oa', False)):
            reasons.append("Open access publication")
        
        # Cost reason
        apc = journal.get('apc_amount') or journal.get('apc_usd')
        if not apc or apc == 0:
            reasons.append("No publication fees")
        elif apc < 1000:
            reasons.append(f"Low publication fees (${apc})")
        
        # Citation reason
        if manuscript_analysis.get('citation_analysis'):
            journal_name = journal.get('display_name', '')
            citations = manuscript_analysis['citation_analysis'].cited_journals
            if journal_name in citations:
                count = citations[journal_name]
                reasons.append(f"Cited {count} times in your references")
        
        return reasons[:5]  # Limit to top 5 reasons
    
    def _identify_risk_factors(self, journal: Dict[str, Any],
                             manuscript_analysis: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors."""
        
        risks = []
        
        # High APC risk
        apc = journal.get('apc_amount') or journal.get('apc_usd')
        if apc and apc > 3000:
            risks.append(f"High publication fees (${apc})")
        
        # Low acceptance rate (if available)
        journal_name = journal.get('display_name', '')
        acceptance_data = self.acceptance_rates.get(journal_name)
        if acceptance_data and acceptance_data.get('rate', 1.0) < 0.2:
            risks.append(f"Low acceptance rate ({acceptance_data['rate']*100:.1f}%)")
        
        # Prestige mismatch
        ranking_analysis = manuscript_analysis.get('ranking_analysis')
        if ranking_analysis:
            manuscript_prestige = ranking_analysis.target_prestige_level
            journal_prestige = journal.get('ranking_metrics', {}).get('prestige_level', 'average')
            
            try:
                journal_prestige_enum = PrestigeLevel(journal_prestige)
                prestige_order = [
                    PrestigeLevel.EMERGING, PrestigeLevel.AVERAGE, PrestigeLevel.GOOD,
                    PrestigeLevel.EXCELLENT, PrestigeLevel.PREMIER, PrestigeLevel.ELITE
                ]
                
                manuscript_level = prestige_order.index(manuscript_prestige)
                journal_level = prestige_order.index(journal_prestige_enum)
                
                if journal_level > manuscript_level + 2:
                    risks.append("Journal prestige may be too high for manuscript quality")
                elif journal_level < manuscript_level - 1:
                    risks.append("Journal prestige below manuscript potential")
            except (ValueError, KeyError):
                pass
        
        # New or small journal
        works_count = journal.get('works_count', 0)
        if works_count < 100:
            risks.append("Relatively new journal with limited track record")
        
        return risks[:3]  # Limit to top 3 risks
    
    def _estimate_acceptance_probability(self, journal: Dict[str, Any],
                                       manuscript_analysis: Dict[str, Any]) -> float:
        """Estimate acceptance probability."""
        
        # Base probability on similarity and compatibility
        similarity = journal.get('similarity_score', 0)
        compatibility = journal.get('manuscript_compatibility', 0.5)
        
        base_prob = (similarity + compatibility) / 2
        
        # Adjust based on prestige mismatch
        ranking_analysis = manuscript_analysis.get('ranking_analysis')
        if ranking_analysis:
            manuscript_prestige = ranking_analysis.target_prestige_level
            journal_prestige = journal.get('ranking_metrics', {}).get('prestige_level', 'average')
            
            prestige_adjustment = self._calculate_prestige_match(journal, manuscript_analysis)
            base_prob = base_prob * (0.5 + prestige_adjustment * 0.5)
        
        # Adjust based on known acceptance rates
        journal_name = journal.get('display_name', '')
        acceptance_data = self.acceptance_rates.get(journal_name)
        if acceptance_data:
            known_rate = acceptance_data.get('rate', 0.5)
            base_prob = base_prob * (0.5 + known_rate * 0.5)
        
        return min(base_prob, 0.95)  # Cap at 95%
    
    def _estimate_publication_time(self, journal: Dict[str, Any]) -> Optional[int]:
        """Estimate time to publication in days."""
        
        journal_name = journal.get('display_name', '')
        speed_data = self.publication_speeds.get(journal_name)
        
        if speed_data:
            return speed_data.get('average_days')
        
        # Default estimates based on journal type
        if journal.get('is_oa', False):
            return 120  # OA journals often faster
        
        ranking_data = journal.get('ranking_metrics', {})
        prestige = ranking_data.get('prestige_level', 'average')
        
        if prestige in ['elite', 'premier']:
            return 200  # High prestige journals often slower
        else:
            return 150  # Average estimate
    
    def _analyze_costs(self, journal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze publication costs."""
        
        apc = journal.get('apc_amount') or journal.get('apc_usd')
        if apc is None:
            apc = 0
        currency = journal.get('apc_currency', 'USD')
        
        analysis = {
            'apc_amount': apc,
            'currency': currency,
            'is_free': apc == 0,
            'cost_category': self._categorize_cost(apc)
        }
        
        # Add cost context
        if apc == 0:
            analysis['cost_note'] = "No publication fees"
        elif apc < 500:
            analysis['cost_note'] = "Low cost publication"
        elif apc < 2000:
            analysis['cost_note'] = "Moderate cost publication"
        elif apc < 4000:
            analysis['cost_note'] = "High cost publication"
        else:
            analysis['cost_note'] = "Very high cost publication"
        
        return analysis
    
    def _categorize_cost(self, apc: Optional[float]) -> str:
        """Categorize APC cost."""
        if apc is None or apc == 0:
            return "free"
        elif apc < 500:
            return "low"
        elif apc < 2000:
            return "moderate"
        elif apc < 4000:
            return "high"
        else:
            return "very_high"
    
    def _generate_match_explanation(self, journal: Dict[str, Any],
                                  manuscript_analysis: Dict[str, Any]) -> str:
        """Generate detailed match explanation."""
        
        explanations = []
        
        # Similarity explanation
        similarity = journal.get('similarity_score', 0)
        if similarity >= 0.7:
            explanations.append("excellent content alignment")
        elif similarity >= 0.5:
            explanations.append("good content alignment")
        else:
            explanations.append("moderate content alignment")
        
        # Quality explanation
        ranking_data = journal.get('ranking_metrics', {})
        quality = ranking_data.get('quality_score', 0)
        if quality >= 0.7:
            explanations.append("high journal quality")
        elif quality >= 0.5:
            explanations.append("good journal quality")
        
        # Prestige explanation
        prestige = ranking_data.get('prestige_level', 'average')
        if prestige in ['elite', 'premier']:
            explanations.append("high prestige venue")
        elif prestige in ['excellent', 'good']:
            explanations.append("well-regarded venue")
        
        return ", ".join(explanations[:3])
    
    def _is_aspirational(self, recommendation: RecommendationResult) -> bool:
        """Check if recommendation is aspirational."""
        ranking_data = recommendation.journal_data.get('ranking_metrics', {})
        prestige = ranking_data.get('prestige_level', 'average')
        return prestige in ['elite', 'premier']
    
    def _is_cost_effective(self, recommendation: RecommendationResult) -> bool:
        """Check if recommendation is cost-effective."""
        apc = recommendation.journal_data.get('apc_amount') or recommendation.journal_data.get('apc_usd')
        if apc is None:
            apc = 0
        return apc < 1000  # Under $1000 APC
    
    def _is_open_access(self, recommendation: RecommendationResult) -> bool:
        """Check if recommendation is open access."""
        return recommendation.journal_data.get('oa_status', recommendation.journal_data.get('is_oa', False))
    
    def _assess_content_quality(self, text: str) -> Dict[str, float]:
        """Assess manuscript content quality."""
        
        quality = {}
        text_lower = text.lower()
        
        # Length assessment
        word_count = len(text.split())
        if word_count > 5000:
            quality['length_adequacy'] = 1.0
        elif word_count > 2000:
            quality['length_adequacy'] = 0.8
        else:
            quality['length_adequacy'] = 0.6
        
        # Structure assessment
        sections = ['abstract', 'introduction', 'method', 'result', 'discussion', 'conclusion']
        section_count = sum(1 for section in sections if section in text_lower)
        quality['structure_completeness'] = min(section_count / len(sections), 1.0)
        
        # Reference assessment
        ref_patterns = [r'references', r'\[\d+\]', r'\(\w+\s+et\s+al\.,\s+\d{4}\)']
        has_references = any(re.search(pattern, text_lower) for pattern in ref_patterns)
        quality['citation_adequacy'] = 1.0 if has_references else 0.3
        
        return quality
    
    def _detect_subject_areas(self, text: str) -> List[str]:
        """Detect subject areas from manuscript text."""
        
        subject_keywords = {
            'medicine': ['patient', 'clinical', 'medical', 'disease', 'treatment', 'therapy'],
            'computer_science': ['algorithm', 'machine learning', 'neural network', 'artificial intelligence'],
            'biology': ['cell', 'protein', 'gene', 'organism', 'biological', 'molecular'],
            'physics': ['quantum', 'particle', 'energy', 'wave', 'matter'],
            'chemistry': ['molecule', 'chemical', 'reaction', 'compound', 'synthesis'],
            'psychology': ['behavior', 'cognitive', 'mental', 'psychological', 'brain'],
            'engineering': ['system', 'design', 'optimization', 'performance', 'technology']
        }
        
        text_lower = text.lower()
        detected_subjects = []
        
        for subject, keywords in subject_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score >= 2:  # Require at least 2 matching keywords
                detected_subjects.append(subject)
        
        return detected_subjects
    
    def _create_analysis_summary(self, manuscript_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of manuscript analysis."""
        
        summary = {
            'word_count': manuscript_analysis.get('word_count', 0),
            'text_length': manuscript_analysis.get('text_length', 0)
        }
        
        # Study classification summary
        study_classification = manuscript_analysis.get('study_classification')
        if study_classification:
            summary['study_type'] = study_classification.primary_type.value
            summary['study_confidence'] = study_classification.confidence
        
        # Ranking analysis summary
        ranking_analysis = manuscript_analysis.get('ranking_analysis')
        if ranking_analysis:
            summary['target_prestige'] = ranking_analysis.target_prestige_level.value
            summary['quality_score'] = ranking_analysis.quality_alignment_score
        
        # Citation analysis summary
        citation_analysis = manuscript_analysis.get('citation_analysis')
        if citation_analysis:
            summary['references_found'] = len(citation_analysis.references)
            summary['unique_journals'] = len(citation_analysis.cited_journals)
        
        # Content quality summary
        content_quality = manuscript_analysis.get('content_quality', {})
        if content_quality:
            summary['content_quality_avg'] = np.mean(list(content_quality.values()))
        
        return summary
    
    def _create_filter_summary(self, filter_criteria: FilterCriteria) -> Dict[str, Any]:
        """Create summary of applied filters."""
        
        summary = {}
        
        if filter_criteria.min_prestige_level:
            summary['min_prestige'] = filter_criteria.min_prestige_level.value
        
        if filter_criteria.open_access_only:
            summary['open_access_only'] = True
        
        if filter_criteria.max_apc:
            summary['max_apc'] = filter_criteria.max_apc
        
        if filter_criteria.required_subjects:
            summary['required_subjects'] = filter_criteria.required_subjects
        
        if filter_criteria.exclude_predatory:
            summary['exclude_predatory'] = True
        
        return summary
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


def create_recommendation_filters(
    strategy: RecommendationStrategy = RecommendationStrategy.BALANCED,
    **kwargs
) -> FilterCriteria:
    """
    Create filter criteria based on strategy and custom parameters.
    
    Args:
        strategy: Recommendation strategy
        **kwargs: Custom filter parameters
        
    Returns:
        FilterCriteria object
    """
    
    # Strategy-based defaults
    if strategy == RecommendationStrategy.CONSERVATIVE:
        filters = FilterCriteria(
            min_prestige_level=PrestigeLevel.GOOD,
            min_quality_score=0.6,
            exclude_predatory=True,
            require_peer_review=True
        )
    elif strategy == RecommendationStrategy.AMBITIOUS:
        filters = FilterCriteria(
            min_prestige_level=PrestigeLevel.EXCELLENT,
            min_quality_score=0.7,
            exclude_predatory=True
        )
    elif strategy == RecommendationStrategy.COST_CONSCIOUS:
        filters = FilterCriteria(
            max_apc=1000,
            no_apc_only=False,
            exclude_predatory=True
        )
    elif strategy == RecommendationStrategy.OPEN_ACCESS:
        filters = FilterCriteria(
            open_access_only=True,
            doaj_only=False,
            exclude_predatory=True
        )
    else:  # BALANCED
        filters = FilterCriteria(
            min_prestige_level=PrestigeLevel.AVERAGE,
            min_quality_score=0.5,
            exclude_predatory=True
        )
    
    # Apply custom overrides
    for key, value in kwargs.items():
        if hasattr(filters, key):
            setattr(filters, key, value)
    
    return filters


def analyze_recommendation_suite(suite: RecommendationSuite) -> Dict[str, Any]:
    """
    Analyze recommendation suite to provide insights.
    
    Args:
        suite: Recommendation suite
        
    Returns:
        Analysis insights
    """
    
    insights = {
        'total_recommendations': (
            len(suite.primary_recommendations) +
            len(suite.alternative_recommendations) +
            len(suite.aspirational_recommendations) +
            len(suite.cost_effective_recommendations) +
            len(suite.open_access_recommendations)
        ),
        'strategy_used': suite.recommendation_strategy.value,
        'journals_considered': suite.total_journals_considered
    }
    
    # Analyze score distributions
    all_recs = (suite.primary_recommendations + 
                suite.alternative_recommendations +
                suite.aspirational_recommendations)
    
    if all_recs:
        scores = [r.recommendation_score for r in all_recs]
        insights['score_stats'] = {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
        
        # Analyze confidence
        confidences = [r.confidence for r in all_recs]
        insights['confidence_stats'] = {
            'mean': np.mean(confidences),
            'high_confidence_count': sum(1 for c in confidences if c >= 0.7)
        }
        
        # Analyze acceptance probabilities
        accept_probs = [r.estimated_acceptance_probability for r in all_recs]
        insights['acceptance_stats'] = {
            'mean': np.mean(accept_probs),
            'high_probability_count': sum(1 for p in accept_probs if p >= 0.6)
        }
    
    return insights