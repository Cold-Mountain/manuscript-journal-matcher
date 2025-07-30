"""
Ensemble matching module for Manuscript Journal Matcher.

This module implements ensemble methods that combine multiple matching approaches
to provide more robust and accurate journal recommendations. It includes semantic
similarity, keyword matching, study type matching, and content structure matching.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import re

try:
    from .config import DATA_DIR
    from .embedder import embed_text, cosine_similarity_single
    from .utils import clean_text, extract_keywords
    from .study_classifier import StudyTypeClassifier, StudyType
    from .multimodal_analyzer import MultiModalContentAnalyzer, ContentSection
    from .match_journals import JournalMatcher
    from .citation_analyzer import CitationNetworkAnalyzer
except ImportError:
    from config import DATA_DIR
    from embedder import embed_text, cosine_similarity_single
    from utils import clean_text, extract_keywords
    from study_classifier import StudyTypeClassifier, StudyType
    from multimodal_analyzer import MultiModalContentAnalyzer, ContentSection
    from match_journals import JournalMatcher
    from citation_analyzer import CitationNetworkAnalyzer

# Set up logging
logger = logging.getLogger(__name__)


class MatchingMethod(Enum):
    """Enumeration of available matching methods."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_MATCHING = "keyword_matching"
    STUDY_TYPE_MATCHING = "study_type_matching"
    STRUCTURAL_MATCHING = "structural_matching"
    SUBJECT_MATCHING = "subject_matching"
    PUBLISHER_PREFERENCE = "publisher_preference"
    IMPACT_WEIGHTED = "impact_weighted"
    MULTIMODAL_ANALYSIS = "multimodal_analysis"
    CITATION_NETWORK = "citation_network"


@dataclass
class MatchingResult:
    """Result from a single matching method."""
    method: MatchingMethod
    journal_scores: Dict[str, float]  # journal_id -> score
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class EnsembleResult:
    """Final ensemble matching result."""
    journal_id: str
    journal_data: Dict[str, Any]
    ensemble_score: float
    individual_scores: Dict[MatchingMethod, float]
    confidence: float
    rank: int
    explanation: str


class EnsembleJournalMatcher:
    """
    Advanced ensemble matcher that combines multiple matching strategies.
    
    Uses weighted voting from different matching methods to provide
    more robust and accurate journal recommendations.
    """
    
    def __init__(self, method_weights: Optional[Dict[MatchingMethod, float]] = None):
        """
        Initialize the ensemble matcher.
        
        Args:
            method_weights: Custom weights for different matching methods
        """
        # Default method weights (can be customized)
        self.method_weights = method_weights or {
            MatchingMethod.SEMANTIC_SIMILARITY: 0.22,
            MatchingMethod.MULTIMODAL_ANALYSIS: 0.18,
            MatchingMethod.CITATION_NETWORK: 0.15,
            MatchingMethod.STUDY_TYPE_MATCHING: 0.13,
            MatchingMethod.KEYWORD_MATCHING: 0.12,
            MatchingMethod.SUBJECT_MATCHING: 0.08,
            MatchingMethod.STRUCTURAL_MATCHING: 0.07,
            MatchingMethod.IMPACT_WEIGHTED: 0.03,
            MatchingMethod.PUBLISHER_PREFERENCE: 0.02
        }
        
        # Initialize component matchers
        self.journal_matcher = JournalMatcher()
        self.study_classifier = StudyTypeClassifier()
        self.multimodal_analyzer = MultiModalContentAnalyzer()
        self.citation_analyzer = CitationNetworkAnalyzer()
        
        # Load journal database
        self.journal_matcher.load_database()
        self.journals = self.journal_matcher.journals
        
        logger.info(f"EnsembleJournalMatcher initialized with {len(self.method_weights)} methods")
    
    def find_matching_journals(self, query_text: str, 
                             top_k: int = 10,
                             min_confidence: float = 0.3,
                             filters: Optional[Dict[str, Any]] = None,
                             methods_to_use: Optional[List[MatchingMethod]] = None) -> List[EnsembleResult]:
        """
        Find matching journals using ensemble of methods.
        
        Args:
            query_text: Manuscript text to analyze
            top_k: Number of top results to return
            min_confidence: Minimum confidence threshold
            filters: Additional filters to apply
            methods_to_use: Specific methods to use (default: all)
            
        Returns:
            List of EnsembleResult objects sorted by ensemble score
        """
        if not query_text or not query_text.strip():
            return []
        
        # Use all methods if not specified
        if methods_to_use is None:
            methods_to_use = list(self.method_weights.keys())
        
        logger.info(f"Starting ensemble matching with {len(methods_to_use)} methods")
        
        # Run each matching method
        method_results = {}
        
        for method in methods_to_use:
            try:
                logger.debug(f"Running {method.value} matching")
                result = self._run_matching_method(method, query_text, filters)
                if result:
                    method_results[method] = result
                    logger.debug(f"{method.value}: {len(result.journal_scores)} journals scored")
            except Exception as e:
                logger.warning(f"Method {method.value} failed: {e}")
                continue
        
        if not method_results:
            logger.warning("No matching methods succeeded")
            return []
        
        # Combine results using ensemble approach
        ensemble_results = self._combine_method_results(method_results, top_k * 3)  # Get more for filtering
        
        # Apply confidence filtering
        ensemble_results = [r for r in ensemble_results if r.confidence >= min_confidence]
        
        # Apply additional filters if provided
        if filters:
            ensemble_results = self._apply_ensemble_filters(ensemble_results, filters)
        
        # Sort by ensemble score and limit results
        ensemble_results = sorted(ensemble_results, key=lambda x: x.ensemble_score, reverse=True)
        ensemble_results = ensemble_results[:top_k]
        
        # Update ranks
        for i, result in enumerate(ensemble_results, 1):
            result.rank = i
        
        logger.info(f"Ensemble matching completed: {len(ensemble_results)} results")
        return ensemble_results
    
    def _run_matching_method(self, method: MatchingMethod, 
                           query_text: str, 
                           filters: Optional[Dict[str, Any]] = None) -> Optional[MatchingResult]:
        """
        Run a specific matching method.
        
        Args:
            method: Matching method to run
            query_text: Query text
            filters: Optional filters
            
        Returns:
            MatchingResult or None if method fails
        """
        if method == MatchingMethod.SEMANTIC_SIMILARITY:
            return self._semantic_similarity_matching(query_text)
        
        elif method == MatchingMethod.MULTIMODAL_ANALYSIS:
            return self._multimodal_analysis_matching(query_text)
        
        elif method == MatchingMethod.KEYWORD_MATCHING:
            return self._keyword_matching(query_text)
        
        elif method == MatchingMethod.STUDY_TYPE_MATCHING:
            return self._study_type_matching(query_text)
        
        elif method == MatchingMethod.SUBJECT_MATCHING:
            return self._subject_matching(query_text)
        
        elif method == MatchingMethod.STRUCTURAL_MATCHING:
            return self._structural_matching(query_text)
        
        elif method == MatchingMethod.IMPACT_WEIGHTED:
            return self._impact_weighted_matching(query_text)
        
        elif method == MatchingMethod.PUBLISHER_PREFERENCE:
            return self._publisher_preference_matching(query_text)
        
        elif method == MatchingMethod.CITATION_NETWORK:
            return self._citation_network_matching(query_text)
        
        else:
            logger.warning(f"Unknown matching method: {method}")
            return None
    
    def _semantic_similarity_matching(self, query_text: str) -> MatchingResult:
        """Semantic similarity matching using embeddings."""
        # Generate query embedding
        query_embedding = embed_text(query_text)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        journal_scores = {}
        
        # Calculate similarities with all journals
        for i, journal in enumerate(self.journals):
            if i < len(self.journal_matcher.embeddings):
                journal_embedding = self.journal_matcher.embeddings[i]
                journal_embedding = journal_embedding / np.linalg.norm(journal_embedding)
                
                similarity = np.dot(query_embedding, journal_embedding)
                journal_scores[journal.get('id', str(i))] = float(similarity)
        
        return MatchingResult(
            method=MatchingMethod.SEMANTIC_SIMILARITY,
            journal_scores=journal_scores,
            confidence=0.85,
            metadata={'embedding_dimension': len(query_embedding)}
        )
    
    def _multimodal_analysis_matching(self, query_text: str) -> MatchingResult:
        """Multi-modal content analysis matching."""
        # Perform multi-modal analysis
        analysis = self.multimodal_analyzer.analyze_content(query_text)
        
        journal_scores = {}
        
        # Use combined embedding for similarity calculation
        query_embedding = analysis.combined_embedding
        
        for i, journal in enumerate(self.journals):
            if i < len(self.journal_matcher.embeddings):
                journal_embedding = self.journal_matcher.embeddings[i]
                journal_embedding = journal_embedding / np.linalg.norm(journal_embedding)
                
                similarity = np.dot(query_embedding, journal_embedding)
                
                # Boost score based on content quality
                quality_boost = analysis.content_quality_score * 0.1
                final_score = similarity + quality_boost
                
                journal_scores[journal.get('id', str(i))] = float(final_score)
        
        return MatchingResult(
            method=MatchingMethod.MULTIMODAL_ANALYSIS,
            journal_scores=journal_scores,
            confidence=analysis.content_quality_score,
            metadata={
                'sections_analyzed': len(analysis.sections),
                'content_quality': analysis.content_quality_score
            }
        )
    
    def _keyword_matching(self, query_text: str) -> MatchingResult:
        """Keyword-based matching using TF-IDF style scoring."""
        # Extract keywords from query
        query_keywords = set(extract_keywords(query_text, top_k=20))
        query_keywords.update(word.lower() for word in query_text.split() 
                            if len(word) > 3 and word.isalpha())
        
        journal_scores = {}
        
        for i, journal in enumerate(self.journals):
            score = 0.0
            
            # Check title keywords
            title = journal.get('display_name', '').lower()
            title_words = set(word for word in title.split() if len(word) > 3)
            title_matches = len(query_keywords.intersection(title_words))
            score += title_matches * 0.3
            
            # Check subject keywords
            subjects = journal.get('subjects', [])
            subject_text = ' '.join([
                s.get('name', '') if isinstance(s, dict) else str(s) 
                for s in subjects
            ]).lower()
            subject_words = set(word for word in subject_text.split() if len(word) > 3)
            subject_matches = len(query_keywords.intersection(subject_words))
            score += subject_matches * 0.2
            
            # Check publisher keywords
            publisher = journal.get('publisher', '').lower()
            publisher_words = set(word for word in publisher.split() if len(word) > 3)
            publisher_matches = len(query_keywords.intersection(publisher_words))
            score += publisher_matches * 0.1
            
            # Normalize score
            total_possible = len(query_keywords)
            normalized_score = score / total_possible if total_possible > 0 else 0
            
            journal_scores[journal.get('id', str(i))] = min(normalized_score, 1.0)
        
        return MatchingResult(
            method=MatchingMethod.KEYWORD_MATCHING,
            journal_scores=journal_scores,
            confidence=0.7,
            metadata={'query_keywords_count': len(query_keywords)}
        )
    
    def _study_type_matching(self, query_text: str) -> MatchingResult:
        """Study type based matching."""
        # Classify study type
        classification = self.study_classifier.classify_study_type(query_text)
        
        journal_scores = {}
        
        # Define study type preferences for different journal types
        study_type_preferences = {
            StudyType.RANDOMIZED_CONTROLLED_TRIAL: ['clinical', 'medical', 'trial', 'medicine'],
            StudyType.META_ANALYSIS: ['review', 'systematic', 'evidence', 'cochrane'],
            StudyType.COMPUTATIONAL: ['computer', 'algorithm', 'artificial', 'data'],
            StudyType.CASE_REPORT: ['case', 'clinical', 'medical', 'report'],
            StudyType.SURVEY: ['social', 'psychology', 'public', 'health'],
            StudyType.EXPERIMENTAL: ['experimental', 'laboratory', 'research']
        }
        
        preferred_terms = study_type_preferences.get(classification.primary_type, [])
        
        for i, journal in enumerate(self.journals):
            score = 0.0
            
            # Check if journal title/subjects match study type preferences
            journal_text = ' '.join([
                journal.get('display_name', ''),
                ' '.join([s.get('name', '') if isinstance(s, dict) else str(s) 
                         for s in journal.get('subjects', [])])
            ]).lower()
            
            for term in preferred_terms:
                if term in journal_text:
                    score += 0.2
            
            # Boost score based on classification confidence
            score *= classification.confidence
            
            journal_scores[journal.get('id', str(i))] = min(score, 1.0)
        
        return MatchingResult(
            method=MatchingMethod.STUDY_TYPE_MATCHING,
            journal_scores=journal_scores,
            confidence=classification.confidence,
            metadata={
                'detected_study_type': classification.primary_type.value,
                'classification_confidence': classification.confidence
            }
        )
    
    def _subject_matching(self, query_text: str) -> MatchingResult:
        """Subject area matching."""
        # Extract subject-related terms from query
        subject_terms = self._extract_subject_terms(query_text)
        
        journal_scores = {}
        
        for i, journal in enumerate(self.journals):
            score = 0.0
            
            # Get journal subjects
            journal_subjects = []
            for subject in journal.get('subjects', []):
                if isinstance(subject, dict):
                    journal_subjects.append(subject.get('name', '').lower())
                else:
                    journal_subjects.append(str(subject).lower())
            
            # Calculate subject overlap
            for query_term in subject_terms:
                for journal_subject in journal_subjects:
                    if query_term in journal_subject or journal_subject in query_term:
                        score += 0.3
                    elif any(word in journal_subject for word in query_term.split()):
                        score += 0.1
            
            # Normalize by number of query terms
            normalized_score = score / len(subject_terms) if subject_terms else 0
            journal_scores[journal.get('id', str(i))] = min(normalized_score, 1.0)
        
        return MatchingResult(
            method=MatchingMethod.SUBJECT_MATCHING,
            journal_scores=journal_scores,
            confidence=0.75,
            metadata={'subject_terms_found': len(subject_terms)}
        )
    
    def _structural_matching(self, query_text: str) -> MatchingResult:
        """Structural content matching based on manuscript structure."""
        # Analyze manuscript structure
        structure_indicators = {
            'has_abstract': bool(re.search(r'\babstract\b', query_text.lower())),
            'has_methods': bool(re.search(r'\b(methods?|methodology)\b', query_text.lower())),
            'has_results': bool(re.search(r'\bresults?\b', query_text.lower())),
            'has_discussion': bool(re.search(r'\bdiscussion\b', query_text.lower())),
            'has_conclusion': bool(re.search(r'\bconclusions?\b', query_text.lower())),
            'has_references': bool(re.search(r'\breferences?\b', query_text.lower())),
            'has_figures': bool(re.search(r'\b(figure|fig\.)\s*\d+\b', query_text.lower())),
            'has_tables': bool(re.search(r'\btable\s*\d+\b', query_text.lower()))
        }
        
        # Calculate structure completeness score
        structure_score = sum(structure_indicators.values()) / len(structure_indicators)
        
        journal_scores = {}
        
        # Prefer journals that typically publish well-structured papers
        for i, journal in enumerate(self.journals):
            base_score = 0.5  # Base score for all journals
            
            # Boost score for journals that prefer structured content
            journal_name = journal.get('display_name', '').lower()
            publisher = journal.get('publisher', '').lower()
            
            # Academic/research journals typically prefer structured content
            if any(term in journal_name or term in publisher for term in 
                   ['research', 'science', 'academic', 'journal', 'review']):
                base_score += 0.3
            
            # Apply structure completeness
            final_score = base_score * structure_score
            
            journal_scores[journal.get('id', str(i))] = min(final_score, 1.0)
        
        return MatchingResult(
            method=MatchingMethod.STRUCTURAL_MATCHING,
            journal_scores=journal_scores,
            confidence=structure_score,
            metadata=structure_indicators
        )
    
    def _impact_weighted_matching(self, query_text: str) -> MatchingResult:
        """Impact factor weighted matching."""
        journal_scores = {}
        
        for i, journal in enumerate(self.journals):
            # Get impact indicators
            h_index = journal.get('h_index', 0)
            citations = journal.get('cited_by_count', 0)
            works_count = journal.get('works_count', 0)
            
            # Calculate impact score
            impact_score = 0.2  # Base score
            
            if h_index > 0:
                impact_score += min(h_index / 100, 0.3)  # H-index contribution
            
            if citations > 0:
                impact_score += min(np.log10(citations) / 10, 0.3)  # Citation contribution
            
            if works_count > 0:
                impact_score += min(np.log10(works_count) / 8, 0.2)  # Productivity contribution
            
            journal_scores[journal.get('id', str(i))] = min(impact_score, 1.0)
        
        return MatchingResult(
            method=MatchingMethod.IMPACT_WEIGHTED,
            journal_scores=journal_scores,
            confidence=0.6,
            metadata={'scoring_method': 'h_index_citations_works'}
        )
    
    def _publisher_preference_matching(self, query_text: str) -> MatchingResult:
        """Publisher preference matching based on content type."""
        # Analyze content to determine preferred publisher types
        query_lower = query_text.lower()
        
        publisher_preferences = {
            'elsevier': 0.5,
            'springer': 0.5,
            'wiley': 0.5,
            'nature': 0.3,
            'plos': 0.7 if 'open access' in query_lower else 0.4,
            'ieee': 0.8 if any(term in query_lower for term in ['computer', 'engineering', 'technology']) else 0.3,
            'acm': 0.8 if any(term in query_lower for term in ['computer', 'computing', 'algorithm']) else 0.3
        }
        
        journal_scores = {}
        
        for i, journal in enumerate(self.journals):
            publisher = journal.get('publisher', '').lower()
            score = 0.4  # Base score
            
            # Apply publisher-specific preferences
            for pub_name, preference in publisher_preferences.items():
                if pub_name in publisher:
                    score = preference
                    break
            
            journal_scores[journal.get('id', str(i))] = score
        
        return MatchingResult(
            method=MatchingMethod.PUBLISHER_PREFERENCE,
            journal_scores=journal_scores,
            confidence=0.5,
            metadata={'preferences_applied': len(publisher_preferences)}
        )
    
    def _citation_network_matching(self, query_text: str) -> MatchingResult:
        """Citation network analysis matching."""
        try:
            # Perform citation analysis
            citation_analysis = self.citation_analyzer.analyze_citations(query_text)
            
            journal_scores = {}
            
            # Score journals based on citation network analysis
            for i, journal in enumerate(self.journals):
                journal_name = journal.get('display_name', '')
                score = 0.0
                
                # Direct citation matching
                if citation_analysis.cited_journals:
                    for cited_journal, count in citation_analysis.cited_journals.items():
                        if self._journals_similar(journal_name, cited_journal):
                            score += (count / sum(citation_analysis.cited_journals.values())) * 0.4
                
                # Network centrality boost
                compatibility_score = self.citation_analyzer.get_journal_compatibility_score(
                    citation_analysis, journal_name
                )
                score += compatibility_score * 0.3
                
                # Subject area alignment
                journal_subjects = self._get_journal_subjects(journal)
                if journal_subjects and citation_analysis.subject_areas:
                    subject_overlap = 0
                    for journal_subject in journal_subjects:
                        for citation_subject, relevance in citation_analysis.subject_areas.items():
                            if citation_subject in journal_subject.lower():
                                subject_overlap += relevance
                    
                    if subject_overlap > 0:
                        score += min(subject_overlap, 1.0) * 0.2
                
                # Research ecosystem alignment
                ecosystem_factor = citation_analysis.research_ecosystem_score
                score += ecosystem_factor * 0.1
                
                journal_scores[journal.get('id', str(i))] = min(score, 1.0)
            
            # Calculate confidence based on citation analysis quality
            confidence = min(
                citation_analysis.research_ecosystem_score + 
                len(citation_analysis.cited_journals) / 20 +  # Normalize by 20 journals
                citation_analysis.impact_metrics.get('reference_quality', 0.0),
                1.0
            )
            
            return MatchingResult(
                method=MatchingMethod.CITATION_NETWORK,
                journal_scores=journal_scores,
                confidence=confidence,
                metadata={
                    'total_references': citation_analysis.metadata.get('total_references', 0),
                    'cited_journals': len(citation_analysis.cited_journals),
                    'ecosystem_score': citation_analysis.research_ecosystem_score,
                    'prestigious_citations': citation_analysis.metadata.get('high_impact_citations', 0)
                }
            )
            
        except Exception as e:
            logger.warning(f"Citation network analysis failed: {e}")
            # Return minimal scores for all journals
            return MatchingResult(
                method=MatchingMethod.CITATION_NETWORK,
                journal_scores={journal.get('id', str(i)): 0.1 for i, journal in enumerate(self.journals)},
                confidence=0.1,
                metadata={'error': str(e)}
            )
    
    def _journals_similar(self, journal1: str, journal2: str) -> bool:
        """Check if two journal names are similar."""
        # Normalize names
        j1_norm = journal1.lower().strip()
        j2_norm = journal2.lower().strip()
        
        # Direct match
        if j1_norm == j2_norm:
            return True
        
        # Check if one contains the other (for abbreviations)
        if j1_norm in j2_norm or j2_norm in j1_norm:
            return True
        
        # Check common words
        j1_words = set(j1_norm.split())
        j2_words = set(j2_norm.split())
        
        # Remove common stop words
        stop_words = {'of', 'the', 'and', 'for', 'in', 'on', 'journal', 'review', 'international'}
        j1_words -= stop_words
        j2_words -= stop_words
        
        if j1_words and j2_words:
            overlap = len(j1_words & j2_words)
            union = len(j1_words | j2_words)
            similarity = overlap / union if union > 0 else 0
            return similarity > 0.3
        
        return False
    
    def _get_journal_subjects(self, journal: Dict[str, Any]) -> List[str]:
        """Extract subject areas from journal metadata."""
        subjects = []
        
        # From subjects field
        for subject in journal.get('subjects', []):
            if isinstance(subject, dict):
                subjects.append(subject.get('name', '').lower())
            elif isinstance(subject, str):
                subjects.append(subject.lower())
        
        # From journal name
        journal_name = journal.get('display_name', '').lower()
        subjects.append(journal_name)
        
        return [s for s in subjects if s]
    
    def _extract_subject_terms(self, text: str) -> List[str]:
        """Extract subject area terms from text."""
        # Common academic subject terms
        subject_patterns = [
            r'\b(biology|biological|life sciences?)\b',
            r'\b(medicine|medical|clinical|healthcare)\b',
            r'\b(computer science|computing|informatics)\b',
            r'\b(physics|physical sciences?)\b',
            r'\b(chemistry|chemical)\b',
            r'\b(engineering|technology)\b',
            r'\b(mathematics|mathematical|statistics)\b',
            r'\b(psychology|psychological|behavioral)\b',
            r'\b(sociology|social sciences?)\b',
            r'\b(economics|economic)\b',
            r'\b(neuroscience|neurology|brain)\b',
            r'\b(genetics|genomics|molecular)\b'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for pattern in subject_patterns:
            matches = re.findall(pattern, text_lower)
            found_terms.extend(matches)
        
        return list(set(found_terms))
    
    def _combine_method_results(self, method_results: Dict[MatchingMethod, MatchingResult], 
                              max_results: int) -> List[EnsembleResult]:
        """
        Combine results from multiple matching methods using weighted voting.
        
        Args:
            method_results: Results from each matching method
            max_results: Maximum number of results to return
            
        Returns:
            List of EnsembleResult objects
        """
        # Collect all unique journal IDs
        all_journal_ids = set()
        for result in method_results.values():
            all_journal_ids.update(result.journal_scores.keys())
        
        # Create ensemble results
        ensemble_results = []
        
        for journal_id in all_journal_ids:
            # Calculate weighted ensemble score
            ensemble_score = 0.0
            individual_scores = {}
            total_weight = 0.0
            
            for method, result in method_results.items():
                if journal_id in result.journal_scores:
                    score = result.journal_scores[journal_id]
                    weight = self.method_weights.get(method, 0.1)
                    confidence_adjusted_weight = weight * result.confidence
                    
                    ensemble_score += score * confidence_adjusted_weight
                    individual_scores[method] = score
                    total_weight += confidence_adjusted_weight
            
            # Normalize ensemble score
            if total_weight > 0:
                ensemble_score /= total_weight
            
            # Calculate confidence based on method agreement
            confidence = self._calculate_ensemble_confidence(individual_scores, method_results)
            
            # Find journal data
            journal_data = None
            journal_index = None
            try:
                journal_index = int(journal_id) if journal_id.isdigit() else None
                if journal_index is not None and journal_index < len(self.journals):
                    journal_data = self.journals[journal_index]
            except (ValueError, IndexError):
                # Handle string IDs or missing journals
                for journal in self.journals:
                    if journal.get('id') == journal_id:
                        journal_data = journal
                        break
            
            if journal_data:
                # Generate explanation
                explanation = self._generate_explanation(individual_scores, method_results)
                
                ensemble_results.append(EnsembleResult(
                    journal_id=journal_id,
                    journal_data=journal_data,
                    ensemble_score=ensemble_score,
                    individual_scores=individual_scores,
                    confidence=confidence,
                    rank=0,  # Will be set later
                    explanation=explanation
                ))
        
        # Sort by ensemble score and limit
        ensemble_results.sort(key=lambda x: x.ensemble_score, reverse=True)
        return ensemble_results[:max_results]
    
    def _calculate_ensemble_confidence(self, individual_scores: Dict[MatchingMethod, float],
                                     method_results: Dict[MatchingMethod, MatchingResult]) -> float:
        """Calculate confidence based on method agreement and individual confidences."""
        if not individual_scores:
            return 0.0
        
        # Base confidence from method results
        method_confidences = [
            method_results[method].confidence 
            for method in individual_scores.keys()
        ]
        base_confidence = np.mean(method_confidences)
        
        # Agreement factor (how much methods agree)
        scores = list(individual_scores.values())
        if len(scores) > 1:
            agreement = 1.0 - np.std(scores)  # Lower std = higher agreement
            agreement = max(0.0, agreement)
        else:
            agreement = 0.5  # Neutral for single method
        
        # Coverage factor (how many methods contributed)
        coverage = len(individual_scores) / len(self.method_weights)
        
        # Combine factors
        final_confidence = base_confidence * 0.5 + agreement * 0.3 + coverage * 0.2
        
        return min(final_confidence, 1.0)
    
    def _generate_explanation(self, individual_scores: Dict[MatchingMethod, float],
                            method_results: Dict[MatchingMethod, MatchingResult]) -> str:
        """Generate human-readable explanation for the match."""
        explanations = []
        
        # Sort methods by their contribution
        sorted_methods = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)
        
        for method, score in sorted_methods[:3]:  # Top 3 contributing methods
            if score > 0.1:  # Only include meaningful contributions
                method_name = method.value.replace('_', ' ').title()
                if score > 0.7:
                    explanations.append(f"Strong {method_name.lower()} match")
                elif score > 0.4:
                    explanations.append(f"Good {method_name.lower()} match")
                else:
                    explanations.append(f"Moderate {method_name.lower()} match")
        
        if not explanations:
            return "General compatibility"
        
        return "; ".join(explanations)
    
    def _apply_ensemble_filters(self, results: List[EnsembleResult], 
                              filters: Dict[str, Any]) -> List[EnsembleResult]:
        """Apply filters to ensemble results."""
        filtered_results = []
        
        for result in results:
            journal = result.journal_data
            
            # Apply standard filters (similar to base matcher)
            if 'open_access_only' in filters and filters['open_access_only']:
                if not journal.get('oa_status', journal.get('is_oa', False)):
                    continue
            
            if 'min_h_index' in filters and filters['min_h_index'] is not None:
                if journal.get('h_index', 0) < filters['min_h_index']:
                    continue
            
            # Add ensemble-specific filters
            if 'min_ensemble_score' in filters:
                if result.ensemble_score < filters['min_ensemble_score']:
                    continue
            
            if 'required_methods' in filters:
                required = set(filters['required_methods'])
                if not required.issubset(set(result.individual_scores.keys())):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def get_method_weights(self) -> Dict[MatchingMethod, float]:
        """Get current method weights."""
        return self.method_weights.copy()
    
    def set_method_weights(self, weights: Dict[MatchingMethod, float]) -> None:
        """Set custom method weights."""
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.method_weights = {method: weight/total_weight 
                                 for method, weight in weights.items()}
        logger.info(f"Updated method weights: {self.method_weights}")


def create_ensemble_matcher(custom_weights: Optional[Dict[MatchingMethod, float]] = None) -> EnsembleJournalMatcher:
    """
    Convenience function to create an ensemble matcher.
    
    Args:
        custom_weights: Optional custom method weights
        
    Returns:
        EnsembleJournalMatcher instance
    """
    return EnsembleJournalMatcher(method_weights=custom_weights)