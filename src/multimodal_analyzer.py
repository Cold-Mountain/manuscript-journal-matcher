"""
Multi-modal content analysis module for Manuscript Journal Matcher.

This module implements sophisticated content analysis by processing different
sections of manuscripts (title, abstract, methodology, conclusions) separately
and combining them with weighted scores for enhanced journal matching.
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path

try:
    from .config import DATA_DIR
    from .embedder import embed_text, cosine_similarity_single
    from .utils import clean_text, extract_keywords
    from .study_classifier import StudyTypeClassifier, StudyType
except ImportError:
    from config import DATA_DIR
    from embedder import embed_text, cosine_similarity_single
    from utils import clean_text, extract_keywords
    from study_classifier import StudyTypeClassifier, StudyType

# Set up logging
logger = logging.getLogger(__name__)


class ContentSection(Enum):
    """Enumeration of manuscript content sections."""
    TITLE = "title"
    ABSTRACT = "abstract" 
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    KEYWORDS = "keywords"
    REFERENCES = "references"


@dataclass
class SectionAnalysis:
    """Analysis result for a specific content section."""
    section: ContentSection
    content: str
    embedding: np.ndarray
    keywords: List[str]
    importance_score: float
    quality_score: float
    metadata: Dict[str, Any]


@dataclass
class MultiModalAnalysis:
    """Complete multi-modal analysis result."""
    sections: Dict[ContentSection, SectionAnalysis]
    combined_embedding: np.ndarray
    content_quality_score: float
    study_classification: Optional[Any]  # StudyClassification
    content_fingerprint: str
    metadata: Dict[str, Any]


class MultiModalContentAnalyzer:
    """
    Advanced content analyzer that processes manuscripts using multi-modal analysis.
    
    Analyzes different sections of manuscripts separately, then combines them
    with intelligent weighting for enhanced journal matching accuracy.
    """
    
    def __init__(self, weights_config: Optional[Dict[str, float]] = None):
        """
        Initialize the multi-modal content analyzer.
        
        Args:
            weights_config: Custom weights for different content sections
        """
        # Default section weights (can be customized)
        self.section_weights = weights_config or {
            ContentSection.TITLE: 0.25,
            ContentSection.ABSTRACT: 0.30,
            ContentSection.METHODOLOGY: 0.20,
            ContentSection.RESULTS: 0.10,
            ContentSection.CONCLUSION: 0.10,
            ContentSection.KEYWORDS: 0.05
        }
        
        # Initialize study classifier for methodology analysis
        self.study_classifier = StudyTypeClassifier()
        
        # Section detection patterns
        self._define_section_patterns()
        
        logger.info("MultiModalContentAnalyzer initialized")
    
    def _define_section_patterns(self) -> None:
        """Define regex patterns for detecting manuscript sections."""
        
        self.section_patterns = {
            ContentSection.TITLE: [
                r'^(.{10,200}?)(?:\n|$)',  # First line, 10-200 chars
                r'title[:\s]*(.+?)(?:\n|$)',
                r'^([A-Z][^.!?]*[.!?]?)(?:\n|Abstract)',
            ],
            
            ContentSection.ABSTRACT: [
                r'abstract[:\s]*(.+?)(?=\n\n|\nkey|\nintro|\n1\.|\nmethod)',
                r'summary[:\s]*(.+?)(?=\n\n|\nkey|\nintro)',
                r'(?:^|\n)(.{100,2000}?)(?=\n\n|\nkey|\nintro|\nmethod)',
            ],
            
            ContentSection.INTRODUCTION: [
                r'introduction[:\s]*(.+?)(?=\nmethod|\nmaterial|\nresult)',
                r'(?:^|\n)1\.?\s*introduction[:\s]*(.+?)(?=\n2\.|\nmethod)',
                r'background[:\s]*(.+?)(?=\nmethod|\nmaterial)',
            ],
            
            ContentSection.METHODOLOGY: [
                r'method(?:ology)?[:\s]*(.+?)(?=\nresult|\ndiscuss|\nconclus)',
                r'materials?\s+and\s+methods?[:\s]*(.+?)(?=\nresult)',
                r'experimental\s+(?:design|procedure)[:\s]*(.+?)(?=\nresult)',
                r'(?:^|\n)(?:2\.|ii\.)\s*method[:\s]*(.+?)(?=\n3\.|\nresult)',
            ],
            
            ContentSection.RESULTS: [
                r'results?[:\s]*(.+?)(?=\ndiscuss|\nconclus|\nref)',
                r'findings?[:\s]*(.+?)(?=\ndiscuss|\nconclus)',
                r'(?:^|\n)(?:3\.|iii\.)\s*results?[:\s]*(.+?)(?=\n4\.|\ndiscuss)',
            ],
            
            ContentSection.DISCUSSION: [
                r'discussion[:\s]*(.+?)(?=\nconclus|\nref|\nacknow)',
                r'(?:^|\n)(?:4\.|iv\.)\s*discussion[:\s]*(.+?)(?=\n5\.|\nconclus)',
            ],
            
            ContentSection.CONCLUSION: [
                r'conclusions?[:\s]*(.+?)(?=\nref|\nacknow|\nappend)',
                r'(?:^|\n)(?:5\.|v\.)\s*conclusions?[:\s]*(.+?)(?=\nref)',
                r'summary\s+and\s+conclusions?[:\s]*(.+?)(?=\nref)',
            ],
            
            ContentSection.KEYWORDS: [
                r'keywords?[:\s]*(.+?)(?=\n\n|\nintro|\nabstract)',
                r'key\s+words?[:\s]*(.+?)(?=\n\n|\nintro)',
                r'terms?[:\s]*(.+?)(?=\n\n|\nintro)',
            ]
        }
    
    def analyze_content(self, text: str, 
                       include_study_classification: bool = True) -> MultiModalAnalysis:
        """
        Perform comprehensive multi-modal analysis of manuscript content.
        
        Args:
            text: Full manuscript text
            include_study_classification: Whether to include study type analysis
            
        Returns:
            MultiModalAnalysis with detailed section analysis
        """
        if not text or not text.strip():
            return self._create_empty_analysis()
        
        logger.info("Starting multi-modal content analysis")
        
        # Clean and preprocess text
        clean_text_content = clean_text(text, remove_extra_whitespace=True)
        
        # Extract sections from text
        sections_content = self._extract_sections(clean_text_content)
        
        # Analyze each section
        section_analyses = {}
        
        for section, content in sections_content.items():
            if content and content.strip():
                analysis = self._analyze_section(section, content)
                section_analyses[section] = analysis
                logger.debug(f"Analyzed {section.value}: {len(content)} chars")
        
        # Perform study type classification if requested
        study_classification = None
        if include_study_classification:
            study_classification = self.study_classifier.classify_study_type(
                clean_text_content
            )
            logger.info(f"Study classification: {study_classification.primary_type.value}")
        
        # Generate combined embedding
        combined_embedding = self._create_combined_embedding(section_analyses)
        
        # Calculate content quality score
        quality_score = self._calculate_content_quality(section_analyses)
        
        # Generate content fingerprint
        fingerprint = self._generate_content_fingerprint(section_analyses)
        
        # Create metadata
        metadata = {
            'total_sections': len(section_analyses),
            'section_lengths': {
                section.value: len(analysis.content) 
                for section, analysis in section_analyses.items()
            },
            'analysis_timestamp': np.datetime64('now').astype(str),
            'quality_breakdown': {
                section.value: analysis.quality_score
                for section, analysis in section_analyses.items()
            }
        }
        
        return MultiModalAnalysis(
            sections=section_analyses,
            combined_embedding=combined_embedding,
            content_quality_score=quality_score,
            study_classification=study_classification,
            content_fingerprint=fingerprint,
            metadata=metadata
        )
    
    def _extract_sections(self, text: str) -> Dict[ContentSection, str]:
        """
        Extract different sections from manuscript text.
        
        Args:
            text: Full manuscript text
            
        Returns:
            Dictionary mapping sections to their content
        """
        sections = {}
        text_lower = text.lower()
        
        for section, patterns in self.section_patterns.items():
            section_content = ""
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.DOTALL | re.IGNORECASE)
                if matches:
                    # Take the longest match (likely most complete)
                    section_content = max(matches, key=len).strip()
                    break
            
            if section_content:
                # Clean up the extracted content
                section_content = re.sub(r'\s+', ' ', section_content)
                section_content = section_content[:2000]  # Limit length
                sections[section] = section_content
        
        # Fallback: if no abstract found, use first paragraph
        if ContentSection.ABSTRACT not in sections:
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if 50 < len(para) < 1000:  # Reasonable abstract length
                    sections[ContentSection.ABSTRACT] = para.strip()
                    break
        
        # Fallback: if no title found, use first line
        if ContentSection.TITLE not in sections:
            first_line = text.split('\n')[0].strip()
            if 10 < len(first_line) < 200:
                sections[ContentSection.TITLE] = first_line
        
        return sections
    
    def _analyze_section(self, section: ContentSection, content: str) -> SectionAnalysis:
        """
        Analyze a specific content section.
        
        Args:
            section: Section type
            content: Section content
            
        Returns:
            SectionAnalysis with detailed analysis
        """
        # Generate embedding for this section
        embedding = embed_text(content)
        
        # Extract keywords specific to this section
        keywords = self._extract_section_keywords(section, content)
        
        # Calculate importance score based on section type and content
        importance_score = self._calculate_importance_score(section, content)
        
        # Calculate quality score for this section
        quality_score = self._calculate_section_quality(section, content)
        
        # Generate section-specific metadata
        metadata = {
            'word_count': len(content.split()),
            'char_count': len(content),
            'sentence_count': len(re.findall(r'[.!?]+', content)),
            'contains_numbers': bool(re.search(r'\d+', content)),
            'contains_citations': bool(re.search(r'\[\d+\]|\(\d{4}\)', content)),
            'technical_terms': self._count_technical_terms(content)
        }
        
        return SectionAnalysis(
            section=section,
            content=content,
            embedding=embedding,
            keywords=keywords,
            importance_score=importance_score,
            quality_score=quality_score,
            metadata=metadata
        )
    
    def _extract_section_keywords(self, section: ContentSection, content: str) -> List[str]:
        """
        Extract keywords specific to a content section.
        
        Args:
            section: Section type
            content: Section content
            
        Returns:
            List of relevant keywords
        """
        # Base keyword extraction
        base_keywords = extract_keywords(content, top_k=15)
        
        # Section-specific keyword filtering and enhancement
        if section == ContentSection.METHODOLOGY:
            # Enhance with methodology-specific terms
            method_terms = [
                'randomized', 'controlled', 'prospective', 'retrospective',
                'cohort', 'case-control', 'cross-sectional', 'survey',
                'experimental', 'observational', 'qualitative', 'quantitative'
            ]
            content_lower = content.lower()
            method_keywords = [term for term in method_terms if term in content_lower]
            base_keywords.extend(method_keywords)
        
        elif section == ContentSection.RESULTS:
            # Focus on statistical and outcome terms
            content_lower = content.lower()
            stat_terms = ['significant', 'correlation', 'regression', 'p-value', 
                         'confidence', 'mean', 'median', 'standard deviation']
            stat_keywords = [term for term in stat_terms if term in content_lower]
            base_keywords.extend(stat_keywords)
        
        elif section == ContentSection.TITLE:
            # Title keywords are especially important
            title_keywords = extract_keywords(content, top_k=10)
            # Boost title keyword importance by duplicating
            base_keywords.extend(title_keywords)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in base_keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        return unique_keywords[:12]  # Limit to top 12
    
    def _calculate_importance_score(self, section: ContentSection, content: str) -> float:
        """
        Calculate importance score for a section based on type and content quality.
        
        Args:
            section: Section type
            content: Section content
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        # Base importance from section weights
        base_score = self.section_weights.get(section, 0.1)
        
        # Content quality factors
        word_count = len(content.split())
        char_count = len(content)
        
        # Length quality factor (not too short, not too long)
        if section == ContentSection.TITLE:
            optimal_length = 100  # characters
            length_factor = 1.0 - abs(char_count - optimal_length) / optimal_length
        elif section == ContentSection.ABSTRACT:
            optimal_length = 800  # characters
            length_factor = 1.0 - abs(char_count - optimal_length) / optimal_length
        else:
            # For other sections, longer is generally better up to a point
            length_factor = min(word_count / 200, 1.0)
        
        length_factor = max(0.2, length_factor)  # Minimum 0.2
        
        # Information density (presence of technical terms, numbers, etc.)
        density_score = 0.5  # base
        if re.search(r'\d+', content):  # Contains numbers
            density_score += 0.1
        if re.search(r'\[\d+\]|\(\d{4}\)', content):  # Contains citations
            density_score += 0.1
        if self._count_technical_terms(content) > 3:  # Technical content
            density_score += 0.2
        
        # Combine factors
        final_score = base_score * length_factor * density_score
        
        return min(final_score, 1.0)
    
    def _calculate_section_quality(self, section: ContentSection, content: str) -> float:
        """
        Calculate quality score for a specific section.
        
        Args:
            section: Section type
            content: Section content
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        quality_score = 0.5  # Base score
        
        # Length appropriateness
        word_count = len(content.split())
        if section == ContentSection.TITLE:
            if 5 <= word_count <= 20:
                quality_score += 0.2
        elif section == ContentSection.ABSTRACT:
            if 50 <= word_count <= 300:
                quality_score += 0.2
        else:
            if word_count >= 20:
                quality_score += 0.1
        
        # Structural indicators
        if section == ContentSection.METHODOLOGY:
            # Look for methodology indicators
            method_indicators = ['participants', 'subjects', 'data', 'analysis', 
                               'procedure', 'design', 'statistical']
            indicator_count = sum(1 for ind in method_indicators 
                                if ind in content.lower())
            quality_score += min(indicator_count * 0.05, 0.2)
        
        # Grammar and completeness (basic checks)
        if content.endswith('.'):  # Proper sentence ending
            quality_score += 0.05
        if content[0].isupper():  # Proper capitalization
            quality_score += 0.05
        
        # Technical content indicator
        if self._count_technical_terms(content) > 0:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _count_technical_terms(self, content: str) -> int:
        """Count technical/scientific terms in content."""
        technical_patterns = [
            r'\b\w+(?:tion|sion|ment|ness|ity|ism|ics)\b',  # Technical suffixes
            r'\b(?:analysis|method|approach|technique|algorithm)\b',
            r'\b(?:significant|correlation|regression|hypothesis)\b',
            r'\b\w*(?:bio|neuro|cardio|patho|physio)\w*\b',  # Medical prefixes
            r'\b\w+(?:gene|protein|cell|tissue|organ)\w*\b'   # Biological terms
        ]
        
        count = 0
        content_lower = content.lower()
        for pattern in technical_patterns:
            matches = re.findall(pattern, content_lower)
            count += len(matches)
        
        return count
    
    def _create_combined_embedding(self, section_analyses: Dict[ContentSection, SectionAnalysis]) -> np.ndarray:
        """
        Create combined embedding from all section embeddings with weighted averaging.
        
        Args:
            section_analyses: Dictionary of section analyses
            
        Returns:
            Combined embedding vector
        """
        if not section_analyses:
            # Return zero embedding if no sections
            return np.zeros(384)  # Standard embedding dimension
        
        # Collect embeddings and weights
        embeddings = []
        weights = []
        
        for section, analysis in section_analyses.items():
            embeddings.append(analysis.embedding)
            # Weight by both section importance and content quality
            weight = analysis.importance_score * analysis.quality_score
            weights.append(weight)
        
        # Stack embeddings and normalize weights
        embeddings_array = np.stack(embeddings)
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()  # Normalize
        
        # Weighted average
        combined = np.average(embeddings_array, axis=0, weights=weights_array)
        
        # Normalize the combined embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def _calculate_content_quality(self, section_analyses: Dict[ContentSection, SectionAnalysis]) -> float:
        """
        Calculate overall content quality score.
        
        Args:
            section_analyses: Dictionary of section analyses
            
        Returns:
            Overall quality score (0.0 to 1.0)
        """
        if not section_analyses:
            return 0.0
        
        # Weighted average of section quality scores
        total_weighted_quality = 0.0
        total_weight = 0.0
        
        for section, analysis in section_analyses.items():
            weight = self.section_weights.get(section, 0.1)
            total_weighted_quality += analysis.quality_score * weight
            total_weight += weight
        
        base_quality = total_weighted_quality / total_weight if total_weight > 0 else 0.0
        
        # Bonus for having multiple important sections
        important_sections = [ContentSection.TITLE, ContentSection.ABSTRACT, 
                            ContentSection.METHODOLOGY]
        sections_present = sum(1 for sec in important_sections if sec in section_analyses)
        completeness_bonus = sections_present / len(important_sections) * 0.2
        
        return min(base_quality + completeness_bonus, 1.0)
    
    def _generate_content_fingerprint(self, section_analyses: Dict[ContentSection, SectionAnalysis]) -> str:
        """
        Generate a unique fingerprint for the content based on key characteristics.
        
        Args:
            section_analyses: Dictionary of section analyses
            
        Returns:
            Content fingerprint string
        """
        fingerprint_components = []
        
        # Add section presence pattern
        sections_present = sorted([sec.value for sec in section_analyses.keys()])
        fingerprint_components.append('|'.join(sections_present))
        
        # Add key terms from title and abstract
        key_terms = []
        for section in [ContentSection.TITLE, ContentSection.ABSTRACT]:
            if section in section_analyses:
                key_terms.extend(section_analyses[section].keywords[:3])
        
        if key_terms:
            fingerprint_components.append('~'.join(sorted(key_terms)[:6]))
        
        # Add content length pattern
        lengths = [len(analysis.content) for analysis in section_analyses.values()]
        length_pattern = ''.join([str(min(l//100, 9)) for l in lengths])
        fingerprint_components.append(length_pattern)
        
        return '#'.join(fingerprint_components)
    
    def _create_empty_analysis(self) -> MultiModalAnalysis:
        """Create empty analysis for invalid input."""
        return MultiModalAnalysis(
            sections={},
            combined_embedding=np.zeros(384),
            content_quality_score=0.0,
            study_classification=None,
            content_fingerprint="empty",
            metadata={'error': 'No valid content provided'}
        )
    
    def compare_analyses(self, analysis1: MultiModalAnalysis, 
                        analysis2: MultiModalAnalysis) -> Dict[str, float]:
        """
        Compare two multi-modal analyses for similarity.
        
        Args:
            analysis1: First analysis
            analysis2: Second analysis
            
        Returns:
            Dictionary of similarity scores
        """
        similarities = {}
        
        # Overall embedding similarity
        similarities['overall'] = cosine_similarity_single(
            analysis1.combined_embedding, 
            analysis2.combined_embedding
        )
        
        # Section-wise similarities
        common_sections = set(analysis1.sections.keys()) & set(analysis2.sections.keys())
        
        for section in common_sections:
            emb1 = analysis1.sections[section].embedding
            emb2 = analysis2.sections[section].embedding
            similarities[f'section_{section.value}'] = cosine_similarity_single(emb1, emb2)
        
        # Content quality similarity
        quality_diff = abs(analysis1.content_quality_score - analysis2.content_quality_score)
        similarities['quality'] = 1.0 - quality_diff
        
        return similarities
    
    def get_section_importance_ranking(self, analysis: MultiModalAnalysis) -> List[Tuple[ContentSection, float]]:
        """
        Get sections ranked by their importance scores.
        
        Args:
            analysis: Multi-modal analysis
            
        Returns:
            List of (section, importance_score) tuples, sorted by importance
        """
        section_scores = [
            (section, section_analysis.importance_score)
            for section, section_analysis in analysis.sections.items()
        ]
        
        return sorted(section_scores, key=lambda x: x[1], reverse=True)


def analyze_manuscript_content(text: str, 
                             custom_weights: Optional[Dict[ContentSection, float]] = None) -> MultiModalAnalysis:
    """
    Convenience function to analyze manuscript content using multi-modal analysis.
    
    Args:
        text: Full manuscript text
        custom_weights: Optional custom section weights
        
    Returns:
        MultiModalAnalysis result
    """
    analyzer = MultiModalContentAnalyzer(custom_weights)
    return analyzer.analyze_content(text)


def extract_content_sections(text: str) -> Dict[ContentSection, str]:
    """
    Convenience function to extract content sections from text.
    
    Args:
        text: Full manuscript text
        
    Returns:
        Dictionary mapping sections to content
    """
    analyzer = MultiModalContentAnalyzer()
    return analyzer._extract_sections(text)