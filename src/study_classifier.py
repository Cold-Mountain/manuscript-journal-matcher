"""
Study type classification module for Manuscript Journal Matcher.

This module implements automatic classification of research study types
to improve journal matching by understanding manuscript methodology
and matching it with journal preferences.
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path

try:
    from .config import DATA_DIR
    from .embedder import embed_text, get_model
    from .utils import clean_text, extract_keywords
except ImportError:
    from config import DATA_DIR
    from embedder import embed_text, get_model
    from utils import clean_text, extract_keywords

# Set up logging
logger = logging.getLogger(__name__)


class StudyType(Enum):
    """Enumeration of research study types."""
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    OBSERVATIONAL_COHORT = "observational_cohort"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    CASE_REPORT = "case_report"
    EXPERIMENTAL = "experimental"
    COMPUTATIONAL = "computational"
    THEORETICAL = "theoretical"
    SURVEY = "survey"
    QUALITATIVE = "qualitative"
    UNKNOWN = "unknown"


@dataclass
class StudyClassification:
    """Result of study type classification."""
    primary_type: StudyType
    confidence: float
    secondary_types: List[Tuple[StudyType, float]]
    evidence: Dict[str, Any]
    methodology_keywords: List[str]


class StudyTypeClassifier:
    """
    Classifier for identifying research study types from manuscript text.
    
    Uses a combination of keyword matching, pattern recognition, and 
    semantic analysis to determine the methodology of research papers.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the study type classifier.
        
        Args:
            model_path: Optional path to saved classification model
        """
        self.model_path = model_path or (DATA_DIR / "study_classifier_model.pkl")
        
        # Define study type patterns and keywords
        self._define_classification_patterns()
        
        # Load or initialize classification model
        self.classification_model = None
        self._load_or_initialize_model()
        
        logger.info("StudyTypeClassifier initialized")
    
    def _define_classification_patterns(self) -> None:
        """Define patterns and keywords for each study type."""
        
        self.study_patterns = {
            StudyType.RANDOMIZED_CONTROLLED_TRIAL: {
                "keywords": [
                    "randomized controlled trial", "randomized control trial", "rct", 
                    "randomization", "randomisation", "placebo controlled", 
                    "double blind", "single blind", "control group", "treatment group",
                    "intervention group", "allocated", "random allocation"
                ],
                "patterns": [
                    r"randomized?\s+controlled?\s+trial",
                    r"double\s*-?\s*blind",
                    r"placebo\s+controlled?",
                    r"randomly?\s+assigned?",
                    r"treatment\s+and\s+control\s+group"
                ],
                "evidence_indicators": [
                    "CONSORT", "trial registration", "primary endpoint", 
                    "secondary endpoint", "randomization ratio"
                ]
            },
            
            StudyType.OBSERVATIONAL_COHORT: {
                "keywords": [
                    "cohort study", "longitudinal study", "prospective study",
                    "follow up", "follow-up", "cohort", "longitudinal",
                    "prospective", "incident", "baseline characteristics"
                ],
                "patterns": [
                    r"cohort\s+study",
                    r"longitudinal\s+study",
                    r"prospective\s+study",
                    r"follow\s*-?\s*up\s+study",
                    r"baseline\s+and\s+follow\s*-?\s*up"
                ],
                "evidence_indicators": [
                    "hazard ratio", "incidence rate", "person-years",
                    "Kaplan-Meier", "survival analysis"
                ]
            },
            
            StudyType.CASE_CONTROL: {
                "keywords": [
                    "case control", "case-control", "retrospective study",
                    "cases and controls", "matched controls", "odds ratio"
                ],
                "patterns": [
                    r"case\s*-?\s*control\s+study",
                    r"retrospective\s+study",
                    r"cases?\s+and\s+controls?",
                    r"matched\s+controls?"
                ],
                "evidence_indicators": [
                    "odds ratio", "matching criteria", "case definition",
                    "control selection"
                ]
            },
            
            StudyType.CROSS_SECTIONAL: {
                "keywords": [
                    "cross sectional", "cross-sectional", "prevalence study",
                    "survey study", "descriptive study"
                ],
                "patterns": [
                    r"cross\s*-?\s*sectional\s+study",
                    r"prevalence\s+study",
                    r"descriptive\s+study"
                ],
                "evidence_indicators": [
                    "prevalence", "cross-sectional analysis", "point estimate"
                ]
            },
            
            StudyType.SYSTEMATIC_REVIEW: {
                "keywords": [
                    "systematic review", "literature review", "evidence synthesis",
                    "PRISMA", "search strategy", "inclusion criteria", "exclusion criteria"
                ],
                "patterns": [
                    r"systematic\s+review",
                    r"literature\s+review",
                    r"evidence\s+synthesis",
                    r"search\s+strategy"
                ],
                "evidence_indicators": [
                    "PRISMA", "search terms", "database search", "study selection",
                    "quality assessment", "risk of bias"
                ]
            },
            
            StudyType.META_ANALYSIS: {
                "keywords": [
                    "meta analysis", "meta-analysis", "pooled analysis",
                    "forest plot", "heterogeneity", "fixed effects", "random effects"
                ],
                "patterns": [
                    r"meta\s*-?\s*analysis",
                    r"pooled\s+analysis",
                    r"forest\s+plot",
                    r"fixed\s+effects?\s+model",
                    r"random\s+effects?\s+model"
                ],
                "evidence_indicators": [
                    "I² statistic", "heterogeneity", "forest plot", "funnel plot",
                    "publication bias", "Cochrane"
                ]
            },
            
            StudyType.CASE_REPORT: {
                "keywords": [
                    "case report", "case series", "case study", "clinical case",
                    "patient case", "rare case"
                ],
                "patterns": [
                    r"case\s+reports?",
                    r"case\s+series",
                    r"case\s+study",
                    r"clinical\s+case",
                    r"patient\s+case"
                ],
                "evidence_indicators": [
                    "patient history", "clinical presentation", "diagnosis",
                    "treatment outcome", "rare condition"
                ]
            },
            
            StudyType.EXPERIMENTAL: {
                "keywords": [
                    "experimental study", "laboratory study", "in vitro",
                    "in vivo", "animal model", "cell culture", "experimental design"
                ],
                "patterns": [
                    r"experimental\s+study",
                    r"laboratory\s+study",
                    r"in\s+vitro\s+study",
                    r"in\s+vivo\s+study",
                    r"animal\s+model"
                ],
                "evidence_indicators": [
                    "cell lines", "experimental conditions", "control conditions",
                    "laboratory protocols", "animal subjects"
                ]
            },
            
            StudyType.COMPUTATIONAL: {
                "keywords": [
                    "computational study", "simulation", "modeling", "machine learning",
                    "artificial intelligence", "algorithm", "bioinformatics",
                    "computational analysis", "data mining"
                ],
                "patterns": [
                    r"computational\s+study",
                    r"simulation\s+study",
                    r"machine\s+learning",
                    r"artificial\s+intelligence",
                    r"computational\s+analysis"
                ],
                "evidence_indicators": [
                    "algorithm", "model performance", "validation dataset",
                    "computational methods", "software implementation"
                ]
            },
            
            StudyType.THEORETICAL: {
                "keywords": [
                    "theoretical study", "mathematical model", "conceptual framework",
                    "theoretical analysis", "hypothesis", "theory"
                ],
                "patterns": [
                    r"theoretical\s+study",
                    r"mathematical\s+model",
                    r"conceptual\s+framework",
                    r"theoretical\s+analysis"
                ],
                "evidence_indicators": [
                    "mathematical equations", "theoretical framework",
                    "conceptual model", "hypothesis testing"
                ]
            },
            
            StudyType.SURVEY: {
                "keywords": [
                    "survey study", "questionnaire", "survey", "interview",
                    "self reported", "self-reported", "response rate"
                ],
                "patterns": [
                    r"survey\s+study",
                    r"questionnaire\s+study",
                    r"interview\s+study",
                    r"self\s*-?\s*reported"
                ],
                "evidence_indicators": [
                    "response rate", "questionnaire validation", "survey design",
                    "participant recruitment"
                ]
            },
            
            StudyType.QUALITATIVE: {
                "keywords": [
                    "qualitative study", "qualitative research", "thematic analysis",
                    "phenomenological", "ethnographic", "grounded theory",
                    "content analysis", "interview"
                ],
                "patterns": [
                    r"qualitative\s+study",
                    r"qualitative\s+research",
                    r"thematic\s+analysis",
                    r"phenomenological\s+study",
                    r"ethnographic\s+study"
                ],
                "evidence_indicators": [
                    "thematic analysis", "coding", "saturation", "phenomenology",
                    "participant interviews"
                ]
            }
        }
    
    def _load_or_initialize_model(self) -> None:
        """Load existing model or initialize a new one."""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.classification_model = pickle.load(f)
                logger.info(f"Loaded classification model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Initializing new model.")
                self._initialize_default_model()
        else:
            self._initialize_default_model()
    
    def _initialize_default_model(self) -> None:
        """Initialize default classification model."""
        self.classification_model = {
            'version': '1.0',
            'weights': {study_type: 1.0 for study_type in StudyType},
            'threshold': 0.3
        }
        logger.info("Initialized default classification model")
    
    def classify_study_type(self, text: str, 
                          include_evidence: bool = True) -> StudyClassification:
        """
        Classify the study type of a research manuscript.
        
        Args:
            text: Full manuscript text or abstract
            include_evidence: Whether to include detailed evidence
            
        Returns:
            StudyClassification with results
        """
        if not text or not text.strip():
            return StudyClassification(
                primary_type=StudyType.UNKNOWN,
                confidence=0.0,
                secondary_types=[],
                evidence={},
                methodology_keywords=[]
            )
        
        # Clean and preprocess text
        clean_text_content = clean_text(text, remove_extra_whitespace=True)
        text_lower = clean_text_content.lower()
        
        # Score each study type
        study_scores = {}
        evidence_details = {}
        
        for study_type, config in self.study_patterns.items():
            score, evidence = self._score_study_type(
                text_lower, clean_text_content, study_type, config
            )
            study_scores[study_type] = score
            if include_evidence:
                evidence_details[study_type.value] = evidence
        
        # Determine primary and secondary classifications
        sorted_scores = sorted(
            study_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        primary_type = sorted_scores[0][0]
        primary_confidence = sorted_scores[0][1]
        
        # If confidence is too low, classify as unknown
        threshold = self.classification_model.get('threshold', 0.3)
        if primary_confidence < threshold:
            primary_type = StudyType.UNKNOWN
            primary_confidence = 0.0
        
        # Get secondary types (above threshold)
        secondary_types = [
            (study_type, score) 
            for study_type, score in sorted_scores[1:5]  # Top 4 secondary
            if score >= threshold * 0.5  # Lower threshold for secondary
        ]
        
        # Extract methodology keywords
        methodology_keywords = self._extract_methodology_keywords(clean_text_content)
        
        return StudyClassification(
            primary_type=primary_type,
            confidence=primary_confidence,
            secondary_types=secondary_types,
            evidence=evidence_details if include_evidence else {},
            methodology_keywords=methodology_keywords
        )
    
    def _score_study_type(self, text_lower: str, original_text: str, 
                         study_type: StudyType, config: Dict) -> Tuple[float, Dict]:
        """
        Score how well the text matches a specific study type.
        
        Args:
            text_lower: Lowercase text for pattern matching
            original_text: Original text for context
            study_type: Study type to score
            config: Configuration for this study type
            
        Returns:
            Tuple of (score, evidence_dict)
        """
        total_score = 0.0
        evidence = {
            'keyword_matches': [],
            'pattern_matches': [],
            'evidence_indicators': [],
            'context_score': 0.0
        }
        
        # Keyword matching (weighted by importance)
        keyword_score = 0.0
        for keyword in config['keywords']:
            if keyword in text_lower:
                # Weight based on keyword specificity
                weight = len(keyword.split()) * 0.2 + 0.3  # Multi-word = higher weight
                keyword_score += weight
                evidence['keyword_matches'].append(keyword)
        
        # Pattern matching (regex patterns)
        pattern_score = 0.0
        for pattern in config['patterns']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                pattern_score += len(matches) * 0.4
                evidence['pattern_matches'].extend(matches)
        
        # Evidence indicators (specialized terms)
        evidence_score = 0.0
        for indicator in config['evidence_indicators']:
            if indicator.lower() in text_lower:
                evidence_score += 0.3
                evidence['evidence_indicators'].append(indicator)
        
        # Contextual scoring using embeddings (if available)
        context_score = self._calculate_contextual_score(
            original_text, study_type, config
        )
        evidence['context_score'] = context_score
        
        # Combine scores with weights
        total_score = (
            keyword_score * 0.4 +      # 40% keyword matching
            pattern_score * 0.3 +      # 30% pattern matching
            evidence_score * 0.2 +     # 20% evidence indicators
            context_score * 0.1        # 10% contextual similarity
        )
        
        # Apply study type specific weights
        model_weight = self.classification_model['weights'].get(study_type, 1.0)
        total_score *= model_weight
        
        # Normalize score to 0-1 range
        total_score = min(total_score, 1.0)
        
        return total_score, evidence
    
    def _calculate_contextual_score(self, text: str, study_type: StudyType, 
                                  config: Dict) -> float:
        """
        Calculate contextual similarity score using embeddings.
        
        Args:
            text: Original text
            study_type: Study type being evaluated
            config: Study type configuration
            
        Returns:
            Contextual similarity score
        """
        try:
            # Create representative text for this study type
            study_description = f"{study_type.value.replace('_', ' ')} study methodology"
            study_keywords = ' '.join(config['keywords'][:5])  # Top 5 keywords
            study_context = f"{study_description}. {study_keywords}"
            
            # Generate embeddings
            text_embedding = embed_text(text[:1000])  # Limit text length
            study_embedding = embed_text(study_context)
            
            # Calculate cosine similarity
            from .embedder import cosine_similarity_single
            similarity = cosine_similarity_single(text_embedding, study_embedding)
            
            # Convert to 0-1 range and apply scaling
            contextual_score = max(0, (similarity + 1) / 2)  # Convert from [-1,1] to [0,1]
            
            return contextual_score
            
        except Exception as e:
            logger.debug(f"Contextual scoring failed: {e}")
            return 0.0
    
    def _extract_methodology_keywords(self, text: str) -> List[str]:
        """
        Extract methodology-related keywords from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of methodology keywords
        """
        # Define methodology-related terms
        methodology_terms = {
            'design', 'method', 'approach', 'technique', 'procedure',
            'protocol', 'analysis', 'statistical', 'sample', 'population',
            'data', 'measurement', 'instrument', 'validation', 'reliability',
            'randomization', 'blinding', 'control', 'intervention', 'outcome',
            'primary', 'secondary', 'endpoint', 'hypothesis', 'significance',
            'confidence', 'power', 'effect', 'bias', 'confounding'
        }
        
        # Extract general keywords from text
        general_keywords = extract_keywords(text, top_k=20)
        
        # Filter for methodology-related keywords
        methodology_keywords = [
            keyword for keyword in general_keywords
            if any(term in keyword.lower() for term in methodology_terms)
        ]
        
        # Add any methodology terms found directly
        text_lower = text.lower()
        for term in methodology_terms:
            if term in text_lower and term not in methodology_keywords:
                methodology_keywords.append(term)
        
        return methodology_keywords[:10]  # Limit to top 10
    
    def batch_classify(self, texts: List[str]) -> List[StudyClassification]:
        """
        Classify multiple texts in batch.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of StudyClassification results
        """
        results = []
        
        logger.info(f"Batch classifying {len(texts)} texts")
        
        for i, text in enumerate(texts):
            try:
                classification = self.classify_study_type(text, include_evidence=False)
                results.append(classification)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Classified {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Classification failed for text {i}: {e}")
                # Add unknown classification for failed texts
                results.append(StudyClassification(
                    primary_type=StudyType.UNKNOWN,
                    confidence=0.0,
                    secondary_types=[],
                    evidence={},
                    methodology_keywords=[]
                ))
        
        logger.info(f"Batch classification completed: {len(results)} results")
        return results
    
    def get_study_type_summary(self, classifications: List[StudyClassification]) -> Dict[str, Any]:
        """
        Generate summary statistics for a batch of classifications.
        
        Args:
            classifications: List of classification results
            
        Returns:
            Summary statistics dictionary
        """
        if not classifications:
            return {'total': 0, 'study_types': {}, 'average_confidence': 0.0}
        
        # Count study types
        type_counts = {}
        total_confidence = 0.0
        
        for classification in classifications:
            study_type = classification.primary_type.value
            type_counts[study_type] = type_counts.get(study_type, 0) + 1
            total_confidence += classification.confidence
        
        # Calculate percentages
        total = len(classifications)
        type_percentages = {
            study_type: (count / total) * 100
            for study_type, count in type_counts.items()
        }
        
        # Sort by frequency
        sorted_types = sorted(
            type_percentages.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total': total,
            'study_types': {
                'counts': type_counts,
                'percentages': type_percentages,
                'ranked': sorted_types
            },
            'average_confidence': total_confidence / total,
            'high_confidence_count': sum(
                1 for c in classifications if c.confidence >= 0.7
            ),
            'unknown_count': sum(
                1 for c in classifications 
                if c.primary_type == StudyType.UNKNOWN
            )
        }
    
    def save_model(self) -> None:
        """Save the classification model to disk."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.classification_model, f)
            logger.info(f"Saved classification model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")


def classify_manuscript_study_type(text: str) -> StudyClassification:
    """
    Convenience function to classify a single manuscript.
    
    Args:
        text: Manuscript text or abstract
        
    Returns:
        StudyClassification result
    """
    classifier = StudyTypeClassifier()
    return classifier.classify_study_type(text)


def get_study_type_display_name(study_type: StudyType) -> str:
    """
    Get human-readable display name for study type.
    
    Args:
        study_type: StudyType enum value
        
    Returns:
        Display name string
    """
    display_names = {
        StudyType.RANDOMIZED_CONTROLLED_TRIAL: "Randomized Controlled Trial",
        StudyType.OBSERVATIONAL_COHORT: "Observational Cohort Study",
        StudyType.CASE_CONTROL: "Case-Control Study",
        StudyType.CROSS_SECTIONAL: "Cross-Sectional Study",
        StudyType.SYSTEMATIC_REVIEW: "Systematic Review",
        StudyType.META_ANALYSIS: "Meta-Analysis",
        StudyType.CASE_REPORT: "Case Report/Series",
        StudyType.EXPERIMENTAL: "Experimental Study",
        StudyType.COMPUTATIONAL: "Computational Study",
        StudyType.THEORETICAL: "Theoretical Study",
        StudyType.SURVEY: "Survey Study",
        StudyType.QUALITATIVE: "Qualitative Research",
        StudyType.UNKNOWN: "Unknown/Other"
    }
    
    return display_names.get(study_type, study_type.value)


def get_study_type_description(study_type: StudyType) -> str:
    """
    Get detailed description of study type.
    
    Args:
        study_type: StudyType enum value
        
    Returns:
        Description string
    """
    descriptions = {
        StudyType.RANDOMIZED_CONTROLLED_TRIAL: 
            "A prospective study with random allocation of participants to treatment groups",
        StudyType.OBSERVATIONAL_COHORT: 
            "A longitudinal study following participants over time to observe outcomes",
        StudyType.CASE_CONTROL: 
            "A retrospective study comparing cases with a condition to controls without",
        StudyType.CROSS_SECTIONAL: 
            "A study examining a population at a single point in time",
        StudyType.SYSTEMATIC_REVIEW: 
            "A comprehensive review of literature following systematic methodology",
        StudyType.META_ANALYSIS: 
            "Statistical analysis combining results from multiple independent studies",
        StudyType.CASE_REPORT: 
            "Detailed report of one or more individual cases or case series",
        StudyType.EXPERIMENTAL: 
            "Laboratory-based study with controlled experimental conditions",
        StudyType.COMPUTATIONAL: 
            "Study using computational methods, modeling, or machine learning",
        StudyType.THEORETICAL: 
            "Study developing or testing theoretical frameworks or mathematical models",
        StudyType.SURVEY: 
            "Study collecting data through questionnaires or structured interviews",
        StudyType.QUALITATIVE: 
            "Study using qualitative methods to understand experiences or phenomena",
        StudyType.UNKNOWN: 
            "Study type could not be determined or does not fit standard categories"
    }
    
    return descriptions.get(study_type, "No description available")