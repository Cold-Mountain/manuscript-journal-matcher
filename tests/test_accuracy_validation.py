"""
Accuracy validation tests using real-world-like academic papers.

This module tests the system's accuracy using carefully crafted test fixtures
that represent different academic domains and validates that the matching
results are meaningful and relevant.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from extractor import extract_manuscript_data, extract_title_and_abstract
    from embedder import embed_text, cosine_similarity_single
    from match_journals import JournalMatcher, format_search_results
    from utils import extract_keywords, clean_text
    from config import get_embedding_dimension
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestAccuracyValidation:
    """Test system accuracy using realistic academic paper fixtures."""
    
    @pytest.fixture
    def fixtures_dir(self):
        """Get the fixtures directory path."""
        return Path(__file__).parent / "fixtures"
    
    @pytest.fixture
    def expected_matches(self, fixtures_dir):
        """Load expected match data."""
        with open(fixtures_dir / "expected_matches.json", 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def sample_papers(self, fixtures_dir):
        """Load all sample paper contents."""
        papers = {}
        for paper_file in fixtures_dir.glob("sample_*.txt"):
            with open(paper_file, 'r', encoding='utf-8') as f:
                papers[paper_file.name] = f.read()
        return papers
    
    @pytest.fixture
    def mock_domain_journals(self):
        """Create domain-specific mock journals for testing."""
        journals = []
        
        # Medical/Healthcare journals
        medical_journals = [
            {
                'id': 'med1',
                'display_name': 'Journal of Medical AI',
                'publisher': 'Medical Publishers',
                'is_oa': True,
                'subjects': [{'name': 'Medicine', 'score': 0.9}, {'name': 'Artificial Intelligence', 'score': 0.8}],
                'semantic_fingerprint': 'Medical artificial intelligence machine learning healthcare clinical diagnosis patient treatment',
                'embedding': None  # Will be generated
            },
            {
                'id': 'med2', 
                'display_name': 'Cardiovascular Research',
                'publisher': 'Heart Foundation',
                'is_oa': False,
                'subjects': [{'name': 'Cardiology', 'score': 0.95}, {'name': 'Medicine', 'score': 0.9}],
                'semantic_fingerprint': 'Cardiovascular heart disease medical research clinical trials patient outcomes',
                'embedding': None
            }
        ]
        
        # Computer Science journals
        cs_journals = [
            {
                'id': 'cs1',
                'display_name': 'Neural Networks and AI',
                'publisher': 'Tech Publications',
                'is_oa': True,
                'subjects': [{'name': 'Computer Science', 'score': 0.95}, {'name': 'Artificial Intelligence', 'score': 0.9}],
                'semantic_fingerprint': 'Neural networks machine learning artificial intelligence deep learning algorithms computer science',
                'embedding': None
            },
            {
                'id': 'cs2',
                'display_name': 'Natural Language Processing Review',
                'publisher': 'ACM',
                'is_oa': False,
                'subjects': [{'name': 'Natural Language Processing', 'score': 0.95}, {'name': 'Computer Science', 'score': 0.9}],
                'semantic_fingerprint': 'Natural language processing text analysis machine translation neural networks attention mechanisms',
                'embedding': None
            }
        ]
        
        # Biology/Genetics journals
        bio_journals = [
            {
                'id': 'bio1',
                'display_name': 'Gene Therapy and Editing',
                'publisher': 'Nature Publishing',
                'is_oa': True,
                'subjects': [{'name': 'Genetics', 'score': 0.95}, {'name': 'Gene Therapy', 'score': 0.9}],
                'semantic_fingerprint': 'Gene therapy CRISPR gene editing genetic diseases molecular biology therapeutic applications',
                'embedding': None
            },
            {
                'id': 'bio2',
                'display_name': 'Molecular Biology Research',
                'publisher': 'Science Publishers',
                'is_oa': False,
                'subjects': [{'name': 'Molecular Biology', 'score': 0.95}, {'name': 'Biology', 'score': 0.9}],
                'semantic_fingerprint': 'Molecular biology cellular mechanisms protein function genetic research laboratory techniques',
                'embedding': None
            }
        ]
        
        all_journals = medical_journals + cs_journals + bio_journals
        
        # Generate embeddings for each journal
        for journal in all_journals:
            journal['embedding'] = embed_text(journal['semantic_fingerprint']).tolist()
        
        embeddings = np.array([journal['embedding'] for journal in all_journals])
        
        return all_journals, embeddings
    
    def test_title_extraction_accuracy(self, sample_papers, expected_matches):
        """Test accuracy of title extraction from sample papers."""
        for paper_name, paper_content in sample_papers.items():
            if paper_name in expected_matches['test_fixtures']:
                expected_title = expected_matches['test_fixtures'][paper_name]['expected_title']
                
                # Extract title
                title, _ = extract_title_and_abstract(paper_content)
                
                # Verify title extraction
                assert title is not None, f"Failed to extract title from {paper_name}"
                assert len(title.strip()) > 0, f"Empty title extracted from {paper_name}"
                
                # Check if extracted title matches expected (allowing for minor variations)
                title_similarity = cosine_similarity_single(
                    embed_text(title),
                    embed_text(expected_title)
                )
                
                assert title_similarity > 0.8, f"Extracted title doesn't match expected for {paper_name}: {title_similarity:.3f}"
                print(f"✅ Title extraction for {paper_name}: {title_similarity:.3f} similarity")
    
    def test_abstract_extraction_accuracy(self, sample_papers, expected_matches):
        """Test accuracy of abstract extraction from sample papers."""
        for paper_name, paper_content in sample_papers.items():
            if paper_name in expected_matches['test_fixtures']:
                expected_keywords = expected_matches['test_fixtures'][paper_name]['expected_abstract_keywords']
                
                # Extract abstract
                _, abstract = extract_title_and_abstract(paper_content)
                
                # Verify abstract extraction
                assert abstract is not None, f"Failed to extract abstract from {paper_name}"
                assert len(abstract.strip()) > expected_matches['quality_metrics']['min_abstract_length'], \
                    f"Abstract too short for {paper_name}"
                
                # Check if abstract contains expected keywords
                abstract_lower = abstract.lower()
                found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in abstract_lower)
                keyword_coverage = found_keywords / len(expected_keywords)
                
                assert keyword_coverage >= 0.6, f"Abstract missing too many expected keywords for {paper_name}: {keyword_coverage:.2f}"
                print(f"✅ Abstract extraction for {paper_name}: {keyword_coverage:.2f} keyword coverage")
    
    def test_domain_specific_matching_accuracy(self, sample_papers, mock_domain_journals, expected_matches):
        """Test that papers match to appropriate domain-specific journals."""
        journals, embeddings = mock_domain_journals
        
        with patch('journal_db_builder.load_journal_database') as mock_load_db:
            mock_load_db.return_value = (journals, embeddings)
            
            matcher = JournalMatcher()
            matcher.load_database()
            
            for paper_name, paper_content in sample_papers.items():
                if paper_name in expected_matches['test_fixtures']:
                    expected_types = expected_matches['test_fixtures'][paper_name]['expected_journal_types']
                    
                    # Extract abstract and search for journals
                    _, abstract = extract_title_and_abstract(paper_content)
                    results = matcher.search_similar_journals(abstract, top_k=3)
                    
                    assert len(results) > 0, f"No matches found for {paper_name}"
                    
                    # Check if top results are from appropriate domains
                    top_result = results[0]
                    top_journal_subjects = [s['name'].lower() for s in top_result.get('subjects', [])]
                    
                    # Check if any expected type matches journal subjects
                    domain_match = any(
                        any(expected_type.lower() in subject for subject in top_journal_subjects)
                        for expected_type in expected_types
                    )
                    
                    assert domain_match, f"Top result for {paper_name} doesn't match expected domain types: {expected_types}"
                    print(f"✅ Domain matching for {paper_name}: {top_result['display_name']} ({top_result['similarity_score']:.3f})")
    
    def test_similarity_score_quality(self, sample_papers, mock_domain_journals):
        """Test that similarity scores are meaningful and well-distributed."""
        journals, embeddings = mock_domain_journals
        
        with patch('journal_db_builder.load_journal_database') as mock_load_db:
            mock_load_db.return_value = (journals, embeddings)
            
            matcher = JournalMatcher()
            matcher.load_database()
            
            all_scores = []
            
            for paper_name, paper_content in sample_papers.items():
                _, abstract = extract_title_and_abstract(paper_content)
                results = matcher.search_similar_journals(abstract, top_k=len(journals))
                
                scores = [result['similarity_score'] for result in results]
                all_scores.extend(scores)
                
                # Test score properties
                assert all(0 <= score <= 1 for score in scores), f"Invalid similarity scores for {paper_name}"
                assert scores == sorted(scores, reverse=True), f"Results not sorted by similarity for {paper_name}"
                
                # Check score distribution
                if len(scores) > 1:
                    score_range = max(scores) - min(scores)
                    assert score_range > 0.1, f"Similarity scores too similar for {paper_name}: range={score_range:.3f}"
                
                print(f"✅ Score quality for {paper_name}: range=[{min(scores):.3f}, {max(scores):.3f}]")
            
            # Test overall score distribution
            if all_scores:
                mean_score = np.mean(all_scores)
                std_score = np.std(all_scores)
                
                assert 0.2 < mean_score < 0.8, f"Mean similarity score seems unrealistic: {mean_score:.3f}"
                assert std_score > 0.05, f"Similarity scores lack sufficient variation: {std_score:.3f}"
    
    def test_keyword_extraction_relevance(self, sample_papers, expected_matches):
        """Test that extracted keywords are relevant to the paper content."""
        for paper_name, paper_content in sample_papers.items():
            if paper_name in expected_matches['test_fixtures']:
                expected_keywords = expected_matches['test_fixtures'][paper_name]['expected_abstract_keywords']
                
                # Extract abstract and keywords
                _, abstract = extract_title_and_abstract(paper_content)
                extracted_keywords = extract_keywords(abstract, top_k=10)
                
                assert len(extracted_keywords) > 0, f"No keywords extracted from {paper_name}"
                
                # Check relevance by comparing with expected keywords
                relevance_scores = []
                for extracted_kw in extracted_keywords[:5]:  # Check top 5
                    max_similarity = 0
                    for expected_kw in expected_keywords:
                        try:
                            similarity = cosine_similarity_single(
                                embed_text(extracted_kw),
                                embed_text(expected_kw)
                            )
                            max_similarity = max(max_similarity, similarity)
                        except:
                            # If embedding fails, use string matching
                            if extracted_kw.lower() in expected_kw.lower() or expected_kw.lower() in extracted_kw.lower():
                                max_similarity = max(max_similarity, 0.7)
                    
                    relevance_scores.append(max_similarity)
                
                avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
                assert avg_relevance > 0.3, f"Extracted keywords not relevant enough for {paper_name}: {avg_relevance:.3f}"
                
                print(f"✅ Keyword relevance for {paper_name}: {avg_relevance:.3f} average similarity")
    
    def test_cross_domain_discrimination(self, sample_papers, mock_domain_journals):
        """Test that the system can discriminate between different academic domains."""
        journals, embeddings = mock_domain_journals
        
        with patch('journal_db_builder.load_journal_database') as mock_load_db:
            mock_load_db.return_value = (journals, embeddings)
            
            matcher = JournalMatcher()
            matcher.load_database()
            
            # Test medical paper
            medical_paper = sample_papers.get('sample_medical_paper.txt')
            if medical_paper:
                _, medical_abstract = extract_title_and_abstract(medical_paper)
                medical_results = matcher.search_similar_journals(medical_abstract, top_k=2)
                
                # Top result should be medical journal
                top_result = medical_results[0]
                top_subjects = [s['name'].lower() for s in top_result.get('subjects', [])]
                is_medical = any('medicine' in subject or 'medical' in subject for subject in top_subjects)
                
                assert is_medical, f"Medical paper didn't match medical journal: {top_result['display_name']}"
            
            # Test CS paper
            cs_paper = sample_papers.get('sample_cs_paper.txt')
            if cs_paper:
                _, cs_abstract = extract_title_and_abstract(cs_paper)
                cs_results = matcher.search_similar_journals(cs_abstract, top_k=2)
                
                # Top result should be CS journal
                top_result = cs_results[0]
                top_subjects = [s['name'].lower() for s in top_result.get('subjects', [])]
                is_cs = any('computer' in subject or 'artificial intelligence' in subject for subject in top_subjects)
                
                assert is_cs, f"CS paper didn't match CS journal: {top_result['display_name']}"
            
            # Test Biology paper  
            bio_paper = sample_papers.get('sample_biology_paper.txt')
            if bio_paper:
                _, bio_abstract = extract_title_and_abstract(bio_paper)
                bio_results = matcher.search_similar_journals(bio_abstract, top_k=2)
                
                # Top result should be biology journal
                top_result = bio_results[0]
                top_subjects = [s['name'].lower() for s in top_result.get('subjects', [])]
                is_bio = any('biology' in subject or 'genetics' in subject or 'gene' in subject for subject in top_subjects)
                
                assert is_bio, f"Biology paper didn't match biology journal: {top_result['display_name']}"
            
            print("✅ Cross-domain discrimination test passed")
    
    def test_result_formatting_consistency(self, sample_papers, mock_domain_journals):
        """Test that result formatting is consistent and complete."""
        journals, embeddings = mock_domain_journals
        
        with patch('journal_db_builder.load_journal_database') as mock_load_db:
            mock_load_db.return_value = (journals, embeddings)
            
            matcher = JournalMatcher()
            matcher.load_database()
            
            for paper_name, paper_content in sample_papers.items():
                _, abstract = extract_title_and_abstract(paper_content)
                results = matcher.search_similar_journals(abstract, top_k=3)
                formatted_results = format_search_results(results)
                
                assert len(formatted_results) == len(results), f"Formatting changed result count for {paper_name}"
                
                # Check required fields
                required_fields = ['journal_name', 'similarity_score', 'rank', 'publisher', 'is_open_access']
                
                for i, result in enumerate(formatted_results):
                    for field in required_fields:
                        assert field in result, f"Missing field {field} in result {i} for {paper_name}"
                    
                    # Check data types
                    assert isinstance(result['similarity_score'], (int, float)), f"Invalid similarity score type for {paper_name}"
                    assert isinstance(result['rank'], int), f"Invalid rank type for {paper_name}"
                    assert isinstance(result['is_open_access'], bool), f"Invalid open access type for {paper_name}"
                    
                    # Check value ranges
                    assert 0 <= result['similarity_score'] <= 1, f"Similarity score out of range for {paper_name}"
                    assert result['rank'] > 0, f"Invalid rank value for {paper_name}"
                
                print(f"✅ Result formatting consistent for {paper_name}")


class TestRobustnessValidation:
    """Test system robustness with edge cases and unusual inputs."""
    
    def test_short_abstract_handling(self):
        """Test handling of very short abstracts."""
        short_abstracts = [
            "Short abstract.",
            "This is a brief study.",
            "Machine learning application.",
            "Gene therapy research."
        ]
        
        for abstract in short_abstracts:
            try:
                embedding = embed_text(abstract)
                assert embedding.shape[0] == get_embedding_dimension()
                print(f"✅ Short abstract handled: '{abstract[:30]}...'")
            except Exception as e:
                pytest.fail(f"Failed to handle short abstract '{abstract}': {e}")
    
    def test_long_abstract_handling(self):
        """Test handling of very long abstracts."""
        # Create a very long abstract by repeating content
        base_text = "This research investigates machine learning applications in healthcare. "
        long_abstract = base_text * 100  # Very long text
        
        try:
            embedding = embed_text(long_abstract)
            assert embedding.shape[0] == get_embedding_dimension()
            print(f"✅ Long abstract handled: {len(long_abstract)} characters")
        except Exception as e:
            pytest.fail(f"Failed to handle long abstract: {e}")
    
    def test_special_characters_handling(self):
        """Test handling of abstracts with special characters and symbols."""
        special_abstracts = [
            "Abstract with symbols: α, β, γ, δ equations and μ-values.",
            "Research on C++ programming and .NET framework applications.",
            "Study of pH levels (7.4±0.2) and CO₂ concentrations at 37°C.",
            "Analysis of gene expression: IL-6, TNF-α, and NF-κB pathways."
        ]
        
        for abstract in special_abstracts:
            try:
                embedding = embed_text(abstract)
                assert embedding.shape[0] == get_embedding_dimension()
                print(f"✅ Special characters handled: '{abstract[:50]}...'")
            except Exception as e:
                pytest.fail(f"Failed to handle special characters in '{abstract}': {e}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])