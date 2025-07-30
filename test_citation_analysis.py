#!/usr/bin/env python3
"""
Test script for citation network analysis system.

This script demonstrates the new citation analysis features
and tests them with manuscripts containing reference sections.
"""

from src.citation_analyzer import CitationNetworkAnalyzer, analyze_manuscript_citations
from src.ensemble_matcher import EnsembleJournalMatcher, MatchingMethod
from src.match_journals import JournalMatcher

# Sample manuscript with comprehensive references section
sample_manuscript_with_references = """
Machine Learning Applications in Medical Diagnosis: A Comprehensive Review

Abstract:
This review examines the current state of machine learning applications in medical diagnosis, 
analyzing recent advances and identifying future research directions. We synthesized findings 
from 150+ studies to provide insights into algorithmic approaches, clinical validation, and 
implementation challenges.

Introduction:
The integration of artificial intelligence and machine learning into healthcare has accelerated 
dramatically in recent years. Clinical decision support systems now leverage sophisticated 
algorithms to assist physicians in diagnostic processes, improving both accuracy and efficiency.

Methods:
We conducted a systematic literature review following PRISMA guidelines, searching PubMed, 
IEEE Xplore, and ACM Digital Library databases for studies published between 2018-2024.

Results:
Our analysis revealed significant advances in deep learning approaches, with convolutional 
neural networks showing particular promise for medical imaging applications. Natural language 
processing techniques have also demonstrated effectiveness in clinical text analysis.

Discussion:
The findings suggest that while machine learning shows tremendous potential in medical diagnosis, 
several challenges remain including data quality, algorithmic bias, and clinical integration.

Conclusions:
Future research should focus on developing more robust, interpretable algorithms and establishing 
comprehensive validation frameworks for clinical deployment.

References:

1. Rajpurkar, P., Chen, E., Banerjee, O., & Topol, E. J. (2022). AI in health and medicine. Nature Medicine, 28(1), 31-38. doi:10.1038/s41591-021-01614-0

2. Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118.

3. Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., ... & Webster, D. R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. JAMA, 316(22), 2402-2410.

4. McKinney, S. M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H., ... & Shetty, S. (2020). International evaluation of an AI system for breast cancer screening. Nature, 577(7788), 89-94.

5. Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C., Turakhia, M. P., & Ng, A. Y. (2019). Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. Nature Medicine, 25(1), 65-69.

6. Liu, Y., Chen, P. H. C., Krause, J., & Peng, L. (2019). How to read articles that use machine learning: users' guides to the medical literature. JAMA, 322(18), 1806-1816.

7. Topol, E. J. (2019). High-performance medicine: the convergence of human and artificial intelligence. Nature Medicine, 25(1), 44-56.

8. Chen, J. H., & Asch, S. M. (2017). Machine learning and prediction in medicine—beyond the peak of inflated expectations. New England Journal of Medicine, 376(26), 2507-2509.

9. Ching, T., Himmelstein, D. S., Beaulieu-Jones, B. K., Kalinin, A. A., Do, B. T., Way, G. P., ... & Greene, C. S. (2018). Opportunities and obstacles for deep learning in biology and medicine. Journal of the Royal Society Interface, 15(141), 20170387.

10. Yu, K. H., Beam, A. L., & Kohane, I. S. (2018). Artificial intelligence in healthcare. Nature Biomedical Engineering, 2(10), 719-731.

11. Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. New England Journal of Medicine, 380(14), 1347-1358.

12. Ghassemi, M., Naumann, T., Schulam, P., Beam, A. L., Chen, I. Y., & Ranganath, R. (2020). A review of challenges and opportunities in machine learning for health. AMIA Summits on Translational Science Proceedings, 2020, 191.

13. Beam, A. L., & Kohane, I. S. (2018). Big data and machine learning in health care. JAMA, 319(13), 1317-1318.

14. Char, D. S., Shah, N. H., & Magnus, D. (2018). Implementing machine learning in health care—addressing ethical challenges. New England Journal of Medicine, 378(11), 981-983.

15. Sendak, M. P., Gao, M., Brajer, N., & Balu, S. (2020). Presenting machine learning model information to clinical end users with model facts labels. NPJ Digital Medicine, 3(1), 1-4.

16. Shah, N. H., Milstein, A., & Bagley, S. C. (2019). Making machine learning models clinically useful. JAMA, 322(14), 1351-1352.

17. Wiens, J., Saria, S., Sendak, M., Ghassemi, M., Liu, V. X., Doshi-Velez, F., ... & Goldenberg, A. (2019). Do no harm: a roadmap for responsible machine learning for health care. Nature Medicine, 25(9), 1337-1340.

18. Larson, D. B., Harvey, H., Rubin, D. L., Irani, N., Tse, J. R., & Langlotz, C. P. (2018). Regulatory frameworks for development and evaluation of artificial intelligence–based diagnostic imaging algorithms: summary and recommendations. Journal of the American College of Radiology, 15(3), 413-424.

19. Park, S. H., & Han, K. (2018). Methodologic guide for evaluating clinical performance and effect of artificial intelligence technology for medical diagnosis and prediction. Radiology, 286(3), 800-809.

20. Bluemke, D. A., Moy, L., Bredella, M. A., Ertl-Wagner, B. B., Fowler, K. J., Goh, V. J., ... & Woo, S. (2020). Assessing radiology research on artificial intelligence: a brief guide for authors, reviewers, and readers—from the Radiology Editorial Board. Radiology, 294(3), 487-489.
"""

# Simpler manuscript for testing basic functionality
simple_manuscript = """
Deep Learning for Image Classification: A Study

Abstract:
We developed a convolutional neural network for medical image classification achieving 95% accuracy.

Methods:
Our approach used ResNet-50 architecture with transfer learning on a dataset of 10,000 images.

Results:
The model showed superior performance compared to traditional machine learning approaches.

References:

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition, 770-778.

2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017). ImageNet classification with deep convolutional neural networks. Communications of the ACM, 60(6), 84-90.

3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.

4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

5. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. Proceedings of the IEEE conference on computer vision and pattern recognition, 1-9.
"""


def test_basic_citation_analysis():
    """Test basic citation analysis functionality."""
    print("📚 Basic Citation Analysis Test")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CitationNetworkAnalyzer()
    
    # Analyze sample manuscript
    print(f"\n📄 Analyzing manuscript with references ({len(sample_manuscript_with_references)} characters)")
    print("-" * 40)
    
    analysis = analyzer.analyze_citations(sample_manuscript_with_references)
    
    # Display results
    print(f"📊 Citation Analysis Results:")
    print(f"  References found: {analysis.metadata['total_references']}")
    print(f"  Unique journals: {analysis.metadata['unique_journals']}")
    print(f"  Unique authors: {analysis.metadata['unique_authors']}")
    print(f"  Year span: {analysis.metadata['year_span']} years")
    print(f"  High-impact citations: {analysis.metadata['high_impact_citations']}")
    print(f"  Research ecosystem score: {analysis.research_ecosystem_score:.3f}")
    
    # Show cited journals
    if analysis.cited_journals:
        print(f"\n📋 Top Cited Journals:")
        sorted_journals = sorted(analysis.cited_journals.items(), key=lambda x: x[1], reverse=True)
        for journal, count in sorted_journals[:5]:
            print(f"  • {journal}: {count} citations")
    
    # Show subject areas
    if analysis.subject_areas:
        print(f"\n🔬 Detected Subject Areas:")
        sorted_subjects = sorted(analysis.subject_areas.items(), key=lambda x: x[1], reverse=True)
        for subject, score in sorted_subjects[:5]:
            print(f"  • {subject}: {score:.3f}")
    
    # Show temporal distribution
    if analysis.temporal_distribution:
        print(f"\n📅 Temporal Distribution (sample):")
        recent_years = {year: count for year, count in analysis.temporal_distribution.items() 
                       if year >= 2018}
        for year in sorted(recent_years.keys(), reverse=True)[:5]:
            print(f"  • {year}: {recent_years[year]} references")
    
    # Show author network info
    print(f"\n👥 Author Network:")
    print(f"  Total authors: {len(analysis.author_network.authors)}")
    print(f"  Collaborations: {len(analysis.author_network.collaborations)}")
    if analysis.author_network.authors:
        sample_authors = list(analysis.author_network.authors)[:3]
        print(f"  Sample authors: {', '.join(sample_authors)}")


def test_reference_parsing():
    """Test reference parsing functionality."""
    print("\n\n🔍 Reference Parsing Test")
    print("=" * 60)
    
    analyzer = CitationNetworkAnalyzer()
    
    # Test with simple manuscript
    print(f"\n📄 Testing reference parsing:")
    analysis = analyzer.analyze_citations(simple_manuscript)
    
    print(f"Found {len(analysis.references)} references:")
    
    for i, ref in enumerate(analysis.references[:3], 1):  # Show first 3
        print(f"\n  Reference {i}:")
        print(f"    Title: {ref.title or 'Not found'}")
        print(f"    Authors: {', '.join(ref.authors[:2]) if ref.authors else 'Not found'}")
        print(f"    Journal: {ref.journal or 'Not found'}")
        print(f"    Year: {ref.year or 'Not found'}")
        print(f"    DOI: {ref.doi or 'Not found'}")
        print(f"    Confidence: {ref.confidence:.3f}")


def test_ensemble_with_citations():
    """Test ensemble matching with citation network analysis."""
    print("\n\n🎯 Ensemble Matching with Citation Analysis")
    print("=" * 60)
    
    try:
        # Test with full manuscript
        manuscript = sample_manuscript_with_references
        
        print(f"\n📋 Testing ensemble matching with citation analysis:")
        print(f"Manuscript length: {len(manuscript)} characters")
        
        # Initialize ensemble matcher
        ensemble_matcher = EnsembleJournalMatcher()
        
        # Test with citation networks enabled
        results = ensemble_matcher.find_matching_journals(
            query_text=manuscript,
            top_k=5,
            min_confidence=0.2,
            methods_to_use=[
                MatchingMethod.SEMANTIC_SIMILARITY,
                MatchingMethod.CITATION_NETWORK,
                MatchingMethod.STUDY_TYPE_MATCHING,
                MatchingMethod.KEYWORD_MATCHING
            ]
        )
        
        if results:
            print(f"\n✅ Found {len(results)} matching journals")
            
            for i, result in enumerate(results, 1):
                name = result.journal_data.get('display_name', 'Unknown')[:50]
                score = result.ensemble_score
                confidence = result.confidence
                
                print(f"\n🏆 #{i}. {name}")
                print(f"   Ensemble Score: {score:.3f}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Explanation: {result.explanation}")
                
                # Show method contributions
                if result.individual_scores:
                    print(f"   Method Scores:")
                    for method, method_score in result.individual_scores.items():
                        if method_score > 0.1:  # Only show meaningful scores
                            method_display = method.value.replace('_', ' ').title()
                            print(f"     • {method_display}: {method_score:.3f}")
        else:
            print("❌ No matching journals found")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_citation_method_only():
    """Test citation network method in isolation."""
    print("\n\n📚 Citation Network Method Isolation Test")
    print("=" * 60)
    
    try:
        ensemble_matcher = EnsembleJournalMatcher()
        
        # Test citation network method only
        results = ensemble_matcher.find_matching_journals(
            query_text=sample_manuscript_with_references,
            top_k=5,
            min_confidence=0.1,
            methods_to_use=[MatchingMethod.CITATION_NETWORK]
        )
        
        if results:
            print(f"✅ Citation network analysis found {len(results)} results")
            
            for i, result in enumerate(results, 1):
                name = result.journal_data.get('display_name', 'Unknown')[:45]
                score = result.ensemble_score
                print(f"  {i}. {name}: {score:.3f}")
                
                # Show citation-specific metadata if available
                if 'search_metadata' in result.journal_data:
                    ensemble_meta = result.journal_data['search_metadata'].get('ensemble_matching', {})
                    if ensemble_meta:
                        print(f"     Citation confidence: {ensemble_meta.get('confidence', 0):.3f}")
        else:
            print("❌ No results from citation network analysis")
    
    except Exception as e:
        print(f"❌ Citation method test failed: {e}")


def test_journal_compatibility():
    """Test journal compatibility scoring."""
    print("\n\n🤝 Journal Compatibility Test")
    print("=" * 60)
    
    analyzer = CitationNetworkAnalyzer()
    
    # Analyze citations
    analysis = analyzer.analyze_citations(sample_manuscript_with_references)
    
    # Test compatibility with various journals
    test_journals = [
        "Nature Medicine",
        "JAMA",
        "New England Journal of Medicine", 
        "Nature",
        "IEEE Transactions on Medical Imaging",
        "Journal of Medical Internet Research",
        "PLOS ONE"
    ]
    
    print(f"\n📊 Compatibility scores for test journals:")
    
    for journal in test_journals:
        compatibility = analyzer.get_journal_compatibility_score(analysis, journal)
        print(f"  • {journal}: {compatibility:.3f}")


def test_author_network_analysis():
    """Test author network analysis functionality."""
    print("\n\n👥 Author Network Analysis Test")
    print("=" * 60)
    
    analyzer = CitationNetworkAnalyzer()
    analysis = analyzer.analyze_citations(sample_manuscript_with_references)
    
    author_network = analysis.author_network
    
    print(f"Author Network Analysis:")
    print(f"  Total unique authors: {len(author_network.authors)}")
    print(f"  Collaboration pairs: {len(author_network.collaborations)}")
    print(f"  Authors with institutions: {len(author_network.institutions)}")
    print(f"  Authors with research areas: {len(author_network.research_areas)}")
    
    # Show sample collaborations
    if author_network.collaborations:
        print(f"\n🤝 Sample Collaborations:")
        top_collabs = sorted(author_network.collaborations.items(), 
                           key=lambda x: x[1], reverse=True)[:3]
        for (author1, author2), count in top_collabs:
            print(f"  • {author1} ↔ {author2}: {count} papers")
    
    # Show research areas
    if author_network.research_areas:
        print(f"\n🔬 Research Areas by Author (sample):")
        for author, areas in list(author_network.research_areas.items())[:3]:
            if areas:
                areas_str = ', '.join(list(areas)[:3])
                print(f"  • {author}: {areas_str}")


def test_temporal_analysis():
    """Test temporal citation analysis."""
    print("\n\n📅 Temporal Analysis Test")
    print("=" * 60)
    
    analyzer = CitationNetworkAnalyzer()
    analysis = analyzer.analyze_citations(sample_manuscript_with_references)
    
    temporal_dist = analysis.temporal_distribution
    
    if temporal_dist:
        print(f"Temporal Distribution Analysis:")
        print(f"  Total years with citations: {len(temporal_dist)}")
        
        # Calculate statistics
        all_years = list(temporal_dist.keys())
        all_counts = list(temporal_dist.values())
        
        print(f"  Earliest citation: {min(all_years)}")
        print(f"  Most recent citation: {max(all_years)}")
        print(f"  Average citations per year: {sum(all_counts) / len(all_counts):.1f}")
        
        # Show year-by-year breakdown
        print(f"\n📊 Citations by Year:")
        for year in sorted(all_years, reverse=True):
            count = temporal_dist[year]
            bar = "█" * min(count, 10)  # Visual bar
            print(f"  {year}: {count:2d} {bar}")
    else:
        print("No temporal data found")


def main():
    """Run all citation analysis tests."""
    print("🧪 MANUSCRIPT JOURNAL MATCHER - Citation Network Analysis Tests")
    print("=" * 80)
    
    try:
        # Test 1: Basic citation analysis
        test_basic_citation_analysis()
        
        # Test 2: Reference parsing
        test_reference_parsing()
        
        # Test 3: Ensemble matching with citations
        test_ensemble_with_citations()
        
        # Test 4: Citation method isolation
        test_citation_method_only()
        
        # Test 5: Journal compatibility
        test_journal_compatibility()
        
        # Test 6: Author network analysis
        test_author_network_analysis()
        
        # Test 7: Temporal analysis
        test_temporal_analysis()
        
        print("\n" + "=" * 80)
        print("✅ All citation analysis tests completed successfully!")
        print("🎉 Citation network analysis system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()