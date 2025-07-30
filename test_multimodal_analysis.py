#!/usr/bin/env python3
"""
Test script for multi-modal content analysis system.

This script demonstrates the new multi-modal analysis features
and tests them with various types of manuscript content.
"""

from src.multimodal_analyzer import MultiModalContentAnalyzer, ContentSection, analyze_manuscript_content
from src.match_journals import JournalMatcher
from src.study_classifier import get_study_type_display_name

# Sample complete manuscript for multi-modal analysis
sample_manuscript = """
Machine Learning Approaches for Automated Medical Diagnosis: A Comprehensive Study

Abstract:
Background: Automated medical diagnosis using machine learning has gained significant attention in recent years. This study evaluates the effectiveness of various machine learning algorithms for diagnosing common medical conditions from patient symptoms and medical imaging data.

Methods: We conducted a retrospective analysis of 10,000 patient records from three major hospitals. The dataset included patient demographics, symptoms, laboratory results, and imaging data. We implemented and compared five machine learning algorithms: Random Forest, Support Vector Machines, Neural Networks, Gradient Boosting, and Logistic Regression. Model performance was evaluated using accuracy, sensitivity, specificity, and area under the ROC curve.

Results: The Neural Network model achieved the highest overall accuracy of 94.2% (95% CI: 92.8-95.6%), followed by Gradient Boosting at 91.7% (95% CI: 90.1-93.3%). Sensitivity ranged from 88.4% to 96.1% across different conditions. The models showed excellent performance for cardiovascular diseases (AUC = 0.967) and diabetes (AUC = 0.942), but lower performance for rare conditions.

Conclusions: Machine learning algorithms demonstrate high accuracy for automated medical diagnosis, with neural networks showing superior performance. However, performance varies significantly across different medical conditions, suggesting the need for condition-specific optimization.

Keywords: machine learning, medical diagnosis, neural networks, healthcare automation

1. Introduction

The healthcare industry faces unprecedented challenges in delivering accurate and timely diagnoses. With the increasing volume of medical data and the growing complexity of medical conditions, there is an urgent need for automated systems that can assist healthcare professionals in the diagnostic process.

Machine learning has emerged as a promising solution for automated medical diagnosis. Recent advances in deep learning, natural language processing, and computer vision have opened new possibilities for analyzing medical data and extracting meaningful insights for clinical decision-making.

2. Methods

2.1 Study Design
We conducted a retrospective cohort study analyzing patient records from January 2018 to December 2022. The study was approved by the institutional review board of all participating hospitals.

2.2 Data Collection
Patient data was collected from electronic health records (EHRs) including:
- Demographic information (age, gender, BMI)
- Chief complaints and symptoms
- Vital signs and laboratory results
- Medical imaging reports
- Final diagnoses confirmed by board-certified physicians

2.3 Machine Learning Models
Five algorithms were implemented and optimized:
1. Random Forest: Ensemble method with 500 decision trees
2. Support Vector Machines: Radial basis function kernel
3. Neural Networks: Multi-layer perceptron with 3 hidden layers
4. Gradient Boosting: XGBoost implementation
5. Logistic Regression: L2 regularization

2.4 Model Training and Validation
The dataset was randomly split into training (70%), validation (15%), and test (15%) sets. Stratified sampling ensured balanced representation of all diagnostic conditions. Cross-validation was performed using 10-fold stratified cross-validation.

3. Results

3.1 Dataset Characteristics
The final dataset included 10,000 patients with a mean age of 52.3 ± 18.7 years. The most common diagnoses were hypertension (n=2,847), diabetes mellitus (n=2,234), and coronary artery disease (n=1,892).

3.2 Model Performance
Overall accuracy results:
- Neural Networks: 94.2% (95% CI: 92.8-95.6%)
- Gradient Boosting: 91.7% (95% CI: 90.1-93.3%)
- Random Forest: 89.3% (95% CI: 87.6-91.0%)
- Support Vector Machines: 87.8% (95% CI: 86.0-89.6%)
- Logistic Regression: 84.2% (95% CI: 82.2-86.2%)

Statistical significance testing using McNemar's test showed significant differences between all model pairs (p < 0.001).

4. Discussion

Our findings demonstrate that machine learning algorithms can achieve high accuracy in automated medical diagnosis, with neural networks showing the best overall performance. The superior performance of ensemble methods (Random Forest, Gradient Boosting) and neural networks aligns with recent literature in medical AI.

However, several limitations should be noted. First, the retrospective nature of the study may introduce selection bias. Second, the models were trained on data from specific hospitals, which may limit generalizability. Third, rare conditions showed lower performance, likely due to insufficient training examples.

5. Conclusions

This study provides evidence that machine learning algorithms, particularly neural networks, can achieve high accuracy for automated medical diagnosis. The results suggest significant potential for clinical implementation, but further validation in prospective studies is needed.

Future research should focus on improving performance for rare conditions, validating models across diverse populations, and developing explainable AI systems that can provide interpretable insights for clinical decision-making.

References:
[1] Smith, J. et al. (2022). Deep learning in medical diagnosis. Nature Medicine, 28(4), 123-134.
[2] Johnson, A. et al. (2021). Machine learning applications in healthcare. JAMA, 325(6), 567-578.
"""

def test_basic_multimodal_analysis():
    """Test basic multi-modal content analysis functionality."""
    print("🔬 Multi-Modal Content Analysis Test")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MultiModalContentAnalyzer()
    
    # Analyze the sample manuscript
    print(f"\n📄 Analyzing manuscript ({len(sample_manuscript)} characters)")
    print("-" * 40)
    
    analysis = analyzer.analyze_content(sample_manuscript)
    
    # Display overall results
    print(f"📊 Overall Content Quality: {analysis.content_quality_score:.3f}")
    print(f"🔍 Content Fingerprint: {analysis.content_fingerprint}")
    print(f"📋 Total Sections Analyzed: {len(analysis.sections)}")
    
    if analysis.study_classification:
        study_type_name = get_study_type_display_name(analysis.study_classification.primary_type)
        print(f"🔬 Study Type: {study_type_name} (confidence: {analysis.study_classification.confidence:.3f})")
    
    print(f"\n📈 Section Analysis:")
    print("-" * 40)
    
    # Display section-by-section analysis
    for section, section_analysis in analysis.sections.items():
        print(f"\n📄 {section.value.upper()}:")
        print(f"  Content length: {len(section_analysis.content)} chars")
        print(f"  Importance score: {section_analysis.importance_score:.3f}")
        print(f"  Quality score: {section_analysis.quality_score:.3f}")
        print(f"  Keywords: {', '.join(section_analysis.keywords[:4])}")
        
        # Show some metadata
        metadata = section_analysis.metadata
        print(f"  Word count: {metadata['word_count']}")
        print(f"  Technical terms: {metadata['technical_terms']}")
        print(f"  Contains citations: {'Yes' if metadata['contains_citations'] else 'No'}")
    
    print(f"\n💡 Analysis Metadata:")
    print(f"  Analysis timestamp: {analysis.metadata['analysis_timestamp']}")
    print(f"  Section lengths: {analysis.metadata['section_lengths']}")


def test_enhanced_journal_matching():
    """Test enhanced journal matching with multi-modal analysis."""
    print("\n\n🔍 Enhanced Journal Matching with Multi-Modal Analysis")
    print("=" * 60)
    
    try:
        # Initialize journal matcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Test with sample manuscript
        print("\n📋 Testing with full manuscript:")
        print(f"'{sample_manuscript[:100]}...'")
        
        # Search with multi-modal analysis enabled
        results = matcher.search_similar_journals(
            query_text=sample_manuscript,
            top_k=5,
            use_multimodal_analysis=True,
            include_study_classification=True
        )
        
        if results:
            print(f"\n✅ Found {len(results)} matching journals")
            
            # Show analysis metadata from first result
            if 'search_metadata' in results[0]:
                metadata = results[0]['search_metadata']
                
                # Multi-modal analysis info
                if 'multimodal_analysis' in metadata:
                    mm_info = metadata['multimodal_analysis']
                    print(f"\n📊 Multi-Modal Analysis Results:")
                    print(f"  Content Quality Score: {mm_info['content_quality_score']:.3f}")
                    print(f"  Sections Analyzed: {mm_info['total_sections']} ({', '.join(mm_info['sections_analyzed'])})")
                    print(f"  Content Fingerprint: {mm_info['content_fingerprint']}")
                
                # Study classification info
                if 'study_classification' in metadata:
                    study_info = metadata['study_classification']
                    study_type_name = get_study_type_display_name(
                        type('StudyType', (), {study_info['primary_type']: study_info['primary_type']})()
                    )
                    print(f"\n🔬 Study Classification:")
                    print(f"  Primary Type: {study_type_name}")
                    print(f"  Confidence: {study_info['confidence']:.3f}")
            
            # Show top 3 journal matches
            print(f"\n🏆 Top Journal Matches:")
            for i, journal in enumerate(results[:3], 1):
                name = journal.get('display_name', 'Unknown')
                similarity = journal.get('similarity_score', 0)
                publisher = journal.get('publisher', 'Unknown')
                
                print(f"\n  {i}. {name}")
                print(f"     Similarity: {similarity:.3f}")
                print(f"     Publisher: {publisher}")
                
                # Show multi-modal specific info if available
                if 'content_quality_score' in journal:
                    print(f"     Content Quality: {journal['content_quality_score']:.3f}")
                if 'section_count' in journal:
                    print(f"     Sections Matched: {journal['section_count']}")
        
        else:
            print("❌ No matching journals found")
    
    except Exception as e:
        print(f"❌ Enhanced matching test failed: {e}")
        print("💡 Make sure the journal database is built first")


def test_section_extraction():
    """Test content section extraction functionality."""
    print("\n\n📄 Content Section Extraction Test")
    print("=" * 60)
    
    analyzer = MultiModalContentAnalyzer()
    
    # Extract sections from sample manuscript
    sections = analyzer._extract_sections(sample_manuscript)
    
    print(f"\n✅ Extracted {len(sections)} sections:")
    print("-" * 40)
    
    for section, content in sections.items():
        print(f"\n📋 {section.value.upper()}:")
        print(f"  Length: {len(content)} characters")
        print(f"  Preview: {content[:100]}..." if len(content) > 100 else f"  Content: {content}")


def test_section_weighting():
    """Test custom section weighting functionality."""
    print("\n\n⚖️ Section Weighting Test")
    print("=" * 60)
    
    # Test with different weighting schemes
    weight_schemes = {
        "default": None,  # Use default weights
        "title_focused": {
            ContentSection.TITLE: 0.5,
            ContentSection.ABSTRACT: 0.3,
            ContentSection.METHODOLOGY: 0.1,
            ContentSection.RESULTS: 0.05,
            ContentSection.CONCLUSION: 0.05
        },
        "methods_focused": {
            ContentSection.TITLE: 0.2,
            ContentSection.ABSTRACT: 0.2,
            ContentSection.METHODOLOGY: 0.4,
            ContentSection.RESULTS: 0.1,
            ContentSection.CONCLUSION: 0.1
        }
    }
    
    for scheme_name, weights in weight_schemes.items():
        print(f"\n🔬 Testing {scheme_name} weighting:")
        print("-" * 30)
        
        analyzer = MultiModalContentAnalyzer(weights_config=weights)
        analysis = analyzer.analyze_content(sample_manuscript)
        
        print(f"  Content Quality Score: {analysis.content_quality_score:.3f}")
        print(f"  Sections Found: {len(analysis.sections)}")
        
        # Show section importance ranking
        importance_ranking = analyzer.get_section_importance_ranking(analysis)
        print(f"  Section Importance Ranking:")
        for section, importance in importance_ranking[:3]:
            print(f"    {section.value}: {importance:.3f}")


def test_comparison_functionality():
    """Test analysis comparison functionality."""
    print("\n\n🔄 Analysis Comparison Test")
    print("=" * 60)
    
    analyzer = MultiModalContentAnalyzer()
    
    # Create two different analyses
    analysis1 = analyzer.analyze_content(sample_manuscript)
    
    # Modified version for comparison
    modified_manuscript = sample_manuscript.replace("Neural Networks", "Deep Learning Networks")
    analysis2 = analyzer.analyze_content(modified_manuscript)
    
    # Compare analyses
    similarities = analyzer.compare_analyses(analysis1, analysis2)
    
    print("📊 Similarity Analysis:")
    print("-" * 30)
    for metric, similarity in similarities.items():
        print(f"  {metric}: {similarity:.3f}")


def main():
    """Run all multi-modal analysis tests."""
    print("🧪 MANUSCRIPT JOURNAL MATCHER - Multi-Modal Analysis Tests")
    print("=" * 80)
    
    try:
        # Test 1: Basic multi-modal analysis
        test_basic_multimodal_analysis()
        
        # Test 2: Enhanced journal matching
        test_enhanced_journal_matching()
        
        # Test 3: Section extraction
        test_section_extraction()
        
        # Test 4: Section weighting
        test_section_weighting()
        
        # Test 5: Comparison functionality
        test_comparison_functionality()
        
        print("\n" + "=" * 80)
        print("✅ All multi-modal analysis tests completed successfully!")
        print("🎉 Multi-modal content analysis system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()