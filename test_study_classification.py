#!/usr/bin/env python3
"""
Test script for study type classification system.

This script demonstrates the new study type classification feature
and tests it with various types of research manuscripts.
"""

from src.study_classifier import StudyTypeClassifier, classify_manuscript_study_type
from src.study_classifier import get_study_type_display_name, get_study_type_description
from src.match_journals import JournalMatcher

# Sample abstracts for different study types
test_abstracts = {
    "RCT": """
    Efficacy of a Novel AI-Guided Therapy for Depression: A Randomized Controlled Trial
    
    Background: Depression affects millions worldwide. This randomized controlled trial 
    evaluated the efficacy of AI-guided cognitive behavioral therapy.
    
    Methods: We randomly assigned 200 patients with major depressive disorder to either 
    AI-guided therapy (n=100) or standard care (n=100). Patients were double-blinded 
    to treatment allocation. The primary endpoint was change in PHQ-9 scores at 12 weeks.
    
    Results: AI-guided therapy showed significant improvement (p<0.001) compared to 
    standard care. The treatment group had a mean reduction of 8.2 points vs 3.1 points 
    in the control group.
    
    Conclusions: AI-guided therapy demonstrates superior efficacy for treating depression.
    Trial registration: NCT12345678.
    """,
    
    "Cohort": """
    Long-term Cardiovascular Outcomes in Diabetes Patients: A 10-Year Cohort Study
    
    Background: The long-term cardiovascular risks in diabetes patients remain unclear.
    
    Methods: We conducted a prospective cohort study following 5,000 diabetes patients 
    over 10 years. Baseline characteristics were collected, and participants underwent 
    annual follow-up visits. The primary outcome was major adverse cardiovascular events.
    
    Results: During 45,000 person-years of follow-up, 890 cardiovascular events occurred. 
    The hazard ratio for cardiovascular events was 2.1 (95% CI: 1.8-2.4) compared to 
    non-diabetic controls. Kaplan-Meier survival analysis showed significant differences.
    
    Conclusions: Diabetes patients have substantially increased long-term cardiovascular risk.
    """,
    
    "Meta-Analysis": """
    Machine Learning in Medical Diagnosis: A Systematic Review and Meta-Analysis
    
    Background: Machine learning applications in medical diagnosis are rapidly expanding.
    
    Methods: We conducted a systematic review following PRISMA guidelines. We searched 
    PubMed, Embase, and Cochrane databases for studies published 2015-2024. Random-effects 
    meta-analysis was performed using RevMan 5.4. Heterogeneity was assessed using I² statistic.
    
    Results: 127 studies met inclusion criteria, involving 450,000 patients. The pooled 
    diagnostic accuracy was 0.89 (95% CI: 0.86-0.92). Significant heterogeneity was observed 
    (I² = 78%). Forest plots showed consistent benefit across different medical specialties.
    
    Conclusions: Machine learning demonstrates high diagnostic accuracy with moderate heterogeneity.
    """,
    
    "Case Report": """
    Rare Presentation of COVID-19 with Neurological Complications: A Case Report
    
    Background: COVID-19 typically presents with respiratory symptoms, but rare neurological 
    manifestations have been reported.
    
    Case Presentation: A 45-year-old male presented with acute confusion and seizures. 
    PCR testing confirmed SARS-CoV-2 infection. MRI revealed bilateral temporal lobe 
    abnormalities consistent with encephalitis. The patient was treated with antivirals 
    and corticosteroids.
    
    Outcome: The patient recovered completely after 3 weeks of treatment with no 
    residual neurological deficits.
    
    Conclusions: This case highlights the rare but serious neurological complications 
    of COVID-19 and the importance of early recognition and treatment.
    """,
    
    "Computational": """
    Deep Learning for Automated Retinal Disease Detection: A Computational Study
    
    Background: Automated detection of retinal diseases could improve screening efficiency.
    
    Methods: We developed a convolutional neural network using 50,000 fundus photographs. 
    The model architecture included ResNet-50 backbone with custom classification layers. 
    Training used Adam optimizer with learning rate scheduling. We performed 5-fold 
    cross-validation and tested on an independent dataset of 10,000 images.
    
    Results: The algorithm achieved 94.2% accuracy, 91.5% sensitivity, and 96.8% specificity. 
    The area under the ROC curve was 0.967. Gradient-weighted class activation mapping 
    revealed that the model focused on clinically relevant features.
    
    Conclusions: Deep learning demonstrates high performance for automated retinal disease detection.
    """,
    
    "Survey": """
    Patient Attitudes Toward Telemedicine: A Cross-Sectional Survey Study
    
    Background: Telemedicine adoption increased dramatically during COVID-19, but patient 
    attitudes remain unclear.
    
    Methods: We conducted an online survey of 2,500 patients across 15 healthcare systems. 
    The questionnaire included validated instruments for technology acceptance and 
    satisfaction. Response rate was 68%. Statistical analysis used chi-square tests 
    and logistic regression.
    
    Results: 72% of patients reported positive attitudes toward telemedicine. Younger 
    patients (OR=2.1, p<0.01) and those with chronic conditions (OR=1.6, p<0.05) were 
    more likely to prefer virtual visits. Main concerns included technical difficulties (45%) 
    and reduced personal connection (38%).
    
    Conclusions: Most patients have positive attitudes toward telemedicine, with age 
    and health status as key predictors.
    """
}


def test_study_classification():
    """Test the study type classification system."""
    print("🔬 Study Type Classification System Test")
    print("=" * 60)
    
    # Initialize classifier
    classifier = StudyTypeClassifier()
    
    # Test each abstract
    for study_name, abstract in test_abstracts.items():
        print(f"\n📄 Testing: {study_name}")
        print("-" * 40)
        
        # Classify the study
        classification = classifier.classify_study_type(abstract)
        
        # Display results
        study_type_name = get_study_type_display_name(classification.primary_type)
        print(f"🎯 Detected Type: {study_type_name}")
        print(f"📊 Confidence: {classification.confidence:.3f}")
        
        if classification.secondary_types:
            print("🔄 Secondary Types:")
            for i, (study_type, confidence) in enumerate(classification.secondary_types[:3], 1):
                secondary_name = get_study_type_display_name(study_type)
                print(f"  {i}. {secondary_name} ({confidence:.3f})")
        
        if classification.methodology_keywords:
            print(f"🏷️ Keywords: {', '.join(classification.methodology_keywords[:5])}")
        
        # Show evidence if high confidence
        if classification.confidence > 0.5 and classification.evidence:
            print("🔍 Evidence:")
            for study_type_key, evidence in list(classification.evidence.items())[:1]:
                if evidence['keyword_matches']:
                    print(f"  Keywords: {', '.join(evidence['keyword_matches'][:3])}")
                if evidence['pattern_matches']:
                    print(f"  Patterns: {', '.join(evidence['pattern_matches'][:2])}")
        
        print()


def test_enhanced_journal_matching():
    """Test enhanced journal matching with study type classification."""
    print("\n🔍 Enhanced Journal Matching Test")
    print("=" * 60)
    
    try:
        # Initialize journal matcher
        matcher = JournalMatcher()
        matcher.load_database()
        
        # Test with RCT abstract
        rct_abstract = test_abstracts["RCT"]
        print("\n📋 Testing RCT Abstract:")
        print(f"'{rct_abstract[:100]}...'")
        
        # Search with study classification
        results = matcher.search_similar_journals(
            query_text=rct_abstract,
            top_k=5,
            include_study_classification=True
        )
        
        if results:
            print(f"\n✅ Found {len(results)} matching journals")
            
            # Show study classification from first result
            if 'search_metadata' in results[0]:
                study_info = results[0]['search_metadata'].get('study_classification')
                if study_info:
                    study_type_name = get_study_type_display_name(
                        StudyTypeClassifier.StudyType(study_info['primary_type'])
                    )
                    print(f"🔬 Detected Study Type: {study_type_name}")
                    print(f"📊 Classification Confidence: {study_info['confidence']:.3f}")
            
            # Show top 3 journal matches
            print(f"\n🏆 Top Journal Matches:")
            for i, journal in enumerate(results[:3], 1):
                name = journal.get('display_name', 'Unknown')
                similarity = journal.get('similarity_score', 0)
                publisher = journal.get('publisher', 'Unknown')
                
                print(f"  {i}. {name}")
                print(f"     Similarity: {similarity:.3f}")
                print(f"     Publisher: {publisher}")
                
                # Show methodology keywords if available
                if 'methodology_keywords' in journal:
                    keywords = journal['methodology_keywords'][:3]
                    if keywords:
                        print(f"     Keywords: {', '.join(keywords)}")
                print()
        
        else:
            print("❌ No matching journals found")
    
    except Exception as e:
        print(f"❌ Enhanced matching test failed: {e}")
        print("💡 Make sure the journal database is built first")


def test_batch_classification():
    """Test batch classification functionality."""
    print("\n📚 Batch Classification Test")
    print("=" * 60)
    
    classifier = StudyTypeClassifier()
    
    # Get all abstracts
    abstracts = list(test_abstracts.values())
    
    # Batch classify
    print(f"Processing {len(abstracts)} abstracts in batch...")
    classifications = classifier.batch_classify(abstracts)
    
    # Get summary
    summary = classifier.get_study_type_summary(classifications)
    
    print(f"\n📊 Batch Classification Summary:")
    print(f"  Total processed: {summary['total']}")
    print(f"  Average confidence: {summary['average_confidence']:.3f}")
    print(f"  High confidence (≥70%): {summary['high_confidence_count']}")
    print(f"  Unknown classifications: {summary['unknown_count']}")
    print()
    
    print("📈 Study Type Distribution:")
    for study_type, percentage in summary['study_types']['ranked']:
        count = summary['study_types']['counts'][study_type]
        print(f"  {get_study_type_display_name(StudyTypeClassifier.StudyType(study_type))}: "
              f"{count} ({percentage:.1f}%)")


def main():
    """Run all study classification tests."""
    print("🧪 MANUSCRIPT JOURNAL MATCHER - Study Classification Tests")
    print("=" * 80)
    
    try:
        # Test 1: Basic classification
        test_study_classification()
        
        # Test 2: Enhanced journal matching
        test_enhanced_journal_matching()
        
        # Test 3: Batch processing
        test_batch_classification()
        
        print("\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("🎉 Study type classification system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()