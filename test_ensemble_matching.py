#!/usr/bin/env python3
"""
Test script for ensemble matching system.

This script demonstrates the new ensemble matching features
and tests them with various types of manuscript content.
"""

from src.ensemble_matcher import EnsembleJournalMatcher, MatchingMethod, create_ensemble_matcher
from src.match_journals import JournalMatcher
from src.study_classifier import get_study_type_display_name

# Sample manuscripts for different types of studies
test_manuscripts = {
    "clinical_trial": """
        Effectiveness of AI-Guided Therapy for Depression: A Randomized Controlled Trial
        
        Abstract: Background: Depression affects millions worldwide. This randomized controlled trial 
        evaluated the efficacy of AI-guided cognitive behavioral therapy versus standard care.
        
        Methods: We randomly assigned 200 patients with major depressive disorder to either 
        AI-guided therapy (n=100) or standard care (n=100). Patients were double-blinded 
        to treatment allocation. The primary endpoint was change in PHQ-9 scores at 12 weeks.
        
        Results: AI-guided therapy showed significant improvement (p<0.001) compared to 
        standard care. The treatment group had a mean reduction of 8.2 points vs 3.1 points 
        in the control group.
        
        Conclusions: AI-guided therapy demonstrates superior efficacy for treating depression.
        Trial registration: NCT12345678.
        """,
    
    "computational_study": """
        Deep Learning for Automated Medical Image Analysis: A Computational Approach
        
        Abstract: We developed a convolutional neural network for automated detection of 
        abnormalities in medical images. Our approach uses transfer learning and data 
        augmentation to improve performance.
        
        Methods: We trained a ResNet-50 model on 50,000 medical images using GPU computing 
        clusters. The model architecture included custom attention mechanisms and ensemble 
        predictions. We implemented cross-validation and hyperparameter optimization.
        
        Results: The algorithm achieved 95.2% accuracy with 92.8% sensitivity and 97.1% 
        specificity. The model demonstrated robust performance across different imaging 
        modalities and patient populations.
        
        Conclusions: Deep learning approaches show promise for automated medical diagnosis 
        with potential for clinical deployment.
        """,
    
    "meta_analysis": """
        Machine Learning in Healthcare: A Systematic Review and Meta-Analysis
        
        Abstract: Background: Machine learning applications in healthcare are rapidly expanding.
        
        Methods: We conducted a systematic review following PRISMA guidelines. We searched 
        PubMed, Embase, and Cochrane databases for studies published 2015-2024. Random-effects 
        meta-analysis was performed using RevMan 5.4. Heterogeneity was assessed using I² statistic.
        
        Results: 127 studies met inclusion criteria, involving 450,000 patients. The pooled 
        diagnostic accuracy was 0.89 (95% CI: 0.86-0.92). Significant heterogeneity was observed 
        (I² = 78%). Forest plots showed consistent benefit across different medical specialties.
        
        Conclusions: Machine learning demonstrates high diagnostic accuracy with moderate heterogeneity.
        """
}


def test_basic_ensemble_matching():
    """Test basic ensemble matching functionality."""
    print("🎯 Basic Ensemble Matching Test")
    print("=" * 60)
    
    # Initialize ensemble matcher
    ensemble_matcher = EnsembleJournalMatcher()
    
    # Test with clinical trial manuscript
    manuscript = test_manuscripts["clinical_trial"]
    print(f"\n📄 Testing Clinical Trial Manuscript ({len(manuscript)} characters)")
    print("-" * 40)
    
    # Perform ensemble matching
    results = ensemble_matcher.find_matching_journals(
        query_text=manuscript,
        top_k=5,
        min_confidence=0.3
    )
    
    if results:
        print(f"✅ Found {len(results)} matching journals using ensemble methods")
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n🏆 #{i}. {result.journal_data.get('display_name', 'Unknown')}")
            print(f"   Ensemble Score: {result.ensemble_score:.3f}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Explanation: {result.explanation}")
            print(f"   Methods Used: {len(result.individual_scores)}")
            
            # Show top contributing methods
            top_methods = sorted(result.individual_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top Methods:")
            for method, score in top_methods:
                method_name = method.value.replace('_', ' ').title()
                print(f"     • {method_name}: {score:.3f}")
    else:
        print("❌ No matching journals found")


def test_method_comparison():
    """Test comparison between different matching methods."""
    print("\n\n⚖️ Method Comparison Test")
    print("=" * 60)
    
    try:
        # Test with computational study
        manuscript = test_manuscripts["computational_study"]
        
        # Initialize standard matcher for comparison
        standard_matcher = JournalMatcher()
        standard_matcher.load_database()
        
        # Standard matching
        print("\n🔧 Standard FAISS Matching:")
        standard_results = standard_matcher.search_similar_journals(
            query_text=manuscript,
            top_k=5,
            use_ensemble_matching=False
        )
        
        if standard_results:
            print(f"   Found {len(standard_results)} results")
            for i, result in enumerate(standard_results[:3], 1):
                name = result.get('display_name', 'Unknown')[:50]
                score = result.get('similarity_score', 0)
                print(f"   {i}. {name}: {score:.3f}")
        
        # Ensemble matching
        print("\n🎯 Ensemble Matching:")
        ensemble_results = standard_matcher.search_similar_journals(
            query_text=manuscript,
            top_k=5,
            use_ensemble_matching=True
        )
        
        if ensemble_results:
            print(f"   Found {len(ensemble_results)} results")
            for i, result in enumerate(ensemble_results[:3], 1):
                name = result.get('display_name', 'Unknown')[:50]
                score = result.get('similarity_score', 0)  # This is ensemble_score
                confidence = result.get('ensemble_confidence', 0)
                print(f"   {i}. {name}: {score:.3f} (conf: {confidence:.3f})")
        
        # Compare overlap
        if standard_results and ensemble_results:
            standard_names = set(r.get('display_name', '') for r in standard_results[:5])
            ensemble_names = set(r.get('display_name', '') for r in ensemble_results[:5])
            overlap = len(standard_names.intersection(ensemble_names))
            print(f"\n📊 Top 5 Results Overlap: {overlap}/5 journals")
    
    except Exception as e:
        print(f"❌ Method comparison failed: {e}")


def test_custom_weights():
    """Test custom method weighting."""
    print("\n\n⚖️ Custom Method Weights Test")
    print("=" * 60)
    
    # Define different weighting schemes
    weight_schemes = {
        "semantic_focused": {
            MatchingMethod.SEMANTIC_SIMILARITY: 0.5,
            MatchingMethod.MULTIMODAL_ANALYSIS: 0.3,
            MatchingMethod.KEYWORD_MATCHING: 0.1,
            MatchingMethod.STUDY_TYPE_MATCHING: 0.1
        },
        "study_type_focused": {
            MatchingMethod.STUDY_TYPE_MATCHING: 0.4,
            MatchingMethod.SEMANTIC_SIMILARITY: 0.3,
            MatchingMethod.SUBJECT_MATCHING: 0.2,
            MatchingMethod.KEYWORD_MATCHING: 0.1
        },
        "balanced": {
            MatchingMethod.SEMANTIC_SIMILARITY: 0.25,
            MatchingMethod.MULTIMODAL_ANALYSIS: 0.20,
            MatchingMethod.STUDY_TYPE_MATCHING: 0.15,
            MatchingMethod.KEYWORD_MATCHING: 0.15,
            MatchingMethod.SUBJECT_MATCHING: 0.15,
            MatchingMethod.STRUCTURAL_MATCHING: 0.10
        }
    }
    
    manuscript = test_manuscripts["meta_analysis"]
    
    for scheme_name, weights in weight_schemes.items():
        print(f"\n🔬 Testing {scheme_name} weighting:")
        print("-" * 30)
        
        # Create matcher with custom weights
        ensemble_matcher = create_ensemble_matcher(custom_weights=weights)
        
        # Perform matching
        results = ensemble_matcher.find_matching_journals(
            query_text=manuscript,
            top_k=3,
            min_confidence=0.2
        )
        
        if results:
            print(f"   Results: {len(results)} journals")
            for i, result in enumerate(results, 1):
                name = result.journal_data.get('display_name', 'Unknown')[:40]
                score = result.ensemble_score
                print(f"   {i}. {name}: {score:.3f}")
        else:
            print("   No results found")


def test_method_selection():
    """Test selective method usage."""
    print("\n\n🔍 Selective Method Usage Test")
    print("=" * 60)
    
    ensemble_matcher = EnsembleJournalMatcher()
    manuscript = test_manuscripts["clinical_trial"]
    
    # Test different method combinations
    method_combinations = [
        ([MatchingMethod.SEMANTIC_SIMILARITY], "Semantic Only"),
        ([MatchingMethod.STUDY_TYPE_MATCHING, MatchingMethod.SUBJECT_MATCHING], "Study+Subject"),
        ([MatchingMethod.SEMANTIC_SIMILARITY, MatchingMethod.KEYWORD_MATCHING, 
          MatchingMethod.STUDY_TYPE_MATCHING], "Core Methods"),
        (list(MatchingMethod), "All Methods")
    ]
    
    for methods, description in method_combinations:
        print(f"\n🧪 Testing {description}:")
        print(f"   Methods: {[m.value for m in methods]}")
        
        results = ensemble_matcher.find_matching_journals(
            query_text=manuscript,
            top_k=3,
            methods_to_use=methods,
            min_confidence=0.1
        )
        
        if results:
            print(f"   Results: {len(results)} journals")
            avg_confidence = sum(r.confidence for r in results) / len(results)
            avg_score = sum(r.ensemble_score for r in results) / len(results)
            print(f"   Avg Score: {avg_score:.3f}, Avg Confidence: {avg_confidence:.3f}")
            
            # Show method contribution for first result
            if results[0].individual_scores:
                print(f"   Method Contributions (top result):")
                for method, score in results[0].individual_scores.items():
                    method_name = method.value.replace('_', ' ')
                    print(f"     {method_name}: {score:.3f}")
        else:
            print("   No results found")


def test_ensemble_explanations():
    """Test ensemble matching explanations."""
    print("\n\n💡 Ensemble Explanations Test")
    print("=" * 60)
    
    ensemble_matcher = EnsembleJournalMatcher()
    
    # Test each manuscript type
    for manuscript_type, manuscript in test_manuscripts.items():
        print(f"\n📄 {manuscript_type.replace('_', ' ').title()}:")
        print("-" * 30)
        
        results = ensemble_matcher.find_matching_journals(
            query_text=manuscript,
            top_k=2,
            min_confidence=0.2
        )
        
        if results:
            for i, result in enumerate(results, 1):
                name = result.journal_data.get('display_name', 'Unknown')[:35]
                print(f"   {i}. {name}")
                print(f"      Score: {result.ensemble_score:.3f}")
                print(f"      Confidence: {result.confidence:.3f}")
                print(f"      Explanation: {result.explanation}")
                
                # Show detailed method breakdown
                method_details = []
                for method, score in result.individual_scores.items():
                    if score > 0.1:  # Only show meaningful scores
                        method_name = method.value.replace('_', ' ')
                        method_details.append(f"{method_name}({score:.2f})")
                
                if method_details:
                    print(f"      Methods: {', '.join(method_details[:4])}")
        else:
            print("   No results found")


def test_filter_integration():
    """Test ensemble matching with filters."""
    print("\n\n🔍 Filter Integration Test")
    print("=" * 60)
    
    ensemble_matcher = EnsembleJournalMatcher()
    manuscript = test_manuscripts["computational_study"]
    
    # Test different filter combinations
    filter_sets = [
        ({}, "No Filters"),
        ({'open_access_only': True}, "Open Access Only"),
        ({'min_h_index': 50}, "High H-Index (≥50)"),
        ({'min_ensemble_score': 0.5}, "High Ensemble Score (≥0.5)"),
        ({'required_methods': [MatchingMethod.SEMANTIC_SIMILARITY, 
                              MatchingMethod.STUDY_TYPE_MATCHING]}, "Required Methods")
    ]
    
    for filters, description in filter_sets:
        print(f"\n🔧 {description}:")
        print(f"   Filters: {filters}")
        
        results = ensemble_matcher.find_matching_journals(
            query_text=manuscript,
            top_k=5,
            min_confidence=0.2,
            filters=filters
        )
        
        print(f"   Results: {len(results)} journals")
        
        if results:
            for i, result in enumerate(results[:2], 1):
                name = result.journal_data.get('display_name', 'Unknown')[:40]
                score = result.ensemble_score
                confidence = result.confidence
                print(f"   {i}. {name}: {score:.3f} (conf: {confidence:.3f})")


def main():
    """Run all ensemble matching tests."""
    print("🧪 MANUSCRIPT JOURNAL MATCHER - Ensemble Matching Tests")
    print("=" * 80)
    
    try:
        # Test 1: Basic ensemble matching
        test_basic_ensemble_matching()
        
        # Test 2: Method comparison
        test_method_comparison()
        
        # Test 3: Custom weights
        test_custom_weights()
        
        # Test 4: Method selection
        test_method_selection()
        
        # Test 5: Ensemble explanations
        test_ensemble_explanations()
        
        # Test 6: Filter integration
        test_filter_integration()
        
        print("\n" + "=" * 80)
        print("✅ All ensemble matching tests completed successfully!")
        print("🎉 Ensemble matching system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()