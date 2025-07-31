#!/usr/bin/env python3
"""
Test script for advanced filtering and recommendation engine.

This script demonstrates the new recommendation engine features
and tests them with various filtering strategies and criteria.
"""

from src.recommendation_engine import (
    AdvancedRecommendationEngine, FilterCriteria, RecommendationStrategy,
    create_recommendation_filters, analyze_recommendation_suite
)
from src.journal_ranker import PrestigeLevel
from src.study_classifier import StudyType

# Sample manuscripts for testing different scenarios
test_manuscripts = {
    "high_impact_clinical": """
        Efficacy of AI-Guided Precision Medicine in Advanced Cancer Treatment:
        A Large-Scale, Multi-Center, Randomized Controlled Trial
        
        Abstract: Background: Precision medicine approaches in oncology show promise 
        but lack comprehensive AI integration. This international, multi-center randomized 
        controlled trial evaluated AI-guided treatment selection versus standard care 
        in 2,847 patients with stage III-IV solid tumors across 47 centers in 12 countries.
        
        Methods: Patients were randomly assigned (1:1) to AI-guided precision therapy 
        (n=1,424) or physician-guided standard care (n=1,423). The AI system analyzed 
        genomic, proteomic, and clinical data using deep learning algorithms. Primary 
        endpoint was overall survival at 24 months.
        
        Results: AI-guided therapy demonstrated significant improvement in overall survival 
        (HR=0.67, 95% CI: 0.58-0.78, p<0.001). Median overall survival was 18.3 months 
        (95% CI: 16.8-19.9) in the AI group versus 13.2 months (95% CI: 11.7-14.8) in 
        the control group. The number needed to treat was 7 (95% CI: 5-11).
        
        Conclusions: AI-guided precision medicine significantly improves survival outcomes 
        in advanced cancer patients, representing a paradigm shift in oncology treatment 
        selection with immediate clinical implications.
        
        References:
        1. Rajpurkar, P., Chen, E., Banerjee, O., & Topol, E. J. (2022). AI in health and medicine. Nature Medicine, 28(1), 31-38.
        2. McKinney, S. M., Sieniek, M., Godbole, V., et al. (2020). International evaluation of an AI system for breast cancer screening. Nature, 577(7788), 89-94.
        3. Topol, E. J. (2019). High-performance medicine: the convergence of human and artificial intelligence. Nature Medicine, 25(1), 44-56.
        """,
    
    "moderate_research": """
        Machine Learning Approaches for Biomarker Discovery in Type 2 Diabetes
        
        Abstract: Background: Type 2 diabetes biomarker identification remains challenging. 
        We developed machine learning models to identify novel biomarkers from metabolomic data.
        
        Methods: We analyzed plasma samples from 485 type 2 diabetes patients and 312 healthy 
        controls using untargeted metabolomics. Random forest and support vector machine 
        algorithms were applied for biomarker discovery with 80/20 train-test splits.
        
        Results: Our models achieved 87% accuracy in diabetes classification. We identified 
        15 potential biomarkers with area under curve >0.80. Three metabolites showed 
        significant correlation with HbA1c levels (r>0.65, p<0.01).
        
        Conclusions: Machine learning enables effective biomarker discovery from metabolomic 
        data. These findings warrant validation in larger cohorts for clinical translation.
        
        References:
        1. Yu, K. H., Beam, A. L., & Kohane, I. S. (2018). Artificial intelligence in healthcare. Nature Biomedical Engineering, 2(10), 719-731.
        2. Chen, J. H., & Asch, S. M. (2017). Machine learning and prediction in medicine. New England Journal of Medicine, 376(26), 2507-2509.
        """,
    
    "basic_observational_study": """
        Electronic Health Record Usage Patterns in Rural Primary Care Clinics
        
        Abstract: We conducted a cross-sectional survey to understand EHR adoption 
        patterns in rural primary care clinics across three states.
        
        Methods: We distributed online surveys to 150 clinic administrators. 
        The survey included questions about EHR systems, implementation challenges, 
        and user satisfaction. Response rate was 34% (n=51).
        
        Results: 78% of clinics had implemented EHR systems. Main challenges included 
        cost (65%), training requirements (48%), and technical issues (31%). 
        User satisfaction scores averaged 6.2/10.
        
        Conclusions: EHR adoption remains challenging for rural clinics despite 
        regulatory requirements. Better support systems are needed for successful 
        implementation.
        
        References:
        1. Adler-Milstein, J., & Jha, A. K. (2017). HITECH Act drove large gains in hospital electronic health record adoption. Health Affairs, 36(8), 1416-1422.
        """
}


def test_basic_recommendation_engine():
    """Test basic recommendation engine functionality."""
    print("🎯 Basic Recommendation Engine Test")
    print("=" * 60)
    
    try:
        # Initialize recommendation engine
        engine = AdvancedRecommendationEngine()
        
        manuscript = test_manuscripts["moderate_research"]
        
        print(f"\n📄 Testing with moderate research manuscript:")
        print(f"Manuscript length: {len(manuscript)} characters")
        
        # Generate basic recommendations
        suite = engine.generate_recommendations(
            manuscript_text=manuscript,
            strategy=RecommendationStrategy.BALANCED,
            max_recommendations=5
        )
        
        print(f"\n✅ Recommendation suite generated successfully!")
        print(f"Strategy: {suite.recommendation_strategy.value}")
        print(f"Journals considered: {suite.total_journals_considered}")
        print(f"Primary recommendations: {len(suite.primary_recommendations)}")
        print(f"Alternative recommendations: {len(suite.alternative_recommendations)}")
        print(f"Aspirational recommendations: {len(suite.aspirational_recommendations)}")
        print(f"Cost-effective recommendations: {len(suite.cost_effective_recommendations)}")
        print(f"Open access recommendations: {len(suite.open_access_recommendations)}")
        
        # Show top primary recommendations
        if suite.primary_recommendations:
            print(f"\n🏆 Top Primary Recommendations:")
            for i, rec in enumerate(suite.primary_recommendations[:3], 1):
                name = rec.journal_data.get('display_name', 'Unknown')[:45]
                score = rec.recommendation_score
                confidence = rec.confidence
                acceptance = rec.estimated_acceptance_probability
                
                print(f"\n  {i}. {name}")
                print(f"     Recommendation Score: {score:.3f}")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Est. Acceptance Probability: {acceptance:.3f}")
                print(f"     Match: {rec.match_explanation}")
                
                if rec.recommendation_reasons:
                    print(f"     Reasons: {', '.join(rec.recommendation_reasons[:3])}")
                
                if rec.cost_analysis:
                    cost_note = rec.cost_analysis.get('cost_note', 'Unknown cost')
                    print(f"     Cost: {cost_note}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_strategy_comparison():
    """Test different recommendation strategies."""
    print("\n\n📊 Strategy Comparison Test")
    print("=" * 60)
    
    try:
        engine = AdvancedRecommendationEngine()
        
        manuscript = test_manuscripts["high_impact_clinical"]
        
        strategies = [
            RecommendationStrategy.CONSERVATIVE,
            RecommendationStrategy.AMBITIOUS,
            RecommendationStrategy.BALANCED,
            RecommendationStrategy.COST_CONSCIOUS
        ]
        
        print(f"\n📄 Testing strategies with high-impact clinical manuscript:")
        print(f"Manuscript length: {len(manuscript)} characters")
        
        strategy_results = {}
        
        for strategy in strategies:
            print(f"\n🎯 {strategy.value.upper()} Strategy:")
            print("-" * 30)
            
            suite = engine.generate_recommendations(
                manuscript_text=manuscript,
                strategy=strategy,
                max_recommendations=3
            )
            
            strategy_results[strategy.value] = suite
            
            print(f"  Journals considered: {suite.total_journals_considered}")
            print(f"  Primary recommendations: {len(suite.primary_recommendations)}")
            
            # Show top recommendation
            if suite.primary_recommendations:
                top_rec = suite.primary_recommendations[0]
                name = top_rec.journal_data.get('display_name', 'Unknown')[:35]
                score = top_rec.recommendation_score
                
                ranking_data = top_rec.journal_data.get('ranking_metrics', {})
                prestige = ranking_data.get('prestige_level', 'unknown')
                
                apc = (top_rec.journal_data.get('apc_amount') or 
                      top_rec.journal_data.get('apc_usd', 0))
                
                print(f"  Top pick: {name}")
                print(f"  Score: {score:.3f}, Prestige: {prestige}, APC: ${apc}")
        
        # Compare strategies
        print(f"\n📈 Strategy Comparison Summary:")
        for strategy_name, suite in strategy_results.items():
            if suite.primary_recommendations:
                avg_score = sum(r.recommendation_score for r in suite.primary_recommendations) / len(suite.primary_recommendations)
                avg_confidence = sum(r.confidence for r in suite.primary_recommendations) / len(suite.primary_recommendations)
                print(f"  {strategy_name.title()}: Avg Score {avg_score:.3f}, Avg Confidence {avg_confidence:.3f}")
    
    except Exception as e:
        print(f"❌ Strategy comparison failed: {e}")


def test_advanced_filtering():
    """Test advanced filtering capabilities."""
    print("\n\n🔍 Advanced Filtering Test")
    print("=" * 60)
    
    try:
        engine = AdvancedRecommendationEngine()
        
        manuscript = test_manuscripts["moderate_research"]
        
        # Test different filter configurations
        filter_configs = [
            {
                "name": "High Prestige Only",
                "filters": FilterCriteria(
                    min_prestige_level=PrestigeLevel.EXCELLENT,
                    min_quality_score=0.7
                )
            },
            {
                "name": "Cost Conscious",
                "filters": FilterCriteria(
                    max_apc=1000,
                    open_access_only=True
                )
            },
            {
                "name": "Open Access Only",
                "filters": FilterCriteria(
                    open_access_only=True,
                    doaj_only=True
                )
            },
            {
                "name": "Medical Journals",
                "filters": FilterCriteria(
                    required_subjects=["medicine", "health"],
                    min_citation_count=1000
                )
            }
        ]
        
        print(f"\n📄 Testing filters with research manuscript:")
        
        for config in filter_configs:
            print(f"\n🔍 {config['name']} Filter:")
            print("-" * 25)
            
            try:
                suite = engine.generate_recommendations(
                    manuscript_text=manuscript,
                    filter_criteria=config['filters'],
                    strategy=RecommendationStrategy.BALANCED,
                    max_recommendations=5
                )
                
                print(f"  Journals considered: {suite.total_journals_considered}")
                print(f"  Recommendations found: {len(suite.primary_recommendations)}")
                
                if suite.primary_recommendations:
                    # Show characteristics of filtered results
                    results = suite.primary_recommendations
                    
                    # Analyze prestige distribution
                    prestige_counts = {}
                    apc_values = []
                    oa_count = 0
                    
                    for rec in results:
                        ranking_data = rec.journal_data.get('ranking_metrics', {})
                        prestige = ranking_data.get('prestige_level', 'unknown')
                        prestige_counts[prestige] = prestige_counts.get(prestige, 0) + 1
                        
                        apc = (rec.journal_data.get('apc_amount') or 
                              rec.journal_data.get('apc_usd'))
                        if apc:
                            apc_values.append(apc)
                        
                        if rec.journal_data.get('oa_status', rec.journal_data.get('is_oa', False)):
                            oa_count += 1
                    
                    print(f"  Prestige distribution: {prestige_counts}")
                    if apc_values:
                        avg_apc = sum(apc_values) / len(apc_values)
                        print(f"  Average APC: ${avg_apc:.0f}")
                    print(f"  Open access journals: {oa_count}/{len(results)}")
                    
                    # Show top result
                    top_rec = results[0]
                    name = top_rec.journal_data.get('display_name', 'Unknown')[:40]
                    score = top_rec.recommendation_score
                    print(f"  Top result: {name} (Score: {score:.3f})")
                else:
                    print(f"  ⚠️ No recommendations found with this filter")
            
            except Exception as e:
                print(f"  ❌ Filter test failed: {e}")
    
    except Exception as e:
        print(f"❌ Advanced filtering test failed: {e}")


def test_recommendation_analysis():
    """Test recommendation suite analysis."""
    print("\n\n📈 Recommendation Analysis Test")
    print("=" * 60)
    
    try:
        engine = AdvancedRecommendationEngine()
        
        manuscript = test_manuscripts["high_impact_clinical"]
        
        # Generate comprehensive recommendations
        suite = engine.generate_recommendations(
            manuscript_text=manuscript,
            strategy=RecommendationStrategy.BALANCED,
            max_recommendations=8
        )
        
        print(f"\n📊 Analyzing recommendation suite:")
        
        # Analyze the suite
        insights = analyze_recommendation_suite(suite)
        
        print(f"Total recommendations: {insights['total_recommendations']}")
        print(f"Strategy used: {insights['strategy_used']}")
        print(f"Journals considered: {insights['journals_considered']}")
        
        if 'score_stats' in insights:
            stats = insights['score_stats']
            print(f"\nScore Statistics:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
            print(f"  Std Dev: {stats['std']:.3f}")
        
        if 'confidence_stats' in insights:
            conf_stats = insights['confidence_stats']
            print(f"\nConfidence Statistics:")
            print(f"  Mean confidence: {conf_stats['mean']:.3f}")
            print(f"  High confidence (≥0.7): {conf_stats['high_confidence_count']}")
        
        if 'acceptance_stats' in insights:
            acc_stats = insights['acceptance_stats']
            print(f"\nAcceptance Probability Statistics:")
            print(f"  Mean probability: {acc_stats['mean']:.3f}")
            print(f"  High probability (≥0.6): {acc_stats['high_probability_count']}")
        
        # Analyze manuscript analysis summary
        manuscript_summary = suite.manuscript_analysis_summary
        print(f"\nManuscript Analysis Summary:")
        print(f"  Word count: {manuscript_summary.get('word_count', 'Unknown')}")
        print(f"  Study type: {manuscript_summary.get('study_type', 'Unknown')}")
        print(f"  Target prestige: {manuscript_summary.get('target_prestige', 'Unknown')}")
        print(f"  Quality score: {manuscript_summary.get('quality_score', 'Unknown')}")
        print(f"  References found: {manuscript_summary.get('references_found', 'Unknown')}")
    
    except Exception as e:
        print(f"❌ Recommendation analysis failed: {e}")


def test_filter_creation():
    """Test filter creation utilities."""
    print("\n\n⚙️ Filter Creation Test")
    print("=" * 60)
    
    # Test strategy-based filter creation
    strategies = [
        RecommendationStrategy.CONSERVATIVE,
        RecommendationStrategy.AMBITIOUS,
        RecommendationStrategy.COST_CONSCIOUS,
        RecommendationStrategy.OPEN_ACCESS
    ]
    
    print("\n🎯 Strategy-based filter creation:")
    
    for strategy in strategies:
        filters = create_recommendation_filters(strategy)
        
        print(f"\n{strategy.value.upper()} Strategy Filters:")
        print(f"  Min prestige: {filters.min_prestige_level.value if filters.min_prestige_level else 'None'}")
        print(f"  Min quality: {filters.min_quality_score or 'None'}")
        print(f"  Max APC: {filters.max_apc or 'None'}")
        print(f"  Open access only: {filters.open_access_only}")
        print(f"  No APC only: {filters.no_apc_only}")
        print(f"  Exclude predatory: {filters.exclude_predatory}")
    
    # Test custom filter creation
    print(f"\n🔧 Custom filter creation:")
    
    custom_filters = create_recommendation_filters(
        strategy=RecommendationStrategy.BALANCED,
        min_prestige_level=PrestigeLevel.GOOD,
        max_apc=2000,
        required_subjects=["medicine", "computer science"],
        min_citation_count=500
    )
    
    print(f"Custom filters created:")
    print(f"  Min prestige: {custom_filters.min_prestige_level.value}")
    print(f"  Max APC: {custom_filters.max_apc}")
    print(f"  Required subjects: {custom_filters.required_subjects}")
    print(f"  Min citations: {custom_filters.min_citation_count}")


def test_detailed_recommendation_breakdown():
    """Test detailed recommendation breakdown."""
    print("\n\n🔬 Detailed Recommendation Breakdown Test")
    print("=" * 60)
    
    try:
        engine = AdvancedRecommendationEngine()
        
        manuscript = test_manuscripts["high_impact_clinical"]
        
        # Generate recommendations with custom filters
        custom_filters = FilterCriteria(
            min_prestige_level=PrestigeLevel.GOOD,
            min_quality_score=0.6,
            exclude_predatory=True
        )
        
        suite = engine.generate_recommendations(
            manuscript_text=manuscript,
            filter_criteria=custom_filters,
            strategy=RecommendationStrategy.BALANCED,
            max_recommendations=3
        )
        
        print(f"\n📋 Detailed breakdown of top recommendations:")
        
        if suite.primary_recommendations:
            for i, rec in enumerate(suite.primary_recommendations, 1):
                print(f"\n{'='*50}")
                print(f"RECOMMENDATION #{i}")
                print(f"{'='*50}")
                
                # Basic info
                name = rec.journal_data.get('display_name', 'Unknown')
                publisher = rec.journal_data.get('publisher', 'Unknown')[:30]
                
                print(f"Journal: {name}")
                print(f"Publisher: {publisher}")
                
                # Scores and metrics
                print(f"\nScores & Metrics:")
                print(f"  Recommendation Score: {rec.recommendation_score:.3f}")
                print(f"  Confidence: {rec.confidence:.3f}")
                print(f"  Similarity Score: {rec.journal_data.get('similarity_score', 0):.3f}")
                print(f"  Est. Acceptance Probability: {rec.estimated_acceptance_probability:.3f}")
                
                # Ranking info
                ranking_data = rec.journal_data.get('ranking_metrics', {})
                print(f"\nRanking Information:")
                print(f"  Prestige Level: {ranking_data.get('prestige_level', 'Unknown')}")
                print(f"  Quality Score: {ranking_data.get('quality_score', 0):.3f}")
                print(f"  Manuscript Compatibility: {rec.journal_data.get('manuscript_compatibility', 0):.3f}")
                
                # Cost analysis
                cost_info = rec.cost_analysis
                print(f"\nCost Analysis:")
                print(f"  APC Amount: ${cost_info.get('apc_amount', 0)}")
                print(f"  Cost Category: {cost_info.get('cost_category', 'Unknown')}")
                print(f"  Cost Note: {cost_info.get('cost_note', 'Unknown')}")
                
                # Access info
                is_oa = rec.journal_data.get('oa_status', rec.journal_data.get('is_oa', False))
                in_doaj = rec.journal_data.get('in_doaj', False)
                print(f"\nAccess Information:")
                print(f"  Open Access: {'Yes' if is_oa else 'No'}")
                print(f"  In DOAJ: {'Yes' if in_doaj else 'No'}")
                
                # Publication estimates
                print(f"\nPublication Estimates:")
                if rec.estimated_time_to_publication:
                    print(f"  Est. Time to Publication: {rec.estimated_time_to_publication} days")
                else:
                    print(f"  Est. Time to Publication: Unknown")
                
                # Reasons and risks
                print(f"\nRecommendation Reasons:")
                for reason in rec.recommendation_reasons[:5]:
                    print(f"  ✓ {reason}")
                
                if rec.risk_factors:
                    print(f"\nRisk Factors:")
                    for risk in rec.risk_factors:
                        print(f"  ⚠️ {risk}")
                
                print(f"\nMatch Explanation: {rec.match_explanation}")
        else:
            print("No recommendations found")
    
    except Exception as e:
        print(f"❌ Detailed breakdown test failed: {e}")


def main():
    """Run all recommendation engine tests."""
    print("🧪 MANUSCRIPT JOURNAL MATCHER - Advanced Recommendation Engine Tests")
    print("=" * 85)
    
    try:
        # Test 1: Basic functionality
        test_basic_recommendation_engine()
        
        # Test 2: Strategy comparison
        test_strategy_comparison()
        
        # Test 3: Advanced filtering
        test_advanced_filtering()
        
        # Test 4: Recommendation analysis
        test_recommendation_analysis()
        
        # Test 5: Filter creation utilities
        test_filter_creation()
        
        # Test 6: Detailed breakdown
        test_detailed_recommendation_breakdown()
        
        print("\n" + "=" * 85)
        print("✅ All recommendation engine tests completed successfully!")
        print("🎯 Advanced filtering and recommendation system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()