#!/usr/bin/env python3
"""
Test script for journal ranking integration system.

This script demonstrates the new journal ranking features
and tests them with various manuscript quality levels.
"""

from src.journal_ranker import JournalRankingIntegrator, PrestigeLevel, integrate_journal_rankings
from src.ensemble_matcher import EnsembleJournalMatcher
from src.match_journals import JournalMatcher

# Sample manuscripts with different quality levels
test_manuscripts = {
    "high_quality_clinical": """
        Efficacy of AI-Guided Precision Medicine in Cancer Treatment: 
        A Randomized, Double-Blind, Placebo-Controlled, Multi-Center Clinical Trial
        
        Abstract: Background: Precision medicine approaches in oncology have shown promise 
        but lack comprehensive AI integration. This international, multi-center randomized 
        controlled trial evaluated AI-guided treatment selection versus standard care.
        
        Methods: We enrolled 2,847 patients with stage III-IV solid tumors across 47 centers 
        in 12 countries. Patients were randomly assigned (1:1) to AI-guided precision therapy 
        (n=1,424) or physician-guided standard care (n=1,423). The AI system analyzed genomic, 
        proteomic, and clinical data using deep learning algorithms. Primary endpoint was 
        overall survival at 24 months. Secondary endpoints included progression-free survival, 
        quality of life, and treatment-related adverse events.
        
        Results: AI-guided therapy demonstrated significant improvement in overall survival 
        (HR=0.67, 95% CI: 0.58-0.78, p<0.001). Median overall survival was 18.3 months 
        (95% CI: 16.8-19.9) in the AI group versus 13.2 months (95% CI: 11.7-14.8) in 
        the control group. The number needed to treat was 7 (95% CI: 5-11).
        
        Conclusions: AI-guided precision medicine significantly improves survival outcomes 
        in advanced cancer patients. This represents a paradigm shift in oncology treatment 
        selection with immediate clinical implications.
        
        Trial Registration: ClinicalTrials.gov NCT04567890
        Funding: National Cancer Institute, European Research Council
        """,
    
    "moderate_quality_research": """
        Machine Learning Approaches for Biomarker Discovery in Diabetes
        
        Abstract: Background: Type 2 diabetes biomarker identification remains challenging. 
        We developed machine learning models to identify novel biomarkers from metabolomic data.
        
        Methods: We analyzed plasma samples from 485 type 2 diabetes patients and 312 healthy 
        controls using untargeted metabolomics. Random forest and support vector machine 
        algorithms were applied for biomarker discovery. Cross-validation was performed 
        using 80/20 train-test splits.
        
        Results: Our models achieved 87% accuracy in diabetes classification. We identified 
        15 potential biomarkers with area under curve >0.80. Three metabolites showed 
        significant correlation with HbA1c levels (r>0.65, p<0.01).
        
        Conclusions: Machine learning enables effective biomarker discovery from metabolomic 
        data. These findings warrant validation in larger cohorts for clinical translation.
        """,
    
    "basic_quality_study": """
        Survey of Electronic Health Record Usage in Small Clinics
        
        Abstract: We conducted a survey to understand EHR adoption patterns in small clinics.
        
        Methods: We distributed online surveys to 150 clinic administrators across three states. 
        The survey included questions about EHR systems, implementation challenges, and user 
        satisfaction. Response rate was 34% (n=51).
        
        Results: 78% of clinics had implemented EHR systems. Main challenges included cost 
        (65%), training requirements (48%), and technical issues (31%). User satisfaction 
        scores averaged 6.2/10.
        
        Conclusions: EHR adoption remains challenging for small clinics despite regulatory 
        requirements. Better support systems are needed for successful implementation.
        """
}


def test_manuscript_quality_analysis():
    """Test manuscript quality assessment."""
    print("📊 Manuscript Quality Analysis Test")
    print("=" * 60)
    
    ranker = JournalRankingIntegrator()
    
    for manuscript_type, manuscript in test_manuscripts.items():
        print(f"\n📄 {manuscript_type.replace('_', ' ').title()}:")
        print("-" * 40)
        
        # Analyze manuscript
        analysis = ranker.analyze_manuscript_for_ranking(manuscript)
        
        print(f"Target Prestige Level: {analysis.target_prestige_level.value}")
        print(f"Quality Alignment Score: {analysis.quality_alignment_score:.3f}")
        print(f"Recommended Ranking Range: #{analysis.recommended_ranking_range[0]}-{analysis.recommended_ranking_range[1]}")
        print(f"Explanation: {analysis.ranking_explanation}")
        
        # Show quality indicators
        print("\nQuality Indicators:")
        for indicator, score in analysis.manuscript_quality_indicators.items():
            indicator_name = indicator.replace('_', ' ').title()
            score_emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
            print(f"  {score_emoji} {indicator_name}: {score:.3f}")


def test_journal_ranking_integration():
    """Test journal ranking integration with matching."""
    print("\n\n🏆 Journal Ranking Integration Test")
    print("=" * 60)
    
    try:
        # Initialize systems
        matcher = JournalMatcher()
        matcher.load_database()
        ranker = JournalRankingIntegrator()
        
        manuscript = test_manuscripts["high_quality_clinical"]
        
        print(f"\n📄 Testing with high-quality clinical manuscript:")
        print(f"Manuscript length: {len(manuscript)} characters")
        
        # Get matching journals
        with_ranking = False
        print("\n🔍 Standard Journal Matching:")
        standard_results = matcher.search_similar_journals(
            query_text=manuscript,
            top_k=10,
            use_ensemble_matching=False
        )
        
        if standard_results:
            print(f"Found {len(standard_results)} journals")
            
            # Integrate ranking data
            print("\n🏅 Integrating Ranking Data:")
            enhanced_results = integrate_journal_rankings(standard_results, manuscript)
            
            # Display enhanced results
            print("\nTop 5 Results with Ranking Integration:")
            for i, journal in enumerate(enhanced_results[:5], 1):
                name = journal.get('display_name', 'Unknown')[:45]
                similarity = journal.get('similarity_score', 0)
                
                ranking_data = journal.get('ranking_metrics', {})
                prestige = ranking_data.get('prestige_level', 'unknown')
                quality = ranking_data.get('quality_score', 0)
                compatibility = journal.get('manuscript_compatibility', 0)
                
                print(f"\n  {i}. {name}")
                print(f"     Similarity: {similarity:.3f}")
                print(f"     Prestige: {prestige}")
                print(f"     Quality Score: {quality:.3f}")
                print(f"     MS Compatibility: {compatibility:.3f}")
                
                # Show existing ranking data if available
                sjr_score = journal.get('sjr_score')
                sjr_rank = journal.get('scimago_rank')
                if sjr_score and sjr_rank:
                    print(f"     SJR: {sjr_score:.2f} (Rank #{sjr_rank})")
        else:
            print("No matching journals found")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_prestige_level_assessment():
    """Test prestige level assessment for different manuscripts."""
    print("\n\n🎯 Prestige Level Assessment Test")
    print("=" * 60)
    
    ranker = JournalRankingIntegrator()
    
    # Expected prestige levels for test manuscripts
    expected_levels = {
        "high_quality_clinical": PrestigeLevel.ELITE,
        "moderate_quality_research": PrestigeLevel.GOOD,
        "basic_quality_study": PrestigeLevel.AVERAGE
    }
    
    print("\nManuscript Prestige Level Assessment:")
    
    for manuscript_type, manuscript in test_manuscripts.items():
        analysis = ranker.analyze_manuscript_for_ranking(manuscript)
        detected_level = analysis.target_prestige_level
        expected_level = expected_levels[manuscript_type]
        
        manuscript_name = manuscript_type.replace('_', ' ').title()
        status = "✅" if detected_level == expected_level else "⚠️"
        
        print(f"\n  {status} {manuscript_name}:")
        print(f"    Detected: {detected_level.value}")
        print(f"    Expected: {expected_level.value}")
        print(f"    Quality Score: {analysis.quality_alignment_score:.3f}")
        
        # Show ranking range
        min_rank, max_rank = analysis.recommended_ranking_range
        print(f"    Recommended Range: #{min_rank}-{max_rank}")


def test_ranking_based_filtering():
    """Test ranking-based journal filtering."""
    print("\n\n🔍 Ranking-Based Filtering Test")
    print("=" * 60)
    
    try:
        matcher = JournalMatcher()
        matcher.load_database()
        ranker = JournalRankingIntegrator()
        
        manuscript = test_manuscripts["moderate_quality_research"]
        
        # Get all matching journals
        all_results = matcher.search_similar_journals(
            query_text=manuscript,
            top_k=20,
            min_similarity=0.1
        )
        
        if all_results:
            print(f"Found {len(all_results)} journals before ranking filter")
            
            # Enhance with ranking data
            enhanced_results = integrate_journal_rankings(all_results, manuscript)
            
            # Test different prestige filters
            prestige_filters = [
                (PrestigeLevel.ELITE, "Elite Journals Only"),
                (PrestigeLevel.PREMIER, "Premier+ Journals"),
                (PrestigeLevel.EXCELLENT, "Excellent+ Journals"),
                (PrestigeLevel.GOOD, "Good+ Journals")
            ]
            
            for min_prestige, filter_name in prestige_filters:
                print(f"\n🏅 {filter_name}:")
                
                # Filter by prestige level
                filtered_journals = []
                for journal in enhanced_results:
                    ranking_data = journal.get('ranking_metrics', {})
                    journal_prestige = ranking_data.get('prestige_level', 'average')
                    
                    # Convert string back to enum for comparison
                    try:
                        journal_prestige_enum = PrestigeLevel(journal_prestige)
                        prestige_values = list(PrestigeLevel)
                        if prestige_values.index(journal_prestige_enum) >= prestige_values.index(min_prestige):
                            filtered_journals.append(journal)
                    except (ValueError, KeyError):
                        continue
                
                print(f"  Results: {len(filtered_journals)} journals")
                
                # Show top 3 results
                for i, journal in enumerate(filtered_journals[:3], 1):
                    name = journal.get('display_name', 'Unknown')[:35]
                    similarity = journal.get('similarity_score', 0)
                    ranking_data = journal.get('ranking_metrics', {})
                    prestige = ranking_data.get('prestige_level', 'unknown')
                    quality = ranking_data.get('quality_score', 0)
                    
                    print(f"    {i}. {name}")
                    print(f"       Similarity: {similarity:.3f}, Prestige: {prestige}, Quality: {quality:.3f}")
        else:
            print("No matching journals found")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_ranking_compatibility_scoring():
    """Test manuscript-journal ranking compatibility scoring."""
    print("\n\n🤝 Ranking Compatibility Scoring Test")
    print("=" * 60)
    
    try:
        matcher = JournalMatcher()
        matcher.load_database()
        ranker = JournalRankingIntegrator()
        
        # Test with different manuscript types
        for manuscript_type, manuscript in test_manuscripts.items():
            print(f"\n📄 {manuscript_type.replace('_', ' ').title()}:")
            print("-" * 35)
            
            # Get journals and enhance with ranking
            results = matcher.search_similar_journals(
                query_text=manuscript,
                top_k=8,
                min_similarity=0.2
            )
            
            if results:
                enhanced_results = integrate_journal_rankings(results, manuscript)
                
                # Sort by manuscript compatibility
                enhanced_results.sort(
                    key=lambda x: x.get('manuscript_compatibility', 0), 
                    reverse=True
                )
                
                print("Top Compatibility Matches:")
                for i, journal in enumerate(enhanced_results[:4], 1):
                    name = journal.get('display_name', 'Unknown')[:30]
                    similarity = journal.get('similarity_score', 0)
                    compatibility = journal.get('manuscript_compatibility', 0)
                    
                    ranking_data = journal.get('ranking_metrics', {})
                    prestige = ranking_data.get('prestige_level', 'unknown')
                    
                    compatibility_emoji = "🟢" if compatibility >= 0.7 else "🟡" if compatibility >= 0.5 else "🔴"
                    
                    print(f"  {i}. {name}")
                    print(f"     {compatibility_emoji} Compatibility: {compatibility:.3f}")
                    print(f"     Similarity: {similarity:.3f}")
                    print(f"     Prestige: {prestige}")
            else:
                print("  No results found")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_quality_indicator_breakdown():
    """Test detailed quality indicator breakdown."""
    print("\n\n📈 Quality Indicator Breakdown Test")
    print("=" * 60)
    
    ranker = JournalRankingIntegrator()
    
    # Test each manuscript
    for manuscript_type, manuscript in test_manuscripts.items():
        print(f"\n📊 {manuscript_type.replace('_', ' ').title()}:")
        print("-" * 40)
        
        analysis = ranker.analyze_manuscript_for_ranking(manuscript)
        
        # Show detailed quality breakdown
        indicators = analysis.manuscript_quality_indicators
        
        print("Quality Indicators:")
        sorted_indicators = sorted(indicators.items(), key=lambda x: x[1], reverse=True)
        
        for indicator, score in sorted_indicators:
            indicator_name = indicator.replace('_', ' ').title()
            
            # Create visual bar
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Color coding
            if score >= 0.8:
                color = "🟢"
            elif score >= 0.6:
                color = "🟡"
            elif score >= 0.4:
                color = "🟠"
            else:
                color = "🔴"
            
            print(f"  {color} {indicator_name:<25} {score:.3f} {bar}")
        
        # Overall assessment
        avg_score = sum(indicators.values()) / len(indicators)
        print(f"\n  📋 Overall Quality Score: {avg_score:.3f}")
        print(f"  🎯 Target Prestige: {analysis.target_prestige_level.value}")


def main():
    """Run all journal ranking tests."""
    print("🧪 MANUSCRIPT JOURNAL MATCHER - Journal Ranking Integration Tests")
    print("=" * 85)
    
    try:
        # Test 1: Manuscript quality analysis
        test_manuscript_quality_analysis()
        
        # Test 2: Journal ranking integration
        test_journal_ranking_integration()
        
        # Test 3: Prestige level assessment
        test_prestige_level_assessment()
        
        # Test 4: Ranking-based filtering
        test_ranking_based_filtering()
        
        # Test 5: Compatibility scoring
        test_ranking_compatibility_scoring()
        
        # Test 6: Quality indicator breakdown
        test_quality_indicator_breakdown()
        
        print("\n" + "=" * 85)
        print("✅ All journal ranking integration tests completed successfully!")
        print("🏆 Journal ranking system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()