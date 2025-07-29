#!/usr/bin/env python3
"""
CSV Data Processor

Handles advanced processing, validation, and quality assurance
for Medicine Journal Rankings CSV data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

class CSVDataProcessor:
    """Advanced processing and validation for CSV journal data."""
    
    def __init__(self):
        """Initialize data processor."""
        self.validation_results = {
            'total_processed': 0,
            'validation_errors': [],
            'quality_warnings': [],
            'statistics': {}
        }
    
    def validate_journal_quality(self, journal: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate journal data quality and flag potential issues.
        
        Args:
            journal: Journal data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Critical validations (would exclude journal)
        if not journal.get('display_name'):
            issues.append("CRITICAL: Missing journal title")
        
        if not journal.get('issn') or len(journal.get('issn', [])) == 0:
            issues.append("CRITICAL: Missing ISSN")
        
        if journal.get('scimago_rank') and journal['scimago_rank'] <= 0:
            issues.append("CRITICAL: Invalid Scimago rank")
        
        # Quality warnings (journal included but flagged)
        if not journal.get('publisher'):
            issues.append("WARNING: Missing publisher information")
        
        if not journal.get('subjects') or len(journal.get('subjects', [])) == 0:
            issues.append("WARNING: No subject categories")
        
        if journal.get('works_count', 0) < 5:
            issues.append("WARNING: Very low publication count")
        
        if not journal.get('sjr_score') or journal.get('sjr_score', 0) <= 0:
            issues.append("WARNING: Missing or invalid SJR score")
        
        if not journal.get('h_index') or journal.get('h_index', 0) <= 0:
            issues.append("WARNING: Missing or invalid H-index")
        
        if not journal.get('country'):
            issues.append("WARNING: Missing country information")
            
        # Check for suspicious patterns
        title = journal.get('display_name', '').lower()
        if 'fake' in title or 'predatory' in title:
            issues.append("WARNING: Potentially problematic journal name")
            
        # Check for extreme values
        if journal.get('h_index', 0) > 2000:
            issues.append("WARNING: Extremely high H-index - verify data")
            
        if journal.get('sjr_score', 0) > 200:
            issues.append("WARNING: Extremely high SJR score - verify data")
        
        # Check for critical issues
        critical_issues = [issue for issue in issues if issue.startswith("CRITICAL")]
        is_valid = len(critical_issues) == 0
        
        return is_valid, issues
    
    def detect_duplicates(self, journals: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
        """
        Detect potential duplicate journals.
        
        Args:
            journals: List of journal dictionaries
            
        Returns:
            List of tuples indicating duplicate pairs (index1, index2, reason)
        """
        duplicates = []
        
        # Group by ISSN
        issn_groups = {}
        for i, journal in enumerate(journals):
            issns = journal.get('issn', [])
            for issn in issns:
                if issn not in issn_groups:
                    issn_groups[issn] = []
                issn_groups[issn].append(i)
        
        # Find ISSN duplicates
        for issn, indices in issn_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        duplicates.append((indices[i], indices[j], f"Same ISSN: {issn}"))
        
        # Group by title similarity
        title_groups = {}
        for i, journal in enumerate(journals):
            title = journal.get('display_name', '').lower().strip()
            # Simple normalization
            normalized_title = re.sub(r'[^\w\s]', '', title)
            normalized_title = re.sub(r'\s+', ' ', normalized_title)
            
            if normalized_title and len(normalized_title) > 3:
                if normalized_title not in title_groups:
                    title_groups[normalized_title] = []
                title_groups[normalized_title].append(i)
        
        # Find title duplicates
        for title, indices in title_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        # Avoid adding same pair twice
                        pair_issn = (indices[i], indices[j], f"Same title: {title[:50]}")
                        reverse_pair_issn = (indices[j], indices[i], f"Same title: {title[:50]}")
                        if pair_issn not in duplicates and reverse_pair_issn not in duplicates:
                            # Check if it's not already flagged by ISSN
                            already_flagged = any(
                                (d[0] == indices[i] and d[1] == indices[j]) or 
                                (d[0] == indices[j] and d[1] == indices[i])
                                for d in duplicates
                            )
                            if not already_flagged:
                                duplicates.append(pair_issn)
        
        logger.info(f"Found {len(duplicates)} potential duplicate pairs")
        return duplicates
    
    def generate_quality_report(self, journals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for journal data.
        
        Args:
            journals: List of processed journal dictionaries
            
        Returns:
            Quality report dictionary
        """
        report = {
            'total_journals': len(journals),
            'timestamp': datetime.now().isoformat(),
            'quality_metrics': {},
            'data_completeness': {},
            'distribution_analysis': {},
            'recommendations': []
        }
        
        if not journals:
            return report
        
        # Quality metrics
        valid_count = 0
        warning_count = 0
        critical_count = 0
        
        for journal in journals:
            is_valid, issues = self.validate_journal_quality(journal)
            if is_valid:
                valid_count += 1
            if any('WARNING' in issue for issue in issues):
                warning_count += 1
            if any('CRITICAL' in issue for issue in issues):
                critical_count += 1
        
        report['quality_metrics'] = {
            'valid_journals': valid_count,
            'journals_with_warnings': warning_count,
            'journals_with_critical_issues': critical_count,
            'quality_score': (valid_count / len(journals)) * 100 if journals else 0,
            'completeness_score': ((len(journals) - critical_count) / len(journals)) * 100 if journals else 0
        }
        
        # Data completeness
        fields_to_check = [
            'display_name', 'issn', 'publisher', 'subjects', 'scimago_rank',
            'sjr_score', 'h_index', 'works_count', 'country', 'sjr_quartile',
            'areas', 'coverage_years'
        ]
        
        completeness = {}
        for field in fields_to_check:
            complete_count = 0
            for j in journals:
                value = j.get(field)
                if value is not None and value != '' and value != []:
                    if isinstance(value, list) and len(value) > 0:
                        complete_count += 1
                    elif not isinstance(value, list):
                        complete_count += 1
            
            completeness[field] = {
                'count': complete_count,
                'percentage': (complete_count / len(journals)) * 100
            }
        
        report['data_completeness'] = completeness
        
        # Distribution analysis
        # Quartile distribution
        quartile_dist = Counter(j.get('sjr_quartile') for j in journals if j.get('sjr_quartile'))
        report['distribution_analysis']['quartile_distribution'] = dict(quartile_dist)
        
        # Country distribution (top 15)
        country_dist = Counter(j.get('country') for j in journals if j.get('country'))
        report['distribution_analysis']['top_countries'] = dict(country_dist.most_common(15))
        
        # Publisher distribution (top 15)
        publisher_dist = Counter(j.get('publisher') for j in journals if j.get('publisher'))
        report['distribution_analysis']['top_publishers'] = dict(publisher_dist.most_common(15))
        
        # Subject area distribution (top 10)
        all_subjects = []
        for j in journals:
            if j.get('subjects'):
                for subject in j['subjects']:
                    if subject.get('name'):
                        all_subjects.append(subject['name'])
        subject_dist = Counter(all_subjects)
        report['distribution_analysis']['top_subjects'] = dict(subject_dist.most_common(10))
        
        # Research area distribution
        all_areas = []
        for j in journals:
            if j.get('areas'):
                all_areas.extend(j['areas'])
        area_dist = Counter(all_areas)
        report['distribution_analysis']['top_research_areas'] = dict(area_dist.most_common(10))
        
        # Numerical statistics
        numerical_fields = ['scimago_rank', 'sjr_score', 'h_index', 'works_count', 'cited_by_count']
        for field in numerical_fields:
            values = [j.get(field) for j in journals if j.get(field) is not None and j.get(field) > 0]
            if values:
                report['distribution_analysis'][f'{field}_stats'] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        # Impact distribution
        impact_tiers = {'Very High (H>200)': 0, 'High (H>100)': 0, 'Medium (H>50)': 0, 'Low (H>20)': 0, 'Very Low (H<=20)': 0}
        for j in journals:
            h_index = j.get('h_index', 0)
            if h_index > 200:
                impact_tiers['Very High (H>200)'] += 1
            elif h_index > 100:
                impact_tiers['High (H>100)'] += 1
            elif h_index > 50:
                impact_tiers['Medium (H>50)'] += 1
            elif h_index > 20:
                impact_tiers['Low (H>20)'] += 1
            else:
                impact_tiers['Very Low (H<=20)'] += 1
        
        report['distribution_analysis']['impact_tiers'] = impact_tiers
        
        # Generate recommendations
        recommendations = []
        
        if report['quality_metrics']['quality_score'] < 90:
            recommendations.append("Consider additional data cleaning - quality score below 90%")
        
        if completeness['publisher']['percentage'] < 95:
            recommendations.append("Many journals missing publisher information")
        
        if completeness['subjects']['percentage'] < 95:
            recommendations.append("Many journals missing subject classifications")
        
        duplicate_count = len(self.detect_duplicates(journals))
        if duplicate_count > 0:
            recommendations.append(f"Found {duplicate_count} potential duplicate journals - review before processing")
        
        if impact_tiers['Very Low (H<=20)'] / len(journals) > 0.3:
            recommendations.append("High percentage of low-impact journals - consider filtering")
            
        if quartile_dist.get('Q4', 0) / len(journals) > 0.4:
            recommendations.append("High percentage of Q4 journals - consider quality thresholds")
        
        report['recommendations'] = recommendations
        
        return report
    
    def filter_high_quality_journals(self, journals: List[Dict[str, Any]], 
                                   min_works: int = 10, 
                                   min_h_index: int = 5,
                                   require_sjr: bool = True,
                                   max_rank: Optional[int] = None,
                                   allowed_quartiles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Filter journals based on quality criteria.
        
        Args:
            journals: List of journal dictionaries
            min_works: Minimum number of works published
            min_h_index: Minimum H-index
            require_sjr: Whether SJR score is required
            max_rank: Maximum acceptable Scimago rank
            allowed_quartiles: List of allowed quartiles (e.g., ['Q1', 'Q2'])
            
        Returns:
            Filtered list of high-quality journals
        """
        filtered = []
        
        for journal in journals:
            # Basic validation
            is_valid, _ = self.validate_journal_quality(journal)
            if not is_valid:
                continue
            
            # Quality filters
            if journal.get('works_count', 0) < min_works:
                continue
            
            if journal.get('h_index', 0) < min_h_index:
                continue
            
            if require_sjr and (not journal.get('sjr_score') or journal.get('sjr_score', 0) <= 0):
                continue
            
            if max_rank and journal.get('scimago_rank', float('inf')) > max_rank:
                continue
                
            if allowed_quartiles and journal.get('sjr_quartile') not in allowed_quartiles:
                continue
            
            filtered.append(journal)
        
        logger.info(f"Filtered {len(journals)} journals to {len(filtered)} high-quality journals")
        return filtered
    
    def optimize_for_embedding(self, journals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize journal data for embedding generation.
        
        Args:
            journals: List of journal dictionaries
            
        Returns:
            Optimized journal list
        """
        optimized = []
        
        for journal in journals:
            # Ensure semantic fingerprint exists
            if not journal.get('semantic_fingerprint'):
                logger.warning(f"Journal {journal.get('display_name', 'Unknown')} missing semantic fingerprint")
                continue
            
            # Create enhanced semantic fingerprint
            fingerprint = journal['semantic_fingerprint']
            
            # Add additional context for better embeddings
            enhancements = []
            
            # Add quartile information
            if journal.get('sjr_quartile'):
                enhancements.append(f"Journal quality: {journal['sjr_quartile']} quartile")
            
            # Add ranking context
            if journal.get('scimago_rank'):
                rank = journal['scimago_rank']
                if rank <= 10:
                    enhancements.append(f"Elite journal (rank {rank})")
                elif rank <= 50:
                    enhancements.append(f"Top tier journal (rank {rank})")
                elif rank <= 100:
                    enhancements.append(f"High ranked journal (rank {rank})")
                elif rank <= 500:
                    enhancements.append(f"Well ranked journal (rank {rank})")
            
            # Add citation context
            if journal.get('h_index'):
                h_index = journal['h_index']
                if h_index > 200:
                    enhancements.append(f"Highly cited journal (H-index: {h_index})")
                elif h_index > 100:
                    enhancements.append(f"Well cited journal (H-index: {h_index})")
                elif h_index > 50:
                    enhancements.append(f"Moderately cited journal (H-index: {h_index})")
            
            # Add publication activity context
            if journal.get('works_count'):
                works = journal['works_count']
                if works > 500:
                    enhancements.append("High publication volume")
                elif works > 100:
                    enhancements.append("Active publication")
            
            # Combine enhancements
            if enhancements:
                enhanced_fingerprint = fingerprint + " | " + " | ".join(enhancements)
                journal['semantic_fingerprint'] = enhanced_fingerprint
            
            optimized.append(journal)
        
        return optimized
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing operations."""
        return self.validation_results.copy()

# Usage Example
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    processor = CSVDataProcessor()
    
    # Example: Generate quality report
    sample_journals = [
        {
            'display_name': 'Test Medical Journal',
            'issn': ['1234-5678'],
            'publisher': 'Test Publisher',
            'scimago_rank': 1,
            'sjr_score': 10.5,
            'sjr_quartile': 'Q1',
            'h_index': 150,
            'works_count': 1000,
            'subjects': [{'name': 'Medicine'}],
            'country': 'United States',
            'areas': ['Medicine']
        },
        {
            'display_name': 'Lower Quality Journal',
            'issn': ['8765-4321'],
            'publisher': 'Small Publisher',
            'scimago_rank': 500,
            'sjr_score': 1.2,
            'sjr_quartile': 'Q3',
            'h_index': 25,
            'works_count': 50,
            'subjects': [{'name': 'Medicine'}],
            'country': 'Germany',
            'areas': ['Medicine']
        }
    ]
    
    report = processor.generate_quality_report(sample_journals)
    print(f"Quality report generated for {report['total_journals']} journals")
    print(f"Quality score: {report['quality_metrics']['quality_score']:.1f}%")
    print(f"Completeness score: {report['quality_metrics']['completeness_score']:.1f}%")
    
    # Test filtering
    high_quality = processor.filter_high_quality_journals(
        sample_journals, 
        min_works=100, 
        min_h_index=50,
        allowed_quartiles=['Q1', 'Q2']
    )
    print(f"High quality journals: {len(high_quality)}")
    
    # Test duplicate detection
    duplicates = processor.detect_duplicates(sample_journals)
    print(f"Duplicates found: {len(duplicates)}")