#!/usr/bin/env python3
"""
Test suite for CSV integration functionality.

Tests CSV parsing, data processing, schema mapping,
and integration with existing system.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import sys
from unittest.mock import Mock, patch
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from csv_journal_importer import CSVJournalImporter
    from csv_schema_mapper import CSVSchemaMapper
    from csv_data_processor import CSVDataProcessor
except ImportError as e:
    pytest.skip(f"CSV modules not available: {e}", allow_module_level=True)

class TestCSVJournalImporter:
    """Test CSV journal importer functionality."""
    
    def create_test_csv(self, content: str) -> str:
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(content)
            return f.name
    
    def test_csv_loading(self):
        """Test basic CSV loading functionality."""
        csv_content = """Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Total Docs. (2024);Total Docs. (3years);Total Refs.;Total Cites (3years);Citable Docs. (3years);Cites / Doc. (2years);Ref. / Doc.;%Female;Overton;SDG;Country;Region;Publisher;Coverage;Categories;Areas
1;28773;"Test Journal";journal;"1234-5678";10,5;Q1;100;50;150;1000;5000;100;20,5;30,2;45,5;5;10;United States;Northern America;"Test Publisher";"2000-2024";"Medicine (Q1)";"Medicine"
2;28774;"Another Journal";journal;"8765-4321";5,2;Q2;50;25;75;500;2500;50;15,3;25,1;55,2;3;8;United Kingdom;Western Europe;"Another Publisher";"1990-2024";"Biology (Q2)";"Life Sciences" """
        
        csv_file = self.create_test_csv(csv_content)
        importer = CSVJournalImporter(csv_file)
        
        try:
            df = importer.load_csv()
            
            assert len(df) == 2
            assert 'Title' in df.columns
            assert df.iloc[0]['Title'] == '"Test Journal"'
            assert df.iloc[1]['Title'] == '"Another Journal"'
            
        finally:
            Path(csv_file).unlink()  # Cleanup
    
    def test_data_cleaning(self):
        """Test data cleaning and validation."""
        csv_content = """Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Total Docs. (2024);Total Docs. (3years);Total Refs.;Total Cites (3years);Citable Docs. (3years);Cites / Doc. (2years);Ref. / Doc.;%Female;Overton;SDG;Country;Region;Publisher;Coverage;Categories;Areas
1;28773;"Test Journal";journal;"1234-5678";10,5;Q1;100;50;150;1000;5000;100;20,5;30,2;45,5;5;10;United States;Northern America;"Test Publisher";"2000-2024";"Medicine (Q1)";"Medicine"
2;28774;"Discontinued Journal (discontinued)";journal;"8765-4321";5,2;Q2;50;25;75;500;2500;50;15,3;25,1;55,2;3;8;United Kingdom;Western Europe;"Test Publisher";"1990-2024";"Biology (Q2)";"Life Sciences"
3;28775;"No ISSN Journal";journal;"";3,1;Q3;25;10;30;200;1000;20;10,2;20,5;40,1;2;5;Germany;Western Europe;"Test Publisher";"1980-2024";"Chemistry (Q3)";"Physical Sciences" """
        
        csv_file = self.create_test_csv(csv_content)
        importer = CSVJournalImporter(csv_file)
        
        try:
            importer.load_csv()
            cleaned_df = importer.clean_and_validate()
            
            # Should filter out discontinued and no-ISSN journals
            assert len(cleaned_df) == 1
            assert cleaned_df.iloc[0]['Title'] == '"Test Journal"'
            
            # Check numeric conversion
            assert isinstance(cleaned_df.iloc[0]['SJR'], float)
            assert cleaned_df.iloc[0]['SJR'] == 10.5  # Comma converted to dot
            
        finally:
            Path(csv_file).unlink()  # Cleanup
    
    def test_chunking(self):
        """Test data chunking functionality."""
        csv_content = """Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Total Docs. (2024);Total Docs. (3years);Total Refs.;Total Cites (3years);Citable Docs. (3years);Cites / Doc. (2years);Ref. / Doc.;%Female;Overton;SDG;Country;Region;Publisher;Coverage;Categories;Areas
1;1;"Journal 1";journal;"1111-1111";10,5;Q1;100;50;150;1000;5000;100;20,5;30,2;45,5;5;10;United States;Northern America;"Test Publisher";"2000-2024";"Medicine (Q1)";"Medicine"
2;2;"Journal 2";journal;"2222-2222";5,2;Q2;50;25;75;500;2500;50;15,3;25,1;55,2;3;8;United Kingdom;Western Europe;"Test Publisher";"1990-2024";"Biology (Q2)";"Life Sciences"
3;3;"Journal 3";journal;"3333-3333";3,1;Q3;25;10;30;200;1000;20;10,2;20,5;40,1;2;5;Germany;Western Europe;"Test Publisher";"1980-2024";"Chemistry (Q3)";"Physical Sciences"
4;4;"Journal 4";journal;"4444-4444";2,5;Q4;15;5;15;100;500;10;8,1;15,3;35,2;1;3;France;Western Europe;"Test Publisher";"1970-2024";"Physics (Q4)";"Physical Sciences"
5;5;"Journal 5";journal;"5555-5555";1,8;Q4;10;3;10;50;200;5;5,2;12,1;30,5;1;2;Italy;Western Europe;"Test Publisher";"1960-2024";"Mathematics (Q4)";"Mathematics" """
        
        csv_file = self.create_test_csv(csv_content)
        importer = CSVJournalImporter(csv_file)
        
        try:
            importer.load_csv()
            importer.clean_and_validate()
            chunks = importer.get_processed_chunks(chunk_size=2)
            
            assert len(chunks) == 3  # 5 journals in chunks of 2
            assert len(chunks[0]) == 2
            assert len(chunks[1]) == 2
            assert len(chunks[2]) == 1
            
        finally:
            Path(csv_file).unlink()  # Cleanup
    
    def test_statistics(self):
        """Test statistics generation."""
        csv_content = """Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Total Docs. (2024);Total Docs. (3years);Total Refs.;Total Cites (3years);Citable Docs. (3years);Cites / Doc. (2years);Ref. / Doc.;%Female;Overton;SDG;Country;Region;Publisher;Coverage;Categories;Areas
1;1;"Journal 1";journal;"1111-1111";10,5;Q1;100;50;150;1000;5000;100;20,5;30,2;45,5;5;10;United States;Northern America;"Test Publisher";"2000-2024";"Medicine (Q1)";"Medicine"
2;2;"Journal 2";journal;"2222-2222";5,2;Q1;50;25;75;500;2500;50;15,3;25,1;55,2;3;8;United Kingdom;Western Europe;"Test Publisher";"1990-2024";"Biology (Q1)";"Life Sciences"
3;3;"Journal 3";journal;"3333-3333";3,1;Q2;25;10;30;200;1000;20;10,2;20,5;40,1;2;5;Germany;Western Europe;"Test Publisher";"1980-2024";"Chemistry (Q2)";"Physical Sciences" """
        
        csv_file = self.create_test_csv(csv_content)
        importer = CSVJournalImporter(csv_file)
        
        try:
            importer.load_csv()
            importer.clean_and_validate()
            stats = importer.get_statistics()
            
            assert stats['total_journals'] == 3
            assert stats['q1_journals'] == 2
            assert stats['q2_journals'] == 1
            assert stats['top_100'] == 3
            assert stats['countries'] == 3
            assert stats['publishers'] == 1
            
        finally:
            Path(csv_file).unlink()  # Cleanup


class TestCSVSchemaMapper:
    """Test CSV to database schema mapping."""
    
    def test_basic_mapping(self):
        """Test basic journal mapping."""
        mapper = CSVSchemaMapper()
        
        csv_row = {
            'Rank': '1',
            'Sourceid': '28773',
            'Title': '"Test Journal"',
            'Issn': '"1234-5678, 8765-4321"',
            'SJR': '10.5',
            'SJR Best Quartile': 'Q1',
            'H index': '100',
            'Publisher': '"Test Publisher"',
            'Country': 'United States',
            'Categories': '"Medicine (Q1); Biology (Q2)"',
            'Areas': '"Medicine; Life Sciences"'
        }
        
        journal = mapper.map_journal(csv_row)
        
        assert journal is not None
        assert journal['display_name'] == 'Test Journal'
        assert journal['scimago_rank'] == 1
        assert journal['sjr_score'] == 10.5
        assert journal['sjr_quartile'] == 'Q1'
        assert len(journal['issn']) == 2
        assert journal['issn_l'] == '1234-5678'
        assert len(journal['subjects']) == 2
        assert len(journal['areas']) == 2
        assert journal['csv_source'] is True
    
    def test_issn_processing(self):
        """Test ISSN processing and formatting."""
        mapper = CSVSchemaMapper()
        
        # Test multiple ISSNs
        issn_data = mapper._process_issns('"12345678, 87654321"')
        assert len(issn_data['issn_list']) == 2
        assert issn_data['issn_list'][0] == '1234-5678'
        assert issn_data['issn_list'][1] == '8765-4321'
        assert issn_data['issn_l'] == '1234-5678'
        
        # Test empty ISSN
        issn_data = mapper._process_issns('')
        assert issn_data['issn_list'] == []
        assert issn_data['issn_l'] is None
        
        # Test single ISSN
        issn_data = mapper._process_issns('"12345678"')
        assert len(issn_data['issn_list']) == 1
        assert issn_data['issn_list'][0] == '1234-5678'
    
    def test_categories_processing(self):
        """Test categories processing with quartiles."""
        mapper = CSVSchemaMapper()
        
        categories = mapper._process_categories('"Medicine (Q1); Biology (Q2); Chemistry"')
        
        assert len(categories) == 3
        assert categories[0]['name'] == 'Medicine'
        assert categories[0]['quartile'] == 'Q1'
        assert categories[0]['score'] == 0.9
        
        assert categories[1]['name'] == 'Biology'
        assert categories[1]['quartile'] == 'Q2'
        assert categories[1]['score'] == 0.7
        
        assert categories[2]['name'] == 'Chemistry'
        assert categories[2]['quartile'] is None
        assert categories[2]['score'] == 0.5
    
    def test_semantic_fingerprint_creation(self):
        """Test semantic fingerprint creation."""
        mapper = CSVSchemaMapper()
        
        journal = {
            'display_name': 'Test Medical Journal',
            'publisher': 'Test Publisher',
            'scimago_rank': 1,
            'sjr_quartile': 'Q1',
            'subjects': [{'name': 'Medicine'}, {'name': 'Biology'}],
            'country': 'United States',
            'areas': ['Medicine', 'Life Sciences'],
            'h_index': 150,
            'sjr_score': 25.5,
            'works_count': 500
        }
        
        fingerprint = mapper.create_semantic_fingerprint(journal)
        
        assert 'Test Medical Journal' in fingerprint
        assert 'Test Publisher' in fingerprint
        assert 'Scimago Rank: 1' in fingerprint
        assert 'SJR Quartile: Q1' in fingerprint
        assert 'Medicine, Biology' in fingerprint
        assert 'H-index: 150' in fingerprint
        assert 'Top 10 journal' in fingerprint
        assert 'High impact journal' in fingerprint
    
    def test_country_code_mapping(self):
        """Test country code mapping."""
        mapper = CSVSchemaMapper()
        
        assert mapper._map_country_code('United States') == 'US'
        assert mapper._map_country_code('United Kingdom') == 'GB'
        assert mapper._map_country_code('Germany') == 'DE'
        assert mapper._map_country_code('Unknown Country') is None
    
    def test_safe_conversions(self):
        """Test safe type conversions."""
        mapper = CSVSchemaMapper()
        
        # Test safe_int
        assert mapper._safe_int('123') == 123
        assert mapper._safe_int('123.45') == 123
        assert mapper._safe_int('') is None
        assert mapper._safe_int('invalid') is None
        assert mapper._safe_int(None) is None
        
        # Test safe_float
        assert mapper._safe_float('123.45') == 123.45
        assert mapper._safe_float('123') == 123.0
        assert mapper._safe_float('') is None
        assert mapper._safe_float('invalid') is None
        assert mapper._safe_float(None) is None


class TestCSVDataProcessor:
    """Test CSV data processing and validation."""
    
    def test_journal_validation(self):
        """Test journal quality validation."""
        processor = CSVDataProcessor()
        
        # Valid journal
        valid_journal = {
            'display_name': 'Test Journal',
            'issn': ['1234-5678'],
            'publisher': 'Test Publisher',
            'scimago_rank': 1,
            'sjr_score': 10.5,
            'subjects': [{'name': 'Medicine'}],
            'works_count': 100,
            'h_index': 50,
            'country': 'United States'
        }
        
        is_valid, issues = processor.validate_journal_quality(valid_journal)
        assert is_valid
        assert len([i for i in issues if 'CRITICAL' in i]) == 0
        
        # Invalid journal (missing critical fields)
        invalid_journal = {
            'display_name': '',  # Missing title
            'issn': [],  # Missing ISSN
            'scimago_rank': -1  # Invalid rank
        }
        
        is_valid, issues = processor.validate_journal_quality(invalid_journal)
        assert not is_valid
        assert len([i for i in issues if 'CRITICAL' in i]) > 0
    
    def test_duplicate_detection(self):
        """Test duplicate journal detection."""
        processor = CSVDataProcessor()
        
        journals = [
            {
                'display_name': 'Test Journal',
                'issn': ['1234-5678']
            },
            {
                'display_name': 'Different Journal',
                'issn': ['8765-4321']
            },
            {
                'display_name': 'Test Journal',  # Same title
                'issn': ['9999-9999']
            },
            {
                'display_name': 'Another Journal',
                'issn': ['1234-5678']  # Same ISSN
            }
        ]
        
        duplicates = processor.detect_duplicates(journals)
        
        # Should find duplicates based on both title and ISSN
        assert len(duplicates) > 0
        
        # Check for specific duplicate types
        duplicate_reasons = [d[2] for d in duplicates]
        assert any('Same ISSN' in reason for reason in duplicate_reasons)
        assert any('Same title' in reason for reason in duplicate_reasons)
    
    def test_quality_report_generation(self):
        """Test quality report generation."""
        processor = CSVDataProcessor()
        
        journals = [
            {
                'display_name': 'Complete Journal',
                'issn': ['1234-5678'],
                'publisher': 'Publisher 1',
                'scimago_rank': 1,
                'sjr_score': 10.5,
                'sjr_quartile': 'Q1',
                'h_index': 100,
                'works_count': 500,
                'country': 'United States',
                'subjects': [{'name': 'Medicine'}],
                'areas': ['Medicine']
            },
            {
                'display_name': 'Incomplete Journal',
                'issn': ['8765-4321'],
                # Missing several fields
                'scimago_rank': 2,
                'sjr_quartile': 'Q2'
            }
        ]
        
        report = processor.generate_quality_report(journals)
        
        assert report['total_journals'] == 2
        assert 'quality_metrics' in report
        assert 'data_completeness' in report
        assert 'distribution_analysis' in report
        assert 'recommendations' in report
        
        # Check completeness metrics
        assert report['data_completeness']['display_name']['count'] == 2
        assert report['data_completeness']['publisher']['count'] == 1
        
        # Check quartile distribution
        assert 'Q1' in report['distribution_analysis']['quartile_distribution']
        assert report['distribution_analysis']['quartile_distribution']['Q1'] == 1
    
    def test_high_quality_filtering(self):
        """Test high-quality journal filtering."""
        processor = CSVDataProcessor()
        
        journals = [
            {
                'display_name': 'High Quality Journal',
                'issn': ['1234-5678'],
                'works_count': 1000,
                'h_index': 100,
                'sjr_score': 15.5,
                'scimago_rank': 50,
                'sjr_quartile': 'Q1'
            },
            {
                'display_name': 'Low Quality Journal',
                'issn': ['8765-4321'],
                'works_count': 5,  # Too low
                'h_index': 3,      # Too low
                'sjr_score': 0.5,
                'scimago_rank': 2000,
                'sjr_quartile': 'Q4'
            },
            {
                'display_name': 'Medium Quality Journal',
                'issn': ['9999-9999'],
                'works_count': 200,
                'h_index': 50,
                'sjr_score': 5.0,
                'scimago_rank': 500,
                'sjr_quartile': 'Q2'
            }
        ]
        
        # Test basic filtering
        filtered = processor.filter_high_quality_journals(
            journals, 
            min_works=50, 
            min_h_index=20, 
            require_sjr=True
        )
        
        # Should include high and medium quality journals
        assert len(filtered) == 2
        assert filtered[0]['display_name'] == 'High Quality Journal'
        assert filtered[1]['display_name'] == 'Medium Quality Journal'
        
        # Test quartile filtering
        filtered_q1 = processor.filter_high_quality_journals(
            journals,
            min_works=50,
            min_h_index=20,
            allowed_quartiles=['Q1']
        )
        
        # Should only include Q1 journal
        assert len(filtered_q1) == 1
        assert filtered_q1[0]['sjr_quartile'] == 'Q1'
    
    def test_embedding_optimization(self):
        """Test embedding optimization."""
        processor = CSVDataProcessor()
        
        journals = [
            {
                'display_name': 'Test Journal',
                'semantic_fingerprint': 'Original fingerprint',
                'sjr_quartile': 'Q1',
                'scimago_rank': 10,
                'h_index': 150,
                'works_count': 800
            },
            {
                'display_name': 'Another Journal',
                'semantic_fingerprint': 'Another fingerprint',
                'sjr_quartile': 'Q3',
                'scimago_rank': 1000,
                'h_index': 30,
                'works_count': 100
            }
        ]
        
        optimized = processor.optimize_for_embedding(journals)
        
        assert len(optimized) == 2
        
        # Check that fingerprints were enhanced
        assert 'Original fingerprint' in optimized[0]['semantic_fingerprint']
        assert 'Q1 quartile' in optimized[0]['semantic_fingerprint']
        assert 'Elite journal' in optimized[0]['semantic_fingerprint']
        assert 'Highly cited journal' in optimized[0]['semantic_fingerprint']
        
        assert 'Another fingerprint' in optimized[1]['semantic_fingerprint']
        assert 'Q3 quartile' in optimized[1]['semantic_fingerprint']


@pytest.mark.integration
class TestCSVIntegrationWorkflow:
    """Integration tests for complete CSV workflow."""
    
    def create_full_test_csv(self) -> str:
        """Create a complete test CSV with realistic data."""
        csv_content = """Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Total Docs. (2024);Total Docs. (3years);Total Refs.;Total Cites (3years);Citable Docs. (3years);Cites / Doc. (2years);Ref. / Doc.;%Female;Overton;SDG;Country;Region;Publisher;Coverage;Categories;Areas
1;28773;"Ca-A Cancer Journal for Clinicians";journal;"15424863, 00079235";145,004;Q1;223;43;122;2704;40834;81;168,71;62,88;48,21;4;37;United States;Northern America;"John Wiley and Sons Inc";"1950-2025";"Hematology (Q1); Oncology (Q1)";"Medicine"
2;19434;"Nature Reviews Drug Discovery";journal;"14741784, 14741776";30,506;Q1;412;247;718;8808;14603;136;16,64;35,66;26,67;1;58;United Kingdom;Western Europe;"Nature Research";"2002-2025";"Drug Discovery (Q1); Medicine (miscellaneous) (Q1); Pharmacology (Q1)";"Medicine; Pharmacology, Toxicology and Pharmaceutics"
3;20425;"Journal of Clinical Medicine";journal;"20776409";2,998;Q2;98;3456;9234;45678;23456;2345;2,54;13,21;42,15;45;89;Switzerland;Western Europe;"MDPI AG";"2012-2025";"Medicine (miscellaneous) (Q2)";"Medicine" """
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            return f.name
    
    def test_complete_workflow(self):
        """Test complete CSV processing workflow."""
        csv_file = self.create_full_test_csv()
        
        try:
            # Step 1: Import CSV
            importer = CSVJournalImporter(csv_file)
            importer.load_csv()
            importer.clean_and_validate()
            
            # Step 2: Get chunks
            chunks = importer.get_processed_chunks(chunk_size=10)
            assert len(chunks) == 1
            
            # Step 3: Map to schema
            mapper = CSVSchemaMapper()
            mapped_journals = []
            
            for chunk in chunks:
                for _, row in chunk.iterrows():
                    journal = mapper.map_journal(row.to_dict())
                    if journal:
                        fingerprint = mapper.create_semantic_fingerprint(journal)
                        journal['semantic_fingerprint'] = fingerprint
                        mapped_journals.append(journal)
            
            assert len(mapped_journals) == 3
            
            # Verify journal data
            first_journal = mapped_journals[0]
            assert first_journal['display_name'] == 'Ca-A Cancer Journal for Clinicians'
            assert first_journal['scimago_rank'] == 1
            assert first_journal['sjr_quartile'] == 'Q1'
            assert len(first_journal['issn']) == 2
            assert first_journal['issn'][0] == '1542-4863'
            
            # Step 4: Process and validate
            processor = CSVDataProcessor()
            report = processor.generate_quality_report(mapped_journals)
            
            assert report['total_journals'] == 3
            assert report['quality_metrics']['quality_score'] > 80
            
            # Step 5: Optimize for embeddings
            optimized = processor.optimize_for_embedding(mapped_journals)
            assert len(optimized) == 3
            
            # Verify semantic fingerprints are enhanced
            for journal in optimized:
                assert 'semantic_fingerprint' in journal
                assert len(journal['semantic_fingerprint']) > 0
                if journal.get('sjr_quartile') == 'Q1':
                    assert 'Q1 quartile' in journal['semantic_fingerprint']
        
        finally:
            Path(csv_file).unlink()  # Cleanup
    
    def test_quality_filtering_workflow(self):
        """Test workflow with quality filtering."""
        csv_file = self.create_full_test_csv()
        
        try:
            # Complete workflow with quality filtering
            importer = CSVJournalImporter(csv_file)
            importer.load_csv()
            importer.clean_and_validate()
            
            mapper = CSVSchemaMapper()
            processor = CSVDataProcessor()
            
            # Map all journals
            all_journals = []
            chunks = importer.get_processed_chunks()
            
            for chunk in chunks:
                for _, row in chunk.iterrows():
                    journal = mapper.map_journal(row.to_dict())
                    if journal:
                        fingerprint = mapper.create_semantic_fingerprint(journal)
                        journal['semantic_fingerprint'] = fingerprint
                        all_journals.append(journal)
            
            # Apply quality filtering
            high_quality = processor.filter_high_quality_journals(
                all_journals,
                min_works=100,
                min_h_index=90,
                allowed_quartiles=['Q1', 'Q2']
            )
            
            # Should filter to high quality journals only
            assert len(high_quality) <= len(all_journals)
            
            # Verify all filtered journals meet criteria
            for journal in high_quality:
                assert journal.get('works_count', 0) >= 100
                assert journal.get('h_index', 0) >= 90
                assert journal.get('sjr_quartile') in ['Q1', 'Q2']
        
        finally:
            Path(csv_file).unlink()  # Cleanup


@pytest.mark.slow
class TestCSVPerformance:
    """Performance tests for CSV processing."""
    
    def create_large_test_csv(self, num_journals: int = 100) -> str:
        """Create a large test CSV for performance testing."""
        header = "Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Total Docs. (2024);Total Docs. (3years);Total Refs.;Total Cites (3years);Citable Docs. (3years);Cites / Doc. (2years);Ref. / Doc.;%Female;Overton;SDG;Country;Region;Publisher;Coverage;Categories;Areas\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(header)
            
            for i in range(1, num_journals + 1):
                quartile = ['Q1', 'Q2', 'Q3', 'Q4'][i % 4]
                row = f'{i};{i};"Test Journal {i}";journal;"{1000+i:04d}-{2000+i:04d}";{10.0-i*0.1:.3f};{quartile};{100-i};{50+i};{150+i*2};{1000+i*10};{5000+i*50};{100+i};{20.0+i*0.1:.2f};{30.0+i*0.2:.2f};{45.0+i*0.3:.2f};{i%10};{i%20};United States;Northern America;"Test Publisher {i%5}";"2000-2024";"Medicine ({quartile})";"Medicine"\n'
                f.write(row)
            
            return f.name
    
    def test_large_csv_processing(self):
        """Test processing of larger CSV files."""
        csv_file = self.create_large_test_csv(100)
        
        try:
            import time
            
            start_time = time.time()
            
            # Process CSV
            importer = CSVJournalImporter(csv_file)
            importer.load_csv()
            importer.clean_and_validate()
            
            load_time = time.time() - start_time
            
            # Map to schema
            mapper = CSVSchemaMapper()
            mapped_journals = []
            
            chunks = importer.get_processed_chunks(chunk_size=25)
            
            for chunk in chunks:
                for _, row in chunk.iterrows():
                    journal = mapper.map_journal(row.to_dict())
                    if journal:
                        mapped_journals.append(journal)
            
            mapping_time = time.time() - start_time - load_time
            
            # Validate performance
            assert len(mapped_journals) == 100
            assert load_time < 5.0  # Should load within 5 seconds
            assert mapping_time < 10.0  # Should map within 10 seconds
            
            # Test memory efficiency
            processor = CSVDataProcessor()
            report = processor.generate_quality_report(mapped_journals)
            
            assert report['total_journals'] == 100
        
        finally:
            Path(csv_file).unlink()  # Cleanup


if __name__ == '__main__':
    pytest.main([__file__, '-v'])