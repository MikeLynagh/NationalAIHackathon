"""
Test module for MPRN Data Parser
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_parser import MPRNDataParser, parse_mprn_file, validate_mprn_data

class TestMPRNDataParser:
    """Test cases for MPRNDataParser class."""
    
    def setup_method(self):
        """Set up test data."""
        self.parser = MPRNDataParser()
        self.sample_data = pd.DataFrame({
            'MPRN': ['10008632585'] * 4,
            'Meter Serial Number': ['000000000032791870'] * 4,
            'Read Value': [0.015, 0.000, 0.014, 0.000],
            'Read Type': ['Active Import Interval (kW)', 'Active Export Interval (kW)', 'Active Import Interval (kW)', 'Active Export Interval (kW)'],
            'Read Date and End Time': ['20-08-2025 03:30', '20-08-2025 03:30', '20-08-2025 03:00', '20-08-2025 03:00']
        })
    
    def test_validate_mprn_data_valid(self):
        """Test validation with valid data."""
        result = self.parser.validate_mprn_data(self.sample_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_mprn_data_missing_columns(self):
        """Test validation with missing columns."""
        invalid_data = self.sample_data.drop(columns=['MPRN'])
        result = self.parser.validate_mprn_data(invalid_data)
        
        assert result['is_valid'] is False
        assert 'Missing required columns' in str(result['errors'])
    
    def test_validate_mprn_data_invalid_read_type(self):
        """Test validation with invalid read type."""
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'Read Type'] = 'Invalid Type'
        result = self.parser.validate_mprn_data(invalid_data)
    
        # With our new flexible validation, this should still pass as long as we have Import data
        assert result['is_valid'] is True
        # The validation now only requires Import data, not Export data
    
    def test_validate_mprn_data_invalid_date_format(self):
        """Test validation with invalid date format."""
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'Read Date and End Time'] = 'invalid-date'
        result = self.parser.validate_mprn_data(invalid_data)
        
        assert result['is_valid'] is False
        assert 'Invalid date format' in str(result['errors'])
    
    def test_clean_and_resample(self):
        """Test data cleaning and resampling."""
        cleaned_data = self.parser.clean_and_resample(self.sample_data)
        
        # Check that we have the expected columns (now simplified to Import-only)
        expected_columns = ['timestamp', 'import_kw']
        for col in expected_columns:
            assert col in cleaned_data.columns
        
        # Check that data is sorted by timestamp
        timestamps = cleaned_data['timestamp']
        assert timestamps.is_monotonic_increasing
    
    def test_pivot_import_export(self):
        """Test import/export pivoting (now Import-only)."""
        # First clean the data to get proper datetime
        df_clean = self.sample_data.copy()
        df_clean['Read Date and End Time'] = pd.to_datetime(
            df_clean['Read Date and End Time'], format='%d-%m-%Y %H:%M'
        )
        
        pivoted = self.parser._pivot_import_export(df_clean)
        
        # Now we only expect Import data
        assert 'timestamp' in pivoted.columns
        assert 'import_kw' in pivoted.columns
        assert len(pivoted) > 0  # Should have Import records
    
    def test_resample_to_30min(self):
        """Test 30-minute resampling."""
        # Create data with the new structure
        df_pivot = pd.DataFrame({
            'timestamp': pd.date_range(
                '2025-08-20 03:00', 
                '2025-08-20 06:00', 
                freq='1H'
            ),
            'import_kw': [0.015, 0.016, 0.014, 0.015]
        })
        
        resampled = self.parser._resample_to_30min(df_pivot)
        
        # Should have more rows due to 30-minute intervals
        assert len(resampled) > len(df_pivot)
        # Check that intervals are 30 minutes
        time_diffs = resampled['timestamp'].diff().dropna()
        assert all(diff == pd.Timedelta(minutes=30) for diff in time_diffs)
    
    def test_fill_gaps(self):
        """Test gap filling functionality."""
        # Create data with the new structure
        df_resampled = pd.DataFrame({
            'timestamp': pd.date_range(
                '2025-08-20 03:00', 
                '2025-08-20 06:00', 
                freq='30T'
            ),
            'import_kw': [0.015, 0.016, 0.014, 0.015, 0.016, 0.014, 0.015]
        })
        
        # Introduce some NaN values
        df_resampled.loc[2:3, 'import_kw'] = None
        
        filled = self.parser._fill_gaps(df_resampled)
        
        # Should have no NaN values
        assert not filled['import_kw'].isna().any()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test data for convenience function tests."""
        self.sample_data = pd.DataFrame({
            'MPRN': ['10008632585'] * 4,
            'Meter Serial Number': ['000000000032791870'] * 4,
            'Read Value': [0.015, 0.000, 0.014, 0.000],
            'Read Type': ['Active Import Interval (kW)', 'Active Export Interval (kW)', 'Active Import Interval (kW)', 'Active Export Interval (kW)'],
            'Read Date and End Time': ['20-08-2025 03:30', '20-08-2025 03:30', '20-08-2025 03:00', '20-08-2025 03:00']
        })
    
    def test_parse_mprn_file(self):
        """Test parse_mprn_file convenience function."""
        # This test should work with our new structure
        try:
            result = parse_mprn_file('data/sample_mprn.csv')
            assert isinstance(result, pd.DataFrame)
            assert 'timestamp' in result.columns
            assert 'import_kw' in result.columns
        except Exception as e:
            pytest.fail(f"parse_mprn_file failed: {e}")
    
    def test_validate_mprn_data(self):
        """Test validate_mprn_data convenience function."""
        result = validate_mprn_data(self.sample_data)
        assert isinstance(result, dict)
        assert 'is_valid' in result
    
    def test_clean_and_resample(self):
        """Test clean_and_resample convenience function."""
        # This test should work with our new structure
        try:
            from data_parser import clean_and_resample
            result = clean_and_resample(self.sample_data)
            assert isinstance(result, pd.DataFrame)
            assert 'timestamp' in result.columns
            assert 'import_kw' in result.columns
        except Exception as e:
            pytest.fail(f"clean_and_resample failed: {e}") 