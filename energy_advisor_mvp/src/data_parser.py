"""
MPRN Data Parser Module

This module handles parsing, validation, and cleaning of Irish MPRN smart meter data.
Supports both Active Import and Active Export readings in 30-minute intervals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MPRNDataParser:
    """Parser for Irish MPRN smart meter data files."""
    
    def __init__(self):
        self.expected_columns = [
            'MPRN', 'Meter Serial Number', 'Read Value', 
            'Read Type', 'Read Date and End Time'
        ]
        self.expected_read_types = [
            'Active Import Interval (kW)'  # Only care about Import data
        ]
    
    def parse_mprn_file(self, uploaded_file) -> pd.DataFrame:
        """
        Parse uploaded MPRN CSV file into a structured DataFrame.
        
        Args:
            uploaded_file: Streamlit uploaded file object or file path
            
        Returns:
            pd.DataFrame: Parsed and structured MPRN data
        """
        try:
            # Read CSV file
            if hasattr(uploaded_file, 'read'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            logger.info(f"Successfully loaded {len(df)} rows from MPRN file")
            
            # Validate basic structure
            validation_result = self.validate_mprn_data(df)
            if not validation_result['is_valid']:
                raise ValueError(f"Data validation failed: {validation_result['errors']}")
            
            # Clean and structure data
            cleaned_df = self.clean_and_resample(df)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error parsing MPRN file: {str(e)}")
            raise
    
    def validate_mprn_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate MPRN data structure and content.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            Dict: Validation results with success status and error details
        """
        errors = []
        
        # Check required columns
        missing_columns = set(self.expected_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if 'Read Value' in df.columns:
            try:
                df['Read Value'] = pd.to_numeric(df['Read Value'], errors='coerce')
                if df['Read Value'].isna().any():
                    errors.append("Non-numeric values found in 'Read Value' column")
            except:
                errors.append("Cannot convert 'Read Value' to numeric")
        
        # Check read types - be more flexible, just ensure we have Import data
        if 'Read Type' in df.columns:
            read_types = set(df['Read Type'].unique())
            if 'Active Import Interval (kW)' not in read_types:
                errors.append("No Import data found - 'Active Import Interval (kW)' is required")
            # Don't reject Export data, we'll just ignore it later
        
        # Check date format
        if 'Read Date and End Time' in df.columns:
            try:
                # Parse dates with dayfirst=True for DD-MM-YYYY format
                df['Read Date and End Time'] = pd.to_datetime(
                    df['Read Date and End Time'], dayfirst=True, errors='coerce'
                )
                if df['Read Date and End Time'].isna().any():
                    errors.append("Invalid date format in 'Read Date and End Time' column")
            except:
                errors.append("Cannot parse 'Read Date and End Time' column")
        
        # Check for outliers
        if 'Read Value' in df.columns and df['Read Value'].notna().any():
            values = df['Read Value'].dropna()
            q99 = values.quantile(0.99)
            outliers = values[values > q99 * 2]  # Values > 2x 99th percentile
            if len(outliers) > 0:
                errors.append(f"Found {len(outliers)} extreme outliers (>2x 99th percentile)")
        
        # Check data completeness
        if len(df) == 0:
            errors.append("File contains no data")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'row_count': len(df),
            'date_range': self._get_date_range(df) if 'Read Date and End Time' in df.columns else None
        }
    
    def clean_and_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and resample MPRN data to ensure 30-minute intervals.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            pd.DataFrame: Cleaned and resampled data
        """
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Ensure datetime column is properly formatted
        df_clean['Read Date and End Time'] = pd.to_datetime(
            df_clean['Read Date and End Time'], dayfirst=True
        )
        
        # Sort by timestamp
        df_clean = df_clean.sort_values('Read Date and End Time')
        
        # Pivot data to separate import/export columns
        df_pivot = self._pivot_import_export(df_clean)
        
        # Handle missing intervals
        df_resampled = self._resample_to_30min(df_pivot)
        
        # Fill small gaps and flag large ones
        df_filled = self._fill_gaps(df_resampled)
        
        logger.info(f"Cleaned data: {len(df_filled)} rows, {len(df_filled.columns)} columns")
        
        return df_filled
    
    def _pivot_import_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process only Import data, ignoring Export data."""
        # Filter to only Import data
        import_df = df[df['Read Type'] == 'Active Import Interval (kW)'].copy()
        
        if import_df.empty:
            raise ValueError("No Import data found in the file")
        
        # Create clean DataFrame with just Import data
        clean_df = pd.DataFrame({
            'timestamp': import_df['Read Date and End Time'],
            'import_kw': import_df['Read Value']
        })
        
        # Ensure timestamp is datetime
        clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'], dayfirst=True)
        
        # Sort by timestamp
        clean_df = clean_df.sort_values('timestamp')
        
        logger.info(f"Processed {len(clean_df)} Import records")
        
        return clean_df
    
    def _resample_to_30min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to ensure 30-minute intervals."""
        # Set timestamp as index
        df_indexed = df.set_index('timestamp')
        
        # Resample to 30-minute intervals
        df_resampled = df_indexed.resample('30T').asfreq()
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill small gaps and flag large ones."""
        # Forward fill small gaps (up to 2 hours)
        df_filled = df.copy()
        
        # Calculate time differences
        time_diffs = df_filled['timestamp'].diff()
        
        # Flag large gaps (>2 hours)
        large_gaps = time_diffs > timedelta(hours=2)
        large_gap_indices = large_gaps[large_gaps].index
        
        if len(large_gap_indices) > 0:
            logger.warning(f"Found {len(large_gap_indices)} large gaps (>2 hours)")
            # Add gap flag column
            df_filled['large_gap'] = False
            df_filled.loc[large_gap_indices, 'large_gap'] = True
        
        # Forward fill small gaps
        df_filled = df_filled.fillna(method='ffill', limit=4)  # Max 2 hours forward fill
        
        # Fill remaining NaN with 0
        df_filled = df_filled.fillna(0)
        
        return df_filled
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict:
        """Get the date range of the data."""
        if 'Read Date and End Time' not in df.columns:
            return None
        
        try:
            dates = pd.to_datetime(df['Read Date and End Time'], dayfirst=True)
            return {
                'start': dates.min(),
                'end': dates.max(),
                'duration_days': (dates.max() - dates.min()).days
            }
        except:
            return None


def parse_mprn_file(uploaded_file) -> pd.DataFrame:
    """
    Convenience function to parse MPRN file.
    
    Args:
        uploaded_file: Streamlit uploaded file object or file path
        
    Returns:
        pd.DataFrame: Parsed and structured MPRN data
    """
    parser = MPRNDataParser()
    return parser.parse_mprn_file(uploaded_file)


def validate_mprn_data(df: pd.DataFrame) -> Dict:
    """
    Convenience function to validate MPRN data.
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        Dict: Validation results
    """
    parser = MPRNDataParser()
    return parser.validate_mprn_data(df)


def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to clean and resample MPRN data.
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        pd.DataFrame: Cleaned and resampled data
    """
    parser = MPRNDataParser()
    return parser.clean_and_resample(df) 