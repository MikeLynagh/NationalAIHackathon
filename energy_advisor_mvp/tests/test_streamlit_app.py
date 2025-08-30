"""
Tests for the Streamlit application
"""

import pytest
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from streamlit_app import (
    show_basic_statistics,
    show_validation_results,
    show_usage_patterns_page,
    show_appliance_detection_page,
    show_recommendations_page,
)


class TestStreamlitApp:
    """Test cases for Streamlit app functions"""

    def setup_method(self):
        """Set up test data"""
        # Create sample DataFrame for testing
        dates = pd.date_range("2025-01-01", periods=48, freq="30min")
        self.sample_df = pd.DataFrame(
            {
                "timestamp": dates,
                "import_kw": [0.5 + i * 0.1 for i in range(48)],
                "export_kw": [0.0] * 48,
                "MPRN": ["10008632585"] * 48,
                "Meter Serial Number": ["000000000032791870"] * 48,
            }
        )

        # Mock file object for testing
        self.mock_file = Mock()
        self.mock_file.seek = Mock()

    def test_show_basic_statistics(self):
        """Test basic statistics display function"""
        # This function should not raise any errors
        try:
            show_basic_statistics(self.sample_df)
            assert True
        except Exception as e:
            pytest.fail(f"show_basic_statistics raised {e} unexpectedly!")

    def test_show_basic_statistics_with_missing_columns(self):
        """Test basic statistics with missing columns"""
        df_missing = pd.DataFrame({"timestamp": [datetime.now()]})
        try:
            show_basic_statistics(df_missing)
            assert True
        except Exception as e:
            pytest.fail(f"show_basic_statistics raised {e} unexpectedly!")

    def test_show_validation_results(self):
        """Test validation results display function"""
        # Mock validation results
        with patch("streamlit_app.validate_mprn_data") as mock_validate:
            mock_validate.return_value = {
                "is_valid": True,
                "total_records": 48,
                "date_range": "2025-01-01 to 2025-01-01",
                "errors": [],
            }

            try:
                show_validation_results(self.mock_file)
                assert True
            except Exception as e:
                pytest.fail(f"show_validation_results raised {e} unexpectedly!")

    def test_show_validation_results_with_errors(self):
        """Test validation results display with validation errors"""
        with patch("streamlit_app.validate_mprn_data") as mock_validate:
            mock_validate.return_value = {
                "is_valid": False,
                "total_records": 0,
                "errors": ["Invalid date format", "Missing required columns"],
            }

            try:
                show_validation_results(self.mock_file)
                assert True
            except Exception as e:
                pytest.fail(f"show_validation_results raised {e} unexpectedly!")

    def test_show_usage_patterns_page(self):
        """Test usage patterns page function"""
        # Mock session state
        with patch("streamlit_app.st.session_state", {"parsed_data": self.sample_df}):
            try:
                show_usage_patterns_page()
                assert True
            except Exception as e:
                pytest.fail(f"show_usage_patterns_page raised {e} unexpectedly!")

    def test_show_usage_patterns_page_no_data(self):
        """Test usage patterns page with no data"""
        with patch("streamlit_app.st.session_state", {}):
            try:
                show_usage_patterns_page()
                assert True
            except Exception as e:
                pytest.fail(f"show_usage_patterns_page raised {e} unexpectedly!")

    def test_show_appliance_detection_page(self):
        """Test appliance detection page function"""
        with patch("streamlit_app.st.session_state", {"parsed_data": self.sample_df}):
            try:
                show_appliance_detection_page()
                assert True
            except Exception as e:
                pytest.fail(f"show_appliance_detection_page raised {e} unexpectedly!")

    def test_show_recommendations_page(self):
        """Test recommendations page function"""
        with patch("streamlit_app.st.session_state", {"parsed_data": self.sample_df}):
            try:
                show_recommendations_page()
                assert True
            except Exception as e:
                pytest.fail(f"show_recommendations_page raised {e} unexpectedly!")


class TestStreamlitAppIntegration:
    """Integration tests for Streamlit app"""

    def test_app_imports(self):
        """Test that the Streamlit app can be imported without errors"""
        try:
            import streamlit_app

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import streamlit_app: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing streamlit_app: {e}")

    def test_app_functions_exist(self):
        """Test that all required functions exist"""
        import streamlit_app

        required_functions = [
            "main",
            "show_data_upload_page",
            "show_basic_statistics",
            "show_validation_results",
            "show_usage_patterns_page",
            "show_appliance_detection_page",
            "show_recommendations_page",
        ]

        for func_name in required_functions:
            assert hasattr(streamlit_app, func_name), f"Function {func_name} not found"
            assert callable(
                getattr(streamlit_app, func_name)
            ), f"Function {func_name} is not callable"
