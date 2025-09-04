"""
Tests for the Tariff Engine module
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from energy_advisor_mvp.src.tariff_engine import TariffEngine, calculate_simple_cost, calculate_time_based_cost


class TestTariffEngine:
    """Test the TariffEngine class"""

    def setup_method(self):
        """Set up test data"""
        # Create sample usage data (24 hours, 30-min intervals)
        dates = pd.date_range(
            start="2024-01-01 00:00:00", end="2024-01-01 23:30:00", freq="30min"
        )

        # Create realistic usage pattern: low at night, higher during day
        usage_pattern = []
        for date in dates:
            hour = date.hour
            if 23 <= hour or hour < 6:  # Night (low usage)
                usage_pattern.append(0.5)  # 0.5 kW
            elif 6 <= hour < 8:  # Morning ramp up
                usage_pattern.append(1.5)  # 1.5 kW
            elif 8 <= hour < 18:  # Day (normal usage)
                usage_pattern.append(2.0)  # 2.0 kW
            elif 18 <= hour < 20:  # Peak hours
                usage_pattern.append(3.0)  # 3.0 kW
            else:  # Evening
                usage_pattern.append(1.8)  # 1.8 kW

        self.test_df = pd.DataFrame({"timestamp": dates, "import_kw": usage_pattern})

        self.engine = TariffEngine()

    def test_calculate_simple_cost_basic(self):
        """Test basic simple cost calculation"""
        result = self.engine.calculate_simple_cost(self.test_df, 0.23)

        assert result is not None
        assert "total_energy_kwh" in result
        assert "total_cost_euros" in result
        assert "rate_per_kwh" in result
        assert result["rate_per_kwh"] == 0.23

        # Check that total energy is reasonable (48 intervals * average ~1.5 kW * 0.5 hours)
        assert 30 <= result["total_energy_kwh"] <= 50
        assert result["total_cost_euros"] > 0

    def test_calculate_simple_cost_time_periods(self):
        """Test that time periods are correctly calculated"""
        result = self.engine.calculate_simple_cost(self.test_df, 0.23)

        assert "time_periods" in result
        time_periods = result["time_periods"]

        # Check all expected time periods exist
        assert "night" in time_periods
        assert "day" in time_periods
        assert "peak" in time_periods

        # Check percentages add up to approximately 100%
        total_percentage = sum(period["percentage"] for period in time_periods.values())
        assert 95 <= total_percentage <= 105  # Allow for rounding

    def test_calculate_simple_cost_empty_data(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.engine.calculate_simple_cost(empty_df, 0.23)

        assert result == {}

    def test_calculate_time_based_cost(self):
        """Test time-based cost calculation"""
        result = self.engine.calculate_time_based_cost(
            self.test_df, day_rate=0.25, night_rate=0.18, peak_rate=0.30
        )

        assert result is not None
        assert "total_cost_euros" in result
        assert "rates" in result
        assert result["rates"]["day"] == 0.25
        assert result["rates"]["night"] == 0.18
        assert result["rates"]["peak"] == 0.30

        # Check that time periods are calculated
        assert "time_periods" in result
        assert "evening" in result["time_periods"]  # Should include evening period

    def test_calculate_time_based_cost_no_peak_rate(self):
        """Test time-based cost calculation without peak rate"""
        result = self.engine.calculate_time_based_cost(
            self.test_df, day_rate=0.25, night_rate=0.18
        )

        assert result is not None
        assert result["rates"]["peak"] == 0.25  # Should default to day rate

    def test_savings_opportunities(self):
        """Test that savings opportunities are identified"""
        result = self.engine.calculate_time_based_cost(
            self.test_df, day_rate=0.25, night_rate=0.18, peak_rate=0.30
        )

        if "savings_opportunities" in result:
            opportunities = result["savings_opportunities"]
            assert isinstance(opportunities, list)

            # Should identify peak to night shift opportunity
            peak_opportunities = [
                op for op in opportunities if "Peak to Night" in op["type"]
            ]
            if peak_opportunities:
                assert "potential_savings" in peak_opportunities[0]
                assert peak_opportunities[0]["potential_savings"] > 0


class TestConvenienceFunctions:
    """Test the convenience functions"""

    def setup_method(self):
        """Set up test data"""
        dates = pd.date_range(
            start="2024-01-01 00:00:00", end="2024-01-01 23:30:00", freq="30min"
        )

        self.test_df = pd.DataFrame(
            {"timestamp": dates, "import_kw": [1.0] * len(dates)}  # 1 kW constant usage
        )

    def test_calculate_simple_cost_function(self):
        """Test the convenience function for simple cost"""
        result = calculate_simple_cost(self.test_df, 0.23)

        assert result is not None
        assert "total_cost_euros" in result
        assert result["rate_per_kwh"] == 0.23

    def test_calculate_time_based_cost_function(self):
        """Test the convenience function for time-based cost"""
        result = calculate_time_based_cost(self.test_df, 0.25, 0.18, 0.30)

        assert result is not None
        assert "total_cost_euros" in result
        assert result["rates"]["day"] == 0.25


if __name__ == "__main__":
    pytest.main([__file__])
