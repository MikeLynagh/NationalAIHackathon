"""
Unit tests for Usage Analyzer module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from usage_analyzer import (
    UsageAnalyzer,
    analyze_daily_patterns,
    identify_peaks,
    calculate_usage_stats,
    classify_user_type,
)


class TestUsageAnalyzer:
    """Test cases for UsageAnalyzer class."""

    def setup_method(self):
        """Set up test data."""
        self.analyzer = UsageAnalyzer()

        # Create sample usage data (24 hours, 30-min intervals)
        dates = pd.date_range("2025-01-01 00:00", periods=48, freq="30min")

        # Create realistic usage pattern: higher during day, lower at night
        usage_values = []
        for i, date in enumerate(dates):
            hour = date.hour
            if 6 <= hour <= 8:  # Morning ramp-up
                usage = 0.5 + np.random.normal(0, 0.1)
            elif 9 <= hour <= 16:  # Day usage
                usage = 1.0 + np.random.normal(0, 0.2)
            elif 17 <= hour <= 19:  # Peak hours
                usage = 2.0 + np.random.normal(0, 0.3)
            elif 20 <= hour <= 22:  # Evening
                usage = 1.2 + np.random.normal(0, 0.2)
            else:  # Night (23:00 - 05:00)
                usage = 0.3 + np.random.normal(0, 0.1)

            usage_values.append(max(0.01, usage))  # Ensure positive values

        self.sample_data = pd.DataFrame({"timestamp": dates, "import_kw": usage_values})

        # Add some clear peak periods that will definitely be detected
        # Make peaks much higher than the rolling average
        self.sample_data.loc[34:36, "import_kw"] = [8.0, 10.0, 9.0]  # Peak around 17:00
        self.sample_data.loc[42:44, "import_kw"] = [
            7.5,
            9.5,
            8.5,
        ]  # Another peak around 21:00

    def test_analyze_daily_patterns(self):
        """Test daily pattern analysis."""
        result = self.analyzer.analyze_daily_patterns(self.sample_data)

        assert isinstance(result, dict)
        assert "hourly_pattern" in result
        assert "daily_pattern" in result
        assert "weekend_vs_weekday" in result
        assert "time_of_use" in result
        assert "total_records" in result
        assert "date_range" in result

        # Check specific values
        assert result["total_records"] == 48
        assert len(result["hourly_pattern"]) == 24  # 24 hours

        # Check time-of-use breakdown
        time_of_use = result["time_of_use"]
        assert "night_percentage" in time_of_use
        assert "day_percentage" in time_of_use
        assert "peak_percentage" in time_of_use

        # Percentages should sum to approximately 100%
        total_percentage = (
            time_of_use["night_percentage"] + time_of_use["day_percentage"]
        )
        assert abs(total_percentage - 100) < 5  # Allow small rounding differences

    def test_identify_peaks(self):
        """Test peak identification."""
        peaks = self.analyzer.identify_peaks(self.sample_data)

        assert isinstance(peaks, list)
        assert len(peaks) > 0  # Should identify our artificial peaks

        # Check peak structure
        for peak in peaks:
            assert "start_time" in peak
            assert "end_time" in peak
            assert "duration_hours" in peak
            assert "peak_value" in peak
            assert "average_value" in peak
            assert "total_energy" in peak
            assert "magnitude_factor" in peak

            # Check data types
            assert isinstance(peak["start_time"], pd.Timestamp)
            assert isinstance(peak["end_time"], pd.Timestamp)
            assert isinstance(peak["duration_hours"], (int, float))
            assert isinstance(peak["peak_value"], (int, float))

            # Check logical constraints
            assert peak["start_time"] <= peak["end_time"]
            assert peak["duration_hours"] > 0
            assert peak["peak_value"] > 0
            assert peak["magnitude_factor"] > 1  # Peak should be above average

    def test_calculate_usage_stats(self):
        """Test usage statistics calculation."""
        stats = self.analyzer.calculate_usage_stats(self.sample_data)

        assert isinstance(stats, dict)
        assert "basic" in stats
        assert "peak_hours" in stats
        assert "night_vs_day" in stats
        assert "efficiency" in stats

        # Check basic stats
        basic = stats["basic"]
        assert basic["total_records"] == 48
        assert basic["total_energy_kwh"] > 0
        assert basic["average_power_kw"] > 0
        assert basic["max_power_kw"] > basic["average_power_kw"]
        assert basic["min_power_kw"] >= 0

        # Check peak hours stats
        peak_hours = stats["peak_hours"]
        assert "peak_hours_usage" in peak_hours
        assert "peak_hours_percentage" in peak_hours
        assert peak_hours["peak_hours_percentage"] > 0

        # Check night vs day stats
        night_day = stats["night_vs_day"]
        assert "night_usage_avg" in night_day
        assert "day_usage_avg" in night_day
        assert "night_day_ratio" in night_day

        # Check efficiency stats
        efficiency = stats["efficiency"]
        assert "usage_variability" in efficiency
        assert "peak_to_average_ratio" in efficiency
        assert efficiency["peak_to_average_ratio"] > 1

    def test_classify_user_type(self):
        """Test user classification."""
        # Get stats first
        stats = self.analyzer.calculate_usage_stats(self.sample_data)

        # Test classification
        user_type = self.analyzer.classify_user_type(stats)

        assert isinstance(user_type, str)
        assert len(user_type) > 0
        assert user_type != "Unknown"

        # Should be one of the expected types
        expected_types = [
            "Night-Heavy (Consistent)",
            "Night-Heavy (Variable)",
            "Day-Heavy (Consistent)",
            "Day-Heavy (Variable)",
            "Balanced (Consistent)",
            "Balanced (Variable)",
        ]
        assert user_type in expected_types

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=["timestamp", "import_kw"])

        # All methods should handle empty data gracefully
        patterns = self.analyzer.analyze_daily_patterns(empty_df)
        peaks = self.analyzer.identify_peaks(empty_df)
        stats = self.analyzer.calculate_usage_stats(empty_df)

        assert patterns == {}
        assert peaks == []
        assert stats == {}

    def test_missing_columns(self):
        """Test handling of missing columns."""
        # Create data with wrong column names
        wrong_df = pd.DataFrame(
            {
                "time": self.sample_data["timestamp"],
                "power": self.sample_data["import_kw"],
            }
        )

        # Should handle gracefully (return empty results)
        patterns = self.analyzer.analyze_daily_patterns(wrong_df)
        peaks = self.analyzer.identify_peaks(wrong_df)
        stats = self.analyzer.calculate_usage_stats(wrong_df)

        # These should return empty results due to missing columns
        assert patterns == {}
        assert peaks == []
        assert stats == {}

    def test_time_periods(self):
        """Test time period definitions."""
        assert self.analyzer.night_period == (
            datetime.strptime("23:00", "%H:%M").time(),
            datetime.strptime("08:00", "%H:%M").time(),
        )
        assert self.analyzer.day_period == (
            datetime.strptime("08:00", "%H:%M").time(),
            datetime.strptime("23:00", "%H:%M").time(),
        )
        assert self.analyzer.peak_period == (
            datetime.strptime("17:00", "%H:%M").time(),
            datetime.strptime("19:00", "%H:%M").time(),
        )


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Set up test data for convenience function tests."""
        dates = pd.date_range("2025-01-01 00:00", periods=24, freq="1h")
        usage_values = [0.5 + 0.5 * np.sin(i * np.pi / 12) for i in range(24)]

        self.sample_data = pd.DataFrame({"timestamp": dates, "import_kw": usage_values})

    def test_analyze_daily_patterns_function(self):
        """Test analyze_daily_patterns convenience function."""
        result = analyze_daily_patterns(self.sample_data)

        assert isinstance(result, dict)
        assert "hourly_pattern" in result
        assert "total_records" in result
        assert result["total_records"] == 24

    def test_identify_peaks_function(self):
        """Test identify_peaks convenience function."""
        peaks = identify_peaks(self.sample_data)

        assert isinstance(peaks, list)
        # May or may not have peaks depending on the sine wave pattern

    def test_calculate_usage_stats_function(self):
        """Test calculate_usage_stats convenience function."""
        stats = calculate_usage_stats(self.sample_data)

        assert isinstance(stats, dict)
        assert "basic" in stats
        assert "efficiency" in stats

    def test_classify_user_type_function(self):
        """Test classify_user_type convenience function."""
        stats = calculate_usage_stats(self.sample_data)
        user_type = classify_user_type(stats)

        assert isinstance(user_type, str)
        assert len(user_type) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
