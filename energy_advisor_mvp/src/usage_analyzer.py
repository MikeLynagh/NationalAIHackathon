"""
Usage Analysis Engine for Energy Advisor MVP

This module provides comprehensive analysis of energy usage patterns,
including daily/weekly patterns, peak identification, and user classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
import logging

# Set up logging
logger = logging.getLogger(__name__)


class UsageAnalyzer:
    """
    Analyzes energy usage patterns and provides insights for optimization.
    """

    def __init__(self):
        """Initialize the UsageAnalyzer."""
        # Define time periods for analysis
        self.night_period = (time(23, 0), time(8, 0))  # 23:00 - 08:00
        self.day_period = (time(8, 0), time(23, 0))  # 08:00 - 23:00
        self.peak_period = (time(17, 0), time(19, 0))  # 17:00 - 19:00

        # Thresholds for peak detection
        self.peak_threshold_multiplier = 2.0  # Peak is 2x average
        self.sustained_peak_duration = pd.Timedelta(
            hours=1
        )  # Minimum duration for sustained peak

        logger.info("UsageAnalyzer initialized with time periods and thresholds")

    def analyze_daily_patterns(self, usage_df: pd.DataFrame) -> Dict:
        """
        Analyze daily usage patterns and return comprehensive insights.

        Args:
            usage_df: DataFrame with 'timestamp' and 'import_kw' columns

        Returns:
            Dictionary containing daily pattern analysis
        """
        if usage_df.empty:
            logger.warning("Empty DataFrame provided for daily pattern analysis")
            return {}

        try:
            # Ensure timestamp is datetime
            df = usage_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Extract time components
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.day_name()
            df["is_weekend"] = df["timestamp"].dt.weekday >= 5

            # Calculate hourly averages
            hourly_pattern = (
                df.groupby("hour")["import_kw"]
                .agg(["mean", "std", "min", "max"])
                .round(3)
            )
            # Convert to proper format for each hour
            hourly_pattern_dict = {}
            for hour in range(24):
                if hour in hourly_pattern.index:
                    hourly_pattern_dict[hour] = {
                        "mean": hourly_pattern.loc[hour, "mean"],
                        "std": hourly_pattern.loc[hour, "std"],
                        "min": hourly_pattern.loc[hour, "min"],
                        "max": hourly_pattern.loc[hour, "max"],
                    }
                else:
                    hourly_pattern_dict[hour] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }

            # Calculate day-of-week patterns
            daily_pattern = (
                df.groupby("day_of_week")["import_kw"]
                .agg(["mean", "std", "min", "max"])
                .round(3)
            )

            # Weekend vs weekday comparison
            weekend_vs_weekday = (
                df.groupby("is_weekend")["import_kw"].agg(["mean", "std"]).round(3)
            )

            # Time-of-use breakdown
            time_of_use = self._calculate_time_of_use_breakdown(df)

            analysis = {
                "hourly_pattern": hourly_pattern_dict,
                "daily_pattern": daily_pattern.to_dict(),
                "weekend_vs_weekday": weekend_vs_weekday.to_dict(),
                "time_of_use": time_of_use,
                "total_records": len(df),
                "date_range": {
                    "start": df["timestamp"].min().isoformat(),
                    "end": df["timestamp"].max().isoformat(),
                    "duration_days": (
                        df["timestamp"].max() - df["timestamp"].min()
                    ).days,
                },
            }

            logger.info(f"Daily pattern analysis completed for {len(df)} records")
            return analysis

        except Exception as e:
            logger.error(f"Error in daily pattern analysis: {e}")
            return {}

    def identify_peaks(self, usage_df: pd.DataFrame) -> List[Dict]:
        """
        Identify peak usage periods and magnitudes.

        Args:
            usage_df: DataFrame with 'timestamp' and 'import_kw' columns

        Returns:
            List of peak periods with details
        """
        if usage_df.empty:
            return []

        try:
            df = usage_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Use a simpler peak detection approach
            # Look for values that are significantly above the overall average
            overall_mean = df["import_kw"].mean()
            overall_std = df["import_kw"].std()

            # Peak threshold: 2 standard deviations above mean
            peak_threshold = overall_mean + (2 * overall_std)

            # Find peaks
            peak_mask = df["import_kw"] > peak_threshold

            peaks = []
            if peak_mask.any():
                # Group consecutive peak periods
                peak_groups = self._group_consecutive_peaks(df, peak_mask)

                for group in peak_groups:
                    peak_data = df.iloc[group]
                    peak_info = {
                        "start_time": peak_data["timestamp"].min(),
                        "end_time": peak_data["timestamp"].max(),
                        "duration_hours": (
                            peak_data["timestamp"].max() - peak_data["timestamp"].min()
                        ).total_seconds()
                        / 3600,
                        "peak_value": peak_data["import_kw"].max(),
                        "average_value": peak_data["import_kw"].mean(),
                        "total_energy": peak_data["import_kw"].sum()
                        * 0.5,  # 30-min intervals = 0.5 hours
                        "magnitude_factor": peak_data["import_kw"].max() / overall_mean,
                    }
                    peaks.append(peak_info)

                # Sort by magnitude
                peaks.sort(key=lambda x: x["magnitude_factor"], reverse=True)

            logger.info(f"Identified {len(peaks)} peak periods")
            return peaks

        except Exception as e:
            logger.error(f"Error in peak identification: {e}")
            return []

    def calculate_usage_stats(self, usage_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive usage statistics.

        Args:
            usage_df: DataFrame with 'timestamp' and 'import_kw' columns

        Returns:
            Dictionary containing usage statistics
        """
        if usage_df.empty:
            return {}

        try:
            df = usage_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Basic statistics
            basic_stats = {
                "total_records": len(df),
                "total_energy_kwh": (df["import_kw"] * 0.5).sum(),  # 30-min intervals
                "average_power_kw": df["import_kw"].mean(),
                "max_power_kw": df["import_kw"].max(),
                "min_power_kw": df["import_kw"].min(),
                "std_power_kw": df["import_kw"].std(),
            }

            # Time-based statistics
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.day_name()

            # Peak hours (17:00-19:00)
            peak_mask = (df["hour"] >= 17) & (df["hour"] < 19)
            peak_stats = {
                "peak_hours_usage": (
                    df[peak_mask]["import_kw"].mean() if peak_mask.any() else 0
                ),
                "peak_hours_percentage": (peak_mask.sum() / len(df)) * 100,
            }

            # Night vs day usage
            night_mask = (df["hour"] >= 23) | (df["hour"] < 8)
            day_mask = (df["hour"] >= 8) & (df["hour"] < 23)

            night_day_stats = {
                "night_usage_avg": (
                    df[night_mask]["import_kw"].mean() if night_mask.any() else 0
                ),
                "day_usage_avg": (
                    df[day_mask]["import_kw"].mean() if day_mask.any() else 0
                ),
                "night_day_ratio": (
                    (
                        df[night_mask]["import_kw"].mean()
                        / df[day_mask]["import_kw"].mean()
                    )
                    if day_mask.any() and df[day_mask]["import_kw"].mean() > 0
                    else 0
                ),
            }

            # Efficiency metrics
            efficiency_stats = {
                "usage_variability": (
                    df["import_kw"].std() / df["import_kw"].mean()
                    if df["import_kw"].mean() > 0
                    else 0
                ),
                "peak_to_average_ratio": (
                    df["import_kw"].max() / df["import_kw"].mean()
                    if df["import_kw"].mean() > 0
                    else 0
                ),
            }

            stats = {
                "basic": basic_stats,
                "peak_hours": peak_stats,
                "night_vs_day": night_day_stats,
                "efficiency": efficiency_stats,
            }

            logger.info(f"Usage statistics calculated for {len(df)} records")
            return stats

        except Exception as e:
            logger.error(f"Error in usage statistics calculation: {e}")
            return {}

    def classify_user_type(self, usage_stats: Dict) -> str:
        """
        Classify user based on usage patterns.

        Args:
            usage_stats: Dictionary from calculate_usage_stats()

        Returns:
            User classification string
        """
        try:
            if not usage_stats or "night_vs_day" not in usage_stats:
                return "Unknown"

            night_day = usage_stats["night_vs_day"]
            efficiency = usage_stats.get("efficiency", {})

            # Extract key metrics
            night_ratio = night_day.get("night_day_ratio", 0)
            variability = efficiency.get("usage_variability", 0)
            peak_ratio = efficiency.get("peak_to_average_ratio", 0)

            # Classification logic
            if night_ratio > 1.5:
                if variability < 0.5:
                    return "Night-Heavy (Consistent)"
                else:
                    return "Night-Heavy (Variable)"
            elif night_ratio < 0.7:
                if variability < 0.5:
                    return "Day-Heavy (Consistent)"
                else:
                    return "Day-Heavy (Variable)"
            else:
                if variability < 0.5:
                    return "Balanced (Consistent)"
                else:
                    return "Balanced (Variable)"

        except Exception as e:
            logger.error(f"Error in user classification: {e}")
            return "Unknown"

    def _calculate_time_of_use_breakdown(self, df: pd.DataFrame) -> Dict:
        """Calculate time-of-use breakdown percentages."""
        try:
            df["hour"] = df["timestamp"].dt.hour

            # Night usage (23:00 - 08:00)
            night_mask = (df["hour"] >= 23) | (df["hour"] < 8)
            night_percentage = (night_mask.sum() / len(df)) * 100

            # Day usage (08:00 - 23:00)
            day_mask = (df["hour"] >= 8) & (df["hour"] < 23)
            day_percentage = (day_mask.sum() / len(df)) * 100

            # Peak usage (17:00 - 19:00)
            peak_mask = (df["hour"] >= 17) & (df["hour"] < 19)
            peak_percentage = (peak_mask.sum() / len(df)) * 100

            return {
                "night_percentage": round(night_percentage, 1),
                "day_percentage": round(day_percentage, 1),
                "peak_percentage": round(peak_percentage, 1),
            }

        except Exception as e:
            logger.error(f"Error in time-of-use breakdown: {e}")
            return {}

    def _group_consecutive_peaks(
        self, df: pd.DataFrame, peak_mask: pd.Series
    ) -> List[List[int]]:
        """Group consecutive peak periods."""
        try:
            groups = []
            current_group = []

            for i, is_peak in enumerate(peak_mask):
                if is_peak:
                    current_group.append(i)
                elif current_group:
                    if len(current_group) >= 2:  # Minimum 2 consecutive peaks
                        groups.append(current_group)
                    current_group = []

            # Handle case where last group ends at end of data
            if current_group and len(current_group) >= 2:
                groups.append(current_group)

            return groups

        except Exception as e:
            logger.error(f"Error in grouping consecutive peaks: {e}")
            return []


# Convenience functions for easy access
def analyze_daily_patterns(usage_df: pd.DataFrame) -> Dict:
    """Convenience function to analyze daily patterns."""
    analyzer = UsageAnalyzer()
    return analyzer.analyze_daily_patterns(usage_df)


def identify_peaks(usage_df: pd.DataFrame) -> List[Dict]:
    """Convenience function to identify peaks."""
    analyzer = UsageAnalyzer()
    return analyzer.identify_peaks(usage_df)


def calculate_usage_stats(usage_df: pd.DataFrame) -> Dict:
    """Convenience function to calculate usage stats."""
    analyzer = UsageAnalyzer()
    return analyzer.calculate_usage_stats(usage_df)


def classify_user_type(usage_stats: Dict) -> str:
    """Convenience function to classify user type."""
    analyzer = UsageAnalyzer()
    return analyzer.classify_user_type(usage_stats)
