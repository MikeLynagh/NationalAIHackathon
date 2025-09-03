"""
Tariff Engine for Energy Advisor MVP

This module provides cost calculations for energy usage, starting with simple rate input
and building toward complex tariff plans with time-of-use pricing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import logging

# Set up logging
logger = logging.getLogger(__name__)


class TariffEngine:
    """
    Calculates energy costs using various tariff structures.
    """

    def __init__(self):
        """Initialize the TariffEngine."""
        # Default Irish time periods
        self.night_period = (time(23, 0), time(8, 0))  # 23:00 - 08:00
        self.day_period = (time(8, 0), time(23, 0))  # 08:00 - 23:00
        self.peak_period = (time(17, 0), time(19, 0))  # 17:00 - 19:00

        logger.info("TariffEngine initialized with Irish time periods")

    def calculate_simple_cost(
        self, usage_df: pd.DataFrame, rate_per_kwh: float
    ) -> Dict:
        """
        Calculate basic energy costs using a simple flat rate.

        Args:
            usage_df: DataFrame with 'timestamp' and 'import_kw' columns
            rate_per_kwh: Rate in euros per kWh

        Returns:
            Dictionary containing cost breakdown
        """
        if usage_df.empty:
            logger.warning("Empty DataFrame provided for cost calculation")
            return {}

        try:
            # Ensure timestamp is datetime
            df = usage_df.copy()
            df["timestamp"] = pd.to_datetime(df["future_datetime"])

            # Calculate total energy consumption (kWh)
            # Each row represents 30 minutes, so multiply by 0.5 hours
            df["energy_kwh"] = df["forecasted_consumption"] * 0.5

            # Basic cost calculations
            total_energy_kwh = df["energy_kwh"].sum()
            total_cost_euros = total_energy_kwh * rate_per_kwh

            # Time-based breakdown
            df["hour"] = df["timestamp"].dt.hour

            # Night usage (23:00 - 08:00)
            night_mask = (df["hour"] >= 23) | (df["hour"] < 8)
            night_energy = df[night_mask]["energy_kwh"].sum()
            night_cost = night_energy * rate_per_kwh

            # Day usage (08:00 - 17:00, excluding peak)
            day_mask = (df["hour"] >= 8) & (df["hour"] < 17)
            day_energy = df[day_mask]["energy_kwh"].sum()
            day_cost = day_energy * rate_per_kwh

            # Evening usage (19:00 - 23:00)
            evening_mask = (df["hour"] >= 19) & (df["hour"] < 23)
            evening_energy = df[evening_mask]["energy_kwh"].sum()
            evening_cost = evening_energy * rate_per_kwh

            # Peak usage (17:00 - 19:00)
            peak_mask = (df["hour"] >= 17) & (df["hour"] < 19)
            peak_energy = df[peak_mask]["energy_kwh"].sum()
            peak_cost = peak_energy * rate_per_kwh

            # Cost breakdown
            cost_breakdown = {
                "total_energy_kwh": round(total_energy_kwh, 2),
                "total_cost_euros": round(total_cost_euros, 2),
                "rate_per_kwh": rate_per_kwh,
                "time_periods": {
                    "night": {
                        "energy_kwh": round(night_energy, 2),
                        "cost_euros": round(night_cost, 2),
                        "percentage": round((night_energy / total_energy_kwh) * 100, 1),
                    },
                    "day": {
                        "energy_kwh": round(day_energy, 2),
                        "cost_euros": round(day_cost, 2),
                        "percentage": round((day_energy / total_energy_kwh) * 100, 1),
                    },
                    "peak": {
                        "energy_kwh": round(peak_energy, 2),
                        "cost_euros": round(peak_cost, 2),
                        "percentage": round((peak_energy / total_energy_kwh) * 100, 1),
                    },
                    "evening": {
                        "energy_kwh": round(evening_energy, 2),
                        "cost_euros": round(evening_cost, 2),
                        "percentage": round(
                            (evening_energy / total_energy_kwh) * 100, 1
                        ),
                    },
                },
                "daily_average": {
                    "energy_kwh": round(
                        total_energy_kwh
                        / max(1, (df["timestamp"].max() - df["timestamp"].min()).days),
                        2,
                    ),
                    "cost_euros": round(
                        total_cost_euros
                        / max(1, (df["timestamp"].max() - df["timestamp"].min()).days),
                        2,
                    ),
                },
                "monthly_projection": {
                    "energy_kwh": round(
                        total_energy_kwh
                        * 30
                        / max(1, (df["timestamp"].max() - df["timestamp"].min()).days),
                        2,
                    ),
                    "cost_euros": round(
                        total_cost_euros
                        * 30
                        / max(1, (df["timestamp"].max() - df["timestamp"].min()).days),
                        2,
                    ),
                },
            }

            logger.info(
                f"Simple cost calculation completed: €{total_cost_euros:.2f} for {total_energy_kwh:.2f} kWh"
            )
            return cost_breakdown

        except Exception as e:
            logger.error(f"Error in simple cost calculation: {e}")
            return {}

    def calculate_time_based_cost(
        self,
        usage_df: pd.DataFrame,
        day_rate: float,
        night_rate: float,
        peak_rate: Optional[float] = None,
    ) -> Dict:
        """
        Calculate costs using day/night rates with optional peak rate.

        Args:
            usage_df: DataFrame with 'timestamp' and 'import_kw' columns
            day_rate: Rate for day hours (€/kWh)
            night_rate: Rate for night hours (€/kWh)
            peak_rate: Optional peak rate (€/kWh), defaults to day_rate

        Returns:
            Dictionary containing cost breakdown with time-based rates
        """
        if usage_df.empty:
            return {}

        try:
            # Use day rate for peak if not specified
            if peak_rate is None:
                peak_rate = day_rate

            # Ensure timestamp is datetime
            df = usage_df.copy()
            df["timestamp"] = pd.to_datetime(df["future_datetime"])
            df["energy_kwh"] = df["forecasted_consumption"] * 0.5
            df["hour"] = df["timestamp"].dt.hour

            # Calculate costs for each time period
            night_mask = (df["hour"] >= 23) | (df["hour"] < 8)
            day_mask = (df["hour"] >= 8) & (df["hour"] < 17)  # Day excluding peak
            peak_mask = (df["hour"] >= 17) & (df["hour"] < 19)
            evening_mask = (df["hour"] >= 19) & (df["hour"] < 23)  # Evening after peak

            # Energy and costs for each period
            night_energy = df[night_mask]["energy_kwh"].sum()
            night_cost = night_energy * night_rate

            day_energy = df[day_mask]["energy_kwh"].sum()
            day_cost = day_energy * day_rate

            peak_energy = df[peak_mask]["energy_kwh"].sum()
            peak_cost = peak_energy * peak_rate

            evening_energy = df[evening_mask]["energy_kwh"].sum()
            evening_cost = evening_energy * day_rate

            # Totals
            total_energy = night_energy + day_energy + peak_energy + evening_energy
            total_cost = night_cost + day_cost + peak_cost + evening_cost

            cost_breakdown = {
                "total_energy_kwh": round(total_energy, 2),
                "total_cost_euros": round(total_cost, 2),
                "rates": {"night": night_rate, "day": day_rate, "peak": peak_rate},
                "time_periods": {
                    "night": {
                        "energy_kwh": round(night_energy, 2),
                        "cost_euros": round(night_cost, 2),
                        "percentage": round((night_energy / total_energy) * 100, 1),
                    },
                    "day": {
                        "energy_kwh": round(day_energy, 2),
                        "cost_euros": round(day_cost, 2),
                        "percentage": round((day_energy / total_energy) * 100, 1),
                    },
                    "peak": {
                        "energy_kwh": round(peak_energy, 2),
                        "cost_euros": round(peak_cost, 2),
                        "percentage": round((peak_energy / total_energy) * 100, 1),
                    },
                    "evening": {
                        "energy_kwh": round(evening_energy, 2),
                        "cost_euros": round(evening_cost, 2),
                        "percentage": round((evening_energy / total_energy) * 100, 1),
                    },
                },
                "savings_opportunities": self._identify_savings_opportunities(
                    night_energy,
                    day_energy,
                    peak_energy,
                    evening_energy,
                    night_rate,
                    day_rate,
                    peak_rate,
                ),
            }

            logger.info(f"Time-based cost calculation completed: €{total_cost:.2f}")
            return cost_breakdown

        except Exception as e:
            logger.error(f"Error in time-based cost calculation: {e}")
            return {}

    def _identify_savings_opportunities(
        self,
        night_energy: float,
        day_energy: float,
        peak_energy: float,
        evening_energy: float,
        night_rate: float,
        day_rate: float,
        peak_rate: float,
    ) -> List[Dict]:
        """
        Identify potential savings opportunities based on usage patterns.

        Args:
            Energy consumption and rates for each time period

        Returns:
            List of savings opportunities with estimated amounts
        """
        opportunities = []

        # Peak to night shift opportunity
        if peak_rate > night_rate and peak_energy > 0:
            potential_savings = peak_energy * (peak_rate - night_rate)
            opportunities.append(
                {
                    "type": "Peak to Night Shift",
                    "description": f"Move {peak_energy:.1f} kWh from peak hours to night hours",
                    "potential_savings": round(potential_savings, 2),
                    "difficulty": "Medium",
                    "action": "Schedule high-energy activities (dishwasher, EV charging) for night hours",
                }
            )

        # Day to night shift opportunity
        if day_rate > night_rate and day_energy > 0:
            potential_savings = (
                day_energy * (day_rate - night_rate) * 0.3
            )  # Assume 30% can be shifted
            opportunities.append(
                {
                    "type": "Day to Night Shift",
                    "description": f"Shift 30% of day usage ({day_energy * 0.3:.1f} kWh) to night hours",
                    "potential_savings": round(potential_savings, 2),
                    "difficulty": "Easy",
                    "action": "Use timers for washing machines, dishwashers, and other flexible loads",
                }
            )

        # Peak avoidance opportunity
        if peak_rate > day_rate and peak_energy > 0:
            potential_savings = peak_energy * (peak_rate - day_rate)
            opportunities.append(
                {
                    "type": "Peak Avoidance",
                    "description": f"Reduce usage during peak hours (17:00-19:00)",
                    "potential_savings": round(potential_savings, 2),
                    "difficulty": "Hard",
                    "action": "Avoid cooking, laundry, and other high-energy activities during peak hours",
                }
            )

        return opportunities

    def load_tariff_plans(self, excel_path: str) -> pd.DataFrame:
        """
        Load tariff plans from Excel file (placeholder for future enhancement).

        Args:
            excel_path: Path to Excel file containing tariff plans

        Returns:
            DataFrame with tariff plan information
        """
        # TODO: Implement Excel loading for complex tariff plans
        logger.info("Tariff plan loading not yet implemented")
        return pd.DataFrame()

    def parse_tariff_plan(self, plan_row: pd.Series) -> Dict:
        """
        Parse tariff plan details (placeholder for future enhancement).

        Args:
            plan_row: Row from tariff plans DataFrame

        Returns:
            Dictionary containing parsed tariff plan
        """
        # TODO: Implement tariff plan parsing
        logger.info("Tariff plan parsing not yet implemented")
        return {}

    def generate_price_timeseries(
        self, plan: Dict, date_range: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Generate price timeseries for given date range (placeholder for future enhancement).

        Args:
            plan: Tariff plan dictionary
            date_range: DatetimeIndex for the period

        Returns:
            Series with prices for each timestamp
        """
        # TODO: Implement price timeseries generation
        logger.info("Price timeseries generation not yet implemented")
        return pd.Series()

    def calculate_bill(
        self, usage_df: pd.DataFrame, price_series: pd.Series, plan_details: Dict
    ) -> Dict:
        """
        Calculate detailed bill with complex tariff (placeholder for future enhancement).

        Args:
            usage_df: Usage DataFrame
            price_series: Price timeseries
            plan_details: Tariff plan details

        Returns:
            Dictionary containing detailed bill breakdown
        """
        # TODO: Implement complex bill calculation
        logger.info("Complex bill calculation not yet implemented")
        return {}


# Convenience functions for easy access
def calculate_simple_cost(usage_df: pd.DataFrame, rate_per_kwh: float) -> Dict:
    """Convenience function to calculate simple costs."""
    engine = TariffEngine()
    return engine.calculate_simple_cost(usage_df, rate_per_kwh)


def calculate_time_based_cost(
    usage_df: pd.DataFrame,
    day_rate: float,
    night_rate: float,
    peak_rate: Optional[float] = None,
) -> Dict:
    """Convenience function to calculate time-based costs."""
    engine = TariffEngine()
    return engine.calculate_time_based_cost(usage_df, day_rate, night_rate, peak_rate)


if __name__ == "__main__":
    # Test the tariff engine
    print("Tariff Engine Module")
    print("Available functions:")
    print("- calculate_simple_cost(usage_df, rate_per_kwh)")
    print("- calculate_time_based_cost(usage_df, day_rate, night_rate, peak_rate)")
