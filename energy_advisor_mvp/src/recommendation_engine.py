"""
AI-Powered Recommendation Engine for Energy Advisor MVP

This module provides intelligent energy savings recommendations by analyzing usage patterns,
detecting appliances, and suggesting optimal timing and tariff strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import json
import re

# Import our existing modules
from ai_engine import AIEngine
from tariff_engine import TariffEngine
from usage_analyzer import UsageAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Generates AI-powered energy saving recommendations.
    """

    def __init__(self, ai_engine: Optional[AIEngine] = None):
        self.ai_engine = ai_engine
        self.tariff_engine = TariffEngine()
        self.usage_analyzer = UsageAnalyzer()
        self.savings_thresholds = {
            'low_impact': 5.0,   # €5+ monthly savings
            'medium_impact': 20.0, # €20+ monthly savings
            'high_impact': 50.0,  # €50+ monthly savings
        }
        logger.info("RecommendationEngine initialized with AI integration")

    def generate_recommendations(
        self,
        usage_df: pd.DataFrame,
        current_rate: float,
        user_preferences: Optional[Dict] = None
    ) -> Dict:
        """
        Generates a comprehensive set of energy saving recommendations.

        Args:
            usage_df: DataFrame with usage data
            current_rate: Current electricity rate (€/kWh)
            user_preferences: Optional user preferences and constraints

        Returns:
            Dictionary containing recommendations and analysis
        """
        if usage_df.empty:
            logger.warning("Empty DataFrame provided for recommendations")
            return {'recommendations': [], 'analysis': {}, 'total_potential_savings': 0}

        try:
            # Analyze usage patterns
            daily_patterns = self.usage_analyzer.analyze_daily_patterns(usage_df)
            usage_stats = self.usage_analyzer.calculate_usage_stats(usage_df)

            # Calculate current costs
            current_costs = self.tariff_engine.calculate_simple_cost(usage_df, current_rate)

            recommendations = []

            # 1. Tariff optimization recommendations
            tariff_recs = self._generate_tariff_recommendations(
                usage_df, current_rate, current_costs
            )
            recommendations.extend(tariff_recs)

            # 2. Time-shifting recommendations
            time_shift_recs = self._generate_time_shifting_recommendations(
                usage_df, current_rate, daily_patterns
            )
            recommendations.extend(time_shift_recs)

            # 3. Load balancing recommendations
            load_balance_recs = self._generate_load_balancing_recommendations(
                usage_df, current_rate, usage_stats
            )
            recommendations.extend(load_balance_recs)

            # 4. Behavioral change recommendations
            behavioral_recs = self._generate_behavioral_recommendations(
                usage_df, current_rate, usage_stats
            )
            recommendations.extend(behavioral_recs)

            # 5. AI-powered insights (if AI engine is available)
            ai_insights = []
            if self.ai_engine and hasattr(self.ai_engine, 'generate_appliance_insights_ai'):
                try:
                    ai_insights = self.ai_engine.generate_appliance_insights_ai(
                        usage_df, [] # Placeholder for detected appliances
                    )
                except Exception as e:
                    logger.warning(f"AI insights generation failed: {e}")

            # Sort recommendations by impact
            recommendations = self._sort_recommendations_by_impact(recommendations)

            # Calculate total potential savings
            total_savings = sum(rec.get('monthly_savings', 0) for rec in recommendations)

            return {
                'recommendations': recommendations,
                'ai_insights': ai_insights,
                'total_potential_savings': round(total_savings, 2),
                'analysis': {
                    'current_costs': current_costs,
                    'usage_stats': usage_stats,
                    'daily_patterns': daily_patterns,
                },
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'recommendations': [], 'analysis': {}, 'error': str(e)}

    def _generate_tariff_recommendations(
        self,
        usage_df: pd.DataFrame,
        current_rate: float,
        current_costs: Dict
    ) -> List[Dict]:
        """Generate tariff optimization recommendations."""
        recommendations = []

        # Simulate different tariff scenarios
        tariff_scenarios = [
            {'name': 'Night Saver', 'day_rate': 0.25, 'night_rate': 0.15, 'peak_rate': 0.30},
            {'name': 'Smart Time', 'day_rate': 0.22, 'night_rate': 0.18, 'peak_rate': 0.35},
            {'name': 'Green Energy', 'day_rate': 0.24, 'night_rate': 0.20, 'peak_rate': 0.28},
        ]

        current_monthly = current_costs.get('monthly_projection', {}).get('cost_euros', 0)

        for scenario in tariff_scenarios:
            # Calculate costs with this tariff
            scenario_costs = self.tariff_engine.calculate_time_based_cost(
                usage_df,
                scenario['day_rate'],
                scenario['night_rate'],
                scenario['peak_rate']
            )

            # Use total_cost_euros directly from time-based calculation
            scenario_monthly = scenario_costs.get('total_cost_euros', 0)
            monthly_savings = current_monthly - scenario_monthly

            if monthly_savings > 5:  # Only recommend if savings > €5/month
                recommendations.append({
                    'type': 'tariff_switch',
                    'title': f"Switch to {scenario['name']} Tariff",
                    'description': f"Switch to {scenario['name']} tariff with time-based rates",
                    'monthly_savings': round(monthly_savings, 2),
                    'annual_savings': round(monthly_savings * 12, 2),
                    'difficulty': 'Easy',
                    'time_to_implement': '1-2 weeks',
                    'action_items': [
                        f"Contact your energy provider about {scenario['name']} tariff",
                        f"Compare rates: Day €{scenario['day_rate']:.2f}/kWh, Night €{scenario['night_rate']:.2f}/kWh, Peak €{scenario['peak_rate']:.2f}/kWh",
                        "Review contract terms and switching fees"
                    ],
                    'scenario_details': scenario,
                    'impact_level': self._get_impact_level(monthly_savings)
                })

        return recommendations

    def _generate_time_shifting_recommendations(
        self,
        usage_df: pd.DataFrame,
        current_rate: float,
        daily_patterns: Dict
    ) -> List[Dict]:
        """Generate time-shifting recommendations."""
        recommendations = []

        # Analyze peak usage patterns
        hourly_pattern_data = daily_patterns.get('hourly_pattern', {})
        if hourly_pattern_data:
            # Extract mean usage for comparison
            hourly_means = {hour: data['mean'] for hour, data in hourly_pattern_data.items()}
            
            if not hourly_means:
                return recommendations # No hourly data to analyze

            mean_overall_usage = np.mean(list(hourly_means.values()))

            peak_hours = [hour for hour, usage_mean in hourly_means.items()
                        if usage_mean > mean_overall_usage * 1.5]

            if peak_hours:
                # Find the highest usage hour
                max_hour_data = max(hourly_means.items(), key=lambda x: x[1])
                peak_hour = max_hour_data[0]
                peak_usage_mean = max_hour_data[1]

                # Calculate potential savings from shifting peak usage
                # Assume we can shift 30% of peak usage to night hours
                shiftable_usage = peak_usage_mean * 0.3
                night_rate = current_rate * 0.7  # Assume night rate is 30% cheaper
                hourly_savings = shiftable_usage * (current_rate - night_rate) * 0.5  # 0.5 hours per interval
                monthly_savings = hourly_savings * 30  # 30 days

                if monthly_savings > 5:
                    recommendations.append({
                        'type': 'time_shifting',
                        'title': 'Shift Peak Hour Usage to Night',
                        'description': f"Move high-energy activities from {peak_hour}:00 to night hours",
                        'monthly_savings': round(monthly_savings, 2),
                        'annual_savings': round(monthly_savings * 12, 2),
                        'difficulty': 'Medium',
                        'time_to_implement': 'Immediate',
                        'action_items': [
                            f"Identify activities causing high usage at {peak_hour}:00",
                            "Use timers to delay high-energy appliances until night",
                            "Consider scheduling EV charging, washing, or heating for night hours",
                            "Monitor usage patterns after implementation"
                        ],
                        'peak_hour': peak_hour,
                        'peak_usage': round(peak_usage_mean, 2),
                        'shiftable_usage': round(shiftable_usage, 2),
                        'impact_level': self._get_impact_level(monthly_savings)
                    })

        return recommendations

    def _generate_load_balancing_recommendations(
        self,
        usage_df: pd.DataFrame,
        current_rate: float,
        usage_stats: Dict
    ) -> List[Dict]:
        """Generate load balancing recommendations."""
        recommendations = []

        # Analyze usage variability
        efficiency = usage_stats.get('efficiency', {})
        variability = efficiency.get('usage_variability', 0)
        peak_ratio = efficiency.get('peak_to_average_ratio', 0)

        if peak_ratio > 2.0:  # High peak-to-average ratio
            # Calculate potential savings from load balancing
            # Assume we can reduce peak usage by 20% through better distribution
            current_monthly = usage_stats.get('basic', {}).get('total_energy_kwh', 0) * current_rate * 30
            peak_reduction = current_monthly * 0.1  # 10% reduction in peak charges
            monthly_savings = peak_reduction * 0.2  # 20% of peak reduction

            if monthly_savings > 5:
                recommendations.append({
                    'type': 'load_balancing',
                    'title': 'Balance Energy Load Throughout Day',
                    'description': 'Distribute energy usage more evenly to reduce peak charges',
                    'monthly_savings': round(monthly_savings, 2),
                    'annual_savings': round(monthly_savings * 12, 2),
                    'difficulty': 'Medium',
                    'time_to_implement': '1-2 weeks',
                    'action_items': [
                        "Stagger high-energy appliances throughout the day",
                        "Avoid running multiple appliances simultaneously",
                        "Use smart home automation to distribute loads",
                        "Monitor peak usage and adjust timing accordingly"
                    ],
                    'current_variability': round(variability, 2),
                    'peak_ratio': round(peak_ratio, 2),
                    'impact_level': self._get_impact_level(monthly_savings)
                })

        return recommendations

    def _generate_behavioral_recommendations(
        self,
        usage_df: pd.DataFrame,
        current_rate: float,
        usage_stats: Dict
    ) -> List[Dict]:
        """Generate behavioral change recommendations."""
        recommendations = []

        # Analyze usage patterns for behavioral insights
        basic_stats = usage_stats.get('basic', {})
        total_energy = basic_stats.get('total_energy_kwh', 0)
        avg_usage = basic_stats.get('average_power_kw', 0)

        # Check for high baseline usage (standby power)
        if avg_usage > 0.5:  # High baseline usage
            # Calculate potential savings from reducing standby power
            standby_reduction = avg_usage * 0.3  # 30% reduction in standby
            monthly_savings = standby_reduction * current_rate * 24 * 30  # 24/7 for 30 days

            if monthly_savings > 5:
                recommendations.append({
                    'type': 'behavioral',
                    'title': 'Reduce Standby Power Consumption',
                    'description': 'Turn off or unplug devices when not in use',
                    'monthly_savings': round(monthly_savings, 2),
                    'annual_savings': round(monthly_savings * 12, 2),
                    'difficulty': 'Easy',
                    'time_to_implement': 'Immediate',
                    'action_items': [
                        "Unplug chargers and devices when not in use",
                        "Use power strips with switches for entertainment centers",
                        "Enable power-saving modes on computers and TVs",
                        "Consider smart plugs for automatic control"
                    ],
                    'current_baseline': round(avg_usage, 2),
                    'potential_reduction': round(standby_reduction, 2),
                    'impact_level': self._get_impact_level(monthly_savings)
                })

        return recommendations

    def _get_impact_level(self, monthly_savings: float) -> str:
        """Determine the impact level based on monthly savings."""
        if monthly_savings >= self.savings_thresholds['high_impact']:
            return 'High Impact'
        elif monthly_savings >= self.savings_thresholds['medium_impact']:
            return 'Medium Impact'
        elif monthly_savings >= self.savings_thresholds['low_impact']:
            return 'Low Impact'
        else:
            return 'Minimal Impact'

    def _sort_recommendations_by_impact(self, recommendations: List[Dict]) -> List[Dict]:
        """Sort recommendations by impact level and monthly savings."""
        impact_order = {'High Impact': 0, 'Medium Impact': 1, 'Low Impact': 2, 'Minimal Impact': 3}

        return sorted(
            recommendations,
            key=lambda x: (impact_order.get(x.get('impact_level', 'Minimal Impact'), 3), -x.get('monthly_savings', 0))
        )

    @staticmethod
    def extract_json_from_ai_response(ai_response):
        # Use regex to find the JSON block between ```json and ```
        match = re.search(r"```json(.*?)```", ai_response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                return None
        else:
            print("No JSON block found in ai_response.")
            return None

    def generate_ai_powered_recommendations(
        self,
        usage_df: pd.DataFrame,
        current_rate: float,
        appliance_insights_prompt: str,
        user_preferences: Optional[Dict] = None
    ) -> Dict:
        """
        Generates AI-powered recommendations by calling an LLM.

        Args:
            usage_df: DataFrame with usage data
            current_rate: Current electricity rate (€/kWh)
            user_preferences: Optional user preferences and constraints

        Returns:
            Dictionary containing AI-generated recommendations and analysis
        """
        if usage_df.empty:
            logger.warning("Empty DataFrame provided for AI recommendations")
            return {"recommendations": [], "analysis": {}, "total_potential_savings": 0}

        # Initialize AI engine if not provided
        if not self.ai_engine:
            self.ai_engine = AIEngine(model="gemini-2.5-pro")

        if not self.ai_engine.client:
            logger.warning("AI Engine not initialized. Falling back to data-driven recommendations.")
            # Fallback to existing data-driven recommendations
            return self.generate_recommendations(usage_df, current_rate, user_preferences)

        # try:
        # 1. Get comprehensive analysis data
        daily_patterns = self.usage_analyzer.analyze_daily_patterns(usage_df)
        usage_stats = self.usage_analyzer.calculate_usage_stats(usage_df)
        current_costs = self.tariff_engine.calculate_simple_cost(usage_df, current_rate)
        peaks = self.usage_analyzer.identify_peaks(usage_df)

        # Prepare data for LLM prompt
        household_profile = {
            "total_energy_kwh": round(usage_stats.get('basic', {}).get('total_energy_kwh', 0), 2),
            "average_power_kw": round(usage_stats.get('basic', {}).get('average_power_kw', 0), 2),
            "current_rate_euros_per_kwh": current_rate,
            "current_monthly_cost_euros": round(current_costs.get('monthly_projection', {}).get('cost_euros', 0), 2)
        }

        usage_patterns = {
            "peak_hours_usage_kw": round(usage_stats.get('peak_hours', {}).get('peak_hours_usage', 0), 2),
            "night_usage_avg_kw": round(usage_stats.get('night_vs_day', {}).get('night_usage_avg', 0), 2),
            "day_usage_avg_kw": round(usage_stats.get('night_vs_day', {}).get('day_usage_avg', 0), 2),
            "night_day_ratio": round(usage_stats.get('night_vs_day', {}).get('night_day_ratio', 0), 2),
            "peak_to_average_ratio": round(usage_stats.get('efficiency', {}).get('peak_to_average_ratio', 0), 2)
        }

        cost_breakdown = {
            "night_percentage": current_costs.get('time_periods', {}).get('night', {}).get('percentage', 0),
            "day_percentage": current_costs.get('time_periods', {}).get('day', {}).get('percentage', 0),
            "peak_percentage": current_costs.get('time_periods', {}).get('peak', {}).get('percentage', 0),
            "evening_percentage": current_costs.get('time_periods', {}).get('evening', {}).get('percentage', 0)
        }

        # Summarize hourly patterns
        hourly_summary = {hour: round(data['mean'], 2) for hour, data in daily_patterns.get('hourly_pattern', {}).items()}

        # Summarize top peaks
        peak_summary = []
        for peak in peaks[:5]:  # Top 5 peaks
            peak_summary.append({
                "timestamp": peak.get('timestamp', ''),
                "power_kw": round(peak.get('power_kw', 0), 2),
                "duration_minutes": peak.get('duration_minutes', 0)
            })

        # Construct the prompt
        prompt_data = {
            "household_profile": household_profile,
            "usage_patterns": usage_patterns,
            "cost_breakdown": cost_breakdown,
            "hourly_usage_summary": hourly_summary,
            "top_peak_events": peak_summary,
            "user_preferences": user_preferences,
            "appliance_insights": appliance_insights_prompt
        }

        prompt_template = f"""
        You are an expert energy advisor analyzing Irish household smart meter data.
        Based on the following JSON data, provide specific, actionable energy-saving recommendations.
        The recommendations should be tailored to the user's usage patterns and the Irish electricity market.
        
        For each recommendation, include:
        1. A clear title.
        2. A detailed description of the action.
        3. An EXACT estimated monthly savings in Euros (€/month).
        4. An estimated annual savings in Euros (€/year).
        5. A difficulty level (Easy, Medium, Hard).
        6. An estimated time to implement (e.g., Immediate, 1-2 weeks, 1-3 months).
        7. Specific action items the user can take.
        8. Where applicable, an ROI analysis or payback period.
        
        The output should be a JSON array of recommendation objects.

        Add in a key "total_potential_savings" in the JSON which gives a exact number of total potential savings in a month keep the number as integer data type
        
        Here is the user's energy data:
        {prompt_data}
        
        Ensure all monetary values are clearly in Euros and savings are calculated based on the provided current rate and usage data.
        If no specific recommendations are found, return an empty array.
        """
        # Call the AI engine
        logger.info("🤖 Calling OpenAI for AI-powered recommendations...")
        ai_response = self.ai_engine.call_ai_analysis(prompt_template)
        print(ai_response)
        ai_response_json = self.extract_json_from_ai_response(ai_response)

        # Return raw AI response (no JSON parsing)
        logger.info(f"✅ AI generated response: {len(ai_response)} characters")

        return {
            'recommendations': ai_response_json.get("recommendations"),  # Empty for now, we'll show raw response
            'ai_insights': [ai_response],  # Raw AI response
            'ai_insights_json': ai_response_json,
            'total_potential_savings': ai_response_json.get("total_potential_savings"),  # Will calculate from raw response if needed
            'analysis': {
                'current_costs': current_costs,
                'usage_stats': usage_stats,
                'daily_patterns': daily_patterns,
                'llm_prompt_data': prompt_data
            },
            'generated_at': datetime.now().isoformat()
        }

        # except Exception as e:
        #     logger.error(f"Error generating AI-powered recommendations: {e}")
        #     # Fallback to data-driven recommendations
        #     logger.info("Falling back to data-driven recommendations...")
        #     return self.generate_recommendations(usage_df, current_rate, user_preferences)


    def generate_action_plan(self, recommendations: List[Dict]) -> Dict:
        """
        Generate a prioritized action plan from recommendations.

        Args:
            recommendations: List of recommendation dictionaries

        Returns:
            Dictionary containing prioritized action plan
        """
        if not recommendations:
            return {'action_plan': [], 'timeline': {}, 'total_savings': 0}

        # Group recommendations by implementation timeline
        immediate_actions = []
        short_term_actions = []
        long_term_actions = []

        for rec in recommendations:
            time_to_implement = rec.get('time_to_implement', 'Unknown')
            if 'Immediate' in time_to_implement:
                immediate_actions.append(rec)
            elif 'week' in time_to_implement.lower():
                short_term_actions.append(rec)
            else:
                long_term_actions.append(rec)

        # Calculate cumulative savings
        total_immediate_savings = sum(rec.get('monthly_savings', 0) for rec in immediate_actions)
        total_short_term_savings = sum(rec.get('monthly_savings', 0) for rec in short_term_actions)
        total_long_term_savings = sum(rec.get('monthly_savings', 0) for rec in long_term_actions)

        return {
            'action_plan': {
                'immediate': immediate_actions,
                'short_term': short_term_actions,
                'long_term': long_term_actions
            },
            'timeline': {
                'immediate': {
                    'timeframe': 'This week',
                    'savings': round(total_immediate_savings, 2),
                    'count': len(immediate_actions)
                },
                'short_term': {
                    'timeframe': 'Next 1-4 weeks',
                    'savings': round(total_short_term_savings, 2),
                    'count': len(short_term_actions)
                },
                'long_term': {
                    'timeframe': '1+ month',
                    'savings': round(total_long_term_savings, 2),
                    'count': len(long_term_actions)
                }
            },
            'total_savings': round(total_immediate_savings + total_short_term_savings + total_long_term_savings, 2),
        }


# Convenience functions for easy access
def generate_recommendations(
    usage_df: pd.DataFrame, 
    current_rate: float,
    user_preferences: Optional[Dict] = None
) -> Dict:
    """Convenience function to generate recommendations."""
    engine = RecommendationEngine()
    return engine.generate_recommendations(usage_df, current_rate, user_preferences)


def generate_action_plan(recommendations: List[Dict]) -> Dict:
    """Convenience function to generate action plan."""
    engine = RecommendationEngine()
    return engine.generate_action_plan(recommendations)


if __name__ == "__main__":
    # Test the recommendation engine
    print("AI-Powered Recommendation Engine")
    print("Available functions:")
    print("- generate_recommendations(usage_df, current_rate, user_preferences)")
    print("- generate_action_plan(recommendations)")
