"""
Tests for the AI-Powered Recommendation Engine module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recommendation_engine import RecommendationEngine, generate_recommendations, generate_action_plan


class TestRecommendationEngine:
    """Test the RecommendationEngine class"""
    
    def setup_method(self):
        """Set up test data"""
        # Create sample usage data (24 hours, 30-min intervals)
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-01 23:30:00',
            freq='30min'
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
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'import_kw': usage_pattern
        })
        
        self.engine = RecommendationEngine()
    
    def test_generate_recommendations_basic(self):
        """Test basic recommendation generation"""
        result = self.engine.generate_recommendations(self.test_df, 0.23)
        
        assert result is not None
        assert 'recommendations' in result
        assert 'analysis' in result
        assert 'total_potential_savings' in result
        assert isinstance(result['recommendations'], list)
        assert isinstance(result['analysis'], dict)
    
    def test_generate_recommendations_empty_data(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.engine.generate_recommendations(empty_df, 0.23)
        
        assert result['recommendations'] == []
        assert result['analysis'] == {}
    
    def test_tariff_recommendations(self):
        """Test tariff optimization recommendations"""
        result = self.engine.generate_recommendations(self.test_df, 0.23)
        recommendations = result['recommendations']
        
        # Should have tariff recommendations
        tariff_recs = [rec for rec in recommendations if rec['type'] == 'tariff_switch']
        assert len(tariff_recs) > 0
        
        # Check tariff recommendation structure
        for rec in tariff_recs:
            assert 'title' in rec
            assert 'monthly_savings' in rec
            assert 'annual_savings' in rec
            assert 'difficulty' in rec
            assert 'action_items' in rec
            assert 'impact_level' in rec
            assert rec['monthly_savings'] > 0
    
    def test_time_shifting_recommendations(self):
        """Test time-shifting recommendations"""
        result = self.engine.generate_recommendations(self.test_df, 0.23)
        recommendations = result['recommendations']
        
        # Should have time-shifting recommendations
        time_shift_recs = [rec for rec in recommendations if rec['type'] == 'time_shifting']
        
        # May or may not have time-shifting recommendations depending on data
        if time_shift_recs:
            for rec in time_shift_recs:
                assert 'title' in rec
                assert 'monthly_savings' in rec
                assert 'peak_hour' in rec
                assert 'action_items' in rec
    
    def test_behavioral_recommendations(self):
        """Test behavioral recommendations"""
        result = self.engine.generate_recommendations(self.test_df, 0.23)
        recommendations = result['recommendations']
        
        # Should have behavioral recommendations
        behavioral_recs = [rec for rec in recommendations if rec['type'] == 'behavioral']
        
        # May or may not have behavioral recommendations depending on data
        if behavioral_recs:
            for rec in behavioral_recs:
                assert 'title' in rec
                assert 'monthly_savings' in rec
                assert 'action_items' in rec
                assert 'current_baseline' in rec
    
    def test_impact_level_classification(self):
        """Test impact level classification"""
        # Test different savings amounts
        assert self.engine._get_impact_level(60.0) == 'High Impact'
        assert self.engine._get_impact_level(30.0) == 'Medium Impact'
        assert self.engine._get_impact_level(10.0) == 'Low Impact'
        assert self.engine._get_impact_level(2.0) == 'Minimal Impact'
    
    def test_recommendation_sorting(self):
        """Test recommendation sorting by impact"""
        recommendations = [
            {'impact_level': 'Low Impact', 'monthly_savings': 10.0},
            {'impact_level': 'High Impact', 'monthly_savings': 60.0},
            {'impact_level': 'Medium Impact', 'monthly_savings': 30.0},
        ]
        
        sorted_recs = self.engine._sort_recommendations_by_impact(recommendations)
        
        # Should be sorted by impact level first, then by savings
        assert sorted_recs[0]['impact_level'] == 'High Impact'
        assert sorted_recs[1]['impact_level'] == 'Medium Impact'
        assert sorted_recs[2]['impact_level'] == 'Low Impact'
    
    def test_generate_action_plan(self):
        """Test action plan generation"""
        recommendations = [
            {
                'type': 'behavioral',
                'title': 'Reduce Standby Power',
                'monthly_savings': 15.0,
                'time_to_implement': 'Immediate',
                'difficulty': 'Easy'
            },
            {
                'type': 'tariff_switch',
                'title': 'Switch Tariff',
                'monthly_savings': 25.0,
                'time_to_implement': '1-2 weeks',
                'difficulty': 'Easy'
            },
            {
                'type': 'efficiency',
                'title': 'Upgrade Appliances',
                'monthly_savings': 40.0,
                'time_to_implement': '1-3 months',
                'difficulty': 'Hard'
            }
        ]
        
        action_plan = self.engine.generate_action_plan(recommendations)
        
        assert 'action_plan' in action_plan
        assert 'timeline' in action_plan
        assert 'total_savings' in action_plan
        
        # Check timeline structure
        timeline = action_plan['timeline']
        assert 'immediate' in timeline
        assert 'short_term' in timeline
        assert 'long_term' in timeline
        
        # Check immediate actions
        immediate = action_plan['action_plan']['immediate']
        assert len(immediate) == 1
        assert immediate[0]['type'] == 'behavioral'
        
        # Check short term actions
        short_term = action_plan['action_plan']['short_term']
        assert len(short_term) == 1
        assert short_term[0]['type'] == 'tariff_switch'
        
        # Check long term actions
        long_term = action_plan['action_plan']['long_term']
        assert len(long_term) == 1
        assert long_term[0]['type'] == 'efficiency'
        
        # Check total savings
        assert action_plan['total_savings'] == 80.0  # 15 + 25 + 40


class TestConvenienceFunctions:
    """Test the convenience functions"""
    
    def setup_method(self):
        """Set up test data"""
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-01 23:30:00',
            freq='30min'
        )
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'import_kw': [1.0] * len(dates)  # 1 kW constant usage
        })
    
    def test_generate_recommendations_function(self):
        """Test the convenience function for generating recommendations"""
        result = generate_recommendations(self.test_df, 0.23)
        
        assert result is not None
        assert 'recommendations' in result
        assert 'analysis' in result
    
    def test_generate_action_plan_function(self):
        """Test the convenience function for generating action plans"""
        recommendations = [
            {
                'type': 'behavioral',
                'title': 'Test Recommendation',
                'monthly_savings': 20.0,
                'time_to_implement': 'Immediate',
                'difficulty': 'Easy'
            }
        ]
        
        action_plan = generate_action_plan(recommendations)
        
        assert action_plan is not None
        assert 'action_plan' in action_plan
        assert 'total_savings' in action_plan


if __name__ == "__main__":
    pytest.main([__file__])


class TestRecommendationEngineDataStructures:
    """Test recommendation engine with different data structures to catch type errors"""
    
    def setup_method(self):
        """Set up test data with realistic usage patterns"""
        # Create sample usage data (24 hours, 30-min intervals)
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-01 23:30:00',
            freq='30min'
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
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'import_kw': usage_pattern
        })
        
        self.engine = RecommendationEngine()
    
    def test_hourly_pattern_data_structure(self):
        """Test that hourly pattern data structure is handled correctly"""
        # This test specifically checks for the dict+dict error we fixed
        daily_patterns = self.engine.usage_analyzer.analyze_daily_patterns(self.test_df)
        hourly_pattern = daily_patterns.get('hourly_pattern', {})
        
        # Verify hourly_pattern contains dictionaries with expected keys
        assert isinstance(hourly_pattern, dict)
        for hour, data in hourly_pattern.items():
            assert isinstance(data, dict)
            assert 'mean' in data
            assert 'std' in data
            assert 'min' in data
            assert 'max' in data
            assert isinstance(data['mean'], (int, float, np.number))
    
    def test_time_shifting_with_dict_hourly_pattern(self):
        """Test time-shifting recommendations with dict-based hourly pattern"""
        daily_patterns = self.engine.usage_analyzer.analyze_daily_patterns(self.test_df)
        
        # This should not raise a TypeError about dict + dict
        recommendations = self.engine._generate_time_shifting_recommendations(
            self.test_df, 0.23, daily_patterns
        )
        
        assert isinstance(recommendations, list)
        # Should handle the dict structure without errors
    
    def test_tariff_costs_data_structure(self):
        """Test that tariff cost data structures are handled correctly"""
        current_costs = self.engine.tariff_engine.calculate_simple_cost(self.test_df, 0.23)
        
        # Verify current_costs structure
        assert isinstance(current_costs, dict)
        assert 'monthly_projection' in current_costs
        assert 'cost_euros' in current_costs['monthly_projection']
        assert isinstance(current_costs['monthly_projection']['cost_euros'], (int, float, np.number))
        
        # Test time-based costs structure
        time_costs = self.engine.tariff_engine.calculate_time_based_cost(
            self.test_df, 0.25, 0.15, 0.30
        )
        
        assert isinstance(time_costs, dict)
        assert 'total_cost_euros' in time_costs
        assert isinstance(time_costs['total_cost_euros'], (int, float, np.number))
    
    def test_tariff_recommendations_with_correct_structures(self):
        """Test tariff recommendations with correct data structures"""
        current_costs = self.engine.tariff_engine.calculate_simple_cost(self.test_df, 0.23)
        
        # This should not raise any type errors
        recommendations = self.engine._generate_tariff_recommendations(
            self.test_df, 0.23, current_costs
        )
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert 'monthly_savings' in rec
            assert 'annual_savings' in rec
            assert isinstance(rec['monthly_savings'], (int, float))
            assert isinstance(rec['annual_savings'], (int, float))
    
    def test_usage_stats_data_structure(self):
        """Test that usage stats data structure is handled correctly"""
        usage_stats = self.engine.usage_analyzer.calculate_usage_stats(self.test_df)
        
        # Verify usage_stats structure
        assert isinstance(usage_stats, dict)
        assert 'basic' in usage_stats
        assert 'efficiency' in usage_stats
        
        basic_stats = usage_stats['basic']
        assert 'total_energy_kwh' in basic_stats
        assert 'average_power_kw' in basic_stats
        assert isinstance(basic_stats['total_energy_kwh'], (int, float, np.number))
        assert isinstance(basic_stats['average_power_kw'], (int, float, np.number))
        
        efficiency_stats = usage_stats['efficiency']
        assert 'usage_variability' in efficiency_stats
        assert 'peak_to_average_ratio' in efficiency_stats
    
    def test_load_balancing_with_correct_structures(self):
        """Test load balancing recommendations with correct data structures"""
        usage_stats = self.engine.usage_analyzer.calculate_usage_stats(self.test_df)
        
        # This should not raise any type errors
        recommendations = self.engine._generate_load_balancing_recommendations(
            self.test_df, 0.23, usage_stats
        )
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert 'monthly_savings' in rec
            assert isinstance(rec['monthly_savings'], (int, float))
    
    def test_behavioral_recommendations_with_correct_structures(self):
        """Test behavioral recommendations with correct data structures"""
        usage_stats = self.engine.usage_analyzer.calculate_usage_stats(self.test_df)
        
        # This should not raise any type errors
        recommendations = self.engine._generate_behavioral_recommendations(
            self.test_df, 0.23, usage_stats
        )
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert 'monthly_savings' in rec
            assert isinstance(rec['monthly_savings'], (int, float))
    
    def test_full_recommendation_generation_no_errors(self):
        """Test that full recommendation generation works without type errors"""
        # This should not raise the "unsupported operand type(s) for +: 'dict' and 'dict'" error
        result = self.engine.generate_recommendations(self.test_df, 0.23)
        
        assert isinstance(result, dict)
        assert 'recommendations' in result
        assert 'total_potential_savings' in result
        assert isinstance(result['recommendations'], list)
        assert isinstance(result['total_potential_savings'], (int, float))
        
        # Verify all recommendations have correct data types
        for rec in result['recommendations']:
            assert 'monthly_savings' in rec
            assert 'annual_savings' in rec
            assert isinstance(rec['monthly_savings'], (int, float))
            assert isinstance(rec['annual_savings'], (int, float))
    
    def test_action_plan_generation_no_errors(self):
        """Test that action plan generation works without type errors"""
        # Generate recommendations first
        result = self.engine.generate_recommendations(self.test_df, 0.23)
        recommendations = result['recommendations']
        
        # This should not raise any type errors
        action_plan = self.engine.generate_action_plan(recommendations)
        
        assert isinstance(action_plan, dict)
        assert 'action_plan' in action_plan
        assert 'timeline' in action_plan
        assert 'total_savings' in action_plan
        assert isinstance(action_plan['total_savings'], (int, float))


class TestRecommendationEngineEdgeCases:
    """Test recommendation engine with edge cases and error conditions"""
    
    def setup_method(self):
        """Set up test data"""
        # Create minimal test data
        dates = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-01-01 23:30:00',
            freq='30min'
        )
        
        self.minimal_df = pd.DataFrame({
            'timestamp': dates,
            'import_kw': [0.1] * len(dates)  # Very low usage
        })
        
        self.engine = RecommendationEngine()
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.engine.generate_recommendations(empty_df, 0.23)
        
        assert result['recommendations'] == []
        assert result['total_potential_savings'] == 0
    
    def test_minimal_usage_data(self):
        """Test with minimal usage data that might not generate recommendations"""
        result = self.engine.generate_recommendations(self.minimal_df, 0.23)
        
        assert isinstance(result, dict)
        assert 'recommendations' in result
        # Should handle gracefully even if no recommendations are generated
    
    def test_very_high_rates(self):
        """Test with very high electricity rates"""
        result = self.engine.generate_recommendations(self.minimal_df, 1.0)  # €1/kWh
        
        assert isinstance(result, dict)
        assert 'recommendations' in result
        # Should handle high rates without errors
    
    def test_zero_rate(self):
        """Test with zero rate (edge case)"""
        result = self.engine.generate_recommendations(self.minimal_df, 0.0)
        
        assert isinstance(result, dict)
        assert 'recommendations' in result
        # Should handle zero rates gracefully
