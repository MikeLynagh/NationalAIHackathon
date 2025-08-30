"""
AI Engine Module

This module integrates with OpenAI GPT-4 or Anthropic Claude to provide
advanced pattern analysis, appliance detection insights, and optimization recommendations.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import time

# Try to import AI libraries
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIEngine:
    """AI-powered analysis engine for energy usage patterns."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize AI engine.

        Args:
            api_key: API key for OpenAI or Anthropic
            model: Model to use (gpt-4, claude-3-sonnet, etc.)
        """
        self.api_key = (
            api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.client = None
        self.provider = None

        # Initialize AI client
        self._setup_ai_client()

        # Configuration
        self.max_tokens = int(os.getenv("AI_MAX_TOKENS", 4000))
        self.temperature = float(os.getenv("AI_TEMPERATURE", 0.3))
        self.timeout = int(os.getenv("AI_TIMEOUT_SECONDS", 30))
        self.enable_fallback = os.getenv("ENABLE_AI_FALLBACK", "true").lower() == "true"

    def _setup_ai_client(self):
        """Setup AI client based on available providers and API key."""
        if not self.api_key:
            logger.warning("No API key provided. AI features will be disabled.")
            return

        # Try OpenAI first
        if OPENAI_AVAILABLE and self.api_key.startswith("sk-"):
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                self.provider = "openai"
                logger.info("OpenAI client initialized successfully")
                return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

        # Try Anthropic
        if ANTHROPIC_AVAILABLE and self.api_key.startswith("sk-ant-"):
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.provider = "anthropic"
                logger.info("Anthropic client initialized successfully")
                return
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

        logger.error("No valid AI provider could be initialized")

    def analyze_usage_patterns_ai(self, usage_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze usage patterns using AI.

        Args:
            usage_df: Processed usage DataFrame with import/export columns

        Returns:
            Dict: AI-generated analysis results
        """
        if not self.client:
            logger.warning("AI client not available, using fallback analysis")
            return self._fallback_pattern_analysis(usage_df)

        try:
            # Prepare data summary for AI
            data_summary = self._prepare_data_summary(usage_df)

            # Create prompt for pattern analysis
            prompt = self._create_pattern_analysis_prompt(data_summary)

            # Get AI response
            response = self._call_ai_api(prompt)

            # Parse and structure response
            analysis = self._parse_ai_response(response, "pattern_analysis")

            return analysis

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            if self.enable_fallback:
                return self._fallback_pattern_analysis(usage_df)
            else:
                raise

    def generate_appliance_insights_ai(
        self, usage_df: pd.DataFrame, detected_appliances: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate AI-powered insights about detected appliances.

        Args:
            usage_df: Processed usage DataFrame
            detected_appliances: List of detected appliances

        Returns:
            Dict: AI-generated appliance insights
        """
        if not self.client:
            return self._fallback_appliance_insights(detected_appliances)

        try:
            # Prepare appliance data for AI
            appliance_summary = self._prepare_appliance_summary(
                usage_df, detected_appliances
            )

            # Create prompt for appliance insights
            prompt = self._create_appliance_insights_prompt(appliance_summary)

            # Get AI response
            response = self._call_ai_api(prompt)

            # Parse and structure response
            insights = self._parse_ai_response(response, "appliance_insights")

            return insights

        except Exception as e:
            logger.error(f"AI appliance insights failed: {e}")
            return self._fallback_appliance_insights(detected_appliances)

    def optimize_recommendations_ai(
        self, usage_df: pd.DataFrame, current_recommendations: List[Dict]
    ) -> List[Dict]:
        """
        Optimize recommendations using AI.

        Args:
            usage_df: Processed usage DataFrame
            current_recommendations: List of current recommendations

        Returns:
            List: AI-optimized recommendations
        """
        if not self.client:
            return current_recommendations

        try:
            # Prepare recommendation data for AI
            rec_summary = self._prepare_recommendation_summary(
                usage_df, current_recommendations
            )

            # Create prompt for optimization
            prompt = self._create_optimization_prompt(rec_summary)

            # Get AI response
            response = self._call_ai_api(prompt)

            # Parse and structure response
            optimized = self._parse_ai_response(response, "recommendations")

            return optimized.get("recommendations", current_recommendations)

        except Exception as e:
            logger.error(f"AI optimization failed: {e}")
            return current_recommendations

    def generate_narrative_report_ai(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate narrative report using AI.

        Args:
            analysis_results: Complete analysis results

        Returns:
            str: AI-generated narrative report
        """
        if not self.client:
            return self._fallback_narrative_report(analysis_results)

        try:
            # Create prompt for narrative generation
            prompt = self._create_narrative_prompt(analysis_results)

            # Get AI response
            response = self._call_ai_api(prompt)

            # Extract narrative text
            narrative = self._extract_narrative_text(response)

            return narrative

        except Exception as e:
            logger.error(f"AI narrative generation failed: {e}")
            return self._fallback_narrative_report(analysis_results)

    def _call_ai_api(self, prompt: str) -> str:
        """Make API call to AI provider."""
        start_time = time.time()

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            raise

        finally:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logger.warning(
                    f"AI API call took {elapsed:.2f}s (exceeded {self.timeout}s timeout)"
                )

    def _prepare_data_summary(self, usage_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data summary for AI analysis."""
        return {
            "total_rows": len(usage_df),
            "date_range": {
                "start": usage_df["Read Date and End Time"].min().isoformat(),
                "end": usage_df["Read Date and End Time"].max().isoformat(),
            },
            "import_stats": {
                "mean": float(usage_df["import_kw"].mean()),
                "max": float(usage_df["import_kw"].max()),
                "total": float(usage_df["import_kw"].sum()),
            },
            "export_stats": {
                "mean": float(usage_df["export_kw"].mean()),
                "max": float(usage_df["export_kw"].max()),
                "total": float(usage_df["export_kw"].sum()),
            },
            "hourly_patterns": self._calculate_hourly_patterns(usage_df),
        }

    def _calculate_hourly_patterns(
        self, usage_df: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Calculate hourly usage patterns."""
        usage_df["hour"] = usage_df["Read Date and End Time"].dt.hour

        hourly_import = usage_df.groupby("hour")["import_kw"].mean().tolist()
        hourly_export = usage_df.groupby("hour")["export_kw"].mean().tolist()

        return {"import_by_hour": hourly_import, "export_by_hour": hourly_export}

    def _create_pattern_analysis_prompt(self, data_summary: Dict) -> str:
        """Create prompt for pattern analysis."""
        return f"""
        Analyze this Irish smart meter data and provide insights about energy usage patterns:

        Data Summary:
        - Total readings: {data_summary['total_rows']}
        - Date range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}
        - Import: Mean {data_summary['import_stats']['mean']:.3f} kW, Max {data_summary['import_stats']['max']:.3f} kW
        - Export: Mean {data_summary['export_stats']['mean']:.3f} kW, Max {data_summary['export_stats']['max']:.3f} kW

        Hourly Patterns:
        Import by hour: {data_summary['hourly_patterns']['import_by_hour']}
        Export by hour: {data_summary['hourly_patterns']['export_by_hour']}

        Please provide:
        1. Key usage patterns and trends
        2. Peak usage times and likely causes
        3. Solar generation patterns (if export > 0)
        4. Recommendations for energy optimization
        5. Estimated cost implications

        Format your response as JSON with these keys:
        - patterns: list of identified patterns
        - peaks: peak usage analysis
        - solar_analysis: solar generation insights
        - recommendations: optimization suggestions
        - cost_implications: estimated financial impact
        """

    def _parse_ai_response(self, response: str, response_type: str) -> Dict[str, Any]:
        """Parse AI response based on expected format."""
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # Fallback: return raw response
                return {"raw_response": response, "type": response_type}
        except json.JSONDecodeError:
            logger.warning(f"Could not parse AI response as JSON: {response[:100]}...")
            return {"raw_response": response, "type": response_type}

    def _fallback_pattern_analysis(self, usage_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback pattern analysis when AI is unavailable."""
        return {
            "patterns": ["AI analysis unavailable - using basic statistical analysis"],
            "peaks": {"message": "Basic peak detection only"},
            "solar_analysis": {"message": "Basic solar pattern detection"},
            "recommendations": ["Enable AI integration for advanced insights"],
            "cost_implications": {"message": "Basic cost calculation only"},
        }

    def _fallback_appliance_insights(
        self, detected_appliances: List[Dict]
    ) -> Dict[str, Any]:
        """Fallback appliance insights when AI is unavailable."""
        return {
            "insights": ["AI insights unavailable - using basic appliance detection"],
            "optimization_tips": ["Enable AI integration for advanced recommendations"],
            "appliances": detected_appliances,
        }

    def _fallback_narrative_report(self, analysis_results: Dict[str, Any]) -> str:
        """Fallback narrative report when AI is unavailable."""
        return f"""
        Energy Usage Analysis Report
        
        This report was generated using basic analysis methods.
        Enable AI integration for more detailed insights and recommendations.
        
        Summary: {len(analysis_results)} analysis components completed.
        """

    def _prepare_appliance_summary(
        self, usage_df: pd.DataFrame, detected_appliances: List[Dict]
    ) -> Dict[str, Any]:
        """Prepare appliance summary for AI analysis."""
        return {
            "detected_appliances": detected_appliances,
            "usage_context": self._prepare_data_summary(usage_df),
        }

    def _create_appliance_insights_prompt(self, appliance_summary: Dict) -> str:
        """Create prompt for appliance insights."""
        return f"""
        Analyze these detected appliances and provide optimization insights:

        Detected Appliances: {appliance_summary['detected_appliances']}
        
        Please provide:
        1. Appliance-specific optimization tips
        2. Timing recommendations for cost savings
        3. Load shifting opportunities
        4. Estimated savings potential
        
        Format as JSON with keys: insights, optimization_tips, load_shifting, savings_estimate
        """

    def _prepare_recommendation_summary(
        self, usage_df: pd.DataFrame, recommendations: List[Dict]
    ) -> Dict[str, Any]:
        """Prepare recommendation summary for AI optimization."""
        return {
            "current_recommendations": recommendations,
            "usage_context": self._prepare_data_summary(usage_df),
        }

    def _create_optimization_prompt(self, rec_summary: Dict) -> str:
        """Create prompt for recommendation optimization."""
        return f"""
        Optimize these energy recommendations based on usage patterns:

        Current Recommendations: {rec_summary['current_recommendations']}
        
        Please provide:
        1. Prioritized recommendations by impact
        2. Specific timing suggestions
        3. Estimated savings with timing
        4. Implementation difficulty ratings
        
        Format as JSON with key 'recommendations' containing optimized list
        """

    def _create_narrative_prompt(self, analysis_results: Dict[str, Any]) -> str:
        """Create prompt for narrative report generation."""
        return f"""
        Generate a comprehensive narrative report for this energy analysis:

        Analysis Results: {analysis_results}
        
        Please create a professional report that:
        1. Summarizes key findings
        2. Explains patterns in simple terms
        3. Provides actionable recommendations
        4. Estimates financial impact
        5. Suggests next steps
        
        Write in clear, professional language suitable for homeowners.
        """

    def _extract_narrative_text(self, response: str) -> str:
        """Extract narrative text from AI response."""
        # Remove any JSON formatting if present
        if response.startswith("```json"):
            response = response.split("```json")[1]
        if response.endswith("```"):
            response = response[:-3]

        return response.strip()


# Convenience functions
def setup_ai_client(api_key: str, model: str) -> AIEngine:
    """Setup AI client with specified configuration."""
    return AIEngine(api_key=api_key, model=model)


def analyze_usage_patterns_ai(usage_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze usage patterns using AI."""
    engine = AIEngine()
    return engine.analyze_usage_patterns_ai(usage_df)


def generate_appliance_insights_ai(
    usage_df: pd.DataFrame, detected_appliances: List[Dict]
) -> Dict[str, Any]:
    """Generate AI-powered appliance insights."""
    engine = AIEngine()
    return engine.generate_appliance_insights_ai(usage_df, detected_appliances)


def optimize_recommendations_ai(
    usage_df: pd.DataFrame, current_recommendations: List[Dict]
) -> List[Dict]:
    """Optimize recommendations using AI."""
    engine = AIEngine()
    return engine.optimize_recommendations_ai(usage_df, current_recommendations)


def generate_narrative_report_ai(analysis_results: Dict[str, Any]) -> str:
    """Generate narrative report using AI."""
    engine = AIEngine()
    return engine.generate_narrative_report_ai(analysis_results)
