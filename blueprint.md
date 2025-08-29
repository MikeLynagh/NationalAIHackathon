Perfect! I have all the decisions needed. Here's the complete blueprint.md:

Energy Advisor MVP - 3-Day Build Blueprint
Project Overview

Goal: Personal energy analyzer that uploads Irish MPRN smart meter data, analyzes usage patterns, detects appliances, and provides specific money-saving
recommendations.

Timeline: 3 days (48 hours)

Tech Stack: Streamlit, Python, pandas, plotly

Approach: Heuristic appliance detection + public profiles, advanced savings calculations

Technical Decisions Summary
- [ ] **Appliance Detection**: AI-powered pattern recognition + heuristic rules for EV, washing, heating
- [ ] **Personalization**: AI clustering analysis + 50 similar households benchmarking
- [ ] **Recommendations**: AI-generated insights with specific timing windows and €€ impact
- [ ] **Savings Calculation**: AI-optimized usage patterns + advanced tariff optimization
- [ ] **Data Processing**: AI-enhanced analysis + client-side Streamlit, supports annual data (~17k rows)
- [ ] **File Format**: Your MPRN sample format (30-min intervals, Import/Export kW)
- [ ] **AI Integration**: OpenAI GPT-4 or Claude for advanced pattern analysis and insights

Day 1: Core Data Pipeline (16 hours)
Module 1: File Parser & Validation (4 hours) ✅ COMPLETED

File:
src/data_parser.py

- [x] def parse_mprn_file(uploaded_file) -> pd.DataFrame
- [x] def validate_mprn_data(df) -> dict  # Returns validation results
- [x] def clean_and_resample(df) -> pd.DataFrame  # 30-min intervals, handle missing

Deliverables:

- [x] Parse MPRN CSV format (Read Value, Read Type, Read Date and End Time)
- [x] Pivot Import/Export by timestamp
- [x] Validate: check date ranges, detect outliers >15kW, missing intervals
- [x] Clean: fill small gaps, flag large gaps, ensure 30-min intervals
- [x] Error handling: clear messages for bad files

Module 2: Tariff Engine (6 hours)

File:
src/tariff_engine.py

- [ ] def load_tariff_plans(excel_path) -> pd.DataFrame
- [ ] def parse_tariff_plan(plan_row) -> dict  # Extract time windows, rates
- [ ] def generate_price_timeseries(plan, date_range) -> pd.Series
- [ ] def calculate_bill(usage_df, price_series, plan_details) -> dict

Deliverables:

- [ ] Load your Tariff Plans.xlsx
- [ ] Parse time windows (Day: 8:00-23:00, Peak: 17:00-19:00, etc.)
- [ ] Handle overlapping windows (precedence: Peak > EV > Day > Night)
- [ ] Generate 30-min price timeseries for any date range
- [ ] Calculate: energy cost, standing charges, PSO, discounts, export credits

Module 3: Usage Analysis Engine (6 hours)

File:
src/usage_analyzer.py

- [ ] def analyze_daily_patterns(usage_df) -> dict
- [ ] def identify_peaks(usage_df) -> list  # Peak times and magnitudes
- [ ] def calculate_usage_stats(usage_df) -> dict  # Total, avg, max, time-of-use breakdown
- [ ] def classify_user_type(usage_stats) -> str  # Night-heavy, day-heavy, balanced

Deliverables:

- [ ] Daily/weekly pattern analysis
- [ ] Peak usage identification (when, how much)
- [ ] Time-of-use breakdown (day/night/peak percentages)
- [ ] User classification for benchmarking

End of Day 1 Target: Can upload MPRN file, apply tariff, show basic usage stats and current bill breakdown.

Day 2: Appliance Detection & Recommendations (16 hours)
Module 4: AI-Powered Appliance Detection (8 hours)

File:
src/appliance_detector.py

- [ ] def detect_ev_charging(usage_df) -> list  # AI + heuristic: High sustained loads at night
- [ ] def detect_washing_cycles(usage_df) -> list  # AI + heuristic: 1-2kW bursts, 90min duration
- [ ] def detect_heating_loads(usage_df) -> list  # AI + heuristic: 2-4kW sustained, often night
- [ ] def get_appliance_profiles() -> dict  # Load typical profiles from data
- [ ] def ai_analyze_patterns(usage_df) -> dict  # AI-powered pattern recognition
- [ ] def ai_identify_anomalies(usage_df) -> list  # Detect unusual usage patterns
- [ ] def ai_classify_appliances(usage_df) -> dict  # AI classification of detected loads

AI Integration:
- [ ] def call_ai_analysis(usage_data, prompt_template) -> dict  # OpenAI/Claude API calls
- [ ] def generate_ai_insights(usage_df, detected_appliances) -> list  # AI-generated insights
- [ ] def ai_optimize_detection_rules(usage_df) -> dict  # AI-optimized detection parameters

Appliance Profiles (AI-Enhanced):

- [ ] EV: AI-learned patterns + 7kW average, 4-6 hours, flexibility: high (any night window)
- [ ] Washing machine: AI-learned patterns + 1.8kW average, 90 minutes, flexibility: medium (6-hour window)
- [ ] Immersion heater: AI-learned patterns + 3kW average, 2-4 hours, flexibility: high (night rate window)
- [ ] Solar generation: AI-detected patterns, weather correlation, seasonal variations
- [ ] Unknown loads: AI classification of unidentified high-consumption patterns

Detection Logic (AI + Heuristic):

- [ ] EV: AI pattern recognition + consecutive slots >6kW for >3 hours, 80% during 22:00-08:00
- [ ] Washing: AI pattern recognition + 1-2.5kW bursts lasting 60-120 minutes, identifiable start spike
- [ ] Heating: AI pattern recognition + sustained 2-4kW loads, often coinciding with low temperature periods
- [ ] Solar: AI correlation with daylight hours, weather data, seasonal patterns
- [ ] Anomalies: AI detection of unusual consumption spikes, missing patterns, outliers

Module 5: AI-Powered Recommendation Engine (8 hours)

File:
src/recommendation_engine.py

- [ ] def find_optimal_tariff(usage_df, all_tariffs) -> dict
- [ ] def optimize_appliance_timing(usage_df, appliances, tariff) -> list
- [ ] def calculate_savings_potential(current_bill, optimized_usage, new_tariff) -> dict
- [ ] def generate_action_plan(recommendations) -> list
- [ ] def ai_generate_recommendations(usage_df, appliances, tariffs) -> dict  # AI-powered insights
- [ ] def ai_optimize_load_shifting(usage_df, detected_appliances) -> list  # AI-optimized timing
- [ ] def ai_calculate_roi_analysis(usage_df, recommendations) -> dict  # AI ROI calculations

AI-Enhanced Recommendation Types:

- [ ] **Tariff Switch**: AI analysis of usage patterns + tariff comparison, rank by savings
- [ ] **Time Shifting**: AI-optimized appliance timing with specific €€ impact and timing windows
- [ ] **Load Balancing**: AI-predicted peak avoidance strategies with behavioral insights
- [ ] **Behavioral**: AI-generated specific actions with time windows, savings estimates, and difficulty ratings
- [ ] **Predictive**: AI forecasting of future usage patterns and seasonal optimization opportunities
- [ ] **Anomaly-Based**: AI detection of unusual patterns leading to specific optimization recommendations

AI-Enhanced Savings Calculation:

- [ ] **Baseline**: Current usage pattern + current tariff = current bill
- [ ] **Tariff optimization**: AI-analyzed usage + best tariff = tariff savings
- [ ] **Usage optimization**: AI-optimized appliance timing + best tariff = total savings
- [ ] **AI insights**: Advanced pattern recognition + behavioral recommendations = enhanced savings
- [ ] **Predictive modeling**: AI forecasting of seasonal variations + optimization opportunities
- [ ] **ROI analysis**: AI-calculated payback periods, investment recommendations, future value projections

End of Day 2 Target: Can detect major appliances, recommend optimal tariff, suggest time shifts with savings estimates.

Day 3: UI, Benchmarking & Integration (16 hours)
Module 6: Benchmarking Engine (4 hours)

File:
src/benchmarking.py

- [ ] def load_reference_households(sample_data_path) -> pd.DataFrame  # 50 houses
- [ ] def find_similar_households(user_stats, reference_data) -> list
- [ ] def calculate_percentiles(user_stats, similar_households) -> dict
- [ ] def generate_comparison_insights(percentiles) -> list

Clustering Approach:

- [ ] Features: total monthly usage, night/day ratio, peak usage, weekday/weekend pattern
- [ ] Find 10-15 most similar households from your 50-house dataset
- [ ] Calculate percentiles: usage rank, cost rank, efficiency rank
- [ ] Generate insights: "You use 25% more than similar homes", "Your night usage is above average"

Module 7: Streamlit UI (8 hours)

File:
streamlit_app.py

# Page structure:
# 1. Upload & Setup (tariff selection)
# 2. Usage Analysis (patterns, bill breakdown)  
# 3. Savings Opportunities (recommendations with €€ impact)
# 4. Action Plan (prioritized steps)

Page 1: Upload & Setup

- [ ] File upload widget with validation feedback
- [ ] Tariff selection dropdown (from Excel, filtered by Urban/Rural)
- [ ] Data preview: show parsed usage, date range, basic stats

Page 2: Usage Analysis

- [ ] Daily usage pattern chart (24-hour profile)
- [ ] Weekly pattern chart (7 days overlay)
- [ ] Bill breakdown: energy cost, standing charges, peak charges
- [ ] Appliance detection results: detected devices with typical run times
- [ ] Comparison to similar homes: percentile rankings

Page 3: Savings Opportunities

- [ ] Tariff comparison table: current vs top 3 alternatives with annual savings
- [ ] Time-shifting recommendations: specific appliances with optimal windows
- [ ] Peak avoidance suggestions: spread usage to avoid peak charges
- [ ] Total savings potential: monthly and annual estimates

Page 4: Action Plan

- [ ] Prioritized recommendations by savings impact
- [ ] Specific instructions: "Move EV charging to 2:00-6:00 AM window to save €25/month"
- [ ] Implementation difficulty ratings: Easy/Medium/Hard
- [ ] Monthly tracking: estimated vs actual savings (placeholder for future)

Module 8: AI Integration & Testing (4 hours) ✅ COMPLETED

File:
src/ai_engine.py

- [x] def setup_ai_client(api_key: str, model: str) -> object  # OpenAI/Claude client setup
- [x] def analyze_usage_patterns_ai(usage_df: pd.DataFrame) -> dict  # AI pattern analysis
- [x] def generate_appliance_insights_ai(usage_df: pd.DataFrame, detected_appliances: list) -> dict  # AI insights
- [x] def optimize_recommendations_ai(usage_df: pd.DataFrame, current_recommendations: list) -> list  # AI optimization
- [x] def generate_narrative_report_ai(analysis_results: dict) -> str  # AI-generated narrative

File:
tests/test_pipeline.py

# End-to-end testing with your sample data
# Edge case handling: incomplete data, extreme values, missing tariff info
# Performance testing: annual data processing speed
# UI testing: file upload, navigation, chart rendering
# AI integration testing: API calls, response handling, error scenarios

Test Cases:

- [ ] Valid MPRN file → complete analysis
- [ ] File with missing intervals → graceful handling
- [ ] Extreme usage values → outlier flagging
- [ ] Different tariff types → correct price calculation
- [ ] No detected appliances → general recommendations only
- [ ] AI API integration → successful analysis and insights
- [ ] AI error handling → graceful degradation when API unavailable
- [ ] AI response parsing → correct extraction of insights and recommendations

End of Day 3 Target: Complete working MVP with polished UI, tested on sample data, ready for demo.

File Structure
- [x] energy_advisor_mvp/
- [x] ├── streamlit_app.py           # Main Streamlit application (TO BE CREATED)
- [x] ├── src/
- [x] │   ├── data_parser.py         # MPRN file parsing & validation ✅
- [ ] │   ├── tariff_engine.py       # Tariff calculations & pricing
- [ ] │   ├── usage_analyzer.py      # Usage pattern analysis
- [ ] │   ├── appliance_detector.py  # AI-powered appliance detection
- [ ] │   ├── recommendation_engine.py # AI-powered savings recommendations
- [ ] │   ├── benchmarking.py        # Similar household comparisons
- [x] │   └── ai_engine.py           # AI integration and analysis engine ✅
- [x] ├── data/
- [ ] │   ├── tariff_plans.xlsx      # Your tariff dataset
- [ ] │   ├── appliance_profiles.json # Typical appliance signatures
- [ ] │   ├── reference_households.csv # 50 houses for benchmarking
- [x] │   └── sample_mprn.csv        # Your sample MPRN data ✅
- [x] ├── tests/
- [x] │   └── test_data_parser.py    # Data parser testing ✅
- [x] ├── .env.example               # Environment variables template ✅
- [x] ├── requirements.txt           # Dependencies ✅
- [ ] └── README.md                  # Setup and usage instructions

Dependencies (requirements.txt)
- [ ] streamlit>=1.28.0
- [ ] pandas>=2.0.0
- [ ] numpy>=1.24.0
- [ ] plotly>=5.15.0
- [ ] openpyxl>=3.1.0  # For Excel reading
- [ ] python-dateutil>=2.8.0
- [ ] openai>=1.0.0  # For GPT-4 API integration
- [ ] anthropic>=0.7.0  # For Claude API integration (alternative)
- [ ] python-dotenv>=1.0.0  # For environment variable management
- [ ] requests>=2.31.0  # For API calls
- [ ] scikit-learn>=1.3.0  # For ML clustering and pattern recognition

Key Data Flows
1. Upload → Analysis Flow
- [ ] MPRN CSV → parse_mprn_file() → validate_mprn_data() → clean_and_resample()
- [ ] → analyze_daily_patterns() → detect_appliances() → classify_user_type()

2. Tariff → Pricing Flow
- [ ] Tariff Excel → load_tariff_plans() → user selects plan → parse_tariff_plan()
- [ ] → generate_price_timeseries() → calculate_bill() (current cost)

3. Recommendations Flow
- [ ] Usage + Tariffs → find_optimal_tariff() → optimize_appliance_timing()
- [ ] → calculate_savings_potential() → generate_action_plan()

4. Benchmarking Flow
- [ ] User stats → find_similar_households() → calculate_percentiles()
- [ ] → generate_comparison_insights() → display in UI

Success Criteria

- [ ] Day 1: Core pipeline works, can show current bill breakdown
- [ ] Day 2: Appliance detection works, can recommend tariff + time shifts with savings
- [ ] Day 3: Complete UI, benchmarking, tested and demo-ready

Demo Script:

- [ ] Upload sample MPRN file → shows parsed data preview
- [ ] Select current tariff → shows current bill breakdown
- [ ] View analysis → shows patterns, detected appliances, vs similar homes
- [ ] View recommendations → shows tariff savings + time-shift savings
- [ ] View action plan → shows specific steps to save €X/month

Risk Mitigation
- [ ] Appliance detection accuracy: Start with conservative rules, refine based on testing
- [ ] Savings calculation errors: Show ranges (±20%) and methodology transparency
- [ ] Performance issues: Test with full annual datasets, optimize pandas operations
- [ ] UI complexity: Keep simple, focus on key insights, avoid feature creep
- [ ] Data quality: Robust validation and clear error messages for bad uploads