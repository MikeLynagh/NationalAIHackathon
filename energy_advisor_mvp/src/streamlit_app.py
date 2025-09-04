"""
Energy Advisor MVP - Streamlit Application
Main application for analyzing Irish MPRN smart meter data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
load_dotenv()
import logging
from datetime import datetime
import sys
import os
import asyncio
import time
import random
from typing import Dict
import joblib

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_parser import (
    MPRNDataParser,
    parse_mprn_file,
    validate_mprn_data,
    clean_and_resample,
)
from usage_analyzer import UsageAnalyzer
from tariff_engine import calculate_simple_cost, calculate_time_based_cost
from forecast_consumption import ForecastConsumption

# Import the smart agent components
try:
    from main import EnhancedSmartPlugAgent
    SMART_AGENT_AVAILABLE = True
except Exception as e:
    SMART_AGENT_AVAILABLE = False
    logging.warning(f"Smart Agent not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Energy Advisor MVP",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main application function"""
    st.title("⚡ Energy Advisor MVP")
    st.markdown("**Personal energy analyzer for Irish MPRN smart meter data**")

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "📊 Data Upload & Analysis",
            "💡 Suggested Tariff Plan",
            "💰 Usage Patterns",
            "💰 Forecast & Cost Analysis",
            "🔍 Appliance Detection",
            "💡 Recommendations",
            "🤖 Smart Agent Scheduler",
        ],
    )

    if page == "📊 Data Upload & Analysis":
        show_data_upload_page()
    elif page == "💡 Suggested Tariff Plan":
        tariff_comparison_page()
    elif page == "💰 Usage Patterns":
        show_usage_patterns_page()
    elif page == "💰 Forecast & Cost Analysis":
        show_cost_analysis_page()
    elif page == "🔍 Appliance Detection":
        show_appliance_detection_page()
    elif page == "💡 Recommendations":
        show_recommendations_page()
    elif page == "🤖 Smart Agent Scheduler":
        show_smart_agent_scheduler_page()


def show_data_upload_page():
    """Data upload and initial analysis page"""
    st.header("📊 Data Upload & Analysis")

    # File upload section
    st.subheader("Upload MPRN Smart Meter Data")
    st.markdown(
        """
    Upload your Irish MPRN smart meter data file (CSV format).
    The file should contain columns: MPRN, Meter Serial Number, Read Value, Read Type, Read Date and End Time
    """
    )

    """Data upload and initial analysis page"""
    st.header("📊 Data Upload & Analysis")

    # Use columns to place the slider and file uploader side-by-side
    # The slider will be in the first column (col1) and the file uploader in the second (col2)

    st.subheader("Select Data Type")
    # Slider to select between Urban and Rural data
    selected_type = st.select_slider(
        "Select the data's geographic type:",
        options=['urban', 'rural'],
        help="This setting affects subsequent analysis and calculations."
    )
    # Store the selected value in session state
    st.session_state['type'] = selected_type

    st.info(f"You have selected: **{st.session_state['type']}**")

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], help="Upload your MPRN smart meter data file"
    )

    if uploaded_file is not None:
        try:
            # Parse the uploaded file
            with st.spinner("Parsing MPRN data..."):
                df = parse_mprn_file(uploaded_file)

            if df is not None and not df.empty:
                st.success("✅ File parsed successfully!")

                # Store in session state for other pages
                st.session_state["parsed_data"] = df
                
                st.session_state["uploaded_file"] = uploaded_file.name

                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Show basic statistics
                show_basic_statistics(df)

                # Show validation results
                show_validation_results(uploaded_file)

            else:
                st.error("❌ Failed to parse file. Please check the file format.")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.error(f"File processing error: {e}")

    # Sample data option
    elif st.button("📁 Load 24-Hour Sample"):
        try:
            sample_path = "data/sample_mprn.csv"
            if os.path.exists(sample_path):
                with open(sample_path, "r") as f:
                    df = parse_mprn_file(f)

                if df is not None and not df.empty:
                    st.success("✅ 24-hour sample data loaded successfully!")
                    st.session_state["parsed_data"] = df
                    st.session_state["uploaded_file"] = "24-Hour Sample Data"

                    # Debug: Show what columns we actually have
                    st.subheader("🔍 Data Structure")
                    st.write(f"**Columns found:** {list(df.columns)}")
                    st.write(f"**Data shape:** {df.shape}")

                    st.subheader("📋 Sample Data Preview")
                    st.dataframe(df.head(10), width="stretch")

                    # Show basic statistics
                    show_basic_statistics(df)

                    # Show validation results for sample data
                    st.subheader("🔍 Sample Data Validation")
                    try:
                        with open(sample_path, "r") as f:
                            validation_results = validate_mprn_data(f)

                        if validation_results["is_valid"]:
                            st.success("✅ Sample data validation passed!")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Validation Details:**")
                                for key, value in validation_results.items():
                                    if key != "is_valid" and key != "errors":
                                        st.write(f"- {key}: {value}")

                            with col2:
                                if validation_results.get("errors"):
                                    st.warning("⚠️ Validation Warnings:")
                                    for error in validation_results["errors"]:
                                        st.write(f"- {error}")
                        else:
                            st.error("❌ Sample data validation failed!")
                            for error in validation_results.get("errors", []):
                                st.error(f"- {error}")
                    except Exception as e:
                        st.warning(f"⚠️ Validation check failed: {str(e)}")
                        st.info(
                            "This is normal for sample data - the parser will handle it automatically."
                        )
                else:
                    st.error("❌ Failed to load 24-hour sample data.")
            else:
                st.error("❌ 24-hour sample data file not found.")
        except Exception as e:
            st.error(f"❌ Error loading 24-hour sample data: {str(e)}")
            st.exception(e)  # Show full error details

    elif st.button("📁 Load 20-Day Sample"):
        try:
            sample_path = "data/twenty_day_sample_mprn_fixed.csv"
            if os.path.exists(sample_path):
                with open(sample_path, "r") as f:
                    df = parse_mprn_file(f)

                if df is not None and not df.empty:
                    st.success("✅ 20-day sample data loaded successfully!")
                    st.session_state["parsed_data"] = df
                    st.session_state["uploaded_file"] = "20-Day Sample Data"

                    # Debug: Show what columns we actually have
                    st.subheader("🔍 Data Structure")
                    st.write(f"**Columns found:** {list(df.columns)}")
                    st.write(f"**Data shape:** {df.shape}")

                    st.subheader("📋 Sample Data Preview")
                    st.dataframe(df.head(10), width="stretch")

                    # Show basic statistics
                    show_basic_statistics(df)

                    # Show validation results for sample data
                    st.subheader("🔍 Sample Data Validation")
                    try:
                        with open(sample_path, "r") as f:
                            validation_results = validate_mprn_data(f)

                        if validation_results["is_valid"]:
                            st.success("✅ Sample data validation passed!")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Validation Details:**")
                                for key, value in validation_results.items():
                                    if key != "is_valid" and key != "errors":
                                        st.write(f"- {key}: {value}")

                            with col2:
                                if validation_results.get("errors"):
                                    st.warning("⚠️ Validation Warnings:")
                                    for error in validation_results["errors"]:
                                        st.write(f"- {error}")
                        else:
                            st.error("❌ Sample data validation failed!")
                            for error in validation_results.get("errors", []):
                                st.error(f"- {error}")
                    except Exception as e:
                        st.warning(f"⚠️ Validation check failed: {str(e)}")
                        st.info(
                            "This is normal for sample data - the parser will handle it automatically."
                        )
                else:
                    st.error("❌ Failed to load 20-day sample data.")
            else:
                st.error("❌ 20-day sample data file not found.")
        except Exception as e:
            st.error(f"❌ Error loading 20-day sample data: {str(e)}")
            st.exception(e)  # Show full error details



# --- Helper: classify time-of-use bands (simplified Summer schedule) ---
def classify_period(ts):
    hour = ts.hour
    weekday = ts.weekday()  # Monday=0, Sunday=6
    if 0 <= hour < 9:
        return "night"
    elif weekday < 5 and 17 <= hour < 19:
        return "peak"
    else:
        return "day"

# --- Main Streamlit page ---
def tariff_comparison_page():
    region = st.session_state.get("type", "urban")
    mprn_df = st.session_state.get("parsed_data")

    if mprn_df is None:
        st.error("No MPRN data found in session_state['parsed_data'].")
        return

    # Convert timestamp to datetime
    mprn_df["timestamp"] = pd.to_datetime(mprn_df["timestamp"])

    # Convert kW to kWh for each 30-min interval
    mprn_df["kwh"] = mprn_df["import_kw"] * 0.5

    # Classify into periods
    mprn_df["period"] = mprn_df["timestamp"].apply(classify_period)

    # Aggregate 20-day totals
    agg = mprn_df.groupby("period")["kwh"].sum()

    # Scale to annual (365/20 days)
    scale = 365 / mprn_df["timestamp"].dt.date.nunique()
    annual_use = agg * scale
    total_import = annual_use.sum()

    # --- Read tariff file ---
    tariff_df = pd.read_csv("data/tariff.csv")

    # Filter by region

    tariffs = tariff_df[tariff_df["supply_region"] == region]

    results = []
    for _, t in tariffs.iterrows():
    
        # Apply discount to unit rates
        day_rate_euro = (t["day_unit"] / 100) * (1 - t["discount"])
        peak_rate_euro = (t["peak_unit"] / 100) * (1 - t["discount"])
        night_rate_euro = (t["night_unit"] / 100) * (1 - t["discount"])

        import_cost = (
            annual_use.get("day", 0) * day_rate_euro
            + annual_use.get("peak", 0) * peak_rate_euro
            + annual_use.get("night", 0) * night_rate_euro
        )

        standing_cost = t["standing_charge"]
        pso_cost = t["el_pso_levy"]
        cash_bonus = t["cash_bonus"] if not pd.isna(t["cash_bonus"]) else 0

        total_cost = import_cost + standing_cost + pso_cost - cash_bonus
        results.append({
            "Supplier": t["supplier"],
            "Tariff Name": t["tariff_name"],
            "Supply Region": t["supply_region"],
            "Plan Type": t["plan_type"],
            "Discount": t["discount"],
            "Contract Duration": t["duration"],
            "Total Annual Cost": total_cost,
            "Standing Charge": standing_cost,
            "PSO levy": pso_cost,
            "Cash Bonus": cash_bonus,
            "Total import unit costs": import_cost,
            "Total export unit costs": 0,
            "Extra": t["extra"] if "extra" in t else "",
            "Last Updated Date": t["update_date"],
        })



    result_df = pd.DataFrame(results).sort_values("Total Annual Cost")

    # --- Summary Metrics ---
    avg_cost = result_df["Total Annual Cost"].mean()
    best_plan = result_df.iloc[0]

    st.subheader("Best Annual Tariff Plan Based on your usage")
    col1, col2 = st.columns(2)
    col1.metric("Estimated Annual Import", f"{total_import:,.0f} kWh")
    col2.metric(
        "Best Plan Annual Cost",
        f"€{best_plan['Total Annual Cost']:.2f}",
        f"€{avg_cost - best_plan['Total Annual Cost']:.2f} cheaper than average",
    )

    # --- Highlight DataFrame ---
    def highlight_cost(val):
        min_cost = result_df["Total Annual Cost"].min()
        max_cost = result_df["Total Annual Cost"].max()
        # Normalize 0 → green, 1 → red
        ratio = (val - min_cost) / (max_cost - min_cost + 1e-9)
        if ratio < 0.25:
            color = "#50C878"  # green
        elif ratio < 0.5:
            color = "#FFEA00"  # yellow
        elif ratio < 0.75:
            color = "#ffd27f"  # orange
        else:
            color = "#ff9999"  # red
        return f"background-color: {color}"

    st.dataframe(result_df.style.applymap(highlight_cost, subset=["Total Annual Cost"]))


def show_basic_statistics(df):
    """Display basic statistics about the parsed data"""
    st.subheader("📈 Basic Statistics")

    # Debug: Show what we're working with
    st.write(f"**Available columns:** {list(df.columns)}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        if "import_kw" in df.columns:
            total_import = df["import_kw"].sum()
            st.metric("Total Import", f"{total_import:.2f} kWh")
        elif "Active Import Interval (kW)" in df.columns:
            total_import = df["Active Import Interval (kW)"].sum()
            st.metric("Total Import", f"{total_import:.2f} kWh")
        else:
            st.metric("Total Import", "Column not found")

    with col3:
        if "export_kw" in df.columns:
            total_export = df["export_kw"].sum()
            st.metric("Total Export", f"{total_export:.2f} kWh")
        else:
            st.metric("Total Export", "No export data")

    with col4:
        if "import_kw" in df.columns:
            avg_import = df["import_kw"].mean()
            st.metric("Avg Import", f"{avg_import:.3f} kW")
        elif "Active Import Interval (kW)" in df.columns:
            avg_import = df["Active Import Interval (kW)"].mean()
            st.metric("Avg Import", f"{avg_import:.3f} kW")
        else:
            st.metric("Avg Import", "Column not found")

    # Date range information
    if "timestamp" in df.columns:
        st.subheader("📅 Data Time Range")
        date_range = df["timestamp"].agg(["min", "max"])
        st.info(
            f"**Start:** {date_range['min'].strftime('%Y-%m-%d %H:%M')} | **End:** {date_range['max'].strftime('%Y-%m-%d %H:%M')}"
        )
    elif "Read Date and End Time" in df.columns:
        st.subheader("📅 Data Time Range")
        date_range = df["Read Date and End Time"].agg(["min", "max"])
        st.info(
            f"**Start:** {date_range['min'].strftime('%Y-%m-%d %H:%M')} | **End:** {date_range['max'].strftime('%Y-%m-%d %H:%M')}"
        )

    # Show first few rows for debugging
    st.subheader("🔍 Raw Data Sample")
    st.dataframe(df.head(5), width="stretch")


def show_validation_results(uploaded_file):
    """Show validation results for the uploaded file"""
    st.subheader("🔍 Data Validation")

    try:
        # Re-upload file for validation (since parse_mprn_file consumes the file)
        uploaded_file.seek(0)
        validation_results = validate_mprn_data(uploaded_file)

        if validation_results["is_valid"]:
            st.success("✅ Data validation passed!")

            # Show validation details
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Validation Details:**")
                for key, value in validation_results.items():
                    if key != "is_valid" and key != "errors":
                        st.write(f"- {key}: {value}")

            with col2:
                if validation_results.get("errors"):
                    st.warning("⚠️ Validation Warnings:")
                    for error in validation_results["errors"]:
                        st.write(f"- {error}")
        else:
            st.error("❌ Data validation failed!")
            for error in validation_results.get("errors", []):
                st.error(f"- {error}")

    except Exception as e:
        st.error(f"❌ Error during validation: {str(e)}")


def show_usage_patterns_page():
    """Display usage patterns and charts"""
    st.header("💰 Usage Patterns")

    if "parsed_data" not in st.session_state:
        st.warning("⚠️ Please upload data first on the Data Upload page.")
        return

    df = st.session_state["parsed_data"]

    if df is None or df.empty:
        st.error("❌ No data available for analysis.")
        return

    # Daily usage pattern
    st.subheader("📊 Daily Usage Pattern")

    # Now we have consistent column names from the data parser
    if "timestamp" in df.columns and "import_kw" in df.columns:
        # Create a copy for analysis to avoid modifying the original
        df_analysis = df.copy()

        # Ensure timestamp column is datetime
        df_analysis["timestamp"] = pd.to_datetime(df_analysis["timestamp"])

        # Resample to hourly for daily pattern
        df_hourly = (
            df_analysis.set_index("timestamp").resample("1h")["import_kw"].mean()
        )

        fig = px.line(
            df_hourly,
            title="Average Hourly Usage Pattern",
            labels={"value": "Average Import (kW)", "index": "Hour of Day"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

        # Weekly pattern
        st.subheader("📅 Weekly Usage Pattern")
        df_analysis["day_of_week"] = df_analysis["timestamp"].dt.day_name()
        df_analysis["hour"] = df_analysis["timestamp"].dt.hour

        weekly_pattern = (
            df_analysis.groupby(["day_of_week", "hour"])["import_kw"].mean().unstack()
        )

        fig_weekly = px.imshow(
            weekly_pattern,
            title="Weekly Usage Heatmap (kW)",
            labels={"x": "Hour of Day", "y": "Day of Week", "color": "Import (kW)"},
        )
        fig_weekly.update_layout(height=400)
        st.plotly_chart(fig_weekly, width="stretch")

        # Add half-hourly usage chart
        st.subheader("⏰ Half-Hourly Usage Pattern")
        # Sample every 4th point for better visualization (every 2 hours)
        df_sample = df_analysis.iloc[::4].copy()

        fig_halfhourly = px.line(
            df_sample,
            x="timestamp",
            y="import_kw",
            title="Half-Hourly Energy Usage (kW)",
            labels={"import_kw": "Import (kW)", "timestamp": "Time"},
        )
        fig_halfhourly.update_layout(height=400)
        st.plotly_chart(fig_halfhourly, width="stretch")

        # Enhanced Usage Analysis
        st.subheader("🔍 Enhanced Usage Analysis")

        try:
            # Initialize usage analyzer
            analyzer = UsageAnalyzer()

            # Get comprehensive analysis
            with st.spinner("Analyzing usage patterns..."):
                daily_patterns = analyzer.analyze_daily_patterns(df_analysis)
                peaks = analyzer.identify_peaks(df_analysis)
                usage_stats = analyzer.calculate_usage_stats(df_analysis)
                user_type = analyzer.classify_user_type(usage_stats)

            if daily_patterns and usage_stats:
                # Display user classification
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("User Type", user_type)
                with col2:
                    st.metric(
                        "Total Energy",
                        f"{usage_stats['basic']['total_energy_kwh']:.2f} kWh",
                    )
                with col3:
                    st.metric(
                        "Peak Hours Usage",
                        f"{usage_stats['peak_hours']['peak_hours_usage']:.3f} kW",
                    )

                # Time-of-use breakdown
                st.subheader("⏰ Time-of-Use Breakdown")
                time_of_use = daily_patterns["time_of_use"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Night Usage", f"{time_of_use['night_percentage']:.1f}%")
                with col2:
                    st.metric("Day Usage", f"{time_of_use['day_percentage']:.1f}%")
                with col3:
                    st.metric("Peak Hours", f"{time_of_use['peak_percentage']:.1f}%")

                # Peak detection results
                if peaks:
                    st.subheader("📈 Peak Usage Detection")
                    st.write(f"**Found {len(peaks)} peak periods:**")

                    for i, peak in enumerate(peaks[:3]):  # Show top 3 peaks
                        with st.expander(f"Peak {i+1}: {peak['peak_value']:.2f} kW"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(
                                    f"**Start:** {peak['start_time'].strftime('%H:%M')}"
                                )
                                st.write(
                                    f"**End:** {peak['end_time'].strftime('%H:%M')}"
                                )
                                st.write(
                                    f"**Duration:** {peak['duration_hours']:.1f} hours"
                                )
                            with col2:
                                st.write(f"**Peak Value:** {peak['peak_value']:.2f} kW")
                                st.write(f"**Average:** {peak['average_value']:.2f} kW")
                                st.write(
                                    f"**Magnitude:** {peak['magnitude_factor']:.1f}x normal"
                                )
                else:
                    st.info("ℹ️ No significant peak periods detected in this data.")

                # Efficiency metrics
                st.subheader("📊 Efficiency Metrics")
                efficiency = usage_stats["efficiency"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Usage Variability", f"{efficiency['usage_variability']:.2f}"
                    )
                    st.caption("Lower values = more consistent usage")
                with col2:
                    st.metric(
                        "Peak-to-Average Ratio",
                        f"{efficiency['peak_to_average_ratio']:.2f}",
                    )
                    st.caption("Higher values = more peaky usage")

        except Exception as e:
            st.error(f"❌ Error in enhanced analysis: {str(e)}")
            logger.error(f"Enhanced analysis error: {e}")

    else:
        st.error(f"❌ Required columns not found for pattern analysis.")
        st.write(f"**Available columns:** {list(df.columns)}")
        st.write(f"**Looking for:** timestamp and import_kw")


def show_cost_analysis_page():
    """Cost analysis page with tariff calculations"""
    st.header("💰 Forecast & Cost Analysis")

    if "parsed_data" not in st.session_state:
        st.warning("⚠️ Please upload data first on the Data Upload page.")
        return

    df = st.session_state["parsed_data"]

    if df is None or df.empty:
        st.error("❌ No data available for cost analysis.")
        return

    st.subheader("⚡ Forecasted Energy Consumption (Next 30 Days)")

    forecast = ForecastConsumption()
    forecast_result = forecast.run_forecast_modelling(df)
    fig = px.line(
                forecast_result,
                x='future_datetime',
                y='forecasted_consumption',
                title='Forecasted Energy Consumption (Next 30 Days, 30-min Intervals)',
                labels={'future_datetime': 'Datetime', 'forecasted_consumption': 'Forecasted Consumption (kWh)'},
                template='plotly_dark'
            )
    st.plotly_chart(fig, use_container_width=True)

    # Rate input section
    st.subheader("⚡ Enter Your Electricity Rate")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Simple Rate Input**
        
        Enter your current electricity rate to see how much your usage costs.
        You can find this on your energy bill.
        """
        )

        rate_per_kwh = st.number_input(
            "Rate per kWh (€):",
            min_value=0.01,
            max_value=1.00,
            value=0.23,
            step=0.01,
            help="Enter your electricity rate in euros per kWh (e.g., 0.23 for €0.23/kWh)",
        )

    with col2:
        st.markdown(
            """
        **Advanced Options**
        
        For more accurate calculations, you can specify different rates for different times.
        """
        )

        use_time_based = st.checkbox("Use time-based rates (Day/Night/Peak)")

        if use_time_based:
            day_rate = st.number_input("Day Rate (€/kWh):", value=0.25, step=0.01)
            night_rate = st.number_input("Night Rate (€/kWh):", value=0.18, step=0.01)
            peak_rate = st.number_input("Peak Rate (€/kWh):", value=0.30, step=0.01)

    # Calculate costs
    if st.button("💰 Calculate Costs", type="primary"):
        with st.spinner("Calculating costs..."):
            if use_time_based:
                cost_breakdown = calculate_time_based_cost(
                    forecast_result, day_rate, night_rate, peak_rate
                )
            else:
                cost_breakdown = calculate_simple_cost(forecast_result, rate_per_kwh)

            if cost_breakdown:
                st.success("✅ Cost calculation completed!")
                show_cost_results(cost_breakdown, df)
            else:
                st.error("❌ Failed to calculate costs. Please check your data.")


def show_cost_results(cost_breakdown: Dict, df: pd.DataFrame):
    """Display cost analysis results"""
    st.subheader("📊 Cost Breakdown")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Energy",
            f"{cost_breakdown['total_energy_kwh']} kWh",
            help="Total energy consumption in the period",
        )

    with col2:
        st.metric(
            "Total Cost",
            f"€{cost_breakdown['total_cost_euros']}",
            help="Total cost for the period",
        )

    with col3:
        daily_cost = cost_breakdown.get("daily_average", {}).get("cost_euros", 0)
        st.metric("Daily Average", f"€{daily_cost}", help="Average daily cost")

    with col4:
        monthly_cost = cost_breakdown.get("monthly_projection", {}).get("cost_euros", 0)
        st.metric(
            "Monthly Projection",
            f"€{monthly_cost}",
            help="Projected monthly cost based on current usage",
        )

    # Time period breakdown
    st.subheader("⏰ Cost by Time Period")

    if "time_periods" in cost_breakdown:
        time_periods = cost_breakdown["time_periods"]

        # Create a DataFrame for better display
        period_data = []
        for period, data in time_periods.items():
            period_data.append(
                {
                    "Time Period": period.title(),
                    "Energy (kWh)": data["energy_kwh"],
                    "Cost (€)": data["cost_euros"],
                    "Percentage": f"{data['percentage']}%",
                }
            )

        period_df = pd.DataFrame(period_data)
        st.dataframe(period_df, width="stretch")

        # Visual breakdown
        col1, col2 = st.columns(2)

        with col1:
            # Energy breakdown pie chart
            energy_values = [data["energy_kwh"] for data in time_periods.values()]
            energy_labels = [period.title() for period in time_periods.keys()]

            import plotly.express as px

            fig_energy = px.pie(
                values=energy_values,
                names=energy_labels,
                title="Energy Usage by Time Period",
            )
            st.plotly_chart(fig_energy, width="stretch")

        with col2:
            # Cost breakdown pie chart
            cost_values = [data["cost_euros"] for data in time_periods.values()]
            cost_labels = [period.title() for period in time_periods.keys()]

            fig_cost = px.pie(
                values=cost_values,
                names=cost_labels,
                title="Cost Breakdown by Time Period",
            )
            st.plotly_chart(fig_cost, width="stretch")

    # Savings opportunities
    if (
        "savings_opportunities" in cost_breakdown
        and cost_breakdown["savings_opportunities"]
    ):
        st.subheader("💡 Potential Savings Opportunities")

        for i, opportunity in enumerate(cost_breakdown["savings_opportunities"]):
            with st.expander(
                f"💰 {opportunity['type']} - Save €{opportunity['potential_savings']}"
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Description:** {opportunity['description']}")
                    st.write(f"**Difficulty:** {opportunity['difficulty']}")

                with col2:
                    st.write(
                        f"**Potential Savings:** €{opportunity['potential_savings']}"
                    )
                    st.write(f"**Action:** {opportunity['action']}")

    # Insights and recommendations
    st.subheader("🔍 Cost Insights")

    # Calculate some insights
    total_cost = cost_breakdown["total_cost_euros"]
    total_energy = cost_breakdown["total_energy_kwh"]

    # Simple insights
    if total_cost > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.info(
                f"**Your forecasted energy usage costs €{total_cost:.2f} for {total_energy:.1f} kWh**"
            )

            if "time_periods" in cost_breakdown:
                peak_percentage = (
                    cost_breakdown["time_periods"].get("peak", {}).get("percentage", 0)
                )
                if peak_percentage > 20:
                    st.warning(
                        f"⚠️ **{peak_percentage}% of your usage is during peak hours** - Consider shifting to cheaper times"
                    )
                else:
                    st.success(
                        f"✅ **Good peak management** - Only {peak_percentage}% during expensive peak hours"
                    )

        with col2:
            # Monthly projection insights
            monthly_cost = cost_breakdown.get("monthly_projection", {}).get(
                "cost_euros", 0
            )
            if monthly_cost > 0:
                st.info(
                    f"**At this rate, you'll spend approximately €{monthly_cost:.0f} per month**"
                )

                if monthly_cost > 150:
                    st.warning(
                        "💡 **High monthly costs detected** - Consider energy efficiency improvements"
                    )
                elif monthly_cost < 80:
                    st.success(
                        "✅ **Efficient energy usage** - Your costs are below average"
                    )
                else:
                    st.info("📊 **Moderate energy usage** - Room for optimization")


def show_appliance_detection_page():
    """Display appliance detection analysis"""
    st.header("🔍 Appliance Detection")

    if "parsed_data" not in st.session_state:
        st.warning("⚠️ Please upload data first on the Data Upload page.")
        return

    df = st.session_state["parsed_data"]

    if df is None or df.empty:
        st.error("❌ No data available for analysis.")
        return

    try:
        # Prepare data for model
        df_model = df.copy()
        df_model["power"] = df_model["import_kw"]
        df_model["power_diff"] = df_model["power"].diff().fillna(0)
        df_model["rolling_mean_power"] = df_model["power"].rolling(window=3).mean().fillna(0)
        df_model["rolling_std_power"] = df_model["power"].rolling(window=3).std().fillna(0)
        df_model.dropna(inplace=True)

        # Load the model
        with st.spinner("Loading appliance detection model..."):
            model = joblib.load("model/appliance_prediction_model.pkl")

        # Make predictions
        predictions = model.predict(df_model[["power", "power_diff", "rolling_mean_power", "rolling_std_power"]])
        df_model["predicted_appliance"] = predictions

        # Process predictions into time segments
        appliance_segments = []
        current_appliance = None
        start_time = None

        for index, row in df_model.iterrows():
            if current_appliance is None:
                current_appliance = row["predicted_appliance"]
                start_time = row["timestamp"]
            elif row["predicted_appliance"] != current_appliance:
                appliance_segments.append({
                    "appliance_name": current_appliance,
                    "start_time": start_time,
                    "end_time": row["timestamp"]
                })
                current_appliance = row["predicted_appliance"]
                start_time = row["timestamp"]

        # Add final segment
        if current_appliance is not None:
            appliance_segments.append({
                "appliance_name": current_appliance,
                "start_time": start_time,
                "end_time": df_model.iloc[-1]["timestamp"]
            })

        # Store in session state
        st.session_state["appliance_segments"] = appliance_segments

        # Display results
        st.subheader("📊 Detected Appliance Usage")

        # Create timeline chart using plotly
        fig = go.Figure()

        # Color map for different appliances
        colors = px.colors.qualitative.Set3

        # Get unique appliances
        unique_appliances = list(set(segment["appliance_name"] for segment in appliance_segments))
        color_map = dict(zip(unique_appliances, colors[:len(unique_appliances)]))

        for segment in appliance_segments:
            fig.add_trace(go.Bar(
                x=[segment["start_time"], segment["end_time"]],
                y=[segment["appliance_name"], segment["appliance_name"]],
                orientation="h",
                name=segment["appliance_name"],
                marker_color=color_map[segment["appliance_name"]],
                showlegend=False
            ))

        fig.update_layout(
            title="Appliance Usage Timeline",
            xaxis_title="Time",
            yaxis_title="Appliance",
            height=400,
            barmode="overlay"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display summary table
        st.subheader("📋 Usage Summary")
        summary_data = []
        for appliance in unique_appliances:
            appliance_times = [
                segment for segment in appliance_segments 
                if segment["appliance_name"] == appliance
            ]
            total_duration = sum(
                (segment["end_time"] - segment["start_time"]).total_seconds() / 3600 
                for segment in appliance_times
            )
            usage_count = len(appliance_times)
            
            summary_data.append({
                "Appliance": appliance,
                "Usage Count": usage_count,
                "Total Hours": f"{total_duration:.2f}"
            })

        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(f"❌ Error in appliance detection: {str(e)}")
        st.exception(e)

def show_recommendations_page():
    """AI-powered recommendations page"""
    st.header("💡 AI-Powered Energy Saving Recommendations")
    
    if 'parsed_data' not in st.session_state:
        st.warning("⚠️ Please upload data first on the Data Upload page.")
        return
    
    df = st.session_state['parsed_data']
    
    if df is None or df.empty:
        st.error("❌ No data available for recommendations.")
        return
    
    # Rate input section
    st.subheader("⚡ Enter Your Current Electricity Rate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_rate = st.number_input(
            "Current Rate per kWh (€):",
            min_value=0.01,
            max_value=1.00,
            value=0.23,
            step=0.01,
            help="Enter your current electricity rate in euros per kWh"
        )
    
    with col2:
        st.markdown("""
        **AI Analysis**
        
        Our AI will analyze your usage patterns and provide personalized recommendations with specific savings estimates.
        """)
    
    # Generate recommendations
    if st.button("🤖 Generate AI Recommendations", type="primary"):
        with st.spinner("🤖 AI is analyzing your usage patterns..."):
            try:
                from recommendation_engine import RecommendationEngine, generate_action_plan
                
                # Generate recommendations
                engine = RecommendationEngine()
                recommendations_data = engine.generate_ai_powered_recommendations(df, current_rate)
                
                if recommendations_data and 'recommendations' in recommendations_data:
                    st.success("✅ AI analysis completed!")
                    show_recommendations_results(recommendations_data)
                else:
                    st.error("❌ Failed to generate recommendations. Please check your data.")
                    
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                logger.error(f"Recommendations generation error: {e}")


def show_recommendations_results(recommendations_data):
    """Display AI-powered recommendations results"""
    recommendations = recommendations_data.get('recommendations', [])
    total_savings = recommendations_data.get('total_potential_savings', 0)
    analysis = recommendations_data.get('analysis', {})
    
    # Summary metrics
    st.subheader("📊 AI Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Recommendations",
            len(recommendations),
            help="Number of personalized recommendations generated"
        )
    
    with col2:
        st.metric(
            "Monthly Savings Potential",
            f"€{total_savings}",
            help="Total potential monthly savings from all recommendations"
        )
    
    with col3:
        annual_savings = total_savings * 12
        st.metric(
            "Annual Savings Potential",
            f"€{annual_savings:.0f}",
            help="Total potential annual savings"
        )
    
    with col4:
        current_monthly = analysis.get('current_costs', {}).get('monthly_projection', {}).get('cost_euros', 0)
        savings_percentage = (total_savings / current_monthly * 100) if current_monthly > 0 else 0
        st.metric(
            "Savings Percentage",
            f"{savings_percentage:.1f}%",
            help="Percentage reduction in monthly energy costs"
        )
    
    # Check if we have AI insights (raw response) or structured recommendations
    ai_insights = recommendations_data.get('ai_insights', [])
    has_ai_response = any(insight and len(insight.strip()) > 50 for insight in ai_insights)
    
    if not recommendations and not has_ai_response:
        st.info("ℹ️ No specific recommendations found for your usage pattern. Your energy usage appears to be already well optimized!")
        return
    
    # If we have AI insights but no structured recommendations, show the AI response
    if not recommendations and has_ai_response:
        st.subheader("🤖 AI-Powered Energy Analysis")
        for insight in ai_insights:
            if insight and len(insight.strip()) > 50:  # Only show substantial responses
                st.markdown("### DeepSeek AI Analysis:")
                st.markdown(insight)
        return
    
    # Recommendations by impact level
    st.subheader("🎯 Recommendations by Impact Level")
    
    # Group recommendations by impact level
    impact_groups = {}
    for rec in recommendations:
        impact = rec.get('impact_level', 'Minimal Impact')
        if impact not in impact_groups:
            impact_groups[impact] = []
        impact_groups[impact].append(rec)
    
    # Display recommendations by impact level
    impact_order = ['High Impact', 'Medium Impact', 'Low Impact', 'Minimal Impact']
    
    for impact_level in impact_order:
        if impact_level in impact_groups:
            impact_recs = impact_groups[impact_level]
            
            # Impact level header
            if impact_level == 'High Impact':
                st.success(f"🔥 **{impact_level}** - {len(impact_recs)} recommendations")
            elif impact_level == 'Medium Impact':
                st.info(f"⚡ **{impact_level}** - {len(impact_recs)} recommendations")
            elif impact_level == 'Low Impact':
                st.warning(f"💡 **{impact_level}** - {len(impact_recs)} recommendations")
            else:
                st.info(f"📝 **{impact_level}** - {len(impact_recs)} recommendations")
            
            # Display recommendations for this impact level
            for i, rec in enumerate(impact_recs):
                with st.expander(f"{rec.get('title', 'Recommendation')} - Save €{rec.get('monthly_savings', 0)}/month"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Description:** {rec.get('description', 'No description available')}")
                        st.write(f"**Monthly Savings:** €{rec.get('monthly_savings', 0)}")
                        st.write(f"**Annual Savings:** €{rec.get('annual_savings', 0)}")
                        st.write(f"**Difficulty:** {rec.get('difficulty', 'Unknown')}")
                        st.write(f"**Time to Implement:** {rec.get('time_to_implement', 'Unknown')}")
                    
                    with col2:
                        if 'action_items' in rec and rec['action_items']:
                            st.write("**Action Items:**")
                            for action in rec['action_items']:
                                st.write(f"• {action}")
                        
                        # Show additional details if available
                        if 'peak_hour' in rec:
                            st.write(f"**Peak Hour:** {rec['peak_hour']}:00")
                        if 'current_baseline' in rec:
                            st.write(f"**Current Baseline:** {rec['current_baseline']} kW")
    
    # Action Plan
    st.subheader("📋 Personalized Action Plan")
    
    try:
        from recommendation_engine import generate_action_plan
        action_plan = generate_action_plan(recommendations)
        
        if action_plan and 'timeline' in action_plan:
            timeline = action_plan['timeline']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "This Week",
                    f"€{timeline.get('immediate', {}).get('savings', 0)}",
                    help=f"{timeline.get('immediate', {}).get('count', 0)} immediate actions"
                )
            
            with col2:
                st.metric(
                    "Next 1-4 Weeks",
                    f"€{timeline.get('short_term', {}).get('savings', 0)}",
                    help=f"{timeline.get('short_term', {}).get('count', 0)} short-term actions"
                )
            
            with col3:
                st.metric(
                    "Next 1-3 Months",
                    f"€{timeline.get('long_term', {}).get('savings', 0)}",
                    help=f"{timeline.get('long_term', {}).get('count', 0)} long-term actions"
                )
            
            # Detailed action plan
            action_plan_details = action_plan.get('action_plan', {})
            
            if action_plan_details.get('immediate'):
                st.subheader("⚡ Immediate Actions (This Week)")
                for rec in action_plan_details['immediate']:
                    st.write(f"• **{rec.get('title', 'Action')}** - Save €{rec.get('monthly_savings', 0)}/month")
            
            if action_plan_details.get('short_term'):
                st.subheader("📅 Short-term Actions (Next 1-4 Weeks)")
                for rec in action_plan_details['short_term']:
                    st.write(f"• **{rec.get('title', 'Action')}** - Save €{rec.get('monthly_savings', 0)}/month")
            
            if action_plan_details.get('long_term'):
                st.subheader("🎯 Long-term Actions (Next 1-3 Months)")
                for rec in action_plan_details['long_term']:
                    st.write(f"• **{rec.get('title', 'Action')}** - Save €{rec.get('monthly_savings', 0)}/month")
    
    except Exception as e:
        st.warning(f"Could not generate action plan: {str(e)}")
    
    # AI Insights
    ai_insights = recommendations_data.get('ai_insights', [])
    if ai_insights:
        st.subheader("🧠 AI Insights")
        for insight in ai_insights:
            st.info(f"💡 {insight}")
    if "parsed_data" in st.session_state:
        st.write("Data is available for generating recommendations.")
    else:
        st.warning("⚠️ Please upload data first on the Data Upload page.")


def show_smart_agent_scheduler_page():
    """Smart Agent Scheduler page for automated energy management"""
    st.header("🤖 Smart Agent Scheduler")
    st.markdown("**Intelligent automation for optimal energy usage and cost savings**")
    
    if 'parsed_data' not in st.session_state:
        st.warning("⚠️ Please upload data first on the Data Upload page.")
        return
    
    df = st.session_state['parsed_data']
    
    if df is None or df.empty:
        st.error("❌ No data available for smart scheduling.")
        return
    
    # Initialize agent session state
    if 'smart_agent' not in st.session_state:
        st.session_state['smart_agent'] = None
        st.session_state['agent_status'] = "🔴 Inactive"
        st.session_state['command_history'] = []
        st.session_state['agent_active'] = False
        st.session_state['schedules_today'] = 0
        st.session_state['monthly_savings'] = 0
        st.session_state['carbon_saved'] = 0
    
    # Initialize the smart agent if available and not already initialized
    if SMART_AGENT_AVAILABLE and st.session_state['smart_agent'] is None:
        try:
            with st.spinner("🤖 Initializing Smart Agent..."):
                st.session_state['smart_agent'] = EnhancedSmartPlugAgent()
                st.session_state['agent_status'] = "🟢 Ready"
            st.success("✅ Smart Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize Smart Agent: {str(e)}")
            st.session_state['agent_status'] = "🔴 Error"
    elif not SMART_AGENT_AVAILABLE:
        st.warning("⚠️ Smart Agent hardware interface not available. Using simulation mode.")
    
    # Show agent status
    if st.session_state['smart_agent'] is not None:
        st.info(f"🤖 **Smart Agent Status:** {st.session_state['agent_status']}")
    else:
        st.info("🤖 **Smart Agent Status:** Simulation Mode")
    
    # Smart Agent Command Interface Section
    st.subheader("🎯 Smart Agent Command Interface")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Direct Device Control**
        
        Use natural language commands to control your smart devices:
        - "Turn on dishwasher"
        - "Set coffee maker at 7:00 AM"
        - "Turn off lights in 30 minutes"
        - "Status" - Show system status
        - "Recommendations" - Get smart recommendations
        """)
        
        # Command input
        command_input = st.text_input(
            "Enter command:",
            placeholder="e.g., turn on dishwasher",
            key="agent_command"
        )
        
        col_cmd1, col_cmd2, col_cmd3 = st.columns([1, 1, 1])
        
        # Initialize session state for command results
        if 'command_result' not in st.session_state:
            st.session_state['command_result'] = None
        if 'result_type' not in st.session_state:
            st.session_state['result_type'] = None
        
        with col_cmd1:
            if st.button("🚀 Execute Command", type="primary"):
                if command_input.strip():
                    with st.spinner("🤖 Processing command..."):
                        result = execute_smart_agent_command(command_input.strip())
                        st.session_state['command_result'] = result
                        st.session_state['result_type'] = 'execute'
                        # Add to history
                        st.session_state['command_history'].append({
                            'timestamp': datetime.now(),
                            'command': command_input.strip(),
                            'result': result
                        })
                else:
                    st.warning("Please enter a command")
        
        with col_cmd2:
            if st.button("📊 Get Status"):
                with st.spinner("📊 Getting system status..."):
                    result = execute_smart_agent_command("status")
                    st.session_state['command_result'] = result
                    st.session_state['result_type'] = 'status'
        
        with col_cmd3:
            if st.button("💡 Get Recommendations"):
                with st.spinner("💡 Getting recommendations..."):
                    result = execute_smart_agent_command("recommendations")
                    st.session_state['command_result'] = result
                    st.session_state['result_type'] = 'recommendations'
    
    # Display command result in a larger, horizontal layout
    if st.session_state.get('command_result') is not None:
        st.markdown("---")
        st.subheader(f"📋 Command Result - {st.session_state['result_type'].title()}")
        
        # Create a large container for the result
        with st.container():
            display_command_result_horizontal(st.session_state['command_result'])
        
        # Add clear button
        if st.button("🗑️ Clear Result", key="clear_command_result"):
            st.session_state['command_result'] = None
            st.session_state['result_type'] = None
            st.rerun()
    
    with col2:
        st.markdown("### 🎮 Quick Actions")
        
        if st.button("🔌 Turn On Dishwasher", use_container_width=True):
            result = execute_smart_agent_command("turn on dishwasher")
            st.session_state['command_result'] = result
            st.session_state['result_type'] = 'quick_action'
        
        if st.button("☕ Turn On Coffee Maker", use_container_width=True):
            result = execute_smart_agent_command("turn on coffee maker")
            st.session_state['command_result'] = result
            st.session_state['result_type'] = 'quick_action'
        
        if st.button("💡 Turn On Lights", use_container_width=True):
            result = execute_smart_agent_command("turn on lights")
            st.session_state['command_result'] = result
            st.session_state['result_type'] = 'quick_action'
        
        if st.button("🚨 Emergency Stop", use_container_width=True):
            result = execute_smart_agent_command("emergency")
            st.session_state['command_result'] = result
            st.session_state['result_type'] = 'emergency'
    
    # Command History Section
    if st.session_state['command_history']:
        st.subheader("📝 Command History")
        
        # Show last 5 commands
        recent_commands = st.session_state['command_history'][-5:]
        
        for i, cmd_data in enumerate(reversed(recent_commands)):
            with st.expander(f"🎯 {cmd_data['command']} - {cmd_data['timestamp'].strftime('%H:%M:%S')}"):
                display_command_result(cmd_data['result'], show_in_expander=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state['command_history'] = []
            st.rerun()
    
    st.divider()
    
    # Agent Configuration Section
    st.subheader("⚙️ Smart Agent Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Energy Goals")
        
        # Goal selection
        primary_goal = st.selectbox(
            "Primary Goal:",
            [
                "💰 Minimize Cost",
                "🌱 Minimize Carbon Footprint", 
                "⚡ Maximize Efficiency",
                "🏠 Maintain Comfort"
            ]
        )
        
        # Cost savings target
        target_savings = st.slider(
            "Target Monthly Savings (€):",
            min_value=5,
            max_value=200,
            value=30,
            step=5
        )
        
        # Priority appliances
        priority_appliances = st.multiselect(
            "Priority Appliances (never interrupt):",
            [
                "🌡️ Heating/Cooling",
                "🔆 Essential Lighting",
                "🍽️ Refrigerator",
                "💻 Home Office",
                "🏥 Medical Equipment"
            ],
            default=["🌡️ Heating/Cooling", "🍽️ Refrigerator"]
        )
    
    with col2:
        st.markdown("### 🕒 Scheduling Preferences")
        
        # Time preferences
        preferred_hours = st.select_slider(
            "Preferred Hours for High-Usage Tasks:",
            options=[
                "Early Morning (6-9 AM)",
                "Morning (9-12 PM)", 
                "Afternoon (12-6 PM)",
                "Evening (6-9 PM)",
                "Late Evening (9-11 PM)",
                "Night (11 PM-6 AM)"
            ],
            value="Night (11 PM-6 AM)"
        )
        
        # Automation level
        automation_level = st.radio(
            "Automation Level:",
            [
                "🔔 Notify Only (recommendations)",
                "⚡ Semi-Automatic (ask permission)",
                "🤖 Fully Automatic (trusted actions)"
            ]
        )
        
        # Smart plug integration
        enable_smart_plugs = st.checkbox(
            "🔌 Enable Smart Plug Control",
            value=True,
            help="Allow the agent to control compatible smart plugs"
        )
    
    # Schedule Preview Section
    st.subheader("📅 AI-Generated Schedule Preview")
    
    if st.button("🧠 Generate Smart Schedule", type="primary"):
        with st.spinner("🤖 AI Agent is analyzing your usage patterns and creating optimal schedule..."):
            
            # Simulate AI schedule generation
            time.sleep(2)  # Simulate processing
            
            # Generate mock schedule (in real implementation, this would be AI-generated)
            schedule_data = generate_mock_schedule(df, primary_goal, target_savings)
            
            if schedule_data:
                st.success("✅ Smart schedule generated successfully!")
                show_schedule_results(schedule_data)
            else:
                st.error("❌ Failed to generate schedule. Please try different settings.")
    
    # Smart Agent Status Section
    st.subheader("📊 Smart Agent Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Agent Status",
            st.session_state.get('agent_status', "🔴 Inactive")
        )
    
    with col2:
        st.metric(
            "Schedules Today",
            st.session_state.get('schedules_today', 0)
        )
    
    with col3:
        st.metric(
            "Savings This Month",
            f"€{st.session_state.get('monthly_savings', 0)}"
        )
    
    with col4:
        st.metric(
            "Carbon Saved",
            f"{st.session_state.get('carbon_saved', 0)} kg CO₂"
        )
    
    # Agent Controls
    st.subheader("🎮 Agent Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Agent", type="primary"):
            st.session_state['agent_active'] = True
            st.session_state['agent_status'] = "🟢 Active"
            st.success("🤖 Smart Agent activated!")
    
    with col2:
        if st.button("⏸️ Pause Agent"):
            st.session_state['agent_active'] = False
            st.session_state['agent_status'] = "🟡 Paused"
            st.info("⏸️ Smart Agent paused.")
    
    with col3:
        if st.button("🔄 Reset Settings"):
            # Reset agent settings
            for key in ['agent_active', 'schedules_today', 'monthly_savings', 'carbon_saved']:
                if key in st.session_state:
                    st.session_state[key] = 0
            st.session_state['agent_status'] = "🔴 Inactive"
            st.session_state['command_history'] = []
            st.info("🔄 Agent settings reset.")


def run_async_command(agent, command_func, *args, **kwargs):
    """Helper function to run async commands in Streamlit"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(command_func(*args, **kwargs))
        finally:
            loop.close()
    except Exception as e:
        return {
            'success': False,
            'error': f'Error running async command: {str(e)}',
            'type': 'error'
        }


def execute_smart_agent_command(command: str) -> Dict:
    """Execute a command using the smart agent"""
    try:
        agent = st.session_state.get('smart_agent')
        
        if agent is None and SMART_AGENT_AVAILABLE:
            return {
                'success': False,
                'error': 'Smart Agent not initialized',
                'suggestion': 'Please wait for agent initialization or refresh the page'
            }
        
        # Use real agent if available, otherwise simulate
        if agent is not None and SMART_AGENT_AVAILABLE:
            # Handle different command types with the real agent
            if command.lower() == "status":
                # Get comprehensive status from the real agent
                result = run_async_command(agent, agent.get_comprehensive_status)
                if result.get('error'):
                    return {
                        'success': False,
                        'error': result['error'],
                        'type': 'error'
                    }
                
                return {
                    'success': True,
                    'type': 'status',
                    'data': result
                }
            
            elif command.lower() == "recommendations":
                # Get recommendations from the real agent
                result = run_async_command(agent, agent.get_device_recommendations)
                return {
                    'success': True,
                    'type': 'recommendations',
                    'data': result if isinstance(result, list) else []
                }
            
            elif command.lower() == "emergency":
                # Execute emergency shutdown
                result = run_async_command(agent, agent.emergency_shutdown)
                return {
                    'success': result.get('success', False),
                    'type': 'emergency',
                    'message': 'Emergency shutdown completed' if result.get('success') else 'Emergency shutdown failed',
                    'data': result
                }
            
            else:
                # Process natural language command
                result = run_async_command(agent, agent.process_natural_language_command, command)
                
                if result.get('success'):
                    return {
                        'success': True,
                        'type': 'device_control',
                        'action': result.get('action', 'executed'),
                        'device': result.get('device', command),
                        'message': f"Command '{command}' executed successfully",
                        'confidence': result.get('llm_confidence', 0.0),
                        'result_details': result
                    }
                else:
                    return {
                        'success': False,
                        'error': result.get('error', 'Command execution failed'),
                        'suggestion': 'Try commands like: turn on dishwasher, status, recommendations'
                    }
        
        else:
            # Fallback to simulation mode
            return execute_simulated_command(command)
    
    except Exception as e:
        return {
            'success': False,
            'error': f'Error executing command: {str(e)}',
            'type': 'error'
        }


def execute_simulated_command(command: str) -> Dict:
    """Execute simulated commands when real agent is not available"""
    if command.lower() == "status":
        return {
            'success': True,
            'type': 'status',
            'data': {
                'summary': {
                    'active_devices': 3,
                    'total_devices': 11,
                    'power_usage': 1440.0,
                    'pending_jobs': 2,
                    'llm_provider': 'simulation'
                },
                'devices_by_location': {
                    'Kitchen': [
                        {'device_id': 'kitchen_dishwasher', 'friendly_name': 'Dishwasher', 'is_on': True, 'power_consumption': 1440.0},
                        {'device_id': 'kitchen_coffee_maker', 'friendly_name': 'Coffee Maker', 'is_on': False, 'power_consumption': 0.0}
                    ],
                    'Living Room': [
                        {'device_id': 'living_room_lights', 'friendly_name': 'Main Lights', 'is_on': True, 'power_consumption': 120.0},
                        {'device_id': 'living_room_tv', 'friendly_name': 'Smart TV', 'is_on': False, 'power_consumption': 0.0}
                    ]
                },
                'scheduled_jobs': [
                    {'job_id': 'job_001', 'device': 'coffee_maker', 'action': 'turn_on', 'scheduled_time': '07:00', 'status': 'pending'},
                    {'job_id': 'job_002', 'device': 'dishwasher', 'action': 'turn_off', 'scheduled_time': '14:30', 'status': 'pending'}
                ],
                'power_limit': 15000.0,
                'safety_margin': 0.8
            }
        }
    elif command.lower() == "recommendations":
        recommendations = [
            "Consider turning on the coffee maker for your morning routine",
            "Good time to charge your EV with lower electricity rates",
            "High power usage detected. Consider turning off non-essential devices"
        ]
        return {
            'success': True,
            'type': 'recommendations',
            'data': recommendations
        }
    elif command.lower() == "emergency":
        return {
            'success': True,
            'type': 'emergency',
            'message': 'Emergency shutdown completed (simulation)',
            'data': {'shutdown_devices': 3, 'cancelled_jobs': 2}
        }
    elif "turn on" in command.lower():
        device = command.lower().replace("turn on", "").strip()
        return {
            'success': True,
            'type': 'device_control',
            'action': 'turn_on',
            'device': device,
            'message': f'{device} action completed (simulation)',
            'confidence': 0.95
        }
    elif "turn off" in command.lower():
        device = command.lower().replace("turn off", "").strip()
        return {
            'success': True,
            'type': 'device_control',
            'action': 'turn_off',
            'device': device,
            'message': f'{device} turned off (simulation)',
            'confidence': 0.95
        }
    else:
        return {
            'success': False,
            'error': 'Could not understand the command (simulation mode)',
            'suggestion': 'Try commands like: turn on dishwasher, status, recommendations'
        }


def display_command_result_horizontal(result: Dict):
    """Display command result in a large, horizontal layout"""
    
    if result['success']:
        if result.get('type') == 'status':
            # Status display in horizontal layout
            st.success("📊 **SYSTEM STATUS RETRIEVED**")
            
            data = result['data']
            summary = data['summary']
            
            # Main metrics in a large horizontal container
            st.markdown("### 🏠 **Smart Home System Status**")
            
            # System Overview in larger metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="🔌 **Active Devices**", 
                    value=f"{summary['active_devices']}/{summary['total_devices']}",
                    help="Number of devices currently turned on"
                )
            
            with col2:
                # Handle both old simulation and new real data formats
                power_value = summary.get('power_usage', summary.get('total_power_consumption', 0))
                st.metric(
                    label="⚡ **Power Usage**", 
                    value=f"{power_value:.1f}W",
                    help="Current total power consumption"
                )
            
            with col3:
                st.metric(
                    label="📅 **Pending Jobs**", 
                    value=summary['pending_jobs'],
                    help="Number of scheduled actions waiting"
                )
            
            with col4:
                # Handle both old simulation and new real data formats
                llm_provider = data.get('primary_llm_provider', summary.get('llm_provider', 'openai'))
                st.metric(
                    label="🧠 **LLM Provider**", 
                    value=llm_provider.upper(),
                    help="AI provider for command processing"
                )
            
            # Power Management in horizontal layout
            st.markdown("### ⚡ **Power Management**")
            power_usage = summary.get('power_usage', summary.get('total_power_consumption', 0))
            power_limit = summary.get('power_limit', data.get('power_limit', 15000.0))
            usage_percentage = (power_usage / power_limit) * 100
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.progress(usage_percentage / 100)
                st.write(f"**Power Usage: {power_usage:.0f}W / {power_limit:.0f}W ({usage_percentage:.1f}%)**")
            with col2:
                if usage_percentage < 50:
                    st.markdown("### 🟢 **OPTIMAL**")
                elif usage_percentage < 75:
                    st.markdown("### 🟡 **MODERATE**")
                else:
                    st.markdown("### 🔴 **HIGH**")
            
            # Devices by location in horizontal tabs
            st.markdown("### 🏠 **Devices by Location**")
            devices_by_location = data['devices_by_location']
            
            if devices_by_location:
                tabs = st.tabs(list(devices_by_location.keys()))
                
                for i, (location, devices) in enumerate(devices_by_location.items()):
                    with tabs[i]:
                        for device in devices:
                            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                            
                            with col1:
                                status_icon = "🟢" if device['is_on'] else "🔴"
                                st.write(f"{status_icon} **{device['friendly_name']}**")
                            
                            with col2:
                                st.write(f"ID: `{device['device_id']}`")
                            
                            with col3:
                                power_consumption = device.get('power_consumption', 0)
                                st.write(f"⚡ {power_consumption:.0f}W")
                            
                            with col4:
                                status_text = "ON" if device['is_on'] else "OFF"
                                color = "green" if device['is_on'] else "red"
                                st.markdown(f"<span style='color: {color}; font-weight: bold;'>{status_text}</span>", unsafe_allow_html=True)
            
            # Scheduled jobs in horizontal layout
            scheduled_jobs = data.get('scheduled_jobs', [])
            if scheduled_jobs:
                st.markdown("### 📅 **Scheduled Jobs**")
                for job in scheduled_jobs:
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    
                    with col1:
                        job_id = job.get('job_id', job.get('id', 'unknown'))
                        st.write(f"🆔 **{job_id}**")
                    
                    with col2:
                        device_name = job.get('device', job.get('device_id', 'unknown'))
                        st.write(f"🔌 **{device_name}**")
                    
                    with col3:
                        action = job.get('action', 'unknown')
                        action_icon = "🟢" if action == 'turn_on' else "🔴"
                        st.write(f"{action_icon} **{action}**")
                    
                    with col4:
                        scheduled_time = job.get('scheduled_time', job.get('time', 'unknown'))
                        st.write(f"⏰ **{scheduled_time}**")
            else:
                st.info("📅 No scheduled jobs pending")
        
        elif result.get('type') == 'recommendations':
            st.success("💡 **SMART RECOMMENDATIONS RETRIEVED**")
            
            recommendations = result.get('data', [])
            if recommendations:
                # Display recommendations in horizontal cards
                st.markdown("### 🎯 **Smart Device Recommendations**")
                
                for i, rec in enumerate(recommendations):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{i+1}.** {rec}")
                        with col2:
                            st.write("💡")
            else:
                st.info("No recommendations available at this time")
        
        elif result.get('type') == 'device_control':
            st.success("🔌 **DEVICE CONTROL SUCCESS**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Device", result.get('device', 'Unknown'))
            with col2:
                st.metric("Action", result.get('action', 'Unknown'))
            with col3:
                confidence = result.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")
            
            if result.get('message'):
                st.info(f"📋 **Result:** {result['message']}")
        
        elif result.get('type') == 'emergency':
            st.error("🚨 **EMERGENCY ACTION COMPLETED**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Message:** {result.get('message', 'Emergency completed')}")
            with col2:
                data = result.get('data', {})
                if data:
                    st.write(f"**Devices affected:** {data.get('shutdown_devices', 0)}")
        
        else:
            st.success(f"✅ **COMMAND COMPLETED**")
            if result.get('message'):
                st.info(result['message'])
    
    else:
        st.error("❌ **COMMAND FAILED**")
        st.error(f"**Error:** {result.get('error', 'Unknown error')}")
        if result.get('suggestion'):
            st.info(f"💡 **Suggestion:** {result['suggestion']}")


def display_command_result(result: Dict, show_in_expander: bool = False):
    """Display the result of a command execution"""
    
    if not show_in_expander:
        st.markdown("### � Command Result")
    
    if result['success']:
        if result.get('type') == 'status':
            st.success("📊 **SYSTEM STATUS RETRIEVED**")
            
            # Create a prominent status display
            st.markdown("---")
            st.markdown("## 🏠 Smart Home System Status")
            
            data = result['data']
            summary = data['summary']
            
            # Main metrics in a highlighted container
            with st.container():
                st.markdown("### 📈 **System Overview**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="🔌 **Active Devices**", 
                        value=f"{summary['active_devices']}/{summary['total_devices']}",
                        help="Number of devices currently turned on"
                    )
                
                with col2:
                    # Handle both old simulation and new real data formats
                    power_value = summary.get('power_usage', summary.get('total_power_consumption', 0))
                    st.metric(
                        label="⚡ **Power Usage**", 
                        value=f"{power_value:.1f}W",
                        help="Current total power consumption"
                    )
                
                with col3:
                    st.metric(
                        label="📅 **Pending Jobs**", 
                        value=summary['pending_jobs'],
                        help="Number of scheduled actions waiting"
                    )
                
                with col4:
                    # Handle both old simulation and new real data formats
                    llm_provider = data.get('primary_llm_provider', summary.get('llm_provider', 'openai'))
                    st.metric(
                        label="🧠 **LLM Provider**", 
                        value=llm_provider.upper(),
                        help="AI provider for command processing"
                    )
            
            # Power status with visual indicator
            st.markdown("### ⚡ **Power Management**")
            # Handle both old simulation and new real data formats
            power_usage = summary.get('power_usage', summary.get('total_power_consumption', 0))
            power_limit = summary.get('power_limit', data.get('power_limit', 15000.0))  # Get from summary first, then data
            usage_percentage = (power_usage / power_limit) * 100
            
            # Power usage bar
            if usage_percentage < 50:
                bar_color = "🟢"
                status_text = "OPTIMAL"
            elif usage_percentage < 75:
                bar_color = "🟡"
                status_text = "MODERATE"
            else:
                bar_color = "🔴"
                status_text = "HIGH"
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(usage_percentage / 100)
                st.write(f"**Power Usage: {power_usage:.0f}W / {power_limit:.0f}W ({usage_percentage:.1f}%)**")
            with col2:
                st.markdown(f"### {bar_color} **{status_text}**")
            
            # Devices by location
            st.markdown("### 🏠 **Devices by Location**")
            devices_by_location = data['devices_by_location']
            
            for location, devices in devices_by_location.items():
                with st.expander(f"📍 **{location}** ({len([d for d in devices if d['is_on']])}/{len(devices)} active)"):
                    for device in devices:
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                        
                        with col1:
                            status_icon = "🟢" if device['is_on'] else "🔴"
                            st.write(f"{status_icon} **{device['friendly_name']}**")
                        
                        with col2:
                            st.write(f"ID: `{device['device_id']}`")
                        
                        with col3:
                            if device['is_on']:
                                st.write(f"⚡ {device['power_consumption']:.0f}W")
                            else:
                                st.write("⚡ 0W")
                        
                        with col4:
                            status_text = "ON" if device['is_on'] else "OFF"
                            if device['is_on']:
                                st.markdown(f"<span style='color: green; font-weight: bold;'>{status_text}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color: red; font-weight: bold;'>{status_text}</span>", unsafe_allow_html=True)
            
            # Scheduled jobs
            scheduled_jobs = data.get('scheduled_jobs', [])
            if scheduled_jobs:
                st.markdown("### 📅 **Scheduled Jobs**")
                for job in scheduled_jobs:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                        
                        with col1:
                            job_id = job.get('job_id', job.get('id', 'unknown'))
                            st.write(f"🆔 **{job_id}**")
                        
                        with col2:
                            device_name = job.get('device', job.get('device_id', 'unknown'))
                            st.write(f"🔌 **{device_name}**")
                        
                        with col3:
                            action = job.get('action', 'unknown')
                            action_icon = "🟢" if action == 'turn_on' else "🔴"
                            st.write(f"{action_icon} **{action}**")
                        
                        with col4:
                            scheduled_time = job.get('scheduled_time', job.get('time', 'unknown'))
                            st.write(f"⏰ **{scheduled_time}**")
            else:
                st.info("📅 No scheduled jobs pending")
            
            st.markdown("---")
            st.success("✅ **Status information updated successfully!**")
        
        elif result.get('type') == 'recommendations':
            st.success("💡 **SMART RECOMMENDATIONS RETRIEVED**")
            
            st.markdown("---")
            st.markdown("## 🤖 **AI-Powered Energy Recommendations**")
            
            recommendations = result['data']
            
            st.info(f"📋 **Found {len(recommendations)} personalized recommendations for your home**")
            
            for i, rec in enumerate(recommendations, 1):
                # Create a visually appealing recommendation card
                with st.container():
                    st.markdown(f"### 💡 **Recommendation #{i}**")
                    
                    # Determine recommendation type and icon
                    if "coffee" in rec.lower():
                        icon = "☕"
                        priority = "Low"
                    elif "power" in rec.lower() or "high" in rec.lower():
                        icon = "⚠️"
                        priority = "High"
                    elif "charge" in rec.lower() or "ev" in rec.lower():
                        icon = "🔋"
                        priority = "Medium"
                    else:
                        icon = "💡"
                        priority = "Medium"
                    
                    # Priority color coding
                    if priority == "High":
                        st.error(f"{icon} **{rec}**")
                    elif priority == "Medium":
                        st.warning(f"{icon} **{rec}**")
                    else:
                        st.info(f"{icon} **{rec}**")
                    
                    # Add potential savings if applicable
                    if "power" in rec.lower():
                        st.caption("💰 Potential savings: €15-25/month")
                    elif "charge" in rec.lower():
                        st.caption("💰 Potential savings: €8-12/month")
                    elif "coffee" in rec.lower():
                        st.caption("⏰ Convenience improvement")
            
            st.markdown("---")
            st.success("✅ **Recommendations updated successfully!**")
        
        elif result.get('type') == 'device_control':
            action_emoji = "🟢" if result['action'] == 'turn_on' else "🔴"
            action_text = "TURNED ON" if result['action'] == 'turn_on' else "TURNED OFF"
            
            # Create a prominent success message
            if result['action'] == 'turn_on':
                st.success(f"✅ **DEVICE ACTIVATED SUCCESSFULLY**")
            else:
                st.info(f"⏹️ **DEVICE DEACTIVATED SUCCESSFULLY**")
            
            # Device details in a container
            with st.container():
                st.markdown(f"### {action_emoji} **{result['device'].upper()} {action_text}**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"📱 **Device:** {result['device'].title()}")
                    st.write(f"🎯 **Action:** {action_text}")
                
                with col2:
                    if 'confidence' in result:
                        confidence_percentage = result['confidence'] * 100
                        st.metric(
                            label="🤖 **AI Confidence**", 
                            value=f"{confidence_percentage:.0f}%",
                            help="How confident the AI is about understanding your command"
                        )
                
                # Add estimated power impact
                if result['action'] == 'turn_on':
                    if 'dishwasher' in result['device'].lower():
                        st.info("⚡ **Estimated Power Draw:** 1440W")
                        st.caption("💰 Running cost: ~€0.35/hour at current rates")
                    elif 'coffee' in result['device'].lower():
                        st.info("⚡ **Estimated Power Draw:** 800W")
                        st.caption("💰 Running cost: ~€0.18/hour at current rates")
                    elif 'lights' in result['device'].lower():
                        st.info("⚡ **Estimated Power Draw:** 120W")
                        st.caption("💰 Running cost: ~€0.03/hour at current rates")
                    else:
                        st.info("⚡ **Device activated successfully**")
                else:
                    st.info("⚡ **Power consumption reduced to 0W**")
                    st.caption("💰 Energy saving mode activated")
            
            st.markdown("---")
        
        elif result.get('type') == 'schedule':
            st.success(f"📅 {result['message']}")
            if 'scheduled_time' in result:
                st.info(f"⏰ Scheduled for: {result['scheduled_time']}")
        
        elif result.get('type') == 'emergency':
            st.error("🚨 **EMERGENCY SHUTDOWN EXECUTED**")
            
            # Create a prominent emergency status display
            with st.container():
                st.markdown("### 🚨 **EMERGENCY PROTOCOL ACTIVATED**")
                st.warning(f"**Status:** {result['message']}")
                
                if 'data' in result:
                    data = result['data']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="🔌 **Devices Shutdown**", 
                            value=data.get('shutdown_devices', 0),
                            help="Number of devices that were turned off"
                        )
                    
                    with col2:
                        st.metric(
                            label="📅 **Jobs Cancelled**", 
                            value=data.get('cancelled_jobs', 0),
                            help="Number of scheduled jobs that were cancelled"
                        )
                
                st.error("⚠️ **All non-essential devices have been turned off for safety**")
                st.info("ℹ️ Essential devices (heating, refrigeration) remain active")
            
            st.markdown("---")
        
        else:
            st.success(f"✅ {result.get('message', 'Command executed successfully')}")
    
    else:
        st.error(f"❌ {result.get('error', 'Unknown error')}")
        if 'suggestion' in result:
            st.info(f"💡 Suggestion: {result['suggestion']}")
    
    # Update session state based on successful actions
    if result['success']:
        if result.get('type') == 'device_control':
            st.session_state['schedules_today'] = st.session_state.get('schedules_today', 0) + 1
            st.session_state['monthly_savings'] = st.session_state.get('monthly_savings', 0) + random.uniform(0.5, 2.0)
            st.session_state['carbon_saved'] = st.session_state.get('carbon_saved', 0) + random.uniform(0.1, 0.5)


def generate_mock_schedule(df, primary_goal, target_savings):
    """Generate a mock smart schedule for demonstration"""
    from datetime import datetime, timedelta
    import random
    
    # Analyze current usage patterns
    if 'timestamp' not in df.columns or 'import_kw' not in df.columns:
        return None
    
    # Generate schedule for next 7 days
    schedule = {
        'optimization_summary': {
            'projected_savings': target_savings * 0.8,  # 80% of target
            'efficiency_gain': random.uniform(15, 25),
            'carbon_reduction': random.uniform(5, 15),
            'comfort_score': random.uniform(85, 95)
        },
        'daily_schedules': [],
        'recommendations': []
    }
    
    # Generate daily schedules
    for day in range(7):
        date = datetime.now() + timedelta(days=day)
        
        daily_schedule = {
            'date': date.strftime('%Y-%m-%d'),
            'day_name': date.strftime('%A'),
            'scheduled_actions': []
        }
        
        # Generate some scheduled actions
        actions = [
            {
                'time': '02:00',
                'action': '🔌 Start dishwasher (off-peak rates)',
                'device': 'Dishwasher',
                'savings': f'€{random.uniform(0.5, 1.5):.2f}',
                'priority': 'Low'
            },
            {
                'time': '03:30',
                'action': '🧺 Start washing machine (cheapest rates)',
                'device': 'Washing Machine',
                'savings': f'€{random.uniform(0.8, 2.0):.2f}',
                'priority': 'Medium'
            },
            {
                'time': '06:00',
                'action': '🌡️ Pre-heat house before peak rates',
                'device': 'Heating System',
                'savings': f'€{random.uniform(1.0, 3.0):.2f}',
                'priority': 'High'
            },
            {
                'time': '22:30',
                'action': '🔋 Charge electric devices (night rate)',
                'device': 'Multiple Devices',
                'savings': f'€{random.uniform(0.3, 1.0):.2f}',
                'priority': 'Low'
            }
        ]
        
        # Randomly select 2-4 actions per day
        selected_actions = random.sample(actions, random.randint(2, 4))
        daily_schedule['scheduled_actions'] = selected_actions
        
        schedule['daily_schedules'].append(daily_schedule)
    
    # Generate recommendations
    recommendations = [
        {
            'title': 'Shift High-Usage Activities',
            'description': 'Move energy-intensive tasks to off-peak hours (11 PM - 7 AM)',
            'impact': 'High',
            'savings': f'€{random.uniform(15, 25):.0f}/month'
        },
        {
            'title': 'Smart Thermostat Integration',
            'description': 'Optimize heating/cooling based on occupancy and rate schedules',
            'impact': 'Medium',
            'savings': f'€{random.uniform(8, 15):.0f}/month'
        },
        {
            'title': 'Load Balancing',
            'description': 'Distribute appliance usage to avoid peak demand charges',
            'impact': 'Medium',
            'savings': f'€{random.uniform(5, 12):.0f}/month'
        }
    ]
    
    schedule['recommendations'] = recommendations
    
    return schedule


def show_schedule_results(schedule_data):
    """Display the generated smart schedule results"""
    
    # Optimization Summary
    st.subheader("📈 Optimization Summary")
    
    summary = schedule_data['optimization_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Projected Savings",
            f"€{summary['projected_savings']:.0f}/month"
        )
    
    with col2:
        st.metric(
            "Efficiency Gain",
            f"{summary['efficiency_gain']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Carbon Reduction",
            f"{summary['carbon_reduction']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Comfort Score",
            f"{summary['comfort_score']:.0f}/100"
        )
    
    # Weekly Schedule
    st.subheader("📅 7-Day Smart Schedule")
    
    # Create tabs for each day
    tab_names = [schedule['day_name'] for schedule in schedule_data['daily_schedules']]
    tabs = st.tabs(tab_names)
    
    for i, (tab, daily_schedule) in enumerate(zip(tabs, schedule_data['daily_schedules'])):
        with tab:
            st.markdown(f"### {daily_schedule['date']} - {daily_schedule['day_name']}")
            
            if daily_schedule['scheduled_actions']:
                for action in daily_schedule['scheduled_actions']:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 4, 2, 2])
                        
                        with col1:
                            st.write(f"**{action['time']}**")
                        
                        with col2:
                            st.write(action['action'])
                        
                        with col3:
                            st.write(f"💰 {action['savings']}")
                        
                        with col4:
                            priority_color = {
                                'High': '🔴',
                                'Medium': '🟡', 
                                'Low': '🟢'
                            }
                            st.write(f"{priority_color.get(action['priority'], '⚪')} {action['priority']}")
                
                # Daily summary
                daily_savings = sum(float(action['savings'].replace('€', '')) for action in daily_schedule['scheduled_actions'])
                st.info(f"💰 **Daily Savings: €{daily_savings:.2f}**")
            else:
                st.info("No optimizations needed for this day - already efficient!")
    
    # Smart Recommendations
    st.subheader("💡 Additional Smart Recommendations")
    
    for i, rec in enumerate(schedule_data['recommendations']):
        with st.expander(f"{rec['title']} - {rec['savings']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**Impact Level:** {rec['impact']}")
            
            with col2:
                st.write(f"**Monthly Savings:** {rec['savings']}")
                
                # Add action button
                if st.button(f"✅ Implement", key=f"implement_{i}"):
                    st.success(f"✅ {rec['title']} has been added to your smart schedule!")
    
    # Schedule Actions
    st.subheader("🎯 Schedule Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📅 Apply This Schedule", type="primary"):
            st.success("✅ Smart schedule applied! The agent will now manage your devices automatically.")
            st.session_state['agent_active'] = True
            st.session_state['schedules_today'] = len(schedule_data['daily_schedules'][0]['scheduled_actions'])
    
    with col2:
        if st.button("📝 Modify Schedule"):
            st.info("🔄 Returning to configuration... (Feature coming soon)")
    
    with col3:
        if st.button("💾 Save as Template"):
            st.info("💾 Schedule saved as template for future use!")


if __name__ == "__main__":
    main()
