"""
Energy Advisor MVP - Streamlit Application
Main application for analyzing Irish MPRN smart meter data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_parser import (
    MPRNDataParser,
    parse_mprn_file,
    validate_mprn_data,
    clean_and_resample,
)
from usage_analyzer import UsageAnalyzer

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
            "💰 Usage Patterns",
            "🔍 Appliance Detection",
            "💡 Recommendations",
        ],
    )

    if page == "📊 Data Upload & Analysis":
        show_data_upload_page()
    elif page == "💰 Usage Patterns":
        show_usage_patterns_page()
    elif page == "🔍 Appliance Detection":
        show_appliance_detection_page()
    elif page == "💡 Recommendations":
        show_recommendations_page()


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


def show_appliance_detection_page():
    """Appliance detection page (placeholder for now)"""
    st.header("🔍 Appliance Detection")
    st.info("🚧 This feature is under development. Coming in the next iteration!")

    if "parsed_data" in st.session_state:
        st.write("Data is available for appliance detection analysis.")
    else:
        st.warning("⚠️ Please upload data first on the Data Upload page.")


def show_recommendations_page():
    """Recommendations page (placeholder for now)"""
    st.header("💡 Energy Saving Recommendations")
    st.info("🚧 This feature is under development. Coming in the next iteration!")

    if "parsed_data" in st.session_state:
        st.write("Data is available for generating recommendations.")
    else:
        st.warning("⚠️ Please upload data first on the Data Upload page.")


if __name__ == "__main__":
    main()
