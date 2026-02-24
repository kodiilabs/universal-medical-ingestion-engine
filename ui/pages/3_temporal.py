# ============================================================================
# ui/pages/3_temporal.py
# ============================================================================
"""
Temporal Analysis Page

Displays lab value trends over time.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.components.trend_graph import (
    render_trend_chart,
    render_panel_trends,
    render_trend_summary,
    render_critical_value_alert
)


def generate_sample_trend_data(test_name: str, num_points: int = 10):
    """Generate sample trend data for demo."""
    import random

    base_values = {
        'Hemoglobin': (14.0, 1.0),
        'Glucose': (95, 15),
        'Creatinine': (1.0, 0.2),
        'WBC': (7.0, 1.5),
        'Platelets': (250, 50),
        'Sodium': (140, 3),
        'Potassium': (4.2, 0.4),
    }

    base, std = base_values.get(test_name, (100, 10))

    data_points = []
    start_date = datetime.now() - timedelta(days=365)

    for i in range(num_points):
        date = start_date + timedelta(days=i * (365 // num_points))
        value = base + random.gauss(0, std)
        data_points.append({
            'date': date.isoformat(),
            'value': round(value, 2)
        })

    return data_points


def main():
    st.title("📈 Temporal Analysis")

    st.markdown("""
    Analyze lab value trends over time. Track changes and identify patterns.
    """)

    st.divider()

    # Test selection
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_panel = st.selectbox(
            "Select Panel",
            ["CBC", "CMP", "Lipid Panel", "Thyroid Panel", "All Tests"]
        )

    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["Last 30 days", "Last 90 days", "Last year", "All time"]
        )

    st.divider()

    # Panel-specific tests
    panel_tests = {
        "CBC": ["Hemoglobin", "Hematocrit", "WBC", "Platelets", "RBC"],
        "CMP": ["Glucose", "Creatinine", "Sodium", "Potassium", "Calcium", "BUN"],
        "Lipid Panel": ["Total Cholesterol", "LDL", "HDL", "Triglycerides"],
        "Thyroid Panel": ["TSH", "Free T4", "Free T3"]
    }

    reference_ranges = {
        "Hemoglobin": (13.5, 17.5),
        "Hematocrit": (38.8, 50.0),
        "WBC": (4.5, 11.0),
        "Platelets": (150, 400),
        "RBC": (4.5, 5.9),
        "Glucose": (70, 100),
        "Creatinine": (0.7, 1.3),
        "Sodium": (136, 145),
        "Potassium": (3.5, 5.0),
        "Calcium": (8.5, 10.5),
        "BUN": (7, 20),
        "Total Cholesterol": (0, 200),
        "LDL": (0, 100),
        "HDL": (40, 200),
        "Triglycerides": (0, 150),
        "TSH": (0.4, 4.0),
        "Free T4": (0.8, 1.8),
        "Free T3": (2.3, 4.2)
    }

    units = {
        "Hemoglobin": "g/dL",
        "Hematocrit": "%",
        "WBC": "K/uL",
        "Platelets": "K/uL",
        "RBC": "M/uL",
        "Glucose": "mg/dL",
        "Creatinine": "mg/dL",
        "Sodium": "mEq/L",
        "Potassium": "mEq/L",
        "Calcium": "mg/dL",
        "BUN": "mg/dL",
        "Total Cholesterol": "mg/dL",
        "LDL": "mg/dL",
        "HDL": "mg/dL",
        "Triglycerides": "mg/dL",
        "TSH": "mIU/L",
        "Free T4": "ng/dL",
        "Free T3": "pg/mL"
    }

    if selected_panel == "All Tests":
        tests = [t for tests in panel_tests.values() for t in tests]
    else:
        tests = panel_tests.get(selected_panel, [])

    # Test selector
    selected_test = st.selectbox("Select Test", tests)

    st.divider()

    # Generate sample data
    trend_data = generate_sample_trend_data(selected_test)
    ref_range = reference_ranges.get(selected_test)
    unit = units.get(selected_test, "")

    # Render trend chart
    render_trend_chart(
        data_points=trend_data,
        test_name=selected_test,
        unit=unit,
        reference_range=ref_range
    )

    st.divider()

    # Critical value check
    if trend_data:
        latest = trend_data[-1]['value']

        # Check for critical values (using double the reference range as critical)
        if ref_range:
            critical_low = ref_range[0] * 0.7
            critical_high = ref_range[1] * 1.3

            if latest < critical_low:
                render_critical_value_alert(
                    selected_test,
                    latest,
                    unit,
                    (critical_low, critical_high),
                    "low"
                )
            elif latest > critical_high:
                render_critical_value_alert(
                    selected_test,
                    latest,
                    unit,
                    (critical_low, critical_high),
                    "high"
                )

    st.divider()

    # Panel overview
    st.subheader(f"{selected_panel} Overview")

    if selected_panel != "All Tests":
        panel_data = {}
        for test in tests[:4]:  # Limit to 4 for display
            data = generate_sample_trend_data(test, 5)
            if len(data) >= 2:
                panel_data[test] = {
                    'current': data[-1]['value'],
                    'previous': data[-2]['value'],
                    'unit': units.get(test, '')
                }

        if panel_data:
            render_panel_trends(selected_panel, panel_data)

    st.divider()

    # Trend summary
    st.subheader("All Test Summary")

    trends = {}
    for test in tests[:6]:
        trends[test] = generate_sample_trend_data(test, 5)

    render_trend_summary(trends, reference_ranges)


if __name__ == "__main__":
    main()
else:
    main()
