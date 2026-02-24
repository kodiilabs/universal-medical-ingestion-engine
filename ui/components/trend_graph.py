# ============================================================================
# ui/components/trend_graph.py
# ============================================================================
"""
Trend Graph Component

Displays lab value trends over time with reference ranges.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime


def render_trend_chart(
    data_points: List[Dict[str, Any]],
    test_name: str,
    unit: str = "",
    reference_range: Optional[tuple] = None
) -> None:
    """
    Render a trend chart for lab values.

    Args:
        data_points: List of dicts with 'date' and 'value' keys
        test_name: Name of the test
        unit: Unit of measurement
        reference_range: Tuple of (low, high) reference values
    """
    import pandas as pd

    if not data_points:
        st.info(f"No data available for {test_name}")
        return

    # Prepare data
    df = pd.DataFrame(data_points)

    if 'date' not in df.columns or 'value' not in df.columns:
        st.error("Invalid data format. Expected 'date' and 'value' columns.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    st.subheader(f"{test_name} Trend")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Latest", f"{df['value'].iloc[-1]:.2f} {unit}")

    with col2:
        st.metric("Min", f"{df['value'].min():.2f} {unit}")

    with col3:
        st.metric("Max", f"{df['value'].max():.2f} {unit}")

    with col4:
        st.metric("Avg", f"{df['value'].mean():.2f} {unit}")

    # Reference range info
    if reference_range:
        low, high = reference_range
        st.info(f"Reference Range: {low} - {high} {unit}")

        # Check if latest value is in range
        latest = df['value'].iloc[-1]
        if latest < low:
            st.warning(f"Latest value is below reference range")
        elif latest > high:
            st.warning(f"Latest value is above reference range")
        else:
            st.success(f"Latest value is within reference range")

    # Line chart
    chart_data = df.set_index('date')['value']
    st.line_chart(chart_data)

    # Data table
    with st.expander("View Data"):
        display_df = df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df)


def render_multi_trend_chart(
    trends: Dict[str, List[Dict[str, Any]]],
    title: str = "Lab Value Trends"
) -> None:
    """
    Render multiple trends on a single chart.

    Args:
        trends: Dict of test_name -> list of data points
        title: Chart title
    """
    import pandas as pd

    st.subheader(title)

    if not trends:
        st.info("No trend data available")
        return

    # Combine all trends into a single DataFrame
    all_data = []
    for test_name, data_points in trends.items():
        for dp in data_points:
            all_data.append({
                'date': dp.get('date'),
                'value': dp.get('value'),
                'test': test_name
            })

    if not all_data:
        st.info("No data points available")
        return

    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])

    # Pivot for chart
    pivot_df = df.pivot(index='date', columns='test', values='value')
    pivot_df = pivot_df.sort_index()

    st.line_chart(pivot_df)

    # Legend
    st.write("**Tests:**")
    for test_name in trends.keys():
        st.write(f"- {test_name}")


def render_trend_comparison(
    current: float,
    previous: float,
    test_name: str,
    unit: str = ""
) -> None:
    """
    Render a comparison between current and previous values.

    Args:
        current: Current value
        previous: Previous value
        test_name: Test name
        unit: Unit of measurement
    """
    if previous == 0:
        delta_pct = 0
    else:
        delta_pct = ((current - previous) / previous) * 100

    delta = current - previous

    st.metric(
        label=test_name,
        value=f"{current:.2f} {unit}",
        delta=f"{delta:+.2f} ({delta_pct:+.1f}%)"
    )


def render_panel_trends(
    panel_name: str,
    tests: Dict[str, Dict[str, Any]]
) -> None:
    """
    Render trends for a panel of tests.

    Args:
        panel_name: Name of the panel (e.g., "CBC", "CMP")
        tests: Dict of test_name -> test data with 'current', 'previous', 'unit'
    """
    st.subheader(f"{panel_name} Trends")

    cols = st.columns(min(len(tests), 4))

    for i, (test_name, data) in enumerate(tests.items()):
        col_idx = i % len(cols)
        with cols[col_idx]:
            render_trend_comparison(
                current=data.get('current', 0),
                previous=data.get('previous', 0),
                test_name=test_name,
                unit=data.get('unit', '')
            )


def render_critical_value_alert(
    test_name: str,
    value: float,
    unit: str,
    critical_range: tuple,
    direction: str = "high"
) -> None:
    """
    Render alert for critical values.

    Args:
        test_name: Test name
        value: Current value
        unit: Unit of measurement
        critical_range: Tuple of (low, high) critical values
        direction: "high" or "low" indicating which threshold was crossed
    """
    low, high = critical_range

    if direction == "high":
        st.error(f"""
        **CRITICAL VALUE ALERT**

        **{test_name}**: {value} {unit}

        Value exceeds critical high threshold of {high} {unit}

        Immediate clinical attention recommended.
        """)
    else:
        st.error(f"""
        **CRITICAL VALUE ALERT**

        **{test_name}**: {value} {unit}

        Value below critical low threshold of {low} {unit}

        Immediate clinical attention recommended.
        """)


def render_sparkline(
    values: List[float],
    label: str = ""
) -> None:
    """
    Render a simple sparkline chart.

    Args:
        values: List of values
        label: Optional label
    """
    import pandas as pd

    if not values:
        return

    if label:
        st.write(f"**{label}**")

    df = pd.DataFrame({'value': values})
    st.area_chart(df, height=100)


def render_trend_summary(
    trends: Dict[str, List[Dict[str, Any]]],
    reference_ranges: Dict[str, tuple]
) -> None:
    """
    Render a summary of all trends with status indicators.

    Args:
        trends: Dict of test_name -> list of data points
        reference_ranges: Dict of test_name -> (low, high) tuple
    """
    st.subheader("Trend Summary")

    for test_name, data_points in trends.items():
        if not data_points:
            continue

        latest = data_points[-1].get('value', 0)
        ref_range = reference_ranges.get(test_name)

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(f"**{test_name}**")

        with col2:
            st.write(f"{latest:.2f}")

        with col3:
            if ref_range:
                low, high = ref_range
                if latest < low:
                    st.write("🔵 Low")
                elif latest > high:
                    st.write("🔴 High")
                else:
                    st.write("🟢 Normal")
            else:
                st.write("➖")

        # Mini sparkline
        values = [dp.get('value', 0) for dp in data_points[-10:]]
        render_sparkline(values)
