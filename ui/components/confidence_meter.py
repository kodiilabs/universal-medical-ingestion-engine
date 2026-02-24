# ============================================================================
# ui/components/confidence_meter.py
# ============================================================================
"""
Confidence Meter Component

Displays extraction confidence scores with visual indicators.
"""

import streamlit as st
from typing import Dict, Any, Optional


def render_confidence_meter(
    confidence: float,
    label: str = "Confidence",
    show_percentage: bool = True
) -> None:
    """
    Render a confidence meter with color-coded progress bar.

    Args:
        confidence: Confidence score between 0 and 1
        label: Label to display
        show_percentage: Whether to show percentage text
    """
    # Clamp confidence to valid range
    confidence = max(0.0, min(1.0, confidence))

    # Determine color based on confidence level
    if confidence >= 0.9:
        color = "green"
        status = "High"
    elif confidence >= 0.7:
        color = "blue"
        status = "Good"
    elif confidence >= 0.5:
        color = "orange"
        status = "Medium"
    else:
        color = "red"
        status = "Low"

    # Display label and status
    col1, col2 = st.columns([3, 1])

    with col1:
        st.write(f"**{label}**")

    with col2:
        if show_percentage:
            st.write(f"{confidence * 100:.1f}%")

    # Progress bar
    st.progress(confidence)

    # Status indicator
    if confidence >= 0.9:
        st.success(f"{status} confidence")
    elif confidence >= 0.7:
        st.info(f"{status} confidence")
    elif confidence >= 0.5:
        st.warning(f"{status} confidence")
    else:
        st.error(f"{status} confidence - manual review recommended")


def render_confidence_breakdown(
    scores: Dict[str, float],
    title: str = "Confidence Breakdown"
) -> None:
    """
    Render a breakdown of multiple confidence scores.

    Args:
        scores: Dict of label -> confidence score
        title: Section title
    """
    st.subheader(title)

    for label, score in scores.items():
        render_confidence_meter(score, label=label)
        st.divider()


def render_extraction_confidence(
    extraction_result: Dict[str, Any]
) -> None:
    """
    Render confidence meters for an extraction result.

    Args:
        extraction_result: Extraction result with confidence scores
    """
    st.subheader("Extraction Confidence")

    # Overall confidence
    overall = extraction_result.get('overall_confidence', 0.0)
    render_confidence_meter(overall, label="Overall")

    st.divider()

    # Field-level confidence
    fields = extraction_result.get('fields', {})
    if fields:
        st.write("**Field-Level Confidence:**")

        for field_name, field_data in fields.items():
            if isinstance(field_data, dict):
                conf = field_data.get('confidence', 0.0)
            else:
                conf = 1.0

            with st.expander(field_name):
                render_confidence_meter(conf, label=field_name, show_percentage=True)


def render_validation_confidence(
    validation_result: Dict[str, Any]
) -> None:
    """
    Render confidence for validation results.

    Args:
        validation_result: Validation result with scores
    """
    st.subheader("Validation Confidence")

    # Overall validation score
    is_valid = validation_result.get('is_valid', False)
    errors = validation_result.get('errors', [])
    warnings = validation_result.get('warnings', [])

    # Calculate confidence from validation
    if is_valid and not warnings:
        confidence = 1.0
    elif is_valid and warnings:
        confidence = 0.8 - (len(warnings) * 0.1)
    else:
        confidence = 0.3 - (len(errors) * 0.1)

    confidence = max(0.0, confidence)

    render_confidence_meter(confidence, label="Validation Score")

    # Show issues
    if errors:
        st.error(f"**Errors ({len(errors)}):**")
        for error in errors:
            st.write(f"- {error}")

    if warnings:
        st.warning(f"**Warnings ({len(warnings)}):**")
        for warning in warnings:
            st.write(f"- {warning}")


def confidence_badge(confidence: float, size: str = "small") -> str:
    """
    Return HTML for a confidence badge.

    Args:
        confidence: Confidence score
        size: Badge size (small, medium, large)

    Returns:
        HTML string for badge
    """
    if confidence >= 0.9:
        color = "#28a745"
        text = "HIGH"
    elif confidence >= 0.7:
        color = "#17a2b8"
        text = "GOOD"
    elif confidence >= 0.5:
        color = "#ffc107"
        text = "MED"
    else:
        color = "#dc3545"
        text = "LOW"

    font_size = {"small": "10px", "medium": "12px", "large": "14px"}.get(size, "10px")

    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: {font_size};
        font-weight: bold;
    ">{text} {confidence * 100:.0f}%</span>
    """
