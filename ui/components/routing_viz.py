# ============================================================================
# ui/components/routing_viz.py
# ============================================================================
"""
Routing Visualization Component

Displays document classification and routing decisions.
"""

import streamlit as st
from typing import Dict, Any, List, Optional


DOCUMENT_TYPE_ICONS = {
    'lab_report': '🧪',
    'prescription': '💊',
    'radiology': '📷',
    'pathology': '🔬',
    'discharge_summary': '📋',
    'clinical_note': '📝',
    'unknown': '❓'
}

DOCUMENT_TYPE_COLORS = {
    'lab_report': '#28a745',
    'prescription': '#17a2b8',
    'radiology': '#6f42c1',
    'pathology': '#fd7e14',
    'discharge_summary': '#20c997',
    'clinical_note': '#6c757d',
    'unknown': '#dc3545'
}


def render_classification_result(
    classification: Dict[str, Any],
    show_scores: bool = True
) -> None:
    """
    Render document classification result.

    Args:
        classification: Classification result dict
        show_scores: Whether to show detailed scores
    """
    doc_type = classification.get('document_type', 'unknown')
    confidence = classification.get('confidence', 0.0)
    scores = classification.get('scores', {})

    icon = DOCUMENT_TYPE_ICONS.get(doc_type, '❓')
    color = DOCUMENT_TYPE_COLORS.get(doc_type, '#6c757d')

    # Main classification
    st.markdown(f"""
    <div style="
        background-color: {color}20;
        border-left: 4px solid {color};
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    ">
        <h3 style="margin: 0; color: {color};">
            {icon} {doc_type.replace('_', ' ').title()}
        </h3>
        <p style="margin: 5px 0 0 0; color: #666;">
            Confidence: {confidence * 100:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed scores
    if show_scores and scores:
        with st.expander("Classification Scores"):
            for doc_type_key, score in sorted(scores.items(), key=lambda x: -x[1]):
                icon = DOCUMENT_TYPE_ICONS.get(doc_type_key, '❓')
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{icon} {doc_type_key.replace('_', ' ').title()}")
                with col2:
                    st.write(f"{score * 100:.1f}%")
                st.progress(score)


def render_routing_decision(
    routing: Dict[str, Any]
) -> None:
    """
    Render routing decision visualization.

    Args:
        routing: Routing decision dict
    """
    st.subheader("Routing Decision")

    source = routing.get('source', 'Document')
    processor = routing.get('processor', 'Unknown')
    confidence = routing.get('confidence', 0.0)
    reason = routing.get('reason', '')

    # Flow visualization
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown("""
        <div style="
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        ">
            <h4>📄 Input</h4>
            <p>{}</p>
        </div>
        """.format(source), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 20px;
        ">
            <h2>➡️</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background-color: #e7f5ff;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        ">
            <h4>⚙️ Processor</h4>
            <p>{}</p>
        </div>
        """.format(processor), unsafe_allow_html=True)

    # Details
    if reason:
        st.info(f"**Reason:** {reason}")

    st.metric("Routing Confidence", f"{confidence * 100:.1f}%")


def render_processing_pipeline(
    stages: List[Dict[str, Any]]
) -> None:
    """
    Render processing pipeline visualization.

    Args:
        stages: List of processing stage dicts
    """
    st.subheader("Processing Pipeline")

    for i, stage in enumerate(stages):
        name = stage.get('name', f'Stage {i+1}')
        status = stage.get('status', 'pending')
        duration = stage.get('duration_ms', 0)

        # Status indicator
        if status == 'completed':
            icon = '✅'
            color = '#28a745'
        elif status == 'in_progress':
            icon = '🔄'
            color = '#ffc107'
        elif status == 'failed':
            icon = '❌'
            color = '#dc3545'
        else:
            icon = '⏳'
            color = '#6c757d'

        col1, col2, col3 = st.columns([1, 4, 1])

        with col1:
            st.write(f"**{i+1}**")

        with col2:
            st.markdown(f"""
            <div style="
                border-left: 3px solid {color};
                padding-left: 10px;
            ">
                {icon} <strong>{name}</strong>
                <br>
                <small style="color: #666;">Status: {status}</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if duration:
                st.write(f"{duration}ms")

        if i < len(stages) - 1:
            st.markdown("<div style='text-align: center;'>↓</div>", unsafe_allow_html=True)


def render_document_type_legend() -> None:
    """Render legend for document types."""
    st.subheader("Document Types")

    cols = st.columns(3)
    types = list(DOCUMENT_TYPE_ICONS.items())

    for i, (doc_type, icon) in enumerate(types):
        col_idx = i % 3
        with cols[col_idx]:
            color = DOCUMENT_TYPE_COLORS.get(doc_type, '#6c757d')
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                margin: 5px 0;
            ">
                <span style="font-size: 20px; margin-right: 8px;">{icon}</span>
                <span style="color: {color};">{doc_type.replace('_', ' ').title()}</span>
            </div>
            """, unsafe_allow_html=True)


def render_classification_comparison(
    classifications: List[Dict[str, Any]]
) -> None:
    """
    Render comparison of multiple classifications.

    Args:
        classifications: List of classification results
    """
    st.subheader("Classification Comparison")

    if not classifications:
        st.info("No classifications to compare.")
        return

    # Create comparison table
    data = []
    for i, c in enumerate(classifications):
        data.append({
            'Document': f"Doc {i+1}",
            'Type': c.get('document_type', 'unknown'),
            'Confidence': f"{c.get('confidence', 0) * 100:.1f}%"
        })

    st.table(data)
