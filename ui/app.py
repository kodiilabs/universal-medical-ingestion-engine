
#!/usr/bin/env python3
# ============================================================================
# ui/app.py
# ============================================================================
"""
Medical Document Ingestion Engine - Streamlit UI

Main application entry point.

Usage:
    streamlit run ui/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def init_session_state():
    """Initialize session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []

    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None

    if 'audit_log' not in st.session_state:
        st.session_state.audit_log = []

    if 'fhir_bundles' not in st.session_state:
        st.session_state.fhir_bundles = []


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Medical Document Ingestion Engine",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # Main page content
    st.title("🏥 Medical Document Ingestion Engine")

    st.markdown("""
    Welcome to the **Medical Document Ingestion Engine** - an intelligent system for
    processing and standardizing medical documents.

    ### Features

    - **Document Classification**: Automatically identifies document types (lab reports,
      prescriptions, radiology reports, etc.)
    - **Data Extraction**: Extracts structured data using MedGemma AI
    - **FHIR Conversion**: Converts extracted data to FHIR R4 format
    - **Quality Assurance**: Validates extracted values against reference ranges
    - **Temporal Analysis**: Tracks lab values over time

    ### Getting Started

    1. **Upload** your medical documents using the Upload page
    2. **Review** the extracted results on the Results page
    3. **Analyze** temporal trends on the Temporal page
    4. **Validate** quality metrics on the Quality page
    5. **Export** FHIR bundles on the FHIR page
    6. **Review** the audit log on the Audit page

    Use the sidebar to navigate between pages.
    """)

    # Quick stats
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Documents Uploaded",
            value=len(st.session_state.uploaded_files)
        )

    with col2:
        st.metric(
            label="Documents Processed",
            value=len(st.session_state.processing_results)
        )

    with col3:
        st.metric(
            label="FHIR Bundles",
            value=len(st.session_state.fhir_bundles)
        )

    with col4:
        st.metric(
            label="Audit Events",
            value=len(st.session_state.audit_log)
        )

    # System status
    st.divider()
    st.subheader("System Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.success("Document Classifier: Ready")
        st.success("Lab Processor: Ready")
        st.success("Prescription Processor: Ready")

    with status_col2:
        st.success("Radiology Processor: Ready")
        st.success("FHIR Builder: Ready")
        st.info("MedGemma: Requires model download")


if __name__ == "__main__":
    main()
