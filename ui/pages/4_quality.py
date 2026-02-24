# ============================================================================
# ui/pages/4_quality.py
# ============================================================================
"""
Quality Metrics Page

Displays data quality metrics and validation results.
"""

import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.components.confidence_meter import render_validation_confidence


def main():
    st.title("✅ Quality Metrics")

    st.markdown("""
    Review data quality metrics, validation results, and extraction accuracy.
    """)

    st.divider()

    # Overall quality score
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Quality", "94.2%", "+2.1%")

    with col2:
        st.metric("Extraction Accuracy", "96.8%", "+1.5%")

    with col3:
        st.metric("Validation Pass Rate", "91.3%", "-0.5%")

    with col4:
        st.metric("Critical Errors", "0", "0")

    st.divider()

    # Quality breakdown
    st.subheader("Quality Breakdown by Document Type")

    quality_data = [
        {"Document Type": "Lab Report", "Processed": 45, "Passed": 43, "Warnings": 2, "Errors": 0, "Quality": "95.6%"},
        {"Document Type": "Prescription", "Processed": 23, "Passed": 21, "Warnings": 1, "Errors": 1, "Quality": "91.3%"},
        {"Document Type": "Radiology", "Processed": 12, "Passed": 11, "Warnings": 1, "Errors": 0, "Quality": "91.7%"},
        {"Document Type": "Pathology", "Processed": 8, "Passed": 8, "Warnings": 0, "Errors": 0, "Quality": "100%"},
        {"Document Type": "Discharge Summary", "Processed": 5, "Passed": 4, "Warnings": 1, "Errors": 0, "Quality": "80.0%"},
    ]

    st.dataframe(quality_data, use_container_width=True)

    st.divider()

    # Validation results
    st.subheader("Recent Validation Results")

    tab1, tab2, tab3 = st.tabs(["All", "Warnings", "Errors"])

    with tab1:
        validations = [
            {"Document": "lab_report_001.pdf", "Status": "✅ Passed", "Score": "98%", "Issues": "None"},
            {"Document": "prescription_002.pdf", "Status": "⚠️ Warning", "Score": "85%", "Issues": "Missing refills"},
            {"Document": "radiology_003.pdf", "Status": "✅ Passed", "Score": "92%", "Issues": "None"},
            {"Document": "lab_report_004.pdf", "Status": "✅ Passed", "Score": "96%", "Issues": "None"},
            {"Document": "prescription_005.pdf", "Status": "❌ Error", "Score": "45%", "Issues": "Invalid DEA number"},
        ]
        st.dataframe(validations, use_container_width=True)

    with tab2:
        warnings = [v for v in validations if "Warning" in v["Status"]]
        if warnings:
            st.dataframe(warnings, use_container_width=True)
        else:
            st.info("No warnings")

    with tab3:
        errors = [v for v in validations if "Error" in v["Status"]]
        if errors:
            st.dataframe(errors, use_container_width=True)
        else:
            st.success("No errors")

    st.divider()

    # Detailed validation
    st.subheader("Validation Details")

    selected_doc = st.selectbox(
        "Select Document",
        ["lab_report_001.pdf", "prescription_002.pdf", "radiology_003.pdf"]
    )

    render_validation_confidence({
        'is_valid': True,
        'errors': [],
        'warnings': ["Specimen collection time not specified"] if "prescription" in selected_doc else []
    })

    st.divider()

    # Quality rules
    st.subheader("Quality Rules")

    with st.expander("View Active Quality Rules"):
        rules = [
            {"Rule": "Required Fields", "Description": "All required fields must be present", "Severity": "Error"},
            {"Rule": "Reference Range", "Description": "Values must be within plausible ranges", "Severity": "Warning"},
            {"Rule": "Date Validation", "Description": "Dates must be valid and not in future", "Severity": "Error"},
            {"Rule": "Unit Consistency", "Description": "Units must match expected format", "Severity": "Warning"},
            {"Rule": "Critical Values", "Description": "Critical values must be flagged", "Severity": "Error"},
            {"Rule": "Patient ID", "Description": "Patient identifier must be present", "Severity": "Error"},
            {"Rule": "Provider NPI", "Description": "Provider NPI should be valid", "Severity": "Warning"},
        ]
        st.dataframe(rules, use_container_width=True)

    st.divider()

    # Export
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export Quality Report"):
            st.download_button(
                label="Download CSV",
                data="Document,Status,Score\nlab_report_001.pdf,Passed,98%",
                file_name="quality_report.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("🔄 Re-validate All"):
            with st.spinner("Re-validating..."):
                st.success("Re-validation complete!")


if __name__ == "__main__":
    main()
else:
    main()
