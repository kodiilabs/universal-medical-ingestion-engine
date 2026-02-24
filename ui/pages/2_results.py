# ============================================================================
# ui/pages/2_results.py
# ============================================================================
"""
Results Page

Displays extraction results from processed documents using actual data.
"""

import streamlit as st
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.components.confidence_meter import render_confidence_meter
from ui.components.routing_viz import render_classification_result


def main():
    st.title("📊 Extraction Results")

    st.markdown("""
    View and review extracted data from processed medical documents.
    """)

    st.divider()

    # Check for results
    results = st.session_state.get('processing_results', [])

    if not results:
        st.info("No documents have been processed yet. Go to the **Upload** page to process documents.")
        return

    # Document selector
    doc_names = [r.get('display_name') or r.get('file_name') or f"Document {i+1}" for i, r in enumerate(results)]
    selected_idx = st.selectbox("Select Document", range(len(doc_names)), format_func=lambda x: doc_names[x])

    selected_result = results[selected_idx]

    st.divider()

    # Document overview
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Document Info")
        st.write(f"**File:** {selected_result.get('display_name') or selected_result.get('file_name', 'Unknown')}")
        st.write(f"**Document ID:** {selected_result.get('document_id', 'N/A')}")
        st.write(f"**Type:** {selected_result.get('document_type', 'unknown').replace('_', ' ').title()}")
        st.write(f"**Status:** {'✅ Success' if selected_result.get('success') else '❌ Failed'}")
        st.write(f"**Processing Time:** {selected_result.get('processing_time', 0):.2f}s")

        if selected_result.get('requires_review'):
            st.warning(f"⚠️ Requires review: {selected_result.get('review_priority', 'medium')}")

    with col2:
        st.subheader("Classification")
        render_classification_result({
            'document_type': selected_result.get('document_type', 'unknown'),
            'confidence': selected_result.get('confidence', 0.0),
            'scores': {}
        }, show_scores=False)

    st.divider()

    # Confidence and warnings
    col1, col2 = st.columns(2)

    with col1:
        render_confidence_meter(
            selected_result.get('confidence', 0),
            label="Overall Confidence"
        )

    with col2:
        # Warnings and flags
        warnings = selected_result.get('warnings', [])
        quality_flags = selected_result.get('quality_flags', [])
        critical = selected_result.get('critical_findings', [])

        if critical:
            st.error(f"**Critical Findings:** {len(critical)}")
            for cf in critical:
                st.write(f"🚨 {cf}")

        if warnings:
            st.warning(f"**Warnings:** {len(warnings)}")
            for w in warnings:
                st.write(f"⚠️ {w}")

        if quality_flags:
            st.info(f"**Quality Flags:** {len(quality_flags)}")
            for qf in quality_flags:
                st.write(f"🔍 {qf}")

        if not critical and not warnings and not quality_flags:
            st.success("No issues detected")

    st.divider()

    # Extracted data
    st.subheader("Extracted Data")

    extracted = selected_result.get('extracted_data', {})
    doc_type = selected_result.get('document_type', 'unknown')

    # Dynamic tabs based on document type
    if doc_type == 'lab_report':
        tab_names = ["Lab Results", "Patient Info", "Raw JSON", "Audit Trail"]
    elif doc_type == 'radiology':
        tab_names = ["Findings", "Patient Info", "Raw JSON", "Audit Trail"]
    elif doc_type == 'prescription':
        tab_names = ["Medications", "Patient Info", "Raw JSON", "Audit Trail"]
    else:
        tab_names = ["Extracted Data", "Patient Info", "Raw JSON", "Audit Trail"]

    tab1, tab2, tab3, tab4 = st.tabs(tab_names)

    with tab1:
        # Content varies by document type
        if doc_type == 'lab_report':
            # Lab results table
            if 'lab_results' in extracted:
                lab_results = extracted['lab_results']
                st.write(f"**{len(lab_results)} lab values extracted:**")

                table_data = []
                for result in lab_results:
                    flag = result.get('flag', '')
                    flag_display = ""
                    if flag == 'H':
                        flag_display = "🔴 High"
                    elif flag == 'L':
                        flag_display = "🔵 Low"
                    elif flag == 'HH' or flag == 'CRITICAL HIGH':
                        flag_display = "🚨 Critical High"
                    elif flag == 'LL' or flag == 'CRITICAL LOW':
                        flag_display = "🚨 Critical Low"

                    table_data.append({
                        "Test": result.get('test', ''),
                        "Value": result.get('value', ''),
                        "Unit": result.get('unit', ''),
                        "Reference": result.get('reference', ''),
                        "Flag": flag_display
                    })

                st.dataframe(table_data, use_container_width=True, hide_index=True)
            else:
                st.info("No lab results extracted. Check the Raw JSON tab for available data.")

        elif doc_type == 'radiology':
            # Radiology findings
            if extracted.get('findings'):
                st.write("**Findings:**")
                findings = extracted['findings']
                if isinstance(findings, list):
                    for finding in findings:
                        st.write(f"- {finding}")
                else:
                    st.write(findings)

            if extracted.get('impression'):
                st.write("**Impression:**")
                st.write(extracted['impression'])

            if extracted.get('comparison'):
                st.write("**Comparison:**")
                st.write(extracted['comparison'])

            if extracted.get('recommendations'):
                st.write("**Recommendations:**")
                recs = extracted['recommendations']
                if isinstance(recs, list):
                    for rec in recs:
                        st.write(f"- {rec}")
                else:
                    st.write(recs)

            if not extracted.get('findings') and not extracted.get('impression'):
                st.info("No radiology findings extracted. Check the Raw JSON tab for available data.")

        elif doc_type == 'prescription':
            # Prescription medications
            if 'medications' in extracted:
                meds = extracted['medications']
                st.write(f"**{len(meds)} medication(s) found:**")

                for med in meds:
                    raw_name = med.get('medication_name') or med.get('name', 'Unknown')
                    rxnorm_name = med.get('rxnorm_name')
                    display_name = rxnorm_name or raw_name
                    was_corrected = (
                        rxnorm_name
                        and rxnorm_name.lower() != raw_name.lower()
                        and not raw_name.lower().startswith(rxnorm_name.lower())
                    )
                    strength = med.get('strength', '')
                    validation_status = med.get('validation_status', '')

                    # Status indicator
                    status_badge = {
                        'verified': '🟢 Verified',
                        'ocr_corrected': '🟡 OCR Corrected',
                        'medgemma_verified': '🔵 AI Verified',
                        'strength_mismatch': '🟠 Strength Mismatch',
                        'unverified': '🔴 Unverified',
                    }.get(validation_status, '')

                    st.write(f"### 💊 {display_name} {strength}")
                    if was_corrected:
                        st.caption(f"Originally read as: {raw_name}")
                    if status_badge:
                        st.caption(status_badge)

                    if validation_status == 'unverified':
                        st.warning("Not verified against any drug database. Manual review required.")
                    elif validation_status == 'strength_mismatch':
                        st.warning("Strength does not match known dosages for this drug. Possible misread — manual review required.")

                    if med.get('route'):
                        st.write(f"- **Route:** {med['route']}")
                    if med.get('frequency'):
                        st.write(f"- **Frequency:** {med['frequency']}")
                    if med.get('quantity'):
                        st.write(f"- **Quantity:** {med['quantity']}")
                    if med.get('refills'):
                        st.write(f"- **Refills:** {med['refills']}")
                    if med.get('directions'):
                        st.write(f"- **Directions:** {med['directions']}")

            if extracted.get('drug_interactions'):
                st.warning("**⚠️ Drug Interactions:**")
                for interaction in extracted['drug_interactions']:
                    st.write(f"- {interaction}")

            if extracted.get('contraindications'):
                st.error("**🚨 Contraindications:**")
                for contra in extracted['contraindications']:
                    st.write(f"- {contra}")

            if not extracted.get('medications'):
                st.info("No medications extracted. Check the Raw JSON tab for available data.")

        else:
            # Generic/Unknown document type
            if extracted:
                for key, value in extracted.items():
                    if key not in ['patient', 'prescriber']:
                        st.write(f"**{key.replace('_', ' ').title()}:**")
                        if isinstance(value, list):
                            for item in value:
                                st.write(f"- {item}")
                        elif isinstance(value, dict):
                            st.json(value)
                        else:
                            st.write(value)
            else:
                st.info("No structured data extracted. Check the Raw JSON tab for available data.")

    with tab2:
        # Patient and provider info (common to all types)
        patient = extracted.get('patient', {})
        prescriber = extracted.get('prescriber', {})

        if patient:
            st.write("**Patient Information:**")
            for key, value in patient.items():
                if value:
                    st.write(f"- **{key.replace('_', ' ').title()}:** {value}")

        if prescriber:
            st.write("**Provider Information:**")
            for key, value in prescriber.items():
                if value:
                    st.write(f"- **{key.replace('_', ' ').title()}:** {value}")

        # Clinical summary (if available)
        if selected_result.get('summary'):
            summary = selected_result['summary']
            if summary.get('clinical_summary'):
                st.write("**Clinical Summary:**")
                st.write(summary['clinical_summary'])

        if not patient and not prescriber:
            st.info("No patient/provider information extracted.")

    with tab3:
        # Full JSON view
        st.json(selected_result)

    with tab4:
        # Audit trail
        audit_trail = selected_result.get('audit_trail', [])

        if audit_trail:
            st.write("**Processing Steps:**")
            for i, step in enumerate(audit_trail):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"{i+1}. **{step.get('agent', 'Unknown')}**")
                with col2:
                    st.write(step.get('action', ''))
                with col3:
                    conf = step.get('confidence', 0)
                    if conf:
                        st.write(f"{conf * 100:.0f}%")
        else:
            st.info("No audit trail available.")

    st.divider()

    # Actions
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download JSON"):
            json_str = json.dumps(selected_result, indent=2, default=str)
            st.download_button(
                label="Download",
                data=json_str,
                file_name=f"{(selected_result.get('display_name') or selected_result.get('document_id', 'result')).replace(' ', '_')}_extracted.json",
                mime="application/json",
                key="download_json"
            )

    with col2:
        if st.button("📋 View FHIR Bundle"):
            st.session_state['view_fhir_doc'] = selected_idx
            st.switch_page("pages/5_fhir.py")

    with col3:
        if st.button("🗑️ Remove Result"):
            st.session_state.processing_results.pop(selected_idx)
            st.rerun()


if __name__ == "__main__":
    main()
else:
    main()
