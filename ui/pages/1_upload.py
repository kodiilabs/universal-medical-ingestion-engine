# ============================================================================
# ui/pages/1_upload.py
# ============================================================================
"""
Upload Page

Handles document upload and processing via the core ingestion engine.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.services.processing_service import ProcessingService
from ui.services.audit_service import AuditService


# Initialize services
processing_service = ProcessingService()
audit_service = AuditService()


def main():
    st.title("📤 Upload Documents")

    st.markdown("""
    Upload medical documents for processing. Supported formats:
    - **PDF** - Lab reports, radiology reports, prescriptions
    - **Images** (PNG, JPG) - Scanned documents
    - **Text** (TXT) - Plain text medical documents
    """)

    st.divider()

    # Patient context (optional)
    with st.expander("Patient Context (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID / MRN", key="patient_id")
            patient_name = st.text_input("Patient Name", key="patient_name")
        with col2:
            patient_dob = st.date_input("Date of Birth", value=None, key="patient_dob")
            patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="patient_gender")

    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader(f"Uploaded {len(uploaded_files)} file(s)")

        # Preview and classification
        for idx, uploaded_file in enumerate(uploaded_files):
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer

            # Use index + filename for unique keys (handles duplicate filenames)
            file_key = f"{idx}_{uploaded_file.name}"

            with st.expander(f"📄 {uploaded_file.name}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**File name:** {uploaded_file.name}")
                    st.write(f"**Type:** {uploaded_file.type}")
                    st.write(f"**Size:** {uploaded_file.size:,} bytes")

                with col2:
                    # Auto-classify document
                    if st.button(f"🔍 Classify", key=f"classify_{file_key}"):
                        with st.spinner("Classifying..."):
                            classification = processing_service.classify_document(
                                file_content, uploaded_file.name
                            )
                            st.session_state[f"classification_{file_key}"] = classification

                    # Show classification result
                    if f"classification_{file_key}" in st.session_state:
                        cls = st.session_state[f"classification_{file_key}"]
                        st.write(f"**Detected Type:** {cls['document_type']}")
                        st.write(f"**Confidence:** {cls['confidence'] * 100:.1f}%")

                    # Manual override
                    doc_type = st.selectbox(
                        "Document Type",
                        options=[
                            "Auto-detect",
                            "Lab Report",
                            "Prescription",
                            "Radiology Report",
                            "Pathology Report",
                            "Discharge Summary",
                            "Clinical Note"
                        ],
                        key=f"doc_type_{file_key}"
                    )

        st.divider()

        # Process button
        col1, col2 = st.columns([1, 3])

        with col1:
            process_btn = st.button("🚀 Process Documents", type="primary")

        with col2:
            clear_btn = st.button("🗑️ Clear All")

        if process_btn:
            # Build patient context
            patient_context = None
            if patient_id or patient_name:
                patient_context = {
                    'patient_id': patient_id or None,
                    'name': patient_name or None,
                    'dob': patient_dob.isoformat() if patient_dob else None,
                    'gender': patient_gender or None
                }

            with st.spinner("Processing documents with medical ingestion engine..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")

                    # Read file content
                    file_content = uploaded_file.read()
                    uploaded_file.seek(0)

                    # Store file info
                    file_info = {
                        'name': uploaded_file.name,
                        'type': uploaded_file.type,
                        'size': len(file_content),
                        'content': file_content,
                        'upload_time': datetime.now().isoformat(),
                        'status': 'processing'
                    }

                    if 'uploaded_files' not in st.session_state:
                        st.session_state.uploaded_files = []
                    st.session_state.uploaded_files.append(file_info)

                    # Process with actual engine
                    result = processing_service.process_uploaded_file(
                        file_content=file_content,
                        file_name=uploaded_file.name,
                        patient_context=patient_context
                    )

                    # Add file_name to result for reference
                    result['file_name'] = uploaded_file.name

                    if 'processing_results' not in st.session_state:
                        st.session_state.processing_results = []
                    st.session_state.processing_results.append(result)

                    # Store FHIR bundle
                    if result.get('fhir_bundle'):
                        if 'fhir_bundles' not in st.session_state:
                            st.session_state.fhir_bundles = []
                        st.session_state.fhir_bundles.append(result['fhir_bundle'])

                    # Log to audit
                    audit_service.log_event(
                        action="document_processed",
                        details=f"File: {uploaded_file.name}, Type: {result.get('document_type')}, Confidence: {result.get('confidence', 0):.2f}",
                        document_id=result.get('document_id'),
                        status="success" if result.get('success') else "error"
                    )

                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.empty()

                # Show summary
                successful = sum(1 for r in st.session_state.processing_results[-len(uploaded_files):] if r.get('success'))
                failed = len(uploaded_files) - successful

                if failed == 0:
                    st.success(f"Successfully processed {successful} document(s)!")
                else:
                    st.warning(f"Processed {successful} document(s), {failed} failed.")

                # Show critical findings if any
                for result in st.session_state.processing_results[-len(uploaded_files):]:
                    if result.get('critical_findings'):
                        display = result.get('display_name') or result.get('file_name', 'Unknown')
                        st.error(f"🚨 Critical findings in {display}: {result['critical_findings']}")

                    if result.get('requires_review'):
                        display = result.get('display_name') or result.get('file_name', 'Unknown')
                        st.warning(f"⚠️ {display} requires manual review (confidence: {result.get('confidence', 0) * 100:.0f}%)")

                st.info("Navigate to the **Results** page to view extracted data.")

        if clear_btn:
            st.session_state.uploaded_files = []
            st.session_state.processing_results = []
            st.session_state.fhir_bundles = []
            st.rerun()

    # Processing history
    st.divider()
    st.subheader("Processing History")

    if st.session_state.get('processing_results'):
        for result in reversed(st.session_state.processing_results[-10:]):
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

            with col1:
                status_icon = "✅" if result.get('success') else "❌"
                st.write(f"{status_icon} {result.get('display_name') or result.get('file_name', 'Unknown')}")

            with col2:
                doc_type = result.get('document_type', 'unknown')
                st.write(doc_type.replace('_', ' ').title())

            with col3:
                confidence = result.get('confidence', 0)
                st.write(f"{confidence * 100:.0f}%")

            with col4:
                if result.get('requires_review'):
                    st.write("⚠️ Review")
                else:
                    st.write("✓")
    else:
        st.info("No documents processed yet. Upload files above to get started.")


if __name__ == "__main__":
    main()
else:
    main()
