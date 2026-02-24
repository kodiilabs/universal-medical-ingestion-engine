# ============================================================================
# ui/pages/6_audit.py
# ============================================================================
"""
Audit Log Page

Displays system audit trail from the actual audit database.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.services.audit_service import AuditService

# Initialize audit service
audit_service = AuditService()


def main():
    st.title("📋 Audit Log")

    st.markdown("""
    View system activity, document processing history, and audit trail.
    """)

    st.divider()

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        date_range = st.selectbox(
            "Time Range",
            ["Last hour", "Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
        )

    with col2:
        action_filter = st.multiselect(
            "Action Type",
            ["document_processed", "document_classified", "data_extracted",
             "validation_completed", "validation_warning", "fhir_bundle_generated",
             "critical_value_detected", "processing_complete", "processing_error"],
            default=[]
        )

    with col3:
        status_filter = st.multiselect(
            "Status",
            ["success", "warning", "critical", "info", "error"],
            default=[]
        )

    st.divider()

    # Get audit log from service
    db_events = audit_service.get_recent_events(limit=100)

    # Combine with session state audit log
    session_log = st.session_state.get('audit_log', [])

    # Merge events
    all_events = db_events + session_log

    # Apply filters
    filtered_log = all_events

    if action_filter:
        filtered_log = [e for e in filtered_log if e.get('action') in action_filter or e.get('event_type') in action_filter]

    if status_filter:
        filtered_log = [e for e in filtered_log if e.get('status') in status_filter or (e.get('success') == True and 'success' in status_filter)]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    # Get stats
    stats = audit_service.get_processing_stats()

    with col1:
        st.metric("Total Events", len(filtered_log))

    with col2:
        success_count = len([e for e in filtered_log if e.get('status') == 'success' or e.get('success') == True])
        st.metric("Successful", success_count)

    with col3:
        warning_count = len([e for e in filtered_log if e.get('status') == 'warning'])
        st.metric("Warnings", warning_count)

    with col4:
        st.metric("Docs Processed", stats.get('total_processed', 0))

    st.divider()

    # Audit log table
    st.subheader("Activity Log")

    if not filtered_log:
        st.info("No audit events found. Process documents to generate audit trail.")
    else:
        for entry in filtered_log:
            timestamp = entry.get('timestamp', '')
            action = entry.get('action') or entry.get('event_type', '')
            details = entry.get('details', '')
            status = entry.get('status', 'info')
            doc_id = entry.get('document_id', '')

            # Determine status from success field if not explicitly set
            if 'success' in entry and status == 'info':
                status = 'success' if entry['success'] else 'error'

            # Status indicator
            if status == 'success' or entry.get('success') == True:
                icon = '✅'
            elif status == 'warning':
                icon = '⚠️'
            elif status == 'critical':
                icon = '🚨'
            elif status == 'error' or entry.get('success') == False:
                icon = '❌'
            else:
                icon = 'ℹ️'

            # Format timestamp
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = str(timestamp)

            with st.container():
                col1, col2, col3 = st.columns([1, 2, 3])

                with col1:
                    st.write(f"**{formatted_time}**")

                with col2:
                    st.markdown(f"{icon} `{action}`")

                with col3:
                    if details:
                        st.write(details)
                    elif doc_id:
                        st.write(f"Document: {doc_id}")

                st.divider()

    st.divider()

    # Processing Statistics
    if stats.get('type_breakdown'):
        st.subheader("Processing Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Documents by Type:**")
            for doc_type, count in stats['type_breakdown'].items():
                st.write(f"- {doc_type}: {count}")

        with col2:
            st.write(f"**Success Rate:** {stats.get('success_rate', 0) * 100:.1f}%")
            st.write(f"**Avg Confidence:** {stats.get('avg_confidence', 0) * 100:.1f}%")
            st.write(f"**Needs Review:** {stats.get('needs_review', 0)}")

    st.divider()

    # Export options
    st.subheader("Export Audit Log")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = audit_service.export_audit_log(format='csv')
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name="audit_log.csv",
            mime="text/csv"
        )

    with col2:
        json_data = audit_service.export_audit_log(format='json')
        st.download_button(
            label="📥 Export JSON",
            data=json_data,
            file_name="audit_log.json",
            mime="application/json"
        )

    with col3:
        if st.button("🗑️ Clear Log"):
            audit_service.clear_audit_log()
            st.session_state.audit_log = []
            st.success("Audit log cleared")
            st.rerun()

    st.divider()

    # System info
    with st.expander("System Information"):
        st.write("**Application:** Medical Document Ingestion Engine")
        st.write("**Version:** 1.0.0")
        st.write(f"**Server Time:** {datetime.now().isoformat()}")
        st.write("**FHIR Version:** R4")
        st.write(f"**Audit DB Path:** {audit_service._audit_logger.db_path}")


if __name__ == "__main__":
    main()
else:
    main()
