# ui/pages/7_templates.py
"""
Template Management Page

Admin interface for:
- Viewing pending draft templates
- Reviewing auto-generated templates
- Approving/rejecting templates
- Managing active templates
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime


def render_templates_page():
    """Render the template management page."""
    st.title("Template Management")
    st.markdown("Review and manage document extraction templates")

    # Tabs for different template states
    tab1, tab2, tab3 = st.tabs([
        "Pending Review",
        "Active Templates",
        "Rejected"
    ])

    with tab1:
        render_pending_templates()

    with tab2:
        render_active_templates()

    with tab3:
        render_rejected_templates()


def render_pending_templates():
    """Render pending templates awaiting review."""
    st.subheader("Draft Templates Pending Review")

    try:
        from src.medical_ingestion.processors.template_generator import TemplateApprovalManager
        manager = TemplateApprovalManager()
        pending = manager.list_pending_templates()

        if not pending:
            st.info("No templates pending review")
            return

        st.write(f"**{len(pending)} templates awaiting review**")

        for template_info in pending:
            with st.expander(
                f"{template_info['vendor']} - {template_info['description']}",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Template ID:** `{template_info['id']}`")
                    st.markdown(f"**Type:** {template_info['test_type']}")
                    st.markdown(f"**Fields:** {template_info['field_count']}")
                    st.markdown(f"**Quality:** {template_info.get('quality', 'unknown')}")
                    st.markdown(f"**Source:** {template_info.get('source_document', 'N/A')}")
                    st.markdown(f"**Created:** {template_info.get('created_at', 'N/A')}")

                with col2:
                    # Action buttons
                    if st.button("Review", key=f"review_{template_info['id']}"):
                        st.session_state['review_template_id'] = template_info['id']
                        st.rerun()

        # Review modal
        if 'review_template_id' in st.session_state:
            render_template_review(st.session_state['review_template_id'])

    except Exception as e:
        st.error(f"Error loading pending templates: {e}")


def render_template_review(template_id: str):
    """Render detailed template review interface."""
    st.divider()
    st.subheader(f"Reviewing: {template_id}")

    try:
        from src.medical_ingestion.processors.template_generator import TemplateApprovalManager
        manager = TemplateApprovalManager()
        template = manager.get_template_for_review(template_id)

        if not template:
            st.error("Template not found")
            if st.button("Close"):
                del st.session_state['review_template_id']
                st.rerun()
            return

        # Display template details
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Template Info")
            st.json({
                "id": template.get('id'),
                "vendor": template.get('vendor'),
                "test_type": template.get('test_type'),
                "version": template.get('version'),
                "description": template.get('description')
            })

            st.markdown("### Matching Patterns")
            st.code(f"Header: {template.get('header_pattern', 'None')}")
            st.write("**Vendor Markers:**")
            for marker in template.get('vendor_markers', []):
                st.write(f"  - {marker}")

        with col2:
            st.markdown("### Field Mappings")
            field_mappings = template.get('field_mappings', {})

            if field_mappings:
                for field_name, config in list(field_mappings.items())[:10]:
                    st.markdown(f"**{field_name}**")
                    st.write(f"  PDF Name: {config.get('pdf_name')}")
                    st.write(f"  Unit: {config.get('unit', 'N/A')}")
                    st.write(f"  LOINC: {config.get('loinc_code') or 'Not set'}")

                if len(field_mappings) > 10:
                    st.info(f"... and {len(field_mappings) - 10} more fields")
            else:
                st.warning("No field mappings defined")

        # Validation notes
        if template.get('extraction_notes'):
            st.markdown("### Notes")
            for note in template['extraction_notes']:
                st.write(f"- {note}")

        # Editable fields for modification before approval
        st.divider()
        st.markdown("### Modifications (Optional)")

        with st.form(key=f"modify_{template_id}"):
            new_header = st.text_input(
                "Header Pattern (regex)",
                value=template.get('header_pattern', ''),
                help="Regex pattern to match document header"
            )

            new_vendor_markers = st.text_area(
                "Vendor Markers (one per line)",
                value='\n'.join(template.get('vendor_markers', [])),
                help="Unique text patterns to identify this vendor"
            )

            reviewer_name = st.text_input(
                "Reviewer Name",
                value="",
                help="Your name for audit trail"
            )

            col_approve, col_reject, col_cancel = st.columns(3)

            with col_approve:
                approve_btn = st.form_submit_button(
                    "Approve",
                    type="primary"
                )

            with col_reject:
                reject_btn = st.form_submit_button("Reject")

            with col_cancel:
                cancel_btn = st.form_submit_button("Cancel")

            if approve_btn:
                if not reviewer_name:
                    st.error("Please enter reviewer name")
                else:
                    modifications = {}
                    if new_header != template.get('header_pattern', ''):
                        modifications['header_pattern'] = new_header
                    if new_vendor_markers.strip():
                        modifications['vendor_markers'] = [
                            m.strip() for m in new_vendor_markers.split('\n') if m.strip()
                        ]

                    success, msg = manager.approve_template(
                        template_id,
                        reviewer_name,
                        modifications if modifications else None
                    )

                    if success:
                        st.success(f"Template approved! {msg}")
                        del st.session_state['review_template_id']
                        st.rerun()
                    else:
                        st.error(f"Approval failed: {msg}")

            if reject_btn:
                if not reviewer_name:
                    st.error("Please enter reviewer name")
                else:
                    rejection_reason = st.session_state.get(
                        f'rejection_reason_{template_id}',
                        'Template quality insufficient'
                    )
                    success, msg = manager.reject_template(
                        template_id,
                        reviewer_name,
                        rejection_reason
                    )

                    if success:
                        st.success("Template rejected and archived")
                        del st.session_state['review_template_id']
                        st.rerun()
                    else:
                        st.error(f"Rejection failed: {msg}")

            if cancel_btn:
                del st.session_state['review_template_id']
                st.rerun()

    except Exception as e:
        st.error(f"Error loading template: {e}")


def render_active_templates():
    """Render list of active templates."""
    st.subheader("Active Templates")

    try:
        from src.medical_ingestion.config import base_settings

        template_dirs = ['lab', 'radiology', 'prescription']
        total_templates = 0

        for doc_type in template_dirs:
            templates_dir = base_settings.get_processor_template_dir(doc_type)

            if not templates_dir.exists():
                continue

            templates = list(templates_dir.glob("*.json"))
            if not templates:
                continue

            with st.expander(f"{doc_type.title()} Templates ({len(templates)})", expanded=True):
                for template_file in sorted(templates):
                    try:
                        with open(template_file, 'r') as f:
                            template = json.load(f)

                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.markdown(f"**{template.get('id', template_file.stem)}**")
                            st.write(f"{template.get('description', 'No description')}")

                        with col2:
                            st.write(f"Vendor: {template.get('vendor', 'Unknown')}")
                            st.write(f"Fields: {len(template.get('field_mappings', {}))}")

                        with col3:
                            auto_generated = template.get('auto_generated', False)
                            if auto_generated:
                                st.caption("Auto-generated")
                            st.write(f"v{template.get('version', 1)}")

                        total_templates += 1

                    except Exception as e:
                        st.warning(f"Error reading {template_file.name}: {e}")

        if total_templates == 0:
            st.info("No active templates found")
        else:
            st.success(f"Total: {total_templates} active templates")

    except Exception as e:
        st.error(f"Error loading templates: {e}")


def render_rejected_templates():
    """Render rejected templates archive."""
    st.subheader("Rejected Templates")

    try:
        from src.medical_ingestion.config import base_settings

        rejected_dir = base_settings.get_drafts_dir() / "rejected"

        if not rejected_dir.exists():
            st.info("No rejected templates")
            return

        templates = list(rejected_dir.glob("*.json"))

        if not templates:
            st.info("No rejected templates")
            return

        st.write(f"**{len(templates)} rejected templates**")

        for template_file in sorted(templates, reverse=True):
            try:
                with open(template_file, 'r') as f:
                    template = json.load(f)

                metadata = template.get('_draft_metadata', {})

                with st.expander(f"{template.get('id', template_file.stem)}"):
                    st.markdown(f"**Vendor:** {template.get('vendor')}")
                    st.markdown(f"**Type:** {template.get('test_type')}")
                    st.markdown(f"**Rejected by:** {metadata.get('reviewed_by', 'Unknown')}")
                    st.markdown(f"**Rejected at:** {metadata.get('rejected_at', 'Unknown')}")
                    st.markdown(f"**Reason:** {metadata.get('rejection_reason', 'No reason provided')}")

                    if st.button("Delete permanently", key=f"delete_{template_file.stem}"):
                        template_file.unlink()
                        st.success("Deleted")
                        st.rerun()

            except Exception as e:
                st.warning(f"Error reading {template_file.name}: {e}")

    except Exception as e:
        st.error(f"Error loading rejected templates: {e}")


# Page entry point
if __name__ == "__main__":
    render_templates_page()
else:
    render_templates_page()
