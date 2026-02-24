# ============================================================================
# ui/components/fhir_preview.py
# ============================================================================
"""
FHIR Preview Component

Displays FHIR resources in a readable format.
"""

import streamlit as st
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, date


def _json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def render_fhir_resource(
    resource: Dict[str, Any],
    expanded: bool = False
) -> None:
    """
    Render a single FHIR resource.

    Args:
        resource: FHIR resource dict
        expanded: Whether to expand by default
    """
    resource_type = resource.get('resourceType', 'Unknown')
    resource_id = resource.get('id', 'N/A')

    with st.expander(f"{resource_type} ({resource_id})", expanded=expanded):
        # Resource summary
        _render_resource_summary(resource)

        st.divider()

        # Raw JSON
        st.write("**Raw JSON:**")
        st.json(resource)


def _render_resource_summary(resource: Dict[str, Any]) -> None:
    """Render a summary of the resource based on type."""
    resource_type = resource.get('resourceType', '')

    if resource_type == 'Observation':
        _render_observation_summary(resource)
    elif resource_type == 'DiagnosticReport':
        _render_diagnostic_report_summary(resource)
    elif resource_type == 'MedicationRequest':
        _render_medication_request_summary(resource)
    elif resource_type == 'Patient':
        _render_patient_summary(resource)
    else:
        st.write(f"Resource Type: **{resource_type}**")


def _render_observation_summary(obs: Dict[str, Any]) -> None:
    """Render Observation summary."""
    # Code
    code = obs.get('code', {})
    coding = code.get('coding', [{}])[0] if code.get('coding') else {}
    display = coding.get('display', code.get('text', 'Unknown'))

    st.write(f"**Test:** {display}")

    # Value
    if 'valueQuantity' in obs:
        vq = obs['valueQuantity']
        value = vq.get('value', 'N/A')
        unit = vq.get('unit', '')
        st.write(f"**Value:** {value} {unit}")
    elif 'valueString' in obs:
        st.write(f"**Value:** {obs['valueString']}")
    elif 'valueCodeableConcept' in obs:
        cc = obs['valueCodeableConcept']
        text = cc.get('text', cc.get('coding', [{}])[0].get('display', 'N/A'))
        st.write(f"**Value:** {text}")

    # Reference range
    ref_ranges = obs.get('referenceRange', [])
    if ref_ranges:
        rr = ref_ranges[0]
        low = rr.get('low', {}).get('value', '')
        high = rr.get('high', {}).get('value', '')
        if low or high:
            st.write(f"**Reference Range:** {low} - {high}")

    # Status
    status = obs.get('status', 'unknown')
    st.write(f"**Status:** {status}")

    # Interpretation
    interpretation = obs.get('interpretation', [])
    if interpretation:
        interp = interpretation[0]
        interp_text = interp.get('text', interp.get('coding', [{}])[0].get('display', ''))
        if interp_text:
            st.write(f"**Interpretation:** {interp_text}")


def _render_diagnostic_report_summary(report: Dict[str, Any]) -> None:
    """Render DiagnosticReport summary."""
    # Code
    code = report.get('code', {})
    display = code.get('text', code.get('coding', [{}])[0].get('display', 'Unknown'))
    st.write(f"**Report Type:** {display}")

    # Status
    status = report.get('status', 'unknown')
    st.write(f"**Status:** {status}")

    # Conclusion
    conclusion = report.get('conclusion', '')
    if conclusion:
        st.write(f"**Conclusion:** {conclusion}")

    # Results count
    results = report.get('result', [])
    st.write(f"**Results:** {len(results)} observations")


def _render_medication_request_summary(med: Dict[str, Any]) -> None:
    """Render MedicationRequest summary."""
    # Medication
    med_cc = med.get('medicationCodeableConcept', {})
    med_name = med_cc.get('text', med_cc.get('coding', [{}])[0].get('display', 'Unknown'))
    st.write(f"**Medication:** {med_name}")

    # Status
    status = med.get('status', 'unknown')
    st.write(f"**Status:** {status}")

    # Dosage
    dosage = med.get('dosageInstruction', [])
    if dosage:
        dose = dosage[0]
        text = dose.get('text', '')
        if text:
            st.write(f"**Dosage:** {text}")

    # Dispense
    dispense = med.get('dispenseRequest', {})
    quantity = dispense.get('quantity', {})
    if quantity:
        qty_value = quantity.get('value', '')
        qty_unit = quantity.get('unit', '')
        st.write(f"**Quantity:** {qty_value} {qty_unit}")


def _render_patient_summary(patient: Dict[str, Any]) -> None:
    """Render Patient summary."""
    # Name
    names = patient.get('name', [])
    if names:
        name = names[0]
        given = ' '.join(name.get('given', []))
        family = name.get('family', '')
        st.write(f"**Name:** {given} {family}")

    # Gender
    gender = patient.get('gender', 'unknown')
    st.write(f"**Gender:** {gender}")

    # Birth date
    dob = patient.get('birthDate', '')
    if dob:
        st.write(f"**Birth Date:** {dob}")

    # Identifiers
    identifiers = patient.get('identifier', [])
    if identifiers:
        for ident in identifiers:
            system = ident.get('system', 'Unknown')
            value = ident.get('value', '')
            st.write(f"**{system}:** {value}")


def render_fhir_bundle(
    bundle: Dict[str, Any],
    title: str = "FHIR Bundle"
) -> None:
    """
    Render a FHIR Bundle with all entries.

    Args:
        bundle: FHIR Bundle dict
        title: Section title
    """
    st.subheader(title)

    # Bundle metadata
    bundle_type = bundle.get('type', 'unknown')
    entries = bundle.get('entry', [])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Type", bundle_type)
    with col2:
        st.metric("Entries", len(entries))
    with col3:
        st.metric("ID", bundle.get('id', 'N/A')[:8] + "...")

    st.divider()

    # Resource type breakdown
    resource_types = {}
    for entry in entries:
        resource = entry.get('resource', {})
        rt = resource.get('resourceType', 'Unknown')
        resource_types[rt] = resource_types.get(rt, 0) + 1

    if resource_types:
        st.write("**Resource Types:**")
        for rt, count in resource_types.items():
            st.write(f"- {rt}: {count}")

    st.divider()

    # Render each entry
    st.write("**Entries:**")
    for i, entry in enumerate(entries):
        resource = entry.get('resource', {})
        render_fhir_resource(resource, expanded=(i == 0))


def render_fhir_json(
    data: Dict[str, Any],
    title: str = "FHIR JSON"
) -> None:
    """
    Render FHIR JSON with download option.

    Args:
        data: FHIR data dict
        title: Section title
    """
    st.subheader(title)

    # JSON display - use custom serializer for datetime objects
    json_str = json.dumps(data, indent=2, default=_json_serializer)

    st.code(json_str, language="json")

    # Download button
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name="fhir_bundle.json",
        mime="application/json"
    )


def render_resource_list(
    resources: List[Dict[str, Any]],
    title: str = "Resources"
) -> None:
    """
    Render a list of FHIR resources.

    Args:
        resources: List of FHIR resource dicts
        title: Section title
    """
    st.subheader(title)

    if not resources:
        st.info("No resources to display.")
        return

    for resource in resources:
        render_fhir_resource(resource)
