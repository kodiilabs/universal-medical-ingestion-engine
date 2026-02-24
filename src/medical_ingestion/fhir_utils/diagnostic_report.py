# ============================================================================
# src/fhir_utils/diagnostic_report.py
# ============================================================================
"""
FHIR DiagnosticReport resource builder for radiology and pathology reports.
"""

from typing import Optional, List
from datetime import datetime
from uuid import uuid4

from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference
from fhir.resources.extension import Extension

from medical_ingestion.core.context.processing_context import ProcessingContext


def create_radiology_report(
    context: ProcessingContext,
    patient_id: Optional[str] = None,
    conclusion: Optional[str] = None
) -> DiagnosticReport:
    """
    Create FHIR DiagnosticReport for radiology imaging study.

    Args:
        context: Processing context
        patient_id: Optional patient identifier
        conclusion: Optional impression/conclusion text

    Returns:
        FHIR DiagnosticReport resource
    """
    report = DiagnosticReport(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[
                Coding(
                    system="http://loinc.org",
                    code="18748-4",
                    display="Diagnostic Imaging Report"
                )
            ],
            text="Radiology Report"
        ),
        issued=datetime.now().isoformat()
    )

    # Add patient reference
    if patient_id or context.patient_id:
        report.subject = Reference(
            reference=f"Patient/{patient_id or context.patient_id}"
        )

    # Add conclusion
    if conclusion or context.clinical_summary:
        report.conclusion = conclusion or context.clinical_summary

    return report


def create_pathology_report(
    context: ProcessingContext,
    patient_id: Optional[str] = None,
    conclusion: Optional[str] = None,
    coded_diagnosis: Optional[List[CodeableConcept]] = None
) -> DiagnosticReport:
    """
    Create FHIR DiagnosticReport for pathology study.

    Args:
        context: Processing context
        patient_id: Optional patient identifier
        conclusion: Optional diagnosis text
        coded_diagnosis: Optional SNOMED-coded diagnoses

    Returns:
        FHIR DiagnosticReport resource
    """
    report = DiagnosticReport(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[
                Coding(
                    system="http://loinc.org",
                    code="11526-1",
                    display="Pathology Report"
                )
            ],
            text="Pathology Report"
        ),
        issued=datetime.now().isoformat()
    )

    # Add patient reference
    if patient_id or context.patient_id:
        report.subject = Reference(
            reference=f"Patient/{patient_id or context.patient_id}"
        )

    # Add conclusion
    if conclusion or context.clinical_summary:
        report.conclusion = conclusion or context.clinical_summary

    # Add coded diagnosis
    if coded_diagnosis:
        report.conclusionCode = coded_diagnosis

    return report


def create_generic_diagnostic_report(
    context: ProcessingContext,
    loinc_code: str,
    display_name: str,
    patient_id: Optional[str] = None,
    conclusion: Optional[str] = None
) -> DiagnosticReport:
    """
    Create FHIR DiagnosticReport with custom LOINC code.

    Args:
        context: Processing context
        loinc_code: LOINC code for report type
        display_name: Display name for report
        patient_id: Optional patient identifier
        conclusion: Optional conclusion text

    Returns:
        FHIR DiagnosticReport resource
    """
    report = DiagnosticReport(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[
                Coding(
                    system="http://loinc.org",
                    code=loinc_code,
                    display=display_name
                )
            ],
            text=display_name
        ),
        issued=datetime.now().isoformat()
    )

    # Add patient reference
    if patient_id or context.patient_id:
        report.subject = Reference(
            reference=f"Patient/{patient_id or context.patient_id}"
        )

    # Add conclusion
    if conclusion:
        report.conclusion = conclusion

    return report


def add_result_observation(
    report: DiagnosticReport,
    observation_id: str
) -> DiagnosticReport:
    """
    Add reference to an Observation to the DiagnosticReport.

    Args:
        report: DiagnosticReport to update
        observation_id: ID of Observation resource

    Returns:
        Updated DiagnosticReport
    """
    if report.result is None:
        report.result = []

    report.result.append(
        Reference(reference=f"Observation/{observation_id}")
    )

    return report


def add_media_attachment(
    report: DiagnosticReport,
    media_id: str
) -> DiagnosticReport:
    """
    Add reference to media (images) to the DiagnosticReport.

    Args:
        report: DiagnosticReport to update
        media_id: ID of Media resource

    Returns:
        Updated DiagnosticReport
    """
    if report.media is None:
        report.media = []

    report.media.append(
        Reference(reference=f"Media/{media_id}")
    )

    return report
