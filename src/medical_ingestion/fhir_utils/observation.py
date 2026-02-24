# ============================================================================
# src/fhir_utils/observation.py
# ============================================================================
"""
FHIR Observation resource builder for lab results.
"""

from typing import Optional, List
from datetime import datetime
from uuid import uuid4

from fhir.resources.observation import Observation, ObservationReferenceRange
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.quantity import Quantity
from fhir.resources.reference import Reference
from fhir.resources.extension import Extension

from medical_ingestion.constants.loinc import LOINC_CODES
from medical_ingestion.core.context.extracted_value import ExtractedValue
from medical_ingestion.core.context.processing_context import ProcessingContext
from medical_ingestion.config.fhir_config import fhir_settings


def create_lab_observation(
    extracted: ExtractedValue,
    context: ProcessingContext,
    patient_id: Optional[str] = None
) -> Observation:
    """
    Create FHIR Observation resource for a lab test result.

    Args:
        extracted: Extracted lab value with metadata
        context: Processing context
        patient_id: Optional patient identifier

    Returns:
        FHIR Observation resource
    """
    # Get LOINC code
    loinc_code = LOINC_CODES.get(extracted.field_name)

    # Build code (what was measured)
    code = CodeableConcept(
        coding=[
            Coding(
                system="http://loinc.org",
                code=loinc_code,
                display=extracted.field_name.replace('_', ' ').title()
            )
        ],
        text=extracted.field_name.replace('_', ' ').title()
    )

    # Build value (the result)
    value_quantity = Quantity(
        value=float(extracted.value) if extracted.value else None,
        unit=extracted.unit,
        system="http://unitsofmeasure.org",
        code=extracted.unit
    )

    # Build reference range
    reference_range = build_reference_range(
        extracted.reference_min,
        extracted.reference_max,
        extracted.unit
    )

    # Build interpretation (H/L/N flags)
    interpretation = build_interpretation(extracted.abnormal_flag)

    # Build extensions
    extensions = build_extensions(extracted)

    # Create Observation
    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=code,
        valueQuantity=value_quantity,
        referenceRange=reference_range,
        interpretation=interpretation,
        issued=datetime.now().isoformat(),
        extension=extensions if extensions else None
    )

    # Add patient reference
    if patient_id or context.patient_id:
        observation.subject = Reference(
            reference=f"Patient/{patient_id or context.patient_id}"
        )

    # Add warnings as notes
    if extracted.warnings:
        observation.note = [
            {"text": warning} for warning in extracted.warnings
        ]

    return observation


def build_reference_range(
    min_value: Optional[float],
    max_value: Optional[float],
    unit: Optional[str]
) -> Optional[List[ObservationReferenceRange]]:
    """
    Build FHIR reference range for lab test.

    Args:
        min_value: Lower bound of normal range
        max_value: Upper bound of normal range
        unit: Unit of measurement

    Returns:
        List with single ObservationReferenceRange or None
    """
    if min_value is None or max_value is None:
        return None

    return [
        ObservationReferenceRange(
            low=Quantity(
                value=min_value,
                unit=unit,
                system="http://unitsofmeasure.org"
            ),
            high=Quantity(
                value=max_value,
                unit=unit,
                system="http://unitsofmeasure.org"
            ),
            text=f"{min_value}-{max_value} {unit}"
        )
    ]


def build_interpretation(abnormal_flag: Optional[str]) -> Optional[List[CodeableConcept]]:
    """
    Build FHIR interpretation from abnormal flag.

    Args:
        abnormal_flag: H/L/N/CRITICAL flag from lab report

    Returns:
        List with CodeableConcept or None
    """
    if not abnormal_flag:
        return None

    interp_code = map_abnormal_flag(abnormal_flag)

    return [
        CodeableConcept(
            coding=[
                Coding(
                    system="http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    code=interp_code,
                    display=abnormal_flag
                )
            ]
        )
    ]


def map_abnormal_flag(flag: str) -> str:
    """
    Map custom abnormal flags to FHIR interpretation codes.

    Args:
        flag: Custom flag from lab report

    Returns:
        FHIR standard code
    """
    mapping = {
        "H": "H",
        "HIGH": "H",
        "L": "L",
        "LOW": "L",
        "N": "N",
        "NORMAL": "N",
        "CRITICAL": "HH",
        "CRITICAL HIGH": "HH",
        "CRITICAL LOW": "LL",
        "HH": "HH",
        "LL": "LL"
    }
    return mapping.get(flag.upper(), "N")


def build_extensions(extracted: ExtractedValue) -> List[Extension]:
    """
    Build FHIR extensions for custom metadata.

    Args:
        extracted: Extracted value with metadata

    Returns:
        List of Extension resources
    """
    extensions = []

    # Confidence score
    if fhir_settings.FHIR_INCLUDE_CONFIDENCE:
        extensions.append(Extension(
            url="http://medical-ingestion-engine.local/confidence",
            valueDecimal=extracted.confidence
        ))

    # Extraction method
    extensions.append(Extension(
        url="http://medical-ingestion-engine.local/extraction-method",
        valueString=extracted.extraction_method
    ))

    # Source location
    if fhir_settings.FHIR_INCLUDE_PROVENANCE and extracted.source_location:
        extensions.append(Extension(
            url="http://medical-ingestion-engine.local/source-location",
            valueString=extracted.source_location
        ))

    # Validation conflict flag
    if extracted.validation_conflict:
        extensions.append(Extension(
            url="http://medical-ingestion-engine.local/validation-conflict",
            valueBoolean=True
        ))

    return extensions
