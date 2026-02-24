# ============================================================================
# src/fhir_utils/medication_request.py
# ============================================================================
"""
FHIR MedicationRequest resource builder for prescriptions.

Compatible with fhir.resources pydantic v2 which changed how choice types work.
- medication[x] now uses 'medication' field with CodeableConcept or Reference
- subject is required
"""

from typing import Optional, List, Union
from datetime import datetime, timezone
from uuid import uuid4
import logging

from fhir.resources.medicationrequest import MedicationRequest, MedicationRequestDispenseRequest
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference
from fhir.resources.dosage import Dosage
from fhir.resources.quantity import Quantity

from medical_ingestion.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


def create_medication_request(
    medication_name: str,
    context: ProcessingContext,
    rxnorm_code: Optional[str] = None,
    patient_id: Optional[str] = None,
    dosage_text: Optional[str] = None,
    quantity: Optional[int] = None,
    refills: Optional[int] = None
) -> MedicationRequest:
    """
    Create FHIR MedicationRequest for a prescription.

    Args:
        medication_name: Name of medication
        context: Processing context
        rxnorm_code: Optional RxNorm code
        patient_id: Optional patient identifier
        dosage_text: Optional dosage instructions
        quantity: Optional quantity to dispense
        refills: Optional number of refills

    Returns:
        FHIR MedicationRequest resource
    """
    # Build medication coding
    medication_concept = build_medication_concept(medication_name, rxnorm_code)

    # Build subject reference (required field)
    # Use provided patient_id, context patient_id, or anonymous placeholder
    subject_id = patient_id or getattr(context, 'patient_id', None) or "anonymous"
    subject_ref = Reference(reference=f"Patient/{subject_id}")

    # Build the request data dict first to handle choice type properly
    # Use timezone-aware datetime for FHIR compliance
    request_data = {
        "id": str(uuid4()),
        "status": "active",
        "intent": "order",
        "subject": subject_ref,
        "authoredOn": datetime.now(timezone.utc).isoformat(),
    }

    # Handle medication choice type for FHIR R5 / fhir.resources pydantic v2
    # In R5, medication[x] is replaced by 'medication' which accepts CodeableReference
    # We need to wrap the CodeableConcept in a CodeableReference structure
    try:
        from fhir.resources.codeablereference import CodeableReference
        # FHIR R5 style - use CodeableReference
        request_data["medication"] = CodeableReference(concept=medication_concept)
        request = MedicationRequest(**request_data)
    except (ImportError, Exception) as e:
        # Fallback for older FHIR versions or different library versions
        logger.debug(f"CodeableReference not available or failed: {e}")
        try:
            # Try direct CodeableConcept assignment
            request_data["medication"] = medication_concept
            request = MedicationRequest(**request_data)
        except Exception as e2:
            logger.debug(f"Direct medication assignment failed: {e2}")
            # Last resort: try medicationCodeableConcept for R4 compatibility
            if "medication" in request_data:
                del request_data["medication"]
            request_data["medicationCodeableConcept"] = medication_concept
            request = MedicationRequest(**request_data)

    # Add dosage instructions
    if dosage_text:
        request.dosageInstruction = [
            Dosage(text=dosage_text)
        ]

    # Add dispense request (quantity and refills)
    if quantity or refills:
        request.dispenseRequest = build_dispense_request(quantity, refills)

    return request


def build_medication_concept(
    medication_name: str,
    rxnorm_code: Optional[str] = None
) -> CodeableConcept:
    """
    Build CodeableConcept for medication.

    Args:
        medication_name: Name of medication
        rxnorm_code: Optional RxNorm code

    Returns:
        CodeableConcept with medication coding
    """
    codings = []

    # Add RxNorm coding if available
    if rxnorm_code:
        codings.append(
            Coding(
                system="http://www.nlm.nih.gov/research/umls/rxnorm",
                code=rxnorm_code,
                display=medication_name
            )
        )

    return CodeableConcept(
        coding=codings if codings else None,
        text=medication_name
    )


def build_dispense_request(
    quantity: Optional[int] = None,
    refills: Optional[int] = None
) -> MedicationRequestDispenseRequest:
    """
    Build dispense request with quantity and refills.

    Args:
        quantity: Quantity to dispense
        refills: Number of refills

    Returns:
        MedicationRequestDispenseRequest
    """
    dispense_request = MedicationRequestDispenseRequest()

    if quantity:
        dispense_request.quantity = Quantity(
            value=quantity
        )

    if refills is not None:
        dispense_request.numberOfRepeatsAllowed = refills

    return dispense_request


def add_dosage_instruction(
    request: MedicationRequest,
    instruction_text: str,
    dose_value: Optional[float] = None,
    dose_unit: Optional[str] = None,
    frequency: Optional[str] = None
) -> MedicationRequest:
    """
    Add detailed dosage instruction to MedicationRequest.

    Args:
        request: MedicationRequest to update
        instruction_text: Human-readable dosage instruction
        dose_value: Optional dose amount
        dose_unit: Optional dose unit
        frequency: Optional timing frequency

    Returns:
        Updated MedicationRequest
    """
    dosage = Dosage(text=instruction_text)

    # Add dose quantity
    if dose_value and dose_unit:
        dosage.doseAndRate = [{
            "doseQuantity": Quantity(
                value=dose_value,
                unit=dose_unit
            )
        }]

    # Add timing
    if frequency:
        dosage.timing = {"code": {"text": frequency}}

    if request.dosageInstruction is None:
        request.dosageInstruction = []

    request.dosageInstruction.append(dosage)

    return request


def create_medication_statement(
    medication_name: str,
    context: ProcessingContext,
    rxnorm_code: Optional[str] = None,
    patient_id: Optional[str] = None,
    status: str = "active"
) -> dict:
    """
    Create FHIR MedicationStatement for medication history.

    Args:
        medication_name: Name of medication
        context: Processing context
        rxnorm_code: Optional RxNorm code
        patient_id: Optional patient identifier
        status: Medication status

    Returns:
        MedicationStatement dict
    """
    medication_concept = build_medication_concept(medication_name, rxnorm_code)

    # Build subject reference (required)
    subject_id = patient_id or getattr(context, 'patient_id', None) or "anonymous"

    statement = {
        "resourceType": "MedicationStatement",
        "id": str(uuid4()),
        "status": status,
        "medication": medication_concept.model_dump(exclude_none=True),
        "subject": {
            "reference": f"Patient/{subject_id}"
        },
        "effectiveDateTime": datetime.now(timezone.utc).isoformat()
    }

    return statement
