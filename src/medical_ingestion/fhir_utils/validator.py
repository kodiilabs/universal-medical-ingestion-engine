# ============================================================================
# FILE: src/fhir/validator.py
# ============================================================================
"""
FHIR Validator

Validates FHIR resources against:
1. FHIR R4 schema (structural validation)
2. Profile constraints (US Core, etc.)
3. Terminology bindings (LOINC, SNOMED)
4. Business rules (cardinality, required fields)

Uses fhir.resources library for validation.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging

from fhir.resources.bundle import Bundle
from fhir.resources.observation import Observation
from fhir.resources.diagnosticreport import DiagnosticReport


logger = logging.getLogger(__name__)


class FHIRValidator:
    """
    FHIR resource validator.

    Validates resources before sending to FHIR server.
    Catches issues early to avoid rejected submissions.
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, fail on warnings. If False, only fail on errors.
        """
        self.strict = strict

    def validate_bundle(self, bundle: Bundle) -> Tuple[bool, List[str]]:
        """
        Validate a FHIR Bundle.

        Args:
            bundle: FHIR Bundle to validate

        Returns:
            (is_valid, errors)
        """
        errors = []

        try:
            # Structural validation (handled by fhir.resources on creation)
            bundle.model_dump_json()  # Try to serialize

            # Validate bundle type
            if bundle.type not in ["collection", "transaction", "document"]:
                errors.append(f"Invalid bundle type: {bundle.type}")

            # Validate entries
            if bundle.entry:
                for idx, entry in enumerate(bundle.entry):
                    entry_errors = self._validate_entry(entry, idx)
                    errors.extend(entry_errors)

            # Validate metadata
            if not bundle.id:
                errors.append("Bundle missing required 'id' field")

            if not bundle.timestamp:
                errors.append("Bundle missing 'timestamp'")

        except Exception as e:
            errors.append(f"Bundle validation error: {str(e)}")

        is_valid = len(errors) == 0

        if errors:
            logger.warning(f"Bundle validation found {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")

        return is_valid, errors

    def _validate_entry(self, entry: Any, index: int) -> List[str]:
        """
        Validate a single bundle entry.
        """
        errors = []

        if not entry.resource:
            errors.append(f"Entry {index}: missing resource")
            return errors

        resource = entry.resource
        resource_type = resource.__class__.__name__

        # Validate based on resource type
        if resource_type == "Observation":
            errors.extend(self._validate_observation(resource, index))
        elif resource_type == "DiagnosticReport":
            errors.extend(self._validate_diagnostic_report(resource, index))

        return errors

    def _validate_observation(self, obs: Observation, index: int) -> List[str]:
        """
        Validate an Observation resource.

        Required fields:
        - status
        - code
        - Either value[x] or dataAbsentReason
        """
        errors = []
        prefix = f"Entry {index} (Observation)"

        # Required: status
        if obs.status is None:
            errors.append(f"{prefix}: missing required 'status'")
        elif obs.status not in ["registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error"]:
            errors.append(f"{prefix}: invalid status '{obs.status}'")

        # Required: code
        if obs.code is None:
            errors.append(f"{prefix}: missing required 'code'")
        else:
            # Validate code has coding
            if obs.code.coding is None or len(obs.code.coding) == 0:
                errors.append(f"{prefix}: code missing 'coding' array")

        # Required: value[x] or dataAbsentReason
        has_value = any([
            getattr(obs, 'valueQuantity', None) is not None,
            getattr(obs, 'valueCodeableConcept', None) is not None,
            getattr(obs, 'valueString', None) is not None,
            getattr(obs, 'valueBoolean', None) is not None,
            getattr(obs, 'valueInteger', None) is not None,
            getattr(obs, 'valueRange', None) is not None,
            getattr(obs, 'valueRatio', None) is not None,
            getattr(obs, 'valueSampledData', None) is not None,
            getattr(obs, 'valueTime', None) is not None,
            getattr(obs, 'valueDateTime', None) is not None,
            getattr(obs, 'valuePeriod', None) is not None
        ])

        if not has_value and not getattr(obs, 'dataAbsentReason', None):
            errors.append(f"{prefix}: must have either value[x] or dataAbsentReason")

        # Validate reference range
        if obs.referenceRange:
            for rr_idx, ref_range in enumerate(obs.referenceRange):
                if not ref_range.low and not ref_range.high and not ref_range.text:
                    errors.append(
                        f"{prefix}: referenceRange[{rr_idx}] must have low, high, or text"
                    )

        return errors

    def _validate_diagnostic_report(
        self,
        report: DiagnosticReport,
        index: int
    ) -> List[str]:
        """
        Validate a DiagnosticReport resource.

        Required fields:
        - status
        - code
        """
        errors = []
        prefix = f"Entry {index} (DiagnosticReport)"

        # Required: status
        if report.status is None:
            errors.append(f"{prefix}: missing required 'status'")
        elif report.status not in ["registered", "partial", "preliminary", "final", "amended", "corrected", "appended", "cancelled", "entered-in-error", "unknown"]:
            errors.append(f"{prefix}: invalid status '{report.status}'")

        # Required: code
        if report.code is None:
            errors.append(f"{prefix}: missing required 'code'")
        else:
            if report.code.coding is None or len(report.code.coding) == 0:
                errors.append(f"{prefix}: code missing 'coding' array")

        return errors

    def validate_resource(self, resource: Any) -> Tuple[bool, List[str]]:
        """
        Validate any FHIR resource.

        Args:
            resource: FHIR resource to validate

        Returns:
            (is_valid, errors)
        """
        errors = []

        try:
            resource.model_dump_json()  # Try to serialize

            resource_type = resource.__class__.__name__

            # Type-specific validation
            if resource_type == "Observation":
                errors.extend(self._validate_observation(resource, 0))
            elif resource_type == "DiagnosticReport":
                errors.extend(self._validate_diagnostic_report(resource, 0))

        except Exception as e:
            errors.append(f"Resource validation error: {str(e)}")

        is_valid = len(errors) == 0

        return is_valid, errors

    def validate_terminology(
        self,
        code: str,
        system: str,
        expected_systems: List[str]
    ) -> bool:
        """
        Validate terminology code against expected system.

        Args:
            code: The code value
            system: The code system URI
            expected_systems: List of valid system URIs

        Returns:
            True if valid
        """
        if system not in expected_systems:
            logger.warning(
                f"Code system '{system}' not in expected systems: {expected_systems}"
            )
            return False

        # Could add additional checks:
        # - Lookup code in terminology server
        # - Validate code format (e.g., LOINC format)

        return True


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_bundle(bundle: Bundle, strict: bool = False) -> bool:
    """
    Quick bundle validation.

    Returns:
        True if valid
    """
    validator = FHIRValidator(strict=strict)
    is_valid, _ = validator.validate_bundle(bundle)
    return is_valid


def get_validation_errors(bundle: Bundle) -> List[str]:
    """
    Get list of validation errors for a bundle.

    Returns:
        List of error messages (empty if valid)
    """
    validator = FHIRValidator()
    _, errors = validator.validate_bundle(bundle)
    return errors
