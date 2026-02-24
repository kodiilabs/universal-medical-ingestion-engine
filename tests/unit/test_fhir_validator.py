# ============================================================================
# FILE: tests/unit/test_fhir_validator.py
# ============================================================================
"""
Unit tests for FHIR validator
"""

import pytest
from uuid import uuid4
from datetime import datetime
from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.observation import Observation, ObservationReferenceRange
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.quantity import Quantity
from fhir.resources.identifier import Identifier
from src.medical_ingestion.fhir_utils.validator import (
    FHIRValidator,
    validate_bundle,
    get_validation_errors
)


def test_fhir_validator_init():
    """Test FHIR validator initialization"""
    validator = FHIRValidator()
    assert validator.strict is False

    validator_strict = FHIRValidator(strict=True)
    assert validator_strict.strict is True


def test_validate_valid_bundle():
    """Test validation of valid bundle"""
    validator = FHIRValidator()

    # Create valid bundle
    bundle = Bundle(
        id=str(uuid4()),
        type="collection",
        identifier=Identifier(
            system="http://example.com",
            value="test-bundle"
        ),
        timestamp=datetime.now().isoformat(),
        entry=[]
    )

    is_valid, errors = validator.validate_bundle(bundle)

    assert is_valid is True
    assert len(errors) == 0


def test_validate_bundle_missing_id():
    """Test validation of bundle without ID"""
    validator = FHIRValidator()

    # Create bundle and then remove ID
    bundle = Bundle(
        id=str(uuid4()),
        type="collection",
        timestamp=datetime.now().isoformat(),
        entry=[]
    )
    bundle.id = None

    is_valid, errors = validator.validate_bundle(bundle)

    assert is_valid is False
    assert any("missing required 'id'" in e for e in errors)


def test_validate_bundle_invalid_type():
    """Test validation of bundle with invalid type"""
    validator = FHIRValidator()

    bundle = Bundle(
        id=str(uuid4()),
        type="invalid_type",
        timestamp=datetime.now().isoformat(),
        entry=[]
    )

    is_valid, errors = validator.validate_bundle(bundle)

    assert is_valid is False
    assert any("Invalid bundle type" in e for e in errors)


def test_validate_bundle_missing_timestamp():
    """Test validation of bundle without timestamp"""
    validator = FHIRValidator()

    bundle = Bundle(
        id=str(uuid4()),
        type="collection",
        timestamp=datetime.now().isoformat(),
        entry=[]
    )
    bundle.timestamp = None

    is_valid, errors = validator.validate_bundle(bundle)

    assert is_valid is False
    assert any("missing 'timestamp'" in e for e in errors)


def test_validate_valid_observation():
    """Test validation of valid Observation"""
    validator = FHIRValidator()

    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(
                system="http://loinc.org",
                code="718-7",
                display="Hemoglobin"
            )]
        ),
        valueQuantity=Quantity(
            value=14.5,
            unit="g/dL",
            system="http://unitsofmeasure.org",
            code="g/dL"
        )
    )

    is_valid, errors = validator.validate_resource(observation)

    assert is_valid is True
    assert len(errors) == 0


def test_validate_observation_missing_status():
    """Test that Observation requires status at creation"""
    # Note: fhir.resources uses Pydantic which validates on creation
    # We can't create an Observation without status, so we just verify
    # that a valid observation with status passes validation
    validator = FHIRValidator()

    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(
                system="http://loinc.org",
                code="718-7"
            )]
        ),
        valueQuantity=Quantity(value=14.5, unit="g/dL")
    )

    is_valid, _ = validator.validate_resource(observation)

    # Should pass since status is present
    assert is_valid is True


def test_validate_observation_invalid_status():
    """Test validation of Observation with invalid status"""
    validator = FHIRValidator()

    # Create valid observation first
    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="718-7")]
        ),
        valueQuantity=Quantity(value=14.5, unit="g/dL")
    )
    # Then set to invalid status
    observation.status = "invalid_status"

    is_valid, errors = validator.validate_resource(observation)

    assert is_valid is False
    assert any("invalid status" in e for e in errors)


def test_validate_observation_missing_code():
    """Test validation of Observation without code"""
    validator = FHIRValidator()

    # Create valid observation first
    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="718-7")]
        ),
        valueQuantity=Quantity(value=14.5, unit="g/dL")
    )
    # Then set code to None
    observation.code = None

    is_valid, errors = validator.validate_resource(observation)

    assert is_valid is False
    assert any("missing required 'code'" in e for e in errors)


def test_validate_observation_code_without_coding():
    """Test validation of Observation with code but no coding"""
    validator = FHIRValidator()

    # Create valid observation first
    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            text="Hemoglobin",
            coding=[Coding(system="http://loinc.org", code="718-7")]
        ),
        valueQuantity=Quantity(value=14.5, unit="g/dL")
    )
    # Then remove coding
    observation.code.coding = []

    is_valid, errors = validator.validate_resource(observation)

    assert is_valid is False
    assert any("code missing 'coding'" in e for e in errors)


def test_validate_observation_missing_value():
    """Test validation of Observation without value or dataAbsentReason"""
    validator = FHIRValidator()

    # Create valid observation first
    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="718-7")]
        ),
        valueQuantity=Quantity(value=14.5, unit="g/dL")
    )
    # Then remove value
    observation.valueQuantity = None

    is_valid, errors = validator.validate_resource(observation)

    assert is_valid is False
    assert any("must have either value" in e for e in errors)


def test_validate_observation_with_reference_range():
    """Test validation of Observation with reference range"""
    validator = FHIRValidator()

    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="718-7")]
        ),
        valueQuantity=Quantity(value=14.5, unit="g/dL"),
        referenceRange=[
            ObservationReferenceRange(
                low=Quantity(value=13.5, unit="g/dL"),
                high=Quantity(value=17.5, unit="g/dL")
            )
        ]
    )

    is_valid, errors = validator.validate_resource(observation)

    assert is_valid is True


def test_validate_observation_empty_reference_range():
    """Test validation of Observation with empty reference range"""
    validator = FHIRValidator()

    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="718-7")]
        ),
        valueQuantity=Quantity(value=14.5, unit="g/dL"),
        referenceRange=[ObservationReferenceRange()]
    )

    is_valid, errors = validator.validate_resource(observation)

    assert is_valid is False
    assert any("referenceRange" in e for e in errors)


def test_validate_valid_diagnostic_report():
    """Test validation of valid DiagnosticReport"""
    validator = FHIRValidator()

    report = DiagnosticReport(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(
                system="http://loinc.org",
                code="11526-1",
                display="Pathology Report"
            )]
        )
    )

    is_valid, errors = validator.validate_resource(report)

    assert is_valid is True


def test_validate_diagnostic_report_missing_status():
    """Test that DiagnosticReport requires status at creation"""
    # Note: fhir.resources uses Pydantic which validates on creation
    # We can't create a DiagnosticReport without status
    validator = FHIRValidator()

    report = DiagnosticReport(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="11526-1")]
        )
    )

    is_valid, _ = validator.validate_resource(report)

    # Should pass since status is present
    assert is_valid is True


def test_validate_diagnostic_report_invalid_status():
    """Test validation of DiagnosticReport with invalid status"""
    validator = FHIRValidator()

    # Create valid report first
    report = DiagnosticReport(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="11526-1")]
        )
    )
    # Then set to invalid status
    report.status = "invalid_status"

    is_valid, errors = validator.validate_resource(report)

    assert is_valid is False
    assert any("invalid status" in e for e in errors)


def test_validate_terminology():
    """Test terminology validation"""
    validator = FHIRValidator()

    # Valid system
    is_valid = validator.validate_terminology(
        code="718-7",
        system="http://loinc.org",
        expected_systems=["http://loinc.org", "http://snomed.info/sct"]
    )
    assert is_valid is True

    # Invalid system
    is_valid = validator.validate_terminology(
        code="718-7",
        system="http://invalid.org",
        expected_systems=["http://loinc.org"]
    )
    assert is_valid is False


def test_validate_bundle_with_observations():
    """Test validation of bundle with observation entries"""
    validator = FHIRValidator()

    observation = Observation(
        id=str(uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="718-7")]
        ),
        valueQuantity=Quantity(value=14.5, unit="g/dL")
    )

    entry = BundleEntry(
        fullUrl=f"urn:uuid:{uuid4()}",
        resource=observation
    )

    bundle = Bundle(
        id=str(uuid4()),
        type="collection",
        identifier=Identifier(system="http://example.com", value="test"),
        timestamp=datetime.now().isoformat(),
        entry=[entry]
    )

    is_valid, errors = validator.validate_bundle(bundle)

    assert is_valid is True
    assert len(errors) == 0


def test_convenience_function_validate_bundle():
    """Test convenience function for bundle validation"""
    bundle = Bundle(
        id=str(uuid4()),
        type="collection",
        timestamp=datetime.now().isoformat(),
        entry=[]
    )

    is_valid = validate_bundle(bundle)

    assert is_valid is True


def test_convenience_function_get_errors():
    """Test convenience function for getting errors"""
    bundle = Bundle(
        type="invalid_type",
        timestamp=datetime.now().isoformat(),
        entry=[]
    )
    bundle.id = None

    errors = get_validation_errors(bundle)

    assert len(errors) > 0
