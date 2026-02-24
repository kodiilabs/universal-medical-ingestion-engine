# ============================================================================
# FILE: tests/unit/test_rule_validator.py
# ============================================================================
"""
Unit tests for rule-based validator
"""

import pytest
from pathlib import Path
from src.validators.rule_validator import RuleValidator, validate_value
from src.medical_ingestion.core.context.processing_context import ProcessingContext
from src.medical_ingestion.core.context.extracted_value import ExtractedValue


def test_rule_validator_init():
    """Test rule validator initialization"""
    validator = RuleValidator()
    assert validator.plausibility_checker is not None
    assert validator.reference_ranges is not None
    assert validator.critical_values is not None


def test_validate_plausible_hemoglobin():
    """Test validation of plausible hemoglobin"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 35}

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, abnormal_flag = validator.validate(extracted, context)

    assert is_valid is True
    assert abnormal_flag == "N"


def test_validate_implausible_hemoglobin():
    """Test validation of implausible hemoglobin (decimal error)"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=142.0,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, abnormal_flag = validator.validate(extracted, context)

    assert is_valid is False
    assert len(warnings) > 0
    assert "Plausibility" in warnings[0]


def test_validate_high_hemoglobin():
    """Test validation of high hemoglobin"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 35}

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=18.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, abnormal_flag = validator.validate(extracted, context)

    assert is_valid is True
    assert abnormal_flag == "H"


def test_validate_low_hemoglobin():
    """Test validation of low hemoglobin"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "female", "age": 30}

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=10.0,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, abnormal_flag = validator.validate(extracted, context)

    assert is_valid is True
    assert abnormal_flag == "L"


def test_validate_critical_potassium():
    """Test validation of critical potassium"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 45}

    extracted = ExtractedValue(
        field_name="potassium",
        value=6.8,
        unit="mmol/L",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, abnormal_flag = validator.validate(extracted, context)

    assert is_valid is True
    assert abnormal_flag in ["CRITICAL HIGH", "H"]
    assert any("CRITICAL" in w for w in warnings)


def test_validate_normal_potassium():
    """Test validation of normal potassium"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 45}

    extracted = ExtractedValue(
        field_name="potassium",
        value=4.2,
        unit="mmol/L",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, abnormal_flag = validator.validate(extracted, context)

    assert is_valid is True
    assert abnormal_flag == "N"


def test_validate_unknown_test():
    """Test validation of unknown test"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="unknown_test",
        value=99.9,
        unit="units",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, abnormal_flag = validator.validate(extracted, context)

    assert is_valid is True


def test_validate_wrong_unit():
    """Test validation with wrong unit"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.2,
        unit="mg/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, abnormal_flag = validator.validate(extracted, context)

    assert is_valid is False
    assert any("Unit mismatch" in w for w in warnings)


def test_cross_field_validation_hgb_hct():
    """Test hemoglobin/hematocrit correlation"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))

    # Add hemoglobin
    hgb = ExtractedValue(
        field_name="hemoglobin",
        value=14.0,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    context.add_extracted_value(hgb)

    # Add hematocrit (should be ~42, but we'll give bad value)
    hct = ExtractedValue(
        field_name="hematocrit",
        value=30.0,
        unit="%",
        confidence=0.95,
        extraction_method="template"
    )
    context.add_extracted_value(hct)

    is_valid, warnings, abnormal_flag = validator.validate(hgb, context)

    assert any("Hgb/Hct correlation" in w for w in warnings)


def test_cross_field_validation_anion_gap():
    """Test anion gap calculation"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))

    # Add sodium
    na = ExtractedValue(
        field_name="sodium",
        value=140,
        unit="mmol/L",
        confidence=0.95,
        extraction_method="template"
    )
    context.add_extracted_value(na)

    # Add chloride
    cl = ExtractedValue(
        field_name="chloride",
        value=105,
        unit="mmol/L",
        confidence=0.95,
        extraction_method="template"
    )
    context.add_extracted_value(cl)

    is_valid, warnings, abnormal_flag = validator.validate(na, context)

    # Should calculate anion gap (140 - 105 = 35, which is normal)
    assert is_valid is True


def test_batch_validate():
    """Test batch validation"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 45}

    # Add multiple values
    context.add_extracted_value(ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    ))

    context.add_extracted_value(ExtractedValue(
        field_name="potassium",
        value=4.2,
        unit="mmol/L",
        confidence=0.95,
        extraction_method="template"
    ))

    context.add_extracted_value(ExtractedValue(
        field_name="glucose",
        value=95,
        unit="mg/dL",
        confidence=0.95,
        extraction_method="template"
    ))

    results = validator.batch_validate(context)

    assert len(results) == 3
    assert results["hemoglobin"]["valid"] is True
    assert results["potassium"]["valid"] is True
    assert results["glucose"]["valid"] is True


def test_convenience_function():
    """Test convenience function"""
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid = validate_value(extracted, context)

    assert is_valid is True


def test_reference_range_sex_specific():
    """Test sex-specific reference ranges"""
    validator = RuleValidator()

    # Male hemoglobin (13.5-17.5)
    context_male = ProcessingContext(document_path=Path("test.pdf"))
    context_male.patient_demographics = {"sex": "male", "age": 35}

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=13.0,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, flag = validator.validate(extracted, context_male)
    assert flag == "L"

    # Female hemoglobin (12.0-15.5)
    context_female = ProcessingContext(document_path=Path("test.pdf"))
    context_female.patient_demographics = {"sex": "female", "age": 35}

    extracted2 = ExtractedValue(
        field_name="hemoglobin",
        value=13.0,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid2, warnings2, flag2 = validator.validate(extracted2, context_female)
    assert flag2 == "N"


def test_glucose_fasting_vs_random():
    """Test glucose validation"""
    validator = RuleValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))

    # Normal glucose
    extracted = ExtractedValue(
        field_name="glucose",
        value=90,
        unit="mg/dL",
        confidence=0.95,
        extraction_method="template"
    )

    is_valid, warnings, flag = validator.validate(extracted, context)
    assert is_valid is True
