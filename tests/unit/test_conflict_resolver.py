# ============================================================================
# FILE: tests/unit/test_conflict_resolver.py
# ============================================================================
"""
Unit tests for conflict resolver
"""

import pytest
from pathlib import Path
from src.validators.conflict_resolver import (
    ConflictResolver,
    ConflictResolution,
    resolve_conflict
)
from src.medical_ingestion.core.context.processing_context import ProcessingContext
from src.medical_ingestion.core.context.extracted_value import ExtractedValue


def test_conflict_resolver_init():
    """Test conflict resolver initialization"""
    resolver = ConflictResolver()
    assert resolver.confidence_threshold == 0.7
    assert resolver.human_review_threshold == 0.85


def test_conflict_resolver_custom_config():
    """Test conflict resolver with custom config"""
    config = {
        "confidence_threshold": 0.8,
        "human_review_threshold": 0.9
    }
    resolver = ConflictResolver(config)
    assert resolver.confidence_threshold == 0.8
    assert resolver.human_review_threshold == 0.9


def test_resolve_agreement_valid():
    """Test resolution when both validators agree (valid)"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    extracted.rule_validation = True
    extracted.ai_validation = True

    resolution, reasoning = resolver.resolve(extracted, context)

    assert resolution == ConflictResolution.ACCEPT_RULE
    assert "agree" in reasoning.lower()


def test_resolve_agreement_invalid():
    """Test resolution when both validators agree (invalid)"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=142.0,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    extracted.rule_validation = False
    extracted.ai_validation = False

    resolution, reasoning = resolver.resolve(extracted, context)

    assert resolution == ConflictResolution.REJECT_VALUE
    assert "agree" in reasoning.lower()


def test_resolve_conflict_low_confidence():
    """Test resolution with low confidence"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.5,  # Low confidence
        extraction_method="template"
    )
    extracted.rule_validation = True
    extracted.ai_validation = False

    resolution, reasoning = resolver.resolve(extracted, context)

    assert resolution == ConflictResolution.FLAG_HUMAN_REVIEW
    assert "Low confidence" in reasoning


def test_resolve_conflict_plausibility_failure():
    """Test resolution when rule fails plausibility"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=142.0,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    extracted.rule_validation = False
    extracted.ai_validation = True
    extracted.warnings = ["Plausibility check failed: value too high"]

    resolution, reasoning = resolver.resolve(extracted, context)

    assert resolution == ConflictResolution.ACCEPT_RULE
    assert "plausibility" in reasoning.lower()


def test_resolve_conflict_ai_high_confidence_invalid():
    """Test resolution when AI says invalid with high confidence"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.9,  # High confidence
        extraction_method="template"
    )
    extracted.rule_validation = True
    extracted.ai_validation = False

    resolution, reasoning = resolver.resolve(extracted, context)

    assert resolution == ConflictResolution.ACCEPT_AI
    assert "clinical implausibility" in reasoning.lower()


def test_resolve_conflict_ai_overrides_rule():
    """Test AI overriding rule-based rejection"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.9,
        extraction_method="template"
    )
    extracted.rule_validation = False
    extracted.ai_validation = True
    extracted.warnings = ["Out of reference range"]

    resolution, reasoning = resolver.resolve(extracted, context)

    assert resolution == ConflictResolution.ACCEPT_AI
    assert "overrides" in reasoning.lower()


def test_resolve_conflict_unresolved():
    """Test unresolved conflict (medium confidence)"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.75,  # Medium confidence
        extraction_method="template"
    )
    extracted.rule_validation = True
    extracted.ai_validation = False

    resolution, reasoning = resolver.resolve(extracted, context)

    assert resolution == ConflictResolution.FLAG_HUMAN_REVIEW
    assert "Unresolved" in reasoning


def test_batch_resolve():
    """Test batch conflict resolution"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    # Add values with different scenarios
    val1 = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    val1.rule_validation = True
    val1.ai_validation = True
    context.add_extracted_value(val1)

    val2 = ExtractedValue(
        field_name="potassium",
        value=4.2,
        unit="mmol/L",
        confidence=0.5,
        extraction_method="template"
    )
    val2.rule_validation = True
    val2.ai_validation = False
    context.add_extracted_value(val2)

    val3 = ExtractedValue(
        field_name="glucose",
        value=95,
        unit="mg/dL",
        confidence=0.95,
        extraction_method="template"
    )
    val3.rule_validation = False
    val3.ai_validation = False
    context.add_extracted_value(val3)

    results = resolver.batch_resolve(context)

    assert len(results) == 3
    assert results["hemoglobin"]["resolution"] == ConflictResolution.ACCEPT_RULE.value
    assert results["potassium"]["needs_review"] is True
    assert results["glucose"]["resolution"] == ConflictResolution.REJECT_VALUE.value


def test_get_summary():
    """Test getting conflict resolution summary"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    # Add values
    val1 = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    val1.rule_validation = True
    val1.ai_validation = True
    context.add_extracted_value(val1)

    val2 = ExtractedValue(
        field_name="potassium",
        value=4.2,
        unit="mmol/L",
        confidence=0.5,
        extraction_method="template"
    )
    val2.rule_validation = True
    val2.ai_validation = False
    context.add_extracted_value(val2)

    summary = resolver.get_summary(context)

    assert summary["total_values"] == 2
    assert summary["conflicts"] == 1
    assert summary["needs_review"] >= 0
    assert summary["accepted"] >= 0
    assert summary["rejected"] >= 0


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
    extracted.rule_validation = True
    extracted.ai_validation = True

    should_accept = resolve_conflict(extracted, context)

    assert should_accept is True


def test_validation_conflict_flag():
    """Test that validation_conflict flag is set"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    extracted.rule_validation = True
    extracted.ai_validation = False
    context.add_extracted_value(extracted)

    results = resolver.batch_resolve(context)

    assert extracted.validation_conflict is True


def test_warnings_added():
    """Test that warnings are added to extracted values"""
    resolver = ConflictResolver()
    context = ProcessingContext(document_path=Path("test.pdf"))

    # Low confidence conflict
    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.5,
        extraction_method="template"
    )
    extracted.rule_validation = True
    extracted.ai_validation = False

    resolution, reasoning = resolver.resolve(extracted, context)

    assert len(extracted.warnings) > 0
    assert any("human review" in w.lower() for w in extracted.warnings)
