# ============================================================================
# FILE: tests/unit/test_ai_validator.py
# ============================================================================
"""
Unit tests for AI validator
"""

import pytest
from pathlib import Path
from src.validators.ai_validator import AIValidator
from src.medical_ingestion.core.context.processing_context import ProcessingContext
from src.medical_ingestion.core.context.extracted_value import ExtractedValue


def test_ai_validator_init():
    """Test AI validator initialization"""
    validator = AIValidator()
    assert validator.medgemma is not None


def test_build_validation_prompt():
    """Test prompt building for validation"""
    validator = AIValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 45}

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template",
        reference_min=13.5,
        reference_max=17.5
    )

    prompt = validator._build_validation_prompt(extracted, context)

    assert "hemoglobin" in prompt.lower()
    assert "14.5" in prompt
    assert "g/dL" in prompt
    assert "male" in prompt.lower()
    assert "45" in prompt
    assert "JSON" in prompt


def test_build_validation_prompt_with_other_values():
    """Test prompt building with other lab values"""
    validator = AIValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 45}

    # Add multiple values
    context.add_extracted_value(ExtractedValue(
        field_name="potassium",
        value=4.2,
        unit="mmol/L",
        confidence=0.95,
        extraction_method="template"
    ))

    context.add_extracted_value(ExtractedValue(
        field_name="sodium",
        value=140,
        unit="mmol/L",
        confidence=0.95,
        extraction_method="template"
    ))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template",
        reference_min=13.5,
        reference_max=17.5
    )

    prompt = validator._build_validation_prompt(extracted, context)

    assert "potassium" in prompt.lower()
    assert "sodium" in prompt.lower()


def test_parse_response_valid_json():
    """Test parsing valid JSON response"""
    validator = AIValidator()

    response_text = '''
    {
        "plausible": true,
        "reasoning": "Value is within normal range",
        "confidence": 0.95
    }
    '''

    result = validator._parse_response(response_text)

    assert result is not None
    assert result["plausible"] is True
    assert "reasoning" in result
    assert result["confidence"] == 0.95


def test_parse_response_with_extra_text():
    """Test parsing JSON with extra text"""
    validator = AIValidator()

    response_text = '''
    Here is my analysis:
    {
        "plausible": false,
        "reasoning": "Value seems too high",
        "confidence": 0.85
    }
    Extra text after JSON
    '''

    result = validator._parse_response(response_text)

    assert result is not None
    assert result["plausible"] is False


def test_parse_response_invalid_json():
    """Test parsing invalid JSON"""
    validator = AIValidator()

    response_text = "This is not JSON at all"

    result = validator._parse_response(response_text)

    assert result is None


def test_parse_response_missing_fields():
    """Test parsing JSON with missing required fields"""
    validator = AIValidator()

    response_text = '''
    {
        "plausible": true
    }
    '''

    result = validator._parse_response(response_text)

    # Should fail because 'reasoning' is missing
    assert result is None


@pytest.mark.asyncio
async def test_validate_mock_success():
    """Test validation with mocked MedGemma (success)"""
    validator = AIValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 45}

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template",
        reference_min=13.5,
        reference_max=17.5
    )

    # Mock the medgemma generate method
    async def mock_generate(prompt, max_tokens, temperature):
        return {
            "text": '{"plausible": true, "reasoning": "Normal value", "confidence": 0.9}'
        }

    validator.medgemma.generate = mock_generate

    is_valid, reasoning, confidence = await validator.validate(extracted, context)

    assert is_valid is True
    assert "Normal value" in reasoning
    assert confidence == 0.9


@pytest.mark.asyncio
async def test_validate_mock_failure():
    """Test validation with mocked MedGemma (failure)"""
    validator = AIValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=142.0,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    # Mock the medgemma generate method
    async def mock_generate(prompt, max_tokens, temperature):
        return {
            "text": '{"plausible": false, "reasoning": "Likely decimal error", "confidence": 0.95}'
        }

    validator.medgemma.generate = mock_generate

    is_valid, reasoning, confidence = await validator.validate(extracted, context)

    assert is_valid is False
    assert "decimal" in reasoning.lower()


@pytest.mark.asyncio
async def test_validate_error_handling():
    """Test validation with error in AI call"""
    validator = AIValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )

    # Mock the medgemma generate method to raise error
    async def mock_generate(prompt, max_tokens, temperature):
        raise Exception("AI service unavailable")

    validator.medgemma.generate = mock_generate

    is_valid, reasoning, confidence = await validator.validate(extracted, context)

    # Should default to valid when AI fails
    assert is_valid is True
    assert "failed" in reasoning.lower()
    assert confidence == 0.5


@pytest.mark.asyncio
async def test_batch_validate():
    """Test batch validation"""
    validator = AIValidator()
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

    # Mock the medgemma generate method
    async def mock_generate(prompt, max_tokens, temperature):
        return {
            "text": '{"plausible": true, "reasoning": "Valid", "confidence": 0.9}'
        }

    validator.medgemma.generate = mock_generate

    results = await validator.batch_validate(context)

    assert len(results) == 2
    assert results["hemoglobin"]["valid"] is True
    assert results["potassium"]["valid"] is True


@pytest.mark.asyncio
async def test_validate_with_explanation():
    """Test validation with detailed explanation"""
    validator = AIValidator()
    context = ProcessingContext(document_path=Path("test.pdf"))
    context.patient_demographics = {"sex": "male", "age": 45}

    extracted = ExtractedValue(
        field_name="hemoglobin",
        value=14.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template",
        reference_min=13.5,
        reference_max=17.5
    )
    extracted.rule_validation = True

    # Mock the medgemma generate method
    async def mock_generate(prompt, max_tokens, temperature):
        return {
            "text": '{"plausible": true, "reasoning": "Normal value", "confidence": 0.9}'
        }

    validator.medgemma.generate = mock_generate

    result = await validator.validate_with_explanation(extracted, context)

    assert result["field_name"] == "hemoglobin"
    assert result["value"] == 14.5
    assert result["unit"] == "g/dL"
    assert result["valid"] is True
    assert result["reasoning"] == "Normal value"
    assert result["confidence"] == 0.9
    assert result["reference_range"] == "13.5-17.5"
    assert result["rule_validation"] is True
