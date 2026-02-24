# ============================================================================
# FILE: tests/unit/test_validators.py
# ============================================================================
"""
Unit tests for validators
"""

def test_validation_agent():
    """Test validation agent functionality"""
    print("=" * 70)
    print("TEST: Validation Agent")
    print("=" * 70)
    
    from src.medical_ingestion.processors.lab.agents.validator import ValidationAgent
    from src.medical_ingestion.core.context.processing_context import ProcessingContext, ExtractedValue
    from pathlib import Path
    
    agent = ValidationAgent({})
    print(f"✓ Validation Agent initialized: {agent.get_name()}")
    
    context = ProcessingContext(document_path=Path("test.pdf"))
    
    # Test plausible value
    value = ExtractedValue(
        field_name="hemoglobin",
        value=14.2,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    
    result = agent._validate_with_rules(value, context)
    assert result == True
    print("✓ Plausible hemoglobin value validated")
    
    # Test implausible value (decimal error)
    value2 = ExtractedValue(
        field_name="hemoglobin",
        value=142.0,  # Likely should be 14.2
        unit="g/dL",
        confidence=0.95,
        extraction_method="template"
    )
    
    result2 = agent._validate_with_rules(value2, context)
    assert result2 == False
    print("✓ Implausible hemoglobin value rejected")
    
    # Test potassium validation
    value3 = ExtractedValue(
        field_name="potassium",
        value=4.2,
        unit="mmol/L",
        confidence=0.95,
        extraction_method="template"
    )
    
    result3 = agent._validate_with_rules(value3, context)
    assert result3 == True
    print("✓ Normal potassium value validated")
    
    print("\n✅ Validation Agent test PASSED\n")