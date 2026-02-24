# ============================================================================
# TEST 3: Processing Context
# ============================================================================

def test_context():
    """Test ProcessingContext creation and manipulation"""
    print("=" * 70)
    print("TEST 3: Processing Context")
    print("=" * 70)
    
    from pathlib import Path
    from src.medical_ingestion.core.context.processing_context import (
        ProcessingContext, ExtractedValue, ConfidenceLevel, ReviewPriority
    )
    
    # Create context
    ctx = ProcessingContext(
        document_path=Path("test_lab.pdf"),
        patient_demographics={"age": 45, "sex": "F"}
    )
    
    print(f"✓ Context created with ID: {ctx.document_id}")
    print(f"  - Document path: {ctx.document_path}")
    print(f"  - Patient demographics: {ctx.patient_demographics}")
    
    # Add extracted value
    value = ExtractedValue(
        field_name="hemoglobin",
        value=12.5,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template",
        reference_min=12.0,
        reference_max=15.5,
        abnormal_flag=None
    )
    
    ctx.add_extracted_value(value)
    print(f"\n✓ Added extracted value: {value.field_name} = {value.value} {value.unit}")
    print(f"  - Confidence: {value.confidence}")
    print(f"  - Overall confidence updated: {ctx.overall_confidence}")
    
    # Add warnings and flags
    ctx.add_warning("Example warning")
    ctx.add_quality_flag("hemolysis_detected", severity="warning")
    print(f"\n✓ Added warnings and quality flags")
    print(f"  - Warnings: {ctx.warnings}")
    print(f"  - Quality flags: {ctx.quality_flags}")
    
    # Test confidence calculation
    ctx.calculate_confidence_level()
    print(f"\n✓ Confidence level calculated: {ctx.confidence_level}")
    
    # Test summary
    summary = ctx.get_summary()
    print(f"\n✓ Context summary generated:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    
    print("\n✅ Context test PASSED\n")