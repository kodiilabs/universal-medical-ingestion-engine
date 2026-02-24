# ============================================================================
# FILE: tests/unit/test_fhir_builder.py
# ============================================================================
"""
Unit tests for FHIR builder
"""

def test_fhir_builder():
    """Test FHIR builder functionality"""
    print("=" * 70)
    print("TEST: FHIR Builder")
    print("=" * 70)
    
    from src.medical_ingestion.fhir_utils.builder import FHIRBuilder
    from src.medical_ingestion.core.context import ProcessingContext, ExtractedValue
    from pathlib import Path
    
    builder = FHIRBuilder()
    print("✓ FHIR Builder initialized")
    
    # Create context with lab data
    context = ProcessingContext(
        document_path=Path("test.pdf"),
        patient_id="12345"
    )
    context.document_type = "lab"
    
    # Add extracted value
    value = ExtractedValue(
        field_name="hemoglobin",
        value=14.2,
        unit="g/dL",
        confidence=0.95,
        extraction_method="template",
        reference_min=13.5,
        reference_max=17.5
    )
    context.add_extracted_value(value)
    
    print("✓ Created context with lab value")
    
    # Test create observation
    observation = builder._create_lab_observation(value, context)
    
    assert observation.status == 'final'
    assert observation.code is not None
    assert observation.valueQuantity is not None
    print("✓ Created FHIR Observation")
    
    # Test abnormal flag mapping
    assert builder._map_abnormal_flag("H") == "H"
    assert builder._map_abnormal_flag("CRITICAL") == "HH"
    print("✓ Abnormal flag mapping works")
    
    print("\n✅ FHIR Builder test PASSED\n")