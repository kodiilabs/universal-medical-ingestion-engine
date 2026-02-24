# ============================================================================
# TEST 6: Template Matching Logic
# ============================================================================

def test_template_matcher():
    """Test template matching without actual templates"""
    print("=" * 70)
    print("TEST 6: Template Matcher")
    print("=" * 70)
    
    from src.medical_ingestion.processors.lab.agents.template_matcher import TemplateMatchingAgent
    
    matcher = TemplateMatchingAgent({})
    print(f"✓ Template Matcher initialized")
    print(f"  - Loaded {len(matcher.templates)} templates")
    
    # Test layout signature checking
    lab_text = """
    Test                Result      Reference Range    Flag
    ----------------------------------------------------------------
    WBC                 7.2         4.5-11.0 K/uL
    Hemoglobin          14.2        13.5-17.5 g/dL      
    """
    
    layout_sig = {
        "columns": 4,
        "has_reference_range": True,
        "has_abnormal_flags": True
    }
    
    score = matcher._check_layout_signature(lab_text, layout_sig)
    print(f"\n✓ Layout signature check: {score:.2f}")
    
    print("""
To test with actual templates, create template JSON files in:
    src/medical_ingestion/processors/lab/templates/
    
Example template structure:
{
    "id": "quest_cbc_v1",
    "vendor": "Quest Diagnostics",
    "test_type": "cbc",
    "header_pattern": "Quest.*Complete Blood Count",
    "vendor_markers": ["Quest", "www.questdiagnostics.com"],
    "required_fields": ["WBC", "RBC", "Hemoglobin"],
    "layout_signature": {
        "columns": 5,
        "has_reference_range": true
    }
}
""")
    
    print("\n✅ Template Matcher structure test PASSED\n")
