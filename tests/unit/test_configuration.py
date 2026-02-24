# ============================================================================
# TEST 1: Configuration and Setup
# ============================================================================

def test_configuration():
    """Test that configuration loads correctly"""
    print("=" * 70)
    print("TEST 1: Configuration Loading")
    print("=" * 70)
    
    from src.medical_ingestion.config import base_settings, threshold_settings, hardware_settings, medgemma_settings
    
    print(f"✓ Configuration loaded successfully")
    print(f"  - Model path: {base_settings.MODEL_PATH}")
    print(f"  - Templates dir: {base_settings.TEMPLATES_DIR}")
    print(f"  - Template threshold: {threshold_settings.TEMPLATE_MATCH_THRESHOLD}")
    print(f"  - Human review threshold: {threshold_settings.HUMAN_REVIEW_THRESHOLD}")
    print(f"  - Use GPU: {hardware_settings.USE_GPU}")
    print(f"  - MedGemma max tokens: {medgemma_settings.MEDGEMMA_MAX_TOKENS}")
    
    # Test directory creation
    base_settings.create_directories()
    print(f"✓ Created necessary directories")
    
    # Test validator
    assert threshold_settings.TEMPLATE_MATCH_THRESHOLD > threshold_settings.HUMAN_REVIEW_THRESHOLD, \
        "Threshold validation failed"
    print(f"✓ Configuration validators working")
    
    print("\n✅ Configuration test PASSED\n")