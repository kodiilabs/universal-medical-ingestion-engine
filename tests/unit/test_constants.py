def test_constants():
    """Test medical constants are loaded"""
    print("=" * 70)
    print("TEST 2: Medical Constants")
    print("=" * 70)
    
    from src.medical_ingestion.constants import (
        LOINC_CODES, REFERENCE_RANGES, UNIT_CONVERSIONS,
        PLAUSIBILITY_RANGES, CRITICAL_VALUES, REFLEX_PROTOCOLS
    )
    
    print(f"✓ LOINC codes loaded: {len(LOINC_CODES)} tests")
    print(f"  Example: Hemoglobin = {LOINC_CODES['hemoglobin']}")
    
    print(f"\n✓ Reference ranges loaded: {len(REFERENCE_RANGES)} tests")
    print(f"  Example: Hemoglobin (male) = {REFERENCE_RANGES['hemoglobin']['male']}")
    
    print(f"\n✓ Unit conversions loaded: {len(UNIT_CONVERSIONS)} conversions")
    
    print(f"\n✓ Plausibility ranges loaded: {len(PLAUSIBILITY_RANGES)} tests")
    print(f"  Example: Potassium plausible range = {PLAUSIBILITY_RANGES['potassium']}")
    
    print(f"\n✓ Critical values loaded: {len(CRITICAL_VALUES)} tests")
    print(f"  Example: Potassium critical = {CRITICAL_VALUES['potassium']}")
    
    print(f"\n✓ Reflex protocols loaded: {len(REFLEX_PROTOCOLS)} protocols")
    print(f"  Example: TSH elevated → {REFLEX_PROTOCOLS['tsh_elevated']['reflex_tests']}")
    
    print("\n✅ Constants test PASSED\n")