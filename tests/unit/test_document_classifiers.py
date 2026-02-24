# ============================================================================
# FILE: tests/unit/test_classifiers.py
# ============================================================================
"""
Unit tests for document classifiers
"""

def test_document_classifier():
    """Test document classifier functionality"""
    print("=" * 70)
    print("TEST: Document Classifier")
    print("=" * 70)
    
    from src.medical_ingestion.classifiers.document_classifier import DocumentClassifier
    
    classifier = DocumentClassifier({})
    print(f"✓ Document Classifier initialized")
    print(f"  - Loaded patterns for {len(classifier.patterns)} document types")
    
    # Test with sample lab text
    lab_text = """
    Quest Diagnostics Laboratory Report
    
    Patient: John Doe
    Date: 2024-01-15
    
    COMPLETE BLOOD COUNT (CBC)
    
    Test                Result      Reference Range    Flag
    ----------------------------------------------------------------
    WBC                 7.2         4.5-11.0 K/uL
    RBC                 4.8         4.5-5.5 M/uL
    Hemoglobin          14.2        13.5-17.5 g/dL
    Hematocrit          42.1        38.8-50.0 %
    Platelets           245         150-400 K/uL       
    """
    
    result = classifier._classify_by_fingerprint(lab_text)
    
    print(f"\n✓ Fingerprint classification complete:")
    print(f"  - Type: {result['type']}")
    print(f"  - Confidence: {result['confidence']:.2f}")
    print(f"  - Method: {result['method']}")
    
    assert result['type'] == 'lab'
    assert result['confidence'] > 0.5
    
    # Test with radiology text
    rad_text = """
    RADIOLOGY REPORT
    
    Examination: Chest X-Ray PA and Lateral
    
    CLINICAL INDICATION: Cough
    
    COMPARISON: None available
    
    FINDINGS:
    The lungs are clear without focal consolidation.
    
    IMPRESSION:
    Normal chest radiograph.
    """
    
    result2 = classifier._classify_by_fingerprint(rad_text)
    print(f"\n✓ Second classification (radiology):")
    print(f"  - Type: {result2['type']}")
    print(f"  - Confidence: {result2['confidence']:.2f}")
    
    assert result2['type'] == 'radiology'
    
    print("\n✅ Document Classifier test PASSED\n")


def test_document_fingerprinter():
    """Test document fingerprinting"""
    print("=" * 70)
    print("TEST: Document Fingerprinter")
    print("=" * 70)
    
    from src.medical_ingestion.classifiers.fingerprinting import DocumentFingerprinter
    
    fingerprinter = DocumentFingerprinter()
    print("✓ Document Fingerprinter initialized")
    
    lab_text = """
    Test                Result      Reference Range
    WBC                 7.2         4.5-11.0 K/uL
    Hemoglobin          14.2        13.5-17.5 g/dL
    """
    
    fingerprint = fingerprinter.analyze(lab_text)
    
    print(f"\n✓ Fingerprint analysis complete:")
    print(f"  - Has table structure: {fingerprint['has_table_structure']}")
    print(f"  - Layout type: {fingerprint['layout_type']}")
    print(f"  - Numeric density: {fingerprint['numeric_density']:.2f}")
    print(f"  - Structural hints: {fingerprint['structural_hints']}")
    
    assert fingerprint['has_table_structure'] == True
    assert 'lab' in fingerprint['structural_hints']
    
    print("\n✅ Document Fingerprinter test PASSED\n")


def test_schema_detector():
    """Test schema detection"""
    print("=" * 70)
    print("TEST: Schema Detector")
    print("=" * 70)
    
    from src.medical_ingestion.classifiers.schema_detector import SchemaDetector
    
    detector = SchemaDetector()
    print("✓ Schema Detector initialized")
    
    cbc_text = "Complete Blood Count WBC Hemoglobin Hematocrit Platelets"
    
    result = detector.detect_schema('lab', cbc_text)
    
    print(f"\n✓ Schema detection complete:")
    print(f"  - Schema: {result['schema']}")
    print(f"  - Confidence: {result['confidence']:.2f}")
    
    assert result['schema'] == 'cbc'
    assert result['confidence'] > 0.5
    
    print("\n✅ Schema Detector test PASSED\n")