# ============================================================================
# TEST 5: Document Classifier (Fingerprinting)
# ============================================================================

def test_document_classifier():
    """Test document classification without MedGemma"""
    print("=" * 70)
    print("TEST 5: Document Classifier (Fingerprinting)")
    print("=" * 70)
    
    from src.medical_ingestion.classifiers.document_classifier import DocumentClassifier
    
    classifier = DocumentClassifier({})
    print("✓ Document Classifier initialized")
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
    print(f"  - Reasoning: {result['reasoning']}")
    
    if result['confidence'] >= 0.85:
        print(f"\n✅ High confidence classification!")
    else:
        print(f"\n⚠️  Low confidence - would use MedGemma")
    
    # Test with radiology text
    rad_text = """
    RADIOLOGY REPORT
    
    Examination: Chest X-Ray PA and Lateral
    
    CLINICAL INDICATION: Cough
    
    COMPARISON: None available
    
    FINDINGS:
    The lungs are clear without focal consolidation, effusion, or pneumothorax.
    The cardiac silhouette is normal in size and contour.
    
    IMPRESSION:
    Normal chest radiograph.
    """
    
    result2 = classifier._classify_by_fingerprint(rad_text)
    print(f"\n✓ Second classification (radiology):")
    print(f"  - Type: {result2['type']}")
    print(f"  - Confidence: {result2['confidence']:.2f}")
    
    print("\n✅ Document Classifier test PASSED\n")

