# ============================================================================
# FILE: tests/unit/test_orchestrator.py
# ============================================================================
"""
Unit tests for orchestrator
"""

def test_orchestrator():
    """Test orchestrator functionality"""
    print("=" * 70)
    print("TEST: Orchestrator")
    print("=" * 70)
    
    from src.medical_ingestion.core.orchestrator import UniversalOrchestrator
    
    orchestrator = UniversalOrchestrator({})
    print("✓ Orchestrator initialized")
    
    # Test processor selection
    processor = orchestrator._select_processor('lab')
    assert processor == 'lab'
    print("✓ Lab processor selected")
    
    processor = orchestrator._select_processor('radiology')
    assert processor == 'radiology'
    print("✓ Radiology processor selected")
    
    processor = orchestrator._select_processor('unknown')
    assert processor == 'fallback'
    print("✓ Fallback processor selected for unknown type")
    
    print("\n✅ Orchestrator test PASSED\n")


def test_review_assessment():
    """Test review assessment logic"""
    print("=" * 70)
    print("TEST: Review Assessment")
    print("=" * 70)
    
    from src.medical_ingestion.core.orchestrator import UniversalOrchestrator
    from src.medical_ingestion.core.context.processing_context import ProcessingContext
    from pathlib import Path
    
    orchestrator = UniversalOrchestrator({})
    context = ProcessingContext(document_path=Path("test.pdf"))
    
    # Test low confidence triggers review
    context.overall_confidence = 0.6
    orchestrator._assess_review_needs(context)
    
    assert context.requires_review == True
    print("✓ Low confidence triggers review")
    
    # Test specimen rejection
    context2 = ProcessingContext(document_path=Path("test2.pdf"))
    context2.specimen_rejected = True
    context2.rejection_reason = "Hemolysis"
    orchestrator._assess_review_needs(context2)
    
    assert context2.requires_review == True
    assert context2.review_priority.value == 'critical'
    print("✓ Specimen rejection triggers critical review")
    
    print("\n✅ Review Assessment test PASSED\n")