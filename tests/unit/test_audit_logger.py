# ============================================================================
# TEST 7: Audit Logger
# ============================================================================

def test_audit_logger():
    """Test audit trail logging"""
    print("=" * 70)
    print("TEST 7: Audit Logger")
    print("=" * 70)
    
    from pathlib import Path
    from src.medical_ingestion.core.audit import AuditLogger
    from src.medical_ingestion.core.context.processing_context import ProcessingContext
    
    # Use test database
    test_db = Path("data/test_audit.db")
    test_db.parent.mkdir(parents=True, exist_ok=True)
    
    audit = AuditLogger(db_path=test_db)
    print(f"✓ Audit Logger initialized")
    print(f"  - Database: {test_db}")
    
    # Create test context
    ctx = ProcessingContext(
        document_path=Path("test.pdf")
    )
    ctx.document_type = "lab"
    ctx.overall_confidence = 0.92
    ctx.processing_duration = 3.5
    
    # Log processing complete
    audit.log_processing_complete(ctx)
    print(f"\n✓ Logged processing completion")
    
    # Retrieve audit trail
    trail = audit.get_trail(ctx.document_id)
    print(f"\n✓ Retrieved audit trail: {len(trail)} entries")
    
    # Clean up test database
    if test_db.exists():
        test_db.unlink()
        print(f"\n✓ Cleaned up test database")
    
    print("\n✅ Audit Logger test PASSED\n")

