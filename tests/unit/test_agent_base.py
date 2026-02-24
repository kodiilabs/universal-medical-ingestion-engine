# ============================================================================
# TEST 8: Agent Base Class
# ============================================================================

def test_agent_base():
    """Test agent base class functionality"""
    print("=" * 70)
    print("TEST 8: Agent Base Class")
    print("=" * 70)
    
    from src.medical_ingestion.core.agent_base import Agent
    from src.medical_ingestion.core.context.processing_context import ProcessingContext
    
    # Create a simple test agent
    class TestAgent(Agent):
        def get_name(self):
            return "TestAgent"
        
        async def execute(self, context):
            return {
                "decision": "test_decision",
                "confidence": 0.95,
                "reasoning": "This is a test"
            }
    
    agent = TestAgent({})
    print(f"✓ Test agent created: {agent.get_name()}")
    
    # Test confidence calculation
    signals = {
        "signal1": 0.9,
        "signal2": 0.85,
        "signal3": 0.95
    }
    confidence = agent.calculate_confidence(signals)
    print(f"\n✓ Confidence calculation: {confidence:.2f}")
    
    # Test threshold checking
    meets_threshold = agent.meets_threshold(0.92, "template")
    print(f"✓ Threshold check (0.92 vs template): {meets_threshold}")
    
    # Test metrics
    metrics = agent.get_metrics()
    print(f"\n✓ Agent metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
    
    print("\n✅ Agent Base Class test PASSED\n")

