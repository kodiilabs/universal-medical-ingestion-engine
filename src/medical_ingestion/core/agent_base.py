# ============================================================================
# src/medical_ingestion/core/agent_base.py
# ============================================================================
"""
Abstract Base Agent Class

All 13+ agents in the system inherit from this base class.
Defines the standard interface and common functionality.

Every agent must implement:
- execute(context): Main processing logic
- get_name(): Agent identifier

Every agent gets:
- Logging
- Error handling
- Confidence scoring utilities
- Audit trail integration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from ..core.context.processing_context import ProcessingContext
from ..core.config import get_config


class Agent(ABC):
    """
    Abstract base class for all processing agents.

    Design principles:
    1. Single responsibility - each agent does ONE thing well
    2. Context-based communication - read from and write to shared context
    3. Confidence scoring - every decision includes confidence
    4. Audit trail - every action is logged
    5. Error resilience - failures don't crash entire pipeline
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize agent with configuration.

        Args:
            config: Configuration dictionary (passed config overrides env defaults)
        """
        # Merge env config with passed config (passed config takes precedence)
        env_config = get_config()
        self.config = {**env_config, **(config or {})}
        self.logger = logging.getLogger(f"{__name__}.{self.get_name()}")
        self._execution_count = 0
        self._total_duration = 0.0
    
    @abstractmethod
    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Main agent execution logic.
        
        This is the core method every agent must implement.
        
        Args:
            context: Shared processing context (read and modify)
            
        Returns:
            Dict containing:
                - decision: Agent's decision/output
                - confidence: Confidence score (0.0-1.0)
                - reasoning: Human-readable explanation
                - metadata: Any additional data
        
        Example return:
            {
                "decision": "use_template",
                "confidence": 0.95,
                "reasoning": "Strong match to Quest CBC format",
                "metadata": {"template_id": "quest_cbc_v1"}
            }
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return agent name for logging and audit trail.
        
        Example: "TemplateMatchingAgent"
        """
        pass
    
    async def run(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Wrapper around execute() that handles logging, timing, and errors.
        
        This method is called by orchestrators, not execute() directly.
        """
        agent_name = self.get_name()
        start_time = datetime.now()
        
        self.logger.info(f"Executing {agent_name}")
        
        try:
            # Execute agent logic
            result = await self.execute(context)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self._execution_count += 1
            self._total_duration += duration
            
            # Log to audit trail
            context.log_agent_execution(
                agent_name=agent_name,
                decision={
                    **result,
                    "duration_seconds": duration,
                    "execution_number": self._execution_count
                }
            )
            
            self.logger.info(
                f"{agent_name} completed in {duration:.2f}s "
                f"(confidence: {result.get('confidence', 0.0):.2f})"
            )
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(f"{agent_name} failed: {str(e)}", exc_info=True)
            
            # Log failure to audit trail
            context.log_agent_execution(
                agent_name=agent_name,
                decision={
                    "error": str(e),
                    "duration_seconds": duration,
                    "execution_number": self._execution_count
                }
            )
            
            # Add warning to context
            context.add_warning(f"{agent_name} failed: {str(e)}")
            
            # Return error result
            return {
                "decision": "error",
                "confidence": 0.0,
                "reasoning": f"Agent execution failed: {str(e)}",
                "error": str(e)
            }
    
    # ========================================================================
    # HELPER METHODS (Available to all agents)
    # ========================================================================
    
    def calculate_confidence(
        self, 
        signals: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted confidence score from multiple signals.
        
        Args:
            signals: Dict of signal_name -> score (0.0-1.0)
            weights: Optional weights (defaults to equal weighting)
            
        Returns:
            Combined confidence score (0.0-1.0)
            
        Example:
            signals = {
                "template_match": 0.95,
                "field_coverage": 0.88,
                "vendor_markers": 0.90
            }
            confidence = self.calculate_confidence(signals)
        """
        if not signals:
            return 0.0
        
        if weights is None:
            # Equal weighting
            weights = {k: 1.0 / len(signals) for k in signals.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        score = sum(signals[k] * normalized_weights.get(k, 0.0) for k in signals.keys())
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def meets_threshold(self, confidence: float, threshold_name: str = "default") -> bool:
        """
        Check if confidence meets configured threshold.
        
        Args:
            confidence: Confidence score to check
            threshold_name: Which threshold to use (from config)
            
        Returns:
            True if confidence >= threshold
        """
        from ..config.thresholds_config import threshold_settings
        
        thresholds = {
            "default": threshold_settings.VALIDATION_CONFIDENCE_THRESHOLD,
            "template": threshold_settings.TEMPLATE_MATCH_THRESHOLD,
            "human_review": threshold_settings.HUMAN_REVIEW_THRESHOLD,
            "classification": threshold_settings.CLASSIFICATION_CONFIDENCE_THRESHOLD,
        }
        
        threshold = thresholds.get(threshold_name, 0.85)
        return confidence >= threshold
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Returns:
            Dict with execution count, total time, average time
        """
        avg_duration = (
            self._total_duration / self._execution_count 
            if self._execution_count > 0 
            else 0.0
        )
        
        return {
            "agent_name": self.get_name(),
            "execution_count": self._execution_count,
            "total_duration_seconds": self._total_duration,
            "average_duration_seconds": avg_duration,
        }