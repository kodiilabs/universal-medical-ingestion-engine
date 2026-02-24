# ============================================================================
# src/medical_ingestion/processors/lab/agents/orchestrator.py
# ============================================================================
"""
Lab Orchestration Agent - Final Assembly

Aggregates results from all previous agents:
1. Calculate overall confidence
2. Build priority review queue
3. Determine review requirements
4. Create final processing summary

This is the "conductor" that brings everything together.
"""

from typing import Dict, Any
import logging

from ....core.agent_base import Agent
from ....core.context import ProcessingContext, ReviewPriority
from ....config import threshold_settings


class LabOrchestrationAgent(Agent):
    """
    Final orchestration for lab processing pipeline.
    
    Responsibilities:
    - Aggregate confidence scores across all agents
    - Prioritize items for human review
    - Set final review requirements
    - Calculate processing quality metrics
    
    This agent makes the final decision on review needs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_name(self) -> str:
        return "LabOrchestrationAgent"
    
    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Orchestrate final processing decisions.
        
        Returns:
            {
                "decision": "processing_complete",
                "confidence": float,
                "reasoning": str,
                "review_required": bool,
                "review_priority": str
            }
        """
        # Calculate overall confidence from all stages
        overall_confidence = self._calculate_overall_confidence(context)
        context.overall_confidence = overall_confidence
        
        # Determine confidence level
        context.calculate_confidence_level()
        
        # Build review queue priorities
        self._prioritize_review_items(context)
        
        # Set final review requirements
        self._finalize_review_requirements(context)
        
        return {
            "decision": "processing_complete",
            "confidence": overall_confidence,
            "reasoning": self._build_reasoning(context),
            "review_required": context.requires_review,
            "review_priority": context.review_priority.value if context.review_priority else "none"
        }
    
    def _calculate_overall_confidence(self, context: ProcessingContext) -> float:
        """
        Calculate overall processing confidence.
        
        Uses minimum confidence across all stages (conservative approach).
        Also factors in warnings, conflicts, and quality issues.
        """
        # Start with confidence scores from each stage
        stage_confidences = list(context.confidence_scores.values())
        
        if not stage_confidences:
            return 0.5  # Default if no scores
        
        # Use minimum (most conservative)
        base_confidence = min(stage_confidences)
        
        # Apply penalties
        confidence = base_confidence
        
        # Penalty for validation conflicts
        conflict_count = sum(
            1 for v in context.extracted_values 
            if v.validation_conflict
        )
        confidence -= conflict_count * 0.1
        
        # Penalty for quality flags
        confidence -= len(context.quality_flags) * 0.05
        
        # Penalty for warnings
        confidence -= len(context.warnings) * 0.02
        
        # Penalty for specimen rejection
        if context.specimen_rejected:
            confidence = min(confidence, 0.3)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def _prioritize_review_items(self, context: ProcessingContext):
        """
        Prioritize which items need review most urgently.
        
        Priority order:
        1. Critical findings (life-threatening)
        2. Specimen rejections (quality issues)
        3. Validation conflicts (rules vs AI disagree)
        4. Low confidence extractions
        5. Abnormal values
        """
        # Build priority score for each extracted value
        for value in context.extracted_values:
            priority_score = 0
            
            # Critical values get highest priority
            if value.abnormal_flag in ['CRITICAL', 'HH', 'LL']:
                priority_score += 100
            
            # Validation conflicts
            if value.validation_conflict:
                priority_score += 50
            
            # Low confidence
            if value.confidence < threshold_settings.HUMAN_REVIEW_THRESHOLD:
                priority_score += 30
            
            # Quality warnings
            if value.warnings:
                priority_score += 20 * len(value.warnings)
            
            # Abnormal flags
            if value.abnormal_flag in ['H', 'L']:
                priority_score += 10
            
            # Store priority score in warnings (for later sorting)
            if priority_score > 0:
                value.warnings.insert(0, f"Priority score: {priority_score}")
    
    def _finalize_review_requirements(self, context: ProcessingContext):
        """
        Make final decision on review requirements.
        
        Escalation triggers:
        - Specimen rejected
        - Critical findings
        - Low overall confidence
        - Validation conflicts
        - Quality flags
        """
        if context.specimen_rejected:
            context.requires_review = True
            context.review_priority = ReviewPriority.CRITICAL
            if "Specimen rejected" not in context.review_reasons:
                context.review_reasons.append("Specimen rejected")
        
        elif context.critical_findings:
            context.requires_review = True
            context.review_priority = ReviewPriority.CRITICAL
            if f"{len(context.critical_findings)} critical findings" not in context.review_reasons:
                context.review_reasons.append(
                    f"{len(context.critical_findings)} critical findings"
                )
        
        elif context.overall_confidence < threshold_settings.HUMAN_REVIEW_THRESHOLD:
            context.requires_review = True
            context.review_priority = ReviewPriority.HIGH
            if f"Low confidence: {context.overall_confidence:.2f}" not in context.review_reasons:
                context.review_reasons.append(
                    f"Low confidence: {context.overall_confidence:.2f}"
                )
        
        elif any(v.validation_conflict for v in context.extracted_values):
            context.requires_review = True
            context.review_priority = ReviewPriority.HIGH
            if "Validation conflicts detected" not in context.review_reasons:
                context.review_reasons.append("Validation conflicts detected")
        
        elif context.quality_flags:
            context.requires_review = True
            if context.review_priority == ReviewPriority.LOW:
                context.review_priority = ReviewPriority.MEDIUM
            if "Specimen quality flags" not in context.review_reasons:
                context.review_reasons.append("Specimen quality flags")
    
    def _build_reasoning(self, context: ProcessingContext) -> str:
        """Build human-readable reasoning summary"""
        parts = []
        
        parts.append(
            f"Processed {len(context.extracted_values)} lab values"
        )
        
        if context.specimen_rejected:
            parts.append(f"SPECIMEN REJECTED: {context.rejection_reason}")
        
        if context.temporal_trends:
            parts.append(f"{len(context.temporal_trends)} temporal trends detected")
        
        if context.reflex_recommendations:
            parts.append(
                f"{len(context.reflex_recommendations)} reflex tests recommended"
            )
        
        parts.append(f"Overall confidence: {context.overall_confidence:.2f}")
        
        if context.requires_review:
            parts.append(
                f"Review required ({context.review_priority.value} priority)"
            )
        
        return "; ".join(parts)