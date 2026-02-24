# ============================================================================
# FILE: src/validators/conflict_resolver.py
# ============================================================================
"""
Conflict Resolver

When rule-based and AI validators disagree, resolve the conflict.

Strategies:
1. Confidence-based (trust higher confidence)
2. Human review flag (when confidence close)
3. Historical data (if available)
4. Consensus voting (if multiple validators)
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_ingestion.core.context.extracted_value import ExtractedValue
from medical_ingestion.core.context.processing_context import ProcessingContext


logger = logging.getLogger(__name__)


class ConflictResolution(Enum):
    """Possible conflict resolution outcomes"""
    ACCEPT_RULE = "accept_rule"
    ACCEPT_AI = "accept_ai"
    FLAG_HUMAN_REVIEW = "flag_human_review"
    REJECT_VALUE = "reject_value"


class ConflictResolver:
    """
    Resolve validation conflicts between different validators.

    Conflict scenarios:
    1. Rule says valid, AI says invalid
    2. Rule says invalid, AI says valid
    3. Multiple AIs disagree
    4. Borderline values (low confidence)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Thresholds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.human_review_threshold = self.config.get('human_review_threshold', 0.85)

    def resolve(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext
    ) -> Tuple[ConflictResolution, str]:
        """
        Resolve conflict for a single value.

        Args:
            extracted: Value with potential conflict
            context: Processing context

        Returns:
            (resolution_decision, reasoning)
        """
        # No conflict if validators agree
        if extracted.rule_validation == extracted.ai_validation:
            return self._handle_agreement(extracted)

        # Conflict exists - resolve
        logger.warning(
            f"Validation conflict for {extracted.field_name}: "
            f"rule={extracted.rule_validation}, AI={extracted.ai_validation}"
        )

        return self._resolve_conflict(extracted, context)

    def _handle_agreement(
        self,
        extracted: ExtractedValue
    ) -> Tuple[ConflictResolution, str]:
        """
        Handle case where validators agree.
        """
        if extracted.rule_validation is True:
            # Both agree it's valid
            return ConflictResolution.ACCEPT_RULE, "Both validators agree - valid"
        else:
            # Both agree it's invalid
            return ConflictResolution.REJECT_VALUE, "Both validators agree - invalid"

    def _resolve_conflict(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext
    ) -> Tuple[ConflictResolution, str]:
        """
        Resolve disagreement between validators.

        Priority:
        1. If confidence very low → human review
        2. If rule fails plausibility → trust rule (obvious error)
        3. If AI confidence high → trust AI (clinical reasoning)
        4. Otherwise → flag for human review
        """
        # Get confidence score
        confidence = extracted.confidence

        # Strategy 1: Very low confidence → always human review
        if confidence < self.confidence_threshold:
            extracted.warnings.append("Low confidence - flagged for human review")
            return (
                ConflictResolution.FLAG_HUMAN_REVIEW,
                f"Low confidence ({confidence:.2f}) - needs human review"
            )

        # Strategy 2: Rule says implausible → trust rule
        # (Rules catch obvious errors like decimal points)
        if extracted.rule_validation is False:
            # Check if it's a plausibility failure
            if any("Plausibility" in w for w in extracted.warnings):
                return (
                    ConflictResolution.ACCEPT_RULE,
                    "Rule-based plausibility check failed - likely data error"
                )

        # Strategy 3: AI says invalid with high confidence → trust AI
        # (AI catches clinical implausibility)
        if extracted.ai_validation is False and confidence > self.human_review_threshold:
            return (
                ConflictResolution.ACCEPT_AI,
                "AI validation failed with high confidence - clinical implausibility"
            )

        # Strategy 4: Rule says invalid, AI says valid → AI wins if confident
        # (Clinical context might explain unusual value)
        if extracted.rule_validation is False and extracted.ai_validation is True:
            if confidence > self.human_review_threshold:
                extracted.warnings.append("Unusual value but clinically plausible per AI")
                return (
                    ConflictResolution.ACCEPT_AI,
                    "AI reasoning overrides rule-based rejection"
                )

        # Strategy 5: Can't resolve confidently → human review
        extracted.warnings.append("Validation conflict - flagged for human review")
        return (
            ConflictResolution.FLAG_HUMAN_REVIEW,
            f"Unresolved conflict (confidence={confidence:.2f})"
        )

    def batch_resolve(
        self,
        context: ProcessingContext
    ) -> Dict[str, Dict[str, Any]]:
        """
        Resolve conflicts for all extracted values.

        Returns:
            {field_name: {
                "resolution": ConflictResolution,
                "reasoning": str,
                "needs_review": bool
            }}
        """
        results = {}

        for extracted in context.extracted_values:
            resolution, reasoning = self.resolve(extracted, context)

            needs_review = resolution == ConflictResolution.FLAG_HUMAN_REVIEW

            results[extracted.field_name] = {
                "resolution": resolution.value,
                "reasoning": reasoning,
                "needs_review": needs_review
            }

            # Update validation conflict flag
            if extracted.rule_validation != extracted.ai_validation:
                extracted.validation_conflict = True

        return results

    def get_summary(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Get summary of conflicts and resolutions.

        Returns:
            {
                "total_values": int,
                "conflicts": int,
                "needs_review": int,
                "accepted": int,
                "rejected": int
            }
        """
        total = len(context.extracted_values)
        conflicts = 0
        needs_review = 0
        accepted = 0
        rejected = 0

        for extracted in context.extracted_values:
            if extracted.rule_validation != extracted.ai_validation:
                conflicts += 1

            resolution, _ = self.resolve(extracted, context)

            if resolution == ConflictResolution.FLAG_HUMAN_REVIEW:
                needs_review += 1
            elif resolution in [ConflictResolution.ACCEPT_RULE, ConflictResolution.ACCEPT_AI]:
                accepted += 1
            elif resolution == ConflictResolution.REJECT_VALUE:
                rejected += 1

        return {
            "total_values": total,
            "conflicts": conflicts,
            "needs_review": needs_review,
            "accepted": accepted,
            "rejected": rejected
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def resolve_conflict(
    extracted: ExtractedValue,
    context: ProcessingContext
) -> bool:
    """
    Quick conflict resolution.

    Returns:
        True if value should be accepted, False if rejected
    """
    resolver = ConflictResolver()
    resolution, _ = resolver.resolve(extracted, context)

    return resolution in [
        ConflictResolution.ACCEPT_RULE,
        ConflictResolution.ACCEPT_AI,
        ConflictResolution.FLAG_HUMAN_REVIEW
    ]
