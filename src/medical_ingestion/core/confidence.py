# ============================================================================
# src/medical_ingestion/core/confidence.py
# ============================================================================
"""
Confidence Scoring and Aggregation

Provides utilities for:
- Calculating confidence scores
- Aggregating multiple confidence values
- Determining confidence levels
- Managing confidence thresholds
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import statistics


class AggregationMethod(Enum):
    """Methods for aggregating multiple confidence scores"""
    MINIMUM = "minimum"  # Most conservative (lowest score)
    AVERAGE = "average"  # Mean of all scores
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted mean
    HARMONIC_MEAN = "harmonic_mean"  # Penalizes low scores
    PRODUCT = "product"  # Product of all scores


@dataclass
class ConfidenceThresholds:
    """Confidence level thresholds"""
    high: float = 0.85
    medium: float = 0.70
    low: float = 0.0

    def get_level(self, score: float) -> str:
        """
        Get confidence level from score.

        Args:
            score: Confidence score (0.0-1.0)

        Returns:
            Level string: "high", "medium", or "low"
        """
        if score >= self.high:
            return "high"
        elif score >= self.medium:
            return "medium"
        else:
            return "low"


class ConfidenceCalculator:
    """
    Utility class for calculating and aggregating confidence scores.
    """

    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        """
        Initialize calculator.

        Args:
            thresholds: Custom confidence thresholds
        """
        self.thresholds = thresholds or ConfidenceThresholds()

    def aggregate(
        self,
        scores: List[float],
        method: AggregationMethod = AggregationMethod.MINIMUM,
        weights: Optional[List[float]] = None,
    ) -> float:
        """
        Aggregate multiple confidence scores into single value.

        Args:
            scores: List of confidence scores (0.0-1.0)
            method: Aggregation method
            weights: Optional weights for weighted average

        Returns:
            Aggregated confidence score (0.0-1.0)
        """
        if not scores:
            return 0.0

        # Filter out invalid scores
        valid_scores = [s for s in scores if 0.0 <= s <= 1.0]
        if not valid_scores:
            return 0.0

        if method == AggregationMethod.MINIMUM:
            return min(valid_scores)

        elif method == AggregationMethod.AVERAGE:
            return statistics.mean(valid_scores)

        elif method == AggregationMethod.WEIGHTED_AVERAGE:
            if weights and len(weights) == len(valid_scores):
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_sum = sum(s * w for s, w in zip(valid_scores, weights))
                    return weighted_sum / total_weight
            # Fallback to regular average
            return statistics.mean(valid_scores)

        elif method == AggregationMethod.HARMONIC_MEAN:
            # Harmonic mean penalizes low scores more than arithmetic mean
            try:
                return statistics.harmonic_mean(valid_scores)
            except statistics.StatisticsError:
                return 0.0

        elif method == AggregationMethod.PRODUCT:
            # Product of scores (very conservative)
            result = 1.0
            for score in valid_scores:
                result *= score
            return result

        return 0.0

    def calculate_from_components(
        self,
        component_scores: Dict[str, float],
        component_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate overall confidence from component scores.

        Args:
            component_scores: Dict of component name -> score
            component_weights: Optional weights for each component

        Returns:
            Dict containing:
                - overall_score: Aggregated score
                - level: Confidence level
                - components: Individual component scores
                - method: Aggregation method used
        """
        scores = list(component_scores.values())

        if component_weights:
            weights = [component_weights.get(k, 1.0) for k in component_scores.keys()]
            overall_score = self.aggregate(scores, AggregationMethod.WEIGHTED_AVERAGE, weights)
            method = "weighted_average"
        else:
            overall_score = self.aggregate(scores, AggregationMethod.MINIMUM)
            method = "minimum"

        return {
            "overall_score": overall_score,
            "level": self.thresholds.get_level(overall_score),
            "components": component_scores,
            "method": method,
        }

    def calculate_extraction_confidence(
        self,
        template_match: float,
        extraction_quality: float,
        validation_result: Optional[bool] = None,
        conflict: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate confidence for a single extracted value.

        Args:
            template_match: Template matching confidence (0.0-1.0)
            extraction_quality: Extraction quality score (0.0-1.0)
            validation_result: Validation pass/fail (optional)
            conflict: Whether validators conflict

        Returns:
            Confidence assessment dict
        """
        # Base confidence from template match and extraction quality
        base_confidence = self.aggregate(
            [template_match, extraction_quality],
            AggregationMethod.HARMONIC_MEAN,
        )

        # Adjust based on validation
        final_confidence = base_confidence

        if conflict:
            # Major penalty for validation conflict
            final_confidence = min(base_confidence, 0.5)
            level = "conflict"
        elif validation_result is not None:
            if validation_result:
                # Boost confidence if validated
                final_confidence = min(base_confidence * 1.1, 1.0)
            else:
                # Penalty if validation failed
                final_confidence = base_confidence * 0.6

        level = self.thresholds.get_level(final_confidence) if not conflict else "conflict"

        return {
            "confidence": final_confidence,
            "level": level,
            "template_match": template_match,
            "extraction_quality": extraction_quality,
            "validated": validation_result,
            "conflict": conflict,
        }

    def calculate_document_confidence(
        self,
        classification_confidence: float,
        template_confidence: float,
        extraction_confidences: List[float],
        quality_score: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate overall document processing confidence.

        Args:
            classification_confidence: Document classification confidence
            template_confidence: Template matching confidence
            extraction_confidences: List of individual extraction confidences
            quality_score: Document quality score (0.0-1.0)

        Returns:
            Overall confidence assessment
        """
        # Base components
        components = {
            "classification": classification_confidence,
            "template_match": template_confidence,
            "quality": quality_score,
        }

        # Add extraction confidence
        if extraction_confidences:
            extraction_confidence = self.aggregate(
                extraction_confidences,
                AggregationMethod.AVERAGE,
            )
            components["extraction"] = extraction_confidence

        # Calculate overall with weights
        weights = {
            "classification": 1.0,
            "template_match": 1.5,  # Higher weight for template matching
            "extraction": 2.0,  # Highest weight for actual extraction
            "quality": 1.0,
        }

        result = self.calculate_from_components(components, weights)

        # Add statistics
        if extraction_confidences:
            result["extraction_stats"] = {
                "count": len(extraction_confidences),
                "min": min(extraction_confidences),
                "max": max(extraction_confidences),
                "mean": statistics.mean(extraction_confidences),
                "median": statistics.median(extraction_confidences),
            }

        return result

    def should_require_review(
        self,
        overall_confidence: float,
        has_conflicts: bool = False,
        has_critical_findings: bool = False,
        quality_flags: int = 0,
    ) -> Dict[str, Any]:
        """
        Determine if document requires human review.

        Args:
            overall_confidence: Overall confidence score
            has_conflicts: Whether there are validation conflicts
            has_critical_findings: Whether critical findings detected
            quality_flags: Number of quality flags

        Returns:
            Review decision dict
        """
        requires_review = False
        priority = "low"
        reasons = []

        # Check confidence level
        level = self.thresholds.get_level(overall_confidence)

        if level == "low":
            requires_review = True
            priority = "medium"
            reasons.append("low_confidence")

        # Check for conflicts
        if has_conflicts:
            requires_review = True
            priority = "high"
            reasons.append("validation_conflict")

        # Check for critical findings
        if has_critical_findings:
            requires_review = True
            priority = "critical"
            reasons.append("critical_findings")

        # Check quality flags
        if quality_flags > 0:
            requires_review = True
            if quality_flags >= 3:
                priority = "high"
            elif priority == "low":
                priority = "medium"
            reasons.append(f"quality_flags ({quality_flags})")

        return {
            "requires_review": requires_review,
            "priority": priority,
            "reasons": reasons,
            "confidence_level": level,
            "confidence_score": overall_confidence,
        }


def calculate_confidence(
    scores: List[float],
    method: str = "minimum",
    weights: Optional[List[float]] = None,
) -> float:
    """
    Convenience function for calculating confidence.

    Args:
        scores: List of confidence scores
        method: Aggregation method name
        weights: Optional weights

    Returns:
        Aggregated confidence score
    """
    calculator = ConfidenceCalculator()
    method_enum = AggregationMethod(method)
    return calculator.aggregate(scores, method_enum, weights)


def get_confidence_level(score: float) -> str:
    """
    Get confidence level from score.

    Args:
        score: Confidence score (0.0-1.0)

    Returns:
        Level string: "high", "medium", or "low"
    """
    thresholds = ConfidenceThresholds()
    return thresholds.get_level(score)


# Default calculator instance
default_calculator = ConfidenceCalculator()
