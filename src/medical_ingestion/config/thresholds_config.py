# ============================================================================
# src/medical_ingestion/config/thresholds.py
# ============================================================================
"""
Confidence Thresholds
- Template matching
- Classification
- Validation
- Human review escalation
- Conflict detection
"""

from pydantic import Field
from pydantic_settings import BaseSettings

class ThresholdSettings(BaseSettings):
    TEMPLATE_MATCH_THRESHOLD: float = Field(
        default=0.85,
        ge=0.0, le=1.0,
        description="Minimum confidence to use template extraction (vs MedGemma). Lowered from 0.90 since visual fallback provides safety net."
    )
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = Field(
        default=0.85,
        ge=0.0, le=1.0,
        description="Minimum confidence for document classification"
    )
    VALIDATION_CONFIDENCE_THRESHOLD: float = Field(
        default=0.85,
        ge=0.0, le=1.0,
        description="Minimum confidence for validation acceptance"
    )
    HUMAN_REVIEW_THRESHOLD: float = Field(
        default=0.70,
        ge=0.0, le=1.0,
        description="Below this confidence, escalate to human review"
    )
    CONFLICT_THRESHOLD: float = Field(
        default=0.15,
        ge=0.0, le=1.0,
        description="Difference threshold to flag validation conflict"
    )
    MIN_CLASSIFICATION_TO_PROCEED: float = Field(
        default=0.40,
        ge=0.0, le=1.0,
        description="Minimum classification confidence to proceed with extraction. Below this, fail early - don't extract potentially wrong data."
    )

threshold_settings = ThresholdSettings()
