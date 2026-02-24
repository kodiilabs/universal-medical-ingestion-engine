# ============================================================================
# src/medical_ingestion/config/clinical.py
# ============================================================================
"""
Clinical Validation Settings
- Dual validation
- Specimen quality
- Temporal analysis
- Reflex testing
"""

from pydantic import Field
from pydantic_settings import BaseSettings

class ClinicalSettings(BaseSettings):
    ENABLE_DUAL_VALIDATION: bool = Field(
        default=True,
        description="Use both rule-based and AI validation"
    )
    ENABLE_SPECIMEN_QUALITY: bool = Field(
        default=True,
        description="Run specimen quality analysis (hemolysis, contamination)"
    )
    ENABLE_TEMPORAL_ANALYSIS: bool = Field(
        default=True,
        description="Analyze trends across patient history"
    )
    ENABLE_REFLEX_PROTOCOLS: bool = Field(
        default=True,
        description="Generate reflex test recommendations"
    )
    TEMPORAL_LOOKBACK_DAYS: int = Field(
        default=365,
        description="Maximum days of history for temporal analysis"
    )

clinical_settings = ClinicalSettings()
