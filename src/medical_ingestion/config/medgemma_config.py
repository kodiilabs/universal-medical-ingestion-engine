# ============================================================================
# src/medical_ingestion/config/medgemma.py
# ============================================================================
"""
MedGemma Configuration (Local Inference)
- Max tokens
- Sampling temperature
- Response cache
- Timeout
"""

from pydantic import Field
from pydantic_settings import BaseSettings

class MedGemmaSettings(BaseSettings):
    MEDGEMMA_MAX_TOKENS: int = Field(
        default=1000,
        description="Maximum tokens for MedGemma generation"
    )
    MEDGEMMA_TEMPERATURE: float = Field(
        default=0.1,
        description="Temperature for MedGemma sampling (0.1 = very deterministic)"
    )
    MEDGEMMA_USE_CACHE: bool = Field(
        default=True,
        description="Cache MedGemma responses for repeated prompts"
    )
    MEDGEMMA_TIMEOUT: int = Field(
        default=30,
        description="Maximum time for MedGemma inference (seconds)"
    )

medgemma_settings = MedGemmaSettings()
