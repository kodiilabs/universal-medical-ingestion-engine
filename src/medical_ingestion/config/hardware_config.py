# ============================================================================
# src/medical_ingestion/config/hardware.py
# ============================================================================
"""
Hardware & Performance Settings
- GPU / CPU usage
- Concurrency
- Memory limits
"""

from pydantic import Field
from pydantic_settings import BaseSettings

class HardwareSettings(BaseSettings):
    USE_GPU: bool = Field(
        default=True,
        description="Use GPU for MedGemma inference if available"
    )
    FORCE_CPU: bool = Field(
        default=False,
        description="Force CPU usage even if GPU detected"
    )
    MAX_CONCURRENT_DOCS: int = Field(
        default=10,
        description="Maximum documents processed simultaneously"
    )
    MAX_MEMORY_GB: int = Field(
        default=16,
        description="Maximum memory for MedGemma model (GB)"
    )

hardware_settings = HardwareSettings()
