# ============================================================================
# src/medical_ingestion/config/logging.py
# ============================================================================
"""
Logging & Monitoring Settings
- Log level
- Audit trail
- Performance metrics
"""

from pydantic import Field
from pydantic_settings import BaseSettings

class LoggingSettings(BaseSettings):
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    ENABLE_AUDIT_TRAIL: bool = Field(
        default=True,
        description="Enable complete audit trail logging"
    )
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable performance metric collection"
    )

logging_settings = LoggingSettings()
