# ============================================================================
# src/medical_ingestion/config/fhir.py
# ============================================================================
"""
FHIR Output Settings
- Version
- Validation
- Provenance
- Confidence scores
"""

from pydantic import Field
from pydantic_settings import BaseSettings

class FHIRSettings(BaseSettings):
    FHIR_VERSION: str = Field(
        default="R4",
        description="FHIR specification version"
    )
    FHIR_VALIDATE: bool = Field(
        default=True,
        description="Validate FHIR bundles against official schema"
    )
    FHIR_INCLUDE_PROVENANCE: bool = Field(
        default=True,
        description="Include provenance (source locations) in FHIR"
    )
    FHIR_INCLUDE_CONFIDENCE: bool = Field(
        default=True,
        description="Include confidence scores in FHIR extensions"
    )

fhir_settings = FHIRSettings()
