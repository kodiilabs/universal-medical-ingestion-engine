# ============================================================================
# src/medical_ingestion/enrichers/__init__.py
# ============================================================================
"""
Type-Specific Enrichers Package

Enrichers add type-specific metadata to already-extracted data:
- LabEnricher: LOINC codes, reference ranges, abnormal flags
- PrescriptionEnricher: RxNorm codes, drug interactions
- RadiologyEnricher: ICD-10 codes, critical finding detection
- PathologyEnricher: ICD-O codes, staging info

These operate AFTER content-agnostic extraction, using classification
to determine which enricher to apply.
"""

from .base import TypeSpecificEnricher, EnrichedExtraction
from .lab_enricher import LabEnricher
from .prescription_enricher import PrescriptionEnricher
from .radiology_enricher import RadiologyEnricher
from .pathology_enricher import PathologyEnricher

__all__ = [
    "TypeSpecificEnricher",
    "EnrichedExtraction",
    "LabEnricher",
    "PrescriptionEnricher",
    "RadiologyEnricher",
    "PathologyEnricher",
]
