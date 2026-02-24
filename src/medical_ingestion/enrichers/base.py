# ============================================================================
# src/medical_ingestion/enrichers/base.py
# ============================================================================
"""
Base Enricher Interface

Enrichers take GenericMedicalExtraction and add type-specific metadata.
They do NOT re-extract data - they enhance already-extracted data with
domain-specific codes, validations, and additional context.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class EnrichedExtraction:
    """
    Enriched extraction result.

    Contains the original extraction plus type-specific enhancements.
    """
    # Original extraction (reference)
    original_extraction: Any = None

    # Enrichment metadata
    enrichment_type: str = ""  # "lab", "prescription", "radiology", "pathology"
    enrichment_timestamp: Optional[datetime] = None
    enrichment_confidence: float = 0.0

    # Enrichment results (type-specific)
    enrichments: Dict[str, Any] = field(default_factory=dict)

    # Validation results
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # Additional flags
    requires_review: bool = False
    review_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enrichment_type": self.enrichment_type,
            "enrichment_timestamp": (
                self.enrichment_timestamp.isoformat()
                if self.enrichment_timestamp else None
            ),
            "enrichment_confidence": self.enrichment_confidence,
            "enrichments": self.enrichments,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "requires_review": self.requires_review,
            "review_reasons": self.review_reasons
        }


class TypeSpecificEnricher(ABC):
    """
    Base class for type-specific enrichers.

    Subclasses implement domain-specific enrichment logic:
    - LabEnricher: LOINC codes, reference ranges
    - PrescriptionEnricher: RxNorm codes, drug interactions
    - RadiologyEnricher: ICD-10 codes, critical findings
    - PathologyEnricher: ICD-O codes, staging
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @property
    @abstractmethod
    def enricher_type(self) -> str:
        """Return the type of enricher (e.g., 'lab', 'prescription')."""
        pass

    @abstractmethod
    async def enrich(self, extraction: Any) -> EnrichedExtraction:
        """
        Enrich the extraction with type-specific metadata.

        Args:
            extraction: GenericMedicalExtraction from ContentAgnosticExtractor

        Returns:
            EnrichedExtraction with added metadata
        """
        pass

    def _create_result(
        self,
        extraction: Any,
        enrichments: Dict[str, Any] = None,
        confidence: float = 0.0
    ) -> EnrichedExtraction:
        """Helper to create EnrichedExtraction result."""
        return EnrichedExtraction(
            original_extraction=extraction,
            enrichment_type=self.enricher_type,
            enrichment_timestamp=datetime.now(),
            enrichment_confidence=confidence,
            enrichments=enrichments or {}
        )
