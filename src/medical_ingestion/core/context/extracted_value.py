# ============================================================================
# src/medical_ingestion/core/context/extracted_value.py
# ============================================================================
"""
Single extracted field representation
- Lab result, finding, or diagnosis
- Stores provenance, confidence, validation, warnings
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple

@dataclass
class ExtractedValue:
    field_name: str
    value: Any
    unit: Optional[str] = None
    confidence: float = 0.0
    extraction_method: str = "unknown"  # "template", "medgemma", etc.

    # Provenance
    source_page: Optional[int] = None
    source_location: Optional[str] = None
    source_text: Optional[str] = None

    # Bounding box for UI highlighting (x0, y0, x1, y1) in PDF coordinates
    # Coordinates are relative to page size (0-1 normalized) or absolute pixels
    bbox: Optional[Tuple[float, float, float, float]] = None
    bbox_normalized: bool = True  # True if bbox is 0-1 normalized, False if absolute

    # Document ordering - used to maintain original document order
    source_row_index: Optional[int] = None  # Row index in extracted table (0-based)

    # Validation
    rule_validation: Optional[bool] = None
    ai_validation: Optional[bool] = None
    validation_conflict: bool = False

    # Reference range
    reference_min: Optional[float] = None
    reference_max: Optional[float] = None
    abnormal_flag: Optional[str] = None  # "H", "L", "CRITICAL"

    # Warnings
    warnings: List[str] = field(default_factory=list)
