# ============================================================================
# src/medical_ingestion/constants/document_types.py
# ============================================================================
"""
Document Types and Processor Mappings
- Supported document types for classification/routing
- Maps document type → processor
"""

from enum import Enum

class DocumentType(str, Enum):
    """
    Supported document types that can be classified and routed.
    Each type routes to a specific processor.
    """
    LAB = "lab"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    PRESCRIPTION = "prescription"
    DISCHARGE_SUMMARY = "discharge_summary"
    OPERATIVE_NOTE = "operative_note"
    UNKNOWN = "unknown"

# Processor mapping must match processor registration
PROCESSOR_MAPPING = {
    DocumentType.LAB: "lab",
    DocumentType.RADIOLOGY: "radiology",
    DocumentType.PATHOLOGY: "pathology",
    DocumentType.PRESCRIPTION: "prescription",
    DocumentType.UNKNOWN: "fallback"
}
