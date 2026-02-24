# ============================================================================
# src/medical_ingestion/core/context.py
# ============================================================================
"""
Processing Context Module

Main exports for document processing context and related utilities.

This module provides:
- ProcessingContext: Main context object passed between agents
- ExtractedValue: Individual field extraction representation
- ConfidenceLevel: Confidence level enumeration
- ReviewPriority: Review priority enumeration
"""

from .context.processing_context import ProcessingContext
from .context.extracted_value import ExtractedValue
from .context.enums import ConfidenceLevel, ReviewPriority

__all__ = [
    "ProcessingContext",
    "ExtractedValue",
    "ConfidenceLevel",
    "ReviewPriority",
]
