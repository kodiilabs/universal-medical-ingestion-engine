# src/medical_ingestion/processors/fallback/__init__.py
"""
Fallback Processor - Universal handler for unknown document types.
"""

from .processor import FallbackProcessor

__all__ = ["FallbackProcessor"]
