# ============================================================================
# ui/services/__init__.py
# ============================================================================
"""
UI Service Layer

Connects Streamlit UI to the core medical ingestion engine.
"""

from .processing_service import ProcessingService
from .audit_service import AuditService

__all__ = ['ProcessingService', 'AuditService']
