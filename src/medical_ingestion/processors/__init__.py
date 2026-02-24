# src/medical_ingestion/processors/__init__.py
"""
Document Processors Module

Contains specialized processors for different document types:
- Lab reports (LabProcessor)
- Radiology reports (RadiologyProcessor)
- Prescriptions (PrescriptionProcessor)
- Fallback for unknown types (FallbackProcessor)

Also includes auto-template generation for learning new formats.
"""

from .base_processor import BaseProcessor
from .lab import LabProcessor
from .radiology import RadiologyProcessor
from .prescription import PrescriptionProcessor
from .fallback import FallbackProcessor
from .template_generator import TemplateGenerator, TemplateApprovalManager

__all__ = [
    "BaseProcessor",
    "LabProcessor",
    "RadiologyProcessor",
    "PrescriptionProcessor",
    "FallbackProcessor",
    "TemplateGenerator",
    "TemplateApprovalManager",
]
