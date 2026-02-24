# src/medical_ingestion/extractors/__init__.py
"""
PDF Extraction Module

Provides robust PDF extraction with:
- Text extraction (pypdfium2, PyPDF2, pdfplumber)
- Table extraction (Camelot)
- OCR for scanned documents (EasyOCR, Tesseract)
- Vision extraction (MedGemma multimodal)
- Pre-flight validation
- Universal text extraction (X2Text pattern)
- Content-agnostic medical extraction
"""

from .pdf_extractor import PDFExtractor, ExtractionResult
from .pdf_validator import PDFValidator, ValidationResult
from .text_extractor import TextExtractor, TextExtractionResult, ExtractedText
from .table_extractor import TableExtractor, ExtractedTable
from .ocr_extractor import OCRExtractor, OCRExtractionResult, OCRResult
from .layout_analyzer import LayoutAnalyzer
from .vision_extractor import VisionExtractor, VisualExtractionResult, HybridExtractor

# Extraction-First Components (Unstract-inspired)
from .universal_text_extractor import (
    UniversalTextExtractor,
    UniversalTextResult,
    DocumentSourceType,
    LayoutInfo,
    PageText,
    TextRegion,
    WordBox,
    extract_text_universal
)
from .content_agnostic_extractor import (
    ContentAgnosticExtractor,
    GenericMedicalExtraction,
    PatientInfo,
    TestResult,
    MedicationInfo,
    ClinicalFinding,
    ProcedureInfo,
    DateInfo,
    ProviderInfo,
    OrganizationInfo,
    extract_medical_content
)
from .consensus_extractor import (
    ConsensusExtractor,
    ConsensusResult,
    extract_with_consensus
)

__all__ = [
    # Main orchestrator
    "PDFExtractor",
    "ExtractionResult",
    # Validation
    "PDFValidator",
    "ValidationResult",
    # Text extraction
    "TextExtractor",
    "TextExtractionResult",
    "ExtractedText",
    # Table extraction
    "TableExtractor",
    "ExtractedTable",
    # OCR
    "OCRExtractor",
    "OCRExtractionResult",
    "OCRResult",
    # Vision (MedGemma multimodal)
    "VisionExtractor",
    "VisualExtractionResult",
    "HybridExtractor",
    # Layout
    "LayoutAnalyzer",
    # Universal Text Extraction (X2Text)
    "UniversalTextExtractor",
    "UniversalTextResult",
    "DocumentSourceType",
    "LayoutInfo",
    "PageText",
    "TextRegion",
    "WordBox",
    "extract_text_universal",
    # Content-Agnostic Extraction
    "ContentAgnosticExtractor",
    "GenericMedicalExtraction",
    "PatientInfo",
    "TestResult",
    "MedicationInfo",
    "ClinicalFinding",
    "ProcedureInfo",
    "DateInfo",
    "ProviderInfo",
    "OrganizationInfo",
    "extract_medical_content",
    # Consensus Extraction (VLM + OCR in parallel)
    "ConsensusExtractor",
    "ConsensusResult",
    "extract_with_consensus",
]
