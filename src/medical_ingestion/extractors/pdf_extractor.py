# src/medical_ingestion/extractors/pdf_extractor.py
"""
PDF Extraction Orchestrator

Provides robust PDF extraction with:
1. Pre-flight validation (encryption, corruption, scan detection)
2. Text extraction cascade (pypdfium2 → PyPDF2 → pdfplumber)
3. OCR fallback for scanned/image pages (EasyOCR → Tesseract)
4. Table extraction (Camelot lattice → stream)
5. Unified result with confidence scoring

Extraction Flow:
    PDF → Validate → Text Extract → OCR (if needed) → Table Extract → Combine
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

from .pdf_validator import PDFValidator, ValidationResult
from .text_extractor import TextExtractor, TextExtractionResult, ExtractedText
from .table_extractor import TableExtractor, ExtractedTable
from .ocr_extractor import OCRExtractor, OCRExtractionResult


@dataclass
class ExtractionResult:
    """Unified extraction result with all extracted content."""
    text: str = ""
    tables: List[ExtractedTable] = field(default_factory=list)
    method: str = "unknown"  # 'text', 'ocr', 'table_lattice', 'table_stream', 'hybrid'
    confidence: float = 0.0
    page_count: int = 0
    pages_ocr: List[int] = field(default_factory=list)  # Pages that used OCR
    is_encrypted: bool = False
    is_scanned: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[ValidationResult] = None
    text_result: Optional[TextExtractionResult] = None
    ocr_result: Optional[OCRExtractionResult] = None


class PDFExtractor:
    """
    Robust PDF extraction orchestrator.

    Features:
    - Pre-extraction validation
    - Multiple extraction strategies with fallback
    - OCR for scanned documents
    - Confidence-based method selection
    - Detailed error reporting
    """

    # Confidence thresholds
    TABLE_LATTICE_THRESHOLD = 0.85
    TABLE_STREAM_THRESHOLD = 0.80
    OCR_THRESHOLD = 100  # Chars per page below which OCR is triggered

    def __init__(self, use_gpu: bool = False):
        """
        Initialize PDF extractor.

        Args:
            use_gpu: Use GPU for OCR (if available)
        """
        self.logger = logging.getLogger(__name__)

        # Initialize extractors
        self.validator = PDFValidator()
        self.text_extractor = TextExtractor()
        self.table_extractor = TableExtractor()
        self._ocr_extractor: Optional[OCRExtractor] = None
        self._use_gpu = use_gpu

    @property
    def ocr_extractor(self) -> OCRExtractor:
        """Lazy-load OCR extractor (heavy initialization)."""
        if self._ocr_extractor is None:
            self._ocr_extractor = OCRExtractor(use_gpu=self._use_gpu)
        return self._ocr_extractor

    def extract_best_effort(
        self,
        pdf_path: Path,
        password: Optional[str] = None,
        enable_ocr: bool = True,
        preserve_layout: bool = True
    ) -> ExtractionResult:
        """
        Extract content using best available method.

        Flow:
        1. VALIDATE - Check encryption, corruption, scan detection
        2. TEXT - Try pypdfium2 → PyPDF2 → pdfplumber
        3. OCR - If text extraction yields little, run OCR on those pages
        4. TABLES - Try Camelot (lattice → stream)
        5. COMBINE - Merge results with confidence scoring

        Args:
            pdf_path: Path to PDF file
            password: Password for encrypted PDFs
            enable_ocr: Enable OCR for scanned pages
            preserve_layout: Preserve text layout

        Returns:
            ExtractionResult with all extracted content
        """
        pdf_path = Path(pdf_path)
        result = ExtractionResult()

        self.logger.info(f"Best-effort extraction for {pdf_path}")

        # Step 1: Validate PDF
        validation = self.validator.validate(pdf_path, password=password)
        result.validation = validation
        result.is_encrypted = validation.is_encrypted
        result.is_scanned = validation.is_scanned
        result.page_count = validation.page_count
        result.metadata = validation.metadata

        if not validation.is_valid:
            result.errors.extend(validation.errors)
            self.logger.error(f"PDF validation failed: {validation.errors}")
            return result

        result.warnings.extend(validation.warnings)

        # Step 2: Text extraction
        try:
            text_result = self.text_extractor.extract_text_detailed(
                pdf_path,
                preserve_layout=preserve_layout,
                password=password
            )
            result.text_result = text_result
            result.text = text_result.text
            result.warnings.extend(text_result.warnings)

            self.logger.info(
                f"Text extraction: {len(result.text)} chars, "
                f"method={text_result.method}, "
                f"needs_ocr={text_result.needs_ocr}"
            )

        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            result.errors.append(f"Text extraction failed: {e}")
            text_result = None

        # Step 3: OCR for pages that need it
        if enable_ocr and text_result and text_result.needs_ocr:
            try:
                ocr_result = self.ocr_extractor.extract_text(
                    pdf_path,
                    pages=text_result.pages_needing_ocr
                )
                result.ocr_result = ocr_result
                result.pages_ocr = text_result.pages_needing_ocr

                # Merge OCR text into result
                if ocr_result.full_text:
                    result.text = self._merge_text_with_ocr(
                        text_result, ocr_result
                    )
                    result.method = "hybrid"
                    self.logger.info(
                        f"OCR extracted {len(ocr_result.full_text)} additional chars "
                        f"from {len(ocr_result.pages)} pages"
                    )

                result.warnings.extend(ocr_result.warnings)
                result.errors.extend(ocr_result.errors)

            except ImportError as e:
                result.warnings.append(f"OCR not available: {e}")
                self.logger.warning(f"OCR not available: {e}")
            except Exception as e:
                result.warnings.append(f"OCR failed: {e}")
                self.logger.warning(f"OCR failed: {e}")

        # Step 4: Table extraction
        tables_result = self._extract_tables_best_effort(pdf_path)

        if tables_result["tables"]:
            result.tables = tables_result["tables"]

            # If tables have high confidence, they may be the primary content
            if tables_result["confidence"] > self.TABLE_LATTICE_THRESHOLD:
                if not result.method:
                    result.method = tables_result["method"]

            self.logger.info(
                f"Table extraction: {len(result.tables)} tables, "
                f"confidence={tables_result['confidence']:.2f}"
            )

        # Step 5: Calculate final confidence
        result.confidence = self._calculate_final_confidence(result)

        # Set method if not already set
        if not result.method or result.method == "unknown":
            if result.tables and not result.text.strip():
                result.method = tables_result.get("method", "table")
            elif result.text.strip():
                result.method = text_result.method if text_result else "text"
            else:
                result.method = "failed"

        self.logger.info(
            f"Extraction complete: method={result.method}, "
            f"confidence={result.confidence:.2f}, "
            f"text={len(result.text)} chars, "
            f"tables={len(result.tables)}"
        )

        return result

    def _extract_tables_best_effort(self, pdf_path: Path) -> Dict[str, Any]:
        """Try table extraction with fallback."""
        # Try lattice first (structured tables with borders)
        try:
            tables_lattice = self.table_extractor.extract_tables(pdf_path, flavor='lattice')
            if tables_lattice and any(t.confidence > self.TABLE_LATTICE_THRESHOLD for t in tables_lattice):
                return {
                    "method": "table_lattice",
                    "tables": tables_lattice,
                    "confidence": max(t.confidence for t in tables_lattice)
                }
        except Exception as e:
            self.logger.warning(f"Lattice table extraction failed: {e}")

        # Try stream (tables defined by whitespace)
        try:
            tables_stream = self.table_extractor.extract_tables(pdf_path, flavor='stream')
            if tables_stream and any(t.confidence > self.TABLE_STREAM_THRESHOLD for t in tables_stream):
                return {
                    "method": "table_stream",
                    "tables": tables_stream,
                    "confidence": max(t.confidence for t in tables_stream)
                }
        except Exception as e:
            self.logger.warning(f"Stream table extraction failed: {e}")

        return {"method": "none", "tables": [], "confidence": 0.0}

    def _merge_text_with_ocr(
        self,
        text_result: TextExtractionResult,
        ocr_result: OCRExtractionResult
    ) -> str:
        """Merge text extraction with OCR results."""
        # Build a dict of OCR results by page
        ocr_by_page = {p.page_number: p.text for p in ocr_result.pages}

        # Merge pages
        merged_pages = []
        for page in text_result.pages:
            if page.page_number in ocr_by_page:
                # Use OCR text for this page
                merged_pages.append(ocr_by_page[page.page_number])
            else:
                # Use original text
                merged_pages.append(page.text)

        return "\n\n".join(merged_pages)

    def _calculate_final_confidence(self, result: ExtractionResult) -> float:
        """Calculate overall extraction confidence."""
        confidences = []

        # Text confidence
        if result.text_result:
            confidences.append(result.text_result.confidence)

        # OCR confidence
        if result.ocr_result and result.ocr_result.average_confidence > 0:
            confidences.append(result.ocr_result.average_confidence)

        # Table confidence
        if result.tables:
            table_conf = max(t.confidence for t in result.tables)
            confidences.append(table_conf)

        if not confidences:
            return 0.0

        # Weighted average (more weight to text)
        return sum(confidences) / len(confidences)

    # Convenience methods (backward compatible)

    def extract_text(
        self,
        pdf_path: Path,
        preserve_layout: bool = False,
        password: Optional[str] = None
    ) -> str:
        """
        Extract text from PDF.

        For detailed results, use extract_best_effort().
        """
        return self.text_extractor.extract_text(
            pdf_path,
            preserve_layout=preserve_layout,
            password=password
        )

    def extract_tables(
        self,
        pdf_path: Path,
        flavor: str = 'lattice'
    ) -> List[ExtractedTable]:
        """Extract tables from PDF."""
        return self.table_extractor.extract_tables(pdf_path, flavor=flavor)

    def validate(
        self,
        pdf_path: Path,
        password: Optional[str] = None
    ) -> ValidationResult:
        """Validate PDF before extraction."""
        return self.validator.validate(pdf_path, password=password)

    def extract_with_ocr(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None
    ) -> OCRExtractionResult:
        """
        Extract text using OCR.

        Use this for scanned documents or when text extraction fails.
        """
        return self.ocr_extractor.extract_text(pdf_path, pages=pages)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about available extraction capabilities."""
        return {
            "text_methods": ["pypdfium2", "pypdf2", "pdfplumber"],
            "table_methods": ["camelot_lattice", "camelot_stream"],
            "ocr": self.ocr_extractor.get_capabilities(),
            "validation": True
        }
