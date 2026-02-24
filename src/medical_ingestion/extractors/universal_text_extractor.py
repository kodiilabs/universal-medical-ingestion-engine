# ============================================================================
# src/medical_ingestion/extractors/universal_text_extractor.py
# ============================================================================
"""
Universal Text Extractor - X2Text Pattern Implementation

Inspired by Unstract's approach: Any document format → clean, structured text
with preserved layout information.

This is the foundation of the extraction-first architecture. It provides a
unified interface for extracting text from ANY document format (PDF, image, DOCX)
and produces consistent, structured output regardless of source.

Key Features:
- Auto-detects document type (digital PDF vs scanned vs image)
- Routes to appropriate extraction adapter
- Preserves layout information (tables, sections, headers)
- Returns word-level bounding boxes for downstream processing
- Works with existing TextExtractor, DocumentPipeline, and OCRRouter
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)


class DocumentSourceType(Enum):
    """Type of document source for extraction routing."""
    DIGITAL_PDF = "digital_pdf"      # Searchable PDF with embedded text
    SCANNED_PDF = "scanned_pdf"      # PDF with images that need OCR
    IMAGE = "image"                   # Image file (PNG, JPG, TIFF, etc.)
    UNKNOWN = "unknown"


@dataclass
class WordBox:
    """A word with its bounding box."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized 0-1
    confidence: float = 1.0
    page: int = 0


@dataclass
class PageText:
    """Text extracted from a single page."""
    page_number: int
    text: str
    width: float
    height: float
    word_boxes: List[WordBox] = field(default_factory=list)
    has_tables: bool = False
    has_images: bool = False


@dataclass
class TextRegion:
    """A detected region within the document."""
    region_type: str  # "table", "paragraph", "header", "list", "handwriting"
    text: str
    bbox: Tuple[float, float, float, float]  # normalized 0-1
    confidence: float
    page: int


@dataclass
class LayoutInfo:
    """Layout information extracted from the document."""
    has_tables: bool = False
    has_sections: bool = False
    has_headers: bool = False
    has_handwriting: bool = False
    table_count: int = 0
    section_headers: List[str] = field(default_factory=list)
    detected_columns: int = 1

    # For medical documents specifically
    has_reference_ranges: bool = False
    has_test_value_columns: bool = False
    has_abnormal_flags: bool = False
    has_lab_units: bool = False


@dataclass
class UniversalTextResult:
    """
    Complete result from universal text extraction.

    This is the standard output regardless of input document format.
    """
    # Core text content
    full_text: str
    pages: List[PageText]

    # Layout information
    layout: LayoutInfo
    regions: List[TextRegion]

    # Metadata
    source_type: DocumentSourceType
    source_path: Path
    confidence: float
    extraction_method: str
    processing_time: float
    page_count: int

    # For downstream bbox mapping
    word_boxes: List[WordBox] = field(default_factory=list)

    # Warnings/errors
    warnings: List[str] = field(default_factory=list)

    def get_text_by_page(self, page_num: int) -> str:
        """Get text for a specific page."""
        for page in self.pages:
            if page.page_number == page_num:
                return page.text
        return ""

    def get_table_regions(self) -> List[TextRegion]:
        """Get all table regions."""
        return [r for r in self.regions if r.region_type == "table"]


class UniversalTextExtractor:
    """
    X2Text Pattern: Universal document → clean, structured text.

    Routes documents to appropriate extraction adapter based on type:
    - Digital PDFs → Direct text extraction (pypdfium2/pdfplumber)
    - Scanned PDFs → Preprocessing + OCR pipeline
    - Images → OCR (TrOCR or PaddleOCR)

    Consensus Mode (USE_CONSENSUS_EXTRACTION=true):
    - Runs both VLM and OCR in parallel
    - Merges results for higher accuracy
    - Best for complex tables, low-quality scans, handwriting

    Usage:
        extractor = UniversalTextExtractor()
        result = await extractor.extract(Path("document.pdf"))
        print(result.full_text)
        print(result.layout.has_tables)
    """

    # Threshold for detecting scanned vs digital PDFs
    MIN_CHARS_PER_PAGE = 100

    # Supported image extensions
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.gif'}

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Lazy-loaded adapters
        self._text_extractor = None
        self._document_pipeline = None
        self._ocr_router = None
        self._vlm_client = None
        self._consensus_extractor = None

        # Configuration
        self.enable_preprocessing = self.config.get('enable_preprocessing', True)
        self.enable_region_detection = self.config.get('enable_region_detection', True)
        self.use_vlm_classification = self.config.get('use_vlm_classification', False)

        # Consensus extraction: run VLM + OCR in parallel and merge
        # Default to False - consensus can cause issues with local Ollama (timeouts, text bloat)
        # Enable explicitly when needed for scanned documents with poor OCR
        self.use_consensus_extraction = self.config.get('use_consensus_extraction', False)

        # VLM fallback settings (used when consensus is disabled)
        self.use_vlm = self.config.get('use_vlm', True)
        self.vlm_fallback_threshold = self.config.get('vlm_fallback_threshold', 0.5)
        self.min_text_for_valid = 50  # Minimum chars to consider extraction valid
        self.force_vlm_all_pages = self.config.get('force_vlm_all_pages', False)

        # VLM Unified: skip PaddleOCR entirely, use VLM for all text extraction
        self.vlm_unified = self.config.get('vlm_unified', False)

    async def extract(self, document_path: Path, progress_callback=None) -> UniversalTextResult:
        """
        Extract text from any supported document format.

        This is the main entry point. It auto-detects document type
        and routes to the appropriate extraction method.

        Args:
            document_path: Path to document file
            progress_callback: Optional callback(current_page, total_pages) for per-page progress

        Returns:
            UniversalTextResult with full extraction results
        """
        import time
        start_time = time.time()

        document_path = Path(document_path)

        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        # Detect document type
        source_type = self._detect_document_type(document_path)

        logger.info(f"Extracting text from {document_path.name} (type: {source_type.value})")

        # =====================================================================
        # VLM Unified Mode: skip PaddleOCR, use VLM for all text extraction
        # =====================================================================
        if self.vlm_unified and source_type in (DocumentSourceType.SCANNED_PDF, DocumentSourceType.IMAGE):
            logger.info(f"VLM Unified mode: using VLM directly for {source_type.value}")
            result = await self._extract_with_vlm(document_path, progress_callback=progress_callback, ocr_only=True)
            if not result or not result.full_text.strip():
                logger.warning("VLM unified extraction returned no text")
                result = UniversalTextResult(
                    full_text="",
                    pages=[],
                    layout=LayoutInfo(),
                    regions=[],
                    source_type=source_type,
                    source_path=document_path,
                    confidence=0.0,
                    extraction_method="vlm_unified_empty",
                    processing_time=0.0,
                    page_count=0,
                    warnings=["VLM unified extraction returned no text"]
                )
            else:
                result.extraction_method = "vlm_unified"

        # =====================================================================
        # Standard Mode: route by document type
        # =====================================================================
        elif source_type == DocumentSourceType.DIGITAL_PDF:
            result = await self._extract_digital_pdf(document_path)
        elif source_type == DocumentSourceType.SCANNED_PDF:
            result = await self._extract_scanned_pdf(document_path, progress_callback=progress_callback)
        elif source_type == DocumentSourceType.IMAGE:
            if progress_callback:
                progress_callback(1, 1)
            result = await self._extract_image(document_path)
        else:
            # Try digital PDF extraction as fallback
            result = await self._extract_digital_pdf(document_path)

        # =====================================================================
        # VLM Post-processing (only in standard mode, skipped in unified mode)
        # =====================================================================
        if not self.vlm_unified:
            # Force VLM for all pages (Accurate mode)
            if self.force_vlm_all_pages and self.use_vlm:
                logger.info(f"Accurate mode: running VLM on all pages ({result.page_count} pages)")
                vlm_result = await self._extract_with_vlm(document_path, progress_callback=progress_callback)
                if vlm_result and vlm_result.full_text.strip():
                    logger.info(
                        f"VLM extraction: {len(vlm_result.full_text)} chars "
                        f"(OCR had {len(result.full_text.strip()) if result.full_text else 0} chars)"
                    )
                    result.full_text = vlm_result.full_text
                    result.confidence = vlm_result.confidence
                    result.extraction_method = f"{result.extraction_method}+vlm_forced"
                    if vlm_result.regions:
                        result.regions.extend(vlm_result.regions)
            else:
                # VLM Fallback: If OCR failed or returned insufficient text, use VLM
                # SKIP VLM for large documents (>5 pages) — VLM processes per-page
                # and would be extremely slow (19 pages × 2 min = 38 min).
                # Large digital PDFs already have good text from pdfplumber/PyMuPDF.
                text_length = len(result.full_text.strip()) if result.full_text else 0
                page_count = result.page_count if hasattr(result, 'page_count') else len(result.pages)
                max_pages_for_vlm = self.config.get('max_pages_for_vlm_fallback', 5)

                needs_vlm_fallback = (
                    self.use_vlm and
                    page_count <= max_pages_for_vlm and
                    (text_length < self.min_text_for_valid or result.confidence < self.vlm_fallback_threshold)
                )

                if self.use_vlm and page_count > max_pages_for_vlm and text_length < self.min_text_for_valid:
                    logger.warning(
                        f"Large document ({page_count} pages) with low text ({text_length} chars) "
                        f"— skipping VLM fallback (max {max_pages_for_vlm} pages). "
                        "Consider improving OCR or splitting the document."
                    )
                    result.warnings.append(
                        f"VLM fallback skipped for {page_count}-page document (limit: {max_pages_for_vlm} pages)"
                    )

                if needs_vlm_fallback:
                    logger.info(
                        f"OCR extraction insufficient ({text_length} chars, {result.confidence:.2f} conf), "
                        "trying VLM extraction..."
                    )
                    vlm_result = await self._extract_with_vlm(document_path, progress_callback=progress_callback)
                    if vlm_result and len(vlm_result.full_text.strip()) > text_length:
                        logger.info(
                            f"VLM extraction successful: {len(vlm_result.full_text)} chars "
                            f"(was {text_length} from OCR)"
                        )
                        # Merge VLM results - VLM text takes priority if better
                        result.full_text = vlm_result.full_text
                        result.confidence = vlm_result.confidence
                        result.extraction_method = f"{result.extraction_method}+vlm"
                        result.warnings.append(
                            f"VLM fallback used (OCR returned only {text_length} chars)"
                        )
                        # Preserve VLM-extracted fields if available
                        if vlm_result.regions:
                            result.regions.extend(vlm_result.regions)

        # Update metadata
        result.source_type = source_type
        result.source_path = document_path
        result.processing_time = time.time() - start_time

        # Analyze layout if not already done
        if not result.layout.has_tables and not result.layout.has_sections:
            result.layout = self._analyze_layout(result.full_text, result.pages)

        logger.info(
            f"Extraction complete: {len(result.full_text)} chars, "
            f"{result.page_count} pages, {result.processing_time:.2f}s"
        )

        return result

    def _detect_document_type(self, document_path: Path) -> DocumentSourceType:
        """
        Detect document type for routing.

        - Checks file extension first
        - For PDFs, checks text content to distinguish digital vs scanned
        """
        suffix = document_path.suffix.lower()

        # Image files
        if suffix in self.IMAGE_EXTENSIONS:
            return DocumentSourceType.IMAGE

        # PDF files - need to check if digital or scanned
        if suffix == '.pdf':
            return self._detect_pdf_type(document_path)

        return DocumentSourceType.UNKNOWN

    def _detect_pdf_type(self, pdf_path: Path) -> DocumentSourceType:
        """
        Detect if PDF is digital (searchable) or scanned (image-based).

        Uses multiple heuristics:
        1. Text character count per page
        2. Ratio of meaningful text vs whitespace/decorative chars
        3. Presence of embedded images covering significant page area

        This handles "hybrid" PDFs that have decorative text but actual
        content in images.
        """
        try:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)

            if num_pages == 0:
                doc.close()
                return DocumentSourceType.UNKNOWN

            total_chars = 0
            total_meaningful_chars = 0
            has_significant_images = False

            for page in doc:
                text = page.get_text()
                total_chars += len(text)

                # Count "meaningful" characters (not whitespace or decorative)
                # Remove common decorative patterns
                clean_text = text.replace('~', '').replace('-', '').replace('=', '')
                clean_text = ' '.join(clean_text.split())  # Collapse whitespace
                total_meaningful_chars += len(clean_text)

                # Check for embedded images
                images = page.get_images()
                if images:
                    # Check if images cover significant page area
                    page_rect = page.rect
                    page_area = page_rect.width * page_rect.height

                    for img_info in images:
                        try:
                            # Get image bbox from the page
                            img_rects = page.get_image_rects(img_info[0])
                            for rect in img_rects:
                                img_area = rect.width * rect.height
                                # If any image covers >30% of page, it's likely content
                                if img_area / page_area > 0.3:
                                    has_significant_images = True
                                    break
                        except Exception:
                            # If we can't get rect, assume significant if multiple images
                            if len(images) >= 1:
                                has_significant_images = True
                    if has_significant_images:
                        break

            doc.close()

            avg_chars_per_page = total_chars / num_pages
            avg_meaningful_per_page = total_meaningful_chars / num_pages

            logger.debug(
                f"PDF detection: {avg_chars_per_page:.0f} chars/page, "
                f"{avg_meaningful_per_page:.0f} meaningful/page, "
                f"has_images={has_significant_images}"
            )

            # Decision logic:
            # A real document page typically has 500+ chars of meaningful content.
            # Headers/footers might have 100-300 chars.
            # If text is sparse AND page has significant images → content is in images
            MIN_MEANINGFUL_FOR_DIGITAL = 500  # Full page of real text content

            if avg_meaningful_per_page >= MIN_MEANINGFUL_FOR_DIGITAL:
                return DocumentSourceType.DIGITAL_PDF
            elif has_significant_images:
                # Low text + images = content is in images
                logger.info(
                    f"PDF has significant images with sparse text ({avg_meaningful_per_page:.0f} chars/page), "
                    "treating as scanned for OCR"
                )
                return DocumentSourceType.SCANNED_PDF
            elif avg_meaningful_per_page >= self.MIN_CHARS_PER_PAGE:
                # Some text but no images - might be sparse digital
                return DocumentSourceType.DIGITAL_PDF
            else:
                return DocumentSourceType.SCANNED_PDF

        except Exception as e:
            logger.warning(f"Error detecting PDF type: {e}")
            return DocumentSourceType.UNKNOWN

    async def _extract_digital_pdf(self, pdf_path: Path) -> UniversalTextResult:
        """
        Extract text from digital (searchable) PDF.

        Uses pypdfium2/PyPDF2 for text and pdfplumber for layout/tables.
        Runs in a thread pool to avoid blocking the asyncio event loop
        (pdfplumber + PyMuPDF are CPU-bound sync operations).
        """
        return await asyncio.to_thread(self._extract_digital_pdf_sync, pdf_path)

    def _extract_digital_pdf_sync(self, pdf_path: Path) -> UniversalTextResult:
        """Synchronous digital PDF extraction (runs in thread pool)."""
        # Get text extractor
        if self._text_extractor is None:
            from .text_extractor import TextExtractor
            self._text_extractor = TextExtractor()

        # Extract basic text
        extraction_result = self._text_extractor.extract_text_detailed(
            pdf_path,
            preserve_layout=True
        )

        # Extract layout with pdfplumber
        pages = []
        regions = []
        all_word_boxes = []
        layout = LayoutInfo()

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Get page dimensions
                    width = page.width
                    height = page.height

                    # Get text with word positions
                    words = page.extract_words() or []
                    word_boxes = []

                    for word in words:
                        wb = WordBox(
                            text=word.get('text', ''),
                            bbox=(
                                word['x0'] / width,
                                word['top'] / height,
                                word['x1'] / width,
                                word['bottom'] / height
                            ),
                            confidence=1.0,
                            page=page_num
                        )
                        word_boxes.append(wb)
                        all_word_boxes.append(wb)

                    # Detect tables
                    tables = page.find_tables() or []
                    has_tables = len(tables) > 0

                    if has_tables:
                        layout.has_tables = True
                        layout.table_count += len(tables)

                        # Add table regions
                        for table in tables:
                            table_bbox = table.bbox
                            table_text = ""
                            try:
                                extracted = table.extract()
                                if extracted:
                                    table_text = "\n".join(
                                        " | ".join(str(cell) for cell in row if cell)
                                        for row in extracted if row
                                    )
                            except Exception:
                                pass

                            regions.append(TextRegion(
                                region_type="table",
                                text=table_text,
                                bbox=(
                                    table_bbox[0] / width,
                                    table_bbox[1] / height,
                                    table_bbox[2] / width,
                                    table_bbox[3] / height
                                ),
                                confidence=0.9,
                                page=page_num
                            ))

                    # Get page text
                    page_text = extraction_result.pages[page_num].text if page_num < len(extraction_result.pages) else ""

                    pages.append(PageText(
                        page_number=page_num,
                        text=page_text,
                        width=width,
                        height=height,
                        word_boxes=word_boxes,
                        has_tables=has_tables,
                        has_images=len(page.images) > 0 if hasattr(page, 'images') else False
                    ))

        except Exception as e:
            logger.warning(f"pdfplumber layout extraction failed: {e}")
            # Fall back to basic extraction result
            for i, page_result in enumerate(extraction_result.pages):
                pages.append(PageText(
                    page_number=i,
                    text=page_result.text,
                    width=612,  # Default letter size
                    height=792,
                    word_boxes=[],
                    has_tables=False,
                    has_images=False
                ))

        # Analyze layout from text
        layout = self._analyze_layout(extraction_result.text, pages, layout)

        return UniversalTextResult(
            full_text=extraction_result.text,
            pages=pages,
            layout=layout,
            regions=regions,
            source_type=DocumentSourceType.DIGITAL_PDF,
            source_path=pdf_path,
            confidence=extraction_result.confidence,
            extraction_method=extraction_result.method,
            processing_time=0.0,
            page_count=len(pages),
            word_boxes=all_word_boxes,
            warnings=extraction_result.warnings
        )

    async def _extract_scanned_pdf(self, pdf_path: Path, progress_callback=None) -> UniversalTextResult:
        """
        Extract text from scanned (image-based) PDF.

        For large PDFs (>5 pages), uses PaddleOCR singleton directly via
        asyncio.to_thread() to avoid blocking the event loop.
        For smaller PDFs, uses the full document pipeline with region detection.
        """
        # Check page count to decide extraction strategy
        page_count = 0
        try:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
        except Exception:
            pass

        max_pages_for_full_pipeline = self.config.get('max_pages_for_full_pipeline', 5)

        if page_count > max_pages_for_full_pipeline:
            # Large PDF: Use PaddleOCR singleton directly in a thread pool.
            # This avoids region detection + per-region OCR overhead and keeps
            # the event loop free for API requests.
            logger.info(
                f"Large scanned PDF ({page_count} pages) — using fast PaddleOCR path "
                f"(skipping region detection to avoid {page_count}×N OCR calls)"
            )
            return await self._extract_scanned_pdf_fast(pdf_path, page_count, progress_callback=progress_callback)

        # Use consensus extraction if enabled (small docs only)
        if self.use_consensus_extraction:
            result = await self._extract_with_consensus(pdf_path)
            if result and len(result.full_text.strip()) >= self.min_text_for_valid:
                return result
            logger.warning("Consensus extraction returned insufficient text, falling back to standard pipeline")

        # Get document pipeline
        if self._document_pipeline is None:
            from ..core.document_pipeline import DocumentPipeline
            self._document_pipeline = DocumentPipeline({
                'enable_preprocessing': self.enable_preprocessing,
                'enable_region_detection': self.enable_region_detection,
                'use_vlm_classification': self.use_vlm_classification
            })

        # Process PDF through pipeline
        pipeline_results = await self._document_pipeline.process_pdf(pdf_path)

        # Convert pipeline results to universal format
        pages = []
        regions = []
        all_word_boxes = []
        all_text_parts = []
        layout = LayoutInfo()
        total_pages = len(pipeline_results)

        for i, result in enumerate(pipeline_results):
            if progress_callback:
                progress_callback(i + 1, total_pages)
            # Collect text
            all_text_parts.append(f"--- Page {i + 1} ---")
            all_text_parts.append(result.full_text)

            # Build word boxes from OCR results
            word_boxes = []
            for ocr_result in result.ocr_result.regions:
                for word_box in ocr_result.word_boxes:
                    if isinstance(word_box, dict):
                        wb = WordBox(
                            text=word_box.get('text', ''),
                            bbox=tuple(word_box.get('bbox', (0, 0, 0, 0))),
                            confidence=word_box.get('confidence', 0.8),
                            page=i
                        )
                        word_boxes.append(wb)
                        all_word_boxes.append(wb)

            # Build page
            pages.append(PageText(
                page_number=i,
                text=result.full_text,
                width=result.preprocessing.processed_size[0],
                height=result.preprocessing.processed_size[1],
                word_boxes=word_boxes,
                has_tables=result.has_tables,
                has_images=True  # Scanned PDFs are images
            ))

            # Build regions
            for ocr_result in result.ocr_result.regions:
                region_type = ocr_result.region.region_type.value
                regions.append(TextRegion(
                    region_type=region_type,
                    text=ocr_result.text,
                    bbox=(
                        ocr_result.region.bbox.x1,
                        ocr_result.region.bbox.y1,
                        ocr_result.region.bbox.x2,
                        ocr_result.region.bbox.y2
                    ),
                    confidence=ocr_result.confidence,
                    page=i
                ))

            # Update layout
            if result.has_tables:
                layout.has_tables = True
                layout.table_count += 1
            if result.has_handwriting:
                layout.has_handwriting = True

        full_text = "\n".join(all_text_parts)

        # Calculate average confidence
        confidences = [r.average_confidence for r in pipeline_results if r.average_confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.7

        # Analyze layout from text
        layout = self._analyze_layout(full_text, pages, layout)

        result = UniversalTextResult(
            full_text=full_text,
            pages=pages,
            layout=layout,
            regions=regions,
            source_type=DocumentSourceType.SCANNED_PDF,
            source_path=pdf_path,
            confidence=avg_confidence,
            extraction_method="document_pipeline_ocr",
            processing_time=0.0,
            page_count=len(pages),
            word_boxes=all_word_boxes,
            warnings=[]
        )

        # Tesseract fallback: if PaddleOCR returned insufficient text, try Tesseract
        if len(full_text.strip()) < self.min_text_for_valid:
            logger.info(
                f"PaddleOCR returned insufficient text ({len(full_text)} chars), "
                "falling back to Tesseract..."
            )
            tesseract_result = await self._extract_with_tesseract(pdf_path)
            if tesseract_result and len(tesseract_result.full_text.strip()) > len(full_text.strip()):
                logger.info(
                    f"Tesseract extraction successful: {len(tesseract_result.full_text)} chars"
                )
                return tesseract_result

        return result

    async def _extract_scanned_pdf_fast(self, pdf_path: Path, page_count: int, progress_callback=None) -> UniversalTextResult:
        """
        Fast extraction for large scanned PDFs.

        Uses PaddleOCR singleton per-page in a thread pool, skipping the full
        document pipeline (region detection, preprocessing) which is too slow
        for 10+ page documents.

        This keeps the asyncio event loop free for API requests.
        """
        from .paddle_ocr import get_paddle_ocr_extractor
        import tempfile
        import os

        def _extract_all_pages():
            """Run all page OCR in a background thread."""
            paddle = get_paddle_ocr_extractor()
            doc = fitz.open(str(pdf_path))
            pages = []
            all_text_parts = []

            for page_num in range(len(doc)):
                logger.info(f"Fast OCR: page {page_num + 1}/{len(doc)}")
                if progress_callback:
                    progress_callback(page_num + 1, len(doc))
                page = doc.load_page(page_num)

                # Render at 2x zoom for OCR quality
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)

                # Save to temp file for PaddleOCR
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    pix.save(f.name)
                    temp_path = f.name

                try:
                    result = paddle.extract_from_image(Path(temp_path))
                    page_text = result.get('text', '')
                finally:
                    os.unlink(temp_path)

                all_text_parts.append(f"--- Page {page_num + 1} ---")
                all_text_parts.append(page_text)

                pages.append(PageText(
                    page_number=page_num,
                    text=page_text,
                    width=pix.width,
                    height=pix.height,
                    word_boxes=[],
                    has_tables=False,
                    has_images=True
                ))

            doc.close()
            return pages, "\n".join(all_text_parts)

        # Run sync OCR in thread pool to avoid blocking event loop
        pages, full_text = await asyncio.to_thread(_extract_all_pages)

        layout = self._analyze_layout(full_text, pages)

        result = UniversalTextResult(
            full_text=full_text,
            pages=pages,
            layout=layout,
            regions=[],
            source_type=DocumentSourceType.SCANNED_PDF,
            source_path=pdf_path,
            confidence=0.8,
            extraction_method="paddle_ocr_fast",
            processing_time=0.0,
            page_count=len(pages),
            word_boxes=[],
            warnings=[]
        )

        # If OCR returned very little text, try also getting embedded text
        if len(full_text.strip()) < 500 and page_count > 0:
            logger.info("Fast OCR returned little text, supplementing with embedded text...")
            try:
                doc = fitz.open(str(pdf_path))
                embedded_parts = []
                for page in doc:
                    embedded_parts.append(page.get_text())
                doc.close()
                embedded_text = "\n".join(embedded_parts)
                if len(embedded_text.strip()) > len(full_text.strip()):
                    result.full_text = embedded_text
                    result.extraction_method = "paddle_ocr_fast+embedded"
                    result.warnings.append("Supplemented OCR with embedded text")
            except Exception as e:
                logger.warning(f"Failed to get embedded text: {e}")

        return result

    async def _extract_with_tesseract(self, document_path: Path) -> Optional[UniversalTextResult]:
        """
        Extract text using Tesseract OCR.

        This is a fallback when PaddleOCR fails or returns insufficient text.
        Tesseract is more reliable for simple documents but less accurate on
        complex layouts.

        Args:
            document_path: Path to document (PDF or image)

        Returns:
            UniversalTextResult or None if extraction fails
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
            return None

        suffix = document_path.suffix.lower()
        images_to_process = []

        # For PDFs, render pages as images
        if suffix == '.pdf':
            try:
                doc = fitz.open(str(document_path))
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # Render at 200 DPI for good OCR quality
                    mat = fitz.Matrix(200/72, 200/72)
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images_to_process.append((img, page_num, pix.width, pix.height))

                doc.close()
            except Exception as e:
                logger.error(f"Failed to convert PDF for Tesseract: {e}")
                return None
        else:
            # Direct image — apply EXIF orientation correction for camera photos
            try:
                from ..utils.image_utils import load_image_for_ocr
                img = load_image_for_ocr(document_path)
                images_to_process.append((img, 0, img.width, img.height))
            except Exception as e:
                logger.error(f"Failed to open image for Tesseract: {e}")
                return None

        # Process each image with Tesseract
        all_text = []
        pages = []

        for img, page_num, width, height in images_to_process:
            try:
                # Run Tesseract
                text = pytesseract.image_to_string(img, lang='eng')

                if text.strip():
                    all_text.append(f"--- Page {page_num + 1} ---")
                    all_text.append(text)

                    pages.append(PageText(
                        page_number=page_num,
                        text=text,
                        width=float(width),
                        height=float(height),
                        word_boxes=[],
                        has_tables=False,
                        has_images=True
                    ))

            except Exception as e:
                logger.warning(f"Tesseract failed on page {page_num}: {e}")
                continue

        if not all_text:
            return None

        full_text = "\n".join(all_text)

        # Build layout from text
        layout = self._analyze_layout(full_text, pages)

        return UniversalTextResult(
            full_text=full_text,
            pages=pages,
            layout=layout,
            regions=[],  # Tesseract doesn't provide region info
            source_type=DocumentSourceType.SCANNED_PDF if suffix == '.pdf' else DocumentSourceType.IMAGE,
            source_path=document_path,
            confidence=0.75,  # Tesseract typically good but not as accurate as PaddleOCR
            extraction_method="tesseract",
            processing_time=0.0,
            page_count=len(pages),
            word_boxes=[],
            warnings=["Extracted using Tesseract fallback"]
        )

    async def _extract_image(self, image_path: Path) -> UniversalTextResult:
        """
        Extract text from image file.

        Strategy (changed to OCR-first for speed):
        - Consensus extraction (VLM + OCR in parallel) if USE_CONSENSUS_EXTRACTION=true
        - force_vlm_all_pages (Accurate mode): VLM as primary extractor
        - Default (Auto/Fast): OCR first, VLM only as fallback if OCR insufficient
        """
        # Use consensus extraction if enabled
        if self.use_consensus_extraction:
            result = await self._extract_with_consensus(image_path)
            if result and len(result.full_text.strip()) >= self.min_text_for_valid:
                return result
            logger.warning("Consensus extraction returned insufficient text, falling back to standard pipeline")

        # Accurate mode: VLM as primary (handles tables, handwriting better)
        if self.force_vlm_all_pages and self.use_vlm:
            logger.info("Accurate mode — using VLM as primary extractor for image")
            vlm_result = await self._extract_with_vlm(image_path)
            if vlm_result and len(vlm_result.full_text.strip()) >= self.min_text_for_valid:
                logger.info(
                    f"VLM primary extraction successful: {len(vlm_result.full_text)} chars, "
                    f"confidence: {vlm_result.confidence:.2f}"
                )
                return vlm_result
            logger.warning("VLM primary extraction failed or insufficient, falling back to OCR pipeline")

        # Get document pipeline
        if self._document_pipeline is None:
            from ..core.document_pipeline import DocumentPipeline
            self._document_pipeline = DocumentPipeline({
                'enable_preprocessing': self.enable_preprocessing,
                'enable_region_detection': self.enable_region_detection,
                'use_vlm_classification': self.use_vlm_classification
            })

        # Process image through pipeline
        result = await self._document_pipeline.process(image_path)
        pipeline_text = result.full_text

        # If DocumentPipeline produced insufficient text, try PaddleOCR directly.
        # The pipeline uses cv2.imread + region detection + OCR routing which can
        # pick TrOCR and miss text. PaddleOCR with load_image_for_ocr handles
        # camera photos (EXIF, HEIC, resize) much more reliably.
        if len(pipeline_text.strip()) < self.min_text_for_valid:
            logger.warning(
                f"DocumentPipeline returned only {len(pipeline_text.strip())} chars, "
                f"trying PaddleOCR directly as final fallback"
            )
            paddle_result = await self._try_paddle_ocr_direct(image_path)
            if paddle_result and len(paddle_result.full_text.strip()) >= self.min_text_for_valid:
                logger.info(f"PaddleOCR direct fallback succeeded: {len(paddle_result.full_text)} chars")
                return paddle_result

        # Build word boxes
        word_boxes = []
        for ocr_result in result.ocr_result.regions:
            for word_box in ocr_result.word_boxes:
                if isinstance(word_box, dict):
                    wb = WordBox(
                        text=word_box.get('text', ''),
                        bbox=tuple(word_box.get('bbox', (0, 0, 0, 0))),
                        confidence=word_box.get('confidence', 0.8),
                        page=0
                    )
                    word_boxes.append(wb)

        # Build page
        page = PageText(
            page_number=0,
            text=result.full_text,
            width=result.preprocessing.processed_size[0],
            height=result.preprocessing.processed_size[1],
            word_boxes=word_boxes,
            has_tables=result.has_tables,
            has_images=True
        )

        # Build regions
        regions = []
        for ocr_result in result.ocr_result.regions:
            region_type = ocr_result.region.region_type.value
            regions.append(TextRegion(
                region_type=region_type,
                text=ocr_result.text,
                bbox=(
                    ocr_result.region.bbox.x1,
                    ocr_result.region.bbox.y1,
                    ocr_result.region.bbox.x2,
                    ocr_result.region.bbox.y2
                ),
                confidence=ocr_result.confidence,
                page=0
            ))

        # Build layout
        layout = LayoutInfo(
            has_tables=result.has_tables,
            has_handwriting=result.has_handwriting,
            table_count=1 if result.has_tables else 0
        )
        layout = self._analyze_layout(result.full_text, [page], layout)

        return UniversalTextResult(
            full_text=result.full_text,
            pages=[page],
            layout=layout,
            regions=regions,
            source_type=DocumentSourceType.IMAGE,
            source_path=image_path,
            confidence=result.average_confidence,
            extraction_method=f"ocr_{'+'.join(result.ocr_result.engines_used)}",
            processing_time=0.0,
            page_count=1,
            word_boxes=word_boxes,
            warnings=[]
        )

    async def _try_paddle_ocr_direct(self, image_path: Path) -> Optional[UniversalTextResult]:
        """
        Direct PaddleOCR fallback — bypasses DocumentPipeline's region detection
        and OCR routing which can pick TrOCR and miss text.

        PaddleOCRExtractor.extract_from_image() applies load_image_for_ocr()
        (EXIF correction, HEIC conversion, resize) and uses PaddleOCR's own
        layout detection, which is more reliable for camera photos.
        """
        try:
            from .paddle_ocr import get_paddle_ocr_extractor

            paddle = get_paddle_ocr_extractor()
            ocr_result = await paddle.extract_from_image_async(image_path)
            text = ocr_result.get('text', '')

            if not text or len(text.strip()) < self.min_text_for_valid:
                return None

            page = PageText(
                page_number=0,
                text=text,
                width=1.0,
                height=1.0,
                word_boxes=[],
                has_tables=False,
                has_images=True
            )
            layout = self._analyze_layout(text, [page])

            return UniversalTextResult(
                full_text=text,
                pages=[page],
                layout=layout,
                regions=[],
                source_type=DocumentSourceType.IMAGE,
                source_path=image_path,
                confidence=0.75,  # Direct PaddleOCR confidence
                extraction_method="ocr_paddleocr_direct",
                processing_time=0.0,
                page_count=1,
                word_boxes=[],
                warnings=["Used direct PaddleOCR fallback (DocumentPipeline produced insufficient text)"]
            )
        except Exception as e:
            logger.error(f"PaddleOCR direct fallback failed: {e}")
            return None

    async def _extract_with_vlm(self, document_path: Path, progress_callback=None, ocr_only=False) -> Optional[UniversalTextResult]:
        """
        Extract text using Vision Language Model (VLM).

        This is used as a fallback when OCR fails or returns insufficient text,
        or as the primary extractor in VLM Unified mode.

        Args:
            document_path: Path to document (PDF or image)
            progress_callback: Optional callback(current_page, total_pages) for per-page progress
            ocr_only: If True, use a simple OCR prompt (just read text faithfully).
                      Used in VLM Unified mode where VLM replaces PaddleOCR.

        Returns:
            UniversalTextResult or None if VLM extraction fails
        """
        try:
            from .vlm_client import VLMClient
        except ImportError:
            logger.warning("VLM client not available")
            return None

        # Initialize VLM client if needed
        if self._vlm_client is None:
            self._vlm_client = VLMClient(self.config)

        # For PDFs, convert to images first
        suffix = document_path.suffix.lower()
        images_to_process = []

        if suffix == '.pdf':
            # Convert PDF pages to images
            import tempfile
            try:
                doc = fitz.open(str(document_path))
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # Render at higher DPI for better VLM results
                    mat = fitz.Matrix(2, 2)  # 2x zoom = ~144 DPI
                    pix = page.get_pixmap(matrix=mat)

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        pix.save(f.name)
                        images_to_process.append((f.name, page_num))

                doc.close()
            except Exception as e:
                logger.error(f"Failed to convert PDF to images for VLM: {e}")
                return None
        else:
            # Direct image — prepare with EXIF correction for camera photos
            import tempfile as _tmpmod
            from ..utils.image_utils import load_image_for_ocr
            _prepared = load_image_for_ocr(document_path)
            with _tmpmod.NamedTemporaryFile(suffix='.png', delete=False) as _f:
                _prepared.save(_f, format='PNG')
                images_to_process.append((_f.name, 0))

        # Process each image with VLM
        all_text = []
        all_fields = {}
        total_confidence = 0.0
        pages = []
        regions = []

        # Per-page VLM timeout (2 min default). For large PDFs, VLM per-page
        # calls can hang if Ollama is memory-constrained.
        per_page_timeout = self.config.get('vlm_per_page_timeout', 120)

        # In OCR-only mode (VLM Unified), use a simple prompt that just reads text
        ocr_prompt = (
            "Read this medical document image and transcribe ALL text exactly as written. "
            "Preserve the original layout, line breaks, and formatting. "
            "Include every word, number, symbol, unit, and label visible. "
            "Do not summarize, interpret, or reorganize — just transcribe faithfully."
        ) if ocr_only else None

        total_pages = len(images_to_process)
        try:
            for image_path, page_num in images_to_process:
                if progress_callback:
                    progress_callback(page_num + 1, total_pages)
                try:
                    result = await asyncio.wait_for(
                        self._vlm_client.extract_from_image(
                            image_path,
                            prompt=ocr_prompt,
                            extract_all=not ocr_only
                        ),
                        timeout=per_page_timeout
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    logger.warning(
                        f"VLM timeout on page {page_num + 1} after {per_page_timeout}s, skipping"
                    )
                    continue
                except Exception as e:
                    logger.warning(f"VLM extraction failed on page {page_num + 1}: {e}")
                    continue

                if result.text:
                    all_text.append(result.text)
                    total_confidence += result.confidence

                    # Create page
                    page = PageText(
                        page_number=page_num,
                        text=result.text,
                        width=1.0,  # Normalized
                        height=1.0,
                        word_boxes=[],
                        has_tables=False,
                        has_images=True
                    )
                    pages.append(page)

                    # Add extracted fields as regions
                    for field_name, field_value in result.fields.items():
                        regions.append(TextRegion(
                            region_type="vlm_field",
                            text=f"{field_name}: {field_value}",
                            bbox=(0, 0, 1, 1),  # No specific bbox from VLM
                            confidence=result.confidence,
                            page=page_num
                        ))

                # Merge fields
                all_fields.update(result.fields)

        finally:
            # Clean up temp files (both PDF page images and EXIF-corrected images)
            import os as _os
            for image_path, _ in images_to_process:
                if image_path != str(document_path):
                    try:
                        _os.unlink(image_path)
                    except Exception:
                        pass

        if not all_text:
            return None

        full_text = "\n\n".join(all_text)
        avg_confidence = total_confidence / len(images_to_process) if images_to_process else 0.0

        # Build layout from VLM text
        layout = self._analyze_layout(full_text, pages)

        return UniversalTextResult(
            full_text=full_text,
            pages=pages,
            layout=layout,
            regions=regions,
            source_type=DocumentSourceType.SCANNED_PDF if suffix == '.pdf' else DocumentSourceType.IMAGE,
            source_path=document_path,
            confidence=avg_confidence,
            extraction_method="vlm",
            processing_time=0.0,
            page_count=len(pages),
            word_boxes=[],
            warnings=[]
        )

    async def _extract_with_consensus(self, document_path: Path) -> Optional[UniversalTextResult]:
        """
        Extract text using consensus of VLM and OCR running in parallel.

        This method runs both extraction methods simultaneously and intelligently
        merges their results for higher accuracy, especially on:
        - Complex tables
        - Low-quality scans
        - Documents with handwriting
        - Mixed printed/handwritten content

        Args:
            document_path: Path to document (PDF or image)

        Returns:
            UniversalTextResult or None if consensus extraction fails
        """
        try:
            from .consensus_extractor import ConsensusExtractor
        except ImportError:
            logger.warning("Consensus extractor not available")
            return None

        # Initialize consensus extractor if needed
        if self._consensus_extractor is None:
            self._consensus_extractor = ConsensusExtractor(self.config)

        try:
            consensus_result = await self._consensus_extractor.extract(document_path)

            if not consensus_result or not consensus_result.full_text:
                return None

            suffix = document_path.suffix.lower()

            # Build page from consensus result
            pages = [PageText(
                page_number=0,
                text=consensus_result.full_text,
                width=1.0,  # Normalized
                height=1.0,
                word_boxes=[],
                has_tables=False,
                has_images=True
            )]

            # Build regions from consensus fields
            regions = []
            for field_name, field_value in consensus_result.fields.items():
                if not field_name.startswith('_'):
                    regions.append(TextRegion(
                        region_type="consensus_field",
                        text=f"{field_name}: {field_value}",
                        bbox=(0, 0, 1, 1),
                        confidence=consensus_result.confidence,
                        page=0
                    ))

            # Build layout from text
            layout = self._analyze_layout(consensus_result.full_text, pages)

            # Build extraction method string
            method_parts = ["consensus"]
            if consensus_result.ocr_contribution > 0:
                method_parts.append(f"ocr:{consensus_result.ocr_contribution:.0%}")
            if consensus_result.vlm_contribution > 0:
                method_parts.append(f"vlm:{consensus_result.vlm_contribution:.0%}")
            extraction_method = "_".join(method_parts)

            # Add consensus metadata to warnings for visibility
            warnings = consensus_result.warnings.copy()
            warnings.append(f"Consensus: primary={consensus_result.primary_source}, "
                          f"OCR={consensus_result.ocr_confidence:.2f}, "
                          f"VLM={consensus_result.vlm_confidence:.2f}")

            logger.info(
                f"Consensus extraction: {len(consensus_result.full_text)} chars, "
                f"source={consensus_result.primary_source}, "
                f"conf={consensus_result.confidence:.2f}"
            )

            return UniversalTextResult(
                full_text=consensus_result.full_text,
                pages=pages,
                layout=layout,
                regions=regions,
                source_type=DocumentSourceType.SCANNED_PDF if suffix == '.pdf' else DocumentSourceType.IMAGE,
                source_path=document_path,
                confidence=consensus_result.confidence,
                extraction_method=extraction_method,
                processing_time=consensus_result.ocr_time + consensus_result.vlm_time + consensus_result.merge_time,
                page_count=1,
                word_boxes=[],
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Consensus extraction failed: {e}")
            return None

    def _analyze_layout(
        self,
        text: str,
        pages: List[PageText],
        existing_layout: LayoutInfo = None
    ) -> LayoutInfo:
        """
        Analyze text to detect layout characteristics.

        Looks for medical document patterns:
        - Reference ranges (e.g., "10-20", "< 100")
        - Test/value columns
        - Abnormal flags (H, L, HIGH, LOW)
        - Lab units (mg/dL, g/dL, etc.)
        - Section headers
        """
        import re

        layout = existing_layout or LayoutInfo()
        text_lower = text.lower()

        # Detect reference ranges
        ref_range_patterns = [
            r'\d+\s*-\s*\d+',  # "10-20"
            r'<\s*\d+',        # "< 100"
            r'>\s*\d+',        # "> 50"
            r'\d+\.\d+\s*-\s*\d+\.\d+',  # "1.0-2.5"
        ]
        for pattern in ref_range_patterns:
            if re.search(pattern, text):
                layout.has_reference_ranges = True
                break

        # Detect lab units
        lab_unit_patterns = [
            r'mg/dl', r'g/dl', r'mmol/l', r'meq/l', r'u/l', r'iu/l',
            r'k/ul', r'cells/ul', r'pg', r'fl', r'%', r'ng/ml', r'pg/ml',
            r'umol/l', r'nmol/l', r'g/l', r'ml/min'
        ]
        for pattern in lab_unit_patterns:
            if re.search(pattern, text_lower):
                layout.has_lab_units = True
                break

        # Detect abnormal flags
        abnormal_patterns = [
            r'\b(H|L|HH|LL|HIGH|LOW|ABNORMAL|CRITICAL)\b',
            r'\*\s*(H|L)',
            r'(H|L)\s*\*'
        ]
        for pattern in abnormal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                layout.has_abnormal_flags = True
                break

        # Detect section headers (common in medical docs)
        section_header_patterns = [
            r'^[A-Z][A-Z\s]+:',  # "CHEMISTRY:"
            r'^[A-Z][A-Z\s]+$',  # "HEMATOLOGY" on its own line
            r'(IMPRESSION|FINDINGS|HISTORY|DIAGNOSIS|ASSESSMENT):?',
            r'(PATIENT INFORMATION|SPECIMEN|COLLECTION):?'
        ]
        headers_found = []
        for line in text.split('\n'):
            line = line.strip()
            for pattern in section_header_patterns:
                if re.match(pattern, line):
                    headers_found.append(line)
                    layout.has_sections = True
                    break

        layout.section_headers = headers_found[:10]  # Limit to first 10

        # Detect if text appears columnar (test/value pattern)
        # Look for lines with test name followed by number
        test_value_pattern = r'^[A-Za-z\s]+\s+\d+\.?\d*\s*(mg|g|mmol|meq|u|k|cells|pg|fl|%|ng|ml)?'
        test_value_matches = sum(1 for line in text.split('\n') if re.match(test_value_pattern, line.strip()))
        if test_value_matches > 3:
            layout.has_test_value_columns = True

        return layout


# Convenience function
async def extract_text_universal(
    document_path: Path,
    config: Dict[str, Any] = None
) -> UniversalTextResult:
    """
    Convenience function to extract text from any document.

    Args:
        document_path: Path to document file
        config: Optional configuration

    Returns:
        UniversalTextResult with full extraction results
    """
    extractor = UniversalTextExtractor(config or {})
    return await extractor.extract(document_path)
