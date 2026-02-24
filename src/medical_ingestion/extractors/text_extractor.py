# src/medical_ingestion/extractors/text_extractor.py
"""
Text extraction from PDFs and images.

Extraction cascade (in order of preference):
1. pypdfium2: Fast, good Unicode support, best for modern PDFs
2. PyPDF2: Fallback, widely compatible
3. pdfplumber: Layout-preserving extraction
4. OCR: For image files and scanned PDFs

Handles:
- Encrypted PDFs (with password)
- Mixed content (text + image pages)
- Various PDF encodings
- Image files (PNG, JPG, JPEG, TIFF, BMP, WEBP)
"""

from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import mimetypes

import pypdfium2
import PyPDF2
import pdfplumber

# Image file extensions that should be routed to OCR
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.gif'}


@dataclass
class ExtractedText:
    """Text extracted from a single page."""
    page_number: int
    text: str
    layout_preserved: bool = False
    char_count: int = 0
    method: str = "unknown"


@dataclass
class TextExtractionResult:
    """Complete text extraction result with metadata."""
    text: str
    pages: List[ExtractedText] = field(default_factory=list)
    method: str = "unknown"  # Primary method used
    confidence: float = 0.0
    page_count: int = 0
    needs_ocr: bool = False  # Some pages appear image-only
    pages_needing_ocr: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TextExtractor:
    """
    Robust text extraction with multiple fallback methods.

    Primary: pypdfium2 (fastest, best Unicode)
    Fallback: PyPDF2 → pdfplumber

    Detects pages that need OCR (low text content).
    """

    # Minimum chars per page to consider it text-based (not scanned)
    MIN_CHARS_PER_PAGE = 50

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text(
        self,
        pdf_path: Path,
        preserve_layout: bool = False,
        password: Optional[str] = None
    ) -> str:
        """
        Extract text from PDF, optionally preserving layout.

        This is the simple interface for backward compatibility.
        For detailed results, use extract_text_detailed().

        Args:
            pdf_path: Path to PDF file
            preserve_layout: Use layout-preserving extraction (slower)
            password: Password for encrypted PDFs

        Returns:
            Extracted text as string
        """
        result = self.extract_text_detailed(pdf_path, preserve_layout, password)

        if result.errors:
            # If there are errors, raise the first one
            raise RuntimeError(result.errors[0])

        return result.text

    def extract_text_detailed(
        self,
        file_path: Path,
        preserve_layout: bool = False,
        password: Optional[str] = None
    ) -> TextExtractionResult:
        """
        Extract text with detailed metadata and error handling.

        Supports both PDF files and image files. Image files are automatically
        routed to OCR extraction.

        Args:
            file_path: Path to PDF or image file
            preserve_layout: Use layout-preserving extraction (PDF only)
            password: Password for encrypted PDFs

        Returns:
            TextExtractionResult with full metadata
        """
        file_path = Path(file_path)
        result = TextExtractionResult(text="")

        self.logger.debug(f"Extracting text from {file_path}")

        # Check if this is an image file
        if self._is_image_file(file_path):
            self.logger.info(f"Detected image file: {file_path.suffix}, using OCR extraction")
            return self._extract_from_image(file_path)

        # PDF extraction flow
        # Try pypdfium2 first (fastest, best Unicode)
        try:
            text, pages = self._extract_with_pypdfium2(file_path, password)
            result.text = text
            result.pages = pages
            result.method = "pypdfium2"
            result.page_count = len(pages)
            self.logger.debug(f"pypdfium2 extracted {len(text)} chars from {len(pages)} pages")

        except Exception as e:
            self.logger.warning(f"pypdfium2 failed, trying PyPDF2: {e}")
            result.warnings.append(f"pypdfium2 failed: {e}")

            # Try PyPDF2 as fallback
            try:
                text, pages = self._extract_with_pypdf2(file_path, password)
                result.text = text
                result.pages = pages
                result.method = "pypdf2"
                result.page_count = len(pages)
                self.logger.debug(f"PyPDF2 extracted {len(text)} chars from {len(pages)} pages")

            except Exception as e2:
                self.logger.error(f"PyPDF2 also failed: {e2}")

                # Check if this might be an image file with wrong extension
                if self._try_as_image(file_path):
                    self.logger.info("File appears to be an image, trying OCR extraction")
                    return self._extract_from_image(file_path)

                result.errors.append(f"All extraction methods failed. pypdfium2: {e}, PyPDF2: {e2}")
                return result

        # Use pdfplumber for layout preservation if requested
        if preserve_layout and result.pages:
            try:
                self._enhance_with_layout(file_path, result, password)
            except Exception as e:
                result.warnings.append(f"Layout preservation failed: {e}")

        # Detect pages that may need OCR
        self._detect_ocr_needed(result)

        # Calculate confidence based on text content
        result.confidence = self._calculate_confidence(result)

        return result

    def _is_image_file(self, file_path: Path) -> bool:
        """Check if file is an image based on extension."""
        return file_path.suffix.lower() in IMAGE_EXTENSIONS

    def _try_as_image(self, file_path: Path) -> bool:
        """
        Try to detect if a file is actually an image by checking magic bytes.
        Used when PDF extraction fails to check if it might be a mislabeled image.
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)

            # Check for common image magic bytes
            # PNG: 89 50 4E 47
            if header.startswith(b'\x89PNG'):
                return True
            # JPEG: FF D8 FF
            if header.startswith(b'\xff\xd8\xff'):
                return True
            # GIF: GIF87a or GIF89a
            if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                return True
            # TIFF: 49 49 2A 00 (little-endian) or 4D 4D 00 2A (big-endian)
            if header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
                return True
            # BMP: 42 4D
            if header.startswith(b'BM'):
                return True
            # WebP: RIFF....WEBP
            if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
                return True

            return False
        except Exception:
            return False

    def _extract_from_image(
        self,
        image_path: Path,
        check_handwriting: bool = True
    ) -> TextExtractionResult:
        """
        Extract text from an image file using OCR.

        Automatically detects handwriting and uses specialized OCR if needed.

        Args:
            image_path: Path to image file
            check_handwriting: Whether to check for handwriting and use specialized OCR
        """
        from .ocr_extractor import OCRExtractor

        result = TextExtractionResult(text="")

        try:
            # Check for handwriting if quality analyzer is available
            is_handwritten = False
            quality_report = None

            if check_handwriting:
                try:
                    from ..utils.image_quality import ImageQualityAnalyzer
                    analyzer = ImageQualityAnalyzer()
                    quality_report = analyzer.analyze(image_path)
                    # Convert numpy bool to Python bool
                    is_handwritten = bool(quality_report.is_likely_handwritten)

                    if is_handwritten:
                        self.logger.info("Handwriting detected, using specialized OCR")
                        result.warnings.append("Handwriting detected - using specialized OCR")

                    if not bool(quality_report.can_process):
                        result.warnings.append(f"Low quality image: {quality_report.summary}")

                except ImportError:
                    self.logger.debug("Image quality analyzer not available")
                except Exception as e:
                    self.logger.debug(f"Quality analysis failed: {e}")

            ocr = OCRExtractor()

            # Use handwriting-specialized OCR if handwriting detected
            if is_handwritten:
                ocr_result = ocr.extract_handwritten_text(image_path, quality_report)
            else:
                ocr_result = ocr.extract_text_from_image(image_path)

            if ocr_result:
                result.text = ocr_result.text
                result.pages = [ExtractedText(
                    page_number=0,
                    text=ocr_result.text,
                    layout_preserved=False,
                    char_count=len(ocr_result.text),
                    method=f"ocr_{ocr_result.method}"
                )]
                result.method = f"ocr_{ocr_result.method}"
                result.page_count = 1
                result.confidence = ocr_result.confidence
                self.logger.info(f"OCR extracted {len(result.text)} chars from image")

                # Add handwriting note to method if applicable
                if is_handwritten:
                    result.method = f"ocr_handwriting_{ocr_result.method}"
            else:
                result.errors.append("OCR extraction returned no results")

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            result.errors.append(f"OCR extraction failed: {e}")

        return result

    def _extract_with_pypdfium2(
        self,
        pdf_path: Path,
        password: Optional[str] = None
    ) -> Tuple[str, List[ExtractedText]]:
        """Extract text using pypdfium2."""
        pdf = pypdfium2.PdfDocument(str(pdf_path), password=password)

        pages = []
        all_text = []

        for page_num in range(len(pdf)):
            page = pdf[page_num]

            try:
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                text = text.strip() if text else ""
            except Exception:
                text = ""

            pages.append(ExtractedText(
                page_number=page_num,
                text=text,
                layout_preserved=False,
                char_count=len(text),
                method="pypdfium2"
            ))
            all_text.append(text)

        pdf.close()

        return "\n\n".join(all_text), pages

    def _extract_with_pypdf2(
        self,
        pdf_path: Path,
        password: Optional[str] = None
    ) -> Tuple[str, List[ExtractedText]]:
        """Extract text using PyPDF2."""
        pages = []
        all_text = []

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Handle encryption
            if reader.is_encrypted:
                if password:
                    reader.decrypt(password)
                else:
                    # Try empty password
                    try:
                        reader.decrypt("")
                    except Exception:
                        raise RuntimeError("PDF is encrypted and requires a password")

            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                pages.append(ExtractedText(
                    page_number=page_num,
                    text=text,
                    layout_preserved=False,
                    char_count=len(text),
                    method="pypdf2"
                ))
                all_text.append(text)

        return "\n\n".join(all_text), pages

    def _enhance_with_layout(
        self,
        pdf_path: Path,
        result: TextExtractionResult,
        password: Optional[str] = None
    ):
        """Enhance extraction with layout preservation using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path, password=password) as pdf:
                enhanced_pages = []
                all_text = []

                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text(layout=True) or ""
                    except Exception:
                        # Fall back to original extraction
                        if page_num < len(result.pages):
                            text = result.pages[page_num].text
                        else:
                            text = ""

                    enhanced_pages.append(ExtractedText(
                        page_number=page_num,
                        text=text,
                        layout_preserved=True,
                        char_count=len(text),
                        method="pdfplumber"
                    ))
                    all_text.append(text)

                result.pages = enhanced_pages
                result.text = "\n\n".join(all_text)
                result.method = "pdfplumber"

        except Exception as e:
            self.logger.warning(f"pdfplumber layout extraction failed: {e}")
            raise

    def _detect_ocr_needed(self, result: TextExtractionResult):
        """Detect pages that appear to be scanned (need OCR)."""
        for page in result.pages:
            if page.char_count < self.MIN_CHARS_PER_PAGE:
                result.pages_needing_ocr.append(page.page_number)

        if result.pages_needing_ocr:
            result.needs_ocr = True
            ratio = len(result.pages_needing_ocr) / result.page_count if result.page_count else 0
            result.warnings.append(
                f"{len(result.pages_needing_ocr)}/{result.page_count} pages "
                f"({ratio:.0%}) appear to need OCR"
            )

    def _calculate_confidence(self, result: TextExtractionResult) -> float:
        """Calculate confidence score based on extraction results."""
        if not result.pages:
            return 0.0

        # Base confidence by method
        method_confidence = {
            "pypdfium2": 0.95,
            "pypdf2": 0.90,
            "pdfplumber": 0.92
        }

        base = method_confidence.get(result.method, 0.8)

        # Reduce confidence if many pages need OCR
        if result.page_count > 0:
            ocr_ratio = len(result.pages_needing_ocr) / result.page_count
            ocr_penalty = ocr_ratio * 0.3  # Max 30% penalty

            base -= ocr_penalty

        return max(0.1, min(1.0, base))

    def extract_text_by_page(
        self,
        pdf_path: Path,
        password: Optional[str] = None
    ) -> List[ExtractedText]:
        """
        Extract text page by page with metadata.

        Args:
            pdf_path: Path to PDF file
            password: Password for encrypted PDFs

        Returns:
            List of ExtractedText objects
        """
        result = self.extract_text_detailed(pdf_path, password=password)
        return result.pages

    # Backward compatibility alias
    def _extract_text_with_layout(self, pdf_path: Path, page_num: int) -> str:
        """Preserve layout using pdfplumber. Legacy method for compatibility."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    return page.extract_text(layout=True) or ""
        except Exception as e:
            self.logger.warning(f"Layout extraction failed for page {page_num}: {e}")
        return ""
