# src/medical_ingestion/extractors/pdf_validator.py
"""
PDF Validation and Pre-flight Analysis

Performs checks before extraction:
- Encryption detection
- Corruption detection
- Scanned/image-only page detection
- Permission checks
- Metadata extraction
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

import pypdf
import pypdfium2


@dataclass
class ValidationResult:
    """Result of PDF validation."""
    is_valid: bool
    is_encrypted: bool
    is_decrypted: bool = False  # True if was encrypted but successfully decrypted
    is_scanned: bool = False  # Likely image-only (needs OCR)
    page_count: int = 0
    scanned_pages: List[int] = field(default_factory=list)  # Pages that appear image-only
    text_pages: List[int] = field(default_factory=list)  # Pages with extractable text
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFValidator:
    """
    Validates PDFs before extraction.

    Checks:
    - File exists and is readable
    - PDF is not corrupt
    - Encryption status
    - Scanned vs text-based pages
    - Basic metadata
    """

    # Minimum characters per page to consider it "text-based"
    MIN_CHARS_PER_PAGE = 50

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(
        self,
        pdf_path: Path,
        password: Optional[str] = None,
        check_scanned: bool = True
    ) -> ValidationResult:
        """
        Validate PDF and detect its characteristics.

        Args:
            pdf_path: Path to PDF file
            password: Optional password for encrypted PDFs
            check_scanned: Whether to check for scanned pages (slower)

        Returns:
            ValidationResult with all detected characteristics
        """
        pdf_path = Path(pdf_path)
        result = ValidationResult(
            is_valid=False,
            is_encrypted=False
        )

        # Check file exists
        if not pdf_path.exists():
            result.errors.append(f"File not found: {pdf_path}")
            return result

        if not pdf_path.is_file():
            result.errors.append(f"Not a file: {pdf_path}")
            return result

        # Try to open with pypdf (good encryption detection)
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)

                # Check encryption
                result.is_encrypted = reader.is_encrypted

                if result.is_encrypted:
                    if password:
                        try:
                            if reader.decrypt(password):
                                result.is_decrypted = True
                                self.logger.info("PDF decrypted successfully with provided password")
                            else:
                                result.errors.append("Incorrect password")
                                return result
                        except Exception as e:
                            result.errors.append(f"Decryption failed: {e}")
                            return result
                    else:
                        # Try empty password (some PDFs use this)
                        try:
                            if reader.decrypt(""):
                                result.is_decrypted = True
                                result.warnings.append("PDF was encrypted with empty password")
                            else:
                                result.errors.append("PDF is encrypted and requires a password")
                                return result
                        except Exception:
                            result.errors.append("PDF is encrypted and requires a password")
                            return result

                # Get page count and metadata
                result.page_count = len(reader.pages)
                result.metadata = self._extract_metadata(reader)

        except pypdf.errors.PdfReadError as e:
            result.errors.append(f"Corrupt or invalid PDF: {e}")
            return result
        except Exception as e:
            result.errors.append(f"Failed to read PDF: {e}")
            return result

        # Check for scanned pages using pypdfium2 (faster text extraction)
        if check_scanned:
            self._detect_scanned_pages(pdf_path, result, password)

        # Determine if mostly scanned
        if result.scanned_pages:
            scanned_ratio = len(result.scanned_pages) / result.page_count
            result.is_scanned = scanned_ratio > 0.5  # More than half pages are scanned
            if result.is_scanned:
                result.warnings.append(
                    f"{len(result.scanned_pages)}/{result.page_count} pages appear to be scanned/image-only"
                )

        result.is_valid = True
        return result

    def _extract_metadata(self, reader: pypdf.PdfReader) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {}

        try:
            if reader.metadata:
                for key in ['/Title', '/Author', '/Subject', '/Creator', '/Producer', '/CreationDate']:
                    value = reader.metadata.get(key)
                    if value:
                        # Clean up the key name
                        clean_key = key.lstrip('/')
                        metadata[clean_key] = str(value)
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata: {e}")

        return metadata

    def _detect_scanned_pages(
        self,
        pdf_path: Path,
        result: ValidationResult,
        password: Optional[str] = None
    ):
        """
        Detect which pages are scanned (image-only) vs text-based.

        Uses pypdfium2 for fast text extraction sampling.
        """
        try:
            pdf = pypdfium2.PdfDocument(str(pdf_path), password=password)

            for page_num in range(len(pdf)):
                page = pdf[page_num]

                # Try to extract text
                try:
                    textpage = page.get_textpage()
                    text = textpage.get_text_range()

                    # Clean and check length
                    text_clean = text.strip() if text else ""

                    if len(text_clean) < self.MIN_CHARS_PER_PAGE:
                        # Low text content - likely scanned
                        result.scanned_pages.append(page_num)
                    else:
                        result.text_pages.append(page_num)

                except Exception:
                    # If text extraction fails, assume scanned
                    result.scanned_pages.append(page_num)

            pdf.close()

        except Exception as e:
            self.logger.warning(f"Scanned page detection failed: {e}")
            result.warnings.append(f"Could not detect scanned pages: {e}")

    def quick_check(self, pdf_path: Path) -> bool:
        """
        Quick check if PDF is readable (no detailed analysis).

        Returns:
            True if PDF can be opened and read
        """
        try:
            pdf = pypdfium2.PdfDocument(str(pdf_path))
            page_count = len(pdf)
            pdf.close()
            return page_count > 0
        except Exception:
            return False

    def is_encrypted(self, pdf_path: Path) -> bool:
        """Check if PDF is encrypted."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                return reader.is_encrypted
        except Exception:
            return False

    def try_decrypt(
        self,
        pdf_path: Path,
        passwords: List[str]
    ) -> Optional[str]:
        """
        Try multiple passwords to decrypt a PDF.

        Args:
            pdf_path: Path to encrypted PDF
            passwords: List of passwords to try

        Returns:
            Working password or None if none work
        """
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)

                if not reader.is_encrypted:
                    return ""  # Not encrypted

                for password in passwords:
                    try:
                        if reader.decrypt(password):
                            return password
                    except Exception:
                        continue

        except Exception as e:
            self.logger.error(f"Decrypt attempt failed: {e}")

        return None
