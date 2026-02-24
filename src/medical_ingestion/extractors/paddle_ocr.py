# ============================================================================
# src/medical_ingestion/extractors/paddle_ocr.py
# ============================================================================
"""
PaddleOCR Extractor for Document Text Extraction

Uses the actual PaddleOCR library for high-quality OCR extraction.
This provides layout-aware text extraction from document images.

Installation:
    pip install paddlepaddle paddleocr

Key capabilities:
- High accuracy text recognition
- Layout-aware extraction
- Table detection
- Multi-language support
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for running sync PaddleOCR in async context
_executor = ThreadPoolExecutor(max_workers=2)

# Module-level singleton — shared across all callers
_singleton_instance: Optional['PaddleOCRExtractor'] = None


def get_paddle_ocr_extractor(config: Optional[Dict[str, Any]] = None) -> 'PaddleOCRExtractor':
    """Get the singleton PaddleOCRExtractor instance (creates on first call)."""
    global _singleton_instance
    if _singleton_instance is None:
        _singleton_instance = PaddleOCRExtractor(config)
    return _singleton_instance


class PaddleOCRExtractor:
    """
    PaddleOCR-based text extractor for medical documents.

    Uses PaddlePaddle's OCR model for accurate text extraction.
    Runs synchronously but can be called from async code.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        self._ocr = None
        self._initialized = False

        # Configuration
        # Note: GPU is auto-detected by PaddlePaddle based on installation
        # (paddlepaddle vs paddlepaddle-gpu). This setting is kept for
        # compatibility and future use when moving to GPU-enabled machines.
        self.use_gpu = config.get('use_gpu', False)
        self.lang = config.get('ocr_lang', 'en')

        # Statistics
        self._inference_count = 0

        logger.info(f"PaddleOCR extractor created (gpu_config={self.use_gpu}, lazy init)")

    def _ensure_initialized(self):
        """Lazily initialize PaddleOCR on first use."""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            # PaddleOCR 3.4+ API — show_log removed, GPU auto-detected
            # orientation classify enabled to handle rotated camera photos
            self._ocr = PaddleOCR(
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                lang=self.lang,
            )
            self._initialized = True
            logger.info("PaddleOCR initialized successfully")

        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Run: "
                "pip install paddlepaddle paddleocr"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def extract_from_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract text from an image file.

        Args:
            image_path: Path to image file (PNG, JPG, etc.)

        Returns:
            Dict with:
                - text: Full extracted text
                - lines: List of text lines with bounding boxes
                - tables: Detected table structures (if any)
        """
        self._ensure_initialized()

        try:
            # Prepare image: EXIF orientation, HEIC conversion, resize for OCR
            import tempfile, os
            from ..utils.image_utils import load_image_for_ocr
            prepared = load_image_for_ocr(image_path)
            # PaddleOCR needs a file path — save the corrected image to a temp file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                prepared.save(f, format='PNG')
                corrected_path = f.name

            try:
                result = self._ocr.predict(input=corrected_path)
            finally:
                os.unlink(corrected_path)

            lines = []
            full_text_parts = []

            # Process results — PaddleOCR 3.4+ returns dict-like OCRResult
            # (supports .get() but NOT attribute access like res.rec_texts)
            for res in result:
                rec_texts = res.get('rec_texts', []) if hasattr(res, 'get') else getattr(res, 'rec_texts', [])
                rec_scores = res.get('rec_scores', []) if hasattr(res, 'get') else getattr(res, 'rec_scores', [])
                dt_polys = res.get('dt_polys', []) if hasattr(res, 'get') else getattr(res, 'dt_polys', [])

                if rec_texts:
                    for i, text in enumerate(rec_texts):
                        if text and text.strip():
                            line_info = {
                                'text': text.strip(),
                                'confidence': float(rec_scores[i]) if i < len(rec_scores) else 1.0,
                            }

                            if dt_polys and i < len(dt_polys):
                                poly = dt_polys[i]
                                line_info['bbox'] = self._poly_to_bbox(poly)

                            lines.append(line_info)
                            full_text_parts.append(text.strip())

            self._inference_count += 1

            return {
                'text': '\n'.join(full_text_parts),
                'lines': lines,
                'line_count': len(lines)
            }

        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            raise

    def extract_from_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text from image bytes.

        Args:
            image_bytes: Raw image bytes (PNG, JPG)

        Returns:
            Dict with extracted text and lines
        """
        import tempfile
        import os

        # Write to temp file (PaddleOCR works with file paths)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(image_bytes)
            temp_path = Path(f.name)

        try:
            return self.extract_from_image(temp_path)
        finally:
            os.unlink(temp_path)

    async def extract_from_image_async(self, image_path: Path) -> Dict[str, Any]:
        """
        Async wrapper for extract_from_image.

        Runs PaddleOCR in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self.extract_from_image,
            image_path
        )

    async def extract_from_image_bytes_async(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Async wrapper for extract_from_image_bytes.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self.extract_from_image_bytes,
            image_bytes
        )

    def extract_structured_lab_results(
        self,
        image_path: Path
    ) -> Dict[str, Any]:
        """
        Extract and structure lab results from an image.

        Attempts to identify test names, values, units, and reference ranges
        from the OCR output.

        Args:
            image_path: Path to lab report image

        Returns:
            Dict with:
                - results: List of extracted lab values
                - raw_text: Full OCR text
                - confidence: Overall extraction confidence
        """
        import re

        # Get raw OCR output
        ocr_result = self.extract_from_image(image_path)
        lines = ocr_result.get('lines', [])

        results = []

        # Pattern for lab values: test_name value unit [reference_range] [flag]
        # Examples:
        #   "WBC 7.5 x10E3/uL 4.0-11.0"
        #   "Glucose 95 mg/dL 70-100"
        #   "Hemoglobin 14.2 g/dL 12.0-16.0 H"

        value_pattern = re.compile(
            r'^(.+?)\s+'           # Test name
            r'([\d.]+)\s*'         # Value
            r'([a-zA-Z%/0-9]+)?\s*'  # Unit (optional)
            r'([\d.]+\s*-\s*[\d.]+)?'  # Reference range (optional)
            r'\s*(H|L|HH|LL|High|Low)?$',  # Flag (optional)
            re.IGNORECASE
        )

        for line_info in lines:
            text = line_info.get('text', '')
            confidence = line_info.get('confidence', 1.0)

            match = value_pattern.match(text)
            if match:
                test_name = match.group(1).strip()
                value_str = match.group(2)
                unit = match.group(3) or ''
                ref_range = match.group(4) or ''
                flag = match.group(5) or ''

                try:
                    value = float(value_str)
                    results.append({
                        'test_name': test_name,
                        'value': value,
                        'unit': unit,
                        'reference_range': ref_range,
                        'flag': flag.upper() if flag else None,
                        'confidence': confidence
                    })
                except ValueError:
                    pass

        return {
            'results': results,
            'raw_text': ocr_result.get('text', ''),
            'line_count': len(lines),
            'result_count': len(results)
        }

    async def extract_structured_lab_results_async(
        self,
        image_path: Path
    ) -> Dict[str, Any]:
        """Async version of extract_structured_lab_results."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self.extract_structured_lab_results,
            image_path
        )

    def _poly_to_bbox(self, poly) -> Dict[str, float]:
        """Convert polygon points to bounding box."""
        try:
            if len(poly) >= 4:
                x_coords = [p[0] for p in poly]
                y_coords = [p[1] for p in poly]
                return {
                    'x1': min(x_coords),
                    'y1': min(y_coords),
                    'x2': max(x_coords),
                    'y2': max(y_coords)
                }
        except Exception:
            pass
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            'initialized': self._initialized,
            'inference_count': self._inference_count,
            'use_gpu': self.use_gpu,
            'language': self.lang
        }

    async def close(self):
        """Cleanup resources (no-op for PaddleOCR, but needed for interface compatibility)."""
        # PaddleOCR doesn't require explicit cleanup
        pass


async def convert_pdf_page_to_image(pdf_path: Path, page_num: int = 0) -> bytes:
    """
    Convert a PDF page to image bytes for OCR processing.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)

    Returns:
        PNG image bytes
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        page = doc[page_num]

        # Render at 150 DPI for good quality
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat)

        image_bytes = pix.tobytes("png")
        doc.close()

        return image_bytes

    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Run: pip install pymupdf")
        raise
    except Exception as e:
        logger.error(f"Failed to convert PDF page to image: {e}")
        raise


async def convert_pdf_to_images(pdf_path: Path) -> List[bytes]:
    """
    Convert all PDF pages to images.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of PNG image bytes, one per page
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))

        doc.close()
        return images

    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Run: pip install pymupdf")
        raise
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise
