# ============================================================================
# src/medical_ingestion/extractors/ocr_router.py
# ============================================================================
"""
OCR Router - Routes document regions to appropriate OCR engines

Region Type → OCR Engine mapping:
- printed_text → PaddleOCR (fast, good for clean printed text)
- handwriting → TrOCR or specialized handwriting OCR
- table → Table-aware extraction (PaddleOCR + structure)
- form_field → Field-aware extraction
- header/footer → PaddleOCR
- stamp/signature → Skip or flag for manual review
- noise → Skip
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import asyncio

from .region_detector import DetectedRegion, RegionType, BoundingBox, RegionDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR on a single region."""
    region: DetectedRegion
    text: str
    confidence: float
    word_boxes: List[Dict[str, Any]]  # Individual word bounding boxes
    ocr_engine: str


@dataclass
class DocumentOCRResult:
    """Combined OCR result for entire document."""
    regions: List[OCRResult]
    full_text: str  # Concatenated text in reading order
    average_confidence: float
    has_low_confidence_regions: bool
    engines_used: List[str]


class OCRRouter:
    """
    Routes document regions to appropriate OCR engines based on region type.

    Supports:
    - PaddleOCR for printed text
    - TrOCR for handwriting (HuggingFace)
    - Table-specific extraction
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # OCR engine instances (lazy loaded)
        self._paddle_ocr = None
        self._trocr = None

        # Configuration
        self.min_confidence = self.config.get('min_ocr_confidence', 0.5)
        self.use_trocr_for_handwriting = self.config.get('use_trocr', True)

    async def process_regions(
        self,
        image: np.ndarray,
        detection_result: RegionDetectionResult
    ) -> DocumentOCRResult:
        """
        Process all detected regions with appropriate OCR.

        Args:
            image: Full page image
            detection_result: Result from region detection

        Returns:
            DocumentOCRResult with all OCR results
        """
        h, w = image.shape[:2]
        ocr_results = []
        engines_used = set()

        # Sort regions by position (top to bottom, left to right) for reading order
        sorted_regions = sorted(
            detection_result.regions,
            key=lambda r: (r.bbox.y1, r.bbox.x1)
        )

        for region in sorted_regions:
            # Extract region image
            px = region.bbox.to_pixels(w, h)
            roi = image[px[1]:px[3], px[0]:px[2]]

            if roi.size == 0:
                continue

            # Route to appropriate OCR
            result = await self._ocr_region(roi, region)

            if result:
                ocr_results.append(result)
                engines_used.add(result.ocr_engine)

        # Combine results
        full_text = self._combine_text(ocr_results)
        avg_confidence = (
            sum(r.confidence for r in ocr_results) / len(ocr_results)
            if ocr_results else 0.0
        )
        has_low_conf = any(r.confidence < self.min_confidence for r in ocr_results)

        return DocumentOCRResult(
            regions=ocr_results,
            full_text=full_text,
            average_confidence=avg_confidence,
            has_low_confidence_regions=has_low_conf,
            engines_used=list(engines_used)
        )

    async def _ocr_region(
        self,
        roi: np.ndarray,
        region: DetectedRegion
    ) -> Optional[OCRResult]:
        """
        Run OCR on a single region.

        Routing strategy:
        - HANDWRITING regions → TrOCR (specialized for handwritten text)
        - PRINTED_TEXT, HEADER, FOOTER → PaddleOCR (best for clean printed text)
        - TABLE → PaddleOCR with cell detection
        - FORM_FIELD → PaddleOCR
        - STAMP/SIGNATURE → Skip
        """
        # Skip noise and blank regions
        if region.region_type in [RegionType.NOISE, RegionType.IMAGE]:
            return None

        # Skip stamps/signatures but flag them
        if region.region_type in [RegionType.STAMP, RegionType.SIGNATURE]:
            return OCRResult(
                region=region,
                text="[STAMP/SIGNATURE]",
                confidence=0.0,
                word_boxes=[],
                ocr_engine="skipped"
            )

        # HANDWRITING → TrOCR (specialized model)
        if region.region_type == RegionType.HANDWRITING:
            if self.use_trocr_for_handwriting:
                return await self._ocr_with_trocr(roi, region)
            else:
                return await self._ocr_with_paddle(roi, region)

        # TABLE → PaddleOCR with cell structure detection
        if region.region_type == RegionType.TABLE:
            return await self._ocr_table(roi, region)

        # PRINTED_TEXT, HEADER, FOOTER, FORM_FIELD → PaddleOCR
        # PaddleOCR PP-OCRv5 is excellent for printed text
        return await self._ocr_with_paddle(roi, region)

    async def _ocr_with_paddle(
        self,
        roi: np.ndarray,
        region: DetectedRegion
    ) -> OCRResult:
        """
        OCR using PaddleOCR PP-OCRv5.

        Good for: printed text, forms, headers/footers
        """
        import tempfile
        import os

        if self._paddle_ocr is None:
            self._paddle_ocr = self._load_paddle_ocr()

        try:
            # PaddleOCR predict() works with file paths, so save ROI temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                from PIL import Image
                # Convert BGR to RGB for PIL
                if len(roi.shape) == 3:
                    pil_image = Image.fromarray(roi[:, :, ::-1])
                else:
                    pil_image = Image.fromarray(roi)
                pil_image.save(f, format='PNG')
                temp_path = f.name

            try:
                # Run PaddleOCR predict
                result = self._paddle_ocr.predict(input=temp_path)
            finally:
                os.unlink(temp_path)

            if not result:
                return OCRResult(
                    region=region,
                    text="",
                    confidence=0.0,
                    word_boxes=[],
                    ocr_engine="paddleocr"
                )

            # Extract text and boxes from new API format
            text_parts = []
            word_boxes = []
            confidences = []
            h, w = roi.shape[:2]

            for res in result:
                # New API returns dict-like OCRResult objects
                rec_texts = res.get('rec_texts', []) if hasattr(res, 'get') else getattr(res, 'rec_texts', [])
                rec_scores = res.get('rec_scores', []) if hasattr(res, 'get') else getattr(res, 'rec_scores', [])
                dt_polys = res.get('dt_polys', []) if hasattr(res, 'get') else getattr(res, 'dt_polys', [])

                for i, text in enumerate(rec_texts):
                    if text and text.strip():
                        text_parts.append(text.strip())

                        conf = float(rec_scores[i]) if i < len(rec_scores) else 0.9
                        confidences.append(conf)

                        # Get bounding box
                        if dt_polys and i < len(dt_polys):
                            poly = dt_polys[i]
                            x_coords = [p[0] for p in poly]
                            y_coords = [p[1] for p in poly]
                            bbox = [
                                min(x_coords) / w,
                                min(y_coords) / h,
                                max(x_coords) / w,
                                max(y_coords) / h
                            ]
                            word_boxes.append({
                                'text': text.strip(),
                                'confidence': conf,
                                'bbox': bbox
                            })

            full_text = ' '.join(text_parts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                region=region,
                text=full_text,
                confidence=avg_conf,
                word_boxes=word_boxes,
                ocr_engine="paddleocr"
            )

        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
            return OCRResult(
                region=region,
                text="",
                confidence=0.0,
                word_boxes=[],
                ocr_engine="paddleocr_error"
            )

    async def _ocr_with_trocr(
        self,
        roi: np.ndarray,
        region: DetectedRegion
    ) -> OCRResult:
        """
        OCR using TrOCR for all text regions.

        TrOCR is a transformer-based OCR model that is more robust
        than PaddleOCR on degraded/faxed documents. While originally
        designed for handwriting, it handles printed text well too,
        especially on noisy, skewed, or low-quality scans.
        """
        if self._trocr is None:
            self._trocr = await self._load_trocr()

        if self._trocr is None:
            logger.error("TrOCR not available - cannot process region")
            return OCRResult(
                region=region,
                text="",
                confidence=0.0,
                word_boxes=[],
                ocr_engine="trocr_unavailable"
            )

        try:
            from PIL import Image

            # Convert to PIL
            if len(roi.shape) == 2:
                pil_image = Image.fromarray(roi).convert('RGB')
            else:
                pil_image = Image.fromarray(roi[:, :, ::-1])

            # Process with TrOCR
            processor, model = self._trocr
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values

            import torch
            device = next(model.parameters()).device
            pixel_values = pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=128)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return OCRResult(
                region=region,
                text=text,
                confidence=0.85,  # TrOCR is robust on degraded docs
                word_boxes=[],
                ocr_engine="trocr"
            )

        except Exception as e:
            logger.error(f"TrOCR failed: {e}")
            return OCRResult(
                region=region,
                text="",
                confidence=0.0,
                word_boxes=[],
                ocr_engine="trocr_error"
            )

    async def _ocr_table(
        self,
        roi: np.ndarray,
        region: DetectedRegion
    ) -> OCRResult:
        """
        OCR for table regions.

        Uses PaddleOCR but with table structure awareness.
        """
        if self._paddle_ocr is None:
            self._paddle_ocr = self._load_paddle_ocr()

        try:
            # For tables, we want to preserve structure
            # First, detect cells using line detection
            cells = self._detect_table_cells(roi)

            if not cells:
                # Fall back to standard OCR
                return await self._ocr_with_paddle(roi, region)

            # OCR each cell
            cell_texts = []
            h, w = roi.shape[:2]

            for cell_box in cells:
                x1, y1, x2, y2 = cell_box
                cell_roi = roi[y1:y2, x1:x2]

                if cell_roi.size == 0:
                    continue

                result = self._paddle_ocr.ocr(cell_roi)

                if result and result[0]:
                    text = ' '.join([line[1][0] for line in result[0] if line])
                    cell_texts.append({
                        'text': text,
                        'bbox': [x1/w, y1/h, x2/w, y2/h]
                    })

            # Combine cell texts
            full_text = '\n'.join([c['text'] for c in cell_texts])

            return OCRResult(
                region=region,
                text=full_text,
                confidence=0.8,
                word_boxes=cell_texts,
                ocr_engine="paddleocr_table"
            )

        except Exception as e:
            logger.warning(f"Table OCR failed: {e}")
            return await self._ocr_with_paddle(roi, region)

    async def _ocr_table_with_trocr(
        self,
        roi: np.ndarray,
        region: DetectedRegion
    ) -> OCRResult:
        """
        OCR for table regions using TrOCR.

        More robust than PaddleOCR on degraded/faxed documents.
        Detects cell structure, then OCRs each cell with TrOCR.
        """
        try:
            # First, detect cells using line detection
            cells = self._detect_table_cells(roi)

            if not cells:
                # No table structure found, OCR the whole region
                return await self._ocr_with_trocr(roi, region)

            # Load TrOCR if needed
            if self._trocr is None:
                self._trocr = await self._load_trocr()

            if self._trocr is None:
                # Fallback to full region TrOCR
                return await self._ocr_with_trocr(roi, region)

            from PIL import Image
            import torch

            processor, model = self._trocr
            device = next(model.parameters()).device

            # OCR each cell with TrOCR
            cell_texts = []
            h, w = roi.shape[:2]

            for cell_box in cells:
                x1, y1, x2, y2 = cell_box
                cell_roi = roi[y1:y2, x1:x2]

                if cell_roi.size == 0:
                    continue

                # Convert to PIL
                if len(cell_roi.shape) == 2:
                    pil_image = Image.fromarray(cell_roi).convert('RGB')
                else:
                    pil_image = Image.fromarray(cell_roi[:, :, ::-1])

                # Process with TrOCR
                pixel_values = processor(pil_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                with torch.no_grad():
                    generated_ids = model.generate(pixel_values, max_length=64)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                if text.strip():
                    cell_texts.append({
                        'text': text.strip(),
                        'bbox': [x1/w, y1/h, x2/w, y2/h]
                    })

            # Combine cell texts
            full_text = '\n'.join([c['text'] for c in cell_texts])

            return OCRResult(
                region=region,
                text=full_text,
                confidence=0.85,
                word_boxes=cell_texts,
                ocr_engine="trocr_table"
            )

        except Exception as e:
            logger.warning(f"Table TrOCR failed: {e}, falling back to region TrOCR")
            return await self._ocr_with_trocr(roi, region)

    def _detect_table_cells(self, roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect table cells using line detection.

        Returns list of cell bounding boxes (x1, y1, x2, y2).
        """
        gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect lines
        edges = cv2.Canny(gray, 50, 150)

        # Horizontal lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)

        # Vertical lines
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)

        # Find line positions
        h_lines = self._find_line_positions(horizontal, axis=0)
        v_lines = self._find_line_positions(vertical, axis=1)

        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        # Create cells from grid intersections
        cells = []
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                x1 = v_lines[j]
                y1 = h_lines[i]
                x2 = v_lines[j + 1]
                y2 = h_lines[i + 1]

                # Filter out very small cells
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    cells.append((x1, y1, x2, y2))

        return cells

    def _find_line_positions(
        self,
        line_image: np.ndarray,
        axis: int
    ) -> List[int]:
        """
        Find positions of lines along an axis.

        axis=0: horizontal lines (find y positions)
        axis=1: vertical lines (find x positions)
        """
        projection = np.sum(line_image, axis=axis)

        # Find peaks in projection
        threshold = np.max(projection) * 0.3
        positions = []

        in_peak = False
        peak_start = 0

        for i, val in enumerate(projection):
            if val > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif val <= threshold and in_peak:
                in_peak = False
                positions.append((peak_start + i) // 2)

        return positions

    def _normalize_box(
        self,
        box: List[List[float]],
        width: int,
        height: int
    ) -> List[float]:
        """Convert PaddleOCR box format to normalized [x1, y1, x2, y2]."""
        # PaddleOCR box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]

        return [
            min(x_coords) / width,
            min(y_coords) / height,
            max(x_coords) / width,
            max(y_coords) / height
        ]

    def _load_paddle_ocr(self):
        """Load PaddleOCR instance."""
        try:
            from paddleocr import PaddleOCR
            import os

            # Suppress PaddleOCR verbose output
            os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

            # Note: use_angle_cls and use_textline_orientation are mutually exclusive
            # in newer PaddleOCR versions. We use use_angle_cls for text direction.
            return PaddleOCR(
                lang='en',
                use_angle_cls=True,  # Enable text direction classification
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            raise

    async def _load_trocr(self):
        """Load TrOCR model for handwriting."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            model_name = self.config.get(
                'trocr_model',
                'microsoft/trocr-base-handwritten'
            )

            logger.info(f"Loading TrOCR model: {model_name}")

            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)

            # Move to appropriate device
            if torch.cuda.is_available():
                model = model.to('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                model = model.to('mps')

            model.eval()

            logger.info("TrOCR loaded successfully")
            return (processor, model)

        except ImportError:
            logger.warning("TrOCR dependencies not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to load TrOCR: {e}")
            return None

    def _combine_text(self, results: List[OCRResult]) -> str:
        """
        Combine OCR results into full text, respecting reading order.
        """
        # Results should already be sorted by position
        text_parts = []

        for result in results:
            if result.text and result.text != "[STAMP/SIGNATURE]":
                text_parts.append(result.text)

        return '\n'.join(text_parts)


# Convenience function
async def ocr_document_regions(
    image: np.ndarray,
    detection_result: RegionDetectionResult,
    config: Dict = None
) -> DocumentOCRResult:
    """
    Convenience function to OCR all regions in a document.

    Args:
        image: Full page image
        detection_result: Result from region detection
        config: Optional configuration

    Returns:
        DocumentOCRResult with all OCR results
    """
    router = OCRRouter(config or {})
    return await router.process_regions(image, detection_result)
