# ============================================================================
# src/medical_ingestion/core/document_pipeline.py
# ============================================================================
"""
Enhanced Document Processing Pipeline

Integrates:
1. OpenCV Preprocessing (deskew, denoise, contrast)
2. Region Detection (OpenCV + optional VLM)
3. Layout-Guided OCR Routing
4. Existing Medical Processing (MedGemma, processors)

Pipeline Flow:
    Document → Preprocess → Detect Regions → Route OCR → Extract → Process
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import asyncio

from ..preprocessors.image_preprocessor import ImagePreprocessor, PreprocessingResult
from ..extractors.region_detector import RegionDetector, RegionDetectionResult, RegionType
from ..extractors.ocr_router import OCRRouter, DocumentOCRResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from the document processing pipeline."""
    # Preprocessing
    preprocessing: PreprocessingResult

    # Region detection
    regions: RegionDetectionResult

    # OCR results
    ocr_result: DocumentOCRResult

    # Combined output
    full_text: str
    document_type_hint: Optional[str]

    # Metadata
    total_regions: int
    has_handwriting: bool
    has_tables: bool
    average_confidence: float

    # For downstream processing
    region_texts: Dict[str, str]  # region_type -> concatenated text


class DocumentPipeline:
    """
    Enhanced document processing pipeline with preprocessing,
    region detection, and intelligent OCR routing.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.preprocessor = ImagePreprocessor(self.config.get('preprocessing', {}))
        self.region_detector = RegionDetector(self.config.get('region_detection', {}))
        self.ocr_router = OCRRouter(self.config.get('ocr', {}))

        # Pipeline options
        self.enable_preprocessing = self.config.get('enable_preprocessing', True)
        self.enable_region_detection = self.config.get('enable_region_detection', True)
        self.use_vlm_classification = self.config.get('use_vlm_classification', False)

    async def process(self, image_path: Path) -> PipelineResult:
        """
        Process a document image through the full pipeline.

        Args:
            image_path: Path to image file

        Returns:
            PipelineResult with all processing outputs
        """
        logger.info(f"Starting document pipeline for: {image_path}")

        # Load image with EXIF orientation correction.
        # cv2.imread() ignores EXIF tags, so iPhone/Android portrait photos
        # appear sideways. Use load_image_for_ocr() to correct orientation,
        # then convert the PIL Image to a numpy array for OpenCV processing.
        try:
            from ..utils.image_utils import load_image_for_ocr
            pil_image = load_image_for_ocr(image_path, max_dimension=4000, ensure_rgb=True)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"load_image_for_ocr failed, falling back to cv2.imread: {e}")
            image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return await self.process_image(image)

    async def process_image(self, image: np.ndarray) -> PipelineResult:
        """
        Process an image array through the full pipeline.

        Args:
            image: Image as numpy array (BGR)

        Returns:
            PipelineResult with all processing outputs
        """
        # Step 1: Preprocessing
        if self.enable_preprocessing:
            logger.info("Step 1: Preprocessing image")
            preprocess_result = self.preprocessor.preprocess_array(image)
            processed_image = preprocess_result.image

            # Convert back to BGR for region detection if grayscale
            if len(processed_image.shape) == 2:
                processed_bgr = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            else:
                processed_bgr = processed_image
        else:
            preprocess_result = PreprocessingResult(
                image=image,
                original_size=(image.shape[1], image.shape[0]),
                processed_size=(image.shape[1], image.shape[0]),
                rotation_angle=0.0,
                operations_applied=[],
                quality_improved=False
            )
            processed_bgr = image
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Step 2: Region Detection
        if self.enable_region_detection:
            logger.info("Step 2: Detecting regions")
            region_result = await self.region_detector.detect_regions(
                processed_bgr,
                classify_with_vlm=self.use_vlm_classification
            )
        else:
            # Create a single region covering the whole image
            from ..extractors.region_detector import DetectedRegion, BoundingBox
            h, w = processed_bgr.shape[:2]
            full_page_region = DetectedRegion(
                region_type=RegionType.PRINTED_TEXT,
                bbox=BoundingBox(0, 0, 1, 1),
                confidence=1.0,
                text_density=1.0,
                has_lines=False,
                is_handwritten=False
            )
            region_result = RegionDetectionResult(
                regions=[full_page_region],
                page_width=w,
                page_height=h,
                document_type_hint=None,
                has_tables=False,
                has_handwriting=False,
                total_text_area=1.0
            )

        logger.info(f"Detected {len(region_result.regions)} regions")

        # Step 3: OCR Routing
        logger.info("Step 3: Running OCR on regions")
        ocr_result = await self.ocr_router.process_regions(processed_bgr, region_result)

        logger.info(f"OCR complete. Engines used: {ocr_result.engines_used}")

        # Step 4: Organize results
        region_texts = self._organize_region_texts(ocr_result)

        return PipelineResult(
            preprocessing=preprocess_result,
            regions=region_result,
            ocr_result=ocr_result,
            full_text=ocr_result.full_text,
            document_type_hint=region_result.document_type_hint,
            total_regions=len(region_result.regions),
            has_handwriting=region_result.has_handwriting,
            has_tables=region_result.has_tables,
            average_confidence=ocr_result.average_confidence,
            region_texts=region_texts
        )

    def _organize_region_texts(self, ocr_result: DocumentOCRResult) -> Dict[str, str]:
        """
        Organize OCR results by region type.

        Returns dict mapping region_type to concatenated text.
        """
        region_texts = {}

        for result in ocr_result.regions:
            rtype = result.region.region_type.value

            if rtype not in region_texts:
                region_texts[rtype] = []

            if result.text:
                region_texts[rtype].append(result.text)

        # Concatenate texts for each type
        return {k: '\n'.join(v) for k, v in region_texts.items()}

    async def process_multipage(
        self,
        image_paths: List[Path]
    ) -> List[PipelineResult]:
        """
        Process multiple pages of a document.

        All pages are assumed to be the same document type.

        Args:
            image_paths: List of paths to page images

        Returns:
            List of PipelineResult, one per page
        """
        results = []

        for i, path in enumerate(image_paths):
            logger.info(f"Processing page {i+1}/{len(image_paths)}")
            result = await self.process(path)
            results.append(result)

        return results

    async def process_pdf(self, pdf_path: Path) -> List[PipelineResult]:
        """
        Process a PDF document.

        Converts PDF pages to images and processes each.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PipelineResult, one per page
        """
        import fitz  # PyMuPDF

        results = []
        doc = fitz.open(str(pdf_path))

        for page_num in range(len(doc)):
            logger.info(f"Processing PDF page {page_num+1}/{len(doc)}")

            # Convert page to image
            page = doc.load_page(page_num)

            # Higher DPI for better OCR
            zoom = 2.0  # 2x zoom = ~144 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to numpy array
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img_data = img_data.reshape(pix.height, pix.width, pix.n)

            # Convert RGB to BGR for OpenCV
            if pix.n == 4:  # RGBA
                image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            # Process
            result = await self.process_image(image)
            results.append(result)

        doc.close()
        return results


# Convenience function
async def process_document(
    document_path: Path,
    config: Dict = None
) -> List[PipelineResult]:
    """
    Convenience function to process a document.

    Handles both images and PDFs.

    Args:
        document_path: Path to document file
        config: Optional configuration

    Returns:
        List of PipelineResult (one per page)
    """
    pipeline = DocumentPipeline(config or {})

    suffix = document_path.suffix.lower()

    if suffix == '.pdf':
        return await pipeline.process_pdf(document_path)
    elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        result = await pipeline.process(document_path)
        return [result]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def combine_multipage_results(results: List[PipelineResult]) -> Dict[str, Any]:
    """
    Combine results from multiple pages into a single document result.

    Args:
        results: List of PipelineResult from each page

    Returns:
        Combined document data
    """
    all_text = []
    all_regions = []
    total_confidence = 0
    has_handwriting = False
    has_tables = False

    for i, result in enumerate(results):
        # Add page marker
        all_text.append(f"--- Page {i+1} ---")
        all_text.append(result.full_text)

        # Collect regions with page info
        for ocr_result in result.ocr_result.regions:
            all_regions.append({
                'page': i + 1,
                'type': ocr_result.region.region_type.value,
                'text': ocr_result.text,
                'confidence': ocr_result.confidence,
                'bbox': [
                    ocr_result.region.bbox.x1,
                    ocr_result.region.bbox.y1,
                    ocr_result.region.bbox.x2,
                    ocr_result.region.bbox.y2
                ]
            })

        total_confidence += result.average_confidence
        has_handwriting = has_handwriting or result.has_handwriting
        has_tables = has_tables or result.has_tables

    return {
        'full_text': '\n'.join(all_text),
        'regions': all_regions,
        'page_count': len(results),
        'average_confidence': total_confidence / len(results) if results else 0,
        'has_handwriting': has_handwriting,
        'has_tables': has_tables,
        'document_type_hint': results[0].document_type_hint if results else None
    }
