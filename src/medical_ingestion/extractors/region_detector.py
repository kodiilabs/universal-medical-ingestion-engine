# ============================================================================
# src/medical_ingestion/extractors/region_detector.py
# ============================================================================
"""
Document Region Detector

Combines OpenCV-based layout detection with VLM classification:
1. OpenCV detects regions (text blocks, tables, etc.) with bounding boxes
2. PaliGemma/VLM classifies each region type
3. Output: regions with bboxes and types for OCR routing

Region Types:
- printed_text: Standard printed text
- handwriting: Handwritten content
- table: Tabular data
- form_field: Form input fields
- stamp: Stamps, seals
- signature: Signatures
- image: Photos, charts, diagrams
- noise: Artifacts to skip
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """Types of regions detected in documents."""
    PRINTED_TEXT = "printed_text"
    HANDWRITING = "handwriting"
    TABLE = "table"
    FORM_FIELD = "form_field"
    STAMP = "stamp"
    SIGNATURE = "signature"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    NOISE = "noise"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Normalized bounding box (0-1 coordinates)."""
    x1: float  # Left
    y1: float  # Top
    x2: float  # Right
    y2: float  # Bottom

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height)
        )

    def area(self) -> float:
        """Calculate normalized area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @classmethod
    def from_pixels(cls, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> 'BoundingBox':
        """Create from pixel coordinates."""
        return cls(
            x1=x1 / width,
            y1=y1 / height,
            x2=x2 / width,
            y2=y2 / height
        )


@dataclass
class DetectedRegion:
    """A detected region in the document."""
    region_type: RegionType
    bbox: BoundingBox
    confidence: float
    text_density: float  # 0-1, how dense is text in this region
    has_lines: bool  # For table detection
    is_handwritten: bool  # Preliminary handwriting detection
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RegionDetectionResult:
    """Result of region detection on a document page."""
    regions: List[DetectedRegion]
    page_width: int
    page_height: int
    document_type_hint: Optional[str]  # Preliminary doc type guess
    has_tables: bool
    has_handwriting: bool
    total_text_area: float  # Fraction of page that is text


class RegionDetector:
    """
    Detects and classifies regions in document images.

    Uses a two-stage approach:
    1. OpenCV for geometric region detection (bounding boxes)
    2. Optional VLM (PaliGemma) for region classification
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.use_vlm = self.config.get('use_vlm_classification', True)
        self.vlm_client = None

        # Detection parameters
        self.min_region_area = self.config.get('min_region_area', 0.001)  # Min 0.1% of page
        self.merge_threshold = self.config.get('merge_threshold', 20)  # Pixels

    async def detect_regions(
        self,
        image: np.ndarray,
        classify_with_vlm: bool = False
    ) -> RegionDetectionResult:
        """
        Detect regions in a document image.

        Args:
            image: Preprocessed grayscale or BGR image
            classify_with_vlm: If True, use VLM to classify ambiguous regions

        Returns:
            RegionDetectionResult with all detected regions
        """
        h, w = image.shape[:2]

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect different region types
        regions = []

        # 1. Detect text blocks
        text_regions = self._detect_text_blocks(gray)
        regions.extend(text_regions)

        # 2. Detect tables
        table_regions = self._detect_tables(gray)
        regions.extend(table_regions)

        # 3. Detect potential handwriting
        handwriting_regions = self._detect_handwriting_regions(gray, text_regions)

        # 4. Merge overlapping regions
        regions = self._merge_overlapping_regions(regions)

        # 5. Classify with VLM if enabled
        if classify_with_vlm and self.use_vlm:
            regions = await self._classify_with_vlm(image, regions)

        # Calculate summary stats
        has_tables = any(r.region_type == RegionType.TABLE for r in regions)
        has_handwriting = any(r.is_handwritten for r in regions)
        total_text_area = sum(r.bbox.area() for r in regions if r.region_type in [
            RegionType.PRINTED_TEXT, RegionType.HANDWRITING
        ])

        # Guess document type based on regions
        doc_type_hint = self._guess_document_type(regions)

        return RegionDetectionResult(
            regions=regions,
            page_width=w,
            page_height=h,
            document_type_hint=doc_type_hint,
            has_tables=has_tables,
            has_handwriting=has_handwriting,
            total_text_area=min(1.0, total_text_area)
        )

    def _detect_text_blocks(self, gray: np.ndarray) -> List[DetectedRegion]:
        """
        Detect text block regions using morphological operations.

        Strategy:
        1. Binarize image
        2. Dilate to connect text characters into blocks
        3. Find contours of blocks
        """
        h, w = gray.shape

        # Adaptive threshold for binarization
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )

        # Dilate horizontally to connect words
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        dilated_h = cv2.dilate(binary, kernel_h)

        # Dilate vertically to connect lines
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        dilated = cv2.dilate(dilated_h, kernel_v)

        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # Filter small regions
            area_ratio = (cw * ch) / (w * h)
            if area_ratio < self.min_region_area:
                continue

            # Create bounding box
            bbox = BoundingBox.from_pixels(x, y, x + cw, y + ch, w, h)

            # Calculate text density in this region
            roi = binary[y:y+ch, x:x+cw]
            text_density = np.sum(roi > 0) / (cw * ch) if cw * ch > 0 else 0

            regions.append(DetectedRegion(
                region_type=RegionType.PRINTED_TEXT,
                bbox=bbox,
                confidence=0.7,  # Base confidence, can be improved by VLM
                text_density=text_density,
                has_lines=False,
                is_handwritten=False
            ))

        return regions

    def _detect_tables(self, gray: np.ndarray) -> List[DetectedRegion]:
        """
        Detect table regions by finding grid lines.

        Tables typically have:
        - Horizontal lines
        - Vertical lines
        - Grid structure
        """
        h, w = gray.shape

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Detect horizontal lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)

        # Detect vertical lines
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)

        # Combine lines
        lines_mask = cv2.add(horizontal, vertical)

        # Dilate to connect grid
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(lines_mask, kernel)

        # Find contours of potential tables
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # Tables should be reasonably sized
            area_ratio = (cw * ch) / (w * h)
            if area_ratio < 0.01:  # At least 1% of page
                continue

            # Check if region has grid structure
            roi_h = horizontal[y:y+ch, x:x+cw]
            roi_v = vertical[y:y+ch, x:x+cw]

            h_lines = np.sum(roi_h > 0)
            v_lines = np.sum(roi_v > 0)

            # Need both horizontal and vertical lines for a table
            if h_lines < 100 or v_lines < 100:
                continue

            bbox = BoundingBox.from_pixels(x, y, x + cw, y + ch, w, h)

            regions.append(DetectedRegion(
                region_type=RegionType.TABLE,
                bbox=bbox,
                confidence=0.8,
                text_density=0.5,
                has_lines=True,
                is_handwritten=False,
                metadata={'horizontal_lines': int(h_lines), 'vertical_lines': int(v_lines)}
            ))

        return regions

    def _detect_handwriting_regions(
        self,
        gray: np.ndarray,
        text_regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """
        Identify regions that may contain handwriting.

        Handwriting characteristics:
        - Irregular stroke widths
        - Less uniform character spacing
        - More curved strokes
        """
        h, w = gray.shape
        handwriting_regions = []

        for region in text_regions:
            px = region.bbox.to_pixels(w, h)
            roi = gray[px[1]:px[3], px[0]:px[2]]

            if roi.size == 0:
                continue

            # Analyze stroke characteristics
            is_handwritten = self._analyze_stroke_characteristics(roi)

            if is_handwritten:
                region.is_handwritten = True
                region.region_type = RegionType.HANDWRITING
                region.confidence = 0.6  # Lower confidence, needs VLM verification
                handwriting_regions.append(region)

        return handwriting_regions

    def _analyze_stroke_characteristics(self, roi: np.ndarray) -> bool:
        """
        Analyze if a region contains handwriting based on stroke characteristics.

        Returns True if likely handwritten.
        """
        # Binarize
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate stroke width variation
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        if np.max(dist_transform) == 0:
            return False

        # Handwriting tends to have more stroke width variation
        stroke_std = np.std(dist_transform[dist_transform > 0])
        stroke_mean = np.mean(dist_transform[dist_transform > 0])

        if stroke_mean == 0:
            return False

        variation_coefficient = stroke_std / stroke_mean

        # Handwriting typically has CV > 0.5
        return variation_coefficient > 0.5

    def _merge_overlapping_regions(
        self,
        regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """
        Merge overlapping regions, preferring more specific types.

        Priority: TABLE > HANDWRITING > PRINTED_TEXT
        """
        if len(regions) <= 1:
            return regions

        # Sort by area (larger first)
        regions = sorted(regions, key=lambda r: -r.bbox.area())

        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            # Check for overlaps with remaining regions
            for j, r2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue

                overlap = self._calculate_overlap(r1.bbox, r2.bbox)

                if overlap > 0.5:  # More than 50% overlap
                    # Keep the more specific type
                    if r2.region_type == RegionType.TABLE:
                        used.add(i)
                        break
                    elif r2.region_type == RegionType.HANDWRITING and r1.region_type == RegionType.PRINTED_TEXT:
                        r1.region_type = RegionType.HANDWRITING
                        r1.is_handwritten = True
                    used.add(j)

            if i not in used:
                merged.append(r1)

        return merged

    def _calculate_overlap(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate IoU (Intersection over Union) of two boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.area()
        area2 = box2.area()
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    async def _classify_with_vlm(
        self,
        image: np.ndarray,
        regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """
        Use VLM (PaliGemma) to classify ambiguous regions.

        Only classifies regions with low confidence.
        """
        if self.vlm_client is None:
            try:
                from .paligemma_client import PaliGemmaClient
                self.vlm_client = PaliGemmaClient(self.config)
            except ImportError:
                logger.warning("PaliGemma client not available, skipping VLM classification")
                return regions

        h, w = image.shape[:2]

        for region in regions:
            if region.confidence >= 0.85:
                continue

            # Extract region image
            px = region.bbox.to_pixels(w, h)
            roi = image[px[1]:px[3], px[0]:px[2]]

            if roi.size == 0:
                continue

            try:
                # Ask VLM to classify
                classification = await self.vlm_client.classify_region(roi)
                if classification:
                    region.region_type = RegionType(classification['type'])
                    region.confidence = classification['confidence']
                    region.is_handwritten = classification.get('is_handwritten', False)
            except Exception as e:
                logger.warning(f"VLM classification failed: {e}")

        return regions

    def _guess_document_type(self, regions: List[DetectedRegion]) -> Optional[str]:
        """
        Make a preliminary guess about document type based on regions.
        """
        has_tables = any(r.region_type == RegionType.TABLE for r in regions)
        has_multiple_tables = sum(1 for r in regions if r.region_type == RegionType.TABLE) > 1
        total_text_area = sum(r.bbox.area() for r in regions)
        has_large_text_block = any(r.bbox.area() > 0.3 for r in regions if r.region_type == RegionType.PRINTED_TEXT)

        if has_multiple_tables:
            return "lab"  # Lab reports often have multiple tables
        elif has_tables and total_text_area < 0.3:
            return "lab"  # Tables with limited text
        elif has_large_text_block and not has_tables:
            return "radiology"  # Narrative text
        elif total_text_area < 0.2 and len(regions) < 5:
            return "prescription"  # Sparse content

        return None


# Convenience function
async def detect_document_regions(
    image_path: Path,
    config: Dict = None
) -> RegionDetectionResult:
    """
    Convenience function to detect regions in a document image.

    Args:
        image_path: Path to image file
        config: Optional configuration

    Returns:
        RegionDetectionResult with detected regions
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    detector = RegionDetector(config or {})
    return await detector.detect_regions(image)
