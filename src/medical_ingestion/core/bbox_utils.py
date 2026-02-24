# ============================================================================
# src/medical_ingestion/core/bbox_utils.py
# ============================================================================
"""
Bounding box utilities for PDF coordinate handling.

This module provides:
- Bbox validation and normalization
- Multi-word bbox merging
- Fuzzy text matching with bboxes
- Bbox confidence scoring
- Coordinate system conversion

Coordinate System:
- All bboxes are stored as (x0, y0, x1, y1) tuples
- Normalized coordinates: 0-1 range where (0,0) is top-left, (1,1) is bottom-right
- x0 < x1 (left to right)
- y0 < y1 (top to bottom)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
from difflib import SequenceMatcher
import re
import logging

logger = logging.getLogger(__name__)

# Type alias for bounding box
BBox = Tuple[float, float, float, float]


@dataclass
class BBoxMatch:
    """Result of a bbox search with confidence scoring."""
    bbox: BBox
    text: str
    confidence: float  # 0-1, how confident we are in this bbox
    source: str  # "table", "ocr", "merged"
    match_type: str  # "exact", "fuzzy", "partial"
    page: Optional[int] = None


def validate_bbox(bbox: Optional[BBox]) -> bool:
    """
    Validate that a bbox is properly formed.

    Args:
        bbox: (x0, y0, x1, y1) tuple

    Returns:
        True if bbox is valid, False otherwise
    """
    if bbox is None:
        return False

    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False

    try:
        x0, y0, x1, y1 = bbox
        # Check all are numbers
        if not all(isinstance(v, (int, float)) for v in bbox):
            return False
        # Check for NaN or infinity
        if any(v != v or abs(v) == float('inf') for v in bbox):  # NaN check: v != v
            return False
        # Check ordering (x0 <= x1, y0 <= y1)
        if x0 > x1 or y0 > y1:
            return False
        return True
    except (TypeError, ValueError):
        return False


def normalize_bbox(
    bbox: BBox,
    page_width: float,
    page_height: float,
    origin: str = "top-left"
) -> Optional[BBox]:
    """
    Normalize bbox to 0-1 range and ensure proper coordinate system.

    Args:
        bbox: (x0, y0, x1, y1) in absolute coordinates
        page_width: Page width in same units as bbox
        page_height: Page height in same units as bbox
        origin: Coordinate origin - "top-left" or "bottom-left"

    Returns:
        Normalized bbox (0-1 range, top-left origin) or None if invalid
    """
    if not bbox or page_width <= 0 or page_height <= 0:
        return None

    try:
        x0, y0, x1, y1 = bbox

        # Normalize to 0-1
        nx0 = x0 / page_width
        ny0 = y0 / page_height
        nx1 = x1 / page_width
        ny1 = y1 / page_height

        # Invert Y if origin is bottom-left (like PDF coordinate system)
        if origin == "bottom-left":
            ny0, ny1 = 1.0 - ny1, 1.0 - ny0

        # Ensure proper ordering
        if nx0 > nx1:
            nx0, nx1 = nx1, nx0
        if ny0 > ny1:
            ny0, ny1 = ny1, ny0

        # Clamp to valid range
        result = (
            max(0.0, min(1.0, nx0)),
            max(0.0, min(1.0, ny0)),
            max(0.0, min(1.0, nx1)),
            max(0.0, min(1.0, ny1))
        )

        return result if validate_bbox(result) else None

    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.debug(f"Bbox normalization failed: {e}")
        return None


def fix_bbox_ordering(bbox: BBox) -> BBox:
    """
    Ensure bbox has correct ordering (x0 < x1, y0 < y1).

    Args:
        bbox: (x0, y0, x1, y1) tuple

    Returns:
        Corrected bbox with proper ordering
    """
    x0, y0, x1, y1 = bbox
    return (
        min(x0, x1),
        min(y0, y1),
        max(x0, x1),
        max(y0, y1)
    )


def merge_bboxes(bboxes: List[BBox]) -> Optional[BBox]:
    """
    Merge multiple bboxes into a single encompassing bbox.

    Useful for multi-word values where each word has its own bbox.

    Args:
        bboxes: List of (x0, y0, x1, y1) tuples

    Returns:
        Merged bbox that encompasses all input bboxes, or None if empty
    """
    if not bboxes:
        return None

    valid_bboxes = [b for b in bboxes if validate_bbox(b)]
    if not valid_bboxes:
        return None

    # Find min/max of all bboxes
    x0 = min(b[0] for b in valid_bboxes)
    y0 = min(b[1] for b in valid_bboxes)
    x1 = max(b[2] for b in valid_bboxes)
    y1 = max(b[3] for b in valid_bboxes)

    return (x0, y0, x1, y1)


def bbox_area(bbox: BBox) -> float:
    """Calculate area of a bbox."""
    if not validate_bbox(bbox):
        return 0.0
    x0, y0, x1, y1 = bbox
    return (x1 - x0) * (y1 - y0)


def bbox_overlap(bbox1: BBox, bbox2: BBox) -> float:
    """
    Calculate overlap ratio between two bboxes.

    Args:
        bbox1, bbox2: (x0, y0, x1, y1) tuples

    Returns:
        Overlap ratio (0-1) based on intersection over union
    """
    if not validate_bbox(bbox1) or not validate_bbox(bbox2):
        return 0.0

    # Calculate intersection
    ix0 = max(bbox1[0], bbox2[0])
    iy0 = max(bbox1[1], bbox2[1])
    ix1 = min(bbox1[2], bbox2[2])
    iy1 = min(bbox1[3], bbox2[3])

    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0  # No overlap

    intersection = (ix1 - ix0) * (iy1 - iy0)

    # Calculate union
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two strings.

    Args:
        text1, text2: Strings to compare

    Returns:
        Similarity ratio 0-1
    """
    if not text1 or not text2:
        return 0.0

    # Normalize strings
    t1 = text1.strip().lower()
    t2 = text2.strip().lower()

    if t1 == t2:
        return 1.0

    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, t1, t2).ratio()


def find_text_in_word_boxes(
    search_text: str,
    word_boxes: List[Any],  # List of objects with .text and .bbox attributes
    fuzzy_threshold: float = 0.8
) -> List[BBoxMatch]:
    """
    Find text in a list of word boxes with fuzzy matching support.

    Handles:
    - Exact matches
    - Fuzzy matches (typos, OCR errors)
    - Multi-word values split across boxes
    - Numeric values with slight variations

    Args:
        search_text: The text to find
        word_boxes: List of word box objects with .text and .bbox
        fuzzy_threshold: Minimum similarity for fuzzy match (0-1)

    Returns:
        List of BBoxMatch results sorted by confidence
    """
    if not search_text or not word_boxes:
        return []

    results = []
    search_normalized = search_text.strip().lower()
    search_words = search_normalized.split()

    # Track which boxes we've used
    used_indices = set()

    # 1. Try exact single-word match first
    for i, wb in enumerate(word_boxes):
        if not hasattr(wb, 'text') or not hasattr(wb, 'bbox'):
            continue

        wb_text = str(wb.text).strip()
        if wb_text.lower() == search_normalized:
            results.append(BBoxMatch(
                bbox=wb.bbox,
                text=wb_text,
                confidence=1.0,
                source="ocr",
                match_type="exact"
            ))
            used_indices.add(i)

    # 2. Try fuzzy single-word match
    if not results:
        for i, wb in enumerate(word_boxes):
            if i in used_indices:
                continue
            if not hasattr(wb, 'text') or not hasattr(wb, 'bbox'):
                continue

            wb_text = str(wb.text).strip()
            similarity = text_similarity(wb_text, search_text)

            if similarity >= fuzzy_threshold:
                results.append(BBoxMatch(
                    bbox=wb.bbox,
                    text=wb_text,
                    confidence=similarity * 0.9,  # Slight penalty for fuzzy
                    source="ocr",
                    match_type="fuzzy"
                ))

    # 3. Try multi-word matching (consecutive words)
    if len(search_words) > 1 and not results:
        for start_idx in range(len(word_boxes) - len(search_words) + 1):
            consecutive_boxes = word_boxes[start_idx:start_idx + len(search_words)]

            # Check if boxes are valid
            if not all(hasattr(wb, 'text') and hasattr(wb, 'bbox') for wb in consecutive_boxes):
                continue

            # Build combined text
            combined_text = " ".join(str(wb.text).strip() for wb in consecutive_boxes)
            similarity = text_similarity(combined_text, search_text)

            if similarity >= fuzzy_threshold:
                # Merge bboxes
                merged = merge_bboxes([wb.bbox for wb in consecutive_boxes])
                if merged:
                    results.append(BBoxMatch(
                        bbox=merged,
                        text=combined_text,
                        confidence=similarity * 0.85,  # Penalty for multi-word
                        source="ocr",
                        match_type="partial"
                    ))

    # 4. Try numeric value matching (handle variations like "5.2" vs "5.20")
    if not results and _is_numeric(search_text):
        search_num = _parse_numeric(search_text)
        if search_num is not None:
            for i, wb in enumerate(word_boxes):
                if not hasattr(wb, 'text') or not hasattr(wb, 'bbox'):
                    continue

                wb_text = str(wb.text).strip()
                if _is_numeric(wb_text):
                    wb_num = _parse_numeric(wb_text)
                    if wb_num is not None and abs(search_num - wb_num) < 0.001:
                        results.append(BBoxMatch(
                            bbox=wb.bbox,
                            text=wb_text,
                            confidence=0.95,
                            source="ocr",
                            match_type="exact"
                        ))

    # Sort by confidence
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


def find_field_value_bbox(
    field_name: str,
    value: str,
    word_boxes: List[Any],
    row_tolerance: float = 0.02
) -> Optional[BBoxMatch]:
    """
    Find bbox for a field value by first locating the field name,
    then searching for the value nearby.

    This improves accuracy by using context (field label) to narrow search.

    Args:
        field_name: The field/test name (e.g., "WBC", "Hemoglobin")
        value: The value to find
        word_boxes: List of word box objects
        row_tolerance: Y-coordinate tolerance for same-row detection

    Returns:
        BBoxMatch for the value, or None
    """
    if not field_name or not value or not word_boxes:
        return None

    # First, find the field name
    field_matches = find_text_in_word_boxes(
        field_name,
        word_boxes,
        fuzzy_threshold=0.7
    )

    if not field_matches:
        # Fall back to just finding the value
        value_matches = find_text_in_word_boxes(value, word_boxes)
        return value_matches[0] if value_matches else None

    # Get field bbox
    field_bbox = field_matches[0].bbox
    field_y_center = (field_bbox[1] + field_bbox[3]) / 2

    # Now search for value, prioritizing boxes on the same row
    value_matches = find_text_in_word_boxes(value, word_boxes)

    if not value_matches:
        return None

    # Score matches by proximity to field
    scored_matches = []
    for match in value_matches:
        value_y_center = (match.bbox[1] + match.bbox[3]) / 2
        y_distance = abs(value_y_center - field_y_center)

        # Same row bonus
        same_row = y_distance <= row_tolerance
        proximity_score = 1.0 - min(y_distance * 5, 0.5)  # Distance penalty

        # Must be to the right of field name
        is_right_of_field = match.bbox[0] > field_bbox[2]

        if same_row and is_right_of_field:
            # Strong match - same row, right of field
            final_confidence = match.confidence * proximity_score * 1.1
        elif same_row:
            final_confidence = match.confidence * proximity_score
        else:
            # Lower confidence for different row
            final_confidence = match.confidence * proximity_score * 0.7

        scored_matches.append(BBoxMatch(
            bbox=match.bbox,
            text=match.text,
            confidence=min(final_confidence, 1.0),
            source=match.source,
            match_type=match.match_type,
            page=match.page
        ))

    # Return best match
    scored_matches.sort(key=lambda x: x.confidence, reverse=True)
    return scored_matches[0] if scored_matches else None


def calculate_bbox_confidence(
    bbox: BBox,
    source: str,
    extraction_confidence: float = 1.0
) -> float:
    """
    Calculate confidence score for a bounding box based on various factors.

    Args:
        bbox: The bounding box
        source: Extraction source ("table", "ocr", "medgemma")
        extraction_confidence: Confidence from the extraction method

    Returns:
        Overall confidence score 0-1
    """
    if not validate_bbox(bbox):
        return 0.0

    # Base confidence from source
    source_weights = {
        "table": 0.95,      # Table extraction is most reliable
        "pdfplumber": 0.93,
        "camelot": 0.85,    # Camelot has coordinate issues
        "ocr": 0.80,        # OCR can have alignment issues
        "merged": 0.75,     # Merged boxes are less precise
        "medgemma": 0.60,   # AI extraction doesn't provide bboxes
        "unknown": 0.50
    }

    base_confidence = source_weights.get(source.lower(), 0.5)

    # Size sanity check - very small or very large boxes are suspicious
    area = bbox_area(bbox)
    if area < 0.0001:  # Too small
        base_confidence *= 0.5
    elif area > 0.5:  # Suspiciously large (>50% of page)
        base_confidence *= 0.7

    # Combine with extraction confidence
    return base_confidence * extraction_confidence


def _is_numeric(text: str) -> bool:
    """Check if text represents a number."""
    try:
        # Remove common lab value decorators
        cleaned = re.sub(r'[<>≤≥]', '', text.strip())
        float(cleaned)
        return True
    except (ValueError, TypeError):
        return False


def _parse_numeric(text: str) -> Optional[float]:
    """Parse numeric value from text."""
    try:
        cleaned = re.sub(r'[<>≤≥]', '', text.strip())
        return float(cleaned)
    except (ValueError, TypeError):
        return None
