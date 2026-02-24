# ============================================================================
# src/medical_ingestion/preprocessors/image_preprocessor.py
# ============================================================================
"""
Image Preprocessing stubs.

The full OpenCV-based preprocessing pipeline is not yet implemented.
These stubs allow the import chain (document_pipeline -> preprocessors) to work.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of image preprocessing."""
    image: Any = None
    original_size: Tuple[int, int] = (0, 0)
    processed_size: Tuple[int, int] = (0, 0)
    rotation_angle: float = 0.0
    operations_applied: List[str] = field(default_factory=list)
    quality_improved: bool = False


class ImagePreprocessor:
    """
    Stub preprocessor — returns images unchanged.

    A full implementation would handle deskew, noise reduction,
    contrast normalization, and binarization for OCR.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def preprocess(self, image_path: Path) -> PreprocessingResult:
        """Pass-through: return image without modification."""
        logger.debug(f"Preprocessing skipped (stub) for {image_path}")
        return PreprocessingResult(
            image=None,
            original_size=(0, 0),
            processed_size=(0, 0),
            rotation_angle=0.0,
            operations_applied=["passthrough"],
            quality_improved=False,
        )
