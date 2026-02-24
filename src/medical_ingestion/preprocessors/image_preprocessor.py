# # ============================================================================
# # src/medical_ingestion/preprocessors/image_preprocessor.py
# # ============================================================================
# """
# OpenCV-based Image Preprocessing for Document Processing

# Handles dirty documents with:
# - Deskew / rotation correction
# - Noise reduction (fax, scan artifacts)
# - Contrast / brightness normalization
# - Shadow removal
# - Binarization for OCR
# """

# import cv2
# import numpy as np
# from pathlib import Path
# from typing import Tuple, Optional, List, Dict, Any
# from dataclasses import dataclass
# import logging

# logger = logging.getLogger(__name__)


# @dataclass
# class PreprocessingResult:
#     """Result of image preprocessing."""
#     image: np.ndarray  # Preprocessed image
#     original_size: Tuple[int, int]  # (width, height)
#     processed_size: Tuple[int, int]
#     rotation_angle: float  # Degrees rotated
#     operations_applied: List[str]
#     quality_improved: bool


# class ImagePreprocessor:
#     """
#     Preprocesses document images for optimal OCR performance.

#     Pipeline:
#     1. Load and convert to grayscale
#     2. Noise reduction
#     3. Deskew correction
#     4. Contrast enhancement
#     5. Shadow removal (optional)
#     6. Adaptive thresholding (optional, for binarization)
#     """

#     def __init__(self, config: Dict[str, Any] = None):
#         self.config = config or {}

#         # Configuration options
#         self.auto_deskew = self.config.get('auto_deskew', True)
#         self.denoise = self.config.get('denoise', True)
#         self.enhance_contrast = self.config.get('enhance_contrast', True)
#         self.remove_shadows = self.config.get('remove_shadows', True)
#         self.target_dpi = self.config.get('target_dpi', 300)

#     def preprocess(
#         self,
#         image_path: Path,
#         binarize: bool = False
#     ) -> PreprocessingResult:
#         """
#         Preprocess a document image.

#         Args:
#             image_path: Path to image file
#             binarize: If True, apply adaptive thresholding for binary output

#         Returns:
#             PreprocessingResult with processed image and metadata
#         """
#         # Load image
#         image = cv2.imread(str(image_path))
#         if image is None:
#             raise ValueError(f"Could not load image: {image_path}")

#         original_size = (image.shape[1], image.shape[0])
#         operations = []

#         # Convert to grayscale for processing
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()

#         # Track rotation
#         rotation_angle = 0.0

#         # Step 1: Noise reduction
#         if self.denoise:
#             gray = self._denoise(gray)
#             operations.append('denoise')

#         # Step 2: Deskew correction
#         if self.auto_deskew:
#             gray, angle = self._deskew(gray)
#             rotation_angle = angle
#             if abs(angle) > 0.1:
#                 operations.append(f'deskew({angle:.1f}°)')

#         # Step 3: Contrast enhancement
#         if self.enhance_contrast:
#             gray = self._enhance_contrast(gray)
#             operations.append('contrast_enhance')

#         # Step 4: Shadow removal
#         if self.remove_shadows:
#             gray = self._remove_shadows(gray)
#             operations.append('shadow_removal')

#         # Step 5: Optional binarization
#         if binarize:
#             gray = self._adaptive_threshold(gray)
#             operations.append('binarize')

#         processed_size = (gray.shape[1], gray.shape[0])

#         # Assess if quality improved
#         quality_improved = len(operations) > 0

#         logger.info(f"Preprocessed image: {operations}")

#         return PreprocessingResult(
#             image=gray,
#             original_size=original_size,
#             processed_size=processed_size,
#             rotation_angle=rotation_angle,
#             operations_applied=operations,
#             quality_improved=quality_improved
#         )

#     def preprocess_array(
#         self,
#         image: np.ndarray,
#         binarize: bool = False
#     ) -> PreprocessingResult:
#         """
#         Preprocess an image array directly.

#         Args:
#             image: Input image as numpy array (BGR or grayscale)
#             binarize: If True, apply adaptive thresholding

#         Returns:
#             PreprocessingResult with processed image
#         """
#         original_size = (image.shape[1], image.shape[0])
#         operations = []

#         # Convert to grayscale if needed
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()

#         rotation_angle = 0.0

#         if self.denoise:
#             gray = self._denoise(gray)
#             operations.append('denoise')

#         if self.auto_deskew:
#             gray, angle = self._deskew(gray)
#             rotation_angle = angle
#             if abs(angle) > 0.1:
#                 operations.append(f'deskew({angle:.1f}°)')

#         if self.enhance_contrast:
#             gray = self._enhance_contrast(gray)
#             operations.append('contrast_enhance')

#         if self.remove_shadows:
#             gray = self._remove_shadows(gray)
#             operations.append('shadow_removal')

#         if binarize:
#             gray = self._adaptive_threshold(gray)
#             operations.append('binarize')

#         return PreprocessingResult(
#             image=gray,
#             original_size=original_size,
#             processed_size=(gray.shape[1], gray.shape[0]),
#             rotation_angle=rotation_angle,
#             operations_applied=operations,
#             quality_improved=len(operations) > 0
#         )

#     def _denoise(self, image: np.ndarray) -> np.ndarray:
#         """
#         Remove noise from image.

#         Uses Non-local Means Denoising which is effective for:
#         - Fax noise
#         - Scan artifacts
#         - Compression artifacts
#         """
#         # Non-local means denoising - good for document images
#         # h=10 is filter strength, templateWindowSize=7, searchWindowSize=21
#         denoised = cv2.fastNlMeansDenoising(
#             image,
#             h=10,
#             templateWindowSize=7,
#             searchWindowSize=21
#         )
#         return denoised

#     def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
#         """
#         Detect and correct skew in document image.

#         Uses Hough Line Transform to detect text lines and
#         calculates the dominant angle for correction.

#         Returns:
#             Tuple of (corrected image, rotation angle in degrees)
#         """
#         # Edge detection
#         edges = cv2.Canny(image, 50, 150, apertureSize=3)

#         # Hough Line Transform
#         lines = cv2.HoughLinesP(
#             edges,
#             rho=1,
#             theta=np.pi / 180,
#             threshold=100,
#             minLineLength=100,
#             maxLineGap=10
#         )

#         if lines is None or len(lines) == 0:
#             return image, 0.0

#         # Calculate angles of all detected lines
#         angles = []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             if x2 - x1 != 0:  # Avoid division by zero
#                 angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
#                 # Only consider near-horizontal lines (text lines)
#                 if abs(angle) < 45:
#                     angles.append(angle)

#         if not angles:
#             return image, 0.0

#         # Use median angle to avoid outliers
#         median_angle = np.median(angles)

#         # Only correct if angle is significant but not too large
#         if abs(median_angle) < 0.5 or abs(median_angle) > 15:
#             return image, 0.0

#         # Rotate image to correct skew
#         h, w = image.shape[:2]
#         center = (w // 2, h // 2)
#         rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

#         # Calculate new image size to avoid cropping
#         cos = np.abs(rotation_matrix[0, 0])
#         sin = np.abs(rotation_matrix[0, 1])
#         new_w = int(h * sin + w * cos)
#         new_h = int(h * cos + w * sin)

#         # Adjust rotation matrix for new size
#         rotation_matrix[0, 2] += (new_w - w) / 2
#         rotation_matrix[1, 2] += (new_h - h) / 2

#         rotated = cv2.warpAffine(
#             image,
#             rotation_matrix,
#             (new_w, new_h),
#             flags=cv2.INTER_CUBIC,
#             borderMode=cv2.BORDER_REPLICATE
#         )

#         return rotated, median_angle

#     def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
#         """
#         Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

#         CLAHE is superior to regular histogram equalization because it:
#         - Works on small regions (tiles) rather than entire image
#         - Limits contrast amplification to reduce noise amplification
#         - Handles varying illumination across document
#         """
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(image)
#         return enhanced

#     def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
#         """
#         Remove shadows from document image.

#         Uses morphological operations to estimate and remove
#         uneven illumination / shadows.
#         """
#         # Dilate to create background estimate
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#         dilated = cv2.dilate(image, kernel)

#         # Apply median blur to smooth the background
#         bg = cv2.medianBlur(dilated, 21)

#         # Divide original by background to normalize illumination
#         # Add small value to avoid division by zero
#         normalized = cv2.divide(image, bg, scale=255)

#         return normalized

#     def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
#         """
#         Apply adaptive thresholding for binarization.

#         Adaptive thresholding is better than global thresholding because
#         it handles varying illumination across the document.
#         """
#         binary = cv2.adaptiveThreshold(
#             image,
#             maxValue=255,
#             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             thresholdType=cv2.THRESH_BINARY,
#             blockSize=11,
#             C=2
#         )
#         return binary

#     def detect_page_orientation(self, image: np.ndarray) -> int:
#         """
#         Detect if page needs 90/180/270 degree rotation.

#         Uses text line detection to determine if page is
#         rotated by 90, 180, or 270 degrees.

#         Returns:
#             Rotation needed in degrees (0, 90, 180, or 270)
#         """
#         # Convert to grayscale if needed
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image

#         # Edge detection
#         edges = cv2.Canny(gray, 50, 150)

#         # Detect lines
#         lines = cv2.HoughLinesP(
#             edges,
#             rho=1,
#             theta=np.pi / 180,
#             threshold=100,
#             minLineLength=50,
#             maxLineGap=10
#         )

#         if lines is None:
#             return 0

#         # Count horizontal vs vertical lines
#         horizontal_count = 0
#         vertical_count = 0

#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

#             if angle < 20 or angle > 160:
#                 horizontal_count += 1
#             elif 70 < angle < 110:
#                 vertical_count += 1

#         # If more vertical lines than horizontal, page might be rotated 90°
#         if vertical_count > horizontal_count * 1.5:
#             return 90

#         return 0

#     def correct_orientation(self, image: np.ndarray) -> np.ndarray:
#         """
#         Correct page orientation (90° rotations).

#         Args:
#             image: Input image

#         Returns:
#             Correctly oriented image
#         """
#         rotation = self.detect_page_orientation(image)

#         if rotation == 0:
#             return image
#         elif rotation == 90:
#             return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#         elif rotation == 180:
#             return cv2.rotate(image, cv2.ROTATE_180)
#         elif rotation == 270:
#             return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

#         return image

#     def remove_borders(self, image: np.ndarray, margin: int = 10) -> np.ndarray:
#         """
#         Remove black borders from scanned documents.

#         Args:
#             image: Input image
#             margin: Additional margin to trim

#         Returns:
#             Image with borders removed
#         """
#         # Convert to grayscale if needed
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image

#         # Threshold to find content
#         _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

#         # Find contours
#         contours, _ = cv2.findContours(
#             thresh,
#             cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE
#         )

#         if not contours:
#             return image

#         # Find bounding box of all content
#         x_min, y_min = gray.shape[1], gray.shape[0]
#         x_max, y_max = 0, 0

#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             x_min = min(x_min, x)
#             y_min = min(y_min, y)
#             x_max = max(x_max, x + w)
#             y_max = max(y_max, y + h)

#         # Add margin
#         x_min = max(0, x_min - margin)
#         y_min = max(0, y_min - margin)
#         x_max = min(gray.shape[1], x_max + margin)
#         y_max = min(gray.shape[0], y_max + margin)

#         # Crop
#         if len(image.shape) == 3:
#             return image[y_min:y_max, x_min:x_max]
#         else:
#             return gray[y_min:y_max, x_min:x_max]


# def preprocess_for_ocr(image_path: Path, config: Dict = None) -> np.ndarray:
#     """
#     Convenience function to preprocess an image for OCR.

#     Args:
#         image_path: Path to image file
#         config: Optional preprocessing configuration

#     Returns:
#         Preprocessed grayscale image ready for OCR
#     """
#     preprocessor = ImagePreprocessor(config or {})
#     result = preprocessor.preprocess(image_path, binarize=False)
#     return result.image
