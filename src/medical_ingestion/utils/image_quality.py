# ============================================================================
# src/medical_ingestion/utils/image_quality.py
# ============================================================================
"""
Image quality analysis for medical documents.

Provides:
- Resolution checking
- Blur detection (Laplacian variance)
- Contrast/brightness analysis
- Skew detection
- Handwriting detection
- Quality scoring with user feedback

Returns actionable feedback to help users capture better images.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Overall quality classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNPROCESSABLE = "unprocessable"


@dataclass
class QualityIssue:
    """A specific quality issue with recommendation."""
    issue: str
    severity: str  # "critical", "warning", "info"
    recommendation: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class QualityReport:
    """Complete image quality analysis report."""
    # Overall assessment
    quality_level: QualityLevel
    quality_score: float  # 0-100
    can_process: bool

    # Detailed metrics
    resolution: Tuple[int, int]  # (width, height)
    estimated_dpi: Optional[int]
    megapixels: float
    blur_score: float  # Higher = sharper
    contrast_score: float  # 0-100
    brightness_score: float  # 0-100 (50 is ideal)

    # Detection flags
    is_likely_handwritten: bool
    has_mixed_content: bool  # Handwriting + printed
    is_skewed: bool
    skew_angle: Optional[float]

    # Issues and recommendations
    issues: List[QualityIssue] = field(default_factory=list)

    # User-friendly message
    summary: str = ""
    tips: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        # Convert numpy types to native Python types for JSON serialization
        return {
            "quality_level": self.quality_level.value,
            "quality_score": float(round(self.quality_score, 1)),
            "can_process": bool(self.can_process),
            "resolution": {"width": int(self.resolution[0]), "height": int(self.resolution[1])},
            "estimated_dpi": int(self.estimated_dpi) if self.estimated_dpi else None,
            "megapixels": float(round(self.megapixels, 2)),
            "blur_score": float(round(self.blur_score, 1)),
            "contrast_score": float(round(self.contrast_score, 1)),
            "brightness_score": float(round(self.brightness_score, 1)),
            "is_likely_handwritten": bool(self.is_likely_handwritten),
            "has_mixed_content": bool(self.has_mixed_content),
            "is_skewed": bool(self.is_skewed),
            "skew_angle": float(round(self.skew_angle, 1)) if self.skew_angle else None,
            "issues": [
                {
                    "issue": i.issue,
                    "severity": i.severity,
                    "recommendation": i.recommendation
                }
                for i in self.issues
            ],
            "summary": self.summary,
            "tips": self.tips
        }


class ImageQualityAnalyzer:
    """
    Analyzes image quality for medical document processing.

    Quality Thresholds:
    - Resolution: Minimum 1500x1000 (acceptable), 2000x1500 (good), 3000x2000 (excellent)
    - Blur: Laplacian variance > 100 (acceptable), > 300 (good), > 500 (excellent)
    - Contrast: std_dev > 30 (acceptable), > 50 (good), > 70 (excellent)
    - Brightness: 30-70% of 255 range is acceptable, 40-60% is ideal
    """

    # Resolution thresholds (in pixels)
    # Note: Resolution alone doesn't determine quality - a sharp 800x600 beats a blurry 3000x2000
    # These are minimums; actual quality depends on blur, contrast, and brightness scores
    MIN_WIDTH = 600
    MIN_HEIGHT = 450
    GOOD_WIDTH = 1600
    GOOD_HEIGHT = 1200
    EXCELLENT_WIDTH = 2500
    EXCELLENT_HEIGHT = 2000

    # Blur thresholds (Laplacian variance)
    BLUR_THRESHOLD_POOR = 50
    BLUR_THRESHOLD_ACCEPTABLE = 100
    BLUR_THRESHOLD_GOOD = 300
    BLUR_THRESHOLD_EXCELLENT = 500

    # Contrast thresholds (standard deviation of grayscale)
    CONTRAST_THRESHOLD_POOR = 20
    CONTRAST_THRESHOLD_ACCEPTABLE = 35
    CONTRAST_THRESHOLD_GOOD = 50

    # Brightness range (0-255 scale)
    BRIGHTNESS_MIN = 50
    BRIGHTNESS_MAX = 205
    BRIGHTNESS_IDEAL_MIN = 100
    BRIGHTNESS_IDEAL_MAX = 180

    # Assumed document size for DPI estimation (letter size: 8.5 x 11 inches)
    ASSUMED_DOC_WIDTH_INCHES = 8.5
    ASSUMED_DOC_HEIGHT_INCHES = 11.0

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, image_path: Path) -> QualityReport:
        """
        Analyze image quality and return detailed report.

        Args:
            image_path: Path to image file

        Returns:
            QualityReport with metrics, issues, and recommendations
        """
        image_path = Path(image_path)
        self.logger.info(f"Analyzing image quality: {image_path}")

        try:
            from PIL import ImageOps
            image = Image.open(image_path)
            # Apply EXIF orientation so resolution check uses correct dimensions
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass
            return self.analyze_image(image)
        except Exception as e:
            self.logger.error(f"Failed to analyze image: {e}")
            return QualityReport(
                quality_level=QualityLevel.UNPROCESSABLE,
                quality_score=0,
                can_process=False,
                resolution=(0, 0),
                estimated_dpi=None,
                megapixels=0,
                blur_score=0,
                contrast_score=0,
                brightness_score=0,
                is_likely_handwritten=False,
                has_mixed_content=False,
                is_skewed=False,
                skew_angle=None,
                issues=[QualityIssue(
                    issue="Failed to open image",
                    severity="critical",
                    recommendation="Please ensure the file is a valid image (PNG, JPG, TIFF)"
                )],
                summary="Unable to analyze image - file may be corrupted or invalid format."
            )

    def analyze_image(self, image: Image.Image) -> QualityReport:
        """Analyze a PIL Image object."""
        issues: List[QualityIssue] = []
        tips: List[str] = []

        # Get basic dimensions
        width, height = image.size
        megapixels = (width * height) / 1_000_000

        # Estimate DPI based on assumed document size
        estimated_dpi = self._estimate_dpi(width, height)

        # Convert to grayscale for analysis
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        gray_array = np.array(gray)

        # Calculate metrics
        blur_score = self._calculate_blur_score(gray_array)
        contrast_score = self._calculate_contrast_score(gray_array)
        brightness_score = self._calculate_brightness_score(gray_array)

        # Detect characteristics
        is_handwritten, handwriting_confidence = self._detect_handwriting(gray_array)
        has_mixed = self._detect_mixed_content(gray_array) if is_handwritten else False
        is_skewed, skew_angle = self._detect_skew(gray_array)

        # Check resolution
        res_issues = self._check_resolution(width, height, estimated_dpi)
        issues.extend(res_issues)

        # Check blur
        blur_issues = self._check_blur(blur_score)
        issues.extend(blur_issues)

        # Check contrast
        contrast_issues = self._check_contrast(contrast_score)
        issues.extend(contrast_issues)

        # Check brightness
        brightness_issues = self._check_brightness(brightness_score)
        issues.extend(brightness_issues)

        # Check skew
        if is_skewed and abs(skew_angle) > 5:
            issues.append(QualityIssue(
                issue="Image is tilted",
                severity="warning",
                recommendation="Try to photograph the document straight-on, not at an angle",
                value=skew_angle
            ))
            tips.append("Hold camera directly above the document")

        # Add handwriting-specific tips
        if is_handwritten:
            issues.append(QualityIssue(
                issue="Handwriting detected",
                severity="info",
                recommendation="Handwritten content requires higher image quality for accurate reading"
            ))
            tips.append("For handwritten prescriptions, ensure text is clearly visible and not smudged")
            if blur_score < self.BLUR_THRESHOLD_GOOD:
                tips.append("Handwriting needs especially sharp images - hold camera steady")

        # Calculate overall quality score
        quality_score = self._calculate_overall_score(
            width, height, blur_score, contrast_score, brightness_score
        )

        # Determine quality level
        quality_level = self._determine_quality_level(quality_score, issues)

        # Can we process this?
        critical_issues = [i for i in issues if i.severity == "critical"]
        can_process = len(critical_issues) == 0 and quality_score >= 20

        # Generate tips if not already added
        if not tips:
            tips = self._generate_tips(issues)

        # Generate summary message
        summary = self._generate_summary(quality_level, is_handwritten, can_process)

        return QualityReport(
            quality_level=quality_level,
            quality_score=quality_score,
            can_process=can_process,
            resolution=(width, height),
            estimated_dpi=estimated_dpi,
            megapixels=megapixels,
            blur_score=blur_score,
            contrast_score=contrast_score,
            brightness_score=brightness_score,
            is_likely_handwritten=is_handwritten,
            has_mixed_content=has_mixed,
            is_skewed=is_skewed,
            skew_angle=skew_angle if is_skewed else None,
            issues=issues,
            summary=summary,
            tips=tips
        )

    def _estimate_dpi(self, width: int, height: int) -> int:
        """Estimate DPI based on assumed document size."""
        # Use the longer dimension to estimate
        if width > height:
            # Landscape orientation
            dpi_from_width = width / self.ASSUMED_DOC_HEIGHT_INCHES
            dpi_from_height = height / self.ASSUMED_DOC_WIDTH_INCHES
        else:
            # Portrait orientation (typical)
            dpi_from_width = width / self.ASSUMED_DOC_WIDTH_INCHES
            dpi_from_height = height / self.ASSUMED_DOC_HEIGHT_INCHES

        return int((dpi_from_width + dpi_from_height) / 2)

    def _calculate_blur_score(self, gray_array: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance.
        Higher score = sharper image.
        """
        try:
            # Simple Laplacian using convolution
            # Laplacian kernel
            laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

            # Convolve (simplified - proper implementation would use scipy)
            from scipy import ndimage
            laplacian = ndimage.convolve(gray_array.astype(float), laplacian_kernel)

            # Variance of the Laplacian is the blur measure
            variance = laplacian.var()
            return float(variance)
        except ImportError:
            # Fallback: use edge detection approximation
            # Calculate local variance as blur proxy
            local_std = np.std(gray_array)
            return float(local_std * 10)  # Scale to approximate Laplacian range
        except Exception:
            return 0.0

    def _calculate_contrast_score(self, gray_array: np.ndarray) -> float:
        """Calculate contrast score (0-100) based on standard deviation."""
        std_dev = np.std(gray_array)
        # Normalize to 0-100 scale (max std_dev for 8-bit is ~127)
        return min(100, (std_dev / 80) * 100)

    def _calculate_brightness_score(self, gray_array: np.ndarray) -> float:
        """Calculate brightness score (0-100, 50 is ideal)."""
        mean_brightness = np.mean(gray_array)
        # Normalize to 0-100
        return (mean_brightness / 255) * 100

    def _detect_handwriting(self, gray_array: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image likely contains handwriting.

        Handwriting characteristics:
        - More irregular stroke widths
        - Less uniform spacing
        - Higher variance in local features

        Returns:
            (is_handwritten, confidence)
        """
        try:
            # Simple heuristic: handwriting has more irregular edges
            # Use edge detection and analyze variance
            from scipy import ndimage

            # Sobel edge detection
            edges_x = ndimage.sobel(gray_array.astype(float), axis=1)
            edges_y = ndimage.sobel(gray_array.astype(float), axis=0)
            edges = np.hypot(edges_x, edges_y)

            # Threshold edges
            edge_threshold = np.percentile(edges, 90)
            edge_mask = edges > edge_threshold

            # Analyze edge characteristics
            # Handwriting tends to have more varied stroke angles
            edge_variance = np.var(edges[edge_mask]) if edge_mask.sum() > 100 else 0

            # High variance suggests handwriting
            # This is a simplified heuristic - be EXTREMELY conservative to avoid
            # misclassifying typed documents with complex formatting (logos, tables)
            # Typed documents with graphics/logos can have variance up to 100k+
            # Only flag as handwritten if variance is extremely high
            # Effectively DISABLE handwriting detection for now - typed docs keep getting misflagged
            is_handwritten = edge_variance > 500000  # Very high threshold - almost never triggers
            confidence = min(1.0, edge_variance / 800000)

            return is_handwritten, confidence

        except ImportError:
            # Without scipy, use simpler heuristic
            # Just check for irregular patterns
            local_vars = []
            h, w = gray_array.shape
            block_size = 50

            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray_array[i:i+block_size, j:j+block_size]
                    local_vars.append(np.var(block))

            if local_vars:
                variance_of_variances = np.var(local_vars)
                # Be EXTREMELY conservative - typed documents with tables/logos can have high variance
                # Effectively DISABLE this detection to avoid false positives on typed documents
                is_handwritten = variance_of_variances > 100000  # Almost never triggers
                confidence = min(1.0, variance_of_variances / 150000)
                return is_handwritten, confidence

            return False, 0.0
        except Exception:
            return False, 0.0

    def _detect_mixed_content(self, gray_array: np.ndarray) -> bool:
        """Detect if image has both handwritten and printed text."""
        # Simplified: just return False for now
        # Full implementation would compare different regions
        return False

    def _detect_skew(self, gray_array: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image is skewed (rotated).

        Returns:
            (is_skewed, angle_degrees)
        """
        try:
            from scipy import ndimage

            # Edge detection
            edges = ndimage.sobel(gray_array.astype(float))

            # Threshold
            threshold = np.percentile(np.abs(edges), 95)
            edge_mask = np.abs(edges) > threshold

            # Find edge points
            points = np.argwhere(edge_mask)

            if len(points) < 100:
                return False, 0.0

            # Simple linear regression to find dominant angle
            # This is a simplified approach
            y_coords = points[:, 0]
            x_coords = points[:, 1]

            # Fit line
            if len(x_coords) > 10:
                coeffs = np.polyfit(x_coords, y_coords, 1)
                angle = math.degrees(math.atan(coeffs[0]))

                is_skewed = abs(angle) > 2
                return is_skewed, angle

            return False, 0.0

        except Exception:
            return False, 0.0

    def _check_resolution(self, width: int, height: int, estimated_dpi: int) -> List[QualityIssue]:
        """Check resolution and return issues."""
        issues = []

        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            issues.append(QualityIssue(
                issue="Resolution too low",
                severity="critical",
                recommendation=f"Image is {width}x{height} pixels. Minimum recommended is {self.GOOD_WIDTH}x{self.GOOD_HEIGHT} pixels",
                value=width * height,
                threshold=self.GOOD_WIDTH * self.GOOD_HEIGHT
            ))
        elif width < self.GOOD_WIDTH or height < self.GOOD_HEIGHT:
            issues.append(QualityIssue(
                issue="Resolution below optimal",
                severity="warning",
                recommendation=f"Image is {width}x{height} pixels. For best results, use at least {self.GOOD_WIDTH}x{self.GOOD_HEIGHT} pixels",
                value=width * height,
                threshold=self.GOOD_WIDTH * self.GOOD_HEIGHT
            ))

        # DPI estimation assumes standard document size - may be inaccurate for cropped/small docs
        # Use warning only since blur/contrast are more reliable quality indicators
        if estimated_dpi and estimated_dpi < 150:
            issues.append(QualityIssue(
                issue="Low effective DPI",
                severity="warning",  # Not critical - blur/contrast are better indicators
                recommendation=f"Estimated {estimated_dpi} DPI. Aim for 200+ DPI for best text recognition",
                value=estimated_dpi,
                threshold=200
            ))

        return issues

    def _check_blur(self, blur_score: float) -> List[QualityIssue]:
        """Check blur and return issues."""
        issues = []

        if blur_score < self.BLUR_THRESHOLD_POOR:
            issues.append(QualityIssue(
                issue="Image is very blurry",
                severity="critical",
                recommendation="Image is too blurry to read. Please retake with camera held steady and focused",
                value=blur_score,
                threshold=self.BLUR_THRESHOLD_ACCEPTABLE
            ))
        elif blur_score < self.BLUR_THRESHOLD_ACCEPTABLE:
            issues.append(QualityIssue(
                issue="Image is somewhat blurry",
                severity="warning",
                recommendation="Image clarity could be better. Try holding camera steady and tap to focus",
                value=blur_score,
                threshold=self.BLUR_THRESHOLD_GOOD
            ))

        return issues

    def _check_contrast(self, contrast_score: float) -> List[QualityIssue]:
        """Check contrast and return issues."""
        issues = []

        if contrast_score < 25:
            issues.append(QualityIssue(
                issue="Very low contrast",
                severity="critical",
                recommendation="Text is hard to distinguish from background. Improve lighting or use flash",
                value=contrast_score,
                threshold=35
            ))
        elif contrast_score < 35:
            issues.append(QualityIssue(
                issue="Low contrast",
                severity="warning",
                recommendation="Consider better lighting to improve text visibility",
                value=contrast_score,
                threshold=50
            ))

        return issues

    def _check_brightness(self, brightness_score: float) -> List[QualityIssue]:
        """Check brightness and return issues."""
        issues = []

        if brightness_score < 20:
            issues.append(QualityIssue(
                issue="Image is too dark",
                severity="critical",
                recommendation="Image is underexposed. Use more light or flash",
                value=brightness_score
            ))
        elif brightness_score < 30:
            issues.append(QualityIssue(
                issue="Image is somewhat dark",
                severity="warning",
                recommendation="More lighting would improve readability",
                value=brightness_score
            ))
        elif brightness_score > 95:
            # Truly washed out - text likely unreadable
            issues.append(QualityIssue(
                issue="Image is overexposed/washed out",
                severity="critical",
                recommendation="Too much light is washing out the text. Reduce lighting or avoid direct flash",
                value=brightness_score
            ))
        elif brightness_score > 85:
            issues.append(QualityIssue(
                issue="Image is quite bright",
                severity="warning",
                recommendation="Consider reducing flash or bright lights to avoid washout",
                value=brightness_score
            ))

        return issues

    def _calculate_overall_score(
        self,
        width: int,
        height: int,
        blur_score: float,
        contrast_score: float,
        brightness_score: float
    ) -> float:
        """Calculate overall quality score (0-100)."""
        # Resolution score (0-30)
        pixels = width * height
        if pixels >= self.EXCELLENT_WIDTH * self.EXCELLENT_HEIGHT:
            res_score = 30
        elif pixels >= self.GOOD_WIDTH * self.GOOD_HEIGHT:
            res_score = 25
        elif pixels >= self.MIN_WIDTH * self.MIN_HEIGHT:
            res_score = 15
        else:
            res_score = max(0, pixels / (self.MIN_WIDTH * self.MIN_HEIGHT) * 10)

        # Blur score (0-30)
        if blur_score >= self.BLUR_THRESHOLD_EXCELLENT:
            blur_component = 30
        elif blur_score >= self.BLUR_THRESHOLD_GOOD:
            blur_component = 25
        elif blur_score >= self.BLUR_THRESHOLD_ACCEPTABLE:
            blur_component = 18
        elif blur_score >= self.BLUR_THRESHOLD_POOR:
            blur_component = 10
        else:
            blur_component = max(0, blur_score / self.BLUR_THRESHOLD_POOR * 5)

        # Contrast score (0-20)
        contrast_component = min(20, contrast_score * 0.2)

        # Brightness score (0-20) - penalize deviation from ideal (50)
        brightness_deviation = abs(brightness_score - 50)
        brightness_component = max(0, 20 - brightness_deviation * 0.4)

        total = res_score + blur_component + contrast_component + brightness_component
        return min(100, max(0, total))

    def _determine_quality_level(
        self,
        score: float,
        issues: List[QualityIssue]
    ) -> QualityLevel:
        """Determine quality level from score and issues."""
        critical_count = sum(1 for i in issues if i.severity == "critical")
        warning_count = sum(1 for i in issues if i.severity == "warning")

        if critical_count > 0 or score < 20:
            return QualityLevel.UNPROCESSABLE if critical_count > 1 else QualityLevel.POOR
        elif score >= 80 and warning_count == 0:
            return QualityLevel.EXCELLENT
        elif score >= 60:
            return QualityLevel.GOOD
        elif score >= 40:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR

    def _generate_tips(self, issues: List[QualityIssue]) -> List[str]:
        """Generate photography tips based on issues."""
        tips = []

        has_blur = any("blur" in i.issue.lower() for i in issues)
        has_dark = any("dark" in i.issue.lower() for i in issues)
        has_bright = any("bright" in i.issue.lower() or "overexposed" in i.issue.lower() for i in issues)
        has_resolution = any("resolution" in i.issue.lower() for i in issues)
        has_contrast = any("contrast" in i.issue.lower() for i in issues)

        if has_blur:
            tips.append("Hold your camera steady - rest your elbows on a surface if possible")
            tips.append("Tap on the text to ensure focus before taking the photo")

        if has_dark or has_contrast:
            tips.append("Use good lighting - natural daylight works best")
            tips.append("Position light source in front of the document, not behind")

        if has_bright:
            tips.append("Avoid direct flash on glossy paper - use indirect lighting")
            tips.append("If using flash, angle the camera slightly to avoid glare")

        if has_resolution:
            tips.append("Move closer to the document or use zoom to fill the frame")
            tips.append("Ensure your camera is set to high resolution mode")

        # General tips
        if not tips:
            tips = [
                "Photograph directly above the document for best results",
                "Ensure all text is within the frame and in focus",
                "Use good, even lighting without harsh shadows"
            ]

        return tips

    def _generate_summary(
        self,
        quality_level: QualityLevel,
        is_handwritten: bool,
        can_process: bool
    ) -> str:
        """Generate user-friendly summary message."""
        if quality_level == QualityLevel.UNPROCESSABLE:
            return "Image quality is too low to process. Please retake the photo following the tips below."
        elif quality_level == QualityLevel.POOR:
            if can_process:
                return "Image quality is low. We'll try to process it, but results may be inaccurate. Consider retaking if possible."
            else:
                return "Image quality is too low for reliable processing. Please retake the photo."
        elif quality_level == QualityLevel.ACCEPTABLE:
            base = "Image quality is acceptable and can be processed."
            if is_handwritten:
                return base + " Note: Handwritten content detected - accuracy may vary."
            return base
        elif quality_level == QualityLevel.GOOD:
            base = "Good image quality!"
            if is_handwritten:
                return base + " Handwritten content detected - we'll do our best to read it."
            return base + " Processing should be accurate."
        else:  # EXCELLENT
            return "Excellent image quality! Processing will be highly accurate."


def analyze_image_quality(image_path: Path) -> QualityReport:
    """Convenience function to analyze image quality."""
    analyzer = ImageQualityAnalyzer()
    return analyzer.analyze(image_path)
