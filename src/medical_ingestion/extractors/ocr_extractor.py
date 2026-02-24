# src/medical_ingestion/extractors/ocr_extractor.py
"""
OCR Extraction for Scanned PDFs and Images

Extracts text from image-based PDF pages using:
1. PaddleOCR (primary) - PP-OCRv5 server models, highest accuracy
2. EasyOCR (fallback) - Pure Python, good accuracy
3. Tesseract (fallback) - Faster, requires system install

Uses pypdfium2 for PDF-to-image conversion.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging
import shutil

import pypdfium2
from PIL import Image
import numpy as np


@dataclass
class WordBox:
    """A word/text segment with its bounding box."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in pixels
    confidence: float


@dataclass
class OCRResult:
    """Result from OCR extraction."""
    page_number: int
    text: str
    confidence: float
    method: str  # 'easyocr' or 'tesseract'
    language: str = 'en'
    word_boxes: List[WordBox] = field(default_factory=list)  # Words with bounding boxes
    page_width: int = 0
    page_height: int = 0

    def get_normalized_boxes(self) -> List[Dict[str, Any]]:
        """Get word boxes with coordinates normalized to 0-1 range."""
        if not self.page_width or not self.page_height:
            return []
        return [
            {
                "text": wb.text,
                "bbox": (
                    wb.bbox[0] / self.page_width,
                    wb.bbox[1] / self.page_height,
                    wb.bbox[2] / self.page_width,
                    wb.bbox[3] / self.page_height,
                ),
                "confidence": wb.confidence
            }
            for wb in self.word_boxes
        ]


@dataclass
class OCRExtractionResult:
    """Complete OCR extraction result."""
    pages: List[OCRResult] = field(default_factory=list)
    full_text: str = ""
    average_confidence: float = 0.0
    method: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class OCRExtractor:
    """
    OCR extraction with PaddleOCR (primary) + EasyOCR/Tesseract fallback.

    PaddleOCR is used as primary because:
    - PP-OCRv5 server models provide highest accuracy
    - Excellent for medical documents including handwriting
    - Better bounding box detection

    EasyOCR is used as first fallback because:
    - Pure Python, no system dependencies
    - Good accuracy on medical documents
    - Supports multiple languages

    Tesseract is used as last fallback because:
    - Faster processing
    - More widely available
    - Better for simple documents
    """

    # Default DPI for PDF rendering (higher = better OCR, slower)
    DEFAULT_DPI = 200

    # Medical document languages
    SUPPORTED_LANGUAGES = ['en']  # Add more as needed

    def __init__(self, use_gpu: bool = False):
        """
        Initialize OCR extractor.

        Args:
            use_gpu: Use GPU for OCR engines (requires CUDA)
        """
        self.logger = logging.getLogger(__name__)
        self.use_gpu = use_gpu

        # Lazy-loaded OCR readers
        self._paddleocr_reader = None
        self._easyocr_reader = None
        self._tesseract_available: Optional[bool] = None
        self._paddleocr_available: Optional[bool] = None

    @property
    def tesseract_available(self) -> bool:
        """Check if Tesseract is installed."""
        if self._tesseract_available is None:
            self._tesseract_available = shutil.which('tesseract') is not None
            if self._tesseract_available:
                self.logger.info("Tesseract OCR is available")
            else:
                self.logger.info("Tesseract OCR not found (optional fallback)")
        return self._tesseract_available

    @property
    def paddleocr_available(self) -> bool:
        """Check if PaddleOCR can be initialized."""
        if self._paddleocr_available is None:
            try:
                _ = self.paddleocr_reader
                self._paddleocr_available = True
            except Exception as e:
                self.logger.warning(f"PaddleOCR not available: {e}")
                self._paddleocr_available = False
        return self._paddleocr_available

    @property
    def paddleocr_reader(self):
        """Lazy-load PaddleOCR reader (PP-OCRv5 server models)."""
        if self._paddleocr_reader is None:
            try:
                from paddleocr import PaddleOCR
                self.logger.info("Initializing PaddleOCR (PP-OCRv5 server models)...")
                self._paddleocr_reader = PaddleOCR(
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    lang='en'
                )
                self.logger.info("PaddleOCR initialized successfully")
            except ImportError:
                self.logger.warning(
                    "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
                )
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise
        return self._paddleocr_reader

    def enhance_image(self, image: Image.Image, aggressive: bool = False) -> Image.Image:
        """
        Enhance image quality for better OCR results.

        Applies:
        - Grayscale conversion
        - Contrast enhancement
        - Sharpening
        - Noise reduction
        - Thresholding (for aggressive mode)

        Args:
            image: PIL Image to enhance
            aggressive: Apply more aggressive preprocessing for very poor quality

        Returns:
            Enhanced PIL Image
        """
        from PIL import ImageEnhance, ImageFilter

        try:
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Convert to grayscale for better OCR
            if image.mode == 'RGB':
                gray = image.convert('L')
            else:
                gray = image

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(1.5)  # Increase contrast by 50%

            # Sharpen the image
            enhanced = enhanced.filter(ImageFilter.SHARPEN)

            if aggressive:
                # Apply additional enhancements for poor quality images

                # More aggressive contrast
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.3)

                # Increase brightness slightly
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.1)

                # Apply unsharp mask for better edge detection
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

                # Apply adaptive thresholding using numpy
                img_array = np.array(enhanced)

                # Simple threshold to clean up
                threshold = 128
                img_array = np.where(img_array > threshold, 255, img_array)

                # Reduce noise with median filter (approximate)
                enhanced = Image.fromarray(img_array.astype(np.uint8))
                enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))

            self.logger.debug(f"Image enhanced: {image.size} -> {enhanced.size}")
            return enhanced

        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image  # Return original if enhancement fails

    def is_poor_quality(self, image: Image.Image) -> bool:
        """
        Detect if an image is of poor quality and needs aggressive enhancement.

        Checks:
        - Low contrast
        - High noise
        - Blurriness

        Args:
            image: PIL Image to check

        Returns:
            True if image quality is poor
        """
        try:
            # Convert to grayscale for analysis
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image

            img_array = np.array(gray)

            # Check contrast (standard deviation of pixel values)
            std_dev = np.std(img_array)
            low_contrast = std_dev < 40  # Low contrast threshold

            # Check if image is mostly uniform (possible blank or overexposed)
            mean_val = np.mean(img_array)
            too_bright = mean_val > 240
            too_dark = mean_val < 15

            # Check for high noise using local variance
            # (simplified - just check overall variance)
            is_noisy = std_dev > 80 and std_dev < 100  # High but not extreme variance

            is_poor = low_contrast or too_bright or too_dark or is_noisy

            if is_poor:
                self.logger.info(
                    f"Poor quality image detected: contrast={std_dev:.1f}, "
                    f"mean={mean_val:.1f}, bright={too_bright}, dark={too_dark}"
                )

            return is_poor

        except Exception as e:
            self.logger.warning(f"Quality check failed: {e}")
            return False

    @property
    def easyocr_available(self) -> bool:
        """Check if EasyOCR can be initialized (models downloaded, SSL works)."""
        if not hasattr(self, '_easyocr_available'):
            try:
                _ = self.easyocr_reader
                self._easyocr_available = True
            except Exception:
                self._easyocr_available = False
        return self._easyocr_available

    @property
    def easyocr_reader(self):
        """Lazy-load EasyOCR reader (heavy initialization)."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self.logger.info("Initializing EasyOCR (this may take a moment)...")
                self._easyocr_reader = easyocr.Reader(
                    self.SUPPORTED_LANGUAGES,
                    gpu=self.use_gpu
                )
                self.logger.info("EasyOCR initialized successfully")
            except ImportError:
                self.logger.warning("EasyOCR not installed. Install with: pip install easyocr")
                raise
            except Exception as e:
                # Handle SSL errors and model download failures
                if 'SSL' in str(e) or 'certificate' in str(e).lower():
                    self.logger.error(
                        f"EasyOCR SSL error - run: /Applications/Python\\ 3.11/Install\\ Certificates.command\n"
                        f"Or: pip install certifi && export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')"
                    )
                self.logger.error(f"Failed to initialize EasyOCR: {e}")
                raise
        return self._easyocr_reader

    def extract_text(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None,
        dpi: int = DEFAULT_DPI,
        prefer_tesseract: bool = False
    ) -> OCRExtractionResult:
        """
        Extract text from PDF using OCR.

        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to OCR (0-indexed), None for all
            dpi: Resolution for rendering (higher = better quality, slower)
            prefer_tesseract: Use Tesseract as primary (faster but may be less accurate)

        Returns:
            OCRExtractionResult with extracted text and metadata
        """
        result = OCRExtractionResult()

        # Convert PDF pages to images
        try:
            images = self._pdf_to_images(pdf_path, pages, dpi)
        except Exception as e:
            result.errors.append(f"Failed to convert PDF to images: {e}")
            return result

        if not images:
            result.warnings.append("No pages to OCR")
            return result

        # OCR each image
        confidences = []
        texts = []

        for page_num, image in images:
            try:
                page_width, page_height = image.size
                method = None
                text = ""
                confidence = 0.0
                word_boxes = []

                # Try PaddleOCR first (unless user prefers Tesseract)
                if not prefer_tesseract:
                    try:
                        text, confidence, word_boxes = self._ocr_with_paddleocr(image, return_boxes=True)
                        method = 'paddleocr'
                    except Exception as paddle_err:
                        self.logger.debug(f"PaddleOCR failed for page {page_num}: {paddle_err}")

                # Fall back to EasyOCR/Tesseract if PaddleOCR failed or returned no text
                if not text.strip():
                    if prefer_tesseract and self.tesseract_available:
                        text, confidence, word_boxes = self._ocr_with_tesseract(image, return_boxes=True)
                        method = 'tesseract'
                    else:
                        try:
                            text, confidence, word_boxes = self._ocr_with_easyocr(image, return_boxes=True)
                            method = 'easyocr'
                        except Exception as e:
                            if self.tesseract_available:
                                text, confidence, word_boxes = self._ocr_with_tesseract(image, return_boxes=True)
                                method = 'tesseract_fallback'
                            else:
                                raise

                ocr_result = OCRResult(
                    page_number=page_num,
                    text=text,
                    confidence=confidence,
                    method=method or 'unknown',
                    word_boxes=word_boxes,
                    page_width=page_width,
                    page_height=page_height
                )
                result.pages.append(ocr_result)
                confidences.append(confidence)
                texts.append(text)

                self.logger.debug(
                    f"Page {page_num}: {len(text)} chars, {len(word_boxes)} boxes, "
                    f"{confidence:.2f} confidence ({method})"
                )

            except Exception as e:
                self.logger.warning(f"OCR failed for page {page_num}: {e}")
                result.errors.append(f"Page {page_num}: OCR failed: {e}")

        # Compile results
        result.full_text = "\n\n".join(texts)
        result.average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        result.method = result.pages[0].method if result.pages else ""

        self.logger.info(
            f"OCR complete: {len(result.pages)} pages, "
            f"{len(result.full_text)} chars, "
            f"{result.average_confidence:.2f} avg confidence"
        )

        return result

    def _pdf_to_images(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        dpi: int
    ) -> List[Tuple[int, Image.Image]]:
        """
        Convert PDF pages to PIL Images using pypdfium2.

        Args:
            pdf_path: Path to PDF
            pages: Page numbers to convert (0-indexed), None for all
            dpi: Rendering resolution

        Returns:
            List of (page_number, PIL.Image) tuples
        """
        images = []
        scale = dpi / 72.0  # PDF points to pixels

        pdf = pypdfium2.PdfDocument(str(pdf_path))

        page_indices = pages if pages is not None else range(len(pdf))

        for page_num in page_indices:
            if page_num >= len(pdf):
                self.logger.warning(f"Page {page_num} out of range, skipping")
                continue

            page = pdf[page_num]

            # Render to bitmap
            bitmap = page.render(scale=scale)

            # Convert to PIL Image
            pil_image = bitmap.to_pil()

            images.append((page_num, pil_image))

        pdf.close()

        return images

    def _ocr_with_paddleocr(
        self, image: Image.Image, return_boxes: bool = False
    ) -> Tuple[str, float, List[WordBox]]:
        """
        OCR using PaddleOCR (PP-OCRv5 server models).

        This provides the highest accuracy for medical documents.

        Args:
            image: PIL Image
            return_boxes: Whether to return word bounding boxes

        Returns:
            (extracted_text, confidence, word_boxes)
        """
        import tempfile
        import os

        # PaddleOCR works with file paths, so save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image.save(f, format='PNG')
            temp_path = f.name

        try:
            # Run PaddleOCR prediction
            result = self.paddleocr_reader.predict(input=temp_path)

            texts = []
            confidences = []
            word_boxes = []

            # Process results - result is a list of dict-like OCRResult objects
            for res in result:
                # OCRResult is dict-like, access via .get() or []
                rec_texts = res.get('rec_texts', []) if hasattr(res, 'get') else getattr(res, 'rec_texts', [])
                rec_scores = res.get('rec_scores', []) if hasattr(res, 'get') else getattr(res, 'rec_scores', [])
                dt_polys = res.get('dt_polys', []) if hasattr(res, 'get') else getattr(res, 'dt_polys', [])

                if rec_texts:
                    for i, text in enumerate(rec_texts):
                        if text and text.strip():
                            texts.append(text.strip())

                            # Get confidence score
                            if rec_scores and i < len(rec_scores):
                                conf = float(rec_scores[i])
                                confidences.append(conf)
                            else:
                                confidences.append(1.0)

                            # Get bounding box
                            if return_boxes and dt_polys and i < len(dt_polys):
                                poly = dt_polys[i]
                                # Convert polygon to bbox (x0, y0, x1, y1)
                                x_coords = [p[0] for p in poly]
                                y_coords = [p[1] for p in poly]
                                bbox = (
                                    float(min(x_coords)),
                                    float(min(y_coords)),
                                    float(max(x_coords)),
                                    float(max(y_coords))
                                )
                                word_boxes.append(WordBox(
                                    text=text.strip(),
                                    bbox=bbox,
                                    confidence=confidences[-1]
                                ))

            if not texts:
                return "", 0.0, []

            full_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return full_text, avg_confidence, word_boxes

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    def _ocr_with_easyocr(
        self, image: Image.Image, return_boxes: bool = False
    ) -> Tuple[str, float, List[WordBox]]:
        """
        OCR using EasyOCR (fallback when PaddleOCR unavailable).

        Args:
            image: PIL Image
            return_boxes: Whether to return word bounding boxes

        Returns:
            (extracted_text, confidence, word_boxes)
        """
        # Convert to numpy array (EasyOCR expects this)
        image_np = np.array(image)

        # Run OCR
        results = self.easyocr_reader.readtext(image_np)

        if not results:
            return "", 0.0, []

        # Extract text, confidence, and bounding boxes
        texts = []
        confidences = []
        word_boxes = []

        for bbox_points, text, confidence in results:
            texts.append(text)
            confidences.append(confidence)

            if return_boxes:
                # EasyOCR bbox is [[x0,y0], [x1,y0], [x1,y1], [x0,y1]] (4 corners)
                # Convert to (x0, y0, x1, y1) format
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                word_boxes.append(WordBox(text=text, bbox=bbox, confidence=confidence))

        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return full_text, avg_confidence, word_boxes

    def _ocr_with_tesseract(
        self, image: Image.Image, return_boxes: bool = False
    ) -> Tuple[str, float, List[WordBox]]:
        """
        OCR using Tesseract.

        Args:
            image: PIL Image
            return_boxes: Whether to return word bounding boxes

        Returns:
            (extracted_text, confidence, word_boxes)
        """
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pytesseract not installed. Install with: pip install pytesseract")

        # Get text with confidence and position data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        texts = []
        confidences = []
        word_boxes = []

        for i, conf in enumerate(data['conf']):
            if conf > 0:  # Valid confidence
                text = data['text'][i].strip()
                if text:
                    texts.append(text)
                    conf_normalized = conf / 100.0  # Normalize to 0-1
                    confidences.append(conf_normalized)

                    if return_boxes:
                        # Tesseract provides left, top, width, height
                        x0 = data['left'][i]
                        y0 = data['top'][i]
                        x1 = x0 + data['width'][i]
                        y1 = y0 + data['height'][i]
                        word_boxes.append(WordBox(
                            text=text,
                            bbox=(x0, y0, x1, y1),
                            confidence=conf_normalized
                        ))

        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return full_text, avg_confidence, word_boxes

    def ocr_single_page(
        self,
        pdf_path: Path,
        page_number: int,
        dpi: int = DEFAULT_DPI
    ) -> OCRResult:
        """
        OCR a single page from a PDF.

        Args:
            pdf_path: Path to PDF
            page_number: Page to OCR (0-indexed)
            dpi: Rendering resolution

        Returns:
            OCRResult for the page
        """
        result = self.extract_text(pdf_path, pages=[page_number], dpi=dpi)

        if result.pages:
            return result.pages[0]

        # Return empty result on failure
        return OCRResult(
            page_number=page_number,
            text="",
            confidence=0.0,
            method="failed"
        )

    def extract_text_from_image(
        self,
        image_path: Path,
        prefer_tesseract: bool = False,
        enhance: bool = True
    ) -> OCRResult:
        """
        Extract text from an image file using OCR.

        Supports: PNG, JPG, JPEG, TIFF, BMP, WEBP, GIF

        Strategy: Try PaddleOCR first (highest accuracy), then fall back to
        EasyOCR or Tesseract if PaddleOCR fails.

        Args:
            image_path: Path to image file
            prefer_tesseract: Use Tesseract as primary OCR engine (skip PaddleOCR/EasyOCR)
            enhance: Apply image enhancement for better OCR (only used with fallback engines)

        Returns:
            OCRResult with extracted text and word boxes
        """
        image_path = Path(image_path)
        self.logger.info(f"Extracting text from image: {image_path}")

        try:
            # Load image with EXIF orientation correction, HEIC support, and resize
            from ..utils.image_utils import load_image_for_ocr
            image = load_image_for_ocr(image_path)

            page_width, page_height = image.size

            # STEP 1: Try PaddleOCR first (highest accuracy for medical documents)
            # Unless user explicitly prefers Tesseract
            if not prefer_tesseract:
                try:
                    text, confidence, word_boxes = self._ocr_with_paddleocr(image, return_boxes=True)
                    method = 'paddleocr'
                    self.logger.info(f"PaddleOCR: {len(text)} chars, {confidence:.2f} confidence")

                    # PaddleOCR usually gives good results, return if we got text
                    if text.strip():
                        return OCRResult(
                            page_number=0,
                            text=text,
                            confidence=confidence,
                            method=method,
                            word_boxes=word_boxes,
                            page_width=page_width,
                            page_height=page_height
                        )

                except Exception as e:
                    self.logger.warning(f"PaddleOCR failed ({e}), falling back to EasyOCR/Tesseract")

            # STEP 2: Fall back to EasyOCR or Tesseract
            if prefer_tesseract and self.tesseract_available:
                text, confidence, word_boxes = self._ocr_with_tesseract(image, return_boxes=True)
                method = 'tesseract'
            else:
                try:
                    text, confidence, word_boxes = self._ocr_with_easyocr(image, return_boxes=True)
                    method = 'easyocr'
                except Exception as e:
                    # EasyOCR failed (likely SSL/model download issue), fall back to Tesseract
                    self.logger.warning(f"EasyOCR failed ({e}), falling back to Tesseract")
                    if self.tesseract_available:
                        text, confidence, word_boxes = self._ocr_with_tesseract(image, return_boxes=True)
                        method = 'tesseract_fallback'
                    else:
                        raise RuntimeError(
                            "All OCR engines failed. Install PaddleOCR: pip install paddlepaddle paddleocr\n"
                            "Or Tesseract: brew install tesseract"
                        )

            self.logger.debug(f"Original image OCR: {len(text)} chars, {confidence:.2f} confidence")

            # STEP 2: If enhancement is enabled and results are poor, try enhanced versions
            if enhance and (len(text.strip()) < 100 or confidence < 0.5):
                self.logger.info(f"Trying enhanced OCR (original: {len(text)} chars, {confidence:.2f} conf)")

                # Reload original image for enhancement (with EXIF correction)
                original_image = load_image_for_ocr(image_path)

                # Try mild enhancement first
                enhanced = self.enhance_image(original_image, aggressive=False)
                if prefer_tesseract and self.tesseract_available:
                    text2, conf2, boxes2 = self._ocr_with_tesseract(enhanced, return_boxes=True)
                else:
                    text2, conf2, boxes2 = self._ocr_with_easyocr(enhanced, return_boxes=True)

                # Use enhanced result if it's better (more text AND reasonable confidence)
                if len(text2) > len(text) * 1.1 and conf2 >= confidence * 0.8:
                    text, confidence, word_boxes = text2, conf2, boxes2
                    method = f"{method}_enhanced"
                    self.logger.debug(f"Using mild enhanced: {len(text)} chars, {confidence:.2f}")

                # If still poor, try aggressive enhancement
                if len(text.strip()) < 100 or confidence < 0.5:
                    original_image = load_image_for_ocr(image_path)

                    aggressive_enhanced = self.enhance_image(original_image, aggressive=True)
                    if prefer_tesseract and self.tesseract_available:
                        text3, conf3, boxes3 = self._ocr_with_tesseract(aggressive_enhanced, return_boxes=True)
                    else:
                        text3, conf3, boxes3 = self._ocr_with_easyocr(aggressive_enhanced, return_boxes=True)

                    # Use aggressive result only if significantly better
                    if len(text3) > len(text) * 1.2 and conf3 >= confidence * 0.7:
                        text, confidence, word_boxes = text3, conf3, boxes3
                        method = f"{method}_aggressive"
                        self.logger.debug(f"Using aggressive enhanced: {len(text)} chars, {confidence:.2f}")

            self.logger.info(
                f"Image OCR: {len(text)} chars, {len(word_boxes)} boxes, "
                f"{confidence:.2f} confidence ({method})"
            )

            return OCRResult(
                page_number=0,
                text=text,
                confidence=confidence,
                method=method,
                word_boxes=word_boxes,
                page_width=page_width,
                page_height=page_height
            )

        except Exception as e:
            self.logger.error(f"Image OCR failed: {e}")

            # Try fallback method
            if not prefer_tesseract and self.tesseract_available:
                try:
                    image = Image.open(image_path)
                    if image.mode not in ('RGB', 'L'):
                        image = image.convert('RGB')
                    page_width, page_height = image.size

                    text, confidence, word_boxes = self._ocr_with_tesseract(image, return_boxes=True)

                    self.logger.info(f"Image OCR fallback (Tesseract): {len(text)} chars")

                    return OCRResult(
                        page_number=0,
                        text=text,
                        confidence=confidence,
                        method='tesseract',
                        word_boxes=word_boxes,
                        page_width=page_width,
                        page_height=page_height
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Tesseract fallback also failed: {fallback_error}")

            # Return empty result on complete failure
            return OCRResult(
                page_number=0,
                text="",
                confidence=0.0,
                method="failed"
            )

    def find_text_bbox(
        self,
        ocr_result: OCRResult,
        search_text: str,
        fuzzy: bool = True
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Find the bounding box for a specific text in OCR results.

        Args:
            ocr_result: OCR result with word boxes
            search_text: Text to find
            fuzzy: Allow partial/fuzzy matching

        Returns:
            Normalized bbox (0-1 coordinates) or None if not found
        """
        if not ocr_result.word_boxes or not ocr_result.page_width:
            return None

        search_lower = search_text.lower().strip()

        # Try exact match first
        for wb in ocr_result.word_boxes:
            if wb.text.lower().strip() == search_lower:
                return (
                    wb.bbox[0] / ocr_result.page_width,
                    wb.bbox[1] / ocr_result.page_height,
                    wb.bbox[2] / ocr_result.page_width,
                    wb.bbox[3] / ocr_result.page_height,
                )

        # Try fuzzy match (contains)
        if fuzzy:
            for wb in ocr_result.word_boxes:
                if search_lower in wb.text.lower() or wb.text.lower() in search_lower:
                    return (
                        wb.bbox[0] / ocr_result.page_width,
                        wb.bbox[1] / ocr_result.page_height,
                        wb.bbox[2] / ocr_result.page_width,
                        wb.bbox[3] / ocr_result.page_height,
                    )

        return None

    def find_value_bbox(
        self,
        ocr_result: OCRResult,
        field_name: str,
        value: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find bounding boxes for both field name and value in OCR results.

        Args:
            ocr_result: OCR result with word boxes
            field_name: The field/test name
            value: The extracted value

        Returns:
            Dict with 'field_bbox', 'value_bbox', 'combined_bbox' (all normalized)
        """
        field_bbox = self.find_text_bbox(ocr_result, field_name)
        value_bbox = self.find_text_bbox(ocr_result, str(value))

        if not field_bbox and not value_bbox:
            return None

        result = {
            "field_bbox": field_bbox,
            "value_bbox": value_bbox,
        }

        # Calculate combined bbox if both found
        if field_bbox and value_bbox:
            result["combined_bbox"] = (
                min(field_bbox[0], value_bbox[0]),
                min(field_bbox[1], value_bbox[1]),
                max(field_bbox[2], value_bbox[2]),
                max(field_bbox[3], value_bbox[3]),
            )
        elif field_bbox:
            result["combined_bbox"] = field_bbox
        else:
            result["combined_bbox"] = value_bbox

        return result

    def extract_handwritten_text(
        self,
        image_path: Path,
        quality_report: Optional[Any] = None
    ) -> OCRResult:
        """
        Extract text from handwritten documents with specialized handling.

        Uses PaddleOCR first (PP-OCRv5 has good handwriting support),
        then falls back to multi-pass EasyOCR/Tesseract if needed.

        Args:
            image_path: Path to image file
            quality_report: Optional QualityReport from image_quality analyzer

        Returns:
            OCRResult optimized for handwritten content
        """
        image_path = Path(image_path)
        self.logger.info(f"Extracting handwritten text from: {image_path}")

        try:
            image = Image.open(image_path)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            page_width, page_height = image.size

            # Try PaddleOCR first - PP-OCRv5 has good handwriting support
            best_text = ""
            best_confidence = 0.0
            best_boxes = []
            best_method = "paddleocr_handwriting"
            paddleocr_failed = False

            try:
                text_p, conf_p, boxes_p = self._ocr_with_paddleocr(image, return_boxes=True)
                if text_p.strip():
                    best_text, best_confidence, best_boxes = text_p, conf_p, boxes_p
                    self.logger.info(f"PaddleOCR handwriting: {len(best_text)} chars, {best_confidence:.2f}")
            except Exception as e:
                self.logger.warning(f"PaddleOCR failed for handwriting ({e})")
                paddleocr_failed = True

            # If PaddleOCR gave poor results, try enhanced multi-pass approach
            if len(best_text) < 50 or best_confidence < 0.5:
                # Apply specialized handwriting preprocessing
                enhanced_image = self._preprocess_for_handwriting(image)
                easyocr_failed = False

                try:
                    # Pass 1: Standard enhanced with EasyOCR
                    text1, conf1, boxes1 = self._ocr_with_easyocr(enhanced_image, return_boxes=True)

                    if len(text1) > len(best_text):
                        best_text, best_confidence, best_boxes = text1, conf1, boxes1
                        best_method = "easyocr_handwriting"

                    # Pass 2: High contrast version
                    high_contrast = self._apply_high_contrast(image)
                    text2, conf2, boxes2 = self._ocr_with_easyocr(high_contrast, return_boxes=True)

                    if len(text2) > len(best_text) or conf2 > best_confidence:
                        best_text, best_confidence, best_boxes = text2, conf2, boxes2
                        best_method = "easyocr_handwriting_highcontrast"

                    # Pass 3: Adaptive threshold version
                    adaptive = self._apply_adaptive_threshold(image)
                    text3, conf3, boxes3 = self._ocr_with_easyocr(adaptive, return_boxes=True)

                    if len(text3) > len(best_text):
                        best_text, best_confidence, best_boxes = text3, conf3, boxes3
                        best_method = "easyocr_handwriting_adaptive"

                except Exception as e:
                    # EasyOCR failed (likely SSL/model download issue)
                    self.logger.warning(f"EasyOCR failed for handwriting ({e})")
                    easyocr_failed = True

                # Use Tesseract if other methods failed or results are poor
                if (easyocr_failed or len(best_text) < 20) and self.tesseract_available:
                    self.logger.info("Using Tesseract for handwriting")
                    text_t, conf_t, boxes_t = self._ocr_handwriting_tesseract(enhanced_image)

                    if len(text_t) > len(best_text):
                        best_text, best_confidence, best_boxes = text_t, conf_t, boxes_t
                        best_method = "tesseract_handwriting"

            # If still no results and all methods failed, raise error
            if not best_text and paddleocr_failed and not self.tesseract_available:
                raise RuntimeError(
                    "EasyOCR failed and Tesseract not available. "
                    "Install Tesseract: brew install tesseract"
                )

            self.logger.info(
                f"Handwriting OCR: {len(best_text)} chars, "
                f"{best_confidence:.2f} confidence ({best_method})"
            )

            return OCRResult(
                page_number=0,
                text=best_text,
                confidence=best_confidence,
                method=best_method,
                word_boxes=best_boxes,
                page_width=page_width,
                page_height=page_height
            )

        except Exception as e:
            self.logger.error(f"Handwriting OCR failed: {e}")
            return OCRResult(
                page_number=0,
                text="",
                confidence=0.0,
                method="failed"
            )

    def _preprocess_for_handwriting(self, image: Image.Image) -> Image.Image:
        """Apply specialized preprocessing for handwritten text."""
        from PIL import ImageEnhance, ImageFilter

        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        # Increase size for better recognition (handwriting needs more pixels)
        width, height = gray.size
        if width < 2000:
            scale = 2000 / width
            new_size = (int(width * scale), int(height * scale))
            gray = gray.resize(new_size, Image.Resampling.LANCZOS)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)

        # Sharpen
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)  # Double sharpen for handwriting

        # Slight denoise
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))

        return enhanced

    def _apply_high_contrast(self, image: Image.Image) -> Image.Image:
        """Apply high contrast preprocessing."""
        from PIL import ImageEnhance

        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        # Very high contrast
        enhancer = ImageEnhance.Contrast(gray)
        return enhancer.enhance(2.5)

    def _apply_adaptive_threshold(self, image: Image.Image) -> Image.Image:
        """Apply adaptive thresholding for handwriting."""
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        img_array = np.array(gray)

        # Simple adaptive threshold using block means
        block_size = 31
        h, w = img_array.shape
        result = np.zeros_like(img_array)

        for i in range(0, h, block_size // 2):
            for j in range(0, w, block_size // 2):
                block = img_array[
                    max(0, i - block_size // 2):min(h, i + block_size // 2),
                    max(0, j - block_size // 2):min(w, j + block_size // 2)
                ]
                threshold = np.mean(block) - 10
                region = img_array[i:min(h, i + block_size // 2), j:min(w, j + block_size // 2)]
                result[i:min(h, i + block_size // 2), j:min(w, j + block_size // 2)] = (
                    np.where(region < threshold, 0, 255)
                )

        return Image.fromarray(result.astype(np.uint8))

    def _ocr_handwriting_tesseract(
        self, image: Image.Image
    ) -> Tuple[str, float, List[WordBox]]:
        """OCR with Tesseract configured for handwriting."""
        try:
            import pytesseract
        except ImportError:
            return "", 0.0, []

        # Use PSM 6 (single uniform block) which often works better for handwriting
        # Use OEM 1 (LSTM neural net only)
        custom_config = r'--oem 1 --psm 6'

        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=custom_config
        )

        texts = []
        confidences = []
        word_boxes = []

        for i, conf in enumerate(data['conf']):
            if conf > 0:
                text = data['text'][i].strip()
                if text:
                    texts.append(text)
                    conf_normalized = conf / 100.0
                    confidences.append(conf_normalized)

                    x0 = data['left'][i]
                    y0 = data['top'][i]
                    x1 = x0 + data['width'][i]
                    y1 = y0 + data['height'][i]
                    word_boxes.append(WordBox(
                        text=text,
                        bbox=(x0, y0, x1, y1),
                        confidence=conf_normalized
                    ))

        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return full_text, avg_confidence, word_boxes

    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about available OCR capabilities."""
        return {
            "paddleocr_available": self.paddleocr_available,  # Primary OCR (highest accuracy)
            "easyocr_available": True,  # Fallback (pure Python)
            "tesseract_available": self.tesseract_available,  # Fallback (system install)
            "gpu_enabled": self.use_gpu,
            "supported_languages": self.SUPPORTED_LANGUAGES,
            "default_dpi": self.DEFAULT_DPI,
            "handwriting_support": True,
            "primary_engine": "paddleocr" if self.paddleocr_available else "easyocr"
        }
