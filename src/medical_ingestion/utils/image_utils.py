# ============================================================================
# src/medical_ingestion/utils/image_utils.py
# ============================================================================
"""
Image utilities for the medical ingestion engine.

Provides:
- Image to PDF conversion for display
- Image format detection
- Image preprocessing
- EXIF orientation correction (critical for iPhone camera photos)
- HEIC/HEIF format conversion
- OCR-optimized image loading
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
from io import BytesIO

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Supported image extensions (includes HEIC/HEIF for iPhone photos)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.gif', '.heic', '.heif'}

# Max dimension for OCR processing — images larger than this get downscaled.
# iPhone 15 Pro shoots 4032x3024; most OCR engines work best at 1500-2500px.
OCR_MAX_DIMENSION = 2500


def is_image_file(file_path: Path) -> bool:
    """Check if a file is an image based on extension."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def detect_image_type(file_path: Path) -> Optional[str]:
    """
    Detect image type by reading magic bytes.

    Returns:
        Image type string ('png', 'jpeg', 'gif', 'tiff', 'bmp', 'webp', 'heic') or None
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)

        if header.startswith(b'\x89PNG'):
            return 'png'
        if header.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
            return 'gif'
        if header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
            return 'tiff'
        if header.startswith(b'BM'):
            return 'bmp'
        if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
            return 'webp'
        # HEIC/HEIF detection — ftyp box with heic/heix/mif1 brand
        if len(header) >= 12 and header[4:8] == b'ftyp':
            brand = header[8:12]
            if brand in (b'heic', b'heix', b'mif1', b'hevc'):
                return 'heic'

        return None
    except Exception:
        return None


def _convert_heic_to_pil(file_path: Path) -> Image.Image:
    """
    Convert a HEIC/HEIF file to a PIL Image.

    Tries pillow-heif first, falls back to ImageMagick convert.
    """
    # Try pillow-heif (pip install pillow-heif)
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        img = Image.open(file_path)
        logger.info(f"HEIC opened via pillow-heif: {img.size}")
        return img
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"pillow-heif failed: {e}")

    # Fallback: use subprocess to call ImageMagick or sips (macOS)
    import subprocess
    import tempfile

    # Try sips (built into macOS)
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ['sips', '-s', 'format', 'jpeg', str(file_path), '--out', tmp_path],
            capture_output=True, check=True, timeout=30
        )
        img = Image.open(tmp_path)
        img.load()  # Load into memory before we delete the temp file
        Path(tmp_path).unlink(missing_ok=True)
        logger.info(f"HEIC converted via sips: {img.size}")
        return img
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"sips conversion failed: {e}")
        Path(tmp_path).unlink(missing_ok=True)

    raise RuntimeError(
        f"Cannot open HEIC file: {file_path}. "
        "Install pillow-heif (pip install pillow-heif) or use macOS with sips."
    )


def load_image_for_ocr(
    file_path: Path,
    max_dimension: int = OCR_MAX_DIMENSION,
    ensure_rgb: bool = True,
) -> Image.Image:
    """
    Load an image with all corrections needed for reliable OCR.

    This is the single entry point for loading camera/phone images.
    It handles:
    1. HEIC/HEIF conversion (iPhone native format)
    2. EXIF orientation correction (critical for iPhone portrait photos)
    3. Downscaling oversized images (iPhone 12MP+ photos)
    4. Color mode conversion to RGB

    Args:
        file_path: Path to image file
        max_dimension: Maximum width or height (larger images are downscaled)
        ensure_rgb: Convert to RGB mode if True

    Returns:
        Corrected PIL Image ready for OCR
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Step 1: Handle HEIC/HEIF format
    if suffix in ('.heic', '.heif') or detect_image_type(file_path) == 'heic':
        image = _convert_heic_to_pil(file_path)
    else:
        image = Image.open(file_path)

    # Step 2: Apply EXIF orientation correction
    # iPhone stores photos in landscape pixel orientation with an EXIF tag
    # saying "rotate 90° for display". Without this call, OCR sees sideways text.
    try:
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        logger.warning(f"EXIF transpose failed (non-fatal): {e}")

    # Step 3: Downscale if too large for OCR
    w, h = image.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"Resized {w}x{h} -> {new_w}x{new_h} for OCR")

    # Step 4: Ensure RGB
    if ensure_rgb and image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')

    logger.debug(f"Image loaded for OCR: {image.size}, mode={image.mode}")
    return image


def save_prepared_image(image: Image.Image, output_path: Path) -> Path:
    """
    Save a prepared PIL Image to disk as PNG (lossless, OCR-friendly).

    Args:
        image: PIL Image (already EXIF-corrected, resized, etc.)
        output_path: Where to save

    Returns:
        The output path
    """
    image.save(output_path, format='PNG')
    return output_path


def convert_image_to_pdf(
    image_path: Path,
    output_path: Optional[Path] = None,
    dpi: int = 150,
    quality: int = 95
) -> Path:
    """
    Convert an image file to PDF format.

    Args:
        image_path: Path to the input image
        output_path: Path for output PDF (if None, uses same name with .pdf extension)
        dpi: Resolution for the PDF (affects file size and quality)
        quality: JPEG quality for compression (1-100)

    Returns:
        Path to the created PDF file
    """
    image_path = Path(image_path)

    if output_path is None:
        output_path = image_path.with_suffix('.pdf')
    else:
        output_path = Path(output_path)

    logger.info(f"Converting image to PDF: {image_path} -> {output_path}")

    try:
        # Open image with EXIF orientation correction
        image = load_image_for_ocr(image_path, max_dimension=99999, ensure_rgb=False)

        # Convert to RGB if necessary (PDF doesn't support RGBA well)
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Save as PDF
        image.save(
            output_path,
            'PDF',
            resolution=dpi,
            quality=quality
        )

        logger.info(f"Image converted to PDF successfully: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to convert image to PDF: {e}")
        raise


def convert_image_to_pdf_bytes(
    image_path: Path,
    dpi: int = 150,
    quality: int = 95
) -> bytes:
    """
    Convert an image file to PDF and return as bytes.

    Useful for API responses without writing to disk.

    Args:
        image_path: Path to the input image
        dpi: Resolution for the PDF
        quality: JPEG quality for compression

    Returns:
        PDF content as bytes
    """
    image_path = Path(image_path)

    try:
        # Open image with EXIF orientation correction
        image = load_image_for_ocr(image_path, max_dimension=99999, ensure_rgb=False)

        # Convert to RGB if necessary
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to bytes buffer
        buffer = BytesIO()
        image.save(
            buffer,
            'PDF',
            resolution=dpi,
            quality=quality
        )

        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Failed to convert image to PDF bytes: {e}")
        raise


def get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions (EXIF-corrected).

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            return img.size
    except Exception:
        return None


def ensure_pdf_for_display(file_path: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Ensure a file can be displayed as PDF.

    If the file is already a PDF, returns the original path.
    If it's an image, converts to PDF and returns the new path.

    Args:
        file_path: Path to the file
        output_dir: Directory for converted PDFs (defaults to same directory)

    Returns:
        Path to a PDF file
    """
    file_path = Path(file_path)

    # Already a PDF
    if file_path.suffix.lower() == '.pdf':
        return file_path

    # Check if it's an image
    if is_image_file(file_path) or detect_image_type(file_path):
        if output_dir:
            output_path = output_dir / f"{file_path.stem}.pdf"
        else:
            output_path = file_path.with_suffix('.pdf')

        return convert_image_to_pdf(file_path, output_path)

    # Unknown file type, return as-is
    logger.warning(f"Unknown file type, cannot convert to PDF: {file_path}")
    return file_path
