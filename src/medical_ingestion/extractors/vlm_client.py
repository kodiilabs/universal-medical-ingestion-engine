# ============================================================================
# src/medical_ingestion/extractors/vlm_client.py
# ============================================================================
"""
Vision Language Model (VLM) Client

Uses VLM for direct image-to-text extraction when OCR fails or confidence is low.
Supports multiple backends with HuggingFace GGUF preferred over Ollama hub.

Supported models:
- moondream2 (1.8B) - Fastest, smallest, good for basic VQA
- minicpm-v (3B) - Better quality, efficient
- llava (7B) - Best quality but slower

Architecture:
1. Try HuggingFace GGUF first (via local Ollama import)
2. Fall back to Ollama hub if HF not available
"""

import asyncio
import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class VLMExtractionResult:
    """Result from VLM extraction."""
    text: str
    raw_response: str
    confidence: float
    model: str
    extraction_time: float
    fields: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class VLMClient:
    """
    Vision Language Model client for document extraction.

    Uses Ollama as runtime but prefers HuggingFace GGUF models.
    """

    # Model preferences (HuggingFace GGUF names -> Ollama names)
    SUPPORTED_MODELS = {
        # Model name: (HF repo, GGUF file, Ollama name, size_gb)
        'moondream2': ('vikhyatk/moondream2', 'moondream2-text-model-f16.gguf', 'moondream', 1.8),
        'minicpm-v': ('openbmb/MiniCPM-V-2_6-gguf', 'MiniCPM-V-2_6-Q4_K_M.gguf', 'minicpm-v', 3.0),
        'llava-phi3': ('xtuner/llava-phi-3-mini-gguf', 'llava-phi-3-mini-f16.gguf', 'llava-phi3', 3.8),
    }

    DEFAULT_MODEL = 'minicpm-v'

    # Max image dimension for VLM input. Vision models tile images into patches
    # (336-448px each). A 2000px image = ~25 tiles = slow. 1024px = ~6 tiles = fast.
    # 1024px is more than enough resolution for reading text in medical documents.
    VLM_MAX_IMAGE_DIM = 1024

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize VLM client."""
        self.config = config or {}
        self.host = self.config.get('ollama_host', os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
        self.model_name = self.config.get('vlm_model', os.getenv('VLM_MODEL', self.DEFAULT_MODEL))
        self.timeout = self.config.get('vlm_timeout', int(os.getenv('VLM_TIMEOUT', '180')))
        self._session: Optional[aiohttp.ClientSession] = None
        self._model_verified = False

        logger.info(f"VLM Client initialized: {self.host} / {self.model_name}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def ensure_model(self) -> bool:
        """
        Ensure the VLM model is available.

        Strategy:
        1. Check if model already exists in Ollama
        2. If not, try to pull from Ollama hub
        3. Return True if model is available
        """
        if self._model_verified:
            return True

        session = await self._get_session()

        # Get Ollama model name
        ollama_name = self._get_ollama_name()

        try:
            # Check if model exists
            async with session.get(f"{self.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get('name', '').split(':')[0] for m in data.get('models', [])]

                    if ollama_name.split(':')[0] in models:
                        logger.info(f"VLM model {ollama_name} is available")
                        self._model_verified = True
                        return True

            # Model not found, try to pull
            logger.info(f"Pulling VLM model {ollama_name}...")
            async with session.post(
                f"{self.host}/api/pull",
                json={"name": ollama_name, "stream": False}
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully pulled {ollama_name}")
                    self._model_verified = True
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Failed to pull model: {error}")
                    return False

        except Exception as e:
            logger.error(f"Error ensuring VLM model: {e}")
            return False

    def _get_ollama_name(self) -> str:
        """Get Ollama model name for the configured model."""
        if self.model_name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[self.model_name][2]
        return self.model_name

    def _prepare_image(self, image_path: Path) -> str:
        """
        Apply EXIF orientation correction, resize to VLM_MAX_IMAGE_DIM,
        and return base64-encoded JPEG.

        Vision models tile images into fixed-size patches. Sending a 2000px image
        creates ~4x more tiles (and ~4x slower inference) than 1024px, with
        negligible quality difference for text extraction.
        """
        try:
            from PIL import Image as PILImage, ImageOps
            import io

            img = PILImage.open(image_path)

            # Apply EXIF orientation correction — critical for iPhone/Android
            # camera photos that store pixels in landscape with a rotation tag
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            w, h = img.size
            max_dim = self.VLM_MAX_IMAGE_DIM

            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
                logger.info(f"VLM: resized image {w}x{h} → {new_w}x{new_h} for faster inference")

            # Encode as JPEG to minimize payload size
            buf = io.BytesIO()
            img_format = 'JPEG'
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(buf, format=img_format, quality=85)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        except ImportError:
            logger.warning("Pillow not available, sending original image (may be slow)")
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Image resize failed ({e}), sending original")
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

    async def extract_from_image(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        extract_all: bool = True
    ) -> VLMExtractionResult:
        """
        Extract text and data from an image using VLM.

        Args:
            image_path: Path to the image file
            prompt: Custom extraction prompt (uses default if None)
            extract_all: If True, uses comprehensive extraction prompt

        Returns:
            VLMExtractionResult with extracted data
        """
        import time
        start_time = time.time()

        # Ensure model is available
        if not await self.ensure_model():
            return VLMExtractionResult(
                text="",
                raw_response="",
                confidence=0.0,
                model=self.model_name,
                extraction_time=0.0,
                warnings=["VLM model not available"]
            )

        # Read, resize, and encode image
        image_path = Path(image_path)
        if not image_path.exists():
            return VLMExtractionResult(
                text="",
                raw_response="",
                confidence=0.0,
                model=self.model_name,
                extraction_time=0.0,
                warnings=[f"Image not found: {image_path}"]
            )

        image_data = self._prepare_image(image_path)

        # Build prompt — track if caller provided a custom one (e.g. OCR-only)
        custom_prompt = prompt is not None
        if prompt is None:
            if extract_all:
                prompt = self._get_comprehensive_prompt()
            else:
                prompt = "Extract all text visible in this image. Include every word, number, and label you can see."

        # Call Ollama
        session = await self._get_session()
        ollama_name = self._get_ollama_name()

        try:
            # Custom prompts (OCR-only mode) need more tokens — a full page
            # of medical text can be 3000+ tokens.
            token_limit = 4096 if custom_prompt else 2000

            payload = {
                "model": ollama_name,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": token_limit
                }
            }

            async with session.post(
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    return VLMExtractionResult(
                        text="",
                        raw_response=error,
                        confidence=0.0,
                        model=ollama_name,
                        extraction_time=time.time() - start_time,
                        warnings=[f"Ollama error: {error}"]
                    )

                data = await response.json()

            raw_response = data.get('response', '')
            extraction_time = time.time() - start_time

            # Parse response — for custom prompts (OCR mode), the raw
            # response IS the text, skip structured parsing
            if custom_prompt:
                text = raw_response.strip()
                fields = {}
                confidence = 0.75 if len(text) > 100 else (0.5 if text else 0.0)
            else:
                text, fields, confidence = self._parse_response(raw_response)

            logger.info(
                f"VLM extraction complete: {len(text)} chars, "
                f"{len(fields)} fields, {extraction_time:.2f}s"
            )

            return VLMExtractionResult(
                text=text,
                raw_response=raw_response,
                confidence=confidence,
                model=ollama_name,
                extraction_time=extraction_time,
                fields=fields
            )

        except asyncio.TimeoutError:
            return VLMExtractionResult(
                text="",
                raw_response="",
                confidence=0.0,
                model=ollama_name,
                extraction_time=time.time() - start_time,
                warnings=["VLM extraction timed out"]
            )
        except Exception as e:
            logger.error(f"VLM extraction failed: {e}")
            return VLMExtractionResult(
                text="",
                raw_response="",
                confidence=0.0,
                model=ollama_name,
                extraction_time=time.time() - start_time,
                warnings=[f"VLM extraction error: {str(e)}"]
            )

    async def extract_from_image_bytes(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        extract_all: bool = True
    ) -> VLMExtractionResult:
        """
        Extract from image bytes directly.

        Args:
            image_bytes: Raw image bytes
            prompt: Custom extraction prompt
            extract_all: If True, uses comprehensive extraction prompt

        Returns:
            VLMExtractionResult
        """
        import tempfile
        import time

        start_time = time.time()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            return await self.extract_from_image(temp_path, prompt, extract_all)
        finally:
            os.unlink(temp_path)

    def _get_comprehensive_prompt(self) -> str:
        """Get comprehensive extraction prompt for documents."""
        return """Look at this document image carefully and extract ALL information you can see.

List EVERY piece of text, number, and data visible. Be extremely thorough.

Extract and organize:
1. HEADER/TITLE - Any title, logo text, company name at the top
2. NAMES - All person names, organization names
3. IDENTIFIERS - Reference numbers, IDs, account numbers, claim numbers
4. DATES - All dates in any format
5. AMOUNTS - All dollar amounts, fees, totals
6. ADDRESSES - Full addresses, cities, postal codes
7. CONTACT INFO - Phone numbers, emails, fax numbers
8. SERVICE DETAILS - What service/product, description, quantity
9. PAYMENT INFO - Payment method, card info, status
10. LABELS AND VALUES - Any "Label: Value" pairs
11. TABLE DATA - If there's a table, list all rows and columns
12. FOOTER - Any footer text, page numbers, fine print

Format your response as:
EXTRACTED TEXT:
[All visible text in reading order]

KEY-VALUE PAIRS:
[Field]: [Value]
[Field]: [Value]
...

Be complete. Do not skip any visible text."""

    def _parse_response(self, response: str) -> tuple[str, Dict[str, Any], float]:
        """
        Parse VLM response into structured data.

        Returns:
            (full_text, fields_dict, confidence)
        """
        import re

        if not response:
            return "", {}, 0.0

        text = response
        fields = {}
        confidence = 0.7  # Default confidence for VLM extraction

        # Try to extract key-value section
        kv_match = re.search(r'KEY-VALUE PAIRS:(.+?)(?=\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        if kv_match:
            kv_section = kv_match.group(1)
            # Parse key-value pairs
            for line in kv_section.strip().split('\n'):
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower().replace(' ', '_')
                        value = parts[1].strip()
                        if key and value and value.lower() not in ['n/a', 'none', 'null', '']:
                            fields[key] = value

        # Try to extract text section
        text_match = re.search(r'EXTRACTED TEXT:(.+?)(?=KEY-VALUE|$)', response, re.DOTALL | re.IGNORECASE)
        if text_match:
            text = text_match.group(1).strip()

        # Adjust confidence based on content
        if len(fields) > 10:
            confidence = 0.85
        elif len(fields) > 5:
            confidence = 0.75
        elif len(text) > 500:
            confidence = 0.7
        elif len(text) > 100:
            confidence = 0.6
        else:
            confidence = 0.5

        return text, fields, confidence


async def extract_with_vlm(
    image_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    prompt: Optional[str] = None
) -> VLMExtractionResult:
    """
    Convenience function for VLM extraction.

    Args:
        image_path: Path to image
        config: Optional configuration
        prompt: Optional custom prompt

    Returns:
        VLMExtractionResult
    """
    client = VLMClient(config)
    try:
        return await client.extract_from_image(image_path, prompt)
    finally:
        await client.close()
