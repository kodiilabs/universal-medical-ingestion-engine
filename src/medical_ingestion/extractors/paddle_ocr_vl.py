# ============================================================================
# src/medical_ingestion/extractors/paddle_ocr_vl.py
# ============================================================================
"""
PaddleOCR-VL Client for Document Extraction

Uses PaddleOCR-VL (0.9B) via Ollama for vision-language document parsing.
This is a SOTA model specifically designed for document understanding.

Key capabilities:
- Native table/form understanding
- Layout-aware text extraction
- Structured data extraction directly from images

Model: MedAIBase/PaddleOCR-VL:0.9b
"""

import aiohttp
import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Default model name
DEFAULT_MODEL = "MedAIBase/PaddleOCR-VL:0.9b"


class PaddleOCRVLClient:
    """
    PaddleOCR-VL client using Ollama for vision-language document extraction.

    This model understands document layout and can extract structured data
    directly from images without separate OCR + parsing steps.

    Use as async context manager for proper cleanup:
        async with PaddleOCRVLClient(config) as client:
            result = await client.extract_from_image(path)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        self.host = config.get('ollama_host', 'http://localhost:11434')
        self.model_name = config.get('paddle_ocr_vl_model', DEFAULT_MODEL)
        self.timeout = config.get('timeout', 180)  # VLMs can be slower

        self._session: Optional[aiohttp.ClientSession] = None
        self._session_loop: Optional[asyncio.AbstractEventLoop] = None

        # Statistics
        self._inference_count = 0
        self._total_inference_time = 0.0

        logger.info(f"Initialized PaddleOCR-VL client: {self.host} / {self.model_name}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures session cleanup."""
        await self.close()
        return False

    def __del__(self):
        """Destructor - attempt to close session if not already closed."""
        if self._session is not None and not self._session.closed:
            try:
                # Can't await in __del__, so schedule cleanup
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass  # Best effort cleanup

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        needs_new_session = (
            self._session is None
            or self._session.closed
            or self._session_loop is None
            or self._session_loop != current_loop
            or (self._session_loop is not None and self._session_loop.is_closed())
        )

        if needs_new_session:
            if self._session is not None and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._session_loop = current_loop

        return self._session

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._session_loop = None

    async def health_check(self) -> Dict[str, Any]:
        """Check if Ollama server is running and model is available."""
        try:
            session = await self._get_session()

            async with session.get(f"{self.host}/api/tags") as response:
                if response.status != 200:
                    return {
                        "healthy": False,
                        "backend": "ollama",
                        "model": self.model_name,
                        "details": f"Ollama server returned status {response.status}"
                    }

                data = await response.json()
                models = [m.get('name', '') for m in data.get('models', [])]

                model_available = any(self.model_name in m for m in models)

                if not model_available:
                    return {
                        "healthy": False,
                        "backend": "ollama",
                        "model": self.model_name,
                        "details": f"Model not found. Run: ollama pull {self.model_name}"
                    }

                return {
                    "healthy": True,
                    "backend": "ollama",
                    "model": self.model_name,
                    "details": "PaddleOCR-VL model available"
                }

        except aiohttp.ClientConnectorError:
            return {
                "healthy": False,
                "backend": "ollama",
                "model": self.model_name,
                "details": f"Cannot connect to Ollama at {self.host}"
            }
        except Exception as e:
            return {
                "healthy": False,
                "backend": "ollama",
                "model": self.model_name,
                "details": f"Health check failed: {str(e)}"
            }

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for Ollama API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_image_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def extract_from_image(
        self,
        image_path: Path,
        extraction_type: str = "lab_results"
    ) -> Dict[str, Any]:
        """
        Extract structured data from a document image.

        Args:
            image_path: Path to image file (PNG, JPG, etc.)
            extraction_type: Type of extraction ("lab_results", "prescription", "general")

        Returns:
            Dict with extracted values, units, and metadata
        """
        start_time = datetime.now()

        # Build prompt based on extraction type
        prompt = self._build_extraction_prompt(extraction_type)

        try:
            session = await self._get_session()

            # Encode image
            image_b64 = self._encode_image(image_path)

            # Ollama vision API payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2000,
                }
            }

            async with session.post(
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama error ({response.status}): {error_text}")

                data = await response.json()

            # Parse response
            response_text = data.get('response', '')
            inference_time = (datetime.now() - start_time).total_seconds()

            # Update stats
            self._inference_count += 1
            self._total_inference_time += inference_time

            # Parse JSON response
            try:
                extracted = json.loads(response_text)
            except json.JSONDecodeError:
                try:
                    from json_repair import repair_json
                    extracted = repair_json(response_text, return_objects=True)
                except Exception:
                    extracted = {"raw_text": response_text, "parse_error": True}

            logger.info(
                f"PaddleOCR-VL extracted data in {inference_time:.2f}s "
                f"from {image_path.name}"
            )

            return {
                "extracted": extracted,
                "inference_time": inference_time,
                "model": self.model_name,
                "source": str(image_path)
            }

        except aiohttp.ClientConnectorError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            logger.error(f"PaddleOCR-VL extraction failed: {e}")
            raise

    async def extract_from_image_bytes(
        self,
        image_bytes: bytes,
        extraction_type: str = "lab_results"
    ) -> Dict[str, Any]:
        """
        Extract structured data from image bytes.

        Args:
            image_bytes: Raw image bytes
            extraction_type: Type of extraction

        Returns:
            Dict with extracted values
        """
        start_time = datetime.now()
        prompt = self._build_extraction_prompt(extraction_type)

        try:
            session = await self._get_session()

            image_b64 = self._encode_image_bytes(image_bytes)

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2000,
                }
            }

            async with session.post(
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama error ({response.status}): {error_text}")

                data = await response.json()

            response_text = data.get('response', '')
            inference_time = (datetime.now() - start_time).total_seconds()

            self._inference_count += 1
            self._total_inference_time += inference_time

            try:
                extracted = json.loads(response_text)
            except json.JSONDecodeError:
                try:
                    from json_repair import repair_json
                    extracted = repair_json(response_text, return_objects=True)
                except Exception:
                    extracted = {"raw_text": response_text, "parse_error": True}

            return {
                "extracted": extracted,
                "inference_time": inference_time,
                "model": self.model_name
            }

        except Exception as e:
            logger.error(f"PaddleOCR-VL extraction failed: {e}")
            raise

    def _build_extraction_prompt(self, extraction_type: str) -> str:
        """Build extraction prompt based on document type."""

        if extraction_type == "lab_results":
            return """Analyze this medical lab report image and extract ALL lab test results.

Return a JSON object with this exact structure:
{
    "results": [
        {
            "test_name": "exact test name as shown",
            "value": numeric_value_only,
            "unit": "unit string",
            "reference_range": "range if shown",
            "flag": "H/L/Normal if shown"
        }
    ],
    "patient_info": {
        "name": "if visible",
        "dob": "if visible",
        "collection_date": "if visible"
    },
    "lab_info": {
        "lab_name": "if visible",
        "report_date": "if visible"
    }
}

Extract EVERY lab value visible in the document. Be precise with numbers.
For values like "1024 High", extract value as 1024 and flag as "H".
"""

        elif extraction_type == "prescription":
            return """Analyze this prescription image and extract medication information.

Return a JSON object with this structure:
{
    "medications": [
        {
            "drug_name": "medication name",
            "dosage": "dosage amount",
            "frequency": "how often",
            "duration": "for how long",
            "instructions": "any special instructions"
        }
    ],
    "prescriber": {
        "name": "doctor name if visible",
        "npi": "NPI if visible"
    },
    "patient": {
        "name": "patient name if visible"
    },
    "date": "prescription date if visible"
}
"""

        else:  # general
            return """Analyze this medical document image and extract all structured information.

Return a JSON object containing:
{
    "document_type": "type of document",
    "content": {
        // All extracted fields and values
    },
    "tables": [
        // Any tables found with headers and rows
    ],
    "text_blocks": [
        // Major text sections
    ]
}

Be thorough and extract all visible information.
"""

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0 else 0
        )

        return {
            "model": self.model_name,
            "inference_count": self._inference_count,
            "total_inference_time": self._total_inference_time,
            "average_inference_time": avg_time,
            "ollama_host": self.host
        }


async def convert_pdf_page_to_image(pdf_path: Path, page_num: int = 0) -> bytes:
    """
    Convert a PDF page to image bytes for VLM processing.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)

    Returns:
        PNG image bytes
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        page = doc[page_num]

        # Render at 150 DPI for good quality
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat)

        image_bytes = pix.tobytes("png")
        doc.close()

        return image_bytes

    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Run: pip install pymupdf")
        raise
    except Exception as e:
        logger.error(f"Failed to convert PDF page to image: {e}")
        raise


async def convert_pdf_to_images(pdf_path: Path) -> List[bytes]:
    """
    Convert all PDF pages to images.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of PNG image bytes, one per page
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))

        doc.close()
        return images

    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Run: pip install pymupdf")
        raise
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise
