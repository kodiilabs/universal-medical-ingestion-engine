# ============================================================================
# src/medical_ingestion/extractors/paligemma_client.py
# ============================================================================
"""
PaliGemma Client for Local Visual Language Model Inference

Uses HuggingFace transformers to run PaliGemma locally for:
- Region classification (printed text, handwriting, table, etc.)
- Document type detection
- Visual question answering about document content

Requires: transformers, torch, PIL
Model: google/paligemma-3b-pt-224 or google/paligemma-3b-mix-224
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Lazy loading for heavy imports
_model = None
_processor = None
_model_loaded = False


class PaliGemmaClient:
    """
    Client for running PaliGemma VLM locally.

    Uses HuggingFace transformers with local model caching.
    """

    # Supported region types for classification
    REGION_TYPES = [
        "printed_text",
        "handwriting",
        "table",
        "form_field",
        "stamp",
        "signature",
        "image",
        "header",
        "footer",
        "noise"
    ]

    # Document type prompts
    DOCUMENT_TYPES = [
        "lab_report",
        "radiology_report",
        "prescription",
        "pathology_report",
        "medical_form",
        "other"
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_name = self.config.get(
            'paligemma_model',
            'google/paligemma-3b-pt-224'
        )
        self.device = self.config.get('device', 'auto')
        self.model = None
        self.processor = None

    async def _ensure_model_loaded(self):
        """Load model if not already loaded."""
        global _model, _processor, _model_loaded

        if _model_loaded:
            self.model = _model
            self.processor = _processor
            return

        # Run model loading in thread pool to not block
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self):
        """Synchronously load the model."""
        global _model, _processor, _model_loaded

        if _model_loaded:
            return

        try:
            import torch
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

            logger.info(f"Loading PaliGemma model: {self.model_name}")

            # Determine device
            if self.device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            else:
                device = self.device

            logger.info(f"Using device: {device}")

            # Load processor
            _processor = AutoProcessor.from_pretrained(self.model_name)

            # Load model with appropriate dtype
            if device == 'cuda':
                _model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                _model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                _model = _model.to(device)

            _model.eval()
            _model_loaded = True

            self.model = _model
            self.processor = _processor

            logger.info("PaliGemma model loaded successfully")

        except ImportError as e:
            logger.error(f"Missing dependencies for PaliGemma: {e}")
            logger.error("Install with: pip install transformers torch pillow")
            raise
        except Exception as e:
            logger.error(f"Failed to load PaliGemma model: {e}")
            raise

    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if len(image.shape) == 2:
            # Grayscale
            return Image.fromarray(image, mode='L').convert('RGB')
        elif image.shape[2] == 3:
            # BGR to RGB
            return Image.fromarray(image[:, :, ::-1], mode='RGB')
        else:
            return Image.fromarray(image, mode='RGB')

    async def classify_region(self, region_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify a document region.

        Args:
            region_image: Numpy array of the region (BGR or grayscale)

        Returns:
            Dict with 'type', 'confidence', 'is_handwritten'
        """
        await self._ensure_model_loaded()

        pil_image = self._numpy_to_pil(region_image)

        # Prompt for region classification
        prompt = "What type of content is in this image? Is it: printed text, handwriting, a table, a form field, a stamp, a signature, an image/chart, or noise/artifacts? Answer with just the type."

        result = await self._generate(pil_image, prompt)

        # Parse response
        response_lower = result.lower()

        detected_type = "printed_text"  # Default
        confidence = 0.7
        is_handwritten = False

        for rtype in self.REGION_TYPES:
            if rtype.replace('_', ' ') in response_lower or rtype in response_lower:
                detected_type = rtype
                confidence = 0.85
                break

        if 'handwrit' in response_lower or 'hand writ' in response_lower:
            is_handwritten = True
            detected_type = "handwriting"

        return {
            'type': detected_type,
            'confidence': confidence,
            'is_handwritten': is_handwritten,
            'raw_response': result
        }

    async def classify_document_type(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify the document type from the full page image.

        Args:
            image: Full page image as numpy array

        Returns:
            Dict with 'document_type', 'confidence'
        """
        await self._ensure_model_loaded()

        pil_image = self._numpy_to_pil(image)

        prompt = "What type of medical document is this? Is it a: lab report (blood test results, chemistry panel), radiology report (X-ray, CT, MRI findings), prescription (medication orders), pathology report (biopsy results), or a medical form? Answer with just the document type."

        result = await self._generate(pil_image, prompt)

        response_lower = result.lower()

        doc_type = "unknown"
        confidence = 0.6

        if 'lab' in response_lower or 'blood' in response_lower or 'chemistry' in response_lower:
            doc_type = "lab"
            confidence = 0.85
        elif 'radiology' in response_lower or 'x-ray' in response_lower or 'ct' in response_lower or 'mri' in response_lower:
            doc_type = "radiology"
            confidence = 0.85
        elif 'prescription' in response_lower or 'medication' in response_lower or 'rx' in response_lower:
            doc_type = "prescription"
            confidence = 0.85
        elif 'pathology' in response_lower or 'biopsy' in response_lower:
            doc_type = "pathology"
            confidence = 0.85

        return {
            'document_type': doc_type,
            'confidence': confidence,
            'raw_response': result
        }

    async def describe_region(self, region_image: np.ndarray) -> str:
        """
        Get a description of what's in a region.

        Useful for understanding complex or ambiguous content.
        """
        await self._ensure_model_loaded()

        pil_image = self._numpy_to_pil(region_image)
        prompt = "Describe what you see in this image section of a medical document."

        return await self._generate(pil_image, prompt)

    async def extract_text_hint(self, region_image: np.ndarray) -> str:
        """
        Ask VLM to read any visible text in the region.

        This can help verify OCR results or provide hints for
        difficult-to-read content.
        """
        await self._ensure_model_loaded()

        pil_image = self._numpy_to_pil(region_image)
        prompt = "Read and transcribe any text visible in this image."

        return await self._generate(pil_image, prompt)

    async def _generate(self, image: Image.Image, prompt: str) -> str:
        """
        Generate response from the VLM.

        Args:
            image: PIL Image
            prompt: Text prompt

        Returns:
            Generated text response
        """
        import torch

        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        # Decode
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response if present
        if prompt in response:
            response = response.replace(prompt, '').strip()

        return response

    async def is_available(self) -> bool:
        """Check if PaliGemma is available and can be loaded."""
        try:
            await self._ensure_model_loaded()
            return True
        except Exception as e:
            logger.warning(f"PaliGemma not available: {e}")
            return False


# Alternative: Ollama-based VLM client (for llava, bakllava, etc.)
class OllamaVLMClient:
    """
    Alternative VLM client using Ollama with vision models.

    Supports models like:
    - llava
    - bakllava
    - llava-llama3
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = self.config.get('ollama_vlm_model', 'llava')
        self.base_url = self.config.get('ollama_url', 'http://localhost:11434')

    async def classify_region(self, region_image: np.ndarray) -> Dict[str, Any]:
        """Classify region using Ollama vision model."""
        import aiohttp
        import base64

        # Convert to base64
        pil_image = Image.fromarray(
            region_image[:, :, ::-1] if len(region_image.shape) == 3 else region_image
        )
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        prompt = "What type of content is this? Answer with one of: printed_text, handwriting, table, form_field, stamp, signature, image, noise"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data.get('response', '')

                    # Parse response
                    response_lower = response.lower()
                    detected_type = "printed_text"

                    for rtype in PaliGemmaClient.REGION_TYPES:
                        if rtype in response_lower:
                            detected_type = rtype
                            break

                    return {
                        'type': detected_type,
                        'confidence': 0.75,
                        'is_handwritten': 'handwrit' in response_lower,
                        'raw_response': response
                    }
                else:
                    raise Exception(f"Ollama request failed: {resp.status}")


def create_vlm_client(config: Dict = None) -> PaliGemmaClient:
    """
    Factory function to create the appropriate VLM client.

    Tries PaliGemma first, falls back to Ollama if available.
    """
    config = config or {}

    # Check if user prefers Ollama
    if config.get('use_ollama_vlm', False):
        return OllamaVLMClient(config)

    # Default to PaliGemma
    return PaliGemmaClient(config)
