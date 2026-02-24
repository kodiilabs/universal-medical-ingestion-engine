# ============================================================================
# src/medical_ingestion/medgemma/base.py
# ============================================================================
"""
Base MedGemma Client Interface

Defines the abstract interface that all MedGemma backends must implement.
Supported backends:
- local: HuggingFace Transformers (requires GPU/CPU, large RAM)
- ollama: Ollama server (lightweight, easy setup)
- api: External API (cloud-based)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
import logging
import json
import re

from json_repair import repair_json


class BackendType(Enum):
    """Supported inference backends."""
    LOCAL = "local"      # HuggingFace Transformers
    OLLAMA = "ollama"    # Ollama server
    API = "api"          # External API


class BaseMedGemmaClient(ABC):
    """
    Abstract base class for MedGemma inference clients.

    All backends must implement:
    - generate(): Async text generation
    - get_statistics(): Return inference stats
    - health_check(): Verify backend is available
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Common statistics
        self._inference_count = 0
        self._cache_hits = 0
        self._total_inference_time = 0.0

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate response from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            use_cache: Whether to use response cache
            json_mode: If True, constrain output to valid JSON (Ollama: format="json")

        Returns:
            {
                "text": str,           # Generated text
                "prompt_tokens": int,  # Input token count
                "generated_tokens": int,  # Output token count
                "model": str,          # Model identifier
                "backend": str,        # Backend type
                "inference_time": float,  # Seconds
                "cached": bool         # Whether from cache
            }
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the backend is available and ready.

        Returns:
            {
                "healthy": bool,
                "backend": str,
                "model": str,
                "details": str
            }
        """
        pass

    def extract_json(self, response_text: str) -> Optional[Dict]:
        """
        Extract JSON object from generated text.

        LLMs often return JSON embedded in text like:
        "Based on the data, here is the analysis: {"key": "value"}"

        This extracts the JSON portion. Uses json_repair as fallback for
        malformed JSON (single quotes, trailing commas, etc).
        """
        if not response_text or not response_text.strip():
            self.logger.warning("Empty response text, no JSON to extract")
            return None

        # Try 1: Direct parse of entire response
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Try 2: Use json_repair on entire response
        try:
            repaired = repair_json(response_text, return_objects=True)
            if isinstance(repaired, dict):
                self.logger.debug("json_repair fixed entire response")
                return repaired
        except Exception:
            pass

        # Try 3: Extract JSON block by brace matching
        try:
            start_idx = response_text.find('{')
            if start_idx == -1:
                self.logger.warning("No JSON found in response")
                return None

            # Count braces to find matching close
            depth = 0
            end_idx = start_idx
            for i, char in enumerate(response_text[start_idx:], start=start_idx):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break

            json_str = response_text[start_idx:end_idx + 1]

            # Try direct parse of extracted JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            # Try json_repair on extracted JSON
            try:
                repaired = repair_json(json_str, return_objects=True)
                if isinstance(repaired, dict):
                    self.logger.debug("json_repair fixed extracted JSON block")
                    return repaired
            except Exception as e:
                self.logger.debug(f"json_repair failed on extracted block: {e}")

        except Exception as e:
            self.logger.warning(f"JSON extraction failed: {e}")

        self.logger.warning(f"Could not parse JSON from response: {response_text[:200]}...")
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference and cache statistics."""
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0.0
        )

        cache_hit_rate = (
            self._cache_hits / (self._inference_count + self._cache_hits)
            if (self._inference_count + self._cache_hits) > 0
            else 0.0
        )

        return {
            "backend": self.backend_type.value,
            "model": self.model_name,
            "inference_count": self._inference_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "total_inference_time": self._total_inference_time,
            "average_inference_time": avg_time,
        }
