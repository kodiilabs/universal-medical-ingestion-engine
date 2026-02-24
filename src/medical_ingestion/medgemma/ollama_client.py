# ============================================================================
# src/medical_ingestion/medgemma/ollama_client.py
# ============================================================================
"""
Ollama MedGemma Client

Uses Ollama for local inference. Ollama handles model management,
quantization, and provides a simple HTTP API.

Benefits:
- Easy setup (just install ollama and pull the model)
- Efficient quantization (Q4_K_S = ~2.5GB instead of 8GB)
- No Python dependencies for inference
- Works well on CPU and Apple Silicon

Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull model: ollama pull MedAIBase/MedGemma1.5:4b-it-q8_0
    3. Start server: ollama serve (or it runs automatically)
"""

import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .base import BaseMedGemmaClient, BackendType
from .cache import PromptCache


# Default Ollama model for MedGemma
DEFAULT_OLLAMA_MODEL = "MedAIBase/MedGemma1.5:4b-it-q8_0"


class OllamaMedGemmaClient(BaseMedGemmaClient):
    """
    Ollama-based MedGemma inference client.

    Config options:
        ollama_host: Ollama server URL (default: http://localhost:11434)
        ollama_model: Model name (default: MedAIBase/MedGemma1.5:4b-it-q8_0)
        max_tokens: Default max tokens (default: 1000)
        temperature: Default temperature (default: 0.1)
        timeout: Request timeout in seconds (default: 120)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Ollama configuration
        self.host = self.config.get('ollama_host', 'http://localhost:11434')
        self._model_name = self.config.get('ollama_model', DEFAULT_OLLAMA_MODEL)

        # Generation defaults
        self.default_max_tokens = self.config.get('max_tokens', 1000)
        self.default_temperature = self.config.get('temperature', 0.1)
        self.timeout = self.config.get('timeout', 120)

        # Response cache — avoids re-running identical prompts
        use_cache = self.config.get('use_cache', True)
        if use_cache:
            cache_dir = self.config.get('cache_dir', Path('./ollama_cache'))
            if isinstance(cache_dir, str):
                cache_dir = Path(cache_dir)
            self.cache = PromptCache(
                max_size=self.config.get('cache_max_size', 500),
                default_ttl=self.config.get('cache_ttl', 3600),
                cache_dir=cache_dir,
                auto_persist=True,
                persist_interval=5,
            )
        else:
            self.cache = None

        # HTTP session (created lazily, tied to event loop)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_loop: Optional[asyncio.AbstractEventLoop] = None

        self.logger.info(f"Initialized Ollama client: {self.host} / {self._model_name}")

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Check if we need a new session (none exists, closed, or different event loop)
        needs_new_session = (
            self._session is None
            or self._session.closed
            or self._session_loop is None
            or self._session_loop != current_loop
            or (self._session_loop is not None and self._session_loop.is_closed())
        )

        if needs_new_session:
            # Close old session if it exists and isn't already closed
            if self._session is not None and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass  # Ignore errors closing stale session

            timeout = aiohttp.ClientTimeout(
                total=None,       # No overall limit — let inference run
                sock_connect=30,  # 30s to establish connection
                sock_read=600,    # 10min max between chunks from Ollama
            )
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
        """
        Check if Ollama server is running and model is available.
        """
        try:
            session = await self._get_session()

            # Check server is running
            async with session.get(f"{self.host}/api/tags") as response:
                if response.status != 200:
                    return {
                        "healthy": False,
                        "backend": "ollama",
                        "model": self._model_name,
                        "details": f"Ollama server returned status {response.status}"
                    }

                data = await response.json()
                models = [m.get('name', '') for m in data.get('models', [])]

                # Check if our model is available
                model_available = any(self._model_name in m for m in models)

                if not model_available:
                    return {
                        "healthy": False,
                        "backend": "ollama",
                        "model": self._model_name,
                        "details": f"Model not found. Available: {models}. Run: ollama pull {self._model_name}"
                    }

                return {
                    "healthy": True,
                    "backend": "ollama",
                    "model": self._model_name,
                    "details": "Ollama server running and model available"
                }

        except aiohttp.ClientConnectorError:
            return {
                "healthy": False,
                "backend": "ollama",
                "model": self._model_name,
                "details": f"Cannot connect to Ollama at {self.host}. Is it running? Try: ollama serve"
            }
        except Exception as e:
            return {
                "healthy": False,
                "backend": "ollama",
                "model": self._model_name,
                "details": f"Health check failed: {str(e)}"
            }

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate response using Ollama.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cache
            json_mode: If True, constrain output to valid JSON using Ollama's format option

        Returns:
            Response dict with text, tokens, timing info
        """
        start_time = datetime.now()

        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get_response(prompt, max_tokens, temperature)
            if cached is not None:
                cached["cached"] = True
                cached["inference_time"] = 0.0
                self.logger.debug("Cache hit — returning cached response")
                return cached

        # Per-request timeout: cap each LLM call so large documents can't hang
        # the pipeline indefinitely. Default 5 minutes per call.
        request_timeout = self.config.get('request_timeout', 300)

        try:
            session = await self._get_session()

            # Ollama generate API
            payload = {
                "model": self._model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            }

            # Enable JSON mode - Ollama will constrain output to valid JSON
            if json_mode:
                payload["format"] = "json"
                self.logger.debug("JSON mode enabled for generation")

            async def _do_request():
                async with session.post(
                    f"{self.host}/api/generate",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama error ({response.status}): {error_text}")
                    return await response.json()

            data = await asyncio.wait_for(_do_request(), timeout=request_timeout)

            # Extract response
            generated_text = data.get('response', '')
            inference_time = (datetime.now() - start_time).total_seconds()

            # Ollama provides some metrics
            prompt_tokens = data.get('prompt_eval_count', 0)
            generated_tokens = data.get('eval_count', 0)

            # Update statistics
            self._inference_count += 1
            self._total_inference_time += inference_time

            # Calculate tokens/sec if available
            eval_duration = data.get('eval_duration', 0)  # nanoseconds
            if eval_duration > 0 and generated_tokens > 0:
                tokens_per_sec = generated_tokens / (eval_duration / 1e9)
            else:
                tokens_per_sec = generated_tokens / inference_time if inference_time > 0 else 0

            self.logger.info(
                f"Generated {generated_tokens} tokens in {inference_time:.2f}s "
                f"({tokens_per_sec:.1f} tokens/sec)"
            )

            result = {
                "text": generated_text.strip(),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "model": self._model_name,
                "backend": "ollama",
                "inference_time": inference_time,
                "tokens_per_second": tokens_per_sec,
                "cached": False
            }

            # Store in cache for future identical prompts
            if use_cache and self.cache:
                self.cache.set_response(prompt, max_tokens, temperature, result)

            return result

        except asyncio.TimeoutError:
            self.logger.error(
                f"Ollama request timed out after {request_timeout}s "
                f"(model={self._model_name}, max_tokens={max_tokens})"
            )
            raise TimeoutError(
                f"LLM request timed out after {request_timeout}s. "
                "The model may be overloaded or the document too large."
            )
        except aiohttp.ClientConnectorError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            self.logger.error(f"Ollama inference failed: {e}")
            raise

    async def pull_model(self) -> bool:
        """
        Pull the model if not already available.

        Returns:
            True if model is now available
        """
        try:
            session = await self._get_session()

            self.logger.info(f"Pulling model {self._model_name}...")

            payload = {
                "name": self._model_name,
                "stream": False
            }

            async with session.post(
                f"{self.host}/api/pull",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3600)  # 1 hour for large models
            ) as response:
                if response.status == 200:
                    self.logger.info(f"Model {self._model_name} pulled successfully")
                    return True
                else:
                    error = await response.text()
                    self.logger.error(f"Failed to pull model: {error}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to pull model: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = super().get_statistics()
        stats["ollama_host"] = self.host
        if self.cache:
            stats["cache"] = self.cache.get_statistics()
        return stats
