# ============================================================================
# src/medical_ingestion/medgemma/client.py
# ============================================================================
"""
MedGemma Client Factory

Provides a unified interface for creating MedGemma inference clients.
Supports multiple backends:
- ollama: Ollama server (recommended for easy setup)
- transformers: HuggingFace Transformers (no timeouts, full control)
- local: Alias for transformers
- api: External API (not yet implemented)

Usage:
    from src.medical_ingestion.medgemma.client import create_client

    # Ollama backend (recommended for easy setup)
    client = create_client({'backend': 'ollama'})

    # HuggingFace Transformers backend (recommended for production)
    client = create_client({'backend': 'transformers'})

    # Local HuggingFace backend (alias for transformers)
    client = create_client({'backend': 'local', 'model_path': './models/cache/medgemma'})

    # Generate response
    result = await client.generate("What are normal glucose levels?")
"""

from typing import Dict, Any, Optional

from .base import BaseMedGemmaClient, BackendType
from .ollama_client import OllamaMedGemmaClient, DEFAULT_OLLAMA_MODEL
from .local_client import LocalMedGemmaClient
from ..core.config import get_config


# Default backend
DEFAULT_BACKEND = "ollama"

# Singleton cache — keyed by (backend, host/model) so the same Ollama
# HTTP session and prompt cache are reused across all documents.
_client_cache: Dict[tuple, BaseMedGemmaClient] = {}

import logging
_logger = logging.getLogger(__name__)


def create_client(config: Optional[Dict[str, Any]] = None) -> BaseMedGemmaClient:
    """
    Factory function to create a MedGemma client.

    Returns a cached singleton when backend + connection params match,
    so Ollama HTTP sessions and prompt caches are reused across documents.

    Configuration is loaded from .env file and merged with any passed config.
    Passed config values take precedence over .env values.

    Args:
        config: Configuration dict with at minimum:
            - backend: "ollama" | "transformers" | "local" | "api" (default: "ollama")

            Ollama-specific:
            - ollama_host: Server URL (default: http://localhost:11434)
            - ollama_model: Model name (default: MedAIBase/MedGemma1.5:4b-it-q8_0)

            Transformers/Local-specific:
            - transformers_model: HuggingFace model name or local path (default: google/medgemma-4b-it)
            - model_path: Alias for transformers_model (legacy)
            - use_gpu: Use GPU if available (default: True)
            - force_cpu: Force CPU usage (default: False)

            Common:
            - max_tokens: Default max tokens (default: 1000)
            - temperature: Default temperature (default: 0.1)
            - use_cache: Enable caching (default: True)

    Returns:
        Configured MedGemma client instance

    Raises:
        ValueError: If backend type is not supported
    """
    # Merge env config with passed config (passed config takes precedence)
    env_config = get_config()
    config = {**env_config, **(config or {})}
    backend = config.get('backend', DEFAULT_BACKEND).lower()

    # Build cache key from backend + connection-defining params
    if backend == "ollama":
        cache_key = (backend, config.get('ollama_host'), config.get('ollama_model'))
    elif backend in ("local", "transformers"):
        cache_key = (backend, config.get('transformers_model'))
    else:
        cache_key = None

    # Return cached client if available
    if cache_key and cache_key in _client_cache:
        _logger.debug(f"Reusing cached {backend} client: {cache_key}")
        return _client_cache[cache_key]

    # Create new client
    if backend == "ollama":
        client = OllamaMedGemmaClient(config)

    elif backend in ("local", "transformers"):
        client = LocalMedGemmaClient(config)

    elif backend == "api":
        raise NotImplementedError(
            "API backend not yet implemented. "
            "Use 'ollama', 'transformers', or 'local' backend instead."
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported backends: ollama, transformers, local, api"
        )

    # Cache for future reuse
    if cache_key:
        _client_cache[cache_key] = client
        _logger.info(f"Created and cached {backend} client: {cache_key}")

    return client


# Backward compatibility alias
MedGemmaLocalClient = LocalMedGemmaClient


# Convenience function to get default config
def get_default_config(backend: str = DEFAULT_BACKEND) -> Dict[str, Any]:
    """
    Get default configuration for a backend.

    Args:
        backend: Backend type ("ollama", "local", "api")

    Returns:
        Default configuration dict
    """
    base_config = {
        "backend": backend,
        "max_tokens": 1000,
        "temperature": 0.1,
        "use_cache": True,
    }

    if backend == "ollama":
        base_config.update({
            "ollama_host": "http://localhost:11434",
            "ollama_model": DEFAULT_OLLAMA_MODEL,
            "timeout": 120,
        })

    elif backend in ("local", "transformers"):
        base_config.update({
            "transformers_model": "google/medgemma-4b-it",
            "model_path": "./models/cache/medgemma",
            "use_gpu": True,
            "force_cpu": False,
            "cache_dir": "./medgemma",
        })

    return base_config


__all__ = [
    "create_client",
    "get_default_config",
    "BaseMedGemmaClient",
    "BackendType",
    "OllamaMedGemmaClient",
    "LocalMedGemmaClient",
    "MedGemmaLocalClient",  # Backward compatibility
    "DEFAULT_BACKEND",
    "DEFAULT_OLLAMA_MODEL",
]
