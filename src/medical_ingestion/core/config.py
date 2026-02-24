# ============================================================================
# src/medical_ingestion/core/config.py
# ============================================================================
"""
Centralized Configuration Management

Loads configuration from environment variables (.env file) with sensible defaults.
All config values flow from this single source of truth.

Usage:
    from medical_ingestion.core.config import get_config, Config

    # Get full config dict
    config = get_config()

    # Or use Config class for attribute access
    cfg = Config()
    print(cfg.ollama_host)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache


def _load_dotenv():
    """Load .env file if it exists."""
    try:
        from dotenv import load_dotenv

        # Look for .env in project root
        env_path = Path(__file__).parent.parent.parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            return True

        # Also check current working directory
        cwd_env = Path.cwd() / '.env'
        if cwd_env.exists():
            load_dotenv(cwd_env)
            return True

    except ImportError:
        pass

    return False


def _get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def _get_int(key: str, default: int = 0) -> int:
    """Get integer from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(key: str, default: float = 0.0) -> float:
    """Get float from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class Config:
    """
    Configuration container with attribute access.

    All values are loaded from environment variables with defaults.
    """

    # General
    data_dir: str = field(default_factory=lambda: os.getenv('DATA_DIR', 'data'))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))

    # LLM Backend
    backend: str = field(default_factory=lambda: os.getenv('BACKEND', 'ollama'))

    # Ollama
    ollama_host: str = field(default_factory=lambda: os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
    ollama_model: str = field(default_factory=lambda: os.getenv('OLLAMA_MODEL', 'MedAIBase/MedGemma1.5:4b-it-q8_0'))

    # Transformers / Local model
    transformers_model: str = field(default_factory=lambda: os.getenv('TRANSFORMERS_MODEL', 'google/medgemma-4b-it'))
    model_path: str = field(default_factory=lambda: os.getenv('MODEL_PATH', './models/cache/medgemma'))

    # Generation
    max_tokens: int = field(default_factory=lambda: _get_int('MAX_TOKENS', 1000))
    temperature: float = field(default_factory=lambda: _get_float('TEMPERATURE', 0.1))
    timeout: int = field(default_factory=lambda: _get_int('TIMEOUT', 120))

    # Extraction
    # Consensus extraction runs VLM + OCR in parallel and merges results
    # Default to False - consensus can cause issues with local Ollama (timeouts, text bloat)
    use_consensus_extraction: bool = field(default_factory=lambda: _get_bool('USE_CONSENSUS_EXTRACTION', False))
    classification_only: bool = field(default_factory=lambda: _get_bool('CLASSIFICATION_ONLY', False))
    confidence_threshold: float = field(default_factory=lambda: _get_float('CONFIDENCE_THRESHOLD', 0.7))
    human_review_threshold: float = field(default_factory=lambda: _get_float('HUMAN_REVIEW_THRESHOLD', 0.85))
    vector_store_min_confidence: float = field(default_factory=lambda: _get_float('VECTOR_STORE_MIN_CONFIDENCE', 0.85))
    use_ocr: bool = field(default_factory=lambda: _get_bool('USE_OCR', True))
    use_vision: bool = field(default_factory=lambda: _get_bool('USE_VISION', True))
    # Max text length per chunk for extraction (4000 needed for Quest-style multi-page reports)
    max_text_length: int = field(default_factory=lambda: _get_int('MAX_TEXT_LENGTH', 4000))

    # Vector Store (mxbai-embed-large is top performer on MTEB benchmarks)
    embedding_backend: str = field(default_factory=lambda: os.getenv('EMBEDDING_BACKEND', 'ollama'))
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'mxbai-embed-large'))
    embedding_dim: int = field(default_factory=lambda: _get_int('EMBEDDING_DIM', 1024))

    # VLM (Vision Language Model) for image-based extraction
    use_vlm: bool = field(default_factory=lambda: _get_bool('USE_VLM', True))
    vlm_model: str = field(default_factory=lambda: os.getenv('VLM_MODEL', 'moondream'))
    vlm_timeout: int = field(default_factory=lambda: _get_int('VLM_TIMEOUT', 180))
    vlm_fallback_threshold: float = field(default_factory=lambda: _get_float('VLM_FALLBACK_THRESHOLD', 0.5))

    # VLM Unified: single VLM replaces PaddleOCR + VLM + LLM stack
    vlm_unified: bool = field(default_factory=lambda: _get_bool('VLM_UNIFIED', False))

    # Caching
    use_cache: bool = field(default_factory=lambda: _get_bool('USE_CACHE', True))
    cache_dir: str = field(default_factory=lambda: os.getenv('CACHE_DIR', './medgemma'))
    cache_max_size: int = field(default_factory=lambda: _get_int('CACHE_MAX_SIZE', 1000))
    cache_ttl: int = field(default_factory=lambda: _get_int('CACHE_TTL', 3600))
    cache_persist_interval: int = field(default_factory=lambda: _get_int('CACHE_PERSIST_INTERVAL', 10))

    # Concurrency — max parallel chunk extractions (match OLLAMA_NUM_PARALLEL)
    # Default 1 (sequential). Set to 2+ only if Ollama runs with OLLAMA_NUM_PARALLEL=2+
    max_concurrent_chunks: int = field(default_factory=lambda: _get_int('MAX_CONCURRENT_CHUNKS', 1))

    # Hardware
    use_gpu: bool = field(default_factory=lambda: _get_bool('USE_GPU', True))
    force_cpu: bool = field(default_factory=lambda: _get_bool('FORCE_CPU', False))

    # Cloud (Azure OpenAI) - when enabled, bypasses local VLM/OCR with GPT-4o vision
    use_cloud: bool = field(default_factory=lambda: _get_bool('USE_CLOUD', False))
    azure_deployment: str = field(default_factory=lambda: os.getenv('AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT', 'gpt-4o'))
    azure_endpoint: str = field(default_factory=lambda: os.getenv('AZURE_OPENAI_ENDPOINT', ''))
    azure_api_key: str = field(default_factory=lambda: os.getenv('AZURE_OPENAI_API_KEY', ''))
    azure_api_version: str = field(default_factory=lambda: os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'))

    def __post_init__(self):
        """Ensure .env is loaded before accessing values."""
        _load_dotenv()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for passing to components."""
        return {
            # General
            'data_dir': self.data_dir,
            'log_level': self.log_level,

            # LLM Backend
            'backend': self.backend,
            'ollama_host': self.ollama_host,
            'ollama_model': self.ollama_model,
            'transformers_model': self.transformers_model,
            'model_path': self.model_path,

            # Generation
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout,

            # Extraction
            'use_consensus_extraction': self.use_consensus_extraction,
            'classification_only': self.classification_only,
            'confidence_threshold': self.confidence_threshold,
            'human_review_threshold': self.human_review_threshold,
            'vector_store_min_confidence': self.vector_store_min_confidence,
            'use_ocr': self.use_ocr,
            'use_vision': self.use_vision,
            'max_text_length': self.max_text_length,

            # Vector Store
            'embedding_backend': self.embedding_backend,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,

            # VLM
            'use_vlm': self.use_vlm,
            'vlm_model': self.vlm_model,
            'vlm_timeout': self.vlm_timeout,
            'vlm_fallback_threshold': self.vlm_fallback_threshold,
            'vlm_unified': self.vlm_unified,

            # Caching
            'use_cache': self.use_cache,
            'cache_dir': self.cache_dir,
            'cache_max_size': self.cache_max_size,
            'cache_ttl': self.cache_ttl,
            'cache_persist_interval': self.cache_persist_interval,

            # Concurrency
            'max_concurrent_chunks': self.max_concurrent_chunks,

            # Hardware
            'use_gpu': self.use_gpu,
            'force_cpu': self.force_cpu,

            # Cloud (Azure OpenAI)
            'use_cloud': self.use_cloud,
            'azure_deployment': self.azure_deployment,
            'azure_endpoint': self.azure_endpoint,
            'azure_api_key': self.azure_api_key,
            'azure_api_version': self.azure_api_version,
        }


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    """
    Get configuration dictionary.

    Cached for performance - call once and pass to components.

    Returns:
        Configuration dictionary with all settings
    """
    _load_dotenv()
    return Config().to_dict()


def get_config_instance() -> Config:
    """
    Get Config instance for attribute access.

    Returns:
        Config instance with all settings
    """
    _load_dotenv()
    return Config()


# Convenience: reload config (clears cache)
def reload_config() -> Dict[str, Any]:
    """Reload configuration from environment."""
    get_config.cache_clear()
    return get_config()
