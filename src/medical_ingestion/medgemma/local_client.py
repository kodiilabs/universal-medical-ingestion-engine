# ============================================================================
# src/medical_ingestion/medgemma/local_client.py
# ============================================================================
"""
Local HuggingFace MedGemma Client

Runs MedGemma using HuggingFace Transformers locally.
Requires downloading model weights (~8GB for 4B model).

Benefits:
- Full control over inference
- No external dependencies at runtime
- Works offline after initial download

Requirements:
- Significant RAM (16GB+ recommended)
- GPU recommended for reasonable speed
- Model downloaded via download_models.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BaseMedGemmaClient, BackendType
from .cache import PromptCache


class LocalMedGemmaClient(BaseMedGemmaClient):
    """
    Local HuggingFace Transformers MedGemma client.

    Config options:
        model_path: Path to model weights (default: ./models/cache/medgemma)
        use_cache: Enable response caching (default: True)
        cache_dir: Cache directory (default: ./medgemma)
        max_tokens: Default max tokens (default: 1000)
        temperature: Default temperature (default: 0.1)
        force_cpu: Force CPU even if GPU available (default: False)
        use_gpu: Use GPU if available (default: True)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Model resolution order:
        # 1. If model_path exists as a local directory, use it (pre-downloaded)
        # 2. Otherwise use transformers_model (HF hub name, uses ~/.cache/huggingface/)
        local_path = Path(self.config.get('model_path', './models/cache/medgemma'))
        if local_path.exists() and local_path.is_dir():
            self.model_id = str(local_path)
        else:
            self.model_id = self.config.get('transformers_model') or str(local_path)
        self.model_path = Path(self.model_id)

        # Cache setup
        cache_dir = self.config.get('cache_dir', Path('./medgemma'))
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        use_cache = self.config.get('use_cache', True)
        if use_cache:
            self.cache = PromptCache(
                max_size=self.config.get('cache_max_size', 1000),
                default_ttl=self.config.get('cache_ttl', 3600),
                cache_dir=cache_dir,
                auto_persist=True,
                persist_interval=self.config.get('cache_persist_interval', 10),
            )
        else:
            self.cache = None

        # Device selection
        self.device = self._select_device()

        # Model and tokenizer (loaded lazily)
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        # Generation defaults
        self.default_max_tokens = self.config.get('max_tokens', 1000)
        self.default_temperature = self.config.get('temperature', 0.1)

    @property
    def backend_type(self) -> BackendType:
        return BackendType.LOCAL

    @property
    def model_name(self) -> str:
        return f"medgemma-transformers ({self.model_id})"

    def _select_device(self) -> str:
        """Select compute device."""
        force_cpu = self.config.get('force_cpu', False)
        use_gpu = self.config.get('use_gpu', True)

        if force_cpu:
            self.logger.info("Forcing CPU usage (config setting)")
            return "cpu"

        if torch.cuda.is_available() and use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU detected: {gpu_name} - using CUDA")
            return "cuda"
        elif torch.backends.mps.is_available() and use_gpu:
            self.logger.info("Apple Silicon detected - using MPS")
            return "mps"
        else:
            self.logger.info("No GPU available - using CPU")
            return "cpu"

    def _is_local_path(self) -> bool:
        """Check if model_id points to a local directory (vs HuggingFace Hub name)."""
        return self.model_path.exists() and self.model_path.is_dir()

    def load_model(self):
        """Load model into memory.

        Supports two modes:
        - Local path: loads from a directory on disk (local_files_only=True)
        - HuggingFace Hub: downloads/caches from Hub (e.g. 'google/medgemma-4b-it')
        """
        if self._model_loaded:
            return

        is_local = self._is_local_path()
        source = str(self.model_path) if is_local else self.model_id

        self.logger.info(f"Loading model from {'local path' if is_local else 'HuggingFace Hub'}: {source}")

        if is_local and not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Run: python models/download_models.py --model medgemma"
            )

        try:
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                source,
                local_files_only=is_local,
                trust_remote_code=True
            )

            # Load model (device-specific)
            self.logger.info(f"Loading model to {self.device}...")

            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    local_files_only=is_local,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif self.device == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    local_files_only=is_local,
                    dtype=torch.float16,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    local_files_only=is_local,
                    dtype=torch.float32,
                    trust_remote_code=True
                )

            self.model.eval()
            self._model_loaded = True
            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check if model is available."""
        is_local = self._is_local_path()

        if is_local and not self.model_path.exists():
            return {
                "healthy": False,
                "backend": "transformers",
                "model": self.model_id,
                "details": f"Local model not found at {self.model_path}. Run: python models/download_models.py --model medgemma"
            }

        return {
            "healthy": True,
            "backend": "transformers",
            "model": self.model_id,
            "details": f"Model {'local' if is_local else 'hub'}: {self.model_id}, device: {self.device}, loaded: {self._model_loaded}"
        }

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """Generate response using local model.

        Note: json_mode is accepted for API compatibility but local HuggingFace
        backend doesn't support constrained JSON output. Prompting should be
        used to encourage JSON output instead.
        """
        start_time = datetime.now()

        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        if json_mode:
            self.logger.debug(
                "json_mode requested but local backend doesn't support constrained JSON. "
                "Relying on prompt instructions for JSON output."
            )

        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get_response(prompt, max_tokens, temperature)
            if cached:
                self._cache_hits += 1
                self.logger.info(f"Cache hit (total: {self._cache_hits})")
                cached['cached'] = True
                return cached

        # Load model if needed
        self.load_model()

        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)

            prompt_tokens = inputs['input_ids'].shape[1]

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_ids = outputs[0][prompt_tokens:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            inference_time = (datetime.now() - start_time).total_seconds()
            generated_tokens = len(generated_ids)

            # Update stats
            self._inference_count += 1
            self._total_inference_time += inference_time

            result = {
                "text": generated_text.strip(),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "model": "medgemma-local",
                "backend": "local",
                "device": self.device,
                "inference_time": inference_time,
                "tokens_per_second": generated_tokens / inference_time if inference_time > 0 else 0,
                "cached": False
            }

            # Cache result
            if use_cache and self.cache:
                self.cache.set_response(prompt, max_tokens, temperature, result)

            self.logger.info(
                f"Generated {generated_tokens} tokens in {inference_time:.2f}s "
                f"({generated_tokens/inference_time:.1f} tokens/sec)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics including cache info."""
        stats = super().get_statistics()
        stats["device"] = self.device
        stats["model_loaded"] = self._model_loaded

        if self.cache:
            stats["cache"] = self.cache.get_statistics()

        return stats

    def clear_cache(self):
        """Clear response cache."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared")

    def cleanup_cache(self) -> int:
        """Remove expired cache entries."""
        if self.cache:
            return self.cache.cleanup_expired()
        return 0
