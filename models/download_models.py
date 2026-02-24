#!/usr/bin/env python3
# ============================================================================
# models/download_models.py
# ============================================================================
"""
Model Download Script

Downloads and caches all models used by the pipeline:
- MedGemma LLM (BACKEND=transformers) — from HuggingFace Hub
- Sentence Transformer embeddings (EMBEDDING_BACKEND=sentence_transformers) — from HuggingFace Hub
- VLM model (VLM_MODEL) — from Ollama

Usage:
    python models/download_models.py --list
    python models/download_models.py --model medgemma
    python models/download_models.py --model embeddings
    python models/download_models.py --model vlm
    python models/download_models.py --model all --skip-auth
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Model configurations
MODELS = {
    # === LLM models (BACKEND=transformers) ===
    "medgemma": {
        "repo_id": "google/medgemma-4b-it",
        "description": "MedGemma 4B Instruct - Medical reasoning LLM",
        "size_gb": 8.5,
        "requires_auth": True,
        "category": "llm",
        "backend": "transformers",
        "env_var": "TRANSFORMERS_MODEL=google/medgemma-4b-it",
    },

    # === Embedding models (EMBEDDING_BACKEND=sentence_transformers) ===
    "embeddings": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "all-MiniLM-L6-v2 (384 dims) - Fast, good quality embeddings",
        "size_gb": 0.09,
        "requires_auth": False,
        "category": "embedding",
        "backend": "sentence_transformers",
        "env_var": "EMBEDDING_MODEL=all-MiniLM-L6-v2",
    },

    # === VLM models (Ollama — for image/vision extraction) ===
    "vlm": {
        "ollama_model": "qwen3-vl:latest",
        "description": "Qwen3-VL - Vision Language Model for image extraction",
        "size_gb": 5.3,
        "requires_auth": False,
        "category": "vlm",
        "backend": "ollama",
        "env_var": "VLM_MODEL=qwen3-vl:latest",
    },
}


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def check_dependencies(model_key: str = None) -> bool:
    """Check if required dependencies are installed."""
    config = MODELS.get(model_key, {}) if model_key else {}

    # Ollama models don't need huggingface_hub
    if config.get("backend") == "ollama":
        return True

    if not HF_AVAILABLE:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface-hub")
        return False

    if not TORCH_AVAILABLE:
        print("WARNING: torch not installed. Models will download but cannot be loaded.")

    return True


def check_disk_space(cache_dir: Path, required_gb: float) -> bool:
    """Check if there's enough disk space."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_space = shutil.disk_usage(cache_dir).free / (1024 ** 3)

    if free_space < required_gb * 1.2:  # 20% buffer
        return False

    return True


def check_hf_token() -> Optional[str]:
    """Check for HuggingFace token."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not token:
        token_path = Path.home() / ".huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()

    return token


def _is_ollama_model(model_key: str) -> bool:
    """Check if a model is an Ollama model (vs HuggingFace)."""
    return "ollama_model" in MODELS.get(model_key, {})


def download_ollama_model(
    model_key: str,
    logger: logging.Logger,
    force: bool = False
) -> bool:
    """
    Download a model via Ollama pull.

    Args:
        model_key: Key from MODELS dict
        logger: Logger instance
        force: Force re-download

    Returns:
        True if successful
    """
    config = MODELS[model_key]
    ollama_model = config["ollama_model"]

    logger.info("=" * 60)
    logger.info(f"Pulling Ollama model: {ollama_model}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Size: ~{config['size_gb']} GB")
    logger.info("=" * 60)

    # Check if ollama is available
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            logger.error("Ollama is not running. Start it with: ollama serve")
            return False

        # Check if model already exists
        if not force and ollama_model in result.stdout:
            logger.info(f"Model {ollama_model} already pulled in Ollama")
            logger.info("Use --force to re-pull")
            return True

    except FileNotFoundError:
        logger.error("Ollama not found. Install from: https://ollama.com")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Ollama not responding. Start it with: ollama serve")
        return False

    # Pull model
    try:
        logger.info(f"Pulling {ollama_model}...")
        result = subprocess.run(
            ["ollama", "pull", ollama_model],
            timeout=3600  # 1 hour timeout for large models
        )

        if result.returncode == 0:
            logger.info(f"Pull complete: {ollama_model}")
            return True
        else:
            logger.error(f"Pull failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Pull timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"Pull failed: {e}")
        return False


def download_model(
    model_key: str,
    logger: logging.Logger,
    force: bool = False
) -> bool:
    """
    Download a model (HuggingFace or Ollama).

    Models are cached in HuggingFace's default cache (~/.cache/huggingface/hub/)
    so that from_pretrained() finds them automatically.

    Args:
        model_key: Key from MODELS dict
        logger: Logger instance
        force: Force re-download even if exists

    Returns:
        True if successful
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        logger.info(f"Available models: {', '.join(MODELS.keys())}")
        return False

    # Dispatch to Ollama or HuggingFace download
    if _is_ollama_model(model_key):
        return download_ollama_model(model_key, logger, force)

    config = MODELS[model_key]
    repo_id = config["repo_id"]
    description = config["description"]
    size_gb = config["size_gb"]
    requires_auth = config["requires_auth"]

    logger.info("=" * 60)
    logger.info(f"Downloading: {model_key}")
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Description: {description}")
    logger.info(f"Size: ~{size_gb} GB")
    logger.info("=" * 60)

    # Check disk space in HF cache directory
    hf_cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    if not check_disk_space(hf_cache_base, size_gb):
        logger.error(f"Insufficient disk space. Need ~{size_gb * 1.2:.1f} GB free.")
        return False

    # Check authentication for gated models
    token = None
    if requires_auth:
        token = check_hf_token()
        if not token:
            logger.error(f"Model {model_key} requires authentication.")
            logger.error("Set HF_TOKEN environment variable or run: huggingface-cli login")
            return False

    # Check if already downloaded in HuggingFace cache
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    if hf_cache_dir.exists() and not force:
        snapshots = hf_cache_dir / "snapshots"
        if snapshots.exists() and any(snapshots.iterdir()):
            logger.info(f"Model already cached: {hf_cache_dir}")
            logger.info("Use --force to re-download")
            return True

    # Download model to HuggingFace's default cache (~/.cache/huggingface/hub/)
    # This way from_pretrained() finds it automatically without duplication
    try:
        logger.info("Starting download to HuggingFace cache...")

        snapshot_download(
            repo_id=repo_id,
            token=token,
        )

        logger.info(f"Download complete: {repo_id} (cached in ~/.cache/huggingface/hub/)")
        return True

    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            logger.error("Authentication failed. Check your HF_TOKEN.")
        else:
            logger.error(f"Download failed: {e}")
        return False

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_all_models(
    logger: logging.Logger,
    force: bool = False,
    skip_auth: bool = False
) -> dict:
    """
    Download all models.

    Args:
        logger: Logger instance
        force: Force re-download
        skip_auth: Skip models requiring authentication

    Returns:
        Dict of model_key -> success status
    """
    results = {}

    for model_key, config in MODELS.items():
        if skip_auth and config["requires_auth"]:
            logger.info(f"Skipping {model_key} (requires authentication)")
            results[model_key] = None
            continue

        results[model_key] = download_model(model_key, logger, force)

    return results


def verify_model(model_key: str, logger: logging.Logger) -> bool:
    """
    Verify a downloaded model can be loaded.

    Args:
        model_key: Key from MODELS dict
        logger: Logger instance

    Returns:
        True if model loads successfully
    """
    model_config = MODELS.get(model_key, {})
    category = model_config.get("category", "llm")

    logger.info(f"Verifying {model_key}...")

    if category == "vlm":
        # Verify Ollama model is available
        ollama_model = model_config.get("ollama_model", "")
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )
            if ollama_model in result.stdout:
                logger.info(f"  Ollama model {ollama_model} is available")
                logger.info(f"  Verification passed")
                return True
            else:
                logger.error(f"  Ollama model {ollama_model} not found")
                logger.error(f"  Run: ollama pull {ollama_model}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.error("  Ollama not available")
            return False

    elif category == "embedding":
        # Verify sentence_transformers model by loading with hub name
        repo_id = model_config.get("repo_id", "")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(repo_id)
            test_embedding = model.encode(["test"])
            dim = test_embedding.shape[1]
            logger.info(f"  Embedding model loaded: {dim} dimensions")
            logger.info(f"  Verification passed")
            return True
        except ImportError:
            logger.warning("  sentence-transformers not installed - skipping verification")
            return True
        except Exception as e:
            logger.error(f"  Verification failed: {e}")
            return False
    else:
        # Verify transformers model from HF cache
        repo_id = model_config.get("repo_id", "")
        hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
        if not hf_cache_dir.exists():
            logger.error(f"Model not cached. Run: python models/download_models.py --model {model_key}")
            return False

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - skipping verification")
            return True

        try:
            from transformers import AutoTokenizer, AutoConfig

            config = AutoConfig.from_pretrained(repo_id)
            logger.info(f"  Config loaded: {config.model_type}")

            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            logger.info(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")

            logger.info(f"  Verification passed")
            return True

        except Exception as e:
            logger.error(f"  Verification failed: {e}")
            return False


def list_models() -> None:
    """List available and downloaded models."""
    categories = {
        "llm": "LLM Models (BACKEND=transformers)",
        "embedding": "Embedding Models (EMBEDDING_BACKEND=sentence_transformers)",
        "vlm": "Vision Models (VLM_MODEL, via Ollama)",
    }

    # Check Ollama models
    ollama_models = ""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            ollama_models = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("\nAvailable Models:")
    print("=" * 70)

    for cat_key, cat_label in categories.items():
        cat_models = {k: v for k, v in MODELS.items() if v.get("category") == cat_key}
        if not cat_models:
            continue

        print(f"\n  {cat_label}:")
        print(f"  {'-' * 60}")

        for key, config in cat_models.items():
            # Determine status
            if _is_ollama_model(key):
                ollama_name = config["ollama_model"]
                status = "PULLED" if ollama_name in ollama_models else "NOT PULLED"
            else:
                repo_id = config.get("repo_id", "")
                hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
                status = "CACHED" if hf_cache.exists() else "NOT DOWNLOADED"

            auth = " (auth required)" if config["requires_auth"] else ""
            source = config.get("ollama_model") or config.get("repo_id", "")

            print(f"\n    {key}:")
            print(f"      Source:      {source}")
            print(f"      Description: {config['description']}")
            print(f"      Size:        ~{config['size_gb']} GB")
            print(f"      .env:        {config.get('env_var', '')}")
            print(f"      Status:      {status}{auth}")

    print("\n" + "=" * 70)
    print("\nQuick start:")
    print("  python download_models.py --model medgemma      # LLM (needs HF login)")
    print("  python download_models.py --model embeddings    # Embeddings (no auth)")
    print("  python download_models.py --model vlm           # VLM (needs Ollama running)")
    print("  python download_models.py --model all --skip-auth  # All non-gated models")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download models for medical document processing"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model to download. Options: {', '.join(MODELS.keys())}, all"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded model can be loaded"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )

    parser.add_argument(
        "--skip-auth",
        action="store_true",
        help="Skip models requiring authentication"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    # List models
    if args.list:
        list_models()
        return 0

    # Check dependencies
    if not check_dependencies(args.model):
        return 1

    # Verify mode
    if args.verify:
        if not args.model:
            logger.error("Specify --model to verify")
            return 1

        if args.model == "all":
            success = all(
                verify_model(key, logger)
                for key in MODELS.keys()
            )
        else:
            success = verify_model(args.model, logger)

        return 0 if success else 1

    # Download mode
    if not args.model:
        list_models()
        return 0

    if args.model == "all":
        results = download_all_models(logger, args.force, args.skip_auth)
        success_count = sum(1 for v in results.values() if v is True)
        total_count = len([v for v in results.values() if v is not None])

        logger.info(f"\nDownload complete: {success_count}/{total_count} models")

        return 0 if success_count == total_count else 1

    else:
        success = download_model(args.model, logger, args.force)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
