# ============================================================================
# src/medgemma/model_manager.py
# ============================================================================
"""
MedGemma Model Manager

Handles:
- Model download and verification
- Model metadata management
- Version tracking
- Storage management
- Model health checks
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
from enum import Enum


class ModelStatus(Enum):
    """Model status"""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    READY = "ready"
    CORRUPTED = "corrupted"
    OUTDATED = "outdated"


@dataclass
class ModelMetadata:
    """Model metadata"""
    name: str
    version: str
    model_type: str
    size_bytes: int
    download_date: str
    checksum: Optional[str] = None
    source_url: Optional[str] = None
    description: Optional[str] = None
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(**data)


class ModelManager:
    """
    Manages MedGemma model lifecycle.

    Features:
    - Model download verification
    - Checksum validation
    - Version tracking
    - Storage management
    - Health checks
    """

    METADATA_FILE = "model_metadata.json"
    SUPPORTED_MODELS = {
        "medgemma-7b": {
            "url": "google/medgemma-7b",
            "type": "causal-lm",
            "description": "MedGemma 7B parameter medical language model",
        },
        "medgemma-2b": {
            "url": "google/medgemma-2b",
            "type": "causal-lm",
            "description": "MedGemma 2B parameter medical language model",
        },
    }

    def __init__(self, models_dir: Path):
        """
        Initialize model manager.

        Args:
            models_dir: Directory for storing models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Metadata cache
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        self._load_all_metadata()

    def get_model_status(self, model_name: str) -> ModelStatus:
        """
        Get status of a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelStatus
        """
        model_path = self.models_dir / model_name

        if not model_path.exists():
            return ModelStatus.NOT_DOWNLOADED

        # Check if metadata exists
        metadata_file = model_path / self.METADATA_FILE
        if not metadata_file.exists():
            return ModelStatus.CORRUPTED

        # Check if required files exist
        if not self._verify_model_files(model_path):
            return ModelStatus.CORRUPTED

        # Check if outdated (would need version check)
        # For now, just return READY if files exist
        return ModelStatus.READY

    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelMetadata or None if not found
        """
        if model_name in self._metadata_cache:
            return self._metadata_cache[model_name]

        metadata_file = self.models_dir / model_name / self.METADATA_FILE
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                metadata = ModelMetadata.from_dict(data)
                self._metadata_cache[model_name] = metadata
                return metadata
            except Exception as e:
                self.logger.error(f"Failed to load metadata for {model_name}: {e}")

        return None

    def save_model_metadata(self, model_name: str, metadata: ModelMetadata):
        """
        Save metadata for a model.

        Args:
            model_name: Name of the model
            metadata: Model metadata
        """
        model_path = self.models_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        metadata_file = model_path / self.METADATA_FILE
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

            self._metadata_cache[model_name] = metadata
            self.logger.info(f"Saved metadata for {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to save metadata for {model_name}: {e}")
            raise

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.

        Returns:
            List of model information
        """
        models = []

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                status = self.get_model_status(model_name)
                metadata = self.get_model_metadata(model_name)

                model_info = {
                    "name": model_name,
                    "status": status.value,
                    "path": str(model_dir),
                }

                if metadata:
                    model_info.update({
                        "version": metadata.version,
                        "size_bytes": metadata.size_bytes,
                        "download_date": metadata.download_date,
                        "description": metadata.description,
                    })

                models.append(model_info)

        return models

    def verify_model(self, model_name: str) -> bool:
        """
        Verify model integrity.

        Args:
            model_name: Name of the model

        Returns:
            True if model is valid
        """
        model_path = self.models_dir / model_name

        if not model_path.exists():
            self.logger.error(f"Model not found: {model_name}")
            return False

        # Check required files
        if not self._verify_model_files(model_path):
            self.logger.error(f"Model files incomplete: {model_name}")
            return False

        # Verify checksum if available
        metadata = self.get_model_metadata(model_name)
        if metadata and metadata.checksum:
            actual_checksum = self._calculate_checksum(model_path)
            if actual_checksum != metadata.checksum:
                self.logger.error(f"Checksum mismatch for {model_name}")
                return False

        self.logger.info(f"Model verified: {model_name}")
        return True

    def get_model_size(self, model_name: str) -> int:
        """
        Get total size of model in bytes.

        Args:
            model_name: Name of the model

        Returns:
            Size in bytes
        """
        model_path = self.models_dir / model_name

        if not model_path.exists():
            return 0

        total_size = 0
        for file_path in model_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model.

        Args:
            model_name: Name of the model

        Returns:
            True if deleted successfully
        """
        model_path = self.models_dir / model_name

        if not model_path.exists():
            self.logger.warning(f"Model not found: {model_name}")
            return False

        try:
            import shutil
            shutil.rmtree(model_path)

            # Remove from cache
            if model_name in self._metadata_cache:
                del self._metadata_cache[model_name]

            self.logger.info(f"Deleted model: {model_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model {model_name}: {e}")
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information.

        Returns:
            Storage statistics
        """
        total_size = 0
        model_count = 0

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_count += 1
                total_size += self.get_model_size(model_dir.name)

        return {
            "models_dir": str(self.models_dir),
            "total_models": model_count,
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024**3), 2),
        }

    def create_model_metadata(
        self,
        model_name: str,
        version: str,
        model_type: str,
        source_url: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelMetadata:
        """
        Create metadata for a model.

        Args:
            model_name: Name of the model
            version: Model version
            model_type: Type of model
            source_url: URL where model was downloaded from
            description: Model description
            config: Additional configuration

        Returns:
            ModelMetadata
        """
        model_path = self.models_dir / model_name
        size_bytes = self.get_model_size(model_name) if model_path.exists() else 0
        checksum = self._calculate_checksum(model_path) if model_path.exists() else None

        metadata = ModelMetadata(
            name=model_name,
            version=version,
            model_type=model_type,
            size_bytes=size_bytes,
            download_date=datetime.now().isoformat(),
            checksum=checksum,
            source_url=source_url,
            description=description,
            config=config or {},
        )

        return metadata

    def _verify_model_files(self, model_path: Path) -> bool:
        """
        Verify that required model files exist.

        Args:
            model_path: Path to model directory

        Returns:
            True if all required files exist
        """
        # Check for essential HuggingFace files
        required_files = [
            "config.json",
        ]

        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                self.logger.warning(f"Missing required file: {file_name}")
                return False

        # Check for model weights (at least one should exist)
        weight_patterns = [
            "*.bin",
            "*.safetensors",
            "pytorch_model.bin",
            "model.safetensors",
        ]

        has_weights = False
        for pattern in weight_patterns:
            if list(model_path.glob(pattern)):
                has_weights = True
                break

        if not has_weights:
            self.logger.warning("No model weight files found")
            return False

        return True

    def _calculate_checksum(self, model_path: Path) -> str:
        """
        Calculate checksum for model directory.

        Args:
            model_path: Path to model directory

        Returns:
            MD5 checksum
        """
        md5_hash = hashlib.md5()

        # Hash all files in sorted order for consistency
        for file_path in sorted(model_path.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    # Hash file content in chunks
                    for chunk in iter(lambda: f.read(4096), b""):
                        md5_hash.update(chunk)

        return md5_hash.hexdigest()

    def _load_all_metadata(self):
        """Load metadata for all models"""
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                self.get_model_metadata(model_dir.name)


class ModelDownloader:
    """
    Downloads MedGemma models from HuggingFace.
    """

    def __init__(self, model_manager: ModelManager):
        """
        Initialize downloader.

        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

    def download_model(
        self,
        model_name: str,
        force: bool = False,
        token: Optional[str] = None,
    ) -> bool:
        """
        Download a model from HuggingFace.

        Args:
            model_name: Name of the model
            force: Force re-download even if exists
            token: HuggingFace API token

        Returns:
            True if download successful
        """
        # Check if already exists
        status = self.model_manager.get_model_status(model_name)
        if status == ModelStatus.READY and not force:
            self.logger.info(f"Model already downloaded: {model_name}")
            return True

        # Get model info
        if model_name not in ModelManager.SUPPORTED_MODELS:
            self.logger.error(f"Unsupported model: {model_name}")
            return False

        model_info = ModelManager.SUPPORTED_MODELS[model_name]
        model_url = model_info["url"]

        try:
            from huggingface_hub import snapshot_download

            self.logger.info(f"Downloading model: {model_name} from {model_url}")

            # Download model
            model_path = self.model_manager.models_dir / model_name
            snapshot_download(
                repo_id=model_url,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                token=token,
            )

            # Create metadata
            metadata = self.model_manager.create_model_metadata(
                model_name=model_name,
                version="latest",
                model_type=model_info["type"],
                source_url=model_url,
                description=model_info["description"],
            )

            self.model_manager.save_model_metadata(model_name, metadata)

            # Verify download
            if self.model_manager.verify_model(model_name):
                self.logger.info(f"Successfully downloaded: {model_name}")
                return True
            else:
                self.logger.error(f"Model verification failed: {model_name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            return False


def get_default_model_manager(models_dir: Optional[Path] = None) -> ModelManager:
    """
    Get default model manager instance.

    Args:
        models_dir: Models directory (default: ./models)

    Returns:
        ModelManager instance
    """
    if models_dir is None:
        models_dir = Path("./models")

    return ModelManager(models_dir)
