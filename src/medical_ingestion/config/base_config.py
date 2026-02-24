# ============================================================================
# src/medical_ingestion/config/base_config.py
# ============================================================================
"""
Base Configuration
- Project root
- Paths for models, templates, knowledge bases
- Audit DB
- Cache directories
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class BaseSettingsConfig(BaseSettings):
    # Root project directory
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Root directory of the project"
    )
    
    # Model weights directory
    MODEL_PATH: Path = Field(
        default=Path("models/medgemma"),
        description="Path to MedGemma model weights - MUST be local, no cloud"
    )
    
    # Template libraries
    TEMPLATES_DIR: Path = Field(
        default=Path("src/medical_ingestion/processors"),
        description="Root directory containing all processor templates"
    )
    
    # Knowledge bases
    KNOWLEDGE_DIR: Path = Field(
        default=Path("src/medical_ingestion/knowledge"),
        description="Medical knowledge bases (LOINC, SNOMED, protocols)"
    )
    
    # Audit trail database
    AUDIT_DB_PATH: Path = Field(
        default=Path("data/audit.db"),
        description="SQLite database for audit trail storage"
    )
    
    # Cache directory for MedGemma responses
    CACHE_DIR: Path = Field(
        default=Path("cache/medgemma"),
        description="Response cache for MedGemma (demo reliability)"
    )
    
    def create_directories(self):
        """Create all necessary directories if they don't exist"""
        dirs = [
            self.MODEL_PATH,
            self.TEMPLATES_DIR,
            self.KNOWLEDGE_DIR,
            self.CACHE_DIR,
            self.AUDIT_DB_PATH.parent
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_processor_template_dir(self, processor_name: str) -> Path:
        """Get template directory for specific processor"""
        return self.TEMPLATES_DIR / processor_name / "templates"

    def get_drafts_dir(self) -> Path:
        """Get directory for draft templates pending review"""
        drafts = self.TEMPLATES_DIR / "drafts"
        drafts.mkdir(parents=True, exist_ok=True)
        return drafts

# Global instance
base_settings = BaseSettingsConfig()
