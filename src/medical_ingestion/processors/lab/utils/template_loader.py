# src/medical_ingestion/processors/lab/utils/template_loader.py
"""
Template loading utilities.
"""

import json
from typing import Dict
from pathlib import Path
from ....config import base_settings


def load_template(template_id: str) -> Dict:
    """Load template configuration."""
    template_path = base_settings.get_processor_template_dir("lab") / f"{template_id}.json"

    if template_path.exists():
        with open(template_path) as f:
            return json.load(f)

    return {}
