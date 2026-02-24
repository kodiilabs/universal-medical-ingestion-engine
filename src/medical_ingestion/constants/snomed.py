# ============================================================================
# src/medical_ingestion/constants/snomed.py
# ============================================================================
"""
Common SNOMED Codes
- Pathology diagnoses and common findings
"""

import json
from pathlib import Path

# Load from JSON
# Path: constants/ -> medical_ingestion/ -> knowledge/
_knowledge_dir = Path(__file__).parent.parent / "knowledge"

try:
    with open(_knowledge_dir / "snomed_mappings.json") as f:
        SNOMED_CODES = json.load(f)
except FileNotFoundError:
    SNOMED_CODES = {}
