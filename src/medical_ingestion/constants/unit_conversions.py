# ============================================================================
# src/medical_ingestion/constants/unit_conversions.py
# ============================================================================
"""
Unit Conversion Tables
- Convert lab units between different measurement systems
"""

import json
from pathlib import Path

# Load from JSON
# Path: constants/ -> medical_ingestion/ -> knowledge/
_knowledge_dir = Path(__file__).parent.parent / "knowledge"

try:
    with open(_knowledge_dir / "unit_conversions.json") as f:
        UNIT_CONVERSIONS = json.load(f)
except FileNotFoundError:
    UNIT_CONVERSIONS = {}
